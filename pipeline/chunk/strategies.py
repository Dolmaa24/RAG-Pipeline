"""The four ways a document can be split.

Every strategy returns :class:`~pipeline.chunk.models.Chunk` objects carrying the
same metadata, so the caller can swap strategies without the store noticing.

Imports of the splitters are deferred into the functions. Importing LangChain at
module scope would make ``import pipeline.chunk`` cost a second and pull in
several hundred modules, which the io worker — which imports the task module and
never chunks anything — would pay for nothing.
"""

from __future__ import annotations

from typing import List, Optional

from config import config
from observability import get_logger

from pipeline.chunk.models import Chunk, ChunkMetadata
from pipeline.preprocess.orchestrator import PreprocessedDocument

log = get_logger("chunk")


#: Document metadata that has a named home on ChunkMetadata. Everything else the
#: caller supplied — content hash, resource kind, which tier extracted it — is
#: carried through in ``extra`` rather than dropped, because that is the
#: provenance that makes a retrieval hit traceable back to a stored record.
_NAMED_METADATA = frozenset({"source", "page_no", "section_name"})


def _metadata(doc: PreprocessedDocument, strategy: str, *, section: str = "") -> ChunkMetadata:
    return ChunkMetadata(
        source=doc.metadata.get("source", ""),
        page_no=doc.metadata.get("page_no", -1),
        section_name=section or doc.metadata.get("section_name", ""),
        language=doc.language,
        chunk_strategy=strategy,
        extra={
            key: value
            for key, value in doc.metadata.items()
            if key not in _NAMED_METADATA
        },
    )


# --------------------------------------------------------------------------- #
# 1. Fixed — overlapping windows
# --------------------------------------------------------------------------- #


def fixed_chunk(
    doc: PreprocessedDocument,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """Recursive character windows. The fallback everything else falls back to."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.INDEX_CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.INDEX_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return [
        Chunk(document=text, metadata=_metadata(doc, "fixed"))
        for text in splitter.split_text(doc.clean_text)
    ]


# --------------------------------------------------------------------------- #
# 2. Semantic — split where the meaning turns
# --------------------------------------------------------------------------- #


class _EmbeddingsAdapter:
    """Presents our dense embedder as the interface LangChain expects.

    This exists so semantic chunking reuses the *same* resident model as the
    embedding stage. The first version loaded all-MiniLM separately, which meant
    two sentence-transformers in memory at once on a machine with room for one.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.embed_query(text)


def semantic_chunk(doc: PreprocessedDocument) -> List[Chunk]:
    """Group sentences by embedding similarity, break where it drops."""
    from langchain_experimental.text_splitter import SemanticChunker

    from pipeline.embed.dense import get_dense_embedder

    splitter = SemanticChunker(
        _EmbeddingsAdapter(get_dense_embedder()),
        breakpoint_threshold_type="percentile",
    )
    return [
        Chunk(document=split.page_content, metadata=_metadata(doc, "semantic"))
        for split in splitter.create_documents([doc.clean_text])
    ]


# --------------------------------------------------------------------------- #
# 3. Hierarchical — follow the document's own headings
# --------------------------------------------------------------------------- #


def hierarchical_chunk(doc: PreprocessedDocument) -> List[Chunk]:
    """Split on markdown headers, keeping the heading path as the section name."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    )

    chunks: List[Chunk] = []
    for split in splitter.split_text(doc.clean_text):
        path = " > ".join(
            split.metadata[key]
            for key in ("Header 1", "Header 2", "Header 3")
            if split.metadata.get(key)
        )
        chunks.append(
            Chunk(
                document=split.page_content,
                metadata=_metadata(doc, "hierarchical", section=path),
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# 4. Model-assisted — for text too broken to split mechanically
# --------------------------------------------------------------------------- #

_LLM_SCHEMA = {"chunks": "list of strings"}

_LLM_PROMPT = (
    "The content is a document extracted from the web. It may contain OCR "
    "errors, tables broken across lines, and leftover navigation text. "
    "Repair the text, rebuild any broken table as markdown, drop the "
    "navigation noise, and split what remains into coherent sections. "
    "Return the sections in order as the 'chunks' field."
)


def llm_chunk(doc: PreprocessedDocument, *, local_only: bool = False) -> List[Chunk]:
    """Have a model repair and split text that mechanical splitting mangles.

    Built on :func:`pipeline.extract.llm.get_backend`, which is the same
    Ollama-or-Groq selection, schema validation and ``local_only`` handling the
    extraction cascade uses. The point is that the reply is *parsed and checked*:
    the first version of this function handed the raw completion string back as a
    single chunk, so a request for a JSON array of sections produced exactly one
    section containing the model's entire answer.

    Long documents go through in windows rather than being silently truncated —
    the original cut every input at 3000 characters and said nothing.
    """
    from pipeline.extract.llm import get_backend

    text = doc.clean_text
    if not text.strip():
        return []

    window = config.MAX_CHUNK_SIZE
    windows = [text[i : i + window] for i in range(0, len(text), window)]
    if len(windows) > config.INDEX_LLM_MAX_WINDOWS:
        log.warning(
            "chunk.llm_windows_capped",
            windows=len(windows),
            cap=config.INDEX_LLM_MAX_WINDOWS,
        )
        windows = windows[: config.INDEX_LLM_MAX_WINDOWS]

    try:
        backend = get_backend(local_only=local_only)
    except Exception as exc:
        log.warning("chunk.llm_unavailable", error=repr(exc))
        return fixed_chunk(doc)

    chunks: List[Chunk] = []
    for index, content in enumerate(windows):
        try:
            response = backend.complete_json(
                prompt=_LLM_PROMPT,
                content=content,
                schema_hint=_LLM_SCHEMA,
                json_schema={
                    "type": "object",
                    "properties": {
                        "chunks": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["chunks"],
                },
            )
        except Exception as exc:
            # One bad window should not lose the document. Split it the cheap
            # way and carry on with the rest.
            log.warning("chunk.llm_window_failed", window=index, error=repr(exc))
            chunks.extend(
                _windowed_fallback(doc, content)
            )
            continue

        sections = response.data.get("chunks") or []
        if not isinstance(sections, list):
            log.warning("chunk.llm_bad_shape", window=index, got=type(sections).__name__)
            chunks.extend(_windowed_fallback(doc, content))
            continue

        chunks.extend(
            Chunk(document=str(section), metadata=_metadata(doc, "llm"))
            for section in sections
            if str(section).strip()
        )

    # A model that returned nothing usable is a failure, not an empty document.
    return chunks or fixed_chunk(doc)


def _windowed_fallback(doc: PreprocessedDocument, content: str) -> List[Chunk]:
    """Fixed-split one window, keeping the parent document's metadata."""
    stand_in = doc.model_copy(update={"clean_text": content})
    return fixed_chunk(stand_in)


__all__ = ["fixed_chunk", "hierarchical_chunk", "llm_chunk", "semantic_chunk"]
