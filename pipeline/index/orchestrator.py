"""Indexing: clean text in, retrievable chunks out.

    preprocess -> chunk -> embed -> store

This is to the indexing stages what :mod:`pipeline.runner` is to the extraction
stages: everything above it is a stage, everything below it is a transport. The
Celery task and the synchronous ``main.py`` path both call :func:`index_text`,
so a bug found in one is fixed in both.

The heavy imports live inside the function. Importing this module must stay cheap
because :mod:`tasks` imports it, and the io worker imports :mod:`tasks` while
having no business loading a sentence-transformer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from config import config
from observability import get_logger, metrics

log = get_logger("index")


@dataclass
class IndexReport:
    """What one indexing pass did."""

    source: str
    chunks: int = 0
    stored: int = 0
    strategy: str = "unknown"
    language: str = "unknown"
    pii_masked: bool = False
    truncated: bool = False
    collection: str = ""
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "chunks": self.chunks,
            "stored": self.stored,
            "strategy": self.strategy,
            "language": self.language,
            "pii_masked": self.pii_masked,
            "truncated": self.truncated,
            "collection": self.collection,
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
        }


def index_text(
    text: str,
    *,
    source: str,
    is_html: bool = False,
    page_no: Optional[int] = None,
    section_name: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
    strategy: Optional[str] = None,
    local_only: bool = False,
    store: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    chunker: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> IndexReport:
    """Run one document through preprocessing, chunking, embedding and storage.

    The four collaborators are injectable so a test can substitute a fake
    embedder and stay off the network — the same reason
    :class:`~pipeline.runner.Pipeline` takes its cascade and fetcher as
    arguments.
    """
    from pipeline.chunk.chunker import DocumentChunker
    from pipeline.embed.orchestrator import DocumentEmbedder
    from pipeline.preprocess.orchestrator import DocumentPreprocessor
    from pipeline.store.lance import LanceStore

    report = IndexReport(source=source, collection=config.LANCE_TABLE_NAME)
    started = time.perf_counter()

    if not text or not text.strip():
        report.warnings.append("no text to index")
        return report

    if len(text) > config.INDEX_MAX_TEXT_CHARS:
        log.warning(
            "index.truncated",
            source=source,
            chars=len(text),
            cap=config.INDEX_MAX_TEXT_CHARS,
        )
        report.warnings.append(
            f"text truncated from {len(text)} to {config.INDEX_MAX_TEXT_CHARS} characters"
        )
        report.truncated = True
        text = text[: config.INDEX_MAX_TEXT_CHARS]

    # --- preprocess ---------------------------------------------------- #
    stage = time.perf_counter()
    preprocessor = preprocessor or DocumentPreprocessor()
    document = preprocessor.process(
        text,
        is_html=is_html,
        source=source,
        page_no=page_no,
        section_name=section_name,
        extra_metadata=extra_metadata,
    )
    report.timings_ms["preprocess"] = round((time.perf_counter() - stage) * 1000, 2)
    report.language = document.language
    report.pii_masked = document.pii_masked

    if not document.clean_text.strip():
        report.warnings.append("nothing left after cleaning")
        log.warning("index.empty_after_clean", source=source)
        return report

    # --- chunk ---------------------------------------------------------- #
    stage = time.perf_counter()
    chunker = chunker or DocumentChunker()
    chunked = chunker.chunk(document, strategy=strategy, local_only=local_only)
    report.timings_ms["chunk"] = round((time.perf_counter() - stage) * 1000, 2)
    report.strategy = chunked.strategy
    report.chunks = len(chunked.chunks)

    if not chunked.chunks:
        report.warnings.append("chunking produced nothing")
        return report

    # --- embed ---------------------------------------------------------- #
    stage = time.perf_counter()
    embedder = embedder or DocumentEmbedder()
    chunked = embedder.embed(chunked)
    report.timings_ms["embed"] = round((time.perf_counter() - stage) * 1000, 2)

    # --- store ---------------------------------------------------------- #
    stage = time.perf_counter()
    store = store or LanceStore()
    report.stored = store.upsert_document(chunked)
    report.timings_ms["store"] = round((time.perf_counter() - stage) * 1000, 2)
    report.collection = getattr(store, "table_name", "")

    report.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
    metrics.incr("index.documents")
    metrics.incr("index.chunks", report.stored)
    log.info(
        "index.done",
        source=source,
        strategy=report.strategy,
        chunks=report.chunks,
        stored=report.stored,
        ms=report.timings_ms["total"],
    )
    return report


def preload() -> None:
    """Load the dense model now, so the first document does not pay for it."""
    from pipeline.embed.dense import preload as preload_dense

    preload_dense()


def reset() -> None:
    """Drop cached models. Used by tests."""
    from pipeline.embed.dense import reset as reset_dense
    from pipeline.preprocess.pii_lang import reset as reset_pii

    reset_dense()
    reset_pii()


__all__ = ["IndexReport", "index_text", "preload", "reset"]
