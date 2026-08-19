"""ChromaDB vector storage.

One collection holds one embedding space. That is not a Chroma rule, it is a
geometry one: cosine distance between a 384-dimensional BGE vector and a
512-dimensional CLIP vector is not a smaller or larger number, it is a category
error — and Chroma will not stop you, so the collection records which model
wrote it and a mismatched write is refused here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from config import config
from errors import MissingDependency, PersistError
from observability import get_logger

from pipeline.chunk.models import ChunkedDocument, ChunkMetadata

log = get_logger("store.chroma")

#: Chroma metadata values must be scalars; anything else is JSON-encoded.
_SCALARS = (str, int, float, bool)


class ChromaStore:
    """A persistent local collection, keyed by content hash where one is given."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise MissingDependency("chromadb", "vector storage") from exc

        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME

        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._dimension: Optional[int] = self._recorded_dimension()

    # ------------------------------------------------------------------ #
    # Embedding-space guard
    # ------------------------------------------------------------------ #

    def _recorded_dimension(self) -> Optional[int]:
        recorded = (self.collection.metadata or {}).get("embedding_dim")
        return int(recorded) if recorded is not None else None

    def _check_dimension(self, dimension: int, model: str) -> None:
        """Claim the collection's embedding space, or refuse to corrupt it."""
        if self._dimension is None:
            # Chroma refuses a modify() that carries hnsw: settings, because the
            # distance function is fixed at creation — so the configuration keys
            # are dropped and only our own annotations are sent back.
            metadata = {
                key: value
                for key, value in (self.collection.metadata or {}).items()
                if not key.startswith("hnsw:")
            }
            metadata.update({"embedding_dim": dimension, "embedding_model": model})
            self.collection.modify(metadata=metadata)
            self._dimension = dimension
            log.info(
                "store.chroma.space_claimed",
                collection=self.collection_name,
                dimension=dimension,
                model=model,
            )
            return

        if self._dimension != dimension:
            raise PersistError(
                f"collection {self.collection_name!r} holds "
                f"{self._dimension}-dimensional vectors and this write is "
                f"{dimension}-dimensional ({model}). Use a separate collection "
                "per embedding model — CHROMA_COLLECTION_NAME.",
                collection=self.collection_name,
                expected=self._dimension,
                got=dimension,
            )

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatten_metadata(metadata: ChunkMetadata) -> Dict[str, Any]:
        data = metadata.model_dump()
        flat: Dict[str, Any] = {
            key: data[key]
            for key in (
                "source",
                "page_no",
                "section_name",
                "language",
                "chunk_strategy",
                "embedding_model",
            )
            if data.get(key) is not None
        }
        for key, value in (data.get("extra") or {}).items():
            flat[f"extra_{key}"] = value if isinstance(value, _SCALARS) else json.dumps(value)
        return flat

    def upsert_document(self, chunked_doc: ChunkedDocument, batch_size: int = 100) -> int:
        """Write every chunk. Returns how many landed."""
        chunks = chunked_doc.chunks
        if not chunks:
            return 0

        missing = [chunk.id for chunk in chunks if not chunk.dense_embedding]
        if missing:
            raise PersistError(
                f"{len(missing)} chunk(s) have no dense embedding; run "
                "DocumentEmbedder before storing",
                first=missing[0],
            )

        self._check_dimension(
            len(chunks[0].dense_embedding), chunks[0].metadata.embedding_model or "unknown"
        )

        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.document for chunk in batch],
                embeddings=[chunk.dense_embedding for chunk in batch],
                metadatas=[self._flatten_metadata(chunk.metadata) for chunk in batch],
            )
            written += len(batch)
            log.info(
                "store.chroma.upsert_batch",
                count=len(batch),
                collection=self.collection_name,
            )
        return written

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def search(self, query_embedding: List[float], n_results: int = 5) -> dict:
        """Dense nearest neighbours.

        Dense only. The sparse term maps written alongside each chunk are not
        searchable — see :mod:`pipeline.embed.sparse` for why hybrid retrieval
        needs a corpus-level pass that has not been built yet.
        """
        if self._dimension is not None and len(query_embedding) != self._dimension:
            raise PersistError(
                f"query is {len(query_embedding)}-dimensional but the collection "
                f"holds {self._dimension}-dimensional vectors",
                expected=self._dimension,
                got=len(query_embedding),
            )
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()


__all__ = ["ChromaStore"]
