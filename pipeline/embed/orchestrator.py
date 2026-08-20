"""Attach dense vectors to a chunked document.

There used to be a sparse half here, writing per-document term counts into
metadata that nothing ever read — it could not be read, because ranking with
term frequencies requires the corpus-level inverse document frequency that
per-document counting cannot produce. LanceDB indexes the text column with BM25
directly, computing that IDF across the whole table, so the lexical half of
hybrid search is now the store's job and is done properly.
"""

from __future__ import annotations

from typing import Optional

from config import config
from observability import get_logger, metrics

from pipeline.chunk.models import ChunkedDocument

log = get_logger("embed")


class DocumentEmbedder:
    """Give every chunk a dense vector.

    The model stays resident — see :mod:`pipeline.embed.dense` for why unloading
    it between documents cost far more than it saved.
    """

    def __init__(self, dense_provider: Optional[str] = None) -> None:
        self.dense_provider = dense_provider or config.INDEX_DENSE_PROVIDER

    def embed(self, chunked_doc: ChunkedDocument) -> ChunkedDocument:
        texts = [chunk.document for chunk in chunked_doc.chunks]
        if not texts:
            return chunked_doc

        from pipeline.embed.dense import get_dense_embedder

        embedder = get_dense_embedder(self.dense_provider)
        with metrics.timer("embed.dense"):
            vectors = embedder.embed_documents(texts)

        for chunk, vector in zip(chunked_doc.chunks, vectors):
            chunk.dense_embedding = vector
            chunk.metadata.embedding_model = embedder.model_name

        log.info("embed.done", chunks=len(texts), provider=self.dense_provider)
        return chunked_doc


__all__ = ["DocumentEmbedder"]
