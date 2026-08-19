"""Attach dense and sparse vectors to a chunked document."""

from __future__ import annotations

import gc
import json
from typing import Optional

from config import config
from observability import get_logger, metrics

from pipeline.chunk.models import ChunkedDocument

log = get_logger("embed")


class DocumentEmbedder:
    """Enrich every chunk with a dense vector and a sparse term map.

    The dense model stays resident — see :mod:`pipeline.embed.dense` for why
    unloading it between documents was costing far more than it saved. The
    sparse side is constructed per call because the default (term frequency)
    is free to build, and the one provider that is not free (SPLADE) is exactly
    the one worth unloading afterwards.
    """

    def __init__(
        self,
        dense_provider: Optional[str] = None,
        sparse_provider: Optional[str] = None,
    ) -> None:
        self.dense_provider = dense_provider or config.INDEX_DENSE_PROVIDER
        self.sparse_provider = sparse_provider or config.INDEX_SPARSE_PROVIDER

    def embed(self, chunked_doc: ChunkedDocument) -> ChunkedDocument:
        texts = [chunk.document for chunk in chunked_doc.chunks]
        if not texts:
            return chunked_doc

        from pipeline.embed.dense import get_dense_embedder
        from pipeline.embed.sparse import get_sparse_embedder

        with metrics.timer("embed.dense"):
            dense_embedder = get_dense_embedder(self.dense_provider)
            dense_vectors = dense_embedder.embed_documents(texts)

        with metrics.timer("embed.sparse"):
            sparse_embedder = get_sparse_embedder(self.sparse_provider)
            try:
                sparse_vectors = sparse_embedder.embed_documents(texts)
            finally:
                # SPLADE is the reason this is here; the default costs nothing
                # to tear down and nothing to rebuild.
                sparse_embedder.unload()
                del sparse_embedder
                gc.collect()

        for chunk, dense, sparse in zip(chunked_doc.chunks, dense_vectors, sparse_vectors):
            chunk.dense_embedding = dense
            chunk.sparse_embedding = sparse
            chunk.metadata.embedding_model = dense_embedder.model_name
            # Chroma metadata holds scalars only, so the term map travels as a
            # JSON string. Nothing reads it back yet — see pipeline.embed.sparse.
            if sparse:
                chunk.metadata.extra["sparse_vector"] = json.dumps(sparse)

        log.info(
            "embed.done",
            chunks=len(texts),
            dense=self.dense_provider,
            sparse=self.sparse_provider,
        )
        return chunked_doc


__all__ = ["DocumentEmbedder"]
