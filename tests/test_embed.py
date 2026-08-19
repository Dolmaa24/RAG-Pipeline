"""Embedding: the dense singleton, and the sparse half's honest contract."""

from __future__ import annotations

import json

import pytest

from pipeline.chunk.models import Chunk, ChunkedDocument, ChunkMetadata
from pipeline.embed import dense
from pipeline.embed.orchestrator import DocumentEmbedder
from pipeline.embed.sparse import (
    NullSparseEmbedder,
    TermFrequencyEmbedder,
    get_sparse_embedder,
)


def _document(*texts: str) -> ChunkedDocument:
    return ChunkedDocument(
        chunks=[Chunk(document=text, metadata=ChunkMetadata(source="t")) for text in texts]
    )


# --------------------------------------------------------------------------- #
# Sparse
# --------------------------------------------------------------------------- #


def test_term_frequency_counts_terms():
    [vector] = TermFrequencyEmbedder().embed_documents(["the cat, the hat."])
    assert vector == {"the": 2.0, "cat": 1.0, "hat": 1.0}


def test_term_frequency_is_not_bm25():
    """No IDF: a term common to every document is not down-weighted.

    This is the property that stops these vectors being rankable on their own,
    and the reason the class is not called BM25Embedder.
    """
    embedder = TermFrequencyEmbedder()
    vectors = embedder.embed_documents(["common alpha", "common beta"])
    assert vectors[0]["common"] == vectors[0]["alpha"]


def test_null_sparse_stores_nothing():
    assert NullSparseEmbedder().embed_documents(["a", "b"]) == [{}, {}]


def test_sparse_factory_rejects_the_old_name():
    """'bm25' was never BM25; a config still naming it should fail loudly."""
    with pytest.raises(ValueError, match="unknown sparse provider"):
        get_sparse_embedder("bm25")


# --------------------------------------------------------------------------- #
# Dense
# --------------------------------------------------------------------------- #


def test_dense_embedder_is_built_once_per_process(monkeypatch):
    """The bug this replaced rebuilt a 130 MB model on every document."""
    builds = []

    class _Model:
        def __init__(self, name, device=None):
            builds.append((name, device))

        def get_embedding_dimension(self):
            return 4

        def encode(self, texts, normalize_embeddings=True):
            import numpy

            return numpy.zeros((len(texts), 4))

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _Model)

    first = dense.get_dense_embedder("local_bge")
    second = dense.get_dense_embedder("local_bge")
    assert first is second
    assert len(builds) == 1


def test_dense_factory_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="unknown dense provider"):
        dense.get_dense_embedder("telepathy")


def test_reset_drops_the_cached_model(monkeypatch):
    builds = []

    class _Model:
        def __init__(self, name, device=None):
            builds.append((name, device))

        def get_embedding_dimension(self):
            return 4

        def encode(self, texts, normalize_embeddings=True):
            import numpy

            return numpy.zeros((len(texts), 4))

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _Model)
    dense.get_dense_embedder("local_bge")
    dense.reset()
    dense.get_dense_embedder("local_bge")
    assert len(builds) == 2


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #


def test_embed_attaches_both_halves(monkeypatch, fake_embedder):
    monkeypatch.setattr(
        "pipeline.embed.dense.get_dense_embedder", lambda provider=None: fake_embedder.dense
    )

    document = DocumentEmbedder(sparse_provider="tf").embed(
        _document("the quick brown fox", "apple silicon memory limits")
    )

    for chunk in document.chunks:
        assert len(chunk.dense_embedding) == 8
        assert chunk.metadata.embedding_model == "fake-dense"
        assert chunk.sparse_embedding
        # Chroma metadata holds scalars, so the term map travels as JSON.
        assert json.loads(chunk.metadata.extra["sparse_vector"]) == chunk.sparse_embedding


def test_embed_on_an_empty_document_is_a_no_op():
    assert DocumentEmbedder().embed(ChunkedDocument()).chunks == []


def test_sparse_none_writes_no_metadata(monkeypatch, fake_embedder):
    monkeypatch.setattr(
        "pipeline.embed.dense.get_dense_embedder", lambda provider=None: fake_embedder.dense
    )
    document = DocumentEmbedder(sparse_provider="none").embed(_document("text"))
    assert "sparse_vector" not in document.chunks[0].metadata.extra


@pytest.mark.slow
def test_bge_small_produces_384_dimensions():
    """The real model. Downloads ~130 MB on first run."""
    embedder = dense.get_dense_embedder("local_bge")
    assert embedder.dimension == 384
    [vector] = embedder.embed_documents(["Apple Silicon memory limits."])
    assert len(vector) == 384


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #


def test_a_forked_worker_child_is_pinned_to_cpu(monkeypatch):
    """Metal does not survive fork(); an MPS pipeline in a child SIGABRTs."""

    class _ForkChild:
        name = "ForkPoolWorker-3"

    monkeypatch.setattr("billiard.process.current_process", lambda: _ForkChild())
    assert dense.select_device() == "cpu"


def test_outside_a_fork_pool_torch_chooses(monkeypatch):
    class _Main:
        name = "MainProcess"

    monkeypatch.setattr("billiard.process.current_process", lambda: _Main())
    assert dense.select_device() is None


def test_an_explicit_device_overrides_the_detection(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "INDEX_EMBED_DEVICE", "mps")

    class _ForkChild:
        name = "ForkPoolWorker-1"

    monkeypatch.setattr("billiard.process.current_process", lambda: _ForkChild())
    assert dense.select_device() == "mps"
