"""Embedding: the resident model, and fork-safe device selection.

The sparse half that used to live here is gone. It wrote per-document term
counts that nothing could rank with, because ranking needs corpus-level inverse
document frequency and per-document counting cannot produce it. LanceDB indexes
the text column with BM25 directly — see tests/test_lance.py.
"""

from __future__ import annotations

import pytest

from pipeline.chunk.models import Chunk, ChunkedDocument, ChunkMetadata
from pipeline.embed import dense
from pipeline.embed.orchestrator import DocumentEmbedder


def _document(*texts: str) -> ChunkedDocument:
    return ChunkedDocument(
        chunks=[Chunk(document=text, metadata=ChunkMetadata(source="t")) for text in texts]
    )


class _Model:
    """A SentenceTransformer double that records how often it was built."""

    builds: list[tuple] = []

    def __init__(self, name, device=None):
        type(self).builds.append((name, device))

    def get_embedding_dimension(self):
        return 4

    def encode(self, texts, normalize_embeddings=True):
        import numpy

        return numpy.zeros((len(texts), 4))


@pytest.fixture
def fake_transformer(monkeypatch):
    _Model.builds = []
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _Model)
    return _Model


def test_the_model_is_built_once_per_process(fake_transformer):
    """The first version rebuilt a 130 MB model for every document."""
    first = dense.get_dense_embedder("local_bge")
    second = dense.get_dense_embedder("local_bge")
    assert first is second
    assert len(fake_transformer.builds) == 1


def test_reset_drops_the_cached_model(fake_transformer):
    dense.get_dense_embedder("local_bge")
    dense.reset()
    dense.get_dense_embedder("local_bge")
    assert len(fake_transformer.builds) == 2


def test_switching_provider_rebuilds(fake_transformer):
    dense.get_dense_embedder("local_bge")
    dense.get_dense_embedder("local_e5")
    assert [name for name, _ in fake_transformer.builds] == [
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2",
    ]


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown dense provider"):
        dense.get_dense_embedder("telepathy")


def test_auto_means_cpu():
    """Metal killed both the prefork worker and the uvicorn process."""
    from config import config

    assert config.INDEX_EMBED_DEVICE == "auto"
    assert dense.select_device() == "cpu"


def test_an_explicit_device_is_honoured(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "INDEX_EMBED_DEVICE", "mps")
    assert dense.select_device() == "mps"


def test_the_chosen_device_reaches_the_model(fake_transformer, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "INDEX_EMBED_DEVICE", "cpu")
    dense.get_dense_embedder("local_bge")
    assert fake_transformer.builds[0][1] == "cpu"


def test_every_chunk_gets_a_vector_and_its_model(monkeypatch, fake_embedder):
    monkeypatch.setattr(
        "pipeline.embed.dense.get_dense_embedder", lambda provider=None: fake_embedder.dense
    )
    document = DocumentEmbedder().embed(_document("the quick brown fox", "apple silicon"))

    for chunk in document.chunks:
        assert len(chunk.dense_embedding) == 8
        assert chunk.metadata.embedding_model == "fake-dense"


def test_all_chunks_embed_in_one_batch(monkeypatch, fake_embedder):
    monkeypatch.setattr(
        "pipeline.embed.dense.get_dense_embedder", lambda provider=None: fake_embedder.dense
    )
    DocumentEmbedder().embed(_document("a", "b", "c", "d"))
    assert fake_embedder.dense.calls == 1


def test_an_empty_document_is_a_no_op():
    assert DocumentEmbedder().embed(ChunkedDocument()).chunks == []


@pytest.mark.slow
def test_bge_small_produces_384_dimensions():
    """The real model. Downloads ~130 MB on first run."""
    embedder = dense.get_dense_embedder("local_bge")
    assert embedder.dimension == 384
    [vector] = embedder.embed_documents(["Apple Silicon memory limits."])
    assert len(vector) == 384
