"""Shared fixtures.

The dense model is 130 MB and downloads on first use, so anything that only
needs *a* vector uses :class:`FakeDenseEmbedder` instead. Tests that genuinely
exercise BGE are marked ``slow``.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import pytest

from pipeline.chunk.models import ChunkedDocument
from pipeline.preprocess.orchestrator import PreprocessedDocument


class FakeDenseEmbedder:
    """Deterministic vectors from a hash. No model, no download, no network."""

    name = "fake"
    model_name = "fake-dense"

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([digest[i % len(digest)] / 255.0 for i in range(self.dimension)])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def unload(self) -> None:
        pass


class FakeEmbedder:
    """Stands in for DocumentEmbedder: fills vectors without loading anything."""

    def __init__(self, dimension: int = 8) -> None:
        self.dense = FakeDenseEmbedder(dimension)

    def embed(self, chunked_doc: ChunkedDocument) -> ChunkedDocument:
        texts = [chunk.document for chunk in chunked_doc.chunks]
        if not texts:
            return chunked_doc
        for chunk, vector in zip(chunked_doc.chunks, self.dense.embed_documents(texts)):
            chunk.dense_embedding = vector
            chunk.metadata.embedding_model = self.dense.model_name
        return chunked_doc


class RecordingStore:
    """Captures what would have been written."""

    collection_name = "test-collection"

    def __init__(self) -> None:
        self.documents: List[ChunkedDocument] = []

    def upsert_document(self, chunked_doc: ChunkedDocument, batch_size: int = 100) -> int:
        self.documents.append(chunked_doc)
        return len(chunked_doc.chunks)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def recording_store() -> RecordingStore:
    return RecordingStore()


@pytest.fixture
def make_doc():
    def _make(text: str, **metadata: Dict) -> PreprocessedDocument:
        return PreprocessedDocument(
            clean_text=text,
            language="en",
            metadata={"source": "test://doc", **metadata},
        )

    return _make


@pytest.fixture(autouse=True)
def _reset_caches():
    """No cached model may leak between tests."""
    from pipeline.embed import dense
    from pipeline.preprocess import pii_lang

    dense.reset()
    pii_lang.reset()
    yield
    dense.reset()
    pii_lang.reset()
