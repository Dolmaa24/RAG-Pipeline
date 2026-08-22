"""Shared fixtures.

The dense model is 130 MB and downloads on first use, so anything that only
needs *a* vector uses :class:`FakeDenseEmbedder` instead. Tests that genuinely
exercise BGE are marked ``slow``.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Dict, List, Optional

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


class FakeBackend:
    """Stands in for an Ollama or Groq backend."""

    name = "fake"
    model = "fake-model"

    def __init__(self, data=None, raises=None):
        self._data = data or {}
        self._raises = raises
        self.calls = 0
        self.last_content = None

    def available(self) -> bool:
        return True

    def complete_json(self, *, prompt, content, schema_hint, json_schema):
        self.calls += 1
        self.last_content = content
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(data=self._data, backend=self.name, model=self.model)


class FakeStore:
    """A LanceStore-shaped double that ranks over an in-memory list.

    Dense hits are decided by the caller, lexical hits by substring — enough to
    exercise fusion, filtering and the orchestrator without a database.
    """

    table_name = "fake"

    def __init__(self, rows: Optional[List[dict]] = None) -> None:
        self.rows = rows or []
        self.dense_calls: List[Optional[str]] = []
        self.fts_calls: List[tuple] = []

    def _filtered(self, where):
        if not where:
            return list(self.rows)
        # Only the shapes the compiler emits, which is all the tests produce.
        kept = []
        for row in self.rows:
            ok = True
            for clause in where.split(" AND "):
                if " = " in clause:
                    field, value = clause.split(" = ", 1)
                    if str(row.get(field.strip(), "")) != value.strip().strip("'"):
                        ok = False
                elif " IN " in clause:
                    field, values = clause.split(" IN ", 1)
                    allowed = {v.strip().strip("'") for v in values.strip("() ").split(",")}
                    if str(row.get(field.strip(), "")) not in allowed:
                        ok = False
                elif ">=" in clause:
                    field, value = clause.split(">=", 1)
                    if str(row.get(field.strip(), "")) < value.strip().strip("'"):
                        ok = False
                elif "<=" in clause:
                    field, value = clause.split("<=", 1)
                    if str(row.get(field.strip(), "")) > value.strip().strip("'"):
                        ok = False
            if ok:
                kept.append(row)
        return kept

    def search_dense(self, vector, *, limit=50, where=None):
        self.dense_calls.append(where)
        rows = self._filtered(where)[:limit]
        return [dict(r, score=1.0 - i * 0.1, rank=i) for i, r in enumerate(rows)]

    def search_fts(self, text, *, limit=50, where=None):
        self.fts_calls.append((text, where))
        terms = [t for t in text.lower().split() if len(t) > 2]
        hits = [
            r for r in self._filtered(where)
            if any(t in str(r.get("document", "")).lower() for t in terms)
        ][:limit]
        return [dict(r, score=1.0 - i * 0.1, rank=i) for i, r in enumerate(hits)]


@pytest.fixture
def fake_backend():
    return FakeBackend


@pytest.fixture
def fake_store():
    return FakeStore


@pytest.fixture(autouse=True)
def _offline_defaults(monkeypatch):
    """Keep the suite offline by default.

    ``GRAPH_ENTITY_BACKEND`` ships as "gliner", which downloads and loads a real
    model. Tests that mean to exercise that path monkeypatch
    ``pipeline.graph.ner.get_model`` with a double and set the backend
    themselves; everything else gets the LLM path, whose backend is already
    faked. Without this, adding a test anywhere silently pulls a model.
    """
    from config import config

    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "llm")


@pytest.fixture(autouse=True)
def _reset_caches():
    """No cached model, backend or plan may leak between tests."""
    from pipeline.embed import dense
    from pipeline.preprocess import pii_lang
    # Imported by symbol: `from pipeline.retrieve import understand` gets the
    # function the package re-exports, not this module.
    from pipeline.graph import cache as graph_cache
    from pipeline.graph import ner as graph_ner
    from pipeline.retrieve.understand import clear_cache

    def _clear():
        dense.reset()
        pii_lang.reset()
        clear_cache()
        graph_cache.reset()
        graph_ner.reset()

    _clear()
    yield
    _clear()


@pytest.fixture
def anyio_backend():
    """asyncio only. The MCP server tests are async; trio is not installed."""
    return "asyncio"
