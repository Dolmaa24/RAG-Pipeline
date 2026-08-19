"""ChromaDB storage, including the guard that keeps one collection coherent."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from errors import PersistError
from pipeline.chunk.models import Chunk, ChunkedDocument, ChunkMetadata
from pipeline.store.chroma import ChromaStore


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    instance = ChromaStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")
    yield instance
    shutil.rmtree(tmp_path / "chroma", ignore_errors=True)


def _chunk(text: str, vector: list[float], **metadata) -> Chunk:
    return Chunk(
        document=text,
        metadata=ChunkMetadata(embedding_model="fake-dense", **metadata),
        dense_embedding=vector,
    )


def test_upsert_then_search_returns_the_nearest_chunk(store: ChromaStore):
    near = _chunk("The first chunk.", [0.1, 0.2, 0.3], source="a.html", page_no=1)
    far = _chunk("Another chunk entirely.", [0.9, 0.8, 0.7], source="b.html", page_no=2)

    assert store.upsert_document(ChunkedDocument(chunks=[near, far])) == 2
    assert store.count() == 2

    results = store.search(query_embedding=[0.11, 0.21, 0.31], n_results=1)
    assert results["ids"][0][0] == near.id
    assert results["documents"][0][0] == "The first chunk."


def test_metadata_survives_the_round_trip(store: ChromaStore):
    chunk = _chunk(
        "text",
        [0.1, 0.2, 0.3],
        source="https://example.com/a",
        page_no=7,
        section_name="Results > Revenue",
        language="en",
        chunk_strategy="hierarchical",
    )
    chunk.metadata.extra["content_hash"] = "abc123"
    store.upsert_document(ChunkedDocument(chunks=[chunk]))

    stored = store.search(query_embedding=[0.1, 0.2, 0.3], n_results=1)["metadatas"][0][0]
    assert stored["source"] == "https://example.com/a"
    assert stored["page_no"] == 7
    assert stored["section_name"] == "Results > Revenue"
    assert stored["chunk_strategy"] == "hierarchical"
    assert stored["embedding_model"] == "fake-dense"
    assert stored["extra_content_hash"] == "abc123"


def test_upsert_is_idempotent_on_the_same_id(store: ChromaStore):
    chunk = _chunk("first text", [0.1, 0.2, 0.3])
    store.upsert_document(ChunkedDocument(chunks=[chunk]))

    chunk.document = "revised text"
    store.upsert_document(ChunkedDocument(chunks=[chunk]))

    assert store.count() == 1
    assert store.search([0.1, 0.2, 0.3], n_results=1)["documents"][0][0] == "revised text"


def test_a_second_embedding_model_is_refused(store: ChromaStore):
    """384-d BGE and 512-d CLIP in one collection is a category error."""
    store.upsert_document(ChunkedDocument(chunks=[_chunk("bge text", [0.1, 0.2, 0.3])]))

    wrong = _chunk("clip text", [0.1] * 5)
    wrong.metadata.embedding_model = "clip"

    with pytest.raises(PersistError) as excinfo:
        store.upsert_document(ChunkedDocument(chunks=[wrong]))
    assert "3-dimensional" in str(excinfo.value)
    assert "CHROMA_COLLECTION_NAME" in str(excinfo.value)


def test_a_mismatched_query_is_refused(store: ChromaStore):
    store.upsert_document(ChunkedDocument(chunks=[_chunk("text", [0.1, 0.2, 0.3])]))
    with pytest.raises(PersistError, match="dimensional"):
        store.search([0.1, 0.2, 0.3, 0.4, 0.5])


def test_the_recorded_space_survives_reopening(store: ChromaStore, tmp_path: Path):
    store.upsert_document(ChunkedDocument(chunks=[_chunk("text", [0.1, 0.2, 0.3])]))

    reopened = ChromaStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")
    assert reopened._dimension == 3
    with pytest.raises(PersistError):
        reopened.upsert_document(ChunkedDocument(chunks=[_chunk("x", [0.1] * 8)]))


def test_unembedded_chunks_are_refused(store: ChromaStore):
    bare = Chunk(document="no vector", metadata=ChunkMetadata())
    with pytest.raises(PersistError, match="no dense embedding"):
        store.upsert_document(ChunkedDocument(chunks=[bare]))


def test_an_empty_document_writes_nothing(store: ChromaStore):
    assert store.upsert_document(ChunkedDocument()) == 0
    assert store.count() == 0


def test_batching_writes_every_chunk(store: ChromaStore):
    chunks = [
        _chunk(f"chunk {i}", [i / 250, 0.5, 0.5], page_no=i)
        for i in range(250)
    ]
    assert store.upsert_document(ChunkedDocument(chunks=chunks), batch_size=100) == 250
    assert store.count() == 250
