"""The store: hybrid legs, pre-filtering, the ANN threshold, the space guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import config
from errors import PersistError
from pipeline.chunk.models import Chunk, ChunkedDocument, ChunkMetadata
from pipeline.store.lance import LanceStore
from pipeline.store.schema import FILTER_FIELDS


@pytest.fixture
def store(tmp_path: Path) -> LanceStore:
    return LanceStore(db_dir=str(tmp_path / "lance"), table_name="test")


def _chunk(text: str, vector: list[float], **metadata) -> Chunk:
    return Chunk(
        document=text,
        metadata=ChunkMetadata(embedding_model="fake-dense", **metadata),
        dense_embedding=vector,
    )


CORPUS = [
    _chunk(
        "the quarterly revenue report for ACME Corporation",
        [0.10, 0.20, 0.30],
        source="a.html", department="finance", language="en",
        date="2026-02-01", doc_type="html", author="Chen",
    ),
    _chunk(
        "a treatise on marine biology and coral reefs",
        [0.90, 0.80, 0.70],
        source="b.pdf", department="research", language="en",
        date="2025-06-01", doc_type="document", author="Okafor",
    ),
    _chunk(
        "finance team offsite agenda and budget planning",
        [0.15, 0.25, 0.35],
        source="c.html", department="finance", language="en",
        date="2026-03-15", doc_type="html", author="Chen",
    ),
]


def test_write_then_count(store: LanceStore):
    assert store.upsert_document(ChunkedDocument(chunks=list(CORPUS))) == 3
    assert store.count() == 3


def test_dense_returns_nearest_first(store: LanceStore):
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    hits = store.search_dense([0.11, 0.21, 0.31], limit=2)
    assert "ACME" in hits[0]["document"]
    assert hits[0]["score"] >= hits[1]["score"]


def test_bm25_finds_the_exact_words(store: LanceStore):
    """The case dense retrieval is worst at: rare, literal tokens."""
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    hits = store.search_fts("coral reefs", limit=5)
    assert hits
    assert "marine biology" in hits[0]["document"]


def test_bm25_and_dense_disagree_which_is_the_point(store: LanceStore):
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    # A vector nearest the ACME row, but wording that only matches the reef row.
    dense_top = store.search_dense([0.10, 0.20, 0.30], limit=1)[0]["id"]
    lexical_top = store.search_fts("coral reefs", limit=1)[0]["id"]
    assert dense_top != lexical_top


def test_prefilter_excludes_before_the_search_runs(store: LanceStore):
    """The nearest neighbour is filtered out, not returned and then dropped."""
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    hits = store.search_dense([0.90, 0.80, 0.70], limit=5, where="department = 'finance'")
    assert hits
    assert all(h["department"] == "finance" for h in hits)


def test_a_date_range_filters(store: LanceStore):
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    assert store.count("date >= '2026-01-01'") == 2
    assert store.count("date < '2026-01-01'") == 1


def test_every_filter_column_is_queryable(store: LanceStore):
    """A column the compiler can name must exist in the table."""
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    names = set(store.table.schema.names)
    for field in FILTER_FIELDS:
        assert field in names, field


def test_provenance_survives_the_round_trip(store: LanceStore):
    chunk = _chunk(
        "text", [0.1, 0.2, 0.3],
        source="https://example.com/a", page_no=7,
        section_name="Results > Revenue", chunk_strategy="hierarchical",
        department="legal", region="EMEA", permission_level="internal",
    )
    chunk.metadata.extra["content_hash"] = "abc123"
    chunk.metadata.extra["extraction_tier"] = 1
    store.upsert_document(ChunkedDocument(chunks=[chunk]))

    [row] = store.search_dense([0.1, 0.2, 0.3], limit=1)
    assert row["source"] == "https://example.com/a"
    assert row["page_no"] == 7
    assert row["section_name"] == "Results > Revenue"
    assert row["region"] == "EMEA"
    assert row["permission_level"] == "internal"
    assert row["content_hash"] == "abc123"
    assert row["extraction_tier"] == 1
    assert row["embedding_model"] == "fake-dense"


def test_rewriting_a_chunk_replaces_it(store: LanceStore):
    chunk = _chunk("first text", [0.1, 0.2, 0.3])
    store.upsert_document(ChunkedDocument(chunks=[chunk]))
    chunk.document = "revised text"
    store.upsert_document(ChunkedDocument(chunks=[chunk]))

    assert store.count() == 1
    assert store.search_dense([0.1, 0.2, 0.3], limit=1)[0]["document"] == "revised text"


def test_a_different_embedding_space_is_refused(store: LanceStore):
    """384-d BGE and 512-d CLIP in one table is a category error."""
    store.upsert_document(ChunkedDocument(chunks=[_chunk("bge", [0.1, 0.2, 0.3])]))

    wrong = _chunk("clip", [0.1] * 5)
    wrong.metadata.embedding_model = "clip"
    with pytest.raises(PersistError) as excinfo:
        store.upsert_document(ChunkedDocument(chunks=[wrong]))
    assert "LANCE_TABLE_NAME" in str(excinfo.value)


def test_the_space_guard_survives_reopening(store: LanceStore, tmp_path: Path):
    store.upsert_document(ChunkedDocument(chunks=[_chunk("x", [0.1, 0.2, 0.3])]))
    reopened = LanceStore(db_dir=str(tmp_path / "lance"), table_name="test")
    with pytest.raises(PersistError):
        reopened.upsert_document(ChunkedDocument(chunks=[_chunk("y", [0.1] * 8)]))


def test_unembedded_chunks_are_refused(store: LanceStore):
    bare = Chunk(document="no vector", metadata=ChunkMetadata())
    with pytest.raises(PersistError, match="no dense embedding"):
        store.upsert_document(ChunkedDocument(chunks=[bare]))


def test_an_empty_document_writes_nothing(store: LanceStore):
    assert store.upsert_document(ChunkedDocument()) == 0
    assert store.count() == 0


def test_searching_before_anything_is_written_is_empty(store: LanceStore):
    assert store.search_dense([0.1, 0.2, 0.3]) == []
    assert store.search_fts("anything") == []
    assert store.count() == 0


def test_no_ann_index_below_the_threshold(store: LanceStore):
    """IVF_PQ trains on the data; on three rows a flat scan is exact and faster."""
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    built = store.ensure_indexes()["built"]
    assert "ivf_pq" not in built
    assert "fts" in built
    assert any(name.startswith("scalar:") for name in built)


def test_distinct_reports_the_values_present(store: LanceStore):
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    assert store.distinct("department") == ["finance", "research"]
    assert store.distinct("author") == ["Chen", "Okafor"]


def test_a_scored_row_is_normalised_best_first(store: LanceStore):
    store.upsert_document(ChunkedDocument(chunks=list(CORPUS)))
    hits = store.search_dense([0.10, 0.20, 0.30], limit=3)
    assert hits[0]["score"] == pytest.approx(1.0)
    assert all(0.0 <= h["score"] <= 1.0 for h in hits)
    assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)


def test_removing_a_source_reports_what_it_removed(monkeypatch):
    """Deleting an uploaded file removed the file and left the corpus alone.

    The upload is only needed while it is being extracted; the text lives in
    the store afterwards, and nothing connected the two — so the system went on
    answering from documents the user believed they had removed.
    """
    from pipeline.store.lance import LanceStore

    deleted: list[str] = []

    class FakeTable:
        def delete(self, where):
            deleted.append(where)

    store = LanceStore.__new__(LanceStore)
    store._table = FakeTable()
    monkeypatch.setattr(type(store), "table", property(lambda self: self._table))
    monkeypatch.setattr(type(store), "count", lambda self, where=None: 3)

    assert store.delete_source("upload://ab12-report.pdf/") == 3
    assert deleted == ["source = 'upload://ab12-report.pdf/'"]


def test_removing_a_source_that_is_not_there_removes_nothing(monkeypatch):
    # "Removed 0" is the answer a caller most needs told accurately, so it is
    # counted rather than assumed from a return value LanceDB does not give.
    from pipeline.store.lance import LanceStore

    class FakeTable:
        def delete(self, where):  # pragma: no cover - must not be reached
            raise AssertionError("deleted when there was nothing to delete")

    store = LanceStore.__new__(LanceStore)
    store._table = FakeTable()
    monkeypatch.setattr(type(store), "table", property(lambda self: self._table))
    monkeypatch.setattr(type(store), "count", lambda self, where=None: 0)

    assert store.delete_source("upload://never-existed.pdf/") == 0


def test_a_blank_source_is_refused_rather_than_matching_everything(monkeypatch):
    from pipeline.store.lance import LanceStore

    class FakeTable:
        def delete(self, where):  # pragma: no cover
            raise AssertionError("a blank source must not reach a delete")

    store = LanceStore.__new__(LanceStore)
    store._table = FakeTable()
    monkeypatch.setattr(type(store), "table", property(lambda self: self._table))

    assert store.delete_source("   ") == 0


def test_a_source_with_a_quote_is_escaped(monkeypatch):
    from pipeline.store.lance import LanceStore

    deleted: list[str] = []

    class FakeTable:
        def delete(self, where):
            deleted.append(where)

    store = LanceStore.__new__(LanceStore)
    store._table = FakeTable()
    monkeypatch.setattr(type(store), "table", property(lambda self: self._table))
    monkeypatch.setattr(type(store), "count", lambda self, where=None: 1)

    store.delete_source("upload://o'brien.pdf/")
    assert deleted == ["source = 'upload://o''brien.pdf/'"]
