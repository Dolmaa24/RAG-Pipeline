"""Fusion, and the retriever that feeds it."""

from __future__ import annotations

from pipeline.retrieve.filters import MetadataFilter
from pipeline.retrieve.hybrid import (
    HybridRetriever,
    ScoredChunk,
    alpha_fusion,
    reciprocal_rank_fusion,
)


def _chunk(chunk_id: str, score: float, leg: str = "dense") -> ScoredChunk:
    return ScoredChunk(id=chunk_id, document=chunk_id, score=score, found_by={leg})


ROWS = [
    {
        "id": "r1",
        "document": "the quarterly revenue report for ACME Corporation",
        "department": "finance",
        "language": "en",
        "date": "2026-02-01",
        "source": "a",
    },
    {
        "id": "r2",
        "document": "a treatise on marine biology and coral reefs",
        "department": "research",
        "language": "en",
        "date": "2025-06-01",
        "source": "b",
    },
    {
        "id": "r3",
        "document": "finance team offsite agenda and budget",
        "department": "finance",
        "language": "en",
        "date": "2026-03-15",
        "source": "c",
    },
]


class _Embedder:
    model_name = "fake"
    dimension = 3

    def __init__(self):
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def test_rrf_rewards_a_document_both_legs_found():
    """Agreement between two independent methods is the strongest signal."""
    dense = [_chunk("a", 1.0), _chunk("b", 0.9), _chunk("c", 0.8)]
    lexical = [_chunk("c", 1.0, "bm25"), _chunk("d", 0.9, "bm25")]

    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0].id == "c"
    assert fused[0].found_by == {"dense", "bm25"}


def test_rrf_uses_rank_not_score():
    """Wildly different score scales must not change the ordering."""
    dense = [_chunk("a", 1000.0), _chunk("b", 999.0)]
    lexical = [_chunk("b", 0.002, "bm25"), _chunk("a", 0.001, "bm25")]

    scaled = reciprocal_rank_fusion([dense, lexical])
    dense_flat = [_chunk("a", 1.0), _chunk("b", 1.0)]
    lexical_flat = [_chunk("b", 1.0, "bm25"), _chunk("a", 1.0, "bm25")]
    unscaled = reciprocal_rank_fusion([dense_flat, lexical_flat])

    assert [c.id for c in scaled] == [c.id for c in unscaled]


def test_rrf_keeps_documents_only_one_leg_found():
    fused = reciprocal_rank_fusion([[_chunk("a", 1.0)], [_chunk("b", 1.0, "bm25")]])
    assert {c.id for c in fused} == {"a", "b"}


def test_fusion_preserves_the_rank_each_leg_gave():
    """Ranks are stamped on when rows leave the store; fusion merges them."""
    dense = ScoredChunk(id="target", document="t", score=0.5, found_by={"dense"}, dense_rank=1)
    lexical = ScoredChunk(id="target", document="t", score=1.0, found_by={"bm25"}, bm25_rank=0)

    [found] = reciprocal_rank_fusion([[dense], [lexical]])
    assert found.dense_rank == 1
    assert found.bm25_rank == 0
    assert found.found_by == {"dense", "bm25"}


def test_fusion_keeps_the_best_rank_across_subqueries():
    """The same chunk found by two sub-queries keeps its better placement."""
    worse = ScoredChunk(id="c", document="c", score=0.2, found_by={"dense"}, dense_rank=9)
    better = ScoredChunk(id="c", document="c", score=0.9, found_by={"dense"}, dense_rank=0)

    [found] = reciprocal_rank_fusion([[worse], [better]])
    assert found.dense_rank == 0


def test_alpha_one_is_pure_vector():
    dense = [_chunk("a", 0.9), _chunk("b", 0.1)]
    lexical = [_chunk("b", 1.0, "bm25")]
    assert [c.id for c in alpha_fusion(dense, lexical, alpha=1.0)] == ["a", "b"]


def test_alpha_zero_is_pure_bm25():
    dense = [_chunk("a", 0.9), _chunk("b", 0.1)]
    lexical = [_chunk("b", 1.0, "bm25")]
    assert alpha_fusion(dense, lexical, alpha=0.0)[0].id == "b"


def test_alpha_tuning_moves_the_ordering():
    """The knob has to actually do something across its range."""
    dense = [_chunk("vector_favourite", 1.0), _chunk("lexical_favourite", 0.2)]
    lexical = [_chunk("lexical_favourite", 1.0, "bm25")]

    assert alpha_fusion(dense, lexical, alpha=0.9)[0].id == "vector_favourite"
    assert alpha_fusion(dense, lexical, alpha=0.2)[0].id == "lexical_favourite"


def test_alpha_is_clamped():
    dense = [_chunk("a", 1.0)]
    assert alpha_fusion(dense, [], alpha=5.0)[0].score == 1.0
    assert alpha_fusion(dense, [], alpha=-3.0)[0].score == 0.0


def test_all_subqueries_embed_in_one_batch(fake_store):
    """N sub-queries must not cost N encode() calls."""
    embedder = _Embedder()
    retriever = HybridRetriever(store=fake_store(ROWS), embedder=embedder)

    retriever.retrieve(["first question", "second question", "third question"])

    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == 3


def test_both_legs_run_for_every_query(fake_store):
    store = fake_store(ROWS)
    HybridRetriever(store=store, embedder=_Embedder()).retrieve(["one", "two"])
    assert len(store.dense_calls) == 2
    assert len(store.fts_calls) == 2


def test_the_filter_reaches_both_legs(fake_store):
    store = fake_store(ROWS)
    HybridRetriever(store=store, embedder=_Embedder()).retrieve(
        ["revenue"], filters=MetadataFilter(department=["finance"])
    )
    assert store.dense_calls == ["department = 'finance'"]
    assert store.fts_calls[0][1] == "department = 'finance'"


def test_filtering_excludes_before_ranking(fake_store):
    """Pre-filtering, so a selective filter still returns a full top-k."""
    results = HybridRetriever(store=fake_store(ROWS), embedder=_Embedder()).retrieve(
        ["anything"], filters=MetadataFilter(department=["finance"]), limit=10
    )
    assert {c.id for c in results} == {"r1", "r3"}


def test_bm25_finds_the_lexical_match_dense_would_miss(fake_store):
    results = HybridRetriever(store=fake_store(ROWS), embedder=_Embedder()).retrieve(
        ["coral reefs"], limit=1
    )
    assert results[0].id == "r2"
    assert "bm25" in results[0].found_by


def test_a_failing_leg_does_not_lose_the_query(fake_store):
    class _BrokenFts(fake_store):
        def search_fts(self, *a, **k):
            raise RuntimeError("no fts index")

    results = HybridRetriever(store=_BrokenFts(ROWS), embedder=_Embedder()).retrieve(["x"])
    assert results  # the dense leg still answered


def test_no_queries_returns_nothing(fake_store):
    assert HybridRetriever(store=fake_store(ROWS), embedder=_Embedder()).retrieve([]) == []
    assert HybridRetriever(store=fake_store(ROWS), embedder=_Embedder()).retrieve(["  "]) == []


def test_the_limit_is_respected(fake_store):
    results = HybridRetriever(store=fake_store(ROWS), embedder=_Embedder()).retrieve(
        ["finance revenue"], limit=1
    )
    assert len(results) == 1
