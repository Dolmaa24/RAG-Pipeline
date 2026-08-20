"""The retrieval orchestrator: both legs, degradation, and what it returns."""

from __future__ import annotations

from pipeline.graph.schema import Triple
from pipeline.retrieve.filters import MetadataFilter
from pipeline.retrieve.hybrid import HybridRetriever, ScoredChunk
from pipeline.retrieve.orchestrator import retrieve

ROWS = [
    {"id": "r1", "document": "quarterly revenue for ACME", "department": "finance",
     "language": "en", "date": "2026-02-01", "source": "a"},
    {"id": "r2", "document": "marine biology and coral reefs", "department": "research",
     "language": "en", "date": "2025-06-01", "source": "b"},
]


class _Embedder:
    model_name = "fake"

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _Graph:
    def __init__(self, triples=None, raises=None):
        self.triples = triples or []
        self.raises = raises
        self.calls = 0

    def retrieve(self, question, *, seeds=None, hops=None, limit=None, local_only=False):
        self.calls += 1
        if self.raises:
            raise self.raises
        return list(self.triples), list(seeds or ["seed"])


def _retriever(fake_store):
    return HybridRetriever(store=fake_store(ROWS), embedder=_Embedder())


def test_a_question_returns_chunks(fake_store):
    result = retrieve("what is the revenue", retriever=_retriever(fake_store), use_graph=False)
    assert result.chunks
    assert result.timings_ms["total"] > 0
    assert set(result.timings_ms) >= {"understand", "retrieve", "total"}


def test_an_empty_query_is_reported_not_raised(fake_store):
    result = retrieve("   ", retriever=_retriever(fake_store))
    assert result.chunks == []
    assert result.warnings == ["empty query"]


def test_the_graph_leg_runs_alongside(fake_store):
    triples = [Triple(source="A", relation="KNOWS", target="B")]
    graph = _Graph(triples)
    result = retrieve("who knows B", retriever=_retriever(fake_store), graph=graph, use_graph=True)

    assert graph.calls == 1
    assert result.triples == triples
    assert result.seeds == ["seed"]


def test_the_graph_leg_can_be_skipped(fake_store):
    graph = _Graph()
    retrieve("anything", retriever=_retriever(fake_store), graph=graph, use_graph=False)
    assert graph.calls == 0


def test_a_broken_graph_does_not_lose_the_chunks(fake_store):
    """An empty or absent graph is the normal early state."""
    graph = _Graph(raises=RuntimeError("kuzu missing"))
    result = retrieve("anything", retriever=_retriever(fake_store), graph=graph, use_graph=True)

    assert result.chunks
    assert result.triples == []
    assert any("graph retrieval failed" in w for w in result.warnings)


def test_a_broken_vector_leg_does_not_lose_the_graph(fake_store):
    class _Broken:
        def retrieve(self, *a, **k):
            raise RuntimeError("lance missing")

    graph = _Graph([Triple(source="A", relation="KNOWS", target="B")])
    result = retrieve("anything", retriever=_Broken(), graph=graph, use_graph=True)

    assert result.chunks == []
    assert result.triples
    assert any("vector retrieval failed" in w for w in result.warnings)


def test_filters_reach_the_store(fake_store):
    store = fake_store(ROWS)
    retrieve(
        "revenue",
        retriever=HybridRetriever(store=store, embedder=_Embedder()),
        filters=MetadataFilter(department=["finance"]),
        use_graph=False,
    )
    assert store.dense_calls == ["department = 'finance'"]


def test_filtering_narrows_the_results(fake_store):
    result = retrieve(
        "anything",
        retriever=_retriever(fake_store),
        filters=MetadataFilter(department=["research"]),
        use_graph=False,
    )
    assert [c.id for c in result.chunks] == ["r2"]


def test_rewriting_can_be_turned_off_per_query(fake_store, fake_backend, monkeypatch):
    backend = fake_backend({})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    result = retrieve(
        "how did revenue and headcount change after the merger",
        retriever=_retriever(fake_store),
        use_graph=False,
        rewrite=False,
    )
    assert backend.calls == 0
    assert result.plan.trivial is True


def test_the_result_serialises(fake_store):
    graph = _Graph([Triple(source="A", relation="KNOWS", target="B", description="d")])
    result = retrieve("x", retriever=_retriever(fake_store), graph=graph, use_graph=True)

    import json

    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["chunks"][0]["id"]
    assert payload["triples"][0]["text"] == "(A)-[KNOWS]->(B): d"
    assert "rewritten" in payload["plan"]


def test_context_renders_chunks_and_triples(fake_store):
    graph = _Graph([Triple(source="A", relation="KNOWS", target="B")])
    result = retrieve("x", retriever=_retriever(fake_store), graph=graph, use_graph=True)

    text = result.context()
    assert "quarterly revenue for ACME" in text
    assert "[knowledge graph]" in text
    assert "(A)-[KNOWS]->(B)" in text


def test_the_limit_is_respected(fake_store):
    result = retrieve("x", retriever=_retriever(fake_store), use_graph=False, limit=1)
    assert len(result.chunks) == 1


def test_reranking_reorders_the_shortlist(fake_store, monkeypatch):
    class _Reranker:
        def predict(self, pairs):
            # Score in reverse, so a reranked run must not match the fused order.
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setattr("pipeline.retrieve.rerank.get_reranker", lambda *a, **k: _Reranker())

    plain = retrieve("x", retriever=_retriever(fake_store), use_graph=False)
    reranked = retrieve(
        "x", retriever=_retriever(fake_store), use_graph=False, rerank_results=True
    )
    assert [c.id for c in reranked.chunks] == [c.id for c in plain.chunks][::-1]
    assert "rerank" in reranked.timings_ms


def test_a_broken_reranker_keeps_the_fused_order(fake_store, monkeypatch):
    def _broken(*a, **k):
        raise RuntimeError("no model")

    monkeypatch.setattr("pipeline.retrieve.rerank.get_reranker", _broken)

    result = retrieve("x", retriever=_retriever(fake_store), use_graph=False, rerank_results=True)
    assert result.chunks
