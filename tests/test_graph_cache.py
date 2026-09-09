"""The graph extraction cache — the most expensive call, made once."""

from __future__ import annotations

import pytest

from config import config
from pipeline.graph.cache import GraphCache, content_key
from pipeline.graph.extractor import GraphExtractor
from pipeline.graph.schema import Entity, KnowledgeGraphExtraction, Relationship

TEXT = "Northwind Traders acquired Fabrikam Ltd in 2025."

PAYLOAD = {
    "entities": [
        {"name": "Northwind Traders", "type": "Organization", "description": "a firm"},
        {"name": "Fabrikam Ltd", "type": "Organization", "description": "acquired"},
    ],
    "relationships": [
        {
            "source": "Northwind Traders",
            "target": "Fabrikam Ltd",
            "relation": "ACQUIRED",
            "description": "in 2025",
            "valid_year": "2025",
        }
    ],
}


@pytest.fixture
def cache():
    return GraphCache()


def _extraction() -> KnowledgeGraphExtraction:
    return KnowledgeGraphExtraction(
        entities=[Entity(**e) for e in PAYLOAD["entities"]],
        relationships=[Relationship(**r) for r in PAYLOAD["relationships"]],
    )


def test_a_miss_returns_nothing(cache: GraphCache):
    assert cache.get("hash1", "prompt", "model") is None


def test_what_goes_in_comes_back(cache: GraphCache):
    cache.put("hash1", "prompt", "model", _extraction())
    got = cache.get("hash1", "prompt", "model")

    assert got is not None
    assert [e.name for e in got.entities] == ["Northwind Traders", "Fabrikam Ltd"]
    assert got.relationships[0].relation == "ACQUIRED"
    assert got.relationships[0].valid_year == "2025"


def test_different_content_is_a_different_entry(cache: GraphCache):
    cache.put("hash1", "prompt", "model", _extraction())
    assert cache.get("hash2", "prompt", "model") is None


def test_a_changed_prompt_invalidates(cache: GraphCache):
    """Asking a different question must not be served the old answer."""
    cache.put("hash1", "prompt A", "model", _extraction())
    assert cache.get("hash1", "prompt B", "model") is None


def test_a_different_model_invalidates(cache: GraphCache):
    """llama3.2 and gpt-oss do not extract the same graph from one paragraph."""
    cache.put("hash1", "prompt", "llama3.2:3b", _extraction())
    assert cache.get("hash1", "prompt", "openai/gpt-oss-20b") is None


def test_an_empty_extraction_is_not_cached(cache: GraphCache):
    """Otherwise a transient failure becomes permanent for the whole TTL."""
    cache.put("hash1", "prompt", "model", KnowledgeGraphExtraction())
    assert cache.get("hash1", "prompt", "model") is None


def test_no_content_hash_means_no_caching(cache: GraphCache):
    cache.put("", "prompt", "model", _extraction())
    assert cache.get("", "prompt", "model") is None


def test_the_cache_can_be_turned_off(cache: GraphCache, monkeypatch):
    monkeypatch.setattr(config, "GRAPH_CACHE_ENABLED", False)
    cache.put("hash1", "prompt", "model", _extraction())
    assert cache.get("hash1", "prompt", "model") is None


def test_content_key_is_stable_and_content_addressed():
    assert content_key(TEXT) == content_key(TEXT)
    assert content_key(TEXT) != content_key(TEXT + " ")


def test_the_second_extraction_makes_no_model_call(fake_backend):
    """The point of all this: a re-crawl of unchanged text costs a lookup."""
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())

    first = extractor.extract(TEXT, source_url="u", content_hash="h1")
    second = extractor.extract(TEXT, source_url="u", content_hash="h1")

    assert backend.calls == 1
    assert [e.name for e in second.entities] == [e.name for e in first.entities]
    assert len(second.relationships) == len(first.relationships)


def test_a_domain_guidance_is_part_of_the_key(fake_backend):
    """Two skills ask different questions of the same paragraph.

    The insurance pack asks for policies and exclusions; the school pack asks
    for programmes and subjects. Without the guidance in the key the second
    domain is served the first domain's answer, and the mistake is invisible —
    a plausible graph, extracted for somebody else.
    """
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())

    extractor.extract(TEXT, source_url="u", content_hash="h1", guidance="entities: Policy")
    extractor.extract(TEXT, source_url="u", content_hash="h1", guidance="entities: Subject")

    assert backend.calls == 2


def test_the_same_guidance_still_hits_the_cache(fake_backend):
    """Or every skilled extraction would pay full price on every re-crawl."""
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())

    extractor.extract(TEXT, source_url="u", content_hash="h1", guidance="entities: Policy")
    extractor.extract(TEXT, source_url="u", content_hash="h1", guidance="entities: Policy")

    assert backend.calls == 1


def test_no_guidance_extracts_what_it_always_did(fake_backend):
    """The default path is untouched: a caller naming no skill gets the same
    prompt, the same key and the same graph as before skills existed."""
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())

    extractor.extract(TEXT, source_url="u", content_hash="h1")
    extractor.extract(TEXT, source_url="u", content_hash="h1", guidance="")

    assert backend.calls == 1


def test_the_guidance_reaches_the_model(fake_backend):
    """Keying on it is only half the job; the model has to be told."""
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())
    extractor.extract(TEXT, source_url="u", content_hash="h1", guidance="entities: Policy")

    assert "entities: Policy" in backend.last_prompt
    assert "Extract a knowledge graph" in backend.last_prompt


def test_changed_text_is_extracted_again(fake_backend):
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())

    extractor.extract(TEXT, content_hash="h1")
    extractor.extract("Different text entirely.", content_hash="h2")
    assert backend.calls == 2


def test_text_without_a_hash_is_still_cached(fake_backend):
    """build_graph on a bare string has no content hash; hash the text."""
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())

    extractor.extract(TEXT)
    extractor.extract(TEXT)
    assert backend.calls == 1


def test_provenance_survives_the_cache(fake_backend):
    backend = fake_backend(PAYLOAD)
    extractor = GraphExtractor(backend=backend, cache=GraphCache())

    extractor.extract(TEXT, source_url="https://example.com/a", content_hash="h1")
    cached = extractor.extract(TEXT, source_url="https://example.com/a", content_hash="h1")

    assert cached.entities[0].source_url == "https://example.com/a"
    assert cached.entities[0].content_hash == "h1"


def test_a_failed_extraction_is_not_cached(fake_backend):
    """A rate limit must not poison the entry for the TTL."""
    failing = fake_backend(raises=RuntimeError("rate limited"))
    extractor = GraphExtractor(backend=failing, cache=GraphCache())
    assert extractor.extract(TEXT, content_hash="h1").entities == []

    working = fake_backend(PAYLOAD)
    extractor._backend = working
    assert extractor.extract(TEXT, content_hash="h1").entities
    assert working.calls == 1
