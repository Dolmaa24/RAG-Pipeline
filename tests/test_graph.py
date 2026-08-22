"""The knowledge graph: injection, idempotency, hops, and the read-only guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.graph.cypher import CypherAgent, is_read_only
from pipeline.graph.extractor import GraphExtractor
from pipeline.graph.resolution import EntityResolver, remap_relationships
from pipeline.graph.schema import Entity, Relationship
from pipeline.graph.store import GraphStore


@pytest.fixture
def graph(tmp_path: Path) -> GraphStore:
    return GraphStore(db_path=str(tmp_path / "kuzu"))


def _entity(name: str, **kw) -> Entity:
    return Entity(name=name, type=kw.pop("type", "Person"), description=kw.pop("description", ""), **kw)


def _rel(source: str, target: str, relation: str, **kw) -> Relationship:
    return Relationship(source=source, target=target, relation=relation,
                        description=kw.pop("description", ""), **kw)


CAREER = (
    [_entity(n, type="Organization") for n in ("Tesla", "Apple", "Ford")] + [_entity("Doug Field")],
    [
        _rel("Doug Field", "Tesla", "WORKED_AT", description="SVP Engineering", valid_year="2013"),
        _rel("Doug Field", "Apple", "LED", description="Project Titan", valid_year="2018"),
        _rel("Apple", "Ford", "LOST_TALENT_TO", description="Field departed", valid_year="2021"),
    ],
)


# --------------------------------------------------------------------------- #
# The injection case
# --------------------------------------------------------------------------- #


def test_a_quote_in_an_entity_name_round_trips(graph: GraphStore):
    """The original built Cypher by interpolation; this is what that broke on."""
    graph.upsert([_entity("O'Brien & Co", type="Organization")], [])
    assert "O'Brien & Co" in [e["name"] for e in graph.entities()]


def test_a_cypher_injection_attempt_is_stored_as_a_name(graph: GraphStore):
    hostile = "x'}) DETACH DELETE (n) //"
    graph.upsert([_entity(hostile)], [])
    names = [e["name"] for e in graph.entities()]
    assert hostile in names
    assert graph.count()["entities"] == 1


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def test_reingesting_the_same_document_adds_nothing(graph: GraphStore):
    """The original used CREATE for edges, so every re-crawl doubled them."""
    entities, relationships = CAREER
    graph.upsert(entities, relationships)
    first = graph.count()
    graph.upsert(entities, relationships)
    assert graph.count() == first


def test_a_second_relation_between_the_same_pair_is_a_second_edge(graph: GraphStore):
    graph.upsert(
        [_entity("A"), _entity("B")],
        [_rel("A", "B", "WORKED_AT"), _rel("A", "B", "INVESTED_IN")],
    )
    assert graph.count()["relationships"] == 2


def test_an_edge_updates_in_place(graph: GraphStore):
    graph.upsert([_entity("A"), _entity("B")], [_rel("A", "B", "KNOWS", description="old")])
    graph.upsert([_entity("A"), _entity("B")], [_rel("A", "B", "KNOWS", description="new")])
    [triple] = graph.neighbours(["A"], hops=1)
    assert triple.description == "new"
    assert graph.count()["relationships"] == 1


# --------------------------------------------------------------------------- #
# Traversal
# --------------------------------------------------------------------------- #


def test_one_hop_returns_direct_edges(graph: GraphStore):
    graph.upsert(*CAREER)
    triples = graph.neighbours(["Doug Field"], hops=1)
    assert {t.target for t in triples} == {"Tesla", "Apple"}
    assert all(t.relation != "RELATED_TO" for t in triples)


def test_two_hops_reach_the_indirect_fact(graph: GraphStore):
    """The join no single chunk contains, which is why the graph is here."""
    graph.upsert(*CAREER)
    triples = graph.neighbours(["Doug Field"], hops=2)
    assert any(t.source == "Apple" and t.target == "Ford" for t in triples)


def test_a_multi_hop_path_keeps_each_relation(graph: GraphStore):
    """Collapsing a path to its endpoints throws away the middle."""
    graph.upsert(*CAREER)
    relations = {t.relation for t in graph.neighbours(["Doug Field"], hops=2)}
    assert {"LED", "WORKED_AT", "LOST_TALENT_TO"} <= relations


def test_traversal_does_not_repeat_an_edge(graph: GraphStore):
    graph.upsert(*CAREER)
    triples = graph.neighbours(["Doug Field"], hops=2)
    keys = [(t.source, t.relation, t.target) for t in triples]
    assert len(keys) == len(set(keys))


def test_the_hop_cap_holds(graph: GraphStore, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "GRAPH_MAX_HOPS", 1)
    graph.upsert(*CAREER)
    assert not any(t.source == "Apple" for t in graph.neighbours(["Doug Field"], hops=9))


def test_no_seeds_returns_nothing(graph: GraphStore):
    graph.upsert(*CAREER)
    assert graph.neighbours([]) == []


# --------------------------------------------------------------------------- #
# The read-only guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a:Entity) DETACH DELETE a",
        "MATCH (a:Entity) SET a.name = 'x'",
        "MATCH (a:Entity) RETURN a; DROP TABLE Entity",
        "CREATE (a:Entity {name: 'x'})",
        "MERGE (a:Entity {name: 'x'})",
        "CALL something()",
        "",
        "RETURN 1",
    ],
)
def test_a_write_is_not_read_only(query):
    assert not is_read_only(query)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a:Entity)-[r:CONNECTS_TO]->(b:Entity) RETURN a.name, r.relation, b.name",
        "OPTIONAL MATCH (a:Entity) RETURN a.name",
        "MATCH (a:Entity) WHERE a.name = 'x' RETURN a.name;",
    ],
)
def test_a_plain_read_passes(query):
    assert is_read_only(query)


def test_the_database_refuses_a_write_even_if_the_check_is_wrong(graph: GraphStore):
    """The guard can be out-thought. A read-only database cannot."""
    graph.upsert(*CAREER)
    with pytest.raises(Exception, match="read-only"):
        graph.read_connection.execute("MATCH (e:Entity) DETACH DELETE e")


def test_the_agent_refuses_to_return_a_write(fake_backend):
    agent = CypherAgent(backend=fake_backend({"cypher_query": "MATCH (a:Entity) DELETE a"}))
    assert agent.generate("anything", ["seed"]) is None


def test_the_agent_returns_a_valid_read(fake_backend):
    good = "MATCH (a:Entity)-[r:CONNECTS_TO]->(b:Entity) RETURN a.name, r.relation, b.name"
    agent = CypherAgent(backend=fake_backend({"cypher_query": good}))
    assert agent.generate("anything", ["seed"]) == good


def test_the_agent_declines_without_seeds(fake_backend):
    backend = fake_backend({"cypher_query": "MATCH (a) RETURN a"})
    assert CypherAgent(backend=backend).generate("anything", []) is None
    assert backend.calls == 0


# --------------------------------------------------------------------------- #
# Extraction and resolution
# --------------------------------------------------------------------------- #


def test_extraction_attaches_provenance(fake_backend):
    backend = fake_backend(
        {
            "entities": [{"name": "ACME", "type": "Organization", "description": "a firm"}],
            "relationships": [],
        }
    )
    result = GraphExtractor(backend=backend).extract(
        "text", source_url="https://example.com/a", content_hash="deadbeef"
    )
    assert result.entities[0].source_url == "https://example.com/a"
    assert result.entities[0].content_hash == "deadbeef"


def test_an_edge_naming_an_unlisted_entity_is_dropped(fake_backend):
    """An endpoint with no type and no description is not a node worth having."""
    backend = fake_backend(
        {
            "entities": [{"name": "A", "type": "P", "description": "d"}],
            "relationships": [
                {"source": "A", "target": "Ghost", "relation": "KNOWS", "description": ""}
            ],
        }
    )
    assert GraphExtractor(backend=backend).extract("text").relationships == []


def test_a_failed_extraction_is_empty_not_an_exception(fake_backend):
    backend = fake_backend(raises=RuntimeError("rate limited"))
    result = GraphExtractor(backend=backend).extract("text")
    assert result.entities == [] and result.relationships == []


class _NameEmbedder:
    """Similar strings get similar vectors, so resolution can be tested."""

    model_name = "fake"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            key = text.lower().split()[0] if text.split() else ""
            out.append([1.0, 0.0] if key.startswith("apple") else [0.0, 1.0])
        return out

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_a_case_variant_resolves_without_a_model(fake_backend):
    backend = fake_backend({"is_same": False})
    resolver = EntityResolver(embedder=_NameEmbedder(), backend=backend)
    new, aliases = resolver.resolve([_entity("apple")], [{"name": "Apple", "type": "Org", "description": ""}])
    assert new == []
    assert aliases == {"apple": "Apple"}
    assert backend.calls == 0


def test_the_model_decides_a_near_match_suffixes_cannot(fake_backend):
    """Names that differ by more than a legal suffix still need judgement."""
    backend = fake_backend({"is_same": True})
    resolver = EntityResolver(threshold=0.5, embedder=_NameEmbedder(), backend=backend)
    new, aliases = resolver.resolve(
        [_entity("Apple Computer")], [{"name": "Apple", "type": "Org", "description": ""}]
    )
    assert aliases == {"Apple Computer": "Apple"}
    assert backend.calls == 1


def test_the_model_can_refuse_a_merge(fake_backend):
    """Two different things that share a name must stay two nodes."""
    backend = fake_backend({"is_same": False})
    resolver = EntityResolver(threshold=0.5, embedder=_NameEmbedder(), backend=backend)
    new, aliases = resolver.resolve(
        [_entity("Apple Records")], [{"name": "Apple", "type": "Org", "description": ""}]
    )
    assert aliases == {}
    assert [e.name for e in new] == ["Apple Records"]


def test_the_pool_is_embedded_once_not_once_per_entity():
    """The original re-encoded the whole graph for every new entity."""
    embedder = _NameEmbedder()
    resolver = EntityResolver(embedder=embedder, verify_with_model=False)
    existing = [{"name": f"Entity {i}", "type": "X", "description": ""} for i in range(50)]
    resolver.resolve([_entity(f"New {i}") for i in range(10)], existing)

    # One batch for the pool, then one per genuinely new name. Not 50 per name.
    assert embedder.calls <= 1 + 10 * 2


def test_remapping_points_edges_at_the_surviving_node():
    edges = remap_relationships([_rel("Apple Inc.", "Ford", "SOLD_TO")], {"Apple Inc.": "Apple"})
    assert edges[0].source == "Apple"


def test_remapping_drops_the_self_loop_a_merge_creates():
    edges = remap_relationships(
        [_rel("Apple Inc.", "Apple Corp", "PARTNERED_WITH")],
        {"Apple Inc.": "Apple", "Apple Corp": "Apple"},
    )
    assert edges == []


# --------------------------------------------------------------------------- #
# Suffix resolution — the dominant alias pattern, decided without a model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Apple", "apple"),
        ("Apple Inc.", "apple"),
        ("Apple Corp", "apple"),
        ("APPLE INCORPORATED", "apple"),
        ("Acme Holdings Ltd.", "acme"),
        ("Ford Motor Company", "ford motor"),
        ("Tesla, Inc.", "tesla"),
    ],
)
def test_corporate_suffixes_collapse(name, expected):
    from pipeline.graph.resolution import canonical_key

    assert canonical_key(name) == expected


def test_a_suffix_variant_merges_without_asking_a_model(fake_backend):
    """A 3B model gets 'Apple Inc.' vs 'Apple' wrong. This never asks it."""
    backend = fake_backend({"is_same": False})
    resolver = EntityResolver(embedder=_NameEmbedder(), backend=backend)

    new, aliases = resolver.resolve(
        [_entity("Apple Inc.", type="Organization")],
        [{"name": "Apple", "type": "Organization", "description": ""}],
    )
    assert new == []
    assert aliases == {"Apple Inc.": "Apple"}
    assert backend.calls == 0


def test_a_bare_suffix_word_is_not_stripped_to_nothing():
    from pipeline.graph.resolution import canonical_key

    assert canonical_key("Corp") == "corp"
    assert canonical_key("") == ""


def test_different_companies_are_not_merged_by_suffix_stripping(fake_backend):
    backend = fake_backend({"is_same": False})
    resolver = EntityResolver(embedder=_NameEmbedder(), backend=backend)
    new, aliases = resolver.resolve(
        [_entity("Ford Motor Company", type="Organization")],
        [{"name": "Apple", "type": "Organization", "description": ""}],
    )
    assert aliases == {}
    assert [e.name for e in new] == ["Ford Motor Company"]


def test_an_unbound_relationship_is_rejected_without_a_round_trip():
    """What a small model writes most often: [:CONNECTS_TO] then r.relation."""
    from pipeline.graph.cypher import is_well_formed

    bad = (
        "MATCH (a:Entity)-[:CONNECTS_TO]->(b:Entity) "
        "RETURN a.name, r.relation, b.name"
    )
    good = (
        "MATCH (a:Entity)-[r:CONNECTS_TO]->(b:Entity) "
        "RETURN a.name, r.relation, b.name"
    )
    assert not is_well_formed(bad)
    assert is_well_formed(good)


def test_the_agent_rejects_the_unbound_form(fake_backend):
    backend = fake_backend(
        {"cypher_query": "MATCH (a:Entity)-[:CONNECTS_TO]->(b:Entity) RETURN r.relation"}
    )
    assert CypherAgent(backend=backend).generate("q", ["seed"]) is None


# --------------------------------------------------------------------------- #
# The lock — Kuzu is single-writer and the lock is process-wide
# --------------------------------------------------------------------------- #


def test_a_read_only_store_refuses_to_write(tmp_path: Path):
    path = str(tmp_path / "kuzu")
    with GraphStore(db_path=path) as writer:
        writer.upsert([_entity("A")], [])

    reader = GraphStore(db_path=path, read_only=True)
    from errors import PersistError

    with pytest.raises(PersistError, match="read-only"):
        reader.upsert([_entity("B")], [])
    reader.close()


def test_a_read_only_store_still_reads(tmp_path: Path):
    path = str(tmp_path / "kuzu")
    with GraphStore(db_path=path) as writer:
        writer.upsert(*CAREER)

    with GraphStore(db_path=path, read_only=True) as reader:
        assert reader.count()["entities"] == 4
        assert reader.neighbours(["Doug Field"], hops=1)


def test_closing_releases_the_lock_for_another_writer(tmp_path: Path):
    """A worker holding the write lock would block every reader and writer."""
    path = str(tmp_path / "kuzu")
    with GraphStore(db_path=path) as first:
        first.upsert([_entity("A")], [])

    with GraphStore(db_path=path) as second:
        second.upsert([_entity("B")], [])
        assert second.count()["entities"] == 2


def test_the_builder_releases_the_store_it_opened(tmp_path: Path, fake_backend, monkeypatch):
    """Ingest must not hold the lock past the document it was writing."""
    from pipeline.graph.builder import build_graph

    monkeypatch.setattr("config.config.KUZU_DB_PATH", str(tmp_path / "kuzu"))
    backend = fake_backend(
        {
            "entities": [{"name": "ACME", "type": "Organization", "description": "a firm"}],
            "relationships": [],
        }
    )
    build_graph(
        "text",
        source_url="u",
        extractor=GraphExtractor(backend=backend),
        resolver=EntityResolver(embedder=_NameEmbedder(), verify_with_model=False),
        entity_index=None,
    )
    # If the lock were still held, opening read-write here would raise.
    with GraphStore(db_path=str(tmp_path / "kuzu")) as store:
        assert store.count()["entities"] == 1


# --------------------------------------------------------------------------- #
# Windowing — the whole document reaches the graph, not just its front
# --------------------------------------------------------------------------- #


def test_a_long_document_is_windowed_not_truncated(fake_backend, monkeypatch):
    """The earlier version passed text[:MAX_CHUNK_SIZE] to one call, so a long
    PDF produced a graph of its first few pages and said nothing about the rest.
    """
    from config import config
    from pipeline.graph.cache import GraphCache

    monkeypatch.setattr(config, "MAX_CHUNK_SIZE", 100)
    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "llm")

    backend = fake_backend(
        {"entities": [{"name": "ACME", "type": "Organization", "description": "a firm"}],
         "relationships": []}
    )
    GraphExtractor(backend=backend, cache=GraphCache()).extract("word " * 200)
    assert backend.calls > 1


def test_windowing_is_capped(fake_backend, monkeypatch):
    from config import config
    from pipeline.graph.cache import GraphCache

    monkeypatch.setattr(config, "MAX_CHUNK_SIZE", 50)
    monkeypatch.setattr(config, "GRAPH_MAX_WINDOWS", 3)
    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "llm")

    backend = fake_backend({"entities": [], "relationships": []})
    GraphExtractor(backend=backend, cache=GraphCache()).extract("word " * 500)
    assert backend.calls == 3


def test_entities_are_merged_across_windows(fake_backend, monkeypatch):
    """The same company named in three windows is one node, not three."""
    from config import config
    from pipeline.graph.cache import GraphCache

    monkeypatch.setattr(config, "MAX_CHUNK_SIZE", 100)
    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "llm")

    backend = fake_backend(
        {"entities": [{"name": "ACME", "type": "Organization", "description": "a firm"}],
         "relationships": []}
    )
    result = GraphExtractor(backend=backend, cache=GraphCache()).extract("word " * 200)
    assert [e.name for e in result.entities] == ["ACME"]


def test_a_short_document_is_still_one_call(fake_backend, monkeypatch):
    from config import config
    from pipeline.graph.cache import GraphCache

    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "llm")
    backend = fake_backend({"entities": [], "relationships": []})
    GraphExtractor(backend=backend, cache=GraphCache()).extract("a short sentence")
    assert backend.calls == 1


# --------------------------------------------------------------------------- #
# Reconstructing a variable-length path
# --------------------------------------------------------------------------- #

#: Two hops in a straight line, so that "the middle node" and "the endpoints"
#: are different answers and a shifted expansion cannot pass by luck.
CHAIN = (
    [_entity("Northwind Logistics", type="Organization"),
     _entity("Fabrikam Freight", type="Organization"),
     _entity("Rotterdam", type="Place")],
    [_rel("Northwind Logistics", "Fabrikam Freight", "ACQUIRED"),
     _rel("Fabrikam Freight", "Rotterdam", "LOCATED_IN")],
)

CHAIN_EDGES = {
    ("Northwind Logistics", "ACQUIRED", "Fabrikam Freight"),
    ("Fabrikam Freight", "LOCATED_IN", "Rotterdam"),
}


def _edges(triples) -> set[tuple[str, str, str]]:
    return {(t.source, t.relation, t.target) for t in triples}


def test_a_bound_path_gives_the_hops_not_a_self_loop(graph: GraphStore):
    """``RETURN a.name, p, b.name`` puts *both* endpoints in ``_nodes``.

    Expanded on the intermediate-node rule, a single hop became
    ``[a, a, b, b]`` — a self-loop the graph does not contain, followed by the
    edge read backwards.
    """
    graph.upsert(*CHAIN)
    triples = graph.execute_read(
        "MATCH p = (a:Entity)-[:CONNECTS_TO*1..1]-(b:Entity) "
        "WHERE a.name = 'Northwind Logistics' AND b.name = 'Fabrikam Freight' "
        "RETURN a.name, p, b.name"
    )
    assert _edges(triples) == {("Northwind Logistics", "ACQUIRED", "Fabrikam Freight")}


def test_a_longer_bound_path_invents_no_edges(graph: GraphStore):
    """``*1..3`` between adjacent nodes also walks the detour out and back.

    Those repeats are redundant rather than wrong, so they are kept and
    deduplicated; what must never appear is an edge the graph does not hold.
    """
    graph.upsert(*CHAIN)
    triples = graph.execute_read(
        "MATCH p = (a:Entity)-[:CONNECTS_TO*1..3]-(b:Entity) "
        "WHERE a.name = 'Northwind Logistics' AND b.name = 'Fabrikam Freight' "
        "RETURN a.name, p, b.name"
    )
    assert ("Northwind Logistics", "ACQUIRED", "Fabrikam Freight") in _edges(triples)
    assert _edges(triples) <= CHAIN_EDGES


def test_a_bound_path_keeps_the_middle_node_of_two_hops(graph: GraphStore):
    graph.upsert(*CHAIN)
    triples = graph.execute_read(
        "MATCH p = (a:Entity)-[:CONNECTS_TO*1..3]-(b:Entity) "
        "WHERE a.name = 'Northwind Logistics' AND b.name = 'Rotterdam' "
        "RETURN a.name, p, b.name"
    )
    assert _edges(triples) == CHAIN_EDGES


def test_walking_a_path_backwards_does_not_reverse_the_facts(graph: GraphStore):
    """An undirected match may cross an edge against the way it points.

    Reading direction off the walk turns "Fabrikam Freight is located in
    Rotterdam" into the reverse, which is worse than returning nothing.
    """
    graph.upsert(*CHAIN)
    triples = graph.execute_read(
        "MATCH p = (a:Entity)-[:CONNECTS_TO*1..3]-(b:Entity) "
        "WHERE a.name = 'Rotterdam' AND b.name = 'Northwind Logistics' "
        "RETURN a.name, p, b.name"
    )
    assert _edges(triples) == CHAIN_EDGES


def test_a_bound_recursive_relationship_still_expands(graph: GraphStore):
    """The other convention: ``_nodes`` holds only what is between the ends.

    :meth:`GraphStore.neighbours` runs exactly this shape, so it is the half
    that must not break while the path-variable half is fixed.
    """
    graph.upsert(*CHAIN)
    triples = graph.execute_read(
        "MATCH (a:Entity)-[r:CONNECTS_TO*1..3]->(b:Entity) "
        "WHERE a.name = 'Northwind Logistics' AND b.name = 'Rotterdam' "
        "RETURN a.name, r, b.name"
    )
    assert _edges(triples) == CHAIN_EDGES


def test_path_returns_every_hop_between_two_entities(graph: GraphStore):
    graph.upsert(*CHAIN)
    assert _edges(graph.path("Northwind Logistics", "Rotterdam")) == CHAIN_EDGES


def test_a_path_shape_that_matches_neither_convention_is_dropped(graph: GraphStore):
    """Node and hop counts that fit no rule mean the row cannot be read.

    Guessing an alignment here would emit edges nobody stored.
    """
    from pipeline.graph.store import _path_triples

    nonsense = {"_nodes": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                "_rels": [{"relation": "X"}, {"relation": "Y"}, {"relation": "Z"}]}
    assert _path_triples(nonsense, start="A", end="C") == []
