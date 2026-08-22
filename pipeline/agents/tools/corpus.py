"""Read-only tools: everything answerable from what is already indexed.

These are always available. Nothing here reaches the network or changes stored
state, so a run that never opts into anything else can still do the whole job
of finding and citing evidence.

Every handler is a thin adapter. The retrieval work is done by
:func:`pipeline.retrieve.orchestrator.retrieve` and friends exactly as it is for
the HTTP API — a tool call and a ``POST /api/v1/search`` reach the same code, so
they cannot drift in behaviour or in quality.
"""

from __future__ import annotations

from typing import Any

from observability import get_logger

from pipeline.agents.tools.models import (
    AnswerArgs,
    AnswerResult,
    ChunkArgs,
    ChunkResult,
    GraphResult,
    NeighborsArgs,
    Passage,
    PathArgs,
    ProfileArgs,
    ProfileResult,
    RelationsArgs,
    SearchArgs,
    SearchResult,
)
from pipeline.agents.tools.registry import Effect, tool

log = get_logger("agents.tools.corpus")


@tool(
    name="corpus_profile",
    effect=Effect.READ,
    cost_ms=5,
    description=(
        "What the corpus contains: how many documents, whether a knowledge "
        "graph exists, and which filter values are actually present. Call this "
        "before filtering a search — a filter naming a value the corpus does "
        "not hold returns nothing, and looks the same as a question it cannot "
        "answer."
    ),
)
def corpus_profile(args: ProfileArgs) -> ProfileResult:
    from pipeline.store.schema import FILTER_FIELDS

    try:
        from pipeline.store.lance import LanceStore

        store = LanceStore()
    except Exception as exc:
        return ProfileResult(available=False, error=str(exc))

    if store.table is None:
        return ProfileResult(available=False, error="nothing has been indexed yet")

    filters = {field: store.distinct(field, limit=25) for field in FILTER_FIELDS}
    counts = _graph_counts()

    return ProfileResult(
        available=True,
        chunks=store.count(),
        documents=len(filters.get("source", [])),
        graph_entities=counts["entities"],
        graph_relationships=counts["relationships"],
        relation_kinds=counts.get("kinds") or [],
        filters=filters,
    )


def _graph_counts() -> dict[str, Any]:
    """Graph size and shape, or zeroes. An absent graph is normal, not an error.

    The relationship *kinds* are gathered in the same connection as the counts:
    a model cannot ask graph_relations for a kind it does not know exists, and
    guessing one costs a turn to find out it was wrong.
    """
    from pipeline.graph.store import graph_exists

    empty: dict[str, Any] = {"entities": 0, "relationships": 0, "kinds": []}
    if not graph_exists():
        return empty
    try:
        from pipeline.graph.store import GraphStore

        with GraphStore(read_only=True) as store:
            counts = dict(store.count())
            counts["kinds"] = store.relation_kinds()
            return counts
    except Exception as exc:
        log.warning("agents.tools.graph_count_failed", error=repr(exc))
        return empty


@tool(
    name="search_corpus",
    effect=Effect.READ,
    cost_ms=600,
    description=(
        "Search the indexed documents. Runs vector and keyword search together "
        "and returns numbered passages with their source and a chunk_id. "
        "Filters narrow the search before it runs; call corpus_profile first to "
        "see which values exist. If a filtered search returns nothing, retry "
        "without the filter before concluding the corpus cannot answer."
    ),
)
def search_corpus(args: SearchArgs) -> SearchResult:
    from pipeline.retrieve.orchestrator import retrieve

    result = retrieve(
        args.query,
        filters=args.to_filter(),
        limit=args.limit,
        fusion=args.fusion,
        alpha=args.alpha,
        use_graph=args.use_graph,
    )

    return SearchResult(
        query=args.query,
        passages=[
            Passage(
                chunk_id=str(chunk.id or ""),
                text=chunk.document,
                source=str(chunk.metadata.get("source", "")),
                score=round(float(chunk.score), 6),
                doc_type=str(chunk.metadata.get("doc_type", "")),
            )
            for chunk in result.chunks
        ],
        triples=[triple.render() for triple in result.triples],
        sub_queries=result.plan.sub_queries if result.plan else [],
        warnings=result.warnings,
    )


@tool(
    name="answer_from_corpus",
    effect=Effect.READ,
    cost_ms=4000,
    description=(
        "Answer a question from the corpus in one step: retrieves, then writes "
        "a grounded answer with numbered citations. Slower than search_corpus "
        "because it calls a language model. Prefer it for a self-contained "
        "question; prefer search_corpus when you need the passages themselves "
        "in order to reason across several of them."
    ),
    max_chars=3000,
)
def answer_from_corpus(args: AnswerArgs) -> AnswerResult:
    from pipeline.retrieve.answer import answer_question

    reply = answer_question(
        args.question,
        filters=args.to_filter(),
        limit=args.limit,
        use_graph=args.use_graph,
    )

    return AnswerResult(
        question=args.question,
        answer=reply.answer,
        sufficient=reply.sufficient,
        cited=list(reply.cited),
        sources=[source.origin for source in reply.sources],
    )


@tool(
    name="graph_neighbors",
    effect=Effect.READ,
    cost_ms=60,
    description=(
        "Relationships involving one entity, from the knowledge graph. Use it "
        "when a question is about how things relate — who acquired whom, who "
        "works where — rather than about what a document says. Returns nothing "
        "if no graph has been built."
    ),
)
def graph_neighbors(args: NeighborsArgs) -> GraphResult:
    from pipeline.graph.store import graph_exists

    if not graph_exists():
        return GraphResult(available=False)

    from pipeline.graph.entities import EntityIndex
    from pipeline.graph.store import GraphStore

    # Resolve the name the model typed to names the graph actually holds:
    # "Acme" should find "Acme Corporation" rather than nothing.
    seeds = EntityIndex().seeds(args.entity) or [args.entity]

    with GraphStore(read_only=True) as store:
        triples = store.neighbours(seeds, hops=args.hops, limit=args.limit)

    return GraphResult(seeds=seeds, edges=[t.render() for t in triples])


@tool(
    name="graph_path",
    effect=Effect.READ,
    cost_ms=150,
    description=(
        "How two entities are connected, as the chain of relationships between "
        "them. Use it for questions of the form 'what links A to B'. Returns "
        "nothing when they are unconnected or no graph has been built."
    ),
)
def graph_path(args: PathArgs) -> GraphResult:
    from pipeline.graph.store import graph_exists

    if not graph_exists():
        return GraphResult(available=False)

    from pipeline.graph.entities import EntityIndex
    from pipeline.graph.store import GraphStore

    index = EntityIndex()
    starts = index.seeds(args.start) or [args.start]
    ends = index.seeds(args.end) or [args.end]

    edges: list[str] = []
    with GraphStore(read_only=True) as store:
        for start in starts[:3]:
            for end in ends[:3]:
                if start == end:
                    continue
                for triple in store.path(
                    start, end, max_hops=args.max_hops, limit=args.limit
                ):
                    rendered = triple.render()
                    if rendered not in edges:
                        edges.append(rendered)

    return GraphResult(seeds=[*starts[:3], *ends[:3]], edges=edges[: args.limit])


@tool(
    name="graph_relations",
    effect=Effect.READ,
    cost_ms=80,
    description=(
        "List relationships of one kind across the whole graph — every "
        "acquisition, every location, every appointment. Use it when the "
        "question asks which or how many of something there are, rather than "
        "about one named thing: graph_neighbors and graph_path both need an "
        "entity you already know, and a question like 'which acquisitions are "
        "described' names none."
    ),
)
def graph_relations(args: RelationsArgs) -> GraphResult:
    from pipeline.graph.store import GraphStore, graph_exists

    if not graph_exists():
        return GraphResult(available=False)

    with GraphStore(read_only=True) as store:
        triples = store.relations(args.relation or None, limit=args.limit)
        if not triples and args.relation:
            # Naming a kind the graph does not hold looks identical to a graph
            # with nothing in it. Say which kinds exist instead.
            kinds = store.relation_kinds()
            return GraphResult(
                seeds=kinds,
                edges=[],
            )

    return GraphResult(seeds=[args.relation] if args.relation else [],
                       edges=[t.render() for t in triples])


@tool(
    name="fetch_chunk",
    effect=Effect.READ,
    cost_ms=5,
    description=(
        "The full text of one chunk by its chunk_id, as returned by "
        "search_corpus. Use it to check a claim against its source, or to read "
        "a passage that search returned truncated."
    ),
    max_chars=4000,
)
def fetch_chunk(args: ChunkArgs) -> ChunkResult:
    from pipeline.store.lance import LanceStore
    from pipeline.store.schema import FILTER_FIELDS, TEXT_FIELD

    row = LanceStore().get(args.chunk_id)
    if row is None:
        return ChunkResult(chunk_id=args.chunk_id, found=False)

    return ChunkResult(
        chunk_id=args.chunk_id,
        found=True,
        text=str(row.get(TEXT_FIELD, "")),
        source=str(row.get("source", "")),
        metadata={
            field: row[field] for field in FILTER_FIELDS if row.get(field)
        },
    )


__all__ = [
    "answer_from_corpus",
    "corpus_profile",
    "fetch_chunk",
    "graph_neighbors",
    "graph_path",
    "search_corpus",
]
