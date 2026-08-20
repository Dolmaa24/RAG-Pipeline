"""Hybrid retrieval: dense and BM25, fused.

The two legs fail in opposite directions, which is the whole reason to run both.
Dense search finds "how do I get my money back" in a document that says
"refunds" and never says money; it also confidently returns something vaguely
topical when the answer is not there at all. BM25 finds the exact part number,
the surname, the error code — the tokens where being approximately right is
being wrong — and finds nothing at all when the wording differs.

Fusion happens here rather than in the store because there are two dimensions to
fuse across, not one: the two legs, and the several sub-queries a decomposed
question produced. One implementation covers both, and it can be tested without
a database.

**RRF** is the default. It uses only rank, so it does not care that cosine
similarity and BM25 relevance are unrelated scales — which is exactly the thing
that makes a weighted sum of raw scores misbehave. **alpha** weights normalised
scores instead, and is worth tuning once you have a corpus and some judgements
to tune against.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

from pipeline.retrieve.filters import MetadataFilter

log = get_logger("retrieve.hybrid")


@dataclass
class ScoredChunk:
    """One retrieved chunk, with how it was found."""

    id: str
    document: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Which legs produced it, e.g. {"dense", "bm25"}. A chunk found by both is
    #: usually the best evidence there is, and this is how you can tell.
    found_by: set[str] = field(default_factory=set)
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document": self.document,
            "score": round(self.score, 6),
            "found_by": sorted(self.found_by),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "metadata": self.metadata,
        }


_METADATA_KEYS = (
    "source",
    "doc_type",
    "department",
    "date",
    "author",
    "region",
    "permission_level",
    "language",
    "page_no",
    "section_name",
    "chunk_strategy",
    "embedding_model",
    "content_hash",
    "extraction_tier",
)


def _as_chunk(row: dict, leg: str, rank: int) -> ScoredChunk:
    chunk = ScoredChunk(
        id=str(row.get("id", "")),
        document=str(row.get("document", "")),
        score=float(row.get("score", 0.0)),
        metadata={key: row[key] for key in _METADATA_KEYS if key in row},
        found_by={leg},
    )
    if leg == "dense":
        chunk.dense_rank = rank
    else:
        chunk.bm25_rank = rank
    return chunk


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], *, k: Optional[int] = None
) -> list[ScoredChunk]:
    """Combine ranked lists by rank alone.

    Each list contributes ``1 / (k + rank)`` to every document it ranks. The
    constant damps the top of each list so one leg cannot dominate purely by
    being confident, and because only order is used, lists scored on
    incomparable scales combine correctly.
    """
    constant = k if k is not None else config.RETRIEVE_RRF_K
    merged: dict[str, ScoredChunk] = {}
    totals: dict[str, float] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking):
            if not chunk.id:
                continue
            existing = merged.get(chunk.id)
            if existing is None:
                merged[chunk.id] = chunk
                totals[chunk.id] = 0.0
            else:
                _absorb(existing, chunk)
            totals[chunk.id] += 1.0 / (constant + position + 1)

    for chunk_id, total in totals.items():
        merged[chunk_id].score = total
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


def alpha_fusion(
    dense: list[ScoredChunk], lexical: list[ScoredChunk], *, alpha: Optional[float] = None
) -> list[ScoredChunk]:
    """Weighted sum of normalised scores: ``alpha`` dense, ``1 - alpha`` BM25.

    The store normalises each leg to 0..1 before this sees it, without which the
    sum would be meaningless. A document missing from one leg scores zero there
    rather than being dropped, so a strong lexical-only match still surfaces.
    """
    weight = config.RETRIEVE_ALPHA if alpha is None else alpha
    weight = min(1.0, max(0.0, weight))

    merged: dict[str, ScoredChunk] = {}
    scores: dict[str, float] = {}

    for chunks, factor in ((dense, weight), (lexical, 1.0 - weight)):
        for chunk in chunks:
            if not chunk.id:
                continue
            existing = merged.get(chunk.id)
            if existing is None:
                merged[chunk.id] = chunk
                scores[chunk.id] = 0.0
            else:
                _absorb(existing, chunk)
            scores[chunk.id] += factor * chunk.score

    for chunk_id, total in scores.items():
        merged[chunk_id].score = total
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


def _absorb(target: ScoredChunk, other: ScoredChunk) -> None:
    """Fold a duplicate hit into the one already kept."""
    target.found_by |= other.found_by
    if other.dense_rank is not None:
        target.dense_rank = (
            other.dense_rank if target.dense_rank is None else min(target.dense_rank, other.dense_rank)
        )
    if other.bm25_rank is not None:
        target.bm25_rank = (
            other.bm25_rank if target.bm25_rank is None else min(target.bm25_rank, other.bm25_rank)
        )


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #


class HybridRetriever:
    """Runs both legs for every query and fuses the lot."""

    def __init__(self, store=None, embedder=None) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def store(self):
        if self._store is None:
            from pipeline.store.lance import LanceStore

            self._store = LanceStore()
        return self._store

    @property
    def embedder(self):
        if self._embedder is None:
            from pipeline.embed.dense import get_dense_embedder

            self._embedder = get_dense_embedder()
        return self._embedder

    def retrieve(
        self,
        queries: list[str],
        *,
        filters: Optional[MetadataFilter] = None,
        limit: Optional[int] = None,
        candidates: Optional[int] = None,
        fusion: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> list[ScoredChunk]:
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return []

        top_k = limit or config.RETRIEVE_TOP_K
        per_leg = candidates or config.RETRIEVE_CANDIDATES
        where = (filters or MetadataFilter()).compile()
        mode = fusion or config.RETRIEVE_FUSION

        # One encode() for every sub-query. N separate calls would pay the
        # per-call overhead N times for no reason; the model batches natively.
        with metrics.timer("retrieve.embed_queries"):
            vectors = self.embedder.embed_documents(queries)

        dense_lists: list[list[ScoredChunk]] = []
        lexical_lists: list[list[ScoredChunk]] = []

        # The legs are independent and both wait on something — the vector scan
        # and the inverted index. Running them serially adds their latencies.
        with ThreadPoolExecutor(max_workers=min(8, 2 * len(queries))) as pool:
            dense_futures = [
                pool.submit(self.store.search_dense, vector, limit=per_leg, where=where)
                for vector in vectors
            ]
            lexical_futures = [
                pool.submit(self.store.search_fts, query, limit=per_leg, where=where)
                for query in queries
            ]

            for future in dense_futures:
                rows = _safe(future, "dense")
                dense_lists.append([_as_chunk(r, "dense", i) for i, r in enumerate(rows)])
            for future in lexical_futures:
                rows = _safe(future, "bm25")
                lexical_lists.append([_as_chunk(r, "bm25", i) for i, r in enumerate(rows)])

        if mode == "alpha":
            fused = alpha_fusion(
                _flatten_best(dense_lists), _flatten_best(lexical_lists), alpha=alpha
            )
        else:
            fused = reciprocal_rank_fusion([*dense_lists, *lexical_lists])

        log.info(
            "retrieve.hybrid.done",
            queries=len(queries),
            fusion=mode,
            dense=sum(len(x) for x in dense_lists),
            bm25=sum(len(x) for x in lexical_lists),
            fused=len(fused),
            filtered=bool(where),
        )
        return fused[:top_k]


def _flatten_best(lists: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    """Collapse per-sub-query lists, keeping each chunk's best score."""
    best: dict[str, ScoredChunk] = {}
    for ranking in lists:
        for chunk in ranking:
            existing = best.get(chunk.id)
            if existing is None:
                best[chunk.id] = chunk
            else:
                _absorb(existing, chunk)
                existing.score = max(existing.score, chunk.score)
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def _safe(future, leg: str) -> list[dict]:
    """One failing leg degrades the result; it does not lose the query."""
    try:
        return future.result()
    except Exception as exc:
        log.warning("retrieve.leg_failed", leg=leg, error=repr(exc))
        metrics.incr(f"retrieve.{leg}.failed")
        return []


__all__ = [
    "HybridRetriever",
    "ScoredChunk",
    "alpha_fusion",
    "reciprocal_rank_fusion",
]
