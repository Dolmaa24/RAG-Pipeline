"""Cross-encoder reranking, over a shortlist only.

Bi-encoders — what the index uses — embed the query and the document separately
and compare the two vectors. That is what makes search fast, and also what
limits it: nothing ever compares the query and the document *together*. A cross
encoder does, scoring the pair jointly, which is markedly more accurate and far
too slow to run over a corpus.

So it runs over the top twenty. Reranking a hundred candidates costs five times
as much for a difference concentrated in the first few, and on an 8 GB machine
this is another model resident alongside BGE, Kuzu, LanceDB and possibly a local
LLM — which is why it is off unless asked for.
"""

from __future__ import annotations

import threading
from typing import Optional

from config import config
from errors import MissingDependency
from observability import get_logger, metrics

from pipeline.retrieve.hybrid import ScoredChunk

log = get_logger("retrieve.rerank")

_model = None
_model_name: Optional[str] = None
_lock = threading.Lock()


def get_reranker(model_name: Optional[str] = None):
    """The process-wide cross-encoder, constructed once."""
    global _model, _model_name
    name = model_name or config.RETRIEVE_RERANK_MODEL

    with _lock:
        if _model is not None and _model_name == name:
            return _model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise MissingDependency("sentence-transformers", "reranking") from exc

        from pipeline.embed.dense import select_device

        _model = CrossEncoder(name, device=select_device())
        _model_name = name
        log.info("retrieve.rerank.loaded", model=name)
        return _model


def reset() -> None:
    global _model, _model_name
    with _lock:
        _model, _model_name = None, None


def rerank(
    query: str,
    chunks: list[ScoredChunk],
    *,
    limit: Optional[int] = None,
    candidates: Optional[int] = None,
    model=None,
) -> list[ScoredChunk]:
    """Rescore the top candidates against the query, jointly.

    Anything past the shortlist keeps its fused rank and stays below it. That
    ordering is deliberate: an unreranked candidate has not been shown to be
    worse than a reranked one, only that it was not worth the cost to check.
    """
    if not chunks or not query.strip():
        return chunks

    shortlist_size = candidates or config.RETRIEVE_RERANK_CANDIDATES
    shortlist = chunks[:shortlist_size]
    remainder = chunks[shortlist_size:]

    try:
        encoder = model or get_reranker()
        with metrics.timer("retrieve.rerank"):
            scores = encoder.predict([(query, chunk.document) for chunk in shortlist])
    except Exception as exc:
        log.warning("retrieve.rerank.failed", error=repr(exc))
        metrics.incr("retrieve.rerank.failed")
        return chunks[: limit or len(chunks)]

    for chunk, score in zip(shortlist, scores):
        chunk.score = float(score)
        chunk.found_by.add("rerank")

    ordered = sorted(shortlist, key=lambda c: c.score, reverse=True) + remainder
    return ordered[: limit or len(ordered)]


__all__ = ["get_reranker", "rerank", "reset"]
