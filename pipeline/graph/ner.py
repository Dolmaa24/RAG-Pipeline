"""Entity extraction without generating anything.

Naming the entities in a passage is a *classification* problem — which spans are
entities, and of what kind. Asking a decoder to answer it by autoregressively
emitting JSON is the slowest available way to get a label, and it is why graph
extraction was the most expensive thing in this pipeline.

GLiNER scores spans instead. It is an encoder under 500M parameters, it runs on
the CPU, and on the same paragraph it was measured here at **166 ms against the
local 3B's ~10 s** — while finding the entity that model missed.

What it does not do is relationships. So the graph path uses it for the half it
is good at and keeps a model for the half it is not, which also shrinks that
call: given the entity list, the model only has to emit edges, not re-describe
every node it just found.

Descriptions come from the text rather than from a model. GLiNER returns
character offsets, so the sentence containing the first mention is free to take
and is a real quote — better provenance than a paraphrase, and it cannot
hallucinate.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from config import config
from errors import MissingDependency
from observability import get_logger, metrics

from pipeline.graph.schema import Entity

log = get_logger("graph.ner")

#: What to look for. GLiNER is zero-shot, so these are just words — changing
#: them changes what it finds, with no retraining.
DEFAULT_LABELS = (
    "person",
    "organization",
    "location",
    "product",
    "project",
    "technology",
    "event",
)

#: GLiNER's lowercase labels, mapped to the capitalised types the graph uses.
_LABEL_TYPES = {
    "person": "Person",
    "organization": "Organization",
    "location": "Location",
    "product": "Product",
    "project": "Project",
    "technology": "Technology",
    "event": "Event",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_model = None
_model_name: Optional[str] = None
_lock = threading.Lock()


def get_model(model_name: Optional[str] = None):
    """The process-wide GLiNER model, loaded once.

    Pinned to the device :func:`pipeline.embed.dense.select_device` chooses,
    which is the CPU — this is torch, and Metal has been measured killing both
    the prefork worker and the uvicorn process. At 166 ms a paragraph, the GPU
    is not worth another way to lose the process.
    """
    global _model, _model_name
    name = model_name or config.GRAPH_NER_MODEL

    with _lock:
        if _model is not None and _model_name == name:
            return _model

        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise MissingDependency("gliner", "GLiNER entity extraction") from exc

        from pipeline.embed.dense import select_device

        device = select_device() or "cpu"
        model = GLiNER.from_pretrained(name).to(device)
        model.eval()

        log.info("graph.ner.loaded", model=name, device=device)
        _model, _model_name = model, name
        return model


def reset() -> None:
    """Drop the model. Used by tests and to reclaim the memory."""
    global _model, _model_name
    with _lock:
        _model, _model_name = None, None


def _sentence_at(text: str, offset: int) -> str:
    """The sentence containing a character offset, as the entity description."""
    if not text:
        return ""
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.end() > offset:
            break
        start = match.end()
    end = len(text)
    match = _SENTENCE_END.search(text, offset)
    if match:
        end = match.start()
    return " ".join(text[start:end].split())[:300]


def extract_entities(
    text: str,
    *,
    labels: Optional[tuple[str, ...]] = None,
    threshold: Optional[float] = None,
    source_url: str = "",
    content_hash: str = "",
    model=None,
) -> list[Entity]:
    """Named entities in the text, one per distinct name.

    GLiNER returns a span per *mention*, so a name appearing three times comes
    back three times. They are collapsed here, keeping the highest-scoring
    mention's label and the sentence around the first one.
    """
    if not text or not text.strip():
        return []

    engine = model or get_model()
    wanted = list(labels or config.graph_ner_labels)
    cutoff = config.GRAPH_NER_THRESHOLD if threshold is None else threshold

    with metrics.timer("graph.ner"):
        spans: list[dict[str, Any]] = engine.predict_entities(
            text, wanted, threshold=cutoff
        )

    best: dict[str, dict[str, Any]] = {}
    for span in spans:
        name = str(span.get("text", "")).strip()
        if not name:
            continue
        key = name.lower()
        current = best.get(key)
        if current is None or float(span.get("score", 0.0)) > float(current.get("score", 0.0)):
            # Keep the first mention's offset even when a later one scores
            # higher: the description should come from where it was introduced.
            span = dict(span)
            if current is not None:
                span["start"] = current.get("start", span.get("start", 0))
            best[key] = span

    entities = [
        Entity(
            name=str(span["text"]).strip(),
            type=_LABEL_TYPES.get(str(span.get("label", "")).lower(), "Unknown"),
            description=_sentence_at(text, int(span.get("start", 0))),
            source_url=source_url,
            content_hash=content_hash,
        )
        for span in best.values()
    ]

    log.info("graph.ner.extracted", entities=len(entities), spans=len(spans))
    metrics.incr("graph.ner.entities", len(entities))
    return entities


def preload() -> None:
    get_model()


__all__ = [
    "DEFAULT_LABELS",
    "extract_entities",
    "get_model",
    "preload",
    "reset",
]
