"""Backend selection.

``LLM_BACKEND=auto`` means "the fastest backend that is allowed and reachable":
Groq when a key is configured and the job is not marked local-only, Ollama
otherwise. Naming a backend explicitly is a hard choice — if it is unreachable
the job fails rather than quietly falling back to somewhere the content was not
supposed to go.

**Calls come in two shapes, and the right backend differs.** Measured on this
machine: a local 3B generates at ~35 tok/s, a hosted model at several hundred.
That says "always use the hosted one" until you look at volume:

* **Interactive** — query understanding, graph traversal. Small prompts, a user
  waiting, a handful of calls a minute. Hosted wins outright: ~0.3s against ~8s.
* **Bulk** — extraction, chunking, graph building. One call per chunk, nobody
  waiting, thousands of tokens each. A hosted free tier's tokens-per-minute cap
  turns this into a queue: at 8K TPM and ~3.1K tokens per chunk, roughly two and
  a half chunks a minute. Local is *faster* for bulk despite being slower per
  call.

So callers declare which they are, and ``LLM_INTERACTIVE_BACKEND`` can point
somewhere different from ``LLM_BACKEND``. An interactive preference that is
unavailable falls back to the bulk choice rather than failing: a missing Groq
key should make search slower, not broken.
"""

from __future__ import annotations

import threading
from typing import Optional

from config import config
from errors import ExtractError
from observability import get_logger

from .base import LLMBackend, LLMResponse, build_prompt, extract_json_object
from .groq import GroqBackend
from .ollama import OllamaBackend

log = get_logger("llm")

_cache: dict[str, LLMBackend] = {}
_lock = threading.Lock()


#: Calls a user is waiting on, as opposed to ingest work nobody is waiting on.
INTERACTIVE = "interactive"
BULK = "bulk"


def get_backend(
    name: Optional[str] = None,
    *,
    local_only: bool = False,
    role: str = BULK,
) -> LLMBackend:
    """Return a ready backend.

    ``local_only`` is the per-job override of the global ``LOCAL_ONLY`` setting:
    a caller submitting sensitive content can force the local model without
    reconfiguring the process.

    ``role`` is ``INTERACTIVE`` for calls a user is waiting on and ``BULK`` for
    ingest. See the module docstring for why that changes the answer.
    """
    explicit = name is not None
    choice = (name or _configured(role)).lower()

    if local_only or config.LOCAL_ONLY:
        if choice == "groq" and explicit:
            raise ExtractError("this job is local-only; the Groq backend cannot be used for it")
        choice = "ollama"

    try:
        return _resolve(choice)
    except ExtractError:
        fallback = config.LLM_BACKEND.lower()
        # Only an *interactive preference* falls back. An explicitly named
        # backend still fails loudly, because naming one is a decision about
        # where the content may go, not a preference about speed.
        if explicit or role != INTERACTIVE or fallback == choice:
            raise
        log.warning("llm.interactive_unavailable", wanted=choice, using=fallback)
        return _resolve(fallback)


def _resolve(choice: str) -> LLMBackend:
    """The cached backend for this choice, built once.

    The lock is held around the cache lookup and the construction, and nothing
    else. An earlier version fell back to another backend from *inside* this
    block by calling back into :func:`get_backend`, which deadlocked on a
    non-reentrant lock the moment the preferred backend was unavailable.
    """
    with _lock:
        cached = _cache.get(choice)
        if cached is not None:
            return cached
        backend = _build(choice)
        _cache[choice] = backend
        return backend


def _configured(role: str) -> str:
    """Which backend setting governs this role."""
    if role == INTERACTIVE and config.LLM_INTERACTIVE_BACKEND:
        return config.LLM_INTERACTIVE_BACKEND
    return config.LLM_BACKEND


def _build(choice: str) -> LLMBackend:
    if choice == "groq":
        backend = GroqBackend()
        if not backend.available():
            raise ExtractError(
                "the Groq backend was requested but is unavailable: set GROQ_API_KEY "
                "and `pip install groq`"
            )
        return backend

    if choice == "ollama":
        return OllamaBackend()

    if choice == "auto":
        groq = GroqBackend()
        if groq.available():
            log.info("llm.auto_selected", backend="groq", model=groq.model)
            return groq
        ollama = OllamaBackend()
        log.info("llm.auto_selected", backend="ollama", model=ollama.model)
        return ollama

    raise ExtractError(f"unknown LLM backend {choice!r}; expected auto, ollama or groq")


def reset() -> None:
    """Drop cached backends. Used by tests and after a config change."""
    with _lock:
        _cache.clear()


def status() -> dict:
    """Reachability of each backend, for /health and the dashboard."""
    ollama = OllamaBackend()
    groq = GroqBackend()
    return {
        "configured": config.LLM_BACKEND,
        "local_only": config.LOCAL_ONLY,
        "ollama": {"model": ollama.model, "host": ollama.host, "available": ollama.available()},
        "groq": {
            "model": groq.model,
            "available": groq.available(),
            "schema_enforced": groq.supports_json_schema,
        },
    }


__all__ = [
    "BULK",
    "GroqBackend",
    "INTERACTIVE",
    "LLMBackend",
    "LLMResponse",
    "OllamaBackend",
    "build_prompt",
    "extract_json_object",
    "get_backend",
    "reset",
    "status",
]
