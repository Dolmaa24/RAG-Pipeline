"""Backend selection.

``LLM_BACKEND=auto`` means "the fastest backend that is allowed and reachable":
Groq when a key is configured and the job is not marked local-only, Ollama
otherwise. Naming a backend explicitly is a hard choice — if it is unreachable
the job fails rather than quietly falling back to somewhere the content was not
supposed to go.
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


def get_backend(name: Optional[str] = None, *, local_only: bool = False) -> LLMBackend:
    """Return a ready backend.

    ``local_only`` is the per-job override of the global ``LOCAL_ONLY`` setting:
    a caller submitting sensitive content can force the local model without
    reconfiguring the process.
    """
    choice = (name or config.LLM_BACKEND).lower()
    if local_only or config.LOCAL_ONLY:
        if choice == "groq":
            raise ExtractError("this job is local-only; the Groq backend cannot be used for it")
        choice = "ollama"

    with _lock:
        cached = _cache.get(choice)
        if cached is not None:
            return cached

        backend = _build(choice)
        _cache[choice] = backend
        return backend


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
    "GroqBackend",
    "LLMBackend",
    "LLMResponse",
    "OllamaBackend",
    "build_prompt",
    "extract_json_object",
    "get_backend",
    "reset",
    "status",
]
