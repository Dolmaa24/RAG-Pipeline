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
#: A loop choosing tools. Its own role because the job is different enough that
#: the best model for it is a different model — measured, not assumed.
AGENT = "agent"
#: Checking a drafted answer against its sources. Its own role because it is a
#: third job that a third model wins: it wants scepticism, where the agent role
#: wants a model that knows when to stop.
VERIFY = "verify"
#: Authoring CSS selectors for a domain, once. The strongest case in this
#: pipeline for a hosted model, and the opposite of the volume problem that
#: makes one unaffordable for extraction: it costs one call per *domain*, and
#: its output is replayed free on every page after.
#:
#: The job also asks for something a 3B model is measurably bad at. On
#: books.toscrape.com, qwen2.5:3b answered a three-field schema with
#: ``td[content='£51.77']`` -- an attribute that does not exist in the markup,
#: on an element the skeleton had already shown it as ``p.price_color``. Two of
#: three rules were dead, the spec fell below MIN_FILL_RATE, and every page of
#: the site then paid tier 3. openai/gpt-oss-120b learned two of the three from
#: the same skeleton, which is enough to store and replay.
SELECTOR = "selector"
#: Writing source code. Its own role, and the only one in this project whose
#: setting defaults to the hosted model rather than to LLM_BACKEND.
#:
#: The reason is capacity, not preference. A build is several files that have to
#: import each other and agree on their types, and llama3.2:3b on an 8 GB
#: machine does not hold a contract across five generations -- the failure is
#: not a worse file, it is four modules that do not compose, which makes the
#: whole build worthless rather than weaker. Falling back to local is still
#: correct when the hosted model is unreachable; the caller is warned that the
#: output should be read as a sketch.
CODE = "code"


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
        # Only a *preference* falls back. An explicitly named backend still
        # fails loudly, because naming one is a decision about where the
        # content may go, not a preference about speed. Selector learning is a
        # preference for the same reason interactive is: an unreachable
        # hosted model should make the pipeline slower and worse at tier 2,
        # not stop it extracting.
        if explicit or role not in (INTERACTIVE, SELECTOR, CODE) or fallback == choice:
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
    if role == AGENT and config.LLM_AGENT_BACKEND:
        return config.LLM_AGENT_BACKEND
    if role == VERIFY and config.LLM_VERIFY_BACKEND:
        return config.LLM_VERIFY_BACKEND
    if role == SELECTOR and config.LLM_SELECTOR_BACKEND:
        return config.LLM_SELECTOR_BACKEND
    if role == CODE and config.LLM_CODE_BACKEND:
        return config.LLM_CODE_BACKEND
    return config.LLM_BACKEND


def get_agent_backend(*, local_only: bool = False, role: str = AGENT):
    """A backend ready to be given tools, for the loop.

    ``role`` chooses which setting decides the backend. It defaults to AGENT,
    which is every existing caller; a build passes CODE, because writing five
    files that import each other and picking which corpus tool to call next are
    different enough jobs to deserve different models.

    Two things happen here that :func:`get_backend` does not do.

    The **model** differs from extraction's. Picking a tool and extracting a
    schema are different jobs, and the benchmark says different models win them:
    ``qwen2.5:3b`` extracts better and never stops calling tools, which makes it
    unusable as a supervisor. A configured agent model that is not pulled falls
    back to the extraction model with a warning rather than failing — the same
    principle as an unavailable interactive preference making search slower
    instead of broken.

    And a backend that cannot tool-call is **wrapped rather than rejected**, so
    the loop runs on every model in the stack, well on some and poorly on
    others, rather than only on the ones with a tools array.
    """
    from .toolshim import shim_if_needed

    backend = get_backend(local_only=local_only, role=role)

    wanted = config.AGENT_MODEL_NAME
    if wanted and backend.name == "ollama" and backend.model != wanted:
        candidate = OllamaBackend(model=wanted)
        if candidate.available():
            backend = candidate
        else:
            log.warning(
                "llm.agent_model_missing",
                wanted=wanted,
                using=backend.model,
                hint=f"ollama pull {wanted}",
            )

    return shim_if_needed(backend)


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
    "AGENT",
    "BULK",
    "CODE",
    "SELECTOR",
    "VERIFY",
    "GroqBackend",
    "INTERACTIVE",
    "LLMBackend",
    "LLMResponse",
    "OllamaBackend",
    "build_prompt",
    "extract_json_object",
    "get_agent_backend",
    "get_backend",
    "reset",
    "status",
]
