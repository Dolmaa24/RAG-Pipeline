"""Backend selection: which model answers, and what happens when it cannot."""

from __future__ import annotations

import pytest

from config import config
from errors import ExtractError
from pipeline.extract import llm


@pytest.fixture(autouse=True)
def _clean():
    llm.reset()
    yield
    llm.reset()


@pytest.fixture
def backends(monkeypatch):
    """Stand in for both backends, recording which was built."""
    built: list[str] = []

    class _Groq:
        name = "groq"
        model = "hosted"

        def __init__(self, *a, **k):
            built.append("groq")

        def available(self):
            return _Groq.reachable

    class _Ollama:
        name = "ollama"
        model = "local"

        def __init__(self, *a, **k):
            built.append("ollama")

        def available(self):
            return True

    _Groq.reachable = True
    monkeypatch.setattr(llm, "GroqBackend", _Groq)
    monkeypatch.setattr(llm, "OllamaBackend", _Ollama)
    return built, _Groq


def test_bulk_follows_llm_backend(monkeypatch, backends):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_INTERACTIVE_BACKEND", "groq")
    assert llm.get_backend(role=llm.BULK).name == "ollama"


def test_interactive_can_differ_from_bulk(monkeypatch, backends):
    """The whole point: ingest local, query hosted."""
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_INTERACTIVE_BACKEND", "groq")
    assert llm.get_backend(role=llm.INTERACTIVE).name == "groq"


def test_interactive_defaults_to_the_bulk_setting(monkeypatch, backends):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_INTERACTIVE_BACKEND", None)
    assert llm.get_backend(role=llm.INTERACTIVE).name == "ollama"


def test_an_unavailable_interactive_preference_falls_back(monkeypatch, backends):
    """A missing Groq key should make search slower, not broken."""
    _, groq = backends
    groq.reachable = False
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_INTERACTIVE_BACKEND", "groq")

    assert llm.get_backend(role=llm.INTERACTIVE).name == "ollama"


def test_an_explicitly_named_backend_still_fails_loudly(monkeypatch, backends):
    """Naming one is a decision about where content may go, not a preference."""
    _, groq = backends
    groq.reachable = False
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")

    with pytest.raises(ExtractError, match="unavailable"):
        llm.get_backend("groq", role=llm.INTERACTIVE)


def test_local_only_overrides_the_interactive_preference(monkeypatch, backends):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_INTERACTIVE_BACKEND", "groq")
    assert llm.get_backend(local_only=True, role=llm.INTERACTIVE).name == "ollama"


def test_local_only_with_an_explicit_groq_is_refused(monkeypatch, backends):
    monkeypatch.setattr(config, "LLM_BACKEND", "auto")
    with pytest.raises(ExtractError, match="local-only"):
        llm.get_backend("groq", local_only=True)


def test_a_backend_is_built_once_per_choice(monkeypatch, backends):
    built, _ = backends
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    llm.get_backend(role=llm.BULK)
    llm.get_backend(role=llm.BULK)
    assert built.count("ollama") == 1


def test_the_output_ceiling_is_not_hardcoded_above_a_free_tier():
    """max_tokens is reserved against the tokens-per-minute budget up front.

    Asking for 8192 output tokens on an 8000 TPM account returns 413 for a
    two-word prompt, every time, with a message about the request being too
    large that has nothing to do with the prompt. It was previously hardcoded,
    and the 413 handler read it as a rate limit and retried it with backoff —
    forever, since no amount of waiting shrinks a constant.
    """
    import inspect

    from config import config
    from pipeline.extract.llm import groq

    source = inspect.getsource(groq)
    assert "max_tokens=8192" not in source
    assert "max_tokens=config.GROQ_MAX_OUTPUT_TOKENS" in source
    assert config.GROQ_MAX_OUTPUT_TOKENS <= 8000


def test_selector_learning_can_differ_from_bulk(monkeypatch, backends):
    """The cheapest place in the pipeline to spend a hosted model.

    One call per domain, replayed free on every page after -- the opposite of
    the volume that makes a hosted model unaffordable for extraction.
    """
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_SELECTOR_BACKEND", "groq")
    assert llm.get_backend(role=llm.SELECTOR).name == "groq"
    assert llm.get_backend(role=llm.BULK).name == "ollama"


def test_selector_defaults_to_the_bulk_setting(monkeypatch, backends):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_SELECTOR_BACKEND", None)
    assert llm.get_backend(role=llm.SELECTOR).name == "ollama"


def test_an_unavailable_selector_preference_falls_back(monkeypatch, backends):
    """An unreachable hosted model should make tier 2 worse, not stop ingest.

    Learning would then fail on the local model and the page pays tier 3 --
    slower and more expensive, but it still extracts.
    """
    _, groq = backends
    groq.reachable = False
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_SELECTOR_BACKEND", "groq")
    assert llm.get_backend(role=llm.SELECTOR).name == "ollama"


def test_local_only_overrides_the_selector_preference(monkeypatch, backends):
    """Learning sends a DOM skeleton to the model, so the privacy switch binds."""
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_SELECTOR_BACKEND", "groq")
    assert llm.get_backend(role=llm.SELECTOR, local_only=True).name == "ollama"


def test_the_cascade_asks_for_the_selector_role_when_learning(monkeypatch):
    """Tier 2 and tier 3 are different jobs and must not share a backend.

    Without this the role exists and nothing uses it, which is exactly how the
    setting silently did nothing.
    """
    import inspect

    from pipeline.extract import cascade

    source = inspect.getsource(cascade.ExtractionCascade)
    assert "role=SELECTOR" in source
    # Tier 3 must not have been switched over with it.
    assert source.count("role=SELECTOR") == 1


def test_the_cascade_defaults_to_bulk_for_everything_else(monkeypatch, backends):
    from pipeline.extract.cascade import ExtractionCascade

    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_SELECTOR_BACKEND", "groq")
    cascade = ExtractionCascade()

    assert cascade._get_backend(False).name == "ollama"          # tier 3
    assert cascade._get_backend(False, role=llm.SELECTOR).name == "groq"
