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
