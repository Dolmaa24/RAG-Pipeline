"""Chunking: the strategy router, and the model-assisted path."""

from __future__ import annotations

import pytest

from pipeline.chunk.chunker import DocumentChunker
from pipeline.chunk.strategies import fixed_chunk, hierarchical_chunk, llm_chunk


def test_headings_choose_hierarchical(make_doc):
    text = "\n# One\nalpha\n\n## Two\nbeta\n\n### Three\ngamma\n"
    assert DocumentChunker().decide_strategy(make_doc(text)) == "hierarchical"


def test_short_lines_do_not_reach_the_model_by_default(make_doc):
    """Short lines mean OCR damage in a scan and a nav menu in a web page.

    This test cannot tell them apart and neither can the router, so with
    indexing on by default it must not spend a model call guessing. Measured
    cost of guessing wrong on one ordinary HTML page: 39s and 78 chunks.
    """
    text = "\n".join(f"col{i} 12.{i}" for i in range(40))
    assert DocumentChunker().decide_strategy(make_doc(text)) == "fixed"


def test_short_lines_choose_the_model_when_allowed(make_doc, monkeypatch):
    """Opting in is how a corpus of scanned documents gets the model."""
    from config import config

    monkeypatch.setattr(config, "INDEX_AGENTIC_ALLOW_LLM", True)
    text = "\n".join(f"col{i} 12.{i}" for i in range(40))
    assert DocumentChunker().decide_strategy(make_doc(text)) == "llm"


def test_long_prose_chooses_semantic(make_doc):
    text = ("This is a long paragraph of ordinary prose. " * 80)
    assert DocumentChunker().decide_strategy(make_doc(text)) == "semantic"


def test_short_prose_falls_back_to_fixed(make_doc):
    assert DocumentChunker().decide_strategy(make_doc("One short sentence.")) == "fixed"


def test_empty_text_does_not_crash_the_router(make_doc):
    assert DocumentChunker().decide_strategy(make_doc("   ")) == "fixed"


def test_unknown_strategy_is_rejected(make_doc):
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        DocumentChunker().chunk(make_doc("text"), strategy="telepathy")


def test_fixed_chunk_splits_and_carries_metadata(make_doc):
    doc = make_doc("word " * 2000, page_no=3)
    chunks = fixed_chunk(doc, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.metadata.chunk_strategy == "fixed" for chunk in chunks)
    assert all(chunk.metadata.source == "test://doc" for chunk in chunks)
    assert all(chunk.metadata.page_no == 3 for chunk in chunks)


def test_hierarchical_chunk_records_the_heading_path(make_doc):
    text = "# Report\nintro\n\n## Revenue\nthe number\n\n## Costs\nthe other number\n"
    sections = {chunk.metadata.section_name for chunk in hierarchical_chunk(make_doc(text))}
    assert "Report > Revenue" in sections
    assert "Report > Costs" in sections


def test_chunk_records_the_strategy_it_used(make_doc):
    result = DocumentChunker().chunk(make_doc("word " * 500), strategy="fixed")
    assert result.strategy == "fixed"
    assert len(result.chunks) > 1


class _Response:
    def __init__(self, data):
        self.data = data


class _Backend:
    """Stands in for an Ollama or Groq backend."""

    def __init__(self, data=None, raises=None):
        self._data = data
        self._raises = raises
        self.calls = 0

    def complete_json(self, **_kwargs):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return _Response(self._data)


def test_llm_chunk_parses_the_reply_into_separate_chunks(monkeypatch, make_doc):
    """The bug this replaced returned the whole completion as one chunk."""
    backend = _Backend({"chunks": ["first section", "second section", "third"]})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    chunks = llm_chunk(make_doc("some messy text"))
    assert [chunk.document for chunk in chunks] == [
        "first section",
        "second section",
        "third",
    ]
    assert all(chunk.metadata.chunk_strategy == "llm" for chunk in chunks)


def test_llm_chunk_drops_blank_sections(monkeypatch, make_doc):
    backend = _Backend({"chunks": ["real", "   ", ""]})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)
    assert len(llm_chunk(make_doc("text"))) == 1


def test_llm_chunk_falls_back_when_the_backend_is_unavailable(monkeypatch, make_doc):
    def _unavailable(**_kwargs):
        raise RuntimeError("no model configured")

    monkeypatch.setattr("pipeline.extract.llm.get_backend", _unavailable)
    chunks = llm_chunk(make_doc("word " * 500))
    assert chunks
    assert all(chunk.metadata.chunk_strategy == "fixed" for chunk in chunks)


def test_llm_chunk_falls_back_per_window_when_a_call_fails(monkeypatch, make_doc):
    backend = _Backend(raises=RuntimeError("rate limited"))
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)
    chunks = llm_chunk(make_doc("word " * 500))
    assert chunks
    assert all(chunk.metadata.chunk_strategy == "fixed" for chunk in chunks)


def test_llm_chunk_falls_back_on_a_wrong_shaped_reply(monkeypatch, make_doc):
    backend = _Backend({"chunks": "not a list"})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)
    chunks = llm_chunk(make_doc("word " * 500))
    assert all(chunk.metadata.chunk_strategy == "fixed" for chunk in chunks)


def test_llm_chunk_windows_long_input_instead_of_truncating(monkeypatch, make_doc):
    """The original cut every document at 3000 characters and said nothing."""
    from config import config

    backend = _Backend({"chunks": ["section"]})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    text = "x" * (config.MAX_CHUNK_SIZE * 3)
    llm_chunk(make_doc(text))
    assert backend.calls == 3


def test_llm_chunk_respects_the_window_cap(monkeypatch, make_doc):
    from config import config

    backend = _Backend({"chunks": ["section"]})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    windows = config.INDEX_LLM_MAX_WINDOWS + 4
    llm_chunk(make_doc("x" * (config.MAX_CHUNK_SIZE * windows)))
    assert backend.calls == config.INDEX_LLM_MAX_WINDOWS


def test_a_fallen_back_strategy_reports_what_it_actually_did(monkeypatch, make_doc):
    """A run report claiming "llm" when no model ran is a misleading number."""

    def _unavailable(**_kwargs):
        raise RuntimeError("no model configured")

    monkeypatch.setattr("pipeline.extract.llm.get_backend", _unavailable)
    result = DocumentChunker().chunk(make_doc("word " * 500), strategy="llm")
    assert result.strategy == "fixed"
