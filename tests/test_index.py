"""The indexing facade, end to end with no model and no network."""

from __future__ import annotations

import pytest

from pipeline.index import index_text


def test_index_text_runs_every_stage(fake_embedder, recording_store):
    report = index_text(
        "# Title\n" + ("An ordinary paragraph of prose. " * 60),
        source="https://example.com/report",
        embedder=fake_embedder,
        store=recording_store,
        strategy="fixed",
    )

    assert report.chunks > 1
    assert report.stored == report.chunks
    assert report.strategy == "fixed"
    assert report.language == "en"
    assert report.source == "https://example.com/report"
    assert set(report.timings_ms) == {"preprocess", "chunk", "embed", "store", "total"}
    assert not report.warnings


def test_provenance_reaches_the_stored_chunks(fake_embedder, recording_store):
    index_text(
        "Some prose worth indexing, long enough to survive cleaning.",
        source="https://example.com/a",
        extra_metadata={"content_hash": "deadbeef", "kind": "html"},
        embedder=fake_embedder,
        store=recording_store,
        strategy="fixed",
    )

    [document] = recording_store.documents
    metadata = document.chunks[0].metadata
    assert metadata.source == "https://example.com/a"
    assert metadata.language == "en"
    assert metadata.chunk_strategy == "fixed"
    assert metadata.embedding_model == "fake-dense"
    assert metadata.extra["content_hash"] == "deadbeef"
    assert metadata.extra["kind"] == "html"


def test_empty_text_is_reported_not_raised(fake_embedder, recording_store):
    report = index_text(
        "   ", source="https://example.com/blank", embedder=fake_embedder, store=recording_store
    )
    assert report.chunks == 0
    assert report.stored == 0
    assert report.warnings == ["no text to index"]
    assert recording_store.documents == []


def test_text_that_cleans_away_to_nothing_is_reported(fake_embedder, recording_store):
    report = index_text(
        "<script>track()</script><style>a{}</style>",
        source="https://example.com/js",
        is_html=True,
        embedder=fake_embedder,
        store=recording_store,
    )
    assert report.stored == 0
    assert report.warnings == ["nothing left after cleaning"]


def test_long_text_is_truncated_with_a_warning(monkeypatch, fake_embedder, recording_store):
    from config import config

    monkeypatch.setattr(config, "INDEX_MAX_TEXT_CHARS", 500)
    report = index_text(
        "word " * 400,
        source="https://example.com/long",
        embedder=fake_embedder,
        store=recording_store,
        strategy="fixed",
    )
    assert report.truncated is True
    assert "truncated from 2000 to 500" in report.warnings[0]


def test_html_is_stripped_before_chunking(fake_embedder, recording_store):
    index_text(
        "<html><body><nav>Menu</nav><p>The real content is here.</p></body></html>",
        source="https://example.com/page",
        is_html=True,
        embedder=fake_embedder,
        store=recording_store,
        strategy="fixed",
    )
    [document] = recording_store.documents
    text = document.chunks[0].document
    assert "The real content is here." in text
    assert "<p>" not in text


def test_the_report_is_json_serialisable(fake_embedder, recording_store):
    import json

    report = index_text(
        "Some text.",
        source="https://example.com/a",
        embedder=fake_embedder,
        store=recording_store,
        strategy="fixed",
    )
    assert json.loads(json.dumps(report.to_dict()))["source"] == "https://example.com/a"


def test_the_agentic_router_is_used_when_no_strategy_is_given(
    fake_embedder, recording_store
):
    report = index_text(
        "\n# One\na\n\n## Two\nb\n\n### Three\nc\n",
        source="https://example.com/structured",
        embedder=fake_embedder,
        store=recording_store,
    )
    assert report.strategy == "hierarchical"
