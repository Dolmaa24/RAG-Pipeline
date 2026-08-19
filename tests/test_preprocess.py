"""Preprocessing: cleaning, language, and the PII gate."""

from __future__ import annotations

import pytest

from pipeline.preprocess.cleaners import (
    clean_whitespace,
    fix_encoding,
    process_text,
    remove_ocr_garbage,
    strip_html,
)
from pipeline.preprocess.orchestrator import DocumentPreprocessor


def test_strip_html_drops_chrome_not_content():
    html = """
    <html><head><style>body{color:red}</style></head>
    <body><nav>Home About</nav><script>track()</script>
    <p>The actual sentence.</p><footer>(c) 2026</footer></body></html>
    """
    text = strip_html(html)
    assert "The actual sentence." in text
    for noise in ("track()", "color:red", "Home About", "(c) 2026"):
        assert noise not in text


def test_fix_encoding_repairs_mojibake():
    assert fix_encoding("thereâ€™s") == "there's"


def test_clean_whitespace_collapses_runs_and_blank_lines():
    assert clean_whitespace("a    b\n\n\n\n\nc") == "a b\n\nc"


def test_remove_ocr_garbage_keeps_prose_drops_symbol_soup():
    text = "A normal line of prose.\n|#@$%^&*|~`{}[]<>|\nAnother normal line."
    cleaned = remove_ocr_garbage(text)
    assert "A normal line of prose." in cleaned
    assert "Another normal line." in cleaned
    assert "|#@$%^&*|" not in cleaned


def test_process_text_is_idempotent_on_clean_input():
    once = process_text("Already clean text.")
    assert process_text(once) == once


def test_language_detection_is_stable_across_calls():
    # langdetect samples randomly; the seed in pii_lang is what makes this pass.
    processor = DocumentPreprocessor(apply_pii_removal=False)
    text = "This is an English sentence with enough words to classify."
    languages = {processor.process(text).language for _ in range(5)}
    assert languages == {"en"}


def test_metadata_is_preserved():
    processor = DocumentPreprocessor(apply_pii_removal=False)
    doc = processor.process(
        "Some text.",
        source="https://example.com/a",
        page_no=4,
        section_name="Results",
        extra_metadata={"content_hash": "abc"},
    )
    assert doc.metadata == {
        "content_hash": "abc",
        "source": "https://example.com/a",
        "page_no": 4,
        "section_name": "Results",
    }


def test_pii_removal_is_off_by_default_and_costs_nothing():
    """The default must not import Presidio, let alone load spaCy."""
    import sys

    processor = DocumentPreprocessor()
    assert processor.apply_pii_removal is False

    text = "Call Jane Doe on 555-0100."
    assert processor.process(text).clean_text == text
    assert processor.process(text).pii_masked is False
    assert "presidio_analyzer" not in sys.modules


def test_pii_removal_reports_a_clear_error_when_presidio_is_absent():
    pytest.importorskip  # noqa: B018 - documents intent when presidio IS installed
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("presidio is installed; the missing-dependency path cannot run")

    from errors import MissingDependency

    processor = DocumentPreprocessor(apply_pii_removal=True)
    with pytest.raises(MissingDependency) as excinfo:
        processor.process("Call Jane Doe on 555-0100.")
    assert "presidio-analyzer" in str(excinfo.value)


def test_non_english_text_is_not_masked_silently():
    """Presidio's default recognisers are English-only."""
    processor = DocumentPreprocessor(apply_pii_removal=True)
    doc = processor.process("Ceci est une phrase en francais avec assez de mots.")
    assert doc.language != "en"
    assert doc.pii_masked is False
