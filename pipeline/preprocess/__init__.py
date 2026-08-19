"""Preprocessing: turn a fetched document into text worth indexing.

Cleaning happens before chunking because every later decision reads the text —
the chunk router counts markdown headers, the semantic splitter embeds
sentences — and both are badly served by navigation chrome and mojibake.
"""

from .cleaners import (
    clean_whitespace,
    fix_encoding,
    process_text,
    remove_ocr_garbage,
    strip_html,
)
from .orchestrator import DocumentPreprocessor, PreprocessedDocument
from .pii_lang import detect_language, preload, remove_pii, reset

__all__ = [
    "DocumentPreprocessor",
    "PreprocessedDocument",
    "clean_whitespace",
    "detect_language",
    "fix_encoding",
    "preload",
    "process_text",
    "remove_ocr_garbage",
    "remove_pii",
    "reset",
    "strip_html",
]
