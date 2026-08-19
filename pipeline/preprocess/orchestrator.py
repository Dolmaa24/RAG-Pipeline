"""Preprocessing: a fetched document in, clean indexable text out."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from config import config
from observability import get_logger

from .cleaners import process_text
from .pii_lang import detect_language, remove_pii

log = get_logger("preprocess")


class PreprocessedDocument(BaseModel):
    """Clean text plus what the chunker and the store need to know about it.

    The unmasked original is deliberately *not* carried here. Holding it beside
    the masked copy would defeat the masking the moment anything logged, cached
    or serialised the object — and nothing downstream reads it.
    """

    clean_text: str
    language: str
    pii_masked: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentPreprocessor:
    """Clean, detect the language, and optionally mask PII.

    Deduplication is not done here. The runner already checks every item against
    the store with banded simhash in
    :meth:`~pipeline.runner.Pipeline._check_duplicate` and leaves the verdict on
    the item; asking the same question again with a weaker algorithm would give
    the pipeline two answers and no way to choose between them.
    """

    def __init__(self, *, apply_pii_removal: Optional[bool] = None) -> None:
        self.apply_pii_removal = (
            config.INDEX_PII_REMOVAL if apply_pii_removal is None else apply_pii_removal
        )

    def process(
        self,
        raw_text: str,
        *,
        is_html: bool = False,
        source: str = "",
        page_no: Optional[int] = None,
        section_name: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> PreprocessedDocument:
        """Run the cleaning steps, preserving where the text came from."""
        clean_text = process_text(raw_text, is_html=is_html)
        language = detect_language(clean_text)

        masked = False
        if self.apply_pii_removal and clean_text.strip():
            # Presidio's recogniser set is English-only by default, so a page in
            # another language would be analysed under the wrong rules and come
            # back untouched — silently, which is the worst way to not mask PII.
            if language == "en":
                clean_text = remove_pii(clean_text, language="en")
                masked = True
            else:
                log.warning("preprocess.pii_skipped", language=language, source=source)

        meta: Dict[str, Any] = dict(extra_metadata or {})
        if source:
            meta["source"] = source
        if page_no is not None:
            meta["page_no"] = page_no
        if section_name:
            meta["section_name"] = section_name

        return PreprocessedDocument(
            clean_text=clean_text,
            language=language,
            pii_masked=masked,
            metadata=meta,
        )


__all__ = ["DocumentPreprocessor", "PreprocessedDocument"]
