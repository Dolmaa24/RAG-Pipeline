"""Language detection and PII masking.

Presidio is built lazily, and only when ``INDEX_PII_REMOVAL`` is on. Constructing
an ``AnalyzerEngine`` loads a spaCy NER model — hundreds of megabytes — and the
first version of this module did it as an *import side effect*, so every process
that touched anything under ``pipeline.preprocess`` paid for it whether or not it
ever masked a document. In a prefork worker pool on an 8 GB machine that is the
difference between a worker starting and a worker swapping.

The engines are then held for the life of the process, like the Whisper model in
:mod:`pipeline.transcribe.base`: building them is the expensive part and they are
stateless once built.

Presidio is not in ``requirements.txt``. Masking is off by default, so the
packages are installed only by someone who turned it on, and the error raised
here names the exact commands.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import langdetect
from langdetect import DetectorFactory

from errors import MissingDependency, ParseError
from observability import get_logger

log = get_logger("preprocess.pii")

#: langdetect samples randomly, so the same short input yields different answers
#: across runs unless the factory is seeded. An unstable language tag is worse
#: than a wrong one: it makes a re-run of the same document irreproducible.
DetectorFactory.seed = 0

_engines: Optional[tuple[Any, Any]] = None
_lock = threading.Lock()


def detect_language(text: str) -> str:
    """The dominant language as an ISO code, or ``"unknown"``."""
    if not text or not text.strip():
        return "unknown"
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return "unknown"


def _get_engines() -> tuple[Any, Any]:
    """The process-wide analyzer and anonymizer, constructed once."""
    global _engines
    with _lock:
        if _engines is not None:
            return _engines

        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:
            raise MissingDependency(
                "presidio-analyzer presidio-anonymizer", "PII masking"
            ) from exc

        try:
            engines = (AnalyzerEngine(), AnonymizerEngine())
        except OSError as exc:
            # Presidio installs spaCy but not a model, so this is the failure
            # someone hits immediately after the pip install succeeds.
            raise ParseError(
                "PII masking needs a spaCy model: "
                "python -m spacy download en_core_web_lg"
            ) from exc

        log.info("preprocess.pii.loaded")
        _engines = engines
        return _engines


def remove_pii(text: str, *, language: str = "en") -> str:
    """Mask names, locations, phones, emails, cards and national IDs.

    Each finding is replaced by its entity type — ``<PERSON>``, ``<US_SSN>`` —
    rather than deleted, so a chunk still reads as a sentence and a retrieval
    hit still makes sense to whoever reads it.
    """
    if not text or not text.strip():
        return text

    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(text=text, entities=None, language=language)
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def preload() -> None:
    """Build the engines now rather than on the first masked document."""
    _get_engines()


def reset() -> None:
    """Drop the engines. Used by tests and to reclaim the memory."""
    global _engines
    with _lock:
        _engines = None


__all__ = ["detect_language", "preload", "remove_pii", "reset"]
