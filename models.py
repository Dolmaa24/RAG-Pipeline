"""The data carried between pipeline stages, and the records written out.

``ExtractionItem`` is the single unit of work. Every stage takes one and returns
one; a stage that fails calls :meth:`ExtractionItem.fail` and the rest are
skipped. ``extra="forbid"`` is deliberate — a typo like ``item.cleanedtext``
raises immediately instead of silently creating a dead attribute that the next
stage then reads as ``None``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Stage(str, Enum):
    ROUTE = "ROUTE"
    FETCH = "FETCH"
    DECODE = "DECODE"
    PARSE = "PARSE"
    EXTRACT = "EXTRACT"
    NORMALIZE = "NORMALIZE"
    VALIDATE = "VALIDATE"
    PERSIST = "PERSIST"


class ResourceKind(str, Enum):
    """What the bytes actually are, decided by magic bytes and Content-Type.

    Never by file extension: ``report.pdf`` is routinely an HTML error page,
    and a PDF is routinely served from a URL with no extension at all.
    """

    HTML = "html"
    DOCUMENT = "document"      # pdf, docx, pptx, xlsx, epub, rtf, odt
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    LIVESTREAM = "livestream"
    FEED = "feed"              # rss / atom
    SITEMAP = "sitemap"
    TABULAR = "tabular"        # csv / tsv
    DATA = "data"              # json / jsonl / xml / yaml
    ARCHIVE = "archive"        # zip / tar / 7z / gz
    EMAIL = "email"            # eml / msg
    TEXT = "text"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Which tier of the cascade produced the data. Recorded on every row.

    This is the field you group by when someone asks why the pipeline got
    faster, or which records are worth re-running after a prompt change.
    """

    CACHE = "cache"                    # tier 0 — unchanged since last run
    STRUCTURED_DATA = "structured"     # tier 1 — the publisher's own JSON-LD
    SELECTOR_SPEC = "selector"         # tier 2 — a learned per-domain spec
    LLM = "llm"                        # tier 3 — a model read the text
    NATIVE = "native"                  # the handler produced fields directly
    NONE = "none"


class FetchMode(str, Enum):
    STATIC = "static"
    BROWSER = "browser"
    YTDLP = "ytdlp"
    CACHED = "cached"
    INLINE = "inline"  # bytes handed in directly, e.g. an archive member


class Provenance(BaseModel):
    """Everything needed to decide whether a stored record is still true.

    Without it you cannot tell which of two conflicting rows is stale, or
    whether a field came from the publisher's own markup or from a model's
    reading of rendered text. Both are "the price", and only one of them is
    worth trusting without review.
    """

    model_config = ConfigDict(extra="forbid")

    source_url: str
    canonical_url: Optional[str] = None
    final_url: Optional[str] = None
    redirect_chain: List[str] = Field(default_factory=list)

    fetched_at: datetime = Field(default_factory=_utcnow)
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    content_hash: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None

    kind: ResourceKind = ResourceKind.UNKNOWN
    fetch_mode: FetchMode = FetchMode.STATIC
    handler: Optional[str] = None
    method: ExtractionMethod = ExtractionMethod.NONE
    tier: Optional[int] = None
    confidence: float = 0.0

    llm_backend: Optional[str] = None
    llm_model: Optional[str] = None
    schema_hash: Optional[str] = None
    prompt_hash: Optional[str] = None
    selector_spec_id: Optional[str] = None

    pipeline_version: str = "3.0.0"
    job_id: Optional[str] = None
    parent_url: Optional[str] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class ExtractionItem(BaseModel):
    """One unit of work, carried through every stage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=False, arbitrary_types_allowed=True)

    url: str
    canonical_url: Optional[str] = None
    status_code: Optional[int] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    final_url: Optional[str] = None
    redirect_chain: List[str] = Field(default_factory=list)
    fetch_mode: FetchMode = FetchMode.STATIC

    # --- Stage 1: FETCH ---
    raw_bytes: Optional[bytes] = None
    content_type: Optional[str] = None

    # --- Stage 2: ROUTE ---
    kind: ResourceKind = ResourceKind.UNKNOWN
    handler: Optional[str] = None

    # --- Stage 3: DECODE / PARSE ---
    decoded_text: Optional[str] = None
    parsed_tree: Optional[Dict[str, Any]] = None
    cleaned_text: Optional[str] = None
    #: Embedded structured data found in the document (JSON-LD, microdata,
    #: OpenGraph, __NEXT_DATA__, tables). Tier 1 of the cascade reads this.
    structured: Optional[Dict[str, Any]] = None

    # --- Stage 4: EXTRACT ---
    extracted_data: Optional[Dict[str, Any]] = None
    method: ExtractionMethod = ExtractionMethod.NONE
    tier: Optional[int] = None
    confidence: float = 0.0

    # --- Stage 5: NORMALIZE / VALIDATE ---
    normalized_data: Optional[Dict[str, Any]] = None
    validation_failures: List[str] = Field(default_factory=list)

    # --- Metadata and lineage ---
    metadata: Dict[str, Any] = Field(default_factory=dict)
    #: Items discovered inside this one: archive members, feed entries,
    #: email attachments, sitemap URLs. Recursion is bounded by `depth`.
    children: List["ExtractionItem"] = Field(default_factory=list)
    depth: int = 0
    parent_url: Optional[str] = None

    content_hash: Optional[str] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    job_id: Optional[str] = None

    # --- Failure tracking ---
    error: Optional[str] = None
    error_type: Optional[str] = None
    failed_at_stage: Optional[Stage] = None
    transient: bool = False

    # ------------------------------------------------------------------ #
    # Failure handling
    # ------------------------------------------------------------------ #

    def fail(
        self,
        stage: Stage,
        message: str,
        *,
        error_type: Optional[str] = None,
        transient: bool = False,
    ) -> "ExtractionItem":
        self.error = message
        self.error_type = error_type
        self.failed_at_stage = stage
        self.transient = transient
        return self

    def fail_from(self, exc: Exception) -> "ExtractionItem":
        """Record a raised :class:`~errors.PipelineError` on the item."""
        stage_name = getattr(exc, "stage", None) or "UNKNOWN"
        try:
            stage = Stage(stage_name)
        except ValueError:
            stage = Stage.EXTRACT
        return self.fail(
            stage,
            str(exc),
            error_type=type(exc).__name__,
            transient=bool(getattr(exc, "transient", False)),
        )

    def clear_error(self) -> None:
        self.error = None
        self.error_type = None
        self.failed_at_stage = None
        self.transient = False

    @property
    def ok(self) -> bool:
        return self.error is None

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #

    @property
    def text_for_extraction(self) -> str:
        """Best available text for the extraction stage."""
        if self.cleaned_text:
            return self.cleaned_text
        if self.parsed_tree:
            return self.parsed_tree.get("text_content", "") or ""
        return self.decoded_text or ""

    def compute_content_hash(self) -> Optional[str]:
        """sha256 over the bytes, or over the text when there are no bytes.

        Media transcripts have no raw bytes on the item (the audio is deleted
        with its temp directory), so the transcript itself is what identifies
        the content.
        """
        if self.raw_bytes:
            digest = hashlib.sha256(self.raw_bytes).hexdigest()
        elif self.cleaned_text or self.decoded_text:
            digest = hashlib.sha256(
                (self.cleaned_text or self.decoded_text or "").encode("utf-8")
            ).hexdigest()
        else:
            return None
        self.content_hash = digest
        return digest

    def record_timing(self, name: str, seconds: float) -> None:
        self.timings_ms[name] = round(self.timings_ms.get(name, 0.0) + seconds * 1000, 2)

    def provenance(self, **overrides: Any) -> Provenance:
        """Snapshot everything known about where this record came from."""
        fields: Dict[str, Any] = {
            "source_url": self.url,
            "canonical_url": self.canonical_url,
            "final_url": self.final_url,
            "redirect_chain": list(self.redirect_chain),
            "http_status": self.status_code,
            "content_type": self.content_type,
            "content_length": len(self.raw_bytes) if self.raw_bytes else None,
            "content_hash": self.content_hash,
            "etag": self.headers.get("etag"),
            "last_modified": self.headers.get("last-modified"),
            "kind": self.kind,
            "fetch_mode": self.fetch_mode,
            "handler": self.handler,
            "method": self.method,
            "tier": self.tier,
            "confidence": self.confidence,
            "job_id": self.job_id,
            "parent_url": self.parent_url,
            "timings_ms": dict(self.timings_ms),
            "warnings": list(self.warnings),
        }
        fields.update(overrides)
        return Provenance(**fields)

    def summary(self) -> Dict[str, Any]:
        """Small dict safe to return over the API — no raw bytes, no full text."""
        return {
            "url": self.url,
            "canonical_url": self.canonical_url,
            "kind": self.kind.value,
            "handler": self.handler,
            "method": self.method.value,
            "tier": self.tier,
            "confidence": round(self.confidence, 3),
            "status_code": self.status_code,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "extracted_data": self.normalized_data or self.extracted_data,
            "children": len(self.children),
            "warnings": self.warnings,
            "validation_failures": self.validation_failures,
            "timings_ms": self.timings_ms,
            "error": self.error,
        }


ExtractionItem.model_rebuild()


class RunReport(BaseModel):
    """What one job actually did. Returned by the task and stored alongside it.

    The tier counts are the interesting part: they are how you see the cascade
    working, and how you notice the day a site changes its markup and every
    page starts falling through to the LLM again.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: Optional[str] = None
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None

    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_duplicate: int = 0
    skipped_disallowed: int = 0

    by_kind: Dict[str, int] = Field(default_factory=dict)
    by_method: Dict[str, int] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    def record(self, item: ExtractionItem) -> None:
        self.submitted += 1
        self.by_kind[item.kind.value] = self.by_kind.get(item.kind.value, 0) + 1
        if item.ok:
            self.succeeded += 1
            self.by_method[item.method.value] = self.by_method.get(item.method.value, 0) + 1
        else:
            self.failed += 1
            self.errors.append(
                {
                    "url": item.url,
                    "stage": item.failed_at_stage.value if item.failed_at_stage else None,
                    "type": item.error_type,
                    "message": item.error,
                    "transient": item.transient,
                }
            )
        for name, ms in item.timings_ms.items():
            self.timings_ms[name] = round(self.timings_ms.get(name, 0.0) + ms, 2)
        self.warnings.extend(w for w in item.warnings if w not in self.warnings)

    def finish(self) -> "RunReport":
        self.finished_at = _utcnow()
        return self

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or _utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def llm_avoidance_rate(self) -> float:
        """Share of successes that never touched a model. The headline number."""
        if not self.succeeded:
            return 0.0
        llm = self.by_method.get(ExtractionMethod.LLM.value, 0)
        return round(1.0 - llm / self.succeeded, 3)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["duration_seconds"] = round(self.duration_seconds, 2)
        payload["llm_avoidance_rate"] = self.llm_avoidance_rate
        return payload


__all__ = [
    "ExtractionItem",
    "ExtractionMethod",
    "FetchMode",
    "Provenance",
    "ResourceKind",
    "RunReport",
    "Stage",
]
