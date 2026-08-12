"""Error taxonomy for the pipeline.

Two things every error carries that a bare ``Exception("...")`` does not:

* the **stage** it happened in, so a failure can be attributed without parsing
  a message string;
* a **transient** flag, so the retry decision is a property of the error rather
  than a regex over its text. Celery's ``autoretry_for`` keys off
  :class:`TransientError`; everything else fails immediately and lands in the
  dead-letter collection.

The distinction that matters most is *blocked* vs. *transient*. A 503 is an
outage and you retry it. A challenge page is a refusal, and retrying it harder
is circumventing an access control — so :class:`BlockedError` is deliberately
**not** transient and trips the circuit breaker permanently.
"""

from __future__ import annotations

from typing import Any, Optional


class PipelineError(Exception):
    """Base class. Every pipeline failure is one of these."""

    #: Stage name, e.g. "FETCH". Set by subclasses or at raise site.
    stage: str = "UNKNOWN"
    #: Whether retrying the same operation could plausibly succeed.
    transient: bool = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "stage": self.stage,
            "transient": self.transient,
            "message": self.message,
            **self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        extra = " ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({extra})"


class TransientError(PipelineError):
    """Retrying is reasonable: timeouts, 5xx, connection resets."""

    transient = True


# --------------------------------------------------------------------------- #
# Compliance / access
# --------------------------------------------------------------------------- #


class ComplianceError(PipelineError):
    """The pipeline declined to make the request. Never retried."""

    stage = "FETCH"


class RobotsDisallowed(ComplianceError):
    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"robots.txt disallows {url}: {reason}", url=url, reason=reason)


class SchemeNotAllowed(ComplianceError):
    def __init__(self, url: str, scheme: str) -> None:
        super().__init__(f"scheme {scheme!r} is not fetchable", url=url, scheme=scheme)


class HostNotAllowed(ComplianceError):
    def __init__(self, host: str, reason: str) -> None:
        super().__init__(f"host {host} refused by policy: {reason}", host=host, reason=reason)


class BlockedError(PipelineError):
    """An anti-bot control answered instead of the application.

    Not transient, on purpose: a challenge is a stated refusal to serve
    automated clients, and the supported response is to find an API, a bulk
    export, or another source — not to try again from somewhere else.
    """

    stage = "FETCH"

    def __init__(self, url: str, signal: str) -> None:
        super().__init__(f"{url} is behind an anti-bot control ({signal})", url=url, signal=signal)


class CircuitOpen(PipelineError):
    """This host is cut off after repeated failures."""

    stage = "FETCH"

    def __init__(self, host: str, retry_after: float) -> None:
        super().__init__(
            f"circuit open for {host}; retry in {retry_after:.0f}s",
            host=host,
            retry_after=retry_after,
        )


# --------------------------------------------------------------------------- #
# Per-stage failures
# --------------------------------------------------------------------------- #


class FetchError(PipelineError):
    stage = "FETCH"


class TransientFetchError(TransientError):
    stage = "FETCH"


class TooLarge(FetchError):
    def __init__(self, url: str, size: int, limit: int) -> None:
        super().__init__(
            f"{url} is {size} bytes, over the {limit}-byte limit",
            url=url,
            size=size,
            limit=limit,
        )


class DecodeError(PipelineError):
    stage = "DECODE"


class ParseError(PipelineError):
    stage = "PARSE"


class UnsupportedType(ParseError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"no handler for {kind}{': ' + detail if detail else ''}", kind=kind)


class MissingDependency(ParseError):
    """A handler exists but its optional third-party package is not installed."""

    def __init__(self, package: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} needs the {package!r} package: pip install {package}",
            package=package,
            purpose=purpose,
        )


class ExtractError(PipelineError):
    stage = "EXTRACT"


class TransientExtractError(TransientError):
    """LLM backend was unreachable or rate-limited — worth another go."""

    stage = "EXTRACT"


class SchemaViolation(ExtractError):
    """The model returned JSON that does not match the requested shape."""

    def __init__(self, detail: str, payload: Optional[dict] = None) -> None:
        super().__init__(f"extraction did not match schema: {detail}", detail=detail)
        self.payload = payload


class ValidationFailed(PipelineError):
    stage = "NORMALIZE"

    def __init__(self, failures: list[str]) -> None:
        super().__init__(
            f"{len(failures)} validation rule(s) failed: " + "; ".join(failures[:5]),
            failures=failures,
        )


class PersistError(PipelineError):
    stage = "PERSIST"


class TransientPersistError(TransientError):
    stage = "PERSIST"


__all__ = [
    "BlockedError",
    "CircuitOpen",
    "ComplianceError",
    "DecodeError",
    "ExtractError",
    "FetchError",
    "HostNotAllowed",
    "MissingDependency",
    "ParseError",
    "PersistError",
    "PipelineError",
    "RobotsDisallowed",
    "SchemaViolation",
    "SchemeNotAllowed",
    "TooLarge",
    "TransientError",
    "TransientExtractError",
    "TransientFetchError",
    "TransientPersistError",
    "UnsupportedType",
    "ValidationFailed",
]
