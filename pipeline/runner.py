"""The pipeline itself: one function that takes a URL and returns records.

    acquire → route → handle → extract → normalize → validate → dedupe → persist

Everything above this module is a stage; everything below it is a transport
(Celery, the API, the batch script). Keeping the orchestration here rather than
inside the Celery task is what lets the same code path run synchronously in
``main.py``, in a test with no broker, and in a worker — and be the same code
path, so a bug found in one is fixed in all three.

Two behaviours worth naming:

**Children are first-class.** An archive, a feed, a sitemap or an email produces
child items, and each is run through the whole pipeline in turn, bounded by
``MAX_RECURSION_DEPTH``. That is how one submitted ZIP becomes four extracted
documents without the caller unpacking anything.

**A failed item never stops the run.** Failures are recorded on the
:class:`~models.RunReport` and the next item proceeds. A 200-URL job that hits
one 404 should return 199 records and one explanation, not an exception.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional  # noqa: F401  (Callable used in signatures)

from config import config
from errors import PipelineError, TransientError
from models import ExtractionItem, ExtractionMethod, ResourceKind, RunReport, Stage
from observability import get_logger, job_context, metrics
from pipeline.detect.router import Acquisition, TypeRouter, pre_route
from pipeline.extract.cascade import ExtractionCascade, get_cascade
from pipeline.fetch.client import ResilientFetcher
from pipeline.handlers import registry
from pipeline.normalizer import DataNormalizer
from pipeline.trust.dedupe import Deduplicator
from pipeline.trust.drift import DriftMonitor
from pipeline.trust.validation import validate_record
from urls import canonicalize

log = get_logger("runner")

#: Called with (stage_label, item) so a transport can report progress.
ProgressCallback = Callable[[str, ExtractionItem], None]


class Pipeline:
    """Runs items end to end. One instance per worker process."""

    def __init__(
        self,
        *,
        database=None,
        cascade: Optional[ExtractionCascade] = None,
        fetcher: Optional[ResilientFetcher] = None,
        deduplicator: Optional[Deduplicator] = None,
        drift: Optional[DriftMonitor] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.db = database
        self.cascade = cascade or get_cascade(database)
        self.fetcher = fetcher or ResilientFetcher()
        self.dedupe = deduplicator or Deduplicator(database=database)
        self.drift = drift or DriftMonitor(database=database)
        self.on_progress = on_progress

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #

    def run_url(
        self,
        url: str,
        prompt: str,
        schema: dict,
        *,
        force_dynamic: bool = False,
        local_only: bool = False,
        allowed_tiers: Optional[set[int]] = None,
        job_id: Optional[str] = None,
        follow_children: bool = True,
    ) -> RunReport:
        report = RunReport(job_id=job_id)
        item = ExtractionItem(url=url, job_id=job_id, canonical_url=canonicalize(url))

        with job_context(job_id, url):
            self.run_item(
                item,
                prompt,
                schema,
                report=report,
                force_dynamic=force_dynamic,
                local_only=local_only,
                allowed_tiers=allowed_tiers,
                follow_children=follow_children,
            )

        report.metrics = metrics.snapshot()
        report.finish()
        if self.db is not None:
            self.db.save_run(report)
        return report

    def run_batch(
        self,
        urls: list[str],
        prompt: str,
        schema: dict,
        *,
        job_id: Optional[str] = None,
        **options: Any,
    ) -> RunReport:
        report = RunReport(job_id=job_id)
        for url in urls:
            item = ExtractionItem(url=url, job_id=job_id, canonical_url=canonicalize(url))
            with job_context(job_id, url):
                self.run_item(item, prompt, schema, report=report, **options)
        report.metrics = metrics.snapshot()
        return report.finish()

    # ------------------------------------------------------------------ #
    # One item, all stages
    # ------------------------------------------------------------------ #

    def run_item(
        self,
        item: ExtractionItem,
        prompt: str,
        schema: dict,
        *,
        report: Optional[RunReport] = None,
        force_dynamic: bool = False,
        local_only: bool = False,
        allowed_tiers: Optional[set[int]] = None,
        follow_children: bool = True,
        extract_when: Optional[Callable[[ExtractionItem], bool]] = None,
    ) -> ExtractionItem:
        """Run one item through every stage.

        ``extract_when`` is consulted after the content has been fetched and
        parsed but before extraction, so a caller can decide from the *actual*
        kind rather than from a guess at the URL. A crawl looking for PDFs uses
        it to walk HTML pages for their links without paying to extract fields
        from a navigation menu.
        """
        started = time.perf_counter()
        extracted = False
        try:
            item = self._acquire(item, force_dynamic=force_dynamic)

            if item.ok:
                self._progress("Detecting content type", item)
                TypeRouter.route(item)

                self._progress(f"Reading {item.kind.value}", item)
                item = registry.dispatch(item)

            if item.ok and item.content_hash is None:
                item.compute_content_hash()

            wanted = item.ok and (extract_when is None or extract_when(item))
            if item.ok and not wanted:
                item.metadata["extraction_skipped"] = "not a target for this job"

            if wanted:
                extracted = True
                self._progress("Extracting fields", item)
                item = self.cascade.extract(
                    item, prompt, schema, local_only=local_only, allowed_tiers=allowed_tiers
                )

            if item.ok and extracted:
                self._progress("Normalizing and checking", item)
                item = DataNormalizer.normalize(item)

            if item.ok and extracted:
                self._validate(item, prompt, schema)
                self._check_duplicate(item)
                self._check_drift(item, schema)
                self._persist(item)

        except TransientError:
            raise  # the transport decides whether to retry
        except PipelineError as exc:
            item.fail_from(exc)
        except Exception as exc:
            log.exception("runner.crashed", url=item.url, error=repr(exc))
            item.fail(Stage.EXTRACT, f"unexpected error: {exc}", error_type=type(exc).__name__)

        # Children run whether or not the container itself extracted anything.
        # That is the normal case: a ZIP has no title and no price, and its
        # members are the entire reason it was submitted.
        return self._finish(item, report, started, children_only=follow_children,
                            prompt=prompt, schema=schema, local_only=local_only,
                            allowed_tiers=allowed_tiers)

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def _acquire(self, item: ExtractionItem, *, force_dynamic: bool) -> ExtractionItem:
        """Get the bytes — or, for a media platform, the transcript directly."""
        if item.raw_bytes is not None:
            # A child item: an archive member or an attachment already in hand.
            from models import FetchMode

            item.fetch_mode = FetchMode.INLINE
            return item

        decision = pre_route(item.url)
        item.metadata.setdefault("acquisition", decision.acquisition)

        if decision.acquisition == Acquisition.YTDLP:
            self._progress("Downloading and transcribing media", item)
            from pipeline.media_processor import MediaProcessor

            item.kind = ResourceKind.VIDEO
            item = MediaProcessor().process_media_url(item)
            # The transcript is the content, so the handler stage has nothing
            # left to do; mark it handled and let the cascade take over.
            item.handler = "media"
            return item

        if decision.acquisition == Acquisition.LIVESTREAM:
            self._progress("Capturing live stream", item)
            from pipeline.handlers.livestream import LivestreamHandler

            item.kind = ResourceKind.LIVESTREAM
            handler = LivestreamHandler(on_segment=self._segment_progress(item))
            return handler.handle(item)

        self._progress("Fetching", item)
        return self.fetcher.fetch(item, force_dynamic=force_dynamic)

    def _validate(self, item: ExtractionItem, prompt: str, schema: dict) -> None:
        if not config.VALIDATE_OUTPUT:
            return
        result = validate_record(item.normalized_data or {}, schema_hint=schema, prompt=prompt)
        item.validation_failures = result.errors
        for warning in result.warnings:
            item.warn(warning)

        if result.errors:
            log.warning("runner.validation_failed", url=item.url, errors=result.errors[:3])
            metrics.incr("validate.failed")
            if config.REJECT_ON_VALIDATION_FAILURE:
                item.fail(
                    Stage.VALIDATE,
                    f"validation rejected the record: {'; '.join(result.errors[:3])}",
                )
        else:
            metrics.incr("validate.ok")

    def _check_duplicate(self, item: ExtractionItem) -> None:
        if not config.DEDUPE_ENABLED or not item.ok:
            return
        text = item.text_for_extraction
        if not text.strip() or not item.content_hash:
            return

        verdict = self.dedupe.check(text, item.content_hash, item.canonical_url or item.url)
        if verdict.is_duplicate:
            item.metadata["duplicate_of"] = verdict.matched_url
            item.metadata["duplicate_kind"] = verdict.kind
            item.warn(
                f"{verdict.kind} duplicate of {verdict.matched_url} "
                f"(distance {verdict.distance})"
            )
            metrics.incr(f"dedupe.{verdict.kind}")
        else:
            fingerprint = self.dedupe.add(text, item.content_hash, item.canonical_url or item.url)
            item.metadata["simhash"] = str(fingerprint)

    def _check_drift(self, item: ExtractionItem, schema: dict) -> None:
        if not config.DRIFT_ENABLED or not item.ok:
            return
        alerts = self.drift.observe(
            item.url, item.metadata.get("schema_hash", ""), item.normalized_data or {}
        )
        for alert in alerts:
            item.warn(alert.message())

    def _persist(self, item: ExtractionItem) -> None:
        if self.db is None:
            return
        doc_id = self.db.save_item(item, run_id=item.job_id)
        if doc_id:
            item.metadata["mongo_id"] = doc_id
        raw_id = self.db.store_raw(item)
        if raw_id:
            item.metadata["raw_id"] = raw_id

    # ------------------------------------------------------------------ #
    # Children
    # ------------------------------------------------------------------ #

    def _run_children(
        self,
        item: ExtractionItem,
        report: Optional[RunReport],
        prompt: str,
        schema: dict,
        local_only: bool,
        allowed_tiers: Optional[set[int]],
    ) -> None:
        """Process children whose bytes are already in hand.

        The distinction that matters: an archive member and an email attachment
        arrive *with their bytes*, so processing them inline costs only CPU and
        re-fetching them is not even possible. A feed entry or a sitemap URL is
        only a link — following it inline would make one task perform hundreds
        of serial network fetches, blow through the time limit, and defeat the
        entire point of splitting the queues.

        So URL-only children are *discovered*, not followed. They are surfaced
        for the caller to enqueue as their own tasks, which is what puts them
        back through the rate limiter and across the worker pool.
        """
        if not item.children or item.depth >= config.MAX_RECURSION_DEPTH:
            return

        inline = [child for child in item.children if child.raw_bytes is not None]
        discovered = [child.url for child in item.children if child.raw_bytes is None]

        if discovered:
            item.metadata["discovered_urls"] = discovered[: config.MAX_SITEMAP_URLS]
            item.metadata["discovered_count"] = len(discovered)
            log.info("runner.discovered", url=item.url, urls=len(discovered))

        if not inline:
            return

        log.info("runner.recursing", url=item.url, children=len(inline), depth=item.depth)

        for child in inline:
            child.job_id = item.job_id
            child.canonical_url = child.canonical_url or canonicalize(child.url)
            try:
                self.run_item(
                    child,
                    prompt,
                    schema,
                    report=report,
                    local_only=local_only,
                    allowed_tiers=allowed_tiers,
                    follow_children=True,
                )
            except TransientError as exc:
                # A child's transient failure must not abort the parent's job;
                # the parent has already produced a usable record.
                child.fail_from(exc)
                log.warning("runner.child_failed", url=child.url, error=str(exc)[:200])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _finish(
        self,
        item: ExtractionItem,
        report: Optional[RunReport],
        started: float,
        *,
        children_only: bool = False,
        prompt: str = "",
        schema: Optional[dict] = None,
        local_only: bool = False,
        allowed_tiers: Optional[set[int]] = None,
    ) -> ExtractionItem:
        item.record_timing("total", time.perf_counter() - started)
        if report is not None:
            report.record(item)

        if children_only and item.children and schema is not None:
            self._run_children(item, report, prompt, schema, local_only, allowed_tiers)

        # Child items are kept as summaries: holding their raw bytes would mean
        # a 200-member archive sits in memory in full until the job ends.
        for child in item.children:
            child.raw_bytes = None
            child.decoded_text = None

        if not item.ok:
            log.warning(
                "runner.item_failed",
                url=item.url,
                stage=item.failed_at_stage.value if item.failed_at_stage else None,
                error=item.error,
            )
        else:
            log.info(
                "runner.item_ok",
                url=item.url,
                kind=item.kind.value,
                method=item.method.value,
                tier=item.tier,
                ms=item.timings_ms.get("total"),
            )
        return item

    def _progress(self, label: str, item: ExtractionItem) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(label, item)
            except Exception:  # progress reporting must never break the run
                pass

    def _segment_progress(self, item: ExtractionItem):
        """Report live-stream progress as each segment is transcribed."""
        if self.on_progress is None:
            return None

        def callback(index: int, transcript: str, seconds: float) -> None:
            self._progress(
                f"Live: {index + 1} segments captured ({seconds / 60:.0f} min, "
                f"{len(transcript)} chars)",
                item,
            )

        return callback


def run_once(
    url: str,
    prompt: str,
    schema: dict,
    *,
    database=None,
    **options: Any,
) -> ExtractionItem:
    """Convenience for scripts and tests: one URL in, one item out."""
    pipeline = Pipeline(database=database)
    item = ExtractionItem(url=url, canonical_url=canonicalize(url))
    return pipeline.run_item(item, prompt, schema, **options)


__all__ = ["Pipeline", "ProgressCallback", "run_once"]
