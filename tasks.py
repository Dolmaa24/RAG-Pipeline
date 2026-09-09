"""Celery tasks — thin wrappers around :class:`~pipeline.runner.Pipeline`.

The orchestration lives in the runner, so these do only what a transport should:
route to the right queue, report progress, decide about retries, and shape the
result for the API.

Retry policy is driven by the error taxonomy rather than by pattern-matching
messages. ``autoretry_for=(TransientError,)`` means a timeout or a 503 is retried
with exponential backoff, while a robots.txt denial, a 404 or an anti-bot
challenge fails immediately — because none of those get better on the fourth
attempt, and retrying a challenge harder is circumventing an access control.
"""

from __future__ import annotations

import gc
from typing import Any, Optional

from celery.exceptions import SoftTimeLimitExceeded

from celery_app import celery_app
from config import config
from database import CloudDatabase
from errors import TransientError
from models import ExtractionItem, ExtractionMethod, RunReport
from observability import configure_logging, get_logger, job_context, metrics
from pipeline.runner import Pipeline
from urls import canonicalize

log = get_logger("tasks")
configure_logging(config.LOG_LEVEL, config.LOG_FORMAT)

#: One database handle and one pipeline per worker process. Both hold caches
#: (specs, extractions, dedupe fingerprints) that are exactly what should be
#: shared across the tasks a worker runs.
db = CloudDatabase()
_pipeline: Optional[Pipeline] = None


def get_pipeline(on_progress=None) -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(database=db)
    _pipeline.on_progress = on_progress
    return _pipeline


RETRY_KWARGS = {
    "autoretry_for": (TransientError,),
    "retry_backoff": 5,
    "retry_backoff_max": 600,
    "retry_jitter": True,
    "max_retries": 3,
}


def _progress_reporter(task):
    """Report the current stage back to whoever is polling the task."""

    def report(label: str, item: ExtractionItem) -> None:
        task.update_state(
            state="PROGRESS",
            meta={
                "stage": label,
                "url": item.url,
                "kind": item.kind.value,
                "tier": item.tier,
            },
        )

    return report


def _result(report: RunReport, item: ExtractionItem) -> dict[str, Any]:
    payload = item.summary()
    payload["run"] = report.to_dict()
    if item.children:
        payload["children"] = [child.summary() for child in item.children]
    discovered = item.metadata.get("discovered_urls")
    if discovered:
        payload["discovered_urls"] = discovered
    return payload


def _fan_out(item: ExtractionItem, prompt: str, schema: dict, limit: int, **options: Any) -> list[str]:
    """Enqueue the URLs a feed or sitemap advertised as tasks of their own.

    Following them inside the parent task would make one job perform hundreds
    of serial fetches. Enqueuing puts each back through the router, the rate
    limiter and the queue split, which is what the queues are for.
    """
    urls = item.metadata.get("discovered_urls") or []
    if not urls:
        return []

    queued: list[str] = []
    for url in urls[:limit]:
        try:
            child = extract_url.delay(url, prompt, schema, follow_children=False, **options)
            queued.append(child.id)
        except Exception as exc:
            log.warning("task.fan_out_failed", url=url, error=repr(exc))
            break
    log.info("task.fanned_out", parent=item.url, queued=len(queued), available=len(urls))
    return queued


def _enqueue_index(
    item: ExtractionItem,
    *,
    enabled: Optional[bool],
    graph: Optional[bool] = None,
    supplied_metadata: Optional[dict] = None,
    skill: Optional[str] = None,
) -> Optional[str]:
    """Hand the finished text to the cpu queue for chunking and embedding.

    Indexing is a task rather than a stage for two reasons. Embedding is a
    transformer forward pass and this task runs on the io pool, where it would
    occupy a thread that sixteen fetches are queued behind. And an extraction
    that succeeded is a result the caller is owed — a vector store that is down,
    or a model that will not load, must not turn it into a failure.
    """
    if not (config.INDEX_ENABLED if enabled is None else enabled):
        return None
    if not item.ok:
        return None
    if item.metadata.get("duplicate_of"):
        # The runner already matched this text against the store. Indexing it
        # again under fresh chunk ids would put two copies in front of retrieval.
        log.info("task.index_skipped_duplicate", url=item.url)
        return None

    text = item.text_for_extraction
    if not text.strip():
        return None

    from pipeline.index.metadata import derive

    metadata = {
        "content_hash": item.content_hash or "",
        "extraction_tier": item.tier if item.tier is not None else -1,
        "job_id": item.job_id or "",
    }
    # doc_type, author and date come from what the pipeline already detected and
    # harvested; department, region and permission_level from the caller.
    metadata.update(derive(item, supplied=supplied_metadata))

    try:
        task = index_document.delay(
            text[: config.INDEX_MAX_TEXT_CHARS],
            source=item.canonical_url or item.url,
            metadata=metadata,
            build_graph=graph,
            skill=skill,
        )
    except Exception as exc:
        log.warning("task.index_enqueue_failed", url=item.url, error=repr(exc))
        return None

    log.info("task.index_queued", url=item.url, task_id=task.id)
    return task.id


@celery_app.task(bind=True, name="tasks.extract_url", **RETRY_KWARGS)
def extract_url(
    self,
    url: str,
    prompt: str,
    schema: dict,
    *,
    force_dynamic: bool = False,
    local_only: bool = False,
    allowed_tiers: Optional[list[int]] = None,
    follow_children: bool = True,
    fan_out: int = 0,
    index: Optional[bool] = None,
    build_graph: Optional[bool] = None,
    metadata: Optional[dict] = None,
    skill: Optional[str] = None,
) -> dict:
    """Extract structured data from any URL: page, document, feed, archive, media.

    ``fan_out`` is how a feed or sitemap becomes a crawl: up to that many of the
    URLs it advertises are enqueued as their own tasks. It defaults to 0, so one
    submitted URL means one job unless you ask for more.

    ``index`` overrides ``INDEX_ENABLED`` for this job: the extracted text is
    chunked, embedded and written to the vector store by a separate cpu-queue
    task. ``None`` means follow the configured default.

    ``metadata`` carries the filter fields nothing can infer — department,
    region, permission_level — onto every chunk this document produces.
    ``build_graph`` additionally extracts entities and relationships, which is
    one model call per chunk and therefore never implicit.
    """
    with job_context(self.request.id, url):
        try:
            pipeline = get_pipeline(_progress_reporter(self))
            report = RunReport(job_id=self.request.id)
            item = ExtractionItem(
                url=url, job_id=self.request.id, canonical_url=canonicalize(url)
            )
            item = pipeline.run_item(
                item,
                prompt,
                schema,
                report=report,
                force_dynamic=force_dynamic,
                local_only=local_only,
                allowed_tiers=set(allowed_tiers) if allowed_tiers else None,
                follow_children=follow_children,
            )
            report.metrics = metrics.snapshot()
            report.finish()
            db.save_run(report)

            result = _result(report, item)

            index_task_id = _enqueue_index(
                item,
                enabled=index,
                graph=build_graph,
                supplied_metadata=metadata,
                skill=skill,
            )
            if index_task_id:
                item.metadata["index_task_id"] = index_task_id
                result["index_task_id"] = index_task_id

            if fan_out > 0:
                result["fanned_out"] = _fan_out(
                    item, prompt, schema, fan_out,
                    local_only=local_only, allowed_tiers=allowed_tiers,
                )

            # A container that discovered URLs did its job even if the container
            # itself held nothing matching the schema — a sitemap has no title.
            produced_something = (
                item.ok or item.children or item.metadata.get("discovered_urls")
            )
            if not produced_something:
                # A hard failure is reported as a failed task so the caller's
                # error handling fires, rather than a 200 with an error field.
                raise RuntimeError(f"[{item.failed_at_stage}] {item.error}")

            return result
        except SoftTimeLimitExceeded:
            log.error("task.soft_timeout", url=url, task="extract_url")
            raise
        finally:
            gc.collect()


@celery_app.task(bind=True, name="tasks.extract_media", **RETRY_KWARGS)
def extract_media(self, url: str, prompt: str, schema: dict, *, local_only: bool = False) -> dict:
    """Download and transcribe audio or video, then extract from the transcript."""
    with job_context(self.request.id, url):
        try:
            pipeline = get_pipeline(_progress_reporter(self))
            report = RunReport(job_id=self.request.id)
            item = ExtractionItem(
                url=url, job_id=self.request.id, canonical_url=canonicalize(url)
            )
            item = pipeline.run_item(item, prompt, schema, report=report, local_only=local_only)
            report.metrics = metrics.snapshot()
            report.finish()
            db.save_run(report)

            if not item.ok:
                raise RuntimeError(f"[{item.failed_at_stage}] {item.error}")
            return _result(report, item)
        finally:
            gc.collect()


@celery_app.task(
    bind=True,
    name="tasks.capture_livestream",
    # A live capture is meant to run for as long as the stream does, so it gets
    # its own limits rather than the global 30-minute cap. The soft limit fires
    # first and lets the handler return everything transcribed so far.
    soft_time_limit=config.LIVESTREAM_MAX_MINUTES * 60 + 120,
    time_limit=config.LIVESTREAM_MAX_MINUTES * 60 + 300,
    max_retries=0,  # a stream you reconnect to is a different stream
)
def capture_livestream(self, url: str, prompt: str, schema: dict) -> dict:
    """Capture a live stream in rolling segments, transcribing as it plays."""
    with job_context(self.request.id, url):
        try:
            pipeline = get_pipeline(_progress_reporter(self))
            report = RunReport(job_id=self.request.id)
            item = ExtractionItem(
                url=url, job_id=self.request.id, canonical_url=canonicalize(url)
            )
            item = pipeline.run_item(item, prompt, schema, report=report)
            report.metrics = metrics.snapshot()
            report.finish()
            db.save_run(report)
            if not item.ok:
                raise RuntimeError(f"[{item.failed_at_stage}] {item.error}")
            return _result(report, item)
        except SoftTimeLimitExceeded:
            # Everything transcribed so far is already on the item and saved by
            # the runner, so a timeout is a truncated success, not a loss.
            log.warning("task.livestream_time_limit", url=url)
            raise
        finally:
            gc.collect()


@celery_app.task(bind=True, name="tasks.extract_batch")
def extract_batch(self, urls: list[str], prompt: str, schema: dict, **options: Any) -> dict:
    """Run many URLs in one task, returning a single :class:`RunReport`.

    Useful when the URLs are small and numerous — a sitemap's worth of pages —
    because it amortises the spec and cache lookups across them. For long or
    heterogeneous work, enqueue ``extract_url`` per URL instead so the queues
    can spread it.
    """
    with job_context(self.request.id):
        pipeline = get_pipeline(_progress_reporter(self))
        report = pipeline.run_batch(urls, prompt, schema, job_id=self.request.id, **options)
        report.metrics = metrics.snapshot()
        db.save_run(report)
        return report.to_dict()


@celery_app.task(bind=True, name="tasks.crawl_site")
def crawl_site(
    self,
    start_url: str,
    prompt: str,
    schema: dict,
    *,
    scope_options: Optional[dict] = None,
) -> dict:
    """Start a crawl. Returns immediately with a crawl id to poll.

    This task only seeds the frontier and enqueues the first pages — the crawl
    itself happens across ``crawl_page`` tasks, so it spreads over the worker
    pool and no single task has to finish a whole site inside a time limit.
    """
    from pipeline.discover import CrawlScope, describe_plan, get_frontier, seed_urls

    scope = CrawlScope.build(start_url, **(scope_options or {}))
    crawl_id = self.request.id

    with job_context(crawl_id, start_url):
        frontier = get_frontier(
            crawl_id, start_url=start_url, max_pages=scope.max_pages, scope=scope.to_dict()
        )
        seeds = seed_urls(scope)
        claimed = frontier.claim((url, 0) for url in seeds)

        for url, depth in claimed:
            frontier.bump("in_flight")
            crawl_page.apply_async(
                args=[crawl_id, url, depth, prompt, schema, scope.to_dict()],
                queue=config.IO_QUEUE,
            )

        plan = describe_plan(scope)
        log.info("crawl.started", crawl_id=crawl_id, seeds=len(claimed), plan=plan)
        return {
            "crawl_id": crawl_id,
            "plan": plan,
            "scope": scope.to_dict(),
            "seeded": len(claimed),
            "poll": f"/api/v1/crawls/{crawl_id}",
        }


@celery_app.task(
    bind=True,
    name="tasks.crawl_page",
    autoretry_for=(TransientError,),
    retry_backoff=5,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=2,
    soft_time_limit=300,
    time_limit=360,
)
def crawl_page(
    self,
    crawl_id: str,
    url: str,
    depth: int,
    prompt: str,
    schema: dict,
    scope_dict: dict,
) -> dict:
    """Fetch one page: extract it if it is a target, then enqueue what it links to.

    Runs on the io queue, because for a crawl looking for files this is almost
    pure network waiting. Targets it discovers are handed to ``extract_url`` on
    the **cpu** queue instead — parsing a PDF, running OCR or calling a model is
    exactly the work that must not occupy an io thread.
    """
    from pipeline.discover import CrawlScope, get_frontier, harvest_links, sitemap_urls

    scope = CrawlScope.from_dict(scope_dict)
    frontier = get_frontier(crawl_id)

    with job_context(crawl_id, url):
        if frontier.stop_requested():
            frontier.bump("in_flight", -1)
            return {"url": url, "status": "stopped"}

        try:
            pipeline = get_pipeline(None)
            report = RunReport(job_id=crawl_id)
            item = ExtractionItem(
                url=url, job_id=crawl_id, depth=depth, canonical_url=canonicalize(url)
            )
            item = pipeline.run_item(
                item,
                prompt,
                schema,
                report=report,
                # The authoritative call, made from the magic bytes rather than
                # from the URL: walk pages for their links, extract only targets.
                extract_when=lambda parsed: scope.should_extract(parsed.kind, parsed.url),
                follow_children=False,
            )

            frontier.bump("fetched")

            if not item.ok:
                frontier.bump("failed")
                log.info("crawl.page_failed", url=url, error=item.error)

            # Crucially, this is *not* an early return. A page whose extraction
            # failed — a rate-limited model, a schema nothing matched — was
            # still fetched and parsed, and its links are just as good. Treating
            # an extraction failure as a dead branch lets one 429 on the seed
            # page silently end the entire crawl.
            if item.parsed_tree is None and not item.structured:
                return {"url": url, "status": "failed", "error": item.error}

            if item.ok and item.method is not ExtractionMethod.NONE:
                frontier.bump("collected")
                frontier.record_target(url)

            # A sitemap or feed reached during a crawl is a gift: a ready-made
            # list of exactly the URLs the site wants visited.
            discovered = sitemap_urls(item, scope) if item.structured else []
            harvest = harvest_links(item, scope, depth)

            follow = [(link, depth + 1) for link in harvest.follow]
            follow.extend((link, depth) for link in discovered)
            collect = [(link, depth) for link in harvest.collect]

            queued_pages = _enqueue_pages(frontier, crawl_id, follow, prompt, schema, scope_dict)
            queued_files = _enqueue_targets(frontier, collect, prompt, schema)

            log.info(
                "crawl.page_done",
                url=url,
                depth=depth,
                extracted=item.method is not ExtractionMethod.NONE,
                pages_queued=queued_pages,
                files_queued=queued_files,
            )
            return {
                "url": url,
                "status": "ok" if item.ok else "walked",
                "kind": item.kind.value,
                "extracted": item.ok and item.method is not ExtractionMethod.NONE,
                "extraction_error": None if item.ok else item.error,
                "pages_queued": queued_pages,
                "files_queued": queued_files,
                "skipped": harvest.summary().get("skipped", 0),
            }
        finally:
            frontier.bump("in_flight", -1)
            remaining = frontier.state()
            if remaining.in_flight <= 0:
                frontier.finish("finished")
                log.info(
                    "crawl.finished",
                    crawl_id=crawl_id,
                    fetched=remaining.fetched,
                    collected=remaining.collected,
                )
            gc.collect()


def _enqueue_pages(frontier, crawl_id: str, urls, prompt, schema, scope_dict) -> int:
    """Claim URLs and enqueue a crawl_page for each newly-accepted one.

    The claim is what makes the crawl terminate: a URL already in the frontier
    is refused, so a site whose every page links to every other page is still
    visited once per page rather than once per link.
    """
    claimed = frontier.claim(urls)
    for url, depth in claimed:
        frontier.bump("in_flight")
        crawl_page.apply_async(
            args=[crawl_id, url, depth, prompt, schema, scope_dict], queue=config.IO_QUEUE
        )
    return len(claimed)


def _enqueue_targets(frontier, urls, prompt, schema) -> int:
    """Hand discovered files to the ordinary extraction task, on the cpu queue.

    A found PDF needs the full pipeline — parse, maybe OCR, maybe a model — and
    that is precisely the work the io queue must stay free of.
    """
    claimed = frontier.claim(urls)
    for url, _ in claimed:
        frontier.record_target(url)
        frontier.bump("collected")
        extract_url.apply_async(
            args=[url, prompt, schema],
            kwargs={"follow_children": True},
            queue=config.CPU_QUEUE,
        )
    return len(claimed)


@celery_app.task(bind=True, name="tasks.discover_sitemap", **RETRY_KWARGS)
def discover_sitemap(self, url: str) -> dict:
    """Fetch a sitemap or robots.txt and return the URLs it advertises.

    One request against ``/robots.txt`` or ``/sitemap.xml`` can hand you a
    site's whole URL inventory, which is otherwise a week of crawling to find.
    """
    from pipeline.compliance import robots_gate
    from pipeline.detect import TypeRouter
    from pipeline.fetch import ResilientFetcher
    from pipeline.handlers import registry

    with job_context(self.request.id, url):
        sitemaps = list(robots_gate.sitemaps(url))
        item = ResilientFetcher().fetch(ExtractionItem(url=url))
        urls: list[dict] = []
        if item.ok:
            TypeRouter.route(item)
            item = registry.dispatch(item)
            if item.ok and item.structured:
                urls = item.structured.get("sitemap_urls") or item.structured.get("feed_entries") or []

        return {
            "url": url,
            "advertised_sitemaps": sitemaps,
            "urls": urls,
            "count": len(urls),
            "error": item.error,
        }


@celery_app.task(bind=True, name="tasks.index_document", **RETRY_KWARGS)
def index_document(
    self,
    text: str,
    *,
    source: str,
    metadata: Optional[dict] = None,
    strategy: Optional[str] = None,
    local_only: bool = False,
    build_graph: Optional[bool] = None,
    skill: Optional[str] = None,
) -> dict:
    """Chunk, embed and store one document's text.

    Enqueued by :func:`extract_url` after a successful extraction, and callable
    on its own for text that did not come from a crawl. Thin, like every task
    here: the work is in :func:`pipeline.index.index_text`.
    """
    with job_context(self.request.id, source):
        try:
            from pipeline.index import index_text

            report = index_text(
                text,
                source=source,
                extra_metadata=metadata or {},
                strategy=strategy,
                local_only=local_only,
            )
            payload = report.to_dict()

            if config.GRAPH_ENABLED if build_graph is None else build_graph:
                payload["graph"] = _build_graph(
                    text, source, metadata, local_only, skill
                )
            return payload
        except SoftTimeLimitExceeded:
            log.error("task.soft_timeout", url=source, task="index_document")
            raise
        finally:
            gc.collect()


@celery_app.task(bind=True, name="tasks.search")
def search(
    self,
    query: str,
    *,
    filters: Optional[dict] = None,
    limit: Optional[int] = None,
    fusion: Optional[str] = None,
    alpha: Optional[float] = None,
    use_graph: Optional[bool] = None,
    rerank: Optional[bool] = None,
    rewrite: Optional[bool] = None,
    local_only: bool = False,
) -> dict:
    """Retrieve evidence for a question.

    On the cpu queue: it embeds the query and may load a reranker, and it must
    not do either inside the io pool.
    """
    with job_context(self.request.id, query[:80]):
        try:
            from pipeline.retrieve import MetadataFilter, retrieve

            result = retrieve(
                query,
                filters=MetadataFilter(**filters) if filters else None,
                limit=limit,
                fusion=fusion,
                alpha=alpha,
                use_graph=use_graph,
                rerank_results=rerank,
                rewrite=rewrite,
                local_only=local_only,
            )
            return result.to_dict()
        except SoftTimeLimitExceeded:
            log.error("task.soft_timeout", query=query[:80], task="search")
            raise
        finally:
            gc.collect()


def _build_graph(
    text: str,
    source: str,
    metadata: Optional[dict],
    local_only: bool,
    skill: Optional[str] = None,
) -> dict:
    """Fold this document into the knowledge graph.

    Runs inside the indexing task rather than as one of its own, because it
    needs the same text and the same worker already has the embedder resident.
    A failure here is reported, not raised: the chunks are already stored and
    are useful without a graph.
    """
    try:
        from pipeline.graph.builder import build_graph

        report = build_graph(
            text,
            source_url=source,
            content_hash=str((metadata or {}).get("content_hash", "")),
            local_only=local_only,
            skill=skill,
            database=db,
        )
        return report.to_dict()
    except Exception as exc:
        log.warning("task.graph_failed", source=source[:80], error=repr(exc))
        return {"error": repr(exc)}


@celery_app.task(bind=True, name="tasks.playground_reply")
def playground_reply(
    self,
    thread_id: str,
    question: str,
    *,
    local_only: bool = False,
) -> dict:
    """Answer the newest message in a thread, and store the reply.

    No ``RETRY_KWARGS``, like the other agent tasks: a retry would spend the
    same model calls again for the same reason, and here it would also append a
    second assistant message to a conversation that already has one.

    The user's message is already stored — the API writes it before queuing, so
    it is visible the moment it is sent rather than when the agent finishes.
    """
    with job_context(self.request.id, f"thread {thread_id[:8]}"):
        from playground import threads

        def report(event: dict) -> None:
            self.update_state(state="PROGRESS", meta=event)

        try:
            return threads.reply(
                thread_id, question, local_only=local_only, on_progress=report
            ).to_dict()
        except SoftTimeLimitExceeded:
            log.error("task.soft_timeout", thread=thread_id, task="playground_reply")
            raise
        finally:
            gc.collect()


@celery_app.task(bind=True, name="tasks.build_project")
def build_project(
    self,
    skill: str,
    *,
    intent: str = "",
    allow_execute: bool = False,
) -> dict:
    """Run a skill's roster over a sandboxed project directory.

    No ``RETRY_KWARGS``, for the same reason an investigation has none, and one
    more: a retry would run the workers again over files the first attempt
    already wrote, and a half-finished build re-entered is harder to read than
    a failed one.

    ``allow_execute`` is the second gate. Writing source and running it are
    separate effects, so a build that produces a scaffold for a person to read
    is the default and executing it is asked for.
    """
    with job_context(self.request.id, f"build {skill}"):
        from pipeline.agents.budget import Budget
        from pipeline.agents.builder import Builder
        from pipeline.skills import get as get_skill

        try:
            chosen = get_skill(skill)
        except Exception as exc:
            log.warning("task.build_unknown_skill", skill=skill, error=repr(exc))
            return {"skill": skill, "stopped": str(exc), "files": [], "steps": []}

        base = Budget.from_config()
        budget = Budget(
            max_iterations=base.max_iterations,
            max_tool_calls=base.max_tool_calls,
            # A build is many model calls writing whole files, so the
            # ninety seconds an investigation gets is not the right ceiling.
            max_seconds=base.max_seconds * 6,
            max_tokens=base.max_tokens,
            code_calls=config.BUILD_CODE_CALLS,
            execute_calls=config.BUILD_EXECUTE_CALLS if allow_execute else 0,
        )

        def report(event: dict) -> None:
            self.update_state(state="PROGRESS", meta=event)

        try:
            result = Builder(
                chosen, intent=intent, budget=budget, on_progress=report
            ).build()
            return result.to_dict()
        except SoftTimeLimitExceeded:
            log.error("task.soft_timeout", skill=skill, task="build_project")
            raise
        finally:
            gc.collect()


@celery_app.task(bind=True, name="tasks.investigate")
def investigate(
    self,
    question: str,
    *,
    allow_network: bool = False,
    allow_write: bool = False,
    max_rounds: Optional[int] = None,
    verify: Optional[bool] = None,
    local_only: bool = False,
    skill: Optional[str] = None,
) -> dict:
    """Answer one question with the agent loop, and report the reasoning.

    No ``RETRY_KWARGS``. Retrying is for work whose failure was the network's
    fault; an investigation that failed has already spent its budget on model
    calls, and running it again would spend the same again for the same reason.

    ``allow_network`` and ``allow_write`` become the budget's counters, which
    are also its permission gate — an unbudgeted effect is not merely capped,
    its tools are absent from what the model is shown.

    ``skill`` names a domain pack to run as the gathering specialist. A name
    rather than the composed role: the worker reads the same folder the API
    read, so nothing skill-shaped crosses the broker and an edited skill file
    does not have to match a queued job. An unknown name falls back to the
    generic specialist — a queued job should not be lost to a renamed file.
    """
    with job_context(self.request.id, question[:80]):
        from pipeline.agents.budget import Budget
        from pipeline.agents.supervisor import Supervisor

        base = Budget.from_config()
        budget = Budget(
            max_iterations=base.max_iterations,
            max_tool_calls=base.max_tool_calls,
            max_seconds=base.max_seconds,
            max_tokens=base.max_tokens,
            network_calls=config.AGENT_NETWORK_CALLS if allow_network else 0,
            write_calls=config.AGENT_WRITE_CALLS if allow_write else 0,
        )

        def report(event: dict) -> None:
            self.update_state(state="PROGRESS", meta=event)

        role = None
        if skill:
            try:
                from pipeline.skills import get as get_skill

                role = get_skill(skill).as_role()
            except Exception as exc:
                log.warning("task.skill_unavailable", skill=skill, error=repr(exc))

        try:
            result = Supervisor(
                budget=budget,
                local_only=local_only,
                max_rounds=max_rounds,
                verify_answer=verify,
                role=role,
                on_progress=report,
            ).investigate(question)
            payload = result.to_dict()
            payload["skill"] = role.name if role else None
            return payload
        except SoftTimeLimitExceeded:
            log.error("task.soft_timeout", task="tasks.investigate")
            raise
        finally:
            gc.collect()


@celery_app.task(bind=True, name="tasks.process_web_scrape", **RETRY_KWARGS)
def process_web_scrape_task(
    self, url: str, prompt: str, schema: dict, force_dynamic: bool = False
) -> dict:
    """Original web task name, kept so existing clients keep working."""
    return _run_inline(self, url, prompt, schema, force_dynamic=force_dynamic)


@celery_app.task(bind=True, name="tasks.process_media_scrape", **RETRY_KWARGS)
def process_media_scrape_task(self, media_url: str, prompt: str, schema: dict) -> dict:
    """Original media task name, kept so existing clients keep working."""
    return _run_inline(self, media_url, prompt, schema)


def _run_inline(task, url: str, prompt: str, schema: dict, **options: Any) -> dict:
    """Shared body for the legacy task names.

    Calling another task's ``.run()`` would bind the wrong ``self`` — the
    progress updates and retries would attach to the wrong request — so the
    pipeline is invoked directly instead.
    """
    with job_context(task.request.id, url):
        try:
            pipeline = get_pipeline(_progress_reporter(task))
            report = RunReport(job_id=task.request.id)
            item = ExtractionItem(
                url=url, job_id=task.request.id, canonical_url=canonicalize(url)
            )
            item = pipeline.run_item(item, prompt, schema, report=report, **options)
            report.metrics = metrics.snapshot()
            report.finish()
            db.save_run(report)
            if not item.ok:
                raise RuntimeError(f"[{item.failed_at_stage}] {item.error}")

            result = _result(report, item)
            # The original response shape, for clients that read these keys.
            result.setdefault("type", item.kind.value)
            result.setdefault("mongo_id", item.metadata.get("mongo_id"))
            return result
        finally:
            gc.collect()


__all__ = [
    "capture_livestream",
    "discover_sitemap",
    "extract_batch",
    "extract_media",
    "extract_url",
    "index_document",
    "search",
    "process_media_scrape_task",
    "process_web_scrape_task",
]
