"""The HTTP API.

One endpoint does the work: ``POST /api/v1/extract`` takes any URL — a page, a
PDF, a spreadsheet, an RSS feed, a ZIP, a podcast, a live stream — and queues it
onto the right queue. The caller does not say which kind it is, because the
caller usually does not know and the URL usually does not say.

The rest of the surface exists to make the system inspectable: which tier
answered, what a domain's learned selectors look like, what the pipeline *would*
do with a URL before you commit to running it.
"""

from __future__ import annotations

from typing import Any, Optional

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import shutil
import uuid
from pathlib import Path

from celery_app import celery_app
from config import config
from database import CloudDatabase
from observability import configure_logging, get_logger
from pipeline.detect.router import Acquisition, pre_route
from urls import canonicalize, is_http_url

configure_logging(config.LOG_LEVEL, config.LOG_FORMAT)
log = get_logger("api")

app = FastAPI(
    title="Universal Extraction API",
    version=config.PIPELINE_VERSION,
    description=(
        "Point it at anything on the web and get structured JSON back. "
        "Type detection is by magic bytes, and extraction runs a cascade that "
        "reaches a language model only when the cheaper tiers cannot answer."
    ),
)

db = CloudDatabase()

UPLOAD_DIR = Path("output/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.post("/api/v1/upload", tags=["extract"])
def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "bin"
    file_id = f"{uuid.uuid4().hex}.{ext}"
    dest = UPLOAD_DIR / file_id
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"url": f"http://127.0.0.1:8000/uploads/{file_id}"}


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class ExtractionRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Any http(s) URL.")
    prompt: str = Field(..., min_length=1, description="What to extract, in plain language.")
    schema_template: dict = Field(
        ...,
        description='Field names to types, e.g. {"title": "string", "tags": "list of strings"}.',
    )
    force_dynamic: bool = Field(False, description="Skip the static fetch, render with a browser.")
    local_only: bool = Field(
        False, description="Never send this content to a hosted model."
    )
    follow_children: bool = Field(
        True, description="Recurse into archive members, feed entries and attachments."
    )
    allowed_tiers: Optional[list[int]] = Field(
        None,
        description="Restrict the cascade, e.g. [1] for structured data only, [3] to force the model.",
    )
    fan_out: int = Field(
        0,
        ge=0,
        le=5000,
        description=(
            "For a feed or sitemap: enqueue up to this many of the URLs it advertises "
            "as jobs of their own. 0 means discover them but do not follow them."
        ),
    )
    index: Optional[bool] = Field(
        None,
        description=(
            "Chunk, embed and store the extracted text for retrieval. Runs as a "
            "separate task on the cpu queue. null follows INDEX_ENABLED."
        ),
    )

    @field_validator("url")
    @classmethod
    def _must_be_http(cls, value: str) -> str:
        if not is_http_url(value.strip()):
            raise ValueError("url must start with http:// or https://")
        return value.strip()

    @field_validator("schema_template")
    @classmethod
    def _non_empty_schema(cls, value: dict) -> dict:
        if not value:
            raise ValueError("schema_template must name at least one field")
        return value


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=1000)
    prompt: str = Field(..., min_length=1)
    schema_template: dict
    local_only: bool = False


class CrawlRequest(BaseModel):
    """Walk a site and extract from what it finds.

    The shape of the job is decided by ``collect_extensions``. Set it and the
    crawl is a file hunt: pages are walked for their links but never extracted,
    and only matching files are. Leave it empty and every in-scope page is
    extracted instead.
    """

    start_url: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    schema_template: dict

    collect_extensions: list[str] = Field(
        default_factory=list,
        description='File types to collect, e.g. ["pdf"] or ["pdf","xlsx"]. Empty = every page.',
        examples=[["pdf"]],
    )
    max_depth: int = Field(2, ge=0, le=10, description="Link hops from the start URL.")
    max_pages: int = Field(500, ge=1, le=50_000, description="Hard cap on URLs claimed.")
    same_site: bool = Field(True, description="Stay on the start URL's registrable host.")
    allowed_hosts: list[str] = Field(default_factory=list)
    include_patterns: list[str] = Field(
        default_factory=list, description="Regexes a URL must match to be collected."
    )
    exclude_patterns: list[str] = Field(
        default_factory=list, description="Regexes that exclude a URL entirely."
    )
    follow_html: bool = True

    @field_validator("start_url")
    @classmethod
    def _http_only(cls, value: str) -> str:
        if not is_http_url(value.strip()):
            raise ValueError("start_url must start with http:// or https://")
        return value.strip()

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def _valid_regex(cls, value: list[str]) -> list[str]:
        import re

        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return value

    def scope_options(self) -> dict:
        return {
            "collect_extensions": self.collect_extensions,
            "allowed_hosts": self.allowed_hosts,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "same_site": self.same_site,
            "follow_html": self.follow_html,
        }


# --------------------------------------------------------------------------- #
# Health and introspection
# --------------------------------------------------------------------------- #


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Liveness, plus whether each dependency is actually reachable."""
    try:
        workers = celery_app.control.ping(timeout=1.0) or []
    except Exception as exc:
        log.warning("api.worker_ping_failed", error=repr(exc))
        workers = []

    queues: dict[str, int] = {}
    try:
        import redis

        client = redis.Redis.from_url(config.REDIS_URL)
        for queue in (config.IO_QUEUE, config.CPU_QUEUE):
            queues[queue] = int(client.llen(queue))
        redis_ok = True
    except Exception:
        redis_ok = False

    from pipeline.extract.llm import status as llm_status

    return {
        "status": "ok",
        "version": config.PIPELINE_VERSION,
        "workers_online": len(workers),
        "worker_names": [name for worker in workers for name in worker],
        "queue_depth": queues,
        "redis": redis_ok,
        "mongo": db.ping() if db.is_configured else "not configured",
        "llm": llm_status(),
        "robots_enforced": config.RESPECT_ROBOTS,
    }


@app.get("/api/v1/detect", tags=["ops"])
def detect_url(url: str = Query(..., description="URL to classify without fetching it")) -> dict:
    """What the pipeline would do with this URL, before committing to it."""
    if not is_http_url(url):
        raise HTTPException(400, "url must start with http:// or https://")

    from pipeline.compliance import policy

    decision = pre_route(url)
    verdict = policy.check(url)
    return {
        "url": url,
        "canonical_url": canonicalize(url),
        "acquisition": decision.acquisition,
        "reason": decision.reason,
        "likely_kind": decision.likely_kind.value,
        "allowed": verdict.allowed,
        "policy_reason": verdict.reason,
        "crawl_delay": verdict.crawl_delay,
    }


@app.get("/api/v1/stats", tags=["ops"])
def stats(days: int = Query(7, ge=1, le=90)) -> dict:
    """Which tier answered, over the last N days. The cascade's report card."""
    from pipeline.extract.cascade import get_cascade
    from pipeline.fetch import breaker, http_cache, rate_limiter

    breakdown = db.method_breakdown(days)
    total = sum(breakdown.values())
    llm_calls = breakdown.get("llm", 0)
    return {
        "window_days": days,
        "records": total,
        "by_method": breakdown,
        "llm_avoidance_rate": round(1 - llm_calls / total, 3) if total else None,
        "extraction_cache": get_cascade(db).cache.stats(),
        "http_cache": http_cache.stats(),
        "rate_limiter": rate_limiter.stats(),
        "circuit_breaker": breaker.stats(),
    }


@app.get("/api/v1/specs", tags=["ops"])
def selector_specs(limit: int = Query(50, ge=1, le=500)) -> dict:
    """The per-domain selector specs the pipeline has learned so far."""
    from pipeline.extract.cascade import get_cascade

    specs = get_cascade(db).specs.all(limit)
    return {
        "count": len(specs),
        "specs": [
            {
                "domain": spec.get("domain"),
                "path_prefix": spec.get("path_prefix"),
                "fields": list((spec.get("rules") or {}).keys()),
                "rules": spec.get("rules"),
                "uses": spec.get("uses"),
                "avg_fill_rate": spec.get("avg_fill_rate"),
                "retired": spec.get("retired", False),
                "learned_from": spec.get("learned_from"),
            }
            for spec in specs
        ],
    }


@app.get("/api/v1/records", tags=["data"])
def records(
    limit: int = Query(25, ge=1, le=200),
    domain: Optional[str] = Query(None),
) -> dict:
    return {"records": db.recent(limit, domain=domain)}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


@app.post("/api/v1/extract", tags=["extract"], status_code=202)
def extract(request: ExtractionRequest) -> dict:
    """Queue any URL for extraction. The pipeline works out what it is."""
    decision = pre_route(request.url)

    from tasks import capture_livestream, extract_media, extract_url

    kwargs = {
        "local_only": request.local_only,
        "allowed_tiers": request.allowed_tiers,
    }

    try:
        if decision.acquisition == Acquisition.LIVESTREAM:
            task = capture_livestream.delay(request.url, request.prompt, request.schema_template)
        elif decision.acquisition == Acquisition.YTDLP:
            task = extract_media.delay(
                request.url, request.prompt, request.schema_template,
                local_only=request.local_only,
            )
        else:
            task = extract_url.delay(
                request.url,
                request.prompt,
                request.schema_template,
                force_dynamic=request.force_dynamic,
                follow_children=request.follow_children,
                fan_out=request.fan_out,
                index=request.index,
                **kwargs,
            )
    except Exception as exc:
        # Almost always "Redis is not running".
        log.error("api.enqueue_failed", url=request.url, error=repr(exc))
        raise HTTPException(
            503, f"could not reach the task broker. Is Redis running? ({exc})"
        ) from exc

    log.info("api.queued", url=request.url, task_id=task.id, acquisition=decision.acquisition)
    return {
        "status": "queued",
        "task_id": task.id,
        "acquisition": decision.acquisition,
        "queue": config.CPU_QUEUE if decision.acquisition != Acquisition.HTTP else config.IO_QUEUE,
        "poll": f"/api/v1/tasks/{task.id}",
    }


@app.post("/api/v1/extract/auto", tags=["extract"], status_code=202)
def extract_auto(request: ExtractionRequest) -> dict:
    """Original endpoint name, kept so existing clients keep working."""
    result = extract(request)
    result["detected_type"] = "media" if result["acquisition"] != Acquisition.HTTP else "web"
    return result


@app.post("/api/v1/extract/batch", tags=["extract"], status_code=202)
def extract_batch_endpoint(request: BatchRequest) -> dict:
    from tasks import extract_batch

    invalid = [url for url in request.urls if not is_http_url(url)]
    if invalid:
        raise HTTPException(400, f"{len(invalid)} URL(s) are not http(s): {invalid[:3]}")

    try:
        task = extract_batch.delay(
            request.urls, request.prompt, request.schema_template, local_only=request.local_only
        )
    except Exception as exc:
        raise HTTPException(503, f"could not reach the task broker ({exc})") from exc

    return {"status": "queued", "task_id": task.id, "urls": len(request.urls)}


@app.post("/api/v1/discover", tags=["extract"], status_code=202)
def discover(url: str = Query(..., description="A sitemap, feed, or site root")) -> dict:
    """Ask a site for its URL inventory via robots.txt and its sitemap."""
    from tasks import discover_sitemap

    if not is_http_url(url):
        raise HTTPException(400, "url must start with http:// or https://")
    task = discover_sitemap.delay(url)
    return {"status": "queued", "task_id": task.id}


# --------------------------------------------------------------------------- #
# Crawling
# --------------------------------------------------------------------------- #


@app.post("/api/v1/crawl", tags=["crawl"], status_code=202)
def start_crawl(request: CrawlRequest) -> dict:
    """Walk a site, extracting from the pages or files that match the scope."""
    from pipeline.discover import CrawlScope, describe_plan
    from tasks import crawl_site

    scope = CrawlScope.build(request.start_url, **request.scope_options())
    if not scope.same_site and not scope.allowed_hosts:
        raise HTTPException(
            400,
            "same_site=false needs allowed_hosts — an unbounded crawl of the open "
            "web is not something to start by accident.",
        )

    from pipeline.compliance import policy

    verdict = policy.check(request.start_url)
    if not verdict.allowed:
        raise HTTPException(403, f"the start URL is not fetchable: {verdict.reason}")

    try:
        task = crawl_site.delay(
            request.start_url,
            request.prompt,
            request.schema_template,
            scope_options=request.scope_options(),
        )
    except Exception as exc:
        raise HTTPException(503, f"could not reach the task broker ({exc})") from exc

    log.info("api.crawl_queued", url=request.start_url, crawl_id=task.id)
    return {
        "status": "queued",
        "crawl_id": task.id,
        "plan": describe_plan(scope),
        "poll": f"/api/v1/crawls/{task.id}",
    }


@app.get("/api/v1/crawls/{crawl_id}", tags=["crawl"])
def crawl_status(crawl_id: str, targets: int = Query(200, ge=0, le=5000)) -> dict:
    """Live counters for a crawl, plus the files it has found so far."""
    from pipeline.discover import get_frontier

    frontier = get_frontier(crawl_id)
    state = frontier.state()
    payload = state.to_dict()
    if targets:
        found = frontier.targets(targets)
        payload["targets"] = found
        payload["targets_shown"] = len(found)
    return payload


@app.delete("/api/v1/crawls/{crawl_id}", tags=["crawl"])
def stop_crawl(crawl_id: str) -> dict:
    """Ask a crawl to stop.

    Sets a flag the workers check before each page rather than revoking the
    queued tasks, because Celery does not revoke reliably once a task has been
    prefetched. Pages already in flight finish; nothing new is claimed.
    """
    from pipeline.discover import get_frontier

    frontier = get_frontier(crawl_id)
    frontier.finish("stopped")
    log.info("api.crawl_stopped", crawl_id=crawl_id)
    return {"crawl_id": crawl_id, "status": "stopping"}


# --------------------------------------------------------------------------- #
# Task status
# --------------------------------------------------------------------------- #


@app.get("/api/v1/tasks/{task_id}", tags=["extract"])
def task_status(task_id: str) -> dict:
    result = AsyncResult(task_id, app=celery_app)
    status = result.status
    response: dict[str, Any] = {"task_id": task_id, "status": status}

    if status == "PROGRESS":
        info = result.info if isinstance(result.info, dict) else {}
        response.update({"stage": info.get("stage", "Processing"), **info})
    elif status == "SUCCESS":
        response["result"] = result.result
    elif status == "FAILURE":
        response["error"] = str(result.info)
        response["traceback"] = (result.traceback or "").splitlines()[-3:]
    elif status == "RETRY":
        response["stage"] = "Retrying after a transient failure"

    return response


@app.delete("/api/v1/tasks/{task_id}", tags=["extract"])
def cancel_task(task_id: str) -> dict:
    """Revoke a queued or running task — used to stop a live capture."""
    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    log.info("api.task_revoked", task_id=task_id)
    return {"task_id": task_id, "status": "revoked"}


__all__ = ["app"]
