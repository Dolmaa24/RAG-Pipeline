"""Tools that reach the outside world or change the corpus.

Everything here is refused unless the run that started explicitly allowed the
matching effect. That gate is in :func:`pipeline.agents.tools.registry.invoke`
rather than in these handlers, so a new tool cannot forget to check.

Two rules hold throughout:

* **No handler blocks on long work.** Extraction takes seconds to minutes, so
  these enqueue a Celery task and hand back an id. The durable half of the
  problem is already solved by the queue; a tool call that waited would only add
  a way to time out.
* **No handler bypasses the fetch policy.** ``detect_url`` reports the policy's
  verdict rather than deciding for itself, and the queued tasks run through the
  same pipeline every other job does. An agent persuaded by something it read to
  fetch ``169.254.169.254`` gets the same refusal a user would.
"""

from __future__ import annotations

from observability import get_logger

from pipeline.agents.tools.models import (
    CrawlArgs,
    DetectArgs,
    DetectResult,
    ExtractArgs,
    IndexArgs,
    PollArgs,
    PollResult,
    SitemapArgs,
    TaskResult,
)
from pipeline.agents.tools.registry import Effect, tool

log = get_logger("agents.tools.acquire")

#: What to ask for when a tool has no schema of its own. Extraction needs one,
#: and an agent fetching a page for later retrieval cares about the text rather
#: than about any particular field.
_DEFAULT_SCHEMA: dict[str, str] = {
    "title": "string",
    "summary": "string",
    "key_points": "list of strings",
}


@tool(
    name="detect_url",
    effect=Effect.NETWORK,
    cost_ms=300,
    description=(
        "What the pipeline would do with a URL — how it would fetch it, what "
        "kind of resource it looks like, whether policy allows it at all, and "
        "whether it is already indexed. Costs nothing but a HEAD request. Call "
        "it before extract_url so a refusal costs a moment rather than a job."
    ),
)
def detect_url(args: DetectArgs) -> DetectResult:
    from pipeline.compliance.policy import policy
    from pipeline.detect.router import pre_route

    decision = pre_route(args.url)
    verdict = policy.check(args.url)

    return DetectResult(
        url=args.url,
        acquisition=str(decision.acquisition),
        reason=decision.reason,
        likely_kind=decision.likely_kind.value,
        allowed=verdict.allowed,
        refusal="" if verdict.allowed else verdict.reason,
        already_indexed=_is_indexed(args.url),
    )


def _is_indexed(url: str) -> bool:
    """Whether anything from this URL is already in the corpus."""
    try:
        from pipeline.retrieve.filters import quote_literal
        from pipeline.store.lance import LanceStore

        store = LanceStore()
        if store.table is None:
            return False
        return store.count(where=f"source = {quote_literal(url)}") > 0
    except Exception as exc:
        log.debug("agents.tools.indexed_check_failed", error=repr(exc))
        return False


@tool(
    name="discover_sitemap",
    effect=Effect.NETWORK,
    cost_ms=5000,
    description=(
        "The URLs a site advertises, from its robots.txt and sitemap. Use it to "
        "find out what is available on a site before deciding what to extract, "
        "rather than crawling to find out."
    ),
)
def discover_sitemap(args: SitemapArgs) -> TaskResult:
    from tasks import discover_sitemap as task

    queued = task.delay(args.url)
    return TaskResult(
        task_id=queued.id,
        kind="sitemap discovery",
        note="The result lists URLs; none of them is fetched by this call.",
    )


@tool(
    name="extract_url",
    effect=Effect.WRITE,
    cost_ms=30000,
    description=(
        "Fetch one URL, extract its content, and add it to the searchable "
        "corpus. Takes seconds to a minute, so it returns a task id — poll it "
        "with poll_task, then search once it reports finished. Use detect_url "
        "first if you are unsure the URL is fetchable."
    ),
)
def extract_url(args: ExtractArgs) -> TaskResult:
    from tasks import extract_url as task

    queued = task.delay(
        args.url,
        args.prompt,
        _DEFAULT_SCHEMA,
        index=args.index,
        build_graph=args.build_graph,
    )
    return TaskResult(
        task_id=queued.id,
        kind=f"extraction of {args.url}",
        note=(
            "Indexing runs after extraction, so allow a moment after it "
            "finishes before searching for the content."
            if args.index
            else "Not indexed: this will not be searchable."
        ),
    )


@tool(
    name="crawl_site",
    effect=Effect.WRITE,
    cost_ms=300000,
    description=(
        "Walk a site and extract every page in scope. Expensive — minutes, and "
        "many requests to someone else's server. Prefer extract_url on specific "
        "pages, or discover_sitemap to see what exists first. Returns a task id."
    ),
)
def crawl_site(args: CrawlArgs) -> TaskResult:
    from tasks import crawl_site as task

    queued = task.delay(
        args.start_url,
        args.prompt,
        _DEFAULT_SCHEMA,
        scope_options={
            "max_depth": args.max_depth,
            "max_pages": args.max_pages,
            # Not exposed as an argument. same_site=False with no allowed_hosts
            # is an unbounded crawl of the open web, and that is not a decision
            # to leave anywhere near a model's discretion.
            "same_site": True,
        },
    )
    return TaskResult(
        task_id=queued.id,
        kind=f"crawl of {args.start_url}",
        note=f"Up to {args.max_pages} pages, depth {args.max_depth}.",
    )


@tool(
    name="index_document",
    effect=Effect.WRITE,
    cost_ms=20000,
    description=(
        "Make a piece of text searchable: chunk it, embed it, and add it to the "
        "corpus under a source you name. Use it for text you already have — "
        "pasted, or handed to you — rather than a URL, which extract_url "
        "handles. Returns a task id; poll it before searching for the content. "
        "It only ever adds. It cannot delete, edit or remove anything, and it "
        "cannot send or share anything; if that is what was asked for, say it "
        "is not possible instead of calling this."
    ),
)
def index_document(args: IndexArgs) -> TaskResult:
    from tasks import index_document as task

    metadata = {
        key: value
        for key, value in (
            ("department", args.department.strip()),
            ("region", args.region.strip()),
            ("permission_level", args.permission_level.strip()),
        )
        if value
    }

    queued = task.delay(
        args.text,
        source=args.source,
        metadata=metadata or None,
        build_graph=args.build_graph,
    )
    return TaskResult(
        task_id=queued.id,
        kind=f"indexing of {args.source}",
        note=(
            "Entities and relationships will be extracted too, which takes "
            "considerably longer than indexing alone."
            if args.build_graph
            else "Text only; no graph was built."
        ),
    )


@tool(
    name="poll_task",
    effect=Effect.READ,
    cost_ms=5,
    description=(
        "Whether a queued task has finished, and what it produced. Poll rather "
        "than assuming: extraction is not instant, and searching for content "
        "before its task reports finished will find nothing."
    ),
)
def poll_task(args: PollArgs) -> PollResult:
    from celery.result import AsyncResult

    from celery_app import celery_app

    async_result = AsyncResult(args.task_id, app=celery_app)
    state = str(async_result.state)

    if not async_result.ready():
        return PollResult(task_id=args.task_id, state=state, done=False)

    if async_result.successful():
        return PollResult(
            task_id=args.task_id,
            state=state,
            done=True,
            ok=True,
            summary=_summarise(async_result.result),
        )

    return PollResult(
        task_id=args.task_id,
        state=state,
        done=True,
        ok=False,
        summary=str(async_result.result)[:400],
    )


def _summarise(result: object) -> str:
    """A finished task in a sentence, rather than its whole payload.

    Task results carry the full extraction — every field, every timing. Putting
    that into a message history spends the token budget on things a model does
    not need to decide what to do next.
    """
    if not isinstance(result, dict):
        return str(result)[:400]

    if "urls" in result:
        urls = result.get("urls") or []
        return f"Found {len(urls)} URLs."

    parts = []
    if result.get("url"):
        parts.append(str(result["url"]))
    if result.get("kind"):
        parts.append(f"kind={result['kind']}")
    if result.get("tier") is not None:
        parts.append(f"tier={result['tier']}")
    extracted = result.get("extracted_data")
    if isinstance(extracted, dict) and extracted:
        parts.append("fields: " + ", ".join(sorted(extracted)[:8]))
    if result.get("index_task_id"):
        parts.append("indexing queued")
    return "; ".join(parts) or "Finished."


__all__ = [
    "crawl_site",
    "detect_url",
    "discover_sitemap",
    "extract_url",
    "index_document",
    "poll_task",
]
