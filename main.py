"""Synchronous runner — no Redis, no Celery, no API.

The fastest way to try the pipeline, and the way to debug it: everything runs in
one process, so a traceback is a traceback rather than a task id.

    PYTHONPATH=. ./venv/bin/python main.py
    PYTHONPATH=. ./venv/bin/python main.py https://example.com/a https://example.com/b
    PYTHONPATH=. ./venv/bin/python main.py --tiers 1 https://shop.example/product/1

Concurrency here is threads, and the number is deliberately small. The per-host
rate limiter is the real throttle — raising this raises parallelism *across*
hosts, never against one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from config import config
from database import CloudDatabase
from models import ExtractionItem, RunReport
from observability import configure_logging, get_logger, metrics
from pipeline.runner import Pipeline
from urls import canonicalize

log = get_logger("batch")
OUTPUT_DIR = Path(__file__).parent / "output"

DEFAULT_URLS = [
    "https://quotes.toscrape.com/page/1/",
    "https://quotes.toscrape.com/page/2/",
    "https://quotes.toscrape.com/page/3/",
]
DEFAULT_PROMPT = "Extract the first quote on the page, its author, and its tags."
DEFAULT_SCHEMA = {"quote": "string", "author": "string", "tags": "list of strings"}


def run_batch_pipeline(
    urls: list[str],
    prompt: str,
    schema: dict,
    max_workers: int = 3,
    *,
    database: Optional[CloudDatabase] = None,
    allowed_tiers: Optional[set[int]] = None,
    local_only: bool = False,
) -> RunReport:
    """Run the pipeline over many URLs and write the results to ``output/``."""
    log.info("batch.start", urls=len(urls), workers=max_workers)

    db = database if database is not None else CloudDatabase()
    pipeline = Pipeline(database=db)
    report = RunReport()
    results: list[dict] = []

    def process(url: str) -> ExtractionItem:
        item = ExtractionItem(url=url, canonical_url=canonicalize(url))
        return pipeline.run_item(
            item, prompt, schema, allowed_tiers=allowed_tiers, local_only=local_only
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # a thread blowing up must not kill the batch
                log.exception("batch.thread_failed", url=url, error=repr(exc))
                item = ExtractionItem(url=url).fail_from(exc) if hasattr(exc, "stage") else None
                if item is None:
                    report.failed += 1
                    report.errors.append({"url": url, "message": str(exc)})
                    continue
            report.record(item)
            if item.ok:
                results.append(item.summary())
                _index(item)
            for child in item.children:
                if child.ok:
                    results.append(child.summary())
                    _index(child)

    report.metrics = metrics.snapshot()
    report.finish()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "batch_extracted_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    report_path = OUTPUT_DIR / "batch_run_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    _print_summary(report, results_path, report_path)
    return report


def _index(item: ExtractionItem) -> None:
    """Chunk, embed and store one item's text, if indexing is on.

    The queued path enqueues ``tasks.index_document`` instead. Here there is no
    broker, so the same facade is called directly — which is the point of it
    being a facade: one code path, two transports.
    """
    if not config.INDEX_ENABLED:
        return
    text = item.text_for_extraction
    if not text.strip() or item.metadata.get("duplicate_of"):
        return
    try:
        from pipeline.index import index_text

        from pipeline.index.metadata import derive

        metadata = {
            "content_hash": item.content_hash or "",
            "extraction_tier": item.tier if item.tier is not None else -1,
        }
        metadata.update(derive(item))

        index_text(
            text[: config.INDEX_MAX_TEXT_CHARS],
            source=item.canonical_url or item.url,
            extra_metadata=metadata,
        )
    except Exception as exc:
        # A failed index must not cost you the extraction you already have.
        log.warning("batch.index_failed", url=item.url, error=repr(exc))


def _print_summary(report: RunReport, results_path: Path, report_path: Path) -> None:
    tiers = ", ".join(f"{method}={count}" for method, count in sorted(report.by_method.items()))
    print(
        f"\n{report.succeeded}/{report.submitted} succeeded in "
        f"{report.duration_seconds:.1f}s"
        f"\n  by tier      : {tiers or 'none'}"
        f"\n  model avoided: {report.llm_avoidance_rate:.0%} of successes"
        f"\n  results      : {results_path}"
        f"\n  run report   : {report_path}"
    )
    if report.errors:
        print(f"  failures     : {len(report.errors)}")
        for failure in report.errors[:5]:
            print(f"    - {failure['url']}: [{failure.get('stage')}] {failure.get('message')}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the extraction pipeline synchronously.")
    parser.add_argument("urls", nargs="*", default=None, help="URLs to extract (any type).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--schema",
        default=None,
        help='JSON schema hint, e.g. \'{"title": "string", "price": "string"}\'',
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=None,
        help="Restrict the cascade: 0 cache, 1 structured data, 2 selectors, 3 model.",
    )
    parser.add_argument("--local-only", action="store_true", help="Never use a hosted model.")
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    parser.add_argument("--json-logs", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.log_level, "json" if args.json_logs else "text")
    logging.getLogger("celery").setLevel(logging.WARNING)

    urls = args.urls or DEFAULT_URLS
    schema = json.loads(args.schema) if args.schema else DEFAULT_SCHEMA

    report = run_batch_pipeline(
        urls,
        args.prompt,
        schema,
        max_workers=args.workers,
        allowed_tiers=set(args.tiers) if args.tiers else None,
        local_only=args.local_only,
    )
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
