"""Turning one fetched page into the next set of URLs to visit.

The HTML handler already parses every link on the page, so this does not
re-parse anything: it reads ``item.parsed_tree["links"]``, applies the scope,
and hands back two lists — pages to follow and files to collect.

Keeping this separate from the handler matters. An ordinary extraction job must
not start a crawl because the page happened to have links on it; crawling is
something you ask for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models import ExtractionItem
from observability import get_logger
from urls import canonicalize

from .scope import CrawlScope, LinkVerdict

log = get_logger("discover.crawler")


@dataclass
class LinkHarvest:
    """What one page contributed to the crawl."""

    follow: list[str] = field(default_factory=list)
    collect: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.follow) + len(self.collect)

    def summary(self) -> dict:
        return {
            "follow": len(self.follow),
            "collect": len(self.collect),
            "skipped": sum(self.skipped.values()),
            "skip_reasons": dict(sorted(self.skipped.items(), key=lambda kv: -kv[1])[:5]),
        }


def harvest_links(item: ExtractionItem, scope: CrawlScope, depth: int) -> LinkHarvest:
    """Classify every link on a fetched page against the crawl's scope."""
    harvest = LinkHarvest()
    if not item.parsed_tree:
        return harvest

    links = item.parsed_tree.get("links") or []
    if not links:
        return harvest

    # A page-level `<meta name="robots" content="nofollow">` applies to all of
    # its links, not just the ones carrying rel="nofollow".
    meta_robots = (item.metadata.get("meta_robots") or "").lower()
    page_nofollow = "nofollow" in meta_robots or "none" in meta_robots

    seen: set[str] = set()
    for link in links:
        raw = link.get("url") if isinstance(link, dict) else link
        if not raw:
            continue
        url = canonicalize(raw)
        if not url or url in seen:
            continue
        seen.add(url)

        rel = (link.get("rel") or "") if isinstance(link, dict) else ""
        nofollow = page_nofollow or "nofollow" in rel

        decision = scope.classify(url, depth=depth, nofollow=nofollow)
        if decision.verdict is LinkVerdict.COLLECT:
            harvest.collect.append(url)
        elif decision.verdict is LinkVerdict.FOLLOW:
            harvest.follow.append(url)
        else:
            harvest.skipped[decision.reason] = harvest.skipped.get(decision.reason, 0) + 1

    log.debug(
        "crawler.harvested",
        url=item.url,
        depth=depth,
        **harvest.summary(),
    )
    return harvest


def seed_urls(scope: CrawlScope, *, use_sitemap: bool = True) -> list[str]:
    """Where a crawl starts.

    The start URL always, plus anything robots.txt advertises. One request for
    ``/robots.txt`` can hand over a site's whole sitemap list, which beats
    discovering the same URLs one page at a time — and it is the crawl the site
    itself asked you to do.
    """
    seeds = [canonicalize(scope.start_url)]
    if not use_sitemap:
        return seeds

    try:
        from pipeline.compliance import robots_gate

        for sitemap in robots_gate.sitemaps(scope.start_url):
            url = canonicalize(sitemap)
            if url and url not in seeds:
                seeds.append(url)
    except Exception as exc:
        log.debug("crawler.sitemap_lookup_failed", error=repr(exc))

    if len(seeds) > 1:
        log.info("crawler.seeded_from_robots", start=scope.start_url, sitemaps=len(seeds) - 1)
    return seeds


def sitemap_urls(item: ExtractionItem, scope: CrawlScope) -> list[str]:
    """In-scope URLs from a fetched sitemap or feed.

    A sitemap seed is worth following even though it is not an HTML page: it is
    a list of exactly the URLs the site wants indexed, already deduplicated.
    """
    if not item.structured:
        return []

    entries = item.structured.get("sitemap_urls") or item.structured.get("feed_entries") or []
    found: list[str] = []
    for entry in entries:
        raw = entry.get("url") if isinstance(entry, dict) else entry
        if not raw:
            continue
        url = canonicalize(raw)
        # Depth 0: a sitemap entry is a starting point, not a link found deep
        # in the site, so it should not inherit the sitemap's own depth cost.
        decision = scope.classify(url, depth=0)
        if decision.verdict is not LinkVerdict.SKIP:
            found.append(url)
    return found


def describe_plan(scope: CrawlScope) -> str:
    """One line saying what this crawl will actually do. Shown before it runs."""
    targets = (
        ", ".join(f".{ext}" for ext in sorted(scope.collect_extensions))
        if scope.collect_extensions
        else "every page"
    )
    where = ", ".join(sorted(scope.hosts)) or "anywhere"
    return (
        f"collect {targets} from {where}, following pages up to depth "
        f"{scope.max_depth}, at most {scope.max_pages} URLs"
    )


__all__ = ["LinkHarvest", "describe_plan", "harvest_links", "seed_urls", "sitemap_urls"]
