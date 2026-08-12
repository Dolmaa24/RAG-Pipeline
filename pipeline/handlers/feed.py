"""RSS, Atom and sitemaps — inputs that are really lists of other inputs.

Both are already structured, so neither needs a model. A feed becomes entries
with title, link, author and date; a sitemap becomes a URL inventory. Each entry
is attached as a child item, which is what lets one submitted feed URL fan out
into a crawl without the caller enumerating anything by hand.

Sitemaps are worth pointing out: one request to ``/sitemap.xml`` can hand you a
site's complete URL list, which is otherwise a week of crawling to discover.
"""

from __future__ import annotations

from typing import Any

from config import config
from errors import MissingDependency, ParseError
from models import ExtractionItem, ResourceKind, Stage
from observability import get_logger
from urls import resolve

from .base import BaseHandler, registry

log = get_logger("handlers.feed")


class FeedHandler(BaseHandler):
    name = "feed"
    kinds = (ResourceKind.FEED,)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        try:
            import feedparser
        except ImportError as exc:
            raise MissingDependency("feedparser", "RSS/Atom parsing") from exc

        parsed = feedparser.parse(item.raw_bytes or b"")
        if parsed.bozo and not parsed.entries:
            raise ParseError(f"malformed feed: {parsed.bozo_exception}")

        channel = parsed.feed or {}
        item.metadata.update(
            {
                "title": channel.get("title"),
                "description": channel.get("subtitle") or channel.get("description"),
                "language": channel.get("language"),
                "feed_type": parsed.version or "unknown",
                "entry_count": len(parsed.entries),
            }
        )
        item.metadata = {k: v for k, v in item.metadata.items() if v is not None}

        entries: list[dict[str, Any]] = []
        base = item.final_url or item.url

        for entry in parsed.entries[: config.MAX_FEED_ENTRIES]:
            link = entry.get("link") or ""
            record = {
                "title": entry.get("title"),
                "url": resolve(base, link) if link else None,
                "author": entry.get("author"),
                "published": entry.get("published") or entry.get("updated"),
                "summary": _plain(entry.get("summary")),
                "tags": [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")],
                "content": _entry_content(entry),
            }
            entries.append({k: v for k, v in record.items() if v})

            if record.get("url") and item.depth < config.MAX_RECURSION_DEPTH:
                item.children.append(
                    ExtractionItem(
                        url=record["url"],
                        depth=item.depth + 1,
                        parent_url=item.url,
                        metadata={"from_feed": item.url, "feed_title": record.get("title")},
                    )
                )

        item.structured = {"feed_entries": entries}
        item.cleaned_text = _render(item.metadata.get("title"), entries)
        log.info("feed.parsed", url=item.url, entries=len(entries), children=len(item.children))
        return item


class SitemapHandler(BaseHandler):
    name = "sitemap"
    kinds = (ResourceKind.SITEMAP,)
    #: A sitemap has no prose. Its output is the URL list.
    requires_text = False

    def process(self, item: ExtractionItem) -> ExtractionItem:
        try:
            from lxml import etree
        except ImportError as exc:
            raise MissingDependency("lxml", "sitemap parsing") from exc

        try:
            # resolve_entities=False blocks XXE: a sitemap is attacker-supplied
            # XML, and entity expansion in it can read local files.
            parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
            root = etree.fromstring(item.raw_bytes or b"", parser=parser)
        except etree.XMLSyntaxError as exc:
            raise ParseError(f"malformed sitemap XML: {exc}") from exc
        if root is None:
            raise ParseError("sitemap contained no XML")

        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        is_index = root.tag.endswith("sitemapindex")
        entries: list[dict[str, Any]] = []

        for node in root.iterfind(".//sm:sitemap" if is_index else ".//sm:url", namespace):
            location = node.findtext("sm:loc", namespaces=namespace)
            if not location:
                continue
            entries.append(
                {
                    "url": location.strip(),
                    "lastmod": (node.findtext("sm:lastmod", namespaces=namespace) or "").strip() or None,
                    "changefreq": (node.findtext("sm:changefreq", namespaces=namespace) or "").strip() or None,
                    "priority": (node.findtext("sm:priority", namespaces=namespace) or "").strip() or None,
                }
            )
            if len(entries) >= config.MAX_SITEMAP_URLS:
                item.warn(f"sitemap truncated at {config.MAX_SITEMAP_URLS} URLs")
                break

        if not entries:
            raise ParseError("sitemap contained no <loc> entries")

        item.structured = {"sitemap_urls": entries, "is_index": is_index}
        item.metadata.update({"url_count": len(entries), "sitemap_index": is_index})
        item.cleaned_text = "\n".join(entry["url"] for entry in entries[:1000])

        if item.depth < config.MAX_RECURSION_DEPTH:
            for entry in entries:
                item.children.append(
                    ExtractionItem(
                        url=entry["url"],
                        depth=item.depth + 1,
                        parent_url=item.url,
                        metadata={"from_sitemap": item.url, "lastmod": entry.get("lastmod")},
                    )
                )

        log.info("sitemap.parsed", url=item.url, urls=len(entries), index=is_index)
        return item


def _entry_content(entry) -> str:
    blocks = entry.get("content") or []
    for block in blocks:
        value = block.get("value")
        if value:
            return _plain(value)
    return ""


def _plain(html: str | None) -> str:
    """Feed summaries are HTML fragments. Reduce them to text."""
    if not html:
        return ""
    try:
        from selectolax.lexbor import LexborHTMLParser

        return LexborHTMLParser(html).text(separator=" ", strip=True)
    except Exception:
        import re

        return re.sub(r"<[^>]+>", " ", html).strip()


def _render(title: str | None, entries: list[dict]) -> str:
    lines = [f"# {title}" if title else "# Feed", ""]
    for entry in entries:
        lines.append(f"## {entry.get('title', 'Untitled')}")
        for key in ("author", "published", "url"):
            if entry.get(key):
                lines.append(f"{key}: {entry[key]}")
        body = entry.get("content") or entry.get("summary")
        if body:
            lines.append("")
            lines.append(body[:2000])
        lines.append("")
    return "\n".join(lines)


registry.register(FeedHandler())
registry.register(SitemapHandler())

__all__ = ["FeedHandler", "SitemapHandler"]
