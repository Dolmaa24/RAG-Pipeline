"""HTML: decode, strip the furniture, harvest whatever structure is already there.

The last step is the important one. Most commerce, article, recipe, event and
job pages publish schema.org markup for SEO — the publisher's own structured
description of the page, sitting in a ``<script type="application/ld+json">``
that nobody reads. Harvesting it here is what lets tier 1 of the cascade answer
in milliseconds with data that is *more* accurate than a model's reading of the
rendered text, because it came from the publisher rather than from an inference.
"""

from __future__ import annotations

import re

from selectolax.lexbor import LexborHTMLParser

from config import config
from models import ExtractionItem, ResourceKind, Stage
from observability import get_logger
from urls import resolve

from .base import BaseHandler, registry

log = get_logger("handlers.html")

#: Nodes whose text is chrome, not content. Removed before the text is read.
NOISE_SELECTOR = (
    "script, style, noscript, iframe, svg, canvas, template, "
    "header, footer, nav, form, aside, "
    "[role=navigation], [role=banner], [role=contentinfo], [aria-hidden=true], "
    ".cookie-banner, .cookie-consent, #cookie-banner, .advertisement, .ad-slot"
)

#: Containers that usually hold the actual article, tried in order. Falling
#: back to <body> is fine; trying these first removes a lot of boilerplate.
CONTENT_SELECTORS = (
    "article", "main", "[role=main]", "#content", ".post-content",
    ".article-body", ".entry-content", "#main-content",
)

_WHITESPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class HtmlHandler(BaseHandler):
    name = "html"
    kinds = (ResourceKind.HTML,)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        if item.decoded_text is None:
            item = self._decode(item)
            if not item.ok:
                return item

        tree = LexborHTMLParser(item.decoded_text or "")
        base_url = item.final_url or item.url

        # Read structure *before* stripping noise: JSON-LD lives in <script>
        # tags and OpenGraph in <meta>, and both are about to be removed.
        item.metadata.update(self._page_metadata(tree, base_url))
        item.structured = self._structured(item.decoded_text or "", base_url)
        links = self._links(tree, base_url)

        for node in tree.css(NOISE_SELECTOR):
            node.decompose()

        text = self._readable_text(tree)
        item.parsed_tree = {
            "text_content": text,
            "links": links[:500],
            "headings": self._headings(tree),
            "tables": self._tables(tree),
        }
        item.cleaned_text = text

        if not text.strip() and not item.structured:
            return item.fail(Stage.PARSE, "parsed document contained no readable text")
        return item

    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode(item: ExtractionItem) -> ExtractionItem:
        from pipeline.decoder import TextDecoder

        return TextDecoder.decode(item)

    @staticmethod
    def _structured(html: str, base_url: str) -> dict:
        from pipeline.extract.structured import harvest

        try:
            return harvest(html, base_url)
        except Exception as exc:  # structure is a bonus; never fail the parse
            log.debug("html.structured_failed", url=base_url, error=repr(exc))
            return {}

    @staticmethod
    def _page_metadata(tree: LexborHTMLParser, base_url: str) -> dict:
        metadata: dict = {}
        title = tree.css_first("title")
        if title:
            metadata["title"] = title.text(strip=True)

        for name, key in (
            ("description", "description"),
            ("author", "author"),
            ("keywords", "keywords"),
            ("robots", "meta_robots"),
        ):
            node = tree.css_first(f'meta[name="{name}"]')
            if node:
                value = node.attributes.get("content")
                if value:
                    metadata[key] = value.strip()

        canonical = tree.css_first('link[rel="canonical"]')
        if canonical:
            href = canonical.attributes.get("href")
            if href:
                metadata["canonical"] = resolve(base_url, href)

        html_node = tree.css_first("html")
        if html_node:
            lang = html_node.attributes.get("lang")
            if lang:
                metadata["language"] = lang.strip()

        return metadata

    @staticmethod
    def _readable_text(tree: LexborHTMLParser) -> str:
        """Text from the most content-like container, with block structure kept.

        Newlines between blocks are preserved on purpose: they are most of what
        tells a model where one list item ends and the next begins, and they
        cost nothing.
        """
        node = None
        for selector in CONTENT_SELECTORS:
            candidate = tree.css_first(selector)
            if candidate is not None and len(candidate.text(strip=True)) > 200:
                node = candidate
                break
        if node is None:
            node = tree.body or tree.root
        if node is None:
            return ""

        raw = node.text(separator="\n", strip=True)
        cleaned = _WHITESPACE.sub(" ", raw)
        return _BLANK_LINES.sub("\n\n", cleaned).strip()

    @staticmethod
    def _headings(tree: LexborHTMLParser) -> list[dict]:
        headings = []
        for level in range(1, 4):
            for node in tree.css(f"h{level}"):
                text = node.text(strip=True)
                if text:
                    headings.append({"level": level, "text": text[:300]})
        return headings[:100]

    @staticmethod
    def _links(tree: LexborHTMLParser, base_url: str) -> list[dict]:
        """Every outbound link, absolute, with its ``rel``.

        ``rel`` is kept because the crawler needs it: ``rel="nofollow"`` is the
        page telling you not to treat this link as an endorsement or a path
        worth walking, and honouring it costs nothing.
        """
        links = []
        seen: set[str] = set()
        for node in tree.css("a[href]"):
            href = node.attributes.get("href") or ""
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
                continue
            absolute = resolve(base_url, href)
            if not absolute or absolute in seen:
                continue
            seen.add(absolute)
            entry = {"url": absolute, "text": node.text(strip=True)[:200]}
            rel = (node.attributes.get("rel") or "").strip().lower()
            if rel:
                entry["rel"] = rel
            links.append(entry)
        return links

    @staticmethod
    def _tables(tree: LexborHTMLParser) -> list[dict]:
        """Every ``<table>`` as headers plus rows.

        A table is already structured data. Reading it directly is tier 1;
        asking a model to read a flattened wall of cell text is tier 3 and
        routinely gets the column alignment wrong.
        """
        tables = []
        for table in tree.css("table")[:20]:
            headers = [cell.text(strip=True) for cell in table.css("th")]
            rows = []
            for row in table.css("tr")[:200]:
                cells = [cell.text(strip=True) for cell in row.css("td")]
                if cells:
                    rows.append(cells)
            if not rows:
                continue
            if not headers and rows:
                headers = [f"col_{i}" for i in range(len(rows[0]))]
            tables.append({"headers": headers, "rows": rows[: config.MAX_FEED_ENTRIES]})
        return tables


registry.register(HtmlHandler())

__all__ = ["HtmlHandler"]
