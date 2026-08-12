"""Backwards-compatible import site for the HTML parser.

The real implementation is :class:`~pipeline.handlers.html.HtmlHandler`, which
does everything this did plus harvesting the page's embedded structured data —
the thing tier 1 of the cascade reads. ``DOMParser`` remains as a thin wrapper
so older call sites keep working and get the better behaviour for free.
"""

from __future__ import annotations

from models import ExtractionItem


class DOMParser:
    @staticmethod
    def parse_and_clean(item: ExtractionItem) -> ExtractionItem:
        from pipeline.handlers.html import HtmlHandler

        return HtmlHandler().handle(item)


__all__ = ["DOMParser"]
