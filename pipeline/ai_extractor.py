"""Backwards-compatible wrapper around the extraction cascade.

The original :class:`AIExtractor` went straight to Ollama for every page. It now
delegates to :class:`~pipeline.extract.cascade.ExtractionCascade`, which tries
the cache, the page's own structured data and a learned selector spec first —
so the same call site is unchanged while most pages stop reaching a model at all.
"""

from __future__ import annotations

from typing import Optional

from models import ExtractionItem
from pipeline.extract.cascade import ExtractionCascade, get_cascade


class AIExtractor:
    def __init__(
        self,
        *,
        database=None,
        cascade: Optional[ExtractionCascade] = None,
        local_only: bool = False,
    ) -> None:
        self.cascade = cascade or get_cascade(database)
        self.local_only = local_only

    def extract_with_schema(
        self, item: ExtractionItem, prompt: str, schema: dict
    ) -> ExtractionItem:
        return self.cascade.extract(item, prompt, schema, local_only=self.local_only)


__all__ = ["AIExtractor"]
