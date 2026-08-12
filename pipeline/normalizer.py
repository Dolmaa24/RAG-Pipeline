"""The NORMALIZE stage.

Cleans every extracted value and adds typed companions where the field name
implies one. The original strings survive: ``price`` stays exactly as the page
said it, and ``price_amount`` holds the exact Decimal beside it.
"""

from __future__ import annotations

from models import ExtractionItem, Stage
from pipeline.normalize import normalize_record


class DataNormalizer:
    @staticmethod
    def normalize(item: ExtractionItem, *, typed: bool = True) -> ExtractionItem:
        if not item.extracted_data:
            return item.fail(Stage.NORMALIZE, "no extracted data to normalize")

        item.normalized_data = normalize_record(
            item.extracted_data,
            base_url=item.final_url or item.url,
            typed=typed,
        )
        return item


__all__ = ["DataNormalizer"]
