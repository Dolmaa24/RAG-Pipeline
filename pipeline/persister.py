"""Writing one record to disk.

A file sink for scripts and one-off runs. The pipeline's real persistence path
is :class:`~database.CloudDatabase`, which stores provenance alongside the data
and falls back to JSONL on its own when Mongo is unreachable.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import config
from models import ExtractionItem, Stage


class DataPersister:
    def __init__(self, output_dir: str = "./output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, item: ExtractionItem, filename: str) -> ExtractionItem:
        if not item.normalized_data:
            return item.fail(Stage.PERSIST, "no normalized data available to persist")

        payload = {
            "url": item.url,
            "canonical_url": item.canonical_url,
            "metadata": item.metadata,
            "extracted_data": item.normalized_data,
            # Provenance travels with the record. A file that says only what was
            # extracted, and not how or from which version of the page, cannot
            # be checked against anything later.
            "provenance": item.provenance(
                schema_hash=item.metadata.get("schema_hash"),
                prompt_hash=item.metadata.get("prompt_hash"),
                pipeline_version=config.PIPELINE_VERSION,
            ).model_dump(mode="json"),
            "warnings": item.warnings,
            "validation_failures": item.validation_failures,
        }

        (self.output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return item

    def append_jsonl(self, item: ExtractionItem, filename: str = "extractions.jsonl") -> ExtractionItem:
        """Append one record — the right shape for a long crawl."""
        if not item.normalized_data:
            return item.fail(Stage.PERSIST, "no normalized data available to persist")

        record = {
            "url": item.url,
            "kind": item.kind.value,
            "method": item.method.value,
            "extracted_data": item.normalized_data,
            "content_hash": item.content_hash,
        }
        with open(self.output_dir / filename, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return item


__all__ = ["DataPersister"]
