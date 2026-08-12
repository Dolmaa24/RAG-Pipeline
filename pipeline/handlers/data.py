"""Formats that are already structured: CSV, JSON, JSONL, XML, YAML, plain text.

These need no model at all, and the handler says so: it fills ``item.structured``
directly, so the cascade's tier 1 answers from the parsed rows rather than
asking a model to re-derive them from a stringified table.

A rendered text form is still produced, because a schema field that no key
matches ("what is this dataset about?") is a genuine model question, and the
model needs something to read.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from config import config
from errors import MissingDependency, ParseError
from models import ExtractionItem, ResourceKind
from observability import get_logger

from .base import BaseHandler, registry

log = get_logger("handlers.data")

#: Rows held in memory and rendered. A 2 GB CSV is a database import, not a
#: scrape, and pretending otherwise just exhausts the worker.
MAX_ROWS = 10_000
#: Rows rendered into the text the model sees. More than this is noise.
PREVIEW_ROWS = 100


class TabularHandler(BaseHandler):
    name = "tabular"
    kinds = (ResourceKind.TABULAR,)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        text = self._text(item)
        sample = text[:65536]

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = "\t" if item.metadata.get("detected_subtype") == "tsv" else ","

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        try:
            rows = [row for _, row in zip(range(MAX_ROWS + 1), reader)]
        except csv.Error as exc:
            raise ParseError(f"malformed delimited file: {exc}") from exc

        if not rows:
            raise ParseError("delimited file contained no rows")

        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]

        headers, records = self._as_records(rows)
        item.structured = {"table": {"headers": headers, "records": records}}
        item.metadata.update(
            {
                "row_count": len(records),
                "column_count": len(headers),
                "columns": headers,
                "delimiter": delimiter,
                "truncated": truncated,
            }
        )
        if truncated:
            item.warn(f"table truncated at {MAX_ROWS} rows")

        item.cleaned_text = self._render(headers, records)
        log.info("tabular.parsed", url=item.url, rows=len(records), columns=len(headers))
        return item

    @staticmethod
    def _text(item: ExtractionItem) -> str:
        if item.decoded_text:
            return item.decoded_text
        from pipeline.decoder import TextDecoder

        decoded = TextDecoder.decode(item)
        return decoded.decoded_text or ""

    @staticmethod
    def _as_records(rows: list[list[str]]) -> tuple[list[str], list[dict]]:
        """Treat row 0 as headers when it looks like labels rather than data."""
        first = rows[0]
        looks_like_header = bool(first) and all(
            cell.strip() and not _is_number(cell) for cell in first
        )
        if looks_like_header:
            headers = [cell.strip() or f"col_{i}" for i, cell in enumerate(first)]
            body = rows[1:]
        else:
            headers = [f"col_{i}" for i in range(len(first))]
            body = rows

        records = []
        for row in body:
            if not any(cell.strip() for cell in row):
                continue
            record = {headers[i] if i < len(headers) else f"col_{i}": value for i, value in enumerate(row)}
            records.append(record)
        return headers, records

    @staticmethod
    def _render(headers: list[str], records: list[dict]) -> str:
        lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
        for record in records[:PREVIEW_ROWS]:
            lines.append(" | ".join(str(record.get(header, "")) for header in headers))
        if len(records) > PREVIEW_ROWS:
            lines.append(f"... and {len(records) - PREVIEW_ROWS} more rows")
        return "\n".join(lines)


class StructuredDataHandler(BaseHandler):
    name = "data"
    kinds = (ResourceKind.DATA,)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        text = TabularHandler._text(item)
        subtype = item.metadata.get("detected_subtype", "")

        if subtype == "jsonl":
            payload = self._parse_jsonl(text)
        elif subtype == "yaml":
            payload = self._parse_yaml(text)
        elif subtype == "xml":
            payload = self._parse_xml(item.raw_bytes or text.encode("utf-8"))
        else:
            payload = self._parse_json(text)

        item.structured = {"data": payload}
        item.metadata.update({"data_format": subtype or "json", "top_level_type": type(payload).__name__})
        if isinstance(payload, list):
            item.metadata["record_count"] = len(payload)
        elif isinstance(payload, dict):
            item.metadata["keys"] = list(payload)[:50]

        item.cleaned_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)[
            : config.MAX_CHUNK_SIZE * 4
        ]
        log.info("data.parsed", url=item.url, format=subtype or "json")
        return item

    @staticmethod
    def _parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}") from exc

    @staticmethod
    def _parse_jsonl(text: str) -> list:
        records = []
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # One malformed line in a million-line export should not lose
                # the other 999,999.
                if number <= 3 and not records:
                    raise ParseError(f"invalid JSONL at line {number}")
            if len(records) >= MAX_ROWS:
                break
        return records

    @staticmethod
    def _parse_yaml(text: str) -> Any:
        try:
            import yaml
        except ImportError as exc:
            raise MissingDependency("pyyaml", "YAML parsing") from exc
        try:
            # safe_load, never load: YAML's full loader constructs arbitrary
            # Python objects, and this document came off the internet.
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ParseError(f"invalid YAML: {exc}") from exc

    @staticmethod
    def _parse_xml(data: bytes) -> Any:
        try:
            from lxml import etree
        except ImportError as exc:
            raise MissingDependency("lxml", "XML parsing") from exc
        try:
            parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
            root = etree.fromstring(data, parser=parser)
        except etree.XMLSyntaxError as exc:
            raise ParseError(f"invalid XML: {exc}") from exc
        if root is None:
            raise ParseError("XML document was empty")
        return _xml_to_dict(root)


def _xml_to_dict(node, depth: int = 0) -> Any:
    """XML to nested dicts, keeping attributes under ``@name``."""
    if depth > 20:
        return None
    result: dict[str, Any] = {}
    result.update({f"@{k}": v for k, v in node.attrib.items()})

    children = list(node)
    if not children:
        text = (node.text or "").strip()
        if not result:
            return text or None
        if text:
            result["#text"] = text
        return result

    for child in children[:1000]:
        tag = etree_tag(child)
        value = _xml_to_dict(child, depth + 1)
        if tag in result:
            existing = result[tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[tag] = [existing, value]
        else:
            result[tag] = value
    return result


def etree_tag(node) -> str:
    tag = node.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return str(tag)


class TextHandler(BaseHandler):
    name = "text"
    kinds = (ResourceKind.TEXT, ResourceKind.UNKNOWN)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        text = TabularHandler._text(item)
        item.cleaned_text = text.strip()
        item.metadata.setdefault("char_count", len(text))
        return item


def _is_number(value: str) -> bool:
    try:
        float(value.strip().replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


registry.register(TabularHandler())
registry.register(StructuredDataHandler())
registry.register(TextHandler())

__all__ = ["StructuredDataHandler", "TabularHandler", "TextHandler"]
