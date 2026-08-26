"""PDF, Office, OpenDocument, ePub — everything that is "a document".

Rather than hand-writing eight parsers, this delegates to **MarkItDown**, which
covers PDF, docx, pptx, xlsx, ePub and more behind one API and emits Markdown —
a format that keeps headings, lists and table structure, all of which a model
reads far better than a flattened wall of text.

Two deliberate exceptions to "just use MarkItDown":

* **PDFs go through PyMuPDF first.** It is faster, it reports page count and
  per-page text, and it can tell a text-layer PDF from a scanned one — which is
  the difference between a 40 ms parse and needing OCR.
* **Legacy OLE2 files** (.doc/.xls/.ppt from before 2007) are converted by
  headless LibreOffice, because nothing in Python reads them well.

**Docling** is available as an opt-in for table-heavy or multi-column PDFs,
where its layout model is worth the extra seconds. Set ``DOCUMENT_BACKEND=docling``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from config import config
from errors import MissingDependency, ParseError
from models import ExtractionItem, ResourceKind, Stage
from observability import get_logger

from .base import BaseHandler, registry

log = get_logger("handlers.document")

#: Detected subtype → the extension the parser expects on disk. MarkItDown and
#: LibreOffice both dispatch on the filename, so the temp file needs the right
#: suffix even though we identified the type from magic bytes.
_SUFFIX = {
    "pdf": ".pdf", "docx": ".docx", "pptx": ".pptx", "xlsx": ".xlsx",
    "doc": ".doc", "xls": ".xls", "ppt": ".ppt", "ole2": ".doc",
    "epub": ".epub", "mobi": ".mobi", "rtf": ".rtf", "postscript": ".ps",
    "odt": ".odt", "ods": ".ods", "odp": ".odp", "opendocument": ".odt",
    "ooxml": ".docx", "vsdx": ".vsdx",
}

#: Formats only LibreOffice reads properly, and what to convert them into.
_LIBREOFFICE_TARGETS = {
    ".doc": "docx", ".xls": "xlsx", ".ppt": "pptx",
    ".ps": "pdf", ".vsdx": "pdf", ".wpd": "docx",
}

_LIBREOFFICE_BINARIES = ("soffice", "libreoffice")


class DocumentHandler(BaseHandler):
    name = "document"
    kinds = (ResourceKind.DOCUMENT,)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        if not item.raw_bytes:
            return item.fail(Stage.PARSE, "no bytes to parse")

        subtype = item.metadata.get("detected_subtype", "")
        suffix = _SUFFIX.get(subtype, ".bin")

        workdir = tempfile.mkdtemp(prefix="doc_")
        try:
            path = os.path.join(workdir, f"document{suffix}")
            with open(path, "wb") as handle:
                handle.write(item.raw_bytes)

            if suffix in _LIBREOFFICE_TARGETS:
                path = self._convert_legacy(path, workdir, _LIBREOFFICE_TARGETS[suffix])
                suffix = os.path.splitext(path)[1]

            if suffix == ".pdf" and config.DOCUMENT_BACKEND != "docling":
                text, metadata = self._read_pdf(path, item)
                if text.strip():
                    item.decoded_text = text
                    item.cleaned_text = text
                    item.metadata.update(metadata)
                    return item
                # No text layer: it is a scan, and the pixels are the content.
                item.warn("PDF has no text layer; falling back to OCR")
                return self._ocr_pdf(item, path)

            text, metadata = self._read_generic(path, item)
            item.decoded_text = text
            item.cleaned_text = text
            item.metadata.update(metadata)
            return item
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _read_pdf(self, path: str, item: ExtractionItem) -> tuple[str, dict]:
        try:
            import pymupdf
        except ImportError as exc:
            raise MissingDependency("pymupdf", "PDF text extraction") from exc

        parts: list[str] = []
        tables: list[dict] = []
        metadata: dict = {}

        with pymupdf.open(path) as document:
            info = document.metadata or {}
            metadata = {
                "title": (info.get("title") or "").strip() or None,
                "author": (info.get("author") or "").strip() or None,
                "subject": (info.get("subject") or "").strip() or None,
                "creator": (info.get("creator") or "").strip() or None,
                "page_count": document.page_count,
                "is_encrypted": document.is_encrypted,
            }
            for number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    parts.append(f"## Page {number}\n\n{text}")
                # Tables are structure; pulling them out separately means tier 1
                # can answer from them without a model reading the flattened text.
                if len(tables) < 40:
                    try:
                        for table in page.find_tables().tables:
                            rows = table.extract()
                            if rows and len(rows) > 1:
                                tables.append({"page": number, "rows": rows[:200]})
                    except Exception:  # table finding is best effort
                        pass

        metadata = {k: v for k, v in metadata.items() if v is not None}
        if tables:
            metadata["tables"] = tables
            item.parsed_tree = {"tables": tables}
        return "\n\n".join(parts), metadata

    def _ocr_pdf(self, item: ExtractionItem, path: str) -> ExtractionItem:
        """Render each page and OCR it. Only reached for PDFs with no text."""
        if not config.OCR_ENABLED:
            return item.fail(Stage.PARSE, "PDF has no text layer and OCR is disabled")

        from .image import ocr_bytes

        try:
            import pymupdf
        except ImportError as exc:
            raise MissingDependency("pymupdf", "rendering a scanned PDF for OCR") from exc

        pages: list[str] = []
        with pymupdf.open(path) as document:
            for number, page in enumerate(document, start=1):
                if number > 50:  # a 500-page scan is a batch job, not a request
                    item.warn(f"OCR stopped at page 50 of {document.page_count}")
                    break
                pixmap = page.get_pixmap(dpi=200)
                text = ocr_bytes(pixmap.tobytes("png"))
                if text.strip():
                    pages.append(f"## Page {number}\n\n{text.strip()}")

        if not pages:
            return item.fail(Stage.PARSE, "OCR produced no text from the scanned PDF")
        joined = "\n\n".join(pages)
        item.decoded_text = joined
        item.cleaned_text = joined
        item.metadata["ocr"] = True
        return item

    def _read_generic(self, path: str, item: ExtractionItem) -> tuple[str, dict]:
        if config.DOCUMENT_BACKEND == "docling":
            return self._read_docling(path)
        return self._read_markitdown(path)

    @staticmethod
    def _read_markitdown(path: str) -> tuple[str, dict]:
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise MissingDependency("markitdown[all]", "document conversion") from exc

        converter = MarkItDown(enable_plugins=False)
        result = converter.convert(path)
        text = (result.text_content or "").strip()
        metadata = {"converter": "markitdown"}
        if getattr(result, "title", None):
            metadata["title"] = result.title
        if not text:
            raise ParseError(f"markitdown produced no text from {os.path.basename(path)}")
        return text, metadata

    @staticmethod
    def _read_docling(path: str) -> tuple[str, dict]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise MissingDependency("docling", "layout-aware document conversion") from exc

        converter = DocumentConverter()
        result = converter.convert(path)
        text = result.document.export_to_markdown().strip()
        if not text:
            raise ParseError("docling produced no text")
        return text, {"converter": "docling"}

    @staticmethod
    def _convert_legacy(path: str, workdir: str, target: str) -> str:
        """Convert a pre-2007 Office file with headless LibreOffice."""
        binary = _find_libreoffice()
        if binary is None:
            raise MissingDependency(
                "libreoffice (brew install --cask libreoffice)",
                f"reading legacy {os.path.splitext(path)[1]} files",
            )

        log.info("document.libreoffice_convert", source=os.path.basename(path), target=target)
        try:
            subprocess.run(
                [binary, "--headless", "--norestore", "--convert-to", target, "--outdir", workdir, path],
                check=True,
                capture_output=True,
                timeout=180,
                # LibreOffice needs a writable profile; without an explicit one
                # it can pick up a stale lock from a desktop session and hang.
                env={**os.environ, "HOME": workdir},
            )
        except subprocess.TimeoutExpired as exc:
            raise ParseError("LibreOffice conversion timed out after 180s") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode("utf-8", "ignore")[:300]
            raise ParseError(f"LibreOffice conversion failed: {detail}") from exc

        stem = os.path.splitext(os.path.basename(path))[0]
        converted = os.path.join(workdir, f"{stem}.{target}")
        if not os.path.exists(converted):
            raise ParseError(f"LibreOffice produced no {target} output")
        return converted


def _find_libreoffice() -> Optional[str]:
    for name in _LIBREOFFICE_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    return mac_path if os.path.exists(mac_path) else None


registry.register(DocumentHandler())

__all__ = ["DocumentHandler"]
