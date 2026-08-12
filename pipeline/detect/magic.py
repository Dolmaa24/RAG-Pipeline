"""What are these bytes, actually?

**Detection is by magic bytes and Content-Type, never by file extension.**
That is not pedantry. In practice:

* ``/download?id=8412`` has no extension and is a 40 MB PDF;
* ``quarterly-report.pdf`` is, on a bad day, an HTML login page with a 200
  status and a ``.pdf`` in the URL;
* ``.docx``, ``.xlsx``, ``.pptx``, ``.epub`` and ``.odt`` are all ZIP archives,
  and telling them apart means reading the member list, not the name.

So the order of evidence is: magic bytes first (a file cannot lie about its
first four bytes and still be parseable), then Content-Type, then a structural
probe for the text formats that have no magic at all (JSON, CSV, YAML), and
only then the URL. The extension is used as a tiebreak and never on its own.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from typing import Optional

from models import ResourceKind
from observability import get_logger

log = get_logger("detect.magic")

#: (offset, signature, kind, subtype). Longest match wins, so more specific
#: signatures can share a prefix with more general ones.
_SIGNATURES: tuple[tuple[int, bytes, ResourceKind, str], ...] = (
    # --- documents ---
    (0, b"%PDF-", ResourceKind.DOCUMENT, "pdf"),
    (0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", ResourceKind.DOCUMENT, "ole2"),  # legacy doc/xls/ppt
    (0, b"{\\rtf", ResourceKind.DOCUMENT, "rtf"),
    (0, b"%!PS", ResourceKind.DOCUMENT, "postscript"),
    # --- images ---
    (0, b"\x89PNG\r\n\x1a\n", ResourceKind.IMAGE, "png"),
    (0, b"\xff\xd8\xff", ResourceKind.IMAGE, "jpeg"),
    (0, b"GIF87a", ResourceKind.IMAGE, "gif"),
    (0, b"GIF89a", ResourceKind.IMAGE, "gif"),
    (0, b"BM", ResourceKind.IMAGE, "bmp"),
    (0, b"II*\x00", ResourceKind.IMAGE, "tiff"),
    (0, b"MM\x00*", ResourceKind.IMAGE, "tiff"),
    (0, b"\x00\x00\x01\x00", ResourceKind.IMAGE, "ico"),
    (0, b"<svg", ResourceKind.IMAGE, "svg"),
    # --- audio ---
    (0, b"ID3", ResourceKind.AUDIO, "mp3"),
    (0, b"\xff\xfb", ResourceKind.AUDIO, "mp3"),
    (0, b"\xff\xf3", ResourceKind.AUDIO, "mp3"),
    (0, b"\xff\xf2", ResourceKind.AUDIO, "mp3"),
    (0, b"fLaC", ResourceKind.AUDIO, "flac"),
    (0, b"OggS", ResourceKind.AUDIO, "ogg"),
    (0, b"MThd", ResourceKind.AUDIO, "midi"),
    # --- video ---
    (0, b"\x1a\x45\xdf\xa3", ResourceKind.VIDEO, "matroska"),  # mkv / webm
    (0, b"FLV\x01", ResourceKind.VIDEO, "flv"),
    (0, b"\x00\x00\x01\xba", ResourceKind.VIDEO, "mpeg-ps"),
    (0, b"\x00\x00\x01\xb3", ResourceKind.VIDEO, "mpeg-vs"),
    # --- archives ---
    (0, b"PK\x03\x04", ResourceKind.ARCHIVE, "zip"),  # refined by _refine_zip
    (0, b"PK\x05\x06", ResourceKind.ARCHIVE, "zip-empty"),
    (0, b"\x1f\x8b", ResourceKind.ARCHIVE, "gzip"),
    (0, b"BZh", ResourceKind.ARCHIVE, "bzip2"),
    (0, b"\xfd7zXZ\x00", ResourceKind.ARCHIVE, "xz"),
    (0, b"7z\xbc\xaf\x27\x1c", ResourceKind.ARCHIVE, "7z"),
    (0, b"Rar!\x1a\x07", ResourceKind.ARCHIVE, "rar"),
    (0, b"\x28\xb5\x2f\xfd", ResourceKind.ARCHIVE, "zstd"),
    (257, b"ustar", ResourceKind.ARCHIVE, "tar"),
)

#: ISO-BMFF brands. The container is shared by MP4, M4A, HEIC and AVIF, so the
#: brand at offset 8 is what says whether these bytes are a photo or a film.
_FTYP_BRANDS: tuple[tuple[bytes, ResourceKind, str], ...] = (
    (b"M4A ", ResourceKind.AUDIO, "m4a"),
    (b"M4B ", ResourceKind.AUDIO, "m4b"),
    (b"mp42", ResourceKind.VIDEO, "mp4"),
    (b"mp41", ResourceKind.VIDEO, "mp4"),
    (b"isom", ResourceKind.VIDEO, "mp4"),
    (b"iso2", ResourceKind.VIDEO, "mp4"),
    (b"avc1", ResourceKind.VIDEO, "mp4"),
    (b"qt  ", ResourceKind.VIDEO, "mov"),
    (b"heic", ResourceKind.IMAGE, "heic"),
    (b"heix", ResourceKind.IMAGE, "heic"),
    (b"hevc", ResourceKind.IMAGE, "heic"),
    (b"mif1", ResourceKind.IMAGE, "heif"),
    (b"avif", ResourceKind.IMAGE, "avif"),
)

#: ZIP member paths that identify an OOXML / OpenDocument / ePub payload.
_ZIP_MARKERS: tuple[tuple[str, ResourceKind, str], ...] = (
    ("word/document.xml", ResourceKind.DOCUMENT, "docx"),
    ("ppt/presentation.xml", ResourceKind.DOCUMENT, "pptx"),
    ("xl/workbook.xml", ResourceKind.DOCUMENT, "xlsx"),
    ("xl/worksheets/", ResourceKind.DOCUMENT, "xlsx"),
    ("visio/document.xml", ResourceKind.DOCUMENT, "vsdx"),
)

_CONTENT_TYPE_MAP: tuple[tuple[str, ResourceKind, str], ...] = (
    ("text/html", ResourceKind.HTML, "html"),
    ("application/xhtml", ResourceKind.HTML, "xhtml"),
    ("application/pdf", ResourceKind.DOCUMENT, "pdf"),
    ("application/msword", ResourceKind.DOCUMENT, "doc"),
    ("application/vnd.openxmlformats", ResourceKind.DOCUMENT, "ooxml"),
    ("application/vnd.oasis.opendocument", ResourceKind.DOCUMENT, "opendocument"),
    ("application/vnd.ms-excel", ResourceKind.DOCUMENT, "xls"),
    ("application/vnd.ms-powerpoint", ResourceKind.DOCUMENT, "ppt"),
    ("application/epub", ResourceKind.DOCUMENT, "epub"),
    ("application/rtf", ResourceKind.DOCUMENT, "rtf"),
    ("text/rtf", ResourceKind.DOCUMENT, "rtf"),
    ("application/x-mobipocket", ResourceKind.DOCUMENT, "mobi"),
    ("application/rss+xml", ResourceKind.FEED, "rss"),
    ("application/atom+xml", ResourceKind.FEED, "atom"),
    ("application/feed+json", ResourceKind.FEED, "jsonfeed"),
    ("text/csv", ResourceKind.TABULAR, "csv"),
    ("text/tab-separated-values", ResourceKind.TABULAR, "tsv"),
    ("application/json", ResourceKind.DATA, "json"),
    ("application/ld+json", ResourceKind.DATA, "json"),
    ("application/x-ndjson", ResourceKind.DATA, "jsonl"),
    ("application/yaml", ResourceKind.DATA, "yaml"),
    ("text/yaml", ResourceKind.DATA, "yaml"),
    ("application/xml", ResourceKind.DATA, "xml"),
    ("text/xml", ResourceKind.DATA, "xml"),
    ("message/rfc822", ResourceKind.EMAIL, "eml"),
    ("application/vnd.ms-outlook", ResourceKind.EMAIL, "msg"),
    ("application/vnd.apple.mpegurl", ResourceKind.LIVESTREAM, "hls"),
    ("application/x-mpegurl", ResourceKind.LIVESTREAM, "hls"),
    ("application/dash+xml", ResourceKind.LIVESTREAM, "dash"),
    ("application/zip", ResourceKind.ARCHIVE, "zip"),
    ("application/x-tar", ResourceKind.ARCHIVE, "tar"),
    ("application/gzip", ResourceKind.ARCHIVE, "gzip"),
    ("application/x-7z", ResourceKind.ARCHIVE, "7z"),
    ("image/", ResourceKind.IMAGE, "image"),
    ("audio/", ResourceKind.AUDIO, "audio"),
    ("video/", ResourceKind.VIDEO, "video"),
    ("text/plain", ResourceKind.TEXT, "text"),
)

_HTML_PROBE = re.compile(rb"^\s*(<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>])", re.IGNORECASE)
_XML_PROBE = re.compile(rb"^\s*<\?xml[\s?]", re.IGNORECASE)
_EMAIL_PROBE = re.compile(
    rb"^(return-path|received|message-id|from|to|subject|mime-version|date)\s*:", re.IGNORECASE
)
_M3U8_PROBE = re.compile(rb"^\s*#EXTM3U")

#: Root elements that decide what an XML document is *for*.
_XML_ROOTS: tuple[tuple[bytes, ResourceKind, str], ...] = (
    (b"<urlset", ResourceKind.SITEMAP, "sitemap"),
    (b"<sitemapindex", ResourceKind.SITEMAP, "sitemapindex"),
    (b"<rss", ResourceKind.FEED, "rss"),
    (b"<feed", ResourceKind.FEED, "atom"),
    (b"<rdf:rdf", ResourceKind.FEED, "rdf"),
    (b"<mpd", ResourceKind.LIVESTREAM, "dash"),
)


@dataclass(frozen=True, slots=True)
class Detection:
    kind: ResourceKind
    subtype: str
    #: "magic" | "zip-members" | "content-type" | "probe" | "url" | "default"
    source: str
    confidence: float
    mime: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.kind.value}/{self.subtype} (via {self.source}, {self.confidence:.2f})"


def sniff_bytes(data: bytes) -> Optional[Detection]:
    """Identify content from its leading bytes alone. Highest confidence."""
    if not data:
        return None

    # ISO-BMFF: "....ftyp<brand>" — checked before the table because the brand
    # is what distinguishes a HEIC photo from an MP4 video.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        for candidate, kind, subtype in _FTYP_BRANDS:
            if brand == candidate:
                return Detection(kind, subtype, "magic", 1.0)
        return Detection(ResourceKind.VIDEO, "mp4", "magic", 0.8)

    if len(data) >= 12 and data[:4] == b"RIFF":
        form = data[8:12]
        if form == b"WAVE":
            return Detection(ResourceKind.AUDIO, "wav", "magic", 1.0)
        if form == b"WEBP":
            return Detection(ResourceKind.IMAGE, "webp", "magic", 1.0)
        if form == b"AVI ":
            return Detection(ResourceKind.VIDEO, "avi", "magic", 1.0)

    best: Optional[Detection] = None
    best_length = 0
    for offset, signature, kind, subtype in _SIGNATURES:
        end = offset + len(signature)
        if len(data) >= end and data[offset:end] == signature and len(signature) > best_length:
            best = Detection(kind, subtype, "magic", 1.0)
            best_length = len(signature)

    if best is not None and best.subtype == "zip":
        return _refine_zip(data) or best
    return best


def _refine_zip(data: bytes) -> Optional[Detection]:
    """A ZIP is a container. Read its member list to find out what of."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if "mimetype" in names:
                try:
                    mimetype = archive.read("mimetype").decode("ascii", "ignore").strip()
                except Exception:
                    mimetype = ""
                if mimetype == "application/epub+zip":
                    return Detection(ResourceKind.DOCUMENT, "epub", "zip-members", 1.0)
                if mimetype.startswith("application/vnd.oasis.opendocument"):
                    suffix = mimetype.rsplit(".", 1)[-1]
                    return Detection(ResourceKind.DOCUMENT, f"od{suffix[:1] or 't'}", "zip-members", 1.0)
            lowered = [n.lower() for n in names]
            for marker, kind, subtype in _ZIP_MARKERS:
                if any(name.startswith(marker) for name in lowered):
                    return Detection(kind, subtype, "zip-members", 1.0)
    except (zipfile.BadZipFile, OSError):
        # Truncated download, or a ZIP whose central directory is past the
        # bytes we hold. Treat it as a plain archive and let the handler decide.
        return None
    return Detection(ResourceKind.ARCHIVE, "zip", "zip-members", 0.9)


def from_content_type(content_type: Optional[str]) -> Optional[Detection]:
    """Map a Content-Type header onto a kind. Second-highest confidence."""
    if not content_type:
        return None
    mime = content_type.split(";", 1)[0].strip().lower()
    if not mime or mime == "application/octet-stream":
        return None  # the server's way of saying "I have no idea either"
    for prefix, kind, subtype in _CONTENT_TYPE_MAP:
        if mime.startswith(prefix):
            return Detection(kind, subtype, "content-type", 0.9, mime=mime)
    return None


#: Kinds a Content-Type can name without saying much. `application/xml` covers
#: RSS, Atom, sitemaps and DASH manifests alike; `text/plain` covers everything.
_GENERIC_KINDS = frozenset(
    {ResourceKind.HTML, ResourceKind.TEXT, ResourceKind.DATA, ResourceKind.UNKNOWN}
)


def _more_specific(probed: Detection, declared: Detection) -> bool:
    """Whether a probe result should override what the server declared."""
    if declared.kind not in _GENERIC_KINDS:
        return False  # a concrete declaration (image/, audio/, pdf) is believed
    if probed.kind in _GENERIC_KINDS:
        # Both are generic: only override when the probe is clearly confident,
        # so `application/json` is not downgraded by a weak CSV guess.
        return probed.kind is not declared.kind and probed.confidence >= 0.9
    return True


def probe_text(data: bytes) -> Optional[Detection]:
    """Structural probe for the formats with no magic bytes at all."""
    if not data:
        return None
    head = data[:8192]
    stripped = head.lstrip()

    if _M3U8_PROBE.match(stripped):
        kind = ResourceKind.LIVESTREAM if b"#EXT-X-ENDLIST" not in data[:65536] else ResourceKind.VIDEO
        return Detection(kind, "hls", "probe", 0.95)

    if _HTML_PROBE.match(stripped):
        return Detection(ResourceKind.HTML, "html", "probe", 0.95)

    if _XML_PROBE.match(stripped) or stripped.startswith(b"<"):
        lowered = head.lower()
        for root, kind, subtype in _XML_ROOTS:
            if root in lowered:
                return Detection(kind, subtype, "probe", 0.95)
        if b"<html" in lowered:
            return Detection(ResourceKind.HTML, "html", "probe", 0.9)
        if stripped.startswith(b"<?xml") or b"<" in stripped[:64]:
            return Detection(ResourceKind.DATA, "xml", "probe", 0.8)

    if stripped[:1] in (b"{", b"["):
        try:
            json.loads(head.decode("utf-8", "ignore"))
            return Detection(ResourceKind.DATA, "json", "probe", 0.95)
        except (ValueError, UnicodeDecodeError):
            # A truncated head of a large but valid JSON document still starts
            # with a brace and still has quoted keys.
            if b'":' in head or b'": ' in head:
                return Detection(ResourceKind.DATA, "json", "probe", 0.7)
            # JSONL: each line is its own object.
            first_line = head.split(b"\n", 1)[0]
            try:
                json.loads(first_line.decode("utf-8", "ignore"))
                return Detection(ResourceKind.DATA, "jsonl", "probe", 0.8)
            except (ValueError, UnicodeDecodeError):
                pass

    if _EMAIL_PROBE.match(stripped):
        return Detection(ResourceKind.EMAIL, "eml", "probe", 0.85)

    delimiter = _sniff_delimiter(head)
    if delimiter:
        return Detection(ResourceKind.TABULAR, "tsv" if delimiter == "\t" else "csv", "probe", 0.75)

    return None


def _sniff_delimiter(head: bytes) -> Optional[str]:
    """Consistent column counts across the first lines mean a delimited file.

    Requires at least two rows and more than one column, so a text file whose
    every line happens to contain one comma is not mistaken for a CSV.
    """
    try:
        text = head.decode("utf-8", "ignore")
    except Exception:
        return None
    lines = [line for line in text.splitlines()[:8] if line.strip()]
    if len(lines) < 2:
        return None
    for delimiter in ("\t", ",", ";", "|"):
        counts = [line.count(delimiter) for line in lines]
        if counts[0] >= 1 and len(set(counts)) == 1:
            return delimiter
    return None


def detect(
    data: Optional[bytes] = None,
    *,
    content_type: Optional[str] = None,
    url: Optional[str] = None,
) -> Detection:
    """Best available identification, most trustworthy evidence first."""
    if data:
        found = sniff_bytes(data)
        if found is not None:
            return found

    declared = from_content_type(content_type)
    # A Content-Type is often the laziest true statement a server can make:
    # `text/html` on a JSON API, `application/xml` on an RSS feed or a sitemap.
    # Those are not wrong, just useless — so when the probe finds something
    # *more specific* than the generic kind that was declared, the probe wins.
    if data:
        probed = probe_text(data)
        if probed is not None and (declared is None or _more_specific(probed, declared)):
            return probed
    if declared is not None:
        return declared

    if url:
        guessed = from_url(url)
        if guessed is not None:
            return guessed

    if data:
        probed = probe_text(data)
        if probed is not None:
            return probed
        if _looks_textual(data):
            return Detection(ResourceKind.TEXT, "text", "default", 0.4)

    return Detection(ResourceKind.UNKNOWN, "unknown", "default", 0.0)


#: Extension hints. Deliberately last, and never above 0.5 confidence — this
#: is a tiebreak for bytes we could not otherwise place, not an identification.
_EXTENSION_HINTS: dict[str, tuple[ResourceKind, str]] = {
    "pdf": (ResourceKind.DOCUMENT, "pdf"), "doc": (ResourceKind.DOCUMENT, "doc"),
    "docx": (ResourceKind.DOCUMENT, "docx"), "xls": (ResourceKind.DOCUMENT, "xls"),
    "xlsx": (ResourceKind.DOCUMENT, "xlsx"), "ppt": (ResourceKind.DOCUMENT, "ppt"),
    "pptx": (ResourceKind.DOCUMENT, "pptx"), "odt": (ResourceKind.DOCUMENT, "odt"),
    "epub": (ResourceKind.DOCUMENT, "epub"), "mobi": (ResourceKind.DOCUMENT, "mobi"),
    "rtf": (ResourceKind.DOCUMENT, "rtf"),
    "csv": (ResourceKind.TABULAR, "csv"), "tsv": (ResourceKind.TABULAR, "tsv"),
    "json": (ResourceKind.DATA, "json"), "jsonl": (ResourceKind.DATA, "jsonl"),
    "ndjson": (ResourceKind.DATA, "jsonl"), "xml": (ResourceKind.DATA, "xml"),
    "yaml": (ResourceKind.DATA, "yaml"), "yml": (ResourceKind.DATA, "yaml"),
    "html": (ResourceKind.HTML, "html"), "htm": (ResourceKind.HTML, "html"),
    "txt": (ResourceKind.TEXT, "text"), "md": (ResourceKind.TEXT, "markdown"),
    "png": (ResourceKind.IMAGE, "png"), "jpg": (ResourceKind.IMAGE, "jpeg"),
    "jpeg": (ResourceKind.IMAGE, "jpeg"), "gif": (ResourceKind.IMAGE, "gif"),
    "webp": (ResourceKind.IMAGE, "webp"), "heic": (ResourceKind.IMAGE, "heic"),
    "tif": (ResourceKind.IMAGE, "tiff"), "tiff": (ResourceKind.IMAGE, "tiff"),
    "mp3": (ResourceKind.AUDIO, "mp3"), "wav": (ResourceKind.AUDIO, "wav"),
    "m4a": (ResourceKind.AUDIO, "m4a"), "aac": (ResourceKind.AUDIO, "aac"),
    "flac": (ResourceKind.AUDIO, "flac"), "ogg": (ResourceKind.AUDIO, "ogg"),
    "opus": (ResourceKind.AUDIO, "opus"),
    "mp4": (ResourceKind.VIDEO, "mp4"), "mkv": (ResourceKind.VIDEO, "matroska"),
    "mov": (ResourceKind.VIDEO, "mov"), "avi": (ResourceKind.VIDEO, "avi"),
    "webm": (ResourceKind.VIDEO, "webm"),
    "m3u8": (ResourceKind.LIVESTREAM, "hls"), "mpd": (ResourceKind.LIVESTREAM, "dash"),
    "zip": (ResourceKind.ARCHIVE, "zip"), "tar": (ResourceKind.ARCHIVE, "tar"),
    "gz": (ResourceKind.ARCHIVE, "gzip"), "tgz": (ResourceKind.ARCHIVE, "tar.gz"),
    "bz2": (ResourceKind.ARCHIVE, "bzip2"), "7z": (ResourceKind.ARCHIVE, "7z"),
    "xz": (ResourceKind.ARCHIVE, "xz"), "rar": (ResourceKind.ARCHIVE, "rar"),
    "eml": (ResourceKind.EMAIL, "eml"), "msg": (ResourceKind.EMAIL, "msg"),
}


def from_url(url: str) -> Optional[Detection]:
    from urls import url_filename

    name = url_filename(url).lower()
    if "." not in name:
        return None
    extension = name.rsplit(".", 1)[-1]
    hint = _EXTENSION_HINTS.get(extension)
    if hint is None:
        return None
    kind, subtype = hint
    return Detection(kind, subtype, "url", 0.5)


def _looks_textual(data: bytes) -> bool:
    """Heuristic: mostly printable, no NUL bytes in the first block."""
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    printable = sum(1 for byte in sample if 0x20 <= byte < 0x7F or byte in (9, 10, 13))
    return bool(sample) and printable / len(sample) > 0.85


__all__ = ["Detection", "detect", "from_content_type", "from_url", "probe_text", "sniff_bytes"]
