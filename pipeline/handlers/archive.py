"""Archives: zip, tar, gzip, bzip2, xz, 7z — and what is inside them.

Members are recursed into, because "the report" is routinely a ZIP containing
three PDFs and a spreadsheet, and unpacking it by hand is exactly the manual
step this pipeline exists to remove.

Recursion into attacker-supplied archives needs bounding, and there are three
distinct traps:

* **Compression bombs.** A 42 KB ZIP that expands to 4.5 PB is a real, famous
  file. Members are read against a *running total* of decompressed bytes, and
  the whole archive is abandoned when that total is exceeded — checking each
  member individually does not catch a bomb made of ten thousand small ones.
* **Path traversal.** ``../../etc/passwd`` as a member name. Nothing here is
  written to disk, but the name still becomes metadata, so it is normalised.
* **Unbounded nesting.** A ZIP of a ZIP of a ZIP. Depth is capped by
  ``MAX_RECURSION_DEPTH``.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import posixpath
import tarfile
import zipfile
from typing import Callable, Iterator, Optional

from config import config
from errors import MissingDependency, ParseError
from models import ExtractionItem, ResourceKind
from observability import get_logger

from .base import BaseHandler, registry

log = get_logger("handlers.archive")

#: Total decompressed bytes read from one archive before giving up.
MAX_TOTAL_UNCOMPRESSED = 512 * 1024 * 1024
#: Largest single member worth extracting.
MAX_MEMBER_BYTES = 64 * 1024 * 1024
#: Compression ratio above which a member is treated as a bomb rather than a
#: well-compressed file. Text compresses ~5x; 200x is not a document.
MAX_COMPRESSION_RATIO = 200


class ArchiveHandler(BaseHandler):
    name = "archive"
    kinds = (ResourceKind.ARCHIVE,)
    #: The output is child items, not prose.
    requires_text = False

    def process(self, item: ExtractionItem) -> ExtractionItem:
        data = item.raw_bytes
        if not data:
            raise ParseError("no archive bytes")

        subtype = item.metadata.get("detected_subtype", "")
        members = list(self._iter_members(data, subtype))

        if not members:
            raise ParseError(f"{subtype or 'archive'} contained no readable members")

        listing = []
        for name, payload in members:
            listing.append({"name": name, "bytes": len(payload)})
            if item.depth < config.MAX_RECURSION_DEPTH:
                child = ExtractionItem(
                    url=f"{item.url}!/{name}",
                    depth=item.depth + 1,
                    parent_url=item.url,
                    raw_bytes=payload,
                    metadata={"archive_member": name, "from_archive": item.url},
                )
                child.compute_content_hash()
                item.children.append(child)

        item.structured = {"archive_members": listing}
        item.metadata.update({"member_count": len(listing), "archive_format": subtype})
        item.cleaned_text = "\n".join(
            f"{entry['name']} ({entry['bytes']} bytes)" for entry in listing
        )
        if item.depth >= config.MAX_RECURSION_DEPTH:
            item.warn(f"archive nesting stopped at depth {item.depth}")

        log.info("archive.opened", url=item.url, format=subtype, members=len(listing))
        return item

    # ------------------------------------------------------------------ #

    def _iter_members(self, data: bytes, subtype: str) -> Iterator[tuple[str, bytes]]:
        if subtype in ("zip", "zip-empty") or data[:2] == b"PK":
            yield from self._iter_zip(data)
        elif subtype == "7z":
            yield from self._iter_7z(data)
        elif subtype in ("tar", "tar.gz") or tarfile.is_tarfile(io.BytesIO(data)):
            yield from self._iter_tar(data)
        elif subtype in ("gzip", "bzip2", "xz", "zstd"):
            yield from self._iter_single(data, subtype)
        else:
            raise ParseError(f"unsupported archive format: {subtype or 'unknown'}")

    def _iter_zip(self, data: bytes) -> Iterator[tuple[str, bytes]]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ParseError(f"corrupt ZIP: {exc}") from exc

        with archive:
            total = 0
            for count, info in enumerate(archive.infolist()):
                if count >= config.MAX_ARCHIVE_MEMBERS:
                    log.warning("archive.member_limit", limit=config.MAX_ARCHIVE_MEMBERS)
                    break
                if info.is_dir() or info.file_size == 0:
                    continue
                name = _safe_name(info.filename)
                if name is None:
                    log.warning("archive.unsafe_member_skipped", member=info.filename)
                    continue
                if info.file_size > MAX_MEMBER_BYTES:
                    log.info("archive.member_too_large", member=name, size=info.file_size)
                    continue
                if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                    log.warning(
                        "archive.bomb_suspected",
                        member=name,
                        ratio=round(info.file_size / info.compress_size),
                    )
                    continue
                total += info.file_size
                if total > MAX_TOTAL_UNCOMPRESSED:
                    log.warning("archive.total_size_limit", total=total)
                    break
                try:
                    yield name, archive.read(info)
                except (RuntimeError, zipfile.BadZipFile) as exc:
                    # Encrypted members raise RuntimeError. Skip, don't fail.
                    log.info("archive.member_unreadable", member=name, error=str(exc)[:100])

    def _iter_tar(self, data: bytes) -> Iterator[tuple[str, bytes]]:
        try:
            archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        except tarfile.TarError as exc:
            raise ParseError(f"corrupt tar: {exc}") from exc

        with archive:
            total = 0
            for count, member in enumerate(archive):
                if count >= config.MAX_ARCHIVE_MEMBERS:
                    break
                if not member.isfile() or member.size == 0:
                    continue
                name = _safe_name(member.name)
                if name is None or member.size > MAX_MEMBER_BYTES:
                    continue
                total += member.size
                if total > MAX_TOTAL_UNCOMPRESSED:
                    log.warning("archive.total_size_limit", total=total)
                    break
                handle = archive.extractfile(member)
                if handle is not None:
                    yield name, handle.read()

    def _iter_7z(self, data: bytes) -> Iterator[tuple[str, bytes]]:
        try:
            import py7zr
        except ImportError as exc:
            raise MissingDependency("py7zr", "reading 7z archives") from exc

        try:
            with py7zr.SevenZipFile(io.BytesIO(data)) as archive:
                contents = archive.readall() or {}
        except Exception as exc:
            raise ParseError(f"corrupt 7z: {exc}") from exc

        total = 0
        for count, (raw_name, handle) in enumerate(contents.items()):
            if count >= config.MAX_ARCHIVE_MEMBERS:
                break
            name = _safe_name(raw_name)
            if name is None:
                continue
            payload = handle.read(MAX_MEMBER_BYTES + 1)
            if len(payload) > MAX_MEMBER_BYTES:
                continue
            total += len(payload)
            if total > MAX_TOTAL_UNCOMPRESSED:
                break
            if payload:
                yield name, payload

    def _iter_single(self, data: bytes, subtype: str) -> Iterator[tuple[str, bytes]]:
        """gzip/bzip2/xz wrap exactly one stream — often a tar."""
        decompressors: dict[str, Callable[[bytes], bytes]] = {
            "gzip": gzip.decompress,
            "bzip2": bz2.decompress,
            "xz": lzma.decompress,
        }
        decompress = decompressors.get(subtype)
        if decompress is None:
            raise ParseError(f"no decompressor for {subtype}")

        try:
            payload = decompress(data)
        except (OSError, EOFError, lzma.LZMAError) as exc:
            raise ParseError(f"could not decompress {subtype}: {exc}") from exc

        if len(payload) > MAX_TOTAL_UNCOMPRESSED:
            raise ParseError(
                f"{subtype} stream expands to {len(payload)} bytes, over the "
                f"{MAX_TOTAL_UNCOMPRESSED}-byte limit"
            )

        if tarfile.is_tarfile(io.BytesIO(payload)):
            yield from self._iter_tar(payload)
        else:
            yield "content", payload


def _safe_name(name: str) -> Optional[str]:
    """Normalise a member path, rejecting traversal and absolute paths."""
    cleaned = name.replace("\\", "/").strip()
    if not cleaned or cleaned.startswith("/") or ".." in cleaned.split("/"):
        return None
    if cleaned.startswith("__MACOSX/") or cleaned.endswith("/.DS_Store"):
        return None
    normalized = posixpath.normpath(cleaned)
    return None if normalized.startswith(("/", "..")) else normalized


registry.register(ArchiveHandler())

__all__ = ["ArchiveHandler"]
