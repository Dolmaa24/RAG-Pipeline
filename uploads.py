"""Files handed to us directly, rather than URLs we go and fetch.

An upload is already on disk on the machine the workers run on, so the
pipeline reads it from disk. The first design handed back a loopback HTTP URL
and let the fetch stage download the file from our own API, which the SSRF
policy refused on sight — ``refusing to fetch a private address (127.0.0.1)``.
The policy was right: relaxing it so we could fetch ourselves would have
opened every private address reachable from this host to any caller of
``/api/v1/extract``.

The reference is still a URL, so the rest of the pipeline — which keys caches,
provenance and citations on ``item.url`` — needs no special case:

    upload://3f9a1c2ed4b1-quarterly-report.pdf

Nothing outside :func:`upload_dir` can be named that way. The id is sanitised
on the way in and re-validated against the same pattern on the way out, and
the resolved path is checked to still sit directly inside the directory, so a
caller inventing an ``upload://`` URL of their own cannot read a file we did
not put there.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit

from config import config

UPLOAD_SCHEME = "upload"

#: Anchored at this file rather than the working directory. The API writes the
#: file and a worker in another process reads it; if the two disagreed about
#: where "output/uploads" is, every upload would fail as a missing file.
_ROOT = Path(__file__).resolve().parent

#: Lower-case because :func:`urls.canonicalize` lower-cases the authority, and
#: an id that changed case between being minted and being resolved would not
#: be found on a case-sensitive filesystem.
_ID = re.compile(r"^[0-9a-f]{12}-[a-z0-9][a-z0-9._-]{0,79}$")
_UNSAFE = re.compile(r"[^a-z0-9._-]+")

#: Long enough to keep a recognisable name, short enough to stay inside the
#: 255-byte filename limit once the id prefix is added.
_MAX_NAME = 60


def upload_dir() -> Path:
    """The directory uploads live in, created if it does not exist."""
    configured = Path(config.UPLOAD_DIR).expanduser()
    path = configured if configured.is_absolute() else _ROOT / configured
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def safe_name(filename: str | None) -> str:
    """A filename reduced to characters that cannot mean anything to a path.

    Accents are folded rather than stripped, so ``résumé.docx`` stays
    ``resume.docx`` instead of collapsing to ``r-sum.docx``. A name in a script
    with no ASCII form falls back to ``file``; the id stays unique either way,
    and only the human-readable half is lost.
    """
    name = Path(filename or "").name.lower()
    decomposed = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in decomposed if not unicodedata.combining(c))
    stem, dot, ext = name.rpartition(".")
    if not dot:  # no extension: treat the whole thing as the stem
        stem, ext = name, "bin"

    stem = _UNSAFE.sub("-", stem).strip("-._")[:_MAX_NAME]
    ext = _UNSAFE.sub("", ext)[:16]
    return f"{stem or 'file'}.{ext or 'bin'}"


def save_upload(source: BinaryIO, filename: str | None) -> str:
    """Store an uploaded stream and return the ``upload://`` URL naming it.

    The original name is kept in the id — a citation reading
    ``quarterly-report.pdf`` is worth more to whoever reads the answer than one
    reading a bare UUID — with a random prefix so two people uploading their
    own ``report.pdf`` do not overwrite each other.
    """
    file_id = f"{uuid.uuid4().hex[:12]}-{safe_name(filename)}"
    destination = upload_dir() / file_id

    with destination.open("wb") as handle:
        copied = _copy_capped(source, handle, config.MAX_CONTENT_BYTES)

    if copied > config.MAX_CONTENT_BYTES:
        # Written and then removed rather than measured first: an UploadFile
        # is a stream, and the only honest way to learn its length is to read
        # it. The cap stops that from becoming a way to fill the disk.
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"file is larger than the {config.MAX_CONTENT_BYTES}-byte limit"
        )

    return f"{UPLOAD_SCHEME}://{file_id}"


def _copy_capped(source: BinaryIO, destination: BinaryIO, limit: int) -> int:
    """Copy at most ``limit`` bytes plus one, so the caller can detect overrun."""
    copied = 0
    while chunk := source.read(shutil.COPY_BUFSIZE):
        destination.write(chunk)
        copied += len(chunk)
        if copied > limit:
            return copied
    return copied


def is_upload_url(url: str) -> bool:
    return urlsplit((url or "").strip()).scheme.lower() == UPLOAD_SCHEME


def upload_id(url: str) -> str:
    """The file id in ``url``, or "" if it is not a well-formed upload URL.

    Both the authority and the path are considered, because ``upload://x`` puts
    ``x`` in the authority while canonicalisation appends a trailing slash to
    it, and a caller may reasonably write either.
    """
    parts = urlsplit((url or "").strip())
    if parts.scheme.lower() != UPLOAD_SCHEME:
        return ""
    candidate = (parts.netloc + parts.path).strip("/")
    return candidate if _ID.match(candidate) else ""


def resolve_upload(url: str) -> Path:
    """The path ``url`` names. Raises unless it is one of ours and exists."""
    file_id = upload_id(url)
    if not file_id:
        raise ValueError(f"{url!r} is not a valid upload reference")

    directory = upload_dir()
    path = (directory / file_id).resolve()
    if path.parent != directory:
        # Unreachable through _ID, which admits no separators — kept because
        # the cost of being wrong about that is reading an arbitrary file.
        raise ValueError(f"{url!r} resolves outside the upload directory")
    if not path.is_file():
        raise FileNotFoundError(f"upload {file_id!r} is no longer on disk")
    return path


__all__ = [
    "UPLOAD_SCHEME",
    "is_upload_url",
    "resolve_upload",
    "safe_name",
    "save_upload",
    "upload_dir",
    "upload_id",
]
