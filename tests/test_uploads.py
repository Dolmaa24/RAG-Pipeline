"""Uploaded files: named safely, read from disk, and confined to one directory.

The bug these cover: uploads used to be handed back as ``http://127.0.0.1:8000/
uploads/<id>``, which the SSRF policy refused — correctly, since fetching our
own loopback address is exactly what that policy exists to stop. The fix reads
the file from disk instead, which is only safe if an ``upload://`` URL cannot
be made to name a file we did not write.
"""

from __future__ import annotations

import io

import pytest

from models import ExtractionItem, FetchMode
from pipeline.detect.router import Acquisition, pre_route
from pipeline.fetch.client import ResilientFetcher
import uploads
from uploads import (
    is_upload_url,
    resolve_upload,
    safe_name,
    save_upload,
    upload_id,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the upload directory at a temporary one for the whole test."""
    monkeypatch.setattr(uploads.config, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "given,expected",
    [
        ("Q4 Report (FINAL).PDF", "q4-report-final.pdf"),
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd.bin"),
        ("/absolute/path/notes.txt", "notes.txt"),
        ("no-extension", "no-extension.bin"),
        ("", "file.bin"),
        (None, "file.bin"),
        ("..", "file.bin"),
        ("résumé.docx", "resume.docx"),
        ("Übersicht Q1.xlsx", "ubersicht-q1.xlsx"),
        ("रिपोर्ट.pdf", "file.pdf"),
    ],
)
def test_safe_name(given, expected):
    assert safe_name(given) == expected


def test_safe_name_is_always_a_bare_filename():
    for hostile in ("../x", "a/b/c.pdf", "a\\b.pdf", "....//x.pdf", "~/.ssh/id_rsa"):
        assert "/" not in safe_name(hostile)
        assert "\\" not in safe_name(hostile)


def test_two_uploads_of_the_same_name_do_not_collide(store):
    first = save_upload(io.BytesIO(b"one"), "report.pdf")
    second = save_upload(io.BytesIO(b"two"), "report.pdf")

    assert first != second
    assert resolve_upload(first).read_bytes() == b"one"
    assert resolve_upload(second).read_bytes() == b"two"


def test_the_original_name_survives_in_the_reference(store):
    # A citation reading "quarterly-report.pdf" is worth more than a bare UUID.
    assert save_upload(io.BytesIO(b"x"), "Quarterly Report.pdf").endswith(
        "-quarterly-report.pdf"
    )


def test_is_upload_url():
    assert is_upload_url("upload://abc")
    assert is_upload_url("UPLOAD://abc")
    assert not is_upload_url("https://example.com/a.pdf")
    assert not is_upload_url("file:///etc/passwd")
    assert not is_upload_url("")


def test_upload_id_accepts_the_canonicalised_form():
    from urls import canonicalize

    url = "upload://3f9a1c2ed4b1-report.pdf"
    assert upload_id(url) == "3f9a1c2ed4b1-report.pdf"
    # canonicalize() appends a trailing slash to an empty path; the id must
    # survive that, because the pipeline canonicalises before it fetches.
    assert upload_id(canonicalize(url)) == "3f9a1c2ed4b1-report.pdf"


@pytest.mark.parametrize(
    "hostile",
    [
        "upload://../../../../etc/passwd",
        "upload://..%2f..%2fetc%2fpasswd",
        "upload:///etc/passwd",
        "upload://etc/passwd",
        "upload://3f9a1c2ed4b1-../../../etc/passwd",
        "upload://3f9a1c2ed4b1-report.pdf/../../etc/passwd",
        "upload://UPPERCASE-report.pdf",
        "upload://",
        "https://example.com/a.pdf",
    ],
)
def test_upload_id_rejects_anything_it_did_not_mint(hostile):
    assert upload_id(hostile) == ""


def test_resolve_refuses_an_id_we_never_minted(store):
    with pytest.raises(ValueError):
        resolve_upload("upload://../../etc/passwd")


def test_resolve_reports_a_missing_file_as_missing(store):
    with pytest.raises(FileNotFoundError):
        resolve_upload("upload://000000000000-gone.pdf")


def test_resolve_returns_a_path_inside_the_upload_directory(store):
    url = save_upload(io.BytesIO(b"payload"), "doc.pdf")
    path = resolve_upload(url)

    assert path.parent == store.resolve()
    assert path.read_bytes() == b"payload"


def test_an_oversized_upload_is_rejected_and_leaves_nothing_behind(store, monkeypatch):
    monkeypatch.setattr(uploads.config, "MAX_CONTENT_BYTES", 16)

    with pytest.raises(ValueError, match="larger than"):
        save_upload(io.BytesIO(b"x" * 1024), "big.pdf")

    assert list(store.iterdir()) == []


def test_a_file_at_the_limit_is_kept(store, monkeypatch):
    monkeypatch.setattr(uploads.config, "MAX_CONTENT_BYTES", 16)
    url = save_upload(io.BytesIO(b"x" * 16), "small.pdf")
    assert resolve_upload(url).stat().st_size == 16


def test_fetch_reads_an_upload_from_disk(store):
    url = save_upload(io.BytesIO(b"%PDF-1.7\n"), "report.pdf")
    item = ResilientFetcher().fetch(ExtractionItem(url=url, job_id="t"))

    assert item.ok
    assert item.raw_bytes == b"%PDF-1.7\n"
    assert item.status_code == 200
    assert item.fetch_mode is FetchMode.INLINE
    assert item.content_type == "application/pdf"


def test_fetch_makes_no_network_call_for_an_upload(store, monkeypatch):
    url = save_upload(io.BytesIO(b"bytes"), "doc.txt")

    def explode(*args, **kwargs):  # pragma: no cover - the point is not calling it
        raise AssertionError("an upload must not be fetched over the network")

    monkeypatch.setattr(ResilientFetcher, "_fetch_static", explode)
    monkeypatch.setattr(ResilientFetcher, "_fetch_browser", explode)

    assert ResilientFetcher().fetch(ExtractionItem(url=url, job_id="t")).ok


def test_fetch_fails_cleanly_on_an_invented_reference(store):
    item = ResilientFetcher().fetch(
        ExtractionItem(url="upload://../../etc/passwd", job_id="t")
    )

    assert not item.ok
    assert "not a valid upload reference" in item.error


def test_fetch_still_refuses_the_file_scheme(store):
    # Local reading is reachable only through an id we minted, never through a
    # path the caller chose.
    item = ResilientFetcher().fetch(
        ExtractionItem(url="file:///etc/passwd", job_id="t")
    )
    assert not item.ok
    assert "not an http(s) URL" in item.error


def test_an_upload_over_the_size_limit_is_refused_at_fetch(store, monkeypatch):
    url = save_upload(io.BytesIO(b"x" * 64), "doc.txt")
    monkeypatch.setattr("pipeline.fetch.client.config.MAX_CONTENT_BYTES", 8)

    item = ResilientFetcher().fetch(ExtractionItem(url=url, job_id="t"))
    assert not item.ok
    assert "limit" in item.error


def test_an_upload_is_never_routed_to_the_media_downloader():
    # yt-dlp exists to handle player pages. A filename that happens to read like
    # one is still a file on our disk.
    for name in (
        "upload://3f9a1c2ed4b1-youtube.com.pdf",
        "upload://3f9a1c2ed4b1-live-stream.m3u8",
        "upload://3f9a1c2ed4b1-report.pdf",
    ):
        assert pre_route(name).acquisition == Acquisition.HTTP
