"""Type detection. The rule under test throughout: never trust the extension."""

from __future__ import annotations

import io
import zipfile

import pytest

from models import ResourceKind
from pipeline.detect.magic import detect, from_content_type, probe_text, sniff_bytes
from pipeline.detect.router import Acquisition, pre_route


class TestMagicBytes:
    @pytest.mark.parametrize(
        "data,kind,subtype",
        [
            (b"%PDF-1.7\n", ResourceKind.DOCUMENT, "pdf"),
            (b"\x89PNG\r\n\x1a\n", ResourceKind.IMAGE, "png"),
            (b"\xff\xd8\xff\xe0", ResourceKind.IMAGE, "jpeg"),
            (b"GIF89a", ResourceKind.IMAGE, "gif"),
            (b"ID3\x04\x00", ResourceKind.AUDIO, "mp3"),
            (b"fLaC\x00", ResourceKind.AUDIO, "flac"),
            (b"OggS\x00", ResourceKind.AUDIO, "ogg"),
            (b"\x1a\x45\xdf\xa3", ResourceKind.VIDEO, "matroska"),
            (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", ResourceKind.DOCUMENT, "ole2"),
            (b"{\\rtf1", ResourceKind.DOCUMENT, "rtf"),
            (b"7z\xbc\xaf\x27\x1c", ResourceKind.ARCHIVE, "7z"),
            (b"\x1f\x8b\x08", ResourceKind.ARCHIVE, "gzip"),
            (b"BZh9", ResourceKind.ARCHIVE, "bzip2"),
        ],
    )
    def test_signatures(self, data, kind, subtype):
        found = sniff_bytes(data)
        assert found is not None
        assert (found.kind, found.subtype) == (kind, subtype)

    @pytest.mark.parametrize(
        "form,kind,subtype",
        [(b"WAVE", ResourceKind.AUDIO, "wav"), (b"WEBP", ResourceKind.IMAGE, "webp"),
         (b"AVI ", ResourceKind.VIDEO, "avi")],
    )
    def test_riff_forms(self, form, kind, subtype):
        found = sniff_bytes(b"RIFF\x00\x00\x00\x00" + form)
        assert (found.kind, found.subtype) == (kind, subtype)

    @pytest.mark.parametrize(
        "brand,kind",
        [(b"M4A ", ResourceKind.AUDIO), (b"mp42", ResourceKind.VIDEO),
         (b"heic", ResourceKind.IMAGE), (b"avif", ResourceKind.IMAGE)],
    )
    def test_iso_bmff_brand_decides(self, brand, kind):
        """MP4, M4A, HEIC and AVIF share a container; the brand is the answer."""
        assert sniff_bytes(b"\x00\x00\x00\x18ftyp" + brand).kind is kind

    def test_empty_input(self):
        assert sniff_bytes(b"") is None


class TestZipRefinement:
    def _zip(self, members: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_docx_is_recognised_from_its_members(self):
        data = self._zip({"[Content_Types].xml": b"<x/>", "word/document.xml": b"<w/>"})
        found = detect(data)
        assert (found.kind, found.subtype) == (ResourceKind.DOCUMENT, "docx")

    def test_xlsx(self):
        data = self._zip({"xl/workbook.xml": b"<w/>"})
        assert detect(data).subtype == "xlsx"

    def test_pptx(self):
        data = self._zip({"ppt/presentation.xml": b"<p/>"})
        assert detect(data).subtype == "pptx"

    def test_epub_uses_its_mimetype_member(self):
        data = self._zip({"mimetype": b"application/epub+zip", "OEBPS/x.html": b"<p/>"})
        assert detect(data).subtype == "epub"

    def test_plain_zip_stays_an_archive(self):
        data = self._zip({"notes.txt": b"hello"})
        assert detect(data).kind is ResourceKind.ARCHIVE


class TestTextProbes:
    def test_html(self):
        assert probe_text(b"<!DOCTYPE html><html><body>x</body></html>").kind is ResourceKind.HTML

    def test_json(self):
        assert probe_text(b'{"a": 1, "b": [2]}').kind is ResourceKind.DATA

    def test_jsonl(self):
        found = probe_text(b'{"a":1}\n{"a":2}\n{"a":3}\n')
        assert found.kind is ResourceKind.DATA

    def test_sitemap_vs_feed_by_root_element(self):
        sitemap = b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
        feed = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        assert probe_text(sitemap).kind is ResourceKind.SITEMAP
        assert probe_text(feed).kind is ResourceKind.FEED

    def test_csv_needs_consistent_columns(self):
        assert probe_text(b"a,b,c\n1,2,3\n4,5,6\n").kind is ResourceKind.TABULAR
        # One comma per line by coincidence is not a CSV.
        assert probe_text(b"Hello, world\nThis is prose.\n") is None

    def test_tsv(self):
        assert probe_text(b"a\tb\n1\t2\n3\t4\n").subtype == "tsv"

    def test_email(self):
        assert probe_text(b"From: a@b.test\r\nSubject: hi\r\n\r\nbody").kind is ResourceKind.EMAIL

    def test_live_hls_vs_finished_recording(self):
        live = b"#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXTINF:6,\nseg1.ts\n"
        vod = live + b"#EXT-X-ENDLIST\n"
        assert probe_text(live).kind is ResourceKind.LIVESTREAM
        assert probe_text(vod).kind is ResourceKind.VIDEO


class TestEvidenceOrdering:
    def test_magic_bytes_beat_a_lying_content_type(self):
        found = detect(b"%PDF-1.4\n", content_type="text/html")
        assert found.kind is ResourceKind.DOCUMENT

    def test_magic_bytes_beat_a_lying_extension(self):
        """`report.pdf` that is actually an HTML login page."""
        found = detect(b"<!DOCTYPE html><html><body>Sign in</body></html>",
                       url="https://a.test/report.pdf")
        assert found.kind is ResourceKind.HTML

    def test_content_type_used_when_bytes_are_opaque(self):
        found = detect(b"\x00\x01\x02\x03", content_type="audio/mpeg")
        assert found.kind is ResourceKind.AUDIO

    def test_octet_stream_is_ignored(self):
        """The server's way of saying it has no idea either."""
        assert from_content_type("application/octet-stream") is None

    def test_a_specific_probe_beats_a_generic_content_type(self):
        """`application/xml` covers RSS, sitemaps and DASH alike."""
        rss = b'<?xml version="1.0"?><rss version="2.0"><channel><title>T</title></channel></rss>'
        assert detect(rss, content_type="application/xml").kind is ResourceKind.FEED

        sitemap = (
            b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            b"<url><loc>https://a.test/</loc></url></urlset>"
        )
        assert detect(sitemap, content_type="text/xml").kind is ResourceKind.SITEMAP

    def test_a_concrete_content_type_is_not_second_guessed(self):
        """`image/png` on bytes we cannot sniff is a real statement."""
        assert detect(b"not really a png", content_type="image/png").kind is ResourceKind.IMAGE

    def test_json_declared_as_html_is_still_json(self):
        found = detect(b'{"users": [{"name": "Ada"}]}', content_type="text/html")
        assert found.kind is ResourceKind.DATA

    def test_extension_is_the_last_resort_and_low_confidence(self):
        found = detect(b"\x00\xff\x00\xff", url="https://a.test/a.mp3")
        assert found.kind is ResourceKind.AUDIO
        assert found.confidence <= 0.5

    def test_nothing_identifiable(self):
        assert detect(b"\x00\x01\x02\xff\xfe").kind is ResourceKind.UNKNOWN


class TestPreRoute:
    @pytest.mark.parametrize(
        "url",
        ["https://www.youtube.com/watch?v=x", "https://youtu.be/x",
         "https://open.spotify.com/episode/x", "https://soundcloud.com/a/b"],
    )
    def test_media_platforms_go_to_ytdlp(self, url):
        assert pre_route(url).acquisition == Acquisition.YTDLP

    def test_lookalike_domain_is_not_misrouted(self):
        """A naive substring check would send this to yt-dlp."""
        assert pre_route("https://evil-youtube.com.attacker.net/x").acquisition == Acquisition.HTTP

    @pytest.mark.parametrize("url", ["https://cdn.test/live/a.m3u8", "https://cdn.test/x.mpd"])
    def test_streaming_manifests(self, url):
        assert pre_route(url).acquisition == Acquisition.LIVESTREAM

    def test_ordinary_url(self):
        assert pre_route("https://example.com/article").acquisition == Acquisition.HTTP
