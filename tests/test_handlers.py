"""Handlers, including the safety limits on recursion into archives."""

from __future__ import annotations

import io
import zipfile

import pytest

from config import config
from models import ExtractionItem, ResourceKind
from pipeline.detect import TypeRouter
from pipeline.handlers import registry


def handled(url: str, data: bytes, content_type: str | None = None) -> ExtractionItem:
    item = ExtractionItem(url=url, raw_bytes=data, content_type=content_type)
    TypeRouter.route(item)
    return registry.dispatch(item)


class TestHtmlHandler:
    def test_extracts_text_metadata_and_structure(self, product_html):
        item = handled("https://shop.test/p/1", product_html)
        assert item.ok
        assert item.metadata["title"] == "Blue Widget | ACME"
        assert item.metadata["description"] == "A very blue widget."
        assert "jsonld" in item.structured
        assert "widget" in item.cleaned_text.lower()

    def test_navigation_and_footer_are_stripped(self, product_html):
        item = handled("https://shop.test/p/1", product_html)
        assert "Home Shop Contact" not in item.cleaned_text

    def test_tables_are_kept_as_structure(self, product_html):
        tables = handled("https://shop.test/p/1", product_html).parsed_tree["tables"]
        assert tables[0]["headers"] == ["Spec", "Value"]
        assert ["Colour", "Blue"] in tables[0]["rows"]

    def test_links_are_absolute(self):
        html = b'<html><body><main><p>x</p><a href="/a">A</a><a href="#skip">S</a></main></body></html>'
        links = handled("https://a.test/dir/page", html).parsed_tree["links"]
        assert links == [{"url": "https://a.test/a", "text": "A"}]

    def test_empty_document_fails(self):
        item = handled("https://a.test/x", b"<html><body></body></html>")
        assert not item.ok


class TestDataHandlers:
    def test_csv_becomes_records(self):
        item = handled("https://a.test/d.csv", b"country,gdp\nNepal,40\nBhutan,3\n")
        assert item.ok
        table = item.structured["table"]
        assert table["headers"] == ["country", "gdp"]
        assert table["records"][0] == {"country": "Nepal", "gdp": "40"}

    def test_headerless_csv_gets_positional_columns(self):
        item = handled("https://a.test/d.csv", b"1,2,3\n4,5,6\n7,8,9\n")
        assert item.structured["table"]["headers"] == ["col_0", "col_1", "col_2"]

    def test_tsv_delimiter(self):
        item = handled("https://a.test/d", b"a\tb\n1\t2\n3\t4\n")
        assert item.metadata["delimiter"] == "\t"

    def test_json(self):
        item = handled("https://a.test/api", b'{"users":[{"name":"Ada"}]}')
        assert item.ok
        assert item.structured["data"]["users"][0]["name"] == "Ada"

    def test_malformed_json_fails_clearly(self):
        item = handled("https://a.test/api", b'{"a": ', content_type="application/json")
        assert not item.ok
        assert "JSON" in item.error

    def test_xml_becomes_nested_dicts(self):
        xml = b'<?xml version="1.0"?><root><item id="1">first</item><item>second</item></root>'
        item = handled("https://a.test/d.xml", xml, content_type="application/xml")
        assert item.ok
        assert item.structured["data"]["item"][0]["#text"] == "first"

    def test_xml_external_entities_are_not_expanded(self):
        """XXE: this document came off the internet."""
        xxe = (
            b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            b"<root><a>&x;</a></root>"
        )
        item = handled("https://a.test/d.xml", xxe, content_type="application/xml")
        assert "root:" not in (item.cleaned_text or "")


class TestFeedAndSitemap:
    RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
    <title>Notes</title>
    <item><title>First</title><link>https://a.test/1</link>
          <author>ada@a.test</author><description>&lt;p&gt;Body one&lt;/p&gt;</description></item>
    <item><title>Second</title><link>/2</link></item>
    </channel></rss>"""

    SITEMAP = b"""<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://a.test/x</loc><lastmod>2026-01-01</lastmod></url>
      <url><loc>https://a.test/y</loc></url>
    </urlset>"""

    def test_feed_entries_become_children(self):
        item = handled("https://a.test/feed.xml", self.RSS)
        assert item.ok
        assert item.metadata["title"] == "Notes"
        assert len(item.structured["feed_entries"]) == 2
        assert {child.url for child in item.children} == {"https://a.test/1", "https://a.test/2"}

    def test_feed_html_summaries_are_reduced_to_text(self):
        entries = handled("https://a.test/feed.xml", self.RSS).structured["feed_entries"]
        assert entries[0]["summary"] == "Body one"

    def test_sitemap_yields_a_url_inventory(self):
        item = handled("https://a.test/sitemap.xml", self.SITEMAP)
        assert item.ok
        assert item.metadata["url_count"] == 2
        assert len(item.children) == 2

    def test_recursion_depth_is_respected(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_RECURSION_DEPTH", 0, raising=False)
        item = ExtractionItem(url="https://a.test/sitemap.xml", raw_bytes=self.SITEMAP)
        TypeRouter.route(item)
        item = registry.dispatch(item)
        assert item.ok and item.children == []


class TestEmailHandler:
    EML = (
        b"From: ada@a.test\r\nTo: bob@b.test\r\nSubject: Invoice\r\n"
        b"Date: Mon, 1 Jun 2026 10:00:00 +0000\r\nMIME-Version: 1.0\r\n"
        b'Content-Type: multipart/mixed; boundary="B"\r\n\r\n'
        b"--B\r\nContent-Type: text/plain\r\n\r\nPlease find it attached.\r\n"
        b"--B\r\nContent-Type: text/csv\r\n"
        b'Content-Disposition: attachment; filename="inv.csv"\r\n\r\nitem,total\r\nwidget,42\r\n'
        b"--B--\r\n"
    )

    def test_headers_and_body(self):
        item = handled("https://a.test/m.eml", self.EML)
        assert item.ok
        assert item.metadata["subject"] == "Invoice"
        assert "Please find it attached." in item.cleaned_text

    def test_attachments_become_children(self):
        """The attachment is usually the actual content."""
        item = handled("https://a.test/m.eml", self.EML)
        assert len(item.children) == 1
        child = item.children[0]
        assert child.metadata["filename"] == "inv.csv"

        TypeRouter.route(child)
        parsed = registry.dispatch(child)
        assert parsed.ok and parsed.kind is ResourceKind.TABULAR


class TestArchiveHandler:
    def _zip(self, members: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_members_become_children(self):
        data = self._zip({"a.csv": b"x,y\n1,2\n", "b.json": b'{"k":1}'})
        item = handled("https://a.test/bundle.zip", data)
        assert item.ok
        assert {child.metadata["archive_member"] for child in item.children} == {"a.csv", "b.json"}

    def test_children_are_typed_and_handled(self):
        data = self._zip({"rows.csv": b"a,b\n1,2\n3,4\n"})
        parent = handled("https://a.test/bundle.zip", data)
        child = parent.children[0]
        TypeRouter.route(child)
        assert registry.dispatch(child).kind is ResourceKind.TABULAR

    def test_path_traversal_members_are_skipped(self):
        data = self._zip({"../../etc/passwd": b"root:x:0", "ok.txt": b"fine"})
        item = handled("https://a.test/evil.zip", data)
        names = {child.metadata["archive_member"] for child in item.children}
        assert names == {"ok.txt"}

    def test_absolute_paths_are_skipped(self):
        data = self._zip({"/etc/shadow": b"secret", "ok.txt": b"fine"})
        item = handled("https://a.test/evil.zip", data)
        assert all("shadow" not in child.url for child in item.children)

    def test_compression_bomb_member_is_refused(self):
        """42 KB in, petabytes out, is a real file."""
        data = self._zip({"bomb.txt": b"\x00" * (5 * 1024 * 1024), "ok.txt": b"fine"})
        item = handled("https://a.test/bomb.zip", data)
        names = {child.metadata["archive_member"] for child in item.children}
        assert "bomb.txt" not in names
        assert "ok.txt" in names

    def test_member_count_is_bounded(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_ARCHIVE_MEMBERS", 3, raising=False)
        data = self._zip({f"f{i}.txt": b"x" * 100 for i in range(20)})
        item = handled("https://a.test/many.zip", data)
        assert len(item.children) <= 3

    def test_gzip_wrapping_a_single_file(self):
        import gzip

        data = gzip.compress(b"name,qty\nwidget,3\n")
        item = handled("https://a.test/rows.csv.gz", data)
        assert item.ok and len(item.children) == 1

    def test_corrupt_archive_fails_clearly(self):
        item = handled("https://a.test/x.zip", b"PK\x03\x04" + b"\x00" * 40)
        assert not item.ok


class TestUnknownContent:
    def test_plain_text_still_produces_text(self):
        item = handled("https://a.test/notes", b"The heron stood still for eleven minutes.")
        assert item.ok
        assert "heron" in item.cleaned_text

    def test_low_confidence_detection_is_flagged(self):
        item = handled("https://a.test/x", b"\x00\x01\x02\xff\xfe\xfd")
        assert any("low-confidence" in warning for warning in item.warnings)
