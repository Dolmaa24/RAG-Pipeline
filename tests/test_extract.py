"""The extraction cascade: its tiers, its ordering, and its schema handling."""

from __future__ import annotations

import pytest

from config import config
from models import ExtractionItem, ExtractionMethod, ResourceKind
from pipeline.detect import TypeRouter
from pipeline.extract import schema as schema_module
from pipeline.extract.cascade import ExtractionCascade
from pipeline.extract.selectors import (
    SelectorRule,
    SelectorSpec,
    SpecStore,
    apply_spec,
    dom_skeleton,
    _selector_is_sane,
)
from pipeline.extract.structured import harvest, map_to_schema
from pipeline.handlers import registry

PRODUCT_SCHEMA = {
    "title": "string",
    "price": "string",
    "brand": "string",
    "rating": "string",
}


def parsed(url: str, html: bytes) -> ExtractionItem:
    item = ExtractionItem(url=url, raw_bytes=html, content_type="text/html")
    TypeRouter.route(item)
    return registry.dispatch(item)


# --------------------------------------------------------------------------- #
# Tier 1: the publisher's own structured data
# --------------------------------------------------------------------------- #


class TestStructuredData:
    def test_jsonld_graph_is_flattened(self, product_html):
        found = harvest(product_html.decode())
        types = {node.get("@type") for node in found["jsonld"]}
        assert "Product" in types

    def test_opengraph_is_harvested(self, product_html):
        assert harvest(product_html.decode())["opengraph"]["og:site_name"] == "ACME Store"

    def test_nested_values_are_reachable(self, product_html):
        """`offers.price` and `brand.name` are where the answers actually live."""
        data, fill = map_to_schema(harvest(product_html.decode()), PRODUCT_SCHEMA)
        assert data["price"] == "19.99"
        assert data["brand"] == "ACME"
        assert data["rating"] == "4.6"
        assert fill == 1.0

    def test_the_page_subject_wins_over_the_site(self, product_html):
        """A page carries Organization and BreadcrumbList nodes too."""
        data, _ = map_to_schema(harvest(product_html.decode()), {"title": "string"})
        assert data["title"] == "Blue Widget"

    def test_synonyms_map_schema_words_to_schema_org_words(self):
        html = """<html><head><script type="application/ld+json">
        {"@type":"NewsArticle","headline":"Heron sighted","author":{"name":"Ada"},
         "datePublished":"2026-03-01"}</script></head><body>x</body></html>"""
        data, fill = map_to_schema(harvest(html), {"title": "string", "author": "string", "date": "string"})
        assert data == {"title": "Heron sighted", "author": "Ada", "date": "2026-03-01"}
        assert fill == 1.0

    def test_list_fields_are_coerced(self):
        html = """<html><head><script type="application/ld+json">
        {"@type":"Article","keywords":"birds, rivers, patience"}</script></head><body>x</body></html>"""
        data, _ = map_to_schema(harvest(html), {"tags": "list of strings"})
        assert data["tags"] == ["birds", "rivers", "patience"]

    def test_next_data_is_unwrapped_to_page_props(self):
        html = """<html><body><script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"name":"Widget","price":"9.99"}}}</script></body></html>"""
        assert harvest(html)["next_data"] == {"name": "Widget", "price": "9.99"}

    def test_malformed_jsonld_with_trailing_commas_is_repaired(self):
        html = """<html><head><script type="application/ld+json">
        {"@type":"Product","name":"Widget",}</script></head><body>x</body></html>"""
        assert harvest(html)["jsonld"][0]["name"] == "Widget"

    def test_page_with_no_structure_yields_nothing(self, plain_html):
        assert harvest(plain_html.decode()) == {}


# --------------------------------------------------------------------------- #
# Schema compilation and repair
# --------------------------------------------------------------------------- #


class TestSchemaCompilation:
    def test_friendly_hint_becomes_json_schema(self):
        compiled = schema_module.compile_schema({"title": "string", "count": "integer"})
        assert compiled["properties"]["title"]["type"] == ["string", "null"]
        assert compiled["properties"]["count"]["type"] == ["integer", "null"]
        assert compiled["additionalProperties"] is False

    def test_every_field_is_required_so_absence_is_explicit(self):
        compiled = schema_module.compile_schema({"a": "string", "b": "string"})
        assert set(compiled["required"]) == {"a", "b"}

    def test_list_hints(self):
        compiled = schema_module.compile_schema({"tags": "list of strings"})
        assert compiled["properties"]["tags"]["type"] == ["array", "null"]
        assert compiled["properties"]["tags"]["items"]["type"] == ["string", "null"]

    def test_nested_objects(self):
        compiled = schema_module.compile_schema({"author": {"name": "string", "url": "string"}})
        assert "name" in compiled["properties"]["author"]["properties"]

    def test_an_existing_json_schema_passes_through(self):
        original = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert schema_module.compile_schema(original) is original

    def test_hash_is_order_independent(self):
        assert schema_module.schema_hash({"a": "string", "b": "string"}) == schema_module.schema_hash(
            {"b": "string", "a": "string"}
        )


class TestSchemaValidation:
    def test_missing_field_is_caught(self):
        compiled = schema_module.compile_schema({"a": "string", "b": "string"})
        assert schema_module.validate({"a": "x"}, compiled)

    def test_unexpected_field_is_caught(self):
        compiled = schema_module.compile_schema({"a": "string"})
        assert schema_module.validate({"a": "x", "surprise": 1}, compiled)

    def test_null_is_a_valid_answer(self):
        compiled = schema_module.compile_schema({"a": "string"})
        assert schema_module.validate({"a": None}, compiled) == []


class TestSchemaRepair:
    def test_renamed_fields_are_matched_case_insensitively(self):
        """The classic failure: the model returns `Title` where you asked for `title`."""
        compiled = schema_module.compile_schema({"title": "string", "review_count": "integer"})
        repaired = schema_module.coerce_to_schema({"Title": "x", "reviewCount": "7"}, compiled)
        assert repaired == {"title": "x", "review_count": 7}

    def test_single_key_envelope_is_unwrapped(self):
        compiled = schema_module.compile_schema({"title": "string"})
        assert schema_module.coerce_to_schema({"result": {"title": "x"}}, compiled) == {"title": "x"}

    def test_scalar_is_promoted_to_a_list(self):
        compiled = schema_module.compile_schema({"tags": "list of strings"})
        assert schema_module.coerce_to_schema({"tags": "a, b"}, compiled) == {"tags": ["a", "b"]}

    def test_a_bare_list_is_kept_rather_than_truncated(self):
        compiled = schema_module.compile_schema({"title": "string"})
        assert schema_module.coerce_to_schema([{"title": "a"}, {"title": "b"}], compiled) == {
            "items": [{"title": "a"}, {"title": "b"}]
        }

    def test_empty_string_becomes_null(self):
        compiled = schema_module.compile_schema({"a": "string"})
        assert schema_module.coerce_to_schema({"a": ""}, compiled) == {"a": None}


def test_fill_rate():
    hint = {"a": "string", "b": "string", "c": "string", "d": "string"}
    assert schema_module.fill_rate({"a": "x", "b": "", "c": None, "d": []}, hint) == 0.25


# --------------------------------------------------------------------------- #
# Tier 2: selector specs
# --------------------------------------------------------------------------- #


class TestSelectorSpecs:
    def _spec(self) -> SelectorSpec:
        return SelectorSpec(
            domain="shop.test",
            schema_hash="abc",
            rules={
                "title": SelectorRule("h1.product-title"),
                "price": SelectorRule("span.price"),
                "brand": SelectorRule('meta[property="og:site_name"]', attribute="content"),
                "rating": SelectorRule("span.price", regex=r"\$([\d.]+)"),
            },
        )

    def test_apply_reads_text_attributes_and_regex(self, product_html):
        data, fill = apply_spec(self._spec(), product_html.decode(), PRODUCT_SCHEMA)
        assert data["title"] == "Blue Widget"
        assert data["price"] == "$19.99"
        assert data["brand"] == "ACME Store"
        assert data["rating"] == "19.99"
        assert fill == 1.0

    def test_a_spec_that_matches_nothing_reports_zero_fill(self, plain_html):
        _, fill = apply_spec(self._spec(), plain_html.decode(), PRODUCT_SCHEMA)
        assert fill == 0.0

    def test_multiple_collects_every_match(self):
        html = "<html><body><ul><li class='t'>a</li><li class='t'>b</li></ul></body></html>"
        spec = SelectorSpec("a.test", "h", {"tags": SelectorRule("li.t", multiple=True)})
        data, _ = apply_spec(spec, html, {"tags": "list of strings"})
        assert data["tags"] == ["a", "b"]

    def test_a_broken_selector_does_not_fail_the_whole_spec(self, product_html):
        spec = self._spec()
        spec.rules["price"] = SelectorRule("<<< not a selector >>>")
        data, _ = apply_spec(spec, product_html.decode(), PRODUCT_SCHEMA)
        assert data["title"] == "Blue Widget"
        assert data["price"] is None

    @pytest.mark.parametrize(
        "selector",
        ["div.css-1x2y3z", "div.sc-fJbEBl > span", "ul > li:nth-child(3)", "div.styles__wrapper"],
    )
    def test_unstable_selectors_are_rejected(self, selector):
        """A generated class name is stale before the spec is even stored."""
        assert not _selector_is_sane(selector)

    @pytest.mark.parametrize(
        "selector", ["h1.product-title", "[itemprop=price]", "main article > p", "#content .price"]
    )
    def test_stable_selectors_are_accepted(self, selector):
        assert _selector_is_sane(selector)


class TestSpecDrift:
    def test_a_spec_whose_fill_rate_collapses_is_stale(self):
        spec = SelectorSpec("a.test", "h")
        for _ in range(6):
            spec.record_use(0.0)
        assert spec.is_stale

    def test_a_healthy_spec_is_not_stale(self):
        spec = SelectorSpec("a.test", "h")
        for _ in range(10):
            spec.record_use(1.0)
        assert not spec.is_stale

    def test_the_average_is_recent_weighted(self):
        """A spec that broke last week must not be propped up by a good month."""
        spec = SelectorSpec("a.test", "h")
        for _ in range(30):
            spec.record_use(1.0)
        for _ in range(6):
            spec.record_use(0.0)
        assert spec.avg_fill_rate < 0.4

    def test_store_hides_a_stale_spec(self):
        store = SpecStore()
        spec = SelectorSpec("a.test", "abc", {"x": SelectorRule("p")}, path_prefix="/p/")
        store.put(spec)
        assert store.get("https://a.test/p/1", "abc") is not None
        for _ in range(6):
            spec.record_use(0.0)
        assert store.get("https://a.test/p/1", "abc") is None

    def test_specs_are_filed_per_path_prefix(self):
        """A site's /product/ pages and its /blog/ pages share no structure."""
        store = SpecStore()
        store.put(SelectorSpec("a.test", "abc", {"x": SelectorRule("p")}, path_prefix="/product/"))
        assert store.get("https://a.test/product/1", "abc") is not None
        assert store.get("https://a.test/blog/1", "abc") is None


class TestDomSkeleton:
    def test_is_much_smaller_than_the_page(self, product_html):
        skeleton = dom_skeleton(product_html.decode())
        assert 0 < len(skeleton) < len(product_html)

    def test_carries_addresses_and_values(self, product_html):
        skeleton = dom_skeleton(product_html.decode())
        assert "h1.product-title" in skeleton
        assert "Blue Widget" in skeleton

    def test_scripts_and_styles_are_dropped(self, product_html):
        assert "application/ld+json" not in dom_skeleton(product_html.decode())

    def test_is_bounded(self, product_html):
        assert len(dom_skeleton(product_html.decode() * 50, max_chars=2000)) <= 2400


# --------------------------------------------------------------------------- #
# The cascade
# --------------------------------------------------------------------------- #


class TestCascadeOrdering:
    def test_tier1_answers_without_reaching_a_model(self, product_html, exploding_backend):
        cascade = ExtractionCascade(backend=exploding_backend)
        item = cascade.extract(parsed("https://shop.test/p/1", product_html),
                               "Extract the product.", PRODUCT_SCHEMA)
        assert item.ok
        assert item.tier == 1
        assert item.method is ExtractionMethod.STRUCTURED_DATA

    def test_thin_structure_falls_through_to_the_model(self, plain_html, fake_backend):
        backend = fake_backend({"title": "Field notes", "price": None, "brand": None, "rating": None})
        cascade = ExtractionCascade(backend=backend)
        item = cascade.extract(parsed("https://a.test/notes", plain_html),
                               "Extract the product.", PRODUCT_SCHEMA, allowed_tiers={1, 3})
        assert item.tier == 3
        assert item.method is ExtractionMethod.LLM
        assert len(backend.calls) == 1

    def test_failed_learning_is_abandoned_rather_than_retried_forever(
        self, plain_html, fake_backend
    ):
        """A failed learn costs a wasted call; repeating it on every page doubles the crawl."""
        from pipeline.extract.selectors import MAX_LEARN_ATTEMPTS

        backend = fake_backend({})  # returns no usable selectors, so learning fails
        cascade = ExtractionCascade(backend=backend)
        for index in range(6):
            cascade.extract(
                parsed(f"https://a.test/notes/{index}", plain_html),
                "Extract the product.",
                PRODUCT_SCHEMA,
                allowed_tiers={2, 3},
            )
        learn_calls = sum(1 for call in backend.calls if "selector map" in call["prompt"])
        assert learn_calls == MAX_LEARN_ATTEMPTS

    def test_cache_answers_the_second_time(self, product_html, exploding_backend, monkeypatch):
        monkeypatch.setattr(config, "EXTRACTION_CACHE_ENABLED", True, raising=False)
        cascade = ExtractionCascade(backend=exploding_backend)
        first = cascade.extract(parsed("https://shop.test/p/1", product_html), "p", PRODUCT_SCHEMA)
        second = cascade.extract(parsed("https://shop.test/p/1", product_html), "p", PRODUCT_SCHEMA)
        assert first.tier == 1
        assert second.tier == 0
        assert second.method is ExtractionMethod.CACHE
        assert second.extracted_data == first.extracted_data

    def test_a_changed_page_is_not_served_from_cache(self, product_html, monkeypatch, fake_backend):
        monkeypatch.setattr(config, "EXTRACTION_CACHE_ENABLED", True, raising=False)
        cascade = ExtractionCascade(backend=fake_backend({}))
        cascade.extract(parsed("https://shop.test/p/1", product_html), "p", PRODUCT_SCHEMA)
        changed = product_html.replace(b"19.99", b"24.99")
        second = cascade.extract(parsed("https://shop.test/p/1", changed), "p", PRODUCT_SCHEMA)
        assert second.tier == 1
        assert second.extracted_data["price"] == "24.99"

    def test_a_different_schema_is_a_different_cache_entry(self, product_html, monkeypatch):
        monkeypatch.setattr(config, "EXTRACTION_CACHE_ENABLED", True, raising=False)
        cascade = ExtractionCascade(backend=None)
        cascade.extract(parsed("https://shop.test/p/1", product_html), "p", PRODUCT_SCHEMA)
        other = cascade.extract(
            parsed("https://shop.test/p/1", product_html), "p", {"title": "string"}
        )
        assert other.tier == 1  # recomputed, not served from the first entry

    def test_allowed_tiers_can_force_the_model(self, product_html, fake_backend):
        backend = fake_backend({"title": "forced", "price": None, "brand": None, "rating": None})
        cascade = ExtractionCascade(backend=backend)
        item = cascade.extract(
            parsed("https://shop.test/p/1", product_html), "p", PRODUCT_SCHEMA,
            allowed_tiers={3},
        )
        assert item.tier == 3
        assert len(backend.calls) == 1

    def test_disabling_every_tier_fails_rather_than_guessing(self, product_html):
        cascade = ExtractionCascade(backend=None)
        item = cascade.extract(
            parsed("https://shop.test/p/1", product_html), "p", PRODUCT_SCHEMA, allowed_tiers=set()
        )
        assert not item.ok

    def test_a_stored_spec_is_replayed_without_a_model(self, product_html, exploding_backend):
        store = SpecStore()
        cascade = ExtractionCascade(backend=exploding_backend, specs=store)
        hint = {"title": "string"}
        store.put(
            SelectorSpec(
                domain="shop.test",
                schema_hash=schema_module.schema_hash(hint),
                rules={"title": SelectorRule("h1.product-title")},
                path_prefix="/p/",
            )
        )
        item = cascade.extract(parsed("https://shop.test/p/1", product_html), "p", hint,
                               allowed_tiers={2, 3})
        assert item.tier == 2
        assert item.extracted_data["title"] == "Blue Widget"

    def test_content_with_no_text_fails_cleanly(self, fake_backend):
        cascade = ExtractionCascade(backend=fake_backend({}))
        item = ExtractionItem(url="https://a.test/x", kind=ResourceKind.TEXT)
        result = cascade.extract(item, "p", {"a": "string"})
        assert not result.ok

    def test_an_empty_schema_is_rejected(self, product_html):
        cascade = ExtractionCascade(backend=None)
        assert not cascade.extract(parsed("https://a.test/x", product_html), "p", {}).ok
