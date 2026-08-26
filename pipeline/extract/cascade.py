"""The extraction cascade: try the free options before the expensive one.

This replaces "every page costs one model call". The original pipeline had a
single extractor, so a re-crawl of 1,000 unchanged pages cost 1,000 Ollama
calls at up to 300 seconds each — a throughput ceiling that no amount of
concurrency moves, because the bottleneck is the model, not the queue.

    tier 0  content-hash cache        ~1 ms     unchanged pages
    tier 1  embedded structured data  ~5 ms     anything with schema.org markup
    tier 2  learned selector spec     ~10 ms    any site visited more than once
    tier 3  the model                 1-300 s   genuinely unstructured content

Each tier's result is accepted only if it fills at least ``MIN_FILL_RATE`` of
the requested fields; otherwise the cascade falls through. That threshold is
what stops tier 1 from "succeeding" with a title and seven nulls on a page
whose JSON-LD only describes the site's logo.

Which tier fired is recorded on every record. It is the number that tells you
whether the cascade is working, and the first thing to look at on the day a
site changes its markup and everything falls through to tier 3 again.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from config import config
from errors import ExtractError, PipelineError, TransientExtractError
from models import ExtractionItem, ExtractionMethod, ResourceKind, Stage
from observability import get_logger, metrics

from . import schema as schema_module
from .cache import ExtractionCache, cache_key
from .selectors import SelectorSpec, SpecStore, apply_spec, learn_spec
from .structured import absolutize_images, map_to_schema

log = get_logger("extract.cascade")


def _is_scalarish(value: Any) -> bool:
    """Values simple enough to offer as a schema candidate."""
    if isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, list) and all(
        isinstance(entry, (str, int, float, bool)) for entry in value
    )


class ExtractionCascade:
    """Runs the tiers in order and records which one answered."""

    def __init__(
        self,
        *,
        database=None,
        cache: Optional[ExtractionCache] = None,
        specs: Optional[SpecStore] = None,
        backend=None,
    ) -> None:
        self._db = database
        self.cache = cache if cache is not None else ExtractionCache(database)
        self.specs = specs if specs is not None else SpecStore(database)
        self._backend = backend

    # ------------------------------------------------------------------ #

    def extract(
        self,
        item: ExtractionItem,
        prompt: str,
        schema_hint: dict,
        *,
        local_only: bool = False,
        allowed_tiers: Optional[set[int]] = None,
    ) -> ExtractionItem:
        started = time.perf_counter()
        try:
            return self._run(item, prompt, schema_hint, local_only, allowed_tiers)
        except PipelineError as exc:
            return item.fail_from(exc)
        except Exception as exc:
            return item.fail(
                Stage.EXTRACT, f"extraction failed: {exc}", error_type=type(exc).__name__
            )
        finally:
            item.record_timing("extract", time.perf_counter() - started)

    def _run(
        self,
        item: ExtractionItem,
        prompt: str,
        schema_hint: dict,
        local_only: bool,
        allowed_tiers: Optional[set[int]],
    ) -> ExtractionItem:
        if not schema_hint:
            return item.fail(Stage.EXTRACT, "no schema was supplied")

        allowed = allowed_tiers if allowed_tiers is not None else {0, 1, 2, 3}
        schema_hash = schema_module.schema_hash(schema_hint)
        prompt_hash = config.prompt_hash(prompt)
        json_schema = schema_module.compile_schema(schema_hint)
        content_hash = item.content_hash or item.compute_content_hash() or ""
        base_url = item.final_url or item.url

        item.metadata["schema_hash"] = schema_hash
        item.metadata["prompt_hash"] = prompt_hash

        # --- tier 0: unchanged content ---------------------------------- #
        key = cache_key(content_hash, schema_hash, prompt_hash) if content_hash else ""
        if key and 0 in allowed and config.ENABLE_TIER0_CACHE:
            cached = self.cache.get(key)
            if cached is not None and cached.get("data"):
                log.info("cascade.hit", tier=0, method="cache", url=item.url)
                return self._accept(
                    item, cached["data"], ExtractionMethod.CACHE, 0,
                    confidence=float(cached.get("confidence", 0.9)),
                    note=f"served from cache (originally {cached.get('method')})",
                )

        # --- tier 1: the publisher's own structured data ----------------- #
        if 1 in allowed and config.ENABLE_TIER1_STRUCTURED and item.structured:
            data, fill = self._tier1(item, schema_hint, base_url)
            if fill >= config.MIN_FILL_RATE:
                log.info("cascade.hit", tier=1, method="structured", url=item.url,
                         fill_rate=round(fill, 2))
                self._store(key, data, ExtractionMethod.STRUCTURED_DATA, 1, fill, item, schema_hash)
                return self._accept(item, data, ExtractionMethod.STRUCTURED_DATA, 1, confidence=fill)
            if data:
                log.debug("cascade.tier1_thin", url=item.url, fill_rate=round(fill, 2))
                item.metadata["tier1_fill_rate"] = round(fill, 2)

        # --- tier 2: a learned selector spec ----------------------------- #
        if (
            2 in allowed
            and config.ENABLE_TIER2_SELECTORS
            and item.kind is ResourceKind.HTML
            and item.decoded_text
        ):
            result = self._tier2(item, prompt, schema_hint, schema_hash, local_only)
            if result is not None:
                data, fill, learned = result
                log.info("cascade.hit", tier=2, method="selector", url=item.url,
                         fill_rate=round(fill, 2), learned_now=learned)
                self._store(key, data, ExtractionMethod.SELECTOR_SPEC, 2, fill, item, schema_hash)
                return self._accept(
                    item, data, ExtractionMethod.SELECTOR_SPEC, 2, confidence=fill,
                    note="selector spec authored on this visit" if learned else None,
                )

        # --- tier 3: the model ------------------------------------------- #
        if 3 not in allowed or not config.ENABLE_TIER3_LLM:
            return item.fail(
                Stage.EXTRACT,
                "no tier could extract this content and the model tier is disabled",
            )

        return self._tier3(item, prompt, schema_hint, json_schema, key, schema_hash, local_only)

    # ------------------------------------------------------------------ #
    # Tiers
    # ------------------------------------------------------------------ #

    #: Keys the pipeline writes onto ``metadata`` for its own bookkeeping. They
    #: must never be offered to the schema mapper as if they were page content.
    _INTERNAL_METADATA = frozenset(
        {
            "detected_subtype", "detection_source", "detection_confidence", "acquisition",
            "schema_hash", "prompt_hash", "simhash", "mongo_id", "raw_id", "selector_spec",
            "extraction_note", "tier1_fill_rate", "duplicate_of", "duplicate_kind",
            "llm_backend", "llm_model", "llm_seconds", "llm_tokens", "llm_schema_enforced",
            "discovered_urls", "discovered_count", "truncated", "ocr", "ocr_engine",
            "transcript_backend", "transcript_model", "converter", "delimiter",
        }
    )

    def _tier1(self, item: ExtractionItem, schema_hint: dict, base_url: str) -> tuple[dict, float]:
        """Read schema.org / OpenGraph / framework state / parsed rows / handler metadata."""
        sources: dict[str, Any] = dict(item.structured or {})

        # Tables and parsed rows are structure too, and they are where the
        # answer lives for a CSV, a sitemap, or a spec table on a product page.
        if item.parsed_tree and item.parsed_tree.get("tables"):
            sources.setdefault("tables", item.parsed_tree["tables"])

        # A handler's own metadata is structured data by definition — a feed's
        # <title>, a PDF's document title, an email's subject were all read from
        # the format rather than inferred. Internal bookkeeping is excluded.
        handler_fields = {
            key: value
            for key, value in (item.metadata or {}).items()
            if key not in self._INTERNAL_METADATA and _is_scalarish(value)
        }
        if handler_fields:
            sources.setdefault("handler_metadata", handler_fields)

        data, fill = map_to_schema(sources, schema_hint)
        if data:
            data = absolutize_images(data, base_url)
        metrics.incr("extract.tier1_attempt")
        return data, fill

    def _tier2(
        self,
        item: ExtractionItem,
        prompt: str,
        schema_hint: dict,
        schema_hash: str,
        local_only: bool,
    ) -> Optional[tuple[dict, float, bool]]:
        """Replay a stored spec, or author one on the first visit to a domain.

        Learning costs exactly one model call — the same as tier 3 would have
        cost — and produces something reusable. That is the whole trade: pay the
        model once per *domain* instead of once per *page*.
        """
        html = item.decoded_text or ""
        spec = self.specs.get(item.url, schema_hash)

        if spec is not None:
            data, fill = apply_spec(spec, html, schema_hint)
            spec.record_use(fill)
            self.specs.put(spec)
            if fill >= config.MIN_FILL_RATE:
                item.metadata["selector_spec"] = spec.key
                return data, fill, False

            # Drift: the markup moved. Retire the spec so the next page in this
            # job relearns rather than every page returning nulls until someone
            # notices.
            if spec.is_stale:
                self.specs.retire(spec, f"fill rate fell to {spec.avg_fill_rate:.2f}")
                item.warn(f"selector spec for {spec.domain} drifted and was retired")
            metrics.incr("extract.tier2_miss")
            return None

        # Learning is only free when it works: the model call it costs is the
        # one tier 3 was going to make anyway. When it fails, this page paid
        # twice — so a site where learning cannot succeed stops being asked.
        if not self.specs.should_learn(item.url, schema_hash):
            return None

        try:
            from .llm import SELECTOR

            backend = self._get_backend(local_only, role=SELECTOR)
        except ExtractError as exc:
            log.debug("cascade.tier2_no_backend", error=str(exc))
            return None

        learned = learn_spec(item.url, html, schema_hint, schema_hash, backend)
        if learned is None:
            self.specs.record_learn_failure(item.url, schema_hash)
            metrics.incr("extract.tier2_learn_failed")
            return None

        self.specs.put(learned)
        data, fill = apply_spec(learned, html, schema_hint)
        if fill < config.MIN_FILL_RATE:
            self.specs.record_learn_failure(item.url, schema_hash)
            return None
        item.metadata["selector_spec"] = learned.key
        metrics.incr("extract.tier2_learned")
        return data, fill, True

    def _tier3(
        self,
        item: ExtractionItem,
        prompt: str,
        schema_hint: dict,
        json_schema: dict,
        key: str,
        schema_hash: str,
        local_only: bool,
    ) -> ExtractionItem:
        text = item.text_for_extraction
        if not text.strip():
            return item.fail(Stage.EXTRACT, "no text content to extract from")

        truncated = text[: config.MAX_CHUNK_SIZE]
        if len(text) > config.MAX_CHUNK_SIZE:
            item.warn(
                f"content truncated from {len(text)} to {config.MAX_CHUNK_SIZE} characters "
                "for the model"
            )

        backend = self._get_backend(local_only)
        try:
            response = backend.complete_json(
                prompt=prompt,
                content=truncated,
                schema_hint=schema_hint,
                json_schema=json_schema,
            )
        except TransientExtractError:
            raise  # Celery's autoretry decides what happens next
        except ExtractError as exc:
            return item.fail_from(exc)

        for warning in response.warnings:
            item.warn(warning)

        fill = schema_module.fill_rate(response.data, schema_hint)
        item.metadata.update(
            {
                "llm_backend": response.backend,
                "llm_model": response.model,
                "llm_seconds": response.latency_seconds,
                "llm_schema_enforced": response.schema_enforced,
            }
        )
        if response.completion_tokens:
            item.metadata["llm_tokens"] = response.completion_tokens

        log.info(
            "cascade.hit",
            tier=3,
            method="llm",
            url=item.url,
            backend=response.backend,
            seconds=response.latency_seconds,
            fill_rate=round(fill, 2),
        )
        metrics.incr("extract.tier3_llm")
        self._store(key, response.data, ExtractionMethod.LLM, 3, fill, item, schema_hash)
        return self._accept(item, response.data, ExtractionMethod.LLM, 3, confidence=fill)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_backend(self, local_only: bool, role: Optional[str] = None):
        """The backend for one tier's call.

        Tier 2 and tier 3 are different jobs. Authoring selectors happens once
        per domain and wants the best model available; extracting fields
        happens once per document and wants the one with no rate limit.
        """
        if self._backend is not None:
            return self._backend
        from .llm import BULK, get_backend

        return get_backend(local_only=local_only, role=role or BULK)

    def _store(
        self,
        key: str,
        data: dict,
        method: ExtractionMethod,
        tier: int,
        fill: float,
        item: ExtractionItem,
        schema_hash: str,
    ) -> None:
        if not key or not data:
            return
        self.cache.put(
            key,
            data,
            method=method.value,
            tier=tier,
            confidence=fill,
            url=item.url,
            content_hash=item.content_hash or "",
            schema_hash=schema_hash,
        )

    @staticmethod
    def _accept(
        item: ExtractionItem,
        data: dict,
        method: ExtractionMethod,
        tier: int,
        *,
        confidence: float,
        note: Optional[str] = None,
    ) -> ExtractionItem:
        item.extracted_data = data
        item.method = method
        item.tier = tier
        item.confidence = round(min(max(confidence, 0.0), 1.0), 3)
        if note:
            item.metadata["extraction_note"] = note
        metrics.incr(f"extract.tier{tier}")
        return item

    def stats(self) -> dict:
        return {"cache": self.cache.stats(), "specs_cached": len(self.specs._memory)}


#: One cascade per process. It holds the caches, which is exactly what should be
#: shared across the tasks a worker runs.
_default: Optional[ExtractionCascade] = None


def get_cascade(database=None) -> ExtractionCascade:
    global _default
    if _default is None:
        _default = ExtractionCascade(database=database)
    return _default


def reset() -> None:
    global _default
    _default = None


__all__ = ["ExtractionCascade", "SelectorSpec", "get_cascade", "reset"]
