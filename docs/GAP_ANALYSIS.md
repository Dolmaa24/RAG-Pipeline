# Gap analysis: PipelineTry2 vs. scrapekit

Comparison of this project (`WebScraping_PipelineTry2`) against
`CoRover_Project1/WebScraping_Pipeline` (the `scrapekit` package), written to
decide what Try2 still needs.

## They are not the same kind of thing

| | **PipelineTry2** | **scrapekit** |
|---|---|---|
| Shape | Application / service | Installable library + CLI |
| Extraction strategy | LLM reads the text (Ollama) | Declarative selectors + JSON-LD |
| Per-site setup | None — prompt + schema hint | Write an `ItemSpec` per site |
| Source LOC | 1,278 across 18 files | 11,747 across 48 files |
| Tests | 0 | 384, offline, ~81% coverage |
| Packaging | None | `pyproject.toml`, hatchling, `scrapekit` CLI |
| Tooling | None | ruff + mypy configured |
| Version control | None | git |
| Docs | README | README + 3 docs |

Try2 is a **zero-config, any-site, async service**. scrapekit is a
**deterministic, compliance-first extraction library**. Neither replaces the
other; Try2's weaknesses are almost exactly scrapekit's strengths.

## Stage coverage

```
scrapekit:  DISCOVER → FETCH → DECODE → PARSE → EXTRACT → NORMALIZE → PERSIST
                                                    ↑
                                            VALIDATE / MONITOR

Try2:       (none)   → FETCH → DECODE → PARSE → EXTRACT → NORMALIZE → PERSIST
                                                    ↑
                                              (nothing)
```

Try2 has no DISCOVER stage and no VALIDATE/MONITOR loop. Its PERSIST is a
single unconditional Mongo insert.

---

## What Try2 lacks (scrapekit has it)

### Tier 1 — before running against anything you don't own

| Gap | scrapekit module | Why it matters here |
|---|---|---|
| **robots.txt gate** | `compliance/robots.py` | Try2 has zero compliance checking, sends a spoofed Chrome UA, and follows redirects blindly. This is the legal exposure. scrapekit re-checks robots on *every redirect hop* (RFC 9309). |
| **Per-host rate limiting** | `fetch/ratelimit.py` | Token bucket + `Crawl-delay`. Try2's only pacing is exponential backoff *after* it has already been refused. |
| **Anti-bot block detection** | `fetch/blocks.py` | A Cloudflare interstitial returns **HTTP 200**. Try2 treats it as a successful fetch and pays for a 5-minute Ollama call to extract fields from "Checking your browser". |
| **Manual redirect following** | `fetch/client.py` | Try2 uses `follow_redirects=True`, so a redirect onto a disallowed path is invisible. |

### Tier 2 — output you can trust

| Gap | scrapekit module | Why it matters here |
|---|---|---|
| **Embedded structured data** | `parse/embedded.py` | JSON-LD / microdata / `__NEXT_DATA__`. If a page already publishes structured data, reading it is faster, free, and *more accurate* than asking an LLM to re-derive it from rendered text. This is the single biggest accuracy win available to Try2. |
| **Provenance** | `models.Provenance` | Try2 stores `url`, `type`, `created_at`. scrapekit stores content hash, HTTP status, parser version, extraction method, and confidence on every record. Without it you cannot tell which of two conflicting rows is stale. |
| **Validation** | `quality/validation.py` | Try2 does no output checking at all. With a non-deterministic extractor this matters **more**, not less — llama3 will silently invent fields. |
| **Deduplication** | `quality/dedupe.py` | Exact + simhash near-duplicate. Try2 will insert the same page 50 times across 50 runs. |
| **Drift monitoring** | `quality/drift.py` | Alerts when a field's fill rate collapses. Try2's failure mode is silent empty JSON. |
| **Typed normalization** | `normalize/{dates,numbers}.py` | Try2's normalizer only cleans whitespace. Dates stay as freeform strings; money stays as whatever the LLM emitted. scrapekit parses dates and holds money as `Decimal`, serialised to disk as a string so the value round-trips exactly. |

### Tier 3 — operations and scale

| Gap | scrapekit module | Why it matters here |
|---|---|---|
| **HTTP cache with revalidation** | `fetch/cache.py` | ETag / Last-Modified. Try2 re-fetches and re-LLMs everything on every run. At 5-minute timeouts this dominates runtime and cost. |
| **Circuit breaker** | `fetch/breaker.py` | A failing host gets one probe, not 30,000 retries. |
| **Discovery** | `discover/` | Sitemaps, feeds, frontier, crawl-trap detection. Try2 requires you to hand it every URL by hand. |
| **Raw byte store** | `persist/rawstore.py` | Lets you re-parse without re-fetching — invaluable when you change the prompt or schema. |
| **Sink options** | `persist/` | JSONL / SQLite / MultiSink. Try2 is MongoDB-only, so an Atlas outage means the run is lost. |
| **Checkpointing** | `state.py` | Atomic checkpoints and adaptive re-fetch intervals. Try2 cannot resume an interrupted batch. |
| **Structured logging + metrics** | `observability.py` | Try2 uses bare `logging` with f-strings. Nothing is aggregatable. |
| **Error taxonomy** | `errors.py` | Exceptions grouped by stage with a `transient` flag that drives retry decisions. Try2 raises `Exception(str)` and pattern-matches nothing. |
| **URL canonicalisation** | `urls.py` | Try2 has no canonicalisation, so `?utm_source=` variants are distinct records. |
| **Document extraction** | `extract/documents.py` | PDF, docx, xlsx, pptx, CSV via magic-byte detection. Try2 handles HTML and A/V only. |
| **Tests** | `tests/` | 384 offline tests vs. 0. |

---

## What scrapekit lacks (Try2 has it)

| Try2 capability | scrapekit status |
|---|---|
| **LLM extraction** | **Absent entirely.** scrapekit needs a hand-written `ItemSpec` per site. Try2 works zero-config on a site it has never seen. This is Try2's whole reason to exist. |
| **JavaScript rendering** | **Declared but not implemented.** `pyproject.toml` ships a `browser` extra pinning Playwright, but there is no `sync_playwright` call anywhere in `src/` — the extra is a stub. Try2's Playwright fallback actually works. |
| **Async job queue** | Absent. scrapekit is a `ThreadPoolExecutor` in one process. Try2's Celery + Redis survives long jobs and lets a client poll. |
| **HTTP API** | Absent. Try2 has FastAPI with task status polling. |
| **Web UI** | Absent. Try2 has the Streamlit dashboard. |
| **Audio/video transcription** | Weaker. scrapekit uses `youtube-transcript-api` — YouTube only, and only when captions already exist. Try2 does yt-dlp + local Whisper, so it works on any yt-dlp-supported platform and on media with no captions at all. |
| **Shared cloud datastore** | Absent — local files only. Try2 writes to MongoDB Atlas. |

### Honest caveats about scrapekit

- One git commit, yet `version = "1.0.0"` and `Development Status :: 5 - Production/Stable`. That classifier is aspirational.
- The unimplemented `browser` extra is a trap: `pip install "scrapekit[browser]"` installs Playwright and changes nothing.
- Threads rather than asyncio — a documented, defensible trade-off, but a ceiling.
- Selector-based extraction means ongoing per-site maintenance. Selector drift is real recurring work; the LLM path has no such cost.

---

## Recommendation

**Do not reimplement scrapekit inside Try2.** It is already a proper installable
package. Depend on it:

```bash
./venv/bin/pip install -e ../../Claude/CoRover_Project1/WebScraping_Pipeline
```

Then Try2 becomes the layer scrapekit does not have — LLM extraction, Whisper,
Celery, FastAPI, Streamlit, MongoDB — on top of a fetch/parse stack that is
compliance-checked and has 384 tests behind it. Concretely:

1. Replace `pipeline/fetcher.py` + `decoder.py` + `parser.py` with
   `scrapekit.fetch.Fetcher` and `scrapekit.parse`. Keep Try2's Playwright
   fallback, since scrapekit has none.
2. Insert `parse/embedded.py` **before** the LLM call. Use JSON-LD when the page
   provides it; fall through to Ollama only when it does not.
3. Run `quality/validation.py` and `quality/dedupe.py` on `normalized_data`
   before the Mongo insert.
4. Attach `Provenance` to every stored document.
5. Keep `MediaProcessor`, `AIExtractor`, `tasks.py`, `app.py`, `dashboard.py`
   as they are — that is the part scrapekit cannot do.

If depending on scrapekit is not acceptable, port Tier 1 first. Everything else
is an optimisation; robots.txt and rate limiting are not.
