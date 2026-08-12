# Roadmap — universal extraction pipeline

**Goal:** point it at anything on the web — a page, a PDF, a spreadsheet, a
podcast, a live stream — and get structured JSON out, on a Mac, behind
Redis + Celery + FastAPI, stored in MongoDB.

**Constraints taken as fixed:** macOS on Apple Silicon (confirmed arm64),
Redis, Celery, FastAPI, MongoDB. scrapekit stays a separate reference project;
everything below lands in this repo.

---

## 1. Target architecture

```
                    ┌─────────────┐
   POST /extract ──►│  FastAPI    │──► enqueue ──► Redis
                    └─────────────┘                 │
                                        ┌───────────┴───────────┐
                                        ▼                       ▼
                                 ┌────────────┐          ┌────────────┐
                                 │  io queue  │          │ cpu queue  │
                                 │ threads×16 │          │ prefork×2  │
                                 └─────┬──────┘          └─────┬──────┘
                                       │                       │
                    fetch / HEAD / download          transcribe / OCR / LLM
                                       │                       │
                                       └───────────┬───────────┘
                                                   ▼
                                          ┌─────────────────┐
                                          │  TypeRouter     │
                                          └────────┬────────┘
                    ┌──────────┬──────────┬────────┼────────┬──────────┐
                    ▼          ▼          ▼        ▼        ▼          ▼
                  HTML      Document    Image    Media    Feed      Archive
                    │          │          │        │        │          │
                    └──────────┴──────────┴────┬───┴────────┴──────────┘
                                               ▼
                                   ┌───────────────────────┐
                                   │  Extraction cascade   │
                                   │  1 structured data    │
                                   │  2 saved selector spec│
                                   │  3 LLM (last resort)  │
                                   └───────────┬───────────┘
                                               ▼
                              validate → dedupe → provenance → MongoDB
```

The two structural changes from today: **the queue splits in two**, and **the
LLM stops being the only extractor**.

---

## 2. The single biggest speed win: stop calling the LLM

Right now every page costs one Ollama call, up to 300 seconds. That is the
throughput ceiling, and no amount of concurrency fixes it. Replace the single
extractor with a cascade that tries free options first:

| Tier | Method | Cost | Hits on |
|---|---|---|---|
| 0 | **Content-hash cache** — page unchanged since last run | ~0 ms | re-crawls |
| 1 | **Embedded structured data** — JSON-LD, microdata, OpenGraph, `__NEXT_DATA__`, `<table>`, RSS | ~5 ms | a large share of commerce, article, recipe, event and job pages, which publish schema.org markup for SEO |
| 2 | **Saved selector spec** — a per-domain spec learned once (optionally *authored by* the LLM on first visit, then reused) | ~10 ms | any site you scrape more than once |
| 3 | **LLM** | 1–300 s | genuinely unstructured pages, transcripts, novel layouts |

Tier 2 is the interesting one: let the LLM look at a new domain **once**, have it
emit CSS/XPath selectors rather than the data itself, store that spec in Mongo,
and replay it for free on every subsequent page from that domain. Add a drift
check so the spec is regenerated when its fill rate collapses.

This is what turns the project from "an LLM wrapper" into a scraper.

---

## 3. "Scrape anything" — the type router

Detect by **magic bytes and Content-Type, never by file extension.**

| Input | Detection | Handler | Library |
|---|---|---|---|
| Static HTML | `text/html` | fetch → parse | httpx + selectolax *(have)* |
| JS-rendered HTML | empty body / block signature | headless render | Playwright *(have)* |
| PDF (text layer) | `%PDF` | direct text + tables | PyMuPDF |
| PDF (scanned) | PDF with no text layer | OCR | Vision.framework or Tesseract |
| docx / pptx / xlsx | ZIP + OOXML parts | native parse | MarkItDown |
| Legacy doc / xls / ppt | OLE2 magic | convert then parse | LibreOffice `--headless` |
| ePub / MOBI | ZIP + mimetype | chapter text | ebooklib |
| CSV / TSV | delimiter sniff | rows | pandas |
| JSON / JSONL / XML / YAML | first byte + parse probe | direct | stdlib / lxml |
| Images | `\x89PNG`, `\xff\xd8`, HEIC | OCR + caption | Vision.framework |
| Audio / video file | ftyp / ID3 / RIFF | transcribe | mlx-whisper |
| Media platform URL | domain match | download → transcribe | yt-dlp *(have)* |
| **Live stream** | `.m3u8`, DASH, `is_live` | rolling segment capture → chunked transcribe | yt-dlp + ffmpeg |
| RSS / Atom | root tag | entries | feedparser |
| Sitemap | `<urlset>` / `<sitemapindex>` | URL inventory | lxml |
| JSON API | `application/json` | direct | httpx |
| Archives (zip/tar/7z) | magic | recurse into members | stdlib + py7zr |
| Email (.eml/.msg) | RFC822 headers | body + attachments (recurse) | mailparser |

**Don't hand-write eight document handlers.** Microsoft's **MarkItDown** covers
PDF, Office, ebooks, images and audio in one API and emits LLM-friendly Markdown
with minimal preprocessing overhead. Make it the default document handler; add
**Docling** as an opt-in for table-heavy or multi-column PDFs, where its
AI-based layout detection is worth the extra latency. That collapses a whole
phase into one dependency.

**Live streams** are the genuinely new capability and deserve their own design:
capture in rolling N-minute segments, transcribe each segment as it lands, and
append to the record — never wait for the stream to end. A long-running Celery
task with `soft_time_limit` and periodic `update_state` fits this well.

---

## 4. Performance plan (Apple Silicon specifics)

### 4.1 Split the Celery queues — biggest structural win

Today `-c 1` means one 20-minute transcription blocks 200 quick page fetches.

```python
# celery_app.py
task_routes = {
    "tasks.fetch_*":      {"queue": "io"},
    "tasks.transcribe_*": {"queue": "cpu"},
    "tasks.extract_*":    {"queue": "cpu"},
}
```

```bash
# I/O-bound: network waits, high concurrency, threads are fine
celery -A celery_app worker -Q io  --pool=threads -c 16 -n io@%h

# CPU/GPU-bound: Whisper and the LLM, keep it narrow
celery -A celery_app worker -Q cpu --pool=prefork -c 2  -n cpu@%h
```

### 4.2 Switch Whisper to MLX

`faster-whisper` runs on CTranslate2, which has **no Metal backend on macOS** —
so on your M-series machine it is using CPU cores only and leaving the GPU and
Neural Engine idle. MLX-based Whisper implementations use them. Published 2026
Apple Silicon benchmarks report large speedups for MLX and MLX-derived runtimes,
though the numbers vary a lot by implementation and model size, so treat them as
directional and **benchmark both on your own machine and audio** before
committing. Keep `faster-whisper` behind the same interface as a fallback.

### 4.3 Load models once per worker, not once per task

`MediaProcessor` currently constructs `WhisperModel` inside every task — several
seconds of load time per job. Hoist it into a process-global initialised on
`worker_process_init`, and let `worker_max_tasks_per_child` bound the leak.

### 4.4 Cache at two levels

- **HTTP:** ETag / Last-Modified revalidation. A 304 costs almost nothing.
- **Extraction:** key on `sha256(content) + schema_hash + prompt_hash`. An
  unchanged page skips the LLM entirely. This is where re-crawls get cheap.

### 4.5 MongoDB

```javascript
db.extractions.createIndex({url: 1, content_hash: 1, schema_hash: 1}, {unique: true})
db.extractions.createIndex({created_at: -1})
db.raw_store.createIndex({expires_at: 1}, {expireAfterSeconds: 0})  // TTL
```

Unique index gives idempotency for free — resubmitting a URL upserts instead of
duplicating. Store raw bytes in GridFS so you can re-parse without re-fetching.

### 4.6 macOS gotchas

- **Celery prefork on macOS forks after Objective-C runtime init and can hang.**
  Set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` in the worker's environment. This
  one costs people days.
- Raise the descriptor limit before running 16 concurrent fetchers:
  `ulimit -n 4096`.
- `brew install ffmpeg` is required by yt-dlp; the README now says so.
- Ollama already uses Metal — no configuration needed.

---

## 5. LLM backend: use both, deliberately

| | Local Ollama | Groq |
|---|---|---|
| Latency | seconds to minutes | sub-second typical |
| Cost | free | per-token, cheap at the small-model end |
| Privacy | page content never leaves the Mac | page content leaves the Mac |
| Schema guarantee | `format: "json"` — valid JSON, **not necessarily your shape** | `response_format: {"type": "json_schema"}` — conforms to the schema, no retry logic needed |

Make the backend pluggable behind one `LLMBackend` protocol and choose per job:

- **Default: Ollama.** Private, free, already working.
- **Groq when throughput matters** and the content is not sensitive. Its
  structured-outputs mode is a real correctness upgrade over Ollama's JSON mode —
  it removes a whole class of "the model returned valid JSON with the wrong
  fields" bugs that Try2 currently cannot even detect.
- You already have a `GROQ_API_KEY` in the other project's `.env`. Copy the key,
  **not the file**, and keep `.env` gitignored as that project correctly does.

Add schema validation regardless of backend, so a wrong-shaped response is a
caught error rather than a corrupt Mongo document.

---

## 6. Phases

Each phase is independently shippable. Stop after any of them and the pipeline
still works.

### Phase 1 — Safety (do this before scraping anything you don't own)
- robots.txt gate, re-checked on every redirect hop
- per-host token-bucket rate limiter honouring `Crawl-delay`
- anti-bot block detection (a Cloudflare interstitial returns **HTTP 200** and is
  currently fed to Ollama as if it were content)
- manual redirect following so each hop is re-checked
- circuit breaker per host

*Reference: `scrapekit/compliance/`, `scrapekit/fetch/{ratelimit,blocks,breaker}.py`*

### Phase 2 — Universal input
- magic-byte type router
- MarkItDown document handler; Docling opt-in
- image OCR
- archive and email recursion
- live-stream segment capture
- extend `URLRouter` into the full `TypeRouter`

### Phase 3 — Extraction cascade
- embedded structured-data parser (JSON-LD / microdata / `__NEXT_DATA__` / tables)
- per-domain selector specs stored in Mongo, LLM-authored on first visit
- content-hash extraction cache
- pluggable Ollama/Groq backend with schema validation

### Phase 4 — Performance
- split io/cpu queues
- MLX Whisper, benchmarked against faster-whisper on your own audio
- per-worker model preloading
- ETag cache, Mongo indexes, GridFS raw store

### Phase 5 — Trust
- output validation rules
- exact + near-duplicate detection
- provenance on every record (content hash, extraction method, confidence, which
  tier fired)
- drift alerts when a field's fill rate drops
- `RunReport` per job

### Phase 6 — Operations
- Celery `autoretry_for` + `retry_backoff`, dead-letter collection
- Flower for queue visibility
- structured JSON logging
- **a test suite** — currently zero; scrapekit's 384 offline tests using `respx`
  and injected clocks are a good model to copy
- Docker Compose for Redis + Mongo so setup is one command

---

## 7. Dependencies to add

```bash
# Phase 2 — universal input
pip install markitdown[all] pymupdf puremagic feedparser py7zr mail-parser
pip install docling            # optional, table-heavy PDFs

# Phase 3 — extraction
pip install extruct groq jsonschema

# Phase 4 — performance
pip install mlx-whisper        # Apple Silicon only
pip install flower
```

`brew install ffmpeg libreoffice tesseract`

---

## 8. What I'd do first

If you only do one thing: **Phase 1**, because it is the only phase with legal
and IP-ban consequences, and it is roughly a day's work.

If you want the biggest visible improvement: **Phase 3's tier-1 structured-data
parser**. A page with JSON-LD goes from a 30-second Ollama call to a 5-millisecond
dictionary lookup, and the answer is more accurate because it is the publisher's
own data rather than a model's reading of rendered text.
