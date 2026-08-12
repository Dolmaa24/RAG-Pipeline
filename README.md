# Universal Extraction Pipeline

Point it at anything on the web — a page, a PDF, a spreadsheet, an RSS feed, a
ZIP of documents, an email, a podcast, a live stream — and get structured JSON
out. Behind Redis + Celery + FastAPI, stored in MongoDB, running on a Mac.

Two ideas do most of the work:

**Detection is by magic bytes, never by file extension.** `report.pdf` is
routinely an HTML login page; `/download?id=8412` is routinely a 40 MB PDF.
`.docx`, `.xlsx`, `.pptx` and `.epub` are all ZIP files, and telling them apart
means reading the member list.

**A language model is the last resort, not the only tool.** Most pages already
publish their own structured description for search engines, and any site you
scrape twice can be described once by a reusable selector spec. Reaching for a
model on every page is what puts a hard ceiling on throughput.

---

## The extraction cascade

Each tier is tried in order; a tier's answer is accepted only if it fills at
least `MIN_FILL_RATE` of the requested fields, otherwise the next one runs.

| Tier | Method | Typical cost | Hits on |
|---|---|---|---|
| 0 | **Content-hash cache** | ~1 ms | re-crawls of unchanged pages |
| 1 | **Embedded structured data** — JSON-LD, microdata, OpenGraph, `__NEXT_DATA__`, tables, parsed rows | ~5 ms | commerce, article, recipe, event and job pages, feeds, CSVs, JSON APIs |
| 2 | **Learned selector spec** — per domain, authored once by a model, replayed free | ~10 ms | any site visited more than once |
| 3 | **The model** — Ollama locally or Groq hosted | 0.3–300 s | genuinely unstructured pages, transcripts, novel layouts |

Tier 2 is what turns this from an LLM wrapper into a scraper. On the first
visit to a domain the model is shown a *pruned DOM skeleton* and asked for
**CSS selectors rather than data**. That costs exactly one model call — the one
tier 3 would have made anyway — and the answer is reusable, inspectable and
versioned. Measured here on `books.toscrape.com`:

```
learn   0.68s   one model call, per domain
replay  ~10ms   every subsequent page, no model at all
```

Each learned rule is verified against the page it came from and **dropped if it
extracts nothing**, so a hallucinated selector never reaches the store. Specs
carry a fill-rate average; when a site changes its markup the rate collapses,
the spec is retired, and the next page relearns it. If learning fails twice for
a domain it is abandoned rather than retried on every page forever.

Which tier answered is recorded on every record. `GET /api/v1/stats` reports the
share of records that never reached a model — the number that says whether any
of this is working.

---

## What it can read

Detected from magic bytes and `Content-Type`; the URL is a tiebreak at most.

| Input | Handler |
|---|---|
| Static HTML | httpx + selectolax, with structured data harvested before the noise is stripped |
| JS-rendered HTML | Playwright, as a fallback when the body comes back empty or blocked |
| PDF with a text layer | PyMuPDF — text, per-page tables, metadata |
| Scanned PDF | rendered at 200 dpi, then OCR |
| docx / pptx / xlsx / epub / odt | MarkItDown (Docling opt-in for table-heavy PDFs) |
| Legacy doc / xls / ppt | LibreOffice `--headless` conversion, then the above |
| Images | Apple Vision OCR, Tesseract fallback |
| CSV / TSV | delimiter sniffing, header inference, rows as records |
| JSON / JSONL / XML / YAML | parsed directly; XXE disabled |
| RSS / Atom | entries, plus their URLs as discoveries |
| Sitemap / sitemap index | URL inventory |
| Audio / video files | Whisper (MLX or faster-whisper) |
| Media platform URLs | yt-dlp → audio → Whisper |
| **Live streams** | ffmpeg rolling segments, transcribed as they land |
| Archives (zip/tar/gz/bz2/xz/7z) | members recursed into as child items |
| Email (.eml/.msg) | headers, body, and attachments recursed into |

**Containers behave differently depending on what they hold.** An archive member
or an email attachment arrives *with its bytes*, so it is processed inline. A
feed entry or a sitemap URL is only a link — following those inline would make
one task perform hundreds of serial fetches and defeat the queue split, so they
are reported as `discovered_urls` and enqueued separately with `fan_out`.

---

## Crawling a site

Give it a starting URL and it follows links. One distinction decides how a
crawl behaves — every link is **followed**, **collected**, or **skipped**:

```bash
curl -X POST localhost:8000/api/v1/crawl -H 'Content-Type: application/json' -d '{
  "start_url": "https://example.com/investors",
  "prompt": "Extract the report title and the total revenue figure.",
  "schema_template": {"title": "string", "revenue": "string"},
  "collect_extensions": ["pdf"],
  "max_depth": 3,
  "max_pages": 500
}'
```

Name `collect_extensions` and the crawl is a **file hunt**: HTML pages are
walked for their links but never extracted — they are the map, not the
destination — and only matching files go through the full pipeline. Leave it
empty and every in-scope page is extracted instead, which is the "run this
schema over every product page" shape (and the shape where tier 2's learned
selector specs pay off most).

Poll it, and the files it found *are* the answer:

```bash
curl -s localhost:8000/api/v1/crawls/{crawl_id} | python3 -m json.tool
```

```json
{"status": "finished", "claimed": 6, "fetched": 3, "collected": 3, "failed": 0,
 "elapsed_seconds": 0.3,
 "targets": ["https://example.com/docs/q1.pdf", "https://example.com/docs/q2.pdf", "..."]}
```

`DELETE /api/v1/crawls/{id}` stops one. It sets a flag the workers check before
each page rather than revoking queued tasks, because Celery does not revoke
reliably once a task has been prefetched — pages in flight finish, nothing new
is claimed.

### What stops it running away

A crawl is the one feature here that can turn into an outage, so termination is
enforced four separate ways:

- **The frontier claims each URL exactly once.** `SADD` is atomic and reports
  novelty, so two workers discovering the same link cannot both enqueue it.
  Without this, a site with a shared navigation bar re-enqueues every page from
  every page.
- **A hard budget** (`max_pages`). Every claim is counted and refused past it.
- **A depth limit** on following. Targets are still collected at any depth —
  the PDF four hops down is the thing you asked for.
- **Trap detection.** Repeated path segments (`/a/b/a/b/a/b`), calendar URLs,
  session and sort parameters, absurd path depth. A calendar widget generates a
  URL per day forever, and it looks exactly like an ordinary link.

Scope is `same_site` by default; turning it off requires an explicit
`allowed_hosts`, because an unbounded crawl of the open web should not be
startable by accident. `rel="nofollow"` and page-level `<meta name="robots"
content="nofollow">` are honoured.

Discovery is fanned out, not recursive: each page is its own task, so the crawl
spreads across the worker pool and no single task has to finish a site inside a
time limit. Pages run on the **io** queue; files the crawl finds are handed to
`extract_url` on the **cpu** queue, since parsing a PDF or running OCR is
exactly the work an io thread must not be holding.

A page whose *extraction* fails is still harvested for links. Its links are
just as good, and treating an extraction failure as a dead branch lets one
rate-limited model call on the seed page silently end the whole crawl.

---

## Safety

The pipeline asks permission before it fetches, and the gates are checked on
**every redirect hop**, not just the submitted URL.

- **robots.txt**, per RFC 9309 §2.3.1 — including the parts most crawlers skip:
  a 401/403 on `robots.txt` itself means the whole site is disallowed, and
  `Request-rate: 1/10s` is understood despite `urllib.robotparser` silently
  dropping it. Cached per origin, fetched at most once per TTL.
- **Per-host rate limiting** — token bucket plus a concurrency cap, honouring
  `Crawl-delay` and `Retry-After`, and slowing down *before* a 429 by reading
  the server's own `RateLimit-*` headers.
- **Anti-bot detection** — a Cloudflare interstitial returns **HTTP 200**.
  Without this check the pipeline pays for a 300-second model call to extract
  product fields from "Checking your browser". A detected wall raises, trips the
  breaker permanently, and prints what to do instead. It is never retried or
  evaded: that is a stated refusal, and working around it is a different
  activity with different rules.
- **Circuit breaker** per host, so a failing site gets one probe rather than
  30,000 retries.
- **SSRF guard** — a submitted URL cannot make a worker fetch `127.0.0.1` or
  `169.254.169.254`.
- **Size caps** enforced *while streaming*, so a 10 GB response is abandoned
  after 64 MiB rather than after it fills memory.
- **Archive bombs** — a running total of decompressed bytes, a per-member ratio
  check, and path-traversal member names rejected.

The default User-Agent is honest and contactable (`BOT_NAME` + `CONTACT_URL`),
which is what turns "some bot is hammering us" into an email rather than a
firewall rule. `SPOOF_BROWSER_UA=true` restores a Chrome UA if you need it.

---

## Trust

Every record carries **provenance**: content hash, HTTP status, redirect chain,
which tier fired, confidence, the backend and model if one was used, and the
schema and prompt hashes. Without it you cannot tell which of two conflicting
rows is stale, or whether a price came from the publisher's own markup or from
a model's reading of rendered text.

- **Schema validation** on every backend. Ollama's `format: "json"` guarantees
  valid JSON and says nothing about its shape, so the shape is checked here and
  a mismatch is retried once with the specific problems fed back. Groq's
  `json_schema` mode is a real guarantee — and is still verified, because a
  pipeline that trusts an external promise without checking has no way to notice
  the day it changes.
- **Output validation** catches the answers that look like data and are not:
  `"N/A"`, the schema's own type words echoed back, model refusals
  (`"I'm sorry, I cannot..."`), lorem ipsum, and a whole page dumped into one
  field.
- **Deduplication**, exact by content hash and near by banded simhash, so the
  same article on three syndication partners is one record.
- **Drift alerts** when a field's fill rate falls below its established
  baseline. The characteristic failure here is not a crash — it is a column
  that quietly goes empty for three weeks.
- **`RunReport`** per job: counts by kind and by tier, per-stage timings, and
  every failure with its stage.

Storage failures never lose a finished extraction: every write falls back to a
local JSONL file, **one per record type**, so nothing has to be filtered out
before the results are usable:

| File | Holds |
|---|---|
| `output/extractions.jsonl` | successful extractions, with provenance |
| `output/dead_letter.jsonl` | jobs that exhausted their retries, with the error |
| `output/runs.jsonl` | `RunReport` per job |

Read `dead_letter.jsonl` when a run comes back shorter than expected — that is
where the failures are, and they are never mixed in with the results. The unique
index on
`(canonical_url, content_hash, schema_hash)` makes resubmission an upsert, so
idempotency comes from the database rather than from application logic.

---

## Performance

**The queues are split**, because the work has two shapes:

```bash
./run.sh worker-io    # threads × 16 — network waits, cheap to parallelise
./run.sh worker-cpu   # prefork × 2  — Whisper, the model, Chromium, OCR
```

With one queue, a twenty-minute transcription blocks two hundred quick page
fetches behind it.

**Whisper runs on MLX by default.** `faster-whisper` uses CTranslate2, which has
**no Metal backend on macOS** — on an Apple Silicon machine it is CPU-only while
the GPU and Neural Engine sit idle. Published speedups vary a lot by
implementation and model size, so benchmark rather than believe them:

```bash
PYTHONPATH=. ./venv/bin/python -m pipeline.transcribe.bench your-audio.mp3
```

`faster-whisper` stays behind the same interface as the fallback.

Models load **once per worker process** (`worker_process_init`), not once per
task — the original pipeline paid a full model load on every job. Caching is at
two levels: HTTP `ETag`/`Last-Modified` revalidation, where a 304 costs almost
nothing, and extraction keyed on
`sha256(content_hash + schema_hash + prompt_hash)`. Raw bytes go to GridFS so a
prompt change costs a pass over stored bytes rather than a re-crawl.

### macOS specifics, all handled in `run.sh` and `celery_app.py`

- `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` — Celery's prefork pool forks after
  the Objective-C runtime has initialised, and the child then hangs the first
  time it touches Vision, CoreML or MLX. This one costs people days.
- `ulimit -n 4096` — the default 256 is well under what 16 fetchers plus
  Chromium need.
- `brew install ffmpeg` is required by yt-dlp and by live-stream segmenting.
- Ollama already uses Metal; no configuration needed.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

playwright install chromium     # JS-rendered pages
brew install ffmpeg             # yt-dlp audio, live segmenting

cp .env.example .env            # every value has a working default
```

Optional:

```bash
brew install --cask libreoffice # legacy .doc/.xls/.ppt
brew install tesseract          # OCR where Apple Vision is unavailable
pip install docling             # layout-aware PDF parsing
```

Infrastructure — either Homebrew services or one command:

```bash
docker compose up -d            # redis + mongo
docker compose --profile ops up -d   # + flower on :5555
```

A local model, if you want one:

```bash
ollama pull llama3.2:3b
```

Neither MongoDB nor a Groq key is required. Without Mongo, results go to
`output/extractions.jsonl`; without Groq, extraction uses Ollama.

## Running

### Pre-flight Checks
Before starting the pipeline, ensure your background services are actually running. If you're running locally without Docker:

```bash
# 1. Check if Redis is running (should answer PONG)
redis-cli ping

# 2. Check if Ollama is running and has the model
ollama list
```
If Redis is down, Celery workers will fail to connect. If Ollama is down, extraction will time out. *(If you're using Docker for infrastructure, `docker ps` will show if Redis and Mongo are up.)*

### Starting the Pipeline

```bash
./run.sh                # both workers + API + dashboard; Ctrl-C stops all
./run.sh worker-io      # or start pieces individually
./run.sh worker-cpu
./run.sh api            # http://127.0.0.1:8000/docs
./run.sh dashboard      # http://localhost:8501
./run.sh flower         # http://localhost:5555
```

Synchronously, with no broker and no API — the way to debug, since a traceback
stays a traceback:

```bash
PYTHONPATH=. ./venv/bin/python main.py
PYTHONPATH=. ./venv/bin/python main.py https://example.com/a https://example.com/b
PYTHONPATH=. ./venv/bin/python main.py --tiers 1 --schema '{"title":"string"}' https://shop.test/p/1
```

Media only:

```bash
PYTHONPATH=. ./venv/bin/python test_media.py "https://www.youtube.com/watch?v=..."
PYTHONPATH=. ./venv/bin/python test_media.py --live --minutes 10 "https://twitch.tv/someone"
```

## API

```bash
curl -X POST localhost:8000/api/v1/extract -H 'Content-Type: application/json' -d '{
  "url": "https://quotes.toscrape.com/page/1/",
  "prompt": "Extract the first quote, its author and its tags.",
  "schema_template": {"quote": "string", "author": "string", "tags": "list of strings"}
}'
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/extract` | Any URL. Routed to the right queue automatically. |
| `POST /api/v1/extract/batch` | Many URLs, one report. |
| `POST /api/v1/crawl` | Walk a site; collect files or extract every page. |
| `GET /api/v1/crawls/{id}` | Live counters, and the files found so far. |
| `DELETE /api/v1/crawls/{id}` | Stop a crawl. |
| `POST /api/v1/discover` | A site's URL inventory from robots.txt and its sitemap. |
| `GET /api/v1/tasks/{id}` | Poll progress, then the result. |
| `DELETE /api/v1/tasks/{id}` | Revoke — how you stop a live capture. |
| `GET /api/v1/detect` | What the pipeline *would* do with a URL, before committing. |
| `GET /api/v1/stats` | Tier breakdown, cache hit rates, breaker state. |
| `GET /api/v1/specs` | The selector specs learned per domain. |
| `GET /health` | Workers, queue depth, Redis, Mongo, both LLM backends. |

Useful request options: `allowed_tiers` (e.g. `[1]` for structured data only,
`[3]` to force the model), `local_only` (never send content to a hosted model),
`force_dynamic` (browser rendering), `fan_out` (enqueue a feed's or sitemap's
URLs as their own jobs).

## Schemas

Written the friendly way, compiled into real JSON Schema:

```json
{"title": "string", "price": "number", "tags": "list of strings",
 "author": {"name": "string", "url": "string"}}
```

Every field is marked required, unioned with `null`. A field that is genuinely
absent must have a way to say so — without one, a model is pushed into inventing
a plausible value, which is worse than an empty cell.

## Choosing a backend

| | Ollama (local) | Groq (hosted) |
|---|---|---|
| Latency | seconds to minutes | sub-second |
| Cost | free | per token |
| Privacy | content never leaves the Mac | content leaves the Mac |
| Schema | valid JSON, not necessarily your shape | `json_schema` conforms, on supported models |

`LLM_BACKEND=auto` prefers Groq when a key is set. `LOCAL_ONLY=true`, or
`"local_only": true` per request, forces Ollama. Groq's structured-outputs
support is per-model (`openai/gpt-oss-*`, `qwen`); asking an unsupported model
for a schema returns a 400, which the backend recovers from by dropping to JSON
mode and validating here instead.

## Tests

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q
```

385 tests, all offline — HTTP via `respx`, robots.txt via an injected fetcher,
no Ollama and no network required. They cover the safety gates (robots status
handling, redirect re-checking, block detection, SSRF, archive bombs), type
detection, every handler, the cascade's tier ordering, schema repair, the
trust layer, and the crawler's termination properties.

## Layout

```
app.py  tasks.py  celery_app.py  main.py  dashboard.py  run.sh
config.py  models.py  database.py  errors.py  observability.py  urls.py
pipeline/
  compliance/   robots.txt gate, operator policy
  fetch/        client (manual redirects), rate limit, blocks, breaker, cache, browser
  detect/       magic-byte sniffing, the type router
  handlers/     html, document, image, media, livestream, feed, data, archive, email
  discover/     crawl scope, trap detection, the shared frontier
  extract/      cascade, structured data, selector specs, schema, cache, llm/{ollama,groq}
  transcribe/   mlx and faster-whisper behind one interface, plus a benchmark
  normalize/    text, dates, numbers
  trust/        validation, dedupe, drift
  runner.py     the pipeline itself
tests/          385 offline tests
```

## Notes and limitations

- Selector specs need maintenance — that is the cost the pure-LLM path avoids.
  The drift check makes that maintenance automatic rather than something you
  discover months later.
- `MAX_CHUNK_SIZE` truncates long text before the model sees it. Long documents
  are truncated, not chunked-and-merged; a page whose answer sits past the cap
  will miss it, and the item carries a warning saying so.
- Live capture stops at `LIVESTREAM_MAX_MINUTES`. A truncated capture is a
  success with `capture_truncated: true`, not a failure — everything transcribed
  so far is already saved.
- DNS is resolved once for the SSRF check and again by the HTTP client, so a
  hostile server could in principle answer differently the second time. Closing
  that properly needs address pinning, which httpx does not expose.
- Tier 1's field mapping uses a synonym table (`FIELD_SYNONYMS`). A schema using
  unusual field names may fall through to a later tier; adding a synonym is a
  one-line change.
- `GET /api/v1/stats` reads its tier breakdown from MongoDB, so without
  `MONGO_URI` it reports zeros. The cache and rate-limiter figures in the same
  response are per-process, and the API process is not the one doing the
  fetching — the workers' own numbers appear in each job's `RunReport`.
