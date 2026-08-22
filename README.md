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

## Indexing for retrieval

**On by default** — a feature that is off by default is a feature nobody has.
Every document that finishes extraction is also cleaned, split, embedded and
written to a local LanceDB table, and is then searchable from the dashboard's
Search tab or `POST /api/v1/search`. Pass `"index": false` to skip it for a job
that only wants structured JSON:

```bash
curl -X POST localhost:8000/api/v1/extract -H 'Content-Type: application/json' -d '{
  "url": "https://example.com/report",
  "prompt": "Extract the title.",
  "schema_template": {"title": "string"},
  "index": true
}'
```

`INDEX_ENABLED=true` makes it the default for every job; `"index": true` turns
it on per request, and `false` off again.

**How a document gets split is decided from the document.** A page that
publishes its own headings has already said how it wants to be divided, so those
win. Text whose lines are too short to be prose is the OCR-damage case, and the
only one worth a model call. Length on its own justifies splitting by meaning.
Everything else gets fixed windows.

| Shape | Strategy |
|---|---|
| Three or more markdown headings | `hierarchical` — split on headings, heading path kept as the section name |
| Ten or more lines averaging under 50 characters | `fixed` — see below |
| Over 2000 characters of prose | `semantic` — split where the embedding similarity drops |
| Anything else | `fixed` — overlapping character windows |

A run report says which strategy *ran*, not which was chosen: when the model is
unreachable the `llm` strategy falls back to fixed windows and reports `fixed`,
because a number claiming a model ran on a machine where none did is worse than
no number.

**The router will not choose `llm` on its own.** Short lines mean OCR damage in
a scanned document and a navigation menu in a web page, and the test cannot tell
them apart. Since indexing is on by default, guessing wrong means a model call
per document on ordinary pages — measured at 39 seconds and 78 chunks for one
page of quotes, against 85 milliseconds and 7 chunks once the router was
corrected. Set `INDEX_AGENTIC_ALLOW_LLM=true` for a corpus of scans, or ask for
`INDEX_CHUNK_STRATEGY=llm` directly.

**Indexing is a separate task on the cpu queue**, not a stage in the runner.
Embedding is a transformer forward pass, and `extract_url` runs on the io pool
where it would hold a thread that sixteen fetches are queued behind. It also
means a vector store that is down cannot fail an extraction that already
succeeded — the extraction returns, and the index task fails on its own.

Every chunk carries provenance: source URL, content hash, page number, section
path, language, which extraction tier produced the text, and which model
embedded it.

```bash
curl -s localhost:8000/api/v1/index/stats | python3 -m json.tool
```

**One table holds one embedding space.** Cosine distance between a 384-d BGE
vector and a 512-d CLIP vector is not a bigger or smaller number, it is a
category error, and no vector store will stop you — so the table records which
model wrote it and a mismatched write is refused. Use a separate
`LANCE_TABLE_NAME` per model.

**Two macOS traps, on top of the ones below.** `MTLCompilerService` does not
survive `fork()`: a prefork worker child that builds a Metal pipeline dies with
`SIGABRT`, and `OBJC_DISABLE_INITIALIZE_FORK_SAFETY` does not cover it. The
uvicorn process dies the same way on its first embedding call, with no traceback
because it is killed by a signal. So `INDEX_EMBED_DEVICE` defaults to the CPU,
where BGE-small embeds 32 chunks in about a quarter of a second and leaves the
GPU to Whisper, which needs it. And the model is *not* preloaded at process
init: it takes about six seconds to construct and billiard kills a child that
has not reported ready in four, so preloading turns worker startup into an
endless loop of half-loaded children. It loads on the first indexing task
instead and stays resident — the first document costs ~11s, the next ~0.3s.

---

## Retrieval

Hybrid over the same rows, plus a knowledge graph:

```bash
curl -X POST localhost:8000/api/v1/search -H 'Content-Type: application/json' -d '{
  "query": "What was Doug Field's trajectory before joining Ford?",
  "filters": {"department": ["research"], "date_from": "2026-01-01"},
  "limit": 10, "use_graph": true
}'
```

The dashboard's **Search** tab is the same thing with a UI: a question box,
filter pickers populated from what is actually indexed, fusion and rerank
controls, and results showing score, source and which leg found each passage.
The sidebar reports how many chunks and graph entities exist, which is the first
thing to check when a search returns nothing.

Chunks and triples come back with their provenance. **Nothing here writes an
answer** — retrieval returns evidence and the orchestrator above it decides what
to do with that. Keeping synthesis out means this layer can be judged on whether
it found the right evidence, which has a correct answer, rather than on whether
the prose reads well, which does not.

### One model call, three rewriting techniques

Self-query, sub-query decomposition and step-back prompting are usually three
calls. They are three questions about the same sentence, so they are three
fields of one structured answer:

| Technique | What it contributes |
|---|---|
| **Self-query** | Turns "Q3 finance reports by Chen" into a filter. Without it the metadata columns are decoration — something has to decide "finance" is a department and not a search term. |
| **Decomposition** | "How did revenue and headcount change after the merger" retrieves badly as one query and well as two. |
| **Step-back** | The general form of the question, for when the specific wording matches nothing. |

And for most questions the call is skipped entirely: short, single-clause, no
comparison and no date expression means retrieve directly. Plans are cached by
normalised query. Rewriting a question that did not need it is the most
avoidable latency in the whole path.

### Two legs, then fusion

Dense and BM25 fail in opposite directions, which is the reason to run both.
Dense finds "how do I get my money back" in a document that says *refunds* and
never says *money*; it also returns something vaguely topical when the answer is
absent. BM25 finds the part number, the surname, the error code — where being
approximately right is being wrong — and finds nothing when the wording differs.

Fusion is ours rather than the store's because there are two dimensions to fuse
across: the two legs, and the several sub-queries a decomposed question
produced.

- **`rrf`** (default) combines by rank alone, so it is unaffected by cosine
  similarity and BM25 relevance being unrelated scales — the thing that makes a
  weighted sum of raw scores misbehave.
- **`alpha`** weights normalised scores: `alpha` on dense, `1 - alpha` on BM25.
  Worth tuning once you have judgements to tune against.

Both legs and the graph run concurrently. Every sub-query embeds in one batched
`encode()`, not one call each.

### Filters run before the search, not after

```json
{"department": ["finance"], "author": ["Chen"], "date_from": "2026-01-01"}
```

`doc_type`, `department`, `date`, `author`, `region`, `permission_level`,
`language` and `source`, each with a B-tree scalar index so `prefilter=True` is a
lookup rather than a scan. Post-filtering would ask for ten results and then
throw some away, so a selective filter returns fewer than ten — or none.

**Where the values come from is not uniform, and pretending otherwise would be
the bug.** `language` and `doc_type` are already known; `author` and `date` are
harvested from JSON-LD, OpenGraph and PDF metadata the cascade already reads;
`department`, `region` and `permission_level` are supplied by whoever ingested
the document:

```bash
curl -X POST localhost:8000/api/v1/extract -H 'Content-Type: application/json' -d '{
  "url": "https://internal.example/report", "prompt": "...", "schema_template": {...},
  "index": true, "metadata": {"department": "finance", "permission_level": "internal"}
}'
```

Nothing infers a permission level. A security control derived from a guess is
worse than no control, because it looks like one. An unset field does not match
a filter on it — the safe direction for permissions, and the surprising one for
department, where a crawled page simply has none.

### Quantisation, and when not to use it

`IVF_PQ` with 48 sub-vectors compresses a 384-float32 vector from 1536 bytes to
48. `RETRIEVE_NPROBES` sets how many partitions are searched and
`RETRIEVE_REFINE_FACTOR` rescores the shortlist exactly, which recovers most of
what quantisation costs in accuracy.

**The index is not built below `INDEX_ANN_MIN_ROWS` (5000).** IVF_PQ trains on
the data — it clusters to build partitions and learns a codebook — so on a small
table it is slower than a flat scan *and* less accurate than one. Under the
threshold every search is exact brute-force KNN, which on a few thousand vectors
is a millisecond.

### The knowledge graph

Ported from [Dolmaa24/GraphRAG](https://github.com/Dolmaa24/GraphRAG):
LLM extraction → entity resolution → Kuzu → vector-seeded traversal.

It answers what flat retrieval structurally cannot. "What was Doug Field's
trajectory before Ford" needs three facts from three documents joined through a
shared entity; no single chunk contains the answer, so no amount of chunk ranking
finds it. The graph has the join:

```
(Doug Field)-[WORKED_AT (2013)]->(Tesla)
(Doug Field)-[LED (2018)]->(Project Titan)
(Doug Field)-[DEPARTED (2021)]->(Apple)
(Doug Field)-[JOINED (2021)]->(Ford Motor Company)
(Elon Musk)-[SUPERVISED]->(Doug Field)
```

### Entities are not generated, they are scored

Naming the entities in a passage is a classification problem. Asking a decoder
to answer it by emitting JSON token by token is the slowest available way to get
a label, and it was why graph extraction dominated ingest cost.

[GLiNER](https://github.com/urchade/GLiNER) scores spans instead — an encoder
under 500M parameters, on the CPU. Measured on the same paragraph:

| | LLM entities | GLiNER entities |
|---|---|---|
| Entity extraction | ~10 s | **166 ms** |
| Whole extraction, per document | 6.8 s | **3.7 s** |
| Entities found (benchmark) | 10/11 | **11/11** |

The model is still asked for relationships, which GLiNER does not do — and that
call shrinks too, because given the entity list it only has to emit edges.
Descriptions come from the text rather than a model: GLiNER returns offsets, so
the sentence around the first mention is free to take, is a real quote, and
cannot hallucinate.

Set `GRAPH_ENTITY_BACKEND=llm` for the original path. If the package is missing,
it falls back there on its own rather than failing the job.

### Directions are checked, not hoped for

`Northwind ACQUIRED Fabrikam` and its reverse are a fact and a falsehood, and
nothing downstream can tell them apart — the graph *is* the source of truth for
relationships. `llama3.2:3b` reversed two of four directional edges on the
benchmark. Two signals now catch that:

- **Type asymmetry.** `FOUNDED` runs from a person or organisation *to* an
  organisation or project, so `(Project Titan)-[STARTED]->(Apple)` is
  structurally impossible and the flip is structurally fine.
- **Word order.** Types cannot help when both ends are the same kind — two
  organisations either way round. But the sentence says "Northwind Traders
  acquired Fabrikam Ltd", and an active subject precedes its verb. Passive voice
  inverts that, so "was acquired by" is detected before deciding.

The rule throughout is to **act only when one orientation fits and the other does
not**. Ambiguous edges are left exactly as the model wrote them, because
flipping on a guess would replace one wrong-fact generator with another.

Measured on `llama3.2:3b`, the model that got these wrong:

```
without validation   edges ok 2   reversed 2
with validation      edges ok 4   reversed 0
```

Six things changed in the port:

- **Every query is parameterised.** The original built Cypher by string
  interpolation with hand-escaped quotes, on entity names a model invented from
  scraped pages. `O'Brien & Co` broke it; something crafted did worse.
- **Generated Cypher runs on a read-only database**, not merely a prompt asking
  the model to only read. The keyword check is the first guard; a database that
  cannot write is the one that holds when the check is wrong.
- **Edges `MERGE` instead of `CREATE`**, so re-ingesting a document updates the
  graph rather than doubling it.
- **The candidate pool is embedded once**, not re-encoded for every new entity —
  the original was O(N²) model calls in the size of the graph.
- **Entity resolution strips corporate suffixes first.** `Apple Inc.` ≡ `Apple`
  is the most common alias there is, and llama3.2:3b asked directly gets it
  wrong. Deterministic first, similarity second, model only for what is left.
- **Every node and edge carries provenance** — source URL and content hash. A
  triple you cannot trace is one you cannot check or expire.

Graph building is **on by default** — the graph is half of what "hybrid and
graph retrieval" means, and one nobody builds answers no questions. It costs one
model call per `MAX_CHUNK_SIZE` **window** of a document, not per chunk as an
earlier version of this file said, which still makes it the slowest part of
ingest; pass `"build_graph": false` to skip it. Long documents are windowed
rather than truncated — an earlier version passed only the first 12,000
characters, so a fifty-page PDF produced a graph of its first four pages and
said nothing about the rest. Results are **cached on the content hash**, because
this is both the most expensive call here and a perfectly deterministic one.
Measured on the same document:

```
first build    23.7s      extraction 9.8s
re-ingest       0.14s     extraction 0.001s
```

The key covers the content, the prompt and the model name. A different model
does not extract the same graph from the same paragraph, and serving one where
the other was asked for would be a quality regression that looks like a cache
hit. The cache is process-wide and backed by Mongo when one is configured, so it
also survives `worker_max_tasks_per_child` recycling the worker.

**Kuzu is single-writer and the lock is process-wide.** Measured: many read-only
processes coexist, but one read-write handle blocks every other open, including
read-only ones. So retrieval opens read-only and ingest opens, writes and
closes — a worker holding the write lock would make search fail for as long as
it was alive.

---

## Answering

Retrieval finds the evidence. This is the part that answers the question:

```bash
curl -X POST localhost:8000/api/v1/answer -H 'Content-Type: application/json' -d '{
  "query": "Who said the world as we have created it is a process of our thinking?"
}'
```

```json
{"answer": "Albert Einstein", "sufficient": true, "cited": [1],
 "sources": [{"number": 1, "kind": "passage", "origin": "https://quotes.toscrape.com/page/1/"}]}
```

The dashboard's Search tab does the same with **Answer the question** selected,
showing the answer above the passages and an expander listing exactly what the
model was allowed to read, each marked cited or not.

Three properties matter more than the prose:

**Every claim carries a number.** Sources are numbered in the prompt and the
answer refers to them as `[1]`. Citations are read back out of the answer text
rather than trusted from a separate field, because what the model wrote into the
sentence is what it actually used.

**"We don't have that" is a real outcome.** A model handed six passages will
compose something from them whether or not they are relevant, so `sufficient` is
asked for separately. A small model often returns that verdict with no prose at
all, so an empty answer is turned into the sentence the verdict means rather
than shown as a blank box.

**A dead model does not lose the evidence.** If generation fails, the passages
and triples still come back with a warning attached — the retrieval was real
even when the prose is not.

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
| `POST /api/v1/upload` | Send a file instead of a URL; returns a reference to extract. |
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
| `POST /api/v1/answer` | Ask a question; get an answer that cites its sources. |
| `POST /api/v1/investigate` | Ask a harder one; the agent loop searches, judges whether that was enough, and goes back for what was missing. |
| `GET /api/v1/investigations/{id}` | Progress while it runs, then the answer and the reasoning. |
| `GET /health` | Workers, queue depth, Redis, Mongo, both LLM backends. |

## Investigations

`POST /api/v1/answer` retrieves once and answers. That is the right shape for
most questions and takes a few seconds.

A question with two parts often needs two searches, where the second depends on
what the first turned up — and nothing in a single-shot answer notices that the
first was not enough. `POST /api/v1/investigate` runs the agent loop instead:
a specialist searches the corpus and the graph, an answer is drafted, and the
draft's own `sufficient` flag decides whether to go back out with what was
found. Half a minute rather than five seconds, so use it when the question
earns it.

```bash
curl -X POST localhost:8000/api/v1/investigate -H 'Content-Type: application/json' -d '{
  "question": "What did Acme acquire, and what was the quarterly revenue?"
}'
```

Then poll `/api/v1/investigations/{id}` — it reports which stage is running
while it works, and returns the answer, its sources, the verification result and
the full trace of every tool call when it finishes. The dashboard's **Investigate**
tab does the same and shows the trace as it goes.

Investigations run on their own `agents` queue, started by `./run.sh` along with
everything else. A ninety-second run on the `io` queue would sit in a thread
sixteen page fetches are waiting behind.

**Reaching outside the corpus is off by default.** `allow_network` lets a run
fetch a URL named in the question once the corpus has come up short, and
`allow_write` lets it index what it fetched — both are needed for either to do
anything, since fetching without indexing changes nothing. Neither is ever
turned on by the model.

**The answer is checked against its own sources**, and sentences the passages do
not support are marked `[unsupported]` rather than removed. Measured at 82% of
unsupported claims caught and 29% of supported sentences flagged wrongly
(`bench/verify.py`), which is why it annotates rather than deletes — and why
`AGENT_VERIFY=false` is there if the false alarms bother you more than the
misses.

## MCP: using the corpus from Claude Desktop or Claude Code

The same tools the internal agent loop uses are exposed over the Model Context
Protocol, so an external client can search the corpus, traverse the graph and —
if you allow it — extract new pages.

```bash
./run.sh mcp            # stdio, what a desktop client spawns
./run.sh mcp --http     # streamable HTTP on MCP_HTTP_PORT (default 8765)
```

For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "corpus": {
      "command": "/absolute/path/to/WebScraping_PipelineTry2/venv/bin/python",
      "args": ["/absolute/path/to/WebScraping_PipelineTry2/mcp_server.py"],
      "env": { "PYTHONPATH": "/absolute/path/to/WebScraping_PipelineTry2" }
    }
  }
}
```

**Seven read-only tools are exposed by default** — `corpus_profile`,
`search_corpus`, `answer_from_corpus`, `graph_neighbors`, `graph_path`,
`fetch_chunk`, `poll_task`. Tools that reach the network or change the corpus
are *not advertised at all* unless you turn them on:

```bash
MCP_ALLOW_NETWORK=true   # detect_url, discover_sitemap
MCP_ALLOW_WRITE=true     # extract_url, crawl_site, index_document
```

Absent rather than refused, because a client shown a tool it cannot call will
call it, spend a turn learning that, and try again.

Two things worth knowing before enabling write. The workers must be running —
those tools enqueue Celery jobs and return a task id to poll. And a write tool
taking free text is an attractor: asked to *delete* documents, both local 3B
models reached for `index_document` instead of declining. The effect gate, not
the model's judgement, is what protects the corpus.

The corpus is also exposed as MCP resources: `corpus://profile`, and
`corpus://chunk/{chunk_id}` for citing one passage by URI.

Useful request options: `allowed_tiers` (e.g. `[1]` for structured data only,
`[3]` to force the model), `local_only` (never send content to a hosted model),
`force_dynamic` (browser rendering), `fan_out` (enqueue a feed's or sitemap's
URLs as their own jobs).

### Files, rather than URLs

Upload the file, then extract the reference you get back — everything
downstream treats it exactly like a URL:

```bash
curl -X POST localhost:8000/api/v1/upload -F 'file=@quarterly-report.pdf'
# {"url": "upload://3f9a1c2ed4b1-quarterly-report.pdf"}
```

The file is written to `UPLOAD_DIR` and read from disk by whichever worker
picks the job up. It is never fetched over HTTP and never served over HTTP:
the pipeline refuses to fetch private addresses, which is exactly what its own
API is, and uploads are whatever the user gave us — not something to expose on
an API with no authentication. An `upload://` URL can only ever name a file
inside that one directory, so it is not a way to read anything else on the
host.

On a split deployment, `UPLOAD_DIR` must be a volume the API and the workers
both see.

## Schemas

Written the friendly way, compiled into real JSON Schema:

```json
{"title": "string", "price": "number", "tags": "list of strings",
 "author": {"name": "string", "url": "string"}}
```

Every field is marked required, unioned with `null`. A field that is genuinely
absent must have a way to say so — without one, a model is pushed into inventing
a plausible value, which is worse than an empty cell.

## Choosing a model

The extraction model is scored on the jobs this pipeline gives it, not on a
leaderboard — `bench/graph_models.py` runs three documents with known answers,
four alias pairs, and five traversal prompts checked against Kuzu's own parser:

```bash
PYTHONPATH=. ./venv/bin/python -m bench.graph_models llama3.2:3b qwen2.5:3b
```

"Extra edges" counts relationships beyond the ones each case asks about. Not all
are wrong — the text states more than the two or three facts written down — but
fabrication shows up there and nowhere else, which is why the column exists: an
earlier version scored a model perfectly while it was inventing edges.

With everything on — GLiNER entities and direction validation:

| | `llama3.2:3b` | `qwen2.5:3b` |
|---|---|---|
| Entities found | 11/11 | 11/11 |
| Edge directions right | 4 | 3 |
| Edge directions reversed | 0 | 0 |
| Edges missing | 0 | 1 |
| Extra edges | 3 | 3 |
| Alias judgement | 3/4 | **4/4** |
| Cypher Kuzu accepts | 1/5 | **2/5** |
| Time | 32.8s | **22.4s** |

Raw, with neither check, the gap was starker: llama reversed **two of four**
directional edges and missed the `IBM` / `International Business Machines`
identity, while qwen reversed none.

`qwen2.5:3b` is the default, but note what changed. Direction validation now
fixes reversals for *either* model, so the original argument for qwen — that it
does not reverse edges — no longer decides it. What remains is alias judgement
(4/4 against 3/4), which still matters because a split entity breaks exactly the
multi-hop join the graph exists to make, and it compounds as the corpus grows.

Two cautions the benchmark itself taught:

- **Check Cypher against the database, not a regex.** An earlier version of this
  benchmark used only the static guards and scored `qwen2.5:3b` 2/2 on queries
  Kuzu then rejected outright (`[r:r:CONNECTS_TO]`, `{{name: ...}}`).
- **Do not tune the prompt by eye.** Adding "project" to the entity-type list
  recovered a missed entity on the document being looked at — and reversed an
  edge elsewhere. Scored across the whole benchmark it was a net loss, so it was
  not shipped.

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

### Interactive and bulk want different backends

That table says "always use the hosted one" until you look at volume. Calls here
come in two shapes:

| | Interactive | Bulk |
|---|---|---|
| Examples | query understanding, graph traversal | extraction, chunking, graph building |
| Tokens per call | ~400 | ~3,100 |
| Volume | a few a minute | one per chunk |
| Someone waiting | yes | no |

A local 3B was measured here at **~35 tok/s** generating and ~280 tok/s
prefilling; hosted inference runs several hundred to a thousand. For an
interactive call that is ~0.3s against ~8s, and the user feels every bit of it.

For bulk it inverts. A free tier's tokens-per-minute cap — 8–12K depending on
model — divided by ~3.1K tokens per chunk is **roughly two to four chunks a
minute**. Local has no such ceiling, so it is *faster* for ingest despite being
slower per call.

So `LLM_INTERACTIVE_BACKEND` can point somewhere different from `LLM_BACKEND`:

```bash
LLM_BACKEND=ollama              # ingest: unlimited, local, slow per call
LLM_INTERACTIVE_BACKEND=groq    # queries: fast, rate-limited, low volume
```

An interactive preference that is unavailable falls back to the bulk backend
rather than failing — a missing Groq key should make search slower, not broken.
An *explicitly named* backend still fails loudly, because naming one is a
decision about where content may go rather than a preference about speed.

### Ollama settings that matter

Two defaults cost real time and one costs correctness:

- **`OLLAMA_KEEP_ALIVE=30m`.** Ollama unloads the model five minutes after the
  last call. Reloading measured ~2.2s against ~170ms warm, so an intermittent
  pipeline pays it on nearly every call.
- **`OLLAMA_NUM_CTX=8192`.** Prompt and generation share the window and the
  default is 4096. A full `MAX_CHUNK_SIZE` chunk measured 2,919 prompt tokens,
  so a long answer overflows — and Ollama responds by *shifting* the context,
  discarding the front of the prompt where the instructions are. The symptom is
  bad extraction, not an error. The cost is KV cache: `llama3.2:3b` measured
  2.0 GB resident at 4096 and 3.0 GB at 8192.

## Tests

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q
```

262 tests, all offline — no network, no Ollama, no model download. The dense
embedder and the LLM backends are faked where a test only needs *a* vector or
*an* answer; the one test that loads BGE for real is marked `slow`:

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q -m "not slow"
```

They cover the cleaning steps, the chunk router's four branches, the
model-assisted splitter and each of its fallbacks, the dense embedder's
one-load-per-process guarantee, the store's embedding-space guard and ANN
threshold, filter compilation including injection attempts, the query planner's
gate and cache, both fusion strategies, backend routing (an interactive
preference falls back when unavailable; a named backend does not), the
extraction cache (a changed prompt or model invalidates it; an empty extraction
is never stored), and — for the graph — Cypher injection, re-ingest idempotency,
multi-hop path expansion, suffix resolution, the read-only guard and the
write-lock lifecycle.

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
  preprocess/   cleaning, encoding repair, language, optional PII masking
  chunk/        the strategy router and the four splitters
  embed/        dense (local/hosted), multimodal
  store/        the LanceDB table: vectors, BM25, filter columns
  index/        preprocess -> chunk -> embed -> store, as one call
  retrieve/     query understanding, hybrid search, fusion, reranking
  graph/        GLiNER entities, relations, direction checks, Kuzu, cache
  runner.py     the pipeline itself
tests/          262 offline tests; one marked `slow` loads the real model
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
- **Graph building is one model call per `MAX_CHUNK_SIZE` window**, and on
  Groq's free tier (8000 TPM) that is the binding constraint on ingest — not
  CPU. A long document will rate-limit before it finishes. `GRAPH_MAX_WINDOWS`
  caps how far into a very long document the graph goes, and a document past
  that cap logs a warning rather than silently stopping.
- **The Cypher agent is off by default.** Neither local model writes Cypher
  Kuzu reliably accepts — `qwen2.5:3b` 2/5, `llama3.2:3b` 1/5 — so the fixed
  template answers anyway: 39ms against the agent's 1.9s for identical triples.
  Turn it on with a strong hosted model.
- **Entity resolution at 0.70 cosine can still merge distinct entities** with
  similar names. Suffix stripping and the model check reduce this; neither
  removes it.
- **A 3B extraction model still gets things wrong.** Direction validation
  catches reversals and GLiNER catches missed entities, but both models still
  occasionally invent a relationship between two entities the text never
  connects, and neither check can see that. `bench/graph_models.py` is there to
  re-measure when you change the model; do not change the extraction prompt
  without running it, because a one-word change that recovered an entity also
  reversed an edge.
- **Metadata is per-document, so a page with many authors gets one.** The
  `author` on every chunk of a page comes from that page's JSON-LD or OpenGraph.
  On a page of quotes by different people, every chunk carries whichever author
  the markup named. Filter on it accordingly.
- **Direction validation only covers the verbs it knows.** A relation outside
  `RULES` in `pipeline/graph/validate.py`, or one between two same-typed
  entities with no usable word order, is left as the model wrote it. It reduces
  the failure rate; it does not eliminate it.
- **`department`, `region` and `permission_level` are empty for everything
  crawled so far.** Filtered retrieval on them returns nothing until content is
  re-ingested with them supplied.
- **The reranker is the memory straw** on 8 GB, on top of BGE-small, Kuzu,
  LanceDB and a local LLM. Off by default.
- Retrieval is per-question; there is no multi-agent orchestration or answer
  synthesis yet. This layer returns evidence.
- `langchain-experimental` is sunset upstream. It is used for one thing,
  `SemanticChunker`; replacing it means embedding sentences and cutting at a
  percentile breakpoint, which is perhaps forty lines against the dense embedder
  already loaded.
- `GET /api/v1/stats` reads its tier breakdown from MongoDB, so without
  `MONGO_URI` it reports zeros. The cache and rate-limiter figures in the same
  response are per-process, and the API process is not the one doing the
  fetching — the workers' own numbers appear in each job's `RunReport`.
