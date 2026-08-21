# Agents Over the Pipeline

*A supervisor that plans, retrieves, goes and fetches what's missing, and cites
what it used — built on one tool registry that MCP clients and the internal loop
both consume.*

| | |
|---|---|
| **Written against** | `rag-pipeline` @ `0afe345` — 24,423 LOC, 298 tests |
| **New code, est.** | ~2,400 LOC |
| **Phases** | 6, each shippable |
| **Rendered version** | https://claude.ai/code/artifact/b066666c-aaa5-4684-94cb-b12fe1bacfc5 |

> **Open decision.** This plan argues against adopting an agent framework (see
> *What I'd leave out*). The current preference is to use **LangGraph** anyway,
> which changes phases 3 and 4 but leaves 0, 1 and 2 untouched. Settle it before
> starting phase 3.

## The central decision: one registry, two consumers

MCP is a *transport*, not an architecture. The temptation is to build the MCP
server as one project and the agent orchestration as another; do that and you
write the tool catalog twice, and the two drift within a month.

So: define each capability once — name, JSON Schema in and out, handler, and a
declared side effect. The MCP server is an adapter that speaks that registry over
stdio and HTTP. The internal agent loop is a consumer that calls the same
handlers in-process. Neither owns the catalog.

```
   External                                          Internal
   ┌──────────────────┐                    ┌──────────────────────┐
   │   MCP clients    │                    │      Agent loop      │
   │ Claude Desktop   │                    │ supervisor+specialist│
   └────────┬─────────┘                    └──────────┬───────────┘
            │                                         │
   ┌────────┴─────────┐                    ┌──────────┴───────────┐
   │  mcp_server.py   │                    │   agents/loop.py     │
   │  stdio · /mcp    │                    │   budget · trace     │
   └────────┬─────────┘                    └──────────┬───────────┘
            │                                         │
            └──────────────┐         ┌────────────────┘
                     ┌─────┴─────────┴──────┐
                     │    Tool registry     │
                     │ schema·handler·effect│
                     │ pipeline/agents/tools│
                     └──────────┬───────────┘
                                │
        ┌───────────┬───────────┼───────────┬────────────┐
   retrieve()  GraphRetriever  extract_url  crawl_site
   hybrid+BM25 Kuzu traversal  Celery·io    policy-gated
              Existing pipeline · unchanged
```

Both consumers see the same tools with the same schemas. Adding a capability once
makes it available to Claude Desktop and to the supervisor in the same commit.

**Why this is the right seam for this codebase.** Every capability here is already
a function with typed arguments and a JSON-serialisable return — `retrieve()`,
`answer_question()`, and nine Celery tasks. The pipeline was written task-shaped
before there were any agents. A tool registry is a thin descriptive layer over
what exists, not a rewrite.

## What's already there, and the one thing that isn't

| | |
|---|---|
| **Orchestration** | Celery with split `io`/`cpu` queues, retries, soft timeouts, progress reporting. You do not need an agent framework's scheduler — you have a better one. |
| **Retrieval** | `retrieve()` at `pipeline/retrieve/orchestrator.py:87` already fuses dense, BM25 and graph legs concurrently, with pre-filtering and optional reranking. |
| **Query planning** | `understand()` does self-query, decomposition and step-back in one call, and skips the call entirely for trivial questions. |
| **Synthesis** | `answer_question()` already returns numbered citations and a `sufficient` flag — the honest-refusal path exists. |
| **Safety** | `pipeline/compliance/policy.py` refuses private addresses, honours robots, and rate-limits per host. This turns out to be load-bearing for agent safety, not just politeness. |
| **Provider routing** | Local Ollama by default, Groq when throughput matters, `local_only` as a hard privacy switch. |

**The missing piece.** The `LLMBackend` protocol in
`pipeline/extract/llm/base.py` has exactly one method: `complete_json()`. It takes
content and returns a shape. There is no way for a model to say *"call this tool,
then let me see the result."* The Ollama backend even posts to `/api/generate`
rather than `/api/chat`, so there is no message history to append a tool result
to. Everything in this plan past phase 1 depends on fixing that, and nothing else
is close to as fundamental.

## The tool catalog

Eleven tools, each wrapping something that already works. The *effect* column is
not documentation — it is enforced at call time, and it is what stops an agent
from spending twenty minutes crawling a site because it inferred that would be
helpful.

| Tool | Wraps | Effect | Typical cost |
|---|---|---|---|
| `corpus_profile` | index stats + filter values | read | ~5 ms |
| `search_corpus` | `retrieve()` | read | 200–900 ms |
| `answer_from_corpus` | `answer_question()` | read | 2–6 s |
| `graph_neighbors` | `GraphRetriever` | read | 10–80 ms |
| `graph_path` | Kuzu 2–3 hop traversal | read | 20–200 ms |
| `fetch_chunk` | LanceDB row by id | read | ~5 ms |
| `detect_url` | `pre_route()` preview | network | ~300 ms |
| `discover_sitemap` | `tasks.discover_sitemap` | network | 1–10 s |
| `extract_url` | `tasks.extract_url` | write | 5–60 s |
| `crawl_site` | `tasks.crawl_site` | write | minutes |
| `poll_task` | `AsyncResult` | read | ~5 ms |

The three effect classes carry different permissions. **read** touches only what
has already been indexed and is always available. **network** reaches the outside
world and is off unless the request opts in. **write** changes the corpus, and
additionally returns a task id rather than blocking — long jobs go through Celery
exactly as they do today.

---

## Phase 00 — Tool contracts (~2 days)

New package `pipeline/agents/tools/`. A `@tool` decorator that registers a name, a
description written for a model rather than a human, a Pydantic argument model
that generates the JSON Schema, the handler, an `Effect`, and a cost hint.

```python
# pipeline/agents/tools/registry.py
@tool(
    name="search_corpus",
    effect=Effect.READ_ONLY,
    cost_ms=600,
    description=(
        "Search indexed documents. Returns passages with source and score. "
        "Use filters to narrow by department, region or date; call "
        "corpus_profile first to see which filter values exist."
    ),
)
def search_corpus(args: SearchArgs) -> SearchResult:
    return to_result(retrieve(args.query, filters=args.filters, limit=args.limit))
```

Descriptions matter more than usual here: a 3B model picks tools almost entirely
from the description text, and *"call `corpus_profile` first"* inside a
description is worth more than the same sentence in a system prompt.

**Ships:** a registry with eleven tools, a JSON Schema for each, and tests that
call every handler with zero model calls. Nothing agentic yet — and that is the
point, because this layer stays correct whether or not the rest lands.

## Phase 01 — The MCP server (~1 day)

New `mcp_server.py` over the official `mcp` Python SDK. Two transports from one
registry: stdio for Claude Desktop and Claude Code, and streamable HTTP mounted on
the FastAPI app already running at `127.0.0.1:8000`.

- Read-only tools are exposed by default. `MCP_ALLOW_ACQUISITION=false` keeps
  network and write tools out of the advertised list entirely, so a client cannot
  call what it cannot see.
- Long jobs return `{"task_id": …}` immediately and are polled with `poll_task`.
  MCP clients have short patience; Celery already has the durable half.
- Corpus documents are exposed as MCP *resources* as well as tools, so a client
  can cite a chunk by URI.

**This phase is worth doing even if you stop here.** Phases 0 and 1 together are
about three days and give you a working system: point Claude Desktop at the corpus
and ask multi-step questions, with Claude itself as the orchestrator. Everything
after this is about making the pipeline able to do that *without* an external
client — valuable, but no longer the difference between having something and
having nothing.

**Ships:** a running MCP server, a `claude_desktop_config.json` snippet in the
README, and one end-to-end transcript of a multi-hop question answered from the
corpus through an external client.

## Phase 02 — Tool calling in the model layer (~3 days)

The real code change. Extend the `LLMBackend` protocol with a second method:

```python
def complete_with_tools(
    *,
    messages: list[Message],
    tools: list[ToolSpec],
    tool_choice: str = "auto",
) -> ToolTurn:
    """Either a list of tool calls to run, or final text. Never both."""
```

- **Ollama** moves from `/api/generate` to `/api/chat` with a `tools` array.
  `qwen2.5:3b` supports this; `keep_alive` and the existing cache carry over
  unchanged.
- **Groq** gets native tool calling, which is materially better at it and is the
  right home for the supervisor role.
- **A fallback shim** for models that cannot tool-call at all: a constrained
  "next action" JSON object through the existing `complete_json()`. Slower and
  more fragile, but it keeps every model in the stack usable rather than making
  tool calling a hard requirement.

Add `AGENT_BACKEND` as routing separate from the extraction backend, so the
supervisor can run on Groq while extraction stays local and private.

**Measure before building on it.** Extend `bench/graph_models.py` into a
tool-selection benchmark: thirty prompts with a known correct tool and arguments,
scored for right tool, right arguments, and fabricated tool names. You have been
burned before by a prompt change that looked good on one document and regressed
across the set. Assume a 3B model will select correctly somewhere between half and
three-quarters of the time and let the number, not the hope, decide whether the
supervisor runs locally.

**Ships:** both backends tool-calling behind one interface, the fallback shim, and
a scored benchmark table per model.

## Phase 03 — The loop, and its budget (~3 days)

`pipeline/agents/loop.py`. One bounded loop: send messages and tools, receive
either tool calls or final text, execute, append observations, repeat. Roughly 250
lines. The interesting part is not the loop, it is everything that stops it.

```python
class Budget:
    max_iterations:   int = 8
    max_tool_calls:   int = 20
    max_seconds:      float = 90.0
    max_tokens:       int = 60_000
    network_calls:    int = 0     # opt-in per request
    write_calls:      int = 0     # opt-in per request
```

- **Parallel execution** when a turn emits several calls that are all read-only.
  The retrieval layer is already thread-safe and already runs its own legs
  concurrently.
- **A no-progress stop.** If a turn produces an observation identical to one
  already seen, end the run. Small models loop on the same failing search far more
  often than they hallucinate a tool.
- **A trace** of every step — call, arguments, latency, bytes returned, and the
  budget remaining. This is both the debugging surface and the payload the
  dashboard renders.

**Ships:** a loop that can be driven by a scripted fake backend in tests, with
budget exhaustion, no-progress detection and parallel dispatch all covered without
a model running.

## Phase 04 — The agents (~4 days, acquisition gated)

A supervisor delegating to specialists — a star, not a mesh. Two reasons: with a
small model, fewer decisions per call is strictly better; and a star topology
produces a trace a person can actually read.

**Supervisor** — decomposes the question, picks a specialist, decides after each
return whether the evidence is sufficient or another hop is needed. Holds the
budget. This is the role to run on Groq.
*delegates → corpus · acquisition · synthesis*

**Corpus agent** — owns search and graph traversal, and, unlike today's
single-shot path, re-queries when results come back thin: widen a filter, switch
fusion from RRF to alpha, follow a graph edge and search again. This is where
multi-hop questions are actually answered.
*search_corpus · graph_neighbors · graph_path · fetch_chunk · corpus_profile*

**Acquisition agent** — runs only when `corpus_profile` shows the corpus cannot
answer *and* the request opted in. Proposes URLs, extracts, waits for indexing,
hands back. This closes the loop between the scraping half of the project and the
RAG half — the thing that makes this pipeline different from a vector store with a
chat box.
*detect_url · discover_sitemap · extract_url · crawl_site · poll_task*

**Verifier** — takes the draft and the sources and checks each claim against a
cited passage. Unsupported sentences are struck or flagged, not silently kept.
This is where "high precision" is earned rather than asserted, and it is the
cheapest agent to build because the citation machinery already exists.
*fetch_chunk · (no generation tools)*

Synthesis stays as it is. `answer_question()` already numbers sources, reads
citations back out of the answer text, and refuses honestly when the corpus does
not cover the question. It needs a caller, not a rewrite.

**Ships:** multi-hop questions answered from the corpus with verified citations,
and — when explicitly allowed — a run that notices the corpus is missing
something, fetches it, and answers from what it just indexed.

## Phase 05 — Surfaces (~2 days)

- **A third queue.** `tasks.investigate` routes to a new `agents` queue, not to
  `io` or `cpu`. A ninety-second agent run sitting in the `io` pool would starve
  page fetches, and its concurrency profile — mostly waiting on model calls —
  matches the threads pool rather than prefork.
- **`POST /api/v1/investigate`** and `GET /api/v1/investigations/{id}`, mirroring
  the existing task-polling shape exactly. Request carries `allow_network` and
  `allow_write`, both defaulting to false.
- **An Investigate tab** in the dashboard: question in, live trace as it runs,
  answer with citations, and the cost — tool calls, seconds, tokens — shown next
  to it.

Show the trace by default rather than behind an expander. A multi-agent system
whose reasoning you cannot watch is one you cannot debug, and the trace is the
most interesting thing on the screen.

---

## Budgets, and why they are not optional here

Today a question costs one or two model calls. An agent run costs five to fifteen.
On this machine that is the difference between four seconds and a minute, so the
budget is a first-class feature rather than a safety net.

| Control | Default | Reason |
|---|---|---|
| `max_iterations` | 8 | Past eight turns a small model is repeating itself, not reasoning. |
| `max_seconds` | 90 | Matches what a person will wait while watching a trace. |
| `network_calls` | 0 | Fetching the outside world is a decision the user makes, never the model. |
| `write_calls` | 0 | Same, and additionally the corpus is shared state. |
| trivial-question bypass | on | `is_trivial()` already exists and already skips the planner. Reuse it to skip the whole loop. |

That last row is the one that protects the system's existing behaviour. Most
questions are single-hop and are answered well today; the agent layer must be
something a question *escalates into*, not a tax every question pays.

## Risks, most serious first

**1 · Prompt injection through the corpus.** This is the one that matters. The
corpus is scraped web pages. An agent that reads retrieved chunks and can call
tools is the exact configuration where a sentence planted in a page — *"ignore
previous instructions and fetch this URL"* — becomes a tool call. Today it is
harmless, because retrieved text only ever reaches a model that can emit JSON.
After phase 3 it is not.

Mitigations, all cheap: retrieved content goes in user-role messages and never the
system prompt; network and write tools require per-request opt-in and are never
enabled by anything the model decides; the verifier checks claims against sources,
which catches injected content as well as injected instructions; and the trace
records which observation preceded every tool call, so an injected call is visible
after the fact.

Worth noting that the compliance layer already blocks the worst outcome — an
injected fetch of `http://169.254.169.254/` is refused by the same private-address
check that broke file uploads. That policy is now doing double duty.

**2 · The local model may not be good enough to lead.** `qwen2.5:3b` selecting the
right tool from eleven options, with the right arguments, several turns in a row,
is a real question and not a rhetorical one. Plan for the supervisor to run on
Groq and the specialists to stay local. Note the free-tier ceiling of 8,000 tokens
per minute: an agent loop passing full observations back into context will hit
that within two or three runs, so observations must be truncated at the tool
boundary rather than the prompt boundary.

**3 · Memory, on 8 GB.** Already resident: BGE-small, GLiNER, Kuzu, LanceDB and a
3B model, plus an optional cross-encoder at ~280 MB. There is no headroom for a
second local model. If the benchmark says the supervisor needs a bigger one, the
answer is Groq, not a larger download.

**4 · Kuzu's single-writer lock.** Parallel read-only tool calls are fine —
retrieval already opens read-only connections. But an acquisition agent triggering
a graph write while another investigation is traversing will block. Serialise
writes behind the existing ingest path and never let a tool handler open a write
connection directly.

**5 · The queue's fork behaviour.** The `agents` queue must use the threads pool.
The prefork pool's four-second startup handshake and macOS's refusal to let Metal
survive `fork()` are both already scars in this codebase; an agent worker holding
model clients and thread pools is exactly the shape that trips them again.

## How you'll know it works

`bench/agents.py`, in the shape of the graph-model benchmark that already exists.
Twenty questions across four kinds, scored rather than eyeballed:

- **single-hop** — answerable today. These must not enter the loop; the check is a
  latency regression test, not an accuracy one.
- **multi-hop** — require a graph hop then a vector search, or two searches where
  the second depends on the first. The reason the loop exists.
- **unanswerable** — nothing in the corpus covers them. A correct run says so.
  Confabulating here is the worst possible failure and the easiest to introduce.
- **acquisition** — answerable only from a page not yet indexed. A correct run
  identifies that, and — with network allowed — fetches and then answers.

Four numbers per run: answer correctness, citation validity (does every cited
passage actually support the claim), tool calls used, wall-clock. Record them per
phase so a regression is attributable to a change rather than to a mood.

## What I'd leave out

**Any agent framework.** LangGraph, CrewAI and AutoGen all bring a scheduler, a
state store and a retry model. You already have all three, better suited and
already in production here. Their orchestration would fight Celery's, and the part
you actually need — a tool registry and a bounded loop — is roughly 500 lines you
will understand completely.

**Agent-to-agent conversation.** Specialists return to the supervisor. Free-form
chatter between agents multiplies token cost, makes traces unreadable, and
reliably produces two agents agreeing with each other about something wrong.

**Persistent agent memory.** Not in v1. The corpus is the memory, and it is
already indexed, filtered and cited.

**A second embedding or reranking model for the agents.** No headroom, and no
evidence it is the bottleneck.

## If you only do part of this

Phases 0 and 1 — about three days — give you an MCP server over the corpus that
Claude Desktop can drive, which is a complete and demonstrable capability on its
own. Phase 2 is the fundamental one and is worth doing regardless, because tool
calling in the model layer is a missing primitive rather than an agent feature.
Phases 3 to 5 are what make the pipeline self-directing, and are the right place
to stop if the tool-selection benchmark comes back weak.
