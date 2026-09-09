# Skills

A skill is one domain, described in one file. It gives the agent a prompt
written for that domain, a smaller set of tools, the fields worth extracting
from that domain's pages, and the entity and relation types its knowledge graph
should use.

Nothing here is code. A skill is read, never executed.

## Adding one

Make a folder and put a `SKILL.md` in it:

```
skills/
  logistics/
    SKILL.md
```

```markdown
---
name: logistics
description: >
  Shipments, carriers, tracking, customs and delivery times.
triggers: [shipment, consignment, carrier, tracking, freight, customs, bill of lading]
tools: [corpus_profile, search_corpus, answer_from_corpus, graph_relations, fetch_chunk]
requires: read
extraction:
  schema:
    consignment_id: string
    carrier: string
    origin: string
    destination: string
    status: string
  entity_types: [Shipment, Carrier, Port, Customer]
  relation_types: [SHIPPED_BY, ARRIVED_AT, CONSIGNED_TO]
---

You answer questions about shipments from an indexed corpus, using only what
the tools return...
```

Everything above the second `---` is settings. Everything below it becomes the
agent's system prompt, verbatim.

Then check it loaded, and that it routes:

```bash
curl -s "http://127.0.0.1:8000/api/v1/skills" | python -m json.tool
```

```bash
curl -s "http://127.0.0.1:8000/api/v1/skills/match?intent=where+is+my+consignment" | python -m json.tool
```

The API picks up an edited file on its own. **Celery workers do not** — restart
them, or a queued task will run the old prompt.

## The fields

| Field | Required | What it does |
|---|---|---|
| `name` | yes | Lowercase, URL-safe. Used as an API argument. |
| `description` | yes | Compared against an intent when no trigger word fires, so write the domain's vocabulary rather than a sentence about the file. |
| `triggers` | no | Words and phrases that mean this domain outright. This is what does the routing in practice. |
| `tools` | yes | Names from the tool registry. An unknown name fails at load. |
| `requires` | no | `read` (default), `network`, or `write`. See below. |
| `extraction.schema` | no | Field names to types. Fills `schema_template` for an extraction that names this skill. |
| `extraction.entity_types` | no | Added to the graph extractor's prompt for this domain. |
| `extraction.relation_types` | no | The same, for edges. |

## Writing triggers

Triggers are matched on word boundaries, with plurals folded — `policy` matches
"policies", and `order` does **not** match "reordered". A phrase counts double a
single word, so `sum insured` outranks a stray `school`.

Prefer words that mean nothing else. `claim` appears in ordinary speech about
anything; `sum insured` does not.

An intent that fires no trigger falls through to a vector comparison against the
descriptions, and that comparison is deliberately hard to pass — it must lead
the runner-up by a clear margin, or no skill matches at all. That is not a bug.
An unmatched intent runs the generic corpus specialist, which is what every
question got before skills existed and answers most of them perfectly well. A
*wrong* skill is the expensive outcome, because it hands the agent a prompt
about the wrong domain.

## What a skill cannot do

**A skill cannot grant itself a permission.** The tools it names are intersected
with what the request budgeted, so a skill listing `crawl_site` gets it stripped
on a read-only run, and a skill declaring `requires: write` is simply absent
from one. Effects are granted by the request — `allow_network`, `allow_write` —
and by nothing else. This is the same rule the tool registry and the MCP server
already follow, and it is covered by tests in `tests/test_skills.py` that were
checked against a version without the intersection.

**A skill cannot run code.** The frontmatter goes through `yaml.safe_load` and
the body is prompt text. Neither is imported, evaluated or executed.

**A skill file is trusted like source code, because it is read like source
code.** Its body reaches a model's system prompt. Load skills only from this
folder — never from an upload, a URL, or anything a request supplies. Review a
skill file the way you would review `pipeline/agents/roles.py`.

## Drafting one instead of writing it

An intent that matches nothing can be turned into a draft:

```bash
curl -s -X POST localhost:8000/api/v1/skills/draft \
  -H 'Content-Type: application/json' \
  -d '{"intent":"make a Baristo system"}' | python -m json.tool
```

A model proposes the domain — its vocabulary, its extraction fields, its entity
types, its specialist's prompt and its build roster — and the result is written
to `skills/_drafts/<name>.md`, **which the loader ignores**. Read it, edit it,
and install it:

```bash
curl -s -X POST localhost:8000/api/v1/skills/drafts/barista/approve \
  -H 'Content-Type: application/json' -d '{}'
```

The Build tab in the dashboard does the same thing with the file in an editable
box, which is the easier way to actually read it.

**Read the body before approving.** It becomes the agent's system prompt
verbatim. Approving unread is handing a model authorship of its own
instructions, and there is no reviewing that afterwards, because by then it has
run. Two things a draft can never do, enforced in code rather than asked for in
the prompt: `requires` is forced to `read`, and every tool name is checked
against the registry, so an invented one fails at generation with the name in
the message.

## Building the domain

A skill whose frontmatter declares an `agents:` block can be built. The workers
are the domain's own actors, not software job titles:

```yaml
agents:
  - name: receptionist
    purpose: Takes orders, validates them, manages the queue.
    writes: [orders.py]
  - name: machine
    purpose: Drives the espresso machine and tracks brew state.
    writes: [machine.py]
  - name: cleaner
    purpose: Background sweep clearing stale orders.
    writes: [cleanup.py]
```

```bash
curl -s -X POST localhost:8000/api/v1/build \
  -H 'Content-Type: application/json' -d '{"skill":"barista"}'
```

A **contract step runs first** and writes `models.py` and a `README.md` naming
the interfaces; every worker after it is handed that file and told to use those
names. Without it, each worker is an independent generation capped at 4096
output tokens, and four of those invent four incompatible ideas of what an order
is — the build looks finished and none of the modules import each other.

Order in the block is the running order, and a later worker is shown what the
earlier ones wrote. Files land in `workspace/<skill>/`, which is gitignored.

**Writing and running are separate permissions.** A build produces a scaffold
for you to read by default; `"allow_execute": true` additionally runs the
generated tests. The sandbox drops the environment (so nothing model-written is
handed `GROQ_API_KEY`), runs from inside the project with `PYTHONPATH` unset (so
generated code cannot `import pipeline` and reach the corpus), and bounds CPU,
memory and wall clock. **That is mitigation, not isolation** — it is a
subprocess, not a container.

Expect a plausible scaffold, not a working application.

A build uses two models. The **contract** step goes to the hosted one — it is a
single call and everything else is written against it — while the **workers**
run locally, because they are one call per file and are what exhaust a
tokens-per-minute allowance. Measured on Groq's free tier, putting the whole
build there left every step waiting 25 seconds for one call, and a four-worker
build produced three modules and no tests. Each step reports which model wrote
it.

## One skill at a time

An intent activates one skill, not several. That is deliberate: the tool-calling
benchmark measured a small model at 92% across twelve tools, and every tool a
specialist does not need is one more plausible wrong answer. Merging two skills
would produce a larger tool set and a longer prompt, which is the direction that
measured worse. Near-misses are reported as `runners_up` so a wrong match is
visible, and `skill` on the request is how you correct one.
