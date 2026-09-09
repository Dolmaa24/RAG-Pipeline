"""Can a model this small actually drive the tool catalog?

The whole agent layer rests on one assumption that had never been measured in
this repo: that a model small enough to sit in 8 GB alongside BGE, GLiNER, Kuzu
and LanceDB can pick the right tool, with the right arguments, several turns
running. If it cannot, the supervisor belongs on Groq and a good deal of the
agent work is simply not reachable offline.

Six checks, each one a way the loop fails in practice rather than in theory:

1. **Well-formed** — a tool call the API returns as structured data, not prose
   describing one. A model that writes "I would call search_corpus" has not
   called anything, and the loop stalls.
2. **Right tool** — for questions with one defensible answer. Eleven tools is
   more than a 3B model is comfortable with, and the failure is quiet: it picks
   a plausible neighbour and returns confident nonsense.
3. **Right arguments** — required fields present, values taken from the question
   rather than invented. Flat arguments were chosen in phase 00 precisely
   because small models mishandle nested objects; this is where that pays off
   or does not.
4. **No fabrication** — inventing a tool that does not exist. Asking for
   something the catalog cannot do is the honest test, because the model has to
   decline rather than confabulate a `delete_document`.
5. **Knows to stop** — a remark that needs no tool must not trigger one. An
   agent that searches the corpus when thanked burns a turn every time.
6. **Multi-step** — given the result of a first call, does the second follow
   from it? This is the one the whole loop depends on and the one small models
   are worst at.

The schemas come from the phase-00 registry via ``describe_all()``, not from a
copy written here. A benchmark scoring a hand-maintained duplicate would drift
from the catalog it claims to measure, and would then be worse than no
benchmark at all.

Run:  PYTHONPATH=. ./venv/bin/python -m bench.tool_calling
      PYTHONPATH=. ./venv/bin/python -m bench.tool_calling --models qwen2.5:3b,llama3.2:3b
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from config import config
from pipeline.agents.tools import Effect, describe_all

# The loop's real system prompt, near enough. Deliberately short: every token
# here is resent on every turn, and the Groq free tier is 8,000 per minute.
SYSTEM = (
    "You answer questions using the tools provided. Call a tool when it would "
    "help. If no tool fits, say so plainly instead of guessing. Do not invent "
    "tools that are not listed."
)


@dataclass
class Case:
    name: str
    kind: str
    prompt: str
    #: None means "no tool call is the correct behaviour".
    expect: Optional[str]
    #: Extra credit, checked only when the tool was right.
    args_ok: Optional[Callable[[dict[str, Any]], bool]] = None
    #: Turns to replay before the prompt, for multi-step cases.
    history: list[dict[str, Any]] = field(default_factory=list)


def _has(args: dict[str, Any], key: str, *needles: str) -> bool:
    value = str(args.get(key, "") or "").lower()
    return any(needle in value for needle in needles)


CASES: list[Case] = [
    Case(
        "search_plain", "single",
        "What do the indexed documents say about quarterly revenue?",
        "search_corpus",
        lambda a: _has(a, "query", "revenue", "quarterly"),
    ),
    Case(
        "profile", "single",
        "What is actually in the knowledge base? Which departments and "
        "document types are available to filter on?",
        "corpus_profile",
    ),
    Case(
        "graph_neighbors", "single",
        "Which entities is Acme Corporation directly connected to in the "
        "knowledge graph?",
        "graph_neighbors",
        lambda a: _has(a, "entity", "acme"),
    ),
    Case(
        "graph_path", "single",
        "Is there any connection in the graph between Acme Corporation and "
        "Beta Industries? Show the chain linking them.",
        "graph_path",
        lambda a: _has(a, "start", "acme", "beta") and _has(a, "end", "acme", "beta"),
    ),
    Case(
        "cited_answer", "single",
        "Answer with citations: who acquired Beta Industries, and when?",
        "answer_from_corpus",
        lambda a: _has(a, "question", "beta", "acquir"),
    ),
    Case(
        "detect", "single",
        "I have this link: https://example.com/feed.xml — what kind of resource "
        "is it? Do not download or extract it, just tell me what it is.",
        "detect_url",
        lambda a: _has(a, "url", "example.com"),
    ),
    Case(
        "extract", "single",
        "Please extract https://example.com/annual-report and add it to the "
        "corpus so I can search it.",
        "extract_url",
        lambda a: _has(a, "url", "annual-report"),
    ),
    Case(
        "poll", "single",
        "Is task 3f9a1c2e-4b5d-6789-abcd-ef0123456789 finished yet?",
        "poll_task",
        lambda a: _has(a, "task_id", "3f9a1c2e"),
    ),

    Case(
        "filtered_search", "args",
        "Search only the finance department's documents, published in 2026 or "
        "later, for anything about budget overruns.",
        "search_corpus",
        lambda a: _has(a, "department", "finance")
        and (_has(a, "date_from", "2026") or _has(a, "query", "2026")),
    ),
    Case(
        "hops", "args",
        "Show me everything within two hops of Acme Corporation in the graph.",
        "graph_neighbors",
        lambda a: _has(a, "entity", "acme") and int(a.get("hops") or 1) == 2,
    ),

    Case(
        "gratitude", "stop",
        "Great, thanks — that's everything I needed.",
        None,
    ),
    Case(
        "chitchat", "stop",
        "Roughly how many tools do you have available to you?",
        None,
    ),

    Case(
        "no_such_tool", "trap",
        "Delete every document about Acme Corporation from the corpus "
        "permanently.",
        None,
    ),
    Case(
        "no_such_tool_2", "trap",
        "Email the quarterly summary to my manager.",
        None,
    ),

    Case(
        "after_profile", "multi",
        "Now search that engineering department for anything about latency.",
        "search_corpus",
        lambda a: _has(a, "department", "engineering")
        and _has(a, "query", "latency"),
        history=[
            {
                "role": "user",
                "content": "What departments exist in the corpus?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "corpus_profile",
                            "arguments": {},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "chunks": 412,
                        "departments": ["engineering", "finance", "legal"],
                        "doc_types": ["document", "html"],
                    }
                ),
            },
        ],
    ),
    Case(
        "after_neighbors", "multi",
        "Beta Industries looks relevant. Pull the passages about it so I can "
        "read what they say.",
        "search_corpus",
        lambda a: _has(a, "query", "beta"),
        history=[
            {"role": "user", "content": "Who is Acme connected to?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "graph_neighbors",
                            "arguments": {"entity": "Acme Corporation"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "triples": [
                            "Acme Corporation ACQUIRED Beta Industries",
                            "Acme Corporation EMPLOYS Jane Reyes",
                        ]
                    }
                ),
            },
        ],
    ),
]


def ollama_tools() -> list[dict[str, Any]]:
    """The registry's catalog, in the shape Ollama's /api/chat wants."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in describe_all(allowed=[Effect.READ, Effect.NETWORK, Effect.WRITE])
    ]


@dataclass
class Turn:
    calls: list[tuple[str, dict[str, Any]]]
    text: str
    seconds: float
    error: str = ""


def ask(model: str, case: Case, tools: list[dict[str, Any]], host: str) -> Turn:
    messages = [{"role": "system", "content": SYSTEM}, *case.history,
                {"role": "user", "content": case.prompt}]
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        # Deterministic: this measures capability, not sampling luck.
        "options": {"temperature": 0.0, "num_ctx": config.OLLAMA_NUM_CTX},
    }

    started = time.perf_counter()
    try:
        response = httpx.post(f"{host}/api/chat", json=body, timeout=180.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return Turn([], "", time.perf_counter() - started, error=repr(exc)[:120])

    elapsed = time.perf_counter() - started
    message = payload.get("message", {}) or {}
    calls = []
    for call in message.get("tool_calls", []) or []:
        function = call.get("function", {}) or {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__unparseable__": arguments}
        calls.append((function.get("name", ""), arguments or {}))

    return Turn(calls, (message.get("content") or "").strip(), elapsed)


#: Filled in by main(), so the report can print how many trials each column is
#: over. A bare "92%" over twelve cases invites more confidence than it earns.
_TOTAL_CALLS = 0
_TOTAL_STOPS = 0
_TOTAL_TRAPS = 0
_TOTAL_MULTI = 0


@dataclass
class Score:
    model: str
    well_formed: int = 0
    right_tool: int = 0
    right_args: int = 0
    args_applicable: int = 0
    fabricated: int = 0
    stopped_correctly: int = 0
    stop_cases: int = 0
    trap_ok: int = 0
    trap_cases: int = 0
    call_cases: int = 0
    multi_ok: int = 0
    multi_cases: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)


def run(model: str, host: str, verbose: bool, repeat: int = 1) -> Score:
    tools = ollama_tools()
    known = {tool["function"]["name"] for tool in tools}
    score = Score(model=model)

    for case in CASES * repeat:
        turn = ask(model, case, tools, host)
        score.seconds += turn.seconds

        if turn.error:
            score.failures.append(f"{case.name}: transport {turn.error}")
            continue

        called = turn.calls[0][0] if turn.calls else None
        args = turn.calls[0][1] if turn.calls else {}

        if turn.calls and any(name not in known for name, _ in turn.calls):
            score.fabricated += 1
            score.failures.append(
                f"{case.name}: invented {[n for n, _ in turn.calls if n not in known]}"
            )

        if case.expect is None:
            # Two different competences wear the same shape. Declining small
            # talk is about knowing a tool is not wanted; declining an
            # impossible request is about not pretending the catalog can do
            # something it cannot. A model can be good at one and bad at the
            # other, and averaging them hides which.
            if case.kind == "trap":
                score.trap_cases += 1
                if not turn.calls:
                    score.trap_ok += 1
                else:
                    score.failures.append(f"{case.name}: called {called}, expected none")
            else:
                score.stop_cases += 1
                if not turn.calls:
                    score.stopped_correctly += 1
                else:
                    score.failures.append(f"{case.name}: called {called}, expected none")
        else:
            score.call_cases += 1
            if case.kind == "multi":
                score.multi_cases += 1

            if turn.calls:
                score.well_formed += 1
            else:
                score.failures.append(f"{case.name}: no call, said {turn.text[:60]!r}")

            if called == case.expect:
                score.right_tool += 1
                args_ok = True
                if case.args_ok is not None:
                    score.args_applicable += 1
                    try:
                        args_ok = bool(case.args_ok(args))
                    except Exception:
                        args_ok = False
                    if args_ok:
                        score.right_args += 1
                    else:
                        score.failures.append(f"{case.name}: args {args}")
                # Both halves, or it did not follow the first result.
                if case.kind == "multi" and args_ok:
                    score.multi_ok += 1
            elif turn.calls:
                score.failures.append(
                    f"{case.name}: called {called}, expected {case.expect}"
                )

        if verbose:
            print(f"    {case.name:<18} -> {called or '(none)'}  {turn.seconds:.1f}s")

    return score


def report(scores: list[Score]) -> None:
    print()
    print("=" * 78)
    print(f"{'model':<16} {'well':>6} {'tool':>8} {'args':>8} {'stop':>7} "
          f"{'trap':>7} {'multi':>7} {'fabr':>6} {'avg s':>7}")
    print(f"{'':16} {'n=' + str(_TOTAL_CALLS):>6} {'':>8} {'':>8} "
          f"{'n=' + str(_TOTAL_STOPS):>7} {'n=' + str(_TOTAL_TRAPS):>7} "
          f"{'n=' + str(_TOTAL_MULTI):>7}")
    print("-" * 78)
    for s in scores:
        pct = lambda n, d: f"{100 * n / d:.0f}%" if d else "n/a"  # noqa: E731
        print(
            f"{s.model:<16} "
            f"{pct(s.well_formed, s.call_cases):>6} "
            f"{pct(s.right_tool, s.call_cases):>8} "
            f"{pct(s.right_args, s.args_applicable):>8} "
            f"{pct(s.stopped_correctly, s.stop_cases):>7} "
            f"{pct(s.trap_ok, s.trap_cases):>7} "
            f"{pct(s.multi_ok, s.multi_cases):>7} "
            f"{s.fabricated:>6} "
            f"{s.seconds / max(len(CASES), 1):>7.1f}"
        )
    print("=" * 78)

    for s in scores:
        if not s.failures:
            continue
        print(f"\n{s.model} — what went wrong:")
        for line in s.failures:
            print(f"  · {line}")

    print()
    print("Gate: right-tool above ~70% with clean multi-step means the")
    print("supervisor can run locally; 40-70% means local specialists with a")
    print("hosted supervisor; below 40% means the tool shim is the primary")
    print("path for local models rather than a fallback.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen2.5:3b,llama3.2:3b")
    parser.add_argument("--host", default=config.OLLAMA_HOST)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="passes over the case set; 3 gives a usable idea of variance",
    )
    args = parser.parse_args()

    global _TOTAL_CALLS, _TOTAL_STOPS, _TOTAL_TRAPS, _TOTAL_MULTI
    _TOTAL_CALLS = sum(1 for c in CASES if c.expect) * args.repeat
    _TOTAL_STOPS = sum(1 for c in CASES if c.kind == "stop") * args.repeat
    _TOTAL_TRAPS = sum(1 for c in CASES if c.kind == "trap") * args.repeat
    _TOTAL_MULTI = sum(1 for c in CASES if c.kind == "multi") * args.repeat

    host = args.host.rstrip("/")
    scores = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n--- {model} ---")
        scores.append(run(model, host, args.verbose, args.repeat))
    report(scores)


if __name__ == "__main__":
    main()
