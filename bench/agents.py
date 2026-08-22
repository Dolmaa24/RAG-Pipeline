"""Does an investigation beat asking once?

Every number this project has measured so far is about a component — which tool
a model picks, whether a claim survives checking, how long a graph query takes.
None of them answers the question the whole design rests on: **is the agent loop
worth three times the latency of `/api/v1/answer`?**

So both paths answer the same questions and are scored the same way.

Four kinds, because they fail differently and a single average hides that:

* **single-hop** — answerable by one retrieval. These must not get *worse*, and
  the interesting number is the latency they pay for nothing.
* **multi-hop** — need a graph hop then a search, or two searches where the
  second depends on the first. The reason the loop exists.
* **enumeration** — "which" and "how many" questions, which name no entity to
  start from. A whole class that single-shot retrieval answers partially.
* **unanswerable** — nothing in the corpus covers them. A correct run says so.
  Confabulating here is the worst failure and the easiest to introduce, and it
  is scored separately because getting it right by refusing everything is not
  the same as getting it right.

Scoring is substring matching against facts written down in advance, which is
crude and honest: it cannot reward a good answer phrased unusually, and it
cannot be talked into accepting a wrong one. A question is answered when every
one of its required facts appears.

The corpus this expects is the two acquisition documents used throughout the
plan. Run bench/fixtures.py first, or point it at your own corpus and rewrite
CASES — the scoring makes no assumptions beyond the strings.

Run:  PYTHONPATH=. ./venv/bin/python -m bench.agents
      PYTHONPATH=. ./venv/bin/python -m bench.agents --only multi_hop
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Case:
    name: str
    kind: str
    question: str
    #: Every one must appear for the answer to count. Matched case-insensitively.
    must_contain: tuple[str, ...] = ()
    #: True when the corpus cannot answer it and saying so is the correct result.
    unanswerable: bool = False


CASES: list[Case] = [
    # -- single-hop: one retrieval is enough ------------------------------- #
    Case("acme_target", "single_hop",
         "What did Acme Corporation acquire?", ("beta industries",)),
    Case("northwind_price", "single_hop",
         "How much did Northwind Logistics pay for Fabrikam Freight?", ("310",)),
    Case("fabrikam_base", "single_hop",
         "Where is Fabrikam Freight headquartered?", ("rotterdam",)),
    Case("quarterly_revenue", "single_hop",
         "What was the quarterly revenue in the Acme report?", ("42.5",)),

    # -- multi-hop: the second step depends on the first -------------------- #
    Case("acme_two_part", "multi_hop",
         "What did Acme Corporation acquire, and what was the quarterly revenue?",
         ("beta industries", "42.5")),
    Case("northwind_three_part", "multi_hop",
         "What did Northwind Logistics acquire, for how much, and who leads the "
         "combined group?",
         ("fabrikam", "310", "priya raman")),
    Case("target_location", "multi_hop",
         "Acme acquired a company — where is the company Northwind acquired based?",
         ("rotterdam",)),

    # -- enumeration: no entity to start from ------------------------------- #
    Case("which_acquisitions", "enumeration",
         "Which acquisitions does the corpus describe?",
         ("beta industries", "fabrikam")),
    Case("two_acquisitions_amounts", "enumeration",
         "Which two acquisitions are described, and what were the amounts?",
         ("beta industries", "fabrikam")),
    Case("who_acquired_whom", "enumeration",
         "Which companies acquired which other companies?",
         ("acme", "northwind")),

    # -- unanswerable: refusing is the right answer ------------------------- #
    Case("capital_mongolia", "unanswerable",
         "What is the capital of Mongolia?", unanswerable=True),
    Case("acme_founder", "unanswerable",
         "Who founded Acme Corporation, and in what year?", unanswerable=True),
    Case("contoso", "unanswerable",
         "What did Contoso Shipping acquire?", unanswerable=True),
]


# --------------------------------------------------------------------------- #
# Running each path
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    answer: str = ""
    sufficient: bool = False
    seconds: float = 0.0
    tool_calls: int = 0
    sources: int = 0
    flagged: int = 0
    error: str = ""


def ask_once(case: Case) -> Run:
    """The existing single-shot path: retrieve, then answer."""
    from pipeline.retrieve.answer import answer_question

    started = time.perf_counter()
    try:
        reply = answer_question(case.question)
    except Exception as exc:
        return Run(error=repr(exc)[:120], seconds=time.perf_counter() - started)

    return Run(
        answer=reply.answer,
        sufficient=reply.sufficient,
        seconds=time.perf_counter() - started,
        sources=len(reply.sources),
    )


def investigate(case: Case, *, rounds: int, verify: bool) -> Run:
    """The agent loop: gather, judge whether that was enough, repeat."""
    from pipeline.agents.supervisor import Supervisor

    started = time.perf_counter()
    try:
        result = Supervisor(max_rounds=rounds, verify_answer=verify).investigate(
            case.question
        )
    except Exception as exc:
        return Run(error=repr(exc)[:120], seconds=time.perf_counter() - started)

    tool_calls = sum(
        1
        for step in result.trace
        if step.get("kind") == "specialist"
        for inner in step.get("steps", [])
        if inner.get("kind") == "tool"
    )
    return Run(
        answer=result.answer,
        sufficient=result.sufficient,
        seconds=result.seconds,
        tool_calls=tool_calls,
        sources=len(result.sources),
        flagged=len(result.verdict.unsupported) if result.verdict else 0,
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def graded(case: Case, run: Run) -> str:
    """One of: right, partial, wrong, confabulated, error.

    ``partial`` exists because a two-part question answered halfway is the exact
    failure this whole design was built to fix, and scoring it as simply wrong
    would hide whether that is getting better.
    """
    if run.error:
        return "error"

    text = (run.answer or "").lower()

    if case.unanswerable:
        # Correct means declining. Saying so in the answer *or* reporting
        # insufficiency both count: an honest refusal is an honest refusal.
        declined = not run.sufficient or any(
            phrase in text
            for phrase in ("not contain", "does not", "no information", "cannot",
                           "not covered", "not mentioned", "unable")
        )
        return "right" if declined else "confabulated"

    if not case.must_contain:
        return "right" if text.strip() else "wrong"

    hits = sum(1 for fact in case.must_contain if fact.lower() in text)
    if hits == len(case.must_contain):
        return "right"
    return "partial" if hits else "wrong"


@dataclass
class Tally:
    label: str
    right: int = 0
    partial: int = 0
    wrong: int = 0
    confabulated: int = 0
    errors: int = 0
    seconds: float = 0.0
    tool_calls: int = 0
    flagged: int = 0
    total: int = 0
    detail: list[tuple[str, str]] = field(default_factory=list)

    def add(self, case: Case, run: Run, grade: str) -> None:
        self.total += 1
        self.seconds += run.seconds
        self.tool_calls += run.tool_calls
        self.flagged += run.flagged
        setattr(self, {"right": "right", "partial": "partial", "wrong": "wrong",
                       "confabulated": "confabulated", "error": "errors"}[grade],
                getattr(self, {"right": "right", "partial": "partial", "wrong": "wrong",
                               "confabulated": "confabulated", "error": "errors"}[grade]) + 1)
        self.detail.append((case.name, grade))


def report(kinds: list[str], tallies: dict[tuple[str, str], Tally]) -> None:
    print()
    print("=" * 84)
    print(f"{'kind':<14} {'path':<14} {'right':>6} {'partial':>8} {'wrong':>6} "
          f"{'confab':>7} {'avg s':>7} {'tools':>6}")
    print("-" * 84)
    for kind in kinds:
        for path in ("answer", "investigate"):
            t = tallies.get((kind, path))
            if not t or not t.total:
                continue
            print(
                f"{kind if path == 'answer' else '':<14} {path:<14} "
                f"{t.right:>3}/{t.total:<2} {t.partial:>8} {t.wrong:>6} "
                f"{t.confabulated:>7} {t.seconds / t.total:>7.1f} "
                f"{t.tool_calls:>6}"
            )
        print("-" * 84)

    overall = {}
    for (kind, path), t in tallies.items():
        agg = overall.setdefault(path, Tally(label=path))
        for field_name in ("right", "partial", "wrong", "confabulated", "errors", "total"):
            setattr(agg, field_name, getattr(agg, field_name) + getattr(t, field_name))
        agg.seconds += t.seconds
        agg.tool_calls += t.tool_calls
        agg.flagged += t.flagged

    for path, t in overall.items():
        if not t.total:
            continue
        print(
            f"{'ALL':<14} {path:<14} {t.right:>3}/{t.total:<2} {t.partial:>8} "
            f"{t.wrong:>6} {t.confabulated:>7} {t.seconds / t.total:>7.1f} "
            f"{t.tool_calls:>6}"
        )
    print("=" * 84)

    for (kind, path), t in sorted(tallies.items()):
        bad = [name for name, grade in t.detail if grade != "right"]
        if bad:
            print(f"{kind}/{path} not right: {', '.join(bad)}")

    print()
    print("The loop earns its latency on multi_hop and enumeration or it earns")
    print("nothing. On single_hop the number to watch is seconds, not accuracy:")
    print("a question answered well today must not become slow to answer.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="one kind, e.g. multi_hop")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--paths", default="answer,investigate",
        help="which to run; useful for re-measuring one of them",
    )
    args = parser.parse_args()

    cases = [c for c in CASES if not args.only or c.kind == args.only]
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    tallies: dict[tuple[str, str], Tally] = {}

    for case in cases:
        for path in paths:
            run = (
                ask_once(case)
                if path == "answer"
                else investigate(case, rounds=args.rounds, verify=not args.no_verify)
            )
            grade = graded(case, run)
            tallies.setdefault((case.kind, path), Tally(label=path)).add(case, run, grade)
            print(
                f"  {case.kind:<12} {path:<12} {case.name:<24} {grade:<13} "
                f"{run.seconds:>5.1f}s"
            )

    kinds: list[str] = []
    for case in cases:
        if case.kind not in kinds:
            kinds.append(case.kind)
    report(kinds, tallies)


if __name__ == "__main__":
    main()
