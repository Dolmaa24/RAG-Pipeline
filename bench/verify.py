"""Is the verifier better than not having one?

A check that flags everything and a check that flags nothing are both useless,
and they fail in opposite directions, so a single accuracy number hides which
one you have. This scores the two separately:

* **catch rate** — of the sentences the passages genuinely do not support, how
  many were flagged. Missing these is the failure that matters: an unsupported
  claim passing as verified is worse than no verification at all, because it
  carries a badge.
* **false alarms** — of the sentences the passages plainly do support, how many
  were flagged anyway. Every one of these teaches a reader to ignore the flag,
  which costs the catch rate its value.

Cases are whole answers, not lone sentences, because that is how ``verify`` is
called — and the difference is not cosmetic. Scored one sentence at a time,
llama3.2:3b vouched for all fourteen unsupported claims including "Beta
Industries acquired Acme Corporation"; given the same claims mixed into
multi-sentence answers it has something to discriminate against. A benchmark
that tests a component in a mode it is never used in produces a confident number
about nothing.

The claims are deliberately close to the line: verbatim support, support by
paraphrase, a real claim with one invented number, a reversed relationship.
Anything easier would flatter whatever is measured.

Run:  PYTHONPATH=. ./venv/bin/python -m bench.verify
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from pipeline.agents.verify import verify

PASSAGES = [
    "Quarterly Report. Acme Corporation acquired Beta Industries in March 2026. "
    "Revenue for the quarter was 42.5 million dollars.",
    "Beta Industries is a manufacturer of industrial sensors, founded in 2011 "
    "and based in Leeds.",
]


#: (sentence, is_unsupported). Named so a failure points at the claim.
CLAIMS: dict[str, tuple[str, bool]] = {
    "verbatim_acquisition": ("Acme Corporation acquired Beta Industries in March 2026.", False),
    "verbatim_revenue": ("Revenue for the quarter was 42.5 million dollars.", False),
    "reworded_revenue": ("The quarterly revenue was 42.5 million dollars.", False),
    "paraphrased": ("In March 2026, Beta Industries was bought by Acme Corporation.", False),
    "sensor_maker": ("Beta Industries makes industrial sensors.", False),
    "founded_year": ("Beta Industries was founded in 2011.", False),
    "wrong_number": ("Revenue for the quarter was 62.5 million dollars.", True),
    "wrong_month": ("Acme Corporation acquired Beta Industries in July 2026.", True),
    "invented_person": ("Acme Corporation was founded by Jane Reyes in 1998.", True),
    "invented_place": ("Acme Corporation is headquartered in Berlin.", True),
    "invented_event": ("The chief executive of Beta Industries resigned after the deal.", True),
    "invented_cause": ("Acme acquired Beta Industries to enter the automotive market.", True),
    "reversed": ("Beta Industries acquired Acme Corporation in March 2026.", True),
}


@dataclass
class Case:
    """One whole answer, as verify() is actually given it."""

    name: str
    claims: tuple[str, ...]


CASES: list[Case] = [
    # Nothing wrong. Every flag here is a false alarm.
    Case("clean_short", ("verbatim_acquisition", "verbatim_revenue")),
    Case("clean_reworded", ("paraphrased", "reworded_revenue")),
    Case("clean_mixed", ("verbatim_acquisition", "sensor_maker", "founded_year")),
    # One bad claim among good ones: the realistic failure.
    Case("one_wrong_number", ("verbatim_acquisition", "wrong_number")),
    Case("one_invented_person", ("verbatim_acquisition", "invented_person")),
    Case("one_invented_place", ("verbatim_revenue", "invented_place")),
    Case("one_invented_cause", ("verbatim_acquisition", "invented_cause")),
    Case("one_reversed", ("verbatim_revenue", "reversed")),
    Case("one_wrong_month", ("sensor_maker", "wrong_month")),
    # Mostly wrong.
    Case("mostly_invented", ("invented_place", "invented_event", "verbatim_revenue")),
    Case("all_invented", ("invented_person", "invented_place", "invented_event")),
]


@dataclass
class Score:
    model: str
    caught: int = 0
    to_catch: int = 0
    false_alarms: int = 0
    supported: int = 0
    seconds: float = 0.0
    misses: list[str] = field(default_factory=list)
    alarms: list[str] = field(default_factory=list)


def run(model: str | None, repeat: int) -> Score:
    from pipeline.extract.llm.ollama import OllamaBackend

    backend = OllamaBackend(model=model) if model else None
    score = Score(model=model or "default")

    for case in CASES * repeat:
        sentences = [CLAIMS[name][0] for name in case.claims]
        verdict = verify(" ".join(sentences), PASSAGES, backend=backend)
        score.seconds += verdict.seconds
        flagged = set(verdict.unsupported)

        for name in case.claims:
            sentence, unsupported = CLAIMS[name]
            was_flagged = sentence in flagged
            if unsupported:
                score.to_catch += 1
                if was_flagged:
                    score.caught += 1
                else:
                    score.misses.append(name)
            else:
                score.supported += 1
                if was_flagged:
                    score.false_alarms += 1
                    score.alarms.append(name)

    return score


def report(scores: list[Score], repeat: int) -> None:
    print()
    print("=" * 72)
    print(f"{'model':<16} {'catches':>10} {'false alarms':>15} {'avg s':>8}")
    print("-" * 72)
    for s in scores:
        catch = f"{s.caught}/{s.to_catch}" if s.to_catch else "n/a"
        alarm = f"{s.false_alarms}/{s.supported}" if s.supported else "n/a"
        per = s.seconds / max(len(CASES) * repeat, 1)
        print(f"{s.model:<16} {catch:>10} {alarm:>15} {per:>8.1f}")
    print("=" * 72)

    for s in scores:
        if s.misses:
            print(f"\n{s.model} — let through: {', '.join(sorted(set(s.misses)))}")
        if s.alarms:
            print(f"{s.model} — flagged wrongly: {', '.join(sorted(set(s.alarms)))}")

    print()
    print("A verifier is worth having when it catches most of what it should and")
    print("rarely cries wolf. Flagging is an annotation, not a deletion, so a")
    print("miss costs more than an alarm — but alarms are what teach a reader to")
    print("stop reading the flags, which costs the catches their value.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen2.5:3b,llama3.2:3b")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    scores = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n--- {model} ---")
        scores.append(run(model, args.repeat))
    report(scores, args.repeat)


if __name__ == "__main__":
    main()
