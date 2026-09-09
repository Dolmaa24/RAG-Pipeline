"""What a run is allowed to spend, and what it has spent.

Not a safety net. The tool-calling gate measured `qwen2.5:3b` stopping 0 times
out of 6 when no tool was needed — it called ``corpus_profile`` in reply to
"thanks, that's everything I needed", on every pass. Against a supervisor like
that, these limits are not a guard against occasional misbehaviour; they are the
only reason a run ends at all.

The counters double as the permission gate. ``network_calls`` and
``write_calls`` are both a ceiling and a grant: zero means the effect is not
merely unbudgeted but unavailable, and the tools carrying it are left out of the
catalog the model is shown. That keeps "may it?" and "how much?" from drifting
apart, which is what happens when they are separate settings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config import config

from pipeline.agents.tools import Effect


@dataclass(frozen=True, slots=True)
class Budget:
    """The ceiling for one run."""

    #: Past eight turns a small model is repeating itself, not reasoning. At the
    #: ~8s per call measured for llama3.2:3b, this is also what fits in
    #: max_seconds — so it binds first, which is the intent.
    max_iterations: int = 8
    max_tool_calls: int = 20
    #: What a person will wait while watching a trace.
    max_seconds: float = 90.0
    max_tokens: int = 60_000
    #: Both default to zero: reaching the outside world and changing the corpus
    #: are decisions the caller makes, never the model.
    network_calls: int = 0
    write_calls: int = 0
    #: Source files this run may write, and test runs it may start. Zero for
    #: both, like the two above: a retrieval run cannot reach either, and only
    #: a build request grants them.
    code_calls: int = 0
    execute_calls: int = 0

    @classmethod
    def from_config(cls) -> "Budget":
        return cls(
            max_iterations=config.AGENT_MAX_ITERATIONS,
            max_tool_calls=config.AGENT_MAX_TOOL_CALLS,
            max_seconds=config.AGENT_MAX_SECONDS,
            max_tokens=config.AGENT_MAX_TOKENS,
        )

    def effects(self) -> list[Effect]:
        """The effects this budget permits, in registry terms."""
        allowed = [Effect.READ]
        if self.network_calls > 0:
            allowed.append(Effect.NETWORK)
        if self.write_calls > 0:
            allowed.append(Effect.WRITE)
        if self.code_calls > 0:
            allowed.append(Effect.CODE)
        if self.execute_calls > 0:
            allowed.append(Effect.EXECUTE)
        return allowed


@dataclass(slots=True)
class Spend:
    """What one run has used. Mutable, and owned by the run."""

    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    network_calls: int = 0
    write_calls: int = 0
    code_calls: int = 0
    execute_calls: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.started

    def record_turn(self, tokens: int | None) -> None:
        self.iterations += 1
        self.tokens += tokens or 0

    def record_call(self, effect: Effect) -> None:
        self.tool_calls += 1
        if effect is Effect.NETWORK:
            self.network_calls += 1
        elif effect is Effect.WRITE:
            self.write_calls += 1
        elif effect is Effect.CODE:
            self.code_calls += 1
        elif effect is Effect.EXECUTE:
            self.execute_calls += 1

    def exhausted(self, budget: Budget) -> str:
        """Why the run must stop, or "" if it may continue.

        Returns a reason rather than a boolean because the reason is the useful
        half: "ran out of turns" and "ran out of time" call for different
        changes, and a trace that only records that it stopped tells you
        neither.
        """
        if self.iterations >= budget.max_iterations:
            return f"reached the {budget.max_iterations}-turn limit"
        if self.tool_calls >= budget.max_tool_calls:
            return f"reached the {budget.max_tool_calls}-tool-call limit"
        if self.seconds >= budget.max_seconds:
            return f"reached the {budget.max_seconds:.0f}s time limit"
        if self.tokens >= budget.max_tokens:
            return f"reached the {budget.max_tokens}-token limit"
        return ""

    def remaining(self, budget: Budget) -> dict[str, float]:
        """What is left, for the trace. A step without this is hard to read."""
        return {
            "turns": max(0, budget.max_iterations - self.iterations),
            "tool_calls": max(0, budget.max_tool_calls - self.tool_calls),
            "seconds": round(max(0.0, budget.max_seconds - self.seconds), 1),
            "tokens": max(0, budget.max_tokens - self.tokens),
        }


def would_exceed_effect_budget(spend: Spend, budget: Budget, effect: Effect) -> str:
    """Whether one more call with this effect is affordable.

    Checked before the call, not after: a write that has already happened
    cannot be un-budgeted.
    """
    if effect is Effect.NETWORK and spend.network_calls >= budget.network_calls:
        return f"no network calls left (limit {budget.network_calls})"
    if effect is Effect.WRITE and spend.write_calls >= budget.write_calls:
        return f"no write calls left (limit {budget.write_calls})"
    if effect is Effect.CODE and spend.code_calls >= budget.code_calls:
        return f"no source files left to write (limit {budget.code_calls})"
    if effect is Effect.EXECUTE and spend.execute_calls >= budget.execute_calls:
        return f"no test runs left (limit {budget.execute_calls})"
    return ""


__all__ = ["Budget", "Spend", "would_exceed_effect_budget"]
