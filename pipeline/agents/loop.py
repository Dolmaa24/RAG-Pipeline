"""The bounded loop: think, call, observe, repeat — until something stops it.

A LangGraph ``StateGraph`` with three nodes. *act* asks the model for a turn,
*observe* runs whatever it asked for, and *finish* gets an answer out of a run
that ran out of budget before it produced one. The graph is the easy part.
Everything interesting here is the machinery that ends a run, because the models
this has to work with do not end runs on their own.

**Tool execution stays on the registry**, not LangGraph's ``ToolNode``.
``invoke()`` already enforces effects, validates arguments and turns every
failure into an observation a model can read and act on; ``ToolNode`` does none
of that. Routing through it would also mean the MCP server and this loop no
longer share a code path, which is the single thing the registry exists to
prevent.

**No checkpointer.** LangGraph offers durability and so does Celery, and two
retry models over one job is how work gets done twice. Celery owns it.

**Why there is a *finish* node.** Budget exhaustion with nothing to show is a
wasted minute. The gate measured `qwen2.5:3b` never concluding it was done, so
for that model *every* run ends by exhaustion — without a final turn that asks
for an answer from what was gathered, the loop would reliably produce nothing at
all. One extra call, no tools offered, and the run has something to return.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from observability import get_logger, metrics

from pipeline.agents.budget import Budget, Spend, would_exceed_effect_budget
from pipeline.agents.tools import Effect, ToolCall, catalog, describe_all, invoke
from pipeline.extract.llm.base import Message, ToolTurn

log = get_logger("agents.loop")

SYSTEM = """\
You answer questions using the tools provided.

Call a tool when it would help. Read what it returns before deciding what to do
next. When you have enough to answer, answer — do not call another tool to
confirm what you already know.

If no tool fits what is being asked, say so plainly. Do not invent tools, and do
not use a tool for something it does not do.\
"""

#: Sent when the budget runs out. Phrased as an instruction rather than a
#: question so a model that has been calling tools for eight turns switches
#: mode instead of calling a ninth.
FINISH = """\
Stop searching and answer now, using only what is in this conversation. If what
was gathered does not answer the question, say exactly that and say what is
missing.\
"""


@dataclass(slots=True)
class LoopResult:
    """One finished run, with the reasoning that produced it."""

    question: str
    answer: str = ""
    #: Why the loop ended: "answered", "budget", "no progress", or a failure.
    stopped: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    spend: Optional[Spend] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "stopped": self.stopped,
            "trace": self.trace,
            "warnings": self.warnings,
            "cost": {
                "turns": self.spend.iterations if self.spend else 0,
                "tool_calls": self.spend.tool_calls if self.spend else 0,
                "tokens": self.spend.tokens if self.spend else 0,
                "seconds": round(self.spend.seconds, 2) if self.spend else 0.0,
            },
        }

    @property
    def tools_used(self) -> list[str]:
        return [step["tool"] for step in self.trace if step.get("kind") == "tool"]


class LoopState(TypedDict, total=False):
    messages: list[Message]
    trace: list[dict[str, Any]]
    answer: str
    stopped: str
    #: Hashes of observations already seen. Small models loop on the same
    #: failing search far more often than they hallucinate a tool.
    seen: set[str]


class AgentLoop:
    """One configured loop.

    Reusable across runs, but **one run at a time**: the spend and warnings for
    the run in flight live on the instance, because LangGraph state is merged
    between nodes and a mutable tally does not merge meaningfully. Celery gives
    each task its own loop, so this costs nothing there; sharing one instance
    across threads would silently mix two runs' budgets.
    """

    def __init__(
        self,
        *,
        backend=None,
        budget: Optional[Budget] = None,
        local_only: bool = False,
        system: str = SYSTEM,
    ) -> None:
        self.budget = budget or Budget.from_config()
        self.local_only = local_only
        self.system = system
        self._backend = backend
        self._graph = self._build()

    @property
    def backend(self):
        if self._backend is None:
            from pipeline.extract.llm import get_agent_backend

            self._backend = get_agent_backend(local_only=self.local_only)
        return self._backend

    def tools(self) -> list[dict]:
        """The catalog this run may use. Tools it may not are simply absent."""
        return describe_all(allowed=self.budget.effects())

    def _build(self):
        graph = StateGraph(LoopState)
        graph.add_node("act", self._act)
        graph.add_node("observe", self._observe)
        graph.add_node("finish", self._finish)

        graph.add_edge(START, "act")
        graph.add_conditional_edges(
            "act", self._after_act, {"observe": "observe", "finish": "finish", "end": END}
        )
        graph.add_conditional_edges(
            "observe", self._after_observe, {"act": "act", "finish": "finish"}
        )
        graph.add_edge("finish", END)
        # No checkpointer: Celery owns durability. See the module docstring.
        return graph.compile()

    def _act(self, state: LoopState) -> LoopState:
        turn = self._turn(state["messages"], self.tools())
        self._spend.record_turn(turn.completion_tokens)

        step = {
            "kind": "turn",
            "step": len(state["trace"]),
            "backend": turn.backend,
            "model": turn.model,
            "native": turn.native,
            "calls": [{"tool": c.name, "arguments": c.arguments} for c in turn.calls],
            "text": turn.text,
            "seconds": round(turn.latency_seconds, 2),
            "tokens": turn.completion_tokens,
            "budget_left": self._spend.remaining(self.budget),
        }

        return {
            **state,
            "messages": [*state["messages"], turn.as_message()],
            "trace": [*state["trace"], step],
            "answer": turn.text if not turn.wants_tools else state.get("answer", ""),
            "stopped": _stopped_after(turn, state),
        }

    def _observe(self, state: LoopState) -> LoopState:
        requests = state["messages"][-1].tool_calls
        results = self._run_tools(requests)

        messages = list(state["messages"])
        trace = list(state["trace"])
        seen = set(state.get("seen", set()))
        repeated = False

        for request, call in zip(requests, results):
            messages.append(Message.observation(request, call.observation))
            trace.append({**call.to_dict(), "kind": "tool", "step": len(trace)})

            digest = _digest(call.name, call.observation)
            if digest in seen:
                repeated = True
            seen.add(digest)

        return {
            **state,
            "messages": messages,
            "trace": trace,
            "seen": seen,
            "stopped": "no progress" if repeated else state.get("stopped", ""),
        }

    def _finish(self, state: LoopState) -> LoopState:
        """One last turn, with no tools offered, to get an answer out."""
        messages = [*state["messages"], Message.user(FINISH)]
        turn = self._turn(messages, tools=[])
        self._spend.record_turn(turn.completion_tokens)

        step = {
            "kind": "finish",
            "step": len(state["trace"]),
            "backend": turn.backend,
            "model": turn.model,
            "text": turn.text,
            "seconds": round(turn.latency_seconds, 2),
            "budget_left": self._spend.remaining(self.budget),
        }
        return {
            **state,
            "messages": [*messages, turn.as_message()],
            "trace": [*state["trace"], step],
            "answer": turn.text or _NOTHING_TO_SAY,
        }

    def _after_act(self, state: LoopState) -> str:
        stopped = state.get("stopped")
        if stopped == "answered":
            return "end"
        if stopped == "the model call failed":
            # End, and do not try to finish: the finishing turn goes to the
            # same backend that just failed. An earlier version fell through to
            # observe with an empty call list and looped there until the turn
            # budget ran out, reporting the wrong reason.
            self._stop_reason = stopped
            return "end"
        reason = self._spend.exhausted(self.budget)
        if reason:
            self._stop_reason = reason
            return "finish"
        return "observe"

    def _after_observe(self, state: LoopState) -> str:
        if state.get("stopped") == "no progress":
            self._stop_reason = "the same result came back twice"
            return "finish"
        reason = self._spend.exhausted(self.budget)
        if reason:
            self._stop_reason = reason
            return "finish"
        return "act"

    def _turn(self, messages: list[Message], tools: list[dict]) -> ToolTurn:
        try:
            return self.backend.complete_with_tools(messages=messages, tools=tools)
        except Exception as exc:
            # A model failure ends the run, but as a recorded outcome rather
            # than an exception thrown at whoever called run().
            log.error("agents.loop.turn_failed", error=repr(exc))
            metrics.incr("agents.loop.turn_failed")
            self._warnings.append(f"model call failed: {exc}")
            return ToolTurn(text="", finish_reason="error")

    def _run_tools(self, requests) -> list[ToolCall]:
        """Run one turn's calls, in parallel when all of them only read.

        Anything that writes or reaches the network runs one at a time: those
        have side effects on shared state, and interleaving them turns a
        readable trace into a guess about ordering.
        """
        allowed = self.budget.effects()
        specs = {spec.name: spec for spec in catalog(allowed)}
        effects = [
            specs[request.name].effect if request.name in specs else Effect.READ
            for request in requests
        ]

        for request, effect in zip(requests, effects):
            over = would_exceed_effect_budget(self._spend, self.budget, effect)
            if over:
                # Refused as an observation, in the words of the actual reason.
                # Routing this through invoke() with an empty allowance would
                # report "effect not allowed", which is true but is not why.
                return [_refusal(request, effect, over) for request in requests]

        for effect in effects:
            self._spend.record_call(effect)

        if len(requests) > 1 and all(effect is Effect.READ for effect in effects):
            with ThreadPoolExecutor(max_workers=min(len(requests), 4)) as pool:
                return list(
                    pool.map(
                        lambda request: invoke(request.name, request.arguments, allowed=allowed),
                        requests,
                    )
                )

        return [invoke(request.name, request.arguments, allowed=allowed) for request in requests]

    def run(self, question: str) -> LoopResult:
        """Answer one question, spending no more than the budget allows."""
        question = (question or "").strip()
        if not question:
            return LoopResult(question="", stopped="empty question")

        self._spend = Spend()
        self._warnings: list[str] = []
        self._stop_reason = ""

        started = time.perf_counter()
        state: LoopState = {
            "messages": [Message.system(self.system), Message.user(question)],
            "trace": [],
            "answer": "",
            "stopped": "",
            "seen": set(),
        }

        # recursion_limit is LangGraph's own backstop, in node visits rather
        # than turns. Set above what the turn budget allows so ours is what
        # actually binds — hitting LangGraph's would raise instead of finishing.
        final = self._graph.invoke(
            state, {"recursion_limit": self.budget.max_iterations * 3 + 10}
        )

        result = LoopResult(
            question=question,
            answer=final.get("answer", ""),
            stopped=self._stop_reason or final.get("stopped") or "answered",
            trace=final.get("trace", []),
            spend=self._spend,
            warnings=self._warnings,
        )

        log.info(
            "agents.loop.done",
            question=question[:60],
            stopped=result.stopped,
            turns=self._spend.iterations,
            tool_calls=self._spend.tool_calls,
            seconds=round(time.perf_counter() - started, 1),
            tools=result.tools_used,
        )
        metrics.incr("agents.loop.runs")
        return result


#: When even the finishing turn produces nothing. Silence and an empty answer
#: look identical to a caller, and only one of them is honest.
_NOTHING_TO_SAY = (
    "The run ended without an answer. What was gathered is in the trace; if the "
    "answer should be in there, the question may need rewording."
)


def _stopped_after(turn: ToolTurn, state: LoopState) -> str:
    """Why the run would end after this turn, if it does.

    A model call that failed returns no calls and no text, which is the same
    shape as a clean answer and was briefly reported as one.
    """
    if turn.finish_reason == "error":
        return "the model call failed"
    if turn.wants_tools:
        return ""
    return "answered"


def _refusal(request, effect: Effect, reason: str) -> ToolCall:
    return ToolCall(
        name=request.name,
        arguments=request.arguments,
        ok=False,
        observation=f"Refused: {reason}.",
        duration_ms=0.0,
        effect=effect,
        error=reason,
    )


def _digest(tool: str, observation: str) -> str:
    return hashlib.sha256(f"{tool}\x00{observation}".encode("utf-8")).hexdigest()


def run(question: str, **kwargs) -> LoopResult:
    """Answer one question with a fresh loop."""
    return AgentLoop(**kwargs).run(question)


__all__ = ["AgentLoop", "LoopResult", "LoopState", "run"]
