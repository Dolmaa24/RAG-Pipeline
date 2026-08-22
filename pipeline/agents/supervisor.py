"""The star: a supervisor, specialists that return to it, and a check at the end.

Phase 03 left a specific defect. Given "what did Acme acquire, and what was the
quarterly revenue", `llama3.2:3b` found the acquisition through the graph and
then stopped, answering half the question — while the revenue sat in an indexed
passage a single search away. The loop was bounded from above and not from
below: nothing ever asked *is this enough?*

That question is the whole of this phase, and the pipeline could already answer
it. ``answer_question()`` returns an answer, its sources, **and** a ``sufficient``
flag, in one call. So synthesis and assessment are the same step, and a round
that comes back insufficient sends the specialist out again with what it found
carried forward — which is what makes the second hop happen.

The topology is a star. Specialists never speak to each other; they return here.
Acquisition runs only when the corpus has been tried, has come up short, and the
caller budgeted for it — never because a model decided fetching would be helpful.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from config import config
from observability import get_logger, metrics

from pipeline.agents.budget import Budget, Spend
from pipeline.agents.loop import AgentLoop
from pipeline.agents.roles import ACQUISITION, CORPUS, Role, available
from pipeline.agents.verify import Verdict, verify

log = get_logger("agents.supervisor")

#: Names inside a rendered graph edge: "(Acme Corporation)-[ACQUIRED]->(Beta)".
#: Used to carry what one round discovered into the next round's search, which
#: is the mechanism that turns two single-hop rounds into one multi-hop answer.
_ENTITY = re.compile(r"\(([^)]{2,60})\)")

#: A URL in the question. Acquisition needs somewhere to go, and inferring one
#: is not something to leave to a model that has just failed to find an answer.
_URL = re.compile(r"https?://[^\s>)\]}\"']+")


@dataclass(slots=True)
class Investigation:
    """One finished investigation, and everything it took to get there."""

    question: str
    answer: str = ""
    sufficient: bool = False
    stopped: str = ""
    rounds: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sufficient": self.sufficient,
            "stopped": self.stopped,
            "rounds": self.rounds,
            "sources": self.sources,
            "trace": self.trace,
            "verification": self.verdict.to_dict() if self.verdict else None,
            "warnings": self.warnings,
            "seconds": round(self.seconds, 2),
        }


class _State(TypedDict, total=False):
    question: str
    focus: str
    leads: list[str]
    rounds: int
    answer: str
    sufficient: bool
    sources: list[dict[str, Any]]
    passages: list[str]
    trace: list[dict[str, Any]]
    acquired: list[str]


class Supervisor:
    """Runs specialists until the evidence answers the question, or the budget ends."""

    def __init__(
        self,
        *,
        budget: Optional[Budget] = None,
        local_only: bool = False,
        max_rounds: Optional[int] = None,
        verify_answer: Optional[bool] = None,
        backend=None,
        answerer=None,
    ) -> None:
        self.budget = budget or Budget.from_config()
        self.local_only = local_only
        #: Two by default, not eight. A round is a whole specialist loop plus a
        #: synthesis call, so three rounds is already a minute and a half on
        #: this hardware. The second round is where multi-hop happens; a third
        #: rarely adds evidence the second did not.
        self.max_rounds = config.AGENT_MAX_ROUNDS if max_rounds is None else max_rounds
        self.verify_answer = (
            config.AGENT_VERIFY if verify_answer is None else verify_answer
        )
        self._backend = backend
        self._answerer = answerer
        self._graph = self._build()

    # ------------------------------------------------------------------ #

    def _build(self):
        graph = StateGraph(_State)
        graph.add_node("gather", self._gather)
        graph.add_node("synthesise", self._synthesise)
        graph.add_node("acquire", self._acquire)

        graph.add_edge(START, "gather")
        graph.add_edge("gather", "synthesise")
        graph.add_conditional_edges(
            "synthesise",
            self._after_synthesis,
            {"gather": "gather", "acquire": "acquire", "end": END},
        )
        graph.add_edge("acquire", "gather")
        return graph.compile()

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def _gather(self, state: _State) -> _State:
        return self._run_specialist(state, CORPUS, state.get("focus") or state["question"])

    def _acquire(self, state: _State) -> _State:
        urls = _URL.findall(state["question"])
        already = set(state.get("acquired", []))
        fresh = [url for url in urls if url not in already]

        instruction = (
            f"Fetch and index this so it can be searched: {fresh[0]}"
            if fresh
            else f"Find and index a source that answers: {state['question']}"
        )
        updated = self._run_specialist(state, ACQUISITION, instruction)
        updated["acquired"] = [*already, *fresh]
        return updated

    def _run_specialist(self, state: _State, role: Role, task: str) -> _State:
        loop = AgentLoop(
            backend=self._backend,
            budget=self._round_budget(),
            local_only=self.local_only,
            system=role.system,
        )
        # The role decides which tools exist for this specialist. Fewer tools
        # per decision measured better, and a corpus specialist that cannot see
        # crawl_site cannot decide to crawl.
        loop.tools = lambda: role.catalog(self.budget.effects())  # type: ignore[method-assign]

        leads = state.get("leads") or []
        prompt = task if not leads else (
            f"{task}\n\nRelated things found so far, worth searching for: "
            f"{', '.join(leads[:6])}"
        )

        result = loop.run(prompt)
        self._spend.iterations += result.spend.iterations if result.spend else 0
        self._spend.tool_calls += result.spend.tool_calls if result.spend else 0
        self._warnings.extend(result.warnings)

        found = _leads_from(result.trace)
        return {
            **state,
            "rounds": state.get("rounds", 0) + 1,
            "leads": _merge(leads, found),
            "trace": [
                *state.get("trace", []),
                {
                    "kind": "specialist",
                    "role": role.name,
                    "task": prompt,
                    "stopped": result.stopped,
                    "tools": result.tools_used,
                    "steps": result.trace,
                },
            ],
        }

    def _synthesise(self, state: _State) -> _State:
        """Answer from the corpus, and learn whether that was possible.

        One call does both. ``answer_question`` retrieves, writes a grounded
        answer with numbered citations, and reports whether the evidence
        actually covered the question — which is the assessment this phase
        needed, already built and already tested.
        """
        question = state.get("focus") or state["question"]
        started = time.perf_counter()
        reply = self._answer(question)
        elapsed = time.perf_counter() - started

        sources = [source.to_dict() for source in reply.sources]
        return {
            **state,
            "answer": reply.answer,
            "sufficient": reply.sufficient,
            "sources": sources,
            "passages": [str(source.get("text", "")) for source in sources],
            "trace": [
                *state.get("trace", []),
                {
                    "kind": "synthesis",
                    "question": question,
                    "sufficient": reply.sufficient,
                    "cited": reply.cited,
                    "sources": len(sources),
                    "seconds": round(elapsed, 2),
                },
            ],
        }

    def _answer(self, question: str):
        if self._answerer is not None:
            return self._answerer(question)
        from pipeline.retrieve.answer import answer_question

        return answer_question(question, local_only=self.local_only)

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def _after_synthesis(self, state: _State) -> str:
        if state.get("sufficient"):
            self._stopped = "answered"
            return "end"

        if state.get("rounds", 0) >= self.max_rounds:
            self._stopped = f"reached the {self.max_rounds}-round limit"
            return "end"

        if self._spend.exhausted(self.budget):
            self._stopped = self._spend.exhausted(self.budget)
            return "end"

        if self._can_acquire(state):
            self._stopped = ""
            return "acquire"

        # Round two with what round one turned up. This is the hop the loop
        # alone never took: the leads carried forward are the names the graph
        # surfaced, which is exactly what the first search did not know to ask.
        self._stopped = ""
        return "gather"

    def _can_acquire(self, state: _State) -> bool:
        """Acquisition is a last resort, and only ever a permitted one."""
        if ACQUISITION not in available(self.budget):
            return False
        urls = set(_URL.findall(state["question"]))
        return bool(urls - set(state.get("acquired", [])))

    def _round_budget(self) -> Budget:
        """A slice of the whole, so one specialist cannot spend the run."""
        per_round = max(2, self.budget.max_iterations // self.max_rounds)
        return Budget(
            max_iterations=per_round,
            max_tool_calls=max(2, self.budget.max_tool_calls // self.max_rounds),
            max_seconds=self.budget.max_seconds,
            max_tokens=self.budget.max_tokens,
            network_calls=self.budget.network_calls,
            write_calls=self.budget.write_calls,
        )

    # ------------------------------------------------------------------ #

    def investigate(self, question: str) -> Investigation:
        question = (question or "").strip()
        if not question:
            return Investigation(question="", stopped="empty question")

        self._spend = Spend()
        self._warnings: list[str] = []
        self._stopped = ""
        started = time.perf_counter()

        final = self._graph.invoke(
            {
                "question": question,
                "focus": question,
                "leads": [],
                "rounds": 0,
                "trace": [],
                "acquired": [],
            },
            {"recursion_limit": self.max_rounds * 4 + 10},
        )

        verdict = None
        if self.verify_answer and final.get("answer"):
            verdict = verify(
                final["answer"],
                final.get("passages", []),
                local_only=self.local_only,
            )

        result = Investigation(
            question=question,
            answer=verdict.answer if verdict else final.get("answer", ""),
            sufficient=bool(final.get("sufficient")),
            stopped=self._stopped or "answered",
            rounds=final.get("rounds", 0),
            sources=final.get("sources", []),
            trace=final.get("trace", []),
            verdict=verdict,
            warnings=self._warnings,
            seconds=time.perf_counter() - started,
        )

        log.info(
            "agents.supervisor.done",
            question=question[:60],
            stopped=result.stopped,
            rounds=result.rounds,
            sufficient=result.sufficient,
            unsupported=len(verdict.unsupported) if verdict else 0,
            seconds=round(result.seconds, 1),
        )
        metrics.incr("agents.supervisor.runs")
        return result


def _leads_from(trace: list[dict[str, Any]]) -> list[str]:
    """Names a round turned up, to search for in the next one."""
    found: list[str] = []
    for step in trace:
        if step.get("kind") != "tool" or not step.get("ok"):
            continue
        found.extend(_ENTITY.findall(str(step.get("observation", ""))))
    return found


def _merge(existing: list[str], found: list[str]) -> list[str]:
    seen = {lead.lower() for lead in existing}
    merged = list(existing)
    for lead in found:
        cleaned = lead.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            merged.append(cleaned)
    return merged


def investigate(question: str, **kwargs) -> Investigation:
    return Supervisor(**kwargs).investigate(question)


__all__ = ["Investigation", "Supervisor", "investigate"]
