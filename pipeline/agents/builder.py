"""Turning a domain's roster into a project on disk.

A separate thing from :class:`~pipeline.agents.supervisor.Supervisor`, which
exists to decide whether the evidence answers a question and loops until it
does. A build has no such judgement to make: the skill declares its workers, and
they run in the order they are declared. A fixed sequence, not a loop.

    contract  ->  receptionist  ->  machine  ->  delivery  ->  cleaner  ->  tests

**The contract step is why this produces something that imports itself.** Both
backends cap output at 4096 tokens, so each worker is its own generation, and
four independent generations invent four incompatible ideas of what an order is.
So one call goes first and writes the shared types and a README describing the
interfaces; every worker after it is handed that file verbatim and told to use
those names. Without it the predictable output is four modules that do not
compose, which is a worse failure than any of them being individually poor —
the build looks finished and is not.

**Each worker sees what came before.** The file tree so far, and the contract.
Not the full text of every earlier file: that grows with the roster and would
crowd out the instruction by the fourth worker, on an 8000 tokens-per-minute
budget. A worker that needs an earlier file reads it with ``read_source``.

Nothing here decides what it is allowed to do. The budget arrives carrying
``code_calls`` and ``execute_calls``, and a build whose caller granted no
execution simply has no test tool in its catalog.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

from pipeline.agents.budget import Budget, Spend
from pipeline.agents.loop import AgentLoop
from pipeline.agents.tools import Effect, catalog, describe_all
from pipeline.agents.tools import sandbox
from pipeline.agents.tools.build import ActiveBuild
from pipeline.skills.models import BuildAgent, Skill

log = get_logger("agents.builder")

CONTRACT_FILE = "models.py"
README_FILE = "README.md"

#: How much of the contract each worker is sent. Every worker call carries it,
#: and on Groq's free tier -- 8000 tokens per minute -- that is what decides
#: whether a build of four workers fits. Measured: the contract plus the README
#: came to roughly 1500 tokens per call, and a four-worker build spent the
#: whole minute's allowance before the second worker, so every step after it
#: waited. models.py is the half that matters, because it carries the names the
#: workers must agree on; the README is prose about them.
_CONTRACT_CHARS = 4000
_README_CHARS = 1200

_CONTRACT_SYSTEM = """\
You define the shared foundation of a small Python project, before anyone else
writes a line of it.

Write exactly two files with write_source:

1. models.py — the data types every other part of this project will pass around.
   Plain dataclasses and enums. No behaviour beyond validation, no I/O, no
   imports outside the standard library. This file must import nothing from the
   project, because everything else imports it.

2. README.md — one short paragraph on what the project is, then a section per
   worker listing the functions it must provide, with their exact signatures and
   what each returns. This is a contract the other workers are held to, so be
   specific: names, argument types, return types.

Keep models.py well under 200 lines. Cover what the workers listed below
actually need and nothing more. When both files are written, say so and stop.\
"""

_WORKER_SYSTEM = """\
You write one part of a Python project that other people are writing the rest
of. Your part is described below.

The shared types in models.py are already written and are shown to you. Use
those names exactly. Do not redefine them, do not rename them, and do not
invent a different shape for them — the other workers are using the same file.

Write only the files listed as yours, one write_source call each, each one
complete. Import from models, and from the standard library. Do not import
another worker's module unless the contract says it exists.

Write every file in your list, including the test files, and write them all in
one go — issue one write_source call per file, together, in a single reply. Do
not write one and wait.

Test what your own code does, not what another worker's does.

Keep each file under 200 lines. When all your files are written, say so and
stop.\
"""

_TEST_SYSTEM = """\
The project is written. Run its tests with run_tests and report what happened.

If tests fail, read the failure, read the file it points at with read_source,
and write a corrected version with write_source. Fix only what the failure
names. Do not rewrite files that pass, and do not delete a failing test to make
the suite green — a failing test that describes the right behaviour is more
useful than a passing one that does not.

Stop when the tests pass, or when you have tried twice and the failure is not
something you can fix from what you can see. Say which.\
"""


@dataclass(slots=True)
class BuildStep:
    """One worker's turn."""

    name: str
    purpose: str = ""
    #: What the skill said this worker owns.
    declared: list[str] = field(default_factory=list)
    #: What it actually wrote. Kept apart from ``declared`` because a step that
    #: failed would otherwise report the files it was asked for as though it
    #: had produced them, which is the one thing a build report must not do.
    files: list[str] = field(default_factory=list)
    stopped: str = ""
    tools: list[str] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "declared": self.declared,
            "files": self.files,
            "stopped": self.stopped,
            "tools": self.tools,
            "seconds": round(self.seconds, 2),
            "note": self.note,
        }


@dataclass(slots=True)
class BuildResult:
    """One finished build."""

    intent: str
    skill: str
    project: str = ""
    files: list[str] = field(default_factory=list)
    steps: list[BuildStep] = field(default_factory=list)
    tests: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)
    stopped: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "skill": self.skill,
            "project": self.project,
            "files": self.files,
            "steps": [step.to_dict() for step in self.steps],
            "tests": self.tests,
            "warnings": self.warnings,
            "stopped": self.stopped,
            "seconds": round(self.seconds, 2),
        }


class Builder:
    """Runs a skill's roster over one sandboxed project directory."""

    def __init__(
        self,
        skill: Skill,
        *,
        intent: str = "",
        budget: Optional[Budget] = None,
        backend=None,
        on_progress=None,
    ) -> None:
        self.skill = skill
        self.intent = (intent or skill.description).strip()
        self.budget = budget or Budget(code_calls=config.BUILD_CODE_CALLS)
        self._backend = backend
        self._on_progress = on_progress
        self._warnings: list[str] = []

    @property
    def backend(self):
        if self._backend is None:
            from pipeline.extract.llm import CODE, get_agent_backend

            # The agent backend, because these steps call tools; the CODE role
            # decides which one, and a fallback to local is worth saying out
            # loud rather than leaving in a log.
            self._backend = get_agent_backend(role=CODE)
        return self._backend

    def build(self) -> BuildResult:
        result = BuildResult(intent=self.intent, skill=self.skill.name)

        if not self.skill.buildable:
            result.stopped = (
                f"the {self.skill.name!r} skill declares no agents, so there is "
                "nothing to build. Add an 'agents:' block to its SKILL.md."
            )
            return result
        if not config.BUILD_ENABLED:
            result.stopped = "building is off; set BUILD_ENABLED to turn it on"
            return result

        root = sandbox.project_root(self.skill.name)
        result.project = str(root)
        started = time.perf_counter()

        with ActiveBuild(root):
            self._report({"stage": "contract", "project": self.skill.name})
            result.steps.append(self._contract(root))

            for index, agent in enumerate(self.skill.agents, start=1):
                self._report({
                    "stage": "worker",
                    "worker": agent.name,
                    "index": index,
                    "of": len(self.skill.agents),
                })
                result.steps.append(self._worker(agent, root))

            if Effect.EXECUTE in self.budget.effects():
                self._report({"stage": "testing"})
                step, tests = self._tests(root)
                result.steps.append(step)
                result.tests = tests
            else:
                result.warnings.append(
                    "the generated code was not run — grant execution to run its tests"
                )

        result.files = sandbox.tree(root)
        result.warnings.extend(self._warnings)
        result.seconds = time.perf_counter() - started
        result.stopped = result.stopped or "built"

        if not result.files:
            result.stopped = "no files were written"

        log.info(
            "agents.builder.done",
            skill=self.skill.name,
            files=len(result.files),
            steps=len(result.steps),
            tests_ok=(result.tests or {}).get("ok"),
            seconds=round(result.seconds, 1),
        )
        metrics.incr("agents.builder.runs")
        return result

    # --- the three kinds of step -------------------------------------------

    def _contract(self, root: Path) -> BuildStep:
        roster = "\n".join(
            f"  {agent.name}: {agent.purpose} (writes {', '.join(agent.writes) or 'nothing yet'})"
            for agent in self.skill.agents
        )
        prompt = (
            f"The project to build:\n{self.intent}\n\n"
            f"The workers who will write it:\n{roster}\n\n"
            "Write models.py and README.md now."
        )
        return self._run_step(
            BuildStep(name="contract", purpose="the shared types and the interfaces"),
            system=_CONTRACT_SYSTEM,
            prompt=prompt,
            root=root,
            effects=(Effect.CODE,),
        )

    def _worker(self, agent: BuildAgent, root: Path) -> BuildStep:
        contract = self._read(root, CONTRACT_FILE)
        readme = self._read(root, README_FILE)
        written = sandbox.tree(root)

        wanted = _files_for(agent)
        parts = [
            f"The project:\n{self.intent}",
            f"You are the {agent.name}. {agent.purpose}",
            "Your files, all of which you must write:\n"
            + "\n".join(f"  {name}" for name in wanted),
        ]
        if contract:
            parts.append(
                f"The shared types, already written as {CONTRACT_FILE}:\n\n"
                + _clip(contract, _CONTRACT_CHARS)
            )
        else:
            parts.append(
                f"{CONTRACT_FILE} was not written. Define what you need locally and "
                "keep it minimal."
            )
        if readme:
            parts.append(
                f"The interfaces you are held to, from {README_FILE}:\n\n"
                + _clip(readme, _README_CHARS)
            )
        if written:
            parts.append("Already in the project: " + ", ".join(written))

        return self._run_step(
            BuildStep(name=agent.name, purpose=agent.purpose, declared=wanted),
            system=_WORKER_SYSTEM,
            prompt="\n\n".join(parts),
            root=root,
            effects=(Effect.CODE,),
        )

    def _tests(self, root: Path) -> tuple[BuildStep, Optional[dict[str, Any]]]:
        step = self._run_step(
            BuildStep(name="tests", purpose="run what was written and fix what fails"),
            system=_TEST_SYSTEM,
            prompt=(
                f"The project:\n{self.intent}\n\n"
                "Files: " + ", ".join(sandbox.tree(root)) + "\n\nRun the tests."
            ),
            root=root,
            effects=(Effect.CODE, Effect.EXECUTE),
        )
        # Run once more directly, so the reported result is a measurement rather
        # than the model's account of one. A model that says the tests passed
        # and a suite that passes are different claims.
        try:
            final = sandbox.run_tests(root).to_dict()
        except sandbox.SandboxError as exc:
            final = {"ok": False, "exit_code": -1, "output": str(exc), "seconds": 0.0}
        return step, final

    # --- the machinery -----------------------------------------------------

    def _run_step(
        self,
        step: BuildStep,
        *,
        system: str,
        prompt: str,
        root: Path,
        effects: tuple[Effect, ...],
    ) -> BuildStep:
        before = set(sandbox.tree(root))

        loop = AgentLoop(
            backend=self.backend,
            budget=self._step_budget(),
            system=system,
        )
        allowed = [Effect.READ, *effects]
        permitted = {spec.name for spec in catalog(self.budget.effects())}
        offered = [
            tool
            for tool in describe_all(allowed)
            if tool["name"] in permitted and _is_build_tool(tool["name"])
        ]
        loop.tools = lambda: offered  # type: ignore[method-assign]

        started = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                outcome = loop.run(prompt)
                step.stopped = outcome.stopped
                step.tools = outcome.tools_used

                # Retried only when the step produced nothing. A worker that
                # wrote one of its two files and then hit the limit has made
                # progress, and running it again would regenerate the file it
                # already has from a different sample.
                wrote_nothing = not (set(sandbox.tree(root)) - before)
                wait = _rate_limit_wait(outcome.warnings)
                if wait is not None and attempts == 1 and wrote_nothing:
                    self._report({"stage": "waiting", "worker": step.name, "seconds": wait})
                    log.info("agents.builder.rate_limited", step=step.name, waiting=wait)
                    time.sleep(wait)
                    continue

                # Attributed. The loop reports "model call failed" without
                # knowing which worker it was called for, and across five steps
                # that is a warning nobody can act on.
                self._warnings.extend(f"{step.name}: {note}" for note in outcome.warnings)
            except Exception as exc:
                # One worker failing must not lose the ones that already wrote
                # their files. The build reports what it got.
                log.warning("agents.builder.step_failed", step=step.name, error=repr(exc))
                step.stopped = f"failed: {exc}"
                self._warnings.append(f"{step.name} failed: {exc}")
            break

        step.seconds = time.perf_counter() - started
        step.files = sorted(set(sandbox.tree(root)) - before)
        if not step.files and step.name != "tests":
            step.note = "wrote nothing"
        missing = [name for name in step.declared if name not in step.files]
        if missing and step.files:
            step.note = "did not write " + ", ".join(missing)
        self._report({"stage": "wrote", "worker": step.name, "files": step.files})
        return step

    def _step_budget(self) -> Budget:
        """One worker's slice. Small: it writes one or two files and stops."""
        # Enough turns for each declared file plus a closing one. A worker
        # with two files and two turns spends both writing and never says it
        # is done, which reads as a budget failure rather than a finished step.
        per_step = max(4, self.budget.max_iterations // 2)
        return Budget(
            max_iterations=per_step,
            max_tool_calls=max(3, self.budget.max_tool_calls // 3),
            max_seconds=self.budget.max_seconds,
            max_tokens=self.budget.max_tokens,
            code_calls=self.budget.code_calls,
            execute_calls=self.budget.execute_calls,
        )

    @staticmethod
    def _read(root: Path, name: str) -> str:
        path = root / name
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def _report(self, event: dict[str, Any]) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(event)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("agents.builder.progress_failed", error=repr(exc))


#: How long a step waits when the model says it is over its token rate.
#: Bounded: a per-minute overage is worth waiting out, and a daily one is not
#: something a build should sleep through -- it reports and moves on.
_MAX_RATE_WAIT = 90.0
_DEFAULT_RATE_WAIT = 25.0

#: Groq names the wait in the body: "Please try again in 12.5s".
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def _clip(text: str, limit: int) -> str:
    """Trim, saying so. A worker that needs the rest reads it with read_source."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + (
        f"\n\n… [trimmed at {limit} characters — read the whole file with read_source]"
    )


def _files_for(agent: BuildAgent) -> list[str]:
    """A worker's modules, and a test file for each.

    The tests are derived rather than left to the roster or to the prompt. A
    first version asked for them in the system prompt -- "alongside your module,
    write its tests" -- and a real run against gpt-oss-120b wrote four modules
    and not one test, so ``run_tests`` collected nothing and the whole execute
    path had nothing to execute. A file the worker is handed by name in its own
    list is a file it writes.
    """
    modules = list(agent.writes) or [f"{agent.name}.py"]
    wanted = list(modules)
    for name in modules:
        path = Path(name)
        if path.suffix != ".py" or path.name.startswith("test_"):
            continue
        derived = str(Path("tests") / f"test_{path.stem}.py")
        # A roster that already named its test file must not be handed it twice.
        if derived not in wanted:
            wanted.append(derived)
    return wanted


def _rate_limit_wait(warnings: list[str]) -> Optional[float]:
    """Seconds to wait before retrying, if this step hit a token-rate limit.

    A build is six or more whole-file generations in a row, and a free tier
    measured in tokens per minute cannot take them back to back -- measured on
    Groq's 8000 TPM, the contract call plus two workers is already over. That
    is the ordinary case for this feature, not an exceptional one, so a step
    that hits it waits and asks again rather than being recorded as a failure.
    """
    for note in warnings:
        lowered = note.lower()
        if "rate limit" in lowered or "token-rate" in lowered or "429" in lowered:
            found = _RETRY_AFTER.search(note)
            wait = float(found.group(1)) + 1.0 if found else _DEFAULT_RATE_WAIT
            return min(wait, _MAX_RATE_WAIT)
    return None


#: The tools a build step may see. The corpus tools are registered and READ, so
#: without this filter every worker would also be offered search_corpus — twelve
#: more choices for a decision that is only ever "write the file".
_BUILD_TOOLS = frozenset({"write_source", "read_source", "list_workspace", "run_tests"})


def _is_build_tool(name: str) -> bool:
    return name in _BUILD_TOOLS


def build(skill: Skill, **kwargs) -> BuildResult:
    return Builder(skill, **kwargs).build()


__all__ = ["BuildResult", "BuildStep", "Builder", "build"]
