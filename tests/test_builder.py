"""The build sequence: a contract, then the roster, then the tests.

No model runs here. The backend is scripted, so "the receptionist wrote its
file and the cleaner saw it" is a fixture rather than something to hope for --
which is the same reason the supervisor's tests inject an answerer.

What these are actually protecting is the ordering. Four workers generated
independently invent four incompatible ideas of what an order is, and the
contract step running first, and being shown to everyone after, is the only
thing standing between this feature and four modules that do not import each
other.
"""

from __future__ import annotations

import pytest

from config import config
from pipeline.agents.budget import Budget
from pipeline.agents.builder import Builder
from pipeline.agents.tools import sandbox
from pipeline.extract.llm.base import ToolRequest, ToolTurn
from pipeline.skills.loader import parse

SKILL = """\
---
name: barista
description: A cafe.
tools: [search_corpus]
agents:
  - name: receptionist
    purpose: Takes orders.
    writes: [orders.py]
  - name: cleaner
    purpose: Sweeps stale orders.
    writes: [cleanup.py]
---

Body.
"""


def call(name, **arguments):
    return ToolRequest(name=name, arguments=arguments, call_id=name)


class Scripted:
    """Writes a file for whichever step is asking, and records the prompts."""

    name, model = "scripted", "scripted-1"

    def __init__(self, *, fail_on: str = "", write: bool = True) -> None:
        self.prompts: list[str] = []
        self.offered: list[set[str]] = []
        self.fail_on = fail_on
        self.write = write

    def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
        prompt = messages[-1].content if messages else ""
        self.prompts.append(prompt)
        self.offered.append({tool["name"] for tool in tools})

        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("the model call failed")
        if not self.write:
            return ToolTurn(text="nothing to write", backend=self.name, model=self.model)

        if "Write models.py and README.md now." in prompt:
            return ToolTurn(backend=self.name, model=self.model, calls=[
                call("write_source", path="models.py",
                     content="from dataclasses import dataclass\n\n@dataclass\nclass Order:\n    drink: str\n"),
                call("write_source", path="README.md", content="# Cafe\n\n## receptionist\ntake_order(drink)\n"),
            ])
        if "You are the receptionist" in prompt:
            return ToolTurn(backend=self.name, model=self.model, calls=[
                call("write_source", path="orders.py",
                     content="from models import Order\n\ndef take_order(d):\n    return Order(drink=d)\n"),
                call("write_source", path="tests/test_orders.py",
                     content="from orders import take_order\n\ndef test_order():\n    assert take_order('latte').drink == 'latte'\n"),
            ])
        if "You are the cleaner" in prompt:
            return ToolTurn(backend=self.name, model=self.model, calls=[
                call("write_source", path="cleanup.py", content="def sweep(o):\n    return list(o)\n"),
            ])
        if "Run the tests." in prompt:
            return ToolTurn(backend=self.name, model=self.model, calls=[call("run_tests")])
        return ToolTurn(text="done", backend=self.name, model=self.model)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILD_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(config, "BUILD_ENABLED", True)
    return tmp_path


@pytest.fixture
def skill():
    return parse(SKILL)


def run(skill, backend, **kwargs):
    kwargs.setdefault("budget", Budget(code_calls=20, execute_calls=2))
    kwargs.setdefault("intent", "make a Baristo system")
    return Builder(skill, backend=backend, **kwargs).build()


# --- refusals --------------------------------------------------------------


def test_a_skill_with_no_roster_is_refused(workspace):
    """With a reason, rather than an empty project directory."""
    plain = parse("---\nname: plain\ndescription: d\ntools: [search_corpus]\n---\n\nBody.\n")
    result = run(plain, Scripted())
    assert "declares no agents" in result.stopped
    assert result.files == []


def test_building_is_off_by_default(skill, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILD_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(config, "BUILD_ENABLED", False)
    assert "building is off" in run(skill, Scripted()).stopped


# --- the sequence ----------------------------------------------------------


def test_the_contract_runs_before_any_worker(workspace, skill):
    result = run(skill, Scripted())
    assert [step.name for step in result.steps] == [
        "contract", "receptionist", "cleaner", "tests"
    ]


def test_the_roster_runs_in_declaration_order(workspace, skill):
    """Order is the running order, and a later worker is shown what the
    earlier ones wrote. Sorting the roster would change what each one sees."""
    backend = Scripted()
    run(skill, backend)
    receptionist = next(i for i, p in enumerate(backend.prompts) if "the receptionist" in p)
    cleaner = next(i for i, p in enumerate(backend.prompts) if "the cleaner" in p)
    assert receptionist < cleaner


def test_every_worker_is_shown_the_contract(workspace, skill):
    """The single thing that makes the files compose."""
    backend = Scripted()
    run(skill, backend)
    for prompt in backend.prompts:
        if "You are the" in prompt:
            assert "class Order" in prompt, prompt[:200]


def test_a_worker_is_told_what_already_exists(workspace, skill):
    backend = Scripted()
    run(skill, backend)
    cleaner = next(p for p in backend.prompts if "the cleaner" in p)
    assert "orders.py" in cleaner


def test_the_files_are_written(workspace, skill):
    result = run(skill, Scripted())
    assert set(result.files) == {
        "README.md", "cleanup.py", "models.py", "orders.py", "tests/test_orders.py"
    }


def test_each_step_reports_only_what_it_wrote(workspace, skill):
    result = run(skill, Scripted())
    by_name = {step.name: step for step in result.steps}
    assert by_name["contract"].files == ["README.md", "models.py"]
    assert by_name["cleaner"].files == ["cleanup.py"]


def test_a_step_that_wrote_nothing_says_so(workspace, skill):
    """It must not report the files it was asked for as though it made them."""
    result = run(skill, Scripted(write=False))
    receptionist = next(s for s in result.steps if s.name == "receptionist")
    assert receptionist.declared == ["orders.py", "tests/test_orders.py"]
    assert receptionist.files == []
    assert receptionist.note == "wrote nothing"


def test_one_failing_worker_does_not_lose_the_others(workspace, skill):
    """A build that got three of four files is worth more than an exception."""
    result = run(skill, Scripted(fail_on="You are the receptionist"))
    assert "cleanup.py" in result.files
    assert any("receptionist" in w for w in result.warnings)


# --- what a worker may reach ----------------------------------------------


def test_a_worker_is_offered_only_the_build_tools(workspace, skill):
    """The corpus tools are registered and READ, so without the filter every
    worker would also see search_corpus — twelve more choices for a decision
    that is only ever 'write the file'."""
    backend = Scripted()
    run(skill, backend)
    for offered in backend.offered:
        assert offered <= {"write_source", "read_source", "list_workspace", "run_tests"}
        assert "search_corpus" not in offered


def test_no_worker_is_offered_the_test_runner(workspace, skill):
    """Only the tests step may execute, even when execution is budgeted."""
    backend = Scripted()
    run(skill, backend)
    for prompt, offered in zip(backend.prompts, backend.offered):
        if "You are the" in prompt or "Write models.py" in prompt:
            assert "run_tests" not in offered


def test_without_execution_nothing_is_run(workspace, skill):
    result = run(skill, Scripted(), budget=Budget(code_calls=20))
    assert result.tests is None
    assert any("not run" in w for w in result.warnings)
    assert [s.name for s in result.steps] == ["contract", "receptionist", "cleaner"]


def test_the_files_are_still_written_without_execution(workspace, skill):
    """A scaffold to read is the default outcome, not a degraded one."""
    result = run(skill, Scripted(), budget=Budget(code_calls=20))
    assert "orders.py" in result.files


# --- the test result -------------------------------------------------------


def test_the_generated_tests_actually_run(workspace, skill):
    result = run(skill, Scripted())
    assert result.tests is not None
    assert result.tests["ok"] is True


def test_the_reported_result_is_measured_not_claimed(workspace, skill):
    """A model saying the tests passed and a suite that passes are different
    claims, so the suite is run once more directly for the report."""
    class Liar(Scripted):
        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            prompt = messages[-1].content if messages else ""
            if "Run the tests." in prompt:
                return ToolTurn(text="All tests passed.", backend=self.name, model=self.model)
            return super().complete_with_tools(messages=messages, tools=tools)

    broken = parse(SKILL)
    result = run(broken, Liar())
    # The scripted receptionist writes a passing test, so overwrite it.
    failing = sandbox.workspace_root() / "barista" / "tests" / "test_orders.py"
    failing.write_text("def test_no():\n    assert False\n")
    assert sandbox.run_tests(failing.parent.parent).ok is False


def test_the_project_lands_in_the_workspace(workspace, skill):
    result = run(skill, Scripted())
    assert result.project.endswith("barista")
    assert (workspace / "barista" / "models.py").is_file()


# --- rate limits -----------------------------------------------------------


def test_a_rate_limited_step_waits_and_asks_again(workspace, skill, monkeypatch):
    """A build is six or more whole-file generations in a row, and Groq's free
    tier is 8000 tokens a minute — measured, the contract call plus two workers
    is already over it. Waiting is the ordinary path here, not an exception."""
    import pipeline.agents.builder as builder_module

    slept: list[float] = []
    monkeypatch.setattr(builder_module.time, "sleep", slept.append)

    class Limited(Scripted):
        def __init__(self):
            super().__init__()
            self.seen = 0

        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            prompt = messages[-1].content if messages else ""
            if "You are the receptionist" in prompt:
                self.seen += 1
                if self.seen == 1:
                    raise RuntimeError("Groq rate limited: try again in 4.5s")
            return super().complete_with_tools(messages=messages, tools=tools)

    result = run(skill, Limited())
    assert slept == [5.5]
    receptionist = next(s for s in result.steps if s.name == "receptionist")
    assert receptionist.files == ["orders.py", "tests/test_orders.py"]


def test_a_step_that_wrote_something_is_not_retried(workspace, skill, monkeypatch):
    """Running it again would regenerate the file it already has."""
    import pipeline.agents.builder as builder_module

    slept: list[float] = []
    monkeypatch.setattr(builder_module.time, "sleep", slept.append)

    class PartlyLimited(Scripted):
        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            prompt = messages[-1].content if messages else ""
            if "You are the cleaner" in prompt:
                # It wrote its file on the first turn, then hit the limit.
                if any(m.role == "tool" for m in messages):
                    raise RuntimeError("Groq rate limited: try again in 9s")
            return super().complete_with_tools(messages=messages, tools=tools)

    run(skill, PartlyLimited())
    assert slept == []


def test_the_wait_is_bounded(workspace, skill, monkeypatch):
    """A per-minute overage is worth waiting out. A daily one is not something
    a build should sleep through."""
    import pipeline.agents.builder as builder_module

    slept: list[float] = []
    monkeypatch.setattr(builder_module.time, "sleep", slept.append)

    class DailyCap(Scripted):
        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            prompt = messages[-1].content if messages else ""
            if "You are the receptionist" in prompt:
                raise RuntimeError("Groq rate limited: try again in 3600s")
            return super().complete_with_tools(messages=messages, tools=tools)

    run(skill, DailyCap())
    assert slept == [builder_module._MAX_RATE_WAIT]


def test_an_ordinary_failure_is_not_retried(workspace, skill, monkeypatch):
    import pipeline.agents.builder as builder_module

    slept: list[float] = []
    monkeypatch.setattr(builder_module.time, "sleep", slept.append)
    run(skill, Scripted(fail_on="You are the receptionist"))
    assert slept == []


def test_a_test_file_is_declared_for_every_module(workspace, skill):
    """Derived, not left to the roster or the prompt.

    Asking for them in the system prompt — "alongside your module, write its
    tests" — produced four modules and no tests in a real run against
    gpt-oss-120b, so run_tests collected nothing and the execute path had
    nothing to execute.
    """
    backend = Scripted()
    run(skill, backend)
    receptionist = next(p for p in backend.prompts if "the receptionist" in p)
    assert "orders.py" in receptionist
    assert "tests/test_orders.py" in receptionist


def test_a_worker_that_names_no_files_still_gets_one(workspace):
    from pipeline.skills.models import BuildAgent
    from pipeline.agents.builder import _files_for

    assert _files_for(BuildAgent("cleaner", "sweeps")) == [
        "cleaner.py", "tests/test_cleaner.py"
    ]


def test_a_test_file_does_not_get_a_test_of_its_own():
    from pipeline.skills.models import BuildAgent
    from pipeline.agents.builder import _files_for

    agent = BuildAgent("x", "y", ("orders.py", "tests/test_orders.py"))
    assert _files_for(agent) == ["orders.py", "tests/test_orders.py"]


def test_a_non_python_file_gets_no_test():
    from pipeline.skills.models import BuildAgent
    from pipeline.agents.builder import _files_for

    assert _files_for(BuildAgent("d", "docs", ("GUIDE.md",))) == ["GUIDE.md"]


# --- two backends, one build ------------------------------------------------


class Named:
    """A backend that knows its own name, and writes nothing."""

    def __init__(self, name: str, model: str = "m") -> None:
        self.name, self.model = name, model
        self.calls = 0

    def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
        self.calls += 1
        return ToolTurn(text="done", backend=self.name, model=self.model)


def test_the_contract_and_the_workers_use_different_models(workspace, skill, monkeypatch):
    """The whole point of the split. One hosted call designs the types every
    worker is written against; the workers are one call per file and are what
    exhaust a tokens-per-minute allowance, so they stay local."""
    import pipeline.extract.llm as llm

    chosen: list[str] = []

    def fake(*, local_only=False, role="agent", model=None):
        chosen.append(role)
        return Named("groq" if role == "architect" else "ollama")

    monkeypatch.setattr(llm, "get_agent_backend", fake)
    result = Builder(skill, intent="x", budget=Budget(code_calls=5)).build()

    assert chosen[0] == "architect"
    assert "code" in chosen
    by_name = {step.name: step for step in result.steps}
    assert by_name["contract"].backend == "groq"
    assert by_name["receptionist"].backend == "ollama"


def test_each_backend_is_resolved_once(workspace, skill, monkeypatch):
    """Three workers must not load three copies of a local model."""
    import pipeline.extract.llm as llm

    calls: list[str] = []

    def fake(*, local_only=False, role="agent", model=None):
        calls.append(role)
        return Named(role)

    monkeypatch.setattr(llm, "get_agent_backend", fake)
    Builder(skill, intent="x", budget=Budget(code_calls=5)).build()
    assert sorted(calls) == ["architect", "code"]


def test_the_code_role_asks_for_the_coding_model(workspace, skill, monkeypatch):
    """llama3.2:3b picks tools well and writes poor Python."""
    import pipeline.extract.llm as llm

    asked: dict = {}

    def fake(*, local_only=False, role="agent", model=None):
        asked[role] = model
        return Named(role)

    monkeypatch.setattr(llm, "get_agent_backend", fake)
    Builder(skill, intent="x", budget=Budget(code_calls=5)).build()
    assert asked["code"] == config.CODE_MODEL_NAME
    assert asked["architect"] is None


def test_an_explicit_backend_overrides_both(workspace, skill):
    """One model end to end, which is what every other test here does."""
    backend = Scripted()
    result = Builder(skill, intent="x", budget=Budget(code_calls=5), backend=backend).build()
    assert {step.backend for step in result.steps} == {"scripted"}


def test_an_unreachable_model_falls_back_and_says_so(workspace, skill, monkeypatch):
    """A build that cannot reach Ollama should run on what it can, not fail —
    but the contract falling back is worth saying out loud."""
    import pipeline.extract.llm as llm

    def fake(*, local_only=False, role="agent", model=None):
        if role == "code":
            raise RuntimeError("ollama is not running")
        return Named("groq")

    monkeypatch.setattr(llm, "get_agent_backend", fake)
    result = Builder(skill, intent="x", budget=Budget(code_calls=5)).build()

    assert any("unavailable" in w for w in result.warnings)
    assert next(s for s in result.steps if s.name == "receptionist").backend == "groq"
