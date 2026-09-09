"""What may write code, what may run it, and the gap between the two.

The pipeline's rule is that effects are granted by the request and never by the
model. Writing source and executing it are two new effects, and they are
separate on purpose: a build that produces a scaffold for a person to read
needs the first and not the second. Granting the ability to write is not
granting the ability to run, and that has to be asserted rather than assumed,
because the failure is silent -- a run_tests that stayed in the catalog looks
exactly like one that belongs there.
"""

from __future__ import annotations

import pytest

from config import config
from pipeline.agents.budget import Budget, Spend, would_exceed_effect_budget
from pipeline.agents.tools import Effect, catalog, invoke
from pipeline.agents.tools.build import ActiveBuild


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILD_WORKSPACE", str(tmp_path))
    root = tmp_path / "demo"
    root.mkdir()
    with ActiveBuild(root):
        yield root


def names(*effects) -> set[str]:
    return {spec.name for spec in catalog(effects)}


# --- the gate --------------------------------------------------------------


def test_a_default_budget_reaches_neither_effect():
    allowed = Budget().effects()
    assert Effect.CODE not in allowed and Effect.EXECUTE not in allowed


def test_the_writing_tools_are_absent_without_a_code_budget():
    offered = names(*Budget().effects())
    assert "write_source" not in offered
    assert "read_source" not in offered
    assert "list_workspace" not in offered


def test_writing_does_not_imply_running():
    """The important one. A build may be granted files without execution."""
    offered = names(*Budget(code_calls=5).effects())
    assert "write_source" in offered
    assert "run_tests" not in offered


def test_running_is_granted_separately():
    offered = names(*Budget(code_calls=5, execute_calls=1).effects())
    assert "run_tests" in offered


def test_a_retrieval_budget_sees_none_of_them():
    """An investigation with network and write granted is still not a build."""
    offered = names(*Budget(network_calls=4, write_calls=2).effects())
    assert offered.isdisjoint({"write_source", "read_source", "run_tests", "list_workspace"})


def test_the_spend_is_counted_per_effect():
    spend = Spend()
    spend.record_call(Effect.CODE)
    spend.record_call(Effect.CODE)
    spend.record_call(Effect.EXECUTE)
    assert spend.code_calls == 2 and spend.execute_calls == 1


def test_the_file_budget_is_a_ceiling():
    spend = Spend(code_calls=3)
    assert would_exceed_effect_budget(spend, Budget(code_calls=3), Effect.CODE)
    assert not would_exceed_effect_budget(spend, Budget(code_calls=4), Effect.CODE)


def test_invoking_a_build_tool_unbudgeted_is_refused_not_raised(project):
    call = invoke("write_source", {"path": "a.py", "content": "x = 1"}, allowed=[Effect.READ])
    assert call.ok is False
    assert "did not allow" in call.observation


def test_the_mcp_server_never_advertises_them():
    """It is an allowlist of three settings, so this holds by construction --
    and this test is what keeps it that way when a fourth is added."""
    import mcp_server

    assert set(mcp_server.allowed_effects()) <= {Effect.READ, Effect.NETWORK, Effect.WRITE}


# --- writing ---------------------------------------------------------------


CODE_ONLY = [Effect.READ, Effect.CODE]


def test_a_file_is_written(project):
    call = invoke("write_source", {"path": "orders.py", "content": "x = 1\n"}, allowed=CODE_ONLY)
    assert call.ok and (project / "orders.py").read_text() == "x = 1\n"


def test_a_nested_file_creates_its_directory(project):
    call = invoke(
        "write_source",
        {"path": "tests/test_orders.py", "content": "def test_x(): pass\n"},
        allowed=CODE_ONLY,
    )
    assert call.ok and (project / "tests" / "test_orders.py").is_file()


def test_an_escaping_path_is_reported_to_the_model_not_raised(project):
    """A refusal the model can read and correct, like every other tool error
    here. An exception would end the build instead of teaching it anything."""
    call = invoke("write_source", {"path": "../escape.py", "content": "x"}, allowed=CODE_ONLY)
    assert call.ok is True  # the tool ran
    assert "outside" in call.observation
    assert not (project.parent / "escape.py").exists()


def test_an_oversized_file_is_refused(project, monkeypatch):
    monkeypatch.setattr(config, "BUILD_MAX_FILE_BYTES", 100)
    call = invoke("write_source", {"path": "big.py", "content": "x" * 200}, allowed=CODE_ONLY)
    assert "exceeds" in call.observation
    assert not (project / "big.py").exists()


def test_too_many_files_is_refused(project, monkeypatch):
    """A model that misunderstood the task fails by writing many files."""
    monkeypatch.setattr(config, "BUILD_MAX_FILES", 2)
    for name in ("a.py", "b.py"):
        invoke("write_source", {"path": name, "content": "x"}, allowed=CODE_ONLY)
    call = invoke("write_source", {"path": "c.py", "content": "x"}, allowed=CODE_ONLY)
    assert "limit" in call.observation
    assert not (project / "c.py").exists()


def test_rewriting_an_existing_file_is_allowed_at_the_limit(project, monkeypatch):
    """Fixing a file a test failed on must not be blocked by the file count."""
    monkeypatch.setattr(config, "BUILD_MAX_FILES", 1)
    invoke("write_source", {"path": "a.py", "content": "x = 1"}, allowed=CODE_ONLY)
    call = invoke("write_source", {"path": "a.py", "content": "x = 2"}, allowed=CODE_ONLY)
    assert call.ok and (project / "a.py").read_text() == "x = 2"


# --- reading back ----------------------------------------------------------


def test_a_written_file_reads_back(project):
    invoke("write_source", {"path": "models.py", "content": "class Order: pass"}, allowed=CODE_ONLY)
    call = invoke("read_source", {"path": "models.py"}, allowed=CODE_ONLY)
    assert "class Order" in call.observation


def test_reading_something_unwritten_says_what_exists(project):
    invoke("write_source", {"path": "models.py", "content": "x"}, allowed=CODE_ONLY)
    call = invoke("read_source", {"path": "machine.py"}, allowed=CODE_ONLY)
    assert "models.py" in call.observation


def test_reading_outside_the_project_is_refused(project):
    call = invoke("read_source", {"path": "../../config.py"}, allowed=CODE_ONLY)
    assert "outside" in call.observation


def test_the_workspace_listing_shows_what_was_written(project):
    invoke("write_source", {"path": "a.py", "content": "x"}, allowed=CODE_ONLY)
    invoke("write_source", {"path": "pkg/b.py", "content": "x"}, allowed=CODE_ONLY)
    call = invoke("list_workspace", {}, allowed=CODE_ONLY)
    assert "a.py" in call.observation and "pkg/b.py" in call.observation


# --- outside a build -------------------------------------------------------


def test_the_tools_are_inert_outside_a_build(tmp_path, monkeypatch):
    """Nothing sets the project except a build, so a stray call has nowhere to
    write rather than a default somewhere."""
    monkeypatch.setattr(config, "BUILD_WORKSPACE", str(tmp_path))
    call = invoke("write_source", {"path": "a.py", "content": "x"}, allowed=CODE_ONLY)
    assert call.ok is False
    assert "no build in progress" in call.observation
