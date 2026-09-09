"""Where generated code may write, and what it gets when it runs.

This is the security boundary of the build feature, and the tests are written
as attacks rather than as descriptions. The thing being defended against is not
a hostile user -- it is a model that has misunderstood the task, plus whatever
reaches the model through a page it read, and both produce the same paths.

Worth being plain in the tests as well as the module: a subprocess with rlimits
is mitigation, not isolation. These assert the mitigations actually hold.
"""

from __future__ import annotations

import os
import sys

import pytest

from config import config
from pipeline.agents.tools import sandbox
from pipeline.agents.tools.sandbox import SandboxError, resolve_in


@pytest.fixture
def root(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    return project


# --- path confinement ------------------------------------------------------


def test_an_ordinary_relative_path_is_allowed(root):
    assert resolve_in(root, "orders.py").parent == root.resolve()


def test_a_nested_path_is_allowed(root):
    assert resolve_in(root, "pkg/machine.py").name == "machine.py"


def test_an_absolute_path_is_refused(root):
    with pytest.raises(SandboxError, match="absolute"):
        resolve_in(root, "/etc/passwd")


def test_a_traversal_is_refused(root):
    with pytest.raises(SandboxError, match="outside"):
        resolve_in(root, "../escape.py")


def test_a_traversal_buried_mid_path_is_refused(root):
    """Checking only the first segment would pass this one."""
    with pytest.raises(SandboxError, match="outside"):
        resolve_in(root, "pkg/sub/../../../escape.py")


def test_a_symlink_pointing_out_is_refused(root):
    """Why the path is resolved before it is checked, not after.

    A name-only check sees "link.py", which looks like any other file in the
    project, and writing through it lands wherever the link points.
    """
    outside = root.parent / "secret.py"
    outside.write_text("x")
    (root / "link.py").symlink_to(outside)
    with pytest.raises(SandboxError, match="outside"):
        resolve_in(root, "link.py")


def test_a_symlinked_directory_pointing_out_is_refused(root):
    (root / "up").symlink_to(root.parent, target_is_directory=True)
    with pytest.raises(SandboxError, match="outside"):
        resolve_in(root, "up/secret.py")


def test_a_dotfile_directory_is_refused(root):
    for path in (".git/config", ".ssh/id_rsa", ".env"):
        with pytest.raises(SandboxError):
            resolve_in(root, path)


def test_a_shell_script_is_refused(root):
    """The allowlist is about what something else might execute on sight."""
    with pytest.raises(SandboxError, match="not a file type"):
        resolve_in(root, "install.sh")


def test_a_shared_object_is_refused(root):
    with pytest.raises(SandboxError, match="not a file type"):
        resolve_in(root, "payload.so")


def test_an_empty_path_is_refused(root):
    with pytest.raises(SandboxError, match="required"):
        resolve_in(root, "")


def test_a_project_name_must_be_a_single_segment():
    for name in ("../etc", "/absolute", "a/b", "", "Upper"):
        with pytest.raises(SandboxError):
            sandbox.project_root(name)


# --- what the subprocess is handed -----------------------------------------


def _probe(root, code: str) -> str:
    """Run one assertion inside the sandbox and return what pytest saw."""
    (root / "test_probe.py").write_text(code)
    return sandbox.run_tests(root, timeout=30).output


def test_the_environment_carries_no_secrets(root, monkeypatch):
    """The one that matters. Code a model wrote must not be handed the key that
    wrote it, nor anything else out of .env."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-should-never-be-visible")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    output = _probe(root, (
        "import os\n"
        "def test_no_secrets():\n"
        "    leaked = [k for k in os.environ if 'KEY' in k or 'TOKEN' in k "
        "or 'SECRET' in k or 'REDIS' in k]\n"
        "    assert leaked == [], leaked\n"
    ))
    assert "1 passed" in output, output


def test_the_environment_is_an_allowlist(root):
    """Not a denylist. A denylist has to know every name a secret arrives
    under, and .gitignore in this repo records how that goes.

    Three sources of names are tolerated beyond the six passed deliberately:
    pytest sets its own inside the child, and the platform injects some
    regardless of what is handed over -- macOS adds ``__CF_USER_TEXT_ENCODING``
    to every process through CoreFoundation, which no environment argument
    prevents. Neither carries anything from this process.
    """
    output = _probe(root, (
        "import os\n"
        "PASSED = {'PATH', 'HOME', 'TMPDIR', 'PYTHONDONTWRITEBYTECODE', "
        "'PYTHONUNBUFFERED', 'LANG'}\n"
        "def test_small_environment():\n"
        "    unexpected = [k for k in os.environ if k not in PASSED\n"
        "                  and not k.startswith(('PYTEST_', '__'))]\n"
        "    assert unexpected == [], unexpected\n"
    ))
    assert "1 passed" in output, output


def test_generated_code_cannot_import_the_pipeline(root):
    """cwd is the sandbox and PYTHONPATH is unset, so the corpus, LanceDB and
    Kuzu are out of reach of anything written here."""
    output = _probe(root, (
        "import pytest\n"
        "def test_no_pipeline():\n"
        "    with pytest.raises(ImportError):\n"
        "        import pipeline\n"
    ))
    assert "1 passed" in output, output


def test_the_working_directory_is_the_sandbox(root):
    output = _probe(root, (
        "import os\n"
        f"def test_cwd():\n"
        f"    assert os.getcwd() == {str(root.resolve())!r}\n"
    ))
    assert "1 passed" in output, output


# --- limits ----------------------------------------------------------------


def test_a_test_that_never_finishes_is_killed(root, monkeypatch):
    """An infinite loop is among the more common things a model writes by
    accident, so this is the ordinary case rather than an edge one."""
    monkeypatch.setattr(config, "BUILD_TEST_TIMEOUT", 2.0)
    (root / "test_hang.py").write_text(
        "import time\ndef test_forever():\n    time.sleep(600)\n"
    )
    run = sandbox.run_tests(root)
    assert run.timed_out is True
    assert run.ok is False
    assert "killed after" in " ".join(run.warnings)


def test_a_passing_suite_is_reported_as_passing(root):
    (root / "test_ok.py").write_text("def test_true():\n    assert True\n")
    run = sandbox.run_tests(root)
    assert run.ok is True and run.exit_code == 0


def test_a_failing_suite_is_reported_with_its_output(root):
    (root / "test_bad.py").write_text("def test_false():\n    assert 1 == 2\n")
    run = sandbox.run_tests(root)
    assert run.ok is False
    assert "test_false" in run.output


def test_no_tests_at_all_says_so_rather_than_failing(root):
    """pytest exits 5 for an empty run. "the build wrote no tests" is more use
    to a reader than "the tests failed"."""
    run = sandbox.run_tests(root)
    assert run.exit_code == 5
    assert any("no tests" in w for w in run.warnings)


def test_enormous_output_is_trimmed(root):
    """A runaway suite must not fill a model's context or a response body."""
    (root / "test_loud.py").write_text(
        "def test_loud():\n    print('x' * 200000)\n    assert False\n"
    )
    run = sandbox.run_tests(root)
    assert len(run.output) < 20_000
    assert "characters omitted" in run.output


# --- the tree --------------------------------------------------------------


def test_the_tree_lists_what_was_written(root):
    (root / "a.py").write_text("")
    (root / "pkg").mkdir()
    (root / "pkg" / "b.py").write_text("")
    assert sandbox.tree(root) == ["a.py", "pkg/b.py"]


def test_the_tree_hides_caches_and_dotfiles(root):
    """Otherwise the runner's own .home and __pycache__ are reported as though
    the build had produced them."""
    (root / "a.py").write_text("")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "a.pyc").write_text("")
    (root / ".home").mkdir()
    (root / ".home" / "junk").write_text("")
    assert sandbox.tree(root) == ["a.py"]
