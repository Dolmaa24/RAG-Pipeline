"""Where generated code lives, and how it is run.

**This is a subprocess with limits, not a container.** It reduces the blast
radius of code a model wrote; it does not isolate it. A determined escape is
possible, and anyone reading this should know that before turning execution on.
The interface is deliberately small so that swapping the runner for Docker
later touches this file and nothing else.

What it does do:

* **Confines every path.** A file is resolved and then checked to be under the
  sandbox root. Resolution comes first so that a symlink pointing outward is
  caught by the same check as ``..`` and an absolute path -- checking the string
  before resolving catches neither.
* **Runs a fixed argv**, never a shell string. There is nothing for a filename
  to be interpreted as.
* **Runs from inside the sandbox with ``PYTHONPATH`` unset**, so generated code
  cannot ``import pipeline`` and reach the corpus, LanceDB or Kuzu.
* **Hands over almost no environment.** ``PATH``, a ``HOME`` inside the sandbox,
  and two Python switches. Everything else is dropped, and the point of that is
  ``GROQ_API_KEY``: code written by a model must not be handed the key that
  wrote it.
* **Bounds CPU, memory and wall clock.** An infinite loop is one of the more
  common things a model writes by accident, so the timeout is not an edge case.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import config
from observability import get_logger, metrics

log = get_logger("agents.sandbox")

#: What a build may write. Source, config and documentation -- nothing that a
#: browser or an operating system treats as executable on sight.
ALLOWED_SUFFIXES = frozenset(
    {".py", ".md", ".txt", ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml",
     ".html", ".css", ".js", ".sql", ""}
)

#: Names that are a filesystem's business and never a generated project's.
_REFUSED_NAMES = frozenset({".git", ".env", ".ssh", "__pycache__"})


class SandboxError(RuntimeError):
    """A path or a command was refused. Reported to the model, not raised at it."""


def workspace_root() -> Path:
    """Where projects are built. Relative paths resolve against the repository
    root, so a worker started from anywhere writes to the same place."""
    configured = Path(config.BUILD_WORKSPACE).expanduser()
    if configured.is_absolute():
        return configured
    return (Path(__file__).resolve().parents[3] / configured).resolve()


def project_root(project: str) -> Path:
    """One project's directory, created on demand.

    ``project`` is a skill name and reaches this from a request, so it is
    checked as a single path segment rather than trusted as one.
    """
    import re

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project or ""):
        raise SandboxError(f"{project!r} is not a project name")
    root = workspace_root() / project
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_in(root: Path, relative: str) -> Path:
    """``relative`` inside ``root``, or a refusal.

    The security boundary of the whole build feature. Everything a model asks
    to write goes through here.
    """
    text = (relative or "").strip()
    if not text:
        raise SandboxError("a path is required")

    candidate = Path(text)
    if candidate.is_absolute():
        raise SandboxError(f"{text!r} is absolute; give a path inside the project")
    if any(part in _REFUSED_NAMES for part in candidate.parts):
        raise SandboxError(f"{text!r} names a directory a project may not touch")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise SandboxError(
            f"{candidate.suffix!r} is not a file type a build may write; "
            f"allowed: {', '.join(sorted(s for s in ALLOWED_SUFFIXES if s))}"
        )

    base = root.resolve()
    # Resolve first: this follows symlinks, so a link inside the sandbox
    # pointing outside it fails the containment check below rather than passing
    # a check on its own name.
    resolved = (base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise SandboxError(f"{text!r} resolves outside the project directory")
    return resolved


@dataclass(slots=True)
class TestRun:
    """What running the generated tests produced."""

    ok: bool
    exit_code: int
    output: str
    seconds: float
    timed_out: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "output": self.output,
            "seconds": round(self.seconds, 2),
            "timed_out": self.timed_out,
            "warnings": self.warnings,
        }


#: Enough to see which tests failed and why, without a runaway suite filling a
#: model's context or a response body.
_MAX_OUTPUT = 8000


def _clean_environment(root: Path) -> dict[str, str]:
    """The whole environment the generated code gets.

    An allowlist rather than a denylist. A denylist would have to know every
    name a secret might arrive under, and this project's own ``.gitignore``
    already records what happens when you try that: ``.env`` alone did not cover
    ``.env.local``.
    """
    home = root / ".home"
    home.mkdir(exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "TMPDIR": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
        # Unbuffered, so output survives a kill on timeout.
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
    }


def _limits() -> None:  # pragma: no cover - runs in the child process
    """Applied inside the child, before it execs."""
    try:
        cpu = config.BUILD_TEST_CPU_SECONDS
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        memory = config.BUILD_TEST_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        # A platform that refuses a limit must not stop the run; the wall clock
        # is the backstop and it is enforced by the parent.
        pass


def run_tests(root: Path, *, timeout: Optional[float] = None) -> TestRun:
    """Run pytest inside ``root``. Never a shell, never the project's env."""
    import time

    root = root.resolve()
    if not root.is_dir():
        raise SandboxError(f"{root} does not exist")

    argv = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    limit = config.BUILD_TEST_TIMEOUT if timeout is None else timeout
    warnings: list[str] = []
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            env=_clean_environment(root),
            capture_output=True,
            text=True,
            timeout=limit,
            preexec_fn=_limits if hasattr(os, "fork") else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        log.warning("sandbox.timeout", root=str(root), seconds=round(elapsed, 1))
        metrics.incr("sandbox.timeout")
        partial = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return TestRun(
            ok=False,
            exit_code=-1,
            output=_trim(partial if isinstance(partial, str) else partial.decode("utf-8", "replace")),
            seconds=elapsed,
            timed_out=True,
            warnings=[f"killed after {limit:.0f}s — the generated tests did not finish"],
        )
    except FileNotFoundError as exc:
        raise SandboxError(f"could not start the test runner: {exc}") from exc

    elapsed = time.perf_counter() - started
    output = _trim((completed.stdout or "") + (completed.stderr or ""))
    # pytest exits 5 when it collected nothing, which is not a failing suite —
    # it is a build that wrote no tests, and saying so is more use than "failed".
    if completed.returncode == 5:
        warnings.append("no tests were collected; the build wrote none")

    log.info(
        "sandbox.tests",
        root=str(root),
        exit_code=completed.returncode,
        seconds=round(elapsed, 1),
    )
    metrics.incr("sandbox.tests")
    return TestRun(
        ok=completed.returncode == 0,
        exit_code=completed.returncode,
        output=output,
        seconds=elapsed,
        warnings=warnings,
    )


def _trim(text: str) -> str:
    if len(text) <= _MAX_OUTPUT:
        return text
    half = _MAX_OUTPUT // 2
    return f"{text[:half]}\n\n… [{len(text) - _MAX_OUTPUT} characters omitted] …\n\n{text[-half:]}"


def tree(root: Path, *, limit: int = 200) -> list[str]:
    """Relative paths of what has been written, sorted, excluding our own."""
    root = root.resolve()
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        found.append(str(relative))
        if len(found) >= limit:
            break
    return found


def clear(project: str) -> int:
    """Remove one project's directory. Returns how many files went."""
    root = workspace_root() / project
    if not root.is_dir():
        return 0
    count = len(tree(root, limit=10_000))
    shutil.rmtree(root)
    log.info("sandbox.cleared", project=project, files=count)
    return count


__all__ = [
    "ALLOWED_SUFFIXES",
    "SandboxError",
    "TestRun",
    "clear",
    "project_root",
    "resolve_in",
    "run_tests",
    "tree",
    "workspace_root",
]
