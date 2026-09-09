"""Tools that write source into a build's sandbox, and run what they wrote.

Four tools, on two effects, and the split between them is the point. ``CODE``
puts a file on disk. ``EXECUTE`` runs it. A build that produces a scaffold for a
person to read needs the first and not the second, and that is the default —
granting a model the ability to write is not granting it the ability to run.

**The project is not an argument.** The build sets it, and the tools read it
from a context variable. A model asked for a project name would be able to name
a different one, and there is no reason it should choose where its output lands.
Outside a build the variable is unset and every tool here refuses, which is what
makes these safe to have in a registry the retrieval agents also read from.

Everything a model asks to write goes through :func:`sandbox.resolve_in`, which
is the security boundary for this whole feature.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from config import config
from observability import get_logger

from pipeline.agents.tools import sandbox
from pipeline.agents.tools.registry import Effect, ToolError, tool

log = get_logger("agents.tools.build")

#: The project the current build is writing into. Set by the builder for the
#: duration of one run, and unset otherwise, so these tools are inert anywhere
#: else — including if one ever reached the MCP server, which it cannot.
_project: ContextVar[Optional[Path]] = ContextVar("build_project", default=None)


class ActiveBuild:
    """Context manager binding the tools to one project directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._token = None

    def __enter__(self) -> Path:
        self._token = _project.set(self.root)
        return self.root

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _project.reset(self._token)


def current_project() -> Path:
    root = _project.get()
    if root is None:
        raise ToolError(
            "there is no build in progress; this tool only works inside one"
        )
    return root


class WriteSourceArgs(BaseModel):
    path: str = Field(
        ...,
        description=(
            "Where the file goes, relative to the project root, e.g. "
            "'orders.py' or 'tests/test_orders.py'. Never absolute."
        ),
    )
    content: str = Field(..., description="The complete file. Not a fragment, not a diff.")


class WriteResult(BaseModel):
    path: str
    bytes_written: int = 0
    ok: bool = True
    error: str = ""

    def render(self, *, max_chars: int = 2000) -> str:
        if not self.ok:
            return f"Could not write {self.path}: {self.error}"
        return f"Wrote {self.path} ({self.bytes_written} bytes)."


@tool(
    name="write_source",
    description=(
        "Write one complete source file into the project you are building. "
        "Give the whole file, not a fragment — this replaces whatever was "
        "there. Write one file per call."
    ),
    effect=Effect.CODE,
    cost_ms=5,
    max_chars=300,
)
def write_source(args: WriteSourceArgs) -> WriteResult:
    root = current_project()
    try:
        destination = sandbox.resolve_in(root, args.path)
    except sandbox.SandboxError as exc:
        return WriteResult(path=args.path, ok=False, error=str(exc))

    body = args.content or ""
    encoded = body.encode("utf-8")
    if len(encoded) > config.BUILD_MAX_FILE_BYTES:
        return WriteResult(
            path=args.path,
            ok=False,
            error=(
                f"{len(encoded)} bytes exceeds the {config.BUILD_MAX_FILE_BYTES}-byte "
                "limit for one file. Split it."
            ),
        )

    existing = len(sandbox.tree(root, limit=config.BUILD_MAX_FILES + 1))
    if existing >= config.BUILD_MAX_FILES and not destination.exists():
        return WriteResult(
            path=args.path,
            ok=False,
            error=f"this build has already written {existing} files, its limit",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    log.info("build.wrote", path=args.path, bytes=len(encoded))
    return WriteResult(path=args.path, bytes_written=len(encoded))


class ReadSourceArgs(BaseModel):
    path: str = Field(..., description="A file this build has already written.")


class ReadResult(BaseModel):
    path: str
    content: str = ""
    ok: bool = True
    error: str = ""

    def render(self, *, max_chars: int = 4000) -> str:
        if not self.ok:
            return f"Could not read {self.path}: {self.error}"
        body = self.content
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n… [truncated at {max_chars} chars]"
        return f"{self.path}:\n\n{body}"


@tool(
    name="read_source",
    description=(
        "Read back a file this build has already written, so you can match the "
        "names and types another part of the project defined."
    ),
    effect=Effect.CODE,
    cost_ms=2,
    max_chars=4000,
)
def read_source(args: ReadSourceArgs) -> ReadResult:
    root = current_project()
    try:
        path = sandbox.resolve_in(root, args.path)
    except sandbox.SandboxError as exc:
        return ReadResult(path=args.path, ok=False, error=str(exc))
    if not path.is_file():
        return ReadResult(
            path=args.path,
            ok=False,
            error=f"has not been written yet. Files so far: "
            f"{', '.join(sandbox.tree(root)) or 'none'}",
        )
    return ReadResult(path=args.path, content=path.read_text(encoding="utf-8", errors="replace"))


class ListArgs(BaseModel):
    """No arguments. Declared because every handler takes exactly one model."""


class ListResult(BaseModel):
    files: list[str] = Field(default_factory=list)

    def render(self, *, max_chars: int = 2000) -> str:
        if not self.files:
            return "Nothing has been written yet."
        return "Files written so far:\n" + "\n".join(f"  {name}" for name in self.files)


@tool(
    name="list_workspace",
    description=(
        "List every file written into this project so far, with its path. Call "
        "it before writing, to see what the workers before you already made "
        "and what names they used."
    ),
    effect=Effect.CODE,
    cost_ms=2,
    max_chars=2000,
)
def list_workspace(args: ListArgs) -> ListResult:
    return ListResult(files=sandbox.tree(current_project()))


class RunTestsArgs(BaseModel):
    """No arguments: what runs is fixed, and choosing it is not the model's."""


class TestResult(BaseModel):
    ok: bool = False
    exit_code: int = 0
    output: str = ""
    seconds: float = 0.0
    timed_out: bool = False
    warnings: list[str] = Field(default_factory=list)

    def render(self, *, max_chars: int = 4000) -> str:
        head = "Tests passed." if self.ok else f"Tests failed (exit {self.exit_code})."
        if self.timed_out:
            head = "Tests were killed for running too long."
        body = self.output[:max_chars]
        parts = [head, *self.warnings, body]
        return "\n\n".join(part for part in parts if part).strip()


@tool(
    name="run_tests",
    description=(
        "Run the project's tests and report what happened. Read the failures "
        "and fix the files they point at."
    ),
    effect=Effect.EXECUTE,
    cost_ms=20_000,
    max_chars=4000,
)
def run_tests(args: RunTestsArgs) -> TestResult:
    root = current_project()
    try:
        run = sandbox.run_tests(root)
    except sandbox.SandboxError as exc:
        return TestResult(ok=False, exit_code=-1, output=str(exc))
    return TestResult(**run.to_dict())


__all__ = ["ActiveBuild", "current_project"]
