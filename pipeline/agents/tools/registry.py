"""One catalog of capabilities, described once and consumed twice.

An MCP server and an in-process agent loop need the same three things about a
capability: a name, a JSON Schema for its arguments, and something to call. The
temptation is to write that twice — once in the MCP server and once in whatever
drives the loop — and the two copies drift within a month. So they are declared
here, and both consumers read from this registry.

Every tool also declares an :class:`Effect`. That is not documentation. Handlers
are unreachable through :func:`invoke` unless the caller passes the matching
effect in ``allowed``, and the default is read-only. An agent cannot decide to
start crawling a site, because deciding is not what grants the permission — the
request that started the run is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Protocol, Type

from pydantic import BaseModel, ValidationError

from observability import get_logger, metrics

log = get_logger("agents.tools")


class Effect(str, Enum):
    """What calling a tool can change. Ordered by how much trust it needs."""

    #: Reads what is already indexed. Always available.
    READ = "read"
    #: Reaches the public internet. Off unless the request opted in.
    NETWORK = "network"
    #: Changes the corpus. Off unless the request opted in, and returns a task
    #: id rather than blocking.
    WRITE = "write"


#: What a caller gets by default. Deliberately the least dangerous set: a run
#: that never says otherwise can only read things already fetched under a policy
#: someone already agreed to.
READ_ONLY: frozenset[Effect] = frozenset({Effect.READ})


class ToolResult(Protocol):
    """What a handler returns.

    Two renderings, because there are two audiences. ``render()`` is what goes
    back to a model — prose it can quote and cite. ``model_dump()`` is what goes
    to a dashboard or an MCP client's structured content, where scores and ids
    matter and prose does not.
    """

    def render(self, *, max_chars: int) -> str: ...
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


class ToolError(RuntimeError):
    """A tool could not do its job. Reported to the model, not raised at it."""


class EffectNotAllowed(ToolError):
    def __init__(self, name: str, effect: Effect) -> None:
        super().__init__(
            f"{name!r} has effect {effect.value!r}, which this run did not allow"
        )
        self.tool = name
        self.effect = effect


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Everything both consumers need to know about one capability."""

    name: str
    description: str
    effect: Effect
    #: Rough wall-clock, for a budget to reason about before spending it.
    cost_ms: int
    args_model: Type[BaseModel]
    handler: Callable[[BaseModel], Any]
    #: Cap on the rendered observation. Truncating here rather than at the
    #: prompt boundary is deliberate: an 8000 TPM ceiling is spent by whatever
    #: enters the message history, and by then it is already too late to choose.
    max_chars: int = 2000

    def json_schema(self) -> dict[str, Any]:
        """The argument schema, as a tool-calling API wants it."""
        schema = self.args_model.model_json_schema()
        # Pydantic emits $defs/$ref for nested models. Tool-calling endpoints
        # accept them, but small models handle a flat schema markedly better,
        # so every argument model here is deliberately flat and this is a guard
        # against that quietly stopping being true.
        if "$defs" in schema:
            raise ValueError(
                f"{self.name}: argument models must be flat — nested objects "
                "measurably hurt tool selection on small models"
            )
        schema.pop("title", None)
        return schema

    def describe(self) -> dict[str, Any]:
        """The tool as an API advertises it."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema(),
        }


_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    *,
    name: str,
    description: str,
    effect: Effect,
    cost_ms: int,
    max_chars: int = 2000,
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    """Register a handler as a callable tool.

    The argument model is taken from the handler's own annotation, so the schema
    and the signature cannot disagree.

    ``description`` is written for a model, not a person. A small model picks
    almost entirely from this text, so naming the tool to call first — "call
    corpus_profile before filtering" — is worth more here than the same sentence
    in a system prompt, where it competes with everything else.
    """

    def decorate(handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
        args_model = _args_model_of(handler)
        spec = ToolSpec(
            name=name,
            description=description.strip(),
            effect=effect,
            cost_ms=cost_ms,
            args_model=args_model,
            handler=handler,
            max_chars=max_chars,
        )
        if name in _REGISTRY:
            raise ValueError(f"tool {name!r} is already registered")
        spec.json_schema()  # fail at import, not at the first call
        _REGISTRY[name] = spec
        return handler

    return decorate


def _args_model_of(handler: Callable[[Any], Any]) -> Type[BaseModel]:
    from typing import get_type_hints

    try:
        hints = get_type_hints(handler)
    except NameError as exc:
        # Modules here use `from __future__ import annotations`, so the
        # annotation is a string resolved against module globals. A model
        # defined inside a function is not there, and the bare NameError that
        # results says nothing about why.
        raise ValueError(
            f"{handler.__name__}: could not resolve its argument annotation "
            f"({exc}). The argument model must be defined at module level."
        ) from exc
    hints.pop("return", None)
    if len(hints) != 1:
        raise ValueError(
            f"{handler.__name__} must take exactly one annotated argument, "
            f"a BaseModel; got {sorted(hints)}"
        )
    model = next(iter(hints.values()))
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ValueError(f"{handler.__name__}'s argument must be a BaseModel")
    return model


def get(name: str) -> ToolSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ToolError(
            f"no tool named {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def catalog(allowed: Iterable[Effect] = READ_ONLY) -> list[ToolSpec]:
    """The tools a run may use, in a stable order.

    A tool the caller cannot use is left out rather than advertised and refused.
    A model shown a tool it is not allowed to call will call it, spend a turn
    learning that, and often try again.
    """
    permitted = frozenset(allowed)
    return [
        spec
        for _, spec in sorted(_REGISTRY.items())
        if spec.effect in permitted
    ]


def describe_all(allowed: Iterable[Effect] = READ_ONLY) -> list[dict[str, Any]]:
    return [spec.describe() for spec in catalog(allowed)]


def clear() -> None:
    """Empty the registry. For tests that register throwaway tools."""
    _REGISTRY.clear()


@dataclass(slots=True)
class ToolCall:
    """One completed call, in the shape a trace and a message history both want."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    observation: str
    duration_ms: float
    effect: Effect
    data: Optional[dict[str, Any]] = None
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "observation": self.observation,
            "duration_ms": self.duration_ms,
            "effect": self.effect.value,
            "error": self.error,
            **({"meta": self.meta} if self.meta else {}),
        }


def invoke(
    name: str,
    arguments: dict[str, Any],
    *,
    allowed: Iterable[Effect] = READ_ONLY,
) -> ToolCall:
    """Run one tool and return what happened, successfully or not.

    Never raises for anything the model did wrong. A bad tool name, invalid
    arguments and a handler that failed are all observations — the loop's job is
    to let the model read the problem and try something else, and an exception
    would end the run instead of teaching it anything.
    """
    started = time.perf_counter()
    permitted = frozenset(allowed)

    try:
        spec = get(name)
    except ToolError as exc:
        return _failed(name, arguments, str(exc), started, Effect.READ)

    if spec.effect not in permitted:
        exc = EffectNotAllowed(name, spec.effect)
        log.warning("agents.tools.refused", tool=name, effect=spec.effect.value)
        metrics.incr("agents.tools.refused")
        return _failed(name, arguments, str(exc), started, spec.effect)

    try:
        args = spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        return _failed(name, arguments, _explain(exc), started, spec.effect)

    try:
        with metrics.timer(f"agents.tool.{name}"):
            result = spec.handler(args)
    except Exception as exc:
        log.warning("agents.tools.failed", tool=name, error=repr(exc))
        metrics.incr("agents.tools.failed")
        return _failed(name, arguments, f"{name} failed: {exc}", started, spec.effect)

    duration = round((time.perf_counter() - started) * 1000, 2)
    log.info("agents.tools.ok", tool=name, ms=duration, effect=spec.effect.value)
    metrics.incr("agents.tools.ok")

    return ToolCall(
        name=name,
        arguments=args.model_dump(exclude_none=True),
        ok=True,
        observation=result.render(max_chars=spec.max_chars),
        duration_ms=duration,
        effect=spec.effect,
        data=result.model_dump(),
    )


def _failed(
    name: str,
    arguments: dict[str, Any],
    reason: str,
    started: float,
    effect: Effect,
) -> ToolCall:
    return ToolCall(
        name=name,
        arguments=arguments,
        ok=False,
        observation=reason,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        effect=effect,
        error=reason,
    )


def _explain(exc: ValidationError) -> str:
    """A validation failure in a sentence a model can act on."""
    parts = []
    for error in exc.errors()[:4]:
        where = ".".join(str(p) for p in error["loc"]) or "arguments"
        parts.append(f"{where}: {error['msg']}")
    return "invalid arguments — " + "; ".join(parts)


__all__ = [
    "READ_ONLY",
    "Effect",
    "EffectNotAllowed",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "catalog",
    "clear",
    "describe_all",
    "get",
    "invoke",
    "tool",
]
