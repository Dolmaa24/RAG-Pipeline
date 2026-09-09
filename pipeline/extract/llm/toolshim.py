"""Tool calling for models that cannot tool-call.

Not every model this pipeline can reach speaks a tools array. Older Ollama
builds, quantised community models, and anything reached through a plain
completion endpoint all answer the same way: they ignore the tools and write
prose. Making tool calling a hard requirement would mean the agent layer works
on some of the stack and not the rest, and the part it did not work on is the
part someone runs on a laptop with whatever they already pulled.

So this asks for the same decision as a constrained JSON object instead:

    {"tool": "search_corpus", "arguments": {...}}     -- call this next
    {"tool": null, "answer": "..."}                   -- done, here it is

That goes through ``complete_json``, which every backend already has and which
Ollama constrains with a real grammar. It is slower than native tool calling —
the whole catalog is re-rendered into the prompt every turn, where a tools array
is sent once as structured data — and it is worse at multi-step work, because
the conversation has to be flattened into text. It is a fallback, reached
either when a backend admits it cannot tool-call or when one that claimed it
could is caught writing a call out as prose -- see :class:`AdaptiveToolBackend`.

``native=False`` on the resulting turn is not decoration. A trace that cannot
tell a real tool call from a parsed one cannot explain why a run went badly.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from observability import get_logger

from .base import LLMBackend, Message, ToolRequest, ToolTurn, extract_json_object

log = get_logger("llm.toolshim")

#: Deliberately not a free-form object: ``arguments`` is left unconstrained
#: because it differs per tool, but the envelope around it is pinned so the
#: reply can be parsed without guessing.
_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "next_action",
    "properties": {
        "tool": {
            "type": ["string", "null"],
            "description": "Name of the tool to call, or null to answer now.",
        },
        "arguments": {
            # Nullable because a model answering rather than calling writes
            # null here, which is the honest value. Requiring an object made
            # every answer fail validation and pay a retry to say the same
            # thing again.
            "type": ["object", "null"],
            "description": "Arguments for that tool. Null when tool is null.",
        },
        "answer": {
            # Nullable for the same reason as ``arguments``: exactly one of the
            # two halves is used on any turn, and null is what a model writes
            # for the other one. Insisting on a string here failed every turn
            # that chose a tool.
            "type": ["string", "null"],
            "description": "The final answer. Null unless tool is null.",
        },
    },
    "required": ["tool"],
}

_INSTRUCTION = """\
Decide the single next action.

Return {"tool": "<name>", "arguments": {...}} to call one tool, or
{"tool": null, "answer": "..."} to answer now. Never both. Choose a tool only
from the list; if none of them fits, answer instead and say why.
"""


def render_tools(tools: list[dict]) -> str:
    """The catalog as text, since there is nowhere structured to put it.

    Name and description first, arguments after. The first shape this took was
    a signature line — ``search_corpus(doc_type?, department?, author?, …,
    query, limit?)`` — with the description underneath, and against that
    rendering qwen2.5:3b answered "the provided tools do not include a function
    to answer questions about quarterly revenue" while looking straight at
    ``search_corpus``. Nine optional filters ahead of the one required argument
    buried the only part that says what the tool is for.
    """
    lines = []
    for tool in tools:
        schema = tool.get("input_schema") or {}
        properties = schema.get("properties") or {}
        required = [name for name in properties if name in set(schema.get("required") or [])]
        optional = [name for name in properties if name not in set(required)]

        block = [f"- {tool['name']}: {tool.get('description', '')}"]
        if required:
            block.append(
                "    required: "
                + ", ".join(f"{name} ({_type_of(properties[name])})" for name in required)
            )
        if optional:
            block.append("    optional: " + ", ".join(optional))
        lines.append("\n".join(block))
    return "\n".join(lines)


def _type_of(spec: dict) -> str:
    """A readable type, including for the anyOf an optional field generates."""
    if "type" in spec:
        kind = spec["type"]
        return kind if isinstance(kind, str) else "/".join(k for k in kind if k != "null")
    for option in spec.get("anyOf", []):
        if option.get("type") and option["type"] != "null":
            return str(option["type"])
    return "any"


def render_conversation(messages: list[Message]) -> str:
    """The history as text. Tool results are labelled by the tool that ran."""
    lines = []
    for message in messages:
        if message.role == "tool":
            lines.append(f"[result of {message.tool_name or 'tool'}]\n{message.content}")
        elif message.role == "assistant" and message.tool_calls:
            called = ", ".join(
                f"{call.name}({json.dumps(call.arguments)})" for call in message.tool_calls
            )
            lines.append(f"[you called] {called}")
        elif message.content:
            lines.append(f"[{message.role}] {message.content}")
    return "\n\n".join(lines)


class ShimmedBackend:
    """Wraps a backend, adding :meth:`complete_with_tools` built on JSON mode."""

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self.name = f"{inner.name}+shim"
        self.model = inner.model

    def __getattr__(self, item: str) -> Any:
        # Everything else — available(), complete_json(), host — is the inner
        # backend's. Only the tool-calling half is added here.
        return getattr(self._inner, item)

    def supports_tools(self) -> bool:
        return True

    def complete_with_tools(
        self,
        *,
        messages: list[Message],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> ToolTurn:
        started = time.perf_counter()
        response = self._inner.complete_json(
            prompt=_INSTRUCTION,
            content=(
                f"TOOLS:\n{render_tools(tools)}\n\n"
                f"CONVERSATION SO FAR:\n{render_conversation(messages)}"
            ),
            schema_hint={"tool": "string or null", "arguments": "object", "answer": "string"},
            json_schema=_ACTION_SCHEMA,
        )
        elapsed = time.perf_counter() - started

        turn = _turn_from(response.data, tools)
        turn.backend = self.name
        turn.model = self.model
        turn.prompt_tokens = response.prompt_tokens
        turn.completion_tokens = response.completion_tokens
        turn.latency_seconds = elapsed
        turn.native = False

        log.info(
            "llm.tool_turn",
            backend=self.name,
            model=self.model,
            seconds=round(elapsed, 1),
            calls=[call.name for call in turn.calls],
            native=False,
        )
        return turn


def _turn_from(data: dict, tools: list[dict]) -> ToolTurn:
    """One parsed action, or an answer.

    A named tool that is not in the catalog is *not* turned into a call. The
    registry would refuse it anyway, but spending a turn on a refusal for
    something detectable here is waste — and unlike a model's own tool calls,
    this name came out of free-form generation, where invention is likelier.
    """
    name = data.get("tool")
    if not name or not isinstance(name, str):
        return ToolTurn(text=str(data.get("answer") or "").strip())

    known = {tool["name"] for tool in tools}
    if name not in known:
        return ToolTurn(
            text=(
                f"No tool named {name!r} exists. Available: "
                f"{', '.join(sorted(known))}."
            ),
            finish_reason="unknown_tool",
        )

    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolTurn(calls=[ToolRequest(name=name, arguments=arguments)])


def _looks_like_a_missed_call(text: str, tools: list[dict]) -> bool:
    """Whether a call-less turn wrote the *shape* of a tool call as prose.

    A model can advertise ``tools`` and still emit
    ``{"name": "corpus_profile", "parameters": {"`` as ordinary content --
    often truncated, always invisible to the tools array. ``llama3.2:3b`` does
    exactly this against an eight-tool catalog. Native calling has silently
    failed, and asking the same way again cannot fix it.

    Deliberately narrow. An answer that merely mentions a tool by name is not
    this: the text has to *contain a JSON object* naming a tool from the
    catalog before a run pays for a second call.

    Contain, not open with. The first version of this required the text to
    start with ``{``, and neither model that needs it does. Measured against
    the same worker prompt:

        qwen2.5-coder:3b   ```json\n{"name": "write_source", ...
        llama3.2:3b        Here are the JSON function calls...\n\n{"name": ...

    A fence and a sentence of preamble, so the check never fired and every one
    of those turns was discarded. ``base`` already carries the two helpers for
    exactly this shape, and this reuses them rather than adding a third
    opinion about where a model's JSON begins.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    try:
        candidate = extract_json_object(stripped)
    except Exception:
        return False
    if not candidate.startswith(("{", "[")):
        return False
    return any(tool.get("name", "") in candidate for tool in tools)


class AdaptiveToolBackend:
    """Native tool calling, falling back to the shim once native visibly fails.

    ``supports_tools()`` asks the model what it can do, which is a claim about
    the template rather than a measurement of the model. A small model can hold
    the capability and still be unable to use it, and the failure is silent:
    no calls, no error, a turn of prose that reads like a call. The loop then
    burns its whole round budget re-asking a question that will not land.

    So the claim is trusted exactly once. The first turn that comes back
    call-less holding the shape of a call switches this backend to constrained
    decoding for the rest of its life, which is measurably what the same model
    needed all along. Models that really do tool-call never trip it and pay
    nothing: no probe, no extra call, no re-rendered catalog.

    The switch is permanent by design. A model that cannot tool-call on turn
    one will not learn to by turn three, and flapping between the two paths
    would make a trace impossible to read.
    """

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self._shim = ShimmedBackend(inner)
        self._native = True
        self.name = inner.name
        self.model = inner.model

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def supports_tools(self) -> bool:
        return True

    def complete_with_tools(
        self,
        *,
        messages: list[Message],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> ToolTurn:
        if not self._native:
            return self._shim.complete_with_tools(
                messages=messages, tools=tools, tool_choice=tool_choice
            )

        turn = self._inner.complete_with_tools(
            messages=messages, tools=tools, tool_choice=tool_choice
        )
        if turn.calls or not _looks_like_a_missed_call(turn.text, tools):
            return turn

        # Not a warning about this turn -- the turn is retried and lands. It is
        # a warning about the model, which claimed something untrue.
        log.warning(
            "llm.tool_native_failed",
            backend=self._inner.name,
            model=self.model,
            wrote=turn.text[:120],
        )
        self._native = False
        self.name = self._shim.name
        return self._shim.complete_with_tools(
            messages=messages, tools=tools, tool_choice=tool_choice
        )


def shim_if_needed(backend: LLMBackend) -> LLMBackend:
    """Wrapped for constrained decoding now, or ready to be if native fails."""
    supports = getattr(backend, "supports_tools", None)
    try:
        native = bool(supports()) if callable(supports) else False
    except Exception as exc:  # a capability probe must never fail a run
        log.warning("llm.tool_probe_failed", backend=backend.name, error=repr(exc))
        native = False

    if native:
        return AdaptiveToolBackend(backend)

    log.info("llm.tool_shim_engaged", backend=backend.name, model=backend.model)
    return ShimmedBackend(backend)


__all__ = [
    "AdaptiveToolBackend",
    "ShimmedBackend",
    "shim_if_needed",
    "render_tools",
    "render_conversation",
]
