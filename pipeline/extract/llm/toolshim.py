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
the conversation has to be flattened into text. It is a fallback, and
:func:`shim_if_needed` only reaches for it when the backend says it must.

``native=False`` on the resulting turn is not decoration. A trace that cannot
tell a real tool call from a parsed one cannot explain why a run went badly.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from observability import get_logger

from .base import LLMBackend, Message, ToolRequest, ToolTurn

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


def shim_if_needed(backend: LLMBackend) -> LLMBackend:
    """The backend as-is when it tool-calls natively, wrapped when it does not."""
    supports = getattr(backend, "supports_tools", None)
    try:
        native = bool(supports()) if callable(supports) else False
    except Exception as exc:  # a capability probe must never fail a run
        log.warning("llm.tool_probe_failed", backend=backend.name, error=repr(exc))
        native = False

    if native:
        return backend

    log.info("llm.tool_shim_engaged", backend=backend.name, model=backend.model)
    return ShimmedBackend(backend)


__all__ = ["ShimmedBackend", "shim_if_needed", "render_tools", "render_conversation"]
