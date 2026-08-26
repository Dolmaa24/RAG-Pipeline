"""One interface over local and hosted models, chosen per job.

The two backends are not interchangeable, and pretending otherwise is how you
end up with the wrong one:

======================  ===============================  ==========================
                        Ollama (local)                   Groq (hosted)
======================  ===============================  ==========================
Latency                 seconds to minutes               sub-second typical
Cost                    free                             per token, cheap at the
                                                         small-model end
Privacy                 content never leaves the Mac     content leaves the Mac
Schema guarantee        ``format: "json"`` — valid       ``response_format:
                        JSON, *not necessarily your      json_schema`` — conforms
                        shape*                           to the schema
======================  ===============================  ==========================

So: **Ollama by default** (private, free, already working), **Groq when
throughput matters and the content is not sensitive**, and ``LOCAL_ONLY=true``
as a hard switch for content that must never leave the machine.

Schema validation runs regardless of backend. Groq's guarantee is real but it
is a guarantee about *the API's* behaviour, and a pipeline that trusts an
external promise without checking it has no way to notice the day it changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

SYSTEM_PROMPT = """You are a strict JSON extraction assistant.

Extract structured data from the CONTENT below according to the INSTRUCTION,
and return ONLY a JSON object matching the SCHEMA.

Rules:
- Return exactly the fields in the schema. No extra fields, no commentary, no
  markdown fences.
- Copy values from the content. Do not infer, calculate, or complete them from
  your own knowledge.
- If a value is genuinely not present in the content, use null. A null is a
  correct answer; an invented value is not.
- Preserve the original wording and units of what you extract.

SCHEMA:
{schema}

INSTRUCTION: {prompt}
"""


@dataclass(slots=True)
class LLMResponse:
    """One completed model call."""

    data: dict[str, Any]
    backend: str
    model: str
    raw: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_seconds: float = 0.0
    #: True when the backend itself guaranteed the shape (Groq json_schema),
    #: as opposed to us having validated it afterwards.
    schema_enforced: bool = False
    warnings: list[str] = field(default_factory=list)


# Tool calling
#
# complete_json() answers one question and forgets it: content in, shape out.
# An agent needs the other thing — "call this, then let me see what it
# returned" — which means a conversation with a history, and a turn that is
# either tool calls or an answer.
#
# The tool descriptions are passed as plain dicts, not as the registry's
# ToolSpec. This layer must not import pipeline.agents: the agents are built on
# the model layer, and an import the other way would make the two circular and
# the model layer untestable on its own.


@dataclass(slots=True)
class ToolRequest:
    """One tool the model asked for, with the arguments it chose."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Present on APIs that correlate results back to calls (Groq/OpenAI).
    #: Ollama does not issue one, and the empty string is the honest value.
    call_id: str = ""


@dataclass(slots=True)
class Message:
    """One turn of the conversation, in a form both wire formats can render.

    Kept neutral rather than storing each backend's own dict, because a history
    built against one backend would otherwise be unusable if the run failed
    over to the other — which is exactly when it matters.
    """

    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolRequest] = field(default_factory=list)
    #: For role="tool": which call this answers.
    tool_call_id: str = ""
    tool_name: str = ""

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def observation(cls, request: "ToolRequest", content: str) -> "Message":
        """What a tool returned, addressed to the call that asked for it."""
        return cls(
            role="tool",
            content=content,
            tool_call_id=request.call_id,
            tool_name=request.name,
        )


@dataclass(slots=True)
class ToolTurn:
    """One model turn: either tool calls to run, or final text. Never both.

    A model that emits both is answering and asking at once, and acting on
    either half is a guess. The backends drop the text when calls are present,
    which is what every tool-calling API means by it anyway.
    """

    calls: list[ToolRequest] = field(default_factory=list)
    text: str = ""
    backend: str = ""
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_seconds: float = 0.0
    #: False when the shim parsed a JSON object instead of the model emitting
    #: real tool calls. Worth recording: a trace that cannot tell those apart
    #: cannot explain why one model behaves worse than another.
    native: bool = True
    finish_reason: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.calls)

    def as_message(self) -> Message:
        """This turn, ready to append to the history it came from."""
        return Message(role="assistant", content=self.text, tool_calls=list(self.calls))


class LLMBackend(Protocol):
    name: str
    model: str

    def available(self) -> bool:
        """Cheap reachability check. Must not raise."""
        ...

    def complete_json(
        self,
        *,
        prompt: str,
        content: str,
        schema_hint: dict,
        json_schema: dict,
    ) -> LLMResponse:
        """Return a JSON object for this content. Raises on failure."""
        ...

    def supports_tools(self) -> bool:
        """Whether this backend can be given tools natively. Must not raise."""
        ...

    def complete_with_tools(
        self,
        *,
        messages: list[Message],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> ToolTurn:
        """One turn: tool calls to run, or the final text. Raises on failure.

        ``tools`` are dicts of ``{name, description, input_schema}`` — the shape
        :meth:`pipeline.agents.tools.registry.ToolSpec.describe` already returns.
        """
        ...


def build_prompt(prompt: str, schema_hint: dict, content: str) -> str:
    from .. import schema as schema_module

    header = SYSTEM_PROMPT.format(
        schema=schema_module.describe_for_prompt(schema_hint), prompt=prompt
    )
    return f"{header}\nCONTENT:\n{content}"


def strip_fences(raw: str) -> str:
    """Remove ```json fences that models add despite being told not to."""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(raw: str) -> str:
    """Pull the outermost JSON value out of a chatty reply.

    Some models prepend "Here is the JSON:" no matter what the prompt says.
    Scanning for the first balanced object is more reliable than a regex,
    because braces appear inside string values too.
    """
    text = strip_fences(raw)
    if text.startswith(("{", "[")):
        return text

    start = min(
        (index for index in (text.find("{"), text.find("[")) if index != -1),
        default=-1,
    )
    if start == -1:
        return text

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


__all__ = [
    "SYSTEM_PROMPT",
    "LLMBackend",
    "LLMResponse",
    "build_prompt",
    "extract_json_object",
    "strip_fences",
]
