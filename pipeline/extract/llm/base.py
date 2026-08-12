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
