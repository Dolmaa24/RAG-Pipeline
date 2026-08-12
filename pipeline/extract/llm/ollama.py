"""The local backend. Private, free, and the default.

``format: "json"`` makes Ollama emit syntactically valid JSON. It says nothing
about *which* JSON — the model can return every field renamed and Ollama will
still call it a success. So the shape is checked on this side, and a mismatch
is retried once with the specific problems fed back into the prompt, which is
usually enough because the failure is almost always a naming slip rather than a
misreading.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import httpx

from config import config
from errors import ExtractError, SchemaViolation, TransientExtractError
from observability import get_logger

from .base import LLMResponse, build_prompt, extract_json_object

log = get_logger("llm.ollama")


def _format_for(json_schema: dict) -> object:
    """What to send as Ollama's ``format``.

    A schema is only useful here if it constrains something. Handing Ollama an
    open ``{"type": "object", "additionalProperties": true}`` builds a grammar
    that permits *any* JSON, and constrained decoding then happily generates
    until it hits ``num_predict`` — a five-minute timeout instead of a
    two-second answer. So an unconstrained schema falls back to plain JSON mode,
    where the model stops at the closing brace like it should.
    """
    if isinstance(json_schema, dict) and json_schema.get("properties"):
        return json_schema
    return "json"


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None) -> None:
        self.model = model or config.AI_MODEL_NAME
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.timeout = config.AI_TIMEOUT

    # ------------------------------------------------------------------ #

    def available(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            if response.status_code != 200:
                return False
            names = {entry.get("name", "") for entry in response.json().get("models", [])}
            # Ollama reports "llama3:latest" for a model pulled as "llama3".
            return any(name.split(":")[0] == self.model.split(":")[0] for name in names)
        except Exception:
            return False

    def complete_json(
        self,
        *,
        prompt: str,
        content: str,
        schema_hint: dict,
        json_schema: dict,
    ) -> LLMResponse:
        from .. import schema as schema_module

        full_prompt = build_prompt(prompt, schema_hint, content)
        problems: list[str] = []

        for attempt in range(1, config.LLM_MAX_ATTEMPTS + 1):
            if problems:
                full_prompt = (
                    f"{full_prompt}\n\nYour previous answer was rejected:\n"
                    + "\n".join(f"- {issue}" for issue in problems[:8])
                    + "\nReturn the corrected JSON object, using exactly the schema's field names."
                )

            started = time.perf_counter()
            payload = self._call(full_prompt, json_schema)
            elapsed = time.perf_counter() - started

            raw = payload.get("response", "")
            if not raw.strip():
                raise ExtractError("Ollama returned an empty response", model=self.model)

            try:
                parsed = json.loads(extract_json_object(raw))
            except json.JSONDecodeError as exc:
                if attempt >= config.LLM_MAX_ATTEMPTS:
                    raise ExtractError(
                        f"model did not return valid JSON ({exc}); first 200 chars: {raw[:200]}",
                        model=self.model,
                    ) from exc
                problems = [f"the reply was not valid JSON: {exc}"]
                continue

            data = schema_module.coerce_to_schema(parsed, json_schema)
            problems = schema_module.validate(data, json_schema)

            if not problems or attempt >= config.LLM_MAX_ATTEMPTS:
                response = LLMResponse(
                    data=data,
                    backend=self.name,
                    model=self.model,
                    raw=raw[:4000],
                    prompt_tokens=payload.get("prompt_eval_count"),
                    completion_tokens=payload.get("eval_count"),
                    latency_seconds=round(elapsed, 2),
                    schema_enforced=False,
                )
                if problems:
                    # Returned rather than raised: a partially-correct record
                    # with its faults recorded beats no record at all, and the
                    # caller decides whether to store it.
                    response.warnings = [f"schema: {issue}" for issue in problems[:5]]
                    log.warning(
                        "llm.schema_mismatch",
                        backend=self.name,
                        model=self.model,
                        problems=len(problems),
                    )
                log.info(
                    "llm.ok",
                    backend=self.name,
                    model=self.model,
                    seconds=round(elapsed, 1),
                    tokens=payload.get("eval_count"),
                )
                return response

            log.info("llm.retrying_for_schema", attempt=attempt, problems=len(problems))

        raise SchemaViolation("; ".join(problems[:5]))  # pragma: no cover - loop always returns

    # ------------------------------------------------------------------ #

    def _call(self, full_prompt: str, json_schema: dict) -> dict:
        body = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            # Recent Ollama accepts a JSON Schema here and constrains decoding
            # to it. Older builds only understand the string "json" and ignore
            # an object, which is why the shape is still validated afterwards.
            "format": _format_for(json_schema),
            "options": {
                "temperature": 0.0,  # extraction is not a creative task
                "num_predict": 4096,
            },
        }
        try:
            response = httpx.post(f"{self.host}/api/generate", json=body, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise TransientExtractError(
                f"cannot reach Ollama at {self.host}. Is `ollama serve` running? ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise TransientExtractError(f"Ollama timed out after {self.timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise TransientExtractError(f"Ollama request failed: {exc}") from exc

        if response.status_code == 404:
            raise ExtractError(
                f"model {self.model!r} is not pulled. Run: ollama pull {self.model}",
                model=self.model,
            )
        if response.status_code >= 500:
            raise TransientExtractError(f"Ollama returned {response.status_code}")
        if response.status_code != 200:
            raise ExtractError(
                f"Ollama returned {response.status_code}: {response.text[:200]}", model=self.model
            )
        return response.json()


__all__ = ["OllamaBackend"]
