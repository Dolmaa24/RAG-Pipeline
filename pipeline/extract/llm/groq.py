"""The hosted backend, for when throughput matters more than locality.

The reason to reach for it is not only speed. Groq supports
``response_format: {"type": "json_schema", ...}``, where the API itself
constrains decoding to the schema. That removes the failure mode Ollama's JSON
mode cannot even detect — valid JSON with the wrong fields — without any retry
logic on this side.

The trade is that the page content leaves the machine. So this backend is never
selected when ``LOCAL_ONLY`` is set, and never silently: the choice is recorded
in provenance on every record it produces.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from config import config
from errors import ExtractError, TransientExtractError
from observability import get_logger

from .base import LLMResponse, Message, ToolRequest, ToolTurn, build_prompt, extract_json_object

log = get_logger("llm.groq")

#: Models that accept a full JSON Schema rather than only ``json_object``.
#: Groq's structured-outputs support is per-model and the list moves, so a model
#: not named here still works — it falls back to JSON mode and gets the same
#: validation Ollama's output gets. Asking an unsupported model for a schema is
#: a hard 400, which :meth:`GroqBackend._is_schema_unsupported` recovers from.
#: ``openai/gpt-oss`` was here and has been removed: asked for a schema,
#: openai/gpt-oss-120b answers 400, and while :meth:`_is_schema_unsupported`
#: recovers from that, it costs a wasted round trip on the first call of every
#: worker process and the guarantee was never real. Measured, not assumed —
#: the community reports of it *silently* ignoring the schema describe a
#: different failure than the one this account sees.
_JSON_SCHEMA_MODELS = ("moonshotai/kimi", "qwen")

#: Models the API has told us at runtime do not support schemas. Remembered per
#: process so one 400 is paid once rather than on every subsequent call.
_SCHEMA_UNSUPPORTED: set[str] = set()


class GroqBackend:
    name = "groq"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or config.GROQ_MODEL
        self._api_key = (api_key or config.GROQ_API_KEY or "").strip()
        self.timeout = config.GROQ_TIMEOUT
        self._client = None

    def available(self) -> bool:
        if not self._api_key or config.LOCAL_ONLY:
            return False
        try:
            import groq  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise ExtractError("the groq package is not installed: pip install groq") from exc
            self._client = Groq(api_key=self._api_key, timeout=self.timeout, max_retries=0)
        return self._client

    @property
    def supports_json_schema(self) -> bool:
        if self.model in _SCHEMA_UNSUPPORTED:
            return False
        return any(marker in self.model.lower() for marker in _JSON_SCHEMA_MODELS)

    def _create(self, client, user_message: str, response_format: dict):
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_message}],
            response_format=response_format,
            temperature=0.0,
            max_tokens=config.GROQ_MAX_OUTPUT_TOKENS,
        )

    def supports_tools(self) -> bool:
        """Every model Groq serves through this API accepts a tools array.

        Unlike ``json_schema``, which is per-model and moves, tool calling is a
        property of the chat API here. So this reports reachability, and a model
        that turns out not to comply fails as an ordinary API error rather than
        being guessed about in advance.
        """
        return self.available()

    def complete_with_tools(
        self,
        *,
        messages: list[Message],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> ToolTurn:
        if config.LOCAL_ONLY:
            raise ExtractError("LOCAL_ONLY is set; refusing to send content to a hosted model")

        client = self._get_client()
        started = time.perf_counter()
        try:
            # ``tool_choice`` is omitted entirely when there are no tools, not
            # sent as None. Groq rejects a null with "Only allowed string values
            # for 'tool_choice' are [none, auto, required]" -- so the toolless
            # turn, which is how the agent loop asks for a final answer once the
            # budget is spent, failed with a 400 on the one call whose whole job
            # is to salvage something from a run that has already cost a minute.
            request: dict = {
                "model": self.model,
                "messages": [_wire(message) for message in messages],
                "temperature": 0.0,
                "max_tokens": config.GROQ_MAX_OUTPUT_TOKENS,
            }
            if tools:
                request["tools"] = [_as_groq_tool(tool) for tool in tools]
                request["tool_choice"] = tool_choice
            completion = client.chat.completions.create(**request)
        except Exception as exc:
            raise self._classify(exc) from exc

        elapsed = time.perf_counter() - started
        choice = completion.choices[0]
        calls = [
            ToolRequest(
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
                call_id=call.id or "",
            )
            for call in (choice.message.tool_calls or [])
        ]

        usage = getattr(completion, "usage", None)
        turn = ToolTurn(
            calls=calls,
            text="" if calls else (choice.message.content or "").strip(),
            backend=self.name,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_seconds=elapsed,
            finish_reason=str(choice.finish_reason or ""),
        )
        log.info(
            "llm.tool_turn",
            backend=self.name,
            model=self.model,
            seconds=round(elapsed, 2),
            calls=[call.name for call in calls],
            tokens=getattr(usage, "completion_tokens", None),
        )
        return turn

    @staticmethod
    def _is_schema_unsupported(exc: Exception) -> bool:
        """A 400 specifically about ``response_format``, not any other 400."""
        if getattr(exc, "status_code", None) != 400:
            return False
        message = str(exc).lower()
        return "json_schema" in message or "response format" in message or "response_format" in message

    def complete_json(
        self,
        *,
        prompt: str,
        content: str,
        schema_hint: dict,
        json_schema: dict,
    ) -> LLMResponse:
        from .. import schema as schema_module

        if config.LOCAL_ONLY:
            raise ExtractError("LOCAL_ONLY is set; refusing to send content to a hosted model")

        client = self._get_client()
        user_message = build_prompt(prompt, schema_hint, content)
        enforced = self.supports_json_schema

        response_format: dict
        if enforced:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "extraction"),
                    "schema": json_schema,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}

        started = time.perf_counter()
        try:
            completion = self._create(client, user_message, response_format)
        except Exception as exc:
            if enforced and self._is_schema_unsupported(exc):
                # The model list moves. Rather than fail a job over a
                # capability check, drop to JSON mode and validate on this
                # side — the same guarantee Ollama gets.
                log.warning("llm.json_schema_unsupported", model=self.model)
                _SCHEMA_UNSUPPORTED.add(self.model)
                enforced = False
                try:
                    completion = self._create(client, user_message, {"type": "json_object"})
                except Exception as retry_exc:
                    raise self._classify(retry_exc) from retry_exc
            else:
                raise self._classify(exc) from exc
        elapsed = time.perf_counter() - started

        raw = (completion.choices[0].message.content or "").strip()
        if not raw:
            raise ExtractError("Groq returned an empty completion", model=self.model)

        try:
            parsed = json.loads(extract_json_object(raw))
        except json.JSONDecodeError as exc:
            raise ExtractError(
                f"Groq returned unparseable JSON ({exc}); first 200 chars: {raw[:200]}",
                model=self.model,
            ) from exc

        data = schema_module.coerce_to_schema(parsed, json_schema)
        problems = schema_module.validate(data, json_schema)

        usage = getattr(completion, "usage", None)
        result = LLMResponse(
            data=data,
            backend=self.name,
            model=self.model,
            raw=raw[:4000],
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_seconds=round(elapsed, 2),
            schema_enforced=enforced,
        )
        if problems:
            # Worth logging loudly when the API said it would enforce the shape
            # and the result does not match: that is a contract change, not a
            # model quirk.
            level = log.error if enforced else log.warning
            level("llm.schema_mismatch", backend=self.name, model=self.model, enforced=enforced,
                  problems=problems[:3])
            result.warnings = [f"schema: {issue}" for issue in problems[:5]]

        log.info(
            "llm.ok",
            backend=self.name,
            model=self.model,
            seconds=round(elapsed, 2),
            tokens=result.completion_tokens,
            schema_enforced=enforced,
        )
        return result

    @staticmethod
    def _classify(exc: Exception) -> Exception:
        """Map an SDK exception onto the pipeline's transient/permanent split."""
        name = type(exc).__name__
        message = str(exc)
        if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"):
            return TransientExtractError(f"Groq {name}: {message[:200]}")
        status = getattr(exc, "status_code", None)
        if status is not None and status >= 500:
            return TransientExtractError(f"Groq HTTP {status}: {message[:200]}")
        if status == 429:
            return TransientExtractError(f"Groq rate limited: {message[:200]}")
        if status == 413:
            # Groq answers a tokens-per-minute overage with 413, not 429. That
            # one *is* worth retrying after backoff; a genuinely oversized
            # single request is not, and no amount of waiting shrinks it.
            #
            # This only tells those apart correctly because GROQ_MAX_OUTPUT_TOKENS
            # is kept under the TPM limit. Groq reserves max_tokens against the
            # budget before running anything, so a max_tokens larger than the
            # whole allowance produced a permanent 413 that read as a rate
            # limit — and was retried, with backoff, forever.
            if "per minute" in message.lower() or "tpm" in message.lower():
                return TransientExtractError(f"Groq token-rate limit: {message[:200]}")
            return ExtractError(
                f"content too large for {getattr(exc, 'model', 'the model')}; "
                f"lower MAX_CHUNK_SIZE. {message[:200]}"
            )
        if status in (401, 403):
            return ExtractError("Groq rejected the API key; check GROQ_API_KEY")
        return ExtractError(f"Groq call failed: {message[:300]}")


def _wire(message: Message) -> dict:
    """One message in the OpenAI-compatible shape Groq expects.

    A tool result is correlated by ``tool_call_id`` here, where Ollama uses the
    tool's name. Both live on the neutral Message, so the same history renders
    correctly whichever backend the run ends up on.
    """
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }

    wire: dict = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.call_id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for index, call in enumerate(message.tool_calls)
        ]
    return wire


def _as_groq_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _parse_arguments(raw: object) -> dict:
    """Arguments as a dict. This API sends them as a JSON string."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"__unparseable__": str(raw)}
    return parsed if isinstance(parsed, dict) else {"__unparseable__": str(raw)}


__all__ = ["GroqBackend"]
