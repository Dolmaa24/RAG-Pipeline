"""The tool catalog, spoken as MCP.

An adapter, not a second catalog. Every tool advertised here is read from the
registry in :mod:`pipeline.agents.tools` at startup, so a tool added there
appears here without anything being written twice — which is the whole reason
the registry exists.

Two things are worth understanding before changing this file.

**Schemas are flattened deliberately.** The registry's handlers each take one
Pydantic model, and registering such a function with the MCP SDK produces a
nested schema — ``{"args": {"$ref": ...}}``. Phase 00 chose flat arguments
because small models mishandle nesting, and the tool-calling benchmark measured
that choice at 90–100% argument accuracy across two 3B models. Handing them a
nested schema would give that back. So :func:`_flat` synthesises a signature
whose parameters are the model's own fields, and the SDK generates the flat
schema from that.

**Tools beyond the allowance are not advertised.** A client shown a tool it
cannot call will call it, spend a turn learning that, and often try again. So
network and write tools are absent from ``tools/list`` entirely unless the
matching setting is on, rather than listed and refused.

Run:
    ./run.sh mcp                  # stdio, for Claude Desktop and Claude Code
    ./run.sh mcp --http           # streamable HTTP on MCP_HTTP_PORT
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Annotated, Any, Callable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pydantic import BaseModel, Field

from config import config
from observability import configure_logging, get_logger

from pipeline.agents.tools import Effect, ToolSpec, catalog, get, invoke

configure_logging(config.LOG_LEVEL, config.LOG_FORMAT)
log = get_logger("mcp")

INSTRUCTIONS = """\
This server searches a corpus built by a web-extraction pipeline, and — when
permitted — adds to it.

Start with corpus_profile to see what is actually indexed and which filter
values exist. Use search_corpus for passages you want to reason over yourself,
and answer_from_corpus when a written answer with citations is what is wanted.
graph_neighbors and graph_path answer questions about how entities relate,
which vector search answers badly.

Tools that fetch or write return a task id rather than blocking; poll it with
poll_task before searching for what they produced.\
"""


def allowed_effects() -> list[Effect]:
    """Which effects this server exposes, from settings alone.

    Never widened at runtime and never by a request. Deciding to fetch the
    outside world is not something a client earns by asking.
    """
    effects = [Effect.READ]
    if config.MCP_ALLOW_NETWORK:
        effects.append(Effect.NETWORK)
    if config.MCP_ALLOW_WRITE:
        effects.append(Effect.WRITE)
    return effects


def _annotation_of(field: Any) -> Any:
    """The field's type, carrying its description and bounds into the schema.

    Without this the generated schema keeps only names and types, and the
    per-argument descriptions — which is where a small model learns what
    ``alpha`` or ``hops`` means — are silently dropped.
    """
    annotation = field.annotation
    extras = list(field.metadata)
    if field.description:
        extras.append(Field(description=field.description))
    return Annotated[tuple([annotation, *extras])] if extras else annotation


def _flat(spec: ToolSpec, effects: list[Effect]) -> Callable[..., str]:
    """A callable whose signature is the tool's fields, one level deep."""
    parameters = []
    for name, field in spec.args_model.model_fields.items():
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if field.is_required() else field.default,
                annotation=_annotation_of(field),
            )
        )

    def handler(**kwargs: Any) -> str:
        # invoke() validates, enforces the effect, runs the handler and turns
        # every failure into text. Nothing here needs a try block, and adding
        # one would hide the explanation the model is meant to read.
        call = invoke(spec.name, kwargs, allowed=effects)
        log.info(
            "mcp.call", tool=spec.name, ok=call.ok, ms=round(call.duration_ms, 1)
        )
        return call.observation

    handler.__name__ = spec.name
    handler.__signature__ = inspect.Signature(parameters, return_annotation=str)
    handler.__annotations__ = {p.name: p.annotation for p in parameters} | {"return": str}
    handler.__doc__ = spec.description
    return handler


def build_server(effects: list[Effect] | None = None):
    """The server, with one MCP tool per registry tool it may expose."""
    from mcp.server.mcpserver import MCPServer

    effects = effects or allowed_effects()
    server = MCPServer(
        name="universal-extraction-corpus",
        title="Universal Extraction Corpus",
        instructions=INSTRUCTIONS,
        website_url="https://github.com/Dolmaa24/WebScraper",
    )

    exposed = catalog(effects)
    for spec in exposed:
        server.add_tool(
            _flat(spec, effects),
            name=spec.name,
            description=spec.description,
            # Structured output would re-serialise the result as JSON. These
            # handlers already return text shaped for a model to read, with the
            # token budget spent where it matters.
            structured_output=False,
        )

    _add_resources(server)

    log.info(
        "mcp.ready",
        tools=len(exposed),
        effects=[e.value for e in effects],
        names=[s.name for s in exposed],
    )
    return server


def _add_resources(server) -> None:
    """One passage, addressable by URI, so a client can cite a chunk directly."""

    @server.resource("corpus://chunk/{chunk_id}")
    def chunk(chunk_id: str) -> str:
        """The full text of one indexed passage, by its chunk_id."""
        call = invoke("fetch_chunk", {"chunk_id": chunk_id}, allowed=[Effect.READ])
        return call.observation

    @server.resource("corpus://profile")
    def profile() -> str:
        """What the corpus holds, and which filter values exist."""
        return invoke("corpus_profile", {}, allowed=[Effect.READ]).observation


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server over the corpus.")
    parser.add_argument(
        "--http", action="store_true",
        help="Serve streamable HTTP instead of stdio.",
    )
    parser.add_argument("--port", type=int, default=config.MCP_HTTP_PORT)
    args = parser.parse_args()

    server = build_server()

    if args.http:
        import uvicorn

        uvicorn.run(
            server.streamable_http_app(),
            host="127.0.0.1",
            port=args.port,
            log_level=config.LOG_LEVEL.lower(),
        )
    else:
        # stdio: stdout is the protocol channel, so nothing may print to it.
        # Logging is already configured to stderr.
        server.run()


if __name__ == "__main__":
    main()
