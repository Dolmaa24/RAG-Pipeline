"""The MCP adapter: what it advertises, and in what shape.

Two properties matter more than the rest, and both have already been broken
once in development:

* **Flat schemas.** Registering the registry's handlers directly produces
  ``{"args": {"$ref": ...}}``, because each takes one Pydantic model. Phase 00
  chose flat arguments because 3B models mishandle nesting, and the tool-calling
  benchmark measures that choice at 90–100% argument accuracy. A nested schema
  would give it back silently — nothing would fail, models would just get worse.
* **Tools beyond the allowance are absent, not refused.** A client shown a tool
  it cannot call will call it and spend a turn learning that.
"""

from __future__ import annotations

import json

import pytest

import mcp_server
from pipeline.agents.tools import Effect, catalog


ALL_EFFECTS = [Effect.READ, Effect.NETWORK, Effect.WRITE]


# --------------------------------------------------------------------------- #
# What is exposed
# --------------------------------------------------------------------------- #


def test_read_only_by_default(monkeypatch):
    monkeypatch.setattr(mcp_server.config, "MCP_ALLOW_NETWORK", False)
    monkeypatch.setattr(mcp_server.config, "MCP_ALLOW_WRITE", False)
    assert mcp_server.allowed_effects() == [Effect.READ]


def test_network_permission_does_not_grant_write(monkeypatch):
    monkeypatch.setattr(mcp_server.config, "MCP_ALLOW_NETWORK", True)
    monkeypatch.setattr(mcp_server.config, "MCP_ALLOW_WRITE", False)
    effects = mcp_server.allowed_effects()
    assert Effect.NETWORK in effects
    assert Effect.WRITE not in effects


@pytest.mark.anyio
async def test_write_tools_are_absent_not_refused():
    server = mcp_server.build_server([Effect.READ])
    names = {tool.name for tool in await server.list_tools()}

    assert "search_corpus" in names
    assert "extract_url" not in names
    assert "index_document" not in names
    assert "crawl_site" not in names


@pytest.mark.anyio
async def test_every_registry_tool_reaches_mcp():
    # The adapter has no list of its own; if it grows one, this fails.
    server = mcp_server.build_server(ALL_EFFECTS)
    exposed = {tool.name for tool in await server.list_tools()}
    assert exposed == {spec.name for spec in catalog(ALL_EFFECTS)}


# --------------------------------------------------------------------------- #
# Schema shape
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_schemas_are_flat():
    server = mcp_server.build_server(ALL_EFFECTS)
    for tool in await server.list_tools():
        schema = tool.input_schema
        rendered = json.dumps(schema)
        assert "$ref" not in rendered, f"{tool.name} publishes a nested schema"
        assert "$defs" not in rendered, f"{tool.name} publishes a nested schema"
        for name, prop in schema.get("properties", {}).items():
            assert prop.get("type") != "object", f"{tool.name}.{name} is an object"


@pytest.mark.anyio
async def test_argument_descriptions_survive_the_flattening():
    # They are where a small model learns what `alpha` or `hops` mean, and the
    # naive flattening drops them.
    server = mcp_server.build_server(ALL_EFFECTS)
    search = next(t for t in await server.list_tools() if t.name == "search_corpus")
    described = [
        name
        for name, prop in search.input_schema["properties"].items()
        if prop.get("description")
    ]
    assert len(described) >= 5


@pytest.mark.anyio
async def test_tool_descriptions_come_from_the_registry():
    server = mcp_server.build_server(ALL_EFFECTS)
    from pipeline.agents.tools import get

    for tool in await server.list_tools():
        assert tool.description == get(tool.name).description


# --------------------------------------------------------------------------- #
# Calling
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_a_call_returns_the_registry_observation():
    server = mcp_server.build_server([Effect.READ])
    result = await server.call_tool("corpus_profile", {})
    assert result.content[0].text


@pytest.mark.anyio
async def test_a_bad_argument_is_explained_in_terms_the_model_can_act_on():
    """Argument validation happens in the SDK, above invoke(), and that is fine.

    The registry turns bad arguments into an observation so a model can read
    the problem and retry. The SDK validates against the generated schema
    first and raises, which never reaches ``invoke``. Checked through the
    protocol, that surfaces as ``isError: true`` carrying the same explanation
    — a result the model reads, not a transport failure — so the property that
    matters survives. What must not happen is the field name going missing.
    """
    from mcp.server.mcpserver.exceptions import ToolError

    server = mcp_server.build_server([Effect.READ])
    with pytest.raises(ToolError) as caught:
        await server.call_tool("graph_neighbors", {"entity": ""})
    assert "entity" in str(caught.value)

    with pytest.raises(ToolError) as caught:
        await server.call_tool("search_corpus", {})
    assert "query" in str(caught.value)


@pytest.mark.anyio
async def test_a_tool_outside_the_allowance_cannot_be_called_by_name():
    server = mcp_server.build_server([Effect.READ])
    with pytest.raises(Exception):
        await server.call_tool("extract_url", {"url": "https://example.com"})
