"""The tool catalog.

Importing this package registers every tool. Both consumers — the MCP server and
the in-process loop — import it for that side effect and then read the registry,
so neither has its own list to keep in step.
"""

from pipeline.agents.tools import acquire, build, corpus  # noqa: F401  (registration)
from pipeline.agents.tools.registry import (
    READ_ONLY,
    Effect,
    EffectNotAllowed,
    ToolCall,
    ToolError,
    ToolSpec,
    catalog,
    describe_all,
    get,
    invoke,
    tool,
)

__all__ = [
    "READ_ONLY",
    "Effect",
    "EffectNotAllowed",
    "ToolCall",
    "ToolError",
    "ToolSpec",
    "catalog",
    "describe_all",
    "get",
    "invoke",
    "tool",
]
