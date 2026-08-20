"""Letting a model write the traversal query, safely.

A generated Cypher query is useful — it can ask a question the fixed template
cannot, like following a particular relation type or filtering on a year. It is
also arbitrary code written by a model that has just read scraped web pages, so
two things stand between it and the database:

1. **This module rejects anything that is not a single read.** No write
   keyword, no semicolon, no multiple statements, must start with MATCH.
2. **It runs on a connection opened against a read-only database.** If (1) is
   ever wrong — a keyword nobody thought of, a syntax that slips the check —
   Kuzu refuses the write itself.

The second is the one that matters. A guard you can out-think is a guard; a
database that cannot write is a property. The prompt asking the model to only
read is not on the list, because a prompt is a request, not a control.
"""

from __future__ import annotations

import re
from typing import Optional

from observability import get_logger

log = get_logger("graph.cypher")

#: Anything that could change the graph, plus the clauses used to smuggle one in.
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|COPY|INSTALL|LOAD|"
    r"ATTACH|EXPORT|IMPORT|CALL|BEGIN|COMMIT|ROLLBACK|TRANSACTION)\b",
    re.IGNORECASE,
)

_SCHEMA = """Node table: Entity(name STRING, type STRING, description STRING)
Rel table: CONNECTS_TO(FROM Entity TO Entity, relation STRING, description STRING, valid_year STRING)"""

_PROMPT = """Write ONE read-only Kuzu Cypher query that retrieves the facts needed
to answer the question.

Schema:
{schema}

Rules:
- MATCH and RETURN only. Never CREATE, MERGE, SET, DELETE or CALL.
- Bind the relationship as [r:CONNECTS_TO]. Variable-length is allowed up to
  *1..{max_hops}, no further.
- Constrain with a WHERE clause on a.name or b.name against the seed entities.
- RETURN exactly: a.name, r.relation, b.name, r.description, r.valid_year
- One statement. No semicolon.

Seed entities: {seeds}"""


#: A relationship variable referenced in RETURN but never bound in MATCH. Small
#: models produce this constantly: they write `[:CONNECTS_TO]` and then ask for
#: `r.relation`, which Kuzu rejects with "Variable r is not in scope".
_REFERENCES_REL = re.compile(r"\br\.\w+")
_BINDS_REL = re.compile(r"\[\s*r\s*[:\]]")


def is_well_formed(cypher: str) -> bool:
    """Cheap static checks, so an obvious malformation costs no round trip."""
    if _REFERENCES_REL.search(cypher) and not _BINDS_REL.search(cypher):
        return False
    return True


def is_read_only(cypher: str) -> bool:
    """Whether this query only reads.

    Deliberately strict. A query that is fine but looks suspicious costs one
    fallback to the template; a write that gets through costs the graph.
    """
    text = (cypher or "").strip()
    if not text:
        return False
    if ";" in text.rstrip(";"):
        return False
    if _FORBIDDEN.search(text):
        return False
    return text.upper().startswith(("MATCH", "OPTIONAL MATCH"))


def fallback_query(seeds: list[str], *, max_hops: int = 1) -> tuple[str, dict]:
    """The fixed one-hop traversal, as a parameterised query."""
    return (
        """
        MATCH (a:Entity)-[r:CONNECTS_TO]->(b:Entity)
        WHERE a.name IN $seeds OR b.name IN $seeds
        RETURN a.name, r.relation, b.name, r.description, r.valid_year
        LIMIT $limit
        """,
        {"seeds": seeds},
    )


class CypherAgent:
    """Turns a question plus seed entities into a query, or declines to."""

    def __init__(self, backend=None) -> None:
        self._backend = backend

    def generate(
        self,
        question: str,
        seeds: list[str],
        *,
        max_hops: int = 2,
        local_only: bool = False,
    ) -> Optional[str]:
        """A validated read-only query, or ``None`` to use the template."""
        if not seeds:
            return None

        backend = self._backend
        if backend is None:
            try:
                from pipeline.extract.llm import INTERACTIVE, get_backend

                # Interactive: this sits directly in the query path.
                backend = get_backend(local_only=local_only, role=INTERACTIVE)
            except Exception as exc:
                log.warning("graph.cypher.no_backend", error=repr(exc))
                return None

        try:
            response = backend.complete_json(
                prompt=_PROMPT.format(
                    schema=_SCHEMA, max_hops=max_hops, seeds=", ".join(seeds[:8])
                ),
                content=question,
                schema_hint={"cypher_query": "string"},
                json_schema={
                    "type": "object",
                    "properties": {"cypher_query": {"type": "string"}},
                    "required": ["cypher_query"],
                },
            )
            query = str((response.data or {}).get("cypher_query") or "").strip()
        except Exception as exc:
            log.warning("graph.cypher.generation_failed", error=repr(exc))
            return None

        if not query:
            return None

        if not is_read_only(query):
            log.warning("graph.cypher.rejected", reason="not read-only", query=query[:160])
            return None

        if not is_well_formed(query):
            log.warning("graph.cypher.rejected", reason="unbound relationship", query=query[:160])
            return None

        # Seeds still have to reach the query somehow, and a generated query has
        # no parameters to bind to — which is precisely why it runs read-only.
        log.info("graph.cypher.generated", query=query[:160])
        return query


__all__ = ["CypherAgent", "fallback_query", "is_read_only", "is_well_formed"]
