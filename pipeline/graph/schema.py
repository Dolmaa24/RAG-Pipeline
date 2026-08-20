"""Entities, relationships, and where each one came from.

The shape is from ``Dolmaa24/GraphRAG`` and is unchanged, with one addition:
provenance. A triple in a knowledge graph is an assertion, and an assertion you
cannot trace to a source is one you cannot check, correct, or expire. Every node
and edge here carries the URL and content hash of the chunk that produced it.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(description="The canonical, standardized name of the entity.")
    type: str = Field(
        description="Category, e.g. Person, Organization, Location, Technology, Concept."
    )
    description: str = Field(description="Brief summary of what this entity is or does.")

    #: Provenance, filled in by the extractor rather than the model.
    source_url: str = ""
    content_hash: str = ""


class Relationship(BaseModel):
    source: str = Field(description="The exact name of the source entity.")
    target: str = Field(description="The exact name of the target entity.")
    relation: str = Field(
        description="Relationship type in UPPERCASE_SNAKE_CASE, e.g. WORKED_AT, ACQUIRED."
    )
    description: str = Field(description="Contextual details explaining this connection.")
    valid_year: Optional[str] = Field(
        default="UNKNOWN", description="Year or timestamp associated with this relationship."
    )

    source_url: str = ""
    content_hash: str = ""


class KnowledgeGraphExtraction(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class Triple(BaseModel):
    """One edge, flattened for use as retrieval context."""

    source: str
    relation: str
    target: str
    description: str = ""
    valid_year: str = "UNKNOWN"
    source_url: str = ""

    def render(self) -> str:
        year = f" ({self.valid_year})" if self.valid_year and self.valid_year != "UNKNOWN" else ""
        detail = f": {self.description}" if self.description else ""
        return f"({self.source})-[{self.relation}{year}]->({self.target}){detail}"


__all__ = ["Entity", "KnowledgeGraphExtraction", "Relationship", "Triple"]
