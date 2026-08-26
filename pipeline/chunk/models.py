"""The chunk, and what travels with it into the store."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """What a retrieval hit needs in order to be traceable back to its source.

    A chunk without provenance is an assertion with no address: you cannot tell
    which document it came from, which page, or which model embedded it — and so
    you cannot tell whether it is stale.
    """

    source: str = ""
    page_no: int = -1
    section_name: str = ""
    chunk_strategy: str = "unknown"
    embedding_model: str = ""

    # Empty means "not known", which a filter reads as "do not exclude this".
    # A document with no department is not in department "" — it is a document
    # whose department nobody recorded, and filtering it out on that basis
    # would hide it from every departmental query forever.
    language: str = "unknown"
    doc_type: str = ""
    department: str = ""
    #: ISO 8601. Sorts and compares as a string, and reads plainly in a log.
    date: str = ""
    author: str = ""
    region: str = ""
    permission_level: str = ""

    extra: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One passage, ready to embed and store."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document: str
    metadata: ChunkMetadata

    dense_embedding: List[float] = Field(default_factory=list)


class ChunkedDocument(BaseModel):
    """The chunks one document split into."""

    chunks: List[Chunk] = Field(default_factory=list)
    strategy: str = "unknown"


__all__ = ["Chunk", "ChunkMetadata", "ChunkedDocument"]
