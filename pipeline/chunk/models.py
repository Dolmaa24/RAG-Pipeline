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
    language: str = "unknown"
    chunk_strategy: str = "unknown"
    embedding_model: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One passage, ready to embed and store."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document: str
    metadata: ChunkMetadata

    dense_embedding: List[float] = Field(default_factory=list)
    #: Term weights. Chroma stores only dense vectors, so this is serialised
    #: into metadata at write time and is not searchable yet.
    sparse_embedding: Dict[str, float] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    """The chunks one document split into."""

    chunks: List[Chunk] = Field(default_factory=list)
    strategy: str = "unknown"


__all__ = ["Chunk", "ChunkMetadata", "ChunkedDocument"]
