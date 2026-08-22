"""Arguments and results for the tool catalog.

Two rules run through all of it.

**Arguments are flat.** ``retrieve()`` takes a :class:`MetadataFilter` object,
but a tool that asked a model for a nested object would get one wrong more often
than right — small models handle ``department="finance"`` far better than
``filters={"department": ["finance"]}``. The filter is reassembled here, where
getting it right is a matter of code rather than of the model's attention.

**Results render twice.** ``render()`` produces numbered, citable prose in the
same shape :mod:`pipeline.retrieve.answer` already uses, so a passage read
through a tool cites identically to one retrieved directly. ``model_dump()``
keeps the scores and ids that prose throws away.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, Field

from pipeline.retrieve.filters import MetadataFilter


def _clip(text: str, limit: int) -> str:
    """Trim to ``limit`` characters, saying so rather than ending mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)].rstrip() + f"\n… [truncated at {limit} chars]"


def _as_count(value: Any) -> Any:
    """Read a word where a number was asked for.

    Every phrasing of "list the acquisitions" tested here had the model send
    ``limit: "all"``, which failed validation and cost the round. That is not a
    model getting it wrong occasionally — it is the honest way to say "no limit"
    to a field that offers no way to say it, and a schema that punishes the
    honest answer is the thing that is wrong. Numeric strings are left to
    Pydantic, which already coerces them.
    """
    if isinstance(value, str):
        word = value.strip().lower()
        if word in {"all", "every", "everything", "max", "maximum", "none", "no limit", "unlimited"}:
            return _COUNT_MAX
    return value


#: What a word-limit resolves to. Above every tool's own ceiling, so each
#: field's own bound is what actually clamps it.
_COUNT_MAX = 100

#: A count a model can also express in words.
Count = Annotated[int, BeforeValidator(_as_count)]


class FilterArgs(BaseModel):
    """The filter fields, one scalar each. Mixed into the tools that filter."""

    doc_type: Optional[str] = Field(
        None, description="e.g. document, html. One value."
    )
    department: Optional[str] = None
    author: Optional[str] = None
    region: Optional[str] = None
    permission_level: Optional[str] = None
    language: Optional[str] = Field(None, description="ISO code, e.g. en.")
    source: Optional[str] = Field(
        None, description="Exact source URL of one document."
    )
    date_from: Optional[str] = Field(None, description="YYYY-MM-DD, inclusive.")
    date_to: Optional[str] = Field(None, description="YYYY-MM-DD, inclusive.")

    def to_filter(self) -> MetadataFilter:
        """Scalars back into the list-valued filter the retriever wants.

        A document that never recorded a field does not match a filter on it, so
        an unset argument must stay ``None`` rather than becoming an empty list —
        the difference between "any department" and "no department".
        """
        return MetadataFilter(
            doc_type=[self.doc_type] if self.doc_type else None,
            department=[self.department] if self.department else None,
            author=[self.author] if self.author else None,
            region=[self.region] if self.region else None,
            permission_level=(
                [self.permission_level] if self.permission_level else None
            ),
            language=[self.language] if self.language else None,
            source=[self.source] if self.source else None,
            date_from=self.date_from,
            date_to=self.date_to,
        )


# --------------------------------------------------------------------------- #
# corpus_profile
# --------------------------------------------------------------------------- #


class ProfileArgs(BaseModel):
    """No arguments. Declared anyway so every tool has the same shape."""


class ProfileResult(BaseModel):
    available: bool
    chunks: int = 0
    documents: int = 0
    graph_entities: int = 0
    graph_relationships: int = 0
    #: Which kinds of edge the graph holds. Named here because a model cannot
    #: ask graph_relations for a kind it does not know exists, and guessing one
    #: costs a turn to learn it was wrong.
    relation_kinds: list[str] = Field(default_factory=list)
    filters: dict[str, list[str]] = Field(default_factory=dict)
    error: str = ""

    def render(self, *, max_chars: int = 2000) -> str:
        if not self.available:
            return f"The corpus is empty or unavailable. {self.error}".strip()

        lines = [
            f"{self.chunks} chunks from {self.documents} documents.",
            f"Knowledge graph: {self.graph_entities} entities, "
            f"{self.graph_relationships} relationships.",
        ]
        if self.relation_kinds:
            lines.append(
                "Relationship kinds (list them with graph_relations): "
                + ", ".join(self.relation_kinds[:12])
            )
        populated = {k: v for k, v in self.filters.items() if v}
        if populated:
            lines.append("Filter values in use:")
            for name, values in sorted(populated.items()):
                shown = ", ".join(values[:8])
                more = f" (+{len(values) - 8} more)" if len(values) > 8 else ""
                lines.append(f"  {name}: {shown}{more}")
        else:
            lines.append("No filter values are populated; do not filter.")
        return _clip("\n".join(lines), max_chars)


# --------------------------------------------------------------------------- #
# search_corpus
# --------------------------------------------------------------------------- #


class SearchArgs(FilterArgs):
    query: str = Field(..., min_length=1, description="What to search for.")
    limit: Count = Field(8, ge=1, le=50)
    fusion: Optional[str] = Field(
        None,
        description=(
            "'rrf' ranks robustly and is the default; 'alpha' weights the two "
            "legs by score. Use alpha only when rrf returned nothing useful."
        ),
    )
    alpha: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Under alpha fusion: 1.0 pure vector, 0.0 pure keyword.",
    )
    use_graph: Optional[bool] = Field(
        None, description="Include the knowledge-graph leg."
    )


class Passage(BaseModel):
    chunk_id: str = ""
    text: str = ""
    source: str = ""
    score: float = 0.0
    doc_type: str = ""


class SearchResult(BaseModel):
    query: str
    passages: list[Passage] = Field(default_factory=list)
    triples: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def render(self, *, max_chars: int = 2000) -> str:
        if not self.passages and not self.triples:
            hint = (
                "Nothing matched. If a filter was set, the corpus may not record "
                "that field for these documents — try again without it."
            )
            return f"No results for {self.query!r}. {hint}"

        parts = []
        for index, passage in enumerate(self.passages, start=1):
            parts.append(f"[{index}] ({passage.source})\n{passage.text}")
        if self.triples:
            parts.append("[knowledge graph]\n" + "\n".join(self.triples))
        return _clip("\n\n".join(parts), max_chars)


# --------------------------------------------------------------------------- #
# answer_from_corpus
# --------------------------------------------------------------------------- #


class AnswerArgs(FilterArgs):
    question: str = Field(..., min_length=1)
    limit: Count = Field(8, ge=1, le=50)
    use_graph: Optional[bool] = None


class AnswerResult(BaseModel):
    question: str
    answer: str = ""
    sufficient: bool = False
    cited: list[int] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    def render(self, *, max_chars: int = 2000) -> str:
        head = self.answer or "No answer was produced."
        if not self.sufficient:
            head += "\n\n(The corpus did not fully cover this question.)"
        if self.sources:
            cited = self.cited or list(range(1, len(self.sources) + 1))
            lines = [
                f"[{n}] {self.sources[n - 1]}"
                for n in cited
                if 1 <= n <= len(self.sources)
            ]
            if lines:
                head += "\n\nSources:\n" + "\n".join(lines)
        return _clip(head, max_chars)


# --------------------------------------------------------------------------- #
# graph_neighbors / graph_path
# --------------------------------------------------------------------------- #


class NeighborsArgs(BaseModel):
    entity: str = Field(
        ..., min_length=1,
        description="An entity name, e.g. 'Acme Corporation'. Exact-ish match.",
    )
    hops: int = Field(1, ge=1, le=3)
    limit: Count = Field(25, ge=1, le=100)


class RelationsArgs(BaseModel):
    relation: str = Field(
        "",
        description=(
            "Kind of relationship to list, e.g. ACQUIRED or LOCATED_IN. Leave "
            "empty to list every kind. corpus_profile names the kinds that "
            "exist."
        ),
    )
    limit: Count = Field(25, ge=1, le=100)


class PathArgs(BaseModel):
    start: str = Field(..., min_length=1)
    end: str = Field(..., min_length=1)
    max_hops: int = Field(3, ge=1, le=4)
    limit: Count = Field(25, ge=1, le=100)


class GraphResult(BaseModel):
    seeds: list[str] = Field(default_factory=list)
    edges: list[str] = Field(default_factory=list)
    available: bool = True

    def render(self, *, max_chars: int = 2000) -> str:
        if not self.available:
            return (
                "No knowledge graph has been built yet. Use search_corpus "
                "instead, or extract documents with build_graph enabled."
            )
        if not self.edges:
            found = f" (matched: {', '.join(self.seeds)})" if self.seeds else ""
            return f"No relationships found{found}."
        return _clip("\n".join(self.edges), max_chars)


# --------------------------------------------------------------------------- #
# fetch_chunk
# --------------------------------------------------------------------------- #


class ChunkArgs(BaseModel):
    chunk_id: str = Field(
        ..., min_length=1,
        description="A chunk_id from a search_corpus result.",
    )


class ChunkResult(BaseModel):
    chunk_id: str
    found: bool = False
    text: str = ""
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render(self, *, max_chars: int = 4000) -> str:
        if not self.found:
            return f"No chunk with id {self.chunk_id!r}."
        return _clip(f"({self.source})\n{self.text}", max_chars)


# --------------------------------------------------------------------------- #
# detect_url
# --------------------------------------------------------------------------- #


class DetectArgs(BaseModel):
    url: str = Field(..., min_length=1, description="An http(s) URL.")


class DetectResult(BaseModel):
    url: str
    acquisition: str = ""
    reason: str = ""
    likely_kind: str = ""
    allowed: bool = True
    refusal: str = ""
    already_indexed: bool = False

    def render(self, *, max_chars: int = 1000) -> str:
        if not self.allowed:
            return f"{self.url} cannot be fetched: {self.refusal}"
        known = " Already in the corpus." if self.already_indexed else ""
        return _clip(
            f"{self.url} would be acquired by {self.acquisition} "
            f"({self.reason}); likely a {self.likely_kind}.{known}",
            max_chars,
        )


# --------------------------------------------------------------------------- #
# The queued tools
# --------------------------------------------------------------------------- #


class SitemapArgs(BaseModel):
    url: str = Field(..., min_length=1, description="A site root or sitemap URL.")


class ExtractArgs(BaseModel):
    url: str = Field(..., min_length=1)
    prompt: str = Field(
        "Extract the main content and its key fields.",
        description="What to extract, in plain language.",
    )
    index: bool = Field(True, description="Add it to the searchable corpus.")
    build_graph: bool = Field(
        False, description="Also extract entities and relationships. Much slower."
    )


class IndexArgs(BaseModel):
    text: str = Field(
        ..., min_length=1, description="The document's text, as plain text."
    )
    source: str = Field(
        ...,
        min_length=1,
        description=(
            "Where it came from — a URL, a filename, or another stable "
            "identifier. Searches can filter on it and citations will show it."
        ),
    )
    department: str = Field("", description="Optional provenance filter.")
    region: str = Field("", description="Optional provenance filter.")
    permission_level: str = Field("", description="Optional provenance filter.")
    build_graph: bool = Field(
        False,
        description=(
            "Also extract entities and relationships into the knowledge graph. "
            "Much slower — one model call per window of text."
        ),
    )


class CrawlArgs(BaseModel):
    start_url: str = Field(..., min_length=1)
    prompt: str = Field("Extract the main content and its key fields.")
    max_depth: int = Field(1, ge=0, le=3)
    max_pages: int = Field(25, ge=1, le=500)


class TaskResult(BaseModel):
    """A queued job. Long work returns a receipt, never a blocked call."""

    task_id: str
    kind: str = ""
    note: str = ""

    def render(self, *, max_chars: int = 500) -> str:
        return _clip(
            f"Queued {self.kind} as task {self.task_id}. "
            f"Call poll_task with this id to see whether it finished. {self.note}",
            max_chars,
        )


class PollArgs(BaseModel):
    task_id: str = Field(..., min_length=1)


class PollResult(BaseModel):
    task_id: str
    state: str = "PENDING"
    done: bool = False
    ok: bool = False
    summary: str = ""

    def render(self, *, max_chars: int = 1500) -> str:
        if not self.done:
            return f"Task {self.task_id} is {self.state}; not finished yet."
        head = "finished" if self.ok else "failed"
        return _clip(f"Task {self.task_id} {head}. {self.summary}", max_chars)


__all__ = [
    "AnswerArgs", "AnswerResult", "ChunkArgs", "ChunkResult", "CrawlArgs",
    "DetectArgs", "DetectResult", "ExtractArgs", "FilterArgs", "GraphResult",
    "NeighborsArgs", "Passage", "PathArgs", "PollArgs", "PollResult",
    "ProfileArgs", "ProfileResult", "SearchArgs", "SearchResult", "SitemapArgs",
    "TaskResult",
]
