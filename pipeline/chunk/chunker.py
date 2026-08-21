"""Choosing how to split, and doing it."""

from __future__ import annotations

import re
from typing import List, Optional

from config import config
from observability import get_logger, metrics

from pipeline.chunk.models import Chunk, ChunkedDocument
from pipeline.chunk.strategies import (
    fixed_chunk,
    hierarchical_chunk,
    llm_chunk,
    semantic_chunk,
)
from pipeline.preprocess.orchestrator import PreprocessedDocument

log = get_logger("chunk")

#: Below this, splitting by meaning costs an embedding pass to produce one chunk.
_SEMANTIC_MIN_CHARS = 2000
#: Three headings is enough to believe the document is actually structured;
#: one or two show up in prose by accident.
_HIERARCHICAL_MIN_HEADERS = 3
#: Average line shorter than this suggests OCR columns or a mangled table
#: rather than paragraphs — but only once there are enough lines to mean it.
#: A one-line document has a short average line and is simply short.
_NOISY_MAX_LINE_LENGTH = 50
_NOISY_MIN_LINES = 10

#: Anchored per line, so a document that opens with its own title — which most
#: markdown documents do — has that title counted. Searching for "\n# " instead
#: misses any heading sitting at position 0.
_HEADING = re.compile(r"^#{1,3} ", re.MULTILINE)


class DocumentChunker:
    """Picks a strategy from the document's own shape, then applies it."""

    def __init__(self, default_strategy: Optional[str] = None) -> None:
        self.default_strategy = default_strategy or config.INDEX_CHUNK_STRATEGY

    def chunk(
        self,
        doc: PreprocessedDocument,
        strategy: Optional[str] = None,
        *,
        local_only: bool = False,
    ) -> ChunkedDocument:
        chosen = strategy or self.default_strategy
        if chosen == "agentic":
            chosen = self.decide_strategy(doc)

        with metrics.timer(f"chunk.{chosen}"):
            chunks = self._apply(chosen, doc, local_only=local_only)

        # Report what was done, not what was attempted. The model-assisted
        # strategy falls back to fixed windows when the backend is unreachable,
        # and a run report claiming "llm" on a machine where no model ran is the
        # kind of number you later build a wrong conclusion on.
        performed = {chunk.metadata.chunk_strategy for chunk in chunks}
        actual = performed.pop() if len(performed) == 1 else chosen

        log.info(
            "chunk.done",
            strategy=actual,
            requested=chosen,
            chunks=len(chunks),
            chars=len(doc.clean_text),
        )
        return ChunkedDocument(chunks=chunks, strategy=actual)

    def _apply(
        self, strategy: str, doc: PreprocessedDocument, *, local_only: bool
    ) -> List[Chunk]:
        if strategy == "fixed":
            return fixed_chunk(doc)
        if strategy == "semantic":
            return semantic_chunk(doc)
        if strategy == "hierarchical":
            return hierarchical_chunk(doc)
        if strategy == "llm":
            return llm_chunk(doc, local_only=local_only)
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; expected agentic, fixed, "
            "semantic, hierarchical or llm"
        )

    def decide_strategy(self, doc: PreprocessedDocument) -> str:
        """Pick a strategy from what the text looks like.

        Ordered by how much evidence each test needs. A document that publishes
        its own headings has told us how it wants to be split, so that wins;
        text whose lines are too short to be prose is the OCR-damage case a model
        is worth paying for; length alone only justifies splitting by meaning.
        """
        text = doc.clean_text
        if not text.strip():
            return "fixed"

        if len(_HEADING.findall(text)) >= _HIERARCHICAL_MIN_HEADERS:
            return "hierarchical"

        lines = [line for line in text.splitlines() if line.strip()]
        noisy = (
            len(lines) >= _NOISY_MIN_LINES
            and (len(text) / len(lines)) < _NOISY_MAX_LINE_LENGTH
        )
        if noisy:
            # Short lines mean OCR damage in a scanned document and a navigation
            # menu in an HTML one, and this test cannot tell them apart. Since
            # indexing is on by default, guessing wrong costs a model call per
            # document on ordinary pages — so the model-assisted splitter is
            # reachable only by asking for it.
            if config.INDEX_AGENTIC_ALLOW_LLM and config.ENABLE_TIER3_LLM:
                return "llm"
            return "fixed"

        if len(text) > _SEMANTIC_MIN_CHARS:
            return "semantic"

        return "fixed"


__all__ = ["DocumentChunker"]
