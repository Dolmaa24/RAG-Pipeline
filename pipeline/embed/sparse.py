"""Sparse embeddings.

**This is term frequency, not BM25.** The distinction matters enough to put in
the class name. BM25 weights a term by how rare it is across the whole corpus
(its IDF) and normalises by document length; both of those are corpus-level
quantities, and nothing here has seen the corpus — each document is embedded on
its own, as it arrives. What this produces is the raw term counts a real BM25
would be *built from*.

That is still worth storing: the counts go into Chroma metadata now, and a later
retrieval pass can compute IDF over the finished collection and score properly.
What it is not is something you can rank with today, which is why
:mod:`pipeline.store.chroma` searches dense vectors only.

SPLADE learns the weights instead of counting, and needs no corpus pass — but it
is a full transformer, and on an 8 GB host it does not fit alongside the dense
model. It stays opt-in.
"""

from __future__ import annotations

import string
from typing import Dict, List, Protocol

from errors import MissingDependency
from observability import get_logger

log = get_logger("embed.sparse")

_PUNCTUATION = str.maketrans("", "", string.punctuation)


class SparseEmbedder(Protocol):
    name: str

    def embed_documents(self, texts: List[str]) -> List[Dict[str, float]]: ...

    def unload(self) -> None: ...


class TermFrequencyEmbedder:
    """Per-document term counts. Pure Python, no model, no corpus."""

    name = "tf"

    def embed_documents(self, texts: List[str]) -> List[Dict[str, float]]:
        embeddings: List[Dict[str, float]] = []
        for text in texts:
            counts: Dict[str, float] = {}
            for token in text.lower().translate(_PUNCTUATION).split():
                counts[token] = counts.get(token, 0.0) + 1.0
            embeddings.append(counts)
        return embeddings

    def unload(self) -> None:
        pass


class NullSparseEmbedder:
    """Store nothing. For when the sparse half is not wanted at all."""

    name = "none"

    def embed_documents(self, texts: List[str]) -> List[Dict[str, float]]:
        return [{} for _ in texts]

    def unload(self) -> None:
        pass


class SpladeEmbedder:
    """Learned sparse weights. Off by default — it is a full transformer."""

    name = "splade"
    model_id = "naver/splade-cocondenser-ensembledistil"

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise MissingDependency("transformers torch", "SPLADE sparse embeddings") from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
        self.model.eval()
        log.warning("embed.sparse.splade_loaded", note="heavy model; watch memory")

    def embed_documents(self, texts: List[str]) -> List[Dict[str, float]]:
        torch = self._torch
        vocab = self.tokenizer.get_vocab()
        inverse = {index: token for token, index in vocab.items()}

        out: List[Dict[str, float]] = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                logits = self.model(**inputs).logits
                # The SPLADE pooling: log(1 + relu(logits)), maxed over tokens.
                weights = torch.max(
                    torch.log1p(torch.relu(logits))
                    * inputs["attention_mask"].unsqueeze(-1),
                    dim=1,
                ).values.squeeze()
                nonzero = torch.nonzero(weights).squeeze(-1)
                out.append(
                    {inverse[int(i)]: float(weights[i]) for i in nonzero}
                )
        return out

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None


def get_sparse_embedder(provider: str = "tf") -> SparseEmbedder:
    choice = (provider or "tf").lower()
    if choice == "tf":
        return TermFrequencyEmbedder()
    if choice == "none":
        return NullSparseEmbedder()
    if choice == "splade":
        return SpladeEmbedder()
    raise ValueError(f"unknown sparse provider {choice!r}; expected tf, splade or none")


__all__ = [
    "NullSparseEmbedder",
    "SparseEmbedder",
    "SpladeEmbedder",
    "TermFrequencyEmbedder",
    "get_sparse_embedder",
]
