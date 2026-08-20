"""Dense embeddings, local by default.

The model is held for the life of the process, exactly like the Whisper model in
:mod:`pipeline.transcribe.base`. The first version of this module constructed a
``SentenceTransformer`` inside every call and unloaded it afterwards, which meant
paying a ~130 MB model load *per document* — far more expensive on an 8 GB
machine than simply keeping BGE-small resident. Celery's
``worker_max_tasks_per_child`` already recycles the process, which bounds any
leak the model might have.

The heavyweight paths — SPLADE, CLIP — still load and unload, because there the
memory genuinely does not fit alongside everything else.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Protocol

from config import config
from errors import MissingDependency
from observability import get_logger

log = get_logger("embed.dense")

#: Output width per provider, checked against the stored table before a write.
#: Mixing widths in one table is silent nonsense, not an error.
DIMENSIONS = {
    "local_bge": 384,
    "local_e5": 384,
    "cohere": 1024,
    "voyage": 1024,
}


def select_device() -> Optional[str]:
    """Which torch device to put the model on. ``None`` lets torch decide.

    **Metal keeps killing processes, so "auto" means CPU.** Measured here, in
    two of the three contexts this model runs in:

    * A Celery **prefork child** that builds an MPS compute pipeline aborts with
      ``MPSKernel ... Unable to reach MTLCompilerService`` — the connection to
      Metal's compiler daemon is not inherited across ``fork()``. This is the
      same family of macOS trap that ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY``
      defuses in :mod:`celery_app`, and that variable does not cover it.
    * The **uvicorn** process dies the same way on the first embedding call,
      leaving only a leaked-semaphore warning and no traceback, because it is
      killed by a signal rather than raising.

    Only the synchronous ``main.py`` path survives MPS reliably. Given that
    BGE-small is 33M parameters — a batch of 32 takes about a quarter of a
    second on the CPU — the GPU is not worth a crash. It stays free for Whisper,
    which genuinely needs it.

    Set ``INDEX_EMBED_DEVICE=mps`` to force it in a context you have verified.
    """
    configured = config.INDEX_EMBED_DEVICE
    if configured != "auto":
        return configured
    return "cpu"


class DenseEmbedder(Protocol):
    name: str
    model_name: str
    dimension: int

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class LocalDenseEmbedder:
    """sentence-transformers, running on the machine.

    BGE-small is the default: 384 dimensions and roughly 130 MB, which is the
    largest thing that comfortably shares an 8 GB host with Chromium, Whisper
    and a local LLM.
    """

    name = "local"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MissingDependency("sentence-transformers", "dense embeddings") from exc

        self.model_name = model_name
        self.device = select_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        # Renamed in sentence-transformers 6.0; the old name still works but
        # warns, and pinning to either one alone breaks the other version.
        measure = getattr(
            self.model, "get_embedding_dimension", None
        ) or self.model.get_sentence_embedding_dimension
        self.dimension = int(measure())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def unload(self) -> None:
        self.model = None


class CohereEmbedder:
    name = "cohere"
    model_name = "embed-english-v3.0"
    dimension = 1024

    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            import cohere
        except ImportError as exc:
            raise MissingDependency("cohere", "the Cohere embedder") from exc
        self.client = cohere.Client(api_key or os.environ.get("COHERE_API_KEY"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embed(
            texts=texts, model=self.model_name, input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embed(
            texts=[text], model=self.model_name, input_type="search_query"
        )
        return response.embeddings[0]

    def unload(self) -> None:
        pass


class VoyageEmbedder:
    name = "voyage"
    model_name = "voyage-2"
    dimension = 1024

    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            import voyageai
        except ImportError as exc:
            raise MissingDependency("voyageai", "the Voyage embedder") from exc
        self.client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed(texts, model=self.model_name, input_type="document").embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed(
            [text], model=self.model_name, input_type="query"
        ).embeddings[0]

    def unload(self) -> None:
        pass


_instance: Optional[DenseEmbedder] = None
_instance_provider: Optional[str] = None
_lock = threading.Lock()


def get_dense_embedder(provider: Optional[str] = None) -> DenseEmbedder:
    """The process-wide dense embedder, constructed once per provider."""
    global _instance, _instance_provider
    choice = (provider or config.INDEX_DENSE_PROVIDER).lower()

    with _lock:
        if _instance is not None and _instance_provider == choice:
            return _instance

        if choice == "local_bge":
            embedder: DenseEmbedder = LocalDenseEmbedder("BAAI/bge-small-en-v1.5")
        elif choice == "local_e5":
            embedder = LocalDenseEmbedder("intfloat/e5-small-v2")
        elif choice == "cohere":
            embedder = CohereEmbedder()
        elif choice == "voyage":
            embedder = VoyageEmbedder()
        else:
            raise ValueError(
                f"unknown dense provider {choice!r}; expected one of {sorted(DIMENSIONS)}"
            )

        log.info(
            "embed.dense.loaded",
            provider=choice,
            model=embedder.model_name,
            dimension=embedder.dimension,
            device=getattr(embedder, "device", None) or "auto",
        )
        _instance, _instance_provider = embedder, choice
        return embedder


def preload() -> None:
    """Load the model now rather than on the first document."""
    get_dense_embedder()


def reset() -> None:
    """Drop the cached embedder. Used by tests and to reclaim the memory."""
    global _instance, _instance_provider
    with _lock:
        if _instance is not None and hasattr(_instance, "unload"):
            _instance.unload()
        _instance, _instance_provider = None, None


__all__ = [
    "DIMENSIONS",
    "CohereEmbedder",
    "DenseEmbedder",
    "LocalDenseEmbedder",
    "VoyageEmbedder",
    "get_dense_embedder",
    "preload",
    "reset",
    "select_device",
]
