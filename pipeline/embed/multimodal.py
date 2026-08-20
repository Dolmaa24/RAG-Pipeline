"""Multimodal embeddings (CLIP).

Not part of the ingest path. It is here for image-and-text retrieval, and the
thing to know before wiring it in is that CLIP is 512-dimensional while the
default dense model is 384 — so it needs a collection of its own. Writing both
into one table is refused by :class:`~pipeline.store.lance.LanceStore`.
"""

from __future__ import annotations

from typing import List, Protocol

from errors import MissingDependency
from observability import get_logger

log = get_logger("embed.multimodal")


class MultimodalEmbedder(Protocol):
    name: str
    dimension: int

    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...

    def embed_images(self, image_paths: List[str]) -> List[List[float]]: ...

    def unload(self) -> None: ...


class ClipEmbedder:
    """``clip-vit-base-patch32`` — roughly 600 MB, the smallest useful CLIP."""

    name = "clip"
    dimension = 512

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise MissingDependency("transformers", "CLIP embeddings") from exc

        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _normalize(outputs):
        return (outputs / outputs.norm(p=2, dim=-1, keepdim=True)).detach().numpy().tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return self._normalize(self.model.get_text_features(**inputs))

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise MissingDependency("pillow", "image embeddings") from exc

        images = [Image.open(path) for path in image_paths]
        try:
            inputs = self.processor(images=images, return_tensors="pt")
            return self._normalize(self.model.get_image_features(**inputs))
        finally:
            for image in images:
                image.close()

    def unload(self) -> None:
        self.model = None
        self.processor = None


def get_multimodal_embedder(provider: str = "clip") -> MultimodalEmbedder:
    if provider == "clip":
        return ClipEmbedder()
    raise ValueError(f"unknown multimodal provider {provider!r}; expected clip")


__all__ = ["ClipEmbedder", "MultimodalEmbedder", "get_multimodal_embedder"]
