"""Images: read the text in them.

On this machine the right OCR engine is **Vision.framework**. It ships with
macOS, needs no install, runs on the Neural Engine, and is markedly more
accurate on photographed and screenshotted text than Tesseract. It is reached
through PyObjC, which is present in the system Python but usually not in a venv
— so Tesseract stays as the portable fallback, and a missing engine degrades to
a clear message rather than a crash.

Image *captioning* (describing a photo that contains no text) is a model
question, not a parsing one, so it is left to the extraction cascade.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Optional

from config import config
from errors import MissingDependency
from models import ExtractionItem, ResourceKind, Stage
from observability import get_logger

from .base import BaseHandler, registry

log = get_logger("handlers.image")


class ImageHandler(BaseHandler):
    name = "image"
    kinds = (ResourceKind.IMAGE,)
    #: A photograph with no text in it is still a successful parse — the
    #: cascade can caption it. So an empty OCR result is not a failure.
    requires_text = False

    def process(self, item: ExtractionItem) -> ExtractionItem:
        if not item.raw_bytes:
            return item.fail(Stage.PARSE, "no bytes to read")

        item.metadata.update(_image_dimensions(item.raw_bytes))

        if not config.OCR_ENABLED:
            item.warn("OCR is disabled; no text extracted from image")
            item.cleaned_text = ""
            return item

        text = ocr_bytes(item.raw_bytes)
        item.decoded_text = text
        item.cleaned_text = text
        item.metadata["ocr"] = True
        item.metadata["ocr_engine"] = _selected_engine() or "none"
        if not text.strip():
            item.warn("OCR found no text in this image")
        else:
            log.info("image.ocr_ok", url=item.url, chars=len(text), engine=item.metadata["ocr_engine"])
        return item


@lru_cache(maxsize=1)
def _vision_available() -> bool:
    if os.uname().sysname != "Darwin":
        return False
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _tesseract_path() -> Optional[str]:
    return shutil.which("tesseract")


@lru_cache(maxsize=1)
def _selected_engine() -> Optional[str]:
    preference = config.OCR_BACKEND
    if preference == "vision":
        return "vision" if _vision_available() else None
    if preference == "tesseract":
        return "tesseract" if _tesseract_path() else None
    if _vision_available():
        return "vision"
    if _tesseract_path():
        return "tesseract"
    return None


def ocr_bytes(data: bytes) -> str:
    """Read text out of image bytes with the best available engine."""
    engine = _selected_engine()
    if engine == "vision":
        return _ocr_vision(data)
    if engine == "tesseract":
        return _ocr_tesseract(data)
    raise MissingDependency(
        "pyobjc-framework-Vision (or `brew install tesseract`)",
        "optical character recognition",
    )


def _ocr_vision(data: bytes) -> str:
    """Apple Vision, via PyObjC. Accurate, on-device, no network."""
    import Quartz
    import Vision
    from Foundation import NSData

    ns_data = NSData.dataWithBytes_length_(data, len(data))
    source = Quartz.CGImageSourceCreateWithData(ns_data, None)
    if source is None or Quartz.CGImageSourceGetCount(source) == 0:
        return ""
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if image is None:
        return ""

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    success, error = handler.performRequests_error_([request], None)
    if not success:
        log.warning("image.vision_failed", error=str(error))
        return ""

    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if candidates:
            lines.append(str(candidates[0].string()))
    return "\n".join(lines)


def _ocr_tesseract(data: bytes) -> str:
    binary = _tesseract_path()
    if binary is None:  # pragma: no cover - guarded by _selected_engine
        raise MissingDependency("tesseract", "optical character recognition")

    with tempfile.TemporaryDirectory(prefix="ocr_") as workdir:
        source = os.path.join(workdir, "image")
        with open(source, "wb") as handle:
            handle.write(data)
        try:
            completed = subprocess.run(
                [binary, source, "stdout"],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            log.warning("image.tesseract_timeout")
            return ""
        except subprocess.CalledProcessError as exc:
            log.warning("image.tesseract_failed", error=(exc.stderr or b"")[:200].decode("utf-8", "ignore"))
            return ""
    return completed.stdout.decode("utf-8", "replace").strip()


def _image_dimensions(data: bytes) -> dict:
    """Width and height without decoding the whole image, when Pillow is around."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return {"width": image.width, "height": image.height, "image_format": image.format}
    except Exception:
        return {}


registry.register(ImageHandler())

__all__ = ["ImageHandler", "ocr_bytes"]
