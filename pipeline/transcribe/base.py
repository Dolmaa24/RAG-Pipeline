"""Speech-to-text behind one interface, with the backend chosen at runtime.

Why this is not just ``faster-whisper``: faster-whisper runs on CTranslate2,
which has **no Metal backend on macOS**. On an Apple Silicon machine it is
therefore using CPU cores only, while the GPU and the Neural Engine sit idle.
MLX-based Whisper uses both.

Published Apple Silicon benchmarks report large speedups for MLX, but the
numbers vary a lot by implementation, model size and audio, so treat them as
directional: :mod:`pipeline.transcribe.bench` runs both on *your* machine and
*your* audio, which is the only comparison that decides anything. faster-whisper
stays behind the same interface as the fallback, and as the answer on any
machine that is not a Mac.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Protocol

from config import config
from errors import MissingDependency
from observability import get_logger

log = get_logger("transcribe")


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None
    backend: str = ""
    model: str = ""

    def timestamped(self) -> str:
        """Transcript with ``[12.4s - 18.1s]`` prefixes.

        Worth the extra tokens: it lets a model answer "when did they say X"
        and lets a human check a claim against the source.
        """
        if not self.segments:
            return self.text
        return "\n".join(
            f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()}"
            for segment in self.segments
            if segment.text.strip()
        )


class Transcriber(Protocol):
    name: str
    model_name: str

    def transcribe(self, audio_path: str, *, language: Optional[str] = None) -> TranscriptResult: ...


_instance: Optional[Transcriber] = None
_lock = threading.Lock()


def get_transcriber(force: Optional[str] = None) -> Transcriber:
    """The process-wide transcriber, constructed once.

    Constructing a Whisper model costs seconds and hundreds of MB. The original
    pipeline paid that on *every task*; holding one instance per worker process
    is the single cheapest speedup in the media path.
    """
    global _instance
    backend = force or config.WHISPER_BACKEND

    with _lock:
        if _instance is not None and (force is None or _instance.name == force):
            return _instance
        _instance = _build(backend)
        log.info("transcribe.backend_selected", backend=_instance.name, model=_instance.model_name)
        return _instance


def _build(backend: str) -> Transcriber:
    if backend in ("auto", "mlx"):
        try:
            from .mlx_backend import MlxWhisper

            return MlxWhisper()
        except (ImportError, MissingDependency) as exc:
            if backend == "mlx":
                raise
            log.info("transcribe.mlx_unavailable", reason=str(exc))

    from .faster_backend import FasterWhisper

    return FasterWhisper()


def preload() -> None:
    """Warm the model at worker start rather than inside the first task."""
    if not config.PRELOAD_MODELS:
        return
    try:
        get_transcriber()
    except Exception as exc:  # a worker must still start without audio support
        log.warning("transcribe.preload_failed", error=repr(exc))


def reset() -> None:
    """Drop the cached transcriber. Used by tests and by worker recycling."""
    global _instance
    with _lock:
        _instance = None


__all__ = ["Segment", "TranscriptResult", "Transcriber", "get_transcriber", "preload", "reset"]
