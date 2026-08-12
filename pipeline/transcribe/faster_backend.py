"""Whisper on CTranslate2 — the portable fallback.

Kept because it is the right answer everywhere that is not an Apple Silicon
Mac, and because a fallback that shares the interface means switching backends
is a config change rather than a code change.

The model is constructed **once per process** and reused. Building it inside
every task, as the original pipeline did, cost several seconds and a full model
load per job.
"""

from __future__ import annotations

from typing import Optional

from config import config
from errors import MissingDependency
from observability import get_logger

from .base import Segment, TranscriptResult

log = get_logger("transcribe.faster")


class FasterWhisper:
    name = "faster"

    def __init__(self, model_size: Optional[str] = None) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependency("faster-whisper", "CPU transcription") from exc

        self.model_name = model_size or config.WHISPER_MODEL_SIZE
        # int8 on CPU: roughly 4x less memory than float32 for a small accuracy
        # cost that does not survive into extracted fields.
        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        log.info("transcribe.model_loaded", backend=self.name, model=self.model_name)

    def transcribe(self, audio_path: str, *, language: Optional[str] = None) -> TranscriptResult:
        log.info("transcribe.start", backend=self.name, model=self.model_name, path=audio_path)
        segments_iter, info = self._model.transcribe(
            audio_path,
            beam_size=1,
            language=language or config.WHISPER_LANGUAGE,
            vad_filter=True,  # skip silence rather than hallucinate through it
            condition_on_previous_text=False,
        )

        segments = [
            Segment(start=float(s.start), end=float(s.end), text=s.text) for s in segments_iter
        ]
        text = " ".join(segment.text.strip() for segment in segments).strip()

        return TranscriptResult(
            text=text,
            segments=segments,
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
            backend=self.name,
            model=self.model_name,
        )


__all__ = ["FasterWhisper"]
