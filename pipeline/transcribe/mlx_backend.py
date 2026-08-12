"""Whisper on MLX — the Apple Silicon path, using the GPU and Neural Engine."""

from __future__ import annotations

from typing import Optional

from config import config
from errors import MissingDependency
from observability import get_logger

from .base import Segment, TranscriptResult

log = get_logger("transcribe.mlx")


class MlxWhisper:
    """Wraps ``mlx_whisper.transcribe``.

    MLX loads weights lazily and caches them in ``~/.cache/huggingface``, so
    there is no model object to hold: the first call pays the download and the
    graph build, and later calls in the same process reuse both.
    """

    name = "mlx"

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            raise MissingDependency("mlx-whisper", "GPU-accelerated transcription on Apple Silicon") from exc
        self.model_name = model or config.MLX_WHISPER_MODEL

    def transcribe(self, audio_path: str, *, language: Optional[str] = None) -> TranscriptResult:
        import mlx_whisper

        log.info("transcribe.start", backend=self.name, model=self.model_name, path=audio_path)
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.model_name,
            language=language or config.WHISPER_LANGUAGE,
            verbose=None,
            word_timestamps=False,
            condition_on_previous_text=False,  # stops the model looping on silence
        )

        segments = [
            Segment(
                start=float(chunk.get("start", 0.0)),
                end=float(chunk.get("end", 0.0)),
                text=str(chunk.get("text", "")),
            )
            for chunk in result.get("segments", []) or []
        ]
        text = (result.get("text") or "").strip()

        return TranscriptResult(
            text=text,
            segments=segments,
            language=result.get("language"),
            duration=segments[-1].end if segments else None,
            backend=self.name,
            model=self.model_name,
        )


__all__ = ["MlxWhisper"]
