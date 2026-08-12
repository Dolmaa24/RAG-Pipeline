"""Speech-to-text backends behind one interface."""

from .base import Segment, TranscriptResult, Transcriber, get_transcriber, preload, reset

__all__ = [
    "Segment",
    "TranscriptResult",
    "Transcriber",
    "get_transcriber",
    "preload",
    "reset",
]
