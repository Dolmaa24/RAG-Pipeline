"""Live streams: capture rolling segments, transcribe as they land.

The handler itself is thin — the work is in
:class:`~pipeline.media_processor.LiveStreamProcessor`. What matters here is
that the item arrives with ``kind == LIVESTREAM``, which means nothing
downstream ever waits for a file that will not finish.
"""

from __future__ import annotations

from typing import Callable, Optional

from models import ExtractionItem, ResourceKind
from observability import get_logger
from pipeline.media_processor import LiveStreamProcessor

from .base import BaseHandler, registry

log = get_logger("handlers.livestream")


class LivestreamHandler(BaseHandler):
    name = "livestream"
    kinds = (ResourceKind.LIVESTREAM,)

    def __init__(self, on_segment: Optional[Callable[[int, str, float], None]] = None) -> None:
        self.on_segment = on_segment

    def process(self, item: ExtractionItem) -> ExtractionItem:
        # An HLS manifest fetched over HTTP arrives as bytes, but the bytes are
        # a playlist, not audio. Capture works from the URL either way, so the
        # downloaded manifest is simply dropped.
        item.raw_bytes = None
        return LiveStreamProcessor().capture(item, on_segment=self.on_segment)


registry.register(LivestreamHandler())

__all__ = ["LivestreamHandler"]
