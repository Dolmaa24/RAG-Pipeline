"""Audio and video bytes that were fetched directly over HTTP.

The platform path (a YouTube page, a podcast host) goes through
:class:`~pipeline.media_processor.MediaProcessor`, because there the URL's bytes
are a player page rather than the media. This handler is for the other case: a
direct link to an ``.mp3`` or ``.mp4`` that the fetcher already downloaded.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from errors import MissingDependency
from models import ExtractionItem, ResourceKind, Stage
from observability import get_logger
from pipeline.media_processor import apply_transcript
from pipeline.transcribe import get_transcriber

from .base import BaseHandler, registry

log = get_logger("handlers.media")


class MediaHandler(BaseHandler):
    name = "media"
    kinds = (ResourceKind.AUDIO, ResourceKind.VIDEO)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        if not item.raw_bytes:
            return item.fail(Stage.PARSE, "no media bytes to transcribe")

        subtype = item.metadata.get("detected_subtype", "bin")
        workdir = tempfile.mkdtemp(prefix="media_bytes_")
        try:
            # ffmpeg (under Whisper) dispatches on the extension, so the temp
            # file gets the one the magic bytes implied.
            path = os.path.join(workdir, f"media.{subtype}")
            with open(path, "wb") as handle:
                handle.write(item.raw_bytes)

            try:
                result = get_transcriber().transcribe(path)
            except MissingDependency as exc:
                return item.fail_from(exc)

            item.metadata.setdefault("media_bytes", len(item.raw_bytes))
            return apply_transcript(item, result)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


registry.register(MediaHandler())

__all__ = ["MediaHandler"]
