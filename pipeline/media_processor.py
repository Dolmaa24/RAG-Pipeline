"""Acquiring and transcribing media, including streams that never end.

Two shapes of work:

:class:`MediaProcessor`
    A finite recording. Download the audio with yt-dlp, transcribe it, done.

:class:`LiveStreamProcessor`
    A live broadcast. There is no "done", so waiting for the file is not an
    option. ffmpeg writes rolling N-minute segments; each one is transcribed as
    it lands and appended to the record, with a callback after every segment so
    the caller can update task state and persist partial results. A crash or a
    timeout costs you the current segment, not the whole stream.
"""

from __future__ import annotations

import gc
import os
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Callable, Optional

import yt_dlp

from config import config
from errors import FetchError, MissingDependency, TransientFetchError
from models import ExtractionItem, FetchMode, ResourceKind, Stage
from observability import get_logger
from pipeline.transcribe import TranscriptResult, get_transcriber

log = get_logger("media")

#: Called after each live segment with (index, partial_transcript, seconds).
SegmentCallback = Callable[[int, str, float], None]


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise MissingDependency(
            "ffmpeg (brew install ffmpeg)", "extracting and segmenting audio"
        )
    return path


def _ydl_options(workdir: str) -> dict:
    return {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                # 128 kbps is far more than Whisper needs (it resamples to 16 kHz
                # mono anyway) but keeps the file small and the decode fast.
                "preferredquality": "128",
            }
        ],
        "outtmpl": os.path.join(workdir, "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "socket_timeout": 30,
        "user_agent": config.user_agent if config.SPOOF_BROWSER_UA else None,
    }


class MediaProcessor:
    """Download a media URL with yt-dlp and transcribe it locally."""

    def __init__(self, model_size: Optional[str] = None) -> None:
        self.model_size = model_size

    def process_media_url(self, item: ExtractionItem) -> ExtractionItem:
        url = str(item.url)
        started = time.perf_counter()
        try:
            _require_ffmpeg()
        except MissingDependency as exc:
            return item.fail_from(exc)

        # A unique working directory per call, so concurrent workers never
        # collide on a shared temp filename.
        workdir = tempfile.mkdtemp(prefix="media_")
        try:
            info = self._probe(url)
            if info is None:
                return item.fail(Stage.FETCH, f"yt-dlp could not read {url}")

            if info.get("is_live"):
                log.info("media.is_live", url=url)
                item.kind = ResourceKind.LIVESTREAM
                return LiveStreamProcessor().capture(item)

            audio_path = self._download_audio(item, url, workdir, info)
            if item.error:
                return item

            return self._transcribe(item, audio_path)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            item.record_timing("media", time.perf_counter() - started)
            gc.collect()

    # ------------------------------------------------------------------ #

    @staticmethod
    def _probe(url: str) -> Optional[dict]:
        """Ask yt-dlp what this is without downloading it.

        Cheap, and it is the only reliable way to know whether a URL is a live
        broadcast before committing to "download the whole thing first".
        """
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as exc:
            log.warning("media.probe_failed", url=url, error=repr(exc))
            return None

    def _download_audio(
        self, item: ExtractionItem, url: str, workdir: str, info: dict
    ) -> Optional[str]:
        log.info("media.downloading", url=url, title=info.get("title"))
        try:
            with yt_dlp.YoutubeDL(_ydl_options(workdir)) as ydl:
                info = ydl.extract_info(url, download=True) or info
        except yt_dlp.utils.DownloadError as exc:
            message = str(exc)
            # yt-dlp reports geo-blocks, sign-in walls and removals the same
            # way as a network blip; only the latter is worth retrying.
            transient = any(word in message.lower() for word in ("timed out", "connection", "temporarily"))
            item.fail(
                Stage.FETCH,
                f"failed to download media: {message[:300]}",
                error_type="DownloadError",
                transient=transient,
            )
            return None
        except Exception as exc:
            item.fail(Stage.FETCH, f"failed to download media: {exc}", error_type=type(exc).__name__)
            return None

        item.metadata.update(
            {
                "title": info.get("title"),
                "author": info.get("uploader") or info.get("channel"),
                "duration": info.get("duration"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "source_url": info.get("webpage_url") or url,
                "extractor": info.get("extractor_key"),
            }
        )
        item.fetch_mode = FetchMode.YTDLP
        item.metadata = {k: v for k, v in item.metadata.items() if v is not None}

        # The postprocessor renames the file, so locate it rather than assume.
        candidates = [
            os.path.join(workdir, name) for name in os.listdir(workdir) if name.startswith("audio.")
        ]
        mp3s = [path for path in candidates if path.endswith(".mp3")]
        audio_path = (mp3s or candidates or [None])[0]

        if not audio_path or not os.path.exists(audio_path):
            item.fail(Stage.FETCH, "audio download produced no file (ffmpeg conversion failed?)")
            return None
        return audio_path

    def _transcribe(self, item: ExtractionItem, audio_path: str) -> ExtractionItem:
        try:
            transcriber = get_transcriber()
            result = transcriber.transcribe(audio_path)
        except MissingDependency as exc:
            return item.fail_from(exc)
        except Exception as exc:
            return item.fail(Stage.DECODE, f"transcription failed: {exc}", error_type=type(exc).__name__)

        return apply_transcript(item, result)

    # Kept for the original call signature used by test_media.py.
    def process(self, item: ExtractionItem) -> ExtractionItem:
        return self.process_media_url(item)


def apply_transcript(item: ExtractionItem, result: TranscriptResult) -> ExtractionItem:
    """Write a transcript onto an item, with the metadata that came with it."""
    transcript = result.timestamped()
    if not transcript.strip():
        return item.fail(Stage.DECODE, "transcription produced no text (silent media?)")

    # Media skips DECODE/PARSE, so the text fields are populated directly.
    item.decoded_text = transcript
    item.cleaned_text = transcript
    item.metadata.update(
        {
            "transcript_backend": result.backend,
            "transcript_model": result.model,
            "detected_language": result.language,
            "segments": len(result.segments),
        }
    )
    if result.duration:
        item.metadata.setdefault("duration", round(result.duration, 1))
    item.compute_content_hash()
    log.info(
        "media.transcribed",
        url=item.url,
        chars=len(transcript),
        segments=len(result.segments),
        backend=result.backend,
    )
    return item


class LiveStreamProcessor:
    """Capture a live stream in rolling segments and transcribe as it plays.

    The design constraint is that the stream has no end. So: never wait for the
    file. ffmpeg writes fixed-length segments; a segment is known complete once
    the *next* one has appeared, at which point it is transcribed and appended.
    Everything transcribed so far survives a timeout or a cancellation.
    """

    def __init__(
        self,
        *,
        segment_seconds: Optional[int] = None,
        max_segments: Optional[int] = None,
        max_minutes: Optional[int] = None,
    ) -> None:
        self.segment_seconds = segment_seconds or config.LIVESTREAM_SEGMENT_SECONDS
        self.max_segments = max_segments or config.LIVESTREAM_MAX_SEGMENTS
        self.max_minutes = max_minutes or config.LIVESTREAM_MAX_MINUTES

    def capture(
        self, item: ExtractionItem, on_segment: Optional[SegmentCallback] = None
    ) -> ExtractionItem:
        url = str(item.url)
        try:
            ffmpeg = _require_ffmpeg()
        except MissingDependency as exc:
            return item.fail_from(exc)

        stream_url, metadata = self._resolve(url)
        if stream_url is None:
            return item.fail(Stage.FETCH, f"could not resolve a stream URL for {url}")
        item.metadata.update(metadata)
        item.fetch_mode = FetchMode.YTDLP
        item.kind = ResourceKind.LIVESTREAM

        workdir = tempfile.mkdtemp(prefix="live_")
        deadline = time.monotonic() + self.max_minutes * 60
        pieces: list[str] = []
        transcribed: set[str] = set()
        process: Optional[subprocess.Popen] = None

        try:
            transcriber = get_transcriber()
            process = self._start_ffmpeg(ffmpeg, stream_url, workdir)
            log.info(
                "livestream.capturing",
                url=url,
                segment_seconds=self.segment_seconds,
                max_minutes=self.max_minutes,
            )

            while time.monotonic() < deadline and len(transcribed) < self.max_segments:
                if process.poll() is not None and not self._pending(workdir, transcribed):
                    log.info("livestream.ended", url=url, segments=len(transcribed))
                    break

                ready = self._pending(workdir, transcribed)
                if not ready:
                    time.sleep(2.0)
                    continue

                for path in ready:
                    if len(transcribed) >= self.max_segments:
                        break
                    index = len(transcribed)
                    offset = index * self.segment_seconds
                    try:
                        result = transcriber.transcribe(path)
                    except Exception as exc:
                        log.warning("livestream.segment_failed", index=index, error=repr(exc))
                        transcribed.add(path)
                        continue

                    text = self._offset_segment_text(result, offset)
                    transcribed.add(path)
                    os.unlink(path)  # a long stream must not fill the disk
                    if not text.strip():
                        continue

                    pieces.append(text)
                    joined = "\n".join(pieces)
                    item.cleaned_text = joined
                    item.decoded_text = joined
                    log.info("livestream.segment", index=index, chars=len(text), total=len(joined))
                    if on_segment is not None:
                        on_segment(index, joined, offset + self.segment_seconds)

        except Exception as exc:
            if not pieces:
                return item.fail(
                    Stage.DECODE, f"live capture failed: {exc}", error_type=type(exc).__name__
                )
            item.warn(f"live capture ended early: {exc}")
        finally:
            self._stop(process)
            shutil.rmtree(workdir, ignore_errors=True)
            gc.collect()

        if not pieces:
            return item.fail(Stage.DECODE, "live capture produced no transcribable audio")

        transcript = "\n".join(pieces)
        item.decoded_text = transcript
        item.cleaned_text = transcript
        item.metadata.update(
            {
                "live": True,
                "segments_captured": len(pieces),
                "captured_seconds": len(pieces) * self.segment_seconds,
                "capture_truncated": len(transcribed) >= self.max_segments
                or time.monotonic() >= deadline,
            }
        )
        item.compute_content_hash()
        return item

    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve(url: str) -> tuple[Optional[str], dict]:
        """Get a directly playable stream URL, and what is known about it."""
        if url.lower().split("?")[0].endswith((".m3u8", ".mpd")):
            return url, {"live": True, "source_url": url}

        try:
            options = {"quiet": True, "no_warnings": True, "format": "bestaudio/best"}
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            log.warning("livestream.resolve_failed", url=url, error=repr(exc))
            return None, {}

        if not info:
            return None, {}
        stream_url = info.get("url")
        if not stream_url:
            for candidate in info.get("formats", []) or []:
                if candidate.get("acodec") not in (None, "none"):
                    stream_url = candidate.get("url")
        metadata = {
            "title": info.get("title"),
            "author": info.get("uploader") or info.get("channel"),
            "live": bool(info.get("is_live")),
            "source_url": info.get("webpage_url") or url,
        }
        return stream_url, {k: v for k, v in metadata.items() if v is not None}

    def _start_ffmpeg(self, ffmpeg: str, stream_url: str, workdir: str) -> subprocess.Popen:
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "30",
            "-i", stream_url,
            "-vn",                       # audio only: video bytes are never read
            "-acodec", "libmp3lame", "-b:a", "64k", "-ar", "16000", "-ac", "1",
            "-f", "segment",
            "-segment_time", str(self.segment_seconds),
            "-reset_timestamps", "1",
            os.path.join(workdir, "seg%05d.mp3"),
        ]
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # Its own process group, so stopping it does not depend on how the
            # worker was signalled.
            start_new_session=True,
        )

    @staticmethod
    def _pending(workdir: str, done: set[str]) -> list[str]:
        """Segments that are finished being written.

        The newest file is still open, so it is excluded: transcribing a
        half-written MP3 gives you a truncated sentence and a wasted model call.
        """
        try:
            names = sorted(name for name in os.listdir(workdir) if name.startswith("seg"))
        except FileNotFoundError:
            return []
        paths = [os.path.join(workdir, name) for name in names]
        return [path for path in paths[:-1] if path not in done and os.path.getsize(path) > 4096]

    @staticmethod
    def _offset_segment_text(result: TranscriptResult, offset: float) -> str:
        """Re-base a segment's timestamps onto the whole-stream timeline."""
        if not result.segments:
            return result.text.strip()
        return "\n".join(
            f"[{segment.start + offset:.1f}s - {segment.end + offset:.1f}s] {segment.text.strip()}"
            for segment in result.segments
            if segment.text.strip()
        )

    @staticmethod
    def _stop(process: Optional[subprocess.Popen]) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, PermissionError):
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                pass


__all__ = ["LiveStreamProcessor", "MediaProcessor", "apply_transcript"]
