"""Media path only — download, transcribe, extract. No Redis, no API.

    PYTHONPATH=. ./venv/bin/python test_media.py "https://www.youtube.com/watch?v=..."
    PYTHONPATH=. ./venv/bin/python test_media.py --live "https://twitch.tv/someone"

Useful for checking that ffmpeg, yt-dlp and the Whisper backend are all working
before putting a media URL through the queue, where the same failure would
arrive as a task id and a stack trace.
"""

from __future__ import annotations

import argparse
import json
import sys

from config import config
from models import ExtractionItem
from observability import configure_logging, get_logger
from pipeline.extract.cascade import ExtractionCascade
from pipeline.media_processor import LiveStreamProcessor, MediaProcessor
from pipeline.normalizer import DataNormalizer

log = get_logger("test_media")

DEFAULT_PROMPT = (
    "Summarise this recording: its main topic, the key points made, and any "
    "named people or organisations mentioned."
)
DEFAULT_SCHEMA = {
    "topic": "string",
    "summary": "string",
    "key_points": "list of strings",
    "people": "list of strings",
    "organisations": "list of strings",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="A media platform URL, or a direct audio/video link.")
    parser.add_argument("--live", action="store_true",
                        help="Treat it as a live stream: capture rolling segments.")
    parser.add_argument("--minutes", type=int, default=None,
                        help="Cap a live capture (default: LIVESTREAM_MAX_MINUTES).")
    parser.add_argument("--no-extract", action="store_true",
                        help="Stop after the transcript; do not run extraction.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level, config.LOG_FORMAT)
    item = ExtractionItem(url=args.url)

    if args.live:
        processor = LiveStreamProcessor(max_minutes=args.minutes or config.LIVESTREAM_MAX_MINUTES)
        item = processor.capture(
            item,
            on_segment=lambda index, text, seconds: print(
                f"  … segment {index + 1}: {seconds / 60:.0f} min, {len(text)} chars", flush=True
            ),
        )
    else:
        item = MediaProcessor().process_media_url(item)

    if not item.ok:
        print(f"\nFAILED at {item.failed_at_stage}: {item.error}", file=sys.stderr)
        return 1

    transcript = item.cleaned_text or ""
    print("\n--- metadata ---")
    print(json.dumps(item.metadata, indent=2, ensure_ascii=False, default=str))
    print(f"\n--- transcript ({len(transcript)} chars) ---")
    print(transcript[:2000] + ("…" if len(transcript) > 2000 else ""))

    if args.no_extract:
        return 0

    print("\n--- extracting ---")
    item = ExtractionCascade().extract(item, DEFAULT_PROMPT, DEFAULT_SCHEMA)
    if not item.ok:
        print(f"Extraction failed: {item.error}", file=sys.stderr)
        return 1

    item = DataNormalizer.normalize(item)
    print(f"tier {item.tier} ({item.method.value}), confidence {item.confidence}")
    print(json.dumps(item.normalized_data, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
