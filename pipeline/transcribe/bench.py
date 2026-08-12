"""Benchmark the Whisper backends on *your* machine and *your* audio.

Published Apple Silicon numbers for MLX vs. CTranslate2 vary enormously by
implementation, model size, audio length and thermal state. They are worth
treating as directional and nothing more. This runs both on a file you choose
and prints what actually happened here, which is the only comparison that
should decide the setting.

    PYTHONPATH=. ./venv/bin/python -m pipeline.transcribe.bench audio.mp3
    PYTHONPATH=. ./venv/bin/python -m pipeline.transcribe.bench audio.mp3 --repeat 3

What to look at: the realtime factor. 10x means a ten-minute recording takes one
minute. Note also that the *first* MLX run includes weight download and graph
construction, so ``--repeat 2`` or more is what shows steady-state speed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Optional


def audio_duration(path: str) -> Optional[float]:
    """Length in seconds, via ffprobe, so the realtime factor means something."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def run_backend(name: str, path: str, repeat: int) -> dict:
    from .base import _build

    try:
        backend = _build(name)
    except Exception as exc:
        return {"backend": name, "error": f"{type(exc).__name__}: {exc}"}

    timings: list[float] = []
    text = ""
    for index in range(repeat):
        started = time.perf_counter()
        try:
            result = backend.transcribe(path)
        except Exception as exc:
            return {"backend": name, "error": f"{type(exc).__name__}: {exc}"}
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        text = result.text
        label = "cold" if index == 0 else f"run {index + 1}"
        print(f"  {name:<8} {label:<7} {elapsed:7.2f}s", flush=True)

    return {
        "backend": name,
        "model": backend.model_name,
        "cold_seconds": round(timings[0], 2),
        "warm_seconds": round(min(timings[1:]), 2) if len(timings) > 1 else None,
        "chars": len(text),
        "preview": text[:160],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", help="Path to an audio or video file.")
    parser.add_argument("--repeat", type=int, default=2,
                        help="Runs per backend. The first includes model load.")
    parser.add_argument("--backends", nargs="+", default=["mlx", "faster"])
    args = parser.parse_args(argv)

    if not os.path.exists(args.audio):
        print(f"No such file: {args.audio}", file=sys.stderr)
        return 2

    duration = audio_duration(args.audio)
    print(f"\nAudio: {args.audio}")
    print(f"Length: {duration:.1f}s\n" if duration else "Length: unknown (ffprobe not found)\n")

    results = [run_backend(name, args.audio, args.repeat) for name in args.backends]

    print(f"\n{'backend':<10}{'model':<34}{'cold':>9}{'warm':>9}{'realtime':>11}")
    print("-" * 73)
    for result in results:
        if "error" in result:
            print(f"{result['backend']:<10}unavailable — {result['error'][:45]}")
            continue
        warm = result["warm_seconds"] or result["cold_seconds"]
        factor = f"{duration / warm:.1f}x" if duration and warm else "?"
        print(
            f"{result['backend']:<10}{result['model'][:33]:<34}"
            f"{result['cold_seconds']:>8.2f}s"
            f"{warm:>8.2f}s{factor:>11}"
        )

    usable = [r for r in results if "error" not in r]
    if len(usable) > 1:
        best = min(usable, key=lambda r: r["warm_seconds"] or r["cold_seconds"])
        print(f"\nFastest here: {best['backend']}. Set WHISPER_BACKEND={best['backend']} in .env.")
        print("Transcripts differ between backends; compare the previews before deciding.\n")
        for result in usable:
            print(f"  {result['backend']}: {result['preview']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
