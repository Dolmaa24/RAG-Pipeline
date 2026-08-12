"""Backwards-compatible import site for the fetcher.

The implementation moved to :mod:`pipeline.fetch.client` when the compliance
gates were added. This module is deliberately a re-export rather than the old
code: the previous ``ResilientFetcher`` consulted no robots.txt, applied no
rate limiting, followed redirects blindly, and treated a Cloudflare
interstitial as a successful fetch. Leaving it importable would mean a stray
``from pipeline.fetcher import ResilientFetcher`` silently bypassed every gate
Phase 1 exists to enforce.
"""

from pipeline.fetch.client import (
    BROWSER_FALLBACK_STATUS,
    RETRYABLE_STATUS,
    FetchResult,
    ResilientFetcher,
)

__all__ = [
    "BROWSER_FALLBACK_STATUS",
    "RETRYABLE_STATUS",
    "FetchResult",
    "ResilientFetcher",
]
