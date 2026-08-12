"""Backwards-compatible import site for the router.

The real implementation moved to :mod:`pipeline.detect.router` when the two-way
web/media split grew into the full type router. Existing imports of
``pipeline.router.URLRouter`` keep working.
"""

from pipeline.detect.router import (
    MEDIA_DOMAINS,
    Acquisition,
    PreRoute,
    TypeRouter,
    URLRouter,
    pre_route,
)

__all__ = [
    "MEDIA_DOMAINS",
    "Acquisition",
    "PreRoute",
    "TypeRouter",
    "URLRouter",
    "pre_route",
]
