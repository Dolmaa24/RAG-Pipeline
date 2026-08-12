"""Type detection: what these bytes are, and how to have acquired them."""

from .magic import Detection, detect, from_content_type, probe_text, sniff_bytes
from .router import MEDIA_DOMAINS, Acquisition, PreRoute, TypeRouter, URLRouter, pre_route

__all__ = [
    "MEDIA_DOMAINS",
    "Acquisition",
    "Detection",
    "PreRoute",
    "TypeRouter",
    "URLRouter",
    "detect",
    "from_content_type",
    "pre_route",
    "probe_text",
    "sniff_bytes",
]
