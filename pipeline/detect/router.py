"""The TypeRouter: decide how to acquire a URL, then what its bytes are.

Two separate decisions, made at two different moments, because they answer
different questions:

* :func:`pre_route` runs **before** any bytes exist. All it has is the URL, and
  the only thing it decides is *how to acquire* the resource — a plain HTTP GET,
  or yt-dlp because the URL is a media platform page whose bytes are an HTML
  player, not the video. Getting this wrong means downloading a YouTube page
  and extracting fields from its navigation menu.

* :meth:`TypeRouter.route` runs **after** the bytes are in hand, and decides
  what they actually are, using :mod:`.magic`. This is where a ``.pdf`` URL
  that returned an HTML login page gets correctly classified as HTML.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Optional

from models import ExtractionItem, ResourceKind
from observability import get_logger
from urls import registrable_host

from .magic import Detection, detect

log = get_logger("detect.router")

#: Platforms whose page HTML is a player shell. yt-dlp knows how to get the
#: actual media; a plain GET does not.
MEDIA_DOMAINS: frozenset[str] = frozenset(
    {
        "youtube.com", "youtu.be", "youtube-nocookie.com",
        "vimeo.com", "dailymotion.com", "twitch.tv",
        "tiktok.com", "instagram.com", "facebook.com",
        "soundcloud.com", "spotify.com", "mixcloud.com", "audiomack.com",
        "bandcamp.com", "bitchute.com", "rumble.com", "odysee.com",
        "twitter.com", "x.com", "reddit.com",
        "bbc.co.uk", "arte.tv", "ted.com",
        "podcasts.apple.com", "pca.st", "anchor.fm", "buzzsprout.com",
        "libsyn.com", "podbean.com", "megaphone.fm", "simplecast.com",
    }
)

#: Live-streaming manifests. These never end, so they get the segment-capture
#: path rather than "download the file, then transcribe it".
_STREAM_EXTENSIONS = (".m3u8", ".mpd")

#: Hosts where a URL is a live stream far more often than not. Not a
#: certainty — yt-dlp's `is_live` flag is the authority, checked at download.
_LIVE_HINTS = ("/live", "/live/", "twitch.tv", "is_live=1")


class Acquisition(str):
    """How to get the bytes. A str subclass so it serialises straight to JSON."""

    HTTP = "http"
    YTDLP = "ytdlp"
    LIVESTREAM = "livestream"


@dataclass(frozen=True, slots=True)
class PreRoute:
    acquisition: str
    reason: str
    #: Best guess at the kind, refined once bytes exist.
    likely_kind: ResourceKind = ResourceKind.UNKNOWN


def pre_route(url: str) -> PreRoute:
    """Decide how to acquire ``url``, from the URL alone."""
    parsed = urllib.parse.urlparse(url.strip())
    host = registrable_host(parsed.netloc.lower())
    path = parsed.path.lower()

    if path.endswith(_STREAM_EXTENSIONS):
        return PreRoute(Acquisition.LIVESTREAM, "streaming manifest in the URL path", ResourceKind.LIVESTREAM)

    # Suffix match, so `evil-youtube.com.attacker.net` is not misrouted the way
    # a naive substring check would misroute it.
    if any(host == domain or host.endswith("." + domain) for domain in MEDIA_DOMAINS):
        if any(hint in url.lower() for hint in _LIVE_HINTS):
            return PreRoute(Acquisition.YTDLP, f"{host} is a media platform, possibly live", ResourceKind.VIDEO)
        return PreRoute(Acquisition.YTDLP, f"{host} is a media platform", ResourceKind.VIDEO)

    return PreRoute(Acquisition.HTTP, "ordinary HTTP resource")


class TypeRouter:
    """Classify fetched bytes and name the handler that should read them."""

    #: One handler per kind. Registered by :mod:`pipeline.handlers`, so this
    #: module has no import-time dependency on any parser library.
    _handlers: dict[ResourceKind, str] = {}

    @classmethod
    def register(cls, kind: ResourceKind, handler_name: str) -> None:
        cls._handlers[kind] = handler_name

    @staticmethod
    def classify(item: ExtractionItem) -> Detection:
        return detect(
            item.raw_bytes,
            content_type=item.content_type,
            url=item.final_url or item.url,
        )

    @classmethod
    def route(cls, item: ExtractionItem) -> ExtractionItem:
        """Set ``kind`` and ``handler`` on the item from its bytes."""
        detection = cls.classify(item)
        item.kind = detection.kind
        item.handler = cls._handlers.get(detection.kind)
        item.metadata.setdefault("detected_subtype", detection.subtype)
        item.metadata.setdefault("detection_source", detection.source)
        item.metadata.setdefault("detection_confidence", round(detection.confidence, 2))

        if detection.confidence < 0.5:
            item.warn(f"low-confidence type detection: {detection}")

        log.info(
            "route.classified",
            url=item.url,
            kind=detection.kind.value,
            subtype=detection.subtype,
            source=detection.source,
            handler=item.handler,
        )
        return item


class URLRouter:
    """Backwards-compatible shim for the original two-way web/media split.

    Kept so existing callers and the dashboard keep working. New code should
    use :func:`pre_route`, which distinguishes a live stream from a finished
    recording.
    """

    @staticmethod
    def detect_type(url: str) -> str:
        decision = pre_route(url)
        if decision.acquisition in (Acquisition.YTDLP, Acquisition.LIVESTREAM):
            return "media"

        # Last resort: ask the server what it is holding, without downloading it.
        try:
            import httpx

            from config import config

            with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                response = client.head(url, headers={"User-Agent": config.user_agent})
                content_type = response.headers.get("content-type", "").lower()
                if content_type.startswith(("video/", "audio/")):
                    return "media"
        except Exception as exc:
            log.debug("route.head_probe_skipped", url=url, error=repr(exc))

        return "web"


__all__ = ["MEDIA_DOMAINS", "Acquisition", "PreRoute", "TypeRouter", "URLRouter", "pre_route"]
