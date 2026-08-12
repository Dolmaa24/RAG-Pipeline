"""What a crawl is allowed to visit, and what it is looking for.

Every link gets one of three verdicts, and keeping them separate is what makes
"find every PDF on this site" expressible at all:

``FOLLOW``
    An HTML page inside the scope. Fetch it to keep discovering, but do not
    extract from it — you asked for PDFs, not for the navigation page that
    happens to link to one.

``COLLECT``
    A target. Fetch it and run the full extraction pipeline over it.

``SKIP``
    Out of scope, the wrong type, already-known junk (``.css``, ``.js``,
    ``mailto:``), or a crawl trap.

The verdict is a *pre-fetch guess*, made from the URL alone, because fetching
everything to find out what it is defeats the point. It is deliberately
permissive — a URL with no extension is followed as a possible page — and the
authoritative decision is made after the bytes arrive, from the magic bytes.

Crawl traps are the thing that turns a crawl into an outage. A calendar widget
generates a URL per day until the heat death of the universe; a faceted product
filter generates one per combination of six checkboxes. Both look like ordinary
links. :meth:`CrawlScope.is_trap` catches the common shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlsplit

from models import ResourceKind
from urls import canonicalize, registrable_host, resolve

#: Extensions never worth fetching in a crawl: page furniture and media that
#: carries no text. Kept separate from the collect filter so that asking for
#: images explicitly still works.
JUNK_EXTENSIONS: frozenset[str] = frozenset(
    {
        "css", "js", "mjs", "map", "ico", "woff", "woff2", "ttf", "eot", "otf",
        "svg", "webmanifest", "xsl", "rss",
    }
)

#: Extension → the kind it usually indicates. Used only for the pre-fetch
#: guess; magic bytes decide for real once the bytes are in hand.
_EXTENSION_KIND: dict[str, ResourceKind] = {
    "pdf": ResourceKind.DOCUMENT, "doc": ResourceKind.DOCUMENT,
    "docx": ResourceKind.DOCUMENT, "xls": ResourceKind.DOCUMENT,
    "xlsx": ResourceKind.DOCUMENT, "ppt": ResourceKind.DOCUMENT,
    "pptx": ResourceKind.DOCUMENT, "odt": ResourceKind.DOCUMENT,
    "ods": ResourceKind.DOCUMENT, "epub": ResourceKind.DOCUMENT,
    "rtf": ResourceKind.DOCUMENT,
    "csv": ResourceKind.TABULAR, "tsv": ResourceKind.TABULAR,
    "json": ResourceKind.DATA, "xml": ResourceKind.DATA, "yaml": ResourceKind.DATA,
    "zip": ResourceKind.ARCHIVE, "tar": ResourceKind.ARCHIVE, "gz": ResourceKind.ARCHIVE,
    "7z": ResourceKind.ARCHIVE, "bz2": ResourceKind.ARCHIVE,
    "png": ResourceKind.IMAGE, "jpg": ResourceKind.IMAGE, "jpeg": ResourceKind.IMAGE,
    "gif": ResourceKind.IMAGE, "webp": ResourceKind.IMAGE, "tiff": ResourceKind.IMAGE,
    "mp3": ResourceKind.AUDIO, "wav": ResourceKind.AUDIO, "m4a": ResourceKind.AUDIO,
    "mp4": ResourceKind.VIDEO, "mov": ResourceKind.VIDEO, "webm": ResourceKind.VIDEO,
    "eml": ResourceKind.EMAIL,
    "html": ResourceKind.HTML, "htm": ResourceKind.HTML, "php": ResourceKind.HTML,
    "asp": ResourceKind.HTML, "aspx": ResourceKind.HTML, "jsp": ResourceKind.HTML,
}

#: Query parameters that generate a new URL for the same content. Following
#: them is how a crawl discovers ten thousand copies of one page.
_TRAP_PARAMS: frozenset[str] = frozenset(
    {
        "sessionid", "sid", "phpsessid", "jsessionid", "sort", "order", "orderby",
        "view", "display", "print", "share", "replytocom", "page_id_filter",
    }
)

_CALENDAR = re.compile(
    r"/(?:\d{4})/(?:\d{1,2})(?:/(?:\d{1,2}))?/?$|[?&](?:year|month|day|week)=\d+", re.IGNORECASE
)


class LinkVerdict(str, Enum):
    FOLLOW = "follow"
    COLLECT = "collect"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class LinkDecision:
    verdict: LinkVerdict
    reason: str

    def __bool__(self) -> bool:
        return self.verdict is not LinkVerdict.SKIP


@dataclass
class CrawlScope:
    """The rules for one crawl.

    ``collect_extensions`` is the file-type filter. With it set, HTML pages are
    followed but never extracted — the crawl is a search for files, and the
    pages are only the map. Leave it empty and every in-scope page is collected
    instead, which is the "extract this schema from every product page" shape.
    """

    start_url: str
    max_depth: int = 2
    max_pages: int = 500

    #: Restrict to the start URL's registrable host. Turning this off needs
    #: `allowed_hosts`, because an unbounded crawl of the open web is not a
    #: thing this pipeline should make easy to start by accident.
    same_site: bool = True
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)

    #: Extensions to collect, without the dot: {"pdf"}, {"pdf", "xlsx"}. Empty
    #: means "collect every in-scope page".
    collect_extensions: frozenset[str] = field(default_factory=frozenset)
    #: Kinds to collect once the bytes have been identified. Empty derives it
    #: from `collect_extensions`.
    collect_kinds: frozenset[ResourceKind] = field(default_factory=frozenset)

    #: Regexes a URL must match (any) / must not match (any).
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()

    follow_html: bool = True
    #: Honour `<meta name="robots" content="nofollow">` and rel="nofollow".
    respect_nofollow: bool = True
    detect_traps: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "_include", [re.compile(p, re.I) for p in self.include_patterns])
        object.__setattr__(self, "_exclude", [re.compile(p, re.I) for p in self.exclude_patterns])

    # ------------------------------------------------------------------ #

    @property
    def host(self) -> str:
        return registrable_host(self.start_url)

    @property
    def hosts(self) -> frozenset[str]:
        if self.allowed_hosts:
            return self.allowed_hosts
        return frozenset({self.host}) if self.same_site else frozenset()

    @property
    def effective_kinds(self) -> frozenset[ResourceKind]:
        """Kinds to extract, derived from the extension filter when not given."""
        if self.collect_kinds:
            return self.collect_kinds
        if not self.collect_extensions:
            return frozenset()  # empty means "collect everything in scope"
        return frozenset(
            _EXTENSION_KIND[ext] for ext in self.collect_extensions if ext in _EXTENSION_KIND
        )

    @property
    def collects_everything(self) -> bool:
        return not self.collect_extensions and not self.collect_kinds

    # ------------------------------------------------------------------ #

    def classify(self, url: str, *, depth: int, nofollow: bool = False) -> LinkDecision:
        """Decide what to do with one discovered link, before fetching it."""
        if not url or not url.lower().startswith(("http://", "https://")):
            return LinkDecision(LinkVerdict.SKIP, "not an http(s) URL")

        parts = urlsplit(url)
        host = registrable_host(url)
        path = parts.path.lower()
        extension = path.rsplit("/", 1)[-1].rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""

        allowed = self.hosts
        if allowed and host not in allowed:
            return LinkDecision(LinkVerdict.SKIP, f"off-site ({host})")

        if extension in JUNK_EXTENSIONS:
            return LinkDecision(LinkVerdict.SKIP, f"page furniture (.{extension})")

        if self._exclude and any(pattern.search(url) for pattern in self._exclude):
            return LinkDecision(LinkVerdict.SKIP, "matched an exclude pattern")

        if self.detect_traps:
            trap = self.is_trap(url)
            if trap:
                return LinkDecision(LinkVerdict.SKIP, f"crawl trap: {trap}")

        # A target is collected whatever its depth: the whole point of a crawl
        # for PDFs is that the PDF at depth 3 still counts.
        if self._is_target(extension):
            if self._include and not any(pattern.search(url) for pattern in self._include):
                return LinkDecision(LinkVerdict.SKIP, "did not match an include pattern")
            return LinkDecision(LinkVerdict.COLLECT, f"target file (.{extension or '?'})")

        if depth >= self.max_depth:
            return LinkDecision(LinkVerdict.SKIP, f"at the depth limit ({self.max_depth})")

        if nofollow and self.respect_nofollow:
            return LinkDecision(LinkVerdict.SKIP, "rel=nofollow")

        if not self.follow_html:
            return LinkDecision(LinkVerdict.SKIP, "not following pages")

        # No extension, or a page extension: treat as a page worth following.
        kind = _EXTENSION_KIND.get(extension)
        if extension and kind is not ResourceKind.HTML:
            return LinkDecision(LinkVerdict.SKIP, f"not a target and not a page (.{extension})")

        if self._include and not any(pattern.search(url) for pattern in self._include):
            # Still followed: a page outside the include pattern may link to
            # one inside it. The pattern filters what is *collected*.
            return LinkDecision(LinkVerdict.FOLLOW, "page (outside include, followed for links)")

        return LinkDecision(LinkVerdict.FOLLOW, "page")

    def _is_target(self, extension: str) -> bool:
        if self.collects_everything:
            return False  # everything in scope is collected via FOLLOW + extract
        return extension in self.collect_extensions

    def should_extract(self, kind: ResourceKind, url: str) -> bool:
        """Authoritative decision, made after the magic bytes are known."""
        if self._exclude and any(pattern.search(url) for pattern in self._exclude):
            return False
        if self._include and not any(pattern.search(url) for pattern in self._include):
            return False
        kinds = self.effective_kinds
        if not kinds:
            return True  # collect everything in scope
        return kind in kinds

    # ------------------------------------------------------------------ #

    def is_trap(self, url: str) -> Optional[str]:
        """Name the trap shape, or None. Cheap checks first."""
        parts = urlsplit(url)
        segments = [s for s in parts.path.split("/") if s]

        if len(segments) > 12:
            return "path is implausibly deep"

        # /a/b/a/b/a/b — a relative-link bug on the server, and endless.
        for size in (1, 2, 3):
            if len(segments) >= size * 3:
                window = segments[-size * 3 :]
                if window[:size] == window[size : size * 2] == window[size * 2 :]:
                    return "repeated path segments"

        if _CALENDAR.search(url):
            return "calendar-style URL"

        query = parse_qsl(parts.query, keep_blank_values=True)
        if len(query) > 6:
            return "too many query parameters"
        for name, _ in query:
            if name.lower() in _TRAP_PARAMS:
                return f"session or view parameter ({name})"

        if len(url) > 2000:
            return "URL is implausibly long"
        return None

    # ------------------------------------------------------------------ #

    def normalize(self, base_url: str, href: str) -> str:
        """Absolute, canonical form of a link seen on ``base_url``."""
        absolute = resolve(base_url, href)
        return canonicalize(absolute) if absolute else ""

    def to_dict(self) -> dict:
        return {
            "start_url": self.start_url,
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "same_site": self.same_site,
            "allowed_hosts": sorted(self.hosts),
            "collect_extensions": sorted(self.collect_extensions),
            "include_patterns": list(self.include_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "follow_html": self.follow_html,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CrawlScope":
        """Rebuild a scope from :meth:`to_dict`.

        Celery arguments must be JSON, so the scope crosses the queue as a plain
        dict and is rebuilt inside each task. Round-tripping through the same
        two methods keeps that from drifting.
        """
        data = dict(payload or {})
        return cls.build(
            data.get("start_url", ""),
            collect_extensions=data.get("collect_extensions") or (),
            allowed_hosts=data.get("allowed_hosts") or (),
            include_patterns=data.get("include_patterns") or (),
            exclude_patterns=data.get("exclude_patterns") or (),
            max_depth=int(data.get("max_depth", 2)),
            max_pages=int(data.get("max_pages", 500)),
            same_site=bool(data.get("same_site", True)),
            follow_html=bool(data.get("follow_html", True)),
        )

    @classmethod
    def build(
        cls,
        start_url: str,
        *,
        collect_extensions: Optional[Iterable[str]] = None,
        allowed_hosts: Optional[Iterable[str]] = None,
        include_patterns: Optional[Iterable[str]] = None,
        exclude_patterns: Optional[Iterable[str]] = None,
        **options,
    ) -> "CrawlScope":
        """Build a scope from loose input, normalising extensions and hosts."""
        extensions = frozenset(
            ext.lower().lstrip(".") for ext in (collect_extensions or ()) if ext.strip()
        )
        hosts = frozenset(registrable_host(h) for h in (allowed_hosts or ()) if h.strip())
        return cls(
            start_url=start_url,
            collect_extensions=extensions,
            allowed_hosts=hosts,
            include_patterns=tuple(include_patterns or ()),
            exclude_patterns=tuple(exclude_patterns or ()),
            **options,
        )


__all__ = ["JUNK_EXTENSIONS", "CrawlScope", "LinkDecision", "LinkVerdict"]
