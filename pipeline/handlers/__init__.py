"""Handler registry, and the wiring that populates it.

Import cost decides what is eager and what is lazy. The HTML, data, feed,
archive, email, document and image handlers pull in nothing expensive at module
scope — their heavy dependencies (MarkItDown, PyMuPDF, Vision) are imported
inside the methods that need them. The media handlers pull in ``yt_dlp`` and a
Whisper runtime, which is seconds of import time and hundreds of MB, so a
worker that only ever fetches HTML should never pay for them.
"""

from __future__ import annotations

from models import ResourceKind

from .base import BaseHandler, Handler, HandlerRegistry, registry

# Cheap at import time — register immediately.
from . import archive as _archive  # noqa: F401
from . import data as _data  # noqa: F401
from . import document as _document  # noqa: F401
from . import email as _email  # noqa: F401
from . import feed as _feed  # noqa: F401
from . import html as _html  # noqa: F401
from . import image as _image  # noqa: F401


def _load_media() -> BaseHandler:
    from .media import MediaHandler

    return MediaHandler()


def _load_livestream() -> BaseHandler:
    from .livestream import LivestreamHandler

    return LivestreamHandler()


registry.register_lazy((ResourceKind.AUDIO, ResourceKind.VIDEO), _load_media)
registry.register_lazy((ResourceKind.LIVESTREAM,), _load_livestream)

# The router needs to know which handler claims each kind so it can name it on
# the item before dispatch.
from pipeline.detect.router import TypeRouter  # noqa: E402

for _kind, _handler in (
    (ResourceKind.HTML, "html"),
    (ResourceKind.DOCUMENT, "document"),
    (ResourceKind.IMAGE, "image"),
    (ResourceKind.AUDIO, "media"),
    (ResourceKind.VIDEO, "media"),
    (ResourceKind.LIVESTREAM, "livestream"),
    (ResourceKind.FEED, "feed"),
    (ResourceKind.SITEMAP, "sitemap"),
    (ResourceKind.TABULAR, "tabular"),
    (ResourceKind.DATA, "data"),
    (ResourceKind.ARCHIVE, "archive"),
    (ResourceKind.EMAIL, "email"),
    (ResourceKind.TEXT, "text"),
    (ResourceKind.UNKNOWN, "text"),
):
    TypeRouter.register(_kind, _handler)

__all__ = ["BaseHandler", "Handler", "HandlerRegistry", "registry"]
