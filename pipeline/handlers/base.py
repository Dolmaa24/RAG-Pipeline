"""Handler protocol and registry.

A handler turns bytes of one kind into text, metadata, and — where the format
already carries structure — structured fields and child items. It never calls a
model. That separation is what lets the extraction cascade decide *afterwards*
whether a model is needed at all.

Every handler's contract:

* reads  ``item.raw_bytes`` (and ``item.url`` for context);
* writes ``item.decoded_text`` / ``item.cleaned_text``, ``item.metadata``,
  optionally ``item.structured`` and ``item.children``;
* on failure calls ``item.fail(Stage.PARSE, ...)`` rather than raising.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Optional, Protocol

from errors import MissingDependency, PipelineError, UnsupportedType
from models import ExtractionItem, ResourceKind, Stage
from observability import get_logger, metrics

log = get_logger("handlers")


class Handler(Protocol):
    name: str
    kinds: tuple[ResourceKind, ...]

    def handle(self, item: ExtractionItem) -> ExtractionItem: ...


class BaseHandler:
    """Common plumbing: timing, error trapping, and empty-output checking."""

    name: str = "base"
    kinds: tuple[ResourceKind, ...] = ()
    #: Handlers that produce no text (a sitemap yields URLs, not prose) opt out
    #: of the "did you actually extract anything?" check.
    requires_text: bool = True

    def handle(self, item: ExtractionItem) -> ExtractionItem:
        started = time.perf_counter()
        item.handler = self.name
        try:
            item = self.process(item)
            if item.ok and self.requires_text and not item.text_for_extraction.strip():
                if not item.children and not item.structured:
                    return item.fail(Stage.PARSE, f"{self.name} produced no text")
            metrics.incr(f"handler.{self.name}.ok")
            return item
        except MissingDependency as exc:
            metrics.incr(f"handler.{self.name}.missing_dependency")
            return item.fail_from(exc)
        except PipelineError as exc:
            metrics.incr(f"handler.{self.name}.failed")
            return item.fail_from(exc)
        except Exception as exc:
            metrics.incr(f"handler.{self.name}.failed")
            log.warning("handler.crashed", handler=self.name, url=item.url, error=repr(exc))
            return item.fail(
                Stage.PARSE, f"{self.name} failed: {exc}", error_type=type(exc).__name__
            )
        finally:
            item.record_timing(f"handler.{self.name}", time.perf_counter() - started)

    def process(self, item: ExtractionItem) -> ExtractionItem:  # pragma: no cover - abstract
        raise NotImplementedError


class HandlerRegistry:
    """Maps a :class:`~models.ResourceKind` to the handler that reads it."""

    def __init__(self) -> None:
        self._by_kind: dict[ResourceKind, BaseHandler] = {}
        self._by_name: dict[str, BaseHandler] = {}
        self._factories: dict[ResourceKind, Callable[[], BaseHandler]] = {}

    def register(self, handler: BaseHandler) -> BaseHandler:
        self._by_name[handler.name] = handler
        for kind in handler.kinds:
            self._by_kind[kind] = handler
        return handler

    def register_lazy(self, kinds: Iterable[ResourceKind], factory: Callable[[], BaseHandler]) -> None:
        """Defer construction until first use.

        Whisper, MarkItDown and Playwright each cost real time and memory to
        import. A worker that only ever fetches HTML should never pay for them.
        """
        for kind in kinds:
            self._factories[kind] = factory

    def get(self, kind: ResourceKind) -> Optional[BaseHandler]:
        handler = self._by_kind.get(kind)
        if handler is not None:
            return handler
        factory = self._factories.get(kind)
        if factory is None:
            return None
        handler = factory()
        self.register(handler)
        return handler

    def by_name(self, name: str) -> Optional[BaseHandler]:
        return self._by_name.get(name)

    @property
    def kinds(self) -> tuple[ResourceKind, ...]:
        return tuple(sorted(set(self._by_kind) | set(self._factories), key=lambda k: k.value))

    def dispatch(self, item: ExtractionItem) -> ExtractionItem:
        handler = self.get(item.kind)
        if handler is None:
            subtype = item.metadata.get("detected_subtype", "?")
            return item.fail_from(UnsupportedType(item.kind.value, subtype))
        return handler.handle(item)


registry = HandlerRegistry()

__all__ = ["BaseHandler", "Handler", "HandlerRegistry", "registry"]
