"""Headless Chromium rendering for pages that build themselves in JavaScript.

Playwright is expensive — a browser launch is hundreds of milliseconds and a
few hundred MB — so this is a fallback, never the first choice.

**Everything here runs on one dedicated owner thread**, and that is not
optional. Playwright's sync API is built on greenlets bound to the thread that
created them; calling it from a second thread fails with *"Cannot switch to a
different thread"*. The io worker runs ``--pool=threads -c 16``, so that second
thread is guaranteed. A lock is not enough — serialising the calls does not
change which thread owns the objects — so calls are marshalled to a single
long-lived thread that owns the browser for the life of the process.
"""

from __future__ import annotations

import atexit
import queue
import threading
from concurrent.futures import Future
from typing import Callable, Optional

from config import config
from errors import FetchError, TransientFetchError
from observability import get_logger

log = get_logger("fetch.browser")

#: (callable, future). A None callable tells the owner thread to exit.
_work: "queue.Queue[tuple[Optional[Callable], Optional[Future]]]" = queue.Queue()
_owner: Optional[threading.Thread] = None
_owner_lock = threading.Lock()
_playwright = None
_browser = None


def _owner_loop() -> None:
    """Own the browser and run every Playwright call on this one thread."""
    while True:
        job, future = _work.get()
        if job is None:
            _work.task_done()
            return
        try:
            result = job()
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller
            if future is not None and not future.set_running_or_notify_cancel():
                pass
            if future is not None:
                future.set_exception(exc)
        else:
            if future is not None:
                future.set_result(result)
        finally:
            _work.task_done()


def _ensure_owner() -> None:
    global _owner
    with _owner_lock:
        if _owner is not None and _owner.is_alive():
            return
        _owner = threading.Thread(target=_owner_loop, name="playwright-owner", daemon=True)
        _owner.start()


def _submit(job: Callable):
    """Run ``job`` on the owner thread and return its result here."""
    _ensure_owner()
    future: Future = Future()
    _work.put((job, future))
    return future.result()


# --------------------------------------------------------------------------- #
# Runs on the owner thread only
# --------------------------------------------------------------------------- #


def _ensure_browser():
    global _playwright, _browser
    if _browser is not None and _browser.is_connected():
        return _browser

    from playwright.sync_api import sync_playwright

    if _playwright is None:
        _playwright = sync_playwright().start()

    launch_kwargs: dict = {"headless": True, "args": ["--disable-dev-shm-usage"]}
    if config.PROXY_URL:
        launch_kwargs["proxy"] = {"server": config.PROXY_URL}
    _browser = _playwright.chromium.launch(**launch_kwargs)
    log.info("browser.launched")
    return _browser


def _render_on_owner(url: str, wait_until: str) -> tuple[Optional[int], dict, str, str]:
    try:
        browser = _ensure_browser()
    except Exception as exc:
        raise FetchError(f"could not start headless Chromium: {exc}") from exc

    context = None
    try:
        context = browser.new_context(
            user_agent=config.user_agent, locale="en-US", ignore_https_errors=False
        )
        page = context.new_page()
        # Images and fonts are bytes we never read. Blocking them typically
        # halves render time on a media-heavy page.
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "media", "font")
            else route.continue_(),
        )
        response = page.goto(url, wait_until=wait_until, timeout=config.BROWSER_TIMEOUT_MS)
        html = page.content()
        status = response.status if response else None
        headers = {k.lower(): v for k, v in (response.headers if response else {}).items()}
        final_url = page.url
        log.info("browser.rendered", url=url, status=status, bytes=len(html))
        return status, headers, html, final_url
    except Exception as exc:
        name = type(exc).__name__
        if "Timeout" in name:
            raise TransientFetchError(f"browser render timed out: {exc}", url=url) from exc
        raise FetchError(f"browser render failed: {exc}", url=url) from exc
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:  # pragma: no cover - teardown best effort
                pass


def _shutdown_on_owner() -> None:
    global _playwright, _browser
    if _browser is not None:
        try:
            _browser.close()
        except Exception:  # pragma: no cover
            pass
        _browser = None
    if _playwright is not None:
        try:
            _playwright.stop()
        except Exception:  # pragma: no cover
            pass
        _playwright = None


# --------------------------------------------------------------------------- #
# Public API — callable from any thread
# --------------------------------------------------------------------------- #


def render(url: str, *, wait_until: str = "networkidle") -> tuple[Optional[int], dict, str, str]:
    """Render ``url`` and return ``(status, headers, html, final_url)``."""
    return _submit(lambda: _render_on_owner(url, wait_until))


def shutdown() -> None:
    """Close the browser and stop the owner thread. Safe from any thread."""
    global _owner
    with _owner_lock:
        thread = _owner
    if thread is None or not thread.is_alive():
        return
    try:
        _submit(_shutdown_on_owner)
    except Exception:  # pragma: no cover - teardown best effort
        pass
    _work.put((None, None))
    thread.join(timeout=10)
    with _owner_lock:
        _owner = None


atexit.register(shutdown)

__all__ = ["render", "shutdown"]
