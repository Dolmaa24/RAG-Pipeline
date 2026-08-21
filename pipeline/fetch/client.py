"""The HTTP fetcher: one place where every safety gate is applied.

Order of operations for a single request, and why:

1. **Circuit breaker** — cheapest possible rejection for a host already known
   to be down or blocked. No socket, no DNS.
2. **Policy + robots** — asked *before* connecting, and again for the target of
   every redirect. ``follow_redirects=True`` would make a 302 onto a
   disallowed path invisible, which is the ordinary way a well-behaved-looking
   crawler ends up somewhere it was told not to go.
3. **Rate limiter** — per host, holding a slot for the duration of the request.
4. **Conditional request** — send the stored ETag; a 304 costs nothing.
5. **Block detection** — a challenge page returns HTTP 200, so this runs on
   success, not just on errors.
6. **Size cap** — enforced while streaming, so a 10 GB response is abandoned
   after the first 64 MiB rather than after it has filled the machine's RAM.

Retries cover transient failures only. A 403, a robots denial, and a challenge
page are all answers, not outages, and none of them is retried.
"""

from __future__ import annotations

import mimetypes
import random
import re
import time
from typing import Iterable, Optional

import httpx

from config import config
from errors import (
    BlockedError,
    CircuitOpen,
    ComplianceError,
    FetchError,
    TooLarge,
    TransientFetchError,
)
from models import ExtractionItem, FetchMode, Stage
from nettls import client_context, explain
from observability import get_logger, metrics
from pipeline.compliance import policy as default_policy
from uploads import is_upload_url, resolve_upload
from urls import canonicalize, host_of, is_http_url

from .blocks import block_guidance, detect_block
from .breaker import breaker as default_breaker
from .cache import http_cache as default_cache
from .ratelimit import rate_limiter as default_limiter

log = get_logger("fetch.client")

#: Worth retrying with the same client.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
#: Usually "you got bot-blocked" — worth one attempt with a real browser.
BROWSER_FALLBACK_STATUS = frozenset({401, 403, 429, 503})
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


#: Root elements a single-page app mounts into. Their presence alongside no
#: text is about as clear a "the content arrives later" signal as HTML offers.
_APP_ROOTS = (
    b'id="root"', b"id='root'", b'id="app"', b"id='app'",
    b'id="__next"', b'id="__nuxt"', b"data-reactroot", b"ng-app",
)
_TAGS = re.compile(rb"<(script|style|noscript)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)
#: Visible characters below which a document is not carrying its own content.
_SHELL_TEXT_THRESHOLD = 200


def _is_empty_shell(body: bytes) -> bool:
    """True when the markup carries almost no readable text."""
    sample = body[:200_000]
    text = _TAGS.sub(b" ", sample)
    visible = len(b" ".join(text.split()))

    if visible >= _SHELL_TEXT_THRESHOLD:
        return False
    lowered = sample.lower()
    # Little text *and* an app mount point, or little text in a body big enough
    # that the bytes clearly went somewhere other than prose.
    return any(marker in lowered for marker in _APP_ROOTS) or len(sample) > 2048


class FetchResult:
    """Everything one completed HTTP exchange produced."""

    __slots__ = ("body", "final_url", "from_cache", "headers", "mode", "redirects", "status")

    def __init__(
        self,
        status: int,
        headers: dict[str, str],
        body: bytes,
        final_url: str,
        redirects: list[str],
        *,
        from_cache: bool = False,
        mode: FetchMode = FetchMode.STATIC,
    ) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.final_url = final_url
        self.redirects = redirects
        self.from_cache = from_cache
        self.mode = mode


class ResilientFetcher:
    """Static fetch with a headless-browser fallback, behind every gate."""

    def __init__(
        self,
        *,
        policy=None,
        limiter=None,
        breaker=None,
        cache=None,
    ) -> None:
        self.policy = policy or default_policy
        self.limiter = limiter or default_limiter
        self.breaker = breaker or default_breaker
        self.cache = cache or default_cache

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def fetch(self, item: ExtractionItem, force_dynamic: bool = False) -> ExtractionItem:
        started = time.perf_counter()
        item.canonical_url = canonicalize(item.url)
        try:
            if is_upload_url(item.url):
                # A file the user handed us is already on this machine. There
                # is nothing to rate-limit, no robots.txt to consult and no
                # host to trip a circuit breaker, so none of that applies.
                self._apply(item, self._fetch_upload(item.url))
                metrics.incr("fetch.ok")
                return item

            if not is_http_url(item.url):
                raise FetchError(f"{item.url!r} is not an http(s) URL")

            if force_dynamic:
                result = self._fetch_browser(item.url)
            else:
                result = self._fetch_static(item.url)
                if self._should_try_browser(result) and config.USE_BROWSER_FALLBACK:
                    log.info("fetch.browser_fallback", url=item.url, status=result.status)
                    result = self._fetch_browser(item.url)

            self._apply(item, result)
            self.breaker.record_success(item.url)
            metrics.incr("fetch.ok")
            return item

        except (ComplianceError, CircuitOpen) as exc:
            metrics.incr("fetch.refused")
            return item.fail_from(exc)
        except BlockedError as exc:
            self.breaker.record_block(item.url, exc.context.get("signal", "block"))
            log.error("fetch.blocked", url=item.url, guidance=block_guidance(host_of(item.url)))
            metrics.incr("fetch.blocked")
            return item.fail_from(exc)
        except (FetchError, TooLarge) as exc:
            self.breaker.record_failure(item.url)
            metrics.incr("fetch.failed")
            return item.fail_from(exc)
        except Exception as exc:  # never let an unexpected error escape a stage
            self.breaker.record_failure(item.url)
            metrics.incr("fetch.failed")
            return item.fail(Stage.FETCH, f"unexpected fetch error: {exc}", error_type=type(exc).__name__)
        finally:
            item.record_timing("fetch", time.perf_counter() - started)

    # ------------------------------------------------------------------ #
    # Local path: uploads
    # ------------------------------------------------------------------ #

    def _fetch_upload(self, url: str) -> FetchResult:
        """Read an uploaded file from disk as though it had been fetched."""
        try:
            path = resolve_upload(url)
        except (ValueError, FileNotFoundError) as exc:
            raise FetchError(str(exc)) from exc

        size = path.stat().st_size
        if size > config.MAX_CONTENT_BYTES:
            raise TooLarge(url, size, config.MAX_CONTENT_BYTES)

        # A guess from the extension only. Detection proper runs on the bytes
        # in the next stage and overrules this, which is what makes a .pdf
        # that is really a ZIP land in the right handler anyway.
        guessed, _ = mimetypes.guess_type(path.name)
        return FetchResult(
            status=200,
            headers={"content-type": guessed or "application/octet-stream"},
            body=path.read_bytes(),
            final_url=url,
            redirects=[],
            mode=FetchMode.INLINE,
        )

    # ------------------------------------------------------------------ #
    # Static path
    # ------------------------------------------------------------------ #

    def _fetch_static(self, url: str) -> FetchResult:
        backoff = 1.0
        last: Optional[Exception] = None

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                return self._request_with_redirects(url)
            except TransientFetchError as exc:
                last = exc
                self.breaker.record_failure(url)
                if attempt < config.MAX_RETRIES:
                    delay = backoff + random.uniform(0.1, 0.5)
                    log.warning(
                        "fetch.retrying",
                        url=url,
                        attempt=attempt,
                        of=config.MAX_RETRIES,
                        reason=str(exc),
                        sleep=round(delay, 2),
                    )
                    time.sleep(delay)
                    backoff *= 2.0

        raise FetchError(f"static fetch failed after {config.MAX_RETRIES} attempts: {last}")

    def _request_with_redirects(self, url: str) -> FetchResult:
        """One logical fetch, following redirects by hand and re-checking each."""
        redirects: list[str] = []
        current = url

        for hop in range(config.MAX_REDIRECTS + 1):
            self.breaker.check(current)
            verdict = self.policy.enforce(current, hop=hop)
            self.limiter.apply_crawl_delay(current, verdict.crawl_delay)

            cached = self.cache.lookup(current)
            result = self._single_request(current, cached)

            if result.status in _REDIRECT_STATUS:
                location = result.headers.get("location")
                if not location:
                    raise FetchError(f"HTTP {result.status} with no Location header", url=current)
                target = httpx.URL(current).join(location)
                redirects.append(current)
                current = str(target)
                log.debug("fetch.redirect", from_url=redirects[-1], to=current, status=result.status)
                continue

            result.final_url = current
            result.redirects = redirects
            return result

        raise FetchError(f"more than {config.MAX_REDIRECTS} redirects", url=url, chain=redirects)

    def _single_request(self, url: str, cached) -> FetchResult:
        headers = self._headers()
        if cached is not None:
            headers.update(cached.validators())

        client_kwargs: dict = {
            "headers": headers,
            "http2": True,
            "follow_redirects": False,  # handled above, so every hop is checked
            "timeout": config.STATIC_TIMEOUT,
            "verify": client_context(),
        }
        if config.PROXY_URL:
            client_kwargs["proxy"] = config.PROXY_URL

        with self.limiter.slot(url):
            try:
                with httpx.Client(**client_kwargs) as client:
                    with client.stream("GET", url) as response:
                        response_headers = {k.lower(): v for k, v in response.headers.items()}
                        status = response.status_code
                        self.limiter.observe_response(url, status, response_headers)

                        if status == 304 and cached is not None:
                            entry = self.cache.on_not_modified(url, cached)
                            metrics.incr("fetch.not_modified")
                            return FetchResult(
                                200, entry.headers, entry.body, url, [], from_cache=True,
                                mode=FetchMode.CACHED,
                            )

                        if status in _REDIRECT_STATUS:
                            response.close()
                            return FetchResult(status, response_headers, b"", url, [])

                        declared = response_headers.get("content-length")
                        if declared and declared.isdigit() and int(declared) > config.MAX_CONTENT_BYTES:
                            raise TooLarge(url, int(declared), config.MAX_CONTENT_BYTES)

                        body = self._read_capped(response.iter_bytes(), url)

            except (TooLarge, ComplianceError):
                raise
            except httpx.TimeoutException as exc:
                raise TransientFetchError(f"timed out after {config.STATIC_TIMEOUT:.0f}s", url=url) from exc
            except httpx.TransportError as exc:
                raise TransientFetchError(explain(exc), url=url) from exc

        if config.DETECT_BLOCKS:
            signal = detect_block(status, response_headers, body)
            if signal is not None:
                raise BlockedError(url, str(signal))

        if status in RETRYABLE_STATUS:
            raise TransientFetchError(f"HTTP {status}", url=url, status=status)
        if status >= 400:
            raise FetchError(f"HTTP {status}", url=url, status=status)

        self.cache.store(url, status, response_headers, body)
        return FetchResult(status, response_headers, body, url, [])

    @staticmethod
    def _read_capped(chunks: Iterable[bytes], url: str) -> bytes:
        """Read the body, abandoning it the moment it exceeds the cap."""
        buffer = bytearray()
        limit = config.MAX_CONTENT_BYTES
        for chunk in chunks:
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise TooLarge(url, len(buffer), limit)
        return bytes(buffer)

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }

    # ------------------------------------------------------------------ #
    # Browser path
    # ------------------------------------------------------------------ #

    @staticmethod
    def _should_try_browser(result: FetchResult) -> bool:
        """Is a browser render worth several hundred milliseconds and 300 MB?

        Only when the response looks like an *empty shell*: markup that carries
        essentially no text, which is what a client-rendered app serves before
        its JavaScript runs. Judging by raw byte count alone was wrong — plenty
        of perfectly complete pages are under a kilobyte, and rendering every
        one of them in Chromium is a large cost for nothing.
        """
        if result.status in BROWSER_FALLBACK_STATUS:
            return True
        if result.status != 200:
            return False
        if not result.body:
            return True

        content_type = (result.headers.get("content-type") or "").lower()
        if content_type and not content_type.startswith(("text/html", "application/xhtml")):
            return False  # only HTML can be under-rendered

        return _is_empty_shell(result.body)

    def _fetch_browser(self, url: str) -> FetchResult:
        from .browser import render

        self.breaker.check(url)
        verdict = self.policy.enforce(url)
        self.limiter.apply_crawl_delay(url, verdict.crawl_delay)

        with self.limiter.slot(url):
            status, headers, html, final_url = render(url)

        body = html.encode("utf-8")
        if config.DETECT_BLOCKS:
            signal = detect_block(status or 200, headers, body, text=html)
            if signal is not None:
                raise BlockedError(url, str(signal))

        return FetchResult(
            status or 200, headers, body, final_url or url, [], mode=FetchMode.BROWSER
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply(item: ExtractionItem, result: FetchResult) -> None:
        item.status_code = result.status
        item.headers = result.headers
        item.raw_bytes = result.body
        item.final_url = result.final_url
        item.redirect_chain = result.redirects
        item.fetch_mode = result.mode
        item.content_type = result.headers.get("content-type")
        item.compute_content_hash()
        log.info(
            "fetch.ok",
            url=item.url,
            status=result.status,
            bytes=len(result.body),
            mode=result.mode.value,
            hops=len(result.redirects),
            cached=result.from_cache,
        )


__all__ = ["BROWSER_FALLBACK_STATUS", "RETRYABLE_STATUS", "FetchResult", "ResilientFetcher"]
