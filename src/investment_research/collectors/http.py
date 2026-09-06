"""HTTP client: retries, rate limiting, caching, and honest failure reporting.

Requirement 14.  Deliberately built on the standard library so the core has no
third-party runtime dependency.

The most important property here is that a failure is *reported as a failure*.
A blocked host, a timeout or a 404 must never turn into a silently-empty result
that downstream code reads as "nothing to worry about".  Every fetch returns a
:class:`FetchResult` carrying an explicit :class:`FetchOutcome`.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas.enums import FetchOutcome

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "investment-research-agent (contact: set IRA_SEC_USER_AGENT)"


@dataclass
class FetchResult:
    url: str
    outcome: FetchOutcome
    status: int | None = None
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    from_cache: bool = False
    error: str = ""
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome == FetchOutcome.OK

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any | None:
        """Parse JSON, returning ``None`` (not a guess) on malformed payloads."""
        if not self.ok or not self.body:
            return None
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            log.warning("invalid JSON from %s: %s", self.url, exc)
            self.outcome = FetchOutcome.ERROR
            self.error = f"invalid JSON: {exc}"
            return None

    def content_hash(self) -> str:
        return hashlib.sha256(self.body).hexdigest()[:32] if self.body else "UNKNOWN"


class RateLimiter:
    """Simple thread-safe token-bucket-ish spacing limiter."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = self._last + self._min_interval - now
            if delta > 0:
                time.sleep(delta)
            self._last = time.monotonic()


#: Statuses that mean "the request itself is fine, try again later".
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
#: Statuses that mean "an intermediary refused"; retrying is pointless and, for
#: an egress policy denial, actively wrong (see /root/.ccr/README.md).
_BLOCKED_STATUS = {403, 407}


class HttpClient:
    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 30.0,
        max_retries: int = 4,
        rate_limit_rps: float = 5.0,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: int = 6 * 3600,
        offline: bool = False,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(rate_limit_rps)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl_seconds
        self.offline = offline
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fetch_log: list[FetchResult] = []

    # -- cache -------------------------------------------------------------
    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.bin"

    def _read_cache(self, url: str) -> bytes | None:
        path = self._cache_path(url)
        if path is None or not path.is_file():
            return None
        if self.cache_ttl and (time.time() - path.stat().st_mtime) > self.cache_ttl:
            return None
        return path.read_bytes()

    def _write_cache(self, url: str, body: bytes) -> None:
        path = self._cache_path(url)
        if path is not None:
            try:
                path.write_bytes(body)
            except OSError as exc:  # pragma: no cover - disk pressure
                log.warning("cache write failed for %s: %s", url, exc)

    # -- fetch -------------------------------------------------------------
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        use_cache: bool = True,
        accept: str = "application/json, text/html;q=0.9, */*;q=0.8",
    ) -> FetchResult:
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"

        started = time.monotonic()

        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                result = FetchResult(
                    url=url, outcome=FetchOutcome.OK, status=200, body=cached, from_cache=True
                )
                self.fetch_log.append(result)
                return result

        if self.offline:
            result = FetchResult(
                url=url,
                outcome=FetchOutcome.DISABLED,
                error="offline mode: no network call attempted",
            )
            self.fetch_log.append(result)
            return result

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Encoding": "gzip",
        }
        request_headers.update(dict(headers or {}))

        last_error = ""
        status: int | None = None
        outcome = FetchOutcome.ERROR

        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            request = urllib.request.Request(url, headers=request_headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        with contextlib.suppress(OSError):
                            body = gzip.decompress(body)
                    status = int(response.status)
                    result = FetchResult(
                        url=url,
                        outcome=FetchOutcome.OK,
                        status=status,
                        body=body,
                        headers=dict(response.headers),
                        attempts=attempt,
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                    )
                    if use_cache:
                        self._write_cache(url, body)
                    self.fetch_log.append(result)
                    return result
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                last_error = f"HTTP {status}: {exc.reason}"
                if status in _BLOCKED_STATUS:
                    # Do NOT retry a policy denial and do NOT route around it.
                    outcome = FetchOutcome.BLOCKED
                    break
                if status == 404:
                    outcome = FetchOutcome.NOT_FOUND
                    break
                if status == 429:
                    outcome = FetchOutcome.RATE_LIMITED
                elif status in _RETRYABLE_STATUS:
                    outcome = FetchOutcome.ERROR
                else:
                    outcome = FetchOutcome.ERROR
                    break
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                last_error = f"{type(exc).__name__}: {reason}"
                text = str(reason).lower()
                if "timed out" in text or isinstance(exc, TimeoutError):
                    outcome = FetchOutcome.TIMEOUT
                elif "403" in text or "tunnel" in text or "forbidden" in text:
                    # A CONNECT tunnel refusal from an egress proxy.
                    outcome = FetchOutcome.BLOCKED
                    break
                else:
                    outcome = FetchOutcome.ERROR

            if attempt < self.max_retries:
                backoff = 2.0 ** (attempt - 1)
                log.info(
                    "retrying %s in %.1fs (attempt %d/%d): %s",
                    url,
                    backoff,
                    attempt,
                    self.max_retries,
                    last_error,
                )
                time.sleep(backoff)

        result = FetchResult(
            url=url,
            outcome=outcome,
            status=status,
            attempts=self.max_retries if outcome not in (FetchOutcome.BLOCKED, FetchOutcome.NOT_FOUND) else 1,
            error=last_error,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        self.fetch_log.append(result)
        log.warning("fetch failed: %s -> %s (%s)", url, outcome, last_error)
        return result
