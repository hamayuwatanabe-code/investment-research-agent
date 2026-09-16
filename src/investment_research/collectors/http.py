"""HTTP client: retries, rate limiting, caching, and honest failure reporting.

Requirement 14.  Deliberately built on the standard library so the core has no
third-party runtime dependency.

The most important property here is that a failure is *reported as a failure*.
A blocked host, a timeout or a 404 must never turn into a silently-empty result
that downstream code reads as "nothing to worry about".  Every fetch returns a
:class:`FetchResult` carrying an explicit :class:`FetchOutcome`.

Wire URL vs. public URL (Phase 3F.0.3): a caller (e.g.
``research/literature_acquisition_adapter.py``) may pass a URL carrying a
secret query parameter (``email``/``api_key``, NCBI's own courtesy-
identification convention) -- that URL is used for EXACTLY ONE thing, the
actual ``urllib.request.Request`` this module issues. Every ``FetchResult``
this module returns -- including on a cache hit, an offline/disabled
result, a redirect, and every failure/log line -- carries only the
SANITIZED ("public") form of that URL, via :func:`_sanitize_url`: a secret
query parameter is never retained on a returned object, in a log line, or
in an exception message. This has no effect on SEC/ClinicalTrials/Form4
callers, whose URLs never carry a secret-shaped query parameter to begin
with -- sanitizing a URL with none is a no-op, so their existing behavior
and tests are unchanged by this hardening.
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

#: Query-parameter names this module treats as secret-shaped and strips
#: before a URL is ever retained on a returned ``FetchResult``, logged, or
#: placed in an exception message (Phase 3F.0.3 requirement 1/2). Matches
#: ``research/literature_acquisition_adapter.py``'s own
#: ``_public_request_url`` convention exactly -- NCBI's courtesy-
#: identification parameters. ``tool`` is deliberately NOT included: it
#: carries no personal information or credential (a short software-
#: identifier string), matching ``config.Settings.secret_values()``'s own
#: exclusion of it.
_SECRET_QUERY_PARAM_NAMES = frozenset({"email", "api_key"})


def _sanitize_url(url: str) -> str:
    """Strips any query parameter named in ``_SECRET_QUERY_PARAM_NAMES``.
    A no-op for a URL that carries none of them -- which is every SEC/
    ClinicalTrials/Form4 URL today, so this changes nothing for those
    callers. Never raises; a URL this can't parse is returned unchanged
    rather than dropped (this is a redaction pass, not a validator)."""
    if "?" not in url:
        return url
    prefix, _, query = url.partition("?")
    if not any(f"{name}=" in query for name in _SECRET_QUERY_PARAM_NAMES):
        return url
    kept = [
        pair
        for pair in query.split("&")
        if pair and not any(pair.startswith(f"{name}=") for name in _SECRET_QUERY_PARAM_NAMES)
    ]
    return prefix + ("?" + "&".join(kept) if kept else "")


def _strip_wire_url(text: str, wire_url: str, public_url: str) -> str:
    """Belt-and-braces: replaces a literal wire_url substring (e.g. one
    embedded verbatim in a ``urllib`` exception's own message) with its
    already-sanitized public_url form. A no-op when the two are identical
    (the common case -- no secret was ever in the URL)."""
    if wire_url == public_url or wire_url not in text:
        return text
    return text.replace(wire_url, public_url)


@dataclass
class FetchResult:
    #: Always the SANITIZED ("public") URL -- see the module docstring's
    #: "Wire URL vs. public URL" section. Never the raw URL a secret-
    #: bearing caller passed to ``HttpClient.get()``.
    url: str
    outcome: FetchOutcome
    status: int | None = None
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    from_cache: bool = False
    error: str = ""
    elapsed_ms: int = 0
    #: Phase 3F.0.3: the URL the response actually came from, after any
    #: redirect -- sanitized the same way ``url`` is. Empty when no
    #: response was obtained (a pre-send rejection, an offline/disabled
    #: result) or when the transport does not track it (e.g. a cache hit,
    #: which replays a prior result verbatim). Equal to ``url`` when the
    #: response was not redirected.
    final_url: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome == FetchOutcome.OK

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any | None:
        """Parse JSON, returning ``None`` (not a guess) on malformed payloads.

        Phase 3F.0.3: ``self.url`` is already sanitized (see ``FetchResult``'s
        own docstring), so this log line was never a secret-leak path once
        that field itself became safe -- kept explicit here rather than
        relying on that invariant silently. The response BODY is never
        placed in the log line or in ``self.error``: only its length and
        content hash, which are diagnostic without being a body dump.
        ``json.JSONDecodeError``'s own ``str()`` carries a message plus a
        line/column/character position -- never the document text itself.
        """
        if not self.ok or not self.body:
            return None
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            log.warning(
                "invalid JSON from %s (%d bytes, hash=%s): %s",
                self.url, len(self.body), self.content_hash(), exc,
            )
            self.outcome = FetchOutcome.ERROR
            self.error = f"invalid JSON ({len(self.body)} bytes, hash={self.content_hash()}): {exc}"
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
        allowed_hosts: frozenset[str] | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(rate_limit_rps)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl_seconds
        self.offline = offline
        #: Phase 3F.0.3 requirement 4: when set, the FINAL response URL
        #: (after any redirect) is re-checked against this set; a redirect
        #: that leaves it is treated as BLOCKED, never silently accepted.
        #: ``None`` (the default) preserves this class's pre-existing
        #: behavior exactly for every current caller (SEC/ClinicalTrials/
        #: Form4 never pass this) -- opt-in only.
        self.allowed_hosts = allowed_hosts
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
                log.warning("cache write failed for %s: %s", _sanitize_url(url), exc)

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

        # wire_url is used for EXACTLY ONE thing below: the real
        # urllib.request.Request. public_url is what every returned
        # FetchResult, log line, and exception message uses instead (Phase
        # 3F.0.3 -- see the module docstring's "Wire URL vs. public URL").
        wire_url = url
        public_url = _sanitize_url(wire_url)

        started = time.monotonic()

        if use_cache:
            cached = self._read_cache(wire_url)
            if cached is not None:
                result = FetchResult(
                    url=public_url, outcome=FetchOutcome.OK, status=200, body=cached,
                    from_cache=True, final_url=public_url,
                )
                self.fetch_log.append(result)
                return result

        if self.offline:
            result = FetchResult(
                url=public_url,
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
            request = urllib.request.Request(wire_url, headers=request_headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    # Phase 3F.0.3 requirement 4: the URL actually reached,
                    # after any redirect -- re-validated against
                    # allowed_hosts (when the caller opted in) before the
                    # body is ever accepted, mirroring
                    # research/sec_live_smoke.py's AllowlistedHttpClient
                    # defense-in-depth check.
                    final_wire_url = response.geturl()
                    final_public_url = _sanitize_url(final_wire_url)
                    if self.allowed_hosts is not None:
                        final_host = urllib.parse.urlparse(final_wire_url).hostname
                        if final_host not in self.allowed_hosts:
                            result = FetchResult(
                                url=public_url, outcome=FetchOutcome.BLOCKED,
                                final_url=final_public_url, attempts=attempt,
                                error=f"final response host {final_host!r} not allowlisted",
                                elapsed_ms=int((time.monotonic() - started) * 1000),
                            )
                            self.fetch_log.append(result)
                            return result
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        with contextlib.suppress(OSError):
                            body = gzip.decompress(body)
                    status = int(response.status)
                    result = FetchResult(
                        url=public_url,
                        outcome=FetchOutcome.OK,
                        status=status,
                        body=body,
                        headers=dict(response.headers),
                        attempts=attempt,
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                        final_url=final_public_url,
                    )
                    if use_cache:
                        self._write_cache(wire_url, body)
                    self.fetch_log.append(result)
                    return result
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                last_error = _strip_wire_url(f"HTTP {status}: {exc.reason}", wire_url, public_url)
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
                last_error = _strip_wire_url(f"{type(exc).__name__}: {reason}", wire_url, public_url)
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
                    public_url,
                    backoff,
                    attempt,
                    self.max_retries,
                    last_error,
                )
                time.sleep(backoff)

        result = FetchResult(
            url=public_url,
            outcome=outcome,
            status=status,
            attempts=self.max_retries
            if outcome not in (FetchOutcome.BLOCKED, FetchOutcome.NOT_FOUND)
            else 1,
            error=last_error,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        self.fetch_log.append(result)
        log.warning("fetch failed: %s -> %s (%s)", public_url, outcome, last_error)
        return result
