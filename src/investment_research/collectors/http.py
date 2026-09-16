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

Retry semantics and pre-connect redirect validation (Phase 3F.0.4):
``max_retries`` is the number of retries AFTER the first attempt -- so
``max_retries=0`` still makes exactly one real request, never zero, and
``max_retries=N`` makes at most ``N + 1`` attempts in total. This matches
``research/sec_live_smoke.py``'s ``AllowlistedHttpClient``, which already
had this semantics; this module previously did not (a range() off-by-one
made ``max_retries=0`` skip the request entirely -- fixed here). Separately,
when a caller opts into ``allowed_hosts``, a redirect's scheme/userinfo/host
is now validated BEFORE urllib ever connects to it -- the previous
``final_url`` re-check (kept, as defense in depth) only runs AFTER the
connection to the redirect target has already been made, which is too late
to prevent it.
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


class RedirectRejectedError(Exception):
    """Raised by :func:`_validate_redirect_target` when a redirect target
    fails scheme/userinfo/host validation -- checked BEFORE urllib ever
    connects to it (Phase 3F.0.4). Carries only a safe, URL-free reason
    string (host/scheme, never the rejected URL itself), so it is always
    safe to log or place directly in a ``FetchResult.error``."""


class TooManyRedirectsError(Exception):
    """Raised when a redirect chain exceeds the configured cap -- an
    explicit, typed limit independent of urllib's own internal one (10), so
    a redirect loop or an excessive chain surfaces as a clear, catchable
    failure rather than an opaque ``HTTPError`` (Phase 3F.0.4)."""


#: Default cap on redirects followed in one request when a caller opts into
#: ``allowed_hosts`` -- generous enough for a normal same-host redirect
#: chain, low enough that a loop fails fast and explicitly rather than
#: silently riding out urllib's own internal cap of 10.
DEFAULT_MAX_REDIRECTS = 5


def _validate_redirect_target(url: str, allowed_hosts: frozenset[str], *, current_scheme: str) -> None:
    """Validates a redirect target BEFORE urllib ever opens a connection to
    it -- a post-hoc check of ``response.geturl()`` is too late, since the
    connection to the (possibly disallowed) target has already been made by
    the time a response exists to inspect. Raises
    :class:`RedirectRejectedError` rather than returning a boolean, so a
    caller cannot accidentally ignore a rejection.

    ``current_scheme`` is the scheme of the request being redirected FROM
    (``req.type``, or ``"https"`` if that cannot be determined -- the safe
    default that never silently skips the downgrade check below). Passed
    explicitly rather than assumed, so this never breaks a legitimate
    plain-http redirect chain (e.g. a loopback test server) while still
    catching a REAL downgrade -- an https request whose redirect points at
    anything other than https.

    Rejects, in order:

    * a URL carrying userinfo (``user:pass@host``) outright -- rather than
      trusting any one parser's resolution of a ``host@actually-the-host``
      style ambiguity, a userinfo component is simply never valid here;
      none of this module's real callers ever need one.
    * any scheme other than ``http``/``https`` outright (``file://``,
      ``ftp://``, ...), regardless of ``current_scheme``.
    * a redirect FROM ``https`` TO anything other than ``https`` -- this
      would newly expose a secret query parameter in plaintext, and every
      real caller (SEC/NCBI/Europe PMC) is https-only, so this is a
      downgrade, not a legitimate same-scheme redirect.
    * a host (matched case-insensitively; ``.hostname`` is already
      lowercased by :mod:`urllib.parse`, lowercased again here for clarity
      and defense in depth) not present in ``allowed_hosts``.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise RedirectRejectedError("refused redirect: URL carries userinfo")
    if parsed.scheme not in ("http", "https"):
        raise RedirectRejectedError(f"refused redirect: unsupported scheme {parsed.scheme!r}")
    if current_scheme == "https" and parsed.scheme != "https":
        raise RedirectRejectedError(f"refused redirect: downgrade from https to {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if host not in allowed_hosts:
        raise RedirectRejectedError(f"refused redirect to disallowed host {host!r}")


class _PreConnectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates -- and counts -- a redirect BEFORE urllib follows it.
    Mirrors ``research/sec_live_smoke.py``'s own
    ``_AllowlistedRedirectHandler`` (which now shares this same validation
    function). Used by :class:`HttpClient` only when the caller opts into
    ``allowed_hosts`` -- ``None`` (the default) never builds one of these,
    so existing SEC/ClinicalTrials/Form4 callers are unaffected (Phase
    3F.0.4 requirement 2)."""

    def __init__(self, allowed_hosts: frozenset[str], max_redirects: int) -> None:
        super().__init__()
        self._allowed_hosts = allowed_hosts
        self._max_redirects = max_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        #: urllib gives no built-in way to thread state across a redirect
        #: chain, so the count is carried as a dynamic attribute on the
        #: Request object itself -- redirect_request returns a NEW Request
        #: each time, and that new object is what gets passed back in on
        #: the next redirect, so setting it here (below) is what makes the
        #: count visible on the following call.
        count = getattr(req, "_ira_redirect_count", 0) + 1
        if count > self._max_redirects:
            raise TooManyRedirectsError(f"redirect count exceeded cap of {self._max_redirects}")
        current_scheme = getattr(req, "type", None) or "https"
        _validate_redirect_target(newurl, self._allowed_hosts, current_scheme=current_scheme)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            new_req._ira_redirect_count = count
        return new_req


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
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(rate_limit_rps)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl_seconds
        self.offline = offline
        #: Phase 3F.0.3 requirement 4 / Phase 3F.0.4 requirement 2: when
        #: set, (a) the initial request's host and (b) every redirect's
        #: scheme/userinfo/host are validated -- the redirect check BEFORE
        #: urllib ever connects to it, not merely after -- and the FINAL
        #: response URL is re-checked too, as defense in depth. ``None``
        #: (the default) preserves this class's pre-existing behavior
        #: exactly for every current caller (SEC/ClinicalTrials/Form4 never
        #: pass this) -- opt-in only.
        self.allowed_hosts = allowed_hosts
        self.max_redirects = max_redirects
        #: Built only when allowed_hosts is set -- None means "use the
        #: default urllib opener", i.e. exactly the pre-3F.0.4 behavior.
        self._redirect_opener = (
            urllib.request.build_opener(_PreConnectRedirectHandler(self.allowed_hosts, self.max_redirects))
            if self.allowed_hosts is not None
            else None
        )
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

        if self.allowed_hosts is not None:
            # Phase 3F.0.4: the INITIAL host is checked before ever
            # attempting a connection too, not just a redirect target --
            # mirrors research/sec_live_smoke.py's AllowlistedHttpClient.
            initial_host = (urllib.parse.urlparse(wire_url).hostname or "").lower()
            if initial_host not in self.allowed_hosts:
                result = FetchResult(
                    url=public_url, outcome=FetchOutcome.BLOCKED,
                    error=f"refused disallowed host {initial_host!r}",
                )
                self.fetch_log.append(result)
                return result

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Encoding": "gzip",
        }
        request_headers.update(dict(headers or {}))

        # Phase 3F.0.4: max_retries is the number of retries AFTER the
        # first attempt, so max_attempts is always >= 1 -- max_retries=0
        # still makes exactly one real request (previously it made zero:
        # range(1, 0 + 1) never executed, a latent off-by-one this phase
        # fixes). See the module docstring's "Retry semantics" section.
        max_attempts = self.max_retries + 1
        open_ = self._redirect_opener.open if self._redirect_opener is not None else urllib.request.urlopen

        last_error = ""
        status: int | None = None
        outcome = FetchOutcome.ERROR
        attempts_made = 0

        for attempt in range(1, max_attempts + 1):
            attempts_made = attempt
            self.limiter.wait()
            request = urllib.request.Request(wire_url, headers=request_headers, method="GET")
            try:
                with open_(request, timeout=self.timeout) as response:
                    # Defense in depth: the redirect target was already
                    # validated BEFORE it was connected to (via
                    # _PreConnectRedirectHandler, when allowed_hosts is
                    # set) -- this re-checks the URL the response actually
                    # came from, catching anything that path might miss.
                    final_wire_url = response.geturl()
                    final_public_url = _sanitize_url(final_wire_url)
                    if self.allowed_hosts is not None:
                        final_host = (urllib.parse.urlparse(final_wire_url).hostname or "").lower()
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
            except (RedirectRejectedError, TooManyRedirectsError) as exc:
                # A redirect policy refusal -- never retried, never routed
                # around (Phase 3F.0.4 requirement 2).
                last_error = str(exc)
                outcome = FetchOutcome.BLOCKED
                break
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

            if attempt < max_attempts:
                backoff = 2.0 ** (attempt - 1)
                log.info(
                    "retrying %s in %.1fs (attempt %d/%d): %s",
                    public_url,
                    backoff,
                    attempt,
                    max_attempts,
                    last_error,
                )
                time.sleep(backoff)

        result = FetchResult(
            url=public_url,
            outcome=outcome,
            status=status,
            attempts=attempts_made,
            error=last_error,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        self.fetch_log.append(result)
        log.warning("fetch failed: %s -> %s (%s)", public_url, outcome, last_error)
        return result
