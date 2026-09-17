"""Phase 3C: ``AllowlistedHttpClient`` exercised against a local loopback
server -- never the real internet, but real ``urllib`` request/redirect
mechanics rather than a hand-rolled fake.

Mirrors ``tests/integration/test_collectors.py``'s established pattern
(``ThreadingHTTPServer`` on ``127.0.0.1``, module-scoped fixture). The
offline, no-transport-at-all tests (env var gating, orchestration against
``FakeHttpClient``) live in ``tests/unit/test_sec_live_smoke_offline.py``;
this file is specifically about the host-allowlist and request-cap
enforcement that only a real HTTP round trip (with real redirects) can
prove.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.research.capture_manifest import read_manifest
from investment_research.research.sec_live_smoke import (
    AllowlistedHttpClient,
    HostAllowlistError,
    MaxRequestsExceededError,
)
from investment_research.schemas.enums import FetchOutcome

pytestmark = pytest.mark.integration

_AGENT = "investment-research-agent-test smoke-test-contact@example.test"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/ok" or path.startswith("/ok/"):
            body = json.dumps({"ok": True, "path": path}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/redirect-to-disallowed-host":
            # Same server, but addressed by a DIFFERENT hostname string
            # ("localhost" vs "127.0.0.1") -- both resolve to loopback, but
            # the allowlist check is a hostname string match, never a DNS
            # resolution, so this must be refused regardless.
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_port}/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/redirect-to-allowed-host":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/echo":
            # Phase 3F.0.3: records the RAW request line this loopback
            # server itself received, so a test can prove a secret query
            # param really went out over the wire while everything above
            # this layer (FetchResult, the Capture Manifest) stays sanitized.
            Handler.last_raw_path = self.path
            body = json.dumps({"echo": self.path}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/ratelimited":
            self.send_response(429)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/flaky-error":
            # Fails with a retryable 500 twice, then succeeds -- keyed on
            # the FULL request line (query string included), so different
            # tests can get independent counters via a distinct query.
            counts = Handler.state.setdefault("flaky_counts", {})
            counts[self.path] = counts.get(self.path, 0) + 1
            if counts[self.path] < 3:
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        elif path == "/slow":
            time.sleep(1.0)
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                body = b'{"late": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        elif path == "/always-500":
            counts = Handler.state.setdefault("always_500_counts", {})
            counts[self.path] = counts.get(self.path, 0) + 1
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    state: dict = {}
    last_raw_path: str = ""


class AllowedHandler(BaseHTTPRequestHandler):
    """A SEPARATE server from Handler -- Phase 3F.0.4's two-loopback-server
    redirect tests need an independently-countable "disallowed" server, not
    just a different hostname string pointing at the same socket."""

    disallowed_port: int = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/ok":
            self._send(200, b'{"ok": true}')
        elif path == "/redirect-to-disallowed-server":
            self._redirect(
                f"http://localhost:{AllowedHandler.disallowed_port}/ok"
                "?api_key=SHOULD-NOT-REACH-DISALLOWED"
            )
        elif path == "/redirect-loop-a":
            self._redirect("/redirect-loop-b")
        elif path == "/redirect-loop-b":
            self._redirect("/redirect-loop-a")
        elif path == "/redirect-two-hop":
            self._redirect("/redirect-to-ok")
        elif path == "/redirect-to-ok":
            self._redirect("/ok")
        else:
            self._send(404, b"")

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DisallowedHandler(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        DisallowedHandler.state["hits"] = DisallowedHandler.state.get("hits", 0) + 1
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def two_servers():
    DisallowedHandler.state = {}
    disallowed = ThreadingHTTPServer(("127.0.0.1", 0), DisallowedHandler)
    threading.Thread(target=disallowed.serve_forever, daemon=True).start()

    AllowedHandler.disallowed_port = disallowed.server_port
    allowed = ThreadingHTTPServer(("127.0.0.1", 0), AllowedHandler)
    threading.Thread(target=allowed.serve_forever, daemon=True).start()

    yield f"http://127.0.0.1:{allowed.server_port}"
    allowed.shutdown()
    disallowed.shutdown()


@pytest.fixture(scope="module")
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _client(**overrides):
    kwargs = {
        "user_agent": _AGENT,
        "allowed_hosts": frozenset({"127.0.0.1"}),
        "timeout": 2.0,
        "rate_limit_rps": 1000.0,
        "max_retries": 0,
        "max_requests": 6,
    }
    kwargs.update(overrides)
    return AllowlistedHttpClient(**kwargs)


# --- normal fetch, for a control ---------------------------------------------
def test_allowed_host_normal_fetch_succeeds(base_url):
    client = _client()
    result = client.get(f"{base_url}/ok")
    assert result.ok
    assert result.outcome is FetchOutcome.OK
    assert result.json() == {"ok": True, "path": "/ok"}
    assert client.requests_made == 1


# --- on_first_attempt callback (Phase 3F.1 correction 2 requirement 1) ------
# Generic to AllowlistedHttpClient -- exercised here independent of any
# specific caller (SEC/ClinicalTrials/Form4/Literature all reuse this same
# class); Literature Live Smoke's own marker-timing tests live in
# tests/integration/test_literature_live_smoke_transport.py.
def test_on_first_attempt_default_none_preserves_existing_behavior(base_url):
    """A client constructed with no on_first_attempt at all (every existing
    SEC/ClinicalTrials/Form4 caller) behaves exactly as before -- nothing
    is called, nothing changes."""
    client = _client()
    result = client.get(f"{base_url}/ok")
    assert result.ok
    assert client.attempts_made == 1


def test_on_first_attempt_fires_exactly_once_before_the_first_physical_attempt(base_url):
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"))
    result = client.get(f"{base_url}/ok")
    assert result.ok
    assert events == ["callback"]


def test_on_first_attempt_never_fires_when_no_attempt_is_made(base_url):
    """A disallowed-host refusal never reaches the physical-attempt code
    path at all -- on_first_attempt must not fire for it."""
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"), allowed_hosts=frozenset({"only-this-host"}))
    result = client.get(f"{base_url}/ok")
    assert result.outcome is FetchOutcome.BLOCKED
    assert events == []


def test_on_first_attempt_fires_only_once_across_retries_of_the_same_call(base_url):
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"), max_retries=2)
    result = client.get(f"{base_url}/flaky-error?case=on-first-attempt-retries")
    assert result.ok
    assert events == ["callback"]
    assert client.attempts_made == 3  # two failed attempts + the succeeding retry


def test_on_first_attempt_fires_only_once_across_multiple_logical_calls(base_url):
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"))
    client.get(f"{base_url}/ok")
    client.get(f"{base_url}/ok/again")
    assert events == ["callback"]


def test_on_first_attempt_still_fires_when_the_first_attempt_then_times_out(base_url):
    """"Sent, then timed out" must still count as a real attempt having
    started -- the callback fires before the outcome (here, a timeout) is
    known."""
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"), timeout=0.2)
    result = client.get(f"{base_url}/slow")
    assert result.outcome is FetchOutcome.TIMEOUT
    assert events == ["callback"]


def test_on_first_attempt_still_fires_when_the_first_attempt_then_429s(base_url):
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"))
    result = client.get(f"{base_url}/ratelimited")
    assert result.outcome is FetchOutcome.RATE_LIMITED
    assert events == ["callback"]


def test_on_first_attempt_raising_makes_zero_physical_attempts(base_url):
    """If the callback raises, no physical attempt is made at all -- this
    is the property main()'s "marker write failure must not begin
    communication" contract relies on."""
    def _boom() -> None:
        raise OSError("simulated failure")

    client = _client(on_first_attempt=_boom)
    with pytest.raises(OSError):
        client.get(f"{base_url}/ok")
    assert client.attempts_made == 0
    assert client.requests_made == 0


# --- host allowlist -----------------------------------------------------------
def test_disallowed_host_is_refused_without_a_request(base_url):
    """A URL whose host isn't in ``allowed_hosts`` never reaches urllib at
    all -- refused synchronously, and does not count against the request
    cap (Phase 3C requirement 10)."""
    client = _client()
    result = client.get(f"http://localhost:{base_url.rsplit(':', 1)[1]}/ok")
    assert not result.ok
    assert result.outcome is FetchOutcome.BLOCKED
    assert client.requests_made == 0


def test_redirect_to_allowed_host_is_followed(base_url):
    client = _client()
    result = client.get(f"{base_url}/redirect-to-allowed-host")
    assert result.ok
    assert result.json() == {"ok": True, "path": "/ok"}


def test_redirect_to_disallowed_host_is_refused(base_url):
    """The redirect target is re-validated against the allowlist BEFORE it
    is followed (Phase 3C requirement 11) -- proven here with a real 302
    from a real server, not merely a unit-level check of the handler
    class."""
    client = _client()
    result = client.get(f"{base_url}/redirect-to-disallowed-host")
    assert not result.ok
    assert result.outcome is FetchOutcome.BLOCKED
    assert "disallowed host" in result.error


def test_redirect_handler_raises_host_allowlist_error_directly():
    from investment_research.research.sec_live_smoke import _AllowlistedRedirectHandler

    handler = _AllowlistedRedirectHandler(frozenset({"127.0.0.1"}))
    with pytest.raises(HostAllowlistError):
        handler.redirect_request(
            req=None, fp=None, code=302, msg="Found", headers={}, newurl="http://localhost:9/ok"
        )


# --- request cap ---------------------------------------------------------------
def test_max_requests_cap_is_enforced(base_url):
    client = _client(max_requests=2)
    client.get(f"{base_url}/ok?a=1")
    client.get(f"{base_url}/ok?a=2")
    with pytest.raises(MaxRequestsExceededError):
        client.get(f"{base_url}/ok?a=3")


def test_cache_hits_do_not_count_against_the_request_cap(base_url):
    """Requesting the SAME url twice is a cache hit, not a second real GET
    -- so it must never itself trip the request cap (Phase 3C requirement
    16: caching is enabled precisely so a re-read of an already-fetched
    resource is free)."""
    client = _client(max_requests=1)
    first = client.get(f"{base_url}/ok")
    second = client.get(f"{base_url}/ok")
    assert first.ok and second.ok
    assert second.from_cache
    assert client.requests_made == 1
    assert client.cache_hits == 1


# --- Phase 3D.4: Capture Manifest, written against a REAL HTTP round trip ---
def test_capture_manifest_is_written_with_expected_fields(base_url, tmp_path):
    client = _client(out_dir=tmp_path, source="sec")
    url = f"{base_url}/ok"
    result = client.get(url)
    assert result.ok

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    read_result = read_manifest(tmp_path, digest, body=result.body)
    assert read_result.status.value == "VERIFIED"
    manifest = read_result.manifest
    assert manifest is not None
    assert manifest.schema_version >= 1
    assert manifest.source == "sec"
    assert manifest.requested_url == url
    assert manifest.final_url == url  # no redirect on this path
    assert manifest.http_status == 200
    assert manifest.content_hash == hashlib.sha256(result.body).hexdigest()
    assert manifest.content_length == len(result.body)
    # A real UTC timestamp, not a placeholder -- format-checked, never
    # compared to a hardcoded value (the test has no control over wall time).
    assert manifest.capture_retrieved_at.endswith("Z")
    assert len(manifest.capture_retrieved_at) == len("2026-09-14T00:00:00Z")


def test_capture_manifest_final_url_reflects_the_redirect_target(base_url, tmp_path):
    client = _client(out_dir=tmp_path, source="sec")
    url = f"{base_url}/redirect-to-allowed-host"
    result = client.get(url)
    assert result.ok

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    read_result = read_manifest(tmp_path, digest, body=result.body)
    manifest = read_result.manifest
    assert manifest is not None
    assert manifest.requested_url == url
    assert manifest.final_url == f"{base_url}/ok"
    assert manifest.final_url != manifest.requested_url


def test_capture_manifest_never_contains_the_user_agent_or_other_secrets(base_url, tmp_path):
    secret_agent = "investment-research-agent-test secret-contact@example.test"
    client = _client(out_dir=tmp_path, source="sec", user_agent=secret_agent)
    url = f"{base_url}/ok"
    client.get(url)

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    manifest_path = tmp_path / f"{digest}.manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert secret_agent not in manifest_text
    assert "user_agent" not in manifest_text
    assert "user-agent" not in manifest_text.lower()
    assert "header" not in manifest_text.lower()
    assert "api_key" not in manifest_text.lower()
    assert "email" not in manifest_text.lower()


def test_capture_manifest_write_leaves_no_temp_file_behind(base_url, tmp_path):
    """Atomic write via a temp file + os.replace -- confirms the temp file
    used mid-write is never left lying around after a successful write
    (Phase 3D.4 requirement 10)."""
    client = _client(out_dir=tmp_path, source="sec")
    client.get(f"{base_url}/ok")
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_capture_manifest_source_field_reflects_the_caller_e_g_clinicaltrials(base_url, tmp_path):
    """The SAME AllowlistedHttpClient class and _save_response path is used
    by BOTH sec_live_smoke.py and clinicaltrials_live_smoke.py -- source is
    caller-supplied, never inferred from the URL or hardcoded to "sec"
    (Phase 3D.4 requirement 4: one manifest type, one save rule, for both)."""
    client = _client(out_dir=tmp_path, source="clinicaltrials")
    url = f"{base_url}/ok"
    result = client.get(url)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    read_result = read_manifest(tmp_path, digest, body=result.body)
    assert read_result.manifest is not None
    assert read_result.manifest.source == "clinicaltrials"


def test_capture_manifest_never_contains_a_secret_query_param(base_url, tmp_path):
    """Phase 3F.0.3, digest source corrected in the Phase 3F.1 correction
    pass: even when the requested URL itself carries a secret-shaped query
    parameter (as a future Literature Live Smoke tool reusing this client
    might pass), the Capture Manifest written to disk -- and the returned
    FetchResult -- must never contain it. First confirms the loopback
    server itself DID receive the secret (proving this is sanitization,
    not simply never sending the value). The capture filename's own digest
    is derived from the PUBLIC (sanitized) url, not the wire url, so it is
    computed here from result.url -- never a hash of the secret-bearing
    request url itself."""
    secret = "topsecret-manifest-value"
    client = _client(out_dir=tmp_path, source="sec")
    url = f"{base_url}/echo?api_key={secret}&x=1"
    result = client.get(url)
    assert result.ok
    assert secret in Handler.last_raw_path  # the server DID receive it

    assert secret not in result.url
    assert secret not in result.final_url

    digest = hashlib.sha256(result.url.encode()).hexdigest()[:24]
    manifest_path = tmp_path / f"{digest}.manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert secret not in manifest_text

    read_result = read_manifest(tmp_path, digest, body=result.body)
    manifest = read_result.manifest
    assert manifest is not None
    assert secret not in manifest.requested_url
    assert secret not in manifest.final_url
    assert "x=1" in manifest.requested_url


def test_capture_filename_digest_is_derived_from_the_public_url_not_the_wire_url(base_url, tmp_path):
    """Two requests to the SAME logical (public) resource, differing only
    in secret query parameter VALUES, must produce the SAME capture
    filename/digest -- never two spurious duplicate captures keyed off the
    secret (Phase 3F.1 correction requirement 1)."""
    client_a = _client(out_dir=tmp_path, source="sec")
    url_a = f"{base_url}/echo?api_key=secret-value-one&x=1"
    result_a = client_a.get(url_a)
    assert result_a.ok

    client_b = _client(out_dir=tmp_path, source="sec")
    url_b = f"{base_url}/echo?api_key=secret-value-two&x=1"
    result_b = client_b.get(url_b)
    assert result_b.ok

    assert result_a.url == result_b.url  # same public url
    digest_from_public_url = hashlib.sha256(result_a.url.encode()).hexdigest()[:24]
    assert (tmp_path / f"{digest_from_public_url}.manifest.json").is_file()
    # Exactly one capture on disk for this logical resource -- not two.
    assert len(list(tmp_path.glob("*.manifest.json"))) == 1


# --- Phase 3F.0.4: retry semantics -- max_retries is retries AFTER the
# first attempt, audited to already match this (unlike HttpClient before
# this phase's fix) -- these tests prove the counts, not change behavior. --

def test_allowlisted_max_retries_zero_makes_exactly_one_attempt_on_success(base_url):
    client = _client(max_retries=0)
    result = client.get(f"{base_url}/ok")
    assert result.ok
    assert result.attempts == 1


def test_allowlisted_max_retries_zero_makes_exactly_one_attempt_on_404(base_url):
    client = _client(max_retries=0)
    result = client.get(f"{base_url}/does-not-exist")
    assert result.outcome is FetchOutcome.NOT_FOUND
    assert result.attempts == 1


def test_allowlisted_429_is_never_retried_regardless_of_max_retries(base_url):
    """Documented, audited, and deliberately UNCHANGED asymmetry with
    HttpClient (which DOES retry a 429): AllowlistedHttpClient's
    _get_with_retry only retries an ERROR/TIMEOUT outcome -- RATE_LIMITED
    returns immediately, at any max_retries. The COUNT semantics
    (max_retries=N -> at most N+1 attempts) already matched before this
    phase; this asymmetry is about WHICH outcomes are retryable, which
    Phase 3F.0.4 does not unify (see the final report)."""
    client = _client(max_retries=3)
    result = client.get(f"{base_url}/ratelimited")
    assert result.outcome is FetchOutcome.RATE_LIMITED
    assert result.attempts == 1


def test_allowlisted_max_retries_zero_makes_exactly_one_attempt_on_5xx(base_url):
    client = _client(max_retries=0)
    result = client.get(f"{base_url}/flaky-error?t=zero-retry")
    assert result.outcome is FetchOutcome.ERROR
    assert result.attempts == 1


def test_allowlisted_max_retries_two_succeeds_within_budget_on_5xx(base_url):
    """/flaky-error fails twice then succeeds -- exactly matches a
    max_retries=2 budget (1 initial + 2 retries = 3 attempts)."""
    client = _client(max_retries=2)
    result = client.get(f"{base_url}/flaky-error?t=two-retry")
    assert result.ok
    assert result.attempts == 3


def test_allowlisted_max_retries_zero_makes_exactly_one_attempt_on_timeout(base_url):
    client = _client(max_retries=0, timeout=0.2)
    result = client.get(f"{base_url}/slow")
    assert result.outcome is FetchOutcome.TIMEOUT
    assert result.attempts == 1


def test_allowlisted_max_retries_one_timeout_makes_two_attempts(base_url):
    client = _client(max_retries=1, timeout=0.2)
    result = client.get(f"{base_url}/slow")
    assert result.outcome is FetchOutcome.TIMEOUT
    assert result.attempts == 2


def test_allowlisted_max_retries_zero_makes_exactly_one_attempt_on_connection_error():
    client = _client(max_retries=0, timeout=0.5)
    result = client.get("http://127.0.0.1:1/x")
    assert result.attempts == 1


# --- Phase 3F.0.4: redirect allowlist enforced BEFORE connecting, proven
# against two INDEPENDENT loopback servers ----------------------------------

def test_redirect_to_a_second_disallowed_server_makes_zero_requests_to_it(two_servers):
    client = _client()
    result = client.get(f"{two_servers}/redirect-to-disallowed-server")
    assert result.outcome is FetchOutcome.BLOCKED
    assert DisallowedHandler.state.get("hits", 0) == 0
    assert "SHOULD-NOT-REACH-DISALLOWED" not in result.error
    assert "SHOULD-NOT-REACH-DISALLOWED" not in result.url


def test_redirect_within_the_allowlisted_host_still_succeeds(two_servers):
    client = _client()
    result = client.get(f"{two_servers}/redirect-to-ok")
    assert result.ok


def test_multi_hop_redirect_within_the_allowlist_succeeds(two_servers):
    client = _client(max_redirects=5)
    result = client.get(f"{two_servers}/redirect-two-hop")
    assert result.ok


def test_redirect_loop_is_rejected_as_blocked_once_the_cap_is_exceeded(two_servers):
    client = _client(max_redirects=4)
    result = client.get(f"{two_servers}/redirect-loop-a")
    assert result.outcome is FetchOutcome.BLOCKED
    assert "redirect count exceeded" in result.error


def test_capture_manifest_read_manifest_missing_for_a_pre_manifest_capture(base_url, tmp_path):
    """A body saved with NO manifest (e.g. from before Phase 3D.4, or any
    tool that only ever wrote the raw body) must read back as MISSING, not
    raise and never crash -- backward compatibility (Phase 3D.4 requirement
    11 / Phase 3D.4.1 requirement 6: MISSING, not MALFORMED or anything
    else)."""
    url = f"{base_url}/ok"
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text('{"ok": true}', encoding="utf-8")
    result = read_manifest(tmp_path, digest, body=b'{"ok": true}')
    assert result.status.value == "MISSING"
    assert result.manifest is None


# --- Phase 3F.1 correction requirement 1: cache/dedup keyed by public URL --
def test_cache_is_stable_across_different_secret_values_for_the_same_public_url(base_url, tmp_path):
    """Two requests to the SAME logical resource, differing only in a
    secret query parameter's VALUE, must be recognized as the same entry
    for caching/dedup -- the second is a cache hit, not a second real GET,
    and only ONE capture is ever written to disk."""
    client = _client(out_dir=tmp_path, max_requests=1)
    first = client.get(f"{base_url}/echo?api_key=secret-value-one&x=1")
    second = client.get(f"{base_url}/echo?api_key=secret-value-two&x=1")
    assert first.ok and second.ok
    assert second.from_cache
    assert client.requests_made == 1
    assert client.attempts_made == 1
    assert client.cache_hits == 1
    assert len(list(tmp_path.glob("*.manifest.json"))) == 1


def test_final_urls_cache_is_also_keyed_by_public_url(base_url, tmp_path):
    """The internal final_url tracking used for the Capture Manifest is
    keyed by the PUBLIC url (Phase 3F.1 correction requirement 1) -- proven
    via the manifest actually written for a secret-bearing initial request
    that redirects, since FetchResult itself carries no final_url field for
    AllowlistedHttpClient (that field is collectors.http.HttpClient's own;
    this class's Capture Manifest is the only place a redirect's final_url
    is exposed)."""
    client = _client(out_dir=tmp_path)
    result = client.get(f"{base_url}/redirect-to-allowed-host?api_key=secret-value")
    assert result.ok
    assert "secret-value" not in result.url

    digest = hashlib.sha256(result.url.encode()).hexdigest()[:24]
    read_result = read_manifest(tmp_path, digest, body=result.body)
    manifest = read_result.manifest
    assert manifest is not None
    assert "secret-value" not in manifest.final_url
    assert manifest.final_url == f"{base_url}/ok"


# --- Phase 3F.1 correction requirement 3: request cap applies to physical
# GET attempts, including retries, not just logical URLs -------------------
def test_attempts_made_and_requests_made_are_distinct_fields():
    client = _client()
    assert client.attempts_made == 0
    assert client.requests_made == 0


def test_max_requests_caps_physical_attempts_across_retries_within_one_call(base_url):
    """A single logical .get() call that retries 3 times (max_retries=2,
    so up to 3 physical attempts) against an ALWAYS-500 endpoint must never
    make MORE than max_requests real attempts, even though it is only ONE
    logical request."""
    client = _client(max_retries=2, max_requests=2)
    result = client.get(f"{base_url}/always-500?t=cap-within-call")
    assert not result.ok
    assert client.attempts_made == 2  # capped at max_requests, never reached max_retries+1=3
    assert client.requests_made == 1  # exactly one logical call was made


def test_max_requests_caps_physical_attempts_across_multiple_logical_calls(base_url):
    """The physical attempt budget is a SHARED, run-wide total: once a
    FIRST logical call has already exhausted it entirely (here, exactly
    max_requests attempts), a second logical .get() call must be refused
    OUTRIGHT (raising, before making any attempt of its own) rather than
    being allowed to sneak in further attempts."""
    client = _client(max_retries=1, max_requests=2)
    first = client.get(f"{base_url}/always-500?t=multi-call-a")
    assert client.attempts_made == 2  # one logical call, 2 attempts (1 + 1 retry) -- budget now exhausted
    with pytest.raises(MaxRequestsExceededError):
        client.get(f"{base_url}/always-500?t=multi-call-b")
    assert client.attempts_made == 2  # the second call never got to attempt at all
    assert not first.ok


def test_a_second_call_may_still_use_a_single_remaining_slot_then_stops(base_url):
    """When exactly ONE physical slot remains, a new logical call IS
    allowed to use it (never refused outright while capacity remains), but
    is not allowed to retry further once that slot is spent -- the
    mid-retry cap check stops it silently (no raise), distinct from the
    pre-flight refusal above which fires only when NO capacity remains at
    the start of a new logical call."""
    client = _client(max_retries=1, max_requests=3)
    client.get(f"{base_url}/always-500?t=slot-a")  # consumes 2 of 3
    assert client.attempts_made == 2
    second = client.get(f"{base_url}/always-500?t=slot-b")  # gets the last slot, then stops
    assert client.attempts_made == 3
    assert not second.ok
    assert client.requests_made == 2


def test_timeout_retries_never_exceed_the_physical_cap(base_url):
    client = _client(timeout=0.3, max_retries=3, max_requests=1)
    result = client.get(f"{base_url}/slow")
    assert result.outcome is FetchOutcome.TIMEOUT
    assert client.attempts_made == 1  # capped at 1, never reached max_retries+1=4


def test_connection_error_retries_never_exceed_the_physical_cap():
    """A closed port on loopback -- a real connection-level failure --
    proving the physical cap holds for OSError/URLError retries too, not
    just HTTP-level failures."""
    client = _client(timeout=1.0, max_retries=5, max_requests=2)
    result = client.get("http://127.0.0.1:1/unreachable")
    assert not result.ok
    assert client.attempts_made == 2  # capped at 2, never reached max_retries+1=6


def test_429_is_never_retried_and_counts_as_exactly_one_physical_attempt(base_url):
    client = _client(max_retries=3, max_requests=6)
    result = client.get(f"{base_url}/ratelimited")
    assert result.outcome is FetchOutcome.RATE_LIMITED
    assert client.attempts_made == 1
    assert client.requests_made == 1


def test_5xx_retry_then_success_reports_correct_physical_attempts(base_url):
    client = _client(max_retries=2, max_requests=6)
    result = client.get(f"{base_url}/flaky-error?t=success-within-budget")
    assert result.ok
    assert result.attempts == 3
    assert client.attempts_made == 3
    assert client.requests_made == 1
