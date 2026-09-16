"""HTTP client integration tests against a real local server.

Requirement 14.  These run over real sockets rather than mocks, so the retry,
backoff, timeout and JSON-parse paths are genuinely exercised.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.collectors.http import HttpClient
from investment_research.schemas.enums import FetchOutcome

pytestmark = pytest.mark.integration


class Handler(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):  # silence the test server
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        counts = Handler.state.setdefault("counts", {})
        counts[path] = counts.get(path, 0) + 1

        if path == "/ok":
            self._send(200, json.dumps({"hello": "world"}).encode())
        elif path == "/notfound":
            self._send(404, b"missing")
        elif path == "/forbidden":
            self._send(403, b"policy denied")
        elif path == "/flaky":
            # fails twice, then succeeds
            if counts[path] < 3:
                self._send(503, b"try later")
            else:
                self._send(200, json.dumps({"attempt": counts[path]}).encode())
        elif path == "/ratelimited":
            self._send(429, b"slow down")
        elif path == "/badjson":
            Handler.state["last_raw_path"] = self.path
            self._send(200, b"{not json at all")
        elif path == "/slow":
            time.sleep(1.0)
            # the client times out on purpose, so the write may fail
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self._send(200, b"late")
        elif path == "/echo":
            # Phase 3F.0.3: records the RAW request line (query string and
            # all) this loopback server itself received -- lets a test
            # prove the secret really did go out over the wire, while
            # everything the client returns/logs/raises stays sanitized.
            Handler.state["last_raw_path"] = self.path
            self._send(200, json.dumps({"echo": self.path}).encode())
        elif path == "/redirect-to-ok":
            self._redirect("/ok")
        elif path == "/redirect-to-secret":
            # The redirect TARGET's own query reflects a secret -- proves
            # sanitization applies to final_url too, not just the
            # originally-requested url.
            self._redirect("/echo?api_key=SERVER-SIDE-SECRET")
        elif path == "/redirect-to-other-host":
            self._redirect(f"http://localhost:{self.server.server_port}/ok")
        else:
            self._send(404, b"unknown")

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


@pytest.fixture(scope="module")
def server():
    Handler.state = {}
    # Threading, so the deliberately-slow endpoint cannot block the next test's
    # request and make an unrelated assertion fail.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def client(tmp_path):
    return HttpClient(
        timeout=0.8,
        max_retries=4,
        rate_limit_rps=0,
        cache_dir=tmp_path / "cache",
    )


def test_successful_fetch_parses_json(server, client):
    result = client.get(f"{server}/ok")
    assert result.ok
    assert result.json() == {"hello": "world"}
    assert result.content_hash() != "UNKNOWN"


def test_404_is_not_found_and_is_not_retried(server, client):
    Handler.state["counts"] = {}
    result = client.get(f"{server}/notfound", use_cache=False)
    assert result.outcome is FetchOutcome.NOT_FOUND
    assert Handler.state["counts"]["/notfound"] == 1


def test_403_is_blocked_and_is_not_retried(server, client):
    """An egress-policy denial must be reported, never retried or worked around."""
    Handler.state["counts"] = {}
    result = client.get(f"{server}/forbidden", use_cache=False)
    assert result.outcome is FetchOutcome.BLOCKED
    assert Handler.state["counts"]["/forbidden"] == 1


def test_transient_5xx_is_retried_then_succeeds(server, client):
    Handler.state["counts"] = {}
    result = client.get(f"{server}/flaky", use_cache=False)
    assert result.ok
    assert result.attempts == 3
    assert Handler.state["counts"]["/flaky"] == 3


def test_rate_limit_is_reported_after_exhausting_retries(server):
    fast = HttpClient(timeout=0.8, max_retries=2, rate_limit_rps=0, cache_dir=None)
    Handler.state["counts"] = {}
    result = fast.get(f"{server}/ratelimited", use_cache=False)
    assert result.outcome is FetchOutcome.RATE_LIMITED


def test_invalid_json_returns_none_and_marks_an_error(server, client):
    result = client.get(f"{server}/badjson", use_cache=False)
    assert result.ok  # the HTTP request itself succeeded
    assert result.json() is None
    assert result.outcome is FetchOutcome.ERROR
    assert "invalid JSON" in result.error


def test_timeout_is_reported_as_a_timeout(server):
    slow = HttpClient(timeout=0.2, max_retries=1, rate_limit_rps=0, cache_dir=None)
    result = slow.get(f"{server}/slow", use_cache=False)
    assert result.outcome is FetchOutcome.TIMEOUT


def test_cache_prevents_a_second_request(server, client):
    Handler.state["counts"] = {}
    first = client.get(f"{server}/ok")
    second = client.get(f"{server}/ok")
    assert first.ok and second.ok
    assert second.from_cache
    assert Handler.state["counts"]["/ok"] == 1


def test_offline_mode_makes_no_request_and_says_so(server):
    offline = HttpClient(offline=True, cache_dir=None)
    Handler.state["counts"] = {}
    result = offline.get(f"{server}/ok")
    assert result.outcome is FetchOutcome.DISABLED
    assert "/ok" not in Handler.state.get("counts", {})


def test_rate_limiter_spaces_requests(server, tmp_path):
    limited = HttpClient(rate_limit_rps=5.0, cache_dir=None, timeout=2.0)
    started = time.monotonic()
    for _ in range(3):
        limited.get(f"{server}/ok", use_cache=False)
    assert time.monotonic() - started >= 0.35


def test_every_fetch_is_logged(server, client):
    client.fetch_log.clear()
    client.get(f"{server}/ok", use_cache=False)
    client.get(f"{server}/notfound", use_cache=False)
    assert len(client.fetch_log) == 2
    assert {r.outcome for r in client.fetch_log} == {FetchOutcome.OK, FetchOutcome.NOT_FOUND}


# --- Phase 3F.0.3: wire_url vs public_url, over a REAL loopback round trip ---
#
# Every test below sends a genuinely secret-shaped value (never a real
# credential) over the wire to the loopback server, then asserts it survives
# nowhere else: not in the returned FetchResult (any field), not in
# client.fetch_log, not in an exception message, not in repr()/str(). Each
# test that cares whether the secret really left the process first confirms
# the server's own record of the raw request line DID contain it -- proving
# sanitization happens above the transport, not by simply never sending the
# value.

_SECRET = "topsecret-value-should-never-leak"


def test_secret_query_param_never_appears_in_result_url(server, client):
    Handler.state.pop("last_raw_path", None)
    result = client.get(f"{server}/echo?api_key={_SECRET}&x=1", use_cache=False)
    assert result.ok
    assert _SECRET in Handler.state["last_raw_path"]  # the server DID receive it
    assert _SECRET not in result.url
    assert _SECRET not in result.final_url
    assert "x=1" in result.url  # a non-secret param is left alone


def test_percent_encoded_email_never_appears_in_result_url(server, client):
    Handler.state.pop("last_raw_path", None)
    encoded = "name%40example.test"
    result = client.get(f"{server}/echo?email={encoded}", use_cache=False)
    assert result.ok
    assert encoded in Handler.state["last_raw_path"]
    assert "email" not in result.url
    assert encoded not in result.url


def test_secret_never_appears_in_fetch_log(server, client):
    # /ok's own response body never echoes the request path/query, so this
    # isolates "does anything BUT the response body carry the secret" --
    # url, final_url, error, and the dataclass's own repr/str.
    client.fetch_log.clear()
    client.get(f"{server}/ok?api_key={_SECRET}", use_cache=False)
    for entry in client.fetch_log:
        assert _SECRET not in entry.url
        assert _SECRET not in entry.final_url
        assert _SECRET not in entry.error
        assert _SECRET not in repr(entry)


def test_redirect_final_url_is_tracked(server, client):
    result = client.get(f"{server}/redirect-to-ok", use_cache=False)
    assert result.ok
    assert result.final_url == f"{server}/ok"


def test_redirect_to_secret_bearing_target_is_sanitized(server, client):
    """Even when the redirect DESTINATION's own query reflects a secret
    (e.g. a server that echoes the caller's api_key back in a Location
    header), the sanitized final_url must not carry it."""
    result = client.get(f"{server}/redirect-to-secret", use_cache=False)
    assert result.ok
    assert "SERVER-SIDE-SECRET" not in result.final_url
    assert "SERVER-SIDE-SECRET" not in result.url


def test_malformed_json_error_never_contains_secret_or_full_body(server, client):
    Handler.state.pop("last_raw_path", None)
    result = client.get(f"{server}/badjson?api_key={_SECRET}", use_cache=False)
    assert _SECRET in Handler.state["last_raw_path"]
    parsed = result.json()
    assert parsed is None
    assert _SECRET not in result.error
    assert _SECRET not in result.url
    # The body is represented only by a length/hash diagnostic, never the
    # raw text -- "{not json at all" must not appear verbatim.
    assert "{not json at all" not in result.error
    assert str(len(result.body)) in result.error
    assert result.content_hash() in result.error


def test_timeout_error_never_contains_secret(server):
    slow = HttpClient(timeout=0.2, max_retries=1, rate_limit_rps=0, cache_dir=None)
    result = slow.get(f"{server}/slow?api_key={_SECRET}", use_cache=False)
    assert result.outcome is FetchOutcome.TIMEOUT
    assert _SECRET not in result.error
    assert _SECRET not in result.url


def test_connection_error_to_an_unreachable_port_never_contains_secret():
    """A closed port on loopback -- a real connection-level failure, not a
    mock -- proving urllib's own exception text (which can embed the
    request URL) is scrubbed before it reaches FetchResult.error."""
    unreachable = HttpClient(timeout=1.0, max_retries=1, rate_limit_rps=0, cache_dir=None)
    result = unreachable.get(f"http://127.0.0.1:1/path?api_key={_SECRET}", use_cache=False)
    assert result.outcome in (FetchOutcome.ERROR, FetchOutcome.TIMEOUT, FetchOutcome.BLOCKED)
    assert _SECRET not in result.error
    assert _SECRET not in result.url


def test_allowed_hosts_blocks_redirect_to_a_different_host(server):
    client = HttpClient(
        timeout=0.8, max_retries=1, rate_limit_rps=0, cache_dir=None,
        allowed_hosts=frozenset({"127.0.0.1"}),
    )
    result = client.get(f"{server}/redirect-to-other-host", use_cache=False)
    assert result.outcome is FetchOutcome.BLOCKED
    assert "not allowlisted" in result.error


def test_allowed_hosts_none_by_default_does_not_change_existing_behavior(server, client):
    """The opt-in allowed_hosts param defaults to None -- SEC/ClinicalTrials/
    Form4 callers, who never pass it, keep following a same-host redirect
    exactly as before this hardening (Phase 3F.0.3 requirement: no behavior
    change for existing callers)."""
    assert client.allowed_hosts is None
    result = client.get(f"{server}/redirect-to-ok", use_cache=False)
    assert result.ok
