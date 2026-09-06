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
            self._send(200, b"{not json at all")
        elif path == "/slow":
            time.sleep(1.0)
            # the client times out on purpose, so the write may fail
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self._send(200, b"late")
        else:
            self._send(404, b"unknown")

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
