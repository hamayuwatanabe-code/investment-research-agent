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

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

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
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


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
