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

import hashlib
import json
import threading
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


# --- Phase 3D.4: Capture Manifest, written against a REAL HTTP round trip ---
def test_capture_manifest_is_written_with_expected_fields(base_url, tmp_path):
    client = _client(out_dir=tmp_path, source="sec")
    url = f"{base_url}/ok"
    result = client.get(url)
    assert result.ok

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    manifest = read_manifest(tmp_path, digest)
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
    manifest = read_manifest(tmp_path, digest)
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
    client.get(url)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    manifest = read_manifest(tmp_path, digest)
    assert manifest is not None
    assert manifest.source == "clinicaltrials"


def test_capture_manifest_read_manifest_is_none_for_a_pre_manifest_capture(base_url, tmp_path):
    """A body saved with NO manifest (e.g. from before Phase 3D.4, or any
    tool that only ever wrote the raw body) must read back as None, not
    raise -- backward compatibility (Phase 3D.4 requirement 11)."""
    url = f"{base_url}/ok"
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text('{"ok": true}', encoding="utf-8")
    assert read_manifest(tmp_path, digest) is None
