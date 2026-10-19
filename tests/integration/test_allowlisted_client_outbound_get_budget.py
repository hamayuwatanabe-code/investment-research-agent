"""Phase 4.3K: ``AllowlistedHttpClient.max_requests`` caps ACTUAL outbound
GETs -- initial GETs, retries, and followed redirect hops all count.

Before this phase only ``_do_get`` entries were counted, so redirect hops
(up to ``max_redirects`` per attempt) were sent outside the budget: a
loopback check showed 6 server hits under ``max_requests=1``.

Loopback only (127.0.0.1 / localhost servers started here); no external
network. Every assertion on "what was sent" is the SERVER's own hit log,
never the client's opinion of itself.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise

import pytest

from investment_research.collectors import http as http_module
from investment_research.collectors.http import RateLimiter
from investment_research.research import sec_live_smoke
from investment_research.research.sec_live_smoke import (
    AllowlistedHttpClient,
    MaxRequestsExceededError,
)
from investment_research.schemas.enums import FetchOutcome

pytestmark = pytest.mark.integration

_AGENT = "phase-4-3k-budget-test-agent/1.0"
_SECRET_EMAIL = "budget-secret-contact@example.test"
_SECRET_API_KEY = "budgetSecretApiKey0123456789"


class _Handler(BaseHTTPRequestHandler):
    hits: list[str] = []
    #: time.monotonic() at each GET's arrival, same clock as the client
    arrivals: list[float] = []
    #: path -> redirect target path (same server)
    chain: dict[str, str] = {}
    #: path -> HTTP status to fail with (no redirect)
    fail: dict[str, int] = {}
    disallowed_port: int = 0
    #: seconds the FIRST connection is held up before its request is read
    #: (simulates cold-start latency: first socket/getaddrinfo/thread spawn)
    first_connection_delay: float = 0.0

    def setup(self):
        if type(self).first_connection_delay and not type(self).hits:
            time.sleep(type(self).first_connection_delay)
        super().setup()

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        type(self).arrivals.append(time.monotonic())
        type(self).hits.append(path)
        if path == "/to-disallowed":
            self._redirect(
                f"http://localhost:{type(self).disallowed_port}/ok"
                f"?email={_SECRET_EMAIL}&api_key={_SECRET_API_KEY}"
            )
        elif path in type(self).chain:
            self._redirect(type(self).chain[path])
        elif path in type(self).fail:
            self._send(type(self).fail[path], b"")
        else:
            self._send(200, b'{"ok": true}')

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _DisallowedHandler(BaseHTTPRequestHandler):
    hits: list[str] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).hits.append(self.path)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


@pytest.fixture
def base_url():
    _Handler.hits = []
    _Handler.arrivals = []
    _Handler.chain = {}
    _Handler.fail = {}
    _Handler.first_connection_delay = 0.0
    _DisallowedHandler.hits = []
    disallowed = ThreadingHTTPServer(("127.0.0.1", 0), _DisallowedHandler)
    threading.Thread(target=disallowed.serve_forever, daemon=True).start()
    _Handler.disallowed_port = disallowed.server_port
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    disallowed.shutdown()


def _client(**overrides) -> AllowlistedHttpClient:
    kwargs = {
        "user_agent": _AGENT, "allowed_hosts": frozenset({"127.0.0.1"}), "timeout": 2.0,
        "rate_limit_rps": 1000.0, "max_retries": 0, "max_requests": 6, "max_redirects": 5,
        "source": "literature",
    }
    kwargs.update(overrides)
    return AllowlistedHttpClient(**kwargs)


def test_budget_of_one_never_sends_the_redirect_target(base_url):
    _Handler.chain = {"/a": "/ok"}
    client = _client(max_requests=1)

    result = client.get(f"{base_url}/a")

    assert _Handler.hits == ["/a"]  # the redirect target was never requested
    assert result.outcome is FetchOutcome.ERROR and not result.ok
    assert "redirect target was not requested" in result.error
    assert client.attempts_made == 1
    assert client.requests_made == 1
    with pytest.raises(MaxRequestsExceededError):
        client.get(f"{base_url}/other")
    assert _Handler.hits == ["/a"]


def test_followed_redirect_hops_are_counted_as_actual_gets(base_url):
    """attempts_made = actual GETs (initial + each followed hop);
    requests_made = logical .get() calls."""
    _Handler.chain = {"/a": "/b", "/b": "/c", "/c": "/ok"}
    client = _client(max_requests=6)

    result = client.get(f"{base_url}/a")

    assert result.ok
    assert _Handler.hits == ["/a", "/b", "/c", "/ok"]
    assert client.attempts_made == 4 == len(_Handler.hits)
    assert client.requests_made == 1


def test_retries_and_redirects_together_never_exceed_six_actual_gets(base_url):
    """Each attempt walks a 3-GET redirect chain ending in a 500, and is
    retried: without the hop-aware budget this sends 3 x 3 = 9 GETs."""
    _Handler.chain = {"/r1": "/r2", "/r2": "/fail"}
    _Handler.fail = {"/fail": 500}
    client = _client(max_requests=6, max_retries=2)

    result = client.get(f"{base_url}/r1")

    assert not result.ok
    assert len(_Handler.hits) == 6
    assert client.attempts_made == 6
    assert _Handler.hits == ["/r1", "/r2", "/fail", "/r1", "/r2", "/fail"]


def test_many_logical_requests_with_redirects_stop_at_exactly_six_gets(base_url):
    _Handler.chain = {f"/p{i}": f"/q{i}" for i in range(10)}
    client = _client(max_requests=6)

    outcomes = []
    for i in range(10):
        try:
            outcomes.append(client.get(f"{base_url}/p{i}").outcome)
        except MaxRequestsExceededError:
            break

    assert len(_Handler.hits) == 6
    assert client.attempts_made == 6
    assert outcomes == [FetchOutcome.OK] * 3
    assert client.requests_made == 3


def test_budget_exhausted_mid_chain_ends_without_a_seventh_get(base_url):
    """Two full 2-GET fetches, then a third whose redirect would be the
    7th GET: it stops after its first GET, incomplete."""
    _Handler.chain = {"/x": "/ok", "/y": "/ok2", "/z1": "/z2", "/z2": "/ok3"}
    client = _client(max_requests=5, max_retries=3)

    assert client.get(f"{base_url}/x").ok
    assert client.get(f"{base_url}/y").ok
    third = client.get(f"{base_url}/z1")

    assert third.outcome is FetchOutcome.ERROR
    assert _Handler.hits == ["/x", "/ok", "/y", "/ok2", "/z1"]
    assert client.attempts_made == 5


def test_redirect_to_a_disallowed_host_is_refused_before_connecting(base_url):
    client = _client(max_requests=6)

    result = client.get(f"{base_url}/to-disallowed")

    assert result.outcome is FetchOutcome.BLOCKED
    assert _DisallowedHandler.hits == []  # never connected
    assert _Handler.hits == ["/to-disallowed"]
    assert client.attempts_made == 1  # a refused target consumes no budget
    for leaked in (_SECRET_EMAIL, _SECRET_API_KEY):
        assert leaked not in result.error
        assert leaked not in result.url
        assert leaked not in repr(result)


def test_redirect_hop_cap_still_applies_within_the_budget(base_url):
    _Handler.chain = {f"/h{i}": f"/h{i + 1}" for i in range(10)}
    client = _client(max_requests=20, max_redirects=3)

    result = client.get(f"{base_url}/h0")

    assert result.outcome is FetchOutcome.BLOCKED
    assert "redirect count exceeded" in result.error
    assert _Handler.hits == ["/h0", "/h1", "/h2", "/h3"]
    assert client.attempts_made == 4


def test_on_first_attempt_fires_once_across_retries_and_redirects(base_url):
    _Handler.chain = {"/r1": "/r2", "/r2": "/fail"}
    _Handler.fail = {"/fail": 503}
    calls: list[int] = []
    client = _client(max_requests=6, max_retries=1, on_first_attempt=lambda: calls.append(1))

    client.get(f"{base_url}/r1")

    assert calls == [1]
    assert client.attempts_made == 6


def test_on_first_attempt_failure_sends_nothing_and_consumes_no_budget(base_url):
    def boom():
        raise RuntimeError("marker write failed")

    client = _client(on_first_attempt=boom)
    with pytest.raises(RuntimeError):
        client.get(f"{base_url}/ok")
    assert _Handler.hits == []
    assert client.attempts_made == 0


def test_secret_query_values_are_stripped_from_results_across_a_redirect(base_url):
    _Handler.chain = {"/s": "/ok"}
    client = _client(max_requests=6, additional_secret_query_params=frozenset({"tool"}))

    result = client.get(
        f"{base_url}/s?db=pubmed&email={_SECRET_EMAIL}&api_key={_SECRET_API_KEY}&tool=budget-tool-x"
    )

    assert result.ok
    for leaked in (_SECRET_EMAIL, _SECRET_API_KEY, "budget-tool-x", "email=", "api_key=", "tool="):
        assert leaked not in result.url
        assert leaked not in result.final_url
        assert all(leaked not in u for u in client.requested_urls)


def test_non_http_scheme_on_an_allowlisted_host_is_refused_before_any_connection(base_url):
    """Only the budgeted HTTP/HTTPS handlers count GETs, so any other scheme
    must never reach the opener at all."""
    client = _client()
    result = client.get("ftp://127.0.0.1/file.txt")
    assert result.outcome is FetchOutcome.BLOCKED
    assert "unsupported scheme" in result.error
    assert client.attempts_made == 0
    assert client.requests_made == 0



# =============================================================================
# Rate limiting applies to EVERY GET actually sent, redirect hops included.
#
# What is measured matters. ``RateLimiter.wait()`` guarantees spacing between
# the instants the CLIENT releases consecutive GETs (``_last`` is stamped on
# the same monotonic clock after the sleep). The time a GET ARRIVES at the
# server is that release plus a per-connection latency (socket creation,
# first-use address lookup, handler-thread spawn, scheduler jitter), so
# ``arrival[n+1] - arrival[n] = release gap + latency[n+1] - latency[n]``:
# a cold first connection can legitimately make the server-side gap shorter
# than the interval (observed once on macOS: 0.0257 s against a 0.05 s limit
# with a 0.015 s tolerance) without the limiter having misbehaved. No fixed
# tolerance bounds that under load, so pacing is asserted on the limiter's
# own release instants, and the server's arrival log is used only for what
# it can prove: one release per GET actually received, and no GET arriving
# before its own release.
# =============================================================================
@pytest.fixture
def releases(monkeypatch) -> list[float]:
    """Release instant (the limiter's own ``_last``) of every GET the client
    sends, in order."""
    log: list[float] = []

    class RecordingLimiter(RateLimiter):
        def wait(self) -> None:
            super().wait()
            log.append(self._last)

    monkeypatch.setattr(sec_live_smoke, "RateLimiter", RecordingLimiter)
    return log


def _gaps() -> list[float]:
    """SERVER-side arrival gaps: informative only, never a pacing bound."""
    return [later - earlier for earlier, later in pairwise(_Handler.arrivals)]


def _assert_paced(releases: list[float], interval: float, *, sent: int) -> None:
    assert len(releases) == sent == len(_Handler.arrivals), (releases, _Handler.arrivals)
    gaps = [later - earlier for earlier, later in pairwise(releases)]
    assert len(gaps) == sent - 1
    # exact: both stamps are the limiter's own, so no tolerance is needed
    # beyond float rounding
    assert min(gaps) >= interval - 1e-6, gaps
    # causality: a GET cannot reach the server before the client released it
    assert all(arrived >= released for released, arrived in zip(releases, _Handler.arrivals, strict=True))


def test_server_arrival_gaps_do_not_bound_the_limiter_spacing(base_url, releases):
    """The flake, reproduced deterministically: hold the FIRST connection up
    at the server (cold-start latency). The limiter keeps its contract, yet
    the server-side arrival gap falls far below the interval -- so a pacing
    assertion on arrival gaps (the previous measurement) fails spuriously."""
    _Handler.first_connection_delay = 0.08
    _Handler.chain = {"/a": "/ok"}
    client = _client(rate_limit_rps=20.0)  # 0.05 s minimum spacing

    assert client.get(f"{base_url}/a").ok

    _assert_paced(releases, 0.05, sent=2)  # the contract holds
    assert _gaps()[0] < 0.05 - 0.015, _gaps()  # ...and arrival gaps cannot show it


def test_redirect_hops_are_paced_like_initial_gets(base_url, releases):
    _Handler.chain = {"/a": "/b", "/b": "/c", "/c": "/ok"}
    client = _client(rate_limit_rps=10.0)  # 0.1 s minimum spacing

    assert client.get(f"{base_url}/a").ok

    assert _Handler.hits == ["/a", "/b", "/c", "/ok"]
    _assert_paced(releases, 0.1, sent=4)


def test_retries_and_redirects_are_paced_and_still_capped_at_six(base_url, releases):
    _Handler.chain = {"/r1": "/r2", "/r2": "/fail"}
    _Handler.fail = {"/fail": 500}
    client = _client(rate_limit_rps=20.0, max_requests=6, max_retries=2)  # 0.05 s

    assert not client.get(f"{base_url}/r1").ok

    assert _Handler.hits == ["/r1", "/r2", "/fail", "/r1", "/r2", "/fail"]
    assert client.attempts_made == 6
    _assert_paced(releases, 0.05, sent=6)


def test_pacing_spans_logical_requests_and_their_hops_up_to_the_cap(base_url, releases):
    _Handler.chain = {f"/p{i}": f"/q{i}" for i in range(10)}
    client = _client(rate_limit_rps=20.0, max_requests=6)

    for i in range(10):
        try:
            client.get(f"{base_url}/p{i}")
        except MaxRequestsExceededError:
            break

    assert len(_Handler.hits) == 6 == client.attempts_made
    _assert_paced(releases, 0.05, sent=6)


def test_a_cold_first_connection_does_not_loosen_the_limiter_across_logical_requests(
    base_url, releases,
):
    """Same cold-start condition across separate .get() calls."""
    _Handler.first_connection_delay = 0.08
    client = _client(rate_limit_rps=20.0, max_requests=6)

    for i in range(3):
        assert client.get(f"{base_url}/ok{i}").ok

    _assert_paced(releases, 0.05, sent=3)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_contract_on_a_fake_clock(monkeypatch):
    """The RateLimiter contract, deterministically: consecutive wait()
    returns are at least ``1/rps`` apart; no wait when the caller was slower."""
    clock = _FakeClock()
    monkeypatch.setattr(http_module, "time", clock)  # the module's own name only
    limiter = RateLimiter(2.0)  # 0.5 s

    stamps = []
    for work in (0.0, 0.1, 0.9, 0.0):  # time the caller spends between calls
        limiter.wait()
        stamps.append(clock.now)
        clock.now += work

    assert all(b - a >= 0.5 - 1e-9 for a, b in pairwise(stamps)), stamps
    assert clock.sleeps == pytest.approx([0.5, 0.4])  # the slow caller (4th call) waits for nothing
    assert RateLimiter(0.0).wait() is None  # rps<=0 disables spacing


def test_a_redirect_refused_by_the_budget_is_never_waited_for(base_url):
    """Budget is checked before the rate-limit wait: a GET that will not be
    sent costs no time."""
    _Handler.chain = {"/a": "/ok"}
    client = _client(rate_limit_rps=2.0, max_requests=1)  # 0.5 s spacing

    started = time.monotonic()
    result = client.get(f"{base_url}/a")
    elapsed = time.monotonic() - started

    assert result.outcome is FetchOutcome.ERROR
    assert _Handler.hits == ["/a"]
    assert elapsed < 0.4, elapsed


def test_a_disallowed_redirect_is_refused_without_waiting_or_connecting(base_url):
    client = _client(rate_limit_rps=2.0)  # 0.5 s spacing

    started = time.monotonic()
    result = client.get(f"{base_url}/to-disallowed")
    elapsed = time.monotonic() - started

    assert result.outcome is FetchOutcome.BLOCKED
    assert _DisallowedHandler.hits == []
    assert client.attempts_made == 1
    assert elapsed < 0.4, elapsed
