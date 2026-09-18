"""Phase 3F.1: Literature Live Smoke's OWN transport dependencies --
``AllowlistedHttpClient`` and the shared Capture Manifest -- exercised
against real loopback sockets, never the real internet.

Most tests here exercise ``AllowlistedHttpClient`` (the SAME class
``run_live_smoke`` constructs in production) and
``literature_live_smoke.analyze_capture`` directly, without going through
``PubMedLiteratureAdapter`` at all, because
``research/literature_acquisition_adapter.py`` hardcodes its own real
NCBI/Europe PMC hostnames (``NCBI_ESEARCH_URL``, ``ALLOWED_HOSTS``, ...)
with no PRODUCTION injection point for a test host.

``test_run_live_smoke_full_orchestration_and_secret_sweep_over_a_real_loopback_server``
(Phase 3F.1 correction requirement 8) is the exception: it drives
``run_live_smoke()``'s FULL orchestration (LOCATE via ESearch -> FETCH via
EFetch + Europe PMC search/full-text -> PARSE) end-to-end over real
sockets, by monkeypatching ``literature_acquisition_adapter``'s
module-level URL/``ALLOWED_HOSTS`` constants for the duration of that one
test. Python resolves a module global at CALL time, so this requires no
production code change or DI seam -- it is a test-only technique, reverted
automatically by pytest's ``monkeypatch`` fixture, never a claim that the
production adapter is actually configurable. That same test also performs
the comprehensive secret-non-leakage sweep Phase 3F.1 correction
requirement 2 asks for (text output, JSON output, ``repr()``, error
strings, Capture Manifest files, and the transport's own cache-key set)
against a REAL end-to-end run, since only a real run populates every one
of those surfaces meaningfully.

Every scenario Phase 3F.1 requirement 8 asks for that ``AllowlistedHttpClient``
itself is responsible for (pre-connect redirect validation, an explicit
redirect-count cap, the physical attempt cap, secret sanitization) is ALSO
covered generically by ``tests/integration/test_sec_live_smoke_transport.py``
-- the other tests in this file are not a second from-scratch proof of
that machinery, but a proof that THIS module's own choices (which
host/path shapes it builds, how ``analyze_capture`` classifies and
replays them) compose with it correctly.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import pytest

from investment_research.research import literature_live_smoke as live
from investment_research.research.capture_manifest import read_manifest
from investment_research.research.sec_live_smoke import AllowlistedHttpClient, _safe

pytestmark = pytest.mark.integration

_AGENT = "investment-research-agent-literature-live-smoke-test/1.0"
_SECRET_EMAIL = "secret-contact@example.test"
_SECRET_API_KEY = "secret-ncbi-api-key-12345"
_SECRET_TOOL = "secret-tool-value"


class Handler(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        counts = Handler.state.setdefault("counts", {})
        counts[path] = counts.get(path, 0) + 1
        Handler.state["last_raw_path"] = self.path
        Handler.state.setdefault("raw_paths", []).append(self.path)

        if path == "/entrez/eutils/esearch.fcgi":
            self._send(200, json.dumps({"esearchresult": {"idlist": ["90000008"]}}).encode())
        elif path == "/entrez/eutils/efetch.fcgi":
            self._send(200, _EFETCH_BODY.encode())
        elif path == "/europepmc/webservices/rest/search":
            self._send(
                200,
                json.dumps(
                    {"resultList": {"result": [{"pmid": "90000008", "pmcid": "PMC9990008", "isOpenAccess": "Y", "inEPMC": "Y"}]}}
                ).encode(),
            )
        elif path == "/europepmc/webservices/rest/PMC9990008/fullTextXML":
            self._send(200, _FULLTEXT_BODY.encode())
        elif path == "/redirect-to-disallowed-server":
            self._redirect(
                f"http://localhost:{Handler.state['disallowed_port']}/ok"
                f"?email={_SECRET_EMAIL}&api_key={_SECRET_API_KEY}"
            )
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


_EFETCH_BODY = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
<MedlineCitation Status="MEDLINE" Owner="NLM">
<PMID Version="1">90000008</PMID>
<Article PubModel="Print-Electronic">
<Journal><Title>Fictional Journal of Full Identifiers</Title></Journal>
<ArticleTitle>A fictional article carrying PMID, PMCID, and DOI together</ArticleTitle>
<PublicationTypeList><PublicationType UI="D016428">Journal Article</PublicationType></PublicationTypeList>
</Article>
</MedlineCitation>
<PubmedData>
<ArticleIdList>
<ArticleId IdType="pubmed">90000008</ArticleId>
<ArticleId IdType="pmc">PMC9990008</ArticleId>
</ArticleIdList>
</PubmedData>
</PubmedArticle>
</PubmedArticleSet>"""

_FULLTEXT_BODY = "<article><body><sec><title>Results</title><p>Fictional full text.</p></sec></body></article>"


@pytest.fixture
def base_url():
    Handler.state = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def two_servers():
    DisallowedHandler.state = {}
    disallowed = ThreadingHTTPServer(("127.0.0.1", 0), DisallowedHandler)
    threading.Thread(target=disallowed.serve_forever, daemon=True).start()

    Handler.state = {"disallowed_port": disallowed.server_port}
    allowed = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=allowed.serve_forever, daemon=True).start()

    yield f"http://127.0.0.1:{allowed.server_port}"
    allowed.shutdown()
    disallowed.shutdown()


def _client(**overrides):
    kwargs = {
        "user_agent": _AGENT, "allowed_hosts": frozenset({"127.0.0.1"}), "timeout": 2.0,
        "rate_limit_rps": 1000.0, "max_retries": 0, "max_requests": 10, "source": "literature",
    }
    kwargs.update(overrides)
    return AllowlistedHttpClient(**kwargs)


# --- secret non-leakage over a REAL wire, raw and percent-encoded ----------
def test_secret_raw_and_percent_encoded_values_never_leak_past_the_transport(base_url, tmp_path):
    client = _client(out_dir=tmp_path)
    encoded_email = "secret-contact%40example.test"
    url = f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&email={encoded_email}&api_key={_SECRET_API_KEY}&tool=t"
    result = client.get(url)
    assert result.ok

    # The server itself DID receive both, raw and percent-encoded --
    # confirms this is sanitization, not simply never sending the value.
    assert _SECRET_API_KEY in Handler.state["last_raw_path"]
    assert encoded_email in Handler.state["last_raw_path"]

    for leaked in (_SECRET_API_KEY, _SECRET_EMAIL, encoded_email, "secret-contact"):
        assert leaked not in result.url
        assert leaked not in result.final_url
        assert leaked not in result.error
        assert leaked not in repr(result)

    import hashlib

    # The capture digest is derived from the PUBLIC url (Phase 3F.1
    # correction requirement 1), never the wire url that carried the
    # secret -- so it is computed here from result.url, not `url` itself.
    digest = hashlib.sha256(result.url.encode()).hexdigest()[:24]
    manifest_text = (tmp_path / f"{digest}.manifest.json").read_text(encoding="utf-8")
    for leaked in (_SECRET_API_KEY, _SECRET_EMAIL, encoded_email):
        assert leaked not in manifest_text


# --- disallowed redirect: two INDEPENDENT servers, hit count = 0 -----------
def test_redirect_to_a_disallowed_server_makes_zero_requests_and_leaks_no_secret(two_servers):
    client = _client()
    result = client.get(f"{two_servers}/redirect-to-disallowed-server")
    assert not result.ok
    assert result.outcome.value == "BLOCKED"
    assert DisallowedHandler.state.get("hits", 0) == 0
    assert _SECRET_EMAIL not in result.error
    assert _SECRET_API_KEY not in result.error
    assert _SECRET_EMAIL not in result.url


# --- Capture Manifest: real write, then literature_live_smoke's OWN offline
# replay, classifying every captured request kind correctly --------------
def test_capture_then_offline_replay_classifies_every_request_kind(base_url, tmp_path):
    client = _client(out_dir=tmp_path)

    esearch_url = f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&term=NCT09990001%5Bsi%5D&retmode=json&retmax=3&tool=t&email={_SECRET_EMAIL}"
    efetch_url = f"{base_url}/entrez/eutils/efetch.fcgi?db=pubmed&id=90000008&retmode=xml&tool=t&email={_SECRET_EMAIL}"
    epmc_search_url = f"{base_url}/europepmc/webservices/rest/search?query=ext_id%3A90000008+AND+src%3Amed&format=json"
    epmc_fulltext_url = f"{base_url}/europepmc/webservices/rest/PMC9990008/fullTextXML"

    for url in (esearch_url, efetch_url, epmc_search_url, epmc_fulltext_url):
        result = client.get(url)
        assert result.ok, f"loopback GET failed for {url}: {result.error}"

    # analyze_capture is entirely offline from here -- no client, no
    # credentials, no network -- just the manifests/bodies just written.
    replay = live.analyze_capture(tmp_path)
    assert replay["found"] is True
    assert replay["esearch"]["pmids_found"] == ["90000008"]
    assert _SECRET_EMAIL not in replay["esearch"]["requested_url"]

    assert len(replay["efetch"]) == 1
    assert replay["efetch"][0]["parsed_ok"] is True
    assert replay["efetch"][0]["articles"][0]["pmid"] == "90000008"
    assert _SECRET_EMAIL not in replay["efetch"][0]["requested_url"]

    assert len(replay["europepmc_search"]) == 1
    assert replay["europepmc_search"][0]["results"][0]["pmid"] == "90000008"
    assert replay["europepmc_search"][0]["results"][0]["is_open_access"] is True

    assert len(replay["europepmc_fulltext"]) == 1
    assert replay["europepmc_fulltext"][0]["parsed_ok"] is True
    assert replay["europepmc_fulltext"][0]["section_count"] == 1
    # The full-text SECTION TEXT ("Fictional full text.") must never
    # surface anywhere in the replay result.
    assert "Fictional full text" not in json.dumps(replay)

    assert replay["evidence_integrity_failures"] == []


def test_capture_manifest_fields_preserved_across_the_real_round_trip(base_url, tmp_path):
    client = _client(out_dir=tmp_path)
    url = f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&term=x&retmode=json&retmax=3"
    result = client.get(url)
    assert result.ok

    import hashlib

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    read_result = read_manifest(tmp_path, digest, body=result.body)
    assert read_result.status.value == "VERIFIED"
    manifest = read_result.manifest
    assert manifest is not None
    assert manifest.source == "literature"
    assert manifest.http_status == 200
    assert manifest.content_hash == hashlib.sha256(result.body).hexdigest()
    assert manifest.content_length == len(result.body)
    assert manifest.capture_retrieved_at.endswith("Z")


def test_offline_replay_of_a_corrupted_capture_is_reported_as_an_evidence_integrity_failure(base_url, tmp_path):
    client = _client(out_dir=tmp_path)
    url = f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&term=x&retmode=json&retmax=3"
    result = client.get(url)
    assert result.ok

    import hashlib

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    body_path = tmp_path / f"{digest}.bin"
    original = body_path.read_bytes()
    body_path.write_bytes(original[:-1] + (b"X" if original[-1:] != b"X" else b"Y"))

    replay = live.analyze_capture(tmp_path)
    statuses = {f["status"] for f in replay["evidence_integrity_failures"]}
    assert "HASH_MISMATCH" in statuses


# --- Phase 3F.1 correction 2 requirement 1: on_first_attempt marker timing,
# against a REAL loopback server (not FakeHttpClient) ------------------------
def test_on_first_attempt_fires_before_the_first_real_socket_attempt(base_url):
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"))
    assert Handler.state.get("counts", {}) == {}
    result = client.get(f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&term=x&retmode=json")
    assert result.ok
    assert events == ["callback"]
    assert Handler.state["counts"] == {"/entrez/eutils/esearch.fcgi": 1}


def test_on_first_attempt_fires_only_once_across_multiple_logical_requests(base_url):
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"))
    client.get(f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&term=x&retmode=json")
    client.get(f"{base_url}/entrez/eutils/efetch.fcgi?db=pubmed&id=1&retmode=xml")
    assert events == ["callback"]


def test_on_first_attempt_fires_even_though_the_request_then_fails(base_url):
    """A definitive, non-OK outcome (404, here) still means the first real
    socket-level attempt happened -- the callback already fired by the
    time that outcome is known (Phase 3F.1 correction 2 requirement 1:
    "sent, then failed" must never erase that a real attempt was made)."""
    events: list[str] = []
    client = _client(on_first_attempt=lambda: events.append("callback"))
    result = client.get(f"{base_url}/does-not-exist")
    assert result.outcome.value == "NOT_FOUND"
    assert events == ["callback"]


def test_on_first_attempt_raising_makes_zero_physical_attempts_and_zero_server_hits(base_url):
    """If the callback itself raises (standing in for "the marker write
    failed"), the real transport must make ZERO server hits -- proving
    "HTTP communication must not begin" all the way down at the socket
    level, not merely at some higher, mockable layer (Phase 3F.1
    correction 2 requirement 1, test scenario (e))."""
    def _boom() -> None:
        raise OSError("simulated marker write failure")

    client = _client(on_first_attempt=_boom)
    with pytest.raises(OSError):
        client.get(f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&term=x&retmode=json")
    assert Handler.state.get("counts", {}) == {}
    assert client.attempts_made == 0
    assert client.requests_made == 0


# --- Phase 3F.1 correction 2 requirement 2: direct _cache inspection -------
def test_real_client_cache_keys_never_contain_a_secret_value(base_url):
    """Directly inspects AllowlistedHttpClient's own private ``_cache``
    dict KEYS (not report.requested_urls, which is merely populated FROM
    it) -- with literature_live_smoke's own stricter
    ``additional_secret_query_params={"tool"}`` policy applied, exactly as
    ``run_live_smoke`` configures it in production."""
    client = _client(additional_secret_query_params=frozenset({"tool"}))
    encoded_email = quote(_SECRET_EMAIL, safe="")
    url = (
        f"{base_url}/entrez/eutils/esearch.fcgi?db=pubmed&email={encoded_email}"
        f"&api_key={_SECRET_API_KEY}&tool={_SECRET_TOOL}"
    )
    result = client.get(url)
    assert result.ok
    assert client._cache  # not empty -- there is something to inspect
    for key in client._cache:
        assert _SECRET_API_KEY not in key
        assert _SECRET_EMAIL not in key
        assert encoded_email not in key
        assert _SECRET_TOOL not in key


# --- Phase 3F.2 correction requirements 5 + 6: failed-response Capture
# Manifest + offline replay, over a REAL loopback 404 ------------------------
def test_a_real_404_fulltext_response_is_captured_and_offline_replay_reports_it(base_url, tmp_path):
    """Literature Live Smoke's own opt-in (save_failed_responses=True):
    a fullTextXML 404 -- this run's own real-world finding -- gets an
    offline-auditable Capture Manifest, never a silent absence. The
    manifest is VERIFIED (Evidence Integrity is about "does the body
    match its own manifest", never "did the acquisition succeed") while
    analyze_capture's own per-entry result explicitly reports
    http_status=404, parsed_ok=False, acquisition_failed=True."""
    client = _client(out_dir=tmp_path, save_failed_responses=True)
    # An unregistered PMCID -- the loopback Handler's fallback branch
    # returns a plain 404 with no body, exactly like a real Europe PMC
    # 404 for a PMCID this test never registered a success response for.
    result = client.get(f"{base_url}/europepmc/webservices/rest/PMC0000000/fullTextXML")
    assert not result.ok
    assert result.status == 404

    manifests = list(tmp_path.glob("*.manifest.json"))
    assert len(manifests) == 1

    replay = live.analyze_capture(tmp_path)
    assert replay["found"] is True
    assert replay["evidence_integrity_failures"] == []  # VERIFIED, not corrupted
    assert len(replay["europepmc_fulltext"]) == 1
    entry = replay["europepmc_fulltext"][0]
    assert entry["http_status"] == 404
    assert entry["parsed_ok"] is False
    assert entry["acquisition_failed"] is True
    assert entry["capture_manifest_status"] == "VERIFIED"
    assert entry["section_count"] == 0


def test_a_corrupted_failed_capture_is_still_an_evidence_integrity_failure(base_url, tmp_path):
    """A saved failed-response body that is later tampered with is still
    caught as an Evidence Integrity failure -- the same guarantee a
    successful capture already had, now extended to a failed one (Phase
    3F.2 correction requirement 6: corrupted body/manifest stays an
    Evidence Integrity failure regardless of http_status)."""
    client = _client(out_dir=tmp_path, save_failed_responses=True)
    result = client.get(f"{base_url}/europepmc/webservices/rest/PMC0000001/fullTextXML")
    assert not result.ok
    assert result.status == 404

    import hashlib

    digest = hashlib.sha256(result.url.encode()).hexdigest()[:24]
    body_path = tmp_path / f"{digest}.bin"
    body_path.write_bytes(b"tampered-after-capture")

    replay = live.analyze_capture(tmp_path)
    statuses = {f["status"] for f in replay["evidence_integrity_failures"]}
    assert statuses  # non-empty -- a real integrity failure was detected
    assert statuses <= {"HASH_MISMATCH", "CONTENT_LENGTH_MISMATCH"}


def test_timeout_never_produces_a_capture_even_with_save_failed_responses(tmp_path):
    client = _client(out_dir=tmp_path, save_failed_responses=True, timeout=1.0)
    result = client.get("http://127.0.0.1:1/unreachable")
    assert not result.ok
    assert result.status is None
    assert list(tmp_path.glob("*.manifest.json")) == []


# --- Phase 3F.1 correction requirements 2 + 8: full orchestration over a
# real loopback server, PLUS the comprehensive secret-non-leakage sweep ----
def test_run_live_smoke_full_orchestration_and_secret_sweep_over_a_real_loopback_server(base_url, tmp_path, monkeypatch):
    """Monkeypatches literature_acquisition_adapter's module-level URL/
    ALLOWED_HOSTS constants (restored automatically after this test) so
    run_live_smoke()'s FULL LOCATE->FETCH->PARSE orchestration runs against
    the real loopback server, using a REAL AllowlistedHttpClient end to
    end -- then sweeps every output surface (text, JSON, repr, error
    strings, Capture Manifest files, transport cache keys) for a
    configured secret, in both raw and percent-encoded form."""
    import investment_research.research.literature_acquisition_adapter as adapter_module

    loopback_hosts = frozenset({"127.0.0.1"})
    monkeypatch.setattr(adapter_module, "ALLOWED_HOSTS", loopback_hosts)
    monkeypatch.setattr(adapter_module, "NCBI_ESEARCH_URL", f"{base_url}/entrez/eutils/esearch.fcgi")
    monkeypatch.setattr(adapter_module, "NCBI_EFETCH_URL", f"{base_url}/entrez/eutils/efetch.fcgi")
    monkeypatch.setattr(adapter_module, "EUROPEPMC_SEARCH_URL", f"{base_url}/europepmc/webservices/rest/search")
    monkeypatch.setattr(
        adapter_module, "EUROPEPMC_FULLTEXT_URL", f"{base_url}/europepmc/webservices/rest/{{pmcid}}/fullTextXML",
    )
    # literature_live_smoke.py imported ALLOWED_HOSTS into its OWN
    # namespace (`from .literature_acquisition_adapter import ALLOWED_HOSTS`)
    # -- a separate bound name, so it must be patched too, for the client
    # run_live_smoke() constructs ITSELF (never injected below, so this
    # test exercises the exact production client-construction call,
    # additional_secret_query_params included).
    monkeypatch.setattr(live, "ALLOWED_HOSTS", loopback_hosts)

    secret_email = "sweep-secret@example.test"
    secret_api_key = "sweep-api-key-12345"
    secret_tool = "sweep-tool-value"
    env = {"IRA_NCBI_TOOL": secret_tool, "IRA_NCBI_EMAIL": secret_email, "IRA_NCBI_API_KEY": secret_api_key}

    report = live.run_live_smoke(
        mode="discovery", nct_id="NCT09990001", max_articles=3, max_fulltext_fetches=1,
        out_dir=tmp_path, env=env,
    )

    # 1. Full orchestration actually happened, end to end, over real sockets.
    assert not report.refused, report.refused_reason
    assert report.smoke_run_completed is True
    assert report.status is live.LiveSmokeStatus.COMPLETED
    assert report.pmids_located == ["90000008"]
    assert report.articles and report.articles[0]["full_text_acquired"] is True
    assert report.attempt_count >= report.request_count > 0

    # 2. The loopback server itself DID receive the secret, raw and
    # percent-encoded -- confirms every check below is sanitization, not
    # simply never sending the value.
    all_raw_paths = " ".join(Handler.state.get("raw_paths", []))
    assert secret_api_key in all_raw_paths

    encoded_email = quote(secret_email, safe="")
    secrets_to_check = [secret_email, secret_api_key, secret_tool, encoded_email]

    # -- text output --
    printed_text = live.format_report_for_print(report, secrets=[secret_tool, secret_email, secret_api_key])
    for value in secrets_to_check:
        assert value not in printed_text, f"{value!r} leaked into text output"

    # -- JSON output, through the SAME _safe() scrub main() applies --
    printed_json = _safe(
        json.dumps(live.report_to_jsonable(report), default=str),
        [secret_tool, secret_email, secret_api_key],
    )
    for value in secrets_to_check:
        assert value not in printed_json, f"{value!r} leaked into json output"

    # -- repr() --
    for value in secrets_to_check:
        assert value not in repr(report), f"{value!r} leaked into repr(report)"

    # -- error strings --
    combined_errors = " ".join(report.errors)
    for value in secrets_to_check:
        assert value not in combined_errors, f"{value!r} leaked into an error string"

    # -- Capture Manifest files on disk --
    manifest_paths = list(tmp_path.glob("*.manifest.json"))
    assert manifest_paths
    for manifest_path in manifest_paths:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        for value in secrets_to_check:
            assert value not in manifest_text, f"{value!r} leaked into {manifest_path.name}"

    # -- the transport's own internal cache-key set -- report.requested_urls
    # is populated (via _collect_transport_diagnostics) directly from the
    # client's own requested_urls list, which is exactly the set of keys
    # used in the client's internal `_cache` dict (Phase 3F.1 correction
    # requirement 1: cache is keyed by public_url) -- so this is the same
    # check as inspecting the private _cache dict itself, without reaching
    # into the client run_live_smoke() builds and never exposes.
    assert report.requested_urls
    for cache_key in report.requested_urls:
        for value in secrets_to_check:
            assert value not in cache_key, f"{value!r} leaked into a cache key"


# --- Phase 3F.2 correction 3 requirement 4: the fail-closed credential-echo
# gate, exercised end to end through run_live_smoke() itself -- a dedicated
# server/handler (not the shared Handler/base_url above) so this scenario's
# Europe PMC full-text response, which deliberately echoes the literature
# client's own User-Agent back, can never affect any other test in this
# file. ---------------------------------------------------------------------
_CREDENTIAL_ECHO_EFETCH_BODY = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
<MedlineCitation Status="MEDLINE" Owner="NLM">
<PMID Version="1">90000009</PMID>
<Article PubModel="Print-Electronic">
<Journal><Title>Fictional Journal of Credential Echo Handling</Title></Journal>
<ArticleTitle>A fictional article whose full text server echoes request credentials</ArticleTitle>
<PublicationTypeList><PublicationType UI="D016428">Journal Article</PublicationType></PublicationTypeList>
</Article>
</MedlineCitation>
<PubmedData>
<ArticleIdList>
<ArticleId IdType="pubmed">90000009</ArticleId>
<ArticleId IdType="pmc">PMC9990009</ArticleId>
</ArticleIdList>
</PubmedData>
</PubmedArticle>
</PubmedArticleSet>"""


class CredentialEchoHandler(BaseHTTPRequestHandler):
    """Identical LOCATE/FETCH shape to ``Handler`` above (a single PMID that
    resolves to an open-access, in-EPMC PMCID), except its fullTextXML
    route echoes the request's own User-Agent header value into a
    genuinely-200 JSON body -- standing in for a real provider's error/
    debug page that echoes request metadata back to the caller."""

    state: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        CredentialEchoHandler.state.setdefault("raw_paths", []).append(self.path)
        if path == "/entrez/eutils/esearch.fcgi":
            self._send(200, json.dumps({"esearchresult": {"idlist": ["90000009"]}}).encode())
        elif path == "/entrez/eutils/efetch.fcgi":
            self._send(200, _CREDENTIAL_ECHO_EFETCH_BODY.encode())
        elif path == "/europepmc/webservices/rest/search":
            self._send(
                200,
                json.dumps(
                    {"resultList": {"result": [{"pmid": "90000009", "pmcid": "PMC9990009", "isOpenAccess": "Y", "inEPMC": "Y"}]}}
                ).encode(),
            )
        elif path == "/europepmc/webservices/rest/PMC9990009/fullTextXML":
            ua = self.headers.get("User-Agent", "")
            body = json.dumps({"echoed_user_agent": ua}).encode()
            self._send(200, body)
        else:
            self._send(404, b"")

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def credential_echo_base_url():
    CredentialEchoHandler.state = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), CredentialEchoHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_run_live_smoke_rejects_a_credential_echoing_europepmc_fulltext_response(
    credential_echo_base_url, tmp_path, monkeypatch,
):
    """Phase 3F.2 correction 3 requirement 4: drives run_live_smoke() end to
    end (real loopback sockets, LOCATE->FETCH->PARSE) against a Europe PMC
    fullTextXML response that echoes the literature client's own
    User-Agent -- confirming the fail-closed gate's downstream effects at
    the LITERATURE level (not just AllowlistedHttpClient's own FetchResult,
    already covered by test_sec_live_smoke_transport.py):
      - the parser (parse_europepmc_fulltext_xml) is never invoked for this
        response,
      - no fulltext Document is produced (full_text_acquired stays False),
      - report.coverage_complete is False,
      - report.status is BLOCKED,
      - this article's fulltext fetch never enters live_verified_candidates.
    None of this required any change to literature_live_smoke.py or
    literature_acquisition_adapter.py -- it is the existing Phase 3F.2
    correction machinery (transport_outcomes -> _compute_status,
    europepmc_fulltext_failures -> coverage_complete, live_verified_
    candidates gated on full_text_acquired) automatically propagating a
    FetchOutcome.BLOCKED result up from AllowlistedHttpClient."""
    import investment_research.research.literature_acquisition_adapter as adapter_module

    parse_calls: list[str] = []
    real_parse = adapter_module.parse_europepmc_fulltext_xml

    def _spy_parse(text: str):
        parse_calls.append(text)
        return real_parse(text)

    loopback_hosts = frozenset({"127.0.0.1"})
    monkeypatch.setattr(adapter_module, "ALLOWED_HOSTS", loopback_hosts)
    monkeypatch.setattr(adapter_module, "NCBI_ESEARCH_URL", f"{credential_echo_base_url}/entrez/eutils/esearch.fcgi")
    monkeypatch.setattr(adapter_module, "NCBI_EFETCH_URL", f"{credential_echo_base_url}/entrez/eutils/efetch.fcgi")
    monkeypatch.setattr(
        adapter_module, "EUROPEPMC_SEARCH_URL", f"{credential_echo_base_url}/europepmc/webservices/rest/search",
    )
    monkeypatch.setattr(
        adapter_module, "EUROPEPMC_FULLTEXT_URL",
        f"{credential_echo_base_url}/europepmc/webservices/rest/{{pmcid}}/fullTextXML",
    )
    monkeypatch.setattr(adapter_module, "parse_europepmc_fulltext_xml", _spy_parse)
    monkeypatch.setattr(live, "ALLOWED_HOSTS", loopback_hosts)

    # Deliberately realistic (multi-character) credential values -- a
    # single-character IRA_NCBI_TOOL like the bare "t" some other tests use
    # would itself, as a secret value, match almost any English text via
    # the detection gate, producing a false-positive rejection on the
    # ESearch/EFetch legs unrelated to this test's actual scenario.
    env = {
        "IRA_NCBI_TOOL": "credential-echo-smoke-test-tool",
        "IRA_NCBI_EMAIL": "credential-echo-smoke@example.test",
        "IRA_NCBI_API_KEY": "credential-echo-smoke-key-12345",
    }

    report = live.run_live_smoke(
        mode="discovery", nct_id="NCT09990002", max_articles=3, max_fulltext_fetches=1,
        out_dir=tmp_path, env=env,
    )

    # The server genuinely received a real request for the full-text route,
    # and genuinely echoed the User-Agent back -- this is a real rejection
    # of real server behaviour, not "the request never happened".
    assert any(
        "/europepmc/webservices/rest/PMC9990009/fullTextXML" in p
        for p in CredentialEchoHandler.state.get("raw_paths", [])
    )

    # 1. The parser was never invoked for the rejected response.
    assert parse_calls == []

    # 2. No fulltext Document was produced for this article.
    assert report.articles, "PubMed article step must still have run"
    article_entry = report.articles[0]
    assert article_entry["pmid"] == "90000009"
    assert article_entry["europepmc_search_succeeded"] is True
    assert article_entry["full_text_acquired"] is False

    # 3. coverage_complete is False -- a genuine provider failure, not a
    # budget skip, but incompleteness either way.
    assert report.coverage_complete is False
    assert any(f.get("kind") == "BLOCKED" and f.get("pmid") == "90000009" for f in report.europepmc_failures)

    # 4. The whole-run status reflects the BLOCKED transport outcome.
    assert report.status is live.LiveSmokeStatus.BLOCKED

    # 5. This article's fulltext fetch never became a "live verified"
    # candidate -- only its search half did (search_result really did
    # come back OA/inEPMC; the fulltext half is what got rejected).
    assert "target_lit_smoke:europepmc_search:90000009" in report.live_verified_candidates
    assert "target_lit_smoke:europepmc_fulltext:90000009" not in report.live_verified_candidates

    # 6. The fixed, secret-free rejection message reached the report's own
    # error surface, never the raw echoed User-Agent value.
    combined_errors = " ".join(report.errors)
    assert "response rejected because it echoed request credentials" in combined_errors
    assert live.LITERATURE_LIVE_SMOKE_USER_AGENT not in combined_errors
    assert live.LITERATURE_LIVE_SMOKE_USER_AGENT not in repr(report)
