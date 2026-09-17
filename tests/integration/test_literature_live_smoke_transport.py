"""Phase 3F.1: Literature Live Smoke's OWN transport dependencies --
``AllowlistedHttpClient`` and the shared Capture Manifest -- exercised
against real loopback sockets, never the real internet.

Why this file cannot drive ``literature_live_smoke.run_live_smoke()``
end-to-end: ``research/literature_acquisition_adapter.py`` hardcodes its
own real NCBI/Europe PMC hostnames (``NCBI_ESEARCH_URL``, ``ALLOWED_HOSTS``,
...) with no injection point for a test host, so the adapter itself can
never be pointed at a loopback server without either monkeypatching module
constants or a real DNS trick -- neither of which this suite does. Instead,
this file proves the actual transport/replay MACHINERY
``literature_live_smoke.py`` depends on -- ``AllowlistedHttpClient`` (the
SAME class ``run_live_smoke`` constructs in production) and
``literature_live_smoke.analyze_capture`` (which reads whatever
``AllowlistedHttpClient`` actually wrote) -- against real sockets, using
literature-shaped URLs (NCBI/Europe-PMC-style paths and courtesy query
params) so the write-then-replay cycle is proven genuinely end-to-end.

Every scenario Phase 3F.1 requirement 8 asks for that ``AllowlistedHttpClient``
itself is responsible for (pre-connect redirect validation, an explicit
redirect-count cap, secret sanitization) is ALSO covered generically by
``tests/integration/test_sec_live_smoke_transport.py`` -- the tests here
are not a second from-scratch proof of that machinery, but a proof that
THIS module's own choices (which host/path shapes it builds, how
``analyze_capture`` classifies and replays them) compose with it correctly.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.research import literature_live_smoke as live
from investment_research.research.capture_manifest import read_manifest
from investment_research.research.sec_live_smoke import AllowlistedHttpClient

pytestmark = pytest.mark.integration

_AGENT = "investment-research-agent-literature-live-smoke-test/1.0"
_SECRET_EMAIL = "secret-contact@example.test"
_SECRET_API_KEY = "secret-ncbi-api-key-12345"


class Handler(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        counts = Handler.state.setdefault("counts", {})
        counts[path] = counts.get(path, 0) + 1
        Handler.state["last_raw_path"] = self.path

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
        elif path == "/europepmc/webservices/rest/PMC/PMC9990008/fullTextXML":
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

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
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
    epmc_fulltext_url = f"{base_url}/europepmc/webservices/rest/PMC/PMC9990008/fullTextXML"

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
