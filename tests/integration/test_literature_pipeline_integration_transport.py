"""Phase 4.2A: Literature Pipeline Integration's OWN transport dependencies
-- ``AllowlistedHttpClient`` (the SAME class every other live-communication-
capable module in this repository uses) and ``PubMedLiteratureAdapter`` --
exercised against real loopback sockets, never the real internet.

Mirrors ``tests/integration/test_literature_live_smoke_transport.py``'s own
two end-to-end tests (full orchestration + secret sweep; credential-echo
rejection) adapted to drive ``run_literature_pipeline_acquisition`` instead
of ``run_live_smoke`` -- proving THIS module's production wiring reuses
the same, already-hardened transport correctly: pre-connect redirect
validation, the fail-closed credential-echo gate, and secret/URL
non-leakage all hold when reached through the production entry point, not
just through Literature Live Smoke's own manual tool.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import pytest

from investment_research.research import literature_pipeline_integration as lpi

pytestmark = pytest.mark.integration

_EFETCH_BODY = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
<MedlineCitation Status="MEDLINE" Owner="NLM">
<PMID Version="1">90000008</PMID>
<Article PubModel="Print-Electronic">
<Journal><Title>Fictional Journal of Full Identifiers</Title>
<JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
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

_FULLTEXT_BODY = """<?xml version="1.0"?>
<article><body>
<sec><title>Introduction</title><p>Fictional full-text introduction content.</p></sec>
<sec><title>Methods</title><p>Fictional methods content.</p></sec>
<sec><title>Results</title><p>Fictional results content.</p></sec>
</body></article>"""


class Handler(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        Handler.state.setdefault("raw_paths", []).append(self.path)
        if path == "/entrez/eutils/efetch.fcgi":
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
            self._redirect(f"http://localhost:{Handler.state['disallowed_port']}/ok")
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

    yield allowed
    allowed.shutdown()
    disallowed.shutdown()


def _monkeypatch_hosts(monkeypatch, base_url: str) -> None:
    """Mirrors Literature Live Smoke's own loopback technique exactly:
    monkeypatch literature_acquisition_adapter's module-level URL/
    ALLOWED_HOSTS constants (restored automatically after the test) so the
    real AllowlistedHttpClient + PubMedLiteratureAdapter this module
    constructs in production run against the loopback server instead of
    the real NCBI/Europe PMC hosts. literature_pipeline_integration.py
    imported ALLOWED_HOSTS into its OWN namespace too -- a separate bound
    name, patched here as well, for the client run_literature_pipeline_
    acquisition() constructs ITSELF (never injected in these tests)."""
    import investment_research.research.literature_acquisition_adapter as adapter_module

    loopback_hosts = frozenset({"127.0.0.1"})
    monkeypatch.setattr(adapter_module, "ALLOWED_HOSTS", loopback_hosts)
    monkeypatch.setattr(adapter_module, "NCBI_ESEARCH_URL", f"{base_url}/entrez/eutils/esearch.fcgi")
    monkeypatch.setattr(adapter_module, "NCBI_EFETCH_URL", f"{base_url}/entrez/eutils/efetch.fcgi")
    monkeypatch.setattr(adapter_module, "EUROPEPMC_SEARCH_URL", f"{base_url}/europepmc/webservices/rest/search")
    monkeypatch.setattr(
        adapter_module, "EUROPEPMC_FULLTEXT_URL", f"{base_url}/europepmc/webservices/rest/{{pmcid}}/fullTextXML",
    )
    monkeypatch.setattr(lpi, "ALLOWED_HOSTS", loopback_hosts)


# --- targeted PMID, full orchestration, over real loopback sockets ---------
def test_run_literature_pipeline_acquisition_full_orchestration_over_a_real_loopback_server(
    base_url, monkeypatch
):
    _monkeypatch_hosts(monkeypatch, base_url)

    secret_email = "sweep-secret@example.test"
    secret_api_key = "sweep-api-key-12345"
    secret_tool = "sweep-tool-value"
    env = {"IRA_NCBI_TOOL": secret_tool, "IRA_NCBI_EMAIL": secret_email, "IRA_NCBI_API_KEY": secret_api_key}

    request = lpi.LiteraturePipelineRequest(
        reference_mode=lpi.LiteratureReferenceMode.PMID, pmid="90000008", nct_id=None,
        max_articles=lpi.DEFAULT_MAX_ARTICLES, max_fulltext_fetches=lpi.DEFAULT_MAX_FULLTEXT_FETCHES,
    )
    bundle = lpi.run_literature_pipeline_acquisition(request, ticker="DEMOBIO", env=env)

    # 1. Full orchestration actually happened, end to end, over real sockets.
    assert bundle.refused_reason is None
    assert bundle.document_count == 3  # pubmed + europepmc search + europepmc fulltext
    assert bundle.chunk_count > 0
    assert bundle.physical_attempt_count >= bundle.logical_request_count > 0

    # 2. The loopback server itself DID receive the secret -- confirms
    # every check below is sanitization, not simply never sending the value.
    all_raw_paths = " ".join(Handler.state.get("raw_paths", []))
    encoded_email = quote(secret_email, safe="")
    assert secret_api_key in all_raw_paths
    assert secret_tool in all_raw_paths
    assert encoded_email in all_raw_paths
    secrets_to_check = [secret_email, secret_api_key, secret_tool, encoded_email]

    diagnostics = lpi.bundle_diagnostics(bundle, enabled=True)
    diagnostics_json = json.dumps(diagnostics, default=str)
    for value in secrets_to_check:
        assert value not in diagnostics_json, f"{value!r} leaked into bundle_diagnostics JSON"
        assert value not in repr(bundle), f"{value!r} leaked into repr(bundle)"
        for reason in bundle.unresolved_reasons:
            assert value not in reason, f"{value!r} leaked into an unresolved reason"
        for source in bundle.collection_result.sources:
            assert value not in source.url, f"{value!r} leaked into a Source url"
        for raw_fact in bundle.collection_result.raw_facts:
            assert value not in raw_fact.source.url, f"{value!r} leaked into a RawFact source url"


# --- disallowed redirect: two INDEPENDENT servers, hit count = 0 -----------
def test_disallowed_redirect_during_literature_pipeline_acquisition_makes_zero_requests(
    two_servers, monkeypatch
):
    base_url = f"http://127.0.0.1:{two_servers.server_port}"
    import investment_research.research.literature_acquisition_adapter as adapter_module

    loopback_hosts = frozenset({"127.0.0.1"})
    monkeypatch.setattr(adapter_module, "ALLOWED_HOSTS", loopback_hosts)
    # Route EFetch itself at the redirect-to-disallowed path.
    monkeypatch.setattr(
        adapter_module, "NCBI_EFETCH_URL", f"{base_url}/redirect-to-disallowed-server",
    )
    monkeypatch.setattr(adapter_module, "NCBI_ESEARCH_URL", f"{base_url}/entrez/eutils/esearch.fcgi")
    monkeypatch.setattr(adapter_module, "EUROPEPMC_SEARCH_URL", f"{base_url}/europepmc/webservices/rest/search")
    monkeypatch.setattr(
        adapter_module, "EUROPEPMC_FULLTEXT_URL", f"{base_url}/europepmc/webservices/rest/{{pmcid}}/fullTextXML",
    )
    monkeypatch.setattr(lpi, "ALLOWED_HOSTS", loopback_hosts)

    env = {"IRA_NCBI_TOOL": "t", "IRA_NCBI_EMAIL": "e@example.com"}
    request = lpi.LiteraturePipelineRequest(
        reference_mode=lpi.LiteratureReferenceMode.PMID, pmid="90000008", nct_id=None,
        max_articles=lpi.DEFAULT_MAX_ARTICLES, max_fulltext_fetches=lpi.DEFAULT_MAX_FULLTEXT_FETCHES,
    )
    bundle = lpi.run_literature_pipeline_acquisition(request, ticker="DEMOBIO", env=env)

    assert DisallowedHandler.state.get("hits", 0) == 0  # never reached the disallowed server
    assert bundle.collection_result.degraded
    assert bundle.coverage_complete is False
    assert bundle.document_count == 0


# --- credential-echo response: BLOCKED, parser never reached ---------------
_CREDENTIAL_ECHO_EFETCH_BODY = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
<MedlineCitation Status="MEDLINE" Owner="NLM">
<PMID Version="1">90000009</PMID>
<Article PubModel="Print-Electronic">
<Journal><Title>Fictional Journal of Credential Echo Handling</Title>
<JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
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
    state: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        CredentialEchoHandler.state.setdefault("raw_paths", []).append(self.path)
        if path == "/entrez/eutils/efetch.fcgi":
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


def test_credential_echoing_europepmc_fulltext_response_is_rejected_parser_never_reached(
    credential_echo_base_url, monkeypatch
):
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
    monkeypatch.setattr(lpi, "ALLOWED_HOSTS", loopback_hosts)

    env = {
        "IRA_NCBI_TOOL": "credential-echo-pipeline-test-tool",
        "IRA_NCBI_EMAIL": "credential-echo-pipeline@example.test",
    }
    request = lpi.LiteraturePipelineRequest(
        reference_mode=lpi.LiteratureReferenceMode.PMID, pmid="90000009", nct_id=None,
        max_articles=lpi.DEFAULT_MAX_ARTICLES, max_fulltext_fetches=lpi.DEFAULT_MAX_FULLTEXT_FETCHES,
    )
    bundle = lpi.run_literature_pipeline_acquisition(request, ticker="DEMOBIO", env=env)

    # The server genuinely received a real request for the full-text route.
    assert any(
        "/europepmc/webservices/rest/PMC9990009/fullTextXML" in p
        for p in CredentialEchoHandler.state.get("raw_paths", [])
    )
    # 1. The parser was never invoked for the rejected response.
    assert parse_calls == []
    # 2. No fulltext Document was produced -- only PubMed + search documents.
    assert bundle.document_count == 2
    # 3. coverage_complete is False -- a genuine provider rejection.
    assert bundle.coverage_complete is False
    assert bundle.collection_result.degraded
    assert any("version chain corrupted" not in r for r in bundle.unresolved_reasons)
    combined = " ".join(bundle.unresolved_reasons)
    assert "response rejected because it echoed request credentials" in combined
    assert lpi.LITERATURE_PIPELINE_USER_AGENT not in combined
