"""Phase 3E.2: Form4Adapter exercised over a REAL local loopback HTTP
round trip via ``sec_live_smoke.AllowlistedHttpClient`` -- never the real
internet, but real ``urllib`` request/redirect mechanics rather than a
hand-rolled fake, proving Form4Adapter genuinely interoperates with the
transport class ``form4_live_smoke.py`` actually uses (every other Form4
test drives the adapter with ``FakeHttpClient``).

``AllowlistedHttpClient``'s own generic host-allowlist/redirect/request-
cap/Capture-Manifest behavior is already proven source-agnostically (and
specifically parametrized with a non-"sec" ``source``) in
``test_sec_live_smoke_transport.py`` -- this file does not repeat that;
it proves Form4Adapter's LOCATE->FETCH->PARSE chain and its Capture
Manifest writes specifically.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.research.acquisition_executor import ExecutionContext
from investment_research.research.capture_manifest import read_manifest
from investment_research.research.document_store import DocumentStore
from investment_research.research.form4_acquisition_adapter import (
    Form4Adapter,
    Form4IssuerReference,
)
from investment_research.research.form4_live_smoke import _build_dummy_target, _step
from investment_research.research.sec_live_smoke import AllowlistedHttpClient
from investment_research.research.source_routing import StepKind, StepStatus

pytestmark = pytest.mark.integration

_AGENT = "investment-research-agent-test smoke-test-contact@example.test"
_ISSUER_CIK = 1112223
_ACCESSION = "0001112223-26-000001"
_ACCESSION_NODASH = _ACCESSION.replace("-", "")

_OWNERSHIP_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <issuer>
        <issuerCik>0001112223</issuerCik>
        <issuerName>Sample Biotech Holdings, Inc.</issuerName>
        <issuerTradingSymbol>SAMPB</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0005556667</rptOwnerCik>
            <rptOwnerName>Sample Reporting Person</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector><isOfficer>0</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-02-27</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>12.34</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts><sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
            <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <remarks>None</remarks>
    <ownerSignature><signatureName>/s/ Sample Reporting Person</signatureName><signatureDate>2026-03-01</signatureDate></ownerSignature>
</ownershipDocument>
"""

_SUBMISSIONS = {
    "cik": _ISSUER_CIK, "name": "Sample Biotech Holdings, Inc.", "tickers": ["SAMPB"],
    "filings": {
        "recent": {
            "accessionNumber": [_ACCESSION], "form": ["4"], "primaryDocument": ["primary_doc.xml"],
            "filingDate": ["2026-03-01"], "reportDate": ["2026-03-01"],
        },
        "files": [],
    },
}

_DIRECTORY = {"directory": {"item": [{"name": "primary_doc.xml", "type": "4"}]}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == f"/submissions/CIK{_ISSUER_CIK:010d}.json":
            self._send_json(_SUBMISSIONS)
        elif path == f"/Archives/edgar/data/{_ISSUER_CIK}/{_ACCESSION_NODASH}/index.json":
            self._send_json(_DIRECTORY)
        elif path == f"/Archives/edgar/data/{_ISSUER_CIK}/{_ACCESSION_NODASH}/primary_doc.xml":
            body = _OWNERSHIP_XML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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


def test_form4_adapter_locate_fetch_parse_over_real_loopback_transport(base_url, tmp_path, monkeypatch):
    """Patches the two SEC URL templates Form4Adapter actually issues GETs
    against to point at the loopback server -- never patches Form4Adapter's
    own logic. Proves LOCATE->FETCH->PARSE genuinely completes over a real
    HTTP round trip, and that every one of the 3 real GETs (submissions,
    directory index, ownership XML body) gets its own verified Capture
    Manifest with source="form4"."""
    import investment_research.collectors.sec_edgar as sec_edgar

    monkeypatch.setattr(sec_edgar, "SUBMISSIONS_URL", base_url + "/submissions/CIK{cik:010d}.json")
    monkeypatch.setattr(sec_edgar, "FILING_INDEX_URL", base_url + "/Archives/edgar/data/{cik}/{accession_nodash}/{document}")
    import investment_research.research.sec_acquisition_adapters as sec_adapters

    monkeypatch.setattr(sec_adapters, "SUBMISSIONS_URL", sec_edgar.SUBMISSIONS_URL)
    monkeypatch.setattr(sec_adapters, "FILING_INDEX_URL", sec_edgar.FILING_INDEX_URL)

    client = AllowlistedHttpClient(
        user_agent=_AGENT, allowed_hosts=frozenset({"127.0.0.1"}), timeout=2.0,
        rate_limit_rps=1000.0, max_retries=0, max_requests=6, out_dir=tmp_path, source="form4",
    )
    store = DocumentStore()
    adapter = Form4Adapter(client)
    ref = Form4IssuerReference(issuer_cik=_ISSUER_CIK, max_candidates=5)
    context = ExecutionContext(target=_build_dummy_target(), document_store=store, form4_references={"target_form4_live_smoke": ref})

    l1 = _step("l1", StepKind.LOCATE)
    locate_result = adapter.execute(l1, context)
    assert locate_result.status is StepStatus.URL_RESOLVED
    context.payloads[l1.step_id] = locate_result.payload

    f = _step("f", StepKind.FETCH, depends_on=(l1.step_id,))
    fetch_result = adapter.execute(f, context)
    assert fetch_result.status is StepStatus.BODY_FETCHED
    context.payloads[f.step_id] = fetch_result.payload

    p = _step("p", StepKind.PARSE, depends_on=(f.step_id,))
    parse_result = adapter.execute(p, context)
    assert parse_result.status is StepStatus.PARSED
    parsed = parse_result.payload["parsed_documents"][0]["parsed"]
    assert parsed["issuer_cik"] == "0001112223"
    assert len(parsed["non_derivative_transactions"]) == 1

    assert client.requests_made == 3  # submissions + directory index + document body
    for url in client.requested_urls:
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        cached = client._cache[url]
        read_result = read_manifest(tmp_path, digest, body=cached.body)
        assert read_result.status.value == "VERIFIED"
        assert read_result.manifest is not None
        assert read_result.manifest.source == "form4"
