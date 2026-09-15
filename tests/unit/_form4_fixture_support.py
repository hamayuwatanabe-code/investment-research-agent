"""Shared support for tests driving Form4Adapter against the real-format
fixtures in ``tests/fixtures/form4_real_format/`` (see that directory's
MANIFEST.md for provenance). Not a test module itself -- no ``test_*``
functions live here, so pytest never collects it directly.

``submissions.json``/``index.json`` bodies are built here programmatically
(rather than as 19 near-identical static files) -- the SHAPE still matches
SEC EDGAR's real, documented schema exactly (the same parallel-array
``filings.recent`` structure ``tests/fixtures/sec_edgar_real_format``
models as static files), only the construction method differs. The
interesting content -- the ownership XML bodies themselves -- are real
static fixture files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from investment_research.collectors.sec_edgar import FILING_INDEX_URL, SUBMISSIONS_URL
from investment_research.schemas.enums import FetchOutcome

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "form4_real_format"

#: Wholly fictional issuer/reporting-owner identifiers -- never a real
#: company, ticker, CIK, or accession number (Phase 3E requirement 14).
ISSUER_CIK = "0001112223"
ISSUER_NAME = "Sample Biotech Holdings, Inc."
ISSUER_TICKER = "SAMPB"
FILER_CIK = 5556667  # int, as Form4FilingReference.filer_cik expects
FILER_CIK_PADDED = "0005556667"
OWNER_NAME = "Sample Reporting Person"

PRIMARY_DOCUMENT = "primary_doc.xml"

#: scenario name -> (accession suffix, authoritative `form` value SEC's own
#: submissions metadata records, fixture XML filename or None). `form`
#: values other than "4"/"4/A" (the metadata_says_form3 case) exist
#: specifically to prove LOCATE refuses BEFORE any body fetch is attempted.
_SCENARIOS: dict[str, tuple[str, str, str | None]] = {
    "normal_market_purchase": ("000001", "4", "normal_market_purchase.xml"),
    "normal_market_sale": ("000002", "4", "normal_market_sale.xml"),
    "option_exercise": ("000003", "4", "option_exercise.xml"),
    "tax_withholding": ("000004", "4", "tax_withholding.xml"),
    "grant_award": ("000005", "4", "grant_award.xml"),
    "gift": ("000006", "4", "gift.xml"),
    "derivative_transaction": ("000007", "4", "derivative_transaction.xml"),
    "indirect_ownership": ("000008", "4", "indirect_ownership.xml"),
    "multiple_transactions": ("000009", "4", "multiple_transactions.xml"),
    "footnote_10b5_1": ("000010", "4", "footnote_10b5_1.xml"),
    "form4a_amendment": ("000011", "4/A", "form4a_amendment.xml"),
    "missing_fields": ("000012", "4", "missing_fields.xml"),
    "metadata_says_form3": ("000013", "3", "normal_market_purchase.xml"),  # never fetched
    "body_says_form3": ("000014", "4", "body_says_form3.xml"),
    "missing_ownership_document": ("000015", "4", "missing_ownership_document.xml"),
    "malformed_xml": ("000016", "4", "malformed_xml.xml"),
    "empty_body": ("000017", "4", "empty_body.xml"),
    "html_error_page": ("000018", "4", "html_error_page.htm"),
    "missing_required_fields": ("000019", "4", "missing_required_fields.xml"),
}

#: An accession that genuinely does not exist anywhere in FILER's
#: submissions listing -- for a NOT_FOUND-at-LOCATE test.
UNKNOWN_ACCESSION = f"{FILER_CIK_PADDED}-26-000999"


def accession(scenario: str) -> str:
    suffix, _form, _fixture = _SCENARIOS[scenario]
    return f"{FILER_CIK_PADDED}-26-{suffix}"


def scenario_form(scenario: str) -> str:
    return _SCENARIOS[scenario][1]


def fixture_xml(scenario: str) -> str:
    _suffix, _form, fixture_name = _SCENARIOS[scenario]
    assert fixture_name is not None
    return fixture_text(fixture_name)


def fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def submissions_url() -> str:
    return SUBMISSIONS_URL.format(cik=FILER_CIK)


def document_url(scenario: str, *, document: str = PRIMARY_DOCUMENT) -> str:
    accession_nodash = accession(scenario).replace("-", "")
    return FILING_INDEX_URL.format(cik=FILER_CIK, accession_nodash=accession_nodash, document=document)


def directory_index_url(scenario: str) -> str:
    return document_url(scenario, document="index.json")


def build_submissions_payload() -> dict:
    """One filer's ``filings.recent`` payload covering every scenario
    accession -- the exact parallel-array shape SEC's real submissions.json
    uses (see tests/fixtures/sec_edgar_real_format/MANIFEST.md)."""
    accessions, forms, documents, filing_dates, report_dates = [], [], [], [], []
    for name in _SCENARIOS:
        accessions.append(accession(name))
        forms.append(scenario_form(name))
        documents.append(PRIMARY_DOCUMENT)
        filing_dates.append("2026-03-01")
        report_dates.append("2026-03-01")
    return {
        "cik": FILER_CIK,
        "name": OWNER_NAME,
        "filings": {
            "recent": {
                "accessionNumber": accessions,
                "form": forms,
                "primaryDocument": documents,
                "filingDate": filing_dates,
                "reportDate": report_dates,
            }
        },
    }


def build_directory_payload(scenario: str) -> dict:
    return {"directory": {"item": [{"name": PRIMARY_DOCUMENT, "type": scenario_form(scenario)}]}}


@dataclass
class FakeResult:
    ok: bool
    outcome: FetchOutcome
    _body: str
    error: str = ""

    @property
    def text(self) -> str:
        return self._body

    def json(self) -> object:
        return json.loads(self._body) if self._body else None


def ok(body: str) -> FakeResult:
    return FakeResult(ok=True, outcome=FetchOutcome.OK, _body=body)


def not_found(error: str = "404") -> FakeResult:
    return FakeResult(ok=False, outcome=FetchOutcome.NOT_FOUND, _body="", error=error)


def failed(outcome: FetchOutcome, error: str) -> FakeResult:
    return FakeResult(ok=False, outcome=outcome, _body="", error=error)


@dataclass
class FakeHttpClient:
    """Records every URL requested, in order, and returns a scripted
    response per URL -- so a test can assert exact URLs and exact request
    counts (never inferred). A URL with no registered response is treated
    as a 404, never silently synthesized."""

    responses: dict[str, FakeResult]
    requested_urls: list[str] = field(default_factory=list)

    def get(self, url: str, **kwargs: object) -> FakeResult:
        self.requested_urls.append(url)
        return self.responses.get(url, not_found("no fake response registered for this URL"))


def default_responses() -> dict[str, FakeResult]:
    """Submissions + every scenario's directory index + body, all wired
    up and ready for any single-scenario or multi-scenario test."""
    responses: dict[str, FakeResult] = {submissions_url(): ok(json.dumps(build_submissions_payload()))}
    for name in _SCENARIOS:
        responses[directory_index_url(name)] = ok(json.dumps(build_directory_payload(name)))
        _suffix, _form, fixture_name = _SCENARIOS[name]
        if fixture_name is not None:
            responses[document_url(name)] = ok(fixture_text(fixture_name))
    return responses
