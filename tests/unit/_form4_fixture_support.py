"""Shared support for tests driving Form4Adapter against the real-format
fixtures in ``tests/fixtures/form4_real_format/`` (see that directory's
MANIFEST.md for provenance). Not a test module itself -- no ``test_*``
functions live here, so pytest never collects it directly.

Phase 3E.1: issuer-driven discovery means every scenario accession now
lives under the ISSUER's own CIK (never a separately-supplied "filer"
CIK) -- SEC's own submissions.json for a company includes the Section 16
(Form 3/4/5) filings made against it, and the same accession's documents
are reachable via that issuer CIK's own Archives directory path.

``submissions.json``/``index.json``/``company_tickers.json`` response
bodies are built here programmatically (rather than as dozens of near-
identical static files) -- the SHAPE still matches SEC EDGAR's real,
documented schema exactly (the same parallel-array ``filings.recent``
structure ``tests/fixtures/sec_edgar_real_format`` already models as
static files), only the construction method differs. The interesting
content -- the ownership XML bodies themselves -- are real static fixture
files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from investment_research.collectors.sec_edgar import (
    FILING_INDEX_URL,
    SUBMISSIONS_PAGE_URL,
    SUBMISSIONS_URL,
    TICKER_MAP_URL,
    cik_for_archives,
)
from investment_research.schemas.enums import FetchOutcome

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "form4_real_format"

#: Wholly fictional issuer/reporting-owner identifiers -- never a real
#: company, ticker, CIK, or accession number (Phase 3E/3E.1 requirement
#: on fixture neutrality).
ISSUER_CIK_INT = 1112223
ISSUER_CIK = "0001112223"  # matches every fixture XML's own <issuerCik>
ISSUER_NAME = "Sample Biotech Holdings, Inc."
ISSUER_TICKER = "SAMPB"
OWNER_NAME = "Sample Reporting Person"
OWNER_CIK_PADDED = "0005556667"  # matches every fixture XML's own <rptOwnerCik>

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
    "form4a_amendment": ("000011", "4/A", "form4a_amendment.xml"),  # unresolved: no accession referenced
    "missing_fields": ("000012", "4", "missing_fields.xml"),
    "metadata_says_form3": ("000013", "3", "normal_market_purchase.xml"),  # never fetched
    "body_says_form3": ("000014", "4", "body_says_form3.xml"),
    "missing_ownership_document": ("000015", "4", "missing_ownership_document.xml"),
    "malformed_xml": ("000016", "4", "malformed_xml.xml"),
    "empty_body": ("000017", "4", "empty_body.xml"),
    "html_error_page": ("000018", "4", "html_error_page.htm"),
    "missing_required_fields": ("000019", "4", "missing_required_fields.xml"),
    "cik_mismatch": ("000020", "4", "cik_mismatch.xml"),
    "reconciled_amendment": ("000021", "4/A", "reconciled_amendment.xml"),  # references normal_market_purchase
}

#: A subset used for the "issuer-driven discovery finds every real Form
#: 4/4-A candidate" happy-path acceptance test -- every scenario whose own
#: XML is genuinely fetchable AND shape-valid AND issuer-CIK-matching.
CLEANLY_ACQUIRABLE_SCENARIOS: tuple[str, ...] = (
    "normal_market_purchase", "normal_market_sale", "option_exercise", "tax_withholding",
    "grant_award", "gift", "derivative_transaction", "indirect_ownership", "multiple_transactions",
    "footnote_10b5_1", "form4a_amendment", "missing_fields", "reconciled_amendment",
)

#: An accession that genuinely does not exist anywhere in the issuer's
#: submissions listing -- for a NOT_FOUND/ZERO_RESULTS-style test.
UNKNOWN_ACCESSION = f"{ISSUER_CIK}-26-000999"

#: The continuation-page ("filings.files") scenario -- an OLDER filing
#: that appears ONLY on a paginated submissions file, never in
#: filings.recent (Phase 3E.1 requirement 1).
_CONTINUATION_SUFFIX = "000098"
_CONTINUATION_FIXTURE = "older_filing_on_continuation_page.xml"
CONTINUATION_PAGE_NAME = f"CIK{ISSUER_CIK}-submissions-001.json"


def accession(scenario: str) -> str:
    suffix, _form, _fixture = _SCENARIOS[scenario]
    return f"{ISSUER_CIK}-26-{suffix}"


def continuation_accession() -> str:
    return f"{ISSUER_CIK}-26-{_CONTINUATION_SUFFIX}"


def scenario_form(scenario: str) -> str:
    return _SCENARIOS[scenario][1]


def fixture_xml(scenario: str) -> str:
    _suffix, _form, fixture_name = _SCENARIOS[scenario]
    assert fixture_name is not None
    return fixture_text(fixture_name)


def fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def submissions_url() -> str:
    return SUBMISSIONS_URL.format(cik=ISSUER_CIK_INT)


def continuation_page_url() -> str:
    return SUBMISSIONS_PAGE_URL.format(name=CONTINUATION_PAGE_NAME)


def ticker_map_url() -> str:
    return TICKER_MAP_URL


def document_url(scenario: str, *, document: str = PRIMARY_DOCUMENT) -> str:
    accession_nodash = accession(scenario).replace("-", "")
    return FILING_INDEX_URL.format(
        cik=cik_for_archives(ISSUER_CIK_INT), accession_nodash=accession_nodash, document=document,
    )


def continuation_document_url() -> str:
    accession_nodash = continuation_accession().replace("-", "")
    return FILING_INDEX_URL.format(
        cik=cik_for_archives(ISSUER_CIK_INT), accession_nodash=accession_nodash, document=PRIMARY_DOCUMENT,
    )


def directory_index_url(scenario: str) -> str:
    return document_url(scenario, document="index.json")


def continuation_directory_index_url() -> str:
    return continuation_document_url().rsplit("/", 1)[0] + "/index.json"


def build_submissions_payload(*, with_continuation_pointer: bool = False) -> dict:
    """The issuer's own ``filings.recent`` payload covering every scenario
    accession -- the exact parallel-array shape SEC's real submissions.json
    uses (see tests/fixtures/sec_edgar_real_format/MANIFEST.md)."""
    accessions, forms, documents, filing_dates, report_dates = [], [], [], [], []
    for name in _SCENARIOS:
        accessions.append(accession(name))
        forms.append(scenario_form(name))
        documents.append(PRIMARY_DOCUMENT)
        filing_dates.append("2026-03-01")
        report_dates.append("2026-03-01")
    payload = {
        "cik": ISSUER_CIK_INT,
        "name": ISSUER_NAME,
        "tickers": [ISSUER_TICKER],
        "filings": {
            "recent": {
                "accessionNumber": accessions,
                "form": forms,
                "primaryDocument": documents,
                "filingDate": filing_dates,
                "reportDate": report_dates,
            },
            "files": (
                [{"name": CONTINUATION_PAGE_NAME, "filingCount": 1, "filingFrom": "2020-01-01", "filingTo": "2020-12-31"}]
                if with_continuation_pointer else []
            ),
        },
    }
    return payload


def build_continuation_page_payload() -> dict:
    """A continuation page's own body has NO ``filings``/``recent``
    wrapper -- it is the parallel-array object directly (Phase 3E.1
    requirement 1)."""
    return {
        "accessionNumber": [continuation_accession()],
        "form": ["4"],
        "primaryDocument": [PRIMARY_DOCUMENT],
        "filingDate": ["2020-06-15"],
        "reportDate": ["2020-06-15"],
    }


def build_ticker_map_payload() -> dict:
    return {"0": {"cik_str": ISSUER_CIK_INT, "ticker": ISSUER_TICKER, "title": ISSUER_NAME}}


def build_directory_payload(scenario: str) -> dict:
    return {"directory": {"item": [{"name": PRIMARY_DOCUMENT, "type": scenario_form(scenario)}]}}


def build_continuation_directory_payload() -> dict:
    return {"directory": {"item": [{"name": PRIMARY_DOCUMENT, "type": "4"}]}}


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


def default_responses(*, with_continuation_pointer: bool = False) -> dict[str, FakeResult]:
    """Submissions + ticker map + every scenario's directory index + body,
    all wired up and ready for any single-scenario or multi-scenario
    test."""
    responses: dict[str, FakeResult] = {
        submissions_url(): ok(json.dumps(build_submissions_payload(with_continuation_pointer=with_continuation_pointer))),
        ticker_map_url(): ok(json.dumps(build_ticker_map_payload())),
    }
    if with_continuation_pointer:
        responses[continuation_page_url()] = ok(json.dumps(build_continuation_page_payload()))
        responses[continuation_directory_index_url()] = ok(json.dumps(build_continuation_directory_payload()))
        responses[continuation_document_url()] = ok(fixture_text(_CONTINUATION_FIXTURE))
    for name in _SCENARIOS:
        responses[directory_index_url(name)] = ok(json.dumps(build_directory_payload(name)))
        _suffix, _form, fixture_name = _SCENARIOS[name]
        if fixture_name is not None:
            responses[document_url(name)] = ok(fixture_text(fixture_name))
    return responses
