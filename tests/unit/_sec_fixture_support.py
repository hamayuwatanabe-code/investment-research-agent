"""Shared support for tests driving the SEC adapters against the real-format
fixtures in ``tests/fixtures/sec_edgar_real_format/`` (see that directory's
MANIFEST.md for provenance). Not a test module itself -- no ``test_*``
functions live here, so pytest never collects it directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from investment_research.collectors.sec_edgar import FILING_INDEX_URL, SUBMISSIONS_URL
from investment_research.schemas.enums import FetchOutcome

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "sec_edgar_real_format"

CIK = 9999999
ACCESSION = "0009999999-26-000123"
ACCESSION_NODASH = ACCESSION.replace("-", "")
PRIMARY_DOCUMENT = "testco-20251231.htm"


def fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def submissions_url() -> str:
    return SUBMISSIONS_URL.format(cik=CIK)


def document_url(document: str, *, accession_nodash: str = ACCESSION_NODASH) -> str:
    return FILING_INDEX_URL.format(cik=CIK, accession_nodash=accession_nodash, document=document)


def directory_index_url() -> str:
    return document_url("index.json")


def filing_detail_url() -> str:
    return document_url(f"{ACCESSION}-index.htm")


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


@dataclass
class FakeHttpClient:
    """Records every URL requested, in order, and returns a scripted
    response per URL -- so a test can assert exact URLs and exact request
    counts (never inferred). A URL with no registered response is treated as
    a 404, never silently synthesized."""

    responses: dict[str, FakeResult]
    requested_urls: list[str] = field(default_factory=list)

    def get(self, url: str, **kwargs: object) -> FakeResult:
        self.requested_urls.append(url)
        return self.responses.get(url, not_found("no fake response registered for this URL"))


def default_responses() -> dict[str, FakeResult]:
    """The full, self-consistent happy-path fixture set: submissions,
    directory index, filing detail, primary document, and the EX-99.1
    exhibit -- everything ``test_phase3b_offline_acceptance.py`` walks end
    to end."""
    return {
        submissions_url(): ok(fixture_text("submissions_testco.json")),
        directory_index_url(): ok(fixture_text("directory_index_10k.json")),
        filing_detail_url(): ok(fixture_text("filing_detail_10k.htm")),
        document_url(PRIMARY_DOCUMENT): ok(fixture_text("primary_10k_ixbrl.htm")),
        document_url("testco-20251231_ex99-1.htm"): ok(fixture_text("exhibit_99_1_press_release.htm")),
        document_url("testco-20251231_ex10-1.htm"): ok(fixture_text("exhibit_10_1_material_agreement.htm")),
        document_url("testco-20251231_ex31-1.htm"): ok(fixture_text("exhibit_31_1_certification.htm")),
    }
