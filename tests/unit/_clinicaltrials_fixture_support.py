"""Shared support for tests driving ``ClinicalTrialsStudyAdapter`` against the
real-format fixtures in ``tests/fixtures/clinicaltrials_v2_real_format/`` (see
that directory's MANIFEST.md for provenance). Not a test module itself -- no
``test_*`` functions live here, so pytest never collects it directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from investment_research.collectors.clinicaltrials import STUDY_DETAIL_URL
from investment_research.schemas.enums import FetchOutcome

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "clinicaltrials_v2_real_format"

NCT_ID = "NCT09990001"
NCT_ID_V2 = NCT_ID  # same id, updated content -- for DocumentStore versioning
NCT_ID_COMPLETED = "NCT09990002"
NCT_ID_TERMINATED = "NCT09990003"
NCT_ID_WITHDRAWN = "NCT09990004"
NCT_ID_MISSING_FIELDS = "NCT09990005"
NCT_ID_PARTIAL_DATES = "NCT09990006"
NCT_ID_MALFORMED = "NCT09990007"
NCT_ID_404 = "NCT00000000"


def fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def study_url(nct_id: str) -> str:
    return STUDY_DETAIL_URL.format(nct_id=nct_id)


@dataclass
class FakeResult:
    """Mirrors ``collectors.http.FetchResult``'s duck-typed surface
    (``ok``/``outcome``/``error``/``.text``/``.json()``) closely enough for
    the adapter, INCLUDING its graceful-``None``-on-malformed-JSON behaviour
    -- unlike ``tests/unit/_sec_fixture_support.py``'s ``FakeResult``, which
    never needed to model that case, this one's ``.json()`` must not raise
    on invalid JSON, since the adapter itself never wraps that call in a
    try/except (it relies on ``FetchResult.json()``'s own contract)."""

    ok: bool
    outcome: FetchOutcome
    _body: str
    error: str = ""

    @property
    def text(self) -> str:
        return self._body

    def json(self) -> object:
        if not self.ok or not self._body:
            return None
        try:
            return json.loads(self._body)
        except json.JSONDecodeError as exc:
            self.outcome = FetchOutcome.ERROR
            self.error = f"invalid JSON: {exc}"
            return None


def ok(body: str) -> FakeResult:
    return FakeResult(ok=True, outcome=FetchOutcome.OK, _body=body)


def not_found(error: str = "404") -> FakeResult:
    return FakeResult(ok=False, outcome=FetchOutcome.NOT_FOUND, _body="", error=error)


def rate_limited(error: str = "429") -> FakeResult:
    return FakeResult(ok=False, outcome=FetchOutcome.RATE_LIMITED, _body="", error=error)


def server_error(error: str = "503") -> FakeResult:
    return FakeResult(ok=False, outcome=FetchOutcome.ERROR, _body="", error=error)


def timed_out(error: str = "timed out") -> FakeResult:
    return FakeResult(ok=False, outcome=FetchOutcome.TIMEOUT, _body="", error=error)


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
    """The happy-path fixture: one normal, recruiting interventional study."""
    return {
        study_url(NCT_ID): ok(fixture_text("study_recruiting_interventional.json")),
    }
