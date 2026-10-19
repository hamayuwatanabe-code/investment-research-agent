"""Synthetic SEC issuers for the Phase 4.3M coverage tests.

Every issuer, accession, date and sentence here is INVENTED. The builders only
lay out WHICH filings exist (forms, dates, item codes, exhibits) so the
selection / budget / ledger / Action Gate behaviour can be tested across
different issuers and item mixes; none of them copies a real filing's text.
Shared by several test modules (like tests/unit/_literature_fixture_support.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import investment_research.orchestrator.pipeline as pipeline_module
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.http import FetchResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research.escalation import EscalationReport
from investment_research.research.provider import DomainCoverage, NullResearchProvider
from investment_research.schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    FetchOutcome,
    SearchStatus,
)
from investment_research.scoring.completeness import CompletenessResult
from investment_research.scoring.evidence_sufficiency import EvidenceSufficiencyMatrix

TODAY = date(2026, 10, 16)
FILLER = (
    " This synthetic paragraph exists only to give the test document a realistic length so "
    "that body validation treats it as a real filing rather than an index or error page."
)


def ex99(sentences: str) -> str:
    return (
        "<html><body><p>SOUTH CITY, Sept. 16, 2026 (SYNTHETIC WIRE) -- Synthetica Therapeutics "
        f"Inc. today reported. {sentences}</p><p>{FILLER}</p></body></html>"
    )


def primary_8k(body: str = "") -> str:
    return (
        "<html><body><p>FORM 8-K CURRENT REPORT</p><p>Date of Report (Date of earliest event "
        "reported): September 16, 2026</p><p>Synthetica Therapeutics Inc.</p>"
        f"<p>{body}{FILLER}</p></body></html>"
    )


def periodic_body(sentences: str) -> str:
    """A long-ish synthetic 10-K / 10-Q body."""
    para = (
        "The Company is a clinical-stage company. Our operations are subject to a number of "
        "risks that are described below. "
    )
    return f"<html><body><p>{para * 40}{sentences}</p><p>{FILLER}</p></body></html>"


@dataclass
class Filing:
    form: str
    filed: str
    items: str
    n: int
    #: EX-99 press-release bodies (already wrapped), by order
    press_releases: list[str] = field(default_factory=list)
    #: (name, type) listed in the index but never read
    other_exhibits: list[tuple[str, str]] = field(default_factory=list)
    primary_text: str = ""
    primary_name: str = "primary.htm"


class Issuer:
    """One synthetic issuer: its submissions ``filings.recent`` block and the
    document routes behind it."""

    def __init__(self, cik: int, filings: list[Filing] | None = None, older_pages: int = 0) -> None:
        self.cik = cik
        self.filings: list[Filing] = list(filings or [])
        self.older_pages = older_pages

    def accession(self, filing: Filing) -> str:
        return f"{self.cik:010d}-26-{filing.n:06d}"

    def base(self, filing: Filing) -> str:
        nodash = self.accession(filing).replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{nodash}"

    def recent(self) -> dict:
        rows = self.filings
        return {
            "form": [f.form for f in rows], "filingDate": [f.filed for f in rows],
            "items": [f.items for f in rows],
            "accessionNumber": [self.accession(f) for f in rows],
            "primaryDocument": [f.primary_name for f in rows],
            "reportDate": [f.filed for f in rows],
        }

    def routes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in self.filings:
            base = self.base(f)
            listing = [
                {"name": f"{self.accession(f)}-index.htm", "type": "text.gif"},
                {"name": f.primary_name, "type": f.form},
            ]
            for i, body in enumerate(f.press_releases, start=1):
                name = f"ex99-{i}.htm"
                listing.append({"name": name, "type": f"EX-99.{i}"})
                out[f"{base}/{name}"] = body
            for name, doc_type in f.other_exhibits:
                listing.append({"name": name, "type": doc_type})
            out[f"{base}/index.json"] = json.dumps({"directory": {"item": listing}})
            out[f"{base}/{f.primary_name}"] = f.primary_text or primary_8k()
        return out


class FakeSec:
    """Dict-backed fake returning real ``FetchResult`` objects; an
    unregistered URL is a 404. Counts every request."""

    def __init__(self, routes: dict[str, str | FetchOutcome]) -> None:
        self.routes = routes
        self.requested: list[str] = []
        self.attempts_made = 0

    def get(self, url: str) -> FetchResult:
        self.requested.append(url)
        self.attempts_made += 1
        route = self.routes.get(url)
        if route is None:
            return FetchResult(url=url, outcome=FetchOutcome.NOT_FOUND, status=404, error="HTTP 404")
        if isinstance(route, FetchOutcome):
            return FetchResult(url=url, outcome=route, error=str(route))
        return FetchResult(url=url, outcome=FetchOutcome.OK, status=200, body=route.encode())


# --- the three issuer layouts the tests share ---------------------------------------
def alpha_biotech() -> Issuer:
    """Clinical-stage layout: results + strategy 8.01s, two listing notices
    (3.01), three earnings releases (2.02), ten officer/director 8-Ks (5.02)."""
    f = [
        Filing("8-K", "2026-09-23", "7.01,8.01,9.01", 1, [ex99(
            "The SYN-301 trial (NCT88880001) did not meet its primary endpoint of change in "
            "walking distance at week 52."
        )]),
        Filing("8-K", "2026-08-12", "2.02,9.01", 2, [ex99("Second quarter results were reported.")]),
        Filing("8-K", "2026-07-02", "3.01,9.01", 3, primary_text=primary_8k(
            "On July 1, 2026, the Company received a letter from the Nasdaq Listing Qualifications "
            "Department notifying it that the bid price of its common stock had closed below the "
            "minimum bid price requirement for continued listing. "
        )),
        Filing("8-K", "2026-05-20", "2.02,9.01", 4, [ex99("First quarter results were reported.")]),
        Filing("8-K", "2026-03-12", "1.01,3.02,9.01", 5, other_exhibits=[("ex10-1.htm", "EX-10.1")]),
        Filing("8-K", "2026-03-02", "2.02,9.01", 6, [ex99("Year-end results were reported.")]),
        Filing("8-K", "2026-02-10", "3.01", 7, primary_text=primary_8k(
            "On February 9, 2026, the Company received a letter from Nasdaq notifying it that it was "
            "not in compliance with the minimum stockholders' equity requirement for continued listing. "
        )),
        Filing("8-K", "2026-01-15", "8.01", 8, [ex99("The Company presented at a conference.")]),
    ]
    for k in range(10):
        f.append(Filing("8-K", f"2026-0{(k % 6) + 3}-{10 + k:02d}", "5.02,9.01", 30 + k))
    f.sort(key=lambda x: (x.filed, x.n), reverse=True)
    return Issuer(1721484, f)


def beta_device() -> Issuer:
    """Financing-only layout, no clinical content: agreements and notes."""
    f = [
        Filing("8-K", "2026-09-02", "1.01,3.02,9.01", 1, other_exhibits=[("ex10-1.htm", "EX-10.1")],
               primary_text=primary_8k(
                   "On September 1, 2026, the Company entered into a securities purchase agreement "
                   "for the sale of common stock. "
               )),
        Filing("8-K", "2026-06-20", "2.03,9.01", 2, primary_text=primary_8k(
            "On June 19, 2026, the Company entered into a term loan. "
        )),
        Filing("8-K", "2026-04-04", "5.07", 3),
    ]
    return Issuer(1802001, f)


def gamma_heavy(n_material: int = 20) -> Issuer:
    """A heavy filer: more MATERIAL 8-Ks than one run may read, plus filings
    with item codes the system cannot interpret."""
    f = [
        Filing("8-K", f"2026-09-{(i % 27) + 1:02d}", "8.01", 100 + i, [ex99(f"Update {i}.")])
        for i in range(n_material)
    ]
    f.append(Filing("8-K", "2026-09-28", "1.05", 300))
    f.append(Filing("8-K", "2026-09-27", "2.01,9.01", 301))
    f.sort(key=lambda x: (x.filed, x.n), reverse=True)
    return Issuer(1900003, f)


# --- noise-free Pipeline harness (copied convention, see other integration tests) -------
class AvailableNullResearchProvider(NullResearchProvider):
    def available(self):
        return True, ""


def install_noise_free_pipeline_harness(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_module, "assess_completeness",
        lambda **_k: CompletenessResult(coverage={
            d: DomainCoverage(domain=d, status=SearchStatus.SEARCHED)
            for d in REQUIRED_RESEARCH_DOMAINS
        }),
    )
    monkeypatch.setattr(
        pipeline_module, "assess_evidence_sufficiency",
        lambda **_k: EvidenceSufficiencyMatrix(decision_grade_fact_total=1),
    )
    monkeypatch.setattr(
        pipeline_module, "escalate",
        lambda facts, _p, *, company, **_k: (list(facts), EscalationReport()),
    )
    real_run_agent = Pipeline._run_agent
    noisy = {"competitive", "microstructure", "kill_agent"}

    def patched(self, agent, guard, result, **kwargs):
        before_status, before_len = result.context.status, len(result.failures)
        output = real_run_agent(self, agent, guard, result, **kwargs)
        if agent.agent_id in noisy:
            del result.failures[before_len:]
            result.context.status = before_status
        return output

    monkeypatch.setattr(Pipeline, "_run_agent", patched)


def demobio_base() -> tuple[str, float, tuple, CollectionResult]:
    fixtures = FixtureCollector(Path(__file__).resolve().parent.parent.parent / "data" / "fixtures")
    metadata = fixtures.metadata("DEMOBIO")
    company = metadata["company_name"]
    return company, metadata["price"], tuple(metadata.get("aliases", ())), fixtures.collect("DEMOBIO", company)


def pipeline_for(repo) -> Pipeline:
    return Pipeline(repo, NullSearchProvider(), today=TODAY, research=AvailableNullResearchProvider())
