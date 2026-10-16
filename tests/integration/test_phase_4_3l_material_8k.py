"""Phase 4.3L: material 8-K bodies (and EX-99 exhibits) become evidence, and
an issuer's own primary-endpoint failure reaches the Action Gate as a
blocking K5 -- never a "different programme" K1.

Every document, company, accession and trial identifier here is SYNTHETIC
("Synthetica Therapeutics", CIK 9900001, NCT88880001...). No real issuer's
filing text is reproduced or invented. Transports are fakes: a dict-backed
fake returning real ``FetchResult`` objects, or the real budgeted
``AllowlistedHttpClient`` with only ``HTTPSHandler.https_open`` faked. No
socket is ever opened.
"""

from __future__ import annotations

import io
import socket
import urllib.request
import urllib.response
from datetime import date
from email.message import Message
from types import SimpleNamespace

import pytest

from investment_research import cli
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.http import FetchResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.config import Settings
from investment_research.orchestrator import pipeline as pipeline_module
from investment_research.orchestrator import resume as resume_module
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.reporting.report import render_report
from investment_research.research import sec_material_8k as m8k
from investment_research.schemas.enums import (
    ContentKind,
    DocumentAuthority,
    FactCategory,
    FetchOutcome,
    KillCategory,
    KillConfirmation,
    KillLevel,
    Provenance,
    ResearchStatus,
    SourceTier,
)
from investment_research.schemas.fact import RawFact, Source
from investment_research.scoring.program_resolution import is_superseding_disclosure
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.integration

TODAY = date(2026, 10, 16)
TICKER = "SYNTH"
CIK = 9900001
ACC = "0009900001-26-000123"
ACC_ND = ACC.replace("-", "")
TRIAL = "NCT88880001"  # synthetic
OLD_TRIAL = "NCT88880000"  # synthetic
BASE = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{ACC_ND}"
INDEX_URL = f"{BASE}/index.json"
PRIMARY = "synth-8k.htm"
EX991 = "synth-ex99-1.htm"
FILLER = (
    " This synthetic paragraph exists only to give the test document a realistic length so "
    "that body validation treats it as a real filing rather than an index or error page."
)

PRIMARY_8K_BODY = (
    "<html><body><p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 8-K CURRENT REPORT</p>"
    "<p>Date of Report (Date of earliest event reported): September 16, 2026</p>"
    "<p>Synthetica Therapeutics Inc.</p><p>Item 8.01 Other Events.</p>"
    "<p>On September 16, 2026, the Company issued a press release, attached as Exhibit 99.1."
    + FILLER + "</p></body></html>"
)


def _ex99(sentences: str) -> str:
    return (
        "<html><body><p>SOUTH CITY, Sept. 16, 2026 (SYNTHETIC WIRE) -- Synthetica "
        f"Therapeutics Inc. today reported topline results. {sentences}</p>"
        "<p>Forward-Looking Statements. This release contains forward-looking statements, "
        "including whether future trials will meet their primary endpoints and whether the "
        "Company may discontinue any program." + FILLER + "</p></body></html>"
    )


FAILURE_EX99 = _ex99(
    f"The SYN-301 trial ({TRIAL}) did not meet its primary endpoint of change in walking "
    "distance at week 52. The Company has initiated a strategic review and is exploring "
    "strategic alternatives."
)


def _index(*entries: tuple[str, str]) -> str:
    import json

    items = [{"name": f"{ACC}-index.htm", "type": "text.gif"}]
    items += [{"name": name, "type": doc_type} for name, doc_type in entries]
    return json.dumps({"directory": {"item": items}})


def _recent(*rows: tuple[str, str, str, str, str]) -> dict:
    """rows: (form, filing_date, items, accession, primary_document)"""
    return {
        "form": [r[0] for r in rows],
        "filingDate": [r[1] for r in rows],
        "items": [r[2] for r in rows],
        "accessionNumber": [r[3] for r in rows],
        "primaryDocument": [r[4] for r in rows],
    }


ONE_8K = _recent(("8-K", "2026-09-23", "7.01,8.01,9.01", ACC, PRIMARY))


class FakeSec:
    """Dict-backed fake returning real ``FetchResult`` objects; an
    unregistered URL is a 404, never a synthesized body."""

    def __init__(self, routes: dict[str, str | FetchOutcome]) -> None:
        self.routes = routes
        self.requested: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.requested.append(url)
        route = self.routes.get(url)
        if route is None:
            return FetchResult(url=url, outcome=FetchOutcome.NOT_FOUND, status=404, error="HTTP 404")
        if isinstance(route, FetchOutcome):
            return FetchResult(url=url, outcome=route, error=str(route))
        return FetchResult(url=url, outcome=FetchOutcome.OK, status=200, body=route.encode())


def _routes(ex99_body: str = FAILURE_EX99, **overrides) -> dict:
    routes = {
        INDEX_URL: _index((PRIMARY, "8-K"), (EX991, "EX-99.1")),
        f"{BASE}/{PRIMARY}": PRIMARY_8K_BODY,
        f"{BASE}/{EX991}": ex99_body,
    }
    routes.update(overrides)
    return routes


def _collect(routes=None, recent=ONE_8K, http="default", **kwargs) -> tuple[CollectionResult, FakeSec]:
    fake = FakeSec(routes if routes is not None else _routes())
    result = m8k.collect_material_8k_disclosures(
        TICKER, cik=CIK, recent=recent, http=fake if http == "default" else http,
        today=TODAY, **kwargs,
    )
    return result, fake


# =============================================================================
# Selection: items choose what to read; nothing here is evidence
# =============================================================================
def test_selection_uses_items_window_and_cap_newest_first():
    rows = [
        ("8-K", "2026-09-23", "7.01,8.01,9.01", "0009900001-26-000001", "a.htm"),
        ("8-K", "2026-08-12", "2.02,9.01", "0009900001-26-000002", "b.htm"),  # earnings only
        ("8-K/A", "2026-07-01", "8.01", "0009900001-26-000003", "c.htm"),
        ("8-K", "2026-06-01", "", "0009900001-26-000004", "d.htm"),  # items unknown: kept
        ("10-Q", "2026-08-12", "", "0009900001-26-000005", "e.htm"),
        ("8-K", "2025-09-01", "8.01", "0009900001-25-000006", "f.htm"),  # outside 365 days
        ("8-K", "not-a-date", "8.01", "0009900001-26-000007", "g.htm"),  # undatable: kept
    ]
    rows += [("8-K", f"2026-05-{d:02d}", "7.01", f"0009900001-26-0001{d:02d}", "h.htm") for d in (1, 2, 3, 4)]
    selected, beyond, start = m8k.select_material_8k_candidates(_recent(*rows), today=TODAY)

    assert start == date(2025, 10, 16)
    chosen = [c.accession for c in selected + beyond]
    assert "0009900001-26-000002" not in chosen and "0009900001-26-000005" not in chosen
    assert "0009900001-25-000006" not in chosen
    assert {"0009900001-26-000004", "0009900001-26-000007"} <= set(chosen)
    assert len(selected) == m8k.MAX_FILINGS
    assert selected[0].accession == "0009900001-26-000001"  # newest dated first
    dated = [c.filing_date for c in selected + beyond if c.filing_date != "UNKNOWN"]
    assert dated == sorted(dated, reverse=True)
    assert (selected + beyond)[-1].accession == "0009900001-26-000007"  # undatable last
    assert len(beyond) == len(chosen) - m8k.MAX_FILINGS


def test_budget_constants_are_consistent():
    assert m8k.GETS_PER_FILING == 2 + m8k.MAX_EXHIBITS_PER_FILING
    assert m8k.MAX_OUTBOUND_GETS == m8k.MAX_FILINGS * m8k.GETS_PER_FILING
    assert pipeline_module._MATERIAL_8K_COLLECTOR_LABEL == m8k.MATERIAL_8K_COLLECTOR
    assert pipeline_module._UNVERIFIED_MATERIAL_8K_PREFIX == m8k.UNVERIFIED_PREFIX


# =============================================================================
# Extraction: only an unqualified, past-tense PRIMARY-endpoint result counts
# =============================================================================
@pytest.mark.parametrize(
    ("sentence", "kinds"),
    [
        ("The SYN-301 trial did not meet its primary endpoint.", ["PRIMARY_ENDPOINT_NOT_MET"]),
        ("Although the trial did not meet its primary endpoint, we believe the data are encouraging.",
         ["PRIMARY_ENDPOINT_NOT_MET"]),
        ("Topline results showed the primary endpoint was not met.", ["PRIMARY_ENDPOINT_NOT_MET"]),
        ("The trial met its primary endpoint.", ["PRIMARY_ENDPOINT_MET"]),
        # future / conditional / hypothetical: never a result
        ("There can be no assurance that the trial will meet its primary endpoint.", []),
        ("If the trial fails to meet its primary endpoint, we may discontinue the program.", []),
        ("We expect topline results in the third quarter.", []),
        # secondary endpoints are not the primary endpoint
        ("The trial did not meet its secondary endpoints.", []),
        ("The trial did not meet its primary endpoint, but it met all key secondary endpoints.",
         ["PRIMARY_ENDPOINT_NOT_MET"]),
        # contradictory / unclassifiable wording: surfaced, never settled
        ("The study met its primary endpoint in Cohort A but did not meet the primary endpoint in Cohort B.",
         ["AMBIGUOUS_PRIMARY_ENDPOINT"]),
        ("The primary endpoint did not show a statistically significant improvement.",
         ["AMBIGUOUS_PRIMARY_ENDPOINT"]),
        # descriptions and forward-looking boilerplate
        ("The primary endpoint is change from baseline at month 12.", []),
        ("Forward-looking statements include whether the trial will meet its primary endpoint.", []),
        ("The Company has initiated a strategic review to explore all options.", ["STRATEGIC_CHANGE"]),
    ],
)
def test_statement_classification(sentence, kinds):
    assert [s.kind for s in m8k.extract_statements(sentence)] == kinds


def test_body_event_date_is_only_what_the_body_states():
    assert m8k.body_event_date(PRIMARY_8K_BODY, is_primary_8k=True) == "2026-09-16"
    assert m8k.body_event_date(FAILURE_EX99, is_primary_8k=False) == "2026-09-16"
    assert m8k.body_event_date("A body with no dateline at all.", is_primary_8k=False) == "UNKNOWN"


# =============================================================================
# Collection over a fake transport
# =============================================================================
def test_failure_and_strategy_become_company_claim_facts_with_full_provenance():
    result, fake = _collect()
    assert result.errors == [] and result.outcome is FetchOutcome.OK
    assert fake.requested == [INDEX_URL, f"{BASE}/{PRIMARY}", f"{BASE}/{EX991}"]

    by_value = {f.value: f for f in result.raw_facts}
    failure = by_value["primary endpoint was not met (issuer 8-K disclosure)"]
    strategy = by_value["strategic review or programme change (issuer 8-K disclosure)"]
    assert failure.claim == (
        f"The SYN-301 trial ({TRIAL}) did not meet its primary endpoint of change in walking "
        "distance at week 52."
    )
    assert failure.category is FactCategory.CLINICAL and strategy.category is FactCategory.OTHER
    for fact in (failure, strategy):
        assert fact.company_claim is True
        assert fact.content_kind is ContentKind.FULL_DOCUMENT
        assert fact.source_authority is DocumentAuthority.COMPANY_IR
        assert fact.source.url == f"{BASE}/{EX991}"
        assert fact.source.accession == ACC
        assert EX991 in fact.source.title and "EX-99.1" in fact.source.title
        assert fact.source.filing_date == "2026-09-23"
        assert fact.source.published_date == "2026-09-23"
        assert fact.source.event_date == "2026-09-16"
        assert fact.source.tier is SourceTier.TIER_1
        assert is_superseding_disclosure(SimpleNamespace(claim=fact.claim, value=fact.value))
    # forward-looking boilerplate in the same release produced nothing
    assert not any("future trials" in f.claim for f in result.raw_facts)


@pytest.mark.parametrize(
    ("routes", "expected"),
    [
        (_routes(**{f"{BASE}/{EX991}": FetchOutcome.TIMEOUT}), "was not read: fetch failed"),
        (_routes(**{f"{BASE}/{EX991}": "<p>too short</p>"}), "failed body validation"),
        (_routes(**{INDEX_URL: FetchOutcome.BLOCKED}), "filing index could not be read"),
        (_routes(**{INDEX_URL: _index((EX991, "EX-99.1"))}), "does not list the primary document"),
    ],
    ids=["exhibit-timeout", "exhibit-invalid-body", "index-blocked", "primary-not-listed"],
)
def test_a_failed_or_invalid_read_is_unverified_never_a_confirmed_failure(routes, expected):
    result, _ = _collect(routes)
    assert result.outcome is FetchOutcome.ERROR
    assert any(e.startswith(m8k.UNVERIFIED_PREFIX) and expected in e and ACC in e for e in result.errors)
    assert not any(f.source.url == f"{BASE}/{EX991}" for f in result.raw_facts)


def test_more_exhibits_than_the_cap_marks_the_filing_partially_unverified():
    extra = {f"{BASE}/synth-ex99-{n}.htm": FAILURE_EX99 for n in (2, 3)}
    routes = _routes(**{
        INDEX_URL: _index((PRIMARY, "8-K"), (EX991, "EX-99.1"),
                          ("synth-ex99-2.htm", "EX-99.2"), ("synth-ex99-3.htm", "EX-99.3")),
        **extra,
    })
    result, fake = _collect(routes)
    assert any("further EX-99 exhibit(s)" in e for e in result.errors)
    assert f"{BASE}/synth-ex99-3.htm" not in fake.requested


def test_contradictory_primary_endpoint_statements_are_both_reported_and_unverified():
    routes = _routes(**{
        INDEX_URL: _index((PRIMARY, "8-K"), (EX991, "EX-99.1"), ("synth-ex99-2.htm", "EX-99.2")),
        f"{BASE}/synth-ex99-2.htm": _ex99("The SYN-301 trial met its primary endpoint."),
    })
    result, _ = _collect(routes)
    values = {f.value for f in result.raw_facts}
    assert "primary endpoint was not met (issuer 8-K disclosure)" in values
    assert "primary endpoint was met (issuer 8-K disclosure)" in values
    assert any("contradictory primary-endpoint statements" in e for e in result.errors)


def test_ambiguous_wording_is_unverified_and_produces_no_failure_fact():
    routes = _routes(_ex99("The primary endpoint did not show a statistically significant improvement."))
    result, _ = _collect(routes)
    assert not any("not met" in str(f.value) for f in result.raw_facts)
    assert any("cannot be classified unambiguously" in e for e in result.errors)


def test_no_client_means_zero_requests_and_every_candidate_unverified():
    result, fake = _collect(http=None, unavailable_reason="no valid SEC User-Agent")
    assert fake.requested == []
    assert result.raw_facts == []
    assert any(ACC in e and "no valid SEC User-Agent" in e for e in result.errors)


def test_an_unobtained_filing_list_makes_the_whole_scope_unverified():
    result = m8k.collect_material_8k_disclosures(
        TICKER, cik=None, recent=None, http=FakeSec({}), today=TODAY,
    )
    assert result.outcome is FetchOutcome.ERROR
    assert result.errors and result.errors[0].startswith(m8k.UNVERIFIED_PREFIX)


def test_candidates_beyond_the_cap_are_unverified_not_dropped():
    rows = [("8-K", f"2026-09-{d:02d}", "8.01", f"0009900001-26-0002{d:02d}", "x.htm") for d in range(1, 9)]
    result, fake = _collect(routes={}, recent=_recent(*rows))
    beyond = [e for e in result.errors if "beyond the per-run cap" in e]
    assert len(beyond) == 8 - m8k.MAX_FILINGS
    assert all("0009900001-26-00020" in e for e in beyond)  # the two OLDEST


# =============================================================================
# The real budgeted client: actual GETs (retries + redirects) never exceed 24
# =============================================================================
@pytest.fixture
def no_socket(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("a real socket connection was attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def _wire_response(url: str, status: int, body: str = "", location: str = ""):
    headers = Message()
    if location:
        headers["Location"] = location
    response = urllib.response.addinfourl(io.BytesIO(body.encode()), headers, url, status)
    response.msg = "OK" if status == 200 else "Error"
    return response


def test_sec_document_client_configuration():
    client = cli.build_sec_document_client("Synthetic Research ops@example.org")
    assert client.allowed_hosts == frozenset({"www.sec.gov"})
    assert client.max_requests == m8k.MAX_OUTBOUND_GETS == 24
    assert client.max_retries == 1
    assert client.rate_limit_rps == 2.0
    assert client.out_dir is None


def test_retries_and_redirects_stop_at_the_actual_get_budget(monkeypatch, no_socket):
    monkeypatch.setattr(cli, "SEC_DOCUMENT_RATE_LIMIT_RPS", 1000.0)  # pacing is tested in 4.3K
    wire: list[str] = []

    def fake_https_open(_handler, req):
        wire.append(req.full_url)
        if req.full_url.endswith("/index.json") and "redirected" not in req.full_url:
            # every index first redirects within www.sec.gov, then errors
            return _wire_response(req.full_url, 302, location=req.full_url + "?redirected=1")
        return _wire_response(req.full_url, 500)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_https_open)
    rows = [("8-K", f"2026-09-{d:02d}", "8.01", f"0009900001-26-0003{d:02d}", "x.htm") for d in range(1, 7)]
    client = cli.build_sec_document_client("Synthetic Research ops@example.org")
    result = m8k.collect_material_8k_disclosures(
        TICKER, cik=CIK, recent=_recent(*rows), http=client, today=TODAY,
    )
    assert len(wire) == client.attempts_made <= m8k.MAX_OUTBOUND_GETS
    assert any("budget of 24 outbound SEC GETs was exhausted" in e or "fetch failed" in e
               for e in result.errors)
    assert len([e for e in result.errors if e.startswith(m8k.UNVERIFIED_PREFIX)]) == 6
    assert result.raw_facts == []


def test_an_off_host_redirect_is_refused_before_connecting(monkeypatch, no_socket):
    monkeypatch.setattr(cli, "SEC_DOCUMENT_RATE_LIMIT_RPS", 1000.0)
    wire: list[str] = []

    def fake_https_open(_handler, req):
        wire.append(req.full_url)
        return _wire_response(req.full_url, 302, location="https://evil.example.org/index.json")

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_https_open)
    client = cli.build_sec_document_client("Synthetic Research ops@example.org")
    result = m8k.collect_material_8k_disclosures(TICKER, cik=CIK, recent=ONE_8K, http=client, today=TODAY)
    assert all("evil.example.org" not in u for u in wire)
    assert any("filing index could not be read" in e for e in result.errors)


# =============================================================================
# CLI wiring: no request without --live or a real User-Agent
# =============================================================================
def test_cli_offline_or_placeholder_user_agent_makes_no_request(monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("no SEC document client may be built")

    monkeypatch.setattr(cli, "build_sec_document_client", explode)
    sec = SimpleNamespace(resolved_cik=CIK, recent_filings=ONE_8K)
    offline = cli.collect_material_8k(TICKER, sec, settings=Settings(), live=False, today=TODAY)
    placeholder = cli.collect_material_8k(
        TICKER, sec, settings=Settings(sec_user_agent="investment-research-agent contact@example.com"),
        live=True, today=TODAY,
    )
    assert any("offline run" in e for e in offline.errors)
    assert any("no valid SEC User-Agent" in e for e in placeholder.errors)


# =============================================================================
# Pipeline: the 8-K failure reaches the Action Gate as a blocking K5
# =============================================================================
def _registry_and_marker() -> CollectionResult:
    def raw(claim, *, value="UNKNOWN", event="2026-06-30", url, company=False):
        return RawFact(
            ticker=TICKER, category=FactCategory.CLINICAL, claim=claim, value=value,
            source=Source(
                source_id=f"src_{abs(hash(url)) % 10**8}", url=url, title="synthetic",
                tier=SourceTier.TIER_1, publisher="SYNTHETIC", published_date=event,
                event_date=event,
            ),
            company_claim=company, collector="synthetic_registry_fixture",
        )

    facts = [
        raw(f"{TRIAL} overall status is ACTIVE_NOT_RECRUITING", value="ACTIVE_NOT_RECRUITING",
            url=f"https://clinicaltrials.gov/study/{TRIAL}"),
        raw(f"{OLD_TRIAL} overall status is COMPLETED", value="COMPLETED", event="2019-12-31",
            url=f"https://clinicaltrials.gov/study/{OLD_TRIAL}"),
        # a stale description of the OLD trial as the lead programme
        raw(f"The company describes {OLD_TRIAL} as its lead programme.", event="2019-01-01",
            url="https://www.sec.gov/synthetic-2019-10k", company=True),
    ]
    return CollectionResult(
        collector="synthetic_registry_fixture", raw_facts=facts,
        sources=[f.source for f in facts], provenance=Provenance.FIXTURE,
    )


def _run(results, *, repo=None, run_id=None, resume=False):
    repo = repo or Repository(open_db(":memory:"))
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    return pipeline.run(TICKER, "Synthetica Therapeutics Inc.", results, run_id=run_id, resume=resume), repo


def _failure_findings(result):
    """The verdict's CLINICAL assessment, the Kill Agent's own published
    trial_failed findings (detail text included), and its KILL payload."""
    clinical = next(a for a in result.verdict.kill_gate.assessments
                    if a.category == KillCategory.CLINICAL_KILL)
    payload = result.bus.channels[Channel.KILL].payload
    failed = [f for f in payload["findings"] if f["title"] == "Trial missed its primary endpoint"]
    return clinical, failed, payload


def test_pipeline_8k_failure_is_a_blocking_k5_not_a_different_programme_k1():
    material, _ = _collect()
    result, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])

    clinical, failed, payload = _failure_findings(result)
    assert failed, "the 8-K failure fact must reach the Kill Gate"
    assert all(f["level"] == str(KillLevel.K5) for f in failed)
    assert not any("DIFFERENT PROGRAMME" in f["detail"] for f in failed)
    assert any("PROGRAM_RELEVANCE_UNRESOLVED" in f["detail"] for f in failed)
    # the stale 2019 lead description of the OLD trial no longer decides
    assert payload["program_resolved"] == "UNKNOWN"
    assert payload["program_relevance_unresolved"] is True
    assert clinical.level is KillLevel.K5
    assert clinical.confirmation is KillConfirmation.PROVISIONAL
    assert any(str(KillCategory.CLINICAL_KILL) in p and "K5" in p
               for p in result.evidence_sufficiency.provisional_kill_findings)
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION

    report = render_report(result)
    assert f"{BASE}/{EX991}" in report  # the exhibit is cited as the source
    assert "event_date=2026-09-16" in report and "published=2026-09-23" in report


def test_pipeline_unverified_material_8k_blocks_with_its_reason():
    material, _ = _collect(_routes(**{f"{BASE}/{EX991}": FetchOutcome.TIMEOUT}))
    result, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    reasons = [r for r in result.verdict.blocking_verification_required
               if r.startswith(m8k.UNVERIFIED_PREFIX)]
    assert reasons and ACC in reasons[0]
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION


def test_resume_judges_the_restored_collections_8k_record_not_this_invocations(monkeypatch):
    monkeypatch.setattr(resume_module, "is_stale", lambda _f, _t: False)
    unread, _ = _collect(_routes(**{f"{BASE}/{EX991}": FetchOutcome.TIMEOUT}))
    first, repo = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), unread],
                       run_id="m8k-resume")
    assert any(r.startswith(m8k.UNVERIFIED_PREFIX) for r in first.verdict.blocking_verification_required)

    read_ok, _ = _collect()  # this invocation's fresh read is clean...
    second, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), read_ok],
                     repo=repo, run_id="m8k-resume", resume=True)
    assert second.resume_plan.resume_from_index > 0
    # ...but the restored evidence is the first invocation's, which was not read
    assert any(r.startswith(m8k.UNVERIFIED_PREFIX) for r in second.verdict.blocking_verification_required)
    assert second.verdict.action is None


def test_resume_of_a_pre_4_3l_collection_with_no_8k_record_is_unverified(monkeypatch):
    monkeypatch.setattr(resume_module, "is_stale", lambda _f, _t: False)
    first, repo = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar")], run_id="m8k-legacy")
    assert not any(r.startswith(m8k.UNVERIFIED_PREFIX) for r in first.verdict.blocking_verification_required)

    second, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar")],
                     repo=repo, run_id="m8k-legacy", resume=True)
    assert any("predates Phase 4.3L" in r for r in second.verdict.blocking_verification_required)


def test_cli_collect_live_appends_the_material_8k_result(monkeypatch):
    """The live SEC path now reads the bodies the listing points to."""
    fake = FakeSec(_routes())

    class FakeSecEdgar:
        name = "sec_edgar"

        def __init__(self, _http):
            self.resolved_cik = CIK
            self.recent_filings = ONE_8K

        def collect(self, ticker, company_name):
            return CollectionResult(collector="sec_edgar")

    class Quiet:
        def __init__(self, _http):
            pass

        def collect(self, ticker, company_name):
            return CollectionResult(collector="other")

    monkeypatch.setattr(cli, "SecEdgarCollector", FakeSecEdgar)
    monkeypatch.setattr(cli, "ClinicalTrialsCollector", Quiet)
    monkeypatch.setattr(cli, "FdaCollector", Quiet)
    monkeypatch.setattr(cli, "build_sec_document_client", lambda _ua: fake)
    results, _ = cli.collect(
        TICKER, "Synthetica Therapeutics Inc.",
        settings=Settings(sec_user_agent="Synthetic Research ops@example.org"),
        http=None, use_fixtures=False, live=True,
    )
    material = [r for r in results if r.collector == m8k.MATERIAL_8K_COLLECTOR]
    assert len(material) == 1 and material[0].raw_facts
    assert fake.requested[0] == INDEX_URL


def test_the_sec_user_agent_never_appears_in_facts_errors_or_notes(monkeypatch, no_socket):
    monkeypatch.setattr(cli, "SEC_DOCUMENT_RATE_LIMIT_RPS", 1000.0)
    agent = "Synthetic Research private-contact@example.org"

    def fake_https_open(_handler, req):
        return _wire_response(req.full_url, 500)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_https_open)
    result = m8k.collect_material_8k_disclosures(
        TICKER, cik=CIK, recent=ONE_8K, http=cli.build_sec_document_client(agent), today=TODAY,
    )
    blob = " ".join([*result.errors, *result.notes, *(f.claim for f in result.raw_facts)])
    assert result.errors and "private-contact@example.org" not in blob
