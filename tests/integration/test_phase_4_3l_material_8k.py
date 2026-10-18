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
    EvidenceClass,
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
    assert client.max_requests == m8k.MAX_OUTBOUND_GETS == 32
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


# =============================================================================
# Correction 1: a trial RESULT and an issuer's report of a REGULATOR'S VIEW are
# different kinds of fact. All sentences below are SYNTHETIC.
# =============================================================================
REG_NOT_SUFFICIENT = (
    "The FDA advised the Company that the primary endpoint, change in SYN-RVEF, is not "
    "sufficient to demonstrate efficacy."
)
REG_NOT_ACCEPTED = (
    "The FDA does not consider the primary endpoint appropriate to establish effectiveness."
)
REGULATOR_EX99 = _ex99(
    f"The SYN-301 trial ({TRIAL}) did not meet its primary endpoint of change in walking "
    f"distance at week 52. {REG_NOT_SUFFICIENT} {REG_NOT_ACCEPTED}"
)


@pytest.mark.parametrize(
    ("sentence", "kinds"),
    [
        (REG_NOT_SUFFICIENT, ["REGULATOR_VIEW_REPORTED"]),
        (REG_NOT_ACCEPTED, ["REGULATOR_VIEW_REPORTED"]),
        # the trial result is still a trial result
        ("The SYN-301 trial did not meet its primary endpoint.", ["PRIMARY_ENDPOINT_NOT_MET"]),
        # future / hypothetical / company expectation: no fact of either kind
        ("If the FDA advised that the primary endpoint is not sufficient to demonstrate efficacy, "
         "we may discontinue the program.", []),
        ("We expect the FDA will advise that the endpoint is not sufficient to demonstrate efficacy.", []),
        ("The FDA could advise that the endpoint is not sufficient to demonstrate efficacy.", []),
        ("Whether the FDA would not consider the primary endpoint appropriate is uncertain.", []),
        ("We believe the FDA will accept the primary endpoint as adequate.", []),
        ("The Company plans to ask the FDA whether the primary endpoint is acceptable.", []),
        # a regulator view and a trial result in ONE sentence is never settled
        ("The trial did not meet its primary endpoint and the FDA advised that the endpoint is "
         "not sufficient to demonstrate efficacy.", ["AMBIGUOUS_PRIMARY_ENDPOINT"]),
        # the company's own opinion is not a regulator's view
        ("The Company believes the endpoint is not sufficient to demonstrate efficacy.", []),
    ],
)
def test_regulator_view_is_a_separate_kind_from_trial_result(sentence, kinds):
    assert [s.kind for s in m8k.extract_statements(sentence)] == kinds


def test_regulator_view_becomes_a_regulatory_company_claim_never_the_regulator():
    result, _ = _collect(_routes(REGULATOR_EX99))
    assert result.errors == [] and result.outcome is FetchOutcome.OK
    regulatory = [f for f in result.raw_facts if f.category is FactCategory.REGULATORY]
    clinical = [f for f in result.raw_facts if f.category is FactCategory.CLINICAL]
    assert [f.claim for f in regulatory] == [REG_NOT_SUFFICIENT, REG_NOT_ACCEPTED]
    assert len(clinical) == 1 and "did not meet" in clinical[0].claim
    for fact in regulatory:
        assert fact.company_claim is True
        assert fact.source_authority is DocumentAuthority.COMPANY_IR  # never REGULATOR
        assert fact.content_kind is ContentKind.FULL_DOCUMENT
        assert fact.source.url == f"{BASE}/{EX991}"
        assert fact.source.accession == ACC
        assert fact.source.filing_date == "2026-09-23"
        assert fact.source.event_date == "2026-09-16"
        assert "not retrieved" in str(fact.value)
        assert "was not met" not in str(fact.value)  # not a trial-result value
    assert not result.errors  # nothing was left unclassified


def test_pipeline_regulator_view_stays_a_provisional_company_claim_beside_the_trial_failure():
    material, _ = _collect(_routes(REGULATOR_EX99))
    result, repo = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])

    # no unverified blocking reason is left over from an unclassifiable sentence
    assert not [r for r in result.verdict.blocking_verification_required
                if r.startswith(m8k.UNVERIFIED_PREFIX)]
    regulatory = next(a for a in result.verdict.kill_gate.assessments
                      if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5
    assert regulatory.confirmation is KillConfirmation.PROVISIONAL  # company claim: never CONFIRMED
    assert regulatory in result.verdict.kill_gate.provisional_major_or_worse
    # the ELPIS II-type trial failure is unchanged
    clinical, failed, _ = _failure_findings(result)
    assert failed and all(f["level"] == str(KillLevel.K5) for f in failed)
    assert clinical.confirmation is KillConfirmation.PROVISIONAL
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    stored = [f for f in repo.facts_for_resume(result.context.run_id) if f.category is FactCategory.REGULATORY
              and f.claim in (REG_NOT_SUFFICIENT, REG_NOT_ACCEPTED)]
    assert len(stored) == 2 and not any(f.is_decision_grade for f in stored)
    assert all(f.evidence_class is EvidenceClass.COMPANY_CLAIM for f in stored)


def test_regulator_view_alone_blocks_the_action_without_a_trial_failure_fact():
    material, _ = _collect(_routes(_ex99(REG_NOT_SUFFICIENT)))
    result, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    regulatory = next(a for a in result.verdict.kill_gate.assessments
                      if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5 and regulatory.confirmation is KillConfirmation.PROVISIONAL
    assert result.verdict.action is None


def test_regulator_view_reaches_programme_resolution_as_a_superseding_disclosure():
    material, _ = _collect(_routes(_ex99(REG_NOT_SUFFICIENT)))
    (fact,) = material.raw_facts
    assert is_superseding_disclosure(SimpleNamespace(claim=fact.claim, value=fact.value))
    assert is_superseding_disclosure(SimpleNamespace(claim="x", value=fact.value))


# =============================================================================
# Correction 2: the wording that actually matters is "not an APPROPRIATE
# endpoint to demonstrate efficacy" -- Correction 1's tests only used "not
# sufficient". Extraction and the Kill Gate must share one vocabulary, so a
# sentence cannot be extracted yet unseen by the gate (or the reverse).
# All sentences are SYNTHETIC; they keep the MEANING of the disclosed wording.
# =============================================================================
V_APPROPRIATE = (
    "The FDA advised the Company that RVEF, the primary endpoint of the SYN-301 trial, is not "
    "an appropriate endpoint to demonstrate efficacy."
)
V_PRIMARY_APPROPRIATE = (
    "The FDA stated that RVEF is not an appropriate primary endpoint to demonstrate efficacy "
    "of SYN-301."
)
V_NOT_APPROPRIATE = "The FDA informed the Company that RVEF is not appropriate to demonstrate efficacy."
V_DOES_NOT_CONSIDER = (
    "The FDA indicated that it does not consider RVEF an appropriate endpoint to establish efficacy."
)
REGULATOR_VIEW_SENTENCES = [V_APPROPRIATE, V_PRIMARY_APPROPRIATE, V_NOT_APPROPRIATE, V_DOES_NOT_CONSIDER]

AGREES_NO_ALTERNATIVE = (
    "The Company agrees with the FDA's view and was unable to reach agreement with the FDA on an "
    "alternative primary endpoint."
)
P_PIVOTAL = "The FDA no longer considers the SYN-301 trial to be pivotal."
P_PIVOTAL_WITH_ENDPOINT = (
    "The FDA no longer considers the SYN-301 trial, whose primary endpoint is RVEF, to be a "
    "pivotal trial."
)
P_REGISTRATIONAL = "The FDA does not consider the SYN-301 trial to be a registrational trial."


def _kinds(sentence):
    return [s.kind for s in m8k.extract_statements(sentence)]


@pytest.mark.parametrize("sentence", REGULATOR_VIEW_SENTENCES)
def test_c2_regulator_view_wording_variants_are_regulator_views_not_trial_results(sentence):
    assert _kinds(sentence) == ["REGULATOR_VIEW_REPORTED"]


@pytest.mark.parametrize("sentence", REGULATOR_VIEW_SENTENCES)
def test_c2_every_extracted_regulator_view_is_seen_by_the_kill_gate(sentence):
    """Structural guard: extraction and Kill Gate rules share one vocabulary."""
    from investment_research.scoring.kill_gate import evaluate_kill_gate

    material, _ = _collect(_routes(_ex99(sentence)))
    (fact,) = [f for f in material.raw_facts if f.category is FactCategory.REGULATORY]
    from tests.conftest import make_fact

    gate = evaluate_kill_gate(
        [make_fact(fact.claim, category=FactCategory.REGULATORY, value=fact.value,
                   evidence_class=EvidenceClass.COMPANY_CLAIM, company_claim=True)],
        [],
    )
    regulatory = next(a for a in gate.assessments if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5
    assert regulatory.confirmation is KillConfirmation.PROVISIONAL


def test_c2_appropriate_wording_in_the_pipeline_is_a_provisional_regulatory_k5_beside_the_failure():
    material, _ = _collect(_routes(_ex99(
        f"The SYN-301 trial ({TRIAL}) did not meet its primary endpoint of change in walking "
        f"distance at week 52. {V_APPROPRIATE}"
    )))
    result, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    assert not [r for r in result.verdict.blocking_verification_required
                if r.startswith(m8k.UNVERIFIED_PREFIX)]
    regulatory = next(a for a in result.verdict.kill_gate.assessments
                      if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5 and regulatory.confirmation is KillConfirmation.PROVISIONAL
    clinical, failed, _ = _failure_findings(result)
    assert failed and clinical.level is KillLevel.K5
    assert result.verdict.action is None


def test_c2_company_agreement_alone_fails_closed_with_a_reason():
    """The company agreeing with the FDA is not an FDA document and not a
    trial result, and says nothing about the outcome: no fact is invented,
    and the reason says why. (Correction 3: a stated failure to AGREE on an
    endpoint is different, and is kept as a company-reported fact.)"""
    sentence = "The Company agrees with the FDA's view."
    assert _kinds(sentence) == ["AMBIGUOUS_REGULATOR_STATEMENT"]
    result, _ = _collect(_routes(_ex99(sentence)))
    assert result.raw_facts == []
    assert result.outcome is FetchOutcome.ERROR
    (error,) = result.errors
    assert error.startswith(m8k.UNVERIFIED_PREFIX) and ACC in error
    assert "regulator" in error and "cannot be classified" in error
    assert "agrees with the FDA" in error  # the sentence is quoted, not paraphrased


def test_c2_view_plus_company_agreement_keeps_the_view_and_the_consultation_outcome():
    sentence = (
        "The Company agrees with the FDA's assessment that RVEF is not an appropriate endpoint to "
        "demonstrate efficacy, and the Company and the FDA were unable to reach agreement on an "
        "alternative primary endpoint."
    )
    assert _kinds(sentence) == [
        "REGULATOR_VIEW_REPORTED", "REGULATOR_CONSULTATION_OUTCOME_REPORTED",
    ]
    assert _kinds(AGREES_NO_ALTERNATIVE) == ["REGULATOR_CONSULTATION_OUTCOME_REPORTED"]


@pytest.mark.parametrize("sentence", [P_PIVOTAL, P_PIVOTAL_WITH_ENDPOINT, P_REGISTRATIONAL])
def test_c2_loss_of_pivotal_status_is_one_regulatory_status_fact_not_a_duplicate_block(sentence):
    assert _kinds(sentence) == ["REGULATOR_STATUS_CHANGE_REPORTED"]
    result, _ = _collect(_routes(_ex99(sentence)))
    (fact,) = result.raw_facts
    assert result.errors == []
    assert fact.category is FactCategory.REGULATORY and fact.company_claim is True
    assert fact.claim == sentence and fact.source_authority is DocumentAuthority.COMPANY_IR
    assert fact.source.accession == ACC and fact.source.event_date == "2026-09-16"
    assert is_superseding_disclosure(SimpleNamespace(claim=fact.claim, value=fact.value))


def test_c2_pivotal_status_loss_reaches_the_kill_gate_and_blocks_the_action():
    material, _ = _collect(_routes(_ex99(P_PIVOTAL_WITH_ENDPOINT)))
    result, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    assert not [r for r in result.verdict.blocking_verification_required
                if r.startswith(m8k.UNVERIFIED_PREFIX)]
    regulatory = next(a for a in result.verdict.kill_gate.assessments
                      if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5 and regulatory.confirmation is KillConfirmation.PROVISIONAL
    assert result.verdict.action is None


@pytest.mark.parametrize(
    "sentence",
    [
        "If the FDA were to determine that RVEF is not an appropriate endpoint to demonstrate "
        "efficacy, we may discontinue the program.",
        "We expect the FDA to confirm that RVEF is not an appropriate endpoint to demonstrate efficacy.",
        "The FDA could conclude that RVEF is not an appropriate endpoint to demonstrate efficacy.",
        "There is a risk that the FDA will advise that RVEF is not an appropriate endpoint.",
        "The Company believes RVEF is an appropriate endpoint to demonstrate efficacy.",
        "The Company is seeking FDA agreement that RVEF is an appropriate primary endpoint.",
        "The Company believes RVEF is not an appropriate endpoint to demonstrate efficacy.",
        "The Company plans to ask the FDA whether it still considers the trial pivotal.",
        "If the FDA no longer considers the trial to be pivotal, the Company may seek alternatives.",
        "We do not expect the FDA to stop considering the SYN-301 trial pivotal.",
    ],
)
def test_c2_future_hypothetical_and_company_only_statements_are_not_classified(sentence):
    assert _kinds(sentence) == []


def test_c2_an_actual_primary_endpoint_failure_is_still_only_a_trial_result():
    assert _kinds("The SYN-301 trial did not meet its primary endpoint.") == ["PRIMARY_ENDPOINT_NOT_MET"]
    assert _kinds("Topline results showed the primary endpoint was not met.") == ["PRIMARY_ENDPOINT_NOT_MET"]


# =============================================================================
# Correction 3 (all text SYNTHETIC; the real filings were not read here)
# =============================================================================
def _march_rows(extra: int):
    """One filing per day from newest to oldest, the OLDEST being the
    financing 8-K that a cap of six pushed out of reach."""
    rows = [("8-K", f"2026-09-{d:02d}", "7.01,8.01", f"0009900001-26-0004{d:02d}", "x.htm")
            for d in range(20, 20 - extra, -1)]
    rows.append(("8-K", "2026-03-12", "1.01,3.02,8.01", "0009900001-26-000430", "march.htm"))
    return rows


def test_c3_a_year_of_filings_around_one_clinical_event_still_reaches_the_older_financing_8k():
    """Six newer 8-Ks used to push the 2026-03-12 financing 8-K beyond the cap."""
    selected, beyond, _ = m8k.select_material_8k_candidates(_recent(*_march_rows(6)), today=TODAY)
    assert "0009900001-26-000430" in [c.accession for c in selected]
    assert beyond == []


def test_c3_the_cap_and_the_actual_get_budget_are_stated_and_consistent():
    assert m8k.MAX_FILINGS == 8
    assert m8k.MAX_OUTBOUND_GETS == 32 == m8k.MAX_FILINGS * m8k.GETS_PER_FILING


def test_c3_beyond_the_new_cap_is_still_an_unverified_blocking_reason_never_dropped():
    selected, beyond, _ = m8k.select_material_8k_candidates(_recent(*_march_rows(8)), today=TODAY)
    assert [c.accession for c in beyond] == ["0009900001-26-000430"]  # the oldest
    result, _ = _collect(routes={}, recent=_recent(*_march_rows(8)))
    errors = [e for e in result.errors if "0009900001-26-000430" in e]
    assert errors and "beyond the per-run cap of 8" in errors[0]
    assert errors[0].startswith(m8k.UNVERIFIED_PREFIX)


def test_c3_real_client_actual_gets_never_exceed_32_with_eight_filings_retries_and_redirects(
    monkeypatch, no_socket,
):
    monkeypatch.setattr(cli, "SEC_DOCUMENT_RATE_LIMIT_RPS", 1000.0)
    wire: list[str] = []

    def fake_https_open(_handler, req):
        wire.append(req.full_url)
        if req.full_url.endswith("/index.json") and "redirected" not in req.full_url:
            return _wire_response(req.full_url, 302, location=req.full_url + "?redirected=1")
        return _wire_response(req.full_url, 500)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_https_open)
    rows = [("8-K", f"2026-09-{d:02d}", "8.01", f"0009900001-26-0005{d:02d}", "x.htm") for d in range(1, 10)]
    client = cli.build_sec_document_client("Synthetic Research ops@example.org")
    result = m8k.collect_material_8k_disclosures(
        TICKER, cik=CIK, recent=_recent(*rows), http=client, today=TODAY,
    )
    assert len(wire) == client.attempts_made <= 32
    # nine in scope: eight attempted (all unverified), the oldest beyond the cap
    assert len([e for e in result.errors if e.startswith(m8k.UNVERIFIED_PREFIX)]) == 9
    assert any("beyond the per-run cap of 8" in e for e in result.errors)
    assert result.raw_facts == []


CONSULT_NO_AGREEMENT = (
    "The Company and the FDA were unable to reach agreement on an alternative primary "
    "endpoint while the trial is ongoing."
)


def test_c3_company_reported_failure_to_agree_on_an_endpoint_is_a_company_claim_fact():
    assert _kinds(CONSULT_NO_AGREEMENT) == ["REGULATOR_CONSULTATION_OUTCOME_REPORTED"]
    result, _ = _collect(_routes(_ex99(CONSULT_NO_AGREEMENT)))
    assert result.errors == [] and result.outcome is FetchOutcome.OK
    (fact,) = result.raw_facts
    assert fact.category is FactCategory.REGULATORY and fact.company_claim is True
    assert fact.claim == CONSULT_NO_AGREEMENT
    assert fact.source_authority is DocumentAuthority.COMPANY_IR  # never the regulator's document
    assert fact.source.accession == ACC and fact.source.event_date == "2026-09-16"
    assert "FDA document not retrieved" in str(fact.value)
    assert "was not met" not in str(fact.value)


def test_c3_the_consultation_fact_is_a_provisional_regulatory_finding_that_blocks_the_action():
    material, _ = _collect(_routes(_ex99(CONSULT_NO_AGREEMENT)))
    result, repo = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    assert not [r for r in result.verdict.blocking_verification_required
                if r.startswith(m8k.UNVERIFIED_PREFIX)]
    regulatory = next(a for a in result.verdict.kill_gate.assessments
                      if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level >= KillLevel.K3
    assert regulatory.confirmation is KillConfirmation.PROVISIONAL
    assert regulatory in result.verdict.kill_gate.provisional_major_or_worse
    assert result.verdict.action is None
    stored = [f for f in repo.facts_for_resume(result.context.run_id)
              if f.claim == CONSULT_NO_AGREEMENT]
    assert len(stored) == 1 and not stored[0].is_decision_grade
    assert stored[0].evidence_class is EvidenceClass.COMPANY_CLAIM


@pytest.mark.parametrize(
    ("sentence", "kinds"),
    [
        # agreement with the FDA alone says nothing about the outcome: still fail-closed
        ("The Company agrees with the FDA's view.", ["AMBIGUOUS_REGULATOR_STATEMENT"]),
        # future / hypothetical / hope / plan
        ("We hope to reach agreement with the FDA on an alternative primary endpoint.", []),
        ("If we are unable to reach agreement with the FDA on a primary endpoint, we may "
         "discontinue the program.", []),
        ("The Company plans to seek FDA agreement on an alternative primary endpoint.", []),
        ("There can be no assurance that we will reach agreement with the FDA on the endpoint.", []),
        # a company disagreement with no regulator is not a regulatory consultation outcome
        ("The investors were unable to agree on an alternative primary endpoint for the plan.", []),
    ],
)
def test_c3_other_consultation_wording_stays_fail_closed_or_unclassified(sentence, kinds):
    assert _kinds(sentence) == kinds


FINANCING_CONTINGENT = (
    "The second tranche of the financing is conditioned upon the Company reaching agreement "
    "with the FDA on the primary endpoint."
)


def test_c3_financing_conditioned_on_the_regulatory_endpoint_is_kept_as_a_company_claim():
    assert _kinds(FINANCING_CONTINGENT) == ["FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE"]
    result, _ = _collect(_routes(_ex99(FINANCING_CONTINGENT)))
    (fact,) = result.raw_facts
    assert result.errors == []
    assert fact.category is FactCategory.LIQUIDITY and fact.company_claim is True
    assert fact.claim == FINANCING_CONTINGENT
    assert fact.source.accession == ACC and fact.source_authority is DocumentAuthority.COMPANY_IR


@pytest.mark.parametrize(
    "sentence",
    [
        "The Company closed a financing of $10 million in gross proceeds.",
        "The tranche will be funded on the closing date.",
        "The FDA granted Fast Track designation to the program.",
        "There is a risk that financing could be conditioned on FDA agreement on the primary endpoint.",
    ],
)
def test_c3_ordinary_financing_or_regulatory_sentences_are_not_milestone_facts(sentence):
    assert _kinds(sentence) == []


# =============================================================================
# Correction 4: wording taken from the REAL filings' reported clauses.
# NOTE ON PROVENANCE: the filings themselves were not retrieved in this
# environment (no external HTTP). The sentences below are COMPOSED around the
# exact phrases reported from the real documents -- "could not be agreed to
# while the trial is still ongoing"; "The Second Closing Trigger Date shall
# occur upon satisfaction ..." followed by the Purchase Agreement, the
# Milestone and the FDA-agreed primary endpoint -- and are NOT verbatim copies.
# =============================================================================
COULD_NOT_A = (
    "The Company proposed a new primary endpoint to the FDA, but the new primary endpoint "
    "could not be agreed to while the trial is still ongoing."
)
COULD_NOT_B = (
    "The Company and the FDA discussed a new primary endpoint for the trial; however, a new "
    "primary endpoint could not be agreed to while the trial is still ongoing."
)
COULD_NOT_AGENCY = (
    "A new primary endpoint could not be agreed to with the Agency while the trial was still ongoing."
)


@pytest.mark.parametrize("sentence", [COULD_NOT_A, COULD_NOT_B, COULD_NOT_AGENCY])
def test_c4_reported_could_not_be_agreed_is_a_consultation_outcome(sentence):
    assert _kinds(sentence) == ["REGULATOR_CONSULTATION_OUTCOME_REPORTED"]


@pytest.mark.parametrize("sentence", [COULD_NOT_A, COULD_NOT_B])
def test_c4_could_not_be_agreed_flows_to_a_stored_company_claim_and_a_provisional_k3(sentence):
    material, _ = _collect(_routes(_ex99(sentence)))
    assert material.errors == [] and material.outcome is FetchOutcome.OK
    (fact,) = material.raw_facts
    assert fact.category is FactCategory.REGULATORY and fact.company_claim is True
    assert fact.claim == sentence
    assert fact.source_authority is DocumentAuthority.COMPANY_IR
    assert "FDA document not retrieved" in str(fact.value)

    result, repo = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    assert not [r for r in result.verdict.blocking_verification_required
                if r.startswith(m8k.UNVERIFIED_PREFIX)]
    regulatory = next(a for a in result.verdict.kill_gate.assessments
                      if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K3
    assert regulatory.confirmation is KillConfirmation.PROVISIONAL
    assert regulatory in result.verdict.kill_gate.provisional_major_or_worse
    assert result.verdict.action is None
    stored = [f for f in repo.facts_for_resume(result.context.run_id) if f.claim == sentence]
    assert len(stored) == 1 and not stored[0].is_decision_grade
    assert stored[0].evidence_class is EvidenceClass.COMPANY_CLAIM


def test_c4_could_not_without_a_reported_anchor_or_a_regulator_fails_closed_with_a_reason():
    # no temporal anchor: could be a past report or a hypothetical -> not settled
    assert _kinds("The FDA could not agree to a new primary endpoint.") == [
        "AMBIGUOUS_REGULATOR_STATEMENT"
    ]
    # no regulator named: with whom it was (not) agreed is unknown -> not settled
    assert _kinds("A new primary endpoint could not be agreed to while the trial is still ongoing.") == [
        "AMBIGUOUS_REGULATOR_STATEMENT"
    ]


@pytest.mark.parametrize(
    "sentence",
    [
        "If a new primary endpoint could not be agreed to with the FDA while the trial is still "
        "ongoing, we may be required to start a new trial.",
        "There is a risk that the FDA could not agree to a new primary endpoint while the trial is "
        "ongoing.",
        "Should the FDA review a new primary endpoint, it could not be agreed to while the trial is "
        "still ongoing.",
        "We expect that a new primary endpoint could not be agreed to with the FDA while the trial "
        "is ongoing.",
        "Even if the Company proposes a new primary endpoint after completing enrollment of the "
        "remaining patients in the trial, the FDA could not agree to it while the trial is ongoing.",
        "The Company believes the FDA would not object to a new primary endpoint.",
    ],
)
def test_c4_future_and_hypothetical_could_not_is_never_a_reported_outcome(sentence):
    assert "REGULATOR_CONSULTATION_OUTCOME_REPORTED" not in _kinds(sentence)
    assert _kinds(sentence) == []


FIN_BASE = (
    "The Second Closing Trigger Date shall occur upon satisfaction of the Milestone (as defined "
    "in the Purchase Agreement), which requires that the Company has reached agreement with the "
    "U.S. Food and Drug Administration (the “FDA”) on the primary endpoint for the trial, "
    "and the Investors shall purchase the second tranche of Shares at the Second Closing"
)
FIN_WAIVER = FIN_BASE + ", which condition may be waived by the Investors in their sole discretion."
FIN_PLAIN = FIN_BASE + "."
FIN_LONG = FIN_BASE + ", " + " ".join(
    f"subject to the representations in Section 3.{i}(b) of the Purchase Agreement being true and "
    "correct in all material respects as of such date" for i in range(1, 14)
) + "."


@pytest.mark.parametrize("sentence", [FIN_PLAIN, FIN_WAIVER])
def test_c4_second_closing_trigger_conditioned_on_the_fda_agreed_endpoint_is_extracted(sentence):
    assert _kinds(sentence) == ["FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE"]
    result, _ = _collect(_routes(_ex99(sentence)))
    (fact,) = result.raw_facts
    assert result.errors == []
    assert fact.category is FactCategory.LIQUIDITY and fact.company_claim is True
    assert fact.claim == sentence
    assert fact.source_authority is DocumentAuthority.COMPANY_IR
    assert fact.source.accession == ACC


def test_c4_an_overlong_financing_condition_sentence_is_never_silently_dropped():
    assert len(FIN_LONG) > 1200
    assert _kinds(FIN_LONG) == ["AMBIGUOUS_FINANCING_STATEMENT"]
    result, _ = _collect(_routes(_ex99(FIN_LONG)))
    assert result.raw_facts == []
    (error,) = result.errors
    assert error.startswith(m8k.UNVERIFIED_PREFIX) and ACC in error
    assert "financing" in error and "cannot be classified" in error


@pytest.mark.parametrize(
    "sentence",
    [
        "The Company closed a financing of $10 million in gross proceeds.",
        "The Second Closing shall occur on the third business day after the First Closing.",
        "There is a risk that the second tranche could be conditioned on FDA agreement on the primary endpoint.",
        "If the FDA does not agree on the primary endpoint, the Investors may decline to fund the second tranche.",
    ],
)
def test_c4_ordinary_or_hypothetical_financing_text_is_not_a_milestone_fact(sentence):
    assert _kinds(sentence) == []


# =============================================================================
# Correction 5: live results on the real filings showed (a) a "potentially"
# in an EARLIER part of the sentence suppressed the reported consultation
# outcome, and (b) the contract-style financing sentence ALSO produced an
# AMBIGUOUS_PRIMARY_ENDPOINT block. As before, the sentences are COMPOSED
# around the phrases reported from the real 8-Ks (not retrieved here).
# =============================================================================
MAY_COMMA = (
    "Following discussions of potentially appropriate endpoints for the trial, the FDA indicated "
    "that a new primary endpoint could not be agreed to while the trial is still ongoing."
)
MAY_NO_PUNCTUATION = (
    "The Company and the FDA discussed potentially appropriate endpoints and the FDA indicated "
    "that a new primary endpoint could not be agreed to while the trial is still ongoing."
)
MAY_EXPECT_ELSEWHERE = (
    "We expect to discuss potentially appropriate endpoints with the FDA; the FDA indicated that "
    "a new primary endpoint could not be agreed to while the trial is still ongoing."
)


@pytest.mark.parametrize("sentence", [MAY_COMMA, MAY_NO_PUNCTUATION, MAY_EXPECT_ELSEWHERE])
def test_c5_an_unrelated_potentially_does_not_hide_the_reported_consultation_outcome(sentence):
    assert _kinds(sentence) == ["REGULATOR_CONSULTATION_OUTCOME_REPORTED"]


def test_c5_the_may_sentence_reaches_a_stored_fact_and_a_provisional_k3_with_no_unverified_reason():
    material, _ = _collect(_routes(_ex99(MAY_COMMA)))
    assert material.errors == [] and material.outcome is FetchOutcome.OK
    (fact,) = material.raw_facts
    assert fact.category is FactCategory.REGULATORY and fact.company_claim is True
    assert fact.claim == MAY_COMMA and "FDA document not retrieved" in str(fact.value)
    result, repo = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    assert not [r for r in result.verdict.blocking_verification_required
                if r.startswith(m8k.UNVERIFIED_PREFIX)]
    regulatory = next(a for a in result.verdict.kill_gate.assessments
                      if a.category == KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K3 and regulatory.confirmation is KillConfirmation.PROVISIONAL
    assert result.verdict.action is None
    stored = [f for f in repo.facts_for_resume(result.context.run_id) if f.claim == MAY_COMMA]
    assert len(stored) == 1 and not stored[0].is_decision_grade


@pytest.mark.parametrize(
    "sentence",
    [
        # a conditional OPENER makes the whole consultation outcome hypothetical
        "If the FDA indicated that a new primary endpoint could not be agreed to while the trial is "
        "still ongoing, the Company may need to start a new trial.",
        "Even if potentially appropriate endpoints are identified, the FDA indicated that a new "
        "primary endpoint could not be agreed to while the trial is still ongoing.",
        "We expect that, after discussing potentially appropriate endpoints, the FDA will indicate "
        "that a new primary endpoint could not be agreed to while the trial is still ongoing.",
        "The Company noted that if it proposed a new primary endpoint, the FDA could not agree to it "
        "while the trial is still ongoing.",
        "There is a risk that, after discussing appropriate endpoints, the FDA will indicate a new "
        "primary endpoint could not be agreed to while the trial is still ongoing.",
    ],
)
def test_c5_a_qualifier_that_governs_the_outcome_still_prevents_the_fact(sentence):
    assert _kinds(sentence) == []


def test_c5_potentially_that_hedges_the_outcome_itself_is_not_a_fact_but_is_surfaced():
    """"potentially could not be agreed" hedges the result: never a reported
    outcome, yet not silently ignored either (it is still about the primary endpoint)."""
    sentence = (
        "The FDA indicated that a new primary endpoint potentially could not be agreed to while "
        "the trial is still ongoing."
    )
    kinds = _kinds(sentence)
    assert "REGULATOR_CONSULTATION_OUTCOME_REPORTED" not in kinds
    assert kinds == ["AMBIGUOUS_PRIMARY_ENDPOINT"]


# ---- the contract-style financing sentence ----------------------------------------
FIN_SIG = (
    "The Second Closing Trigger Date shall occur upon satisfaction or waiver of the Milestone set "
    "forth in the Purchase Agreement, being the Company's receipt of written confirmation from the "
    "FDA that a statistically significant improvement in the primary endpoint of the trial would "
    "be sufficient to support a marketing application, and the Investors shall purchase the "
    "second tranche of Shares at the Second Closing"
)
FIN_SIG_PLAIN = FIN_SIG + "."


def test_c5_a_contractual_condition_naming_statistical_significance_is_one_liquidity_fact():
    assert _kinds(FIN_SIG_PLAIN) == ["FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE"]
    result, _ = _collect(_routes(_ex99(FIN_SIG_PLAIN)))
    (fact,) = result.raw_facts
    assert result.errors == [] and result.outcome is FetchOutcome.OK  # no AMBIGUOUS_PRIMARY_ENDPOINT block
    assert fact.category is FactCategory.LIQUIDITY and fact.company_claim is True
    assert fact.claim == FIN_SIG_PLAIN


def test_c5_the_financing_sentence_alone_does_not_block_the_action_with_an_unverified_reason():
    material, _ = _collect(_routes(_ex99(FIN_SIG_PLAIN)))
    result, _ = _run([_registry_and_marker(), CollectionResult(collector="sec_edgar"), material])
    assert not [r for r in result.verdict.blocking_verification_required
                if r.startswith(m8k.UNVERIFIED_PREFIX)]


def test_c5_an_independent_clear_trial_result_in_the_same_sentence_is_still_reported():
    sentence = FIN_SIG + ", and the trial did not meet its primary endpoint."
    assert sorted(_kinds(sentence)) == sorted(
        ["PRIMARY_ENDPOINT_NOT_MET", "FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE"]
    )


def test_c5_an_independent_unclassifiable_trial_result_in_the_same_sentence_is_still_unverified():
    sentence = FIN_SIG + ", and the trial showed only a numerical trend on the primary endpoint."
    assert sorted(_kinds(sentence)) == sorted(
        ["AMBIGUOUS_PRIMARY_ENDPOINT", "FINANCING_CONTINGENT_ON_REGULATORY_MILESTONE"]
    )


def test_c5_without_a_financing_condition_statistical_significance_stays_ambiguous():
    assert _kinds("The primary endpoint did not show a statistically significant improvement.") == [
        "AMBIGUOUS_PRIMARY_ENDPOINT"
    ]
    assert _kinds("A statistically significant effect on the primary endpoint was observed.") == [
        "AMBIGUOUS_PRIMARY_ENDPOINT"
    ]
