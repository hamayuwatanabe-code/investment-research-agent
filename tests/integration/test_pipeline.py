"""End-to-end pipeline integration tests.

Covers requirement 26's completion criteria: the program runs a ticker end to
end, persists to SQLite, keeps bull and bear independent, judges blind, applies
the kill gate, and stores source citations.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.reporting.report import SECTION_ORDER, render_report
from investment_research.schemas.enums import (
    KillCategory,
    KillLevel,
    Provenance,
    RunStatus,
)
from investment_research.schemas.evaluation import SCORE_DIMENSIONS

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)


@pytest.fixture
def run_result(repo, fixture_dir):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    return pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"],
        aliases=metadata.get("aliases", ()),
        user_preferences={"holdings": "I own 5000 shares at an average cost of 4.20"},
    )


# --- the run completes ------------------------------------------------------
def test_pipeline_runs_end_to_end(run_result):
    assert run_result.verdict is not None
    assert run_result.scorecard is not None
    assert len(run_result.bus.facts) > 20


def test_every_agent_produced_a_run_record(run_result, repo):
    rows = repo.conn.execute(
        "SELECT agent_id, status FROM agent_runs WHERE run_id = ?", (run_result.context.run_id,)
    ).fetchall()
    recorded = {r["agent_id"] for r in rows}
    for agent_id in (
        "fact_collector",
        "evidence_integrity",
        "regulatory",
        "capital_structure",
        "science",
        "competitive",
        "catalyst",
        "microstructure",
        "contradiction",
        "kill_agent",
        "bear_agent",
        "bull_agent",
        "valuation",
        "blind_judge",
    ):
        assert agent_id in recorded, f"{agent_id} has no run record"


# --- persistence ------------------------------------------------------------
def test_facts_persisted_with_their_sources(run_result, repo):
    facts = repo.latest_facts("DEMOBIO")
    assert len(facts) == len(run_result.bus.facts)
    assert all(row["source_url"] for row in facts)
    assert all(row["source_tier"] for row in facts)
    assert all(row["run_id"] == run_result.context.run_id for row in facts)


def test_sources_persisted(run_result, repo):
    rows = repo.conn.execute("SELECT * FROM sources").fetchall()
    assert rows
    assert all(r["provenance"] == str(Provenance.FIXTURE) for r in rows)


def test_kill_gate_scores_scenarios_and_catalysts_persisted(run_result, repo):
    run_id = run_result.context.run_id
    for table in ("kill_assessments", "scores", "scenarios", "catalysts", "contradictions"):
        count = repo.conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        assert count > 0, f"{table} is empty"
    stored = repo.conn.execute(
        "SELECT dimension FROM scores WHERE run_id = ?", (run_id,)
    ).fetchall()
    assert {r["dimension"] for r in stored} == set(SCORE_DIMENSIONS)


def test_capital_structure_row_persisted(run_result, repo):
    row = repo.conn.execute(
        "SELECT * FROM capital_structure WHERE run_id = ?", (run_result.context.run_id,)
    ).fetchone()
    assert row["basic_shares"] == 41_200_000
    assert row["fully_diluted_shares"] == 63_500_000


def test_fetch_attempts_are_logged(run_result, repo):
    assert repo.conn.execute("SELECT COUNT(*) c FROM runs").fetchone()["c"] == 1


def test_structured_domain_tables_are_populated_not_merely_declared(run_result, repo):
    """A declared-but-never-written table is dead schema, not persistence."""
    trials = repo.conn.execute("SELECT * FROM clinical_trials").fetchall()
    assert len(trials) == 1
    trial = trials[0]
    assert trial["nct_id"] == "NCT-FIXTURE-0001"
    assert trial["enrollment"] == 84
    assert trial["randomized"] == "RANDOMIZED"
    assert trial["blinding"] == "DOUBLE"
    assert trial["phase"] == "2b"
    assert "biomarker" in trial["primary_endpoint"]

    events = repo.conn.execute("SELECT * FROM regulatory_events").fetchall()
    assert events
    adverse = [e for e in events if e["event_type"] == "endpoint_not_acceptable"]
    assert adverse, "the decisive regulatory event must be queryable, not only narrated"
    assert adverse[0]["regulator"] == "FDA"
    assert adverse[0]["not_agreed"] != "UNKNOWN"
    assert adverse[0]["event_date"] == "2026-05-19"

    trades = repo.conn.execute("SELECT * FROM insider_trades").fetchall()
    assert len(trades) == 1
    assert trades[0]["shares"] == 310_000
    assert trades[0]["transaction_code"] == "S"
    assert trades[0]["is_10b5_1"] == "true"


def test_unavailable_insider_details_stay_unknown(run_result, repo):
    """Without Form 4 XML the individual is not in the text; do not guess one."""
    trade = repo.conn.execute("SELECT * FROM insider_trades").fetchone()
    assert trade["insider"] == "UNKNOWN"
    assert trade["price"] is None


def test_blocked_collector_still_records_its_attempts(repo):
    """The fetch log is how a blocked run proves what it tried."""
    from investment_research.collectors.base import CollectionResult
    from investment_research.schemas.enums import FetchOutcome

    blocked = CollectionResult(
        collector="sec_edgar",
        outcome=FetchOutcome.BLOCKED,
        errors=["egress policy denied the request"],
        attempted_urls=["https://data.sec.gov/submissions/CIK0000000001.json"],
    )
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run("BLOCKED", "Blocked Corp", [blocked], price=1.0)

    rows = repo.conn.execute(
        "SELECT * FROM fetch_log WHERE run_id = ?", (result.context.run_id,)
    ).fetchall()
    assert rows
    assert rows[0]["outcome"] == str(FetchOutcome.BLOCKED)
    assert "data.sec.gov" in rows[0]["url"]


# --- isolation held throughout ---------------------------------------------
def test_no_isolation_violation_occurred(run_result):
    """The guard raises on breach, so a completed run is itself the assertion."""
    assert run_result.verdict.judged_blind is True


def test_bull_and_bear_both_produced_output(run_result):
    assert Channel.BULL in run_result.bus.channels
    assert Channel.BEAR in run_result.bus.channels
    bull = run_result.bus.channels[Channel.BULL]
    bear = run_result.bus.channels[Channel.BEAR]
    assert bull.summary != bear.summary


def test_bull_case_does_not_reproduce_bear_prose(run_result):
    bull_text = json.dumps(run_result.bus.channels[Channel.BULL].payload, default=str).lower()
    for shingle in run_result.bus.channels[Channel.BEAR].fingerprint_tokens:
        assert shingle.lower() not in bull_text


def test_user_holdings_never_appear_anywhere_in_the_run(run_result):
    blob = json.dumps(
        {name: ev.payload for name, ev in run_result.bus.channels.items()}, default=str
    )
    assert "5000 shares" not in blob
    assert "4.20" not in blob


# --- the kill gate did its job ---------------------------------------------
def test_regulatory_kill_detected_from_a_buried_filing_fact(run_result):
    gate = run_result.verdict.kill_gate
    regulatory = gate.by_category(KillCategory.REGULATORY_KILL)
    assert regulatory.level.level >= 4
    assert gate.max_level is KillLevel.K5


def test_verdict_is_blocked_pending_verification_and_confidence_precedes_it(run_result):
    """The regulatory K5 is CONFIRMED here (a real Tier 1 filing, actually read),

    but a separate, uncorroborated Nasdaq-notice claim leaves a PROVISIONAL K3
    finding outstanding. The Decision-Grade Evidence Gate withholds the Action
    until that is resolved too -- a confirmed disqualifier in one category
    does not license ignoring an unresolved one in another.
    """
    from investment_research.schemas.enums import KillConfirmation, ResearchStatus

    assert run_result.verdict.action is None
    assert run_result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    regulatory = run_result.verdict.kill_gate.by_category(KillCategory.REGULATORY_KILL)
    assert regulatory.confirmation is KillConfirmation.CONFIRMED
    assert run_result.verdict.evidence_confidence is not None
    assert "Evidence confidence" in run_result.verdict.reasoning[0]


def test_fixture_run_is_never_reported_complete(run_result):
    assert run_result.context.use_fixtures
    assert run_result.context.status is RunStatus.INCOMPLETE_RESEARCH


def test_contradictions_found_between_company_framing_and_the_filing(run_result):
    kinds = {c.kind for c in run_result.bus.contradictions}
    assert "company_vs_regulator" in kinds
    assert "basic_vs_fully_diluted" in kinds


# --- report -----------------------------------------------------------------
def test_report_contains_all_twenty_sections_in_order(run_result):
    report = render_report(run_result)
    positions = [report.index(title) for title in SECTION_ORDER]
    assert positions == sorted(positions), "sections are out of the mandated order"


def test_report_puts_thesis_breakers_before_the_bull_case(run_result):
    report = render_report(run_result)
    assert report.index("4. What Would Kill The Thesis") < report.index("7. Bull Case")


def test_report_carries_the_synthetic_data_banner(run_result):
    report = render_report(run_result)
    assert "SYNTHETIC FIXTURE DATA" in report
    assert "INCOMPLETE_RESEARCH" in report


def test_report_cites_a_source_for_the_disqualifying_fact(run_result):
    report = render_report(run_result)
    assert "fixture://sec/DEMOBIO/10-Q-2026Q2-riskfactors" in report


# --- thesis versioning ------------------------------------------------------
def test_second_run_creates_a_new_thesis_version_with_a_diff(repo, fixture_dir):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)

    first = pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"],
    )
    second = pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"],
    )

    assert first.thesis_version == 1
    assert second.thesis_version == 2
    assert second.thesis_diff["is_first"] is False
    assert "No material change" in " ".join(second.thesis_diff["WHAT_CHANGED"])

    versions = repo.conn.execute(
        "SELECT version, action FROM thesis_versions WHERE ticker = 'DEMOBIO' ORDER BY version"
    ).fetchall()
    assert [v["version"] for v in versions] == [1, 2]


def test_facts_are_not_duplicated_across_runs(repo, fixture_dir):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    first = pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
    )
    pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
    )
    assert len(repo.latest_facts("DEMOBIO")) == len(first.bus.facts)


# --- partial failure --------------------------------------------------------
def test_run_with_no_evidence_is_incomplete_and_makes_no_claims(repo):
    from investment_research.collectors.base import CollectionResult
    from investment_research.schemas.enums import FetchOutcome

    blocked = CollectionResult(
        collector="sec_edgar",
        outcome=FetchOutcome.BLOCKED,
        errors=["egress policy denied the request"],
    )
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run("NOTHING", "Nothing Corp", [blocked], price=1.0)

    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.bus.facts == []
    assert result.confidence_breakdown.score == 0.0

    # Phase 2 (requirement P6): with no evidence, no required research domain
    # was examined, so the completeness gate withholds the action label
    # entirely rather than emitting a cautious-sounding one.
    assert result.blocked
    assert result.verdict.action is None
    assert result.verdict.blocked

    report = render_report(result)
    assert "INCOMPLETE_RESEARCH" in report
    assert "No sources retrieved." in report
