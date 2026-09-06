"""Real-ticker end-to-end tests (requirement P5).

These run the full pipeline over the **captured corpora** in ``data/corpus/`` --
real documents about real issuers, with real URLs, captured at a stated time.
They are not synthetic fixtures, and they are not live fetches either; the tests
assert on both the findings and on the system's honesty about that distinction.

LGVN is the important one. It is the live 2026 instance of the exact failure
this system was built after: a company headline reading "Constructive Type C
Meeting" over an FDA position that the primary endpoint cannot demonstrate
efficacy, and a trial that is no longer called pivotal.
"""

from __future__ import annotations

from datetime import date

import pytest

from investment_research.collectors.extraction import DocumentCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.reporting.report import render_report
from investment_research.research.adversarial import build_plan, run_adversarial_search
from investment_research.research.corpus import CorpusResearchProvider
from investment_research.schemas.enums import (
    KillCategory,
    Provenance,
    ResearchPath,
    VerifiedStatus,
)
from investment_research.schemas.evaluation import INVESTMENT_QUALITY_DIMENSIONS

pytestmark = pytest.mark.regression

TODAY = date(2026, 9, 6)
HIGH = 7.0


def run_ticker(repo, corpus_dir, ticker: str, *, adversarial: bool = True):
    corpus = CorpusResearchProvider(corpus_dir, ticker)
    metadata = corpus.metadata(ticker)
    documents = corpus.documents(ticker)
    outcome = None
    if adversarial:
        outcome = run_adversarial_search(
            corpus, build_plan(ticker, metadata.get("company_name", ticker))
        )
    collector = DocumentCollector(documents, provenance=Provenance.CAPTURED)
    results = [collector.collect(ticker, metadata.get("company_name", ticker))]
    pipeline = Pipeline(
        repo,
        NullSearchProvider(),
        today=TODAY,
        research=corpus,
        adversarial=outcome,
        chunks=collector.chunks,
        capture_info=corpus.capture_info(ticker),
    )
    return pipeline.run(
        ticker,
        metadata.get("company_name", ticker),
        results,
        price=metadata.get("price"),
        aliases=metadata.get("aliases", ()),
    )


@pytest.fixture(scope="module")
def corpus_dir():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "data" / "corpus"


@pytest.fixture
def lgvn(repo, corpus_dir):
    return run_ticker(repo, corpus_dir, "LGVN")


# =========================== LGVN ==========================================
def test_lgvn_corpus_is_real_captured_data_not_fixture(lgvn):
    assert lgvn.context.use_fixtures is False
    assert lgvn.capture_info["captured_via"] == "anthropic_websearch"
    assert all(f.provenance is Provenance.CAPTURED for f in lgvn.bus.facts)
    assert all(s.url.startswith("https://") for s in lgvn.bus.sources)


def test_lgvn_finds_the_type_c_meeting(lgvn):
    claims = " ".join(f.claim for f in lgvn.bus.facts)
    assert "Type C meeting" in claims
    # The abbreviation must not truncate the statement.
    assert "U.S. FDA" in claims or "with the U.S." in claims


def test_lgvn_finds_the_rvef_endpoint_issue(lgvn):
    """The decisive fact, in the company's own words."""
    claims = [f.claim for f in lgvn.bus.facts]
    endpoint = [
        c
        for c in claims
        if "not sufficient to demonstrate efficacy" in c or "cannot prove efficacy" in c
    ]
    assert endpoint, "the RVEF endpoint finding was not extracted"
    assert any("right ventricular ejection fraction" in c or "RVEF" in c for c in claims)


def test_lgvn_finds_that_elpis_ii_is_no_longer_called_pivotal(lgvn):
    """The status change is the finding a keyword list would miss."""
    claims = " ".join(f.claim for f in lgvn.bus.facts)
    assert "no longer refers to the ELPIS II trial as pivotal" in claims or (
        "no longer calls ELPIS II pivotal" in claims
    )


def test_lgvn_regulatory_position_is_rejected(lgvn):
    payload = lgvn.bus.channels[Channel.REGULATORY].payload
    assert payload["endpoint_position"] == "REJECTED"
    assert payload["not_agreed"], "the refused list must not be empty for LGVN"


def test_lgvn_regulatory_kill_is_at_least_k3(lgvn):
    """The assertion the brief asks for, on real 2026 evidence."""
    assessment = lgvn.verdict.kill_gate.by_category(KillCategory.REGULATORY_KILL)
    assert assessment is not None
    assert assessment.level.level >= 3, f"regulatory kill was only {assessment.level}"


def test_lgvn_finds_cash_runway_and_going_concern(lgvn):
    claims = " ".join(f.claim for f in lgvn.bus.facts).lower()
    assert "going concern" in claims
    assert "fourth quarter of 2026" in claims or "runway" in claims


def test_lgvn_capital_kill_fires_on_the_balance_sheet(lgvn):
    assessment = lgvn.verdict.kill_gate.by_category(KillCategory.CAPITAL_KILL)
    assert assessment.level.level >= 3


def test_lgvn_finds_financing_and_dilution(lgvn):
    claims = " ".join(f.claim for f in lgvn.bus.facts).lower()
    assert "private placement" in claims
    assert "dilution" in claims


def test_lgvn_finds_the_designations_and_labels_them_procedural(lgvn):
    claims = " ".join(f.claim for f in lgvn.bus.facts)
    for designation in ("Orphan Drug", "Fast Track", "Rare Pediatric Disease"):
        assert designation in claims, f"{designation} designation was not found"

    payload = lgvn.bus.channels[Channel.REGULATORY].payload
    designations = [a for a in payload["agreed"] if "designation" in a.lower()]
    assert designations
    for entry in designations:
        assert "procedural designation only" in entry


def test_lgvn_the_headline_does_not_win(lgvn):
    """The press release says "Constructive". The evidence says otherwise."""
    claims = " ".join(f.claim for f in lgvn.bus.facts)
    assert "constructive" in claims.lower(), "the company's framing should be recorded"
    assert lgvn.bus.channels[Channel.REGULATORY].payload["endpoint_position"] == "REJECTED"


# --- the central prohibition, on real data ---------------------------------
def test_lgvn_does_not_emit_high_explosive_and_high_quality_together(lgvn):
    scores = lgvn.scorecard.scores
    explosive = scores["explosive_potential"]
    quality = {d: scores[d] for d in INVESTMENT_QUALITY_DIMENSIONS if d in scores}
    if explosive >= HIGH:
        offenders = {d: v for d, v in quality.items() if v >= HIGH}
        assert not offenders, (
            f"explosive_potential={explosive} emitted with high investment quality {offenders}"
        )


def test_lgvn_does_not_emit_high_regulatory_certainty(lgvn):
    """ "High Regulatory Certainty" must be impossible here."""
    assert lgvn.scorecard.scores["regulatory_quality"] < 3.0


def test_lgvn_investment_quality_was_capped(lgvn):
    """Every investment-quality dimension sits at or below the kill-gate cap.

    Asserting membership of ``capped_by_kill_gate`` would be weaker: a dimension
    already scored below the cap is never listed there, and being *already* at
    the floor is a stronger result than being pulled down to it.
    """
    from investment_research.scoring.kill_gate import quality_cap

    cap = quality_cap(lgvn.verdict.kill_gate.max_level)
    assert lgvn.scorecard.capped_by_kill_gate
    for dimension in INVESTMENT_QUALITY_DIMENSIONS:
        value = lgvn.scorecard.scores.get(dimension)
        if value is not None:
            assert value <= cap, f"{dimension}={value} exceeds the kill-gate cap {cap}"


def test_lgvn_verdict_is_avoid(lgvn):
    from investment_research.schemas.enums import Action

    assert lgvn.verdict.action is Action.AVOID
    assert not lgvn.verdict.blocked


# --- traceability and escalation on real data -------------------------------
def test_lgvn_every_fact_traces_to_a_real_url(lgvn):
    index = lgvn.traceability
    assert index is not None and index.links
    for link in index.rows():
        assert link.url.startswith("https://"), f"{link.fact_id} has no real URL"
        assert link.source_id
        assert link.publication_date != "" and link.event_date != ""


def test_lgvn_material_claims_are_escalated_and_marked_when_unconfirmed(lgvn):
    """Search summaries cannot confirm themselves."""
    assert lgvn.escalation is not None
    assert lgvn.escalation.attempts, "material claims should have triggered escalation"
    unverified = [
        f for f in lgvn.bus.facts if f.verified_status is VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM
    ]
    assert unverified, "unconfirmed material claims must be marked, not left as merely unverified"
    assert not any(f.is_decision_grade for f in unverified)


def test_lgvn_company_press_release_cannot_confirm_a_regulator_claim(lgvn):
    for attempt in lgvn.escalation.confirmed:
        assert "globenewswire" not in attempt.confirming_url


def test_lgvn_report_states_the_capture_time_and_its_limits(lgvn):
    report = render_report(lgvn)
    assert "CAPTURED CORPUS REPLAY" in report
    assert "2026-09-06" in report
    assert "NOT fetched live at run time" in report
    assert "search-engine summaries rather than the document itself" in report


def test_lgvn_search_coverage_is_reported(lgvn):
    assert lgvn.completeness is not None
    report = render_report(lgvn)
    assert "SEARCH COMPLETENESS GATE" in report
    assert all(
        entry.paths == () or ResearchPath.CORPUS in entry.paths
        for entry in lgvn.completeness.coverage.values()
    )


def test_lgvn_isolation_held(lgvn):
    assert lgvn.verdict.judged_blind is True


# =========================== CNTB ==========================================
@pytest.fixture
def cntb(repo, corpus_dir):
    return run_ticker(repo, corpus_dir, "CNTB")


def test_cntb_finds_the_acute_exacerbation_programme(cntb):
    claims = " ".join(f.claim for f in cntb.bus.facts).lower()
    assert "acute exacerbation" in claims
    assert "rademikibart" in claims


def test_cntb_finds_the_dmc_interim_analysis(cntb):
    claims = " ".join(f.claim for f in cntb.bus.facts)
    assert "Data Monitoring Committee" in claims
    assert "interim analysis" in claims.lower() or "pre-specified" in claims.lower()


def test_cntb_finds_cash_and_runway(cntb):
    claims = " ".join(f.claim for f in cntb.bus.facts).lower()
    assert "cash" in claims
    assert "2027" in claims, "the runway guidance should be captured"


def test_cntb_finds_the_financing(cntb):
    claims = " ".join(f.claim for f in cntb.bus.facts).lower()
    assert "private placement" in claims


def test_cntb_finds_the_upcoming_readouts(cntb):
    claims = " ".join(f.claim for f in cntb.bus.facts).lower()
    assert "september 2026" in claims
    assert "asthma" in claims and "copd" in claims


def test_cntb_is_not_disqualified_the_way_lgvn_is(cntb):
    """A different company must produce a different answer."""
    regulatory = cntb.verdict.kill_gate.by_category(KillCategory.REGULATORY_KILL)
    assert regulatory.level.level < 5
    assert cntb.bus.channels[Channel.REGULATORY].payload["endpoint_position"] != "REJECTED"


# =========================== CRBP ==========================================
@pytest.fixture
def crbp(repo, corpus_dir):
    return run_ticker(repo, corpus_dir, "CRBP")


def test_crbp_finds_the_phase_1a_study(crbp):
    claims = " ".join(f.claim for f in crbp.bus.facts)
    assert "Phase 1a" in claims or "single ascending dose" in claims.lower()
    assert "CRB-913" in claims


def test_crbp_finds_canyon_1(crbp):
    claims = " ".join(f.claim for f in crbp.bus.facts)
    assert "CANYON-1" in claims
    assert "240" in claims or "dose-ranging" in claims.lower()


def test_crbp_finds_the_cb1_class_safety_history(crbp):
    claims = " ".join(f.claim for f in crbp.bus.facts).lower()
    assert "neuropsychiatric" in claims
    assert "abandoned" in claims


def test_crbp_finds_the_monlunabant_comparison(crbp):
    claims = " ".join(f.claim for f in crbp.bus.facts).lower()
    assert "monlunabant" in claims


def test_crbp_finds_crb_701(crbp):
    claims = " ".join(f.claim for f in crbp.bus.facts)
    assert "CRB-701" in claims


def test_crbp_cash_is_reported_unknown_not_guessed(crbp):
    """sec.gov was blocked, so the balance sheet is genuinely not in evidence."""
    capital = crbp.bus.channels[Channel.CAPITAL_STRUCTURE].payload
    assert capital["basic_shares"] is None
    assert capital["fully_diluted_shares"] is None
    report = render_report(crbp)
    assert "UNKNOWN" in report


def test_crbp_thin_evidence_produces_low_confidence_not_a_confident_answer(crbp):
    assert crbp.confidence_breakdown.score < 5.0
