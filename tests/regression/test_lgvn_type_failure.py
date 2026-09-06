"""Regression test for the LGVN-type failure (requirement 18).

The failure being guarded against, restated precisely:

    A small-cap clinical-stage company presented an unusually attractive
    surface -- tiny market cap, near-term catalyst, Fast Track and Orphan Drug
    designations, a DMC recommending continuation, NIH support, a large stated
    TAM, and bullish sell-side coverage.  A coherent and exciting narrative was
    assembled from those signals.  One fact, available in a filing, invalidated
    all of it: the FDA did not consider the primary endpoint appropriate to
    establish effectiveness.

The required behaviour is not "be pessimistic".  It is specific:

  1. the single disqualifying regulatory fact must be FOUND, from the filing,
     not from the press release;
  2. Regulatory Kill must be at least K3;
  3. high Explosive Potential and high Investment Quality must NOT be produced
     together;
  4. the company's own word ("constructive") must not overwrite the regulator's;
  5. the attractive signals must NOT be able to rescue the verdict.

These are asserted against an anonymous Company X so the test cannot pass by
recognising a ticker.
"""

from __future__ import annotations

from datetime import date

import pytest

from investment_research.collectors.base import CollectionResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.reporting.report import render_report
from investment_research.schemas.enums import (
    UNKNOWN,
    Action,
    FactCategory,
    KillCategory,
    Provenance,
    SourceTier,
)
from investment_research.schemas.evaluation import INVESTMENT_QUALITY_DIMENSIONS
from investment_research.schemas.fact import RawFact, Source, make_source_id

pytestmark = pytest.mark.regression

TODAY = date(2026, 9, 6)
# A distinct, synthetic identity: the point is that the Judge sees "Company X"
# because the system anonymised it, not because it was labelled that way here.
TICKER = "NWBX"
NAME = "Northwind Biosciences"

#: Thresholds for "high" in requirement 18's prohibition.
HIGH = 7.0


def _source(url: str, title: str, tier: SourceTier, event: str, published: str) -> Source:
    return Source(
        source_id=make_source_id(url, title),
        url=url,
        title=title,
        tier=tier,
        event_date=event,
        published_date=published,
        filing_date=published,
        provenance=Provenance.FIXTURE,
    )


FILING = _source(
    "fixture://sec/NWBX/10-Q-riskfactors",
    "Form 10-Q, Risk Factors",
    SourceTier.TIER_1,
    "2026-05-19",
    "2026-08-07",
)
FILING_MAIN = _source(
    "fixture://sec/NWBX/10-Q",
    "Form 10-Q",
    SourceTier.TIER_1,
    "2026-06-30",
    "2026-08-07",
)
PRESS = _source(
    "fixture://ir/NWBX/pr",
    "Company press release",
    SourceTier.TIER_2,
    "2026-05-19",
    "2026-05-27",
)
REGISTRY = _source(
    "fixture://ctgov/NCT-X",
    "Registry record",
    SourceTier.TIER_1,
    "2026-11-30",
    "2026-04-02",
)
BROKER = _source(
    "fixture://broker/NOTE",
    "Broker research note",
    SourceTier.TIER_4,
    "2026-08-14",
    "2026-08-14",
)


def raw(claim, source, category, value=UNKNOWN, company_claim=False, unit=UNKNOWN):
    return RawFact(
        ticker=TICKER,
        category=category,
        claim=claim,
        source=source,
        value=value,
        unit=unit,
        company_claim=company_claim,
        collector="regression_fixture",
    )


#: The attractive surface. Every one of these is a real, citable positive.
ATTRACTIVE_SIGNALS = [
    raw(
        "FDA granted Fast Track designation to the lead programme",
        PRESS,
        FactCategory.REGULATORY,
        "fast_track",
        True,
    ),
    raw(
        "FDA granted Orphan Drug designation for the rare disease indication",
        PRESS,
        FactCategory.REGULATORY,
        "orphan_drug",
        True,
    ),
    raw(
        "The independent Data Monitoring Committee completed its planned interim review and "
        "recommended continuation of the trial without modification",
        PRESS,
        FactCategory.CLINICAL,
        "dmc_continue",
        True,
    ),
    raw(
        "NIH awarded a research grant supporting work on the mechanism",
        _source(
            "fixture://nih/GRANT", "NIH award record", SourceTier.TIER_1, "2026-03-01", "2026-03-01"
        ),
        FactCategory.SCIENCE,
        "nih_grant",
        False,
    ),
    raw(
        "Company investor presentation states a total addressable market in excess of "
        "$8,000,000,000",
        PRESS,
        FactCategory.MARKET_SIZE,
        "8000000000",
        True,
        "USD",
    ),
    raw(
        "Topline data are guided for the fourth quarter of 2026",
        PRESS,
        FactCategory.CATALYST,
        "2026-12-15",
        True,
        "date",
    ),
    raw(
        "A sell-side analyst maintains a price target of $18.00",
        BROKER,
        FactCategory.OTHER,
        "18.00",
        False,
        "USD",
    ),
    raw(
        "NCT-X is a Phase 2b trial; allocation is Randomized, masking is DOUBLE",
        REGISTRY,
        FactCategory.CLINICAL,
        "Randomized/DOUBLE",
        True,
        "design",
    ),
    raw(
        "NCT-X enrollment is 84 (ACTUAL)",
        REGISTRY,
        FactCategory.CLINICAL,
        "84",
        True,
        "participants",
    ),
    raw(
        "Basic shares outstanding were 41,200,000",
        FILING_MAIN,
        FactCategory.CAPITAL_STRUCTURE,
        "41200000",
        True,
        "shares",
    ),
    # The company's own characterization of the regulatory interaction.
    raw(
        "Company press release characterizes the meeting outcome as constructive and states "
        "the company is aligned with the agency on the path forward",
        PRESS,
        FactCategory.REGULATORY,
        "constructive",
        True,
        "company_characterization",
    ),
]

#: The three facts that actually decide the case, all in filings.
DISQUALIFYING_FACTS = [
    raw(
        "In written responses following the Type C meeting, FDA stated that it does not "
        "consider the proposed primary endpoint appropriate to establish effectiveness for the "
        "intended indication, and that an additional adequate and well-controlled trial would "
        "be required to support a marketing application",
        FILING,
        FactCategory.REGULATORY,
        "endpoint_not_accepted",
        False,
        "regulatory_position",
    ),
    raw(
        "Existing cash is expected to fund operations into the second quarter of 2027, which is "
        "less than twelve months from the issuance date of these financial statements, raising "
        "substantial doubt about the ability to continue as a going concern",
        FILING_MAIN,
        FactCategory.LIQUIDITY,
        "going_concern_doubt",
        True,
        "status",
    ),
    raw(
        "An at-the-market offering program with $50,000,000 of remaining capacity is in effect "
        "under an effective shelf registration statement",
        FILING_MAIN,
        FactCategory.CAPITAL_STRUCTURE,
        "50000000",
        True,
        "USD",
    ),
]

CASH_FACTS = [
    raw(
        "Cash and cash equivalents were $31,500,000",
        FILING_MAIN,
        FactCategory.FINANCIAL,
        "31500000",
        True,
        "USD",
    ),
    raw(
        "Net cash used in operating activities was $32,800,000 for the six months",
        FILING_MAIN,
        FactCategory.FINANCIAL,
        "32800000",
        True,
        "USD",
    ),
]


def build_collection(include_disqualifying: bool = True) -> CollectionResult:
    facts = list(ATTRACTIVE_SIGNALS) + list(CASH_FACTS)
    if include_disqualifying:
        facts += DISQUALIFYING_FACTS
    sources = {f.source.source_id: f.source for f in facts}
    return CollectionResult(
        collector="regression_fixture",
        raw_facts=facts,
        sources=list(sources.values()),
        provenance=Provenance.FIXTURE,
    )


def run(repo, include_disqualifying: bool = True):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    return pipeline.run(
        TICKER,
        NAME,
        [build_collection(include_disqualifying)],
        price=3.40,
        user_preferences={
            "opinion": "I am tremendously enthusiastic about this particular holding",
            "holdings": "I already own 5000 shares at an average cost of 4.20",
        },
    )


@pytest.fixture
def result(repo):
    return run(repo)


# --- 1. the decisive fact is found ------------------------------------------
def test_the_disqualifying_regulatory_fact_is_found(result):
    payload = result.bus.channels[Channel.REGULATORY].payload
    assert payload["endpoint_position"] == "REJECTED"
    assert payload["not_agreed"], "the 'not agreed' list must never be empty here"


def test_it_is_found_from_the_filing_not_the_press_release(result):
    """The press release said 'constructive'. The filing said what happened."""
    supporting = [
        f
        for f in result.bus.facts
        if "does not consider" in f.claim and f.source_tier is SourceTier.TIER_1
    ]
    assert supporting, "the decisive fact must be sourced to the statutory filing"
    assert "10-Q" in supporting[0].source_title


# --- 2. Regulatory Kill >= K3 ------------------------------------------------
def test_regulatory_kill_is_at_least_k3(result):
    """The explicit assertion requirement 18 asks for."""
    assessment = result.verdict.kill_gate.by_category(KillCategory.REGULATORY_KILL)
    assert assessment is not None
    assert assessment.level.level >= 3, f"regulatory kill was only {assessment.level}"


def test_capital_kill_also_fires_on_runway_and_going_concern(result):
    assessment = result.verdict.kill_gate.by_category(KillCategory.CAPITAL_KILL)
    assert assessment.level.level >= 3


# --- 3. the central prohibition ---------------------------------------------
def test_high_explosive_potential_and_high_investment_quality_are_not_both_produced(result):
    """Requirement 18: these two must never be emitted together.

    Explosive Potential is allowed to stay high -- this company genuinely can
    move violently -- but every investment-quality dimension must be capped.
    """
    scores = result.scorecard.scores
    explosive = scores["explosive_potential"]
    quality = {d: scores[d] for d in INVESTMENT_QUALITY_DIMENSIONS if d in scores}

    if explosive >= HIGH:
        offenders = {d: v for d, v in quality.items() if v >= HIGH}
        assert not offenders, (
            f"explosive_potential={explosive} was emitted alongside high investment quality "
            f"{offenders}; this is the exact failure the system exists to prevent"
        )


def test_investment_quality_dimensions_were_actually_capped_by_the_kill_gate(result):
    assert result.scorecard.capped_by_kill_gate, "the kill gate applied no cap at all"
    for dimension in result.scorecard.capped_by_kill_gate:
        assert dimension in INVESTMENT_QUALITY_DIMENSIONS


def test_there_is_no_overall_score_to_average_the_problem_away(result):
    assert "overall" not in result.scorecard.scores
    assert "total" not in result.scorecard.scores
    assert not hasattr(result.scorecard, "overall_score")


def test_evidence_confidence_is_reported_separately_from_upside(result):
    """ "Explosive 9.5 / Confidence 4" must be expressible."""
    assert result.scorecard.evidence_confidence is not None
    assert result.scorecard.evidence_confidence != result.scorecard.scores["explosive_potential"]


# --- 4. the company's adjective does not win --------------------------------
def test_company_framing_is_contradicted_not_adopted(result):
    kinds = {c.kind for c in result.bus.contradictions}
    assert "company_vs_regulator" in kinds

    payload = result.bus.channels[Channel.REGULATORY].payload
    assert payload["company_framing"], "the company's adjective must be recorded as such"
    assert any("characterization only" in f for f in payload["company_framing"])


def test_procedural_designations_are_labelled_as_saying_nothing_about_efficacy(result):
    payload = result.bus.channels[Channel.REGULATORY].payload
    designations = [a for a in payload["agreed"] if "Designation" in a or "designation" in a]
    assert designations
    for entry in designations:
        assert "procedural designation only" in entry


def test_analyst_target_cannot_settle_the_question(result):
    kinds = {c.kind for c in result.bus.contradictions}
    assert "analyst_vs_primary" in kinds
    broker_facts = [f for f in result.bus.facts if f.source_tier is SourceTier.TIER_4]
    assert broker_facts
    assert not any(f.is_decision_grade for f in broker_facts)


# --- 5. the attractive signals cannot rescue the verdict --------------------
def test_verdict_is_avoid_despite_every_positive_signal(result):
    assert result.verdict.action is Action.AVOID


def test_thesis_breakers_lead_with_the_regulator_position(result):
    assert result.verdict.thesis_breakers
    assert "endpoint" in result.verdict.thesis_breakers[0].lower()


def test_report_states_the_killer_before_the_bull_case(result):
    report = render_report(result)
    breakers = report.index("4. What Would Kill The Thesis")
    bull = report.index("7. Bull Case")
    assert breakers < bull
    killer_section = report[breakers:bull]
    assert "endpoint" in killer_section.lower()


def test_user_enthusiasm_did_not_reach_the_judge(result):
    """The user's enthusiasm and position must not have reached any agent."""
    assert result.verdict.judged_blind
    import json

    blob = json.dumps(
        {name: ev.payload for name, ev in result.bus.channels.items()}, default=str
    ).lower()
    assert "tremendously enthusiastic" not in blob
    assert "5000 shares" not in blob
    assert "average cost" not in blob


# --- control: without the killer fact, the system must behave differently ---
def test_without_the_disqualifying_facts_there_is_no_regulatory_kill(repo):
    """Guards against a test that passes because the system is simply pessimistic.

    Removing the three filing facts must remove the regulatory disqualification.
    The verdict can still be cautious -- an UNKNOWN regulator position and a
    thin evidence set are both real reasons to wait -- but the *reason* must
    change, and the kill gate must no longer report a disqualifier.
    """
    control = run(repo, include_disqualifying=False)
    regulatory = control.verdict.kill_gate.by_category(KillCategory.REGULATORY_KILL)

    assert regulatory.level.level < 3, "the kill must come from the facts, not from pessimism"
    assert control.bus.channels[Channel.REGULATORY].payload["endpoint_position"] == "UNKNOWN"
    # The capital kill legitimately survives -- the short runway is in both
    # datasets -- but the REGULATORY disqualification must be gone.
    assert KillCategory.REGULATORY_KILL not in {
        a.category for a in control.verdict.kill_gate.disqualifying
    }
    assert not any(
        "regulator" in breaker.lower() and "already" in breaker.lower()
        for breaker in control.verdict.thesis_breakers
    ), "the 'already broken' finding must depend on the evidence, not on pessimism"


def test_the_two_runs_differ_in_the_way_that_matters(repo):
    """The decisive facts, and only they, produce the disqualification."""
    with_facts = run(repo)
    without_facts = run(repo, include_disqualifying=False)

    assert with_facts.verdict.kill_gate.max_level.level > (
        without_facts.verdict.kill_gate.max_level.level
    )
    assert any("disqualifier" in reason.lower() for reason in with_facts.verdict.reasoning), (
        "only the run containing the decisive fact reports a disqualifier"
    )
    assert (
        with_facts.scorecard.scores["regulatory_quality"]
        < without_facts.scorecard.scores["regulatory_quality"]
    )

    # The cap itself must be tighter, not merely applied to as many dimensions.
    from investment_research.scoring.kill_gate import quality_cap

    assert quality_cap(with_facts.verdict.kill_gate.max_level) < quality_cap(
        without_facts.verdict.kill_gate.max_level
    )


def test_unknown_endpoint_position_is_not_treated_as_good_news(repo):
    control = run(repo, include_disqualifying=False)
    assert any(
        "endpoint" in question.question.lower() and question.blocking
        for question in control.bus.unresolved
    ), "an unverified regulator position must be raised as a blocking question"
