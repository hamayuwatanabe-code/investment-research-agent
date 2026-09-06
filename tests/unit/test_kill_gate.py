"""Kill gate tests (requirement 5)."""

from __future__ import annotations

import pytest

from investment_research.schemas.agent_io import RiskFlag
from investment_research.schemas.enums import (
    MANDATORY_KILL_CATEGORIES,
    FactCategory,
    KillCategory,
    KillLevel,
    Materiality,
    SourceTier,
)
from investment_research.scoring.kill_gate import evaluate_kill_gate, quality_cap
from tests.conftest import make_fact

ENDPOINT_REJECTED = (
    "FDA stated that it does not consider the proposed primary endpoint appropriate to "
    "establish effectiveness for the intended indication"
)


def test_endpoint_rejection_from_a_primary_source_is_k5():
    gate = evaluate_kill_gate([make_fact(ENDPOINT_REJECTED, category=FactCategory.REGULATORY)], [])
    regulatory = gate.by_category(KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5
    assert gate.max_level is KillLevel.K5


def test_same_claim_from_a_weak_source_is_downgraded():
    """Requirement 2: a Tier 5 rumour cannot disqualify a company on its own."""
    gate = evaluate_kill_gate(
        [
            make_fact(
                ENDPOINT_REJECTED,
                category=FactCategory.REGULATORY,
                tier=SourceTier.TIER_5,
                url="https://seekingalpha.com/a",
            )
        ],
        [],
    )
    assert gate.by_category(KillCategory.REGULATORY_KILL).level is KillLevel.K3


def test_all_mandatory_categories_are_always_reported():
    gate = evaluate_kill_gate([make_fact("Cash was $31.5 million")], [])
    reported = {a.category for a in gate.assessments}
    for category in MANDATORY_KILL_CATEGORIES:
        assert category in reported


def test_unsearched_category_is_labelled_not_examined():
    """K0 for an unsearched category must not read as 'no problem found'."""
    gate = evaluate_kill_gate(
        [make_fact("Cash was $31.5 million")],
        [],
        unsearched_categories=(KillCategory.GOVERNANCE_KILL,),
    )
    assessment = gate.by_category(KillCategory.GOVERNANCE_KILL)
    assert assessment.level is KillLevel.K0
    assert "UNSEARCHED" in assessment.rationale
    assert assessment.evidence_confidence == 0.0


def test_searched_category_with_no_findings_is_a_real_k0():
    gate = evaluate_kill_gate([make_fact("Cash was $31.5 million")], [])
    assessment = gate.by_category(KillCategory.GOVERNANCE_KILL)
    assert assessment.level is KillLevel.K0
    assert "UNSEARCHED" not in assessment.rationale


def test_category_with_a_finding_is_not_reported_as_unsearched():
    """A category that produced a K5 from a filing was, in effect, examined."""
    gate = evaluate_kill_gate(
        [make_fact(ENDPOINT_REJECTED, category=FactCategory.REGULATORY)],
        [],
        unsearched_categories=(KillCategory.REGULATORY_KILL, KillCategory.GOVERNANCE_KILL),
    )
    assert KillCategory.REGULATORY_KILL not in gate.unsearched_categories
    assert KillCategory.GOVERNANCE_KILL in gate.unsearched_categories


@pytest.mark.parametrize(
    "runway,expected",
    [(3.0, KillLevel.K4), (8.0, KillLevel.K3), (15.0, KillLevel.K2), (30.0, KillLevel.K0)],
)
def test_runway_thresholds(runway, expected):
    gate = evaluate_kill_gate([], [], runway_months=runway)
    assert gate.by_category(KillCategory.CAPITAL_KILL).level is expected


@pytest.mark.parametrize(
    "claim,category,level",
    [
        ("The study was placed on clinical hold", KillCategory.CLINICAL_KILL, KillLevel.K5),
        ("The trial did not meet its primary endpoint", KillCategory.CLINICAL_KILL, KillLevel.K5),
        (
            "The company received a Complete Response Letter",
            KillCategory.REGULATORY_KILL,
            KillLevel.K4,
        ),
        (
            "There is substantial doubt about the ability to continue as a going concern",
            KillCategory.CAPITAL_KILL,
            KillLevel.K4,
        ),
        (
            "The company announced a restatement of prior period financial statements",
            KillCategory.ACCOUNTING_KILL,
            KillLevel.K4,
        ),
        (
            "The company received a Nasdaq notice regarding the minimum bid price requirement",
            KillCategory.LIQUIDITY_KILL,
            KillLevel.K3,
        ),
        (
            "The company disclosed an SEC investigation into its disclosures",
            KillCategory.GOVERNANCE_KILL,
            KillLevel.K4,
        ),
    ],
)
def test_individual_kill_rules(claim, category, level):
    gate = evaluate_kill_gate([make_fact(claim)], [])
    assert gate.by_category(category).level is level


def test_benign_facts_produce_no_kill():
    facts = [
        make_fact("Cash and equivalents were $120.0 million at 2026-06-30"),
        make_fact("FDA granted Fast Track designation", category=FactCategory.REGULATORY),
        make_fact("The company reported revenue of $42.1 million"),
    ]
    gate = evaluate_kill_gate(facts, [], runway_months=36.0)
    assert gate.max_level is KillLevel.K0
    assert not gate.major


def test_risk_flags_register_one_level_below_their_severity():
    flag = RiskFlag(
        flag_id="x",
        category=FactCategory.GOVERNANCE,
        title="Board turnover",
        detail="Three directors resigned in one quarter.",
        severity=Materiality.CRITICAL,
    )
    gate = evaluate_kill_gate([], [flag])
    assert gate.by_category(KillCategory.GOVERNANCE_KILL).level is KillLevel.K3


def test_quality_caps_are_monotonic_and_severe():
    caps = [quality_cap(KillLevel.from_level(n)) for n in range(6)]
    assert caps == sorted(caps, reverse=True)
    assert quality_cap(KillLevel.K3) <= 4.5
    assert quality_cap(KillLevel.K5) <= 1.0


def test_disqualifying_and_major_partitions():
    gate = evaluate_kill_gate(
        [
            make_fact(ENDPOINT_REJECTED, category=FactCategory.REGULATORY),
            make_fact(
                "The company received a Nasdaq notice regarding the minimum bid price requirement"
            ),
        ],
        [],
    )
    assert {a.category for a in gate.disqualifying} == {KillCategory.REGULATORY_KILL}
    assert KillCategory.LIQUIDITY_KILL in {a.category for a in gate.major}
