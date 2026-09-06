"""Evidence classification, corroboration and duplicate tests."""

from __future__ import annotations

import pytest

from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import (
    EvidenceClass,
    Materiality,
    SourceTier,
    VerifiedStatus,
)
from tests.conftest import TODAY, make_fact


def run(facts):
    """Assess facts and return them keyed by fact_id (claims can collide)."""
    agent = EvidenceIntegrityAgent(today=TODAY)
    data = AgentInput(
        agent_id="evidence_integrity",
        run_id="r",
        ticker="TEST",
        company_name="Test Co",
        facts=tuple(facts),
    )
    return list(agent.run(data).facts)


def one(facts):
    assessed = run(facts)
    assert len(assessed) == 1
    return assessed[0]


def by_claim(assessed):
    return {f.claim: f for f in assessed}


def test_company_claim_stays_a_company_claim_without_confirmation():
    """Filed in a Tier 1 venue is not the same as independently confirmed."""
    fact = one([make_fact("The company expects approval in 2027", company_claim=True)])
    assert fact.evidence_class is EvidenceClass.COMPANY_CLAIM
    assert fact.verified_status is VerifiedStatus.NOT_VERIFIED
    assert not fact.independent_confirmation


def test_company_claim_upgraded_by_an_independent_primary_source():
    claim = "The trial enrolled 84 participants"
    facts = run(
        [
            make_fact(claim, company_claim=True, url="https://www.sec.gov/a"),
            make_fact(
                claim,
                company_claim=False,
                url="https://clinicaltrials.gov/study/NCT1",
                tier=SourceTier.TIER_1,
            ),
        ]
    )
    company_version = [f for f in facts if f.company_claim][0]
    assert company_version.independent_confirmation
    assert company_version.evidence_class is EvidenceClass.VERIFIED_FACT


def test_reprints_do_not_create_independent_confirmation():
    """Requirement 13: same story, five outlets, still one source."""
    claim = "The data monitoring committee recommended that the study continue"
    facts = run(
        [
            make_fact(
                claim,
                company_claim=True,
                url="https://www.globenewswire.com/a",
                tier=SourceTier.TIER_2,
            ),
            make_fact(
                claim, company_claim=True, url="https://finance.yahoo.com/a", tier=SourceTier.TIER_5
            ),
        ]
    )
    for fact in facts:
        assert not fact.independent_confirmation, "company reprints are not confirmation"


def test_analyst_source_becomes_an_opinion_with_low_confidence():
    fact = one(
        [
            make_fact(
                "A broker maintains a price target of $18.00",
                tier=SourceTier.TIER_4,
                url="https://www.tipranks.com/x",
            )
        ]
    )
    assert fact.evidence_class is EvidenceClass.ANALYST_OPINION
    assert fact.verified_status is VerifiedStatus.INSUFFICIENT_EVIDENCE
    assert fact.confidence <= 0.25
    assert not fact.is_decision_grade


def test_social_source_is_unverified():
    fact = one(
        [
            make_fact(
                "A poster claims a partnership is imminent",
                tier=SourceTier.TIER_5,
                url="https://www.reddit.com/r/x",
            )
        ]
    )
    assert fact.evidence_class is EvidenceClass.UNVERIFIED_CLAIM


@pytest.mark.parametrize(
    "claim",
    [
        "There is substantial doubt about the ability to continue as a going concern",
        "The study was placed on clinical hold",
        "FDA does not consider the endpoint appropriate to establish effectiveness",
        "The company disclosed a material weakness in internal control",
    ],
)
def test_material_phrases_raise_materiality_to_critical(claim):
    assert one([make_fact(claim)]).materiality is Materiality.CRITICAL


def test_confidence_ordering_follows_tier():
    facts = by_claim(
        run(
            [
                make_fact("Claim one", tier=SourceTier.TIER_1, url="https://www.sec.gov/1"),
                make_fact("Claim two", tier=SourceTier.TIER_3, url="https://www.reuters.com/2"),
                make_fact("Claim three", tier=SourceTier.TIER_5, url="https://seekingalpha.com/3"),
            ]
        )
    )
    assert (
        facts["Claim one"].confidence
        > facts["Claim two"].confidence
        > facts["Claim three"].confidence
    )


def test_missing_event_date_is_noted_not_filled_in():
    from investment_research.schemas.enums import UNKNOWN

    fact = one([make_fact("A claim", event_date=UNKNOWN)])
    assert fact.event_date == UNKNOWN
    assert "publication date must not be read as the event date" in fact.notes
