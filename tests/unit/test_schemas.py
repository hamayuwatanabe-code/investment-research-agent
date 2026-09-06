"""Schema validation tests (requirement 17: schema tests)."""

from __future__ import annotations

import pytest

from investment_research.schemas.enums import (
    UNKNOWN,
    EvidenceClass,
    KillLevel,
    Materiality,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.validation import (
    EvaluationLeak,
    SchemaError,
    assert_evaluation_free,
    find_evaluative_language,
    validate_fact,
    validate_source_url,
)
from tests.conftest import make_fact


def test_valid_fact_passes():
    validate_fact(make_fact("Cash was $31.5 million at 2026-06-30"))


def test_confidence_out_of_range_rejected():
    fact = make_fact("Cash was $31.5 million", confidence=1.4)
    with pytest.raises(SchemaError, match="confidence"):
        validate_fact(fact)


def test_empty_claim_rejected():
    with pytest.raises(SchemaError, match="claim"):
        validate_fact(make_fact("   "))


@pytest.mark.parametrize("url", ["", UNKNOWN, "not a url", "javascript:alert(1)"])
def test_malformed_urls_rejected(url):
    with pytest.raises(SchemaError):
        validate_source_url(url)


@pytest.mark.parametrize(
    "url", ["https://www.sec.gov/x", "http://example.com/a", "fixture://sec/X/10-Q"]
)
def test_acceptable_urls(url):
    validate_source_url(url)


def test_company_claim_cannot_be_verified_fact_without_confirmation():
    """A company saying something in a Tier 1 venue is still a company claim."""
    fact = make_fact(
        "The company expects approval in 2027",
        company_claim=True,
        evidence_class=EvidenceClass.VERIFIED_FACT,
        independent_confirmation=False,
    )
    with pytest.raises(SchemaError, match="company claim"):
        validate_fact(fact)


def test_company_claim_verified_when_independently_confirmed():
    fact = make_fact(
        "Cash was $31.5 million",
        company_claim=True,
        evidence_class=EvidenceClass.VERIFIED_FACT,
        independent_confirmation=True,
    )
    validate_fact(fact)


def test_tier4_cannot_alone_establish_a_verified_fact():
    fact = make_fact(
        "A broker set a price target of $18",
        tier=SourceTier.TIER_4,
        url="https://www.tipranks.com/x",
        evidence_class=EvidenceClass.VERIFIED_FACT,
    )
    with pytest.raises(SchemaError, match="cannot alone establish"):
        validate_fact(fact)


def test_bad_date_rejected():
    fact = make_fact("Cash was $31.5 million", event_date="soon")
    with pytest.raises(SchemaError, match="event_date"):
        validate_fact(fact)


def test_unknown_date_accepted():
    validate_fact(make_fact("Cash was $31.5 million", event_date=UNKNOWN))


@pytest.mark.parametrize(
    "text",
    [
        "We recommend buying this name",
        "Explosive Potential 9/10",
        "top pick for 2026",
        "a genuine multi-bagger",
        "この銘柄は買い推奨です",
    ],
)
def test_evaluative_language_detected(text):
    assert find_evaluative_language(text)
    with pytest.raises(EvaluationLeak):
        assert_evaluation_free(text, where="test")


@pytest.mark.parametrize(
    "text",
    [
        "Revenue declined 12% year over year",
        "Cash runway of approximately 9 months",
        "The company reported a net loss of $12.4 million",
        "FDA granted Fast Track designation",
        "Short interest was 16.7 percent of the float",
    ],
)
def test_neutral_facts_are_not_flagged(text):
    """The detector must not fire on ordinary descriptive facts."""
    assert find_evaluative_language(text) == []


def test_kill_level_ordering():
    assert KillLevel.K5.level > KillLevel.K3.level > KillLevel.K0.level
    assert KillLevel.from_level(9) is KillLevel.K5
    assert KillLevel.from_level(-2) is KillLevel.K0


def test_source_tier_ranking_and_primacy():
    assert SourceTier.TIER_1.rank < SourceTier.TIER_5.rank
    assert SourceTier.TIER_1.is_primary and SourceTier.TIER_2.is_primary
    assert not SourceTier.TIER_3.is_primary
    assert SourceTier.UNKNOWN.rank > SourceTier.TIER_5.rank


def test_decision_grade_requires_class_tier_and_status():
    good = make_fact("A regulator stated its position")
    assert good.is_decision_grade

    weak_tier = make_fact(
        "A blogger said something",
        tier=SourceTier.TIER_5,
        url="https://seekingalpha.com/a",
        evidence_class=EvidenceClass.UNVERIFIED_CLAIM,
        verified=VerifiedStatus.INSUFFICIENT_EVIDENCE,
    )
    assert not weak_tier.is_decision_grade

    unverified = make_fact(
        "The company expects approval",
        evidence_class=EvidenceClass.COMPANY_CLAIM,
        verified=VerifiedStatus.NOT_VERIFIED,
        company_claim=True,
    )
    assert not unverified.is_decision_grade


def test_materiality_ordering():
    assert Materiality.CRITICAL.rank > Materiality.HIGH.rank > Materiality.LOW.rank
