"""Programme/thesis-relevance resolution (requirement D).

A live run selected an unrelated historical Phase 2 trial for Science's
design assessment and let a terminated/withdrawn trial in a different
indication drive a company-level kill finding, simply because it was *a*
trial in the evidence, not because it was the thesis-relevant current one.
These tests hold `resolve_current_program` to the evidence-only rule: never
pick the first Phase 2/3 trial encountered, never guess when the evidence
does not resolve to one programme.
"""

from __future__ import annotations

from investment_research.schemas.enums import FactCategory
from investment_research.scoring.program_resolution import resolve_current_program
from tests.conftest import make_fact

OLD_TRIAL = "NCT10000001"
CURRENT_TRIAL = "NCT20000002"


def test_a_single_clinical_programme_is_trivially_resolved():
    facts = [
        make_fact(
            f"{OLD_TRIAL} phase is Phase 2",
            category=FactCategory.CLINICAL,
            value="Phase 2",
        ),
        make_fact(
            f"{OLD_TRIAL} overall status is ACTIVE_NOT_RECRUITING",
            category=FactCategory.CLINICAL,
            value="ACTIVE_NOT_RECRUITING",
        ),
    ]
    resolution = resolve_current_program(facts)
    assert resolution.trial_id == OLD_TRIAL
    assert resolution.relevance_unresolved is False


def test_explicit_lead_marker_wins_over_an_older_unrelated_terminated_trial():
    """The old trial (indication A) is TERMINATED; the current lead trial
    (indication B) is explicitly described as such and remains ACTIVE. The
    old, unrelated trial must never be picked as the current programme."""
    facts = [
        make_fact(
            f"{OLD_TRIAL} phase is Phase 2",
            category=FactCategory.CLINICAL,
            value="Phase 2",
            event_date="2019-01-01",
        ),
        make_fact(
            f"{OLD_TRIAL} overall status is TERMINATED",
            category=FactCategory.CLINICAL,
            value="TERMINATED",
            event_date="2019-06-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} phase is Phase 3",
            category=FactCategory.CLINICAL,
            value="Phase 3",
            event_date="2026-01-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} overall status is RECRUITING",
            category=FactCategory.CLINICAL,
            value="RECRUITING",
            event_date="2026-06-01",
        ),
        make_fact(
            f"Generic Biotech Holdings describes {CURRENT_TRIAL} as its current lead "
            "registrational programme",
            category=FactCategory.CLINICAL,
            company_claim=True,
            event_date="2026-06-01",
        ),
    ]
    resolution = resolve_current_program(facts)
    assert resolution.trial_id == CURRENT_TRIAL
    assert resolution.relevance_unresolved is False


def test_a_genuinely_terminated_current_lead_trial_is_still_resolved_as_current():
    """Status does not disqualify a trial from being the current programme --
    only relevance does. A trial explicitly marked lead/current that later
    terminated is still the current programme, not swapped out."""
    facts = [
        make_fact(
            f"{CURRENT_TRIAL} phase is Phase 3",
            category=FactCategory.CLINICAL,
            value="Phase 3",
        ),
        make_fact(
            f"{CURRENT_TRIAL} overall status is TERMINATED",
            category=FactCategory.CLINICAL,
            value="TERMINATED",
        ),
        make_fact(
            f"Generic Biotech Holdings describes {CURRENT_TRIAL} as its current lead "
            "registrational programme",
            category=FactCategory.CLINICAL,
            company_claim=True,
        ),
        make_fact(
            f"{OLD_TRIAL} phase is Phase 2",
            category=FactCategory.CLINICAL,
            value="Phase 2",
        ),
        make_fact(
            f"{OLD_TRIAL} overall status is COMPLETED",
            category=FactCategory.CLINICAL,
            value="COMPLETED",
        ),
    ]
    resolution = resolve_current_program(facts)
    assert resolution.trial_id == CURRENT_TRIAL
    assert resolution.status == "TERMINATED"


def test_recency_breaks_a_tie_when_no_explicit_lead_marker_exists():
    facts = [
        make_fact(
            f"{OLD_TRIAL} overall status is COMPLETED",
            category=FactCategory.CLINICAL,
            value="COMPLETED",
            event_date="2020-01-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} overall status is RECRUITING",
            category=FactCategory.CLINICAL,
            value="RECRUITING",
            event_date="2026-06-01",
        ),
    ]
    resolution = resolve_current_program(facts)
    assert resolution.trial_id == CURRENT_TRIAL
    assert resolution.relevance_unresolved is False


def test_ambiguous_multiple_programmes_with_no_signal_are_left_unresolved():
    """No lead marker, no distinguishing recency -- never guess."""
    facts = [
        make_fact(
            f"{OLD_TRIAL} overall status is ACTIVE_NOT_RECRUITING",
            category=FactCategory.CLINICAL,
            value="ACTIVE_NOT_RECRUITING",
            event_date="2026-01-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} overall status is ACTIVE_NOT_RECRUITING",
            category=FactCategory.CLINICAL,
            value="ACTIVE_NOT_RECRUITING",
            event_date="2026-01-01",
        ),
    ]
    resolution = resolve_current_program(facts)
    assert resolution.trial_id == "UNKNOWN"
    assert resolution.relevance_unresolved is True
    assert len(resolution.candidates) == 2


def test_no_trial_identifier_at_all_is_unresolved_not_an_error():
    resolution = resolve_current_program(
        [make_fact("The company has a mechanism of action.", category=FactCategory.SCIENCE)]
    )
    assert resolution.trial_id == "UNKNOWN"
    assert resolution.relevance_unresolved is True
    assert resolution.candidates == ()
