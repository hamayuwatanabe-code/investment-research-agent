"""Phase 4.3L: the central programme resolver must not prove a "current"
programme from a stale lead/pivotal description or from a later
(planned) primary-completion date -- because a wrong choice downgrades a
real primary-endpoint failure on ANOTHER trial to a K1 "DIFFERENT
PROGRAMME" finding, which no longer blocks the Action.

All trial identifiers and claim wording here are SYNTHETIC. They are shaped
like the collectors' output (registry facts carry the primary completion
date as ``event_date``) and like an issuer's 8-K/EX-99 sentence (a
``COMPANY_CLAIM``), but they are not text from any real filing.
"""

from __future__ import annotations

from investment_research.research.provider import DomainCoverage
from investment_research.schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    EvidenceClass,
    FactCategory,
    KillCategory,
    KillConfirmation,
    KillLevel,
    SearchStatus,
)
from investment_research.scoring.completeness import CompletenessResult
from investment_research.scoring.evidence_sufficiency import assess_evidence_sufficiency
from investment_research.scoring.kill_gate import evaluate_kill_gate
from investment_research.scoring.program_resolution import resolve_current_program
from tests.conftest import make_fact

HISTORIC = "NCT10000001"  # an old, completed programme
FAILED = "NCT20000002"  # the programme whose primary endpoint was just missed
NEWER = "NCT30000003"  # a registry record with a LATER (planned) completion date
C = FactCategory.CLINICAL


def _registry(nct: str, status: str, primary_completion: str):
    url = f"https://clinicaltrials.gov/study/{nct}"
    return [
        make_fact(f"{nct} phase is PHASE2", category=C, value="PHASE2",
                  event_date=primary_completion, url=url),
        make_fact(f"{nct} overall status is {status}", category=C, value=status,
                  event_date=primary_completion, url=url),
    ]


def _failure(nct: str = FAILED, *, date: str = "2026-09-16"):
    """An issuer's own sentence: COMPANY_CLAIM, not decision-grade."""
    return make_fact(
        f"The {nct} trial did not meet its primary endpoint.",
        category=C, evidence_class=EvidenceClass.COMPANY_CLAIM, company_claim=True,
        event_date=date, publication_date=date, url="https://www.sec.gov/synthetic-ex99-1",
    )


def _marker(nct: str, *, date: str, word: str = "lead"):
    return make_fact(
        f"The company describes {nct} as its {word} programme.",
        category=C, company_claim=True, event_date=date, publication_date=date,
        url=f"https://www.sec.gov/synthetic-marker-{nct}-{date}",
    )


def _strategy(*, date: str):
    return make_fact(
        "The company has initiated a strategic review and is exploring strategic alternatives.",
        category=FactCategory.OTHER, evidence_class=EvidenceClass.COMPANY_CLAIM,
        company_claim=True, event_date=date, publication_date=date,
        url="https://www.sec.gov/synthetic-strategy",
    )


_BASE = (
    _registry(HISTORIC, "COMPLETED", "2019-12-31")
    + _registry(FAILED, "ACTIVE_NOT_RECRUITING", "2026-06-30")
)


def _clinical_findings(facts, resolution):
    current = "UNKNOWN" if resolution.relevance_unresolved else resolution.trial_id
    gate = evaluate_kill_gate(
        list(facts), [], current_program_trial_id=current,
        program_relevance_unresolved=resolution.relevance_unresolved
        and len(resolution.candidates) > 1,
    )
    clinical = next(a for a in gate.assessments if a.category == KillCategory.CLINICAL_KILL)
    failed = [f for f in clinical.findings if f.title == "Trial missed its primary endpoint"]
    return gate, clinical, failed


def _assert_failure_stays_a_blocking_k5(facts, resolution):
    gate, clinical, failed = _clinical_findings(facts, resolution)
    assert failed, "the failure fact must produce a trial_failed finding"
    assert all(f.level is KillLevel.K5 for f in failed), [f.level for f in failed]
    assert all(f.confirmation is KillConfirmation.PROVISIONAL for f in failed)
    assert not any("DIFFERENT PROGRAMME" in f.detail for f in failed)
    assert clinical.level is KillLevel.K5
    assert clinical in gate.provisional_major_or_worse  # what the Action Gate reads
    return gate


# --- the reproduced misselections ----------------------------------------------
def test_stale_lead_marker_on_an_old_trial_no_longer_resolves_after_a_later_failure():
    facts = [*_BASE, _marker(HISTORIC, date="2019-01-01"), _failure()]
    resolution = resolve_current_program(facts)
    assert resolution.relevance_unresolved is True
    assert resolution.trial_id == "UNKNOWN"
    assert "superseded" in resolution.rationale
    _assert_failure_stays_a_blocking_k5(facts, resolution)


def test_later_planned_completion_date_alone_never_picks_the_current_programme():
    facts = [*_BASE, *_registry(NEWER, "NOT_YET_RECRUITING", "2028-12-31"), _failure()]
    resolution = resolve_current_program(facts)
    assert resolution.relevance_unresolved is True
    assert resolution.trial_id == "UNKNOWN"
    _assert_failure_stays_a_blocking_k5(facts, resolution)


def test_stale_pivotal_marker_on_the_failed_trial_itself_is_superseded():
    facts = [*_BASE, _marker(FAILED, date="2025-06-24", word="pivotal"), _failure()]
    resolution = resolve_current_program(facts)
    assert resolution.relevance_unresolved is True
    _assert_failure_stays_a_blocking_k5(facts, resolution)


def test_a_strategic_review_after_the_marker_supersedes_it():
    facts = [*_BASE, _marker(FAILED, date="2025-06-24"), _strategy(date="2026-09-16")]
    resolution = resolve_current_program(facts)
    assert resolution.relevance_unresolved is True
    assert "superseded" in resolution.rationale


def test_an_undated_superseding_disclosure_cannot_be_placed_and_supersedes():
    undated = make_fact(
        "The company discontinued development of the programme.",
        category=C, company_claim=True, event_date="UNKNOWN", publication_date="UNKNOWN",
        url="https://www.sec.gov/synthetic-undated",
    )
    facts = [*_BASE, _marker(FAILED, date="2026-01-01"), undated]
    assert resolve_current_program(facts).relevance_unresolved is True


# --- behaviour that must not change ---------------------------------------------
def test_a_single_programme_is_still_resolved_and_its_failure_is_a_blocking_k5():
    facts = [*_registry(FAILED, "ACTIVE_NOT_RECRUITING", "2026-06-30"), _failure()]
    resolution = resolve_current_program(facts)
    assert resolution.relevance_unresolved is False
    assert resolution.trial_id == FAILED
    _assert_failure_stays_a_blocking_k5(facts, resolution)


def test_a_clear_current_marker_with_no_superseding_disclosure_still_resolves():
    facts = [*_BASE, _marker(FAILED, date="2026-06-01")]
    resolution = resolve_current_program(facts)
    assert resolution.relevance_unresolved is False
    assert resolution.trial_id == FAILED


def test_a_pivot_marker_after_an_old_failure_keeps_the_old_failure_out_of_scope():
    """The company's lead programme was named AFTER the old trial failed:
    that marker is not stale, so the old failure stays a historical K1."""
    facts = [
        *_BASE,
        _failure(HISTORIC, date="2024-03-01"),
        _marker(FAILED, date="2025-01-15"),
    ]
    resolution = resolve_current_program(facts)
    assert resolution.relevance_unresolved is False
    assert resolution.trial_id == FAILED
    _, _, failed = _clinical_findings(facts, resolution)
    assert failed and all(f.level is KillLevel.K1 for f in failed)
    assert all("DIFFERENT PROGRAMME" in f.detail for f in failed)


# --- propagation to the Action Gate's own input ---------------------------------
def test_the_k5_reaches_the_evidence_sufficiency_matrix_and_blocks_the_action():
    facts = [*_BASE, _marker(HISTORIC, date="2019-01-01"), _failure()]
    resolution = resolve_current_program(facts)
    gate = _assert_failure_stays_a_blocking_k5(facts, resolution)
    matrix = assess_evidence_sufficiency(
        facts=facts,
        completeness=CompletenessResult(
            coverage={
                d: DomainCoverage(domain=d, status=SearchStatus.SEARCHED)
                for d in REQUIRED_RESEARCH_DOMAINS
            }
        ),
        gate=gate,
    )
    assert any(
        str(KillCategory.CLINICAL_KILL) in f and str(KillLevel.K5) in f
        for f in matrix.provisional_kill_findings
    ), matrix.provisional_kill_findings
    assert matrix.sufficient is False
