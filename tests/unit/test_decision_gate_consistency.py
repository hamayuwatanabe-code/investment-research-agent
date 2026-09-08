"""ResearchStatus.COMPLETE is the only state that may leave an Action set.

Regression for a defect in commit 00d4a2952a5da8d3a99f70d6a4e6bfeefc1ea006:
the pipeline correctly nulled Verdict.action for BLOCKED_PENDING_VERIFICATION,
but a plain INCOMPLETE run (e.g. some unrelated agent degraded) fell through
an `elif` branch that never nulled the action or scrubbed already-rendered
text -- contradicting ResearchStatus's own documented contract ("the single
gate that decides whether Verdict.action may be non-None").
"""

from __future__ import annotations

from investment_research.schemas.enums import (
    Action,
    KillCategory,
    KillConfirmation,
    KillLevel,
    ResearchStatus,
)
from investment_research.schemas.evaluation import KillAssessment, KillGateResult, Verdict
from investment_research.scoring.decision_gate_consistency import (
    enforce_complete_only_action,
    find_action_labels,
)


def _gate(*, confirmed_k5: bool) -> KillGateResult:
    assessment = KillAssessment(
        category=KillCategory.REGULATORY_KILL,
        level=KillLevel.K5,
        rationale="Regulator states the primary endpoint is not sufficient to demonstrate efficacy",
        confirmation=KillConfirmation.CONFIRMED if confirmed_k5 else KillConfirmation.PROVISIONAL,
    )
    return KillGateResult(assessments=(assessment,))


def _verdict(
    *,
    action: Action | None,
    research_status: ResearchStatus,
    headline: str = "",
    confirmed_k5: bool = False,
) -> Verdict:
    return Verdict(
        action=action,
        evidence_confidence=7.0,
        headline=headline or f"... action {action}." if action else "no action yet",
        reasoning=(f"selected action reason: {action}" if action else "",),
        thesis_breakers=(),
        critical_red_flags=(),
        kill_gate=_gate(confirmed_k5=confirmed_k5),
        research_status=research_status,
    )


def test_incomplete_with_prior_buy_nulls_the_action():
    verdict = _verdict(action=Action.BUY, research_status=ResearchStatus.INCOMPLETE)
    enforce_complete_only_action(verdict)
    assert verdict.action is None
    assert find_action_labels(verdict.headline) == []
    assert all(find_action_labels(r) == [] for r in verdict.reasoning)


def test_incomplete_with_prior_wait_for_event_nulls_the_action():
    verdict = _verdict(action=Action.WAIT_FOR_EVENT, research_status=ResearchStatus.INCOMPLETE)
    enforce_complete_only_action(verdict)
    assert verdict.action is None
    assert find_action_labels(verdict.headline) == []
    assert all(find_action_labels(r) == [] for r in verdict.reasoning)


def test_incomplete_with_confirmed_k5_nulls_action_but_keeps_it_prominent():
    verdict = _verdict(
        action=Action.AVOID, research_status=ResearchStatus.INCOMPLETE, confirmed_k5=True
    )
    enforce_complete_only_action(verdict)
    assert verdict.action is None
    assert verdict.kill_gate.max_confirmed_level is KillLevel.K5
    assert any("CONFIRMED_DISQUALIFYING_EVIDENCE_PRESENT" in c for c in verdict.caveats)
    assert any("K5" in c for c in verdict.caveats)
    # The disqualifier caveat itself must not carry a leaked action label.
    assert all(find_action_labels(c) == [] for c in verdict.caveats)


def test_blocked_pending_verification_with_confirmed_k5_also_stays_prominent():
    verdict = _verdict(
        action=Action.AVOID,
        research_status=ResearchStatus.BLOCKED_PENDING_VERIFICATION,
        confirmed_k5=True,
    )
    enforce_complete_only_action(verdict, ["some domain never searched"])
    assert verdict.action is None
    assert any("CONFIRMED_DISQUALIFYING_EVIDENCE_PRESENT" in c for c in verdict.caveats)


def test_complete_with_valid_evidence_still_permits_a_normal_action():
    verdict = _verdict(action=Action.BUY, research_status=ResearchStatus.COMPLETE)
    enforce_complete_only_action(verdict)
    assert verdict.action is Action.BUY, "COMPLETE must not be touched -- an Action is permitted"


def test_a_confirmed_k5_below_the_disqualifier_threshold_gets_no_extra_caveat():
    """Sanity check: the prominent marker is specifically about a CONFIRMED K5
    (or worse), not any confirmed finding at all."""
    assessment = KillAssessment(
        category=KillCategory.CAPITAL_KILL,
        level=KillLevel.K3,
        rationale="Going concern doubt",
        confirmation=KillConfirmation.CONFIRMED,
    )
    verdict = Verdict(
        action=Action.WAIT_FOR_EVENT,
        evidence_confidence=5.0,
        headline="... action WAIT_FOR_EVENT.",
        reasoning=(),
        thesis_breakers=(),
        critical_red_flags=(),
        kill_gate=KillGateResult(assessments=(assessment,)),
        research_status=ResearchStatus.INCOMPLETE,
    )
    enforce_complete_only_action(verdict)
    assert verdict.action is None
    assert not any("CONFIRMED_DISQUALIFYING_EVIDENCE_PRESENT" in c for c in verdict.caveats)
