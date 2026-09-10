"""source_routing.py: the pure LOCATE/FETCH/PARSE step-DAG functions.

No network, no execution -- these feed hypothetical outcome maps directly.
"""

from __future__ import annotations

import dataclasses

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    ImplementationStatus,
    PlanStatus,
    RequirementPriorityTier,
    SimulationAssumption,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetAcquisitionOutcome,
    TargetKind,
    compute_plan_status,
    is_target_complete,
    priority_tier_for,
    resolve_next_steps,
    target_acquisition_outcome,
    target_plan_status,
)
from investment_research.schemas.enums import ResearchDomain


def _sec_style_target(target_id: str = "t1"):
    l1 = AcquisitionStep(
        step_id="l1", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    l2 = AcquisitionStep(
        step_id="l2", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    f = AcquisitionStep(
        step_id="f", target_id=target_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=("l1", "l2"), completion_condition=StepStatus.BODY_FETCHED,
    )
    p = AcquisitionStep(
        step_id="p", target_id=target_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=("f",), completion_condition=StepStatus.PARSED,
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=("f", "p"), alternative_step_groups=(("l1", "l2"),),
    )
    return target, [l1, l2, f, p]


def test_first_offered_step_is_the_direct_locate():
    target, steps = _sec_style_target()
    next_steps = resolve_next_steps(target, steps, {})
    assert [s.step_id for s in next_steps] == ["l1"]


def test_locate_success_alone_never_completes_the_target():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.URL_RESOLVED}
    assert not is_target_complete(target, steps, outcomes)


def test_locate_success_offers_fetch_next_not_web_alternative():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.URL_RESOLVED}
    next_steps = resolve_next_steps(target, steps, outcomes)
    assert [s.step_id for s in next_steps] == ["f"]


def test_body_fetch_alone_still_incomplete_until_parsed():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.URL_RESOLVED, "f": StepStatus.BODY_FETCHED}
    assert not is_target_complete(target, steps, outcomes)
    next_steps = resolve_next_steps(target, steps, outcomes)
    assert [s.step_id for s in next_steps] == ["p"]


def test_fetch_and_parse_together_complete_the_target():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.URL_RESOLVED, "f": StepStatus.BODY_FETCHED, "p": StepStatus.PARSED}
    assert is_target_complete(target, steps, outcomes)


def test_direct_locate_failure_activates_web_locate_only():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.ZERO_RESULTS}
    next_steps = resolve_next_steps(target, steps, outcomes)
    assert [s.step_id for s in next_steps] == ["l2"]


def test_web_locate_success_still_requires_its_own_fetch():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.ZERO_RESULTS, "l2": StepStatus.URL_RESOLVED}
    assert not is_target_complete(target, steps, outcomes)
    next_steps = resolve_next_steps(target, steps, outcomes)
    assert [s.step_id for s in next_steps] == ["f"]


def test_full_web_fallback_chain_completes_the_target():
    target, steps = _sec_style_target()
    outcomes = {
        "l1": StepStatus.ZERO_RESULTS,
        "l2": StepStatus.URL_RESOLVED,
        "f": StepStatus.BODY_FETCHED,
        "p": StepStatus.PARSED,
    }
    assert is_target_complete(target, steps, outcomes)


def test_downstream_step_depends_on_the_specific_alternative_it_cites():
    """A step that names only ONE member of an alt-group depends on that
    member specifically -- it is not satisfied by the group's OTHER member
    succeeding."""
    target_id = "t2"
    s1 = AcquisitionStep(
        step_id="s1", target_id=target_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
        completion_condition=StepStatus.STRUCTURED_RECORD_RETRIEVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    lw = AcquisitionStep(
        step_id="lw", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    sp = AcquisitionStep(
        step_id="sp", target_id=target_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
        depends_on_step_ids=("s1",), completion_condition=StepStatus.REQUIRED_FIELDS_PARSED,
        failure_policy=FailurePolicy.ALTERNATIVE,
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.CLINICALTRIALS_RECORD,
        alternative_step_groups=(("s1", "lw"), ("sp",)),
    )
    # lw (not s1) succeeded -- sp specifically depends on s1, so it must NOT
    # be offered.
    outcomes = {"s1": StepStatus.ZERO_RESULTS, "lw": StepStatus.URL_RESOLVED}
    next_steps = resolve_next_steps(target, [s1, lw, sp], outcomes)
    assert "sp" not in [s.step_id for s in next_steps]


def test_not_publicly_available_single_step_target_never_reaches_search_and_never_completes():
    """Phase 2.7 correction: NOT_PUBLICLY_AVAILABLE is a conclusive
    non-acquisition resolution, never a completion. It still terminates the
    chain (nothing further is offered) and never sends to web search."""
    target_id = "t3"
    step = AcquisitionStep(
        step_id="npa", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    target = AcquisitionTarget(target_id=target_id, target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE, required_step_ids=("npa",))
    assert [s.step_id for s in resolve_next_steps(target, [step], {})] == ["npa"]
    outcomes = {"npa": StepStatus.NOT_PUBLICLY_AVAILABLE}
    assert not is_target_complete(target, [step], outcomes)
    assert resolve_next_steps(target, [step], outcomes) == []
    assert not step.acquisition_method.requires_web_search_budget
    assert step.never_constitutes_content_acquisition


def test_an_undeclared_outcome_is_conservatively_treated_as_a_dead_end():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.SKIPPED_DUE_TO_BUDGET}
    # l1's own completion_condition (URL_RESOLVED) was not reached, and
    # SKIPPED_DUE_TO_BUDGET is not in l1's terminal set either -- the
    # alt-group should still offer l2 (the untried member), never invent an
    # undeclared behavior.
    next_steps = resolve_next_steps(target, steps, outcomes)
    assert [s.step_id for s in next_steps] == ["l2"]


def test_priority_tier_for_blocking_regulatory_is_top():
    assert priority_tier_for(domain=ResearchDomain.REGULATORY, blocking_if_unresolved=True) is RequirementPriorityTier.BLOCKING_REGULATORY
    assert priority_tier_for(domain=ResearchDomain.REGULATORY, blocking_if_unresolved=False) is not RequirementPriorityTier.BLOCKING_REGULATORY
    assert priority_tier_for(domain=ResearchDomain.CONTRADICTION, blocking_if_unresolved=True) is RequirementPriorityTier.CONTRADICTION_FALSIFICATION
    assert priority_tier_for(domain=ResearchDomain.SCIENCE_TECHNOLOGY, blocking_if_unresolved=False) is RequirementPriorityTier.CURRENT_PROGRAM_SCIENCE
    assert priority_tier_for(domain=ResearchDomain.CAPITAL_STRUCTURE, blocking_if_unresolved=False) is RequirementPriorityTier.CAPITAL_SURVIVAL
    assert priority_tier_for(domain=ResearchDomain.CATALYST, blocking_if_unresolved=False) is RequirementPriorityTier.REMAINING_GAPS
    ranks = [t.rank for t in RequirementPriorityTier]
    assert ranks == sorted(ranks)


def test_target_defaults_are_safe():
    target = AcquisitionTarget(target_id="t", target_kind=TargetKind.GENERIC_WEB_DOCUMENT)
    assert target.required_step_ids == ()
    assert target.alternative_step_groups == ()
    assert target.serves_requirement_ids == ()


def test_subject_scope_import_smoke():
    # SubjectScope is reused unmodified from checks.py in this new module --
    # confirm the import path still resolves.
    assert SubjectScope.COMPANY.value == "COMPANY"


# =========================== Phase 2.7 additions ============================
def _npa_target(target_id: str = "npa_target"):
    step = AcquisitionStep(
        step_id="npa", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    target = AcquisitionTarget(target_id=target_id, target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE, required_step_ids=("npa",))
    return target, [step]


def test_target_acquisition_outcome_not_attempted_when_no_outcomes_recorded():
    target, steps = _sec_style_target()
    assert target_acquisition_outcome(target, steps, {}) is TargetAcquisitionOutcome.NOT_ATTEMPTED


def test_target_acquisition_outcome_in_progress_when_only_locate_succeeded():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.URL_RESOLVED}
    assert target_acquisition_outcome(target, steps, outcomes) is TargetAcquisitionOutcome.IN_PROGRESS


def test_target_acquisition_outcome_acquired_only_after_full_chain():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.URL_RESOLVED, "f": StepStatus.BODY_FETCHED, "p": StepStatus.PARSED}
    assert target_acquisition_outcome(target, steps, outcomes) is TargetAcquisitionOutcome.ACQUIRED


def test_target_acquisition_outcome_exhausted_not_public_never_acquired():
    target, steps = _npa_target()
    outcomes = {"npa": StepStatus.NOT_PUBLICLY_AVAILABLE}
    outcome = target_acquisition_outcome(target, steps, outcomes)
    assert outcome is TargetAcquisitionOutcome.ACQUISITION_EXHAUSTED_NOT_PUBLIC
    assert outcome is not TargetAcquisitionOutcome.ACQUIRED


def test_target_acquisition_outcome_blocked_by_budget():
    target, steps = _sec_style_target()
    outcomes = {"l1": StepStatus.ZERO_RESULTS, "l2": StepStatus.SKIPPED_DUE_TO_BUDGET}
    assert target_acquisition_outcome(target, steps, outcomes) is TargetAcquisitionOutcome.BLOCKED_BY_BUDGET


def test_target_acquisition_outcome_not_implemented_when_exhausted_via_unimplemented_step():
    target_id = "t_ni"
    l1 = AcquisitionStep(
        step_id="l1", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.NOT_IMPLEMENTED,
    )
    target = AcquisitionTarget(target_id=target_id, target_kind=TargetKind.FORM4_FILING, required_step_ids=("l1",))
    outcomes = {"l1": StepStatus.FAILED}
    assert target_acquisition_outcome(target, [l1], outcomes) is TargetAcquisitionOutcome.NOT_IMPLEMENTED


# --- ImplementationStatus / PlanStatus --------------------------------------
def test_target_plan_status_infeasible_when_required_step_not_implemented_with_no_alternative():
    target_id = "t"
    step = AcquisitionStep(
        step_id="s", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        completion_condition=StepStatus.URL_RESOLVED,
        implementation_status=ImplementationStatus.NOT_IMPLEMENTED,
    )
    target = AcquisitionTarget(target_id=target_id, target_kind=TargetKind.FORM4_FILING, required_step_ids=("s",))
    assert target_plan_status(target, [step]) is PlanStatus.INFEASIBLE


def test_target_plan_status_feasible_when_alternative_exists():
    target_id = "t"
    unimplemented = AcquisitionStep(
        step_id="s1", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
        implementation_status=ImplementationStatus.NOT_IMPLEMENTED,
    )
    web = AcquisitionStep(
        step_id="s2", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    target = AcquisitionTarget(target_id=target_id, target_kind=TargetKind.FORM4_FILING, alternative_step_groups=(("s1", "s2"),))
    assert target_plan_status(target, [unimplemented, web]) is PlanStatus.FEASIBLE


def test_target_plan_status_infeasible_for_not_publicly_available_only_target():
    target, steps = _npa_target()
    # Declaring a route is IMPLEMENTED-trivial here, but the method itself
    # never constitutes content acquisition -- INFEASIBLE regardless.
    assert target_plan_status(target, steps) is PlanStatus.INFEASIBLE


def test_compute_plan_status_reports_blocking_gaps_for_non_public_blocking_requirement():
    target, steps = _npa_target()
    requirement = EvidenceRequirement(
        requirement_id="req_npa", serves_legacy_need_ids=("synthetic",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.REGULATORY,
        blocking_if_unresolved=True,
    )
    target = dataclasses.replace(target, serves_requirement_ids=("req_npa",))
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=steps)
    status, blocking = compute_plan_status(graph)
    assert status is PlanStatus.FEASIBLE_WITH_BLOCKING_GAPS
    assert blocking == 1


def test_simulation_assumption_respects_implementation_status_flag():
    assert SimulationAssumption.ASSUME_IMPLEMENTED_ROUTES_SUCCEED.respects_implementation_status is False
    assert SimulationAssumption.CURRENT_IMPLEMENTATION_ONLY.respects_implementation_status is True
    assert SimulationAssumption.DIRECT_FAILURES.respects_implementation_status is True
    assert SimulationAssumption.WORST_CASE.respects_implementation_status is True
