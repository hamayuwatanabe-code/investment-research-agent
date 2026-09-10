"""source_routing.py: the pure LOCATE/FETCH/PARSE step-DAG functions.

No network, no execution -- these feed hypothetical outcome maps directly.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    FailurePolicy,
    RequirementPriorityTier,
    StepKind,
    StepStatus,
    TargetKind,
    is_target_complete,
    priority_tier_for,
    resolve_next_steps,
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


def test_not_publicly_available_single_step_target_completes_without_search():
    target_id = "t3"
    step = AcquisitionStep(
        step_id="npa", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    target = AcquisitionTarget(target_id=target_id, target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE, required_step_ids=("npa",))
    assert [s.step_id for s in resolve_next_steps(target, [step], {})] == ["npa"]
    outcomes = {"npa": StepStatus.NOT_PUBLICLY_AVAILABLE}
    assert is_target_complete(target, [step], outcomes)
    assert resolve_next_steps(target, [step], outcomes) == []
    assert not step.acquisition_method.requires_web_search_budget


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
