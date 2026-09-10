"""routing_budget_scenarios.py: bounded NORMAL/DEGRADED/WORST scenarios.

Covers Phase 2.6 requirement 7 (NORMAL's real feasibility, not just tokens)
and requirement 8 (the run-level MAX_WEB_SEARCH_USES=3 cap with priority
selection, never unbounded search). No network call.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.routing_budget_scenarios import (
    MAX_WEB_SEARCH_USES,
    assess_budget_scenarios,
)
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetKind,
)
from investment_research.schemas.enums import ResearchDomain, ResearchStatus


def _sec_style(n: int, *, domain=ResearchDomain.CAPITAL_STRUCTURE, blocking=False):
    tid, rid = f"target_{n}", f"req_{n}"
    l1 = AcquisitionStep(
        step_id=f"l1_{n}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    l2 = AcquisitionStep(
        step_id=f"l2_{n}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    f = AcquisitionStep(
        step_id=f"f_{n}", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=(f"l1_{n}", f"l2_{n}"), completion_condition=StepStatus.BODY_FETCHED,
    )
    p = AcquisitionStep(
        step_id=f"p_{n}", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=(f"f_{n}",), completion_condition=StepStatus.PARSED,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{n}",), subject_scope=SubjectScope.COMPANY,
        domain=domain, blocking_if_unresolved=blocking,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=(f"f_{n}", f"p_{n}"), alternative_step_groups=((f"l1_{n}", f"l2_{n}"),),
        serves_requirement_ids=(rid,),
    )
    return requirement, target, [l1, l2, f, p]


def _graph_of(n_targets: int, **kwargs) -> SourceRoutingGraph:
    parts = [_sec_style(i, **kwargs) for i in range(n_targets)]
    return SourceRoutingGraph(
        requirements=tuple(p[0] for p in parts),
        targets=tuple(p[1] for p in parts),
        steps=tuple(s for p in parts for s in p[2]),
    )


def test_scenarios_are_computed_separately():
    scenarios = assess_budget_scenarios()
    assert scenarios.normal.scenario == "NORMAL"
    assert scenarios.degraded.scenario == "DEGRADED"
    assert scenarios.worst.scenario == "WORST"


# --- requirement 7: NORMAL's completeness, not just tokens ------------------
def test_normal_direct_success_means_direct_http_fetch_not_zero():
    graph = _graph_of(1)
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.normal.direct_http_body_fetches == 1
    assert scenarios.normal.web_search_locator_uses == 0
    assert scenarios.normal.parsed_full_documents == 1
    assert scenarios.normal.incomplete_targets == 0


def test_normal_safety_gate_requires_completion_not_just_token_fit():
    """A scenario with cheap tokens but an incomplete required target must
    NOT be reported feasible via normal_meets_safety_gate."""
    graph = _graph_of(1)
    # Force incompleteness: corrupt one step's dependency to something that
    # can never be satisfied, by using a target with a step that never gets
    # offered (dangling dependency).
    dangling_step = AcquisitionStep(
        step_id="dangling", target_id="target_0", step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=("does_not_exist",), completion_condition=StepStatus.PARSED,
    )
    broken_target = AcquisitionTarget(
        target_id="target_0", target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=("dangling",),
        serves_requirement_ids=graph.targets[0].serves_requirement_ids,
    )
    broken_graph = SourceRoutingGraph(
        requirements=graph.requirements, targets=(broken_target,), steps=(dangling_step,)
    )
    scenarios = assess_budget_scenarios(broken_graph)
    assert scenarios.normal.incomplete_targets == 1
    assert scenarios.normal_meets_safety_gate is False


def test_normal_metadata_only_is_never_reported_as_a_body_fetch():
    graph = _graph_of(1)
    scenarios = assess_budget_scenarios(graph)
    # metadata_locator_requests counts a successful LOCATE; it is a distinct
    # field from direct_http_body_fetches, never conflated.
    assert scenarios.normal.metadata_locator_requests == 1
    assert scenarios.normal.direct_http_body_fetches == 1


# --- requirement 8: hard cap, priority selection, bounded degradation ------
def test_degraded_never_exceeds_the_hard_cap():
    graph = _graph_of(10)  # 10 independent targets, each needing a fallback under DEGRADED
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.degraded.web_search_locator_uses <= MAX_WEB_SEARCH_USES
    assert scenarios.degraded.executed_web_search_count == MAX_WEB_SEARCH_USES


def test_worst_never_exceeds_the_hard_cap_either():
    graph = _graph_of(10)
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.worst.web_search_locator_uses <= MAX_WEB_SEARCH_USES


def test_fourth_and_beyond_targets_are_recorded_unexecuted_not_deleted():
    graph = _graph_of(5)
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.degraded.unexecuted_web_search_count == 5 - MAX_WEB_SEARCH_USES
    assert len(graph.targets) == 5  # nothing removed from the graph itself


def test_blocking_unexecuted_is_reported_when_a_blocking_target_misses_the_cap():
    # 5 blocking targets compete for only 3 slots.
    graph = _graph_of(5, domain=ResearchDomain.REGULATORY, blocking=True)
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.degraded.blocking_unexecuted_count == 2


def test_completion_is_required_for_blocking_not_just_reduced_count():
    graph = _graph_of(5, domain=ResearchDomain.REGULATORY, blocking=True)
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.degraded.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION


def test_non_blocking_incompleteness_is_incomplete_not_blocked():
    graph = _graph_of(5, domain=ResearchDomain.CATALYST, blocking=False)
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.degraded.blocking_unexecuted_count == 0
    assert scenarios.degraded.research_status is ResearchStatus.INCOMPLETE


def test_degraded_and_worst_never_report_a_token_shortfall_they_deliberately_avoided():
    """The bounded design never lets estimated_actual_high exceed the
    discovery budget in the first place -- shortfall_tokens stays 0, by
    construction, rather than reporting a huge number this module then
    ignores."""
    graph = _graph_of(50, domain=ResearchDomain.REGULATORY, blocking=True)
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.degraded.shortfall_tokens == 0
    assert scenarios.worst.shortfall_tokens == 0


def test_priority_order_picks_blocking_regulatory_before_remaining_gaps():
    blocking_parts = [_sec_style(i, domain=ResearchDomain.REGULATORY, blocking=True) for i in range(2)]
    gap_parts = [_sec_style(i + 100, domain=ResearchDomain.CATALYST, blocking=False) for i in range(2)]
    graph = SourceRoutingGraph(
        requirements=tuple(p[0] for p in (*gap_parts, *blocking_parts)),  # declared in "wrong" order
        targets=tuple(p[1] for p in (*gap_parts, *blocking_parts)),
        steps=tuple(s for p in (*gap_parts, *blocking_parts) for s in p[2]),
    )
    scenarios = assess_budget_scenarios(graph)
    # Only 4 total candidates, well within the cap of 3 minus... exactly 4
    # candidates but cap=3: blocking ones (2) must both be selected before
    # any gap one.
    assert scenarios.degraded.blocking_unexecuted_count == 0
    assert scenarios.degraded.executed_web_search_count == MAX_WEB_SEARCH_USES


# --- split/retry never reduces the total ------------------------------------
def test_repeated_computation_is_stable():
    first = assess_budget_scenarios()
    second = assess_budget_scenarios()
    assert first.normal.estimated_actual_high == second.normal.estimated_actual_high
    assert first.degraded.executed_web_search_count == second.degraded.executed_web_search_count


# --- the real catalog, reported honestly -------------------------------------
def test_real_catalog_normal_scenario_is_now_actually_complete():
    scenarios = assess_budget_scenarios()
    assert scenarios.normal.web_search_locator_uses <= MAX_WEB_SEARCH_USES
    assert scenarios.normal.estimated_actual_high <= 60_000
    assert scenarios.normal.incomplete_targets == 0
    assert scenarios.normal.direct_http_body_fetches > 0
    assert scenarios.normal_meets_safety_gate is True
    assert scenarios.normal.research_status.value == "COMPLETE"


def test_real_catalog_degraded_and_worst_are_bounded_and_blocked():
    scenarios = assess_budget_scenarios()
    assert scenarios.degraded.executed_web_search_count == MAX_WEB_SEARCH_USES
    assert scenarios.worst.executed_web_search_count == MAX_WEB_SEARCH_USES
    assert scenarios.degraded.shortfall_tokens == 0
    assert scenarios.degraded.blocking_unexecuted_count > 0
    assert scenarios.degraded.research_status.value == "BLOCKED_PENDING_VERIFICATION"
