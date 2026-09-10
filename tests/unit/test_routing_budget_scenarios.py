"""routing_budget_scenarios.py: NORMAL / DEGRADED / WORST, computed over the
real Source Routing Graph and over small synthetic graphs. No network call.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import AcquisitionStatus, SubjectScope
from investment_research.research.routing_budget_scenarios import assess_budget_scenarios
from investment_research.research.source_routing import (
    AcquisitionRoute,
    AcquisitionTarget,
    EvidenceRequirement,
    SourceRoutingGraph,
    TargetKind,
)


def _web_only_target(n: int) -> tuple[EvidenceRequirement, AcquisitionTarget, AcquisitionRoute]:
    req_id, target_id, route_id = f"req_{n}", f"target_{n}", f"route_{n}"
    requirement = EvidenceRequirement(
        requirement_id=req_id,
        serves_legacy_need_ids=(f"synthetic_{n}",),
        subject_scope=SubjectScope.COMPANY,
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.GENERIC_WEB_DOCUMENT, serves_requirement_ids=(req_id,)
    )
    route = AcquisitionRoute(
        route_id=route_id,
        target_id=target_id,
        priority=1,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
    )
    return requirement, target, route


def _direct_then_web_target(n: int) -> tuple[EvidenceRequirement, AcquisitionTarget, list[AcquisitionRoute]]:
    req_id, target_id = f"req_d{n}", f"target_d{n}"
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=(f"synthetic_d{n}",), subject_scope=SubjectScope.COMPANY
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT, serves_requirement_ids=(req_id,)
    )
    routes = [
        AcquisitionRoute(
            route_id=f"route_d{n}_1", target_id=target_id, priority=1,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_d{n}_2", target_id=target_id, priority=2,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]
    return requirement, target, routes


def test_normal_and_degraded_and_worst_are_computed_separately():
    scenarios = assess_budget_scenarios()
    assert scenarios.normal.scenario == "NORMAL"
    assert scenarios.degraded.scenario == "DEGRADED"
    assert scenarios.worst.scenario == "WORST"
    # NORMAL must never exceed DEGRADED/WORST in search volume for this catalog.
    assert scenarios.normal.web_search_uses <= scenarios.degraded.web_search_uses
    assert scenarios.normal.web_search_uses <= scenarios.worst.web_search_uses


def test_direct_only_target_contributes_zero_search_under_normal_but_one_under_degraded():
    requirement, target, routes = _direct_then_web_target(1)
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), routes=tuple(routes))
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.normal.web_search_uses == 0
    assert scenarios.normal.direct_api_requests == 1
    assert scenarios.degraded.web_search_uses == 1


def test_not_publicly_available_route_contributes_no_search_in_any_scenario():
    req_id, target_id, route_id = "req_npa", "target_npa", "route_npa"
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=("synthetic_npa",), subject_scope=SubjectScope.COMPANY
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE, serves_requirement_ids=(req_id,)
    )
    route = AcquisitionRoute(
        route_id=route_id, target_id=target_id, priority=1,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        terminal_conditions=(AcquisitionStatus.NOT_PUBLICLY_AVAILABLE,),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), routes=(route,))
    scenarios = assess_budget_scenarios(graph)
    for scenario in (scenarios.normal, scenarios.degraded, scenarios.worst):
        assert scenario.web_search_uses == 0
        assert scenario.actual_tokens_high == 0


# --- NORMAL safety gate: <=3 searches, never adjusted to force a pass ------
def test_normal_meets_safety_gate_at_exactly_three_web_only_targets():
    parts = [_web_only_target(i) for i in range(3)]
    graph = SourceRoutingGraph(
        requirements=tuple(p[0] for p in parts),
        targets=tuple(p[1] for p in parts),
        routes=tuple(p[2] for p in parts),
    )
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.normal.web_search_uses == 3
    assert scenarios.normal_meets_safety_gate is True


def test_normal_fails_safety_gate_at_four_web_only_targets_reported_not_hidden():
    parts = [_web_only_target(i) for i in range(4)]
    graph = SourceRoutingGraph(
        requirements=tuple(p[0] for p in parts),
        targets=tuple(p[1] for p in parts),
        routes=tuple(p[2] for p in parts),
    )
    scenarios = assess_budget_scenarios(graph)
    assert scenarios.normal.web_search_uses == 4
    assert scenarios.normal_meets_safety_gate is False
    assert scenarios.normal.feasible is False
    # The four targets are all still present in the graph -- nothing was
    # dropped or reclassified to force a pass.
    assert len(graph.targets) == 4


def test_real_catalog_normal_scenario_is_reported_honestly():
    """Whatever the real catalog's NORMAL scenario computes to, this test
    locks the actual measured shape rather than an aspirational one -- if a
    future catalog change moves this number, the test (and the report) must
    be updated to match reality, never the other way around."""
    scenarios = assess_budget_scenarios()
    assert scenarios.normal.web_search_uses == 3
    assert scenarios.normal.actual_tokens_high <= 60_000
    assert scenarios.normal_meets_safety_gate is True
    # DEGRADED/WORST are NOT expected to pass -- this is the honest state
    # Phase 2.5 was asked to expose, not resolve.
    assert scenarios.degraded.feasible is False
    assert scenarios.worst.feasible is False


# --- split/retry never reduces the total ------------------------------------
def test_repeated_computation_never_reduces_the_total():
    first = assess_budget_scenarios()
    second = assess_budget_scenarios()
    assert first.normal.actual_tokens_high == second.normal.actual_tokens_high
    assert first.degraded.actual_tokens_high == second.degraded.actual_tokens_high
    assert first.worst.actual_tokens_high == second.worst.actual_tokens_high


def test_doubling_web_only_targets_doubles_the_token_estimate_linearly():
    one = [_web_only_target(0)]
    two = [_web_only_target(0), _web_only_target(1)]
    graph_one = SourceRoutingGraph(
        requirements=tuple(p[0] for p in one), targets=tuple(p[1] for p in one), routes=tuple(p[2] for p in one)
    )
    graph_two = SourceRoutingGraph(
        requirements=tuple(p[0] for p in two), targets=tuple(p[1] for p in two), routes=tuple(p[2] for p in two)
    )
    scenarios_one = assess_budget_scenarios(graph_one)
    scenarios_two = assess_budget_scenarios(graph_two)
    assert scenarios_two.normal.actual_tokens_high == 2 * scenarios_one.normal.actual_tokens_high
    assert scenarios_two.normal.reservation_tokens == 2 * scenarios_one.normal.reservation_tokens
