"""ImplementationStatus semantics (Phase 3B requirement 6): an ordered
maturity ladder (option A), never a set of independently-hand-picked
capability tuples. ``rank`` is the single source of truth; every capability
property is derived from it, so promoting or demoting a step's status can
never produce a contradiction between "could this run" and "how is this
counted" (execution eligibility, routing, coverage, budget projection, and
diagnostics all read capability through the SAME two properties).

This file caught a real bug during Phase 3B: ``is_executor_ready`` was a
hand-picked tuple that omitted PIPELINE_WIRED, which ranks ABOVE
EXECUTOR_WIRED -- meaning a step the real production pipeline already called
would have read as *less* capable than one merely wired to a test executor.
Fixed to be rank-based; the regression test for it lives here.
"""

from __future__ import annotations

import dataclasses

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.routing_budget_scenarios import assess_simulation_projections
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    ImplementationStatus,
    PlanStatus,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetKind,
    compute_plan_status,
    target_plan_status,
)
from investment_research.research.source_routing_catalog import routing_coverage_counts
from investment_research.schemas.enums import ResearchDomain

_LEVELS_LOW_TO_HIGH = (
    ImplementationStatus.DECLARED,
    ImplementationStatus.PRIMITIVE_AVAILABLE,
    ImplementationStatus.ADAPTER_IMPLEMENTED,
    ImplementationStatus.EXECUTOR_WIRED,
    ImplementationStatus.PIPELINE_WIRED,
    ImplementationStatus.OFFLINE_VERIFIED,
    ImplementationStatus.LIVE_VERIFIED,
)


def test_rank_is_strictly_increasing_across_the_maturity_ladder():
    ranks = [level.rank for level in _LEVELS_LOW_TO_HIGH]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_disabled_ranks_below_every_other_level():
    assert ImplementationStatus.DISABLED.rank < ImplementationStatus.DECLARED.rank
    for level in _LEVELS_LOW_TO_HIGH:
        assert ImplementationStatus.DISABLED.rank < level.rank


def test_declared_cannot_execute():
    assert not ImplementationStatus.DECLARED.has_runnable_code
    assert not ImplementationStatus.DECLARED.is_executor_ready
    assert not ImplementationStatus.DECLARED.has_dedicated_adapter


def test_primitive_available_has_runnable_code_but_not_a_dedicated_adapter():
    assert ImplementationStatus.PRIMITIVE_AVAILABLE.has_runnable_code
    assert not ImplementationStatus.PRIMITIVE_AVAILABLE.has_dedicated_adapter
    assert not ImplementationStatus.PRIMITIVE_AVAILABLE.is_executor_ready


def test_adapter_implemented_does_not_by_itself_mean_executor_wired():
    assert ImplementationStatus.ADAPTER_IMPLEMENTED.has_dedicated_adapter
    assert ImplementationStatus.ADAPTER_IMPLEMENTED.has_runnable_code
    assert not ImplementationStatus.ADAPTER_IMPLEMENTED.is_executor_ready


def test_executor_wired_is_executable():
    assert ImplementationStatus.EXECUTOR_WIRED.is_executor_ready
    assert ImplementationStatus.EXECUTOR_WIRED.has_runnable_code
    assert ImplementationStatus.EXECUTOR_WIRED.has_dedicated_adapter
    assert not ImplementationStatus.EXECUTOR_WIRED.is_pipeline_wired
    assert not ImplementationStatus.EXECUTOR_WIRED.is_live_verified


def test_pipeline_wired_includes_executor_wired_capability():
    """The bug this file exists to catch: PIPELINE_WIRED ranks ABOVE
    EXECUTOR_WIRED, so a step the real pipeline calls must read as AT LEAST
    as executor-ready as one only wired to a test executor -- never less."""
    assert ImplementationStatus.PIPELINE_WIRED.is_executor_ready
    assert ImplementationStatus.PIPELINE_WIRED.is_pipeline_wired
    assert ImplementationStatus.PIPELINE_WIRED.has_runnable_code
    assert ImplementationStatus.PIPELINE_WIRED.has_dedicated_adapter


def test_offline_verified_includes_executor_wired_capability():
    assert ImplementationStatus.OFFLINE_VERIFIED.is_executor_ready
    assert ImplementationStatus.OFFLINE_VERIFIED.has_runnable_code
    assert ImplementationStatus.OFFLINE_VERIFIED.has_dedicated_adapter
    assert not ImplementationStatus.OFFLINE_VERIFIED.is_live_verified


def test_live_verified_includes_offline_executor_and_adapter_capability():
    assert ImplementationStatus.LIVE_VERIFIED.is_live_verified
    assert ImplementationStatus.LIVE_VERIFIED.is_executor_ready
    assert ImplementationStatus.LIVE_VERIFIED.has_runnable_code
    assert ImplementationStatus.LIVE_VERIFIED.has_dedicated_adapter
    assert ImplementationStatus.LIVE_VERIFIED.is_pipeline_wired  # ranks above it too


def test_disabled_is_never_executable_regardless_of_any_higher_state_it_once_had():
    """A step that was, e.g., OFFLINE_VERIFIED and is then explicitly
    disabled must read as fully non-executable -- DISABLED overrides
    everything, by construction (rank -1), never by a special case that
    could be forgotten."""
    assert not ImplementationStatus.DISABLED.has_runnable_code
    assert not ImplementationStatus.DISABLED.is_executor_ready
    assert not ImplementationStatus.DISABLED.has_dedicated_adapter
    assert not ImplementationStatus.DISABLED.is_pipeline_wired
    assert not ImplementationStatus.DISABLED.is_live_verified


def test_every_capability_property_is_a_pure_function_of_rank():
    """Phase 3B requirement 6: no capability check may be a hand-picked
    tuple of member names that could silently omit a higher-ranked member.
    This asserts the actual, general relationship holds for every level."""
    for level in ImplementationStatus:
        assert level.has_runnable_code == (level.rank >= ImplementationStatus.PRIMITIVE_AVAILABLE.rank)
        assert level.has_dedicated_adapter == (level.rank >= ImplementationStatus.ADAPTER_IMPLEMENTED.rank)
        assert level.is_executor_ready == (level.rank >= ImplementationStatus.EXECUTOR_WIRED.rank)
        assert level.is_pipeline_wired == (level.rank >= ImplementationStatus.PIPELINE_WIRED.rank)
        assert level.is_live_verified == (level.rank >= ImplementationStatus.LIVE_VERIFIED.rank)
        # Monotonic: is_live_verified implies is_pipeline_wired implies
        # is_executor_ready implies has_dedicated_adapter implies
        # has_runnable_code (or the level is DISABLED, which satisfies none).
        if level.is_live_verified:
            assert level.is_pipeline_wired
        if level.is_pipeline_wired:
            assert level.is_executor_ready
        if level.is_executor_ready:
            assert level.has_dedicated_adapter
        if level.has_dedicated_adapter:
            assert level.has_runnable_code


# --- consistency across routing/coverage/budget/diagnostics -----------------
def _single_step_target(implementation_status: ImplementationStatus):
    step = AcquisitionStep(
        step_id="s", target_id="t", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        completion_condition=StepStatus.URL_RESOLVED, implementation_status=implementation_status,
    )
    target = AcquisitionTarget(target_id="t", target_kind=TargetKind.FORM4_FILING, required_step_ids=("s",))
    return target, [step]


def test_target_plan_status_agrees_with_is_executor_ready_for_every_level():
    """routing (``target_plan_status``) must read exactly the SAME
    executability ``ImplementationStatus.is_executor_ready`` reports --
    never a second, independently-reasoned notion of "ready"."""
    for level in ImplementationStatus:
        target, steps = _single_step_target(level)
        status = target_plan_status(target, steps)
        if level.is_executor_ready:
            assert status is PlanStatus.EXECUTABLE_COMPLETE, level
        else:
            assert status is PlanStatus.NOT_EXECUTABLE, level


def test_compute_plan_status_promotion_and_demotion_never_contradict_target_plan_status():
    """Promoting a step from DECLARED to EXECUTOR_WIRED, or demoting it back
    down (including to DISABLED), must never leave ``compute_plan_status``'s
    aggregate result inconsistent with what ``target_plan_status`` says
    about the very same single-target graph (Phase 3B requirement 6's
    "promote/demote never contradicts execution eligibility")."""
    requirement = EvidenceRequirement(
        requirement_id="req", serves_legacy_need_ids=("synthetic",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    for level in ImplementationStatus:
        target, steps = _single_step_target(level)
        target = dataclasses.replace(target, serves_requirement_ids=("req",))
        graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))
        plan_status, _blocking, _pending = compute_plan_status(graph)
        target_status = target_plan_status(target, steps)
        assert (plan_status is PlanStatus.EXECUTABLE_COMPLETE) == (target_status is PlanStatus.EXECUTABLE_COMPLETE)


def test_routing_coverage_counts_executor_ready_total_matches_is_executor_ready_sum():
    """coverage (``routing_coverage_counts``) must total to exactly the
    number of steps ``ImplementationStatus.is_executor_ready`` reports True
    for -- never a hand-maintained count that could drift out of sync."""
    counts = routing_coverage_counts()
    from investment_research.research.source_routing_catalog import build_source_routing_graph

    graph = build_source_routing_graph()
    expected = sum(1 for s in graph.steps if s.implementation_status.is_executor_ready)
    assert counts.executor_ready_steps == expected
    expected_runnable = sum(1 for s in graph.steps if s.implementation_status.has_runnable_code)
    assert counts.runnable_code_steps == expected_runnable
    # The exact-level histogram sums to the same total as the ladder itself.
    ladder_total = (
        counts.declared_steps + counts.primitive_available_steps + counts.adapter_implemented_steps
        + counts.executor_wired_steps + counts.pipeline_wired_steps + counts.offline_verified_steps
        + counts.live_verified_steps + counts.disabled_steps
    )
    assert ladder_total == counts.locate_steps + counts.fetch_steps + counts.parse_steps
    # executor_ready_steps is never larger than the exact-level buckets that
    # feed it (EXECUTOR_WIRED + PIPELINE_WIRED + OFFLINE_VERIFIED +
    # LIVE_VERIFIED) -- a direct cross-check against the histogram above.
    assert counts.executor_ready_steps == (
        counts.executor_wired_steps + counts.pipeline_wired_steps
        + counts.offline_verified_steps + counts.live_verified_steps
    )


def test_budget_projection_never_lets_a_below_runnable_step_succeed():
    """budget projection (``routing_budget_scenarios.py``) must refuse to
    let a step with ``has_runnable_code is False`` succeed under any
    assumption that respects implementation status -- the same rule
    ``_resolve_non_search_outcome`` encodes, exercised here through the
    public entry point rather than the private function."""
    from investment_research.research.source_routing import FailurePolicy

    tid, rid = "target_x", "req_x"
    l1 = AcquisitionStep(
        step_id="l1", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.DECLARED,
    )
    f = AcquisitionStep(
        step_id="f", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=("l1",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.PRIMITIVE_AVAILABLE,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=("synthetic_x",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=("l1", "f"), serves_requirement_ids=(rid,),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(l1, f))
    projections = assess_simulation_projections(graph)
    current = projections.current_implementation_only
    assert current.incomplete_targets == 1
    assert current.projected_direct_http_body_fetches == 0
    assert current.projected_discovered_url_body_fetches == 0
