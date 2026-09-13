"""Bounded, honestly-labeled projections over the step-DAG Source Routing
Graph -- Phase 2.7 scope only.

Pure arithmetic, driven by ``source_routing.resolve_next_steps``/
``is_target_complete``/``target_plan_status`` (the same pure functions real
execution would use), never a network call, never a claim about what
actually happened.

Corrects Phase 2.6's false-completeness defect: a NORMAL simulation that
assumes every declared route succeeds -- including routes for adapters that
do not exist as code yet -- is a statement about a possible FUTURE design,
never a statement that research is COMPLETE today. This module therefore:

* Never returns ``ResearchStatus`` (that decision belongs to real execution
  plus Evidence Integrity, neither of which happens here). It returns
  ``PlanStatus`` (is a plan structurally achievable given what is actually
  implemented) and per-scenario projected metrics -- nothing this module
  computes is "actual" or "complete" in the ``ResearchStatus`` sense.
* Names every unexecuted quantity ``projected_``/``estimated_`` -- never
  ``actual_``/``completed_``, which are reserved for a caller that really ran
  something (Phase 3+).
* Computes FOUR ``SimulationAssumption`` variants, not three: the aspirational
  ASSUME_IMPLEMENTED_ROUTES_SUCCEED (treats every declared route, built or
  not, as if it worked) alongside CURRENT_IMPLEMENTATION_ONLY/DIRECT_FAILURES/
  WORST_CASE, all three of which refuse to let a NOT_IMPLEMENTED adapter
  succeed. A caller must look at CURRENT_IMPLEMENTATION_ONLY to know what
  would actually happen if this were run today.
* Separates "we chose to stop at 3 searches" (``scheduled_budget_overrun_
  tokens``, always 0 by construction) from "there was more work than 3
  searches could cover" (``budget_gap_to_full_completion_*``, > 0 whenever
  requirements are left unserved) -- the two are never the same claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .acquisition_planning import AcquisitionMethod
from .budget_feasibility import (
    BASE_TOKENS_PER_SEARCH,
    HIGH_TOKENS_PER_SEARCH,
    LOW_TOKENS_PER_SEARCH,
)
from .source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    ImplementationStatus,
    PlanStatus,
    RequirementPriorityTier,
    SimulationAssumption,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    compute_plan_status,
    is_target_complete,
    resolve_next_steps,
)
from .source_routing_catalog import build_source_routing_graph

#: Requirement 8's hard, run-level cap. One shared budget for the whole
#: run's web-search usage, never per-target or per-domain.
MAX_WEB_SEARCH_USES = 3

_NON_SEARCH_TERMINAL_METHODS = (
    AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
    AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED,
)

#: What a non-search, non-terminal-method step resolves to when reached AND
#: usable, per assumption. ``None`` means "its own completion_condition"
#: (there is no single fixed success value across LOCATE/FETCH/PARSE kinds).
_DIRECT_OUTCOME_BY_ASSUMPTION: dict[SimulationAssumption, StepStatus | None] = {
    SimulationAssumption.ASSUME_IMPLEMENTED_ROUTES_SUCCEED: None,
    SimulationAssumption.CURRENT_IMPLEMENTATION_ONLY: None,
    SimulationAssumption.DIRECT_FAILURES: StepStatus.ZERO_RESULTS,
    SimulationAssumption.WORST_CASE: StepStatus.FAILED,
}


def _requirement_priority(graph: SourceRoutingGraph, target: AcquisitionTarget) -> RequirementPriorityTier:
    tiers = [r.priority_tier for r in graph.requirements if r.requirement_id in target.serves_requirement_ids]
    if not tiers:
        return RequirementPriorityTier.REMAINING_GAPS
    return min(tiers, key=lambda t: t.rank)


def _is_discovery_risk(step: AcquisitionStep) -> bool:
    """Whether DEGRADED/WORST's assumed soft/hard failure applies to this
    step: locating something, or a structured API's combined locate+fetch. A
    plain KNOWN_URL_HTTP fetch of an ALREADY-resolved URL, and every PARSE
    step, is mechanical retrieval, not discovery uncertainty -- it succeeds
    once reached (subject to implementation status) in every assumption.
    """
    if step.step_kind is StepKind.LOCATE:
        return True
    return step.step_kind is StepKind.FETCH and step.completion_condition is StepStatus.STRUCTURED_RECORD_RETRIEVED


def _resolve_non_search_outcome(
    step: AcquisitionStep, *, assumption: SimulationAssumption
) -> StepStatus:
    """What a non-search step resolves to when ``resolve_next_steps`` offers
    it, under ``assumption``. A step with no runnable code at all (DECLARED
    or DISABLED) always fails when the assumption respects implementation
    status -- it is never treated as "the route exists, so it can succeed"
    (Phase 3A requirement 1). This checks ``has_runnable_code`` -- whether
    SOME code (a generic primitive or better) could attempt the step if
    called directly, a lower bar than ``is_executor_ready`` (which additionally
    requires an ``AcquisitionExecutor`` to be wired -- that is what
    ``target_plan_status``/``compute_plan_status`` require).
    """
    if assumption.respects_implementation_status and not step.implementation_status.has_runnable_code:
        return StepStatus.FAILED
    if not _is_discovery_risk(step):
        return step.completion_condition
    fixed = _DIRECT_OUTCOME_BY_ASSUMPTION[assumption]
    return step.completion_condition if fixed is None else fixed


def _walk_target(
    target: AcquisitionTarget,
    steps: list[AcquisitionStep],
    *,
    assumption: SimulationAssumption,
    is_selected: Callable[[str], bool] | None,
) -> dict[str, StepStatus]:
    """Walk one target's step DAG to a fixed point under ``assumption``.

    ``is_selected``: ``None`` means "every web-search step reached succeeds"
    (the unbounded discovery pass used to find candidates); a predicate
    means "only execute a web-search step this returns True for -- every
    other one reads SKIPPED_DUE_TO_BUDGET" (the capped pass).
    """
    outcomes: dict[str, StepStatus] = {}
    for _ in range(len(steps) + 1):
        next_batch = resolve_next_steps(target, steps, outcomes)
        if not next_batch:
            break
        progressed = False
        for step in next_batch:
            if step.step_id in outcomes:
                continue
            if step.acquisition_method in _NON_SEARCH_TERMINAL_METHODS:
                outcomes[step.step_id] = step.completion_condition
            elif step.sends_to_web_search:
                executed = is_selected is None or is_selected(step.step_id)
                outcomes[step.step_id] = step.completion_condition if executed else StepStatus.SKIPPED_DUE_TO_BUDGET
            else:
                outcomes[step.step_id] = _resolve_non_search_outcome(step, assumption=assumption)
            progressed = True
        if not progressed:
            break
    return outcomes


def _select_within_cap(graph: SourceRoutingGraph, *, assumption: SimulationAssumption) -> dict[str, bool]:
    """Discover every web-search-requiring step the plan would touch if
    search were unbounded under ``assumption``, tag each with its target's
    priority tier, keep only the top ``MAX_WEB_SEARCH_USES``."""
    candidates: list[tuple[RequirementPriorityTier, int, str]] = []
    for order, target in enumerate(graph.targets):
        steps = graph.steps_for_target(target.target_id)
        outcomes = _walk_target(target, steps, assumption=assumption, is_selected=None)
        tier = _requirement_priority(graph, target)
        by_id = {s.step_id: s for s in steps}
        for step_id, outcome in outcomes.items():
            step = by_id[step_id]
            if step.sends_to_web_search and outcome == step.completion_condition:
                candidates.append((tier, order, step_id))

    candidates.sort(key=lambda c: (c[0].rank, c[1], c[2]))
    selected_ids = {step_id for _tier, _order, step_id in candidates[:MAX_WEB_SEARCH_USES]}
    return {step_id: (step_id in selected_ids) for _tier, _order, step_id in candidates}


@dataclass(frozen=True)
class ScenarioProjection:
    assumption: SimulationAssumption
    plan_status: PlanStatus
    #: REQUIRED-criticality requirements with NO executor-ready (or, under
    #: ASSUME_IMPLEMENTED_ROUTES_SUCCEED, no OFFLINE_VERIFIED-or-better)
    #: path to content acquisition at all (structural --
    #: ``compute_plan_status``). Under every assumption this still counts a
    #: NOT_PUBLICLY_AVAILABLE-only path, since that is never an
    #: implementation gap.
    blocking_unresolved_requirements: int
    #: CONDITIONAL_BLOCKING requirements with no such path either -- reported
    #: separately, never folded into ``blocking_unresolved_requirements``
    #: (Phase 3A requirement 3: never defaulted to blocking or non-blocking).
    pending_materiality_requirements: int

    projected_metadata_locator_requests: int
    projected_resolved_urls: int
    projected_direct_http_body_fetches: int
    projected_discovered_url_body_fetches: int
    projected_parsed_full_documents: int
    projected_web_search_uses: int
    anthropic_web_fetch_uses: int
    incomplete_targets: int
    total_targets: int

    discovery_budget_tokens: int
    preflight_reservation_tokens: int
    estimated_actual_low: int
    estimated_actual_base: int
    estimated_actual_high: int
    #: Always 0: the capped plan (<= MAX_WEB_SEARCH_USES executed) is never
    #: allowed to exceed the discovery budget by construction. "We chose to
    #: stop" is never the same claim as "we had enough".
    scheduled_budget_overrun_tokens: int

    #: What completing EVERY candidate web search this scenario found
    #: (uncapped) would have cost -- distinct from what the capped plan
    #: actually schedules.
    estimated_full_completion_tokens_low: int
    estimated_full_completion_tokens_base: int
    estimated_full_completion_tokens_high: int
    budget_gap_to_full_completion_low: int
    budget_gap_to_full_completion_base: int
    budget_gap_to_full_completion_high: int

    #: Web-search-requiring steps that exist and would be needed, but lost
    #: the priority cap this round -- present in the graph, never deleted.
    unserved_web_search_uses: int
    #: Of those, how many belong to a BLOCKING-tier requirement.
    unserved_blocking_requirements: int


def _tally(
    graph: SourceRoutingGraph,
    assumption: SimulationAssumption,
    per_target_outcomes: dict[str, dict[str, StepStatus]],
    search_decision: dict[str, bool],
    *,
    discovery_budget_tokens: int,
) -> ScenarioProjection:
    by_step_id = {s.step_id: s for s in graph.steps}

    metadata_locator = resolved_urls = direct_fetches = discovered_fetches = 0
    parsed_docs = web_search_uses = web_fetch_uses = 0

    for target in graph.targets:
        target_outcomes = per_target_outcomes[target.target_id]
        for step_id, outcome in target_outcomes.items():
            step = by_step_id[step_id]
            if (
                step.acquisition_method in (AcquisitionMethod.EXISTING_DIRECT_API, AcquisitionMethod.NEW_DIRECT_ADAPTER)
                and outcome == step.completion_condition
            ):
                metadata_locator += 1
            if outcome == StepStatus.URL_RESOLVED:
                resolved_urls += 1
            if step.acquisition_method is AcquisitionMethod.KNOWN_URL_HTTP and outcome == StepStatus.BODY_FETCHED:
                satisfying_deps = [
                    d
                    for d in step.depends_on_step_ids
                    if d in by_step_id and target_outcomes.get(d) == by_step_id[d].completion_condition
                ]
                dep_is_search = bool(satisfying_deps) and all(by_step_id[d].sends_to_web_search for d in satisfying_deps)
                if dep_is_search:
                    discovered_fetches += 1
                else:
                    direct_fetches += 1
            if outcome in (StepStatus.PARSED, StepStatus.REQUIRED_FIELDS_PARSED):
                parsed_docs += 1
            if step.acquisition_method is AcquisitionMethod.WEB_SEARCH_DISCOVERY and outcome == step.completion_condition:
                web_search_uses += 1
            if step.acquisition_method is AcquisitionMethod.ANTHROPIC_WEB_FETCH and outcome == step.completion_condition:
                web_fetch_uses += 1

    unserved = sum(1 for v in search_decision.values() if not v)
    unserved_blocking = 0
    for step_id, is_selected in search_decision.items():
        if is_selected:
            continue
        step = by_step_id[step_id]
        target = next(t for t in graph.targets if t.target_id == step.target_id)
        if _requirement_priority(graph, target) is RequirementPriorityTier.BLOCKING_REGULATORY:
            unserved_blocking += 1

    executed_uses = web_search_uses + web_fetch_uses
    reservation = executed_uses * HIGH_TOKENS_PER_SEARCH
    actual_low = executed_uses * LOW_TOKENS_PER_SEARCH
    actual_base = executed_uses * BASE_TOKENS_PER_SEARCH
    actual_high = executed_uses * HIGH_TOKENS_PER_SEARCH
    scheduled_overrun = max(0, actual_high - discovery_budget_tokens)

    total_candidate_uses = len(search_decision)
    full_low = total_candidate_uses * LOW_TOKENS_PER_SEARCH
    full_base = total_candidate_uses * BASE_TOKENS_PER_SEARCH
    full_high = total_candidate_uses * HIGH_TOKENS_PER_SEARCH
    gap_low = max(0, full_low - discovery_budget_tokens)
    gap_base = max(0, full_base - discovery_budget_tokens)
    gap_high = max(0, full_high - discovery_budget_tokens)

    incomplete = sum(
        0 if is_target_complete(t, graph.steps_for_target(t.target_id), per_target_outcomes[t.target_id]) else 1
        for t in graph.targets
    )

    respect_impl = assumption.respects_implementation_status
    plan_status, blocking_unresolved, pending_materiality = compute_plan_status(
        graph if respect_impl else _implementation_agnostic_view(graph)
    )
    if not respect_impl and plan_status is PlanStatus.EXECUTABLE_BOUNDED_INCOMPLETE:
        # ASSUME_IMPLEMENTED_ROUTES_SUCCEED is a statement about a possible
        # future design, never about what could run today -- a "some
        # targets, not others" mix under this assumption is
        # PROJECTED_FEASIBLE_WITH_GAPS, not EXECUTABLE_BOUNDED_INCOMPLETE
        # (Phase 3A requirement 2).
        plan_status = PlanStatus.PROJECTED_FEASIBLE_WITH_GAPS
    elif plan_status is PlanStatus.EXECUTABLE_COMPLETE and gap_high > 0:
        # Executor-ready (or, aspirationally, fully OFFLINE_VERIFIED), but
        # the search volume this plan actually needs cannot fit the
        # discovery budget even before the hard 3-search cap is applied.
        plan_status = PlanStatus.BUDGET_INFEASIBLE

    return ScenarioProjection(
        assumption=assumption,
        plan_status=plan_status,
        blocking_unresolved_requirements=blocking_unresolved,
        pending_materiality_requirements=pending_materiality,
        projected_metadata_locator_requests=metadata_locator,
        projected_resolved_urls=resolved_urls,
        projected_direct_http_body_fetches=direct_fetches,
        projected_discovered_url_body_fetches=discovered_fetches,
        projected_parsed_full_documents=parsed_docs,
        projected_web_search_uses=web_search_uses,
        anthropic_web_fetch_uses=web_fetch_uses,
        incomplete_targets=incomplete,
        total_targets=len(graph.targets),
        discovery_budget_tokens=discovery_budget_tokens,
        preflight_reservation_tokens=reservation,
        estimated_actual_low=actual_low,
        estimated_actual_base=actual_base,
        estimated_actual_high=actual_high,
        scheduled_budget_overrun_tokens=scheduled_overrun,
        estimated_full_completion_tokens_low=full_low,
        estimated_full_completion_tokens_base=full_base,
        estimated_full_completion_tokens_high=full_high,
        budget_gap_to_full_completion_low=gap_low,
        budget_gap_to_full_completion_base=gap_base,
        budget_gap_to_full_completion_high=gap_high,
        unserved_web_search_uses=unserved,
        unserved_blocking_requirements=unserved_blocking,
    )


def _implementation_agnostic_view(graph: SourceRoutingGraph) -> SourceRoutingGraph:
    """A copy of ``graph`` with every step's implementation_status forced to
    the highest earned-verification rung (OFFLINE_VERIFIED), for computing
    PlanStatus under ASSUME_IMPLEMENTED_ROUTES_SUCCEED -- a
    NOT_PUBLICLY_AVAILABLE-only path is still NOT_EXECUTABLE under this view
    (that is never an implementation gap, see
    ``AcquisitionStep.never_constitutes_content_acquisition``), but a
    DECLARED/DISABLED adapter is not. Never LIVE_VERIFIED: this view is a
    statement about a possible future design, never a claim that anything
    was checked against a real site/API (Phase 3A requirement 1)."""
    from dataclasses import replace

    return SourceRoutingGraph(
        requirements=graph.requirements,
        targets=graph.targets,
        steps=tuple(replace(s, implementation_status=ImplementationStatus.OFFLINE_VERIFIED) for s in graph.steps),
    )


def _run_scenario(
    graph: SourceRoutingGraph, *, assumption: SimulationAssumption, discovery_budget_tokens: int
) -> ScenarioProjection:
    search_decision = _select_within_cap(graph, assumption=assumption)

    def is_selected(step_id: str) -> bool:
        return search_decision.get(step_id, False)

    per_target_outcomes: dict[str, dict[str, StepStatus]] = {}
    for target in graph.targets:
        steps = graph.steps_for_target(target.target_id)
        per_target_outcomes[target.target_id] = _walk_target(
            target, steps, assumption=assumption, is_selected=is_selected
        )
    return _tally(graph, assumption, per_target_outcomes, search_decision, discovery_budget_tokens=discovery_budget_tokens)


@dataclass(frozen=True)
class SimulationProjections:
    """All four ``SimulationAssumption`` projections, computed together so a
    caller can never accidentally report only the aspirational one."""

    assume_implemented_routes_succeed: ScenarioProjection
    current_implementation_only: ScenarioProjection
    direct_failures: ScenarioProjection
    worst_case: ScenarioProjection


def assess_simulation_projections(
    graph: SourceRoutingGraph | None = None, *, discovery_budget_tokens: int = 60_000
) -> SimulationProjections:
    """Compute every ``SimulationAssumption`` projection over ``graph``
    (default: the real 31-group catalog). Read-only; never executes
    anything; contains no ``ResearchStatus`` anywhere in its output.
    """
    graph = graph or build_source_routing_graph()
    return SimulationProjections(
        assume_implemented_routes_succeed=_run_scenario(
            graph,
            assumption=SimulationAssumption.ASSUME_IMPLEMENTED_ROUTES_SUCCEED,
            discovery_budget_tokens=discovery_budget_tokens,
        ),
        current_implementation_only=_run_scenario(
            graph,
            assumption=SimulationAssumption.CURRENT_IMPLEMENTATION_ONLY,
            discovery_budget_tokens=discovery_budget_tokens,
        ),
        direct_failures=_run_scenario(
            graph, assumption=SimulationAssumption.DIRECT_FAILURES, discovery_budget_tokens=discovery_budget_tokens
        ),
        worst_case=_run_scenario(
            graph, assumption=SimulationAssumption.WORST_CASE, discovery_budget_tokens=discovery_budget_tokens
        ),
    )
