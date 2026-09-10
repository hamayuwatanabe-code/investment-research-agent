"""NORMAL / DEGRADED / WORST budget scenarios over the step-DAG Source
Routing Graph, with a hard, priority-ordered cap on web search usage.

Phase 2.6 scope only: pure arithmetic, driven by
``source_routing.resolve_next_steps``/``is_target_complete`` (the same pure
functions real execution would use), never a network call.

Corrects two things about the Phase 2.5 scenario engine:

1. A target's "success" is no longer read off a single flat Route outcome --
   NORMAL now walks the real LOCATE->FETCH->PARSE dependency chain, so an SEC
   target that only ever reaches URL_RESOLVED is correctly reported as
   incomplete, and ``direct_http_body_fetches`` is never zero merely because
   locating a URL was free.
2. DEGRADED/WORST no longer sum every conceivable fallback unbounded (Phase
   2.5's "541,000 tokens needed"). A run has ``MAX_WEB_SEARCH_USES = 3``,
   period. When more web-search-requiring steps exist than that, the
   requirements they serve are ranked by ``RequirementPriorityTier`` (see
   ``source_routing.priority_tier_for``) and only the top 3 are executed; the
   rest are recorded ``SKIPPED_DUE_TO_BUDGET`` -- never silently dropped, and
   never counted toward token cost. Every scenario reports
   ``research_status`` reflecting that incompleteness, never a token
   shortfall this module invents room to imagine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..schemas.enums import ResearchStatus
from .acquisition_planning import AcquisitionMethod
from .budget_feasibility import (
    BASE_TOKENS_PER_SEARCH,
    HIGH_TOKENS_PER_SEARCH,
    LOW_TOKENS_PER_SEARCH,
)
from .source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    RequirementPriorityTier,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    is_target_complete,
    resolve_next_steps,
)
from .source_routing_catalog import build_source_routing_graph

#: Requirement 8's hard, run-level cap. Never a per-target or per-domain
#: cap -- one shared budget for the whole run's web-search usage.
MAX_WEB_SEARCH_USES = 3

_NON_SEARCH_TERMINAL_METHODS = (
    AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
    AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED,
)

#: What a non-search, non-terminal-method step resolves to when attempted,
#: per scenario. NORMAL has no single fixed value (a LOCATE step's success
#: differs from a FETCH's) -- it always uses the step's own
#: completion_condition, expressed here as ``None``.
_DIRECT_OUTCOME_BY_SCENARIO: dict[str, StepStatus | None] = {
    "NORMAL": None,
    "DEGRADED": StepStatus.ZERO_RESULTS,
    "WORST": StepStatus.FAILED,
}


def _requirement_priority(graph: SourceRoutingGraph, target: AcquisitionTarget) -> RequirementPriorityTier:
    """The best (lowest-rank) priority tier among every requirement this
    target serves -- a target never gets deprioritized just because it also
    happens to serve a lower-priority requirement."""
    tiers = [r.priority_tier for r in graph.requirements if r.requirement_id in target.serves_requirement_ids]
    if not tiers:
        return RequirementPriorityTier.REMAINING_GAPS
    return min(tiers, key=lambda t: t.rank)


def _is_discovery_risk(step: AcquisitionStep) -> bool:
    """Whether ``step`` is where DEGRADED/WORST's assumed failure actually
    applies: locating something (LOCATE) or a structured API's combined
    locate+fetch (FETCH reaching STRUCTURED_RECORD_RETRIEVED). A plain
    KNOWN_URL_HTTP body fetch of an ALREADY-resolved URL, and every PARSE
    step, is mechanical retrieval -- not the discovery uncertainty these
    scenarios model -- so it always succeeds once reached, in every
    scenario, exactly like NORMAL.
    """
    if step.step_kind is StepKind.LOCATE:
        return True
    return step.step_kind is StepKind.FETCH and step.completion_condition is StepStatus.STRUCTURED_RECORD_RETRIEVED


def _direct_outcome_for(step: AcquisitionStep, fixed: StepStatus | None) -> StepStatus:
    if fixed is None or not _is_discovery_risk(step):
        return step.completion_condition
    return fixed


def _walk_target(
    target: AcquisitionTarget,
    steps: list[AcquisitionStep],
    *,
    direct_outcome: StepStatus | None,
    is_selected: Callable[[str], bool] | None,
) -> dict[str, StepStatus]:
    """Walk one target's step DAG to a fixed point.

    ``direct_outcome``: ``None`` means "every non-search step reached
    resolves to its OWN completion_condition" (NORMAL); a fixed
    ``StepStatus`` (``ZERO_RESULTS``/``FAILED``) means every non-search step
    reached resolves to that value instead (DEGRADED/WORST). A
    NOT_PUBLICLY_AVAILABLE/MANUAL_VERIFICATION_REQUIRED step always resolves
    to its own condition regardless -- inherent to the method, not a
    scenario assumption.

    ``is_selected``: ``None`` means "every web-search step reached succeeds"
    (the unbounded discovery pass); a predicate means "only execute a
    web-search step this returns True for -- everything else reads
    SKIPPED_DUE_TO_BUDGET" (the capped pass).
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
                outcomes[step.step_id] = _direct_outcome_for(step, direct_outcome)
            progressed = True
        if not progressed:
            break
    return outcomes


def _select_within_cap(graph: SourceRoutingGraph, *, direct_outcome: StepStatus | None) -> dict[str, bool]:
    """Pass 1: discover every web-search-requiring step the plan would touch
    if search were unbounded, tag each with its target's priority tier, keep
    only the top ``MAX_WEB_SEARCH_USES`` (stable within a tier, catalog
    order). Returns step_id -> "selected to actually execute"."""
    candidates: list[tuple[RequirementPriorityTier, int, str]] = []
    for order, target in enumerate(graph.targets):
        steps = graph.steps_for_target(target.target_id)
        outcomes = _walk_target(target, steps, direct_outcome=direct_outcome, is_selected=None)
        tier = _requirement_priority(graph, target)
        by_id = {s.step_id: s for s in steps}
        for step_id, outcome in outcomes.items():
            step = by_id[step_id]
            if step.sends_to_web_search and outcome == step.completion_condition:
                candidates.append((tier, order, step_id))

    candidates.sort(key=lambda c: (c[0].rank, c[1], c[2]))
    selected_ids = {step_id for _tier, _order, step_id in candidates[: MAX_WEB_SEARCH_USES]}
    return {step_id: (step_id in selected_ids) for _tier, _order, step_id in candidates}


@dataclass(frozen=True)
class ScenarioBudget:
    scenario: str
    metadata_locator_requests: int
    resolved_urls: int
    direct_http_body_fetches: int
    discovered_url_body_fetches: int
    parsed_full_documents: int
    web_search_locator_uses: int
    anthropic_web_fetch_uses: int
    executed_web_search_count: int
    unexecuted_web_search_count: int
    blocking_unexecuted_count: int
    incomplete_targets: int
    total_targets: int
    preflight_reservation_tokens: int
    estimated_actual_low: int
    estimated_actual_base: int
    estimated_actual_high: int
    discovery_budget_tokens: int
    feasible: bool
    shortfall_tokens: int
    research_status: ResearchStatus


@dataclass(frozen=True)
class BudgetScenarios:
    normal: ScenarioBudget
    degraded: ScenarioBudget
    worst: ScenarioBudget

    @property
    def normal_meets_safety_gate(self) -> bool:
        """NORMAL's own pass condition (requirement 7): <=3 web searches,
        the high actual-token estimate fits the discovery budget, AND every
        target actually reaches BODY_FETCHED/PARSED (or its structured
        equivalent) -- metadata/URL-only completion never counts.
        """
        return (
            self.normal.web_search_locator_uses <= MAX_WEB_SEARCH_USES
            and self.normal.estimated_actual_high <= self.normal.discovery_budget_tokens
            and self.normal.incomplete_targets == 0
        )


def _tally(
    graph: SourceRoutingGraph,
    per_target_outcomes: dict[str, dict[str, StepStatus]],
    search_decision: dict[str, bool],
    scenario: str,
    *,
    discovery_budget_tokens: int,
) -> ScenarioBudget:
    by_step_id = {s.step_id: s for s in graph.steps}

    metadata_locator = resolved_urls = direct_fetches = discovered_fetches = 0
    parsed_docs = web_search_uses = web_fetch_uses = 0

    for target in graph.targets:
        for step_id, outcome in per_target_outcomes[target.target_id].items():
            step = by_step_id[step_id]
            if (
                step.acquisition_method in (AcquisitionMethod.EXISTING_DIRECT_API, AcquisitionMethod.NEW_DIRECT_ADAPTER)
                and outcome == step.completion_condition
            ):
                metadata_locator += 1
            if outcome == StepStatus.URL_RESOLVED:
                resolved_urls += 1
            if step.acquisition_method is AcquisitionMethod.KNOWN_URL_HTTP and outcome == StepStatus.BODY_FETCHED:
                # Which cited dependency actually SUCCEEDED (not merely
                # cited) decides whether this fetch followed a Direct or a
                # web-search-discovered URL -- a fetch step commonly cites
                # an entire alternative group (e.g. Direct-locate OR
                # web-search-locate), and only one member of it ran.
                target_outcomes = per_target_outcomes[target.target_id]
                satisfying_deps = [
                    d
                    for d in step.depends_on_step_ids
                    if d in by_step_id and target_outcomes.get(d) == by_step_id[d].completion_condition
                ]
                dep_is_search = bool(satisfying_deps) and all(
                    by_step_id[d].sends_to_web_search for d in satisfying_deps
                )
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

    executed = sum(1 for v in search_decision.values() if v)
    unexecuted = sum(1 for v in search_decision.values() if not v)
    blocking_unexecuted = 0
    for step_id, is_selected in search_decision.items():
        if is_selected:
            continue
        step = by_step_id[step_id]
        target = next(t for t in graph.targets if t.target_id == step.target_id)
        if _requirement_priority(graph, target) is RequirementPriorityTier.BLOCKING_REGULATORY:
            blocking_unexecuted += 1

    uses = web_search_uses + web_fetch_uses
    reservation = uses * HIGH_TOKENS_PER_SEARCH
    low = uses * LOW_TOKENS_PER_SEARCH
    base = uses * BASE_TOKENS_PER_SEARCH
    high = uses * HIGH_TOKENS_PER_SEARCH
    shortfall = max(0, high - discovery_budget_tokens)

    incomplete = sum(
        0 if is_target_complete(t, graph.steps_for_target(t.target_id), per_target_outcomes[t.target_id]) else 1
        for t in graph.targets
    )

    if incomplete == 0:
        status = ResearchStatus.COMPLETE
    elif blocking_unexecuted > 0:
        status = ResearchStatus.BLOCKED_PENDING_VERIFICATION
    else:
        status = ResearchStatus.INCOMPLETE

    feasible = uses <= MAX_WEB_SEARCH_USES and high <= discovery_budget_tokens and incomplete == 0

    return ScenarioBudget(
        scenario=scenario,
        metadata_locator_requests=metadata_locator,
        resolved_urls=resolved_urls,
        direct_http_body_fetches=direct_fetches,
        discovered_url_body_fetches=discovered_fetches,
        parsed_full_documents=parsed_docs,
        web_search_locator_uses=web_search_uses,
        anthropic_web_fetch_uses=web_fetch_uses,
        executed_web_search_count=executed,
        unexecuted_web_search_count=unexecuted,
        blocking_unexecuted_count=blocking_unexecuted,
        incomplete_targets=incomplete,
        total_targets=len(graph.targets),
        preflight_reservation_tokens=reservation,
        estimated_actual_low=low,
        estimated_actual_base=base,
        estimated_actual_high=high,
        discovery_budget_tokens=discovery_budget_tokens,
        feasible=feasible,
        shortfall_tokens=shortfall,
        research_status=status,
    )


def _run_scenario(graph: SourceRoutingGraph, *, scenario: str, discovery_budget_tokens: int) -> ScenarioBudget:
    direct_outcome = _DIRECT_OUTCOME_BY_SCENARIO[scenario]
    search_decision = _select_within_cap(graph, direct_outcome=direct_outcome)
    def is_selected(step_id: str) -> bool:
        return search_decision.get(step_id, False)

    per_target_outcomes: dict[str, dict[str, StepStatus]] = {}
    for target in graph.targets:
        steps = graph.steps_for_target(target.target_id)
        per_target_outcomes[target.target_id] = _walk_target(
            target, steps, direct_outcome=direct_outcome, is_selected=is_selected
        )
    return _tally(graph, per_target_outcomes, search_decision, scenario, discovery_budget_tokens=discovery_budget_tokens)


def assess_budget_scenarios(
    graph: SourceRoutingGraph | None = None, *, discovery_budget_tokens: int = 60_000
) -> BudgetScenarios:
    """Compute NORMAL/DEGRADED/WORST over ``graph`` (default: the real
    31-group catalog). Read-only; never mutates ``graph``; never drops a
    step or target from the accounting, however UNSAFE or incomplete.

    NORMAL assumes every Direct/HTTP step succeeds at its own
    completion_condition. DEGRADED assumes every Direct/HTTP step returns
    ``ZERO_RESULTS``; WORST assumes ``FAILED``. Both trigger the SAME
    fallback eligibility in ``resolve_next_steps`` (neither status equals any
    step's completion_condition), so on this catalog's currently-linear
    fallback chains they select the identical capped search set -- a real
    property of this catalog's shape, not a modeling error.
    """
    graph = graph or build_source_routing_graph()
    return BudgetScenarios(
        normal=_run_scenario(graph, scenario="NORMAL", discovery_budget_tokens=discovery_budget_tokens),
        degraded=_run_scenario(graph, scenario="DEGRADED", discovery_budget_tokens=discovery_budget_tokens),
        worst=_run_scenario(graph, scenario="WORST", discovery_budget_tokens=discovery_budget_tokens),
    )
