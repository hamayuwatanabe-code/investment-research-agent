"""NORMAL / DEGRADED / WORST budget scenarios over a Source Routing Graph.

Phase 2.5 scope only: pure arithmetic, driven by ``source_routing.resolve_active_route``
(the same pure state machine ``source_routing.py`` exposes for real use), never
a network call. This module does not decide anything at runtime -- it answers,
for three fixed hypotheses about how Direct routes would go, what a plan's
actual search volume and token cost would be.

The three scenarios differ only in what is ASSUMED about every non-search
route's outcome, walked through ``resolve_active_route`` exactly as a real
executor would:

* NORMAL   -- every Direct/HTTP route succeeds (ACQUIRED) on first try, so
  its own ``terminal_conditions`` halts the chain before any fallback runs.
  Only targets with NO Direct alternative (route priority 1 is itself a
  search method) contribute a search under this scenario.
* DEGRADED -- every Direct/HTTP route comes back ACQUIRED_ZERO_RESULTS (a
  real, coded ``fallback_conditions`` trigger on every route built in
  ``source_routing_catalog.py``), so the chain falls through to its
  search fallback, once.
* WORST    -- every route ever declared on every target is counted, as if
  every fallback that could ever trigger, did (never inflated further than
  what is actually declared in the catalog).

In every scenario, NOT_PUBLICLY_AVAILABLE and MANUAL_VERIFICATION_REQUIRED
routes are walked (they are real, declared routes) but never counted toward
search volume or token cost -- neither method sends anything to search
(``AcquisitionMethod.requires_web_search_budget`` is False for both).
"""

from __future__ import annotations

from dataclasses import dataclass

from .acquisition_planning import AcquisitionMethod
from .budget_feasibility import (
    BASE_TOKENS_PER_SEARCH,
    HIGH_TOKENS_PER_SEARCH,
    INTERPRETIVE_STAGE_BUDGET_TOKENS,
    LOW_TOKENS_PER_SEARCH,
    UNSAFE_SEARCH_COUNT_THRESHOLD,
)
from .checks import AcquisitionStatus
from .source_routing import AcquisitionRoute, SourceRoutingGraph, resolve_active_route
from .source_routing_catalog import build_source_routing_graph

_DIRECT_METHODS = (
    AcquisitionMethod.EXISTING_DIRECT_API,
    AcquisitionMethod.NEW_DIRECT_ADAPTER,
)
_HTTP_METHODS = (AcquisitionMethod.KNOWN_URL_HTTP,)


@dataclass(frozen=True)
class ScenarioBudget:
    scenario: str
    api_messages: int
    web_search_uses: int
    web_fetch_uses: int
    direct_api_requests: int
    direct_http_requests: int
    reservation_tokens: int
    actual_tokens_low: int
    actual_tokens_base: int
    actual_tokens_high: int
    discovery_budget_tokens: int
    feasible: bool
    shortfall_tokens: int


@dataclass(frozen=True)
class BudgetScenarios:
    normal: ScenarioBudget
    degraded: ScenarioBudget
    worst: ScenarioBudget

    @property
    def normal_meets_safety_gate(self) -> bool:
        """NORMAL's own pass condition (requirement 7): <= 3 expected
        web_search uses AND the high actual-token estimate fits the discovery
        budget. This property never adjusts a classification to force a
        pass -- it only reads the already-computed NORMAL scenario."""
        return (
            self.normal.web_search_uses <= 3
            and self.normal.actual_tokens_high <= self.normal.discovery_budget_tokens
        )


def _reached_routes_for_target(
    routes: list[AcquisitionRoute], *, direct_outcome: AcquisitionStatus
) -> list[AcquisitionRoute]:
    """Walk one target's route chain via ``resolve_active_route``, assuming
    ``direct_outcome`` for every Direct/HTTP route reached. A search route,
    once reached, always halts the walk (its own ``terminal_conditions``
    include ACQUIRED/ACQUIRED_ZERO_RESULTS in this catalog); a
    NOT_PUBLICLY_AVAILABLE/MANUAL_VERIFICATION_REQUIRED route is always
    terminal by construction."""
    outcomes: dict[str, AcquisitionStatus] = {}
    reached: list[AcquisitionRoute] = []
    for _ in range(len(routes)):
        active = resolve_active_route(routes, outcomes)
        if active is None:
            break
        reached.append(active)
        method = active.acquisition_method
        if method.requires_web_search_budget:
            outcome = AcquisitionStatus.ACQUIRED
        elif method is AcquisitionMethod.NOT_PUBLICLY_AVAILABLE:
            outcome = AcquisitionStatus.NOT_PUBLICLY_AVAILABLE
        elif method is AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED:
            outcome = AcquisitionStatus.MANUAL_VERIFICATION_REQUIRED
        else:
            outcome = direct_outcome
        outcomes[active.route_id] = outcome
    return reached


def _tally(routes: list[AcquisitionRoute], scenario: str) -> ScenarioBudget:
    web_search = sum(1 for r in routes if r.acquisition_method is AcquisitionMethod.WEB_SEARCH_DISCOVERY)
    web_fetch = sum(1 for r in routes if r.acquisition_method is AcquisitionMethod.ANTHROPIC_WEB_FETCH)
    direct_api = sum(1 for r in routes if r.acquisition_method in _DIRECT_METHODS)
    direct_http = sum(1 for r in routes if r.acquisition_method in _HTTP_METHODS)

    uses = web_search + web_fetch
    # split/retry never reduces the total: this is a flat per-use rate over
    # `uses`, never divided down for however many underlying calls a future
    # batching layer might serve them in (requirement 7).
    reservation = uses * HIGH_TOKENS_PER_SEARCH
    low = uses * LOW_TOKENS_PER_SEARCH
    base = uses * BASE_TOKENS_PER_SEARCH
    high = uses * HIGH_TOKENS_PER_SEARCH

    feasible = uses < UNSAFE_SEARCH_COUNT_THRESHOLD and high <= INTERPRETIVE_STAGE_BUDGET_TOKENS
    shortfall = max(0, high - INTERPRETIVE_STAGE_BUDGET_TOKENS)

    return ScenarioBudget(
        scenario=scenario,
        api_messages=uses,  # one LLM message per search-shaped route; direct/HTTP need no message
        web_search_uses=web_search,
        web_fetch_uses=web_fetch,
        direct_api_requests=direct_api,
        direct_http_requests=direct_http,
        reservation_tokens=reservation,
        actual_tokens_low=low,
        actual_tokens_base=base,
        actual_tokens_high=high,
        discovery_budget_tokens=INTERPRETIVE_STAGE_BUDGET_TOKENS,
        feasible=feasible,
        shortfall_tokens=shortfall,
    )


def assess_budget_scenarios(graph: SourceRoutingGraph | None = None) -> BudgetScenarios:
    """Compute NORMAL/DEGRADED/WORST over ``graph`` (defaults to the real
    31-group catalog graph). Read-only: never mutates ``graph`` and never
    removes a route from any scenario's accounting, however UNSAFE."""
    graph = graph or build_source_routing_graph()
    target_ids = {t.target_id for t in graph.targets}

    normal_reached: list[AcquisitionRoute] = []
    degraded_reached: list[AcquisitionRoute] = []
    for target_id in target_ids:
        routes = graph.routes_for_target(target_id)
        normal_reached.extend(
            _reached_routes_for_target(routes, direct_outcome=AcquisitionStatus.ACQUIRED)
        )
        degraded_reached.extend(
            _reached_routes_for_target(routes, direct_outcome=AcquisitionStatus.ACQUIRED_ZERO_RESULTS)
        )
    # WORST: every declared route, on every target -- every possible
    # fallback counted, nothing invented beyond what the catalog declares.
    worst_reached = list(graph.routes)

    return BudgetScenarios(
        normal=_tally(normal_reached, "NORMAL"),
        degraded=_tally(degraded_reached, "DEGRADED"),
        worst=_tally(worst_reached, "WORST"),
    )
