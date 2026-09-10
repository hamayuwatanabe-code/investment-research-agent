"""source_routing.py: the pure resolve_active_route state machine.

No network, no execution -- these feed hypothetical outcome maps directly.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import AcquisitionStatus
from investment_research.research.document_store import DocumentRole
from investment_research.research.source_routing import (
    AcquisitionRoute,
    AcquisitionTarget,
    RequestCostClass,
    TargetKind,
    TokenCostClass,
    resolve_active_route,
)


def _direct_then_http_then_web(target_id: str = "t1") -> list[AcquisitionRoute]:
    return [
        AcquisitionRoute(
            route_id="r1", target_id=target_id, priority=1,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id="r2", target_id=target_id, priority=2,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id="r3", target_id=target_id, priority=3,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            token_cost_class=TokenCostClass.HIGH, request_cost_class=RequestCostClass.PAID_LLM_CALL,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]


def test_first_route_is_active_before_anything_is_attempted():
    routes = _direct_then_http_then_web()
    active = resolve_active_route(routes, {})
    assert active is not None
    assert active.route_id == "r1"


def test_direct_success_makes_web_fallback_inactive():
    routes = _direct_then_http_then_web()
    active = resolve_active_route(routes, {"r1": AcquisitionStatus.ACQUIRED})
    assert active is None  # r1's own terminal_conditions end the chain


def test_direct_zero_result_activates_the_next_route_only():
    routes = _direct_then_http_then_web()
    active = resolve_active_route(routes, {"r1": AcquisitionStatus.ACQUIRED_ZERO_RESULTS})
    assert active is not None
    assert active.route_id == "r2"


def test_direct_and_http_both_zero_result_finally_activates_web_search():
    routes = _direct_then_http_then_web()
    active = resolve_active_route(
        routes,
        {"r1": AcquisitionStatus.ACQUIRED_ZERO_RESULTS, "r2": AcquisitionStatus.ACQUIRED_ZERO_RESULTS},
    )
    assert active is not None
    assert active.route_id == "r3"


def test_web_search_zero_result_is_itself_terminal_no_further_route():
    routes = _direct_then_http_then_web()
    active = resolve_active_route(
        routes,
        {
            "r1": AcquisitionStatus.ACQUIRED_ZERO_RESULTS,
            "r2": AcquisitionStatus.ACQUIRED_ZERO_RESULTS,
            "r3": AcquisitionStatus.ACQUIRED_ZERO_RESULTS,
        },
    )
    assert active is None


def test_not_publicly_available_route_never_falls_through_to_search():
    """A single-route target whose only route is NOT_PUBLICLY_AVAILABLE has
    no lower-priority route at all -- there is nothing to fall through to,
    by construction, not by a runtime decision."""
    route = AcquisitionRoute(
        route_id="npa1",
        target_id="t_npa",
        priority=1,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        terminal_conditions=(AcquisitionStatus.NOT_PUBLICLY_AVAILABLE,),
    )
    assert resolve_active_route([route], {}) is route
    assert resolve_active_route([route], {"npa1": AcquisitionStatus.NOT_PUBLICLY_AVAILABLE}) is None
    assert not route.acquisition_method.requires_web_search_budget


def test_an_undeclared_outcome_is_conservatively_treated_as_terminal():
    """An outcome that is neither in terminal_conditions nor in the next
    route's fallback_conditions must never cause an infinite/undefined walk
    -- it stops, rather than guessing a behavior nobody declared."""
    routes = _direct_then_http_then_web()
    active = resolve_active_route(routes, {"r1": AcquisitionStatus.SKIPPED_DUE_TO_BUDGET})
    assert active is None


def test_target_and_route_types_construct_with_expected_defaults():
    target = AcquisitionTarget(target_id="t1", target_kind=TargetKind.GENERIC_WEB_DOCUMENT)
    assert target.document_role is DocumentRole.UNKNOWN
    assert target.serves_requirement_ids == ()
