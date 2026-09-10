"""BudgetFeasibility: offline token-cost arithmetic, no API calls.

Covers Phase 2 requirement 7 (offline budget acceptance) and requirement 8
(the interpretive-pack/stage-budget mismatch stays a reported diagnostic,
never a resolved one).
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import (
    AcquisitionMethod,
    AcquisitionPlan,
    AcquisitionTask,
)
from investment_research.research.budget_feasibility import (
    EVIDENCE_PACK_ESTIMATED_TOTAL_TOKENS,
    INTERPRETIVE_STAGE_BUDGET_TOKENS,
    UNSAFE_SEARCH_COUNT_THRESHOLD,
    assess_budget_feasibility,
    interpretive_budget_diagnostic,
)
from investment_research.research.checks import SubjectScope
from investment_research.schemas.enums import ResearchDomain


def _task(method: AcquisitionMethod, n: int) -> AcquisitionTask:
    return AcquisitionTask(
        task_id=f"task_{method}_{n}",
        equivalence_key=f"company synthetic {n}",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY,
        acquisition_method=method,
        serves_legacy_need_ids=(f"synthetic_{n}",),
    )


def test_direct_http_api_tasks_cost_zero_llm_tokens():
    plan = AcquisitionPlan(
        tasks=tuple(_task(AcquisitionMethod.EXISTING_DIRECT_API, i) for i in range(5))
        + tuple(_task(AcquisitionMethod.KNOWN_URL_HTTP, i) for i in range(5, 10))
    )
    result = assess_budget_feasibility(plan)
    assert result.expected_server_tool_uses == 0
    assert result.actual_tokens_low == result.actual_tokens_base == result.actual_tokens_high == 0
    assert result.direct_http_api_tasks == 10
    assert result.feasible is True
    assert result.unsafe is False


def test_one_to_three_searches_are_plan_level_acceptable():
    for n in (1, 2, 3):
        plan = AcquisitionPlan(
            tasks=tuple(_task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, i) for i in range(n))
        )
        result = assess_budget_feasibility(plan)
        assert result.unsafe is False, f"{n} searches must not be flagged UNSAFE"
        assert result.expected_server_tool_uses == n


def test_four_or_more_searches_are_unsafe_against_discovery_budget():
    assert UNSAFE_SEARCH_COUNT_THRESHOLD == 4
    plan = AcquisitionPlan(
        tasks=tuple(_task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, i) for i in range(4))
    )
    result = assess_budget_feasibility(plan)
    assert result.unsafe is True
    assert result.feasible is False


def test_unsafe_plan_keeps_every_task_never_deletes_one():
    plan = AcquisitionPlan(
        tasks=tuple(_task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, i) for i in range(10))
    )
    result = assess_budget_feasibility(plan)
    assert result.unsafe is True
    assert result.feasible is False
    assert result.shortfall_tokens > 0
    # The plan object itself is untouched -- assess_budget_feasibility is read-only.
    assert len(plan.tasks) == 10


def test_shortfall_is_explicit_and_nonzero_when_infeasible():
    plan = AcquisitionPlan(
        tasks=tuple(_task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, i) for i in range(6))
    )
    result = assess_budget_feasibility(plan, discovery_budget_tokens=60_000)
    assert result.feasible is False
    assert result.shortfall_tokens == result.actual_tokens_high - 60_000
    assert result.shortfall_tokens > 0


def test_not_publicly_available_tasks_are_never_counted_toward_search_volume():
    plan = AcquisitionPlan(
        tasks=(
            _task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, 0),
            _task(AcquisitionMethod.NOT_PUBLICLY_AVAILABLE, 1),
            _task(AcquisitionMethod.NOT_PUBLICLY_AVAILABLE, 2),
            _task(AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED, 3),
        )
    )
    result = assess_budget_feasibility(plan)
    assert result.expected_server_tool_uses == 1
    assert result.web_search_tasks == 1


def test_splitting_the_same_task_count_across_calls_does_not_reduce_the_total():
    """This module counts tasks, not calls: a batching layer that serves
    several tasks in fewer underlying API calls must never make this
    function report a lower actual-token estimate, since real cost tracks
    the number of searches actually issued."""
    plan_a = AcquisitionPlan(
        tasks=tuple(_task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, i) for i in range(6))
    )
    plan_b = AcquisitionPlan(
        tasks=tuple(_task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, i) for i in range(6))
    )
    result_a = assess_budget_feasibility(plan_a)
    result_b = assess_budget_feasibility(plan_b)
    assert result_a.actual_tokens_high == result_b.actual_tokens_high
    assert result_a.expected_server_tool_uses == result_b.expected_server_tool_uses == 6


def test_measured_anchors_are_a_band_not_a_fixed_rate():
    plan_low = AcquisitionPlan(
        tasks=tuple(_task(AcquisitionMethod.WEB_SEARCH_DISCOVERY, i) for i in range(2))
    )
    result = assess_budget_feasibility(plan_low)
    # low != base != high: this is a range, never collapsed to one number.
    assert result.actual_tokens_low < result.actual_tokens_base < result.actual_tokens_high


# --- requirement 8: interpretive budget mismatch stays unresolved ----------
def test_interpretive_budget_diagnostic_reports_incompatibility_not_a_fix():
    diagnostic = interpretive_budget_diagnostic()
    assert diagnostic.evidence_pack_estimated_total_tokens == EVIDENCE_PACK_ESTIMATED_TOTAL_TOKENS
    assert diagnostic.interpretive_stage_budget_tokens == INTERPRETIVE_STAGE_BUDGET_TOKENS
    assert diagnostic.compatible is False
    assert diagnostic.shortfall_tokens == (
        EVIDENCE_PACK_ESTIMATED_TOTAL_TOKENS - INTERPRETIVE_STAGE_BUDGET_TOKENS
    )
    assert diagnostic.shortfall_tokens > 0
