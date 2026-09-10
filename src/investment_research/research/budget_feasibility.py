"""BudgetFeasibility: offline, no-API token-cost planning for an AcquisitionPlan.

Phase 2 scope only: arithmetic over real per-search actual-token anchors
measured in a prior live production run (cited, never re-measured here, and
never treated as a fixed per-search unit cost). No network call is made by
this module, and nothing here touches ``search_batch()``/``fetch()``/stage
budget enforcement (``llm/client.py``'s ``LLMBudget``) -- it only estimates,
before anything runs, whether a plan's search volume WOULD fit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .acquisition_planning import AcquisitionMethod, AcquisitionPlan

#: Real, measured actual-token anchors from a prior live production run
#: (the same anchors research/anthropic_web.py's own docstrings and this
#: project's Phase 0B review cite). Two data points define a LOW/HIGH band,
#: never a single fixed per-search rate -- two points cannot support a
#: fixed+variable cost model, so no such model is fit here.
MEASURED_ANCHOR_LOW_SEARCHES = 2
MEASURED_ANCHOR_LOW_TOKENS = 31_492
MEASURED_ANCHOR_HIGH_SEARCHES = 6
MEASURED_ANCHOR_HIGH_TOKENS = 102_192

LOW_TOKENS_PER_SEARCH = MEASURED_ANCHOR_LOW_TOKENS // MEASURED_ANCHOR_LOW_SEARCHES  # 15,746
HIGH_TOKENS_PER_SEARCH = MEASURED_ANCHOR_HIGH_TOKENS // MEASURED_ANCHOR_HIGH_SEARCHES  # 17,032
#: Midpoint of the two measured rates -- a "base" estimate for reporting,
#: never presented as itself measured.
BASE_TOKENS_PER_SEARCH = (LOW_TOKENS_PER_SEARCH + HIGH_TOKENS_PER_SEARCH) // 2

#: A plan needing 4 or more actual searches is UNSAFE against a 60,000-token
#: discovery quota: fitting the preflight RESERVATION ceiling is a different
#: claim from fitting ACTUAL token consumption (see
#: research/anthropic_web.py's ``_batch_reservation`` docstring for the same
#: distinction at the batching layer, and this project's prior production-
#: budget-capacity findings). This constant is a planning-time judgement
#: only -- it enforces nothing; Phase 2 does not touch any enforcement path.
UNSAFE_SEARCH_COUNT_THRESHOLD = 4

#: Phase 0B's own estimate of the Evidence Pack total across every
#: interpretive agent, restated here as a standing architecture diagnostic --
#: NOT re-derived, NOT resolved. Pack content is ordinary LLM INPUT text; it
#: is never treated as a separate token axis from the Interpretive stage's
#: own quota.
EVIDENCE_PACK_ESTIMATED_TOTAL_TOKENS = 90_000
#: 30% of the default 200,000-token production LLMBudget (llm/client.py's
#: stage split), matching the actual Interpretive stage quota today.
INTERPRETIVE_STAGE_BUDGET_TOKENS = 60_000

_SEARCH_METHODS = (AcquisitionMethod.WEB_SEARCH_DISCOVERY, AcquisitionMethod.ANTHROPIC_WEB_FETCH)
_DIRECT_METHODS = (
    AcquisitionMethod.EXISTING_DIRECT_API,
    AcquisitionMethod.NEW_DIRECT_ADAPTER,
    AcquisitionMethod.KNOWN_URL_HTTP,
)


@dataclass(frozen=True)
class BudgetFeasibility:
    direct_http_api_tasks: int
    web_search_tasks: int
    anthropic_web_fetch_tasks: int
    #: web_search_tasks + anthropic_web_fetch_tasks. Direct HTTP/API tasks,
    #: NOT_PUBLICLY_AVAILABLE tasks, and MANUAL_VERIFICATION_REQUIRED tasks
    #: are never counted here (requirement 7: LLM token cost is zero for
    #: direct HTTP/API; a non-public document is never sent to search at all).
    expected_server_tool_uses: int
    reservation_tokens: int
    actual_tokens_low: int
    actual_tokens_base: int
    actual_tokens_high: int
    discovery_budget_tokens: int
    feasible: bool
    shortfall_tokens: int
    #: True once expected_server_tool_uses reaches UNSAFE_SEARCH_COUNT_THRESHOLD,
    #: independent of whether the reservation ceiling happens to still fit.
    unsafe: bool


def _reservation_for(uses: int) -> int:
    """Conservative preflight reservation: the HIGH measured per-search rate
    applied to every use, mirroring the existing scaled-reservation principle
    in ``research/anthropic_web.py``'s ``_batch_reservation`` (reserve for the
    worse-observed case, not the average)."""
    return uses * HIGH_TOKENS_PER_SEARCH


def assess_budget_feasibility(
    plan: AcquisitionPlan, *, discovery_budget_tokens: int = INTERPRETIVE_STAGE_BUDGET_TOKENS
) -> BudgetFeasibility:
    """Pure arithmetic over ``plan.tasks``. Makes no network call, mutates
    nothing, and never removes a task regardless of the result (an UNSAFE or
    infeasible plan is still reported in full, per requirement 7).

    Each ``AcquisitionTask`` here is one server-tool use if it were actually
    executed -- this module does not model call-level batching (several
    tasks served by one underlying API call), because batching several
    searches into fewer CALLS does not reduce the number of searches
    actually issued, and this module must never report a lower total merely
    because some future batching layer could parallelize the calls
    (requirement 7: "split/retryで総tokenが減った扱いにしない").
    """
    direct = sum(1 for t in plan.tasks if t.acquisition_method in _DIRECT_METHODS)
    web_search = sum(
        1 for t in plan.tasks if t.acquisition_method is AcquisitionMethod.WEB_SEARCH_DISCOVERY
    )
    web_fetch = sum(
        1 for t in plan.tasks if t.acquisition_method is AcquisitionMethod.ANTHROPIC_WEB_FETCH
    )
    uses = web_search + web_fetch

    reservation = _reservation_for(uses)
    low = uses * LOW_TOKENS_PER_SEARCH
    base = uses * BASE_TOKENS_PER_SEARCH
    high = uses * HIGH_TOKENS_PER_SEARCH

    unsafe = uses >= UNSAFE_SEARCH_COUNT_THRESHOLD
    feasible = (not unsafe) and high <= discovery_budget_tokens
    shortfall = max(0, high - discovery_budget_tokens)

    return BudgetFeasibility(
        direct_http_api_tasks=direct,
        web_search_tasks=web_search,
        anthropic_web_fetch_tasks=web_fetch,
        expected_server_tool_uses=uses,
        reservation_tokens=reservation,
        actual_tokens_low=low,
        actual_tokens_base=base,
        actual_tokens_high=high,
        discovery_budget_tokens=discovery_budget_tokens,
        feasible=feasible,
        shortfall_tokens=shortfall,
        unsafe=unsafe,
    )


@dataclass(frozen=True)
class InterpretiveBudgetDiagnostic:
    evidence_pack_estimated_total_tokens: int
    interpretive_stage_budget_tokens: int
    compatible: bool
    shortfall_tokens: int


def interpretive_budget_diagnostic() -> InterpretiveBudgetDiagnostic:
    """NOT a resolution -- a standing architecture diagnostic.

    Records that the ~90,000-token Evidence Pack total Phase 0B estimated
    across the interpretive agents does not fit inside the Interpretive
    stage's existing 60,000-token quota. Evidence Pack input is ordinary LLM
    input text, not a separate budget axis, so it is never excluded from
    this comparison. Phase 2 does not touch stage quotas or agent prompts and
    makes no attempt to close this gap; this function exists so the gap
    cannot be silently reported as solved.
    """
    shortfall = max(0, EVIDENCE_PACK_ESTIMATED_TOTAL_TOKENS - INTERPRETIVE_STAGE_BUDGET_TOKENS)
    return InterpretiveBudgetDiagnostic(
        evidence_pack_estimated_total_tokens=EVIDENCE_PACK_ESTIMATED_TOTAL_TOKENS,
        interpretive_stage_budget_tokens=INTERPRETIVE_STAGE_BUDGET_TOKENS,
        compatible=EVIDENCE_PACK_ESTIMATED_TOTAL_TOKENS <= INTERPRETIVE_STAGE_BUDGET_TOKENS,
        shortfall_tokens=shortfall,
    )
