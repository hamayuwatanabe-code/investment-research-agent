"""routing_budget_scenarios.py: bounded, honestly-labeled projections.

Covers Phase 2.7 requirements 1/2/4/8/9: no ResearchStatus anywhere in this
module's output, projected/estimated naming, the four SimulationAssumption
variants (including CURRENT_IMPLEMENTATION_ONLY vs. the aspirational
ASSUME_IMPLEMENTED_ROUTES_SUCCEED), and the scheduled-overrun-vs-full-
completion-gap distinction. No network call.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.routing_budget_scenarios import (
    MAX_WEB_SEARCH_USES,
    assess_simulation_projections,
)
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    ImplementationStatus,
    PlanStatus,
    RequirementCriticality,
    SimulationAssumption,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetKind,
)
from investment_research.schemas.enums import ResearchDomain


def _sec_style(n: int, *, domain=ResearchDomain.CAPITAL_STRUCTURE, blocking=False, direct_implemented=True):
    tid, rid = f"target_{n}", f"req_{n}"
    l1 = AcquisitionStep(
        step_id=f"l1_{n}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
        implementation_status=ImplementationStatus.ADAPTER_IMPLEMENTED if direct_implemented else ImplementationStatus.DECLARED,
    )
    l2 = AcquisitionStep(
        step_id=f"l2_{n}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
        implementation_status=ImplementationStatus.DISABLED,
    )
    f = AcquisitionStep(
        step_id=f"f_{n}", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=(f"l1_{n}", f"l2_{n}"), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.PRIMITIVE_AVAILABLE,
    )
    p = AcquisitionStep(
        step_id=f"p_{n}", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=(f"f_{n}",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.PRIMITIVE_AVAILABLE,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{n}",), subject_scope=SubjectScope.COMPANY,
        domain=domain, criticality=RequirementCriticality.REQUIRED if blocking else RequirementCriticality.BEST_EFFORT,
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


def test_all_four_assumptions_are_computed():
    projections = assess_simulation_projections()
    assert projections.assume_implemented_routes_succeed.assumption is SimulationAssumption.ASSUME_IMPLEMENTED_ROUTES_SUCCEED
    assert projections.current_implementation_only.assumption is SimulationAssumption.CURRENT_IMPLEMENTATION_ONLY
    assert projections.direct_failures.assumption is SimulationAssumption.DIRECT_FAILURES
    assert projections.worst_case.assumption is SimulationAssumption.WORST_CASE


def test_no_research_status_field_exists_anywhere_on_the_projection():
    projections = assess_simulation_projections()
    fields = set(projections.assume_implemented_routes_succeed.__dataclass_fields__)
    assert "research_status" not in fields
    assert not any("research_status" in f for f in fields)


def test_field_names_use_projected_and_estimated_prefixes():
    fields = set(assess_simulation_projections().current_implementation_only.__dataclass_fields__)
    for expected in (
        "projected_metadata_locator_requests",
        "projected_resolved_urls",
        "projected_direct_http_body_fetches",
        "projected_discovered_url_body_fetches",
        "projected_parsed_full_documents",
        "projected_web_search_uses",
        "estimated_actual_low",
        "estimated_actual_base",
        "estimated_actual_high",
    ):
        assert expected in fields
    # Forbidden bare names never appear as field names on this simulation type.
    for forbidden in ("actual_tokens", "completed_targets", "research_status"):
        assert forbidden not in fields


def test_current_implementation_only_never_lets_not_implemented_adapters_succeed():
    graph = _graph_of(1, direct_implemented=False)
    projections = assess_simulation_projections(graph)
    current = projections.current_implementation_only
    # The Direct locate is NOT_IMPLEMENTED; only the web-search fallback can
    # ever resolve a URL here, and that fallback's own fetch is what must be
    # counted as "discovered", never "direct".
    assert current.projected_direct_http_body_fetches == 0
    assert current.projected_discovered_url_body_fetches == 1


def test_assume_implemented_routes_succeed_is_a_pure_design_projection():
    graph = _graph_of(1, direct_implemented=False)
    projections = assess_simulation_projections(graph)
    aspirational = projections.assume_implemented_routes_succeed
    # Under the aspirational assumption the NOT_IMPLEMENTED direct route is
    # still treated as if it worked.
    assert aspirational.projected_direct_http_body_fetches == 1
    assert aspirational.projected_discovered_url_body_fetches == 0


def test_not_publicly_available_never_counts_as_target_acquired():
    npa_step = AcquisitionStep(
        step_id="npa", target_id="t_npa", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    requirement = EvidenceRequirement(
        requirement_id="req_npa", serves_legacy_need_ids=("synthetic_npa",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.REGULATORY,
        criticality=RequirementCriticality.REQUIRED,
    )
    target = AcquisitionTarget(
        target_id="t_npa", target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE,
        required_step_ids=("npa",), serves_requirement_ids=("req_npa",),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(npa_step,))
    projections = assess_simulation_projections(graph)
    for name in ("assume_implemented_routes_succeed", "current_implementation_only", "direct_failures", "worst_case"):
        scenario = getattr(projections, name)
        assert scenario.incomplete_targets == 1
        # The only target in this graph can never constitute content
        # acquisition (its only step is NOT_PUBLICLY_AVAILABLE) -- with
        # every target in the graph unresolvable, the honest aggregate is
        # NOT_EXECUTABLE, never a partial-completion status.
        assert scenario.plan_status is PlanStatus.NOT_EXECUTABLE
        assert scenario.blocking_unresolved_requirements == 1
        assert scenario.pending_materiality_requirements == 0


def test_conditional_blocking_non_public_requirement_is_reported_as_pending_materiality():
    npa_step = AcquisitionStep(
        step_id="npa", target_id="t_npa", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    requirement = EvidenceRequirement(
        requirement_id="req_npa", serves_legacy_need_ids=("synthetic_npa",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.REGULATORY,
        criticality=RequirementCriticality.CONDITIONAL_BLOCKING,
    )
    target = AcquisitionTarget(
        target_id="t_npa", target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE,
        required_step_ids=("npa",), serves_requirement_ids=("req_npa",),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(npa_step,))
    projections = assess_simulation_projections(graph)
    assert projections.current_implementation_only.blocking_unresolved_requirements == 0
    assert projections.current_implementation_only.pending_materiality_requirements == 1


# --- scheduled overrun vs. full-completion gap ------------------------------
def test_scheduled_overrun_is_zero_and_full_completion_gap_is_positive_when_capped():
    graph = _graph_of(10)  # 10 independent web-fallback-needing targets
    projections = assess_simulation_projections(graph)
    degraded = projections.direct_failures
    assert degraded.scheduled_budget_overrun_tokens == 0
    assert degraded.budget_gap_to_full_completion_high > 0
    assert degraded.unserved_web_search_uses == 10 - MAX_WEB_SEARCH_USES


def test_degraded_holds_unserved_count_for_the_real_catalog():
    projections = assess_simulation_projections()
    degraded = projections.direct_failures
    assert degraded.unserved_web_search_uses == 29
    assert degraded.scheduled_budget_overrun_tokens == 0
    assert degraded.budget_gap_to_full_completion_high > 0


def test_worst_case_matches_direct_failures_shape_on_this_linear_catalog():
    projections = assess_simulation_projections()
    assert projections.worst_case.unserved_web_search_uses == projections.direct_failures.unserved_web_search_uses


# --- real catalog: 49/31/18-adjacent invariants and honesty -----------------
def test_real_catalog_current_implementation_only_never_claims_full_completion():
    projections = assess_simulation_projections()
    current = projections.current_implementation_only
    assert current.incomplete_targets > 0
    # Never EXECUTABLE_COMPLETE: no AcquisitionExecutor exists to wire any
    # step to yet (Phase 3A requirement 1/2), so this can never read as
    # fully-executable research today.
    assert current.plan_status is not PlanStatus.EXECUTABLE_COMPLETE
    assert current.blocking_unresolved_requirements > 0
    assert current.pending_materiality_requirements == 6  # the 6 fda_dual regulator sub-requirements


def test_real_catalog_aspirational_projection_still_has_incomplete_targets():
    """Even the BEST-case design projection has 6 incomplete targets -- the
    FDA regulator-confirmation ones, which can never be "acquired" by
    construction (Phase 2.7 requirement 5), regardless of implementation."""
    projections = assess_simulation_projections()
    aspirational = projections.assume_implemented_routes_succeed
    assert aspirational.incomplete_targets == 6


def test_repeated_computation_is_stable():
    first = assess_simulation_projections()
    second = assess_simulation_projections()
    assert first.direct_failures.unserved_web_search_uses == second.direct_failures.unserved_web_search_uses
    assert first.current_implementation_only.projected_direct_http_body_fetches == second.current_implementation_only.projected_direct_http_body_fetches
