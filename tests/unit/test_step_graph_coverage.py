"""step_graph_coverage.py: projecting Source Routing Graph outcomes onto the
CoverageLedger. No network, no execution -- pure projection over supplied
(hypothetical) step outcomes.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import AcquisitionStatus, SubjectScope
from investment_research.research.legacy_catalog import LEGACY_CATALOG
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    ImplementationStatus,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetKind,
)
from investment_research.research.source_routing_catalog import build_source_routing_graph
from investment_research.research.step_graph_coverage import project_step_results_to_coverage
from investment_research.schemas.enums import ResearchDomain


def _document_target(n: int, need_ids: tuple[str, ...]):
    tid, rid = f"target_{n}", f"req_{n}"
    l1 = AcquisitionStep(
        step_id=f"l1_{n}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
    )
    f = AcquisitionStep(
        step_id=f"f_{n}", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=(f"l1_{n}",), completion_condition=StepStatus.BODY_FETCHED,
    )
    p = AcquisitionStep(
        step_id=f"p_{n}", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
        depends_on_step_ids=(f"f_{n}",), completion_condition=StepStatus.PARSED,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=need_ids, subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=(f"l1_{n}", f"f_{n}", f"p_{n}"), serves_requirement_ids=(rid,),
    )
    return requirement, target, [l1, f, p]


def test_every_one_of_the_49_legacy_needs_is_present():
    graph = SourceRoutingGraph()
    ledger = project_step_results_to_coverage(graph, {})
    assert len(ledger.results) == 49
    assert ledger.missing_need_ids() == set()


def test_metadata_only_never_becomes_acquired():
    requirement, target, steps = _document_target(1, ("bear_0",))
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=steps)
    outcomes = {"l1_1": StepStatus.URL_RESOLVED}
    ledger = project_step_results_to_coverage(graph, outcomes)
    assert ledger.results["bear_0"].acquisition_status is not AcquisitionStatus.ACQUIRED
    assert ledger.results["bear_0"].acquisition_status is AcquisitionStatus.UNSEARCHED


def test_full_chain_completion_maps_to_acquired():
    requirement, target, steps = _document_target(2, ("bear_0",))
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=steps)
    outcomes = {"l1_2": StepStatus.URL_RESOLVED, "f_2": StepStatus.BODY_FETCHED, "p_2": StepStatus.PARSED}
    ledger = project_step_results_to_coverage(graph, outcomes)
    assert ledger.results["bear_0"].acquisition_status is AcquisitionStatus.ACQUIRED


def test_one_failed_target_propagates_to_every_served_legacy_need():
    requirement, target, steps = _document_target(3, ("bear_5", "kill_4", "kill_22"))
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=steps)
    outcomes = {"l1_3": StepStatus.ZERO_RESULTS}  # exhausted, no alt-group, no further steps eligible
    ledger = project_step_results_to_coverage(graph, outcomes)
    for need_id in ("bear_5", "kill_4", "kill_22"):
        assert ledger.results[need_id].acquisition_status is AcquisitionStatus.FAILED


def test_not_publicly_available_maps_to_not_publicly_available_status():
    step = AcquisitionStep(
        step_id="npa", target_id="t_npa", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    requirement = EvidenceRequirement(
        requirement_id="req_npa", serves_legacy_need_ids=("bear_0",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY,
    )
    target = AcquisitionTarget(
        target_id="t_npa", target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE,
        required_step_ids=("npa",), serves_requirement_ids=("req_npa",),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(step,))
    outcomes = {"npa": StepStatus.NOT_PUBLICLY_AVAILABLE}
    ledger = project_step_results_to_coverage(graph, outcomes)
    assert ledger.results["bear_0"].acquisition_status is AcquisitionStatus.NOT_PUBLICLY_AVAILABLE
    assert ledger.results["bear_0"].acquisition_status is not AcquisitionStatus.ACQUIRED


def test_budget_skipped_step_maps_to_skipped_due_to_budget():
    requirement, target, steps = _document_target(4, ("bear_0",))
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=steps)
    outcomes = {"l1_4": StepStatus.SKIPPED_DUE_TO_BUDGET}
    ledger = project_step_results_to_coverage(graph, outcomes)
    assert ledger.results["bear_0"].acquisition_status is AcquisitionStatus.SKIPPED_DUE_TO_BUDGET


def test_not_implemented_adapter_maps_to_not_implemented_status():
    step = AcquisitionStep(
        step_id="l1", target_id="t_ni", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        completion_condition=StepStatus.URL_RESOLVED, implementation_status=ImplementationStatus.DECLARED,
    )
    requirement = EvidenceRequirement(
        requirement_id="req_ni", serves_legacy_need_ids=("bear_0",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CONTRADICTION,
    )
    target = AcquisitionTarget(
        target_id="t_ni", target_kind=TargetKind.FORM4_FILING,
        required_step_ids=("l1",), serves_requirement_ids=("req_ni",),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(step,))
    outcomes = {"l1": StepStatus.FAILED}
    ledger = project_step_results_to_coverage(graph, outcomes)
    assert ledger.results["bear_0"].acquisition_status is AcquisitionStatus.NOT_IMPLEMENTED


def test_coverage_summary_always_totals_49():
    from investment_research.research.acquisition_planning import build_acquisition_plan
    from investment_research.research.coverage_ledger import summarize_coverage

    graph = build_source_routing_graph()
    ledger = project_step_results_to_coverage(graph, {})  # nothing attempted at all
    plan = build_acquisition_plan()  # Phase 2's coarser plan, reused only for grouped_acquisition_tasks
    summary = summarize_coverage(ledger, plan)
    assert summary.total_legacy_needs == 49
    assert summary.balanced
    assert summary.unsearched == 49  # nothing attempted -> everything UNSEARCHED


def test_no_silent_drop_across_the_real_49_row_catalog():
    graph = build_source_routing_graph()
    ledger = project_step_results_to_coverage(graph, {})
    assert len(ledger.results) == 49
    assert {n.legacy_need_id for n in LEGACY_CATALOG} == set(ledger.results)
