"""Phase 3B requirement 9: five distinct concepts that must never be
confused with one another:

* ``RequirementCriticality.REQUIRED`` -- always counts as blocking if
  unresolved (an issuer's own statutory disclosure obligation).
* ``RequirementCriticality.CONDITIONAL_BLOCKING`` -- MAY block, pending a
  materiality assessment this repository does not implement yet. Reported
  as ``pending_materiality``/``requires_materiality_assessment``, never
  resolved either way.
* ``RequirementCriticality.BEST_EFFORT`` -- never blocks completion.
* "Not publicly available" (``AcquisitionMethod.NOT_PUBLICLY_AVAILABLE`` /
  ``TargetAcquisitionOutcome.ACQUISITION_EXHAUSTED_NOT_PUBLIC``) -- a
  CONCLUSIVE resolution about a specific document's availability. Distinct
  from criticality: a REQUIRED requirement can resolve not-public (and still
  count as blocking), and a CONDITIONAL_BLOCKING requirement can resolve
  not-public (and count as pending-materiality, never folded into
  blocking_unresolved).

The FDA regulator-confirmation sub-requirement's own non-public correspondence
must never, by itself, force the whole research run to BLOCKED -- see
``compute_plan_status``'s per-requirement accounting and
``AcquisitionExecutor``'s ``not_public_requirements``/
``pending_materiality_requirements`` diagnostics counters.
"""

from __future__ import annotations

from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentStore
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    PlanStatus,
    RequirementCriticality,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetAcquisitionOutcome,
    TargetKind,
    compute_plan_status,
    target_acquisition_outcome,
)
from investment_research.schemas.enums import ResearchDomain


def _npa_target(tag: str, criticality: RequirementCriticality):
    tid, rid = f"target_{tag}", f"req_{tag}"
    step = AcquisitionStep(
        step_id=f"npa_{tag}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{tag}",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.REGULATORY,
        criticality=criticality,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE,
        required_step_ids=(f"npa_{tag}",), serves_requirement_ids=(rid,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(step,))


def test_required_and_not_public_together_count_as_blocking():
    graph = _npa_target("a", RequirementCriticality.REQUIRED)
    status, blocking, pending = compute_plan_status(graph)
    assert status is PlanStatus.NOT_EXECUTABLE
    assert blocking == 1
    assert pending == 0


def test_conditional_blocking_and_not_public_together_count_as_pending_never_blocking():
    graph = _npa_target("b", RequirementCriticality.CONDITIONAL_BLOCKING)
    status, blocking, pending = compute_plan_status(graph)
    assert status is PlanStatus.NOT_EXECUTABLE
    assert blocking == 0
    assert pending == 1


def test_best_effort_and_not_public_together_count_as_neither():
    graph = _npa_target("c", RequirementCriticality.BEST_EFFORT)
    status, blocking, pending = compute_plan_status(graph)
    assert blocking == 0
    assert pending == 0


def test_not_public_outcome_is_the_same_regardless_of_criticality():
    """"Not publicly available" is a fact about the DOCUMENT, not about how
    critical the requirement is -- the TargetAcquisitionOutcome must read
    identically across all three criticality levels."""
    for criticality in RequirementCriticality:
        graph = _npa_target("d", criticality)
        target = graph.targets[0]
        outcome = target_acquisition_outcome(target, graph.steps, {"npa_d": StepStatus.NOT_PUBLICLY_AVAILABLE})
        assert outcome is TargetAcquisitionOutcome.ACQUISITION_EXHAUSTED_NOT_PUBLIC


def test_requires_materiality_assessment_is_true_only_for_conditional_blocking():
    for criticality in RequirementCriticality:
        requirement = EvidenceRequirement(
            requirement_id="r", serves_legacy_need_ids=("x",),
            subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.REGULATORY,
            criticality=criticality,
        )
        assert requirement.requires_materiality_assessment == (criticality is RequirementCriticality.CONDITIONAL_BLOCKING)


def test_executor_diagnostics_distinguish_not_public_from_pending_materiality():
    """A CONDITIONAL_BLOCKING requirement whose target resolves not-public
    increments ``not_public_requirements`` (a conclusive resolution),
    while pending_materiality_requirements is reserved for a
    CONDITIONAL_BLOCKING requirement that is NOT yet conclusively resolved
    either way -- the two counters must never overlap for the same
    requirement in the same run."""
    graph = _npa_target("e", RequirementCriticality.CONDITIONAL_BLOCKING)
    executor = AcquisitionExecutor(adapters={}, document_store=DocumentStore())
    report = executor.run(graph)
    assert report.diagnostics.not_public_requirements == 1
    assert report.diagnostics.pending_materiality_requirements == 0


def test_a_conditional_blocking_requirement_that_merely_failed_is_pending_not_not_public():
    """A CONDITIONAL_BLOCKING requirement whose target FAILED for an
    unrelated reason (no adapter registered) is not "confirmed not public"
    -- that label is reserved for AcquisitionMethod.NOT_PUBLICLY_AVAILABLE's
    own conclusive resolution. Any non-ACQUIRED resolution of a
    CONDITIONAL_BLOCKING requirement reads as pending_materiality, since
    this repository does not yet implement the assessment that would
    resolve it either way (Phase 3A requirement 3)."""
    step = AcquisitionStep(
        step_id="s", target_id="t", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        completion_condition=StepStatus.URL_RESOLVED,
    )
    requirement = EvidenceRequirement(
        requirement_id="r", serves_legacy_need_ids=("x",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY, criticality=RequirementCriticality.CONDITIONAL_BLOCKING,
    )
    target = AcquisitionTarget(target_id="t", target_kind=TargetKind.FORM4_FILING, required_step_ids=("s",), serves_requirement_ids=("r",))
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(step,))
    executor = AcquisitionExecutor(adapters={}, document_store=DocumentStore())
    report = executor.run(graph)
    assert report.outcomes["s"] is StepStatus.FAILED
    assert report.diagnostics.not_public_requirements == 0
    assert report.diagnostics.pending_materiality_requirements == 1
