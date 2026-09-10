"""Projecting Source Routing Graph step outcomes onto the CoverageLedger.

Phase 2.7 scope only: a pure function, no network, no execution. It answers
"given one set of (real or simulated) step outcomes, what does that mean for
every one of the 49 original LegacyResearchNeed rows" -- bridging the
fine-grained step-DAG model (``source_routing.py``) back onto the coarser,
per-legacy-need vocabulary ``coverage_ledger.CoverageLedger`` already uses.

The mapping is deliberately conservative about what counts as ACQUIRED:
only ``TargetAcquisitionOutcome.ACQUIRED`` (full content acquisition, per
``source_routing.is_target_complete``) maps to ``AcquisitionStatus.ACQUIRED``.
Every other outcome -- metadata-only, budget-skipped, not-implemented,
not-publicly-available, still in progress -- maps to something else, so a
legacy need's row can never read ACQUIRED merely because its target reached
some intermediate state.
"""

from __future__ import annotations

from collections.abc import Mapping

from .checks import AcquisitionStatus, LegacyResearchNeed, LegacyResearchNeedResult
from .coverage_ledger import CoverageLedger
from .legacy_catalog import LEGACY_CATALOG
from .source_routing import (
    SourceRoutingGraph,
    StepStatus,
    TargetAcquisitionOutcome,
    target_acquisition_outcome,
)

_OUTCOME_TO_STATUS: dict[TargetAcquisitionOutcome, AcquisitionStatus] = {
    TargetAcquisitionOutcome.ACQUIRED: AcquisitionStatus.ACQUIRED,
    TargetAcquisitionOutcome.ACQUISITION_EXHAUSTED_NOT_PUBLIC: AcquisitionStatus.NOT_PUBLICLY_AVAILABLE,
    TargetAcquisitionOutcome.BLOCKED_BY_BUDGET: AcquisitionStatus.SKIPPED_DUE_TO_BUDGET,
    TargetAcquisitionOutcome.NOT_IMPLEMENTED: AcquisitionStatus.NOT_IMPLEMENTED,
    TargetAcquisitionOutcome.FAILED: AcquisitionStatus.FAILED,
    TargetAcquisitionOutcome.NOT_ATTEMPTED: AcquisitionStatus.UNSEARCHED,
    #: A target reached SOME step (e.g. LOCATE/metadata) but not full content
    #: acquisition. Never ACQUIRED -- reads as UNSEARCHED, the same "not yet
    #: resolved" bucket as NOT_ATTEMPTED, since neither means the claim was
    #: settled (requirement 6: body-fetch-not-run must propagate as
    #: unresolved, not as a partial success).
    TargetAcquisitionOutcome.IN_PROGRESS: AcquisitionStatus.UNSEARCHED,
}

#: Priority when a requirement is served by more than one target and their
#: outcomes disagree: the WORST (most conclusively unresolved) outcome wins,
#: never the best -- a requirement is only as done as its weakest target.
_STATUS_PRIORITY: dict[AcquisitionStatus, int] = {
    AcquisitionStatus.ACQUIRED: 0,
    AcquisitionStatus.ACQUIRED_ZERO_RESULTS: 1,
    AcquisitionStatus.UNSEARCHED: 2,
    AcquisitionStatus.SKIPPED_DUE_TO_BUDGET: 3,
    AcquisitionStatus.SKIPPED_DUE_TO_DIRECT_COVERAGE: 3,
    AcquisitionStatus.NOT_IMPLEMENTED: 4,
    AcquisitionStatus.NOT_PUBLICLY_AVAILABLE: 4,
    AcquisitionStatus.MANUAL_VERIFICATION_REQUIRED: 4,
    AcquisitionStatus.FAILED: 5,
}


def _combine(statuses: list[AcquisitionStatus]) -> AcquisitionStatus:
    if not statuses:
        return AcquisitionStatus.UNSEARCHED
    if all(s == AcquisitionStatus.ACQUIRED for s in statuses):
        return AcquisitionStatus.ACQUIRED
    return max(statuses, key=lambda s: _STATUS_PRIORITY.get(s, 2))


def project_step_results_to_coverage(
    graph: SourceRoutingGraph,
    step_outcomes: Mapping[str, StepStatus],
    legacy_catalog: tuple[LegacyResearchNeed, ...] = LEGACY_CATALOG,
) -> CoverageLedger:
    """Project one set of step outcomes onto a fresh ``CoverageLedger``.

    Every one of ``legacy_catalog``'s 49 rows is present in the result (via
    ``CoverageLedger.initialize``) whether or not the graph's requirements
    reference it -- a legacy need this graph never even raised a requirement
    for stays UNSEARCHED, never silently absent. A target's failure fans out
    to EVERY legacy_need_id its requirement(s) serve, exactly like
    ``CoverageLedger.apply_task_result`` -- no partial/silent drop.
    """
    ledger = CoverageLedger.initialize(legacy_catalog)
    touched: set[str] = set()

    for requirement in graph.requirements:
        target_statuses: list[AcquisitionStatus] = []
        for target in graph.targets_for_requirement(requirement.requirement_id):
            steps = graph.steps_for_target(target.target_id)
            outcome = target_acquisition_outcome(target, steps, step_outcomes)
            target_statuses.append(_OUTCOME_TO_STATUS[outcome])
        status = _combine(target_statuses)

        for need_id in requirement.serves_legacy_need_ids:
            if need_id not in ledger.results:
                continue
            # A need can be served by more than one requirement (Phase 2.6's
            # fda_dual/sec_chain_with_exhibit): combine conservatively so an
            # already-recorded worse outcome for this need is never
            # overwritten by a better one from a different requirement.
            existing = ledger.results[need_id]
            combined = _combine([existing.acquisition_status, status]) if need_id in touched else status
            touched.add(need_id)
            ledger.results[need_id] = LegacyResearchNeedResult(
                legacy_need_id=need_id,
                acquisition_status=combined,
                research_need_id=requirement.requirement_id,
            )

    return ledger
