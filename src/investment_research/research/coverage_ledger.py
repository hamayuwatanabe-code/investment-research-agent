"""CoverageLedger: per-``LegacyResearchNeed`` acquisition state, fanned out
from whichever ``AcquisitionTask`` serves it.

Phase 2 scope only: pure offline bookkeeping, no network, no execution. This
module answers "given one AcquisitionTask's outcome, what does that mean for
every LegacyResearchNeed it serves" and keeps per-need accounting so a
many-to-one fan-out never silently drops one of the original 49 ids.

Never conflates acquisition with evidence or judgment -- see
``checks.py``'s module docstring for the three-way boundary this whole
module respects: a ``CoverageLedger`` row says whether a document was
retrieved, never whether a claim in it was confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .acquisition_planning import AcquisitionPlan, AcquisitionTask
from .checks import AcquisitionStatus, LegacyResearchNeed, LegacyResearchNeedResult
from .legacy_catalog import LEGACY_CATALOG


@dataclass
class CoverageLedger:
    """``legacy_need_id`` -> ``LegacyResearchNeedResult``, one row per
    ORIGINAL legacy need -- however many of them collapse into one
    ``AcquisitionTask``, each keeps its own row.
    """

    results: dict[str, LegacyResearchNeedResult] = field(default_factory=dict)

    @classmethod
    def initialize(
        cls, catalog: tuple[LegacyResearchNeed, ...] = LEGACY_CATALOG
    ) -> CoverageLedger:
        """Every legacy need starts UNSEARCHED -- present and accounted for,
        never silently absent (Phase 2 requirement A: "全件にCoverageLedger
        行がある")."""
        return cls(
            results={
                need.legacy_need_id: LegacyResearchNeedResult(
                    legacy_need_id=need.legacy_need_id,
                    acquisition_status=AcquisitionStatus.UNSEARCHED,
                )
                for need in catalog
            }
        )

    def apply_task_result(
        self,
        task: AcquisitionTask,
        *,
        status: AcquisitionStatus,
        document_ids: tuple[str, ...] = (),
        reason: str = "",
    ) -> None:
        """Fan out one ``AcquisitionTask``'s outcome to every
        ``LegacyResearchNeed`` it serves (Phase 2 requirement D).

        Every id in ``task.serves_legacy_need_ids`` gets its OWN row
        updated -- none silently dropped -- all carrying the identical
        status/document_ids/reason, since they share one underlying
        acquisition. This never promotes any need to an evidence-level
        status: ``status`` here is an ``AcquisitionStatus``, not an
        evaluative verdict.
        """
        for need_id in task.serves_legacy_need_ids:
            self.results[need_id] = LegacyResearchNeedResult(
                legacy_need_id=need_id,
                acquisition_status=status,
                research_need_id=task.task_id,
                document_ids=document_ids,
                reason=reason,
            )

    def status_counts(self) -> dict[AcquisitionStatus, int]:
        counts: dict[AcquisitionStatus, int] = {}
        for result in self.results.values():
            counts[result.acquisition_status] = counts.get(result.acquisition_status, 0) + 1
        return counts

    def missing_need_ids(self, catalog: tuple[LegacyResearchNeed, ...] = LEGACY_CATALOG) -> set[str]:
        """Any of ``catalog``'s ids with no ledger row at all. Always empty
        immediately after ``initialize``, but exposed so the acceptance
        harness asserts this directly rather than trusting it silently."""
        return {need.legacy_need_id for need in catalog} - set(self.results)


@dataclass(frozen=True)
class CoverageSummary:
    """Per-run accounting (Phase 2 requirement E): every legacy need is
    counted in exactly one bucket, and the buckets must sum to the total.
    """

    total_legacy_needs: int
    grouped_acquisition_tasks: int
    acquired: int
    zero_results: int
    failed: int
    unsearched: int
    not_publicly_available: int
    manual_verification_required: int
    skipped_due_to_budget: int
    skipped_due_to_direct_coverage: int

    def accounted_total(self) -> int:
        return (
            self.acquired
            + self.zero_results
            + self.failed
            + self.unsearched
            + self.not_publicly_available
            + self.manual_verification_required
            + self.skipped_due_to_budget
            + self.skipped_due_to_direct_coverage
        )

    @property
    def balanced(self) -> bool:
        return self.accounted_total() == self.total_legacy_needs


def summarize_coverage(ledger: CoverageLedger, plan: AcquisitionPlan) -> CoverageSummary:
    counts = ledger.status_counts()

    def c(status: AcquisitionStatus) -> int:
        return counts.get(status, 0)

    return CoverageSummary(
        total_legacy_needs=len(ledger.results),
        grouped_acquisition_tasks=len(plan.tasks),
        acquired=c(AcquisitionStatus.ACQUIRED),
        zero_results=c(AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        failed=c(AcquisitionStatus.FAILED),
        unsearched=c(AcquisitionStatus.UNSEARCHED),
        not_publicly_available=c(AcquisitionStatus.NOT_PUBLICLY_AVAILABLE),
        manual_verification_required=c(AcquisitionStatus.MANUAL_VERIFICATION_REQUIRED),
        skipped_due_to_budget=c(AcquisitionStatus.SKIPPED_DUE_TO_BUDGET),
        skipped_due_to_direct_coverage=c(AcquisitionStatus.SKIPPED_DUE_TO_DIRECT_COVERAGE),
    )
