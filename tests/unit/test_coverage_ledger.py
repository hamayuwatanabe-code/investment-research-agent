"""CoverageLedger: inventory, fan-out and failure-safety unit tests.

See tests/regression/test_phase2_offline_acceptance.py for the full,
end-to-end Offline Acceptance Harness (sections A-E of the Phase 2
authorization) exercising AcquisitionPlan + CoverageLedger + BudgetFeasibility
together.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import (
    AcquisitionMethod,
    AcquisitionTask,
    build_acquisition_plan,
)
from investment_research.research.checks import (
    AcquisitionStatus,
    LegacyResearchNeedResult,
    SubjectScope,
)
from investment_research.research.coverage_ledger import CoverageLedger, summarize_coverage
from investment_research.schemas.enums import ResearchDomain


# --- A. Inventory ------------------------------------------------------------
def test_initialize_creates_a_row_for_every_one_of_the_49_needs():
    ledger = CoverageLedger.initialize()
    assert len(ledger.results) == 49
    assert ledger.missing_need_ids() == set()
    assert all(r.acquisition_status is AcquisitionStatus.UNSEARCHED for r in ledger.results.values())


def test_grouping_into_acquisition_tasks_does_not_remove_any_original_row():
    plan = build_acquisition_plan()
    ledger = CoverageLedger.initialize()
    # Every legacy_need_id referenced by the plan still has its own ledger row.
    for task in plan.tasks:
        for need_id in task.serves_legacy_need_ids:
            assert need_id in ledger.results


# --- B. Fan-out --------------------------------------------------------------
def test_one_task_result_fans_out_to_all_three_served_needs_identically():
    task = AcquisitionTask(
        task_id="task_fanout",
        equivalence_key="company going concern",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CAPITAL_STRUCTURE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        serves_legacy_need_ids=("bear_6", "kill_5", "kill_23"),
    )
    ledger = CoverageLedger.initialize()
    ledger.apply_task_result(
        task, status=AcquisitionStatus.ACQUIRED, document_ids=("doc_1", "doc_2")
    )

    for need_id in ("bear_6", "kill_5", "kill_23"):
        result = ledger.results[need_id]
        assert result.acquisition_status is AcquisitionStatus.ACQUIRED
        assert result.document_ids == ("doc_1", "doc_2")
        assert result.research_need_id == "task_fanout"
        # The three legacy ids remain distinct rows -- not merged into one.
    assert len({id(ledger.results[n]) for n in ("bear_6", "kill_5", "kill_23")}) == 3


def test_fan_out_never_silently_drops_one_of_the_served_needs():
    task = AcquisitionTask(
        task_id="task_fanout2",
        equivalence_key="company fda concern",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        serves_legacy_need_ids=("bear_0", "kill_0", "kill_18"),
    )
    ledger = CoverageLedger.initialize()
    ledger.apply_task_result(task, status=AcquisitionStatus.ACQUIRED_ZERO_RESULTS)

    updated = {
        need_id
        for need_id, result in ledger.results.items()
        if result.acquisition_status is AcquisitionStatus.ACQUIRED_ZERO_RESULTS
    }
    assert updated == {"bear_0", "kill_0", "kill_18"}


# --- C. Failure safety --------------------------------------------------------
def test_no_results_is_never_recorded_as_acquired():
    task = AcquisitionTask(
        task_id="task_zero",
        equivalence_key="company delisting",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CAPITAL_STRUCTURE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        serves_legacy_need_ids=("bear_9",),
    )
    ledger = CoverageLedger.initialize()
    ledger.apply_task_result(task, status=AcquisitionStatus.ACQUIRED_ZERO_RESULTS)
    assert ledger.results["bear_9"].acquisition_status is AcquisitionStatus.ACQUIRED_ZERO_RESULTS
    assert ledger.results["bear_9"].acquisition_status is not AcquisitionStatus.ACQUIRED


def test_skipped_due_to_budget_is_never_confused_with_failed_or_unsearched():
    task = AcquisitionTask(
        task_id="task_budget",
        equivalence_key="company auditor",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CONTRADICTION,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        serves_legacy_need_ids=("bear_11",),
    )
    ledger = CoverageLedger.initialize()
    ledger.apply_task_result(
        task, status=AcquisitionStatus.SKIPPED_DUE_TO_BUDGET, reason="discovery quota exhausted"
    )
    status = ledger.results["bear_11"].acquisition_status
    assert status is AcquisitionStatus.SKIPPED_DUE_TO_BUDGET
    assert status is not AcquisitionStatus.FAILED
    assert status is not AcquisitionStatus.UNSEARCHED


def test_untouched_task_leaves_its_needs_unsearched_not_implicitly_complete():
    ledger = CoverageLedger.initialize()
    # No apply_task_result call at all for any task -- everything must stay
    # UNSEARCHED, never silently promoted to any completed-looking status.
    assert all(r.acquisition_status is AcquisitionStatus.UNSEARCHED for r in ledger.results.values())


def test_not_publicly_available_is_never_recorded_as_zero_result_search():
    task = AcquisitionTask(
        task_id="task_npa",
        equivalence_key="program type c meeting minutes text",
        subject_scope=SubjectScope.PROGRAM,
        domain=ResearchDomain.REGULATORY,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        serves_legacy_need_ids=("synthetic_minutes",),
    )
    ledger = CoverageLedger(
        results={
            "synthetic_minutes": LegacyResearchNeedResult(
                legacy_need_id="synthetic_minutes", acquisition_status=AcquisitionStatus.UNSEARCHED
            )
        }
    )
    ledger.apply_task_result(
        task,
        status=AcquisitionStatus.NOT_PUBLICLY_AVAILABLE,
        reason="document is not publicly available; never sent to search",
    )
    status = ledger.results["synthetic_minutes"].acquisition_status
    assert status is AcquisitionStatus.NOT_PUBLICLY_AVAILABLE
    assert status is not AcquisitionStatus.ACQUIRED_ZERO_RESULTS


# --- E. Coverage accounting ---------------------------------------------------
def test_coverage_summary_buckets_sum_to_the_total():
    plan = build_acquisition_plan()
    ledger = CoverageLedger.initialize()
    # Drive a mix of outcomes across a few tasks; leave the rest UNSEARCHED.
    statuses = [
        AcquisitionStatus.ACQUIRED,
        AcquisitionStatus.ACQUIRED_ZERO_RESULTS,
        AcquisitionStatus.FAILED,
        AcquisitionStatus.SKIPPED_DUE_TO_BUDGET,
    ]
    for task, status in zip(plan.tasks[:4], statuses, strict=True):
        ledger.apply_task_result(task, status=status)

    summary = summarize_coverage(ledger, plan)
    assert summary.total_legacy_needs == 49
    assert summary.grouped_acquisition_tasks == 31
    assert summary.balanced
    assert summary.accounted_total() == 49
