"""Phase 2 Offline Acceptance Harness (sections A-F of the Phase 2
authorization). Synthetic fixtures only -- "TESTCO"/generic placeholders, no
real ticker or issuer. No network, no LLM call, no pipeline wiring.

This is the single end-to-end test asserting Phase 2's actual pass condition,
as stated by the authorizing message: not "live research succeeds", but --

* all 49 legacy needs never go missing,
* acquisition and evidence are never conflated,
* the same acquisition is never double-planned,
* 4+ searches are never misjudged safe,
* under-researched areas are explicit and can halt the run.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import (
    AcquisitionMethod,
    AcquisitionPlan,
    AcquisitionTask,
    build_acquisition_plan,
)
from investment_research.research.budget_feasibility import assess_budget_feasibility
from investment_research.research.checks import (
    AcquisitionStatus,
    SubjectScope,
)
from investment_research.research.coverage_ledger import CoverageLedger, summarize_coverage
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.research.legacy_catalog import LEGACY_CATALOG
from investment_research.schemas.enums import REQUIRED_RESEARCH_DOMAINS, ResearchDomain
from investment_research.scoring.kill_gate import KILL_RULES


# =========================== A. Inventory ===================================
def test_a_all_49_legacy_needs_exist():
    assert len(LEGACY_CATALOG) == 49
    assert len({n.legacy_need_id for n in LEGACY_CATALOG}) == 49


def test_a_all_49_needs_have_a_coverage_ledger_row():
    ledger = CoverageLedger.initialize()
    assert len(ledger.results) == 49
    assert ledger.missing_need_ids() == set()


def test_a_grouping_into_acquisition_tasks_keeps_every_original_row():
    plan = build_acquisition_plan()
    assert len(plan.tasks) == 31
    assert plan.all_served_legacy_need_ids() == {n.legacy_need_id for n in LEGACY_CATALOG}


def test_a_kill_rules_and_required_domains_are_untouched():
    assert len(KILL_RULES) == 19
    assert len(REQUIRED_RESEARCH_DOMAINS) == 6


# =========================== B. Dedup =======================================
def test_b_one_acquisition_task_serves_multiple_legacy_needs():
    plan = build_acquisition_plan()
    multi = [t for t in plan.tasks if len(t.serves_legacy_need_ids) > 1]
    assert multi, "at least one real group must collapse multiple legacy needs"
    assert any(len(t.serves_legacy_need_ids) == 3 for t in multi)


def test_b_same_document_requested_by_multiple_tasks_is_fetched_once():
    """DocumentStore identity, not AcquisitionPlan, is what prevents a
    duplicate fetch: two tasks resolving to the same URL/content register as
    ONE Document in the store, however many tasks asked for it."""
    store = DocumentStore()
    from investment_research.collectors.documents import Document
    from investment_research.schemas.enums import DocumentAuthority

    url = "https://ir.testco.example/press/shared"
    text = "Shared disclosure both tasks were looking for."
    first = store.put(
        Document(doc_id="d1", url=url, text=text, title="shared", authority=DocumentAuthority.COMPANY_IR),
        document_id="d1",
        canonical_url=url,
    )
    second = store.put(
        Document(doc_id="d2", url=url, text=text, title="shared", authority=DocumentAuthority.COMPANY_IR),
        document_id="d2",
        canonical_url=url,
    )
    # Idempotent: the second "fetch" of identical content at the same URL
    # resolves to the SAME StoredDocument, not a second retrieval.
    assert first.document_id == second.document_id
    assert len(store.resolve_by_canonical_url(url)) == 1


def test_b_same_accession_different_filename_is_a_separate_document():
    store = DocumentStore()
    from investment_research.collectors.documents import Document
    from investment_research.schemas.enums import DocumentAuthority

    store.put(
        Document(doc_id="idx", url="https://sec.gov/x/0001/", title="index", authority=DocumentAuthority.STATUTORY_FILING),
        document_id="idx",
        accession="0001-26-000999",
        filename="",
        document_role=DocumentRole.FILING_INDEX,
    )
    store.put(
        Document(doc_id="primary", url="https://sec.gov/x/0001/primary.htm", title="primary", text="10-Q body", authority=DocumentAuthority.STATUTORY_FILING),
        document_id="primary",
        accession="0001-26-000999",
        filename="primary.htm",
        document_role=DocumentRole.PRIMARY_DOCUMENT,
    )
    rows = store.resolve_by_accession("0001-26-000999")
    assert {r.document_id for r in rows} == {"idx", "primary"}


def test_b_same_content_different_authority_is_a_separate_document():
    store = DocumentStore()
    from investment_research.collectors.documents import Document
    from investment_research.schemas.enums import DocumentAuthority

    same_text = "TESTCO announced a financing update."
    store.put(
        Document(doc_id="issuer", url="https://ir.testco.example/p1", text=same_text, title="p1", authority=DocumentAuthority.COMPANY_IR),
        document_id="issuer",
        canonical_url="https://ir.testco.example/p1",
    )
    store.put(
        Document(doc_id="wire", url="https://wire.example/p1", text=same_text, title="p1", authority=DocumentAuthority.INDEPENDENT),
        document_id="wire",
        canonical_url="https://wire.example/p1",
    )
    assert store.get("issuer") is not None
    assert store.get("wire") is not None
    assert store.get("issuer").document_id != store.get("wire").document_id

    relations = store.duplicate_content_relations()
    assert len(relations) == 1
    assert set(relations[0].document_ids) == {"issuer", "wire"}
    assert relations[0].same_authority is False


def test_b_content_duplicate_is_recorded_as_a_relation_only_never_a_merge():
    store = DocumentStore()
    from investment_research.collectors.documents import Document
    from investment_research.schemas.enums import DocumentAuthority

    same_text = "Duplicate text at two URLs."
    store.put(
        Document(doc_id="a", url="https://x.example/a", text=same_text, title="a", authority=DocumentAuthority.INDEPENDENT),
        document_id="a",
        canonical_url="https://x.example/a",
    )
    store.put(
        Document(doc_id="b", url="https://x.example/b", text=same_text, title="b", authority=DocumentAuthority.INDEPENDENT),
        document_id="b",
        canonical_url="https://x.example/b",
    )
    # Two DISTINCT documents remain -- the relation is informational only.
    assert store.get("a") is not None and store.get("b") is not None
    assert len(store.duplicate_content_relations()) == 1


def test_b_old_document_version_stays_retrievable():
    store = DocumentStore()
    from investment_research.collectors.documents import Document
    from investment_research.schemas.enums import DocumentAuthority

    url = "https://ir.testco.example/press/rolling"
    v1 = store.put(
        Document(doc_id="v1", url=url, text="Original text.", title="p", authority=DocumentAuthority.COMPANY_IR),
        document_id="v1",
        canonical_url=url,
    )
    v2 = store.put(
        Document(doc_id="v2", url=url, text="Corrected text.", title="p", authority=DocumentAuthority.COMPANY_IR),
        document_id="v2",
        canonical_url=url,
    )
    assert v2.previous_version_id == v1.document_id
    assert store.get(v1.document_id) is not None
    assert store.get(v1.document_id).document.text == "Original text."


# =========================== C. Failure safety ==============================
def _task(need_ids: tuple[str, ...], method: AcquisitionMethod = AcquisitionMethod.WEB_SEARCH_DISCOVERY) -> AcquisitionTask:
    return AcquisitionTask(
        task_id=f"task_{'_'.join(need_ids)}",
        equivalence_key="company synthetic",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY,
        acquisition_method=method,
        serves_legacy_need_ids=need_ids,
    )


def test_c_no_results_is_never_acquired():
    ledger = CoverageLedger.initialize()
    task = _task(("bear_0",))
    ledger.apply_task_result(task, status=AcquisitionStatus.ACQUIRED_ZERO_RESULTS)
    assert ledger.results["bear_0"].acquisition_status is AcquisitionStatus.ACQUIRED_ZERO_RESULTS
    assert ledger.results["bear_0"].acquisition_status is not AcquisitionStatus.ACQUIRED


def test_c_document_discovery_alone_is_not_evidenced():
    """Acquisition status has no EVIDENCED/CONTRADICTED value at all --
    finding a document is not confirming a claim (requirement 6)."""
    valid_values = {s.value for s in AcquisitionStatus}
    assert "EVIDENCED" not in valid_values
    assert "CONTRADICTED" not in valid_values


def test_c_metadata_only_content_kind_is_not_confused_with_body_confirmed():
    """ContentKind (how much of a document is in hand) is orthogonal to
    AcquisitionStatus (whether acquisition happened) -- METADATA_ONLY content
    must never be reported as ACQUIRED in a sense that implies the body was
    read; this harness keeps the two questions on two different types."""
    from investment_research.schemas.enums import ContentKind

    assert ContentKind.METADATA_ONLY.is_primary_text is False
    # AcquisitionStatus.ACQUIRED says a task's document(s) were retrieved; it
    # carries no claim about content_kind, which the Document/Fact layer
    # tracks separately.
    ledger = CoverageLedger.initialize()
    task = _task(("bear_1",))
    ledger.apply_task_result(task, status=AcquisitionStatus.ACQUIRED, document_ids=("d1",))
    result = ledger.results["bear_1"]
    assert not hasattr(result, "content_kind")


def test_c_not_publicly_available_is_never_searched_zero_result():
    ledger = CoverageLedger.initialize()
    task = _task(("kill_9",), method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE)
    ledger.apply_task_result(task, status=AcquisitionStatus.NOT_PUBLICLY_AVAILABLE)
    status = ledger.results["kill_9"].acquisition_status
    assert status is AcquisitionStatus.NOT_PUBLICLY_AVAILABLE
    assert status is not AcquisitionStatus.ACQUIRED_ZERO_RESULTS


def test_c_skipped_due_to_budget_is_never_confused_with_incomplete_response():
    """AcquisitionStatus has no INCOMPLETE_RESPONSE value at all (that is
    IntentStatus's concept, at the search-batching layer, deliberately not
    reused here) -- a budget-skip at the AcquisitionTask/CoverageLedger layer
    is SKIPPED_DUE_TO_BUDGET, unambiguous on its own."""
    valid_values = {s.value for s in AcquisitionStatus}
    assert "INCOMPLETE_RESPONSE" not in valid_values
    ledger = CoverageLedger.initialize()
    task = _task(("kill_11",))
    ledger.apply_task_result(task, status=AcquisitionStatus.SKIPPED_DUE_TO_BUDGET)
    assert ledger.results["kill_11"].acquisition_status is AcquisitionStatus.SKIPPED_DUE_TO_BUDGET


def test_c_unprocessed_task_never_becomes_implicitly_complete():
    ledger = CoverageLedger.initialize()
    build_acquisition_plan()  # tasks exist, but none are ever applied below
    assert all(r.acquisition_status is AcquisitionStatus.UNSEARCHED for r in ledger.results.values())


# =========================== D. Fan-out ======================================
def test_d_three_legacy_needs_served_by_one_task_all_get_the_same_result():
    plan = build_acquisition_plan()
    task = plan.task_for_need("bear_0")
    assert task is not None
    assert set(task.serves_legacy_need_ids) == {"bear_0", "kill_0", "kill_18"}

    ledger = CoverageLedger.initialize()
    ledger.apply_task_result(task, status=AcquisitionStatus.ACQUIRED, document_ids=("doc_x",))

    for need_id in ("bear_0", "kill_0", "kill_18"):
        result = ledger.results[need_id]
        assert result.acquisition_status is AcquisitionStatus.ACQUIRED
        assert result.document_ids == ("doc_x",)

    # No silent drop: all three remain independently addressable rows.
    assert len(ledger.results) == 49


# =========================== E. Coverage accounting ==========================
def test_e_every_bucket_sums_to_the_total_across_a_full_synthetic_run():
    plan = build_acquisition_plan()
    ledger = CoverageLedger.initialize()

    outcomes = [
        AcquisitionStatus.ACQUIRED,
        AcquisitionStatus.ACQUIRED_ZERO_RESULTS,
        AcquisitionStatus.FAILED,
        AcquisitionStatus.SKIPPED_DUE_TO_BUDGET,
        AcquisitionStatus.NOT_PUBLICLY_AVAILABLE,
        AcquisitionStatus.MANUAL_VERIFICATION_REQUIRED,
    ]
    for task, status in zip(plan.tasks, outcomes, strict=False):
        ledger.apply_task_result(task, status=status)
    # Remaining tasks (beyond len(outcomes)) are left UNSEARCHED deliberately.

    summary = summarize_coverage(ledger, plan)
    assert summary.total_legacy_needs == 49
    assert summary.grouped_acquisition_tasks == 31
    assert summary.balanced
    assert summary.accounted_total() == summary.total_legacy_needs


# =========================== F. Budget: 4+ searches is never "safe" =========
def test_f_4_or_more_web_search_tasks_is_flagged_unsafe_and_run_can_be_halted():
    plan = AcquisitionPlan(
        tasks=tuple(
            AcquisitionTask(
                task_id=f"task_{i}",
                equivalence_key=f"company synthetic {i}",
                subject_scope=SubjectScope.COMPANY,
                domain=ResearchDomain.REGULATORY,
                acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
                serves_legacy_need_ids=(f"synthetic_{i}",),
            )
            for i in range(4)
        )
    )
    feasibility = assess_budget_feasibility(plan)
    assert feasibility.unsafe is True
    assert feasibility.feasible is False
    # The halting decision itself belongs to a future caller (Phase 2 does
    # not wire this into the pipeline); what this harness asserts is that the
    # information needed to halt -- an explicit, non-silent unsafe/infeasible
    # flag with a shortfall -- exists and is never suppressed.
    assert feasibility.shortfall_tokens > 0


def test_f_real_catalog_plan_is_reported_unsafe_not_silently_optimistic():
    """The real 49/31-task plan, with every task defaulting to
    WEB_SEARCH_DISCOVERY, is reported UNSAFE/infeasible against a 60,000
    discovery budget -- this harness does not let that be hidden."""
    plan = build_acquisition_plan()
    feasibility = assess_budget_feasibility(plan)
    assert feasibility.unsafe is True
    assert feasibility.feasible is False
    assert feasibility.expected_server_tool_uses == 31
