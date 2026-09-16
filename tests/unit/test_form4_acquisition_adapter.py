"""Phase 3E.1: Form4Adapter's issuer-driven discovery against real-Form4-
format fixtures (``tests/fixtures/form4_real_format/``, see that
directory's MANIFEST.md). No real network call anywhere in this file;
every URL and request count is asserted exactly.
"""

from __future__ import annotations

from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.research.form4_acquisition_adapter import (
    FORM4_ADAPTER_ID,
    Form4Adapter,
    Form4IssuerReference,
)
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    ImplementationStatus,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetAcquisitionOutcome,
    TargetKind,
)
from investment_research.schemas.enums import (
    ContentKind,
    DocumentAuthority,
    ResearchDomain,
)

from . import _form4_fixture_support as fx


def _form4_graph(tag: str = "a"):
    tid, rid = f"target_{tag}", f"req_{tag}"
    l1 = AcquisitionStep(
        step_id=f"l1_{tag}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id=FORM4_ADAPTER_ID,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    f = AcquisitionStep(
        step_id=f"f_{tag}", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=FORM4_ADAPTER_ID,
        depends_on_step_ids=(f"l1_{tag}",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    p = AcquisitionStep(
        step_id=f"p_{tag}", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=FORM4_ADAPTER_ID,
        depends_on_step_ids=(f"f_{tag}",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{tag}",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CONTRADICTION,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.FORM4_FILING,
        required_step_ids=(f"l1_{tag}", f"f_{tag}", f"p_{tag}"), serves_requirement_ids=(rid,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(l1, f, p))


def _run(ref, http, *, graph=None):
    graph = graph or _form4_graph()
    target = graph.targets[0]
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={FORM4_ADAPTER_ID: Form4Adapter(http)}, document_store=store)
    report = executor.run(graph, form4_references={target.target_id: ref})
    return report, store, target


def _by_step_prefix(report, prefix):
    return next(r for r in report.target_reports[0].step_results if r.step_id.startswith(prefix))


# --- issuer-driven discovery (Phase 3E.1 requirement 1) --------------------
def test_issuer_cik_alone_discovers_form4_candidates_never_pre_specified():
    """No accession, primary_document, or reporting-owner CIK is ever
    supplied by the caller -- LOCATE must discover them itself."""
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT)
    report, store, target = _run(ref, http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate_result = _by_step_prefix(report, "l1_")
    assert locate_result.payload["candidates"]
    for candidate in locate_result.payload["candidates"]:
        assert candidate["form"] in ("4", "4/A")


def test_issuer_ticker_alone_resolves_cik_and_discovers_candidates():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_ticker=fx.ISSUER_TICKER)
    report, store, target = _run(ref, http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert fx.ticker_map_url() in http.requested_urls
    locate_result = _by_step_prefix(report, "l1_")
    assert locate_result.payload["issuer_cik"] == fx.ISSUER_CIK_INT


def test_unknown_ticker_never_acquired():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_ticker="NOSUCHTICKER")
    report, store, target = _run(ref, http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    locate_result = _by_step_prefix(report, "l1_")
    assert locate_result.status is StepStatus.NOT_FOUND


def test_neither_cik_nor_ticker_fails_never_acquired():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run(Form4IssuerReference(), http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


def test_malformed_issuer_cik_refused():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik="not-a-cik")
    report, store, target = _run(ref, http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


# --- Forms 3/5 excluded (Phase 3E.1 requirement 1/3) -----------------------
def test_form3_never_becomes_a_candidate():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    locate_result = _by_step_prefix(report, "l1_")
    accessions = {c["accession"] for c in locate_result.payload["candidates"]}
    assert fx.accession("metadata_says_form3") not in accessions
    assert store.resolve_by_accession(fx.accession("metadata_says_form3")) == []


# --- zero results, malformed, CIK mismatch, budget exclusion --------------
def test_issuer_with_zero_form4_filings_is_zero_results_never_acquired():
    responses = dict(fx.default_responses())
    empty_submissions = fx.build_submissions_payload()
    empty_submissions["filings"]["recent"] = {
        "accessionNumber": [], "form": [], "primaryDocument": [], "filingDate": [], "reportDate": [],
    }
    import json
    responses[fx.submissions_url()] = fx.ok(json.dumps(empty_submissions))
    http = fx.FakeHttpClient(responses=responses)
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT)
    report, store, target = _run(ref, http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    locate_result = _by_step_prefix(report, "l1_")
    assert locate_result.status is StepStatus.ZERO_RESULTS


def test_cik_mismatch_candidate_excluded_never_acquired_as_evidence():
    """A candidate whose OWN ownership XML declares a DIFFERENT issuerCik
    than requested must never be stored as ACQUIRED evidence for this
    issuer (Phase 3E.1 requirement 1)."""
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    fetch_result = _by_step_prefix(report, "f_")
    mismatch_accession = fx.accession("cik_mismatch")
    fetched_accessions = {e["accession"] for e in fetch_result.payload.get("fetched", [])}
    failure_accessions = {e["accession"] for e in fetch_result.payload.get("fetch_failures", [])}
    assert mismatch_accession not in fetched_accessions
    assert mismatch_accession in failure_accessions
    assert store.resolve_by_accession(mismatch_accession) == []


def test_malformed_candidate_never_acquired_as_evidence():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    fetch_result = _by_step_prefix(report, "f_")
    failure_accessions = {e["accession"] for e in fetch_result.payload.get("fetch_failures", [])}
    assert fx.accession("malformed_xml") in failure_accessions
    assert store.resolve_by_accession(fx.accession("malformed_xml")) == []


def test_budget_exclusion_never_silently_fetches_beyond_max_candidates():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=3)
    report, store, target = _run(ref, http)
    locate_result = _by_step_prefix(report, "l1_")
    assert len(locate_result.payload["candidates"]) == 3
    assert locate_result.payload["excluded_candidates_count"] > 0
    fetch_result = _by_step_prefix(report, "f_")
    assert len(fetch_result.payload["fetched"]) + len(fetch_result.payload["fetch_failures"]) == 3


def test_continuation_page_filings_files_reached_when_present():
    http = fx.FakeHttpClient(responses=fx.default_responses(with_continuation_pointer=True))
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    assert fx.continuation_page_url() in http.requested_urls
    locate_result = _by_step_prefix(report, "l1_")
    accessions = {c["accession"] for c in locate_result.payload["candidates"]}
    assert fx.continuation_accession() in accessions
    assert locate_result.payload["submissions_pages_fetched"] == 1


# --- 4 vs 4/A never double-counted (Phase 3E.1 requirement 3) -------------
def test_amendment_and_original_both_discovered_never_double_counted():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    parse_result = _by_step_prefix(report, "p_")
    parsed_by_accession = {d["accession"]: d for d in parse_result.payload["parsed_documents"]}
    original = fx.accession("normal_market_purchase")
    amendment = fx.accession("reconciled_amendment")
    assert original in parsed_by_accession
    assert amendment in parsed_by_accession
    # Two distinct documents, two distinct transaction lists -- never
    # merged into one, never summed into a single "net" figure.
    assert parsed_by_accession[original]["document_id"] != parsed_by_accession[amendment]["document_id"]
    diagnostics = parse_result.payload["diagnostics"]
    assert diagnostics["form4_amendments_found"] >= 2  # reconciled_amendment + form4a_amendment
    # No "net"/aggregate transaction key exists anywhere in the payload.
    assert "net_shares" not in parse_result.payload
    assert "net_buying" not in parse_result.payload
    assert not any("net" in str(k).lower() for k in parse_result.payload["diagnostics"])


def test_reconciled_amendment_references_its_original():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    parse_result = _by_step_prefix(report, "p_")
    parsed_by_accession = {d["accession"]: d for d in parse_result.payload["parsed_documents"]}
    amendment_doc = parsed_by_accession[fx.accession("reconciled_amendment")]
    assert amendment_doc["reconciliation"].status == "RECONCILED"
    assert amendment_doc["reconciliation"].original_accession == fx.accession("normal_market_purchase")


def test_unresolved_amendment_never_guessed_related():
    """form4a_amendment.xml's remarks name no accession at all -- must
    stay UNRESOLVED, never guessed related to any other filing merely by
    period/owner coincidence (Phase 3E.1 requirement 3)."""
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    parse_result = _by_step_prefix(report, "p_")
    parsed_by_accession = {d["accession"]: d for d in parse_result.payload["parsed_documents"]}
    doc = parsed_by_accession[fx.accession("form4a_amendment")]
    assert doc["reconciliation"].status == "UNRESOLVED"
    assert doc["reconciliation"].original_accession == "UNKNOWN"


def test_plain_form4_reconciliation_is_not_applicable():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    parse_result = _by_step_prefix(report, "p_")
    parsed_by_accession = {d["accession"]: d for d in parse_result.payload["parsed_documents"]}
    doc = parsed_by_accession[fx.accession("normal_market_purchase")]
    assert doc["reconciliation"].status == "NOT_APPLICABLE"


def test_diagnostics_amendments_reconciled_and_unresolved_counted_separately():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    diagnostics = _by_step_prefix(report, "p_").payload["diagnostics"]
    assert diagnostics["amendments_reconciled"] == 1  # reconciled_amendment
    assert diagnostics["amendments_unresolved"] == 1  # form4a_amendment
    assert diagnostics["form4_amendments_found"] == diagnostics["amendments_reconciled"] + diagnostics["amendments_unresolved"]


# --- Rule 10b5-1 checkbox diagnostics (Phase 3E.1 requirement 2/5) --------
def test_ten_b5_1_checkbox_diagnostics_all_unknown_when_absent_everywhere():
    """None of this Phase's fixtures include the (unverified-name)
    checkbox element -- every parsed document's checkbox must resolve to
    UNKNOWN, never a guessed False."""
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    diagnostics = _by_step_prefix(report, "p_").payload["diagnostics"]
    assert diagnostics["ten_b5_1_checkbox_true"] == 0
    assert diagnostics["ten_b5_1_checkbox_false"] == 0
    assert diagnostics["ten_b5_1_checkbox_unknown"] > 0


def test_ten_b5_1_checkbox_never_present_on_individual_transactions():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    parse_result = _by_step_prefix(report, "p_")
    for doc in parse_result.payload["parsed_documents"]:
        for t in doc["parsed"]["non_derivative_transactions"] + doc["parsed"]["derivative_transactions"]:
            assert "ten_b5_1_checkbox" not in t


# --- diagnostics: every named field present and accurate -------------------
def test_all_required_diagnostics_fields_present():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    diagnostics = _by_step_prefix(report, "p_").payload["diagnostics"]
    for key in (
        "parsed_non_derivative_transactions", "parsed_derivative_transactions", "parsed_reporting_owners",
        "form4_amendments_found", "amendments_reconciled", "amendments_unresolved",
        "ten_b5_1_checkbox_true", "ten_b5_1_checkbox_false", "ten_b5_1_checkbox_unknown",
    ):
        assert key in diagnostics, key
        assert isinstance(diagnostics[key], int)


def test_parsed_transaction_counts_match_actual_parsed_documents():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    parse_result = _by_step_prefix(report, "p_")
    diagnostics = parse_result.payload["diagnostics"]
    nd_count = sum(len(d["parsed"]["non_derivative_transactions"]) for d in parse_result.payload["parsed_documents"])
    d_count = sum(len(d["parsed"]["derivative_transactions"]) for d in parse_result.payload["parsed_documents"])
    assert diagnostics["parsed_non_derivative_transactions"] == nd_count
    assert diagnostics["parsed_derivative_transactions"] == d_count


# --- evidence semantics: reporting-person filing never independent -------
def test_form4_raw_fact_never_promoted_to_independent_evidence_via_real_pipeline():
    """The REAL RawFact -> Fact pipeline (EvidenceIntegrityAgent), not
    merely the RawFact's own fields, must never promote a Form 4 fact to
    independent_confirmation=True or EvidenceClass.INDEPENDENT_EVIDENCE
    (Phase 3E.1 requirement 4). Phase 3E.4: it must land on the dedicated
    REPORTING_PERSON_STATUTORY_ASSERTION class, never COMPANY_CLAIM
    either, and never be decision-grade on its own."""
    from datetime import date

    from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
    from investment_research.collectors.form4 import parse_ownership_document, raw_facts_from_form4
    from investment_research.schemas.agent_io import AgentInput
    from investment_research.schemas.enums import DECISION_GRADE_CLASSES, EvidenceClass, SourceTier
    from investment_research.schemas.fact import Fact, Source, make_source_id

    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    source = Source(
        source_id=make_source_id("https://example.test/form4", "Form 4"),
        url="https://example.test/form4", title="Form 4", tier=SourceTier.TIER_1,
    )
    raw_facts = raw_facts_from_form4("SAMPB", parsed, source, document_id="doc_form4_test")
    assert raw_facts
    assert all(f.company_claim is False for f in raw_facts)

    facts = [
        Fact(
            fact_id=rf.fact_id(), ticker=rf.ticker, category=rf.category, claim=rf.claim,
            evidence_class=EvidenceClass.UNVERIFIED_CLAIM, source_id=rf.source.source_id,
            source_url=rf.source.url, source_title=rf.source.title, source_tier=rf.source.tier,
            company_claim=rf.company_claim, document_id=rf.document_id,
        )
        for rf in raw_facts
    ]
    agent = EvidenceIntegrityAgent(today=date(2026, 9, 15))
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="test_run", ticker="SAMPB", company_name=fx.ISSUER_NAME,
        facts=tuple(facts),
    )
    output = agent.run(agent_input)

    for assessed in output.facts:
        assert assessed.independent_confirmation is False, (
            f"Form 4 fact wrongly independently confirmed: {assessed.claim}"
        )
        assert assessed.evidence_class is not EvidenceClass.INDEPENDENT_EVIDENCE, (
            f"Form 4 fact wrongly classified INDEPENDENT_EVIDENCE: {assessed.claim}"
        )
        assert assessed.evidence_class is not EvidenceClass.VERIFIED_FACT, (
            f"Form 4 fact wrongly classified VERIFIED_FACT -- a reporting person's own statutory "
            f"filing is not the regulator's own statement: {assessed.claim}"
        )
        assert assessed.evidence_class is not EvidenceClass.COMPANY_CLAIM, (
            f"Form 4 fact wrongly classified COMPANY_CLAIM -- the reporting person is not the "
            f"issuer (Phase 3E.4): {assessed.claim}"
        )
        assert assessed.evidence_class is EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION
        assert assessed.evidence_class not in DECISION_GRADE_CLASSES
        assert assessed.is_decision_grade is False


# --- Document/authority basics ----------------------------------------------
def test_every_stored_form4_document_is_reporting_person_filing_never_company_ir():
    """Phase 3E.4: Form 4/4-A documents carry their OWN dedicated
    DocumentAuthority (REPORTING_PERSON_FILING), distinct from the
    issuer's own STATUTORY_FILING -- never a company statement."""
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    parse_result = _by_step_prefix(report, "p_")
    for doc in parse_result.payload["parsed_documents"]:
        stored = store.get(doc["document_id"])
        assert stored is not None
        assert stored.authority is DocumentAuthority.REPORTING_PERSON_FILING
        assert stored.authority is not DocumentAuthority.STATUTORY_FILING
        assert stored.document.is_company_ir is False
        assert stored.document.content_kind is ContentKind.FULL_DOCUMENT
        assert stored.document_role is DocumentRole.PRIMARY_DOCUMENT


def test_zero_external_llm_tokens_always():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT)
    report, store, target = _run(ref, http)
    assert report.diagnostics.external_llm_tokens == 0


def test_module_never_imports_pipeline_or_web_search():
    import investment_research.research.form4_acquisition_adapter as module

    assert "pipeline" not in module.__dict__
    assert not hasattr(module, "run_pipeline")
    assert "anthropic" not in "".join(dir(module)).lower()


# --- cross-adapter/same-accession dedup (still true after the rewrite) ----
def test_two_form4_targets_same_issuer_dedup_submissions_fetch():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    graph_a, graph_b = _form4_graph("d1"), _form4_graph("d2")
    graph = SourceRoutingGraph(
        requirements=graph_a.requirements + graph_b.requirements,
        targets=graph_a.targets + graph_b.targets,
        steps=graph_a.steps + graph_b.steps,
    )
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={FORM4_ADAPTER_ID: Form4Adapter(http)}, document_store=store)
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT)
    report = executor.run(
        graph,
        form4_references={graph_a.targets[0].target_id: ref, graph_b.targets[0].target_id: ref},
    )
    assert report.outcome_for(graph_a.targets[0].target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.outcome_for(graph_b.targets[0].target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls.count(fx.submissions_url()) == 1
    assert report.diagnostics.duplicate_acquisition_avoided > 0


# --- lineage: SEC metadata -> Document -> RawFact -> Fact.document_id -----
def test_lineage_document_id_flows_from_document_store_to_raw_facts():
    from investment_research.collectors.form4 import raw_facts_from_form4
    from investment_research.schemas.enums import SourceTier
    from investment_research.schemas.fact import Source, make_source_id

    http = fx.FakeHttpClient(responses=fx.default_responses())
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT, max_candidates=50)
    report, store, target = _run(ref, http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED

    parse_result = _by_step_prefix(report, "p_")
    for doc in parse_result.payload["parsed_documents"]:
        stored = store.get(doc["document_id"])
        assert stored is not None
        source = Source(
            source_id=make_source_id(stored.document.url, stored.document.title),
            url=stored.document.url, title=stored.document.title, tier=SourceTier.TIER_1,
        )
        facts = raw_facts_from_form4("SAMPB", doc["parsed"], source, document_id=doc["document_id"])
        assert facts
        for f in facts:
            resolved = store.get(f.document_id)
            assert resolved is stored
