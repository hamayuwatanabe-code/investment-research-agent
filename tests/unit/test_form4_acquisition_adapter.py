"""Phase 3E: Form4Adapter against real-Form4-format fixtures
(``tests/fixtures/form4_real_format/``, see that directory's MANIFEST.md).
No real network call anywhere in this file; every URL and request count
is asserted exactly, mirroring test_sec_acquisition_adapters.py's own
established pattern.
"""

from __future__ import annotations

from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.research.form4_acquisition_adapter import (
    FORM4_ADAPTER_ID,
    Form4Adapter,
    Form4FilingReference,
)
from investment_research.research.sec_acquisition_adapters import (
    SecFilingReference,
    SecPrimaryDocumentAdapter,
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
    FetchOutcome,
    ResearchDomain,
    SourceTier,
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


def _ref(scenario: str, **overrides) -> Form4FilingReference:
    base = {
        "filer_cik": fx.FILER_CIK, "accession": fx.accession(scenario), "primary_document": fx.PRIMARY_DOCUMENT,
        "issuer_cik": fx.ISSUER_CIK, "issuer_name": fx.ISSUER_NAME, "issuer_ticker": fx.ISSUER_TICKER,
        "reporting_owner_cik": fx.FILER_CIK_PADDED, "reporting_owner_name": fx.OWNER_NAME,
    }
    base.update(overrides)
    return Form4FilingReference(**base)


def _run(scenario: str, http, *, graph=None, target=None, refs=None):
    graph = graph or _form4_graph()
    target = target or graph.targets[0]
    refs = refs if refs is not None else {target.target_id: _ref(scenario)}
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={FORM4_ADAPTER_ID: Form4Adapter(http)}, document_store=store)
    report = executor.run(graph, form4_references=refs)
    return report, store, target


# --- happy path scenarios: every required transaction-type fixture --------
_HAPPY_SCENARIOS = (
    "normal_market_purchase", "normal_market_sale", "option_exercise", "tax_withholding",
    "grant_award", "gift", "derivative_transaction", "indirect_ownership", "multiple_transactions",
    "footnote_10b5_1", "form4a_amendment", "missing_fields",
)


def test_every_happy_path_scenario_acquires_successfully():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    for scenario in _HAPPY_SCENARIOS:
        report, store, target = _run(scenario, http)
        assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED, scenario
        stored = store.resolve_by_accession(fx.accession(scenario))
        assert len(stored) == 1, scenario
        assert stored[0].authority is DocumentAuthority.STATUTORY_FILING
        assert stored[0].document.is_company_ir is False
        assert stored[0].document.content_kind is ContentKind.FULL_DOCUMENT


def test_market_purchase_end_to_end_urls_and_counts_exact():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("normal_market_purchase", http)
    assert http.requested_urls == [
        fx.submissions_url(), fx.directory_index_url("normal_market_purchase"),
        fx.document_url("normal_market_purchase"),
    ]
    diag = report.diagnostics
    # The submissions.json call is inherited from SecPrimaryDocumentAdapter's
    # own composed _fetch_submissions() and is classified as a direct API
    # call there (Phase 3D.1's classification convention); the directory
    # index.json and the XML body itself are both generic HTTP fetches.
    assert diag.direct_api_requests == 1
    assert diag.direct_http_requests == 2
    assert diag.body_fetch_successes == 1
    assert diag.parsed_documents == 1
    assert diag.external_llm_tokens == 0


def test_form4a_amendment_stored_as_separate_document_never_an_overwrite():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report_orig, store, _ = _run("normal_market_purchase", http)
    graph_amend = _form4_graph(tag="amend")
    target_amend = graph_amend.targets[0]
    executor = AcquisitionExecutor(adapters={FORM4_ADAPTER_ID: Form4Adapter(http)}, document_store=store)
    report_amend = executor.run(graph_amend, form4_references={target_amend.target_id: _ref("form4a_amendment")})

    assert report_amend.outcome_for(target_amend.target_id) is TargetAcquisitionOutcome.ACQUIRED
    original_docs = store.resolve_by_accession(fx.accession("normal_market_purchase"))
    amendment_docs = store.resolve_by_accession(fx.accession("form4a_amendment"))
    assert len(original_docs) == 1
    assert len(amendment_docs) == 1
    assert original_docs[0].document_id != amendment_docs[0].document_id
    assert original_docs[0].version == 1
    assert amendment_docs[0].version == 1  # a NEW document, not a new version of the original


# --- transaction classification lands correctly per scenario --------------
def test_option_exercise_parsed_as_derivative_transaction():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, target = _run("option_exercise", http)
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("p_"))
    parsed = parse_result.payload["parsed"]
    assert len(parsed["derivative_transactions"]) == 1
    assert parsed["non_derivative_transactions"] == []
    assert parsed["derivative_transactions"][0]["transaction_code"] == "M"


def test_footnote_10b5_1_reaches_parse_stage_true():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, _target = _run("footnote_10b5_1", http)
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("p_"))
    parsed = parse_result.payload["parsed"]
    assert parsed["non_derivative_transactions"][0]["is_10b5_1_plan"] is True


def test_multiple_transactions_all_present_after_full_chain():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, _target = _run("multiple_transactions", http)
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("p_"))
    parsed = parse_result.payload["parsed"]
    assert len(parsed["non_derivative_transactions"]) == 2


# --- Form 3/4/5 disambiguation ------------------------------------------
def test_metadata_says_form3_refused_at_locate_never_fetched():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("metadata_says_form3", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    # LOCATE refused before any body URL was ever requested.
    assert fx.document_url("metadata_says_form3") not in http.requested_urls
    assert store.resolve_by_accession(fx.accession("metadata_says_form3")) == []
    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.status is StepStatus.FAILED
    assert "form" in locate_result.failure_reason.lower()


def test_body_says_form3_refused_at_fetch_defense_in_depth():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("body_says_form3", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert store.resolve_by_accession(fx.accession("body_says_form3")) == []
    fetch_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("f_"))
    assert fetch_result.status is StepStatus.FAILED
    assert "WRONG_DOCUMENT_TYPE" in fetch_result.failure_reason


# --- other failure states, each distinguishable, never ACQUIRED -----------
def test_unknown_accession_is_not_found_never_acquired():
    responses = dict(fx.default_responses())
    http = fx.FakeHttpClient(responses=responses)
    report, store, target = _run(
        "normal_market_purchase", http,
        refs={_form4_graph().targets[0].target_id: _ref("normal_market_purchase", accession=fx.UNKNOWN_ACCESSION)},
    )
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.status is StepStatus.NOT_FOUND


def test_missing_ownership_document_root_fails_at_fetch():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("missing_ownership_document", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("f_"))
    assert "MISSING_OWNERSHIP_DOCUMENT" in fetch_result.failure_reason


def test_malformed_xml_fails_at_fetch():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("malformed_xml", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("f_"))
    assert "MALFORMED" in fetch_result.failure_reason


def test_empty_body_fails_at_fetch_never_acquired():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("empty_body", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("f_"))
    assert fetch_result.status is StepStatus.FAILED
    assert "empty" in fetch_result.failure_reason.lower()


def test_html_error_page_never_accepted_as_ownership_document():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("html_error_page", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert store.resolve_by_accession(fx.accession("html_error_page")) == []


def test_missing_required_fields_fails_at_parse_not_fetch():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("missing_required_fields", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("f_"))
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("p_"))
    assert fetch_result.status is StepStatus.BODY_FETCHED  # FETCH's coarse gate passed
    assert parse_result.status is StepStatus.FAILED  # PARSE's field-presence check caught it
    assert "issuer CIK" in parse_result.failure_reason
    # The document WAS stored (a fetch genuinely happened) -- never
    # promoted to ACQUIRED, since PARSE never completed.
    assert store.resolve_by_accession(fx.accession("missing_required_fields"))


def test_404_never_acquired():
    responses = dict(fx.default_responses())
    responses[fx.document_url("normal_market_purchase")] = fx.not_found()
    http = fx.FakeHttpClient(responses=responses)
    report, store, target = _run("normal_market_purchase", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("f_"))
    assert fetch_result.status is StepStatus.NOT_FOUND


def test_rate_limited_429_never_acquired():
    responses = dict(fx.default_responses())
    responses[fx.document_url("normal_market_purchase")] = fx.failed(FetchOutcome.RATE_LIMITED, "429 too many requests")
    http = fx.FakeHttpClient(responses=responses)
    report, store, target = _run("normal_market_purchase", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("f_"))
    assert fetch_result.status is StepStatus.FAILED


def test_server_error_5xx_never_acquired():
    responses = dict(fx.default_responses())
    responses[fx.document_url("normal_market_purchase")] = fx.failed(FetchOutcome.ERROR, "500 internal server error")
    http = fx.FakeHttpClient(responses=responses)
    report, store, target = _run("normal_market_purchase", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_timeout_never_acquired():
    responses = dict(fx.default_responses())
    responses[fx.document_url("normal_market_purchase")] = fx.failed(FetchOutcome.TIMEOUT, "timed out")
    http = fx.FakeHttpClient(responses=responses)
    report, store, target = _run("normal_market_purchase", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_no_form4_reference_supplied_fails_never_acquired():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    graph = _form4_graph()
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={FORM4_ADAPTER_ID: Form4Adapter(http)}, document_store=store)
    report = executor.run(graph, form4_references={})
    assert report.outcome_for(graph.targets[0].target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []  # zero HTTP calls with nothing to locate


# --- dedup: same accession/URL/CIK never re-fetched -----------------------
def test_two_targets_same_accession_dedup_to_one_real_fetch():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    graph_a, graph_b = _form4_graph("d1"), _form4_graph("d2")
    graph = SourceRoutingGraph(
        requirements=graph_a.requirements + graph_b.requirements,
        targets=graph_a.targets + graph_b.targets,
        steps=graph_a.steps + graph_b.steps,
    )
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={FORM4_ADAPTER_ID: Form4Adapter(http)}, document_store=store)
    refs = {
        graph_a.targets[0].target_id: _ref("normal_market_purchase"),
        graph_b.targets[0].target_id: _ref("normal_market_purchase"),
    }
    report = executor.run(graph, form4_references=refs)
    assert report.outcome_for(graph_a.targets[0].target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.outcome_for(graph_b.targets[0].target_id) is TargetAcquisitionOutcome.ACQUIRED
    # Exactly one physical GET per distinct URL -- the second target's
    # identical accession/URL is served entirely from request_cache.
    assert http.requested_urls.count(fx.document_url("normal_market_purchase")) == 1
    assert http.requested_urls.count(fx.submissions_url()) == 1
    assert report.diagnostics.duplicate_acquisition_avoided > 0
    # Both targets resolve to the SAME stored document -- never duplicated.
    docs = store.resolve_by_accession(fx.accession("normal_market_purchase"))
    assert len(docs) == 1


def test_form4_and_sec_primary_adapter_share_one_submissions_fetch():
    """Cross-adapter dedup (Phase 3E requirement 2): a Form4 target and an
    SEC primary-document target for the SAME filer CIK share exactly one
    submissions.json GET within one AcquisitionExecutor.run() call."""
    http = fx.FakeHttpClient(responses=fx.default_responses())

    form4_graph = _form4_graph("x1")
    sec_tag = "x2"
    l1 = AcquisitionStep(
        step_id=f"l1_{sec_tag}", target_id=f"target_{sec_tag}", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id="sec_primary_document_adapter",
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    f = AcquisitionStep(
        step_id=f"f_{sec_tag}", target_id=f"target_{sec_tag}", step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=(f"l1_{sec_tag}",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    p = AcquisitionStep(
        step_id=f"p_{sec_tag}", target_id=f"target_{sec_tag}", step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=(f"f_{sec_tag}",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    requirement = EvidenceRequirement(
        requirement_id=f"req_{sec_tag}", serves_legacy_need_ids=(f"synthetic_{sec_tag}",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    sec_target = AcquisitionTarget(
        target_id=f"target_{sec_tag}", target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=(f"l1_{sec_tag}", f"f_{sec_tag}", f"p_{sec_tag}"), serves_requirement_ids=(f"req_{sec_tag}",),
    )
    sec_graph = SourceRoutingGraph(requirements=(requirement,), targets=(sec_target,), steps=(l1, f, p))

    graph = SourceRoutingGraph(
        requirements=form4_graph.requirements + sec_graph.requirements,
        targets=form4_graph.targets + sec_graph.targets,
        steps=form4_graph.steps + sec_graph.steps,
    )
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={FORM4_ADAPTER_ID: Form4Adapter(http), "sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)},
        document_store=store,
    )
    # Same filer CIK/accession/document as the Form4 target below -- both
    # adapters must resolve against the SAME submissions.json/directory
    # listing entries, proving the shared request_cache is what dedups
    # the submissions.json call, not merely two independent fixtures that
    # happen to look alike.
    sec_ref = SecFilingReference(
        cik=fx.FILER_CIK, accession=fx.accession("normal_market_purchase"),
        primary_document=fx.PRIMARY_DOCUMENT, form="4",
    )
    report = executor.run(
        graph,
        form4_references={form4_graph.targets[0].target_id: _ref("normal_market_purchase")},
        filing_references={sec_target.target_id: sec_ref},
    )
    assert report.outcome_for(form4_graph.targets[0].target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.outcome_for(sec_target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls.count(fx.submissions_url()) == 1


# --- lineage: SEC metadata -> Document -> RawFact -> Fact.document_id -----
def test_lineage_document_id_flows_from_document_store_to_raw_facts():
    from investment_research.collectors.form4 import raw_facts_from_form4
    from investment_research.schemas.fact import Source, make_source_id

    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run("normal_market_purchase", http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED

    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id.startswith("p_"))
    document_id = parse_result.document_id
    parsed = parse_result.payload["parsed"]
    assert document_id is not None

    stored = store.get(document_id)
    assert stored is not None
    assert stored.document_role is DocumentRole.PRIMARY_DOCUMENT
    assert stored.document.text  # the actual XML text is in hand

    source = Source(
        source_id=make_source_id(stored.document.url, stored.document.title),
        url=stored.document.url, title=stored.document.title, tier=SourceTier.TIER_1,
    )
    facts = raw_facts_from_form4("SAMPB", parsed, source, document_id=document_id)
    assert facts
    assert all(f.document_id == document_id for f in facts)
    # And the document_id genuinely resolves back to the SAME stored
    # document -- lineage is a real, checkable chain, not a label.
    for f in facts:
        resolved = store.get(f.document_id)
        assert resolved is stored


# --- issuer CIK vs reporting-owner CIK never confused ----------------------
def test_issuer_cik_and_reporting_owner_cik_never_confused_in_payload():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, _target = _run("normal_market_purchase", http)
    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["issuer_cik"] == fx.ISSUER_CIK
    assert locate_result.payload["reporting_owner_cik"] == fx.FILER_CIK_PADDED
    assert locate_result.payload["issuer_cik"] != locate_result.payload["reporting_owner_cik"]
    assert locate_result.payload["filer_cik"] == fx.FILER_CIK


# --- unsafe references refused, never guessed -----------------------------
def test_malformed_accession_refused():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    graph = _form4_graph()
    report, store, target = _run(
        "normal_market_purchase", http,
        refs={graph.targets[0].target_id: _ref("normal_market_purchase", accession="not-an-accession")},
        graph=graph, target=graph.targets[0],
    )
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


def test_unsafe_primary_document_filename_refused():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    graph = _form4_graph()
    report, store, target = _run(
        "normal_market_purchase", http,
        refs={graph.targets[0].target_id: _ref("normal_market_purchase", primary_document="../../etc/passwd")},
        graph=graph, target=graph.targets[0],
    )
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


def test_empty_primary_document_refused():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    graph = _form4_graph()
    report, store, target = _run(
        "normal_market_purchase", http,
        refs={graph.targets[0].target_id: _ref("normal_market_purchase", primary_document="")},
        graph=graph, target=graph.targets[0],
    )
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED


# --- never company_claim / never company IR --------------------------------
def test_form4_document_is_never_marked_company_ir():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    _report, store, _target = _run("normal_market_purchase", http)
    stored = store.resolve_by_accession(fx.accession("normal_market_purchase"))[0]
    assert stored.document.is_company_ir is False
    assert stored.authority is DocumentAuthority.STATUTORY_FILING


# --- never Pipeline.run() connected, never Web Search / LLM ---------------
def test_module_never_imports_pipeline_or_web_search():
    import investment_research.research.form4_acquisition_adapter as module

    assert "pipeline" not in module.__dict__
    assert not hasattr(module, "run_pipeline")
    assert "anthropic" not in "".join(dir(module)).lower()


def test_zero_external_llm_tokens_always():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, _target = _run("normal_market_purchase", http)
    assert report.diagnostics.external_llm_tokens == 0


# --- diagnostics: metadata-only and incomplete-target counts ---------------
def test_diagnostics_metadata_only_result_recorded_on_locate_success():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, _target = _run("normal_market_purchase", http)
    assert report.diagnostics.metadata_only_results >= 1


def test_diagnostics_incomplete_targets_counts_a_failed_target():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, target = _run("malformed_xml", http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert report.diagnostics.incomplete_targets == 1
    assert report.diagnostics.body_fetch_failures == 1


def test_diagnostics_planned_and_executed_steps_reported():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report, _store, _target = _run("normal_market_purchase", http)
    graph = _form4_graph()
    assert report.diagnostics.planned_steps == len(graph.steps)
    assert report.diagnostics.executed_steps == len(graph.steps)
