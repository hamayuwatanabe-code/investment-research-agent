"""SecPrimaryDocumentAdapter/SecExhibitAdapter against real-SEC-format
fixtures (``tests/fixtures/sec_edgar_real_format/``, see that directory's
MANIFEST.md) -- Phase 3B requirements 1/2/3/4/5/9/10/14. No real network
call anywhere in this file; every URL and request count is asserted exactly.
"""

from __future__ import annotations

from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.research.sec_acquisition_adapters import (
    ExhibitCategory,
    SecExhibitAdapter,
    SecExhibitSelectionRequest,
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
from investment_research.schemas.enums import ResearchDomain

from . import _sec_fixture_support as fx
from ._sec_fixture_support import ACCESSION, CIK, PRIMARY_DOCUMENT, FakeHttpClient


def _primary_document_graph(tag: str = "a"):
    tid, rid = f"target_{tag}", f"req_{tag}"
    l1 = AcquisitionStep(
        step_id=f"l1_{tag}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id="sec_primary_document_adapter",
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    f = AcquisitionStep(
        step_id=f"f_{tag}", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=(f"l1_{tag}",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    p = AcquisitionStep(
        step_id=f"p_{tag}", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=(f"f_{tag}",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{tag}",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=(f"l1_{tag}", f"f_{tag}", f"p_{tag}"), serves_requirement_ids=(rid,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(l1, f, p))


def _exhibit_graph(tag: str = "b"):
    tid, rid = f"target_{tag}", f"req_{tag}"
    l1 = AcquisitionStep(
        step_id=f"l1_{tag}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id="sec_exhibit_enumeration",
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    f = AcquisitionStep(
        step_id=f"f_{tag}", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_exhibit_enumeration",
        depends_on_step_ids=(f"l1_{tag}",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    p = AcquisitionStep(
        step_id=f"p_{tag}", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_exhibit_enumeration",
        depends_on_step_ids=(f"f_{tag}",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{tag}",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.SEC_EXHIBIT,
        required_step_ids=(f"l1_{tag}", f"f_{tag}", f"p_{tag}"), serves_requirement_ids=(rid,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(l1, f, p))


def _combined(primary_tag="p", exhibit_tag="e"):
    pg, eg = _primary_document_graph(primary_tag), _exhibit_graph(exhibit_tag)
    return SourceRoutingGraph(
        requirements=pg.requirements + eg.requirements,
        targets=pg.targets + eg.targets,
        steps=pg.steps + eg.steps,
    ), pg.targets[0], eg.targets[0]


def _ref(**overrides) -> SecFilingReference:
    base = {"cik": CIK, "accession": ACCESSION, "primary_document": PRIMARY_DOCUMENT, "form": "10-K"}
    base.update(overrides)
    return SecFilingReference(**base)


def _selection(**overrides) -> SecExhibitSelectionRequest:
    base = {"cik": CIK, "accession": ACCESSION, "primary_document": PRIMARY_DOCUMENT}
    base.update(overrides)
    return SecExhibitSelectionRequest(**base)


# --- end-to-end primary document -------------------------------------------
def test_primary_document_full_chain_lands_in_document_store_with_exact_urls():
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=store,
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref()})

    assert http.requested_urls == [fx.submissions_url(), fx.directory_index_url(), fx.document_url(PRIMARY_DOCUMENT)]
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED

    fetch_result = next(r for r in report.target_reports[0].step_results if r.status == StepStatus.BODY_FETCHED)
    stored = store.get(fetch_result.document_id)
    assert stored is not None
    assert stored.document_role is DocumentRole.PRIMARY_DOCUMENT
    assert stored.accession == ACCESSION
    assert stored.document.authority.value == "STATUTORY_FILING"
    assert stored.document.retrieved_at != stored.document.filing_date
    assert stored.document.retrieved_at != stored.document.event_date
    # The Inline XBRL hidden block/script must never appear in extracted text.
    assert "must never appear" not in stored.document.text
    assert "did not consider" in stored.document.text.lower()


# --- section 3: URL resolution edge cases -----------------------------------
def test_empty_primary_document_is_never_treated_as_a_body():
    http = FakeHttpClient(responses={})
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref(primary_document="")})

    assert http.requested_urls == []  # never even attempted a network call
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_malformed_accession_is_refused_before_any_network_call():
    http = FakeHttpClient(responses={})
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref(accession="not-a-real-accession")})

    assert http.requested_urls == []
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_path_traversal_filename_is_refused_before_any_network_call():
    http = FakeHttpClient(responses={})
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref(primary_document="../../../etc/passwd")})

    assert http.requested_urls == []
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_off_host_absolute_url_as_primary_document_is_refused():
    http = FakeHttpClient(responses={})
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref(primary_document="https://evil.example/x.htm")})

    assert http.requested_urls == []
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_cik_is_zero_padded_in_submissions_url_but_not_in_archives_url():
    """A classic real-world gotcha: the JSON submissions endpoint needs a
    10-digit zero-padded CIK in the filename, but the Archives path segment
    uses the bare (non-padded) CIK."""
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    executor.run(graph, filing_references={target_id: _ref()})

    assert "CIK0009999999.json" in http.requested_urls[0]
    assert "/data/9999999/" in http.requested_urls[1]  # never "/data/0009999999/"


def test_accession_hyphens_are_stripped_only_in_the_archives_path():
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    executor.run(graph, filing_references={target_id: _ref()})

    assert "000999999926000123" in http.requested_urls[1]  # dash-free in the Archives path
    assert "0009999999-26-000123" not in http.requested_urls[1]


def test_directory_missing_the_primary_document_is_not_found_never_acquired():
    """Submissions metadata says the primary document exists, but the
    accession's own directory listing disagrees -- Phase 3B requirement 3:
    never resolve a URL for a file the directory does not actually list."""
    responses = dict(fx.default_responses())
    import json as _json

    directory = _json.loads(fx.fixture_text("directory_index_10k.json"))
    directory["directory"]["item"] = [
        item for item in directory["directory"]["item"] if item["name"] != PRIMARY_DOCUMENT
    ]
    responses[fx.directory_index_url()] = fx.ok(_json.dumps(directory))

    http = FakeHttpClient(responses=responses)
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref()})

    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    # Never fell through to actually fetching a document body.
    assert fx.document_url(PRIMARY_DOCUMENT) not in http.requested_urls


def test_submissions_primary_document_mismatch_is_a_failure_not_a_guess():
    """The caller's expected primaryDocument disagrees with what
    submissions metadata records for this accession -- never silently
    prefer either; report the mismatch."""
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref(primary_document="wrong-document.htm")})

    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert fx.directory_index_url() not in http.requested_urls  # never got that far


def test_accession_not_in_submissions_listing_is_not_found_not_a_body():
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref(accession="0009999999-20-000001")})

    assert http.requested_urls == [fx.submissions_url()]
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_index_page_response_is_rejected_as_full_document():
    responses = dict(fx.default_responses())
    responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.ok(fx.fixture_text("index_page_directory_listing.htm"))
    http = FakeHttpClient(responses=responses)
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=store,
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref()})

    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert len(store.duplicate_content_relations()) == 0
    assert report.diagnostics.body_fetch_failures == 1


def test_sec_access_denial_page_is_never_a_successful_body():
    responses = dict(fx.default_responses())
    responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.ok(fx.fixture_text("error_page_undeclared_automated_tool.htm"))
    http = FakeHttpClient(responses=responses)
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref()})
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_rate_limit_error_page_is_never_a_successful_body():
    responses = dict(fx.default_responses())
    responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.ok(fx.fixture_text("error_page_rate_limited.htm"))
    http = FakeHttpClient(responses=responses)
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref()})
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_http_error_page_is_never_a_successful_body():
    responses = dict(fx.default_responses())
    responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.not_found("404")
    http = FakeHttpClient(responses=responses)
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
    )
    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: _ref()})
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_empty_and_too_short_bodies_are_never_full_document():
    for fixture_name in ("empty_document.htm", "too_short_body.htm"):
        responses = dict(fx.default_responses())
        responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.ok(fx.fixture_text(fixture_name))
        http = FakeHttpClient(responses=responses)
        executor = AcquisitionExecutor(
            adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=DocumentStore(),
        )
        graph = _primary_document_graph()
        target_id = graph.targets[0].target_id
        report = executor.run(graph, filing_references={target_id: _ref()})
        assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED, fixture_name


# --- dedup -------------------------------------------------------------
def test_same_primary_document_requested_by_two_targets_is_fetched_exactly_once():
    """Phase 3A requirement 10: the same document, needed by two different
    EvidenceRequirements/targets, results in exactly 1 HTTP GET, 1
    DocumentStore save, and the SAME document_id for both."""
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http)}, document_store=store,
    )
    graph_a, graph_b = _primary_document_graph("a"), _primary_document_graph("b")
    combined = SourceRoutingGraph(
        requirements=graph_a.requirements + graph_b.requirements,
        targets=graph_a.targets + graph_b.targets,
        steps=graph_a.steps + graph_b.steps,
    )
    target_a, target_b = combined.targets
    report = executor.run(combined, filing_references={target_a.target_id: _ref(), target_b.target_id: _ref()})

    assert http.requested_urls.count(fx.submissions_url()) == 1
    assert http.requested_urls.count(fx.directory_index_url()) == 1
    assert http.requested_urls.count(fx.document_url(PRIMARY_DOCUMENT)) == 1
    assert report.diagnostics.duplicate_acquisition_avoided >= 1

    doc_id_a = next(r.document_id for r in report.target_reports[0].step_results if r.document_id)
    doc_id_b = next(r.document_id for r in report.target_reports[1].step_results if r.document_id)
    assert doc_id_a == doc_id_b
    assert len(store._by_id) == 1  # exactly one DocumentStore save


def test_directory_index_json_is_shared_between_primary_and_exhibit_adapters():
    """Phase 3B: both adapters need the SAME accession's index.json --
    it must be fetched exactly once for the whole run, never once per
    adapter."""
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={
            "sec_primary_document_adapter": SecPrimaryDocumentAdapter(http),
            "sec_exhibit_enumeration": SecExhibitAdapter(http),
        },
        document_store=store,
    )
    combined, primary_target, exhibit_target = _combined()
    report = executor.run(
        combined,
        filing_references={primary_target.target_id: _ref()},
        exhibit_selectors={exhibit_target.target_id: _selection()},
    )
    assert http.requested_urls.count(fx.directory_index_url()) == 1
    assert report.outcome_for(primary_target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.outcome_for(exhibit_target.target_id) is TargetAcquisitionOutcome.ACQUIRED


# --- section 4: exhibit enumeration and selection ---------------------------
def test_exhibit_adapter_selects_press_release_by_type_sequence_description():
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http)}, document_store=store)
    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: _selection()})

    assert http.requested_urls == [
        fx.directory_index_url(), fx.filing_detail_url(), fx.document_url("testco-20251231_ex99-1.htm"),
    ]
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["category"] == ExhibitCategory.PRESS_RELEASE.value
    assert locate_result.payload["detail_html_unavailable"] is False


def test_primary_document_is_never_offered_as_an_exhibit_candidate():
    """Phase 3B requirement 4: the primary document itself must never be
    re-selected as if it were a separate exhibit."""
    selection = _selection(priority_keywords=("10-K", "EX-99"))
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http)}, document_store=DocumentStore())
    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: selection})

    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["filename"] != PRIMARY_DOCUMENT
    assert locate_result.payload["category"] == ExhibitCategory.PRESS_RELEASE.value


def test_graphic_and_xbrl_technical_entries_are_never_selected_as_exhibit_content():
    """EX-101.* linkbases and the .jpg graphic must never be offered as
    narrative exhibit content, however broad the priority keywords."""
    selection = _selection(priority_keywords=("EX-101", "GRAPHIC", "jpg", "xsd"))
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http)}, document_store=DocumentStore())
    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: selection})
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_certification_exhibits_are_classified_but_not_selected_by_default():
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http)}, document_store=DocumentStore())
    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: _selection()})

    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["category"] != ExhibitCategory.CERTIFICATION.value


def test_certification_exhibit_is_selectable_when_explicitly_prioritized():
    selection = _selection(priority_keywords=("certification",))
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http)}, document_store=store)
    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: selection})

    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["category"] == ExhibitCategory.CERTIFICATION.value


def test_exhibits_beyond_the_fetch_limit_are_reported_not_yet_fetched_never_nonexistent():
    selection = _selection(priority_keywords=("EX-99", "EX-10"), fetch_limit=1)  # both ex99-1 and ex10-1 match
    http = FakeHttpClient(responses=fx.default_responses())
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http)}, document_store=DocumentStore())
    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: selection})

    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["not_yet_fetched"] == ["testco-20251231_ex10-1.htm"]
    assert fx.document_url("testco-20251231_ex10-1.htm") not in http.requested_urls
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED


def test_filing_detail_html_unavailable_degrades_but_still_classifies_by_type():
    """Phase 3B's documented limitation: if the filing detail HTML cannot be
    fetched, exhibit selection still works from index.json's Type field
    alone, and says so explicitly rather than silently proceeding."""
    responses = dict(fx.default_responses())
    responses[fx.filing_detail_url()] = fx.not_found("503")
    http = FakeHttpClient(responses=responses)
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http)}, document_store=DocumentStore())
    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: _selection()})

    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["detail_html_unavailable"] is True
    assert locate_result.payload["category"] == ExhibitCategory.PRESS_RELEASE.value  # Type alone was enough here


def test_primary_document_and_exhibit_are_distinguished_within_the_same_accession():
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={
            "sec_primary_document_adapter": SecPrimaryDocumentAdapter(http),
            "sec_exhibit_enumeration": SecExhibitAdapter(http),
        },
        document_store=store,
    )
    combined, primary_target, exhibit_target = _combined()
    report = executor.run(
        combined,
        filing_references={primary_target.target_id: _ref()},
        exhibit_selectors={exhibit_target.target_id: _selection()},
    )

    assert report.outcome_for(primary_target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.outcome_for(exhibit_target.target_id) is TargetAcquisitionOutcome.ACQUIRED

    primary_doc_id = next(r.document_id for r in report.target_reports[0].step_results if r.document_id)
    exhibit_doc_id = next(r.document_id for r in report.target_reports[1].step_results if r.document_id)
    assert primary_doc_id != exhibit_doc_id

    resolved = store.resolve_by_accession(ACCESSION)
    roles = {(r.filename, r.document_role) for r in resolved}
    assert (PRIMARY_DOCUMENT, DocumentRole.PRIMARY_DOCUMENT) in roles
    assert ("testco-20251231_ex99-1.htm", DocumentRole.EXHIBIT) in roles


def test_sec_user_agent_requirement_is_satisfied_via_settings_and_httpclient():
    """Phase 3A requirement 5: neither adapter sets its own headers -- the
    SEC User-Agent requirement is satisfied structurally, by constructing the
    injected HttpClient from Settings.sec_user_agent, exactly like every
    other SEC-facing collector in this repository."""
    from investment_research.collectors.http import HttpClient
    from investment_research.config import Settings

    settings = Settings()
    http = HttpClient(user_agent=settings.sec_user_agent, offline=True)
    assert http.user_agent == settings.sec_user_agent

    primary = SecPrimaryDocumentAdapter(http)
    exhibit = SecExhibitAdapter(http)
    assert primary.http is http
    assert exhibit.http is http


# --- Phase 3D.1: physical vs. logical request classification (regression) --
def test_direct_api_and_http_request_split_matches_submissions_vs_generic_fetches():
    """Locks down SEC's existing, correct split so it is never silently
    disturbed by a future change elsewhere (e.g. the ClinicalTrials fix
    that corrected a step reporting a direct-API call as a generic HTTP
    fetch instead). Primary LOCATE's own submissions.json lookup is the
    ONE genuinely direct-API-classified call (EXISTING_DIRECT_API);
    everything else this combined run touches (the accession directory
    listing, the primary body, the filing detail HTML, the exhibit body)
    is a generic HTTP fetch of a known/resolved URL, never itself "the
    direct API" -- direct_api_requests + direct_http_requests must equal
    the number of distinct physical GETs actually made, never merely
    approximate it."""
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={
            "sec_primary_document_adapter": SecPrimaryDocumentAdapter(http),
            "sec_exhibit_enumeration": SecExhibitAdapter(http),
        },
        document_store=store,
    )
    graph, primary_target, exhibit_target = _combined()
    report = executor.run(
        graph, filing_references={primary_target.target_id: _ref()},
        exhibit_selectors={exhibit_target.target_id: _selection()},
    )
    assert report.diagnostics.direct_api_requests == 1  # submissions.json only
    assert report.diagnostics.direct_http_requests == 4  # directory + primary body + filing detail + exhibit body
    assert (
        report.diagnostics.direct_api_requests + report.diagnostics.direct_http_requests
        == len(http.requested_urls)
    )
