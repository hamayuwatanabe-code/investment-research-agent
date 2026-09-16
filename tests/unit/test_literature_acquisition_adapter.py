"""Phase 3F: PubMedLiteratureAdapter/EuropePmcFullTextAdapter against
real-format fixtures (``tests/fixtures/literature_real_format/``, see that
directory's MANIFEST.md). No real network call anywhere in this file;
every URL and request count is asserted exactly. Mirrors
``tests/unit/test_form4_acquisition_adapter.py``'s own structure.
"""

from __future__ import annotations

import pytest

from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentStore
from investment_research.research.literature_acquisition_adapter import (
    ALLOWED_HOSTS,
    LITERATURE_ADAPTER_ID,
    LiteratureReference,
    PubMedLiteratureAdapter,
    _public_request_url,
    _require_allowed_host,
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
from investment_research.schemas.enums import ContentKind, DocumentAuthority, ResearchDomain

from . import _literature_fixture_support as fx


def _literature_graph(*tags: str) -> SourceRoutingGraph:
    requirements, targets, steps = [], [], []
    for tag in tags:
        tid, rid = f"target_{tag}", f"req_{tag}"
        l1 = AcquisitionStep(
            step_id=f"l1_{tag}", target_id=tid, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id=LITERATURE_ADAPTER_ID,
            completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
            implementation_status=ImplementationStatus.EXECUTOR_WIRED,
        )
        f = AcquisitionStep(
            step_id=f"f_{tag}", target_id=tid, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=LITERATURE_ADAPTER_ID,
            depends_on_step_ids=(f"l1_{tag}",), completion_condition=StepStatus.BODY_FETCHED,
            implementation_status=ImplementationStatus.EXECUTOR_WIRED,
        )
        p = AcquisitionStep(
            step_id=f"p_{tag}", target_id=tid, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=LITERATURE_ADAPTER_ID,
            depends_on_step_ids=(f"f_{tag}",), completion_condition=StepStatus.PARSED,
            implementation_status=ImplementationStatus.EXECUTOR_WIRED,
        )
        requirement = EvidenceRequirement(
            requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{tag}",), subject_scope=SubjectScope.COMPANY,
            domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        )
        target = AcquisitionTarget(
            target_id=tid, target_kind=TargetKind.LITERATURE_ARTICLE,
            required_step_ids=(f"l1_{tag}", f"f_{tag}", f"p_{tag}"), serves_requirement_ids=(rid,),
        )
        requirements.append(requirement)
        targets.append(target)
        steps.extend([l1, f, p])
    return SourceRoutingGraph(requirements=tuple(requirements), targets=tuple(targets), steps=tuple(steps))


def _run(refs: dict, http, *, graph=None, store=None):
    graph = graph or _literature_graph(*[t.replace("target_", "") for t in refs])
    store = store or DocumentStore()
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: PubMedLiteratureAdapter(http)}, document_store=store)
    report = executor.run(graph, literature_references=refs)
    return report, store, graph


def _by_step_prefix(report, target_index, prefix):
    return next(r for r in report.target_reports[target_index].step_results if r.step_id.startswith(prefix))


# --- Priority A/B/C acquisition order ---------------------------------------
def test_priority_a_ct_pmid_skips_esearch_entirely():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert not any("esearch.fcgi" in u for u in http.requested_urls)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"]
    assert parsed[0]["pmid"] == "90000001"


def test_priority_b_nct_id_exact_search():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(
        responses={
            fx.esearch_url(term, 20): fx.ok(fx.esearch_response(["90000007"])),
            fx.efetch_url(["90000007"]): fx.ok(fx.fixture_text("nct_id_present.xml")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.payload["confirmed_same_trial"] is True
    assert locate.payload["priority"] == "B"


def test_priority_c_alias_candidates_never_auto_confirmed_same_trial():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "SampleCompound AND FictionalSyndrome"
    http = fx.FakeHttpClient(
        responses={
            fx.esearch_url(term, 20): fx.ok(fx.esearch_response(["90000001"])),
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        }
    )
    ref = LiteratureReference(alias="SampleCompound", condition="FictionalSyndrome")
    report, store, _ = _run({target_id: ref}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.payload["priority"] == "C"
    assert locate.payload["confirmed_same_trial"] is False
    parse = _by_step_prefix(report, 0, "p_")
    assert parse.payload["confirmed_same_trial"] is False


def test_no_reference_supplied_never_acquired_zero_requests():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={})
    report, store, _ = _run({}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


def test_reference_with_no_usable_field_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={})
    report, store, _ = _run({target_id: LiteratureReference()}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


# --- zero results / malformed / error distinguishing -------------------------
def test_zero_esearch_results_never_acquired():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990099[si]"
    http = fx.FakeHttpClient(responses={fx.esearch_url(term, 20): fx.ok(fx.esearch_response([]))})
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990099")}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.ZERO_RESULTS


def test_malformed_esearch_response_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(responses={fx.esearch_url(term, 20): fx.ok("not json { at all")})
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.FAILED


@pytest.mark.parametrize(
    "outcome_name",
    ["RATE_LIMITED", "ERROR", "TIMEOUT"],
)
def test_esearch_transport_failures_distinguished(outcome_name):
    from investment_research.schemas.enums import FetchOutcome

    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(
        responses={fx.esearch_url(term, 20): fx.failed(FetchOutcome[outcome_name], outcome_name.lower())}
    )
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph)
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.FAILED
    assert outcome_name in locate.failure_reason


def test_malformed_xml_efetch_body_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000012"]): fx.ok(fx.fixture_text("malformed.xml"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000012")}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert "MALFORMED" in fetch.failure_reason


def test_html_error_page_efetch_body_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("html_error_page.htm"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert "HTML_ERROR_PAGE" in fetch.failure_reason


def test_empty_efetch_body_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok("")})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert "EMPTY" in fetch.failure_reason


def test_pmid_mismatch_fails_and_is_recorded():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000013"]): fx.ok(fx.fixture_text("mismatched_pmid.xml"))})
    executor_store = DocumentStore()
    from investment_research.research.acquisition_executor import AcquisitionExecutor as _Exec

    executor = _Exec(adapters={LITERATURE_ADAPTER_ID: PubMedLiteratureAdapter(http)}, document_store=executor_store)
    report = executor.run(graph, literature_references={target_id: LiteratureReference(ct_pmid="90000013")})
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


# --- batch fetch / same-PMID dedup -------------------------------------------
def test_batch_fetch_of_two_pmids_is_one_efetch_get():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "SampleCompound AND FictionalSyndrome"
    http = fx.FakeHttpClient(
        responses={
            fx.esearch_url(term, 20): fx.ok(fx.esearch_response(["90000015", "90000016"])),
            fx.efetch_url(["90000015", "90000016"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        }
    )
    ref = LiteratureReference(alias="SampleCompound", condition="FictionalSyndrome")
    report, store, _ = _run({target_id: ref}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"]
    assert {d["pmid"] for d in parsed} == {"90000015", "90000016"}


def test_same_pmid_requested_by_two_targets_fetched_once():
    graph = _literature_graph("a", "b")
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    refs = {t.target_id: LiteratureReference(ct_pmid="90000001") for t in graph.targets}
    report, store, _ = _run(refs, http, graph=graph)
    for target in graph.targets:
        assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1


# --- Europe PMC open-access full text -----------------------------------------
def test_open_access_europepmc_fulltext_acquired_and_distinguished_from_abstract():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    assert parsed["full_text_acquired"] is True
    fulltext_doc_id = parsed["europepmc_fulltext_document_id"]
    assert fulltext_doc_id is not None
    stored = store.get(fulltext_doc_id)
    assert stored is not None
    assert stored.document.content_kind is ContentKind.FULL_DOCUMENT
    assert stored.document.authority is DocumentAuthority.PEER_REVIEWED_LITERATURE
    assert "fictional full-text introduction content" in stored.document.text.lower()
    # The PubMed abstract-only document is a SEPARATE document, never
    # conflated with the Europe PMC full text.
    pubmed_doc_id = parsed["document_id"]
    assert pubmed_doc_id != fulltext_doc_id


def test_non_open_access_never_marked_full_text_acquired_and_no_fulltext_fetch():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    assert parsed["full_text_acquired"] is False
    assert parsed["europepmc_fulltext_document_id"] is None
    assert not any("fullTextXML" in u for u in http.requested_urls)


def test_pubmed_abstract_document_is_never_full_document_content_kind():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    stored = store.get(parsed["document_id"])
    assert stored.document.content_kind is ContentKind.EXCERPT
    assert stored.document.content_kind is not ContentKind.FULL_DOCUMENT


# --- Document versioning ------------------------------------------------------
def test_refetch_with_different_content_creates_a_new_document_version():
    store = DocumentStore()
    graph1 = _literature_graph("a")
    target_id = graph1.targets[0].target_id
    http1 = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report1, _, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http1, graph=graph1, store=store)
    doc_id_v1 = _by_step_prefix(report1, 0, "p_").payload["parsed_documents"][0]["document_id"]

    graph2 = _literature_graph("a")
    http2 = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("updated_article_v2.xml"))})
    report2, _, _ = _run({graph2.targets[0].target_id: LiteratureReference(ct_pmid="90000001")}, http2, graph=graph2, store=store)
    doc_id_v2 = _by_step_prefix(report2, 0, "p_").payload["parsed_documents"][0]["document_id"]

    assert doc_id_v1 != doc_id_v2
    history = store.version_history(doc_id_v2)
    assert len(history) == 2
    assert history[0].document_id == doc_id_v1  # oldest first
    assert history[-1].document_id == doc_id_v2


# --- HTTP safety ---------------------------------------------------------------
def test_allowed_hosts_are_exactly_the_documented_three():
    assert frozenset({"eutils.ncbi.nlm.nih.gov", "www.ebi.ac.uk", "europepmc.org"}) == ALLOWED_HOSTS


def test_require_allowed_host_rejects_other_hosts():
    with pytest.raises(ValueError):
        _require_allowed_host("https://evil.example.com/esearch.fcgi")
    _require_allowed_host("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")  # must not raise


def test_public_request_url_strips_email_and_api_key():
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&email=secret@example.com&api_key=SECRETKEY123&term=x"
    public = _public_request_url(url)
    assert "secret@example.com" not in public
    assert "SECRETKEY123" not in public
    assert "term=x" in public


# --- external LLM / Pipeline isolation ----------------------------------------
def test_no_external_llm_tokens_used():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    assert report.diagnostics.external_llm_tokens == 0
