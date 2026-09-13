"""SecPrimaryDocumentAdapter/SecExhibitAdapter against a FakeHttpClient --
Phase 3A requirements 5/6/9/10/14. No real network call anywhere in this
file; every URL and request count is asserted exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from investment_research.collectors.sec_edgar import FILING_INDEX_URL, SUBMISSIONS_URL
from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.research.sec_acquisition_adapters import (
    MIN_BODY_CHARS,
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
from investment_research.schemas.enums import FetchOutcome, ResearchDomain

_CIK = 9999999
_ACCESSION = "0009999999-25-000123"
_PRIMARY_DOC = "primary10k.htm"

_SUBMISSIONS_PAYLOAD = {
    "filings": {
        "recent": {
            "accessionNumber": [_ACCESSION],
            "form": ["10-K"],
            "primaryDocument": [_PRIMARY_DOC],
            "filingDate": ["2026-01-15"],
            "reportDate": ["2025-12-31"],
        }
    }
}

_BODY_TEXT = (
    "<html><body><p>" + ("Item 1. Business. " * 40) + "</p></body></html>"
)
assert len(_BODY_TEXT) > MIN_BODY_CHARS

_INDEX_PAGE_TEXT = "<html><title>EDGAR Filing Documents</title><body>Document Format Files listing</body></html>"

_EXHIBIT_INDEX_PAYLOAD = {
    "items": [
        {"filename": "primary10k.htm", "type": "10-K", "sequence": 1, "description": "Annual Report"},
        {"filename": "ex99-1.htm", "type": "EX-99.1", "sequence": 2, "description": "Press Release"},
        {"filename": "ex10-1.htm", "type": "EX-10.1", "sequence": 3, "description": "Material Agreement"},
    ]
}

_EXHIBIT_BODY_TEXT = "<html><body><p>" + ("Exhibit press release text. " * 40) + "</p></body></html>"


@dataclass
class _FakeResult:
    ok: bool
    outcome: FetchOutcome
    _body: str
    error: str = ""

    @property
    def text(self) -> str:
        return self._body

    def json(self):
        return json.loads(self._body) if self._body else None


@dataclass
class FakeHttpClient:
    """Records every URL requested, in order, and returns a scripted
    response per URL. If a URL is requested more than once, the test can
    tell -- exactly what the dedup requirement (item 10) must prevent."""

    responses: dict[str, _FakeResult]
    requested_urls: list[str] = field(default_factory=list)

    def get(self, url: str, **kwargs):
        self.requested_urls.append(url)
        if url not in self.responses:
            return _FakeResult(ok=False, outcome=FetchOutcome.NOT_FOUND, _body="", error="no fake response registered")
        return self.responses[url]


def _submissions_url() -> str:
    return SUBMISSIONS_URL.format(cik=_CIK)


def _primary_url() -> str:
    return FILING_INDEX_URL.format(cik=_CIK, accession_nodash=_ACCESSION.replace("-", ""), document=_PRIMARY_DOC)


def _index_url() -> str:
    return FILING_INDEX_URL.format(cik=_CIK, accession_nodash=_ACCESSION.replace("-", ""), document="index.json")


def _exhibit_url(filename: str) -> str:
    return FILING_INDEX_URL.format(cik=_CIK, accession_nodash=_ACCESSION.replace("-", ""), document=filename)


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


# --- SEC primary document adapter -------------------------------------------
def test_primary_document_full_chain_lands_in_document_store_with_exact_urls():
    http = FakeHttpClient(responses={
        _submissions_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_SUBMISSIONS_PAYLOAD)),
        _primary_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=_BODY_TEXT),
    })
    adapter = SecPrimaryDocumentAdapter(http)
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={"sec_primary_document_adapter": adapter}, document_store=store)
    ref = SecFilingReference(cik=_CIK, accession=_ACCESSION, primary_document=_PRIMARY_DOC, form="10-K")

    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: ref})

    assert http.requested_urls == [_submissions_url(), _primary_url()]
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED

    fetch_result = next(r for r in report.target_reports[0].step_results if r.status == StepStatus.BODY_FETCHED)
    stored = store.get(fetch_result.document_id)
    assert stored is not None
    assert stored.document_role is DocumentRole.PRIMARY_DOCUMENT
    assert stored.accession == _ACCESSION
    assert stored.document.authority.value == "STATUTORY_FILING"
    assert stored.document.retrieved_at != stored.document.filing_date
    assert stored.document.retrieved_at != stored.document.event_date


def test_empty_primary_document_is_never_treated_as_a_body():
    http = FakeHttpClient(responses={})
    adapter = SecPrimaryDocumentAdapter(http)
    executor = AcquisitionExecutor(adapters={"sec_primary_document_adapter": adapter}, document_store=DocumentStore())
    ref = SecFilingReference(cik=_CIK, accession=_ACCESSION, primary_document="")

    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: ref})

    assert http.requested_urls == []  # never even attempted a network call
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_index_page_response_is_rejected_as_full_document():
    http = FakeHttpClient(responses={
        _submissions_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_SUBMISSIONS_PAYLOAD)),
        _primary_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=_INDEX_PAGE_TEXT),
    })
    adapter = SecPrimaryDocumentAdapter(http)
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={"sec_primary_document_adapter": adapter}, document_store=store)
    ref = SecFilingReference(cik=_CIK, accession=_ACCESSION, primary_document=_PRIMARY_DOC)

    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: ref})

    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert len(store.duplicate_content_relations()) == 0
    assert report.diagnostics.body_fetch_failures == 1


def test_http_error_page_is_never_a_successful_body():
    http = FakeHttpClient(responses={
        _submissions_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_SUBMISSIONS_PAYLOAD)),
        _primary_url(): _FakeResult(ok=False, outcome=FetchOutcome.NOT_FOUND, _body="", error="404"),
    })
    adapter = SecPrimaryDocumentAdapter(http)
    executor = AcquisitionExecutor(adapters={"sec_primary_document_adapter": adapter}, document_store=DocumentStore())
    ref = SecFilingReference(cik=_CIK, accession=_ACCESSION, primary_document=_PRIMARY_DOC)

    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: ref})
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_accession_not_in_submissions_listing_is_not_found_not_a_body():
    http = FakeHttpClient(responses={
        _submissions_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps({"filings": {"recent": {"accessionNumber": []}}})),
    })
    adapter = SecPrimaryDocumentAdapter(http)
    executor = AcquisitionExecutor(adapters={"sec_primary_document_adapter": adapter}, document_store=DocumentStore())
    ref = SecFilingReference(cik=_CIK, accession=_ACCESSION, primary_document=_PRIMARY_DOC)

    graph = _primary_document_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, filing_references={target_id: ref})

    # Never resolved a document URL, since the accession wasn't confirmed.
    assert http.requested_urls == [_submissions_url()]
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_same_primary_document_requested_by_two_targets_is_fetched_exactly_once():
    """Phase 3A requirement 10: the same document, needed by two different
    EvidenceRequirements/targets, results in exactly 1 HTTP GET, 1
    DocumentStore save, and the SAME document_id for both."""
    http = FakeHttpClient(responses={
        _submissions_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_SUBMISSIONS_PAYLOAD)),
        _primary_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=_BODY_TEXT),
    })
    adapter = SecPrimaryDocumentAdapter(http)
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={"sec_primary_document_adapter": adapter}, document_store=store)
    ref = SecFilingReference(cik=_CIK, accession=_ACCESSION, primary_document=_PRIMARY_DOC)

    graph_a = _primary_document_graph("a")
    graph_b = _primary_document_graph("b")
    combined = SourceRoutingGraph(
        requirements=graph_a.requirements + graph_b.requirements,
        targets=graph_a.targets + graph_b.targets,
        steps=graph_a.steps + graph_b.steps,
    )
    target_a, target_b = combined.targets
    report = executor.run(combined, filing_references={target_a.target_id: ref, target_b.target_id: ref})

    # Exactly one submissions API call and one document GET, never two.
    assert http.requested_urls.count(_submissions_url()) == 1
    assert http.requested_urls.count(_primary_url()) == 1
    assert report.diagnostics.direct_api_requests == 1  # the one real submissions call
    assert report.diagnostics.duplicate_acquisition_avoided >= 1

    doc_id_a = next(r.document_id for r in report.target_reports[0].step_results if r.document_id)
    doc_id_b = next(r.document_id for r in report.target_reports[1].step_results if r.document_id)
    assert doc_id_a == doc_id_b
    assert len(store._by_id) == 1  # exactly one DocumentStore save


# --- SEC exhibit adapter ------------------------------------------------
def test_exhibit_adapter_selects_by_priority_and_lands_a_separate_document():
    http = FakeHttpClient(responses={
        _index_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_EXHIBIT_INDEX_PAYLOAD)),
        _exhibit_url("ex99-1.htm"): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=_EXHIBIT_BODY_TEXT),
    })
    adapter = SecExhibitAdapter(http)
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": adapter}, document_store=store)
    selection = SecExhibitSelectionRequest(cik=_CIK, accession=_ACCESSION)

    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: selection})

    assert http.requested_urls == [_index_url(), _exhibit_url("ex99-1.htm")]
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED

    fetch_result = next(r for r in report.target_reports[0].step_results if r.status == StepStatus.BODY_FETCHED)
    stored = store.get(fetch_result.document_id)
    assert stored.document_role is DocumentRole.EXHIBIT
    assert stored.filename == "ex99-1.htm"


def test_exhibits_beyond_the_fetch_limit_are_reported_not_yet_fetched_never_nonexistent():
    http = FakeHttpClient(responses={
        _index_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_EXHIBIT_INDEX_PAYLOAD)),
        _exhibit_url("ex99-1.htm"): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=_EXHIBIT_BODY_TEXT),
    })
    adapter = SecExhibitAdapter(http)
    executor = AcquisitionExecutor(adapters={"sec_exhibit_enumeration": adapter}, document_store=DocumentStore())
    selection = SecExhibitSelectionRequest(
        cik=_CIK, accession=_ACCESSION,
        priority_keywords=("EX-99", "EX-10"),  # both ex99-1 and ex10-1 match
        fetch_limit=1,
    )

    graph = _exhibit_graph()
    target_id = graph.targets[0].target_id
    report = executor.run(graph, exhibit_selectors={target_id: selection})

    locate_result = report.target_reports[0].step_results[0]
    assert locate_result.payload["not_yet_fetched"] == ["ex10-1.htm"]
    # Never claimed nonexistent -- only ONE exhibit body was ever requested.
    assert http.requested_urls.count(_exhibit_url("ex10-1.htm")) == 0
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED


def test_primary_document_and_exhibit_are_distinguished_within_the_same_accession():
    http = FakeHttpClient(responses={
        _submissions_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_SUBMISSIONS_PAYLOAD)),
        _primary_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=_BODY_TEXT),
        _index_url(): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=json.dumps(_EXHIBIT_INDEX_PAYLOAD)),
        _exhibit_url("ex99-1.htm"): _FakeResult(ok=True, outcome=FetchOutcome.OK, _body=_EXHIBIT_BODY_TEXT),
    })
    store = DocumentStore()
    primary_adapter = SecPrimaryDocumentAdapter(http)
    exhibit_adapter = SecExhibitAdapter(http)
    executor = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": primary_adapter, "sec_exhibit_enumeration": exhibit_adapter},
        document_store=store,
    )
    primary_graph = _primary_document_graph("primary")
    exhibit_graph = _exhibit_graph("exhibit")
    combined = SourceRoutingGraph(
        requirements=primary_graph.requirements + exhibit_graph.requirements,
        targets=primary_graph.targets + exhibit_graph.targets,
        steps=primary_graph.steps + exhibit_graph.steps,
    )
    primary_target, exhibit_target = combined.targets
    report = executor.run(
        combined,
        filing_references={primary_target.target_id: SecFilingReference(cik=_CIK, accession=_ACCESSION, primary_document=_PRIMARY_DOC)},
        exhibit_selectors={exhibit_target.target_id: SecExhibitSelectionRequest(cik=_CIK, accession=_ACCESSION)},
    )

    assert report.outcome_for(primary_target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.outcome_for(exhibit_target.target_id) is TargetAcquisitionOutcome.ACQUIRED

    primary_doc_id = next(r.document_id for r in report.target_reports[0].step_results if r.document_id)
    exhibit_doc_id = next(r.document_id for r in report.target_reports[1].step_results if r.document_id)
    assert primary_doc_id != exhibit_doc_id

    resolved = store.resolve_by_accession(_ACCESSION)
    roles = {(r.filename, r.document_role) for r in resolved}
    assert (_PRIMARY_DOC, DocumentRole.PRIMARY_DOCUMENT) in roles
    assert ("ex99-1.htm", DocumentRole.EXHIBIT) in roles


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

    # Neither adapter overrides or ignores the injected client's user agent.
    primary = SecPrimaryDocumentAdapter(http)
    exhibit = SecExhibitAdapter(http)
    assert primary.http is http
    assert exhibit.http is http
