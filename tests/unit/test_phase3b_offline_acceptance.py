"""Phase 3B requirement 10: one full offline path through real-SEC-format
fixtures --

    submissions metadata -> filing resolution -> filing detail/index parsing
    -> primary/exhibit selection -> HTTP fixture fetch -> DocumentStore
    -> Chunk -> RawFact -> Fact -> Evidence authority confirmation
    -> Coverage/Acquisition diagnostics

No real network call anywhere in this file (``FakeHttpClient`` only, backed
by ``tests/fixtures/sec_edgar_real_format/`` -- see that directory's
MANIFEST.md for provenance). No Anthropic API, no Web Search, no LLM tokens.
"""

from __future__ import annotations

from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.extraction import DocumentCollector
from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import AcquisitionStatus, SubjectScope
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.research.legacy_catalog import LEGACY_CATALOG
from investment_research.research.sec_acquisition_adapters import (
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
    target_acquisition_outcome,
)
from investment_research.research.step_graph_coverage import project_step_results_to_coverage
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import (
    ContentKind,
    DocumentAuthority,
    Provenance,
    ResearchDomain,
)
from investment_research.schemas.fact import EvidenceClass

from . import _sec_fixture_support as fx
from ._sec_fixture_support import ACCESSION, CIK, PRIMARY_DOCUMENT, FakeHttpClient


def _build_graph():
    primary_id, primary_req = "target_primary", "req_primary"
    exhibit_id, exhibit_req = "target_exhibit", "req_exhibit"

    pl1 = AcquisitionStep(
        step_id="pl1", target_id=primary_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id="sec_primary_document_adapter",
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    pf = AcquisitionStep(
        step_id="pf", target_id=primary_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=("pl1",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    pp = AcquisitionStep(
        step_id="pp", target_id=primary_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=("pf",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    primary_requirement = EvidenceRequirement(
        requirement_id=primary_req, serves_legacy_need_ids=("bear_5",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY,
    )
    primary_target = AcquisitionTarget(
        target_id=primary_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=("pl1", "pf", "pp"), serves_requirement_ids=(primary_req,),
    )

    el1 = AcquisitionStep(
        step_id="el1", target_id=exhibit_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id="sec_exhibit_enumeration",
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    ef = AcquisitionStep(
        step_id="ef", target_id=exhibit_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_exhibit_enumeration",
        depends_on_step_ids=("el1",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    ep = AcquisitionStep(
        step_id="ep", target_id=exhibit_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_exhibit_enumeration",
        depends_on_step_ids=("ef",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    exhibit_requirement = EvidenceRequirement(
        requirement_id=exhibit_req, serves_legacy_need_ids=("bull_1",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    exhibit_target = AcquisitionTarget(
        target_id=exhibit_id, target_kind=TargetKind.SEC_EXHIBIT,
        required_step_ids=("el1", "ef", "ep"), serves_requirement_ids=(exhibit_req,),
    )

    graph = SourceRoutingGraph(
        requirements=(primary_requirement, exhibit_requirement),
        targets=(primary_target, exhibit_target),
        steps=(pl1, pf, pp, el1, ef, ep),
    )
    return graph, primary_id, exhibit_id


def test_full_offline_path_from_submissions_to_fact_and_coverage_diagnostics():
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={
            "sec_primary_document_adapter": SecPrimaryDocumentAdapter(http),
            "sec_exhibit_enumeration": SecExhibitAdapter(http),
        },
        document_store=store,
    )
    graph, primary_id, exhibit_id = _build_graph()
    report = executor.run(
        graph,
        filing_references={primary_id: SecFilingReference(cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT, form="10-K")},
        exhibit_selectors={
            exhibit_id: SecExhibitSelectionRequest(
                cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT,
                # Both EX-99.1 (press release) and EX-10.1 (license
                # agreement, whose Description does not literally contain
                # the phrase "material agreement") match here, so the
                # fetch_limit=1 default genuinely defers one of them --
                # exercising the "not yet fetched" path end to end.
                priority_keywords=("EX-99", "EX-10"),
            ),
        },
    )

    # --- acquisition outcomes: never ACQUIRED on a failure/incomplete path,
    # never FULL_DOCUMENT from metadata/URL alone (both targets went all the
    # way through FETCH+PARSE here, so both legitimately reach ACQUIRED).
    assert report.outcome_for(primary_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.outcome_for(exhibit_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.diagnostics.metadata_only_results == 2  # both LOCATE steps, correctly separated from body-fetch
    assert report.diagnostics.body_fetch_successes == 2
    assert report.diagnostics.parsed_documents == 2
    assert report.diagnostics.incomplete_targets == 0

    # --- zero external communication of any kind -----------------------
    assert report.diagnostics.external_llm_tokens == 0
    # every URL in the request log is one of this fixture's local, fake ones
    for url in http.requested_urls:
        assert url.startswith("https://data.sec.gov/") or url.startswith("https://www.sec.gov/")

    # --- exactly 1 HTTP GET per distinct URL (dedup) --------------------
    assert http.requested_urls.count(fx.directory_index_url()) == 1
    assert http.requested_urls.count(fx.document_url(PRIMARY_DOCUMENT)) == 1
    assert http.requested_urls.count(fx.submissions_url()) == 1

    # --- exactly 1 DocumentStore save per distinct document -------------
    primary_doc_id = next(r.document_id for r in report.target_reports[0].step_results if r.document_id)
    exhibit_doc_id = next(r.document_id for r in report.target_reports[1].step_results if r.document_id)
    assert primary_doc_id != exhibit_doc_id
    assert store.get(primary_doc_id) is not None
    assert store.get(exhibit_doc_id) is not None
    resolved = store.resolve_by_accession(ACCESSION)
    assert len(resolved) == 2  # exactly primary + exhibit, no phantom duplicates
    roles = {(r.filename, r.document_role) for r in resolved}
    assert (PRIMARY_DOCUMENT, DocumentRole.PRIMARY_DOCUMENT) in roles
    assert ("testco-20251231_ex99-1.htm", DocumentRole.EXHIBIT) in roles

    # --- unfetched exhibits (beyond this run's fetch_limit=1 default)
    # reported not-yet-fetched, never "does not exist" -----------------
    exhibit_locate = report.target_reports[1].step_results[0]
    assert "testco-20251231_ex10-1.htm" in exhibit_locate.payload["not_yet_fetched"]

    # --- Document -> Chunk -> RawFact -> Fact, authority/company_claim --
    primary_stored = store.get(primary_doc_id)
    exhibit_stored = store.get(exhibit_doc_id)
    assert primary_stored is not None and exhibit_stored is not None
    assert primary_stored.document.authority is DocumentAuthority.STATUTORY_FILING
    assert exhibit_stored.document.authority is DocumentAuthority.STATUTORY_FILING
    assert primary_stored.document.content_kind is ContentKind.FULL_DOCUMENT
    assert exhibit_stored.document.content_kind is ContentKind.FULL_DOCUMENT

    collector = DocumentCollector(
        [primary_stored.document, exhibit_stored.document], provenance=Provenance.FIXTURE,
    )
    collection_result = collector.collect("TESTCO", "Generic Biotech Holdings")
    assert collection_result.raw_facts
    assert collection_result.raw_fact_count_before_dedup >= len(collection_result.raw_facts)

    agent = FactCollectorAgent(results=[collection_result])
    agent_input = AgentInput(agent_id="fact_collector", run_id="phase3b_acceptance", ticker="TESTCO", company_name="Generic Biotech Holdings")
    output = agent.run(agent_input)
    assert output.facts

    rejection_facts = [f for f in output.facts if "did not consider" in f.claim.lower()]
    assert rejection_facts, "the regulatory-rejection statement must survive Document -> Chunk -> RawFact -> Fact"
    for fact in rejection_facts:
        # document_id lineage
        assert fact.document_id in (primary_doc_id, exhibit_doc_id)
        # issuer disclosure, never independent regulator confirmation --
        # no REGULATOR-authority document was ever fetched in this run.
        assert fact.company_claim is True
        assert fact.evidence_class is EvidenceClass.COMPANY_CLAIM
        assert fact.independent_confirmation is False
        assert fact.content_kind is ContentKind.FULL_DOCUMENT  # never a search snippet

    # --- Coverage/Acquisition diagnostics -------------------------------
    outcomes = {
        "pl1": StepStatus.URL_RESOLVED, "pf": StepStatus.BODY_FETCHED, "pp": StepStatus.PARSED,
        "el1": StepStatus.URL_RESOLVED, "ef": StepStatus.BODY_FETCHED, "ep": StepStatus.PARSED,
    }
    ledger = project_step_results_to_coverage(graph, outcomes)
    assert len(ledger.results) == len(LEGACY_CATALOG)  # every one of the 49 rows present
    assert ledger.results["bear_5"].acquisition_status is AcquisitionStatus.ACQUIRED
    assert ledger.results["bull_1"].acquisition_status is AcquisitionStatus.ACQUIRED
    # A legacy need this graph never raised a requirement for stays
    # UNSEARCHED, never silently dropped or guessed ACQUIRED.
    untouched = next(n for n in LEGACY_CATALOG if n.legacy_need_id not in ("bear_5", "bull_1"))
    assert ledger.results[untouched.legacy_need_id].acquisition_status is AcquisitionStatus.UNSEARCHED


def test_metadata_only_and_url_only_never_read_as_full_document_in_this_same_path():
    """The same full path, but the primary document's body fetch fails --
    confirming the honest incomplete/failure states hold end to end,
    never silently promoted to ACQUIRED."""
    responses = dict(fx.default_responses())
    responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.not_found("503")
    http = FakeHttpClient(responses=responses)
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={
            "sec_primary_document_adapter": SecPrimaryDocumentAdapter(http),
            "sec_exhibit_enumeration": SecExhibitAdapter(http),
        },
        document_store=store,
    )
    graph, primary_id, exhibit_id = _build_graph()
    report = executor.run(
        graph,
        filing_references={primary_id: SecFilingReference(cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT, form="10-K")},
        exhibit_selectors={exhibit_id: SecExhibitSelectionRequest(cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)},
    )
    assert report.outcome_for(primary_id) is not TargetAcquisitionOutcome.ACQUIRED
    # LOCATE (URL_RESOLVED) still succeeded -- confirming metadata/URL
    # resolution alone is recorded distinctly from, and never conflated
    # with, full-document acquisition.
    primary_steps = graph.steps_for_target(primary_id)
    outcomes = {"pl1": StepStatus.URL_RESOLVED, "pf": StepStatus.FAILED}
    assert target_acquisition_outcome(graph.targets[0], primary_steps, outcomes) is not TargetAcquisitionOutcome.ACQUIRED
    assert report.diagnostics.body_fetch_failures == 1
