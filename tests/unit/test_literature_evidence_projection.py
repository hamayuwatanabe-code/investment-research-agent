"""Phase 4.1A (corrected): Literature Evidence Projection Bridge.

Offline end-to-end: AcquisitionExecutor -> PubMedLiteratureAdapter ->
DocumentStore -> literature_evidence_projection -> CollectionResult ->
FactCollectorAgent -> EvidenceIntegrityAgent, against the SAME real-format
fixtures/FakeHttpClient double ``test_literature_acquisition_adapter.py``
uses. No real network call anywhere in this file.

Two Evidence Integrity corrections this file exists to pin down (the
first version of this bridge got both wrong):

1. ``Source.tier`` is ``StoredDocument.document.tier`` verbatim -- never
   a fixed ``SourceTier.TIER_2`` constant. Every literature Document this
   repository's adapter constructs leaves ``tier`` at its own class
   default (``SourceTier.UNKNOWN``), so every literature Source this
   bridge produces today reads ``UNKNOWN`` too.
2. A PubMed EFetch document, an Europe PMC search-response document, and
   an Europe PMC full-text document are three SEPARATE documents. A
   RawFact's ``source``/``document_id`` must be the document that fact's
   claim is actually drawn from -- never borrowed from a different one.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import pytest

from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.orchestrator.isolation import EvidenceBus, IsolationGuard
from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentStore
from investment_research.research.literature_acquisition_adapter import (
    LITERATURE_ADAPTER_ID,
    LiteratureReference,
    PubMedLiteratureAdapter,
)
from investment_research.research.literature_evidence_projection import (
    project_literature_target_reports,
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
    TargetKind,
)
from investment_research.schemas.enums import (
    DocumentAuthority,
    EvidenceClass,
    FetchOutcome,
    PeerReviewStatus,
    ResearchDomain,
    SourceTier,
)

from . import _literature_fixture_support as fx

#: A genuinely-executed Europe PMC search that found nothing -- distinct
#: from an UNREGISTERED route (which FakeHttpClient reports as a 404-style
#: "no fake response registered", a real failure this bridge must NOT
#: treat as clean). Every test below whose subject is the PubMed side only
#: still registers this for the PMID it uses, since
#: PubMedLiteratureAdapter._fetch() always attempts a Europe PMC search
#: for every fetched PMID regardless of that article's own PMCID. A
#: genuinely empty result list means no europepmc_by_pmid entry is ever
#: recorded for this PMID (mirrors "not found"), so no Europe PMC
#: RawFact/Source/Document is produced for it either -- exactly like
#: never having searched Europe PMC for it at all.
_EPMC_EMPTY = '{"resultList": {"result": []}}'


def _literature_graph(*tags: str) -> SourceRoutingGraph:
    """Mirrors test_literature_acquisition_adapter.py's own helper of the
    same shape exactly -- kept as a private local copy rather than a
    shared import, matching this test suite's existing per-file
    convention."""
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


def _run(refs: dict, http, *, graph=None, store=None, **adapter_kwargs):
    graph = graph or _literature_graph(*[t.replace("target_", "") for t in refs])
    store = store or DocumentStore()
    adapter = PubMedLiteratureAdapter(http, **adapter_kwargs)
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    report = executor.run(graph, literature_references=refs)
    return report, store


def _through_fact_collector_and_evidence_integrity(collection_result, *, ticker="DEMOBIO"):
    """The rest of the CANONICAL path this bridge feeds into -- exactly the
    existing agents, exactly as Pipeline.run() itself would call them,
    never a shortcut or a reimplementation."""
    bus = EvidenceBus()
    guard = IsolationGuard(bus, strict=True)
    collector = FactCollectorAgent([collection_result])
    collector_input = guard.project(
        collector.agent_id, ticker=ticker, company_name="Demo Bio", aliases=(), facts=None,
        params={"run_id": "test-run"},
    )
    collector_output = collector.execute(collector_input)

    integrity = EvidenceIntegrityAgent()
    integrity_input = guard.project(
        integrity.agent_id, ticker=ticker, company_name="Demo Bio", aliases=(),
        facts=list(collector_output.facts), params={"run_id": "test-run"},
    )
    integrity_output = integrity.execute(integrity_input)
    return collector_output, integrity_output


# --- PubMed abstract present / absent ---------------------------------------
def test_pubmed_abstract_present_projects_excerpt_facts_and_verifies():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.outcome is FetchOutcome.OK
    assert not cr.degraded
    assert cr.raw_facts
    assert projection.coverage_complete
    assert not projection.unresolved_reasons

    _, integrity_output = _through_fact_collector_and_evidence_integrity(cr)
    assert integrity_output.facts
    for fact in integrity_output.facts:
        assert fact.source_authority is DocumentAuthority.BIOMEDICAL_LITERATURE
        assert fact.company_claim is False
        assert fact.independent_confirmation is False
        assert fact.document_id is not None


def test_no_abstract_still_projects_metadata_only_facts():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000003"]): fx.ok(fx.fixture_text("no_abstract.xml")),
            fx.europepmc_search_url("90000003"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000003")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.outcome is FetchOutcome.OK
    assert cr.raw_facts
    assert all(f.unit != "literature_reported_result_wording" for f in cr.raw_facts)
    assert any(f.unit == "literature_publication_exists" for f in cr.raw_facts)


# --- Europe PMC OA full text / non-OA ---------------------------------------
def test_open_access_fulltext_is_biomedical_publication_assertion_never_decision_grade():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.outcome is FetchOutcome.OK
    # PubMed EFetch doc + Europe PMC search doc + Europe PMC fulltext doc:
    # three DIFFERENT documents, never collapsed to fewer.
    assert len(projection.document_ids) == 3

    fulltext_raw = next(f for f in cr.raw_facts if f.unit == "literature_full_text_availability" and f.value is True)
    assert "3 section(s)" in fulltext_raw.claim  # the REAL parsed section count, never guessed

    _, integrity_output = _through_fact_collector_and_evidence_integrity(cr)
    fulltext_fact = next(
        f for f in integrity_output.facts
        if f.unit == "literature_full_text_availability" and f.value is True
    )
    assert fulltext_fact.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
    assert fulltext_fact.is_decision_grade is False
    assert fulltext_fact.independent_confirmation is False
    # FULL_DOCUMENT content_kind means the body was retrieved -- it must
    # never, on its own, imply peer review was confirmed.
    peer_review_facts = [f for f in integrity_output.facts if "peer review" in f.claim.lower()]
    assert peer_review_facts
    assert all(f"{PeerReviewStatus.CONFIRMED.value}:" not in f.claim for f in peer_review_facts)
    assert all(PeerReviewStatus.CONFIRMED.value not in f.value for f in peer_review_facts if isinstance(f.value, str))


def test_non_open_access_never_produces_fulltext_acquired_fact():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert not any(
        f.unit == "literature_full_text_availability" and f.value is True for f in cr.raw_facts
    )
    assert any(f.unit == "literature_open_access_status" for f in cr.raw_facts)
    # PubMed doc + Europe PMC SEARCH doc -- no fulltext document exists,
    # since it was never fetched.
    assert len(projection.document_ids) == 2
    assert not any("fullTextXML" in u for u in http.requested_urls)  # no fulltext GET was made


# --- Preprint / retraction / correction --------------------------------------
def test_preprint_peer_review_status_is_not_peer_reviewed_never_confirmed():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000022"]): fx.ok(fx.fixture_text("preprint.xml")),
            fx.europepmc_search_url("90000022"): fx.ok(fx.fixture_text("europepmc_search_preprint.json")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000022")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    pubmed_peer_review = next(
        f for f in cr.raw_facts if f.unit == "literature_peer_review_status" and "Europe PMC source=" not in f.claim
    )
    assert pubmed_peer_review.value == PeerReviewStatus.NOT_PEER_REVIEWED.value
    epmc_peer_review = next(
        f for f in cr.raw_facts if f.unit == "literature_peer_review_status" and "Europe PMC source=" in f.claim
    )
    assert epmc_peer_review.value == PeerReviewStatus.NOT_PEER_REVIEWED.value
    assert not any(f.value == PeerReviewStatus.CONFIRMED.value for f in cr.raw_facts)


def test_retraction_recorded_as_its_own_fact_never_silently_dropped():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000011"]): fx.ok(fx.fixture_text("retracted.xml"))})
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000011")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    retraction_facts = [f for f in cr.raw_facts if f.unit == "literature_retraction_correction_status"]
    assert retraction_facts
    assert any("RetractionIn" in f.claim for f in retraction_facts)


def test_correction_recorded_as_its_own_fact():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000009"]): fx.ok(fx.fixture_text("correction.xml"))})
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000009")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    correction_facts = [f for f in cr.raw_facts if f.unit == "literature_retraction_correction_status"]
    assert correction_facts
    assert any("CorrectionIn" in f.claim for f in correction_facts)


# --- Dedup across multiple semantic targets ----------------------------------
def test_same_pmid_across_three_targets_deduplicated_by_bridge_pubmed_only():
    graph = _literature_graph("a", "b", "c")
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    refs = {t.target_id: LiteratureReference(ct_pmid="90000001") for t in graph.targets}
    report, store = _run(refs, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert len(cr.sources) == 1
    assert len(projection.document_ids) == 1
    assert projection.duplicate_source_count == 2
    # RawFact.fact_id() dedup left no duplicate identities.
    fact_ids = [f.fact_id() for f in cr.raw_facts]
    assert len(fact_ids) == len(set(fact_ids))
    # Zero extra HTTP requests from having 3 targets share one PMID --
    # AcquisitionExecutor's own dedup (not this bridge) is what bounds this
    # to one EFetch/one Europe PMC search; the bridge makes zero requests
    # of its own either way.
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1
    search_calls = [u for u in http.requested_urls if "/search" in u]
    assert len(search_calls) == 1

    # Feeding the deduplicated CollectionResult through FactCollectorAgent
    # produces no further duplicates either -- the canonical path's own
    # dedup (fact_id-keyed) agrees with the bridge's.
    collector_output, _ = _through_fact_collector_and_evidence_integrity(cr)
    assert len(collector_output.facts) == len(cr.raw_facts)


def test_same_pmid_across_three_targets_deduplicated_by_bridge_oa_fulltext():
    """The full three-document (PubMed/search/fulltext) case, deduplicated
    across three targets sharing one PMID: each Document is stored once,
    dedup never crosses document identity (a PubMed-doc dedup never
    silently absorbs the search or fulltext document, or vice versa)."""
    graph = _literature_graph("a", "b", "c")
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    refs = {t.target_id: LiteratureReference(ct_pmid="90000008") for t in graph.targets}
    report, store = _run(refs, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert len(cr.sources) == 3
    assert len(projection.document_ids) == 3
    assert projection.duplicate_source_count == 2
    fact_ids = [f.fact_id() for f in cr.raw_facts]
    assert len(fact_ids) == len(set(fact_ids))
    for kind, count in (("efetch.fcgi", 1), ("/search", 1), ("fullTextXML", 1)):
        assert len([u for u in http.requested_urls if kind in u]) == count


# --- Failure / completeness semantics ----------------------------------------
def test_budget_excluded_pmid_recorded_as_unresolved_and_degraded_never_clean():
    graph = _literature_graph("a", "b")
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.efetch_url(["90000002"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    refs = {
        graph.targets[0].target_id: LiteratureReference(ct_pmid="90000001"),
        graph.targets[1].target_id: LiteratureReference(ct_pmid="90000002"),
    }
    report, store = _run(refs, http, graph=graph, max_articles=1)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.outcome is not FetchOutcome.OK
    assert cr.degraded
    assert cr.zero_results is False
    assert projection.coverage_complete is False
    assert any("budget" in r.lower() for r in projection.unresolved_reasons)
    # The one PMID that WAS fetched is still projected -- partial failure
    # degrades the result, it never discards what succeeded.
    assert cr.raw_facts


@pytest.mark.parametrize("outcome_name", ["RATE_LIMITED", "ERROR", "TIMEOUT"])
def test_provider_failure_is_never_reported_as_zero_results(outcome_name):
    from investment_research.schemas.enums import FetchOutcome as _FetchOutcome

    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(
        responses={fx.esearch_url(term, 20): fx.failed(_FetchOutcome[outcome_name], outcome_name.lower())}
    )
    report, store = _run({target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.zero_results is False
    assert cr.outcome is not FetchOutcome.OK
    assert cr.degraded
    assert projection.coverage_complete is False
    assert any(outcome_name in r or "FAILED" in r for r in projection.unresolved_reasons)


def test_malformed_xml_response_is_never_reported_as_zero_results_or_clean():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000012"]): fx.ok(fx.fixture_text("malformed.xml"))})
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000012")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.zero_results is False
    assert cr.degraded
    assert not cr.raw_facts
    assert projection.coverage_complete is False
    assert any("MALFORMED" in r for r in projection.unresolved_reasons)


def test_genuine_zero_result_search_is_reported_as_zero_results_and_not_degraded():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990099[si]"
    http = fx.FakeHttpClient(responses={fx.esearch_url(term, 20): fx.ok(fx.esearch_response([]))})
    report, store = _run({target_id: LiteratureReference(nct_id="NCT09990099")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.outcome is FetchOutcome.OK
    assert cr.degraded is False
    assert cr.zero_results is True
    assert not cr.raw_facts
    assert not projection.unresolved_reasons


def test_missing_document_in_document_store_is_recorded_and_excluded():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, _real_store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    empty_store = DocumentStore()  # simulates a document_id the caller's own store never saw
    projection = project_literature_target_reports("DEMOBIO", report.target_reports, empty_store)
    assert projection.collection_result.degraded
    assert not projection.collection_result.raw_facts
    assert projection.coverage_complete is False
    assert any("not found in DocumentStore" in r for r in projection.unresolved_reasons)


def test_missing_europepmc_search_document_never_falls_back_to_pubmed_document():
    """The search Document itself is missing from DocumentStore (never
    the PubMed one) -- the bridge must exclude the Europe PMC facts, not
    silently attribute them to the PubMed document instead."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    search_document_id = parse_result.payload["europepmc"]["90000008"]["search_document_id"]
    del store._by_id[search_document_id]  # noqa: SLF001 - test-only direct removal

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.degraded
    assert projection.coverage_complete is False
    assert any("not found in DocumentStore" in r for r in projection.unresolved_reasons)
    # PubMed facts still projected (partial success preserved).
    assert any(f.unit == "literature_publication_exists" for f in cr.raw_facts)
    # No Europe PMC-only fact (never produced by the PubMed extractor)
    # was misattributed to the PubMed document. "peer_review_status" is
    # excluded from this check since PubMed's OWN extractor also
    # legitimately produces a fact under that same unit name.
    pubmed_doc_id = parse_result.payload["parsed_documents"][0]["document_id"]
    epmc_only_units = {"open_access_status", "full_text_availability"}
    assert not any(
        f"literature_{u}" == f.unit and f.document_id == pubmed_doc_id
        for f in cr.raw_facts
        for u in epmc_only_units
    )
    assert not any("Europe PMC source=" in f.claim and f.document_id == pubmed_doc_id for f in cr.raw_facts)


def test_missing_fulltext_document_never_falls_back_to_search_or_pubmed_document():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    epmc_entry = parse_result.payload["europepmc"]["90000008"]
    fulltext_document_id = epmc_entry["fulltext_document_id"]
    del store._by_id[fulltext_document_id]  # noqa: SLF001 - test-only direct removal

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.degraded
    assert projection.coverage_complete is False
    assert any("not found in DocumentStore" in r for r in projection.unresolved_reasons)
    # No "full text acquired"/"unavailable" fact was fabricated from a
    # different document.
    assert not any(f.unit == "literature_full_text_availability" for f in cr.raw_facts)
    # Search-derived facts (open-access status etc.) are unaffected.
    assert any(f.unit == "literature_open_access_status" for f in cr.raw_facts)


def test_corrupted_version_chain_is_recorded_and_excluded_never_raised():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    document_id = parse_result.payload["parsed_documents"][0]["document_id"]
    existing = store._by_id[document_id]  # noqa: SLF001 - test-only direct corruption, mirrors document_store's own tests
    store._by_id[document_id] = dc_replace(existing, previous_version_id=document_id)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    assert projection.collection_result.degraded
    assert any("version chain corrupted" in r for r in projection.unresolved_reasons)


def test_corrupted_europepmc_search_document_version_chain_excludes_only_that_document():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    search_document_id = parse_result.payload["europepmc"]["90000008"]["search_document_id"]
    existing = store._by_id[search_document_id]  # noqa: SLF001
    store._by_id[search_document_id] = dc_replace(existing, previous_version_id=search_document_id)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert cr.degraded
    assert any("version chain corrupted" in r for r in projection.unresolved_reasons)
    # PubMed facts unaffected by the search document's own corruption.
    assert any(f.unit == "literature_publication_exists" for f in cr.raw_facts)
    assert not any(f.unit == "literature_open_access_status" for f in cr.raw_facts)


# --- Lineage / Source projection rules ---------------------------------------
def test_source_projection_never_copies_document_body_into_excerpt():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    for source in projection.collection_result.sources:
        assert source.excerpt == ""


def test_source_and_raw_fact_document_id_lineage_preserved_through_to_fact():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    assert all(f.document_id in projection.document_ids for f in cr.raw_facts)

    collector_output, integrity_output = _through_fact_collector_and_evidence_integrity(cr)
    for fact in collector_output.facts:
        assert fact.document_id in projection.document_ids
        assert fact.source_authority is DocumentAuthority.BIOMEDICAL_LITERATURE
    for fact in integrity_output.facts:
        assert fact.document_id in projection.document_ids
        assert fact.source_authority is DocumentAuthority.BIOMEDICAL_LITERATURE


def test_pubmed_fact_lineage_is_the_pubmed_efetch_document():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    pubmed_doc_id = parse_result.payload["parsed_documents"][0]["document_id"]
    pubmed_public_url = store.get(pubmed_doc_id).document.url

    pubmed_facts = [f for f in cr.raw_facts if f.unit.startswith("literature_publication_")]
    assert pubmed_facts
    for f in pubmed_facts:
        assert f.document_id == pubmed_doc_id
        assert f.source.url == pubmed_public_url


def test_europepmc_search_metadata_fact_lineage_is_the_search_document_not_pubmed_or_fulltext():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    epmc_entry = parse_result.payload["europepmc"]["90000008"]
    pubmed_doc_id = parse_result.payload["parsed_documents"][0]["document_id"]
    search_document_id = epmc_entry["search_document_id"]
    fulltext_document_id = epmc_entry["fulltext_document_id"]
    search_public_url = store.get(search_document_id).document.url
    assert "europepmc.org" in search_public_url or "ebi.ac.uk" in search_public_url
    assert "search" in search_public_url

    open_access_fact = next(f for f in cr.raw_facts if f.unit == "literature_open_access_status")
    assert open_access_fact.document_id == search_document_id
    assert open_access_fact.document_id != pubmed_doc_id
    assert open_access_fact.document_id != fulltext_document_id
    assert open_access_fact.source.url == search_public_url


def test_europepmc_fulltext_acquired_fact_lineage_is_the_fulltext_document_not_search_or_pubmed():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    epmc_entry = parse_result.payload["europepmc"]["90000008"]
    pubmed_doc_id = parse_result.payload["parsed_documents"][0]["document_id"]
    search_document_id = epmc_entry["search_document_id"]
    fulltext_document_id = epmc_entry["fulltext_document_id"]
    fulltext_public_url = store.get(fulltext_document_id).document.url
    assert "fullTextXML" in fulltext_public_url

    acquired_fact = next(
        f for f in cr.raw_facts if f.unit == "literature_full_text_availability" and f.value is True
    )
    assert acquired_fact.document_id == fulltext_document_id
    assert acquired_fact.document_id != pubmed_doc_id
    assert acquired_fact.document_id != search_document_id
    assert acquired_fact.source.url == fulltext_public_url


def test_non_oa_unavailable_fact_lineage_is_the_search_document():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    cr = projection.collection_result
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    epmc_entry = parse_result.payload["europepmc"]["90000001"]
    pubmed_doc_id = parse_result.payload["parsed_documents"][0]["document_id"]
    search_document_id = epmc_entry["search_document_id"]
    assert epmc_entry.get("fulltext_document_id") is None

    unavailable_fact = next(
        f for f in cr.raw_facts if f.unit == "literature_full_text_availability" and f.value is False
    )
    assert unavailable_fact.document_id == search_document_id
    assert unavailable_fact.document_id != pubmed_doc_id


# --- Tier: never inferred, never a fixed constant -----------------------------
def test_preprint_source_tier_is_unknown_never_tier_2():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000022"]): fx.ok(fx.fixture_text("preprint.xml")),
            fx.europepmc_search_url("90000022"): fx.ok(fx.fixture_text("europepmc_search_preprint.json")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000022")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    for source in projection.collection_result.sources:
        assert source.tier is SourceTier.UNKNOWN
        assert source.tier is not SourceTier.TIER_2


def test_pubmed_journal_article_source_tier_is_unknown_never_auto_tier_2():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    for source in projection.collection_result.sources:
        assert source.tier is SourceTier.UNKNOWN


def test_europepmc_search_and_fulltext_source_tier_are_both_unknown():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)

    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    assert len(projection.collection_result.sources) == 3
    for source in projection.collection_result.sources:
        assert source.tier is SourceTier.UNKNOWN


def test_tier_never_promotes_evidence_class_or_independent_confirmation():
    """Whatever Source.tier reads, BIOMEDICAL_PUBLICATION_ASSERTION /
    independent_confirmation=False / is_decision_grade=False are driven
    by source_authority, not tier -- unaffected by the tier fix."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    projection = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    _, integrity_output = _through_fact_collector_and_evidence_integrity(projection.collection_result)
    fulltext_fact = next(
        f for f in integrity_output.facts
        if f.unit == "literature_full_text_availability" and f.value is True
    )
    assert fulltext_fact.source_tier is SourceTier.UNKNOWN
    assert fulltext_fact.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
    assert fulltext_fact.independent_confirmation is False
    assert fulltext_fact.is_decision_grade is False


# --- Zero external requests made by the bridge itself ------------------------
def test_bridge_itself_makes_zero_additional_requests():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    before = list(http.requested_urls)

    project_literature_target_reports("DEMOBIO", report.target_reports, store)

    assert http.requested_urls == before  # not one more request was made


# --- Payload no longer carries full-text body/section content ----------------
def test_execution_report_payload_never_carries_parsed_fulltext_body():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    epmc_entry = parse_result.payload["europepmc"]["90000008"]
    assert "parsed_fulltext" not in epmc_entry
    assert epmc_entry.get("fulltext_section_count") == 3
    assert "fictional full-text introduction content" not in str(parse_result.payload)


# --- Production non-connection ------------------------------------------------
def test_cli_never_imports_the_bridge():
    import investment_research.cli as cli_module

    assert "literature_evidence_projection" not in cli_module.__dict__
    assert "project_literature_target_reports" not in cli_module.__dict__


def test_pipeline_never_imports_the_bridge():
    import investment_research.orchestrator.pipeline as pipeline_module

    assert "literature_evidence_projection" not in pipeline_module.__dict__
    assert "project_literature_target_reports" not in pipeline_module.__dict__


def test_bridge_module_never_imports_pipeline_or_cli():
    import investment_research.research.literature_evidence_projection as bridge_module

    assert "pipeline" not in bridge_module.__dict__
    assert "cli" not in bridge_module.__dict__
    assert not hasattr(bridge_module, "Pipeline")


def test_source_routing_catalog_and_catalog_invariants_unchanged():
    from investment_research.research.source_routing_catalog import routing_coverage_counts

    counts = routing_coverage_counts()
    assert counts.offline_verified_steps == 74
    assert counts.live_verified_steps == 9
    assert counts.legacy_needs == 49
    assert counts.equivalence_groups == 31


def test_kill_rules_count_unchanged():
    from investment_research.scoring.kill_gate import KILL_RULES

    assert len(KILL_RULES) == 19
