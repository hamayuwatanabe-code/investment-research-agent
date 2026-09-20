"""Phase 4.1B: Literature Chunk Projection Bridge.

Offline end-to-end: AcquisitionExecutor -> PubMedLiteratureAdapter ->
DocumentStore -> literature_evidence_projection (Phase 4.1A) ->
literature_chunk_projection (Phase 4.1B) -> FactCollectorAgent ->
EvidenceIntegrityAgent -> TraceabilityIndex, against the SAME real-format
fixtures/FakeHttpClient double ``test_literature_acquisition_adapter.py``/
``test_literature_evidence_projection.py`` use. No real network call
anywhere in this file, no real PMID/PMCID/NCT id -- every identifier here
is one of the existing synthetic fixtures' own fictional ids.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.documents import Document, chunk_document
from investment_research.orchestrator.isolation import EvidenceBus, IsolationGuard
from investment_research.reporting.traceability import build_index
from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.document_store import DocumentStore
from investment_research.research.literature_acquisition_adapter import (
    LITERATURE_ADAPTER_ID,
    LiteratureReference,
    PubMedLiteratureAdapter,
)
from investment_research.research.literature_chunk_projection import (
    LiteratureChunkProjection,
    project_literature_chunks,
)
from investment_research.research.literature_evidence_projection import (
    project_literature_target_reports,
)
from investment_research.schemas.enums import (
    ContentKind,
    DocumentAuthority,
    EvidenceClass,
    PeerReviewStatus,
)

from . import _literature_fixture_support as fx
from .test_literature_evidence_projection import _literature_graph

_EPMC_EMPTY = '{"resultList": {"result": []}}'


def _run(refs: dict, http, *, graph=None, **adapter_kwargs):
    graph = graph or _literature_graph(*[t.replace("target_", "") for t in refs])
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(http, **adapter_kwargs)
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    report = executor.run(graph, literature_references=refs)
    return report, store


def _through_fact_collector_and_evidence_integrity(collection_result, *, ticker="DEMOBIO"):
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


# --- PubMed EXCERPT chunk generation ------------------------------------------
def test_pubmed_abstract_present_is_chunked_as_excerpt():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    assert evidence.coverage_complete

    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    assert chunk_projection.coverage_complete
    assert not chunk_projection.unresolved_reasons
    assert chunk_projection.chunks
    assert chunk_projection.duplicate_document_count == 0
    assert set(chunk_projection.projected_document_ids) == set(evidence.document_ids)

    pubmed_doc_id = evidence.document_ids[0]
    stored = store.get(pubmed_doc_id)
    assert stored.document.content_kind is ContentKind.EXCERPT
    for chunk in chunk_projection.chunks:
        assert chunk.doc_id == pubmed_doc_id
        assert chunk.document is stored.document  # the ORIGINAL Document, never a copy


# --- Europe PMC FULL_DOCUMENT chunk generation / search METADATA_ONLY excluded
def test_europepmc_fulltext_is_chunked_search_document_is_excluded():
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
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    assert len(evidence.document_ids) == 3  # pubmed + search + fulltext

    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    assert chunk_projection.coverage_complete
    assert not chunk_projection.unresolved_reasons

    parse_result = next(r for r in report.target_reports[0].step_results if r.step_id == "p_a")
    epmc_entry = parse_result.payload["europepmc"]["90000008"]
    pubmed_doc_id = parse_result.payload["parsed_documents"][0]["document_id"]
    search_doc_id = epmc_entry["search_document_id"]
    fulltext_doc_id = epmc_entry["fulltext_document_id"]

    assert search_doc_id in chunk_projection.excluded_non_primary_document_ids
    assert search_doc_id not in chunk_projection.projected_document_ids
    assert store.get(search_doc_id).document.content_kind is ContentKind.METADATA_ONLY
    assert not any(c.doc_id == search_doc_id for c in chunk_projection.chunks)

    assert pubmed_doc_id in chunk_projection.projected_document_ids
    assert fulltext_doc_id in chunk_projection.projected_document_ids
    assert store.get(fulltext_doc_id).document.content_kind is ContentKind.FULL_DOCUMENT
    assert any(c.doc_id == fulltext_doc_id for c in chunk_projection.chunks)
    assert any(c.doc_id == pubmed_doc_id for c in chunk_projection.chunks)

    # FULL_DOCUMENT never, on its own, implies confirmed peer review or
    # decision-grade evidence -- carried through the canonical path exactly
    # as Phase 4.1A's own equivalent assertions do.
    _, integrity_output = _through_fact_collector_and_evidence_integrity(evidence.collection_result)
    fulltext_fact = next(
        f for f in integrity_output.facts
        if f.unit == "literature_full_text_availability" and f.value is True
    )
    assert fulltext_fact.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
    assert fulltext_fact.is_decision_grade is False
    assert fulltext_fact.independent_confirmation is False
    peer_review_facts = [f for f in integrity_output.facts if "peer review" in f.claim.lower()]
    assert peer_review_facts
    assert all(f"{PeerReviewStatus.CONFIRMED.value}:" not in f.claim for f in peer_review_facts)


# --- non-OA: fulltext never attempted -----------------------------------------
def test_non_open_access_produces_no_fulltext_document_to_chunk():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    assert len(evidence.document_ids) == 2  # pubmed + search only

    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    assert chunk_projection.coverage_complete
    # Only the PubMed EXCERPT document is chunk-eligible; the search
    # document (METADATA_ONLY) is excluded, and no fulltext document
    # exists at all to even consider.
    assert len(chunk_projection.projected_document_ids) == 1
    assert len(chunk_projection.excluded_non_primary_document_ids) == 1


# --- no abstract: PubMed doc is METADATA_ONLY, never chunked -----------------
def test_no_abstract_pubmed_document_is_metadata_only_never_chunked():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000003"]): fx.ok(fx.fixture_text("no_abstract.xml")),
            fx.europepmc_search_url("90000003"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000003")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    pubmed_doc_id = evidence.document_ids[0]
    assert store.get(pubmed_doc_id).document.content_kind is ContentKind.METADATA_ONLY

    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    assert chunk_projection.coverage_complete
    assert not chunk_projection.chunks
    assert pubmed_doc_id in chunk_projection.excluded_non_primary_document_ids
    assert pubmed_doc_id not in chunk_projection.projected_document_ids


# --- duplicate target / duplicate document_id ---------------------------------
def test_duplicate_document_id_input_is_deduplicated_never_chunked_twice():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    pubmed_doc_id = evidence.document_ids[0]

    duplicated_input = (pubmed_doc_id, pubmed_doc_id, pubmed_doc_id)
    chunk_projection = project_literature_chunks(duplicated_input, store)
    assert chunk_projection.duplicate_document_count == 2
    assert chunk_projection.projected_document_ids == (pubmed_doc_id,)
    chunk_ids = [c.chunk_id for c in chunk_projection.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # never chunked twice


def test_same_pmid_across_three_targets_document_ids_dedup_at_evidence_layer():
    """The evidence-layer dedup already collapses a shared PMID to one
    document_id set (Phase 4.1A); the chunk layer's own dedup is then a
    no-op defense-in-depth, not the primary mechanism -- both hold."""
    graph = _literature_graph("a", "b", "c")
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    refs = {t.target_id: LiteratureReference(ct_pmid="90000001") for t in graph.targets}
    report, store = _run(refs, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    assert len(evidence.document_ids) == 1

    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    assert chunk_projection.duplicate_document_count == 0
    assert len(chunk_projection.projected_document_ids) == 1


# --- missing Document ----------------------------------------------------------
def test_missing_document_in_store_is_recorded_never_raised():
    empty_store = DocumentStore()
    chunk_projection = project_literature_chunks(("doc_does_not_exist",), empty_store)
    assert chunk_projection.coverage_complete is False
    assert not chunk_projection.chunks
    assert any("not found in DocumentStore" in r for r in chunk_projection.unresolved_reasons)
    assert "doc_does_not_exist" not in chunk_projection.projected_document_ids


# --- corrupted version chain ----------------------------------------------------
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
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    pubmed_doc_id = evidence.document_ids[0]

    existing = store._by_id[pubmed_doc_id]  # noqa: SLF001 - test-only, mirrors document_store's own tests
    store._by_id[pubmed_doc_id] = dc_replace(existing, previous_version_id=pubmed_doc_id)

    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    assert chunk_projection.coverage_complete is False
    assert not chunk_projection.chunks
    assert any("version chain corrupted" in r for r in chunk_projection.unresolved_reasons)


# --- empty primary-text body -----------------------------------------------------
def test_empty_primary_text_body_is_recorded_never_silently_zero_chunks():
    store = DocumentStore()
    empty_excerpt = Document(
        doc_id="doc_empty_excerpt",
        url="https://pubmed.ncbi.nlm.nih.gov/90000099/",
        title="An Article With No Retrievable Body",
        text="",
        content_kind=ContentKind.EXCERPT,
    )
    store.put(empty_excerpt, document_id=empty_excerpt.doc_id)

    chunk_projection = project_literature_chunks(("doc_empty_excerpt",), store)
    assert chunk_projection.coverage_complete is False
    assert not chunk_projection.chunks
    assert "doc_empty_excerpt" not in chunk_projection.projected_document_ids
    assert "doc_empty_excerpt" not in chunk_projection.excluded_non_primary_document_ids
    assert any("body is empty" in r for r in chunk_projection.unresolved_reasons)


# --- deterministic Chunk ID / order -----------------------------------------------
def test_same_input_produces_identical_chunk_ids_and_order():
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
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)

    first = project_literature_chunks(evidence.document_ids, store)
    second = project_literature_chunks(evidence.document_ids, store)
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert first.projected_document_ids == second.projected_document_ids


# --- Chunk traces correctly to Fact/document/source through Traceability -------
def test_chunk_traces_to_fact_document_and_source_through_traceability_index():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    assert chunk_projection.chunks

    collector_output, integrity_output = _through_fact_collector_and_evidence_integrity(
        evidence.collection_result
    )
    assert integrity_output.facts

    index = build_index(list(integrity_output.facts), evidence.collection_result.sources, chunk_projection.chunks)
    pubmed_doc_id = evidence.document_ids[0]
    facts_for_doc = [f for f in integrity_output.facts if f.document_id == pubmed_doc_id]
    assert facts_for_doc

    resolved_any = False
    for chunk in chunk_projection.chunks:
        link = index.resolve(chunk.chunk_id)
        if link is not None:
            resolved_any = True
            assert link.fact_id in {f.fact_id for f in facts_for_doc}
    assert resolved_any

    # BIOMEDICAL_PUBLICATION_ASSERTION never becomes Decision Grade and
    # independent_confirmation=False is maintained end to end.
    for fact in facts_for_doc:
        assert fact.source_authority is DocumentAuthority.BIOMEDICAL_LITERATURE
        assert fact.independent_confirmation is False
        if fact.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION:
            assert fact.is_decision_grade is False


# --- integrity: Chunk.doc_id / Document.doc_id mismatch is checked, not assumed
def test_chunk_doc_id_matches_document_doc_id_for_every_produced_chunk():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    for chunk in chunk_projection.chunks:
        assert chunk.doc_id == chunk.document.doc_id


# --- Chunk generation is a faithful call to the existing chunker -------------
def test_chunks_match_a_direct_chunk_document_call_on_the_same_document():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    pubmed_doc_id = evidence.document_ids[0]
    document = store.get(pubmed_doc_id).document

    chunk_projection = project_literature_chunks(evidence.document_ids, store)
    expected = chunk_document(document, target_tokens=400, overlap_sentences=1)
    assert [c.chunk_id for c in chunk_projection.chunks] == [c.chunk_id for c in expected]
    assert [c.text for c in chunk_projection.chunks] == [c.text for c in expected]


# --- Production non-connection ------------------------------------------------
def test_cli_never_imports_the_chunk_bridge():
    import investment_research.cli as cli_module

    assert "literature_chunk_projection" not in cli_module.__dict__
    assert "project_literature_chunks" not in cli_module.__dict__


def test_pipeline_never_imports_the_chunk_bridge():
    import investment_research.orchestrator.pipeline as pipeline_module

    assert "literature_chunk_projection" not in pipeline_module.__dict__
    assert "project_literature_chunks" not in pipeline_module.__dict__


def test_chunk_bridge_module_never_imports_pipeline_or_cli():
    import investment_research.research.literature_chunk_projection as bridge_module

    assert "pipeline" not in bridge_module.__dict__
    assert "cli" not in bridge_module.__dict__
    assert not hasattr(bridge_module, "Pipeline")


def test_chunk_bridge_makes_zero_additional_requests():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
        }
    )
    report, store = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    evidence = project_literature_target_reports("DEMOBIO", report.target_reports, store)
    before = list(http.requested_urls)

    project_literature_chunks(evidence.document_ids, store)

    assert http.requested_urls == before


def test_dataclass_shape_matches_required_fields():
    projection = LiteratureChunkProjection(
        chunks=(),
        projected_document_ids=(),
        excluded_non_primary_document_ids=(),
        unresolved_reasons=(),
        coverage_complete=True,
    )
    assert projection.duplicate_document_count == 0
    assert isinstance(projection.chunks, tuple)
