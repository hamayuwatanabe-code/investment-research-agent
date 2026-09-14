"""ClinicalTrialsStudyAdapter (Phase 3D) exercised entirely offline against
the real-format fixtures in tests/fixtures/clinicaltrials_v2_real_format/.

No real network call anywhere in this file -- every test injects
``FakeHttpClient`` from ``_clinicaltrials_fixture_support``.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, fields

from investment_research.collectors.clinicaltrials import parse_study, raw_facts_from_study
from investment_research.research.acquisition_executor import (
    AcquisitionExecutor,
    ExecutionContext,
)
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.clinicaltrials_acquisition_adapter import (
    CLINICALTRIALS_ADAPTER_ID,
    ClinicalTrialsStudyAdapter,
    ClinicalTrialsStudyReference,
)
from investment_research.research.document_store import DocumentRole, DocumentStore
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
from investment_research.schemas.enums import UNKNOWN, DocumentAuthority, ResearchDomain, SourceTier
from investment_research.schemas.fact import Fact, RawFact, Source

from . import _clinicaltrials_fixture_support as fx
from ._clinicaltrials_fixture_support import FakeHttpClient


class _PoisonHttpClient:
    def get(self, url, **kwargs):
        raise AssertionError(f"must never be called -- attempted GET {url}")


def _one_target_graph(target_id: str = "target_ct") -> tuple[SourceRoutingGraph, AcquisitionTarget]:
    """A bespoke LOCATE -> FETCH -> PARSE graph -- separate from
    source_routing_catalog.py's LOCATE-less structured archetype, mirroring
    how sec_live_smoke.py drives the SEC adapters through their own custom
    3-step graph. Step ids are scoped by ``target_id`` -- ``AcquisitionExecutor
    .run()``'s ``outcomes`` dict is keyed by step_id ALONE across the whole
    graph, so two different targets' steps must never share a step_id, on
    pain of one target's outcome silently overwriting/pre-satisfying the
    other's (exactly the fan-out scenario this graph is built to test)."""
    l1, f, p = f"{target_id}_l1", f"{target_id}_f", f"{target_id}_p"
    steps = [
        AcquisitionStep(
            step_id=l1, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id=CLINICALTRIALS_ADAPTER_ID,
            completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
            implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        ),
        AcquisitionStep(
            step_id=f, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id=CLINICALTRIALS_ADAPTER_ID,
            depends_on_step_ids=(l1,), completion_condition=StepStatus.STRUCTURED_RECORD_RETRIEVED,
            implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        ),
        AcquisitionStep(
            step_id=p, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id=CLINICALTRIALS_ADAPTER_ID,
            depends_on_step_ids=(f,), completion_condition=StepStatus.REQUIRED_FIELDS_PARSED,
            implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        ),
    ]
    requirement = EvidenceRequirement(
        requirement_id=f"req_{target_id}", serves_legacy_need_ids=(f"synthetic_{target_id}",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.SCIENCE_TECHNOLOGY,
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.CLINICALTRIALS_RECORD,
        required_step_ids=(l1, f, p), serves_requirement_ids=(requirement.requirement_id,),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))
    return graph, target


def _run(nct_id: str, http, target_id: str = "target_ct"):
    graph, target = _one_target_graph(target_id)
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(http)},
        document_store=store,
    )
    report = executor.run(graph, study_references={target_id: ClinicalTrialsStudyReference(nct_id=nct_id)})
    return report, store, target


# --- LOCATE: zero network, NCT ID validation ---------------------------------
def test_locate_builds_deterministic_url_with_zero_network_calls():
    """LOCATE in isolation -- NOT through the full executor, which would
    immediately advance into FETCH within the same run() call and trip the
    poison client. This calls the adapter directly for exactly one step,
    proving LOCATE itself never touches the transport."""
    graph, target = _one_target_graph()
    adapter = ClinicalTrialsStudyAdapter(_PoisonHttpClient())
    context = ExecutionContext(
        target=target, document_store=DocumentStore(),
        study_references={target.target_id: ClinicalTrialsStudyReference(nct_id=fx.NCT_ID)},
    )
    locate_step = next(s for s in graph.steps if s.step_kind is StepKind.LOCATE)
    result = adapter.execute(locate_step, context)
    assert result.status is StepStatus.URL_RESOLVED
    assert result.payload["url"] == fx.study_url(fx.NCT_ID)


def test_locate_refuses_malformed_nct_id():
    report, _, _ = _run("not-an-nct-id", _PoisonHttpClient())
    assert report.outcomes["target_ct_l1"] is StepStatus.FAILED


def test_locate_fails_without_a_study_reference():
    graph, target = _one_target_graph()
    executor = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(_PoisonHttpClient())},
        document_store=DocumentStore(),
    )
    report = executor.run(graph)  # no study_references supplied at all
    assert report.outcomes["target_ct_l1"] is StepStatus.FAILED


# --- FETCH: success, 404, transient errors, malformed/wrong-shape JSON ------
def test_fetch_stores_document_and_returns_structured_record_retrieved():
    http = FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run(fx.NCT_ID, http)
    assert report.outcomes["target_ct_f"] is StepStatus.STRUCTURED_RECORD_RETRIEVED
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    stored_docs = store.resolve_by_accession(fx.NCT_ID)
    assert len(stored_docs) == 1
    assert stored_docs[0].document.authority is DocumentAuthority.REGISTRY
    # ClinicalTrials.gov is a government-operated REGISTRY, never a
    # company IR channel -- is_company_ir specifically asserts "the
    # issuing company controls this host". Separate from RawFact.
    # company_claim=True (the record's CONTENT is sponsor-submitted),
    # asserted below via raw_facts_from_study.
    assert stored_docs[0].document.is_company_ir is False
    assert stored_docs[0].document_role is DocumentRole.STRUCTURED_API_RECORD
    assert stored_docs[0].canonical_url == fx.study_url(fx.NCT_ID)
    assert stored_docs[0].content_hash() != "UNKNOWN"


def test_fetch_404_is_not_found_never_acquired():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID_404): fx.not_found()})
    report, _, target = _run(fx.NCT_ID_404, http)
    assert report.outcomes["target_ct_f"] is StepStatus.NOT_FOUND
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_fetch_rate_limited_is_failed_never_confused_with_not_found():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID): fx.rate_limited()})
    report, _, _ = _run(fx.NCT_ID, http)
    assert report.outcomes["target_ct_f"] is StepStatus.FAILED
    assert report.outcomes["target_ct_f"] is not StepStatus.NOT_FOUND


def test_fetch_server_error_is_failed():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID): fx.server_error()})
    report, _, _ = _run(fx.NCT_ID, http)
    assert report.outcomes["target_ct_f"] is StepStatus.FAILED


def test_fetch_timeout_is_failed():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID): fx.timed_out()})
    report, _, _ = _run(fx.NCT_ID, http)
    assert report.outcomes["target_ct_f"] is StepStatus.FAILED


def test_fetch_malformed_json_is_failed_never_acquired():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID_MALFORMED): fx.ok(fx.fixture_text("study_malformed.txt"))})
    report, _, target = _run(fx.NCT_ID_MALFORMED, http)
    assert report.outcomes["target_ct_f"] is StepStatus.FAILED
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_fetch_wrong_shape_json_is_failed_distinct_from_malformed():
    """Valid JSON, but not a CT.gov study record at all -- a schema error,
    never silently treated as a successful fetch."""
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID): fx.ok(json.dumps({"unexpected": "shape"}))})
    report, _, _ = _run(fx.NCT_ID, http)
    assert report.outcomes["target_ct_f"] is StepStatus.FAILED


# --- dedup / fan-out ----------------------------------------------------------
def test_same_nct_id_requested_by_three_targets_costs_exactly_one_http_get():
    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    graphs_targets = [_one_target_graph(f"target_{i}") for i in range(3)]
    all_steps = tuple(s for graph, _ in graphs_targets for s in graph.steps)
    all_requirements = tuple(r for graph, _ in graphs_targets for r in graph.requirements)
    all_targets = tuple(t for _, t in graphs_targets)
    combined_graph = SourceRoutingGraph(requirements=all_requirements, targets=all_targets, steps=all_steps)

    executor = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(http)},
        document_store=store,
    )
    study_references = {f"target_{i}": ClinicalTrialsStudyReference(nct_id=fx.NCT_ID) for i in range(3)}
    report = executor.run(combined_graph, study_references=study_references)

    assert len(http.requested_urls) == 1
    for i in range(3):
        assert report.outcome_for(f"target_{i}") is TargetAcquisitionOutcome.ACQUIRED
    assert report.diagnostics.duplicate_acquisition_avoided >= 2
    # Exactly one Document was ever stored for this NCT ID, shared by all three.
    assert len(store.resolve_by_accession(fx.NCT_ID)) == 1


# --- PARSE: required fields, missing status, partial dates -------------------
def test_parse_extracts_fields_and_confirms_required_fields_parsed():
    http = FakeHttpClient(responses=fx.default_responses())
    report, _, _ = _run(fx.NCT_ID, http)
    assert report.outcomes["target_ct_p"] is StepStatus.REQUIRED_FIELDS_PARSED


def test_parse_fails_when_overall_status_is_missing():
    http = FakeHttpClient(
        responses={fx.study_url(fx.NCT_ID_MISSING_FIELDS): fx.ok(fx.fixture_text("study_missing_status_and_dates.json"))}
    )
    report, _, target = _run(fx.NCT_ID_MISSING_FIELDS, http)
    assert report.outcomes["target_ct_f"] is StepStatus.STRUCTURED_RECORD_RETRIEVED  # fetch succeeded --
    assert report.outcomes["target_ct_p"] is StepStatus.FAILED  # but required fields were absent
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_partial_dates_pass_through_verbatim_never_padded():
    study = json.loads(fx.fixture_text("study_partial_dates.json"))
    parsed = parse_study(study)
    assert parsed["start_date"] == "2025"
    assert parsed["primary_completion"] == "2026-03"
    assert parsed["completion_date"] == "2026-09"


# --- status variety: recruiting/completed/terminated/withdrawn ---------------
def test_recruiting_completed_terminated_withdrawn_all_parse_and_acquire_cleanly():
    cases = [
        (fx.NCT_ID_COMPLETED, "study_completed_with_results.json", "COMPLETED"),
        (fx.NCT_ID_TERMINATED, "study_terminated.json", "TERMINATED"),
        (fx.NCT_ID_WITHDRAWN, "study_withdrawn.json", "WITHDRAWN"),
    ]
    for nct_id, filename, _expected_status in cases:
        http = FakeHttpClient(responses={fx.study_url(nct_id): fx.ok(fx.fixture_text(filename))})
        report, store, target = _run(nct_id, http)
        assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED, filename
        parsed = report.outcomes["target_ct_p"]
        assert parsed is StepStatus.REQUIRED_FIELDS_PARSED, filename


def test_completed_status_is_never_interpreted_as_trial_success():
    """Registered COMPLETED status is stored verbatim -- this test proves
    no additional field or interpretation is layered on top of it."""
    study = json.loads(fx.fixture_text("study_completed_with_results.json"))
    parsed = parse_study(study)
    assert parsed["status"] == "COMPLETED"
    assert "success" not in {k.lower() for k in parsed}
    assert "efficacy" not in {k.lower() for k in parsed}


def test_terminated_and_withdrawn_do_not_carry_any_kill_or_program_field():
    """This adapter never decides whether a terminated/withdrawn trial is
    the company's CURRENT programme, and never emits any field that could
    be mistaken for a kill/action verdict -- that judgment belongs entirely
    to scoring/program_resolution.py and the Kill Gate, operating on
    Facts, never to this acquisition-only module."""
    for filename in ("study_terminated.json", "study_withdrawn.json"):
        study = json.loads(fx.fixture_text(filename))
        parsed = parse_study(study)
        forbidden = {"kill", "action", "verdict", "sufficient", "efficacy_confirmed"}
        assert forbidden.isdisjoint({k.lower() for k in parsed})


# --- results section presence, never interpreted -----------------------------
def test_results_section_presence_is_recorded_never_interpreted():
    recruiting = parse_study(json.loads(fx.fixture_text("study_recruiting_interventional.json")))
    completed = parse_study(json.loads(fx.fixture_text("study_completed_with_results.json")))
    assert recruiting["has_results"] is False
    assert completed["has_results"] is True


def test_no_results_section_is_never_read_as_bad_outcome():
    """A study with no results section yet must parse and acquire exactly
    as cleanly as one that has one -- there is no separate, more permissive
    or more restrictive code path for either case."""
    http_no_results = FakeHttpClient(responses=fx.default_responses())
    report_no_results, _, target = _run(fx.NCT_ID, http_no_results)
    assert report_no_results.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED

    http_with_results = FakeHttpClient(
        responses={fx.study_url(fx.NCT_ID_COMPLETED): fx.ok(fx.fixture_text("study_completed_with_results.json"))}
    )
    report_with_results, _, target2 = _run(fx.NCT_ID_COMPLETED, http_with_results)
    assert report_with_results.outcome_for(target2.target_id) is TargetAcquisitionOutcome.ACQUIRED


# --- versioning ---------------------------------------------------------------
def test_document_store_versions_an_updated_study_at_the_same_nct_id():
    http_v1 = FakeHttpClient(responses={fx.study_url(fx.NCT_ID): fx.ok(fx.fixture_text("study_recruiting_interventional.json"))})
    _, store, _ = _run(fx.NCT_ID, http_v1)
    v1_docs = store.resolve_by_accession(fx.NCT_ID)
    assert len(v1_docs) == 1
    v1_id = v1_docs[0].document_id

    # A SEPARATE run reusing the SAME DocumentStore, with updated content at
    # the identical URL -- mirrors a later live re-fetch of the same NCT ID.
    graph, target = _one_target_graph("target_ct_v2")
    http_v2 = FakeHttpClient(responses={fx.study_url(fx.NCT_ID): fx.ok(fx.fixture_text("study_recruiting_interventional_v2.json"))})
    executor = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(http_v2)},
        document_store=store,
    )
    executor.run(graph, study_references={"target_ct_v2": ClinicalTrialsStudyReference(nct_id=fx.NCT_ID)})

    # resolve_by_accession returns EVERY version ever stored under this
    # NCT ID -- both v1 and v2 remain individually retrievable.
    v2_docs = store.resolve_by_accession(fx.NCT_ID)
    assert len(v2_docs) == 2
    latest = next(d for d in v2_docs if d.version == 2)
    assert latest.previous_version_id == v1_id
    history = store.version_history(latest.document_id)
    assert len(history) == 2
    assert history[0].document_id == v1_id
    assert history[-1].document_id == latest.document_id


# --- zero external LLM/AI activity, always -----------------------------------
def test_no_web_search_adapter_needed_when_nct_id_is_known():
    """A run with a known NCT ID never even registers a web-search adapter
    -- proving Web Search truly plays no role, not merely that it wasn't
    invoked this time."""
    http = FakeHttpClient(responses=fx.default_responses())
    graph, target = _one_target_graph()
    executor = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(http)},  # no web-search adapter registered at all
        document_store=DocumentStore(),
    )
    report = executor.run(graph, study_references={target.target_id: ClinicalTrialsStudyReference(nct_id=fx.NCT_ID)})
    assert report.diagnostics.external_llm_tokens == 0
    assert report.diagnostics.web_search_steps_not_executed == 0
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED


# --- Phase 3D.1: authority semantics (REGISTRY != company IR channel) ------
def test_document_host_authority_and_company_channel_are_kept_separate():
    """Four distinct concerns, never collapsed into one boolean or one
    enum value: (1) document host/authority = REGISTRY -- a government-
    operated registry, (2) is_company_ir = False -- the company does not
    control clinicaltrials.gov, (3) the CLAIM's submitter is the sponsor,
    captured at the RawFact level as company_claim=True, (4) independent
    confirmation of anything the sponsor asserted = False, since nothing
    here has independently verified it."""
    http = FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run(fx.NCT_ID, http)
    stored = store.resolve_by_accession(fx.NCT_ID)[0]

    assert stored.document.authority is DocumentAuthority.REGISTRY
    assert stored.document.is_company_ir is False

    raw_facts = raw_facts_from_study(
        "TEST", parse_study(json.loads(fx.fixture_text("study_recruiting_interventional.json"))),
        Source(source_id="s1", url=fx.study_url(fx.NCT_ID), title="t", tier=SourceTier.TIER_1),
        document_id=stored.document_id,
    )
    assert raw_facts
    assert all(f.company_claim is True for f in raw_facts)
    # RawFact carries no independent-confirmation or evidence-class field at
    # all -- the Fact Collector cannot assert either, by construction (see
    # schemas/fact.py's RawFact docstring: "evaluation-free by construction").
    raw_fact_fields = {f.name for f in fields(RawFact)}
    assert "independent_confirmation" not in raw_fact_fields
    assert "evidence_class" not in raw_fact_fields
    # And Fact's OWN dataclass defaults, if one were ever built from this
    # RawFact downstream, never default to an upgraded/confirmed state.
    assert Fact.__dataclass_fields__["independent_confirmation"].default is False
    assert Fact.__dataclass_fields__["evidence_class"].default is MISSING  # required -- never silently defaulted


def test_completed_never_upgrades_to_efficacy_success_or_independent_evidence():
    """A COMPLETED study with a resultsSection present is still only a
    REGISTRY-authority, sponsor-submitted record -- neither the completion
    status nor the presence of results implies peer review or independent
    verification of anything."""
    http = FakeHttpClient(
        responses={fx.study_url(fx.NCT_ID_COMPLETED): fx.ok(fx.fixture_text("study_completed_with_results.json"))}
    )
    report, store, target = _run(fx.NCT_ID_COMPLETED, http)
    stored = store.resolve_by_accession(fx.NCT_ID_COMPLETED)[0]
    assert stored.document.authority is DocumentAuthority.REGISTRY  # never REGULATOR/INDEPENDENT
    assert stored.document.is_company_ir is False

    parsed = parse_study(json.loads(fx.fixture_text("study_completed_with_results.json")))
    assert parsed["has_results"] is True
    assert parsed["status"] == "COMPLETED"
    # No field anywhere claims efficacy, independence, or peer review.
    forbidden = {"efficacy", "peer_reviewed", "independent", "verified", "success"}
    assert forbidden.isdisjoint({k.lower() for k in parsed})


def test_clinicaltrials_alone_never_reported_as_science_or_regulatory_sufficient():
    """This module never imports evidence_sufficiency.py at all -- it
    cannot, structurally, mark any domain SUFFICIENT; that determination
    belongs entirely to scoring/evidence_sufficiency.py, operating on a
    full evidence set this adapter never sees."""
    import investment_research.research.clinicaltrials_acquisition_adapter as module

    assert "evidence_sufficiency" not in module.__dict__
    assert not hasattr(module, "domain_is_sufficient")


def test_registry_can_confirm_status_design_and_endpoint_wording_only():
    """What a successful fetch DOES let a caller confirm -- registered
    status, phase/design, and the verbatim endpoint wording -- versus what
    it never can: FDA acceptance of that endpoint, efficacy, peer review,
    or independent regulatory confirmation of anything the sponsor
    registered (Phase 3D.2 requirement 3)."""
    parsed = parse_study(json.loads(fx.fixture_text("study_recruiting_interventional.json")))
    # Confirmable from the registry alone:
    assert parsed["status"] != UNKNOWN
    assert parsed["phase"] != UNKNOWN
    assert parsed["allocation"] != UNKNOWN
    assert parsed["primary_endpoint"] != UNKNOWN
    # Never present anywhere in the parsed record or in the RawFact claims:
    forbidden_substrings = (
        "fda", "regulator_agree", "regulator agrees", "endpoint accepted",
        "accepted by", "peer review", "peer-reviewed", "independently confirmed",
    )
    parsed_text = " ".join(str(v).lower() for v in parsed.values() if isinstance(v, (str, int, float, bool)))
    for forbidden in forbidden_substrings:
        assert forbidden not in parsed_text

    raw_facts = raw_facts_from_study(
        "TEST", parsed, Source(source_id="s1", url=fx.study_url(fx.NCT_ID), title="t", tier=SourceTier.TIER_1),
    )
    claims_text = " ".join(f.claim.lower() for f in raw_facts)
    for forbidden in forbidden_substrings:
        assert forbidden not in claims_text


# --- Phase 3D.2: physical vs. logical request classification (regression) --
def test_direct_api_requests_reflects_the_actual_direct_api_call():
    """Regression for the bug the real Mac run surfaced: FETCH's own
    acquisition_method is EXISTING_DIRECT_API (CT.gov's API v2 single-study
    endpoint IS the direct/structured API), so its one physical GET must be
    counted via direct_api_requests, never direct_http_requests -- the
    original bug reported the exact opposite split (0/1 instead of 1/0)."""
    http = FakeHttpClient(responses=fx.default_responses())
    report, store, target = _run(fx.NCT_ID, http)
    diag = report.diagnostics
    assert diag.direct_api_requests == 1
    assert diag.direct_http_requests == 0
    assert diag.direct_api_requests + diag.direct_http_requests == len(http.requested_urls)
