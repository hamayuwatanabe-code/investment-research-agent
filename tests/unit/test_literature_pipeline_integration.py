"""Phase 4.2A: Literature Pipeline Integration.

Offline end-to-end: AcquisitionExecutor -> PubMedLiteratureAdapter ->
DocumentStore -> Literature Evidence Projection -> Literature Chunk
Projection -> CollectionResult/canonical chunks -> cli.run_one()/
Pipeline.run(), all driven against the SAME real-format fixtures/
FakeHttpClient double every other literature test in this repository
uses. No real NCBI/Europe PMC communication anywhere in this file.
"""

from __future__ import annotations

from datetime import date
from urllib.parse import urlencode

import pytest

from investment_research.agents.blind_judge import BlindJudgeAgent
from investment_research.agents.llm_agents import LLMRegulatoryAgent
from investment_research.agents.llm_agents2 import LLMBlindJudgeAgent
from investment_research.agents.regulatory import RegulatoryAgent
from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.llm.client import LLMBudget, LLMClient
from investment_research.orchestrator.isolation import policy_for
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research import literature_live_smoke as live
from investment_research.research.escalation import EscalationReport
from investment_research.research.literature_acquisition_adapter import (
    NCBI_EFETCH_URL,
    NCBI_ESEARCH_URL,
    build_single_target_literature_graph,
)
from investment_research.research.literature_pipeline_integration import (
    DEFAULT_MAX_ARTICLES,
    DEFAULT_MAX_FULLTEXT_FETCHES,
    TARGET_ID,
    TARGETED_MAX_FULLTEXT_FETCHES_CAP,
    LiteraturePipelineRequest,
    LiteraturePipelineRequestError,
    LiteratureReferenceMode,
    bundle_diagnostics,
    run_literature_pipeline_acquisition,
    validate_literature_pipeline_request,
)
from investment_research.research.provider import DomainCoverage, NullResearchProvider
from investment_research.research.source_routing import ImplementationStatus
from investment_research.schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    FetchOutcome,
    ResearchStatus,
    RunStatus,
    SearchStatus,
)
from investment_research.scoring.completeness import CompletenessResult
from investment_research.scoring.evidence_sufficiency import EvidenceSufficiencyMatrix

from . import _literature_fixture_support as fx
from ._network_guard import forbid_external_network_autouse  # noqa: F401

TODAY = date(2026, 9, 6)
_EPMC_EMPTY = '{"resultList": {"result": []}}'
_ENV = {"IRA_NCBI_TOOL": "test-tool", "IRA_NCBI_EMAIL": "test@example.com"}


def _efetch_url(pmids: list[str], *, tool: str = "test-tool", email: str = "test@example.com") -> str:
    """Mirrors ``literature_acquisition_adapter._eutils_params``'s exact
    param-insertion order (db/id/retmode/rettype, then tool, then email) --
    this module's own PubMedLiteratureAdapter is constructed WITH real
    ``NcbiCredentials`` (never ``settings=None``), unlike Phase 4.1A/4.1B's
    own fixture tests, so the wire URL genuinely carries these courtesy
    params and the FakeHttpClient response registered here must match it
    exactly. Never used for Europe PMC (which never carries them)."""
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email
    return f"{NCBI_EFETCH_URL}?{urlencode(params)}"


def _esearch_url(
    term: str, max_pmids: int, *, tool: str = "test-tool", email: str = "test@example.com"
) -> str:
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids)}
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email
    return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"


# =============================================================================
# 1. CLI input validation rules (task section 2)
# =============================================================================
def test_flag_off_no_identifiers_returns_none_the_pre_existing_default():
    req = validate_literature_pipeline_request(
        enabled=False, pmid=None, nct_id=None, live=False, use_fixtures=True, use_corpus=False,
        env=_ENV,
    )
    assert req is None


def test_identifier_without_flag_is_rejected_not_silently_ignored():
    with pytest.raises(LiteraturePipelineRequestError, match="without --document-first-literature"):
        validate_literature_pipeline_request(
            enabled=False, pmid="90000001", nct_id=None, live=True, use_fixtures=False,
            use_corpus=False, env=_ENV,
        )


def test_flag_on_no_identifier_is_rejected():
    with pytest.raises(LiteraturePipelineRequestError, match="requires exactly one"):
        validate_literature_pipeline_request(
            enabled=True, pmid=None, nct_id=None, live=True, use_fixtures=False, use_corpus=False,
            env=_ENV,
        )


def test_both_identifiers_is_rejected():
    with pytest.raises(LiteraturePipelineRequestError, match="mutually exclusive"):
        validate_literature_pipeline_request(
            enabled=True, pmid="90000001", nct_id="NCT09990001", live=True, use_fixtures=False,
            use_corpus=False, env=_ENV,
        )


def test_malformed_pmid_is_rejected_pre_send():
    with pytest.raises(LiteraturePipelineRequestError, match="malformed PMID"):
        validate_literature_pipeline_request(
            enabled=True, pmid="not-a-pmid", nct_id=None, live=True, use_fixtures=False,
            use_corpus=False, env=_ENV,
        )


def test_malformed_nct_id_is_rejected_pre_send():
    with pytest.raises(LiteraturePipelineRequestError, match="malformed NCT ID"):
        validate_literature_pipeline_request(
            enabled=True, pmid=None, nct_id="NOT-AN-NCT-ID", live=True, use_fixtures=False,
            use_corpus=False, env=_ENV,
        )


def test_fixtures_combination_is_rejected():
    with pytest.raises(LiteraturePipelineRequestError, match="--fixtures/--corpus"):
        validate_literature_pipeline_request(
            enabled=True, pmid="90000001", nct_id=None, live=True, use_fixtures=True,
            use_corpus=False, env=_ENV,
        )


def test_corpus_combination_is_rejected():
    with pytest.raises(LiteraturePipelineRequestError, match="--fixtures/--corpus"):
        validate_literature_pipeline_request(
            enabled=True, pmid="90000001", nct_id=None, live=True, use_fixtures=False,
            use_corpus=True, env=_ENV,
        )


def test_missing_live_is_rejected_pre_send():
    with pytest.raises(LiteraturePipelineRequestError, match="requires --live"):
        validate_literature_pipeline_request(
            enabled=True, pmid="90000001", nct_id=None, live=False, use_fixtures=False,
            use_corpus=False, env=_ENV,
        )


def test_missing_credential_is_rejected_pre_send():
    with pytest.raises(LiteraturePipelineRequestError, match="IRA_NCBI_TOOL"):
        validate_literature_pipeline_request(
            enabled=True, pmid="90000001", nct_id=None, live=True, use_fixtures=False,
            use_corpus=False, env={},
        )


def test_missing_email_only_is_rejected_pre_send():
    with pytest.raises(LiteraturePipelineRequestError, match="IRA_NCBI_EMAIL"):
        validate_literature_pipeline_request(
            enabled=True, pmid="90000001", nct_id=None, live=True, use_fixtures=False,
            use_corpus=False, env={"IRA_NCBI_TOOL": "t"},
        )


def test_api_key_never_required():
    req = validate_literature_pipeline_request(
        enabled=True, pmid="90000001", nct_id=None, live=True, use_fixtures=False,
        use_corpus=False, env={"IRA_NCBI_TOOL": "t", "IRA_NCBI_EMAIL": "e@example.com"},
    )
    assert req is not None


def test_valid_pmid_request_is_accepted_and_normalized():
    req = validate_literature_pipeline_request(
        enabled=True, pmid=" 90000001 ", nct_id=None, live=True, use_fixtures=False,
        use_corpus=False, env=_ENV,
    )
    assert req == LiteraturePipelineRequest(
        reference_mode=LiteratureReferenceMode.PMID, pmid="90000001", nct_id=None,
        max_articles=DEFAULT_MAX_ARTICLES, max_fulltext_fetches=DEFAULT_MAX_FULLTEXT_FETCHES,
    )


def test_valid_nct_id_request_is_accepted_and_uppercased():
    req = validate_literature_pipeline_request(
        enabled=True, pmid=None, nct_id="nct09990001", live=True, use_fixtures=False,
        use_corpus=False, env=_ENV,
    )
    assert req is not None
    assert req.reference_mode == LiteratureReferenceMode.NCT_ID
    assert req.nct_id == "NCT09990001"


def test_targeted_pmid_max_fulltext_fetches_is_clamped_in_the_request_itself():
    """Phase 3F.1 correction's own point applied here too: what a caller
    displays and what actually executes can never silently disagree."""
    req = validate_literature_pipeline_request(
        enabled=True, pmid="90000001", nct_id=None, live=True, use_fixtures=False,
        use_corpus=False, max_fulltext_fetches=5, env=_ENV,
    )
    assert req.max_fulltext_fetches == TARGETED_MAX_FULLTEXT_FETCHES_CAP


def test_no_http_attempt_on_any_rejection():
    """Every LiteraturePipelineRequestError raise site fires before any
    network-capable object exists -- there is nothing to even count
    requests on. This is verified structurally: validate_literature_
    pipeline_request never imports/constructs DocumentStore, Acquisition
    Executor, or any HTTP client."""
    import investment_research.research.literature_pipeline_integration as mod

    src = mod.validate_literature_pipeline_request.__code__.co_names
    assert "AllowlistedHttpClient" not in src
    assert "DocumentStore" not in src
    assert "AcquisitionExecutor" not in src


# =============================================================================
# 2. Request budget formula parity with Literature Live Smoke (task section 6)
# =============================================================================
def test_targeted_formula_matches_live_smoke_numerically():
    import investment_research.research.literature_pipeline_integration as mod

    mine_limits, mine_total, mine_formula = mod._targeted_request_budget(1)
    live_limits, live_total, live_formula = live._targeted_budget(1)
    assert mine_total == live_total == 3
    assert mine_limits == live_limits


def test_discovery_formula_matches_live_smoke_numerically():
    import investment_research.research.literature_pipeline_integration as mod

    mine_limits, mine_total, _ = mod._discovery_request_budget(3, 1)
    live_limits, live_total, _ = live._discovery_budget(3, 1)
    assert mine_total == live_total == 6
    assert mine_limits == live_limits


def test_targeted_request_cap_is_three_with_defaults():
    import investment_research.research.literature_pipeline_integration as mod

    _limits, total, _formula = mod._targeted_request_budget(DEFAULT_MAX_FULLTEXT_FETCHES)
    assert total == 3  # EFetch(1) + EuropePMC search(1) + fulltext(<=1)


def test_discovery_request_cap_is_six_with_defaults():
    import investment_research.research.literature_pipeline_integration as mod

    _limits, total, _formula = mod._discovery_request_budget(
        DEFAULT_MAX_ARTICLES, DEFAULT_MAX_FULLTEXT_FETCHES
    )
    assert total == 6  # ESearch(1)+EFetch(1)+EuropePMC search(3)+fulltext(1)


def test_targeted_fulltext_cap_never_exceeds_one_regardless_of_request():
    import investment_research.research.literature_pipeline_integration as mod

    _limits, total, _formula = mod._targeted_request_budget(99)
    assert total == 3


# =============================================================================
# 3. Shared single-target graph (task section 5)
# =============================================================================
def test_pipeline_integration_and_live_smoke_use_the_same_graph_builder():
    graph_a = build_single_target_literature_graph("target_a")
    graph_b = build_single_target_literature_graph("target_a")
    assert graph_a.steps == graph_b.steps
    assert [s.step_id for s in graph_a.steps] == ["l1", "f", "p"]


def test_single_target_graph_is_never_the_31_group_master_catalog():
    from investment_research.research.source_routing_catalog import (
        build_source_routing_graph,
        routing_coverage_counts,
    )

    master = build_source_routing_graph()
    mine = build_single_target_literature_graph(TARGET_ID)
    assert mine is not master
    assert len(mine.targets) == 1
    # routing_coverage_counts() defaults to the MASTER graph -- passing mine
    # explicitly proves it is never silently substituted for it either.
    counts_master = routing_coverage_counts()
    counts_mine = routing_coverage_counts(mine)
    assert counts_master.legacy_needs == counts_mine.legacy_needs  # legacy_needs is catalog-global
    assert counts_master.evidence_requirements != counts_mine.evidence_requirements


def test_single_target_graph_steps_are_offline_verified_never_live_verified():
    graph = build_single_target_literature_graph(TARGET_ID)
    for step in graph.steps:
        assert step.implementation_status is ImplementationStatus.OFFLINE_VERIFIED
        assert step.implementation_status is not ImplementationStatus.LIVE_VERIFIED
        # OFFLINE_VERIFIED's own rank already implies PIPELINE_WIRED -- see
        # build_single_target_literature_graph's own docstring.
        assert step.implementation_status.is_pipeline_wired


# =============================================================================
# 4. Offline E2E acquisition (task section 12)
# =============================================================================
def _pmid_request(pmid="90000001", **overrides) -> LiteraturePipelineRequest:
    kwargs = {'reference_mode': LiteratureReferenceMode.PMID, 'pmid': pmid, 'nct_id': None,
                   'max_articles': DEFAULT_MAX_ARTICLES, 'max_fulltext_fetches': DEFAULT_MAX_FULLTEXT_FETCHES}
    kwargs.update(overrides)
    return LiteraturePipelineRequest(**kwargs)


def _nct_request(nct_id="NCT09990001", **overrides) -> LiteraturePipelineRequest:
    kwargs = {'reference_mode': LiteratureReferenceMode.NCT_ID, 'pmid': None, 'nct_id': nct_id,
                   'max_articles': DEFAULT_MAX_ARTICLES, 'max_fulltext_fetches': DEFAULT_MAX_FULLTEXT_FETCHES}
    kwargs.update(overrides)
    return LiteraturePipelineRequest(**kwargs)


def test_targeted_pmid_success():
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.refused_reason is None
    assert bundle.coverage_complete
    assert bundle.raw_fact_count > 0
    assert bundle.document_count == 1
    assert bundle.chunk_count > 0
    assert bundle.current_program_confirmed is False
    assert bundle.target_outcome == "ACQUIRED"


def test_nct_discovery_success_via_pubmed_batch_efetch():
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(responses={
        _esearch_url(term, 20): fx.ok(fx.esearch_response(["90000015", "90000016"])),
        _efetch_url(["90000015", "90000016"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
        fx.europepmc_search_url("90000016"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(_nct_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.refused_reason is None
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1  # one batched EFetch for both PMIDs
    assert bundle.document_count >= 2  # both PubMed documents projected


def test_europepmc_oa_fulltext_chunked():
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
        fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
        fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000008"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.document_count == 3  # pubmed + search + fulltext
    assert bundle.excluded_non_primary_document_count == 1  # the search doc
    assert bundle.chunk_count > 0


def test_non_open_access_no_fulltext_fetch_attempted():
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
    })
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert not any("fullTextXML" in u for u in http.requested_urls)
    assert bundle.coverage_complete


def test_no_abstract_projects_metadata_only_never_chunked():
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000003"]): fx.ok(fx.fixture_text("no_abstract.xml")),
        fx.europepmc_search_url("90000003"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000003"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.chunk_count == 0
    assert bundle.excluded_non_primary_document_count == 1


def test_genuine_zero_results_is_clean_not_degraded():
    term = "NCT09990099[si]"
    http = fx.FakeHttpClient(responses={_esearch_url(term, 20): fx.ok(fx.esearch_response([]))})
    bundle = run_literature_pipeline_acquisition(
        _nct_request(nct_id="NCT09990099"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.coverage_complete
    assert bundle.collection_result.zero_results is True
    assert not bundle.collection_result.degraded


@pytest.mark.parametrize("outcome_name", ["RATE_LIMITED", "ERROR", "TIMEOUT"])
def test_provider_failure_never_reported_as_zero_results(outcome_name):
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(
        responses={_esearch_url(term, 20): fx.failed(FetchOutcome[outcome_name], outcome_name.lower())}
    )
    bundle = run_literature_pipeline_acquisition(_nct_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.collection_result.zero_results is False
    assert bundle.collection_result.degraded
    assert not bundle.coverage_complete


def test_404_never_treated_as_zero_results():
    http = fx.FakeHttpClient(responses={})  # nothing registered -> NOT_FOUND for every URL
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.collection_result.zero_results is False
    assert bundle.collection_result.degraded


def test_request_cap_matches_targeted_formula():
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
        fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
        fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000008"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.logical_request_count == 3  # EFetch + EuropePMC search + fulltext


def test_article_budget_exclusion_recorded_never_not_found():
    http = fx.FakeHttpClient(responses={
        _esearch_url("NCT09990001[si]", 20): fx.ok(fx.esearch_response(["90000001", "90000002"])),
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _nct_request(max_articles=1), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.coverage_complete is False
    assert bundle.collection_result.degraded
    assert any("budget" in r.lower() for r in bundle.unresolved_reasons)
    # Partial success preserved: PMID that WAS fetched is still projected.
    assert bundle.raw_fact_count > 0


def test_partial_success_preserves_acquired_facts():
    http = fx.FakeHttpClient(responses={
        _esearch_url("NCT09990001[si]", 20): fx.ok(fx.esearch_response(["90000001", "90000012"])),
        _efetch_url(["90000001", "90000012"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(_nct_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.raw_fact_count > 0  # PMID 90000001's facts kept despite any batch issue


def test_document_source_fact_chunk_dedup_zero_extra_http_for_repeat_run_scope():
    """A single request never chunks/dedupes across itself -- this test
    pins the WITHIN-one-call dedup guarantee inherited from Phase 4.1A/4.1B
    (never duplicated Documents/Facts/Chunks for one PMID)."""
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.duplicate_document_count == 0
    chunk_ids = [c.chunk_id for c in bundle.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_credential_missing_refused_before_any_request():
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
    })
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env={})
    assert bundle.refused
    assert "IRA_NCBI_TOOL" in bundle.refused_reason
    assert http.requested_urls == []
    assert bundle.logical_request_count == 0
    assert bundle.physical_attempt_count == 0


# =============================================================================
# 5. Diagnostics never carry a body/URL/secret (task sections 4/10)
# =============================================================================
def test_bundle_diagnostics_never_none_disabled_shape():
    """Phase 4.2A correction 1: an empty dict, not a dict carrying
    feature_enabled=False -- so cli.py::result_to_json can omit the
    direct_acquisition_info key entirely by plain truthiness for a run
    that never used --document-first-literature, keeping the default-OFF
    --json output contract identical to before Phase 4.2A."""
    diag = bundle_diagnostics(None, enabled=False)
    assert diag == {}


def test_bundle_diagnostics_are_json_serializable_and_body_free():
    import json

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
        fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
        fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000008"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    diag = bundle_diagnostics(bundle, enabled=True)
    blob = json.dumps(diag)
    assert "fictional full-text introduction" not in blob  # no body text
    assert "eutils.ncbi.nlm.nih.gov" not in blob  # no URL
    assert "test@example.com" not in blob  # no credential value
    assert diag["anthropic_api_calls"] == 0
    assert diag["web_search_calls"] == 0
    assert diag["external_llm_tokens"] == 0
    assert diag["current_program_confirmed"] is False


def test_bundle_never_carries_raw_execution_report_or_parsed_article_types():
    from investment_research.collectors.literature import ParsedPubmedArticle
    from investment_research.research.acquisition_executor import ExecutionReport

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    for value in vars(bundle).values():
        assert not isinstance(value, ExecutionReport)
        assert not isinstance(value, ParsedPubmedArticle)


# =============================================================================
# 6. Pipeline integration: production order, chunk/traceability flow,
#    Blind Judge isolation, Action=NONE propagation (task sections 7-9)
# =============================================================================
def _demobio_collection_result():
    return FixtureCollector.__new__(FixtureCollector)  # placeholder, replaced below


def test_acquisition_completes_before_pipeline_construction(repo, fixture_dir):
    """Production order (task section 7): the literature CollectionResult/
    chunks are fully built BEFORE Pipeline is even constructed -- proven
    here by driving the exact same sequence cli.run_one() uses, directly."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000015"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.collection_result is not None  # acquisition already complete here

    collection_results = [base_result, bundle.collection_result]
    chunks = list(bundle.chunks)

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, chunks=chunks)
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], collection_results, price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )
    # The literature Fact/Chunk made it all the way through FactCollector/
    # EvidenceIntegrity -- present in the bus BEFORE any domain-agent-only
    # artifact could have invented it.
    literature_document_ids = {c.doc_id for c in bundle.chunks}
    literature_facts = [f for f in result.bus.facts if f.document_id in literature_document_ids]
    assert literature_facts
    assert result.chunks == chunks
    assert any(c.doc_id for c in result.chunks)


def test_literature_chunks_reach_result_chunks_and_traceability(repo, fixture_dir):
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000015"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.chunks

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks))
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    assert {c.chunk_id for c in bundle.chunks} <= {c.chunk_id for c in result.chunks}
    # TraceabilityIndex resolves a literature Chunk -> Fact -> Source.
    lit_chunk = bundle.chunks[0]
    link = result.traceability.resolve(lit_chunk.chunk_id)
    assert link is not None
    assert link.fact_id in {f.fact_id for f in result.bus.facts}


def test_blind_judge_never_receives_raw_chunks_even_when_literature_chunks_present(repo):
    """Blind Judge isolation (task section 9): policy.sees_identity is
    False for blind_judge, and Pipeline._agent_for's existing chunk gate
    (`chunks = () if not policy.sees_identity else self.chunks`, unchanged
    by this phase) must exclude literature chunks exactly like every other
    chunk. Contrasted with a non-blind agent (regulatory), which DOES
    receive them -- proving this isn't merely "chunks are always empty"."""
    assert policy_for("blind_judge").sees_identity is False

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.chunks

    budget = LLMBudget(max_total_tokens=1_000_000)
    llm = LLMClient(api_key="sk-ant-test", base_url="http://127.0.0.1:1", effort="low", budget=budget)
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, llm=llm, chunks=list(bundle.chunks))
    pipeline._identity = ("DEMOBIO", "Demo Bio", ())

    class _EmptyBus:
        channels: dict = {}

    blind_agent = pipeline._agent_for("blind_judge", BlindJudgeAgent(), LLMBlindJudgeAgent, _EmptyBus())
    assert isinstance(blind_agent, LLMBlindJudgeAgent)
    assert blind_agent.chunks == []

    sighted_agent = pipeline._agent_for("regulatory", RegulatoryAgent(), LLMRegulatoryAgent, _EmptyBus())
    assert isinstance(sighted_agent, LLMRegulatoryAgent)
    assert len(sighted_agent.chunks) == len(bundle.chunks)


@pytest.mark.parametrize("outcome_name", ["BLOCKED", "RATE_LIMITED", "ERROR"])
def test_incomplete_literature_evidence_forces_action_none(repo, fixture_dir, outcome_name):
    """Action Gate strengthening (task section 8): an explicitly-requested
    Literature acquisition that genuinely failed must withhold the Action
    label -- never silently absorbed into a run that still emits one."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(
        responses={_esearch_url("NCT09990001[si]", 20): fx.failed(FetchOutcome[outcome_name], "x")}
    )
    bundle = run_literature_pipeline_acquisition(_nct_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.collection_result.degraded

    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    assert result.verdict is not None
    assert result.verdict.action is None
    assert result.verdict.research_status in (
        ResearchStatus.BLOCKED_PENDING_VERIFICATION, ResearchStatus.INCOMPLETE,
    )
    assert any("Literature Document-First" in r for r in result.verdict.blocking_verification_required)


def test_clean_literature_evidence_never_blocks_action_on_its_own(repo, fixture_dir):
    """The strengthening never fires for a genuinely complete fetch -- this
    is the 'never weakens, and never over-blocks either' half of the
    guarantee."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000015"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert not bundle.collection_result.degraded
    assert bundle.coverage_complete

    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    literature_reasons = [
        r for r in (result.verdict.blocking_verification_required if result.verdict else ())
        if "Literature Document-First" in r
    ]
    assert literature_reasons == []


# =============================================================================
# 6b. Correction 1: Action=NONE propagation for the two failure shapes a
#    CollectionResult.degraded-only check misses (task section 1)
# =============================================================================
def test_chunk_projection_failure_alone_forces_action_none(repo, fixture_dir):
    """Scenario A: Evidence Projection succeeded
    (CollectionResult.degraded=False) but Chunk Projection itself failed --
    coverage_complete=False is the ONLY signal; the CollectionResult-loop
    branch finds nothing degraded. Must still force Action=None."""
    import dataclasses

    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000015"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert not bundle.collection_result.degraded
    assert bundle.coverage_complete  # genuinely clean before simulating the Chunk Projection failure

    # Simulates exactly what run_literature_pipeline_acquisition itself
    # would produce if project_literature_chunks alone returned
    # coverage_complete=False (e.g. a corrupted version chain discovered
    # after Evidence Projection already succeeded) -- Evidence
    # Projection's own, already-clean CollectionResult is untouched.
    incomplete_bundle = dataclasses.replace(
        bundle, coverage_complete=False,
        unresolved_reasons=("document doc_x: version chain corrupted: simulated for this test",),
    )
    assert not incomplete_bundle.collection_result.degraded  # unchanged by the simulation

    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, chunks=list(incomplete_bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(incomplete_bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, incomplete_bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    assert result.verdict is not None
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert any("Literature Document-First" in r for r in result.verdict.blocking_verification_required)
    assert any("coverage" in r.lower() for r in result.verdict.blocking_verification_required)


def test_refused_bundle_forces_action_none_with_zero_http_attempts(repo, fixture_dir):
    """Scenario B: the defensive credential re-check inside
    run_literature_pipeline_acquisition refused before any request --
    collection_result is None (nothing is ever appended to
    collection_results for this case, mirroring cli.py's own
    `if literature_bundle.collection_result is not None` guard), so only
    direct_acquisition_info's refused=True signal exists."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={})
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(), ticker="DEMOBIO", http_client=http, env={}
    )
    assert bundle.refused
    assert bundle.collection_result is None
    assert http.requested_urls == []
    assert bundle.physical_attempt_count == 0
    assert bundle.logical_request_count == 0

    collection_results = [base_result]
    if bundle.collection_result is not None:  # never true here; mirrors cli.py's own guard exactly
        collection_results.append(bundle.collection_result)

    diagnostics = bundle_diagnostics(bundle, enabled=True)
    assert diagnostics["refused"] is True
    assert "IRA_NCBI_TOOL" in diagnostics["refused_reason"]

    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks),
        direct_acquisition_info=diagnostics,
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], collection_results,
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    assert result.verdict is not None
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    reasons = result.verdict.blocking_verification_required
    assert any("Literature Document-First" in r for r in reasons)
    assert any("refused" in r.lower() for r in reasons)
    assert any("IRA_NCBI_TOOL" in r for r in reasons)


def test_degraded_and_incomplete_coverage_do_not_duplicate_the_blocking_reason(repo, fixture_dir):
    """Both signals present simultaneously (a fully-failed acquisition:
    Evidence Projection degraded AND Chunk Projection incomplete) must
    still contribute exactly ONE Literature blocking reason, never two."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(
        responses={_esearch_url("NCT09990001[si]", 20): fx.failed(FetchOutcome.ERROR, "x")}
    )
    bundle = run_literature_pipeline_acquisition(_nct_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.collection_result.degraded
    assert bundle.coverage_complete is False  # both signals present at once

    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    literature_reasons = [
        r for r in result.verdict.blocking_verification_required if "Literature Document-First" in r
    ]
    assert len(literature_reasons) == 1  # never duplicated


def test_literature_blocking_helper_never_fires_when_flag_unused():
    from investment_research.orchestrator.pipeline import _literature_incomplete_blocking_reasons

    assert _literature_incomplete_blocking_reasons([], None) == []
    assert _literature_incomplete_blocking_reasons([], {}) == []
    assert _literature_incomplete_blocking_reasons(
        [], {"feature_enabled": False, "coverage_complete": False, "refused": True}
    ) == []


def test_literature_blocking_helper_clean_bundle_produces_nothing():
    from investment_research.orchestrator.pipeline import _literature_incomplete_blocking_reasons

    info = {
        "feature_enabled": True, "refused": False, "coverage_complete": True,
        "unresolved_reasons": [],
    }
    assert _literature_incomplete_blocking_reasons([], info) == []


def test_literature_blocking_helper_scenario_a_coverage_incomplete_unit():
    from investment_research.orchestrator.pipeline import _literature_incomplete_blocking_reasons

    info = {
        "feature_enabled": True, "refused": False, "coverage_complete": False,
        "unresolved_reasons": ["document doc_x: version chain corrupted"],
    }
    reasons = _literature_incomplete_blocking_reasons([], info)
    assert len(reasons) == 1
    assert "coverage" in reasons[0].lower()


def test_literature_blocking_helper_scenario_b_refused_unit():
    from investment_research.orchestrator.pipeline import _literature_incomplete_blocking_reasons

    info = {
        "feature_enabled": True, "refused": True,
        "refused_reason": "missing required environment variable(s): IRA_NCBI_TOOL",
    }
    reasons = _literature_incomplete_blocking_reasons([], info)
    assert len(reasons) == 1
    assert "IRA_NCBI_TOOL" in reasons[0]


def test_literature_blocking_helper_dedup_unit():
    from investment_research.collectors.base import CollectionResult
    from investment_research.orchestrator.pipeline import _literature_incomplete_blocking_reasons

    degraded_result = CollectionResult(collector="literature_evidence_projection", outcome=FetchOutcome.ERROR)
    info = {
        "feature_enabled": True, "refused": False, "coverage_complete": False,
        "unresolved_reasons": ["x"],
    }
    reasons = _literature_incomplete_blocking_reasons([degraded_result], info)
    assert len(reasons) == 1


# =============================================================================
# 6c. Correction 1: default-OFF --json output contract (task section 3)
# =============================================================================
def test_result_to_json_omits_direct_acquisition_info_key_when_flag_off(repo, fixture_dir):
    import investment_research.cli as cli_module

    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result], price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )
    payload = cli_module.result_to_json(result)
    assert "direct_acquisition_info" not in payload


def test_result_to_json_includes_direct_acquisition_info_key_when_flag_on(repo, fixture_dir):
    import investment_research.cli as cli_module

    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000015"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    payload = cli_module.result_to_json(result)
    assert "direct_acquisition_info" in payload
    assert payload["direct_acquisition_info"]["feature_enabled"] is True


class _AvailableNullResearchProvider(NullResearchProvider):
    """Reports ``available()=True`` (unlike ``NullResearchProvider``) but
    keeps the same safe no-op ``search()``/``fetch()`` -- used only so the
    escalation stage does not add its own "skipped" noise to
    ``result.failures`` in the non-vacuous tests below.
    ``pipeline_module.escalate`` is ALSO faked (see
    ``_install_noise_free_pipeline_harness``), so this provider's
    ``search``/``fetch`` are never actually invoked either."""

    def available(self):
        return True, ""


def _install_noise_free_pipeline_harness(monkeypatch) -> None:
    """Phase 4.2A correction 2: neutralizes every KNOWN source of
    Literature-UNRELATED ``result.failures``/``ctx.status`` noise in a
    real ``Pipeline.run()`` call, so a test can prove a status/action
    transition is caused BY Literature specifically -- never merely
    coincide with noise the DEMOBIO fixture already produces on its own
    (competitive/microstructure/kill_agent DEGRADED from sparse fixture
    data; an "escalation skipped" message from no research provider being
    configured; the Search Completeness Gate blocking on domains this
    fixture's structured collectors alone never satisfy).

    Verified empirically before being written into this suite (see this
    correction's own commit message): WITHOUT this harness, temporarily
    disabling Correction 2's own fix and re-running the exact Scenario A/B
    setups below still produced ``status=INCOMPLETE_RESEARCH`` --
    vacuously, via unrelated DEMOBIO noise, not via the fix -- exactly the
    trap an audit warned about. WITH this harness and Correction 2
    disabled, the SAME setups instead reproduced the audit's reported
    contradiction exactly (``status=COMPLETE``,
    ``research_status=BLOCKED_PENDING_VERIFICATION``, ``action=None``,
    ``result.incomplete=False``) -- proving this harness is what makes the
    tests below non-vacuous, and that Correction 2 is what fixes that
    contradiction. See ``test_noise_free_harness_reaches_genuine_complete_
    baseline`` for the harness's own self-check (no Literature at all ->
    a genuine, failure-free COMPLETE run).

    Every patch here targets ONLY test noise unrelated to Literature's own
    evidence -- never touches Literature-specific code, and never widens
    what the real completeness/sufficiency gates would accept in
    production (production callers never monkeypatch anything)."""
    import investment_research.orchestrator.pipeline as pipeline_module

    def fake_assess_completeness(**_kwargs):
        return CompletenessResult(
            coverage={
                domain: DomainCoverage(domain=domain, status=SearchStatus.SEARCHED)
                for domain in REQUIRED_RESEARCH_DOMAINS
            }
        )

    def fake_assess_evidence_sufficiency(**_kwargs):
        return EvidenceSufficiencyMatrix(decision_grade_fact_total=1)

    def fake_escalate(facts, _provider, *, company, **_kwargs):
        return list(facts), EscalationReport()

    monkeypatch.setattr(pipeline_module, "assess_completeness", fake_assess_completeness)
    monkeypatch.setattr(
        pipeline_module, "assess_evidence_sufficiency", fake_assess_evidence_sufficiency
    )
    monkeypatch.setattr(pipeline_module, "escalate", fake_escalate)

    real_run_agent = Pipeline._run_agent
    noisy_agent_ids = {"competitive", "microstructure", "kill_agent"}

    def patched_run_agent(self, agent, guard, result, **kwargs):
        # Snapshot/restore rather than skip the real call: every agent's
        # REAL facts/evaluation still publish to the bus normally -- only
        # the generic degraded/failed bookkeeping (_run_agent's own
        # result.failures.append/ctx.status=INCOMPLETE_RESEARCH for a hard
        # failure) is undone, and only for these three agents, whose
        # degradation against this fixture is about missing competitor/
        # microstructure/research-provider data -- never about Literature.
        before_status = result.context.status
        before_len = len(result.failures)
        output = real_run_agent(self, agent, guard, result, **kwargs)
        if agent.agent_id in noisy_agent_ids:
            del result.failures[before_len:]
            result.context.status = before_status
        return output

    monkeypatch.setattr(Pipeline, "_run_agent", patched_run_agent)


def _noise_free_pipeline(repo, monkeypatch, **kwargs) -> Pipeline:
    _install_noise_free_pipeline_harness(monkeypatch)
    return Pipeline(
        repo, NullSearchProvider(), today=TODAY, research=_AvailableNullResearchProvider(), **kwargs
    )


# =============================================================================
# 6d. Correction 2: RunStatus/result.failures propagation -- non-vacuous
#    proof via a noise-free Pipeline.run() baseline (task sections 1-2)
# =============================================================================
def test_noise_free_harness_reaches_genuine_complete_baseline(monkeypatch, repo, fixture_dir):
    """Validates the harness ITSELF: with no Literature involved at all,
    a real Pipeline.run() over DEMOBIO reaches a genuinely COMPLETE,
    failure-free state -- the necessary precondition for every non-vacuous
    claim below (otherwise unrelated DEMOBIO noise could produce the same
    final status coincidentally, exactly the trap an audit flagged)."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])
    pipeline = _noise_free_pipeline(repo, monkeypatch)
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result], price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )
    assert result.context.status is RunStatus.COMPLETE
    assert result.failures == []
    assert result.incomplete is False
    assert result.verdict.research_status is ResearchStatus.COMPLETE
    assert result.verdict.action is not None  # a real Action was actually reached


def test_scenario_a_coverage_incomplete_alone_forces_incomplete_research(monkeypatch, repo, fixture_dir):
    """Scenario A, non-vacuous: starting from the harness's own proven-
    COMPLETE baseline, a Chunk-Projection-only failure
    (CollectionResult.degraded=False, coverage_complete=False) is the ONLY
    thing that differs -- and it alone flips every one of these."""
    import dataclasses

    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000015"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert not bundle.collection_result.degraded
    assert bundle.coverage_complete
    incomplete_bundle = dataclasses.replace(
        bundle, coverage_complete=False,
        unresolved_reasons=("document doc_x: version chain corrupted: simulated for this test",),
    )

    pipeline = _noise_free_pipeline(
        repo, monkeypatch, chunks=list(incomplete_bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(incomplete_bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, incomplete_bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )

    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.run_status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.verdict.action is None
    assert result.incomplete is True
    # Non-vacuous: failures contains ONLY the Literature-attributed entry
    # -- proving nothing else in this run contributed to the transition.
    assert len(result.failures) == 1
    assert "Literature Document-First" in result.failures[0]
    assert "coverage" in result.failures[0].lower()


def test_scenario_b_refused_alone_forces_incomplete_research(monkeypatch, repo, fixture_dir):
    """Scenario B, non-vacuous: the defensive credential re-check refused
    before any request -- collection_result is None, so ONLY direct_
    acquisition_info's refused=True signal exists. Starting from the same
    proven-COMPLETE baseline, this alone flips every one of these."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={})
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env={})
    assert bundle.refused
    assert bundle.collection_result is None
    assert http.requested_urls == []
    assert bundle.physical_attempt_count == 0

    pipeline = _noise_free_pipeline(
        repo, monkeypatch, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result],  # no literature CollectionResult at all
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )

    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.run_status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.verdict.action is None
    assert result.incomplete is True
    assert len(result.failures) == 1
    assert "Literature Document-First" in result.failures[0]
    assert "refused" in result.failures[0].lower()
    assert "IRA_NCBI_TOOL" in result.failures[0]


def test_degraded_and_incomplete_coverage_do_not_duplicate_failures(monkeypatch, repo, fixture_dir):
    """Both signals present simultaneously (a fully-failed acquisition:
    Evidence Projection degraded AND Chunk Projection incomplete) must
    still contribute exactly ONE result.failures entry, never two --
    proven against the same noise-free baseline."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(
        responses={_esearch_url("NCT09990001[si]", 20): fx.failed(FetchOutcome.ERROR, "x")}
    )
    bundle = run_literature_pipeline_acquisition(_nct_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.collection_result.degraded
    assert bundle.coverage_complete is False

    pipeline = _noise_free_pipeline(
        repo, monkeypatch, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    literature_failures = [f for f in result.failures if "Literature Document-First" in f]
    assert len(literature_failures) == 1
    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.run_status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.action is None


def test_flag_off_leaves_status_and_failures_unchanged_non_vacuous(monkeypatch, repo, fixture_dir):
    """Flag unused: direct_acquisition_info is empty -- no Literature-
    attributed change to ctx.status/result.failures/verdict at all,
    verified against the same proven-COMPLETE baseline (not merely "ended
    up INCOMPLETE for some other reason, so nothing could be observed")."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    pipeline = _noise_free_pipeline(repo, monkeypatch)  # no chunks, no direct_acquisition_info
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result], price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )
    assert result.context.status is RunStatus.COMPLETE
    assert result.failures == []
    assert result.incomplete is False
    assert result.verdict.research_status is ResearchStatus.COMPLETE
    assert result.verdict.action is not None


def test_clean_acquisition_leaves_status_and_failures_unchanged_non_vacuous(monkeypatch, repo, fixture_dir):
    """A genuinely complete, non-degraded Literature fetch never
    contributes a status/failures/verdict change on its own -- the 'never
    over-blocks' half of the guarantee, proven against the same
    proven-COMPLETE baseline."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000015"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert not bundle.collection_result.degraded
    assert bundle.coverage_complete

    pipeline = _noise_free_pipeline(
        repo, monkeypatch, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    assert result.context.status is RunStatus.COMPLETE
    assert result.failures == []
    assert result.incomplete is False
    assert result.verdict.research_status is ResearchStatus.COMPLETE
    assert result.verdict.action is not None


def test_result_to_json_never_shows_complete_status_with_blocked_research_status(
    monkeypatch, repo, fixture_dir
):
    """The exact contradiction an audit found (status=COMPLETE,
    research_status=BLOCKED_PENDING_VERIFICATION, action=None) must never
    appear in --json output for a Literature-incomplete run."""
    import investment_research.cli as cli_module

    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={})
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env={})
    assert bundle.refused

    pipeline = _noise_free_pipeline(
        repo, monkeypatch, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result], price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )
    payload = cli_module.result_to_json(result)
    assert payload["research_status"] == str(ResearchStatus.BLOCKED_PENDING_VERIFICATION)
    assert payload["status"] == str(RunStatus.INCOMPLETE_RESEARCH)
    assert payload["action"] is None
    # The specific contradiction: never COMPLETE status with a BLOCKED research_status.
    assert not (payload["status"] == str(RunStatus.COMPLETE) and "BLOCKED" in payload["research_status"])


def test_pipeline_never_calls_literature_acquisition_code_directly():
    """pipeline.py's own blocking-propagation check (Phase 4.2A) reads
    CollectionResult.collector/.degraded/.describe() and the plain dict
    self.direct_acquisition_info (Phase 4.2A correction 1) -- it must never
    import literature_evidence_projection/literature_chunk_projection/
    literature_pipeline_integration, or call any acquisition function
    itself, whatever a run's collection_results happen to contain."""
    import investment_research.orchestrator.pipeline as pipeline_module

    assert "literature_evidence_projection" not in pipeline_module.__dict__
    assert "literature_chunk_projection" not in pipeline_module.__dict__
    assert "literature_pipeline_integration" not in pipeline_module.__dict__
    assert "project_literature_target_reports" not in pipeline_module.__dict__
    assert "project_literature_chunks" not in pipeline_module.__dict__
    assert "run_literature_pipeline_acquisition" not in pipeline_module.__dict__


def test_literature_bridge_collector_label_constant_stays_in_sync():
    from investment_research.orchestrator.pipeline import _LITERATURE_BRIDGE_COLLECTOR_LABEL
    from investment_research.research.literature_evidence_projection import BRIDGE_COLLECTOR_LABEL

    assert _LITERATURE_BRIDGE_COLLECTOR_LABEL == BRIDGE_COLLECTOR_LABEL


# =============================================================================
# 7. cli.py wiring: flag OFF preserves observable output/behavior (facts,
#    chunks, verdict, report, --json keys) -- NOT the Python import graph
#    or object structure, which do differ (see literature_pipeline_
#    integration.py's own module docstring); flag ON follows production
#    order (task sections 2/7)
# =============================================================================
def test_flag_off_run_one_makes_zero_literature_calls(monkeypatch, repo, fixture_dir, tmp_path):
    import argparse

    import investment_research.cli as cli_module

    calls = []
    monkeypatch.setattr(
        cli_module,
        "run_literature_pipeline_acquisition",
        lambda *a, **k: calls.append((a, k)) or (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    settings = _fake_settings(fixture_dir, tmp_path)
    args = argparse.Namespace(
        fixtures=True, corpus=False, live=False, llm=False, token_budget=500_000, adversarial=False,
        full_dd=False, kill_test=False, catalyst=False, update=False, price=None, company_name=None,
        portfolio=None, resume=None,
    )
    result = cli_module.run_one("DEMOBIO", args, settings, repo, http=None, literature_request=None)
    assert calls == []
    assert result.direct_acquisition_info == {}


def _fake_settings(fixture_dir, tmp_path):
    from investment_research.config import Settings

    return Settings(
        fixture_dir=fixture_dir, corpus_dir=tmp_path, db_path=tmp_path / "x.db",
        cache_dir=tmp_path, log_dir=tmp_path,
    )


def test_flag_on_run_one_runs_literature_after_structured_collectors_before_adversarial(
    monkeypatch, repo, fixture_dir, tmp_path
):
    import argparse

    import investment_research.cli as cli_module

    order: list[str] = []

    def fake_collect(*_a, **_k):
        order.append("structured_collectors")
        fixtures = FixtureCollector(fixture_dir)
        metadata = fixtures.metadata("DEMOBIO")
        return [fixtures.collect("DEMOBIO", metadata["company_name"])], metadata

    def fake_literature(request, *, ticker, **_kw):
        order.append("literature_acquisition")
        http = fx.FakeHttpClient(responses={
            _efetch_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
            fx.europepmc_search_url("90000015"): fx.ok(_EPMC_EMPTY),
        })
        return run_literature_pipeline_acquisition(request, ticker=ticker, http_client=http, env=_ENV)

    monkeypatch.setattr(cli_module, "collect", fake_collect)
    monkeypatch.setattr(cli_module, "run_literature_pipeline_acquisition", fake_literature)

    settings = _fake_settings(fixture_dir, tmp_path)
    args = argparse.Namespace(
        fixtures=False, corpus=False, live=True, llm=False, token_budget=500_000, adversarial=False,
        full_dd=False, kill_test=False, catalyst=False, update=False, price=None, company_name=None,
        portfolio=None, resume=None,
    )
    request = _pmid_request(pmid="90000015")
    result = cli_module.run_one("DEMOBIO", args, settings, repo, http=None, literature_request=request)

    assert order == ["structured_collectors", "literature_acquisition"]
    assert len(result.collection_results) == 2  # structured collector + literature bridge
    assert any(c.collector == "literature_evidence_projection" for c in result.collection_results)
    assert result.chunks  # literature chunks reached Pipeline
    assert result.direct_acquisition_info["feature_enabled"] is True
    assert result.direct_acquisition_info["reference_mode"] == "PMID"


# =============================================================================
# 8. Catalog / production non-connection (task sections 11/13)
# =============================================================================
def test_catalog_invariants_unchanged():
    from investment_research.research.source_routing_catalog import routing_coverage_counts

    counts = routing_coverage_counts()
    assert counts.offline_verified_steps == 74
    assert counts.live_verified_steps == 9
    assert counts.legacy_needs == 49
    assert counts.equivalence_groups == 31


def test_kill_rules_count_unchanged():
    from investment_research.scoring.kill_gate import KILL_RULES

    assert len(KILL_RULES) == 19


def test_sec_clinicaltrials_form4_new_adapters_never_imported_by_cli():
    import investment_research.cli as cli_module

    for name in (
        "sec_acquisition_adapters", "clinicaltrials_acquisition_adapter", "form4_acquisition_adapter",
    ):
        assert name not in cli_module.__dict__


def test_web_search_and_master_registry_not_introduced():
    import investment_research.research.literature_pipeline_integration as mod

    assert not hasattr(mod, "WebSearchProvider")
    assert not hasattr(mod, "MasterRegistry")


def test_live_smoke_marker_and_capture_untouched_by_default():
    """run_literature_pipeline_acquisition never writes a Live Smoke marker
    (it never imports literature_live_smoke at all) and never writes a
    Capture Manifest unless a caller explicitly passes out_dir (which
    cli.py never does)."""
    import investment_research.research.literature_pipeline_integration as mod

    assert "literature_live_smoke" not in mod.__dict__


def test_anthropic_web_search_llm_always_zero():
    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        fx.europepmc_search_url("90000001"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(_pmid_request(), ticker="DEMOBIO", http_client=http, env=_ENV)
    assert bundle.anthropic_api_calls == 0
    assert bundle.web_search_calls == 0
    assert bundle.external_llm_tokens == 0


# =============================================================================
# 9. Phase 4.2B correction 1: PubMed date normalization + downstream
#    completeness consistency, reproducing a real Production E2E Live run's
#    finding (a PubDate with a 3-letter English month abbreviation and a
#    day, e.g. "2021-Feb-04", rejected by Source validation -- see
#    collectors/literature.py's own
#    ``_normalize_pubmed_date_parts``/``_pub_date``). The fictional fixtures
#    below (PMIDs 90000030/90000031) reproduce the SHAPE of that finding,
#    never the real issuer/article it came from.
# =============================================================================
def test_pubmed_month_abbreviation_date_normalizes_and_source_survives(repo, fixture_dir):
    """Requirement 2: a real-format PubDate carrying a 3-letter English
    month abbreviation with a day (the exact reported bug shape) is
    normalized to ``"YYYY-MM-DD"`` before it ever reaches Source, passes
    Source validation/repository persistence, is never quarantined, and
    every Fact depending on it survives Evidence Integrity -- with the
    full Document->Source->Fact->Chunk->traceability lineage intact."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000030"]): fx.ok(fx.fixture_text("pubdate_month_abbreviation_with_day.xml")),
        fx.europepmc_search_url("90000030"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000030"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert bundle.refused_reason is None
    assert not bundle.collection_result.degraded
    assert bundle.coverage_complete

    pubmed_sources = [
        s for s in bundle.collection_result.sources if "efetch" in s.url or "pubmed" in s.url.lower()
    ]
    assert pubmed_sources
    # The exact reported bug shape (Year+Month-abbreviation+Day) is the
    # JournalIssue/PubDate this fixture carries -- normalized to full
    # day-precision, never left as "2025-Feb-04".
    assert any(s.published_date == "2025-02-04" for s in pubmed_sources)
    assert not any("Feb" in s.published_date for s in bundle.collection_result.sources)

    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )

    # Never quarantined.
    assert result.quarantined_sources == []
    assert not any("quarantined" in f for f in result.failures)

    # Every Fact the bridge submitted for this PMID survives Evidence
    # Integrity -- tracked via RawFact.fact_id() == Fact.fact_id (see
    # agents/fact_collector.py::FactCollectorAgent._to_fact).
    submitted_fact_ids = {raw.fact_id() for raw in bundle.collection_result.raw_facts}
    assert submitted_fact_ids
    surviving_fact_ids = {f.fact_id for f in result.bus.facts}
    assert submitted_fact_ids <= surviving_fact_ids

    # Document->Source->Fact->Chunk->traceability lineage intact.
    assert bundle.chunks
    lit_chunk = bundle.chunks[0]
    assert lit_chunk.chunk_id in {c.chunk_id for c in result.chunks}
    link = result.traceability.resolve(lit_chunk.chunk_id)
    assert link is not None
    assert link.fact_id in surviving_fact_ids

    # coverage_complete stays True for a genuinely clean fetch (requirement
    # 3's "never affect a normal/clean Literature acquisition" half).
    assert result.direct_acquisition_info["coverage_complete"] is True
    literature_reasons = [
        r for r in (result.verdict.blocking_verification_required if result.verdict else ())
        if "Literature Document-First" in r
    ]
    assert literature_reasons == []


def test_pubmed_date_normalization_matrix_all_survive_source_validation():
    """Phase 4.2B correction 1+2's full matrix, driven directly against
    ``collectors.literature._pub_date`` (the single normalization point):
    numeric month, English month name in multiple letter cases, year-month
    only, year only, a MedlineDate season/range degrading to year-only, a
    cross-year MedlineDate range and other unparseable/ambiguous values
    falling back to UNKNOWN, and a missing element -- every non-UNKNOWN
    output must pass the Date Integrity contract
    (``schemas.fact.parse_date_bounds``), and nothing here is ever a
    guessed date."""
    import xml.etree.ElementTree as ET

    from investment_research.collectors.literature import _pub_date
    from investment_research.schemas.fact import parse_date_bounds

    def _date_el(year=None, month=None, day=None, medline_date=None) -> ET.Element:
        el = ET.Element("PubDate")
        if medline_date is not None:
            ET.SubElement(el, "MedlineDate").text = medline_date
            return el
        if year is not None:
            ET.SubElement(el, "Year").text = year
        if month is not None:
            ET.SubElement(el, "Month").text = month
        if day is not None:
            ET.SubElement(el, "Day").text = day
        return el

    cases = [
        # 1. The exact reported bug shape.
        ({"year": "2025", "month": "Feb", "day": "04"}, "2025-02-04"),
        # 2. Numeric month and English month name, multiple letter cases.
        ({"year": "2025", "month": "feb", "day": "4"}, "2025-02-04"),  # lowercase, unpadded day
        ({"year": "2025", "month": "FEB", "day": "04"}, "2025-02-04"),  # uppercase
        ({"year": "2025", "month": "Feb", "day": "04"}, "2025-02-04"),  # mixed case
        ({"year": "2025", "month": "Dec", "day": "31"}, "2025-12-31"),
        ({"year": "2025", "month": "02", "day": "04"}, "2025-02-04"),  # already-numeric month
        ({"year": "2025", "month": "2", "day": "4"}, "2025-02-04"),  # unpadded numeric
        # 3. Year-month only, year only.
        ({"year": "2025", "month": "Feb"}, "2025-02"),
        ({"year": "2025"}, "2025"),
        # 4. A season -- only the year is certain.
        ({"medline_date": "2025 Winter"}, "2025"),
        # 5. A within-year month range -- only the year is certain.
        ({"medline_date": "2025 Jan-Feb"}, "2025"),
        # 6. A cross-year range -- no single year is certain.
        ({"medline_date": "2024-2025"}, "UNKNOWN"),
        # 7. Genuinely undeterminable text.
        ({"medline_date": "Spring/Summer, exact year uncertain"}, "UNKNOWN"),
        ({"year": "2025", "month": "Xyz", "day": "04"}, "2025"),  # unrecognized month token
    ]
    for kwargs, expected in cases:
        result = _pub_date(_date_el(**kwargs))
        assert result == expected, f"{kwargs} -> {result!r}, expected {expected!r}"
        if result != "UNKNOWN":
            assert parse_date_bounds(result) is not None, f"{expected!r} rejected by Date Integrity"

    # 8. A missing PubDate element entirely.
    assert _pub_date(None) == "UNKNOWN"

    # The MedlineDate free-text string is NEVER passed through verbatim --
    # requirement: never misattribute an imprecise season/range into
    # anything more specific than the year it genuinely names.
    assert _pub_date(_date_el(medline_date="2025 Jan-Feb")) != "2025 Jan-Feb"


def test_pubmed_medline_date_range_degrades_to_year_never_quarantined(monkeypatch, repo, fixture_dir):
    """Phase 4.2B correction 2 (requirements 4/5/9/10): a PubDate that is
    NLM's own MedlineDate range fallback (``"2025 Jan-Feb"``) degrades to
    the one precision genuinely certain -- the year -- rather than being
    rejected. A low date precision alone must never quarantine an
    otherwise-acquired paper: the Source passes validation, is never
    quarantined, its Facts survive Evidence Integrity, and
    coverage_complete is never disturbed for this genuinely clean fetch."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000031"]): fx.ok(fx.fixture_text("pubdate_medline_range_unparseable.xml")),
        fx.europepmc_search_url("90000031"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000031"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert not bundle.collection_result.degraded
    assert bundle.coverage_complete

    pubmed_sources = [
        s for s in bundle.collection_result.sources if "efetch" in s.url or "pubmed" in s.url.lower()
    ]
    assert pubmed_sources
    # Degraded to year-only -- never the raw "2025 Jan-Feb" string, never a
    # guessed month/day.
    assert any(s.published_date == "2025" for s in pubmed_sources)
    assert not any("Jan-Feb" in s.published_date for s in bundle.collection_result.sources)

    pipeline = _noise_free_pipeline(
        repo, monkeypatch, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )

    # Never quarantined -- a low-precision (year-only) date is not, by
    # itself, a validation failure.
    assert result.quarantined_sources == []
    assert not any("quarantined" in f for f in result.failures)

    submitted_fact_ids = {raw.fact_id() for raw in bundle.collection_result.raw_facts}
    assert submitted_fact_ids
    surviving_fact_ids = {f.fact_id for f in result.bus.facts}
    assert submitted_fact_ids <= surviving_fact_ids

    assert result.direct_acquisition_info["coverage_complete"] is True
    assert result.context.status is RunStatus.COMPLETE
    assert result.failures == []
    literature_blocking = [
        r for r in (result.verdict.blocking_verification_required if result.verdict else ())
        if "Literature Document-First" in r
    ]
    assert literature_blocking == []


def test_genuinely_malformed_source_field_still_quarantines_and_flips_coverage_complete(
    monkeypatch, repo, fixture_dir
):
    """Requirement 11: a genuinely invalid field OTHER than the (now
    always-safe) date fields must still be rejected and quarantined by
    Source validation exactly as before -- and the downstream
    completeness-consistency fix (Phase 4.2B correction 1) must still
    correctly flip ``coverage_complete``/``RunStatus``/``research_status``/
    ``action`` in that case. Since ``_pub_date`` can no longer itself
    produce a value Source validation would reject, this simulates the
    remaining failure mode directly: a Source with a malformed ``url``
    (never something ``_pub_date``/the adapter could produce), injected
    into an otherwise-genuine, already-acquired literature bundle -- the
    same technique the Correction 1 test suite uses to simulate a failure
    downstream of a clean acquisition (see
    ``test_chunk_projection_failure_alone_forces_action_none``)."""
    import dataclasses

    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    http = fx.FakeHttpClient(responses={
        _efetch_url(["90000030"]): fx.ok(fx.fixture_text("pubdate_month_abbreviation_with_day.xml")),
        fx.europepmc_search_url("90000030"): fx.ok(_EPMC_EMPTY),
    })
    bundle = run_literature_pipeline_acquisition(
        _pmid_request(pmid="90000030"), ticker="DEMOBIO", http_client=http, env=_ENV
    )
    assert not bundle.collection_result.degraded
    assert bundle.coverage_complete

    corrupted_sources = tuple(
        dataclasses.replace(s, url="not a well-formed url") if "efetch" in s.url else s
        for s in bundle.collection_result.sources
    )
    assert corrupted_sources != bundle.collection_result.sources
    corrupted_collection_result = dataclasses.replace(bundle.collection_result, sources=corrupted_sources)

    pipeline = _noise_free_pipeline(
        repo, monkeypatch, chunks=list(bundle.chunks),
        direct_acquisition_info=bundle_diagnostics(bundle, enabled=True),
    )
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, corrupted_collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )

    # The malformed-url Source really was quarantined -- the general
    # Source-validation/quarantine mechanism is unaffected by this fix.
    assert result.quarantined_sources
    assert any(q.field == "url" for q in result.quarantined_sources)

    # direct_acquisition_info corrected downstream: never stale True.
    assert result.direct_acquisition_info["feature_enabled"] is True
    assert result.direct_acquisition_info["coverage_complete"] is False
    unresolved = result.direct_acquisition_info.get("unresolved_reasons", [])
    assert unresolved
    assert any("quarantined" in r.lower() for r in unresolved)
    blob = " ".join(unresolved)
    assert "http" not in blob  # sanitized: no raw URL

    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.run_status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.verdict.action is None
    assert result.incomplete is True

    literature_failures = [f for f in result.failures if "Literature Document-First" in f]
    assert len(literature_failures) == 1
    literature_blocking = [
        r for r in result.verdict.blocking_verification_required if "Literature Document-First" in r
    ]
    assert len(literature_blocking) == 1


def test_flag_off_pubmed_date_fix_never_fires(monkeypatch, repo, fixture_dir):
    """Requirement 3's other half: the downstream completeness-consistency
    check must never fire when --document-first-literature was never used
    (direct_acquisition_info stays the pre-existing empty dict)."""
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata("DEMOBIO")
    base_result = fixtures.collect("DEMOBIO", metadata["company_name"])

    pipeline = _noise_free_pipeline(repo, monkeypatch)  # no chunks, no direct_acquisition_info
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result], price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )
    assert result.direct_acquisition_info == {}
    assert result.context.status is RunStatus.COMPLETE
    assert result.failures == []
    assert result.verdict.action is not None
