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
from investment_research.research.source_routing import ImplementationStatus
from investment_research.schemas.enums import FetchOutcome, ResearchStatus

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
    diag = bundle_diagnostics(None, enabled=False)
    assert diag == {
        "feature_enabled": False, "anthropic_api_calls": 0, "web_search_calls": 0,
        "external_llm_tokens": 0,
    }


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

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks))
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

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, chunks=list(bundle.chunks))
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"], [base_result, bundle.collection_result],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )
    literature_reasons = [
        r for r in (result.verdict.blocking_verification_required if result.verdict else ())
        if "Literature Document-First" in r
    ]
    assert literature_reasons == []


def test_pipeline_never_calls_literature_acquisition_code_directly():
    """pipeline.py's own blocking-propagation check (Phase 4.2A) reads
    only CollectionResult.collector/.degraded/.describe() -- it must never
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
# 7. cli.py wiring: flag OFF is byte-identical; flag ON follows production
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
    assert result.direct_acquisition_info == {"feature_enabled": False, "anthropic_api_calls": 0,
                                                "web_search_calls": 0, "external_llm_tokens": 0}


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
