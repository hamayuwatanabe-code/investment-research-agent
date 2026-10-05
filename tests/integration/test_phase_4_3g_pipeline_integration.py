"""Phase 4.3G: Program Identity Resolution / Adaptive Acquisition Plan
Production integration -- full ``Pipeline.run()`` tests.

Mirrors ``tests/integration/test_structured_evidence_snapshot.py``'s own
conventions: small, hand-built ``CollectionResult``s carrying Phase 4.3F
structured evidence, run through the real ``Pipeline.run()`` (never a
mock), with ``NullSearchProvider``/no LLM configured -- offline throughout.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

import investment_research.storage.repository as repository_module
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.isolation import IsolationGuard
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.schemas.enums import FactCategory, Provenance, SourceTier
from investment_research.schemas.fact import RawFact, Source
from investment_research.scoring.adaptive_acquisition_plan import AcquisitionPlanStatus
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
    LiteratureCandidateEvidence,
    ProgramCandidateEvidence,
)
from investment_research.scoring.program_identity_resolution import (
    LiteratureLinkStatus,
    ProgramIdentityResolution,
    ProgramIdentityStatus,
)
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository, ResumeSnapshotCorrupted

pytestmark = pytest.mark.integration

TODAY = date(2026, 10, 5)
TICKER = "CNFTST"
COMPANY = "Confirm Test Biotherapeutics Inc"
NCT_ID = "NCT55512345"


def _new_repo() -> Repository:
    return Repository(open_db(":memory:"))


def _source(source_id: str, url: str, *, published_date: str = "2026-09-01") -> Source:
    return Source(
        source_id=source_id, url=url, title="Doc", tier=SourceTier.TIER_1,
        publisher="SEC", published_date=published_date,
    )


def _raw_fact(source: Source, claim: str = "A registered claim.") -> RawFact:
    return RawFact(
        ticker=TICKER, category=FactCategory.CLINICAL, claim=claim, source=source,
        company_claim=True, collector="phase43g_fixture",
    )


def _company_identity(source: Source, *, sec_official_name: str = COMPANY, **kwargs) -> CompanyIdentityEvidence:
    return CompanyIdentityEvidence(
        ticker=TICKER, cik=1234567, sec_official_name=sec_official_name,
        source_id=source.source_id, source_tier=source.tier,
        retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        **kwargs,
    )


def _program_candidate(
    source: Source, *, lead_sponsor: str = COMPANY, nct_id: str = NCT_ID,
    collaborators: tuple[str, ...] = (), interventions: tuple[str, ...] = (),
    supporting_fact_ids: tuple[str, ...] = (),
) -> ProgramCandidateEvidence:
    return ProgramCandidateEvidence(
        nct_id=nct_id, lead_sponsor=lead_sponsor, collaborators=collaborators,
        interventions=interventions, overall_status="RECRUITING",
        source_id=source.source_id, source_tier=source.tier,
        retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        supporting_fact_ids=supporting_fact_ids,
    )


def _literature_candidate(
    source: Source, *, pmid: str, nct_ids: tuple[str, ...] = ()
) -> LiteratureCandidateEvidence:
    return LiteratureCandidateEvidence(
        pmid=pmid, nct_ids=nct_ids, source_id=source.source_id, source_tier=source.tier,
        retrieved_at=source.retrieved_at, content_hash=source.content_hash,
    )


def _collection_result(
    *, company_identity=None, program_candidates=(), literature_candidates=(),
    sources=(), raw_facts=(),
) -> CollectionResult:
    return CollectionResult(
        collector="phase43g_fixture", raw_facts=list(raw_facts), sources=list(sources),
        provenance=Provenance.FIXTURE,
        company_identity_evidence=company_identity,
        program_candidate_evidence=tuple(program_candidates),
        literature_candidate_evidence=tuple(literature_candidates),
    )


def _run(
    collection_results, run_id, *, repo=None, resume=False, direct_acquisition_info=None,
):
    repo = repo or _new_repo()
    pipeline = Pipeline(
        repo, NullSearchProvider(), today=TODAY, direct_acquisition_info=direct_acquisition_info,
    )
    result = pipeline.run(
        ticker=TICKER, company_name=COMPANY, collection_results=collection_results,
        run_id=run_id, resume=resume,
    )
    return repo, result


def _confirmed_scenario_collection_results() -> list[CollectionResult]:
    """Company identity + one sponsor-matched, strict-valid-NCT candidate,
    zero literature evidence -- the baseline CONFIRMED + UNRESOLVED
    literature scenario a plain ``READY`` plan comes from."""
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    fact = _raw_fact(source)
    return [
        _collection_result(
            company_identity=_company_identity(source),
            program_candidates=[
                _program_candidate(source, supporting_fact_ids=(fact.fact_id(),))
            ],
            sources=[source],
            raw_facts=[fact],
        )
    ]


# =============================================================================
# 7/8. CompanyIdentityEvidence cardinality
# =============================================================================
def test_zero_company_identity_evidence_is_unresolved():
    source = _source("src_ct", "https://clinicaltrials.gov/study/NCT55512345")
    collection_results = [
        _collection_result(
            company_identity=None,
            program_candidates=[_program_candidate(source)],
            sources=[source],
        )
    ]
    _, result = _run(collection_results, "p43g-card-zero")
    assert result.bootstrap_program_identity is not None
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.UNRESOLVED
    assert result.adaptive_acquisition_plan.max_requests == 0


def test_multiple_company_identity_evidence_is_unresolved_never_arbitrary_pick():
    source_a = _source("src_sec_a", "https://www.sec.gov/Archives/a/8-K.htm")
    source_b = _source("src_sec_b", "https://www.sec.gov/Archives/b/8-K.htm")
    collection_results = [
        _collection_result(company_identity=_company_identity(source_a), sources=[source_a]),
        _collection_result(
            company_identity=_company_identity(source_b, sec_official_name="Other Name Inc"),
            sources=[source_b],
        ),
    ]
    _, result = _run(collection_results, "p43g-card-multi")
    assert len(result.company_identity_evidence) == 2
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.UNRESOLVED


# =============================================================================
# 9/10. Quarantined Source / missing-mismatched supporting Fact
# =============================================================================
def test_candidate_resting_on_a_quarantined_source_never_confirms():
    identity_source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    # Malformed published_date -> fails schema validation -> quarantined.
    bad_source = _source(
        "src_ct_bad", "https://clinicaltrials.gov/study/NCT55512345",
        published_date="NOT-A-REAL-DATE",
    )
    fact = _raw_fact(bad_source)
    collection_results = [
        _collection_result(
            company_identity=_company_identity(identity_source),
            program_candidates=[
                _program_candidate(bad_source, supporting_fact_ids=(fact.fact_id(),))
            ],
            sources=[identity_source, bad_source],
            raw_facts=[fact],
        )
    ]
    _, result = _run(collection_results, "p43g-quarantine")
    assert any(q.source_id == "src_ct_bad" for q in result.quarantined_sources)
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED
    assert result.adaptive_acquisition_plan.status is not AcquisitionPlanStatus.READY


def test_candidate_with_a_missing_supporting_fact_never_confirms():
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    fabricated_fact_id = "fact_" + "a" * 20
    collection_results = [
        _collection_result(
            company_identity=_company_identity(source),
            program_candidates=[
                _program_candidate(source, supporting_fact_ids=(fabricated_fact_id,))
            ],
            sources=[source],
        )
    ]
    _, result = _run(collection_results, "p43g-missingfact")
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED


def test_candidate_with_a_mismatched_supporting_fact_source_never_confirms():
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    other_source = _source("src_other", "https://www.sec.gov/Archives/y/8-K.htm")
    fact_on_other_source = _raw_fact(other_source)
    collection_results = [
        _collection_result(
            company_identity=_company_identity(source),
            program_candidates=[
                _program_candidate(source, supporting_fact_ids=(fact_on_other_source.fact_id(),))
            ],
            sources=[source, other_source],
            raw_facts=[fact_on_other_source],
        )
    ]
    _, result = _run(collection_results, "p43g-mismatchfact")
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED


# =============================================================================
# 11/12/13. Never-sufficient-alone signals
# =============================================================================
def test_collaborator_match_alone_never_confirms():
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    collection_results = [
        _collection_result(
            company_identity=_company_identity(source),
            program_candidates=[
                _program_candidate(source, lead_sponsor="Unrelated Sponsor Inc", collaborators=(COMPANY,))
            ],
            sources=[source],
        )
    ]
    _, result = _run(collection_results, "p43g-collab-only")
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED


def test_same_drug_name_different_sponsor_never_confirms():
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    collection_results = [
        _collection_result(
            company_identity=_company_identity(source),
            program_candidates=[
                _program_candidate(
                    source, lead_sponsor="Totally Different Sponsor Inc",
                    interventions=("CNFTST-101",),
                )
            ],
            sources=[source],
        )
    ]
    _, result = _run(collection_results, "p43g-samedrug")
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED


def test_alias_match_alone_never_confirms():
    """``lead_sponsor`` matches ONLY ``explicitly_verified_aliases``, never
    ``sec_official_name`` even after normalization -- alias matching is
    disabled since Phase 4.3B Correction 1, and Phase 4.3G reuses that
    resolver unchanged."""
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    collection_results = [
        _collection_result(
            company_identity=_company_identity(
                source, explicitly_verified_aliases=("Alias Pharma Holdings",)
            ),
            program_candidates=[_program_candidate(source, lead_sponsor="Alias Pharma Holdings")],
            sources=[source],
        )
    ]
    _, result = _run(collection_results, "p43g-alias")
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED


# =============================================================================
# 14. Fresh run produces all three internal results
# =============================================================================
def test_fresh_run_produces_all_three_internal_results():
    _, result = _run(_confirmed_scenario_collection_results(), "p43g-fresh-three")
    assert isinstance(result.bootstrap_program_identity, ProgramIdentityResolution)
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.CONFIRMED
    assert result.bootstrap_program_identity.nct_id == NCT_ID
    assert result.bootstrap_literature_link is not None
    assert result.bootstrap_literature_link.status is LiteratureLinkStatus.UNRESOLVED
    assert result.adaptive_acquisition_plan is not None
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.READY
    assert result.adaptive_acquisition_plan.nct_id == NCT_ID
    assert result.adaptive_acquisition_plan.max_requests == 6
    assert result.adaptive_acquisition_plan.requires_external_communication is True


def test_confirmed_with_linked_literature_is_no_action_end_to_end():
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    fact = _raw_fact(source)
    lit_source = _source("src_pubmed", "https://pubmed.ncbi.nlm.nih.gov/90000012/")
    collection_results = [
        _collection_result(
            company_identity=_company_identity(source),
            program_candidates=[_program_candidate(source, supporting_fact_ids=(fact.fact_id(),))],
            literature_candidates=[_literature_candidate(lit_source, pmid="90000012", nct_ids=(NCT_ID,))],
            sources=[source, lit_source],
            raw_facts=[fact],
        )
    ]
    _, result = _run(collection_results, "p43g-linked")
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.CONFIRMED
    assert result.bootstrap_literature_link.status is LiteratureLinkStatus.LINKED
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.NO_ACTION
    assert result.adaptive_acquisition_plan.max_requests == 0
    assert result.adaptive_acquisition_plan.requires_external_communication is False


# =============================================================================
# 15. Fresh/resume semantic equivalence
# =============================================================================
def _crash_after_verify_run(repo, run_id, collection_results, monkeypatch):
    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_verify(self, checkpoint):
        if checkpoint.stage not in ("collect", "verify"):
            raise RuntimeError("simulated crash after verify checkpoint")
        return original_save_checkpoint(self, checkpoint)

    with monkeypatch.context() as m:
        m.setattr(repository_module.Repository, "save_checkpoint", _crash_after_verify)
        pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
        with pytest.raises(RuntimeError, match="simulated crash after verify checkpoint"):
            pipeline.run(
                ticker=TICKER, company_name=COMPANY, collection_results=collection_results,
                run_id=run_id,
            )


def test_fresh_and_resumed_runs_produce_semantically_identical_resolution_and_plan(monkeypatch):
    control_repo = _new_repo()
    _, control = _run(_confirmed_scenario_collection_results(), "p43g-resume-control", repo=control_repo)

    repo = _new_repo()
    run_id = "p43g-resume-subject"
    _crash_after_verify_run(repo, run_id, _confirmed_scenario_collection_results(), monkeypatch)

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(
        ticker=TICKER, company_name=COMPANY, collection_results=[], run_id=run_id, resume=True,
    )

    assert resumed.resume_plan is not None
    assert resumed.resume_plan.snapshot_unavailable_reason is None
    assert resumed.bootstrap_program_identity == control.bootstrap_program_identity
    assert resumed.bootstrap_literature_link == control.bootstrap_literature_link
    assert resumed.adaptive_acquisition_plan == control.adaptive_acquisition_plan


# =============================================================================
# 16. Missing/corrupt snapshot never reaches READY
# =============================================================================
def test_resume_with_missing_collect_snapshot_never_reaches_ready(monkeypatch):
    run_id = "p43g-missing-snapshot"
    repo = _new_repo()
    _crash_after_verify_run(repo, run_id, _confirmed_scenario_collection_results(), monkeypatch)

    repo.conn.execute(
        "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'", (run_id,),
    )
    repo.conn.commit()

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(
        ticker=TICKER, company_name=COMPANY, collection_results=[], run_id=run_id, resume=True,
    )

    assert resumed.resume_plan.snapshot_unavailable_reason is not None
    assert resumed.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED
    assert resumed.adaptive_acquisition_plan.status is AcquisitionPlanStatus.UNRESOLVED
    assert resumed.adaptive_acquisition_plan.status is not AcquisitionPlanStatus.READY
    assert resumed.adaptive_acquisition_plan.max_requests == 0
    assert resumed.adaptive_acquisition_plan.requires_external_communication is False


def test_resume_with_corrupted_collect_snapshot_never_reaches_ready(monkeypatch):
    run_id = "p43g-corrupt-snapshot"
    repo = _new_repo()
    _crash_after_verify_run(repo, run_id, _confirmed_scenario_collection_results(), monkeypatch)

    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = ? AND stage = '_collect_snapshot'",
        ("{not valid json", run_id),
    )
    repo.conn.commit()
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(run_id)

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(
        ticker=TICKER, company_name=COMPANY, collection_results=[], run_id=run_id, resume=True,
    )

    assert resumed.resume_plan.snapshot_unavailable_reason is not None
    assert resumed.bootstrap_program_identity.status is ProgramIdentityStatus.UNRESOLVED
    assert resumed.adaptive_acquisition_plan.status is not AcquisitionPlanStatus.READY
    assert resumed.adaptive_acquisition_plan.max_requests == 0


# =============================================================================
# 17. Explicit PMID/NCT override is never overridden by the auto plan
# =============================================================================
def test_explicit_override_feature_enabled_is_never_overridden_by_auto_plan():
    explicit_info = {
        "feature_enabled": True,
        "reference_mode": "PMID",
        "pmid": "12345678",
        "nct_id": None,
        "current_program_confirmed": False,
        "refused": False,
        "coverage_complete": True,
    }
    _, result = _run(
        _confirmed_scenario_collection_results(), "p43g-explicit-override",
        direct_acquisition_info=explicit_info,
    )
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    assert result.adaptive_acquisition_plan.max_requests == 0
    assert result.adaptive_acquisition_plan.requires_external_communication is False
    # The explicit designation itself is completely untouched.
    assert result.direct_acquisition_info.get("feature_enabled") is True
    assert result.direct_acquisition_info.get("pmid") == "12345678"
    assert result.direct_acquisition_info.get("reference_mode") == "PMID"


# =============================================================================
# 18. Exploding AcquisitionExecutor/adapter/HttpClient/LLM/WebProvider never
#     get called by Phase 4.3G's own code path
# =============================================================================
def test_exploding_execution_paths_are_never_touched(monkeypatch):
    import investment_research.collectors.http as http_module
    import investment_research.llm.client as llm_client_module
    import investment_research.research.acquisition_executor as acquisition_executor_module
    import investment_research.research.anthropic_web as anthropic_web_module
    import investment_research.research.clinicaltrials_acquisition_adapter as ct_adapter_module
    import investment_research.research.literature_pipeline_integration as literature_module

    def _explode(*_args, **_kwargs):
        raise AssertionError("Phase 4.3G must never reach this execution path")

    monkeypatch.setattr(http_module.HttpClient, "__init__", _explode)
    monkeypatch.setattr(llm_client_module.LLMClient, "__init__", _explode)
    monkeypatch.setattr(anthropic_web_module.AnthropicWebResearchProvider, "__init__", _explode)
    monkeypatch.setattr(acquisition_executor_module.AcquisitionExecutor, "__init__", _explode)
    monkeypatch.setattr(acquisition_executor_module.AcquisitionExecutor, "run", _explode)
    monkeypatch.setattr(ct_adapter_module.ClinicalTrialsStudyAdapter, "execute", _explode)
    monkeypatch.setattr(literature_module, "run_literature_pipeline_acquisition", _explode)

    _, result = _run(_confirmed_scenario_collection_results(), "p43g-explode-guard")
    assert result.bootstrap_program_identity.status is ProgramIdentityStatus.CONFIRMED
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.READY


# =============================================================================
# 19. No AgentInput ever carries the new Program Identity/plan
# =============================================================================
def test_no_agent_input_ever_carries_bootstrap_identity_or_plan(monkeypatch):
    captured_params: list[dict] = []
    original_project = IsolationGuard.project

    def _capture(self, agent_id, **kwargs):
        agent_input = original_project(self, agent_id, **kwargs)
        captured_params.append(dict(agent_input.params))
        assert not hasattr(agent_input, "bootstrap_program_identity")
        assert not hasattr(agent_input, "bootstrap_literature_link")
        assert not hasattr(agent_input, "adaptive_acquisition_plan")
        return agent_input

    monkeypatch.setattr(IsolationGuard, "project", _capture)
    _run(_confirmed_scenario_collection_results(), "p43g-agent-leak-guard")

    assert captured_params  # sanity: agents actually ran
    forbidden_names = (
        "bootstrap_program_identity", "bootstrap_literature_link", "adaptive_acquisition_plan",
        "ProgramIdentityResolution", "AdaptiveAcquisitionPlan", "LiteratureLinkResolution",
    )
    for params in captured_params:
        blob = json.dumps(params, default=str)
        for name in forbidden_names:
            assert name not in blob, f"{name!r} leaked into an AgentInput.params: {params}"


# =============================================================================
# 20. The existing central ProgramResolution object identity is unchanged
# =============================================================================
def test_central_program_resolution_is_the_same_object_for_science_and_kill(monkeypatch):
    captured_resolutions: dict[str, object] = {}
    original_run_agent = Pipeline._run_agent

    def _capture(self, agent, guard, result, **kwargs):
        pr = kwargs.get("program_resolution")
        if pr is not None:
            captured_resolutions[agent.agent_id] = pr
        return original_run_agent(self, agent, guard, result, **kwargs)

    monkeypatch.setattr(Pipeline, "_run_agent", _capture)
    _run(_confirmed_scenario_collection_results(), "p43g-central-resolution-identity")

    assert "science" in captured_resolutions or "kill_agent" in captured_resolutions
    values = list(captured_resolutions.values())
    for other in values[1:]:
        assert other is values[0]


# =============================================================================
# 23. Resume never mixes CollectSnapshot-sourced structured evidence with
#     fresh collection_results
# =============================================================================
def test_resume_past_verify_bootstrap_uses_snapshot_evidence_not_fresh_collection_results(
    monkeypatch,
):
    """Resuming with a DIFFERENT (here, empty) collection_results argument
    must not change the bootstrap resolution/plan at all -- they must come
    entirely from the collect snapshot's own restored structured evidence,
    exactly like Phase 4.3F's own Sources/structured-evidence restoration."""
    run_id = "p43g-resume-no-mixing"
    repo = _new_repo()
    _crash_after_verify_run(repo, run_id, _confirmed_scenario_collection_results(), monkeypatch)

    # A decoy CollectionResult that would resolve DIFFERENTLY if it were
    # ever mixed in -- a different, non-matching sponsor.
    decoy_source = _source("src_decoy", "https://www.sec.gov/Archives/decoy/8-K.htm")
    decoy_collection_results = [
        _collection_result(
            company_identity=_company_identity(decoy_source, sec_official_name="Decoy Sponsor Inc"),
            sources=[decoy_source],
        )
    ]

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(
        ticker=TICKER, company_name=COMPANY, collection_results=decoy_collection_results,
        run_id=run_id, resume=True,
    )

    assert resumed.resume_plan.snapshot_unavailable_reason is None
    # Still the ORIGINAL confirmed identity, never the decoy.
    assert resumed.bootstrap_program_identity.status is ProgramIdentityStatus.CONFIRMED
    assert resumed.bootstrap_program_identity.nct_id == NCT_ID
    assert resumed.bootstrap_program_identity.lead_sponsor == COMPANY
    assert len(resumed.company_identity_evidence) == 1
    assert resumed.company_identity_evidence[0].sec_official_name == COMPANY


# =============================================================================
# 22 (partial). CLI/report rendering never references the new fields
# =============================================================================
def test_cli_module_source_never_references_the_new_phase_4_3g_fields():
    from pathlib import Path

    repo_src = Path(__file__).resolve().parent.parent.parent / "src" / "investment_research"
    cli_text = (repo_src / "cli.py").read_text(encoding="utf-8")
    for name in ("bootstrap_program_identity", "bootstrap_literature_link", "adaptive_acquisition_plan"):
        assert name not in cli_text
