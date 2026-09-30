"""Phase 4.3F: structured primary-source evidence (scoring/program_evidence.py)
flows losslessly through CollectionResult -> Pipeline.run() -> the collect
snapshot -> a resumed run.

Mirrors tests/integration/test_pipeline_resume.py's own conventions (a
small, hand-built CollectionResult; a genuine crash-then-resume sequence
via monkeypatched Repository.save_checkpoint) so this file plugs into the
same established resume-testing pattern, deliberately scoped narrowly to
this phase's own new fields rather than duplicating that file's much
broader coverage.
"""

from __future__ import annotations

from datetime import date

import pytest

import investment_research.storage.repository as repository_module
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.schemas.enums import FactCategory, Provenance, SourceTier
from investment_research.schemas.fact import RawFact, Source
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
    LiteratureCandidateEvidence,
    ProgramCandidateEvidence,
)
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)
TICKER = "SEVFIX"
COMPANY = "Structured Evidence Test Biotech"


def _new_repo() -> Repository:
    return Repository(open_db(":memory:"))


def _company_identity() -> CompanyIdentityEvidence:
    return CompanyIdentityEvidence(
        ticker=TICKER,
        cik=1234567,
        sec_official_name="Structured Evidence Test Biotech Inc",
        source_id="src_ticker_map",
        source_tier=SourceTier.TIER_1,
        retrieved_at="2026-09-01T00:00:00+00:00",
        content_hash="tickermaphash",
    )


def _program_candidate() -> ProgramCandidateEvidence:
    return ProgramCandidateEvidence(
        nct_id="NCT70000001",
        lead_sponsor="Structured Evidence Test Biotech Inc",
        collaborators=("Collaborator University",),
        interventions=("Test Compound Y",),
        conditions=("Test Indication",),
        overall_status="RECRUITING",
        phases=("PHASE2",),
        primary_completion_date="2026-12-31",
        completion_date="2027-06-30",
        source_id="src_ct_study",
        source_tier=SourceTier.TIER_1,
        retrieved_at="2026-09-02T00:00:00+00:00",
        content_hash="studyhash",
        supporting_fact_ids=("fact_" + "c" * 20,),
    )


def _literature_candidate() -> LiteratureCandidateEvidence:
    return LiteratureCandidateEvidence(
        pmid="90000012",
        nct_ids=("NCT70000001",),
        source_id="src_pubmed",
        source_tier=SourceTier.UNKNOWN,
        retrieved_at="2026-09-03T00:00:00+00:00",
        content_hash="pubmedhash",
    )


def _run_kwargs() -> dict:
    """A small, hand-built CollectionResult carrying one of each new
    structured-evidence type -- deliberately not run through a real
    collector, mirroring test_pipeline_resume.py's own _run_kwargs()."""
    # Deliberately a generic URL: never embedding TICKER/COMPANY as a
    # substring, which would leak as a "identity marker present in blind
    # input" false positive once this fact's source_url legitimately
    # reaches Blind Judge's evidence pack (facts travel downstream by
    # design -- only the ticker/company NAME itself is denied there).
    company_source = Source(
        source_id="src_company", url="https://www.sec.gov/example-filing",
        title="Form 10-Q", tier=SourceTier.TIER_1, published_date="2026-09-01",
    )
    raw_facts = [
        RawFact(
            ticker=TICKER, category=FactCategory.OTHER,
            claim="The company reported quarterly revenue of $5 million.",
            source=company_source, company_claim=True,
        ),
    ]
    collection_result = CollectionResult(
        collector="structured_evidence_fixture",
        raw_facts=raw_facts,
        sources=[company_source],
        provenance=Provenance.FIXTURE,
        company_identity_evidence=_company_identity(),
        program_candidate_evidence=(_program_candidate(),),
        literature_candidate_evidence=(_literature_candidate(),),
    )
    return {
        "ticker": TICKER,
        "company_name": COMPANY,
        "collection_results": [collection_result],
    }


def _assert_matches_fixture(evidence_list_company, evidence_list_program, evidence_list_lit) -> None:
    assert len(evidence_list_company) == 1
    assert evidence_list_company[0] == _company_identity()
    assert len(evidence_list_program) == 1
    assert evidence_list_program[0] == _program_candidate()
    assert len(evidence_list_lit) == 1
    assert evidence_list_lit[0] == _literature_candidate()


# =============================================================================
# A. A normal (non-resumed) run
# =============================================================================
def test_fresh_run_populates_structured_evidence_on_the_result():
    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run(**_run_kwargs(), run_id="structev-a")

    _assert_matches_fixture(
        result.company_identity_evidence,
        result.program_candidate_evidence,
        result.literature_candidate_evidence,
    )


def test_collectionresult_default_fields_stay_empty_for_a_collector_that_builds_none():
    """A CollectionResult with no structured evidence at all (the
    overwhelming majority of collectors, unchanged) leaves the run's
    aggregated lists empty -- never populated with something invented."""
    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    kwargs = _run_kwargs()
    kwargs["collection_results"][0].company_identity_evidence = None
    kwargs["collection_results"][0].program_candidate_evidence = ()
    kwargs["collection_results"][0].literature_candidate_evidence = ()

    result = pipeline.run(**kwargs, run_id="structev-a-empty")

    assert result.company_identity_evidence == []
    assert result.program_candidate_evidence == []
    assert result.literature_candidate_evidence == []


# =============================================================================
# B. The collect snapshot itself carries every field losslessly
# =============================================================================
def test_collect_snapshot_round_trips_structured_evidence_losslessly():
    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "structev-b"
    pipeline.run(**_run_kwargs(), run_id=run_id)

    snapshot = repo.collect_snapshot_for_resume(run_id)
    assert snapshot is not None
    _assert_matches_fixture(
        snapshot.company_identity_evidence,
        snapshot.program_candidate_evidence,
        snapshot.literature_candidate_evidence,
    )


def test_a_pre_4_3f_snapshot_with_no_structured_evidence_keys_is_not_corrupted():
    """A snapshot saved before this phase existed (no company_identity_
    evidence/program_candidate_evidence/literature_candidate_evidence keys
    in its JSON at all) must still be read back as a valid, non-corrupted
    snapshot with empty structured-evidence lists -- never raise
    ResumeSnapshotCorrupted merely for lacking keys that did not exist yet
    when it was written."""
    repo = _new_repo()
    repo.upsert_company(TICKER, COMPANY)
    repo.save_collect_snapshot(
        "structev-legacy", facts=[], sources=[],
        collectors=[{"collector": "old", "outcome": "OK", "provenance": "LIVE", "errors": [],
                     "attempted_urls": [], "notes": [], "zero_results": False,
                     "raw_fact_count_before_dedup": 0}],
    )
    # Simulate a genuinely pre-4.3F row: strip the new keys entirely rather
    # than merely leaving them at their (already-empty) default, so this
    # test does not just re-prove the default -- it proves a row that could
    # only have been written by OLDER code still reads back cleanly.
    import json

    row = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
        ("structev-legacy",),
    ).fetchone()
    payload = json.loads(row["payload"])
    for key in (
        "company_identity_evidence", "program_candidate_evidence", "literature_candidate_evidence",
    ):
        payload.pop(key, None)
    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = ? AND stage = '_collect_snapshot'",
        (json.dumps(payload), "structev-legacy"),
    )
    repo.conn.commit()

    snapshot = repo.collect_snapshot_for_resume("structev-legacy")
    assert snapshot is not None
    assert snapshot.company_identity_evidence == []
    assert snapshot.program_candidate_evidence == []
    assert snapshot.literature_candidate_evidence == []


# =============================================================================
# C. Resume restores it, from the SAME snapshot, never mixed with anything
#    a resumed invocation happens to have fresh
# =============================================================================
def test_resumed_run_restores_structured_evidence_from_the_collect_snapshot(monkeypatch):
    run_id = "structev-c"
    repo = _new_repo()

    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_collect(self, checkpoint):
        if checkpoint.stage != "collect":
            raise RuntimeError("simulated crash after collect checkpoint")
        return original_save_checkpoint(self, checkpoint)

    with monkeypatch.context() as m:
        m.setattr(repository_module.Repository, "save_checkpoint", _crash_after_collect)
        pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
        with pytest.raises(RuntimeError, match="simulated crash after collect checkpoint"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert resumed.resume_plan is not None
    _assert_matches_fixture(
        resumed.company_identity_evidence,
        resumed.program_candidate_evidence,
        resumed.literature_candidate_evidence,
    )


def test_resume_past_verify_restores_structured_evidence_from_the_collect_snapshot(
    monkeypatch,
):
    """Phase 4.3F correction 1: when resuming past 'verify', the collect
    snapshot's own structured-evidence lists (company_identity_evidence /
    program_candidate_evidence / literature_candidate_evidence) and the
    Sources they reference are now ALSO recovered, on a best-effort basis
    -- see Pipeline.run()'s own comment on this exact point. A run whose
    collect snapshot was actually recorded no longer silently loses this
    lossless Phase 4.3F output merely because it resumed past verify."""
    run_id = "structev-c2"
    repo = _new_repo()

    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_verify(self, checkpoint):
        if checkpoint.stage not in ("collect", "verify"):
            raise RuntimeError("simulated crash after verify checkpoint")
        return original_save_checkpoint(self, checkpoint)

    with monkeypatch.context() as m:
        m.setattr(repository_module.Repository, "save_checkpoint", _crash_after_verify)
        pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
        with pytest.raises(RuntimeError, match="simulated crash after verify checkpoint"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert resumed.resume_plan is not None
    _assert_matches_fixture(
        resumed.company_identity_evidence,
        resumed.program_candidate_evidence,
        resumed.literature_candidate_evidence,
    )


def test_resume_past_verify_survives_a_missing_collect_snapshot(monkeypatch):
    """Finding A's fix is deliberately best-effort/non-blocking: if the
    collect snapshot cannot be recovered when resuming past verify (here,
    simulated by deleting its row outright after the crash), the resume
    itself must still succeed -- Action/status/facts are governed solely by
    verify_snapshot -- and only the structured-evidence lists fall back to
    empty, exactly as a collector that built none already does."""
    run_id = "structev-c3"
    repo = _new_repo()

    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_verify(self, checkpoint):
        if checkpoint.stage not in ("collect", "verify"):
            raise RuntimeError("simulated crash after verify checkpoint")
        return original_save_checkpoint(self, checkpoint)

    with monkeypatch.context() as m:
        m.setattr(repository_module.Repository, "save_checkpoint", _crash_after_verify)
        pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
        with pytest.raises(RuntimeError, match="simulated crash after verify checkpoint"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    repo.conn.execute(
        "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
        (run_id,),
    )
    repo.conn.commit()

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert resumed.resume_plan is not None
    assert resumed.resume_plan.snapshot_unavailable_reason is None
    assert resumed.company_identity_evidence == []
    assert resumed.program_candidate_evidence == []
    assert resumed.literature_candidate_evidence == []


def test_resumed_structured_evidence_equals_a_fresh_uninterrupted_runs(monkeypatch):
    """Full semantic equivalence, narrowly scoped to this phase's own new
    fields -- mirrors test_pipeline_resume.py's own
    _assert_semantically_equivalent pattern."""
    control_repo = _new_repo()
    control_pipeline = Pipeline(control_repo, NullSearchProvider(), today=TODAY)
    control = control_pipeline.run(**_run_kwargs(), run_id="structev-d-control")

    repo = _new_repo()
    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_collect(self, checkpoint):
        if checkpoint.stage != "collect":
            raise RuntimeError("simulated crash after collect checkpoint")
        return original_save_checkpoint(self, checkpoint)

    with monkeypatch.context() as m:
        m.setattr(repository_module.Repository, "save_checkpoint", _crash_after_collect)
        pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
        with pytest.raises(RuntimeError, match="simulated crash after collect checkpoint"):
            pipeline.run(**_run_kwargs(), run_id="structev-d")

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id="structev-d", resume=True)

    assert resumed.company_identity_evidence == control.company_identity_evidence
    assert resumed.program_candidate_evidence == control.program_candidate_evidence
    assert resumed.literature_candidate_evidence == control.literature_candidate_evidence
