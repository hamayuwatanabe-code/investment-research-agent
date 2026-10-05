"""Phase 4.3I: ``orchestrator/adaptive_literature_step.py`` -- connects
Phase 4.3G's ``AdaptiveAcquisitionPlan`` and Phase 4.3H's
``execute_adaptive_literature_plan()`` bridge to a form ``Pipeline.run()``
can call safely, including every resume stop-point. Entirely offline:
driven only against ``tests/unit/_literature_fixture_support.FakeHttpClient``/
``RaisingHttpClient`` -- no real NCBI/Europe PMC communication, no socket
of any kind.
"""

from __future__ import annotations

from urllib.parse import urlencode

import pytest

import investment_research.orchestrator.adaptive_literature_step as step_module
from investment_research.collectors.base import CollectionResult
from investment_research.orchestrator.adaptive_literature_step import (
    AdaptiveLiteratureStepOutcome,
    run_adaptive_literature_step,
)
from investment_research.orchestrator.evidence_integrity_pass import (
    FullIntegrityPassInput,
    run_full_evidence_integrity_pass,
)
from investment_research.research import adaptive_literature_acquisition as ala
from investment_research.research.literature_acquisition_adapter import (
    NCBI_EFETCH_URL,
    NCBI_ESEARCH_URL,
)
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import FactCategory, SourceTier
from investment_research.schemas.fact import RawFact, Source
from investment_research.scoring.adaptive_acquisition_plan import (
    READY_MAX_REQUESTS,
    REFERENCE_MODE_NCT_ID,
    AcquisitionPlanStatus,
    AdaptiveAcquisitionPlan,
)

from . import _literature_fixture_support as fx
from ._network_guard import forbid_external_network_autouse  # noqa: F401

RUN_ID = "p43i-run"
TICKER = "P43I"
COMPANY = "Phase 4.3I Test Co"
NCT_ID = "NCT09990888"
#: Must match the PMIDs baked into tests/fixtures/literature_real_format/
#: batch_articleset.xml -- the EFetch fixture response's own parsed
#: content, not merely the ESearch response this test constructs, is what
#: the acquisition path cross-checks "a requested PMID resolved" against.
PMIDS = ["90000015", "90000016"]
_ENV = {"IRA_NCBI_TOOL": "test-tool", "IRA_NCBI_EMAIL": "test@example.com"}
_EPMC_EMPTY = '{"resultList": {"result": []}}'


# =============================================================================
# Helpers (mirror tests/unit/test_adaptive_literature_acquisition.py exactly)
# =============================================================================
def _efetch_url(pmids: list[str]) -> str:
    params = {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract",
        "tool": "test-tool", "email": "test@example.com",
    }
    return f"{NCBI_EFETCH_URL}?{urlencode(params)}"


def _esearch_url(term: str, max_pmids: int) -> str:
    params = {
        "db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids),
        "tool": "test-tool", "email": "test@example.com",
    }
    return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"


def _ready_plan(**overrides) -> AdaptiveAcquisitionPlan:
    kwargs = {
        "status": AcquisitionPlanStatus.READY,
        "reference_mode": REFERENCE_MODE_NCT_ID,
        "nct_id": NCT_ID,
        "pmid": "UNKNOWN",
        "rationale": "ready for test",
        "max_requests": READY_MAX_REQUESTS,
        "requires_external_communication": True,
    }
    kwargs.update(overrides)
    return AdaptiveAcquisitionPlan(**kwargs)


def _non_ready_plan(status: AcquisitionPlanStatus = AcquisitionPlanStatus.UNRESOLVED) -> AdaptiveAcquisitionPlan:
    return AdaptiveAcquisitionPlan(status=status, rationale="not ready for test")


def _success_fake_http() -> fx.FakeHttpClient:
    term = f"{NCT_ID}[si]"
    return fx.FakeHttpClient(
        responses={
            _esearch_url(term, 20): fx.ok(fx.esearch_response(PMIDS)),
            _efetch_url(PMIDS): fx.ok(fx.fixture_text("batch_articleset.xml")),
            fx.europepmc_search_url(PMIDS[0]): fx.ok(_EPMC_EMPTY),
            fx.europepmc_search_url(PMIDS[1]): fx.ok(_EPMC_EMPTY),
        }
    )


def _incomplete_fake_http() -> fx.FakeHttpClient:
    """ESearch/EFetch succeed but Europe PMC search is never registered
    for either PMID, which a Fake client answers as a 404 -- a genuine
    provider-failure shape, never a crash, that leaves
    ``coverage_complete=False``."""
    term = f"{NCT_ID}[si]"
    return fx.FakeHttpClient(
        responses={
            _esearch_url(term, 20): fx.ok(fx.esearch_response(PMIDS)),
            _efetch_url(PMIDS): fx.ok(fx.fixture_text("batch_articleset.xml")),
        }
    )


def _step(**overrides):
    kwargs = {
        "repo": None,
        "run_id": RUN_ID,
        "ticker": TICKER,
        "company_name": COMPANY,
        "plan": _ready_plan(),
        "verified_facts": (),
        "sources": (),
        "base_evidence_available": True,
        "http_client": None,
        "env": None,
    }
    kwargs.update(overrides)
    return run_adaptive_literature_step(**kwargs)


def _source(source_id: str, *, url: str, tier: SourceTier = SourceTier.TIER_1) -> Source:
    return Source(source_id=source_id, url=url, title="Doc", tier=tier, published_date="2026-01-01")


def _stage1_fact(repo, claim: str, source: Source, run_id: str = RUN_ID):
    from investment_research.agents.fact_collector import FactCollectorAgent

    raw = RawFact(ticker=TICKER, category=FactCategory.CLINICAL, claim=claim, source=source, company_claim=True)
    result = CollectionResult(collector="test_collector", raw_facts=[raw], sources=[source])
    agent = FactCollectorAgent([result])
    output = agent.execute(
        AgentInput(agent_id=agent.agent_id, run_id=run_id, ticker=TICKER, company_name=COMPANY)
    )
    return list(output.facts)


# =============================================================================
# 1. Non-READY / explicit-override: never attempted, zero effect
# =============================================================================
@pytest.mark.parametrize(
    "status",
    [
        AcquisitionPlanStatus.NO_ACTION,
        AcquisitionPlanStatus.UNRESOLVED,
        AcquisitionPlanStatus.CONFLICTED,
        AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE,
        AcquisitionPlanStatus.REFUSED,
    ],
)
def test_non_ready_plan_is_never_attempted_and_bridge_never_called(repo, monkeypatch, status):
    calls = []
    monkeypatch.setattr(
        ala, "execute_adaptive_literature_plan",
        lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    existing = (_stage1_fact(repo, "An existing claim.", _source("s1", url="https://www.sec.gov/one")))[0]
    result = _step(
        repo=repo, plan=_non_ready_plan(status), verified_facts=(existing,),
        http_client=_success_fake_http(), env=_ENV,
    )
    assert result.outcome is AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_NON_READY
    assert result.verified_facts == (existing,)
    assert result.new_chunks == ()
    assert result.blocking_reasons == ()
    assert calls == []
    assert repo.adaptive_literature_started(RUN_ID) is None


# =============================================================================
# 2. READY, no runner injected: never attempted, bridge never called
# =============================================================================
@pytest.mark.parametrize("http_client,env", [(None, _ENV), (None, None)])
def test_ready_plan_without_runner_is_never_attempted(repo, monkeypatch, http_client, env):
    calls = []
    monkeypatch.setattr(
        ala, "execute_adaptive_literature_plan",
        lambda *a, **k: calls.append(1),
    )
    result = _step(repo=repo, http_client=http_client, env=env)
    assert result.outcome is AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_NO_RUNNER
    assert result.blocking_reasons == ()
    assert calls == []
    assert repo.adaptive_literature_started(RUN_ID) is None


def test_ready_plan_with_env_but_no_http_client_is_never_attempted(repo, monkeypatch):
    calls = []
    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", lambda *a, **k: calls.append(1))
    result = _step(repo=repo, http_client=None, env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_NO_RUNNER
    assert calls == []


# =============================================================================
# 3. Base evidence unavailable: never attempted even with a working runner
# =============================================================================
def test_base_evidence_unavailable_is_never_attempted(repo, monkeypatch):
    calls = []
    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", lambda *a, **k: calls.append(1))
    result = _step(repo=repo, base_evidence_available=False, http_client=_success_fake_http(), env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_BASE_EVIDENCE_UNAVAILABLE
    assert calls == []
    assert repo.adaptive_literature_started(RUN_ID) is None


# =============================================================================
# 4. COMPLETE: Fact/Source/Chunk projection, both AgentRunRecords, Integrity
# =============================================================================
def test_complete_fetch_converts_facts_runs_integrity_and_saves_snapshots(repo):
    existing_source = _source("s_old", url="https://www.sec.gov/old")
    existing = _stage1_fact(repo, "An existing company claim.", existing_source)[0]
    # Give the existing fact a real Evidence Integrity pass first (INITIAL),
    # mirroring what Stage 2 would already have done before this step runs.
    pass1 = run_full_evidence_integrity_pass(
        FullIntegrityPassInput(
            ticker=TICKER, company_name=COMPANY, run_id=RUN_ID,
            pre_integrity_facts=(existing,), sources=(existing_source,),
        ),
        repo,
    )
    verified_before = pass1.verified_facts

    result = _step(
        repo=repo, verified_facts=verified_before, sources=(existing_source,),
        http_client=_success_fake_http(), env=_ENV,
    )

    assert result.outcome is AdaptiveLiteratureStepOutcome.COMPLETE
    assert result.blocking_reasons == ()
    assert len(result.verified_facts) > len(verified_before)
    assert all(f.fact_id in {g.fact_id for g in result.verified_facts} for f in verified_before)
    assert result.new_chunks != ()
    assert all(c.document.text for c in result.new_chunks)
    assert result.new_sources != ()
    new_chunk_doc_ids = {c.doc_id for c in result.new_chunks}
    new_source_ids = {s.source_id for s in result.new_sources}
    assert new_chunk_doc_ids or new_source_ids  # literature evidence produced at least one of each

    audit_ids = {r.agent_id for r in result.agent_records}
    assert audit_ids == {"fact_collector_adaptive", "evidence_integrity_adaptive"}

    rows = repo.conn.execute(
        "SELECT agent_id FROM agent_runs WHERE run_id = ? ORDER BY agent_id", (RUN_ID,),
    ).fetchall()
    # "evidence_integrity" (the INITIAL pass this test ran manually above)
    # must survive untouched alongside the two ADAPTIVE rows -- no PK
    # collision.
    assert {"fact_collector_adaptive", "evidence_integrity_adaptive", "evidence_integrity"} <= {
        r["agent_id"] for r in rows
    }

    assert repo.adaptive_literature_started(RUN_ID) is not None
    collect_snap = repo.adaptive_collect_snapshot_for_resume(RUN_ID)
    assert collect_snap is not None
    assert collect_snap.execution_status == "COMPLETE"
    assert len(collect_snap.chunks) == len(result.new_chunks)
    verify_snap = repo.adaptive_verify_snapshot_for_resume(RUN_ID)
    assert verify_snap is not None
    assert verify_snap.blocking_reasons == []


def test_fact_collector_adaptive_never_collides_with_initial_audit_row(repo):
    """agent_runs's PRIMARY KEY (run_id, agent_id) means a naive second
    FactCollectorAgent call under the same run_id would silently overwrite
    the Stage-1 row -- proves the ADAPTIVE audit identity keeps both."""
    from investment_research.schemas.agent_io import AgentRunRecord

    initial_record = AgentRunRecord(
        run_id=RUN_ID, agent_id="fact_collector", status="OK", started_at="t0", finished_at="t1",
        duration_ms=1, fact_count=1, error_count=0,
    )
    repo.save_agent_run(initial_record)

    result = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.COMPLETE

    rows = repo.conn.execute(
        "SELECT agent_id, fact_count FROM agent_runs WHERE run_id = ? ORDER BY agent_id", (RUN_ID,),
    ).fetchall()
    agent_ids = {r["agent_id"] for r in rows}
    assert "fact_collector" in agent_ids
    assert "fact_collector_adaptive" in agent_ids
    by_id = {r["agent_id"]: r for r in rows}
    assert by_id["fact_collector"]["fact_count"] == 1  # the INITIAL row survived untouched


def test_quarantined_malformed_source_from_adaptive_fetch_is_reported(repo, monkeypatch):
    """A malformed Source introduced by the adaptive fetch itself must be
    quarantined by the ADAPTIVE Integrity pass, exactly like Stage 2 would
    quarantine one from Stage 1 -- proven here by corrupting one Source's
    URL on the bundle's own CollectionResult before Integrity runs."""
    import dataclasses

    http = _success_fake_http()
    real_run = ala.run_literature_pipeline_acquisition

    def _corrupting_run(*args, **kwargs):
        bundle = real_run(*args, **kwargs)
        if bundle.collection_result is not None and bundle.collection_result.sources:
            # Source is frozen -- replace the list entry with a corrupted
            # copy rather than mutating it in place.
            bundle.collection_result.sources[0] = dataclasses.replace(
                bundle.collection_result.sources[0], url="not-a-well-formed-url",
            )
        return bundle

    monkeypatch.setattr(ala, "run_literature_pipeline_acquisition", _corrupting_run)
    result = _step(repo=repo, http_client=http, env=_ENV)
    assert result.outcome in (
        AdaptiveLiteratureStepOutcome.COMPLETE, AdaptiveLiteratureStepOutcome.INCOMPLETE,
    )
    assert len(result.quarantined_sources) >= 1


# =============================================================================
# 5. INCOMPLETE: partial fetch runs through Integrity but still blocks
# =============================================================================
def test_incomplete_fetch_runs_integrity_but_still_blocks_action(repo):
    result = _step(repo=repo, http_client=_incomplete_fake_http(), env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.INCOMPLETE
    assert result.blocking_reasons != ()
    # Partial facts ARE still converted/verified -- coverage incompleteness
    # is reported, never silently treated as "nothing happened".
    verify_snap = repo.adaptive_verify_snapshot_for_resume(RUN_ID)
    assert verify_snap is not None
    assert verify_snap.execution_status == "INCOMPLETE"
    assert verify_snap.blocking_reasons != []


# =============================================================================
# 6. REFUSED: attempted, zero HTTP reaching the executor, still blocks,
#    and a later resume restores REFUSED without re-fetching
# =============================================================================
def test_refused_plan_blocks_and_resume_never_refetches(repo, monkeypatch):
    # A READY plan that independently fails this bridge's own re-verification
    # (a non-canonical nct_id) is REFUSED before any HTTP call.
    bad_plan = _ready_plan(nct_id="nct09990888")  # lower-case -- not canonical
    http = _success_fake_http()
    result = _step(repo=repo, plan=bad_plan, http_client=http, env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.REFUSED
    assert result.blocking_reasons != ()
    assert http.requested_urls == []  # zero HTTP, per the bridge's own contract

    def _explode(*a, **k):
        raise AssertionError("must not re-fetch on resume")

    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", _explode)
    resumed = _step(repo=repo, plan=bad_plan, http_client=_success_fake_http(), env=_ENV)
    assert resumed.outcome is AdaptiveLiteratureStepOutcome.REFUSED
    assert resumed.blocking_reasons != ()


# =============================================================================
# 7. Runner exception: never crashes, treated as ambiguous, fails closed
# =============================================================================
def test_transport_failure_absorbed_by_the_adapter_is_a_normal_incomplete(repo):
    """A transport-level failure (connection error) is already caught and
    converted to a structured, non-``coverage_complete`` bundle by the
    existing Acquisition Adapter layer (``AcquisitionExecutor``) well
    below this step -- never reaches this step as a raised exception at
    all. Confirms this step reports it as a normal, attempted INCOMPLETE,
    not an ambiguous or crashed state."""
    raising = fx.RaisingHttpClient()
    result = _step(repo=repo, http_client=raising, env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.INCOMPLETE
    assert result.blocking_reasons != ()
    assert repo.adaptive_collect_snapshot_for_resume(RUN_ID) is not None


def test_uncontained_runner_exception_is_ambiguous_and_never_crashes(repo, monkeypatch):
    """A failure that escapes ALL the way past the bridge itself (never
    absorbed into a structured ``AdaptiveLiteratureExecution``) -- this
    step's own try/except around ``execute_adaptive_literature_plan`` is
    what this test exercises directly."""

    def _explode(*a, **k):
        raise RuntimeError("urlopen error for https://eutils.ncbi.nlm.nih.gov/secret?tool=abc")

    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", _explode)
    result = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE
    assert result.blocking_reasons != ()
    combined = " ".join(result.blocking_reasons)
    assert "RuntimeError" in combined
    assert "secret" not in combined and "eutils.ncbi.nlm.nih.gov" not in combined
    assert repo.adaptive_literature_started(RUN_ID) is not None
    assert repo.adaptive_collect_snapshot_for_resume(RUN_ID) is None

    # A later resume of this SAME ambiguous state must never auto-retry.
    def _must_not_be_called(*a, **k):
        raise AssertionError("must not retry an ambiguous resume state automatically")

    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", _must_not_be_called)
    resumed = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert resumed.outcome is AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE


# =============================================================================
# 8. Ambiguous resume state: started marker alone, no collect snapshot
# =============================================================================
def test_started_marker_alone_fails_closed_and_is_never_auto_retried(repo, monkeypatch):
    repo.save_adaptive_literature_started(RUN_ID, NCT_ID, READY_MAX_REQUESTS)

    calls = []
    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", lambda *a, **k: calls.append(1))
    result = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE
    assert result.blocking_reasons != ()
    assert calls == []  # resume-state is judged before the runner-injection check

    # A second, later resume of the SAME ambiguous state must also never
    # auto-retry and must stay blocked -- never silently "self-heals".
    result2 = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert result2.outcome is AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE
    assert result2.blocking_reasons != ()
    assert calls == []


def test_started_marker_alone_is_ambiguous_even_with_no_runner_injected(repo):
    """Resume-state must be judged BEFORE the runner-injection check --
    an ambiguous state is never reclassified as a normal no-op merely
    because this invocation also happens to have no runner configured."""
    repo.save_adaptive_literature_started(RUN_ID, NCT_ID, READY_MAX_REQUESTS)
    result = _step(repo=repo, http_client=None, env=None)
    assert result.outcome is AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE
    assert result.blocking_reasons != ()


# =============================================================================
# 9. Collect snapshot exists, verify snapshot does not: resume skips the
#    fetch and runs only the second Integrity pass
# =============================================================================
def test_collect_snapshot_without_verify_snapshot_skips_refetch(repo, monkeypatch):
    first = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert first.outcome is AdaptiveLiteratureStepOutcome.COMPLETE

    # Simulate "crashed between collect snapshot and verify snapshot" by
    # deleting the verify snapshot row this run already wrote.
    repo.conn.execute(
        "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = ?",
        (RUN_ID, "_adaptive_verify_snapshot"),
    )
    repo.conn.commit()
    assert repo.adaptive_verify_snapshot_for_resume(RUN_ID) is None
    assert repo.adaptive_collect_snapshot_for_resume(RUN_ID) is not None

    def _explode(*a, **k):
        raise AssertionError("must not re-fetch when a collect snapshot already exists")

    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", _explode)
    second = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert second.outcome is AdaptiveLiteratureStepOutcome.COMPLETE
    assert second.verified_facts == first.verified_facts
    assert {c.chunk_id for c in second.new_chunks} == {c.chunk_id for c in first.new_chunks}
    assert repo.adaptive_verify_snapshot_for_resume(RUN_ID) is not None


# =============================================================================
# 10. Verify snapshot exists: resume restores directly, zero re-fetch,
#     zero re-Integrity, Chunk/Fact equivalence with the original run
# =============================================================================
def test_verify_snapshot_restores_without_refetch_or_reintegrity(repo, monkeypatch):
    first = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert first.outcome is AdaptiveLiteratureStepOutcome.COMPLETE

    def _explode(*a, **k):
        raise AssertionError("must not call the bridge once a verify snapshot exists")

    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", _explode)

    def _explode_integrity(*a, **k):
        raise AssertionError("must not re-run Evidence Integrity once a verify snapshot exists")

    monkeypatch.setattr(step_module, "run_full_evidence_integrity_pass", _explode_integrity)

    resumed = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert resumed.outcome is AdaptiveLiteratureStepOutcome.COMPLETE
    assert {f.fact_id for f in resumed.verified_facts} == {f.fact_id for f in first.verified_facts}
    assert {c.chunk_id for c in resumed.new_chunks} == {c.chunk_id for c in first.new_chunks}
    # Chunk CONTENT (not just id) survives the round-trip losslessly --
    # this is the semantic-equivalence guarantee Domain Agents' Evidence
    # Packs/TraceabilityIndex/the report depend on.
    first_by_id = {c.chunk_id: c for c in first.new_chunks}
    for c in resumed.new_chunks:
        assert c.text == first_by_id[c.chunk_id].text
        assert c.document.text == first_by_id[c.chunk_id].document.text
        assert c.document.doc_id == first_by_id[c.chunk_id].document.doc_id
    assert resumed.agent_records == ()  # no new agent_runs rows on a pure restore
    # Sources round-trip too (Phase 4.3I correction 1) -- same set,
    # restored from the collect snapshot the verify snapshot depends on.
    assert {s.source_id for s in resumed.new_sources} == {s.source_id for s in first.new_sources}
    assert resumed.new_sources != ()


# =============================================================================
# 10b. verify snapshot exists but its collect snapshot dependency is
#      cleanly missing: fails closed, never restores partial evidence
#      as if complete, never re-fetches, never re-runs Integrity
# =============================================================================
def test_verify_snapshot_without_collect_snapshot_fails_closed(repo, monkeypatch):
    first = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert first.outcome is AdaptiveLiteratureStepOutcome.COMPLETE
    assert repo.adaptive_verify_snapshot_for_resume(RUN_ID) is not None
    assert repo.adaptive_collect_snapshot_for_resume(RUN_ID) is not None

    # Simulate the collect snapshot row being cleanly removed (e.g. an
    # operator pruning large Document bodies) while the verify snapshot
    # -- which depends on it -- survives.
    repo.conn.execute(
        "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = ?",
        (RUN_ID, "_adaptive_collect_snapshot"),
    )
    repo.conn.commit()
    assert repo.adaptive_collect_snapshot_for_resume(RUN_ID) is None
    assert repo.adaptive_verify_snapshot_for_resume(RUN_ID) is not None

    def _explode(*a, **k):
        raise AssertionError("must not re-fetch when the dependency state is inconsistent")

    monkeypatch.setattr(step_module, "execute_adaptive_literature_plan", _explode)

    def _explode_integrity(*a, **k):
        raise AssertionError("must not re-run Evidence Integrity on an inconsistent snapshot state")

    monkeypatch.setattr(step_module, "run_full_evidence_integrity_pass", _explode_integrity)

    result = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE
    assert result.blocking_reasons != ()
    assert result.new_chunks == ()
    assert result.new_sources == ()
    # The ORIGINAL (pre-adaptive) verified_facts are returned unchanged --
    # never the previously-restored, now-unbacked verified_facts.
    assert result.verified_facts == ()


# =============================================================================
# 11. Snapshot corruption: distinguishable from "missing", fails closed
# =============================================================================
def test_corrupted_collect_snapshot_fails_closed_distinctly_from_missing(repo):
    repo.save_adaptive_literature_started(RUN_ID, NCT_ID, READY_MAX_REQUESTS)
    repo.conn.execute(
        """INSERT OR REPLACE INTO run_checkpoints(run_id, stage, stage_index, status,
                                                  payload, fact_count, created_at)
           VALUES(?,?,?,?,?,?,?)""",
        (RUN_ID, "_adaptive_collect_snapshot", -1, "OK", "{not valid json", 0, "2026-01-01T00:00:00+00:00"),
    )
    repo.conn.commit()
    result = _step(repo=repo, http_client=_success_fake_http(), env=_ENV)
    assert result.outcome is AdaptiveLiteratureStepOutcome.SNAPSHOT_CORRUPTED
    assert result.blocking_reasons != ()
