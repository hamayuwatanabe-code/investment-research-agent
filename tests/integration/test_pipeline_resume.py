"""Phase 4.3C Correction 2: end-to-end ``--resume`` tests.

Correction 1 fixed (in ``evidence_integrity_pass.py``) the fact that a
call to ``run_full_evidence_integrity_pass`` could never be silently
bypassed. It also found, but did not fix in ``Pipeline.run()`` itself,
that the ``--resume`` wiring of that era would have fed the WRONG (empty)
local variable into that bypass, discarding restored facts. Correction 2
fixes the actual ``Pipeline.run()`` resume path and proves it here,
against the REAL ``Pipeline.run()``, a REAL (in-memory) ``Repository``,
and REAL checkpoint/resume-plan machinery -- never a reimplementation of
either.

A genuine mid-run crash is simulated by monkeypatching exactly one
function/method to raise partway through an otherwise-real
``Pipeline.run()`` call, so everything before the injected raise (fact
persistence, checkpoint commits) happens for real, through the real
``Repository``. This is the only monkeypatching used; the two calls that
matter for each scenario -- the crash and the resume -- are both genuine
``Pipeline.run()`` invocations.

These tests deliberately do NOT use the DEMOBIO fixture set the way
tests/integration/test_pipeline.py does: DEMOBIO's own dates are curated
for a single realistic (non-resumed) run and routinely trip
orchestrator.resume's genuine, independently-tested (test_phase2_gates.py)
per-category staleness gate against any single fixed "today" -- forcing a
full restart regardless of checkpoints, which is a different code path
this phase does not touch. A small, hand-built CollectionResult with
dates controlled to stay within every touched category's freshness
window isolates the collect/verify resume boundary this Correction is
actually about.

Section map:
A. A normal (non-resumed) run: Evidence Integrity executes exactly once,
   filed under the INITIAL identity.
B. Resume after "collect" but before "verify": the pass reruns exactly
   once, on the genuinely-restored, still-NOT_VERIFIED Stage-1 facts --
   never on an empty set.
C. Resume after "verify" already completed: Evidence Integrity does NOT
   rerun (zero executions), and the already-verified restored facts are
   carried forward unchanged.
D. Equivalence: a clean, uninterrupted run and a crashed-then-resumed run
   over the same input data produce the same verified fact IDs,
   quarantine results, and status/failures (run-scoped values such as
   run_id/timestamps excluded).
E. Regression: the exact defect Correction 1 identified in the base
   commit (``collector_output is None`` combined with an empty local
   substituted for ``resume_plan.restored_facts``, silently discarding
   every restored fact) cannot reoccur.
"""

from __future__ import annotations

from datetime import date

import pytest

import investment_research.orchestrator.pipeline as pipeline_module
import investment_research.storage.repository as repository_module
from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.schemas.enums import FactCategory, SourceTier, VerifiedStatus
from investment_research.schemas.fact import RawFact, Source
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)
TICKER = "DEMOBIO"
COMPANY = "Demo Biotherapeutics Inc"


# =============================================================================
# Helpers
# =============================================================================
def _run_kwargs() -> dict:
    """A small, hand-built, deterministic CollectionResult -- two facts on
    the SAME claim, one a company filing (TIER_1, company_claim), one an
    independent registry (TIER_2, non-company) -- so Evidence Integrity's
    real corroboration logic has something genuine to do (the company
    fact gets independently confirmed and upgraded), without any of
    DEMOBIO's own dozens of categories/dates to trip the staleness gate.
    Dates are close to TODAY and every touched category
    (schemas.enums.FactCategory.OTHER) uses orchestrator.resume's generous
    DEFAULT_FRESHNESS_DAYS (90), well clear of TODAY regardless."""
    company_source = Source(
        source_id="src_company",
        url="https://www.sec.gov/demo-resume",
        title="Form 10-Q",
        tier=SourceTier.TIER_1,
        published_date="2026-09-01",
    )
    independent_source = Source(
        source_id="src_indep",
        url="https://www.example-registry.test/entry",
        title="Registry Entry",
        tier=SourceTier.TIER_2,
        published_date="2026-09-02",
    )
    claim = "The company reported quarterly revenue of $10 million."
    raw_facts = [
        RawFact(
            ticker=TICKER, category=FactCategory.OTHER, claim=claim,
            source=company_source, company_claim=True,
        ),
        RawFact(
            ticker=TICKER, category=FactCategory.OTHER, claim=claim,
            source=independent_source, company_claim=False,
        ),
    ]
    collection_result = CollectionResult(
        collector="resume_test_collector",
        raw_facts=raw_facts,
        sources=[company_source, independent_source],
    )
    return {
        "ticker": TICKER,
        "company_name": COMPANY,
        "collection_results": [collection_result],
    }


def _new_repo() -> Repository:
    """A brand-new, independent in-memory Repository.

    Needed whenever a test runs two DIFFERENT run_ids over the SAME
    content: Fact.fact_id is content-derived, not run_id-scoped (see
    schemas/fact.py's make_fact_id), and Repository.save_fact looks up
    the latest row for a fact_id WITHOUT filtering by run_id. Two runs
    sharing one repo would make the second run's save_fact calls silent
    no-ops for any fact whose content is byte-identical to one the first
    run already saved -- and that row would still carry the FIRST run's
    run_id, making Repository.facts_for_resume(second_run_id) miss it
    entirely. A separate repo per run_id keeps run_id boundaries clean,
    which is what "the same resume/checkpoint API a real run uses"
    actually requires when comparing two runs side by side.
    """
    conn = open_db(":memory:")
    return Repository(conn)


def _count_agent_runs(monkeypatch) -> list:
    """Wraps EvidenceIntegrityAgent.run itself (never Agent.execute) so
    the count is a genuine, independent observation of how many times the
    agent's own logic executed, spanning as many Pipeline.run() calls as
    the caller likes -- mirrors tests/unit/test_evidence_integrity_pass.py's
    own helper of the same name."""
    calls: list = []
    original_run = EvidenceIntegrityAgent.run

    def counting_run(self, data):
        calls.append(data)
        return original_run(self, data)

    monkeypatch.setattr(EvidenceIntegrityAgent, "run", counting_run)
    return calls


def _integrity_agent_run_rows(repo: Repository, run_id: str) -> list:
    return repo.conn.execute(
        "SELECT agent_id, fact_count FROM agent_runs "
        "WHERE run_id = ? AND agent_id LIKE 'evidence_integrity%' ORDER BY agent_id",
        (run_id,),
    ).fetchall()


def _crash_inside_evidence_integrity_pass(monkeypatch) -> None:
    """Simulates a crash WHILE Evidence Integrity is running: real Stage 1
    (collection, fact persistence, the "collect" checkpoint) all happen
    for real first; this raises before run_full_evidence_integrity_pass's
    own body -- agent execution, fact persistence, the "verify"
    checkpoint -- ever starts. Scoped to the caller's monkeypatch fixture,
    so it un-patches at test teardown; callers needing it un-patched
    sooner use monkeypatch.context() themselves."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash inside Evidence Integrity")

    monkeypatch.setattr(pipeline_module, "run_full_evidence_integrity_pass", _boom)


def _crash_after_verify_checkpoint(monkeypatch) -> None:
    """Simulates a crash strictly AFTER "verify" is durably checkpointed:
    the real Repository.save_checkpoint still runs for "collect" and
    "verify" (both persisted for real); the next checkpoint attempt
    (this test's NullResearchProvider means the escalation stage finds no
    usable research channel, so this is "escalate") raises instead."""
    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_verify(self, checkpoint):
        if checkpoint.stage not in ("collect", "verify"):
            raise RuntimeError("simulated crash after verify checkpoint")
        return original_save_checkpoint(self, checkpoint)

    monkeypatch.setattr(repository_module.Repository, "save_checkpoint", _crash_after_verify)


# =============================================================================
# A. Normal run
# =============================================================================
def test_normal_run_executes_evidence_integrity_exactly_once_as_initial(repo, monkeypatch):
    calls = _count_agent_runs(monkeypatch)
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-a"

    result = pipeline.run(**_run_kwargs(), run_id=run_id)

    assert len(calls) == 1
    assert result.resume_plan is None
    assert len(result.bus.facts) == 2
    rows = _integrity_agent_run_rows(repo, run_id)
    assert [r["agent_id"] for r in rows] == ["evidence_integrity"]
    assert not any(f.startswith("RESUMED RUN") for f in result.failures)


# =============================================================================
# B. Resume after "collect", before "verify"
# =============================================================================
def test_resume_before_verify_reruns_integrity_once_on_restored_stage1_facts(repo, monkeypatch):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-b"

    with monkeypatch.context() as m:
        _crash_inside_evidence_integrity_pass(m)
        with pytest.raises(RuntimeError, match="simulated crash inside Evidence Integrity"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    # "collect" alone is checkpointed; Stage-1 facts are already persisted
    # (this correction's own change -- previously nothing was written
    # until inside the pass), and they are genuinely pre-Integrity.
    completed_stages = {c.stage for c in repo.checkpoints(run_id) if c.status == "OK"}
    assert completed_stages == {"collect"}
    pre_crash_facts = repo.facts_for_resume(run_id)
    assert len(pre_crash_facts) == 2
    assert all(f.verified_status == VerifiedStatus.NOT_VERIFIED for f in pre_crash_facts)

    calls = _count_agent_runs(monkeypatch)
    resumed = pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert len(calls) == 1  # Evidence Integrity ran exactly once in this call
    assert resumed.resume_plan is not None
    assert resumed.resume_plan.restored_facts  # never empty
    assert len(resumed.resume_plan.restored_facts) == 2
    assert {f.fact_id for f in resumed.resume_plan.restored_facts} == {
        f.fact_id for f in pre_crash_facts
    }

    assert len(resumed.bus.facts) == 2
    assert {f.fact_id for f in resumed.bus.facts} == {f.fact_id for f in pre_crash_facts}
    # The facts that survive are no longer stuck at the pre-Integrity
    # baseline -- Evidence Integrity genuinely ran over them (corroborated
    # company claim upgraded to VERIFIED).
    assert any(f.verified_status == VerifiedStatus.VERIFIED for f in resumed.bus.facts)

    rows = _integrity_agent_run_rows(repo, run_id)
    assert [r["agent_id"] for r in rows] == ["evidence_integrity"]  # one row, one execution
    assert any("collection was not re-executed" in f for f in resumed.failures)


# =============================================================================
# C. Resume after "verify" already completed
# =============================================================================
def test_resume_after_verify_complete_does_not_rerun_integrity(repo, monkeypatch):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-c"

    with monkeypatch.context() as m:
        _crash_after_verify_checkpoint(m)
        with pytest.raises(RuntimeError, match="simulated crash after verify checkpoint"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    completed_stages = {c.stage for c in repo.checkpoints(run_id) if c.status == "OK"}
    assert completed_stages == {"collect", "verify"}
    verified_before_resume = repo.facts_for_resume(run_id)
    assert len(verified_before_resume) == 2
    assert any(f.verified_status == VerifiedStatus.VERIFIED for f in verified_before_resume)
    rows_before = _integrity_agent_run_rows(repo, run_id)
    assert [r["agent_id"] for r in rows_before] == ["evidence_integrity"]

    calls = _count_agent_runs(monkeypatch)
    resumed = pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert len(calls) == 0  # Evidence Integrity did NOT execute this call
    assert resumed.resume_plan is not None
    assert resumed.resume_plan.restored_facts
    assert {f.fact_id for f in resumed.bus.facts} == {f.fact_id for f in verified_before_resume}
    assert len(resumed.bus.facts) == len(verified_before_resume)
    assert any("Evidence Integrity was not re-executed" in f for f in resumed.failures)

    # Still exactly one stored record -- the resumed call added none.
    rows_after = _integrity_agent_run_rows(repo, run_id)
    assert [r["agent_id"] for r in rows_after] == ["evidence_integrity"]


# =============================================================================
# D. Equivalence: uninterrupted run vs. crashed-then-resumed run
# =============================================================================
def test_resume_and_full_run_are_semantically_equivalent(monkeypatch):
    # Two independent repos (see _new_repo's own docstring): fact_id is
    # content-derived, not run_id-scoped, so sharing one repo across two
    # different run_ids for the SAME content would make the second run's
    # saves silently no-op against the first run's rows.
    full_repo = _new_repo()
    resumed_repo = _new_repo()

    full_pipeline = Pipeline(full_repo, NullSearchProvider(), today=TODAY)
    full = full_pipeline.run(**_run_kwargs(), run_id="resume-d-full")

    resumed_pipeline = Pipeline(resumed_repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-d-resumed"
    with monkeypatch.context() as m:
        _crash_inside_evidence_integrity_pass(m)
        with pytest.raises(RuntimeError):
            resumed_pipeline.run(**_run_kwargs(), run_id=run_id)
    resumed = resumed_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    # run_id, timestamps, and other run-scoped values are explicitly out
    # of scope for this comparison.
    assert {f.fact_id for f in resumed.bus.facts} == {f.fact_id for f in full.bus.facts}
    assert {q.source_id for q in resumed.quarantined_sources} == {
        q.source_id for q in full.quarantined_sources
    }
    assert resumed.context.status == full.context.status
    resumed_failures = [f for f in resumed.failures if not f.startswith("RESUMED RUN:")]
    assert resumed_failures == full.failures


# =============================================================================
# E. Regression: a5d5caa's identified defect cannot reoccur
# =============================================================================
def test_resume_after_collect_never_silently_discards_restored_facts_regression_a5d5caa(
    repo, monkeypatch
):
    """Correction 1 found (but, in the base commit a5d5caa, did not fix in
    Pipeline.run() itself) that the pre-existing --resume wiring set
    ``collector_output = None`` on the skip-collection branch and then fed
    the (consequently empty) ``raw_facts`` LOCAL VARIABLE -- never
    ``resume_plan.restored_facts`` -- into the since-removed bypass path,
    which took an empty pass-through as its verified set. A real
    ``--resume`` invocation would have silently ended up with ZERO facts
    regardless of how many were actually restored. This test drives the
    real collect-then-crash-then-resume sequence and asserts the resumed
    run's final fact count is never zero and matches what was actually
    collected before the simulated crash -- the exact failure mode this
    guards against."""
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-e-regression"

    with monkeypatch.context() as m:
        _crash_inside_evidence_integrity_pass(m)
        with pytest.raises(RuntimeError):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    pre_crash_facts = repo.facts_for_resume(run_id)
    pre_crash_count = len(pre_crash_facts)
    assert pre_crash_count > 0  # collection genuinely produced facts before the crash

    resumed = pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    # The historical bug: verified_facts silently became [] here.
    assert resumed.bus.facts != []
    assert len(resumed.bus.facts) > 0
    assert len(resumed.bus.facts) == pre_crash_count
    assert resumed.resume_plan is not None
    assert len(resumed.resume_plan.restored_facts) == pre_crash_count
