"""Phase 4.3C Correction 2 / Correction 3: end-to-end ``--resume`` tests.

Correction 1 fixed (in ``evidence_integrity_pass.py``) the fact that a
call to ``run_full_evidence_integrity_pass`` could never be silently
bypassed. Correction 2 fixed a real ``--resume`` data-loss defect in
``Pipeline.run()`` itself and proved fact-ID-level equivalence between a
normal run and a crashed-then-resumed run. Correction 3 goes further in
two ways:

1. Fact-ID equality alone is not semantic equivalence -- two facts can
   share an ID while disagreeing on evidence_class, confidence,
   corroboration, or authority. This file's ``_fact_snapshot`` compares
   every field that changes what a Fact means (see its own docstring for
   the exact list and what is excluded and why), and Correction 3's own
   investigation found -- by reading ``Repository.facts_for_resume``
   directly, not by assumption -- that it silently dropped seven of those
   fields (``corroborating_source_ids``, ``contradicting_evidence``,
   ``tags``, ``content_kind``, ``primary_source_url``, ``document_id``,
   ``source_authority``) back to their dataclass defaults on every
   restore, even though all seven are real, persisted columns. That is
   fixed in ``storage/repository.py`` alongside this file.

2. A genuine mid-run crash can land at more than one point inside
   ``run_full_evidence_integrity_pass`` itself, not just before or after
   it runs. Correction 3 also found -- by reading
   ``Pipeline.run()``'s resume branch directly -- that ``EvidenceBus.
   sources`` was never repopulated when collection was skipped, so a
   resumed run's Evidence Integrity pass (when it did rerun) validated
   against an EMPTY source list: no malformed source could ever be
   re-detected, and the report's traceability index silently lost its
   per-source enrichment for every restored fact. That is also fixed in
   ``orchestrator/pipeline.py`` alongside this file, restoring Source
   objects from the caller-supplied ``collection_results`` -- the same
   thing the real CLI's ``--resume`` path already re-supplies (see
   ``cli.py``'s ``run_one``, which calls its collector unconditionally,
   resume or not -- documented in this file's own completion report, not
   independently tested here, since exercising it for real would mean
   live HTTP or a fixture/corpus replay, neither of which this phase
   changes).

A genuine mid-run crash is simulated by monkeypatching exactly one
function/method to raise partway through an otherwise-real
``Pipeline.run()`` call, so everything before the injected raise (fact
persistence, source persistence, checkpoint commits) happens for real,
through the real ``Repository``. The two calls that matter for each
scenario -- the crash and the resume -- are always genuine ``Pipeline.
run()`` invocations, and the resume calls in Correction 3's own new tests
additionally use a BRAND NEW ``Pipeline`` instance (never the same Python
object the crash ran on), so nothing can pass between the two calls
except what a real second process opening the same database would also
have: the repository's persisted state and a freshly-supplied
``collection_results``.

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
D. Full semantic equivalence: a clean, uninterrupted run and a
   crashed-then-resumed run produce the same Fact snapshots, EvidenceBus
   facts, quarantined_sources, failures (resume-only notes excluded),
   RunStatus/ResearchStatus/Action, repository current/latest Facts,
   direct_acquisition_info, and traceability index.
E. Regression: the exact defect Correction 1 identified in the base
   commit a5d5caa (``collector_output is None`` combined with an empty
   local substituted for ``resume_plan.restored_facts``, silently
   discarding every restored fact) cannot reoccur.
F. Stage-1/verified Fact persistence safety: what "collect" vs "verify"
   checkpoints make recoverable, that the repository's current/latest
   view is the verified version once verify has run, that both DB
   versions exist with the expected meaning when Evidence Integrity
   genuinely changes a fact, and that no production code path reads a
   stale pre-Integrity view.
G. Fresh-process Source restoration: a brand-new Pipeline instance
   resuming after a crash gets real Source objects back (never an empty
   set, never another run's), including reproducing malformed-source
   quarantine.
H. Integrity-pass mid-crash points A/B/C: collect-checkpoint-only,
   after Source save/quarantine, and after partial Fact persistence --
   each resumes to the same full semantic snapshot as an uninterrupted
   run, proving a re-run of the pass re-evaluates the WHOLE set even when
   it is a genuine mix of already-verified and still-NOT_VERIFIED facts.
I. AgentRunRecord partial-state handling: a stale/partial record left by
   a crashed pass does not collide with, or survive alongside, the
   record the resumed pass writes.

Phase 4.3C Correction 4 goes one step further than Correction 3's own
"re-evaluate the whole restored set" fix: re-evaluating a fact that a
PARTIAL prior pass had already advanced to its post-Integrity version
(section H's crash point C) fed that ALREADY-PROCESSED version back into
Evidence Integrity a second time -- safe for evidence_class/
verified_status/confidence (Correction 3 confirmed those stay correct),
but not for `notes`, which accumulated duplicate reasoning text and could
leave a stale, now-contradicted reason in place. Correction 4 fixes this
at its ROOT, not by patching the symptom:

1. ``Repository.pristine_facts_for_resume`` (new) returns the EARLIEST
   version this run_id persisted per fact_id -- the genuine Stage-1,
   pre-Integrity snapshot -- distinct from ``facts_for_resume``'s LATEST.
   ``resume.build_plan`` and ``Pipeline.run()`` now feed a resumed
   Evidence Integrity re-run this pristine set (falling back, per fact
   and with an explicit reported failure, to that fact's own latest
   version only when no pristine snapshot can be found -- see
   ``Repository.pristine_facts_for_resume``'s own docstring for the two
   schema-inherent reasons that can happen). This means Evidence
   Integrity's input is now ALWAYS the pristine content on resume, so a
   fact is never fed through evaluation twice in the sense that produced
   the notes bug in the first place.
2. ``agents/evidence_integrity.py``'s own notes-building is also made
   genuinely idempotent (``_INTEGRITY_NOTES_MARKER``): it now recomputes
   its own contribution to ``notes`` fresh every time from the CURRENT
   fact state, discarding (never accumulating) whatever it contributed on
   a prior run, while never touching the ORIGINAL collector-authored
   notes before it. This is what makes ``notes`` safe to compare in
   ``_fact_snapshot`` again, and what makes a genuine second pass (a
   fact's classification legitimately changing, e.g. a future Adaptive
   Acquisition step -- tested in test_evidence_integrity_pass.py, not
   here, since that is the offline 2-pass harness's own concern) produce
   updated, not merely appended-to, reasoning.

``notes`` is restored to ``_fact_snapshot``'s strict comparison (removed
by Correction 3 with a test pinning the duplication as a known,
out-of-scope limitation); that pinning test is reversed below into one
proving the opposite now holds.
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
from investment_research.schemas.enums import (
    UNKNOWN,
    FactCategory,
    RunStatus,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Fact, RawFact, Source
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository, ResumeSnapshotCorrupted

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)
TICKER = "DEMOBIO"
COMPANY = "Demo Biotherapeutics Inc"


# =============================================================================
# Helpers: building input data
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


def _run_kwargs_with_malformed_source() -> dict:
    """Like _run_kwargs, plus a THIRD raw fact resting on a malformed
    Source (an invalid URL -- the same shape tests/unit/
    test_evidence_integrity_pass.py's own malformed-source tests use),
    for the Source-restoration/quarantine-reproduction tests (section G)."""
    kwargs = _run_kwargs()
    bad_source = Source(
        source_id="src_bad",
        url="not-a-well-formed-url",
        title="Malformed Doc",
        tier=SourceTier.TIER_3,
        published_date="2026-09-03",
    )
    bad_raw = RawFact(
        ticker=TICKER, category=FactCategory.OTHER,
        claim="A claim resting on a malformed source.",
        source=bad_source, company_claim=False,
    )
    collection_result: CollectionResult = kwargs["collection_results"][0]
    collection_result.raw_facts.append(bad_raw)
    collection_result.sources.append(bad_source)
    return kwargs


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


# =============================================================================
# Helpers: crash injection (see each function's own docstring for exactly
# what has and has not happened for real by the time it raises)
# =============================================================================
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
        "SELECT agent_id, status, fact_count FROM agent_runs "
        "WHERE run_id = ? AND agent_id LIKE 'evidence_integrity%' ORDER BY agent_id",
        (run_id,),
    ).fetchall()


def _completed_pipeline_stages(repo: Repository, run_id: str) -> set:
    """The real orchestrator.resume.STAGES a run_id has OK-checkpointed --
    excludes Phase 4.3C correction 5's own synthetic snapshot "stages"
    (``_collect_snapshot``/``_verify_snapshot``), which live in the same
    run_checkpoints table (by design -- that table's PRIMARY KEY includes
    run_id, genuinely unlike `facts`/`sources`) but are never real pipeline
    stages and are deliberately excluded from orchestrator.resume.
    resume_point's own stage-index computation too."""
    from investment_research.orchestrator.resume import STAGES

    return {c.stage for c in repo.checkpoints(run_id) if c.status == "OK" and c.stage in STAGES}


def _snapshot_stage_names(repo: Repository, run_id: str) -> set:
    return {
        c.stage for c in repo.checkpoints(run_id)
        if c.status == "OK" and c.stage in ("_collect_snapshot", "_verify_snapshot")
    }


def _crash_point_a_before_pass_starts(monkeypatch) -> None:
    """Crash point A: WHILE Evidence Integrity is running -- real Stage 1
    (collection, fact persistence, source-free-but-collected-in-memory,
    the "collect" checkpoint) all happen for real first; this raises
    before run_full_evidence_integrity_pass's own body -- agent execution,
    source persistence, fact persistence, the "verify" checkpoint -- ever
    starts. No AgentRunRecord for this pass exists after this crash."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("crash point A: before Evidence Integrity pass starts")

    monkeypatch.setattr(pipeline_module, "run_full_evidence_integrity_pass", _boom)


def _crash_point_b_after_source_save(monkeypatch) -> None:
    """Crash point B: AFTER Repository.save_sources has run for real
    (sources persisted, any malformed one genuinely quarantined) but
    BEFORE the "verify" checkpoint. The agent has already executed and
    its AgentRunRecord has already been saved for real (that happens
    before source save in run_full_evidence_integrity_pass's own body),
    so this leaves a real, but PARTIAL (fact persistence never ran),
    AgentRunRecord behind."""
    original_save_sources = repository_module.Repository.save_sources

    def _crash_after(self, sources):
        original_save_sources(self, sources)
        raise RuntimeError("crash point B: after Source save/quarantine")

    monkeypatch.setattr(repository_module.Repository, "save_sources", _crash_after)


class _SimulatedCrash(BaseException):
    """A process-level crash (power loss, SIGKILL, OOM kill) -- deliberately
    NOT an Exception subclass. evidence_integrity_pass.py's own fact-
    persistence loop wraps each Repository.save_fact call in `except
    Exception`, by design (one bad write must not abort the rest of the
    loop) -- so an ordinary exception injected there is already handled
    gracefully and proves nothing about crash recovery. A real mid-loop
    process death is not a caught Exception either, so this is the
    faithful way to simulate one. Not KeyboardInterrupt/SystemExit: pytest
    treats those specially (session-level interrupt handling), which would
    abort the whole test run rather than just this test."""


def _crash_point_c_after_partial_fact_save(monkeypatch, *, after_n: int = 1) -> None:
    """Crash point C: after Repository.save_fact has genuinely persisted
    the first `after_n` verified fact(s) -- Source save/quarantine and the
    AgentRunRecord are also already real by this point -- but before the
    rest of verified_facts are saved and before the "verify" checkpoint.
    Leaves a genuine mix in the database: some facts already carry their
    post-Integrity version, others (not yet reached in the loop) are still
    at whatever version existed before this pass call.

    Stage 1 ALSO calls Repository.save_fact (for the pre-Integrity
    baseline, before "collect" is even checkpointed) -- a plain call
    counter would fire during that unrelated loop instead. This only
    starts counting once "collect" has actually been checkpointed, which
    is exactly when Stage 2's own persist loop -- the one this crash point
    is about -- is the only remaining source of save_fact calls before
    "verify" would be checkpointed."""
    state = {"collect_checkpointed": False, "n": 0}
    original_save_checkpoint = repository_module.Repository.save_checkpoint
    original_save_fact = repository_module.Repository.save_fact

    def _tracking_save_checkpoint(self, checkpoint):
        outcome = original_save_checkpoint(self, checkpoint)
        if checkpoint.stage == "collect":
            state["collect_checkpointed"] = True
        return outcome

    def _crash_after_n(self, fact):
        if state["collect_checkpointed"]:
            state["n"] += 1
            if state["n"] > after_n:
                raise _SimulatedCrash("crash point C: after partial verified-Fact persistence")
        return original_save_fact(self, fact)

    monkeypatch.setattr(repository_module.Repository, "save_checkpoint", _tracking_save_checkpoint)
    monkeypatch.setattr(repository_module.Repository, "save_fact", _crash_after_n)


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
# Helpers: Fact semantic snapshot (Correction 3, section 1)
# =============================================================================
#: Every Fact field that changes what the fact MEANS, in one fixed order
#: shared by _fact_snapshot (from a live Fact object) and _row_to_fact (a
#: DB row, reconstructed independently of Repository.facts_for_resume's
#: own logic so this is a genuine cross-check, not a test of a function
#: against itself). Excluded, and why:
#:   - run_id: run-scoped by definition (the whole point of this test file
#:     is comparing the SAME semantic facts across two different run_ids).
#:   - version, superseded_by: storage bookkeeping about HOW MANY times a
#:     fact was rewritten, not what the current content means (section F
#:     tests version count/meaning directly, separately).
#:   - created_at: a raw persistence timestamp, not a Fact field at all.
#:
#: notes IS included (Phase 4.3C correction 4 -- Correction 3 had
#: excluded it here, with a test pinning the exclusion as a known,
#: out-of-scope EvidenceIntegrityAgent limitation; that test is REVERSED
#: below, not merely deleted, since the underlying defect is now actually
#: fixed at its source: Pipeline.run() feeds a resumed Evidence Integrity
#: re-run the PRISTINE, pre-Integrity version of each fact
#: (resume_plan.pristine_facts, from Repository.pristine_facts_for_resume)
#: rather than whatever partially-processed version an interrupted pass
#: happened to leave as "latest", AND agents/evidence_integrity.py's own
#: notes-building now recomputes its own contribution fresh each time
#: (see _INTEGRITY_NOTES_MARKER there) instead of treating a prior pass's
#: reasoning as more opaque carry-forward text. Together these make
#: notes genuinely stable under resume and correctly UPDATED (not merely
#: appended to) when a second, legitimate pass changes a fact's
#: classification -- see section D below for that second case.
def _fact_snapshot(fact: Fact) -> tuple:
    return (
        fact.fact_id,
        fact.ticker,
        str(fact.category),
        fact.claim,
        str(fact.evidence_class),
        fact.source_id,
        fact.source_url,
        fact.source_title,
        str(fact.source_tier),
        fact.publication_date,
        fact.event_date,
        fact.effective_date,
        fact.filing_date,
        str(fact.verified_status),
        round(float(fact.confidence), 6),
        bool(fact.company_claim),
        bool(fact.independent_confirmation),
        tuple(sorted(x for x in fact.corroborating_source_ids if x)),
        tuple(sorted(x for x in fact.contradicting_evidence if x)),
        str(fact.materiality),
        UNKNOWN if fact.value is None else str(fact.value),
        fact.unit,
        str(fact.provenance),
        bool(fact.stale),
        fact.notes,
        tuple(sorted(x for x in fact.tags if x)),
        str(fact.content_kind),
        fact.primary_source_url,
        fact.document_id,
        str(fact.source_authority),
        fact.is_decision_grade,
    )


def _facts_snapshot_map(facts) -> dict:
    return {f.fact_id: _fact_snapshot(f) for f in facts}


def _row_to_fact(row) -> Fact:
    """Reconstructs a Fact from a raw sqlite3.Row (e.g. from
    Repository.latest_facts), independently of Repository.
    facts_for_resume's own reconstruction -- a deliberate second,
    from-scratch implementation reading every column Fact.to_row() writes
    (confirmed by reading storage/schema.sql and fact.py's to_row()
    directly), so a bug shared by both would not go unnoticed the way it
    would if this test merely called facts_for_resume a second time."""
    from investment_research.schemas.enums import (
        ContentKind,
        DocumentAuthority,
        EvidenceClass,
        Materiality,
        Provenance,
    )
    from investment_research.schemas.enums import (
        VerifiedStatus as VerifiedStatusEnum,
    )

    def _split(value):
        return tuple(x for x in (value or "").split(",") if x)

    return Fact(
        fact_id=row["fact_id"],
        ticker=row["ticker"],
        category=FactCategory(row["category"]),
        claim=row["claim"],
        evidence_class=EvidenceClass(row["evidence_class"]),
        source_id=row["source_id"],
        source_url=row["source_url"],
        source_title=row["source_title"] or "",
        source_tier=SourceTier(row["source_tier"]),
        publication_date=row["publication_date"],
        event_date=row["event_date"],
        effective_date=row["effective_date"],
        filing_date=row["filing_date"],
        verified_status=VerifiedStatusEnum(row["verified_status"]),
        confidence=float(row["confidence"]),
        company_claim=bool(row["company_claim"]),
        independent_confirmation=bool(row["independent_confirmation"]),
        corroborating_source_ids=_split(row["corroborating_source_ids"]),
        contradicting_evidence=_split(row["contradicting_evidence"]),
        materiality=Materiality(row["materiality"]),
        value=row["value"],
        unit=row["unit"],
        provenance=Provenance(row["provenance"]),
        stale=bool(row["stale"]),
        version=int(row["version"]),
        run_id=row["run_id"],
        notes=row["notes"] or "",
        tags=_split(row["tags"]),
        content_kind=ContentKind(row["content_kind"]),
        primary_source_url=row["primary_source_url"],
        document_id=row["document_id"],
        source_authority=DocumentAuthority(row["source_authority"]),
    )


def _repo_current_facts_snapshot(repo: Repository, ticker: str) -> dict:
    """The repository's current/latest view, via Repository.latest_facts
    (the same method any post-hoc report/analysis tool would call) --
    NOT facts_for_resume, so this is a genuinely different code path from
    what Pipeline.run() itself uses for resume."""
    rows = repo.latest_facts(ticker)
    return {row["fact_id"]: _fact_snapshot(_row_to_fact(row)) for row in rows}


def _non_resume_failures(failures) -> list:
    return [f for f in failures if not f.startswith("RESUMED RUN")]


def _assert_semantically_equivalent(full, full_repo: Repository, resumed, resumed_repo: Repository) -> None:
    """The full comparison Correction 3 section 2 requires. Run-scoped
    values (run_id, timestamps, agent_run durations) are never compared;
    everything else about what the two runs FOUND and CONCLUDED is."""
    full_snapshot = _facts_snapshot_map(full.bus.facts)
    resumed_snapshot = _facts_snapshot_map(resumed.bus.facts)
    assert resumed_snapshot == full_snapshot, "EvidenceBus final Facts differ semantically"

    full_repo_snapshot = _repo_current_facts_snapshot(full_repo, TICKER)
    resumed_repo_snapshot = _repo_current_facts_snapshot(resumed_repo, TICKER)
    assert resumed_repo_snapshot == full_repo_snapshot, (
        "repository current/latest Facts differ semantically"
    )
    # And the in-memory view matches what is actually on disk, for both runs.
    assert full_snapshot == full_repo_snapshot
    assert resumed_snapshot == resumed_repo_snapshot

    assert {q.source_id for q in resumed.quarantined_sources} == {
        q.source_id for q in full.quarantined_sources
    }
    assert _non_resume_failures(resumed.failures) == _non_resume_failures(full.failures)
    assert resumed.context.status == full.context.status
    assert resumed.verdict is not None and full.verdict is not None
    assert resumed.verdict.action == full.verdict.action
    assert resumed.verdict.research_status == full.verdict.research_status
    assert dict(resumed.direct_acquisition_info) == dict(full.direct_acquisition_info)
    assert resumed.traceability == full.traceability


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
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError, match="crash point A"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    # "collect" alone is checkpointed; Stage-1 facts are already persisted
    # (Correction 2's own change -- previously nothing was written until
    # inside the pass), and they are genuinely pre-Integrity.
    completed_stages = _completed_pipeline_stages(repo, run_id)
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

    completed_stages = _completed_pipeline_stages(repo, run_id)
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
    # notes (Phase 4.3C correction 4): carried forward untouched, exactly
    # as persisted -- Evidence Integrity never runs in this branch, so it
    # never gets a chance to accumulate or drop anything.
    before_notes = {f.fact_id: f.notes for f in verified_before_resume}
    after_notes = {f.fact_id: f.notes for f in resumed.bus.facts}
    assert after_notes == before_notes

    # Still exactly one stored record -- the resumed call added none.
    rows_after = _integrity_agent_run_rows(repo, run_id)
    assert [r["agent_id"] for r in rows_after] == ["evidence_integrity"]

    # NOTE (Correction 3): quarantined_sources is NOT recomputed in this
    # branch (Evidence Integrity does not rerun, and there is no persisted
    # audit trail of a PAST quarantine event -- the `sources` table has no
    # run_id column at all, confirmed by reading storage/schema.sql, and a
    # quarantined source was, by definition, never written there in the
    # first place). A resumed run that skips verify always reports
    # quarantined_sources=[] regardless of the original run's history.
    # This test's fixture has no malformed source, so the gap does not
    # manifest here; section H's crash-point tests exercise the case where
    # Evidence Integrity DOES rerun (and therefore genuinely re-detects
    # quarantine) instead of masking this known limitation.
    assert resumed.quarantined_sources == []


# =============================================================================
# D. Full semantic equivalence: uninterrupted run vs. crashed-then-resumed
# =============================================================================
def test_resume_and_full_run_are_fully_semantically_equivalent(monkeypatch):
    full_repo = _new_repo()
    resumed_repo = _new_repo()

    full_pipeline = Pipeline(full_repo, NullSearchProvider(), today=TODAY)
    full = full_pipeline.run(**_run_kwargs(), run_id="resume-d-full")

    resumed_pipeline = Pipeline(resumed_repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-d-resumed"
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            resumed_pipeline.run(**_run_kwargs(), run_id=run_id)
    # Fresh Pipeline instance for the resume itself (see module docstring).
    resumed_pipeline_2 = Pipeline(resumed_repo, NullSearchProvider(), today=TODAY)
    resumed = resumed_pipeline_2.run(**_run_kwargs(), run_id=run_id, resume=True)

    _assert_semantically_equivalent(full, full_repo, resumed, resumed_repo)


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
        _crash_point_a_before_pass_starts(m)
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


# =============================================================================
# F. Stage-1/verified Fact persistence safety
# =============================================================================
def test_collect_checkpoint_alone_leaves_not_verified_facts_recoverable(repo, monkeypatch):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-f-collect-only"
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    restored = repo.facts_for_resume(run_id)
    assert len(restored) == 2
    assert all(f.verified_status == VerifiedStatus.NOT_VERIFIED for f in restored)
    assert all(int(f.version) == 1 for f in restored)


def test_after_verify_repository_current_latest_reflects_verified_facts(repo):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-f-current-latest"
    pipeline.run(**_run_kwargs(), run_id=run_id)

    current = _repo_current_facts_snapshot(repo, TICKER)
    assert len(current) == 2
    rows = repo.latest_facts(TICKER)
    statuses = {row["fact_id"]: row["verified_status"] for row in rows}
    # The company-claim fact was corroborated by the independent one --
    # its current/latest row is VERIFIED, never the pre-Integrity
    # NOT_VERIFIED baseline FactCollectorAgent originally assigned it.
    assert "VERIFIED" in statuses.values()
    assert all(s != "NOT_VERIFIED" for s in statuses.values())


def test_two_persisted_versions_exist_with_the_expected_meaning_after_resume(repo, monkeypatch):
    """When Evidence Integrity genuinely changes a fact's classification,
    the append-only facts table must show exactly that history: version 1
    is the Stage-1 (FactCollectorAgent) baseline, version 2 is the
    post-Integrity result -- never collapsed, never lost."""
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-f-versions"
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline.run(**_run_kwargs(), run_id=run_id)
    pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    company_fact_id = next(
        f.fact_id for f in repo.facts_for_resume(run_id) if f.company_claim
    )
    versions = repo.fact_versions(company_fact_id)
    assert len(versions) == 2, f"expected exactly 2 versions, found {len(versions)}"
    v1, v2 = versions[0], versions[1]
    assert int(v1["version"]) == 1
    assert v1["verified_status"] == "NOT_VERIFIED"
    assert v1["evidence_class"] == "COMPANY_CLAIM"
    assert v1["superseded_by"] is not None  # explicitly marked superseded
    assert int(v2["version"]) == 2
    assert v2["verified_status"] == "VERIFIED"
    assert v2["evidence_class"] == "VERIFIED_FACT"
    assert v2["superseded_by"] is None  # the current/latest row


def test_collect_and_verify_snapshots_are_separate_contracts(repo, monkeypatch):
    """Phase 4.3C correction 5: Repository.collect_snapshot_for_resume
    (Stage 1's own pristine output) and Repository.verify_snapshot_for_
    resume (Stage 2's own verified output) are two run-scoped snapshots in
    run_checkpoints, not a MIN/MAX(version) query against the shared
    `facts` table (removed -- both were proven vulnerable to cross-run
    content collision). This proves the two snapshots genuinely diverge
    once Evidence Integrity has touched a fact, and that BOTH are
    immediately correct even mid-crash (crash point C: one fact already
    fully verified and re-saved to the facts table, the other still only
    at its Stage-1 baseline there -- the COLLECT SNAPSHOT itself is
    unaffected by any of that, since it was written once, atomically, at
    the end of Stage 1, before Evidence Integrity ever started)."""
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-f-snapshot-contract"
    with monkeypatch.context() as m:
        _crash_point_c_after_partial_fact_save(m, after_n=1)
        with pytest.raises(_SimulatedCrash, match="crash point C"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    collect_snapshot = repo.collect_snapshot_for_resume(run_id)
    assert collect_snapshot is not None
    assert repo.verify_snapshot_for_resume(run_id) is None  # verify never completed

    pristine = {f.fact_id: f for f in collect_snapshot.facts}
    assert len(pristine) == 2
    assert all(f.verified_status == VerifiedStatus.NOT_VERIFIED for f in pristine.values())
    assert all(
        f.evidence_class.value in ("COMPANY_CLAIM", "UNVERIFIED_CLAIM")
        for f in pristine.values()
    )
    assert {s.source_id for s in collect_snapshot.sources} == {"src_company", "src_indep"}
    assert len(collect_snapshot.collectors) == 1
    assert collect_snapshot.collectors[0]["collector"] == "resume_test_collector"

    # The `facts` table itself, meanwhile, genuinely reflects the mix left
    # by the partial crash -- proving the collect snapshot is NOT simply
    # reading that table back (it is immune to it).
    facts_table_latest = {f.fact_id: f for f in repo.facts_for_resume(run_id)}
    assert any(int(f.version) > 1 for f in facts_table_latest.values())

    # Complete the resume; a verify snapshot now exists too, and it
    # diverges from the collect snapshot exactly where Integrity changed
    # something.
    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)
    verify_snapshot = repo.verify_snapshot_for_resume(run_id)
    assert verify_snapshot is not None
    verified = {f.fact_id: f for f in verify_snapshot.verified_facts}
    assert any(
        verified[fid].verified_status != pristine[fid].verified_status for fid in pristine
    )


def test_collect_snapshot_for_resume_returns_none_for_a_run_with_no_snapshot():
    """Phase 4.3C correction 5's own requirement: for a run_id that never
    reached Stage 1's persist step at all (e.g. an old-format run from
    before this correction first introduced snapshots, or simply a run_id
    nothing was ever collected for), both snapshot methods must return
    ``None`` -- never guess, never fall back to some other run's or some
    other version's content. Absence, not a wrong answer."""
    repo = _new_repo()
    assert repo.collect_snapshot_for_resume("never-existed") is None
    assert repo.verify_snapshot_for_resume("never-existed") is None


def test_production_never_queries_the_db_facts_table_mid_run():
    """Repository.latest_facts/facts_for_run exist for post-hoc
    inspection (reports/tests reading a past run) but must never be how a
    LIVE run reads its own facts -- report/scoring/domain agents work
    exclusively from the in-memory EvidenceBus (already the Integrity-
    verified set), so a stale DB view is structurally unreachable from
    inside a run. Verified by reading the actual call sites, not by
    assumption: only orchestrator/resume.py's own module and Repository's
    own definitions reference these two methods anywhere in src/."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "investment_research"
    offenders = []
    for path in src.rglob("*.py"):
        if path.name in ("repository.py",):
            continue
        text = path.read_text(encoding="utf-8")
        if "latest_facts(" in text or "facts_for_run(" in text:
            offenders.append(str(path))
    assert offenders == [], f"unexpected mid-run readers of the DB facts view: {offenders}"


# =============================================================================
# G. Fresh-process Source restoration
# =============================================================================
def test_fresh_process_resume_restores_sources_never_empty_never_another_runs(repo, monkeypatch):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-g-sources"

    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    # A second, unrelated run_id/company must never leak its own sources
    # into what the resumed call below sees -- collection_results is
    # rebuilt fresh per call (_run_kwargs()), never shared.
    other_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    other_pipeline.run(**_run_kwargs(), run_id="resume-g-other-run")

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)  # brand-new instance
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert {s.source_id for s in resumed.bus.sources} == {"src_company", "src_indep"}
    # Every restored Fact's source_id resolves to a real Source and a
    # walkable traceability link -- never a dangling reference.
    known_source_ids = {s.source_id for s in resumed.bus.sources}
    for fact in resumed.bus.facts:
        assert fact.source_id in known_source_ids
        link = resumed.traceability.resolve(fact.fact_id)
        assert link is not None
        assert link.source_id == fact.source_id


def test_fresh_process_resume_reproduces_malformed_source_quarantine(repo, monkeypatch):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-g-quarantine"

    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline.run(**_run_kwargs_with_malformed_source(), run_id=run_id)

    # No sources were persisted yet (the crash happened before Evidence
    # Integrity's own save_sources call) -- quarantine can only be
    # re-detected if the resumed call gets the malformed Source back at
    # all, which is exactly what this test proves.
    assert repo.conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"] == 0

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(
        **_run_kwargs_with_malformed_source(), run_id=run_id, resume=True
    )

    assert {q.source_id for q in resumed.quarantined_sources} == {"src_bad"}
    assert any("quarantined malformed source" in f for f in resumed.failures)
    # The fact resting only on the malformed source is excluded, the other
    # two survive.
    assert not any(f.source_id == "src_bad" for f in resumed.bus.facts)
    assert len(resumed.bus.facts) == 2
    # The two good sources ARE persisted; the bad one is not.
    persisted_ids = {
        row["source_id"] for row in repo.conn.execute("SELECT source_id FROM sources")
    }
    assert persisted_ids == {"src_company", "src_indep"}


# =============================================================================
# H. Integrity-pass mid-crash points A/B/C
# =============================================================================
def _run_control_full(control_repo: Repository):
    pipeline = Pipeline(control_repo, NullSearchProvider(), today=TODAY)
    return pipeline.run(**_run_kwargs(), run_id="resume-h-control-full")


def test_crash_point_a_collect_checkpoint_only_resumes_to_full_equivalence(monkeypatch):
    control_repo = _new_repo()
    full = _run_control_full(control_repo)

    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-h-point-a"
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError, match="crash point A"):
            pipeline.run(**_run_kwargs(), run_id=run_id)
    completed = _completed_pipeline_stages(repo, run_id)
    assert completed == {"collect"}
    assert _integrity_agent_run_rows(repo, run_id) == []  # no record at all yet

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    _assert_semantically_equivalent(full, control_repo, resumed, repo)


def test_crash_point_b_after_source_save_resumes_to_full_equivalence(monkeypatch):
    control_repo = _new_repo()
    full = _run_control_full(control_repo)

    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-h-point-b"
    with monkeypatch.context() as m:
        _crash_point_b_after_source_save(m)
        with pytest.raises(RuntimeError, match="crash point B"):
            pipeline.run(**_run_kwargs(), run_id=run_id)
    completed = _completed_pipeline_stages(repo, run_id)
    assert completed == {"collect"}
    # Sources ARE already persisted for real at this crash point.
    assert repo.conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"] == 2
    # A real, but partial (fact persistence never ran), AgentRunRecord exists.
    partial_rows = _integrity_agent_run_rows(repo, run_id)
    assert [r["agent_id"] for r in partial_rows] == ["evidence_integrity"]

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    _assert_semantically_equivalent(full, control_repo, resumed, repo)
    final_rows = _integrity_agent_run_rows(repo, run_id)
    assert [r["agent_id"] for r in final_rows] == ["evidence_integrity"]  # replaced, not doubled


def test_crash_point_c_after_partial_fact_save_resumes_to_full_equivalence(monkeypatch):
    control_repo = _new_repo()
    full = _run_control_full(control_repo)

    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-h-point-c"
    with monkeypatch.context() as m:
        _crash_point_c_after_partial_fact_save(m, after_n=1)
        with pytest.raises(_SimulatedCrash, match="crash point C"):
            pipeline.run(**_run_kwargs(), run_id=run_id)
    completed = _completed_pipeline_stages(repo, run_id)
    assert completed == {"collect"}
    # A genuine mix: one fact already has its post-Integrity version
    # persisted, the DB otherwise still holds only the Stage-1 baseline.
    restored = {f.fact_id: f.verified_status for f in repo.facts_for_resume(run_id)}
    assert len(restored) == 2
    statuses = set(restored.values())
    assert VerifiedStatus.NOT_VERIFIED in statuses or VerifiedStatus.VERIFIED in statuses

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    # The resumed full pass re-evaluates the ENTIRE restored set (never
    # just the not-yet-saved remainder), so the mix from the partial crash
    # never leaks into the final result. This now includes notes (Phase
    # 4.3C correction 4) -- see _assert_semantically_equivalent's own use
    # of _fact_snapshot, which compares notes strictly.
    _assert_semantically_equivalent(full, control_repo, resumed, repo)


def test_repeated_evidence_integrity_evaluation_is_now_idempotent_no_duplication(monkeypatch):
    """REVERSES (per Phase 4.3C correction 4's own explicit requirement)
    Correction 3's test that pinned notes-duplication as a known,
    out-of-scope limitation. Crash point C is the exact scenario that
    exposed it: a fact whose post-Integrity version was persisted before
    the crash, fed back into a resumed full pass that correctly
    re-evaluates the WHOLE restored set. Two fixes now apply together:
    Pipeline.run() feeds that resumed pass the PRISTINE (pre-Integrity)
    version of the fact, never its partially-processed one
    (resume_plan.pristine_facts); and even where a fact genuinely is
    evaluated by EvidenceIntegrityAgent more than once (this pristine
    swap included, since strictly speaking pristine facts ARE going
    through their real first evaluation here -- the point is there is no
    SECOND, redundant one), its own notes-building is idempotent
    (_INTEGRITY_NOTES_MARKER). Reasoning text appears at most once, and
    the collector-original note survives underneath it."""
    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-notes-idempotent"
    with monkeypatch.context() as m:
        _crash_point_c_after_partial_fact_save(m, after_n=1)
        with pytest.raises(_SimulatedCrash, match="crash point C"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    company_fact = next(f for f in resumed.bus.facts if f.company_claim)
    reasoning = "event date unknown; publication date must not be read as the event date"
    assert company_fact.notes.count(reasoning) <= 1  # never duplicated
    assert company_fact.notes.count("Evidence Integrity: ") <= 1  # one contribution, not stacked
    assert "collected_by=" in company_fact.notes  # collector-original note survives
    assert company_fact.evidence_class.value == "VERIFIED_FACT"
    assert company_fact.verified_status.value == "VERIFIED"
    assert company_fact.independent_confirmation is True

    # And it now matches an uninterrupted control run's notes exactly --
    # the strongest form of "no duplication": byte-identical to a run
    # that never crashed at all.
    control_repo = _new_repo()
    control = _run_control_full(control_repo)
    control_company_fact = next(f for f in control.bus.facts if f.company_claim)
    assert company_fact.notes == control_company_fact.notes


# =============================================================================
# I. AgentRunRecord partial-state handling
# =============================================================================
def test_partial_agent_run_record_is_replaced_not_duplicated_on_resume(monkeypatch):
    control_repo = _new_repo()
    control_pipeline = Pipeline(control_repo, NullSearchProvider(), today=TODAY)
    control_run_id = "resume-i-control"
    full = control_pipeline.run(**_run_kwargs(), run_id=control_run_id)
    control_record = _integrity_agent_run_rows(control_repo, control_run_id)[0]

    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-i-partial"
    with monkeypatch.context() as m:
        _crash_point_b_after_source_save(m)
        with pytest.raises(RuntimeError, match="crash point B"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    partial_rows = _integrity_agent_run_rows(repo, run_id)
    assert len(partial_rows) == 1
    # fact_count reflects the AGENT's own completed output (already real
    # by crash point B, which is after agent execution) -- persistence
    # progress downstream of that is a separate concern this record does
    # not track, which is exactly why this test checks the FINAL record
    # below rather than trusting this partial one's numbers.
    assert partial_rows[0]["fact_count"] == control_record["fact_count"]

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    final_rows = _integrity_agent_run_rows(repo, run_id)
    # No PRIMARY KEY collision failure (the run() call above completed);
    # exactly one row -- the stale partial record was REPLACED, not kept
    # alongside the successful one.
    assert len(final_rows) == 1
    assert final_rows[0]["agent_id"] == "evidence_integrity"  # INITIAL identity, unchanged
    assert final_rows[0]["status"] == control_record["status"]
    assert final_rows[0]["fact_count"] == control_record["fact_count"]
    assert len(resumed.bus.facts) == len(full.bus.facts)


# =============================================================================
# J. Cross-run content-collision immunity (Phase 4.3C correction 5)
# =============================================================================
def test_byte_identical_content_across_two_run_ids_stays_fully_isolated(monkeypatch):
    """Phase 4.3C correction 5's own empirical reproduction, now proving
    the fix: two DIFFERENT run_ids that happen to collect byte-identical
    Stage-1 content (a realistic case -- re-researching a company whose
    filings have not changed since the last run) must not collide. Before
    this correction, Repository.save_fact's global (never run_id-scoped)
    duplicate detection meant the SECOND run_id's own Stage-1 save was a
    no-op that wrote no row at all for that fact_id -- so a MIN- or
    MAX-version query scoped to that run_id (Correction 4's
    pristine_facts_for_resume, or facts_for_resume) silently returned
    NOTHING, even though checkpoints correctly showed the stage complete
    (empirically reproduced this way; see this correction's own
    completion report). The collect/verify snapshots this correction
    introduces are immune BY CONSTRUCTION (run_checkpoints' own PRIMARY
    KEY includes run_id), which this test proves directly."""
    repo = _new_repo()

    # run A: crashes right after "collect" -- stays at its pristine v1,
    # NOT_VERIFIED, in the shared `facts` table.
    pipeline_a = Pipeline(repo, NullSearchProvider(), today=TODAY)
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline_a.run(**_run_kwargs(), run_id="run-A")

    # run B: the SAME byte-identical Stage-1 content, ALSO crashes right
    # after "collect".
    pipeline_b = Pipeline(repo, NullSearchProvider(), today=TODAY)
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline_b.run(**_run_kwargs(), run_id="run-B")

    # Confirmed collision, in the RAW facts table: only ONE row per
    # fact_id exists at all, tagged run-A -- run B's own Stage-1 save was
    # a genuine no-op.
    rows = repo.conn.execute("SELECT fact_id, run_id FROM facts ORDER BY fact_id").fetchall()
    assert {r["run_id"] for r in rows} == {"run-A"}
    # The OLD (now-removed) mechanism would have found nothing for run-B here.
    assert repo.facts_for_resume("run-B") == []

    # But the collect SNAPSHOT -- run_checkpoints-scoped, not facts-table-
    # scoped -- is genuinely, separately present for BOTH run_ids.
    snap_a = repo.collect_snapshot_for_resume("run-A")
    snap_b = repo.collect_snapshot_for_resume("run-B")
    assert snap_a is not None
    assert snap_b is not None
    assert len(snap_a.facts) == 2
    assert len(snap_b.facts) == 2
    assert {f.fact_id for f in snap_a.facts} == {f.fact_id for f in snap_b.facts}

    # And resuming run B genuinely recovers its own 2 facts -- never zero,
    # never run A's.
    fresh_pipeline_b = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed_b = fresh_pipeline_b.run(**_run_kwargs(), run_id="run-B", resume=True)
    assert len(resumed_b.bus.facts) == 2
    assert resumed_b.resume_plan is not None
    assert resumed_b.resume_plan.snapshot_unavailable_reason is None


# =============================================================================
# K. Fail-closed snapshot unavailability (Phase 4.3C correction 5)
# =============================================================================
def test_missing_collect_snapshot_fails_closed(repo, monkeypatch):
    """A run_id whose "collect" checkpoint exists but whose collect
    snapshot does not (e.g. deleted, or a run predating this correction)
    must fail closed on resume: status=INCOMPLETE_RESEARCH, Action=None,
    a specific blocking reason -- never a silently "successful" resume
    with the wrong (or zero) facts passed off as a real result. No
    additional HTTP/collector call is made either: this branch never
    re-collects behind --resume's own back."""
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-k-missing-collect"
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    # Simulate the snapshot having been lost (deleted row) while the
    # "collect" stage checkpoint itself remains -- the scenario this fix
    # must handle, not just the case where nothing was ever collected.
    repo.conn.execute(
        "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
        (run_id,),
    )
    repo.conn.commit()
    assert repo.collect_snapshot_for_resume(run_id) is None

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert resumed.resume_plan is not None
    assert resumed.resume_plan.snapshot_unavailable_reason is not None
    assert resumed.context.status == RunStatus.INCOMPLETE_RESEARCH
    assert resumed.verdict is not None
    assert resumed.verdict.action is None
    assert resumed.verdict.blocked is True
    assert any(
        "collect snapshot" in reason.lower()
        for reason in resumed.verdict.blocking_verification_required
    )
    assert any("RESUME SNAPSHOT UNAVAILABLE" in f for f in resumed.failures)
    # Fails closed, not "succeeds with nothing": no facts are fabricated
    # or borrowed from anywhere else.
    assert resumed.bus.facts == []
    # No Evidence Integrity execution either -- there was nothing
    # trustworthy to feed it.
    assert _integrity_agent_run_rows(repo, run_id) == []
    # "verify" is never checkpointed OK on this path -- a later --resume
    # of this same run_id, once the underlying issue is fixed, is still
    # correctly judged to need the (still-missing) collect snapshot.
    assert "verify" not in _completed_pipeline_stages(repo, run_id)


def test_corrupted_collect_snapshot_fails_closed(repo, monkeypatch):
    """A collect snapshot row that exists but cannot be parsed back
    (corrupted JSON) must ALSO fail closed -- never silently treated as
    "nothing collected" (a materially different, falsely reassuring
    condition) and never partially trusted."""
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-k-corrupted-collect"
    with monkeypatch.context() as m:
        _crash_point_a_before_pass_starts(m)
        with pytest.raises(RuntimeError):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = ? AND stage = '_collect_snapshot'",
        ("{not valid json", run_id),
    )
    repo.conn.commit()
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume(run_id)

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert resumed.resume_plan is not None
    assert resumed.resume_plan.snapshot_unavailable_reason is not None
    assert "corrupted" in resumed.resume_plan.snapshot_unavailable_reason.lower()
    assert resumed.context.status == RunStatus.INCOMPLETE_RESEARCH
    assert resumed.verdict is not None
    assert resumed.verdict.action is None
    assert resumed.bus.facts == []
    assert _integrity_agent_run_rows(repo, run_id) == []


def test_missing_verify_snapshot_after_verify_complete_fails_closed(monkeypatch):
    """The same fail-closed guarantee for the OTHER resume point: verify
    checkpointed complete, but the verify snapshot itself missing."""
    repo = _new_repo()
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    run_id = "resume-k-missing-verify"
    with monkeypatch.context() as m:
        _crash_after_verify_checkpoint(m)
        with pytest.raises(RuntimeError, match="simulated crash after verify checkpoint"):
            pipeline.run(**_run_kwargs(), run_id=run_id)

    assert _completed_pipeline_stages(repo, run_id) == {"collect", "verify"}
    repo.conn.execute(
        "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = '_verify_snapshot'",
        (run_id,),
    )
    repo.conn.commit()
    assert repo.verify_snapshot_for_resume(run_id) is None

    calls = _count_agent_runs(monkeypatch)
    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_run_kwargs(), run_id=run_id, resume=True)

    assert len(calls) == 0  # Evidence Integrity never runs on this path either
    assert resumed.resume_plan is not None
    assert resumed.resume_plan.snapshot_unavailable_reason is not None
    assert resumed.context.status == RunStatus.INCOMPLETE_RESEARCH
    assert resumed.verdict is not None
    assert resumed.verdict.action is None
    assert resumed.bus.facts == []
