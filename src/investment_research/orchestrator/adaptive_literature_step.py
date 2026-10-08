"""Phase 4.3I: connects Phase 4.3G's ``AdaptiveAcquisitionPlan`` and Phase
4.3H's ``execute_adaptive_literature_plan()`` bridge to ``Pipeline.run()``.

Reachable ONLY when a caller injects a real ``http_client``/``env`` into
``Pipeline.__init__`` (``adaptive_http_client``/``adaptive_env``, both
default ``None``) -- neither is wired to any CLI flag or production
transport in this phase (that is explicitly future work; see this
repository's CLAUDE.md). A Production run that never injects a runner
takes the ``NOT_ATTEMPTED_NO_RUNNER`` branch below unconditionally, calls
``execute_adaptive_literature_plan`` zero times, and produces byte-
identical output to every run before this phase existed.

Design precedents this module deliberately follows rather than
reinventing:

* ``orchestrator.evidence_integrity_pass.run_full_evidence_integrity_pass``
  is called directly (``agent.execute(AgentInput(...))``, never
  ``Pipeline._run_agent``) for the SAME reason that module itself gives:
  genuinely callable with no live Pipeline dependency, and already proven
  safe for a second (``IntegrityPassKind.ADAPTIVE``) pass by that module's
  own test suite.
* The second Integrity pass's ``pre_integrity_facts`` input is
  ``verified_facts`` (Stage 2's own INITIAL output -- always populated by
  the time this step runs, in every resume state) UNION the new facts a
  standalone ``FactCollectorAgent`` call produces from the adaptive
  literature ``CollectionResult`` -- never the pristine Stage-1
  ``pre_integrity_facts`` local variable in ``pipeline.py``, which is an
  EMPTY list whenever a run resumes past "verify" (``pipeline.py`` line
  823) and so is not a reliable input across every resume state the way
  ``verified_facts`` is.
* ``FactCollectorAgent``'s own ``.agent_id`` (``"fact_collector"``) is
  never changed -- only the STORED ``AgentRunRecord.agent_id`` a second
  call files its audit row under, via the closed
  :class:`FactCollectionPassKind` enum, mirroring
  ``evidence_integrity_pass.IntegrityPassKind``'s own established
  precedent exactly (a plain string field would let a caller collide with
  an unrelated agent's own ``agent_runs`` row -- that mistake is already
  documented and fixed once in this codebase; this module does not repeat
  it).
* Chunks are stored directly (the exact, already coverage-filtered
  ``LiteraturePipelineBundle.chunks`` tuple), never re-derived from a
  reconstructed ``DocumentStore`` on resume -- see
  ``storage.repository._chunk_to_dict``'s own docstring.

Phase 4.3I correction 2 (snapshot/input binding): every persisted
checkpoint (``_adaptive_literature_started``/``_adaptive_collect_snapshot``/
``_adaptive_verify_snapshot``) now carries a typed
:class:`~investment_research.storage.repository.AdaptiveLiteratureTarget`
(ticker, reference_mode, nct_id, max_requests, and a content-sensitive
fingerprint of the Stage-2 base ``verified_facts`` the checkpoint was
computed against) plus, for the verify snapshot, a dependency digest of
the EXACT collect snapshot it was computed from. Restoring EITHER
snapshot now independently re-verifies this binding against the CURRENT
plan/ticker/base facts before trusting it -- a mismatch (different
ticker, different NCT id, a changed request budget, or the same fact_id
set with different content) is ``SNAPSHOT_INPUT_MISMATCH``: never
re-fetched, never re-run through Integrity, never silently restored. A
checkpoint saved before this correction (lacking the new identity
fields entirely) is corrupted on read, never guessed at.

Phase 4.3I correction 2 (resume-state ordering): resume-state (started
marker / collect snapshot / verify snapshot, and now the input-binding
check above) is judged FIRST, unconditionally -- before
``base_evidence_available``, before ``plan.status``, and before the
runner-injection check. A prior invocation's ambiguous state
(``_adaptive_literature_started`` with no completed collect snapshot)
must never be masked by this invocation's plan happening to be
non-READY, or by this invocation's base evidence happening to be
unavailable. ``base_evidence_available=False`` with a CLEAN (matching)
prior snapshot on file still refuses to restore that snapshot's Facts
-- the current base to compare it against is itself unknown, so the
binding cannot be meaningfully re-verified, and restoring old evidence
against an unknown new base is exactly the kind of stale-but-plausible
result CLAUDE.md rule 7 forbids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from ..agents.base import inputs_hash
from ..agents.fact_collector import FactCollectorAgent
from ..collectors.documents import Chunk
from ..research.adaptive_literature_acquisition import (
    AdaptiveExecutionStatus,
    LiteratureHttpTransport,
    execute_adaptive_literature_plan,
)
from ..schemas.agent_io import AgentInput, AgentRunRecord
from ..schemas.enums import StrEnum
from ..schemas.fact import Fact, Source
from ..schemas.validation import QuarantinedSource
from ..scoring.adaptive_acquisition_plan import AcquisitionPlanStatus, AdaptiveAcquisitionPlan
from ..storage.repository import (
    AdaptiveCollectSnapshot,
    AdaptiveLiteratureStarted,
    AdaptiveLiteratureTarget,
    AdaptiveVerifySnapshot,
    Repository,
    ResumeSnapshotCorrupted,
    adaptive_collect_snapshot_digest,
    fact_content_fingerprint,
    source_content_fingerprint,
)
from .evidence_integrity_pass import (
    FullIntegrityPassInput,
    IntegrityPassKind,
    run_full_evidence_integrity_pass,
)


class FactCollectionPassKind(StrEnum):
    """Closed identity for which audited ``FactCollectorAgent`` call this
    is -- mirrors ``evidence_integrity_pass.IntegrityPassKind`` exactly,
    for the same reason: ``agent_runs``'s ``PRIMARY KEY (run_id, agent_id)``
    means a second call under the SAME run_id must file its audit row
    under a DIFFERENT ``agent_id``, chosen from a closed set, never a
    caller-supplied string."""

    INITIAL = "initial"
    ADAPTIVE = "adaptive"


#: The only mapping from :class:`FactCollectionPassKind` to a stored
#: ``AgentRunRecord.agent_id``. ``FactCollectorAgent.agent_id`` itself
#: (the real agent identity, ``"fact_collector"``) is never touched by
#: this -- only which string a given call's own audit row is filed under.
_FACT_COLLECTOR_AUDIT_AGENT_IDS: dict[FactCollectionPassKind, str] = {
    FactCollectionPassKind.INITIAL: "fact_collector",
    FactCollectionPassKind.ADAPTIVE: "fact_collector_adaptive",
}


class AdaptiveLiteratureStepOutcome(StrEnum):
    """A closed set of outcomes for one ``run_adaptive_literature_step``
    call. Never a bare bool -- mirrors ``AcquisitionPlanStatus``/
    ``AdaptiveExecutionStatus``'s own convention. "Never attempted" is
    always a DIFFERENT value from any attempted-but-incomplete outcome
    (CLAUDE.md rule 8: an unsearched category is UNSEARCHED, never K0)."""

    #: plan.status was not READY (NO_ACTION/UNRESOLVED/CONFLICTED/
    #: SKIPPED_EXPLICIT_OVERRIDE/REFUSED) -- nothing to acquire. Reached
    #: ONLY when no checkpoint of any kind exists yet for this run_id
    #: (Phase 4.3I correction 2: checked AFTER resume-state, never
    #: before). Zero effect on verified_facts/chunks/blocking_reasons.
    NOT_ATTEMPTED_NON_READY = "NOT_ATTEMPTED_NON_READY"
    #: plan.status is READY but no http_client/env was injected into this
    #: Pipeline -- this step never calls execute_adaptive_literature_plan
    #: at all in this case (distinct from the bridge's own REFUSED, which
    #: means a call WAS made). Reached ONLY when no checkpoint exists yet.
    #: Zero effect on verified_facts/chunks/blocking_reasons.
    NOT_ATTEMPTED_NO_RUNNER = "NOT_ATTEMPTED_NO_RUNNER"
    #: The base evidence set itself was unavailable this invocation
    #: (Pipeline's own resume-snapshot fail-closed path) -- fetching new
    #: literature against a known-incomplete base makes no sense, and
    #: restoring an OLD (even cleanly matching) snapshot's Facts against
    #: an unknown new base is refused for the same reason. Zero HTTP,
    #: zero new facts, zero restored facts.
    NOT_ATTEMPTED_BASE_EVIDENCE_UNAVAILABLE = "NOT_ATTEMPTED_BASE_EVIDENCE_UNAVAILABLE"
    #: A "_adaptive_literature_started" marker exists for this run_id with
    #: no completed collect snapshot -- whether HTTP was ever sent cannot
    #: be determined from persisted state. Checked UNCONDITIONALLY, before
    #: base_evidence_available/plan.status are even inspected (Phase 4.3I
    #: correction 2) -- never masked by a later invocation's plan
    #: happening to be non-READY or its base evidence happening to be
    #: unavailable. Fails closed: blocking_reasons is always non-empty,
    #: Action=None. Never auto-retried.
    AMBIGUOUS_RESUME_STATE = "AMBIGUOUS_RESUME_STATE"
    #: A snapshot row exists for this run_id but could not be parsed back
    #: losslessly -- INCLUDING a pre-correction-2 snapshot lacking the
    #: identity fields a restore now requires. Fails closed, same as
    #: AMBIGUOUS_RESUME_STATE.
    SNAPSHOT_CORRUPTED = "SNAPSHOT_CORRUPTED"
    #: Phase 4.3I correction 2: a collect and/or verify snapshot exists
    #: and parses cleanly, but its stored target identity (ticker/
    #: reference_mode/nct_id/max_requests), its base-fact fingerprint, its
    #: execution_status, or (for a verify snapshot) its dependency digest
    #: on the current collect snapshot does NOT match the CURRENT
    #: invocation's plan/ticker/base facts. Never re-fetched, never
    #: re-run through Integrity, never silently restored -- this is a
    #: DIFFERENT condition from AMBIGUOUS_RESUME_STATE (there, whether a
    #: request was ever sent is unknown; here, a request's recorded
    #: result is known but no longer applies to this invocation's input).
    SNAPSHOT_INPUT_MISMATCH = "SNAPSHOT_INPUT_MISMATCH"
    #: The bridge was called and refused (READY re-verification failure,
    #: request validation refusal, or bundle.refused) -- zero HTTP in
    #: every case per execute_adaptive_literature_plan's own contract.
    #: blocking_reasons is always non-empty: an attempted-and-refused
    #: fetch is never silently equivalent to NOT_ATTEMPTED.
    REFUSED = "REFUSED"
    #: The bridge ran and LiteraturePipelineBundle.coverage_complete was
    #: True, AND the second Evidence Integrity pass itself raised no new
    #: incompleteness (no new quarantine). blocking_reasons is empty.
    COMPLETE = "COMPLETE"
    #: The bridge ran but coverage was incomplete, OR the second Evidence
    #: Integrity pass itself found a new incompleteness (e.g. a newly
    #: quarantined literature Source) even though the fetch itself
    #: reported COMPLETE. blocking_reasons is always non-empty -- a
    #: partial fetch is run through Integrity but never treated as having
    #: resolved the coverage gap.
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class AdaptiveLiteratureStepResult:
    outcome: AdaptiveLiteratureStepOutcome
    #: The full verified_facts set after this step -- identical to the
    #: input verified_facts when outcome is any NOT_ATTEMPTED_*/
    #: AMBIGUOUS_RESUME_STATE/SNAPSHOT_CORRUPTED/SNAPSHOT_INPUT_MISMATCH/
    #: REFUSED value.
    verified_facts: tuple[Fact, ...]
    #: New Chunks to append to Pipeline.chunks -- empty for every outcome
    #: except COMPLETE/INCOMPLETE.
    new_chunks: tuple[Chunk, ...]
    #: New Sources (literature-originated) the caller must merge into
    #: ``EvidenceBus.sources`` (via ``bus.add_sources(...)``, which dedups
    #: by ``source_id``) -- empty for every outcome except COMPLETE/
    #: INCOMPLETE/a restore that found one. Mirrors Stage 1's own existing
    #: contract exactly: a quarantined Source is still included here (never
    #: pre-filtered) -- quarantine exclusion is tracked separately via
    #: ``quarantined_sources`` below, and no Fact resting only on a
    #: quarantined Source ever survives into ``verified_facts`` (Evidence
    #: Integrity's own item 6), so a quarantined Source reaching
    #: ``bus.sources`` is never usable as evidence despite being present.
    new_sources: tuple[Source, ...]
    #: Newly quarantined Sources this step's own Integrity pass found --
    #: the caller extends (never replaces) result.quarantined_sources.
    quarantined_sources: tuple[QuarantinedSource, ...]
    agent_records: tuple[AgentRunRecord, ...]
    failures: tuple[str, ...]
    #: Non-empty exactly for REFUSED/INCOMPLETE/AMBIGUOUS_RESUME_STATE/
    #: SNAPSHOT_CORRUPTED/SNAPSHOT_INPUT_MISMATCH -- the caller extends
    #: blocking_reasons (Action Gate) with these, distinct from
    #: direct_acquisition_info (the existing explicit-Literature-path
    #: diagnostic, never mixed with this step's own).
    blocking_reasons: tuple[str, ...]
    #: Safe (body/secret/URL-free) diagnostics -- a SEPARATE typed carrier
    #: from result.direct_acquisition_info (which belongs exclusively to
    #: the explicit PMID/NCT path). Never includes a Document/Chunk body,
    #: a credential, or a raw bundle.
    diagnostics: dict = field(default_factory=dict)


def _not_attempted(
    outcome: AdaptiveLiteratureStepOutcome,
    verified_facts: Sequence[Fact],
    reason: str,
) -> AdaptiveLiteratureStepResult:
    return AdaptiveLiteratureStepResult(
        outcome=outcome,
        verified_facts=tuple(verified_facts),
        new_chunks=(),
        new_sources=(),
        quarantined_sources=(),
        agent_records=(),
        failures=(),
        blocking_reasons=(),
        diagnostics={"outcome": str(outcome), "reason": reason},
    )


def _blocked(
    outcome: AdaptiveLiteratureStepOutcome,
    verified_facts: Sequence[Fact],
    reason: str,
) -> AdaptiveLiteratureStepResult:
    return AdaptiveLiteratureStepResult(
        outcome=outcome,
        verified_facts=tuple(verified_facts),
        new_chunks=(),
        new_sources=(),
        quarantined_sources=(),
        agent_records=(),
        failures=(reason,),
        blocking_reasons=(reason,),
        diagnostics={"outcome": str(outcome), "reason": reason},
    )


def _current_target(
    plan: AdaptiveAcquisitionPlan,
    ticker: str,
    verified_facts: Sequence[Fact],
    sources: Sequence[Source],
) -> AdaptiveLiteratureTarget:
    """Phase 4.3I correction 2 (correction 4 adds ``sources``): the
    identity THIS invocation's plan/ticker/base facts/base sources would
    bind a NEW checkpoint to -- also what any EXISTING checkpoint is
    compared against before being trusted. ``sources`` is the run's own
    in-memory base Source set (e.g. ``bus.sources``), never a fresh query
    against the global ``sources`` table -- the second Evidence Integrity
    pass consumes this same set as ``old_sources``
    (``_finalize_with_integrity`` below), so its content must be bound
    into the target exactly like the base facts already are."""
    return AdaptiveLiteratureTarget(
        ticker=ticker,
        reference_mode=plan.reference_mode,
        nct_id=plan.nct_id,
        max_requests=plan.max_requests,
        base_fact_fingerprint=fact_content_fingerprint(verified_facts),
        base_source_fingerprint=source_content_fingerprint(sources),
    )


def _target_mismatch_reason(
    current: AdaptiveLiteratureTarget, stored: AdaptiveLiteratureTarget,
) -> str:
    """Empty string when every field matches. Never a bare bool -- the
    returned text names exactly which field(s) disagree, so a human
    reading blocking_reasons can tell "the ticker changed" apart from
    "the same fact_id's content changed under this run_id"."""
    diffs: list[str] = []
    if current.ticker != stored.ticker:
        diffs.append(f"ticker (stored={stored.ticker!r}, current={current.ticker!r})")
    if current.reference_mode != stored.reference_mode:
        diffs.append(
            f"reference_mode (stored={stored.reference_mode!r}, current={current.reference_mode!r})"
        )
    if current.nct_id != stored.nct_id:
        diffs.append(f"nct_id (stored={stored.nct_id!r}, current={current.nct_id!r})")
    if current.max_requests != stored.max_requests:
        diffs.append(
            f"max_requests (stored={stored.max_requests!r}, current={current.max_requests!r})"
        )
    if current.base_fact_fingerprint != stored.base_fact_fingerprint:
        diffs.append(
            "base_fact_fingerprint (the Stage-2 base verified_facts this checkpoint was "
            "computed against no longer match the current invocation's -- same run_id, "
            "different or differently-content fact set)"
        )
    if current.base_source_fingerprint != stored.base_source_fingerprint:
        diffs.append(
            "base_source_fingerprint (the Stage-2 base sources this checkpoint was "
            "computed against no longer match the current invocation's -- same run_id, "
            "different or differently-content source set)"
        )
    return "; ".join(diffs)


def _all_target_mismatches(
    current: AdaptiveLiteratureTarget,
    named_targets: Sequence[tuple[str, AdaptiveLiteratureTarget | None]],
) -> list[str]:
    """Phase 4.3I correction 3: checks EVERY persisted checkpoint's own
    target against the SAME ``current`` reference, never only one of
    them -- a verify snapshot's own target matching ``current`` says
    nothing about whether the COLLECT snapshot it depends on (or a
    co-existing started-marker) also still does. Comparing every
    existing target against one common reference is sufficient to catch
    any mutual disagreement too (if A and B both equal ``current``, they
    equal each other; if either does not, that mismatch is reported on
    its own), without needing separate pairwise comparisons. A ``None``
    entry (e.g. ``started`` genuinely absent) is skipped, never treated
    as a mismatch -- Phase 4.3I correction 3 explicitly preserves the
    existing contract that a missing started-marker never blocks an
    otherwise-independently-verified collect snapshot from restoring."""
    reasons: list[str] = []
    for label, target in named_targets:
        if target is None:
            continue
        mismatch = _target_mismatch_reason(current, target)
        if mismatch:
            reasons.append(f"{label}: {mismatch}")
    return reasons


def _run_fact_collector_adaptive(
    repo: Repository,
    *,
    run_id: str,
    ticker: str,
    company_name: str,
    collection_result,
) -> tuple[tuple[Fact, ...], AgentRunRecord]:
    """One standalone ``FactCollectorAgent`` call over the adaptive
    literature ``CollectionResult`` -- called directly
    (``agent.execute(AgentInput(...))``), never through
    ``Pipeline._run_agent``, mirroring ``run_full_evidence_integrity_pass``'s
    own calling convention for ``EvidenceIntegrityAgent``. The real
    agent's own ``.agent_id`` (``"fact_collector"``) is used for the
    ``AgentInput`` (so the agent's internal logic is unaffected); only the
    SAVED ``AgentRunRecord.agent_id`` is overridden to
    ``"fact_collector_adaptive"`` (Phase 4.3I correction 7)."""
    agent = FactCollectorAgent([collection_result])
    agent_input = AgentInput(
        agent_id=agent.agent_id,
        run_id=run_id,
        ticker=ticker,
        company_name=company_name,
        facts=(),
        params={"run_id": run_id},
    )
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output = agent.execute(agent_input)
    resolved_agent_id = _FACT_COLLECTOR_AUDIT_AGENT_IDS[FactCollectionPassKind.ADAPTIVE]
    record = AgentRunRecord(
        run_id=run_id,
        agent_id=resolved_agent_id,
        status=output.status,
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_ms=output.duration_ms,
        fact_count=len(output.facts),
        error_count=len(output.errors),
        errors=" | ".join(output.errors)[:2000],
    )
    record.inputs_hash = inputs_hash(agent_input)
    repo.save_agent_run(record)
    return tuple(output.facts), record


def _finalize_with_integrity(
    repo: Repository,
    *,
    run_id: str,
    ticker: str,
    company_name: str,
    verified_facts: Sequence[Fact],
    old_sources: Sequence[Source],
    collect_snapshot: AdaptiveCollectSnapshot,
    today: date | None,
    stale_after_days: int,
    extra_agent_records: tuple[AgentRunRecord, ...],
    extra_failures: tuple[str, ...],
) -> AdaptiveLiteratureStepResult:
    """Shared tail for both the "fetch just completed" path and the
    "resuming from a completed, input-verified collect snapshot" path:
    run the second (ADAPTIVE) full Evidence Integrity pass over
    verified_facts UNION collect_snapshot.raw_facts, decide this step's
    own blocking_reasons, and save the verify snapshot bound to
    ``collect_snapshot``'s own target/digest (Phase 4.3I correction 2) --
    ``collect_snapshot`` is ALWAYS the exact object just persisted (fresh
    fetch) or just read back and input-verified (resume), never rebuilt
    from loose parts, so the saved digest is always computed from exactly
    what a later restore will re-read."""
    combined_pre_integrity = tuple(verified_facts) + tuple(collect_snapshot.raw_facts)
    combined_sources = tuple(old_sources) + tuple(collect_snapshot.sources)
    pass_input = FullIntegrityPassInput(
        ticker=ticker,
        company_name=company_name,
        run_id=run_id,
        pre_integrity_facts=combined_pre_integrity,
        sources=combined_sources,
        today=today,
        stale_after_days=stale_after_days,
        pass_kind=IntegrityPassKind.ADAPTIVE,
    )
    pass_output = run_full_evidence_integrity_pass(pass_input, repo)

    blocking: list[str] = []
    if collect_snapshot.execution_status == AdaptiveExecutionStatus.INCOMPLETE.value:
        blocking.append(
            "adaptive literature acquisition ran but did not achieve full coverage "
            "(AdaptiveExecutionStatus.INCOMPLETE); the resulting evidence gap is not "
            "resolved merely because it was run through Evidence Integrity"
        )
    if pass_output.status_incomplete:
        blocking.append(
            "adaptive literature Evidence Integrity pass reported incompleteness: "
            + "; ".join(pass_output.failures)[:500]
        )

    repo.save_adaptive_verify_snapshot(
        run_id,
        AdaptiveVerifySnapshot(
            verified_facts=list(pass_output.verified_facts),
            quarantined_sources=list(pass_output.quarantined_sources),
            execution_status=collect_snapshot.execution_status,
            blocking_reasons=list(blocking),
            target=collect_snapshot.target,
            collect_snapshot_digest=adaptive_collect_snapshot_digest(collect_snapshot),
        ),
    )

    outcome = (
        AdaptiveLiteratureStepOutcome.COMPLETE
        if not blocking
        else AdaptiveLiteratureStepOutcome.INCOMPLETE
    )
    return AdaptiveLiteratureStepResult(
        outcome=outcome,
        verified_facts=tuple(pass_output.verified_facts),
        new_chunks=tuple(collect_snapshot.chunks),
        new_sources=tuple(collect_snapshot.sources),
        quarantined_sources=tuple(pass_output.quarantined_sources),
        agent_records=(*extra_agent_records, pass_output.agent_run_record),
        failures=(*extra_failures, *pass_output.failures),
        blocking_reasons=tuple(blocking),
        diagnostics={
            "outcome": str(outcome),
            "execution_status": collect_snapshot.execution_status,
            "new_fact_count": len(collect_snapshot.raw_facts),
            "new_chunk_count": len(collect_snapshot.chunks),
            "verified_fact_count": len(pass_output.verified_facts),
        },
    )


def run_adaptive_literature_step(
    *,
    repo: Repository,
    run_id: str,
    ticker: str,
    company_name: str,
    plan: AdaptiveAcquisitionPlan,
    verified_facts: Sequence[Fact],
    sources: Sequence[Source],
    base_evidence_available: bool,
    http_client: LiteratureHttpTransport | None,
    env: Mapping[str, str] | None,
    today: date | None = None,
    stale_after_days: int = 400,
) -> AdaptiveLiteratureStepResult:
    """Called by ``Pipeline.run()`` once per invocation, immediately after
    the Phase 4.3G bootstrap block and before Stage 2b escalation. Safe to
    call on every invocation, fresh or resumed.

    Phase 4.3I correction 2's decision order (see module docstring for
    the reasoning): persisted checkpoint state is read and judged FIRST,
    unconditionally -- a verify snapshot's missing collect-snapshot
    dependency, an input-binding mismatch, or a started-marker with no
    completed collect snapshot are all judged BEFORE
    ``base_evidence_available``/``plan.status``/the runner-injection
    check are even inspected. Those three checks are reached ONLY when
    NOTHING is persisted yet for this run_id.
    """
    try:
        verify_snapshot = repo.adaptive_verify_snapshot_for_resume(run_id)
        collect_snapshot = repo.adaptive_collect_snapshot_for_resume(run_id)
        started = repo.adaptive_literature_started(run_id)
    except ResumeSnapshotCorrupted as exc:
        return _blocked(
            AdaptiveLiteratureStepOutcome.SNAPSHOT_CORRUPTED,
            verified_facts,
            f"adaptive literature snapshot for run_id={run_id!r} is corrupted: {exc}",
        )

    if verify_snapshot is not None:
        # verify_snapshot's own Integrity pass was run FROM a collect
        # snapshot's pre-Integrity facts/sources/chunks (every one of this
        # module's OWN save paths writes the collect snapshot strictly
        # BEFORE the verify snapshot). A verify snapshot with no collect
        # snapshot is therefore never a "clean" state to restore from: it
        # is missing exactly the Sources/Chunks the restored Facts are
        # supposed to be backed by. Distinguished from SNAPSHOT_CORRUPTED
        # (a row that exists but fails to parse, handled above): this is
        # a row that is cleanly ABSENT, which is just as untrustworthy
        # here, and fails closed identically.
        if collect_snapshot is None:
            return _blocked(
                AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE,
                verified_facts,
                f"an adaptive literature verify snapshot exists for run_id={run_id!r} but its "
                "dependency, the collect snapshot, is missing (cleanly absent, not corrupted) "
                "-- the Sources/Chunks the restored Facts would be backed by cannot be "
                "recovered; failing closed rather than restoring an incomplete evidence set "
                "as if it were complete",
            )
        if not base_evidence_available:
            # Phase 4.3I correction 2: the CURRENT base (verified_facts)
            # is itself unavailable this invocation -- there is nothing
            # trustworthy to re-verify the snapshot's own base_fact_
            # fingerprint against, so this step refuses to restore ANY
            # Facts/Chunks/Sources from a prior invocation's snapshot,
            # clean or not. The overall run's own INCOMPLETE_RESEARCH/
            # Action=None already comes from resume_snapshot_failures
            # (Stage 1/2's own fail-closed path); this step adds no
            # separate blocking reason of its own here, it simply never
            # lets old evidence re-enter verified_facts/bus.facts.
            return _not_attempted(
                AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_BASE_EVIDENCE_UNAVAILABLE,
                verified_facts,
                "a completed adaptive verify snapshot exists for this run_id, but the base "
                "evidence set itself is unavailable this invocation -- restoring its Facts "
                "against an unknown/incomplete current base is refused rather than silently "
                "reusing stale evidence",
            )
        # Phase 4.3I correction 3: every persisted checkpoint's own target
        # is checked against the current input -- not verify_snapshot's
        # alone. collect_snapshot.target and (when present) started.target
        # are checked here too, so a divergence between what collect/
        # started/verify each believe this run_id's target to be is
        # caught directly, never left to the collect_snapshot_digest
        # check alone to notice indirectly.
        current_target = _current_target(plan, ticker, verified_facts, sources)
        target_mismatches = _all_target_mismatches(
            current_target,
            (
                ("verify_snapshot.target", verify_snapshot.target),
                ("collect_snapshot.target", collect_snapshot.target),
                ("started.target", started.target if started is not None else None),
            ),
        )
        current_digest = adaptive_collect_snapshot_digest(collect_snapshot)
        digest_mismatch = current_digest != verify_snapshot.collect_snapshot_digest
        status_mismatch = verify_snapshot.execution_status != collect_snapshot.execution_status
        if target_mismatches or digest_mismatch or status_mismatch:
            reasons = list(target_mismatches)
            if digest_mismatch:
                reasons.append(
                    "collect_snapshot_digest differs -- the verify snapshot was computed from "
                    "a different collect snapshot than the one currently on file"
                )
            if status_mismatch:
                reasons.append(
                    f"execution_status differs (verify={verify_snapshot.execution_status!r}, "
                    f"collect={collect_snapshot.execution_status!r})"
                )
            return _blocked(
                AdaptiveLiteratureStepOutcome.SNAPSHOT_INPUT_MISMATCH,
                verified_facts,
                f"adaptive literature verify snapshot for run_id={run_id!r} no longer matches "
                f"the current input, and is refused rather than restored or re-fetched: "
                + "; ".join(reasons),
            )
        # State F: Integrity already completed in a prior invocation,
        # input binding verified -- restore directly, re-fetch nothing,
        # re-run nothing. The ORIGINAL execution_status category is
        # preserved rather than re-derived from blocking_reasons alone,
        # so a restored REFUSED/SKIPPED fetch is never relabeled
        # INCOMPLETE merely because it also happens to carry a non-empty
        # blocking_reasons tuple.
        if verify_snapshot.execution_status in (
            str(AdaptiveExecutionStatus.SKIPPED), str(AdaptiveExecutionStatus.REFUSED),
        ):
            outcome = AdaptiveLiteratureStepOutcome.REFUSED
        elif not verify_snapshot.blocking_reasons:
            outcome = AdaptiveLiteratureStepOutcome.COMPLETE
        else:
            outcome = AdaptiveLiteratureStepOutcome.INCOMPLETE
        return AdaptiveLiteratureStepResult(
            outcome=outcome,
            verified_facts=tuple(verify_snapshot.verified_facts),
            new_chunks=tuple(collect_snapshot.chunks),
            new_sources=tuple(collect_snapshot.sources),
            quarantined_sources=tuple(verify_snapshot.quarantined_sources),
            agent_records=(),
            failures=tuple(verify_snapshot.blocking_reasons),
            blocking_reasons=tuple(verify_snapshot.blocking_reasons),
            diagnostics={
                "outcome": str(outcome),
                "execution_status": verify_snapshot.execution_status,
                "restored_from": "adaptive_verify_snapshot",
            },
        )

    if collect_snapshot is not None:
        # State D/E: fetch already completed and durably recorded.
        # Phase 4.3I correction 3: this is reached regardless of whether
        # a `started` marker also exists -- collect_snapshot's own save
        # order being "after started" is NEVER, by itself, treated as
        # proof of safety. Safety is established independently, here, by
        # verifying collect_snapshot's OWN stored target/base-fact
        # fingerprint against the CURRENT plan/ticker/base facts.
        if not base_evidence_available:
            return _not_attempted(
                AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_BASE_EVIDENCE_UNAVAILABLE,
                verified_facts,
                "a completed adaptive collect snapshot exists for this run_id, but the base "
                "evidence set itself is unavailable this invocation -- re-running Evidence "
                "Integrity against an unknown/incomplete current base is refused",
            )
        # Phase 4.3I correction 3: collect_snapshot.target is checked
        # against current input AND, when a started-marker also happens
        # to exist, against started.target too -- a missing started
        # marker is never treated as a mismatch (the existing "verified
        # independently, never by save order alone" contract for a
        # genuinely absent started-marker is preserved), but a PRESENT
        # one that disagrees with collect_snapshot's own target is.
        current_target = _current_target(plan, ticker, verified_facts, sources)
        target_mismatches = _all_target_mismatches(
            current_target,
            (
                ("collect_snapshot.target", collect_snapshot.target),
                ("started.target", started.target if started is not None else None),
            ),
        )
        if target_mismatches:
            return _blocked(
                AdaptiveLiteratureStepOutcome.SNAPSHOT_INPUT_MISMATCH,
                verified_facts,
                f"adaptive literature collect snapshot for run_id={run_id!r} no longer matches "
                "the current input, and is refused rather than restored or re-fetched: "
                + "; ".join(target_mismatches),
            )
        return _finalize_with_integrity(
            repo,
            run_id=run_id,
            ticker=ticker,
            company_name=company_name,
            verified_facts=verified_facts,
            old_sources=sources,
            collect_snapshot=collect_snapshot,
            today=today,
            stale_after_days=stale_after_days,
            extra_agent_records=(),
            extra_failures=(),
        )

    if started is not None:
        # State B/C: something was attempted, but whether it reached the
        # network cannot be determined from persisted state -- never
        # auto-retried, never silently treated as "not attempted".
        # Phase 4.3I correction 1: judged UNCONDITIONALLY here, before
        # base_evidence_available/plan.status are inspected at all --
        # this invocation's plan being non-READY (e.g. an explicit
        # PMID/NCT override used this time) or its base evidence being
        # unavailable must never mask a prior invocation's unresolved
        # ambiguity.
        return _blocked(
            AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE,
            verified_facts,
            f"an adaptive literature acquisition 'started' marker exists for run_id="
            f"{run_id!r} with no completed collect snapshot -- whether an external "
            "request was ever sent cannot be determined from persisted state; failing "
            "closed rather than guessing or auto-retrying",
        )

    # Nothing persisted at all for this run_id. base_evidence_available/
    # plan.status/runner-injection are checked ONLY now.
    if not base_evidence_available:
        return _not_attempted(
            AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_BASE_EVIDENCE_UNAVAILABLE,
            verified_facts,
            "the base evidence set itself was unavailable this invocation "
            "(resume snapshot fail-closed) -- adaptive literature acquisition was not "
            "attempted against a known-incomplete base",
        )

    if plan.status is not AcquisitionPlanStatus.READY:
        return _not_attempted(
            AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_NON_READY,
            verified_facts,
            f"plan.status is {plan.status}, not READY -- nothing to acquire",
        )

    if http_client is None or env is None:
        return _not_attempted(
            AdaptiveLiteratureStepOutcome.NOT_ATTEMPTED_NO_RUNNER,
            verified_facts,
            "plan.status is READY but no http_client/env was injected into this "
            "Pipeline -- execute_adaptive_literature_plan was never called",
        )

    current_target = _current_target(plan, ticker, verified_facts, sources)
    repo.save_adaptive_literature_started(run_id, AdaptiveLiteratureStarted(target=current_target))
    try:
        execution = execute_adaptive_literature_plan(
            plan, ticker=ticker, http_client=http_client, env=env,
        )
    except Exception as exc:  # noqa: BLE001
        # The started-marker above is already durably written; no collect
        # snapshot is written here on purpose -- whether the runner sent
        # zero, one, or several requests before raising cannot be known
        # from here, so this invocation reports the SAME AMBIGUOUS_
        # RESUME_STATE a later resume would independently conclude from
        # persisted state alone (never auto-retried within this same
        # call). Never includes str(exc) -- an exception from a transport
        # can embed a raw wire URL/query string; only the exception TYPE
        # name is safe to report.
        return _blocked(
            AdaptiveLiteratureStepOutcome.AMBIGUOUS_RESUME_STATE,
            verified_facts,
            "adaptive literature acquisition's runner raised "
            f"{type(exc).__name__} before returning a result -- whether an external "
            "request was ever sent cannot be determined; failing closed rather than "
            "guessing or auto-retrying",
        )

    if execution.status in (AdaptiveExecutionStatus.SKIPPED, AdaptiveExecutionStatus.REFUSED):
        reason = (
            f"adaptive literature acquisition was attempted and {execution.status}: "
            f"{execution.rationale}"
        )
        refused_collect = AdaptiveCollectSnapshot(
            execution_status=str(execution.status), target=current_target,
        )
        repo.save_adaptive_collect_snapshot(run_id, refused_collect)
        repo.save_adaptive_verify_snapshot(
            run_id,
            AdaptiveVerifySnapshot(
                verified_facts=list(verified_facts),
                execution_status=str(execution.status),
                blocking_reasons=[reason],
                target=current_target,
                collect_snapshot_digest=adaptive_collect_snapshot_digest(refused_collect),
            ),
        )
        return _blocked(AdaptiveLiteratureStepOutcome.REFUSED, verified_facts, reason)

    # COMPLETE or INCOMPLETE: a real acquisition attempt ran.
    bundle = execution.bundle
    assert bundle is not None and bundle.collection_result is not None  # guaranteed by the bridge's own contract for these two statuses
    new_chunks = tuple(bundle.chunks)
    new_facts, fact_collector_record = _run_fact_collector_adaptive(
        repo,
        run_id=run_id,
        ticker=ticker,
        company_name=company_name,
        collection_result=bundle.collection_result,
    )
    collect_snapshot_obj = AdaptiveCollectSnapshot(
        raw_facts=list(new_facts),
        sources=list(bundle.collection_result.sources),
        chunks=list(new_chunks),
        execution_status=str(execution.status),
        target=current_target,
    )
    repo.save_adaptive_collect_snapshot(run_id, collect_snapshot_obj)
    return _finalize_with_integrity(
        repo,
        run_id=run_id,
        ticker=ticker,
        company_name=company_name,
        verified_facts=verified_facts,
        old_sources=sources,
        collect_snapshot=collect_snapshot_obj,
        today=today,
        stale_after_days=stale_after_days,
        extra_agent_records=(fact_collector_record,),
        extra_failures=(),
    )


__all__ = [
    "AdaptiveLiteratureStepOutcome",
    "AdaptiveLiteratureStepResult",
    "FactCollectionPassKind",
    "run_adaptive_literature_step",
]
