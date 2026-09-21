"""Phase 4.3C: the "full Evidence Integrity pass" -- extracted, as one
named unit, from ``orchestrator.pipeline.Pipeline.run()``'s Stage 2 (base
commit ``3996fdf``, lines ~589-695).

Reading that Stage exactly (never approximated) shows it is NOT just an
``EvidenceIntegrityAgent`` call -- it is nine things done together, every
run:

1. ``EvidenceIntegrityAgent`` execution.
2. ``verified_facts`` determination (``list(integrity.facts) or raw_facts``
   -- a genuinely empty Integrity output falls back to the raw set, never
   to an empty one).
3. Replacing ``EvidenceBus.facts`` with the verified set (never merged
   with whatever ``FactCollectorAgent``'s own raw facts already publish
   there).
4. Persisting ``Source`` records (``Repository.save_sources``).
5. Malformed-``Source`` quarantine (a side effect of #4: ``save_sources``
   rejects, rather than persists, a ``Source`` that fails schema
   validation).
6. Excluding any ``Fact`` that rests ONLY on a quarantined ``Source`` --
   before it can reach scoring/completeness/the Evidence Sufficiency
   Matrix.
7. Recording ``quarantined_sources``/``result.failures``/``ctx.status``
   for both #5 and #6.
8. The Phase 4.2B-correction-1 Literature completeness-consistency check:
   when a Literature Source was quarantined (#5) or a Literature Fact was
   excluded (#6 or Evidence Integrity's own exclusion) despite the
   acquisition stage having reported ``coverage_complete=True``,
   ``direct_acquisition_info`` is corrected (``coverage_complete`` flips
   to ``False``, one ``unresolved_reasons`` entry is appended) so it can
   never keep reporting stale success.
9. Persisting every surviving ``Fact`` (``Repository.save_fact``, itself
   idempotent -- an unchanged fact is a no-op, never a duplicate write)
   and the ``EvidenceIntegrityAgent`` call's own audit record
   (``Repository.save_agent_run``).

This module extracts ALL nine as one function operating on plain,
explicitly-passed data (:class:`FullIntegrityPassInput` in,
:class:`FullIntegrityPassOutput` out) plus an injected ``Repository`` --
never a global, never an implicit ``EvidenceBus``/``IsolationGuard``/live
``Pipeline`` dependency. ``EvidenceIntegrityAgent`` has no LLM variant and
its ``run()`` reads only ``AgentInput.facts`` (confirmed by reading
``agents/evidence_integrity.py`` in full) -- its own bus-publish/
fingerprint side effects inside ``Pipeline._run_agent`` are provably inert
for this specific agent (its output never sets ``.sources``/
``.risk_flags``/``.contradictions``/``.unresolved``/``.evaluation``, and
its ``.facts`` publish is immediately overwritten by Stage 2's own
``bus.facts = verified_facts`` regardless), so invoking it directly here
(``agent.execute(AgentInput(...))``, never ``Pipeline._run_agent``) is
behaviorally identical to today's call site for everything that matters,
while making this pass genuinely callable with no live Pipeline at all --
required for the offline 2-pass harness Phase 4.3C's own test suite
builds (never connected to Pipeline/CLI/HTTP/LLM).

Production still calls this exactly ONCE per run (see
``Pipeline.run()``'s Stage 2, updated in this same phase to call
``run_full_evidence_integrity_pass`` once, replacing its previous inline
block) -- this phase changes nothing about production's observable
behavior; see ``tests/unit/test_evidence_integrity_pass.py``'s own
single-pass-compatibility tests. A second pass, for a future Adaptive
Acquisition step, is proven SAFE here (this module, offline, in tests)
but is not connected to Pipeline in this phase.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ..agents.base import inputs_hash
from ..agents.evidence_integrity import EvidenceIntegrityAgent
from ..collectors.base import CollectionResult
from ..schemas.agent_io import AgentInput, AgentRunRecord
from ..schemas.fact import Fact, Source
from ..schemas.validation import QuarantinedSource
from ..storage.repository import Repository

#: Mirrors ``orchestrator.pipeline._LITERATURE_BRIDGE_COLLECTOR_LABEL``
#: and ``research.literature_evidence_projection.BRIDGE_COLLECTOR_LABEL``
#: exactly. Duplicated (not imported) for the same reason ``pipeline.py``'s
#: own copy already gives (see that constant's own docstring): this module
#: must never gain an import edge onto the Literature Acquisition path
#: merely to borrow one string constant used for classification only. A
#: dedicated test asserts all three stay equal.
LITERATURE_BRIDGE_COLLECTOR_LABEL = "literature_evidence_projection"


@dataclass(frozen=True)
class FullIntegrityPassInput:
    """Everything one full Evidence Integrity pass needs -- plain data
    only. Built by ``Pipeline.run()`` for the one production pass, or by
    an offline test harness for a second, later pass over a combined
    (initial UNION new) evidence set -- never by appending new facts
    directly to an already-verified set (Stage 3b's
    ``escalate_unresolved_questions`` pattern is deliberately NOT reused
    here; see this module's own top docstring, item 2)."""

    ticker: str
    company_name: str
    run_id: str
    #: Stage-1 Facts (already through ``FactCollectorAgent``, NOT yet
    #: through Evidence Integrity) -- this pass runs
    #: ``EvidenceIntegrityAgent`` over exactly this set, in full, every
    #: call; it never re-uses a prior pass's already-assessed
    #: ``verified_facts`` as a substitute for re-running Integrity on the
    #: full combined set.
    raw_facts: tuple[Fact, ...]
    #: Every ``Source`` known so far (already deduplicated by the caller,
    #: e.g. ``EvidenceBus.add_sources``'s own dedup) -- this pass persists/
    #: quarantines exactly this set, in full, every call.
    sources: tuple[Source, ...]
    #: Needed only for the Literature completeness-consistency check
    #: (item 8) -- read-only, never mutated, never used to acquire
    #: anything.
    collection_results: tuple[CollectionResult, ...] = ()
    #: The direct_acquisition_info dict going INTO this pass (e.g.
    #: ``result.direct_acquisition_info`` at the time of the call) --
    #: this pass may return an UPDATED copy (see
    #: :attr:`FullIntegrityPassOutput.direct_acquisition_info`), never
    #: mutates this input in place.
    direct_acquisition_info: Mapping[str, Any] = field(default_factory=dict)
    today: date | None = None
    stale_after_days: int = 400
    #: True ONLY for ``Pipeline.run()``'s pre-existing ``--resume`` shortcut
    #: (base commit ``3996fdf``, lines 563-578), which explicitly skips
    #: re-running ``EvidenceIntegrityAgent`` over restored facts -- when
    #: True, ``raw_facts`` is treated as ALREADY verified (no agent is
    #: invoked, no ``AgentRunRecord`` is produced) and this pass performs
    #: only items 3-9. Every other caller, including both passes of the
    #: offline 2-pass harness, MUST leave this ``False`` -- a second pass
    #: that skipped re-running Evidence Integrity would be exactly the
    #: "bypass Integrity" shape Phase 4.3C requirement B/6 forbids.
    already_verified: bool = False


@dataclass(frozen=True)
class FullIntegrityPassOutput:
    verified_facts: tuple[Fact, ...]
    quarantined_sources: tuple[QuarantinedSource, ...]
    #: New, result.failures-shaped messages this pass produced -- the
    #: caller APPENDS these to its own list, never replaces it (this pass
    #: never sees, and cannot know about, failures from any other stage).
    failures: tuple[str, ...]
    #: True when this pass alone requires ``RunStatus.INCOMPLETE_RESEARCH``
    #: (an ``EvidenceIntegrityAgent``-level failure, a fact-persistence
    #: exception, or at least one quarantined source) -- the caller ORs
    #: this into ``ctx.status``, never treats ``False`` here as license to
    #: downgrade a status some OTHER stage already raised.
    status_incomplete: bool
    #: Possibly-updated ``direct_acquisition_info`` (``coverage_complete``
    #: flipped ``False``, one ``unresolved_reasons`` entry appended) -- see
    #: item 8 above. Identical to the input when no Literature
    #: inconsistency was found (including when the input had no Literature
    #: acquisition info at all).
    direct_acquisition_info: Mapping[str, Any]
    #: The ``EvidenceIntegrityAgent`` call's own audit record, or ``None``
    #: for the ``already_verified=True`` (resume) case, where no agent was
    #: invoked at all. The caller is responsible for appending a non-``None``
    #: record to whatever ``result.agent_records`` list production uses;
    #: this module never holds one itself.
    agent_run_record: AgentRunRecord | None


def run_full_evidence_integrity_pass(
    pass_input: FullIntegrityPassInput, repo: Repository,
) -> FullIntegrityPassOutput:
    """The full Evidence Integrity pass (Phase 4.3C). See this module's
    own top docstring for the nine things this reproduces, read directly
    from ``Pipeline.run()``'s existing Stage 2 at base commit ``3996fdf``.
    """
    failures: list[str] = []
    status_incomplete = False
    record: AgentRunRecord | None = None

    if pass_input.already_verified:
        # The --resume shortcut: raw_facts are already-verified facts
        # restored from a prior run; EvidenceIntegrityAgent is never
        # invoked, exactly like today's Pipeline.run() resume branch.
        verified_facts: list[Fact] = list(pass_input.raw_facts)
    else:
        agent = EvidenceIntegrityAgent(
            today=pass_input.today, stale_after_days=pass_input.stale_after_days
        )
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        agent_input = AgentInput(
            agent_id=agent.agent_id,
            run_id=pass_input.run_id,
            ticker=pass_input.ticker,
            company_name=pass_input.company_name,
            facts=tuple(pass_input.raw_facts),
        )
        started_monotonic = time.monotonic()
        output = agent.execute(agent_input)
        elapsed = time.monotonic() - started_monotonic
        if elapsed > agent.timeout_seconds:
            output.degraded = True
            output.errors.append(
                f"agent exceeded its {agent.timeout_seconds}s budget ({elapsed:.1f}s)"
            )

        record = AgentRunRecord(
            run_id=pass_input.run_id,
            agent_id=output.agent_id,
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

        if not output.ok or output.degraded:
            message = f"{agent.agent_id}: {output.status} :: {'; '.join(output.errors)[:300]}"
            failures.append(message)
            if not output.ok:
                status_incomplete = True

        # Item 2: a genuinely empty Integrity output falls back to
        # raw_facts, never to an empty verified set.
        verified_facts = list(output.facts) or list(pass_input.raw_facts)

    # Items 4/5: persist Sources, quarantining (never crashing on) a
    # malformed one.
    quarantined_sources = repo.save_sources(pass_input.sources)
    if quarantined_sources:
        status_incomplete = True
        for q in quarantined_sources:
            failures.append(
                f"quarantined malformed source {q.source_id} ({q.url}): "
                f"{q.field}={q.value!r} -- {q.error}"
            )
        # Item 6: any fact resting only on a quarantined source is
        # evidence this pass cannot actually stand behind -- excluded
        # before it can reach scoring/completeness/the Evidence
        # Sufficiency Matrix.
        quarantined_ids = {q.source_id for q in quarantined_sources}
        before = len(verified_facts)
        verified_facts = [f for f in verified_facts if f.source_id not in quarantined_ids]
        dropped = before - len(verified_facts)
        if dropped:
            failures.append(
                f"{dropped} fact(s) resting only on a quarantined source were excluded "
                "from evidence"
            )

    # Item 8: Literature completeness-consistency (Phase 4.2B correction 1,
    # reproduced verbatim). A clean, complete Literature fetch (or a run
    # that never used the flag at all) is entirely unaffected.
    direct_acquisition_info: Mapping[str, Any] = pass_input.direct_acquisition_info
    if direct_acquisition_info.get("feature_enabled") and direct_acquisition_info.get(
        "coverage_complete"
    ):
        literature_collection_results = [
            c
            for c in pass_input.collection_results
            if c.collector == LITERATURE_BRIDGE_COLLECTOR_LABEL
        ]
        literature_raw_fact_ids = {
            raw.fact_id() for c in literature_collection_results for raw in c.raw_facts
        }
        missing_literature_fact_ids = literature_raw_fact_ids - {f.fact_id for f in verified_facts}
        if missing_literature_fact_ids:
            literature_source_ids = {
                s.source_id for c in literature_collection_results for s in c.sources
            }
            quarantined_literature_source_ids = {
                q.source_id for q in quarantined_sources
            } & literature_source_ids
            if quarantined_literature_source_ids:
                reason = (
                    f"{len(missing_literature_fact_ids)} Literature-sourced fact(s) were "
                    f"excluded after {len(quarantined_literature_source_ids)} Literature "
                    "source(s) were quarantined by Source validation (source_id(s): "
                    f"{', '.join(sorted(quarantined_literature_source_ids))})"
                )
            else:
                reason = (
                    f"{len(missing_literature_fact_ids)} Literature-sourced fact(s) "
                    "submitted by the acquisition bridge did not survive Evidence Integrity"
                )
            updated_direct_acquisition_info = dict(direct_acquisition_info)
            updated_direct_acquisition_info["coverage_complete"] = False
            updated_direct_acquisition_info["unresolved_reasons"] = [
                *updated_direct_acquisition_info.get("unresolved_reasons", []),
                reason,
            ]
            direct_acquisition_info = updated_direct_acquisition_info

    # Item 9: persist every surviving fact. save_fact is idempotent (an
    # unchanged fact is a no-op, never a duplicate row/version).
    for fact in verified_facts:
        try:
            repo.save_fact(fact)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"fact persistence failed for {fact.fact_id}: {exc}")
            status_incomplete = True

    return FullIntegrityPassOutput(
        verified_facts=tuple(verified_facts),
        quarantined_sources=tuple(quarantined_sources),
        failures=tuple(failures),
        status_incomplete=status_incomplete,
        direct_acquisition_info=direct_acquisition_info,
        agent_run_record=record,
    )


__all__ = [
    "LITERATURE_BRIDGE_COLLECTOR_LABEL",
    "FullIntegrityPassInput",
    "FullIntegrityPassOutput",
    "run_full_evidence_integrity_pass",
]
