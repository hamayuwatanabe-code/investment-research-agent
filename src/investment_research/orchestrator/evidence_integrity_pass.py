"""Phase 4.3C: the "full Evidence Integrity pass" -- extracted, as one
named unit, from ``orchestrator.pipeline.Pipeline.run()``'s Stage 2 (base
commit ``3996fdf``, lines ~589-695).

Reading that Stage exactly (never approximated) shows it is NOT just an
``EvidenceIntegrityAgent`` call -- it is nine things done together, every
run:

1. ``EvidenceIntegrityAgent`` execution.
2. ``verified_facts`` determination (``list(integrity.facts) or
   pre_integrity_facts`` -- a genuinely empty Integrity output falls back
   to the pre-Integrity set, never to an empty one).
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

---------------------------------------------------------------------------
Phase 4.3C Correction 1: no bypass, ever
---------------------------------------------------------------------------

Phase 4.3C's own first cut added a boolean skip-flag field (named for the
notion of facts being "already verified") that, when set, skipped
constructing and executing ``EvidenceIntegrityAgent`` entirely -- meant
only for ``Pipeline.run()``'s pre-existing ``--resume`` shortcut.
Investigating it for this correction found it was worse than merely
redundant: ``Pipeline.run()``'s own wiring passed the (empty, in resume
mode) ``raw_facts`` LOCAL VARIABLE into that bypass path rather than
``resume_plan.restored_facts`` -- so a resumed run would have silently
discarded its own restored facts the moment this pass's output was
assigned back to ``verified_facts``. No test caught this because no
existing test drives ``Pipeline.run(resume=True)`` end to end.

This correction removes the bypass completely, as the far safer design:
``run_full_evidence_integrity_pass`` now ALWAYS executes
``EvidenceIntegrityAgent`` for real, on every call, unconditionally --
"calling this function" and "Evidence Integrity actually ran" are now the
same fact, structurally, not by convention. ``Pipeline.run()``'s own
``--resume`` branch is updated (see that module) to feed
``resume_plan.restored_facts`` into this pass as
``pre_integrity_facts`` -- meaning a resumed run now genuinely
RE-VERIFIES its restored facts rather than trusting a stale, unverified
assumption; this is a deliberate, in-scope consequence of removing the
bypass, not an accidental behavior change smuggled in alongside it. That
skip-flag field itself is deleted, not merely unused, so "skip Integrity"
is not just discouraged but structurally impossible to express through
this contract at all.

---------------------------------------------------------------------------
Phase 4.3C Correction 1: same-run_id audit identity
---------------------------------------------------------------------------

``storage/schema.sql``'s ``agent_runs`` table has
``PRIMARY KEY (run_id, agent_id)`` (confirmed by reading it directly) --
two calls to ``Repository.save_agent_run`` sharing BOTH values silently
overwrite each other (``INSERT OR REPLACE``). Since a future Adaptive
Acquisition step would call this pass a second time under the SAME
``run_id``, ``FullIntegrityPassInput.audit_agent_id`` lets a caller give
each call's ``AgentRunRecord`` a distinct ``agent_id`` -- defaulting to
``EvidenceIntegrityAgent.agent_id`` itself (``"evidence_integrity"``),
which is EXACTLY today's (and Phase 4.3C's own) single-pass production
identity, so a caller that never sets this field (production, today)
gets a byte-identical audit record to before this correction. This is a
plain string field, not a new table/column -- no DB schema migration.
The real ``EvidenceIntegrityAgent`` instance's own ``.agent_id``
(``"evidence_integrity"``, a class attribute) is never changed by this --
only the STORED ``AgentRunRecord.agent_id`` a given call chooses to file
its own audit row under.

Production still calls this exactly ONCE per run (see
``Pipeline.run()``'s Stage 2) -- this phase changes nothing about
production's observable output; see
``tests/unit/test_evidence_integrity_pass.py``'s own single-pass-
compatibility tests. A second pass, for a future Adaptive Acquisition
step, is proven SAFE here (this module, offline, in tests) but is not
connected to Pipeline in this phase.
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
    #: Facts already through ``FactCollectorAgent`` (a ``RawFact`` ->
    #: ``Fact`` conversion this helper never performs itself -- that
    #: conversion is exclusively ``FactCollectorAgent._to_fact``'s job;
    #: this field is genuinely ``Fact``, never ``RawFact``), NOT yet
    #: through Evidence Integrity -- this pass runs
    #: ``EvidenceIntegrityAgent`` over exactly this set, in full, EVERY
    #: call, unconditionally (Phase 4.3C correction 1: there is no input
    #: shape or flag that skips this).
    pre_integrity_facts: tuple[Fact, ...]
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
    #: The identity this call's own ``AgentRunRecord.agent_id`` is filed
    #: under (Phase 4.3C correction 1). Defaults to
    #: ``EvidenceIntegrityAgent.agent_id`` itself (``"evidence_integrity"``)
    #: -- production's one call never overrides this, so its audit record
    #: is identical to every prior phase's. A second, later pass sharing
    #: the SAME ``run_id`` (offline harness only, never Production in this
    #: phase) MUST pass a different value here, or its
    #: ``Repository.save_agent_run`` call will silently overwrite the
    #: first pass's own row (``agent_runs``'s ``PRIMARY KEY (run_id,
    #: agent_id)`` -- see this module's own top docstring).
    audit_agent_id: str = "evidence_integrity"


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
    #: The ``EvidenceIntegrityAgent`` call's own audit record -- ALWAYS
    #: populated (Phase 4.3C correction 1: this pass never skips running
    #: the agent, so there is never a case with no record to return). The
    #: caller is responsible for appending it to whatever
    #: ``result.agent_records`` list production uses; this module never
    #: holds one itself.
    agent_run_record: AgentRunRecord


def run_full_evidence_integrity_pass(
    pass_input: FullIntegrityPassInput, repo: Repository,
) -> FullIntegrityPassOutput:
    """The full Evidence Integrity pass (Phase 4.3C). See this module's
    own top docstring for the nine things this reproduces, read directly
    from ``Pipeline.run()``'s existing Stage 2 at base commit ``3996fdf``,
    and for Phase 4.3C correction 1's own two changes (no bypass; a
    same-run_id-safe audit identity).
    """
    failures: list[str] = []
    status_incomplete = False

    agent = EvidenceIntegrityAgent(
        today=pass_input.today, stale_after_days=pass_input.stale_after_days
    )
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    agent_input = AgentInput(
        agent_id=agent.agent_id,
        run_id=pass_input.run_id,
        ticker=pass_input.ticker,
        company_name=pass_input.company_name,
        facts=tuple(pass_input.pre_integrity_facts),
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
        agent_id=pass_input.audit_agent_id,
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
        message = f"{pass_input.audit_agent_id}: {output.status} :: {'; '.join(output.errors)[:300]}"
        failures.append(message)
        if not output.ok:
            status_incomplete = True

    # Item 2: a genuinely empty Integrity output falls back to
    # pre_integrity_facts, never to an empty verified set.
    verified_facts: list[Fact] = list(output.facts) or list(pass_input.pre_integrity_facts)

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
