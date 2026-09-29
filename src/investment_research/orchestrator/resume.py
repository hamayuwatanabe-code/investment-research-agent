"""Run checkpointing and resume (requirement P8).

An interrupted run is expensive to repeat: the collection stage is where the
network calls and the token spend live. So each stage records a checkpoint, and
``--resume RUN_ID`` restarts from the last successful one.

Two properties keep resume honest:

* **Facts are re-used, not re-asserted.** Restored facts come back with their
  original ``run_id`` and provenance intact, so the report still shows when each
  piece of evidence was actually obtained.
* **Stale evidence is refetched.** A fact whose source is older than the
  freshness threshold for its category is dropped from the restored set and
  collected again. Resuming onto a stale regulatory fact would be worse than
  starting over, because it would look current.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..schemas.enums import FactCategory
from ..schemas.fact import Fact, parse_iso_date

log = logging.getLogger(__name__)

#: Ordered pipeline stages. The index is what "resume from" means.
STAGES: tuple[str, ...] = (
    "collect",
    "verify",
    "escalate",
    "domain",
    "escalate_unresolved",
    "contradiction",
    "kill",
    "bear_bull",
    "valuation",
    "score",
    "judge",
    "portfolio",
    "report",
)

#: How long a fact of each category stays usable before it must be refetched.
#: Regulatory and liquidity evidence goes stale fastest because it is what
#: changes a thesis; a registry design field can be months old and still true.
FRESHNESS_DAYS: dict[FactCategory, int] = {
    FactCategory.REGULATORY: 14,
    FactCategory.LIQUIDITY: 30,
    FactCategory.FINANCIAL: 45,
    FactCategory.CAPITAL_STRUCTURE: 45,
    FactCategory.CATALYST: 14,
    FactCategory.MICROSTRUCTURE: 3,
    FactCategory.CLINICAL: 90,
    FactCategory.COMPETITION: 60,
    FactCategory.LEGAL: 30,
    FactCategory.GOVERNANCE: 60,
    FactCategory.ACCOUNTING: 45,
    FactCategory.LISTING: 21,
}
DEFAULT_FRESHNESS_DAYS = 90


@dataclass
class Checkpoint:
    run_id: str
    stage: str
    stage_index: int
    status: str = "OK"
    payload: dict[str, Any] = field(default_factory=dict)
    fact_count: int = 0


@dataclass
class ResumePlan:
    """What a resumed run should do.

    Phase 4.3C correction 5: ``restored_facts`` is now ALWAYS
    snapshot-sourced -- never a ``facts``/``sources`` table query. Reading
    ``storage/repository.py``'s ``save_fact`` directly (never by
    assumption) found that BOTH the ``MAX(version)`` query
    (``Repository.facts_for_resume``) and the correction-4-introduced
    ``MIN(version)`` one (``Repository.pristine_facts_for_resume``, now
    removed) share one fatal defect: ``save_fact`` looks up the latest row
    for a ``fact_id`` GLOBALLY, never scoped by ``run_id``, and treats
    byte-identical content as a no-op that writes nothing. Empirically
    reproduced (see this correction's own completion report): two runs
    that happen to collect byte-identical Stage-1 content -- an entirely
    realistic case, e.g. re-researching a company whose filings have not
    changed since the last run -- leave the SECOND run_id with ZERO rows
    of its own for that fact_id, so a query scoped to that run_id (MIN or
    MAX alike) silently returns nothing, even though checkpoints correctly
    show the stage complete. The prior fallback (silently using the OTHER
    run's version, or an empty set) is exactly the ambiguity this
    correction removes.

    ``restored_facts`` is therefore filled ONLY from an explicit,
    run_id-scoped snapshot (``Repository.collect_snapshot_for_resume`` or
    ``Repository.verify_snapshot_for_resume``, chosen by ``Pipeline.run()``
    from which stage needs resuming) that the caller already fetched
    before calling ``build_plan`` -- never queried by this module itself.
    When no such snapshot could be obtained, the caller passes
    ``snapshot_missing_reason`` instead of facts, and ``restored_facts``
    stays empty: this is NEVER interpreted as "nothing to restore" (which
    would look identical to a genuinely empty collection) by
    ``Pipeline.run()``, which checks ``snapshot_unavailable_reason``
    explicitly and fails closed.
    """

    run_id: str
    resume_from_stage: str
    resume_from_index: int
    restored_facts: list[Fact] = field(default_factory=list)
    stale_fact_ids: list[str] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    reason: str = ""
    #: Set (to a human-readable reason) when a snapshot was NEEDED for
    #: this resume point but could not be obtained -- missing entirely, or
    #: present but corrupted. ``restored_facts`` is empty whenever this is
    #: set, and that emptiness must never be read as "genuinely collected
    #: zero facts": the caller is responsible for treating this as a
    #: fail-closed condition (status=INCOMPLETE_RESEARCH, Action=None, a
    #: specific blocking reason -- never a silent, successful-looking
    #: resume). ``None`` whenever no snapshot was needed (a fresh start)
    #: or the needed one was found intact.
    snapshot_unavailable_reason: str | None = None

    @property
    def is_fresh_start(self) -> bool:
        return self.resume_from_index == 0

    def should_run(self, stage: str) -> bool:
        try:
            return STAGES.index(stage) >= self.resume_from_index
        except ValueError:  # pragma: no cover - programmer error
            return True


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def resume_point(checkpoints: Sequence[Checkpoint]) -> int:
    """The stage index a resume would restart from, judged purely from
    which stages are checkpointed OK -- no fact/snapshot lookup involved.
    ``Pipeline.run()`` calls this BEFORE ``build_plan`` to know which
    snapshot (if any) it needs to fetch: 0 means a fresh start (no
    snapshot needed at all), 1 means "collect done, verify not done" (the
    collect/pristine snapshot is needed), 2+ means "verify done" (the
    verify/verified snapshot is needed). ``build_plan`` calls this too,
    so the two never compute the index differently."""
    completed = [c.stage for c in checkpoints if c.status == "OK"]
    if not completed:
        return 0
    last_index = max(stage_index(stage) for stage in completed if stage in STAGES)
    return min(last_index + 1, len(STAGES) - 1)


def freshness_days(category: FactCategory) -> int:
    return FRESHNESS_DAYS.get(category, DEFAULT_FRESHNESS_DAYS)


def is_stale(fact: Fact, today: date) -> bool:
    """Whether this fact must be refetched before it can be reused.

    Judged on the retrieval-relevant date: the event date if there is one, else
    the publication date. A fact with no usable date at all is treated as stale,
    because an undated fact cannot be shown to be current.
    """
    for candidate in (
        fact.event_date,
        fact.effective_date,
        fact.filing_date,
        fact.publication_date,
    ):
        parsed = parse_iso_date(candidate)
        if parsed:
            return (today - parsed) > timedelta(days=freshness_days(fact.category))
    return True


def build_plan(
    run_id: str,
    checkpoints: Sequence[Checkpoint],
    *,
    snapshot_facts: Sequence[Fact] = (),
    snapshot_missing_reason: str | None = None,
    today: date | None = None,
) -> ResumePlan:
    """Work out where to restart and which facts survive.

    ``snapshot_facts`` (Phase 4.3C correction 5) is whichever run_id-scoped
    snapshot ``Pipeline.run()`` already fetched for this exact resume
    point via ``resume_point`` (the collect/pristine snapshot when
    resuming before "verify", the verify/verified snapshot when resuming
    after it) -- this module never queries the database itself.
    ``snapshot_missing_reason``, set instead by the caller when that fetch
    failed (missing or corrupted), short-circuits straight to a
    ``ResumePlan`` with ``restored_facts`` empty and
    ``snapshot_unavailable_reason`` set -- staleness is not even
    evaluated, since there is nothing trustworthy to evaluate it on.
    """
    today = today or datetime.now(timezone.utc).date()
    completed = [c.stage for c in checkpoints if c.status == "OK"]

    if not completed:
        return ResumePlan(
            run_id=run_id,
            resume_from_stage=STAGES[0],
            resume_from_index=0,
            reason="no completed stages recorded; running from the start",
        )

    last_index = max(stage_index(stage) for stage in completed if stage in STAGES)
    resume_index = min(last_index + 1, len(STAGES) - 1)

    if snapshot_missing_reason is not None:
        return ResumePlan(
            run_id=run_id,
            resume_from_stage=STAGES[resume_index],
            resume_from_index=resume_index,
            completed_stages=completed,
            reason=f"resume snapshot unavailable: {snapshot_missing_reason}",
            snapshot_unavailable_reason=snapshot_missing_reason,
        )

    fresh: list[Fact] = []
    stale: list[str] = []
    for fact in snapshot_facts:
        if is_stale(fact, today):
            stale.append(fact.fact_id)
        else:
            fresh.append(fact)

    if stale:
        # Anything downstream of collection was computed from a fact set that no
        # longer holds, so re-collect rather than resume onto stale evidence.
        resume_index = 0
        reason = (
            f"{len(stale)} restored fact(s) are past their freshness threshold; "
            "re-collecting from the start so nothing stale is presented as current"
        )
    else:
        reason = (
            f"resuming after {STAGES[last_index]!r}; "
            f"{len(fresh)} fact(s) restored, all within freshness thresholds"
        )

    return ResumePlan(
        run_id=run_id,
        resume_from_stage=STAGES[resume_index],
        resume_from_index=resume_index,
        restored_facts=fresh,
        stale_fact_ids=stale,
        completed_stages=completed,
        reason=reason,
    )


def serialize_payload(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, default=str)[:100_000]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "{}"
