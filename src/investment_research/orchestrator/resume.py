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
    """What a resumed run should do."""

    run_id: str
    resume_from_stage: str
    resume_from_index: int
    restored_facts: list[Fact] = field(default_factory=list)
    #: Phase 4.3C correction 4: the PRISTINE, pre-Integrity version of
    #: each fact in ``restored_facts`` that this run_id's own Stage 1
    #: (collect) actually produced -- for use instead of
    #: ``restored_facts`` whenever ``should_run("verify")`` is True (see
    #: ``Pipeline.run()``'s own Stage 2), so a resumed re-run of Evidence
    #: Integrity always evaluates the ORIGINAL collect-time content, never
    #: a partially- or fully-Integrity-processed version an interrupted
    #: pass happened to leave behind. Populated only for the fact_ids
    #: ``Repository.pristine_facts_for_resume`` could actually find (see
    #: that method's own docstring for the two schema-inherent reasons it
    #: cannot always); ``pristine_missing_fact_ids`` names the rest, and
    #: is never silently empty-filled by this dataclass or ``build_plan``.
    pristine_facts: list[Fact] = field(default_factory=list)
    #: fact_ids present in (fresh, non-stale) ``restored_facts`` for which
    #: no pristine version could be found. ``Pipeline.run()`` is
    #: responsible for reporting this explicitly and falling back, per
    #: fact, to that fact's own ``restored_facts`` (latest) version
    #: instead -- this dataclass only records which fact_ids need that
    #: fallback, it does not perform it.
    pristine_missing_fact_ids: list[str] = field(default_factory=list)
    stale_fact_ids: list[str] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    reason: str = ""

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
    facts: Sequence[Fact],
    *,
    pristine_facts: Sequence[Fact] = (),
    today: date | None = None,
) -> ResumePlan:
    """Work out where to restart and which facts survive.

    ``facts`` is the LATEST version of each fact_id this run_id has
    persisted (``Repository.facts_for_resume``'s own contract);
    ``pristine_facts`` (Phase 4.3C correction 4) is the EARLIEST --
    ``Repository.pristine_facts_for_resume``'s own contract. Both are
    filtered by the SAME staleness decision here, since Evidence
    Integrity never rewrites a fact's dates (confirmed by reading
    ``agents/evidence_integrity.py``'s own ``_assess`` directly: its
    ``replace(...)`` call never touches ``event_date``/
    ``publication_date``/``effective_date``/``filing_date``), so the two
    versions of the same fact_id are always equally stale or equally
    fresh.
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

    fresh: list[Fact] = []
    stale: list[str] = []
    for fact in facts:
        if is_stale(fact, today):
            stale.append(fact.fact_id)
        else:
            fresh.append(fact)

    pristine_fresh: list[Fact] = []
    pristine_missing: list[str] = []
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
        pristine_by_id = {f.fact_id: f for f in pristine_facts}
        for fact in fresh:
            pristine = pristine_by_id.get(fact.fact_id)
            if pristine is not None:
                pristine_fresh.append(pristine)
            else:
                pristine_missing.append(fact.fact_id)

    return ResumePlan(
        run_id=run_id,
        resume_from_stage=STAGES[resume_index],
        resume_from_index=resume_index,
        restored_facts=fresh,
        pristine_facts=pristine_fresh,
        pristine_missing_fact_ids=pristine_missing,
        stale_fact_ids=stale,
        completed_stages=completed,
        reason=reason,
    )


def serialize_payload(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, default=str)[:100_000]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "{}"
