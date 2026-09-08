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
    today: date | None = None,
) -> ResumePlan:
    """Work out where to restart and which facts survive."""
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
