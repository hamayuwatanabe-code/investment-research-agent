"""Decision-gate output consistency (requirement G).

When ``ResearchStatus`` is not COMPLETE, no rendered surface -- headline,
reasoning, thesis breakers, red flags, caveats, the report, the JSON export
-- may contain an actionable label. WAIT_FOR_EVENT is itself an ``Action``
and must never stand in for "there is not enough evidence to decide."

The leak this guards against: the Blind Judge (deterministic or LLM-backed)
bakes its internally-selected ``Action`` directly into free text (a headline
reading "... action WAIT_FOR_EVENT.") at judgement time -- before the
pipeline later learns the Search Completeness Gate or the Decision-Grade
Evidence Gate must withhold that Action and nulls ``verdict.action``. Nulling
the field is not enough; the free text was already written and would still
render.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..schemas.enums import Action, ResearchStatus

if TYPE_CHECKING:
    from ..schemas.agent_io import Evaluation
    from ..schemas.evaluation import Verdict

#: Threshold at which a CONFIRMED kill finding must be surfaced as its own
#: prominent, standalone factual disqualifier signal -- independent of, and
#: never converted into, an Action -- even while research_status withholds
#: the Action itself.
_CONFIRMED_DISQUALIFIER_LEVEL = 5

#: Every permitted final action label -- matched as a whole word so "NONE",
#: "BLOCKED_PENDING_VERIFICATION" and similar compliant text never trip this.
_ACTION_LABEL_RE = re.compile(r"\b(" + "|".join(re.escape(a.value) for a in Action) + r")\b")


def find_action_labels(text: str) -> list[str]:
    """Every Action-enum token literally present in ``text``, in order."""
    return _ACTION_LABEL_RE.findall(text or "")


def no_action_headline(research_status: ResearchStatus, blocking_reasons: Sequence[str]) -> str:
    """The ONLY headline a non-COMPLETE verdict may carry.

    Regenerated wholesale rather than edited, because a headline written
    before the final ``research_status`` was known cannot be trusted to have
    avoided asserting a judgement it is no longer entitled to make. States
    the ACTUAL ``research_status`` (BLOCKED_PENDING_VERIFICATION or plain
    INCOMPLETE) rather than assuming one -- ResearchStatus.COMPLETE is the
    only state that may emit an Action.
    """
    reasons = "; ".join(blocking_reasons)[:300] if blocking_reasons else "insufficient evidence"
    return f"RESEARCH_STATUS: {research_status} -- no action label issued ({reasons})"


def scrub_action_labels(text: str) -> str:
    """Replace any leaked Action label with a neutral marker.

    Defense-in-depth for free text that is not simply regenerated outright
    (unlike the headline): reasoning, thesis breakers, red flags, caveats.
    """
    return _ACTION_LABEL_RE.sub("[ACTION LABEL WITHHELD -- RESEARCH INCOMPLETE]", text or "")


def enforce_complete_only_action(verdict: Verdict, blocking_reasons: Sequence[str] = ()) -> None:
    """Enforce that ResearchStatus.COMPLETE is the ONLY state that may leave
    an Action on ``verdict`` (mutates it in place).

    Must be called after ``verdict.research_status`` is set to its final
    value for this run. A no-op when that value is COMPLETE. Otherwise:

    * ``verdict.action`` is nulled -- whatever Action was selected before
      this was known (AVOID, BUY, WAIT_FOR_EVENT, ...) does not survive,
      whether the run is BLOCKED_PENDING_VERIFICATION or plain INCOMPLETE.
    * every free-text surface (headline, reasoning, thesis breakers, red
      flags, caveats) is regenerated/scrubbed of any leaked Action-enum
      token -- the Blind Judge bakes its selection into free text before
      the pipeline knows the final research_status.
    * a CONFIRMED K5-or-worse kill finding is never silently weakened or
      hidden by any of the above: it is appended as its own prominent,
      standalone ``CONFIRMED_DISQUALIFYING_EVIDENCE_PRESENT`` caveat,
      explicitly never converted into an Action.
    """
    if verdict.research_status is ResearchStatus.COMPLETE:
        return
    verdict.action = None
    verdict.headline = no_action_headline(verdict.research_status, blocking_reasons)
    verdict.reasoning = tuple(scrub_action_labels(r) for r in verdict.reasoning)
    verdict.thesis_breakers = tuple(scrub_action_labels(b) for b in verdict.thesis_breakers)
    verdict.critical_red_flags = tuple(scrub_action_labels(f) for f in verdict.critical_red_flags)
    verdict.caveats = tuple(scrub_action_labels(c) for c in verdict.caveats)
    if verdict.kill_gate.max_confirmed_level.level >= _CONFIRMED_DISQUALIFIER_LEVEL:
        verdict.caveats = (
            *verdict.caveats,
            "CONFIRMED_DISQUALIFYING_EVIDENCE_PRESENT: "
            f"{verdict.kill_gate.max_confirmed_level} is CONFIRMED by decision-grade "
            "evidence. Reported as a factual disqualifier signal regardless of whether an "
            "Action can be issued this run.",
        )


def sync_verdict_channel(evaluation: Evaluation, verdict: Verdict) -> None:
    """Propagate a post-hoc corrected ``Verdict`` back into the published
    VERDICT channel ``Evaluation`` (mutates it in place).

    ``enforce_complete_only_action`` corrects the ``Verdict`` object itself,
    but the Blind Judge already published a (now stale) snapshot of its
    action/headline/reasoning into ``bus.channels[Channel.VERDICT]`` before
    the pipeline learned the final ``research_status``. Any agent that reads
    that channel directly -- the Portfolio agent, which runs after the
    verdict is finalized, is the one in this codebase -- would otherwise see
    the pre-correction action leak straight through. Call this immediately
    after ``enforce_complete_only_action`` and before any later stage reads
    the VERDICT channel.
    """
    evaluation.summary = verdict.headline
    evaluation.points = verdict.reasoning
    evaluation.payload["action"] = str(verdict.action) if verdict.action is not None else None
    evaluation.payload["headline"] = verdict.headline
    evaluation.payload["reasoning"] = list(verdict.reasoning)
    evaluation.payload["thesis_breakers"] = list(verdict.thesis_breakers)
    evaluation.payload["critical_red_flags"] = list(verdict.critical_red_flags)
