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

from ..schemas.enums import Action

#: Every permitted final action label -- matched as a whole word so "NONE",
#: "BLOCKED_PENDING_VERIFICATION" and similar compliant text never trip this.
_ACTION_LABEL_RE = re.compile(r"\b(" + "|".join(re.escape(a.value) for a in Action) + r")\b")


def find_action_labels(text: str) -> list[str]:
    """Every Action-enum token literally present in ``text``, in order."""
    return _ACTION_LABEL_RE.findall(text or "")


def blocked_headline(blocking_reasons: Sequence[str]) -> str:
    """The ONLY headline a blocked / non-COMPLETE verdict may carry.

    Regenerated wholesale rather than edited, because a headline written
    before blocking was known cannot be trusted to have avoided asserting a
    judgement it is no longer entitled to make.
    """
    reasons = "; ".join(blocking_reasons)[:300] if blocking_reasons else "insufficient evidence"
    return (
        "RESEARCH_STATUS: BLOCKED_PENDING_VERIFICATION -- no action label issued "
        f"({reasons})"
    )


def scrub_action_labels(text: str) -> str:
    """Replace any leaked Action label with a neutral marker.

    Defense-in-depth for free text that is not simply regenerated outright
    (unlike the headline): reasoning, thesis breakers, red flags, caveats.
    """
    return _ACTION_LABEL_RE.sub("[ACTION LABEL WITHHELD -- RESEARCH INCOMPLETE]", text or "")
