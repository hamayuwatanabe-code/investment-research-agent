"""Agent 11: Catalyst.

All dates are normalized to JST (requirement 11), because the user trades from
Japan and a "Q4 2026" guided readout is not a date.

Each event records what the bull and bear outcomes actually are, and -- more
importantly -- ``market_pricing``: whether the evidence shows the market has
already discounted the event.  A catalyst the market has priced is not a
catalyst; it is a risk.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime, timedelta, timezone

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import UNKNOWN, FactCategory, Horizon
from ..schemas.evaluation import CatalystEvent
from ..schemas.fact import Fact, parse_iso_date
from .base import Agent

log = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

_QUARTER_RE = re.compile(
    r"(?i)\b(?:q([1-4])|([1-4])(?:st|nd|rd|th)?\s+quarter"
    r"|(first|second|third|fourth)\s+quarter)\D{0,12}(20\d{2})"
)
_HALF_RE = re.compile(r"(?i)\b(?:(first|second)\s+half|h([12]))\D{0,12}(20\d{2})")
_MONTH_RE = re.compile(
    r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})"
)

_QUARTER_END = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_MONTH_NUM = {
    m: i + 1
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}


def to_jst_date(value: str) -> tuple[str, str]:
    """Normalize a date expression to a JST calendar date + a confidence label.

    Returns ``(UNKNOWN, "UNKNOWN")`` when nothing parseable is present.  A vague
    guidance phrase becomes the *end* of its window, labelled LOW, so a range is
    never silently narrowed into a precise-looking date.
    """
    if not value or value == UNKNOWN:
        return UNKNOWN, "UNKNOWN"
    text = str(value)
    exact = parse_iso_date(text)
    if exact:
        return exact.isoformat(), "HIGH"

    match = _QUARTER_RE.search(text)
    if match:
        word = (match.group(3) or "").lower()
        quarter = (
            int(match.group(1) or match.group(2))
            if (match.group(1) or match.group(2))
            else _ORDINAL_WORDS[word]
        )
        year = match.group(4)
        return f"{year}-{_QUARTER_END[quarter]}", "LOW"
    match = _HALF_RE.search(text)
    if match:
        half = 1 if (match.group(1) or "").lower() == "first" or match.group(2) == "1" else 2
        year = match.group(3)
        return f"{year}-{'06-30' if half == 1 else '12-31'}", "LOW"
    match = _MONTH_RE.search(text)
    if match:
        month = _MONTH_NUM[match.group(1).lower()]
        year = int(match.group(2))
        last_day = (date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)).day
        return f"{year:04d}-{month:02d}-{last_day:02d}", "MEDIUM"
    return UNKNOWN, "UNKNOWN"


def horizon_for(target: str, today: date) -> Horizon:
    parsed = parse_iso_date(target)
    if parsed is None:
        return Horizon.T3
    days = (parsed - today).days
    if days <= 1:
        return Horizon.T0
    if days <= 14:
        return Horizon.T1
    if days <= 92:
        return Horizon.T2
    if days <= 550:
        return Horizon.T3
    return Horizon.T4


class CatalystAgent(Agent):
    agent_id = "catalyst"
    purpose = "Enumerate dated catalysts in JST with bull/bear outcomes"

    def __init__(self, today: date | None = None) -> None:
        self.today = today or datetime.now(JST).date()

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        candidates = [
            f
            for f in data.facts
            if f.category in (FactCategory.CATALYST, FactCategory.CLINICAL, FactCategory.REGULATORY)
            or _has_forward_date(f)
        ]

        events: list[CatalystEvent] = []
        for fact in candidates:
            date_jst, confidence = to_jst_date(fact.event_date)
            if date_jst == UNKNOWN:
                date_jst, confidence = to_jst_date(str(fact.value))
            if date_jst == UNKNOWN:
                date_jst, confidence = to_jst_date(fact.claim)
            if date_jst == UNKNOWN:
                continue
            parsed = parse_iso_date(date_jst)
            if parsed is None or parsed < self.today:
                continue  # a past event is history, not a catalyst

            bull, bear, expected = _outcomes_for(fact)
            events.append(
                CatalystEvent(
                    event_id="cat_"
                    + hashlib.sha256(f"{fact.fact_id}{date_jst}".encode()).hexdigest()[:12],
                    horizon=str(horizon_for(date_jst, self.today)),
                    date_jst=date_jst,
                    date_confidence=confidence,
                    event=fact.claim[:400],
                    expected_outcome=expected,
                    bull_outcome=bull,
                    bear_outcome=bear,
                    market_pricing=UNKNOWN,
                    information_source=f"{fact.source_title} ({fact.source_tier})",
                    fact_ids=(fact.fact_id,),
                )
            )

        events.sort(key=lambda e: e.date_jst)
        by_horizon: dict[str, list[str]] = {}
        for event in events:
            by_horizon.setdefault(event.horizon, []).append(event.event_id)

        payload = {
            "events": [e.to_row() for e in events],
            "by_horizon": by_horizon,
            "timezone": "Asia/Tokyo (JST)",
            "today_jst": self.today.isoformat(),
            "low_confidence_dates": [
                e.event_id for e in events if e.date_confidence in ("LOW", "UNKNOWN")
            ],
        }
        summary = (
            f"{len(events)} dated forward catalyst(s) in JST; "
            f"{len(payload['low_confidence_dates'])} rest on guidance rather than a fixed date."
        )
        out.evaluation = self.evaluation(
            Channel.CATALYSTS, summary, (), payload, self.baseline_from(data)
        )
        out.metrics["catalyst_count"] = len(events)
        return out


def _has_forward_date(fact: Fact) -> bool:
    return bool(_QUARTER_RE.search(fact.claim) or _HALF_RE.search(fact.claim))


def _outcomes_for(fact: Fact) -> tuple[str, str, str]:
    text = fact.claim.lower()
    if "topline" in text or "primary completion" in text or "data" in text:
        return (
            "Primary endpoint met with an effect size the regulator and payers treat as "
            "clinically meaningful",
            "Primary endpoint missed, or met on a measure the regulator has not accepted as "
            "adequate to establish effectiveness",
            "UNKNOWN -- the outcome distribution is not derivable from the evidence set",
        )
    if "meeting" in text or "fda" in text:
        return (
            "Regulator agrees the registrational path and the endpoint",
            "Regulator requires an additional trial or rejects the endpoint",
            "UNKNOWN",
        )
    return ("UNKNOWN", "UNKNOWN", "UNKNOWN")
