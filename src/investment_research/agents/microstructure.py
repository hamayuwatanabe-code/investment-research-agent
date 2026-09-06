"""Agent 12: Market Microstructure.

Positioning and flow data, reported **separately** from fundamentals
(requirement 12).  The isolation policy denies this agent every fundamental
channel, so it cannot blend a squeeze setup into a business assessment.

Most of these fields need a market-data subscription.  Where one is not
configured, the field is UNKNOWN and is reported as such -- a short interest of
UNKNOWN is not a short interest of zero.
"""

from __future__ import annotations

import logging
from typing import Any

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import UNKNOWN, FactCategory
from ..schemas.fact import UnresolvedQuestion
from .base import Agent
from .capital_structure import parse_number

log = logging.getLogger(__name__)

FIELDS = (
    "short_interest",
    "short_percent_float",
    "days_to_cover",
    "borrow_availability",
    "borrow_fee",
    "options_open_interest",
    "gamma_exposure",
    "average_volume",
    "relative_volume",
    "vwap",
    "atr",
    "rsi",
    "support",
    "resistance",
    "institutional_ownership",
    "insider_ownership",
    "etf_index_exposure",
)

_FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "short_interest": ("short interest",),
    "short_percent_float": ("percent of the public float", "% of float", "percent of float"),
    "days_to_cover": ("days to cover",),
    "borrow_availability": ("borrow availab", "hard to borrow"),
    "borrow_fee": ("borrow fee", "cost to borrow"),
    "options_open_interest": ("open interest",),
    "average_volume": ("average volume", "average daily volume"),
    "relative_volume": ("relative volume",),
    "vwap": ("vwap",),
    "atr": ("average true range", "atr"),
    "rsi": ("rsi", "relative strength index"),
    "institutional_ownership": ("institutional ownership", "institutions hold"),
    "insider_ownership": ("insider ownership", "insiders hold"),
    "etf_index_exposure": ("index inclusion", "etf holds", "russell"),
}


class MicrostructureAgent(Agent):
    agent_id = "microstructure"
    purpose = "Report positioning and flow data, kept apart from fundamentals"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        facts = list(data.facts_in(FactCategory.MICROSTRUCTURE)) + [
            f for f in data.facts if any(
                any(p in f.claim.lower() for p in patterns)
                for patterns in _FIELD_PATTERNS.values()
            )
        ]

        values: dict[str, Any] = {field: UNKNOWN for field in FIELDS}
        citations: dict[str, str] = {}
        for field, patterns in _FIELD_PATTERNS.items():
            for fact in facts:
                if any(p in fact.claim.lower() for p in patterns):
                    parsed = parse_number(fact.value) if fact.value != UNKNOWN else None
                    values[field] = parsed if parsed is not None else fact.claim
                    citations[field] = fact.fact_id
                    break

        unknown_fields = [f for f, v in values.items() if v == UNKNOWN]
        if unknown_fields:
            out.unresolved.append(
                UnresolvedQuestion(
                    question="Microstructure fields with no data: " + ", ".join(unknown_fields),
                    why_it_matters=(
                        "An unknown short interest or borrow cost is not a benign one; any "
                        "squeeze or flow argument built on these fields is unsupported."
                    ),
                    category=FactCategory.MICROSTRUCTURE,
                    raised_by=self.agent_id,
                )
            )
            out.degraded = True
            out.errors.append(
                f"{len(unknown_fields)}/{len(FIELDS)} microstructure fields have no data"
            )

        payload = {
            "values": values,
            "citations": citations,
            "unknown_fields": unknown_fields,
            "separation_note": (
                "Flow and positioning are reported separately from fundamentals and must not "
                "be combined into a single judgement about the business."
            ),
        }
        summary = (
            f"{len(FIELDS) - len(unknown_fields)}/{len(FIELDS)} microstructure fields "
            "populated from evidence."
        )
        out.evaluation = self.evaluation(
            Channel.MICROSTRUCTURE, summary, (), payload, self.baseline_from(data)
        )
        return out
