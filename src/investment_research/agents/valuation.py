"""Agent 10: Valuation.

Sees facts and the capital-structure arithmetic, and nothing else -- no bull
case, no bear case, no kill findings (isolation policy).  It answers one
question: what would have to be true for each return multiple?

Requirement 10 forbids the conclusion "the market cap is small so 10x is easy".
This agent's output makes that conclusion impossible to state, because every
multiple comes attached to the revenue, profit and market share it demands.
"""

from __future__ import annotations

import logging

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import UNKNOWN, FactCategory
from ..schemas.fact import UnresolvedQuestion
from ..valuation.multiple_math import compute_valuation, required_conditions_text
from .base import Agent
from .capital_structure import parse_number

log = logging.getLogger(__name__)


class ValuationAgent(Agent):
    agent_id = "valuation"
    purpose = "Reverse-engineer what each return multiple requires"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        capital = data.channel(Channel.CAPITAL_STRUCTURE)
        payload = capital.payload if capital else {}

        price = _to_float(data.params.get("price"))
        if price is None:
            out.unresolved.append(
                UnresolvedQuestion(
                    question="Current share price is not in evidence",
                    why_it_matters="No market capitalization can be computed without it.",
                    blocking=True,
                    category=FactCategory.FINANCIAL,
                    raised_by=self.agent_id,
                )
            )

        addressable = _to_float(data.params.get("addressable_market"))
        if addressable is None:
            for fact in data.facts_in(FactCategory.MARKET_SIZE):
                if "addressable market" in fact.claim.lower():
                    addressable = parse_number(fact.value) or parse_number(fact.claim)
                    break

        math = compute_valuation(
            price=price,
            basic_shares=_to_float(payload.get("basic_shares")),
            fully_diluted_shares=_to_float(payload.get("fully_diluted_shares")),
            cash=_to_float(payload.get("cash")),
            debt=_to_float(payload.get("debt")),
            dilution_complete=not payload.get("unknown_fields"),
            addressable_market=addressable,
        )

        requirements = [
            required_conditions_text(key, entry) for key, entry in math.multiples.items()
        ]
        unreachable = [
            key
            for key, entry in math.multiples.items()
            if entry.get("required_market_share_pct") is not None
            and entry["required_market_share_pct"] > 100
        ]

        if math.fully_diluted_market_cap is None:
            out.degraded = True
            out.errors.append(
                "fully diluted market cap unavailable; valuation is incomplete"
            )

        result_payload = {
            "price": math.price,
            "basic_shares": math.basic_shares,
            "fully_diluted_shares": math.fully_diluted_shares,
            "basic_market_cap": math.basic_market_cap,
            "fully_diluted_market_cap": math.fully_diluted_market_cap,
            "cash": math.cash,
            "debt": math.debt,
            "enterprise_value": math.enterprise_value,
            "cash_adjusted_ev": math.cash_adjusted_ev,
            "multiples": math.multiples,
            "requirements": requirements,
            "unreachable_multiples": unreachable,
            "assumptions": list(math.assumptions),
            "unknowns": list(math.unknowns),
            "addressable_market_used": addressable,
        }
        summary = (
            f"Basic market cap {_money(math.basic_market_cap)}, fully diluted "
            f"{_money(math.fully_diluted_market_cap)}, EV {_money(math.enterprise_value)}. "
            + (
                f"Multiples requiring more than the entire stated market: {unreachable}."
                if unreachable
                else "No multiple in the computed range exceeds the stated addressable market."
            )
        )
        out.evaluation = self.evaluation(
            Channel.VALUATION, summary, tuple(requirements), result_payload,
            self.baseline_from(data),
        )
        out.metrics["fully_diluted_market_cap"] = math.fully_diluted_market_cap
        return out


def _to_float(value) -> float | None:
    if value is None or value == UNKNOWN:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _money(value: float | None) -> str:
    if value is None:
        return UNKNOWN
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f}{suffix}"
    return f"${value:,.0f}"
