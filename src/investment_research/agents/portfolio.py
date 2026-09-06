"""Final Portfolio Agent (requirement 10).

The **only** component permitted to see the user's holdings, cost basis, tax
position and portfolio concentration -- and it runs *after* the blind verdict is
fixed, so those facts cannot influence what the evidence was judged to mean.

That ordering is the whole point. Knowing that a position is already held, at
what price, and how much the holder wants it to work, is exactly the information
that turns research into rationalisation. It is admitted only once the verdict
can no longer be changed by it.

This agent therefore never revises the verdict. It answers a different question:
*given this verdict, what should happen to the existing position?*
"""

from __future__ import annotations

import logging
from typing import Any

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import UNKNOWN, Action, KillLevel
from .base import Agent

log = logging.getLogger(__name__)

#: Verdicts under which an existing position should be reduced or exited
#: regardless of the entry price. A loss already taken is not a reason to keep
#: taking it.
_EXIT_ACTIONS = {Action.AVOID, Action.CONSIDER_SELL}


class PortfolioAgent(Agent):
    agent_id = "portfolio"
    purpose = "Apply the fixed verdict to the existing position"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        preferences: dict[str, Any] = dict(data.params.get("user_preferences") or {})
        verdict_channel = data.channel(Channel.VERDICT)

        if not verdict_channel:
            out.degraded = True
            out.errors.append("no verdict available; portfolio guidance withheld")
            return out
        if not preferences:
            out.evaluation = self.evaluation(
                "portfolio_guidance",
                "No portfolio information supplied; no position guidance produced.",
                (),
                {"has_position": False},
                self.baseline_from(data),
            )
            return out

        action = Action(verdict_channel.payload.get("action", Action.HOLD.value))
        max_kill = KillLevel(verdict_channel.payload.get("max_kill_level", "K0"))
        confidence = float(verdict_channel.payload.get("evidence_confidence", 0.0))

        shares = _to_float(preferences.get("shares"))
        cost_basis = _to_float(preferences.get("cost_basis"))
        price = _to_float(data.params.get("price"))
        portfolio_value = _to_float(preferences.get("portfolio_value"))

        position_value = shares * price if shares and price else None
        concentration = (
            round(100.0 * position_value / portfolio_value, 2)
            if position_value and portfolio_value
            else None
        )
        unrealised_pct = (
            round(100.0 * (price - cost_basis) / cost_basis, 1) if price and cost_basis else None
        )

        guidance: list[str] = []
        if action in _EXIT_ACTIONS or max_kill.level >= 4:
            guidance.append(
                "The verdict is a disqualification. The entry price is not a reason to hold: "
                "a position is justified by the evidence from here, not by what was paid."
            )
            if unrealised_pct is not None and unrealised_pct < 0:
                guidance.append(
                    f"The position is {abs(unrealised_pct)}% under water. That is a sunk cost "
                    "and is not an input to this decision."
                )
        elif action is Action.WAIT_FOR_EVENT:
            guidance.append(
                "The verdict is to wait. Adding before the blocking question is resolved buys "
                "more of an unresolved risk, not more of an opportunity."
            )
        elif action in (Action.BUY, Action.STRONG_BUY, Action.BUY_ON_PULLBACK):
            guidance.append("The verdict supports a position; size it against the downside case.")

        if confidence < 5.0:
            guidance.append(
                f"Evidence confidence is {confidence}/10. Position size should reflect how much "
                "is actually established, not how large the upside could be."
            )
        if concentration is not None:
            guidance.append(f"This position is about {concentration}% of the stated portfolio.")
            if concentration > 10 and max_kill.level >= 3:
                guidance.append(
                    f"A {concentration}% position carrying a {max_kill} red flag is a "
                    "concentration decision, not a research one, and should be reviewed as such."
                )
        if preferences.get("stop_loss"):
            guidance.append(f"Stated stop: {preferences['stop_loss']}.")
        if preferences.get("tax_notes"):
            guidance.append(f"Tax considerations noted by the holder: {preferences['tax_notes']}.")

        payload = {
            "has_position": bool(shares),
            "shares": shares,
            "cost_basis": cost_basis,
            "position_value": position_value,
            "portfolio_concentration_pct": concentration,
            "unrealised_pct": unrealised_pct,
            "verdict_action": str(action),
            "verdict_max_kill": str(max_kill),
            "guidance": guidance,
            "verdict_unchanged": True,
        }
        out.evaluation = self.evaluation(
            "portfolio_guidance",
            (
                f"Applying the fixed verdict ({action}) to the existing position. "
                "The verdict itself was reached without any knowledge of this position."
            ),
            tuple(guidance),
            payload,
            self.baseline_from(data),
        )
        return out


def _to_float(value: Any) -> float | None:
    if value in (None, "", UNKNOWN):
        return None
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None
