"""Agent 8: Bull Agent.

Builds the case that the market may be undervaluing the asset, using only
verified evidence.  It cannot see the Bear case or the Kill findings (isolation
policy), so its argument is constructed rather than defensive.

Two constraints keep it honest:

* It may only build on facts that are decision-grade, and it records which facts
  each claim rests on.  A bull point with no fact behind it is dropped.
* Its own output passes through the report's citation validator, so an
  unsupported claim cannot reach the reader.
"""

from __future__ import annotations

import logging

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import FactCategory
from ..schemas.fact import UnresolvedQuestion
from .base import Agent

log = logging.getLogger(__name__)

#: Phrases that make a finding adverse, whatever its source quality.
_ADVERSE_MARKERS = (
    "did not correlate",
    "no correlation",
    "does not predict",
    "failed to predict",
    "did not meet",
    "failed to meet",
    "was not met",
    "no significant",
    "not statistically significant",
    "discontinued",
    "terminated",
    "safety signal",
    "did not support",
    "does not support",
)


class BullAgent(Agent):
    agent_id = "bull_agent"
    purpose = "Assess whether verified evidence supports undervaluation"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        decision_grade = list(data.decision_grade_facts())

        points: list[dict] = []
        points += self._asset_points(data)
        points += self._regulatory_points(data)
        points += self._structural_points(data)
        points += self._catalyst_points(data)

        supported = [p for p in points if p["fact_ids"]]
        dropped = [p for p in points if not p["fact_ids"]]

        if not supported:
            summary = (
                "No verified evidence supports an undervaluation case. This is a finding, "
                "not a failure of imagination: the bull case would have to rest on "
                "unverified claims."
            )
            out.unresolved.append(
                UnresolvedQuestion(
                    question="What verifiable evidence would support an undervaluation case?",
                    why_it_matters=(
                        "A bull case that cannot be built from verified facts is a narrative."
                    ),
                    category=FactCategory.OTHER,
                    raised_by=self.agent_id,
                )
            )
        else:
            summary = (
                f"{len(supported)} evidence-backed argument(s) for undervaluation, resting on "
                f"{len(decision_grade)} decision-grade fact(s)."
            )

        payload = {
            "points": supported,
            "dropped_unsupported_points": [p["claim"] for p in dropped],
            "decision_grade_fact_count": len(decision_grade),
            "evidence_used": sorted({fid for p in supported for fid in p["fact_ids"]}),
        }
        out.evaluation = self.evaluation(
            Channel.BULL,
            summary,
            tuple(p["claim"] for p in supported),
            payload,
            self.baseline_from(data),
        )
        out.metrics["supported_points"] = len(supported)
        out.metrics["dropped_points"] = len(dropped)
        return out

    # -- point builders ----------------------------------------------------
    def _asset_points(self, data: AgentInput) -> list[dict]:
        points: list[dict] = []
        science = data.channel(Channel.SCIENCE)
        if science:
            quality = science.payload.get("design_quality_score")
            if isinstance(quality, (int, float)) and quality >= 6.5:
                points.append(
                    {
                        "claim": (
                            f"Trial design quality scores {quality}/10 on checkable design "
                            "attributes (randomization, masking, endpoint type, sample size)."
                        ),
                        "fact_ids": [f.fact_id for f in data.facts_in(FactCategory.CLINICAL)[:3]],
                        "category": "asset_quality",
                    }
                )
            if science.payload.get("signals_present"):
                points.append(
                    {
                        "claim": (
                            "Technical differentiation is evidenced by: "
                            + ", ".join(science.payload["signals_present"])
                        ),
                        "fact_ids": [f.fact_id for f in data.facts_in(FactCategory.TECHNOLOGY)[:3]],
                        "category": "asset_quality",
                    }
                )
        for fact in data.facts_in(FactCategory.SCIENCE):
            text = fact.claim.lower()
            supportive = "grant" in text or "peer-review" in text or "peer reviewed" in text
            # A peer-reviewed paper that CONTRADICTS the mechanism is evidence
            # against the thesis. Citing it as "independent scientific support"
            # because it is independent and scientific is exactly the kind of
            # narrative laundering this system exists to prevent.
            if supportive and any(marker in text for marker in _ADVERSE_MARKERS):
                continue
            if fact.is_decision_grade and supportive:
                points.append(
                    {
                        "claim": f"Independent scientific support: {fact.claim}",
                        "fact_ids": [fact.fact_id],
                        "category": "asset_quality",
                    }
                )
        return points

    def _regulatory_points(self, data: AgentInput) -> list[dict]:
        channel = data.channel(Channel.REGULATORY)
        if not channel:
            return []
        points: list[dict] = []
        if channel.payload.get("endpoint_position") == "AGREED":
            points.append(
                {
                    "claim": (
                        "The regulator has agreed the primary endpoint, removing the largest "
                        "single source of binary risk in a clinical-stage programme."
                    ),
                    "fact_ids": list(channel.payload.get("primary_source_backed", []))[:3],
                    "category": "regulatory",
                }
            )
        # Designations are listed, but explicitly labelled as procedural, so the
        # bull case cannot lean on them as evidence about efficacy acceptance.
        agreed = [a for a in channel.payload.get("agreed", []) if "procedural designation" not in a]
        for entry in agreed[:3]:
            points.append(
                {
                    "claim": f"Favourable regulatory item: {entry}",
                    "fact_ids": list(channel.payload.get("primary_source_backed", []))[:1],
                    "category": "regulatory",
                }
            )
        return points

    def _structural_points(self, data: AgentInput) -> list[dict]:
        channel = data.channel(Channel.CAPITAL_STRUCTURE)
        if not channel:
            return []
        payload = channel.payload
        points: list[dict] = []
        runway = payload.get("runway_months")
        if isinstance(runway, (int, float)) and runway >= 24:
            points.append(
                {
                    "claim": (
                        f"Roughly {runway} months of runway funds the programme past its next "
                        "catalyst without a forced financing."
                    ),
                    "fact_ids": [f.fact_id for f in data.facts_in(FactCategory.FINANCIAL)[:2]],
                    "category": "structure",
                }
            )
        if payload.get("debt") == 0:
            points.append(
                {
                    "claim": "No debt outstanding, so equity holders are not structurally subordinated.",
                    "fact_ids": [
                        f.fact_id
                        for f in data.facts_in(FactCategory.FINANCIAL)
                        if "debt" in f.claim.lower()
                    ][:1],
                    "category": "structure",
                }
            )
        return points

    def _catalyst_points(self, data: AgentInput) -> list[dict]:
        points: list[dict] = []
        for fact in data.facts_in(FactCategory.CATALYST):
            if fact.is_decision_grade or fact.materiality.rank >= 3:
                points.append(
                    {
                        "claim": f"Dated near-term catalyst: {fact.claim}",
                        "fact_ids": [fact.fact_id],
                        "category": "catalyst",
                    }
                )
        return points[:3]
