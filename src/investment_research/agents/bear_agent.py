"""Agent 7: Bear Agent.

Distinct from the Kill Agent: the Kill Agent looks for facts that disqualify;
the Bear Agent constructs the most plausible *scenario* in which the investment
loses money even if nothing disqualifying is found.

It never sees the Bull case (isolation policy denies ``bull_case``), so it is
building an argument, not writing a rebuttal.
"""

from __future__ import annotations

import logging

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import UNKNOWN, EvidenceClass, FactCategory, Materiality
from ..schemas.fact import Fact
from .base import Agent

log = logging.getLogger(__name__)


class BearAgent(Agent):
    agent_id = "bear_agent"
    purpose = "Construct the most reasonable failure scenario from evidence"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        evidence = [f for f in data.facts if f.is_decision_grade or f.materiality.rank >= 4]

        mechanisms: list[dict] = []
        mechanisms += self._regulatory_mechanisms(data)
        mechanisms += self._capital_mechanisms(data)
        mechanisms += self._science_mechanisms(data)
        mechanisms += self._competitive_mechanisms(data)
        mechanisms += self._contradiction_mechanisms(data)

        if not mechanisms:
            mechanisms.append(
                {
                    "mechanism": "Insufficient evidence to construct a specific failure path",
                    "path": (
                        "The bear case cannot be specified because the evidence set is too "
                        "thin. This is a reason for low confidence, not for comfort."
                    ),
                    "fact_ids": [],
                    "severity": str(Materiality.HIGH),
                }
            )

        ordered = sorted(
            mechanisms, key=lambda m: -Materiality(m["severity"]).rank
        )
        primary = ordered[0]
        narrative = (
            f"Most likely failure path: {primary['mechanism']}. {primary['path']}"
        )
        points = tuple(f"{m['mechanism']}: {m['path']}" for m in ordered[:6])

        payload = {
            "mechanisms": ordered,
            "primary_failure_path": primary["mechanism"],
            "evidence_used": [f.fact_id for f in evidence],
            "evidence_count": len(evidence),
            "downside_drivers": [m["mechanism"] for m in ordered],
        }
        out.evaluation = self.evaluation(
            Channel.BEAR, narrative, points, payload, self.baseline_from(data)
        )
        out.metrics["mechanism_count"] = len(ordered)
        return out

    # -- mechanism builders -------------------------------------------------
    def _regulatory_mechanisms(self, data: AgentInput) -> list[dict]:
        channel = data.channel(Channel.REGULATORY)
        if not channel:
            return []
        position = channel.payload.get("endpoint_position", UNKNOWN)
        items: list[dict] = []
        if position == "REJECTED":
            items.append(
                {
                    "mechanism": "The regulator has already rejected the efficacy endpoint",
                    "path": (
                        "Even a statistically positive readout does not produce an approvable "
                        "package. The company must run a further adequate and well-controlled "
                        "trial, which costs years and at least one financing round. The equity "
                        "value implied by a near-term approval disappears without any new bad news."
                    ),
                    "fact_ids": [],
                    "severity": str(Materiality.CRITICAL),
                }
            )
        elif position == "UNKNOWN":
            items.append(
                {
                    "mechanism": "The regulator's position on the endpoint is unverified",
                    "path": (
                        "The market may be pricing an approval path that has never been "
                        "confirmed by the regulator. Disclosure of an adverse position would "
                        "re-rate the equity immediately."
                    ),
                    "fact_ids": [],
                    "severity": str(Materiality.HIGH),
                }
            )
        for entry in channel.payload.get("not_agreed", [])[:3]:
            items.append(
                {
                    "mechanism": f"Unresolved regulatory item: {entry}",
                    "path": "Adds cost, delay or an additional evidentiary requirement.",
                    "fact_ids": [],
                    "severity": str(Materiality.HIGH),
                }
            )
        return items

    def _capital_mechanisms(self, data: AgentInput) -> list[dict]:
        channel = data.channel(Channel.CAPITAL_STRUCTURE)
        if not channel:
            return []
        payload = channel.payload
        items: list[dict] = []
        runway = payload.get("runway_months")
        if runway is not None and runway < 18:
            items.append(
                {
                    "mechanism": f"Financing before the catalyst pays off (runway {runway} months)",
                    "path": (
                        "With an active ATM and an effective shelf, the company can sell into "
                        "any strength. A holder who is right about the science can still lose "
                        "money through the share count."
                    ),
                    "fact_ids": [],
                    "severity": str(
                        Materiality.CRITICAL if runway < 12 else Materiality.HIGH
                    ),
                }
            )
        overhang = payload.get("dilution_overhang_pct")
        if overhang and overhang > 25:
            items.append(
                {
                    "mechanism": f"Dilution overhang of about {overhang}% above basic shares",
                    "path": (
                        "Warrants, options and pre-funded warrants convert into supply on any "
                        "rally, capping the upside the headline market cap suggests."
                    ),
                    "fact_ids": [],
                    "severity": str(Materiality.HIGH),
                }
            )
        if payload.get("going_concern"):
            items.append(
                {
                    "mechanism": "Going concern doubt forces financing on the buyer's terms",
                    "path": "Structured or discounted financing typically follows, at the "
                    "expense of existing holders.",
                    "fact_ids": [],
                    "severity": str(Materiality.CRITICAL),
                }
            )
        return items

    def _science_mechanisms(self, data: AgentInput) -> list[dict]:
        channel = data.channel(Channel.SCIENCE)
        if not channel:
            return []
        payload = channel.payload
        items: list[dict] = []
        if payload.get("endpoint_type") == "SURROGATE_OR_BIOMARKER":
            items.append(
                {
                    "mechanism": "A surrogate endpoint may move without clinical benefit",
                    "path": (
                        "A positive biomarker result that does not translate into clinical "
                        "benefit produces a headline pop followed by a durable de-rating once "
                        "the regulatory implications are understood."
                    ),
                    "fact_ids": [],
                    "severity": str(Materiality.HIGH),
                }
            )
        if payload.get("surrogate_validation") == "NOT_SUPPORTED":
            items.append(
                {
                    "mechanism": "Independent literature contradicts the surrogate",
                    "path": "The mechanism does not connect to outcomes that regulators or "
                    "payers reward.",
                    "fact_ids": [],
                    "severity": str(Materiality.CRITICAL),
                }
            )
        enrollment = payload.get("enrollment")
        if isinstance(enrollment, int) and enrollment < 100:
            items.append(
                {
                    "mechanism": f"Underpowered trial (n={enrollment})",
                    "path": "A real but modest effect can miss significance, and a spurious "
                    "one can reach it.",
                    "fact_ids": [],
                    "severity": str(Materiality.MEDIUM),
                }
            )
        return items

    def _competitive_mechanisms(self, data: AgentInput) -> list[dict]:
        items: list[dict] = []
        for fact in data.facts_in(FactCategory.COMPETITION, FactCategory.COMMERCIAL):
            text = fact.claim.lower()
            if "competitor" in text and ("phase 3" in text or "approved" in text or "marketing application" in text):
                items.append(
                    {
                        "mechanism": "A competitor reaches the market first",
                        "path": (
                            "The first approved product sets the standard of care and the "
                            "reimbursement anchor, and a later entrant must prove superiority "
                            "rather than mere activity."
                        ),
                        "fact_ids": [fact.fact_id],
                        "severity": str(Materiality.HIGH),
                    }
                )
                break
        return items

    def _contradiction_mechanisms(self, data: AgentInput) -> list[dict]:
        items: list[dict] = []
        for contradiction in data.contradictions[:3]:
            items.append(
                {
                    "mechanism": f"Unresolved contradiction: {contradiction.kind}",
                    "path": contradiction.description,
                    "fact_ids": [contradiction.left_fact_id, contradiction.right_fact_id],
                    "severity": str(contradiction.severity),
                }
            )
        return items
