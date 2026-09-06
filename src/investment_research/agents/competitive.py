"""Agent 9: Competitive Intelligence.

Separates TAM / SAM / SOM (requirement 9), because "large TAM" is the single
most common way a small company's addressable revenue gets overstated by two
orders of magnitude.

TAM  -- everyone with the condition or need, globally.
SAM  -- those the company could actually serve given approval, label, geography.
SOM  -- those it could realistically capture given competitors and distribution.

Where the evidence supports only one of the three, the other two are UNKNOWN.
A SAM computed by multiplying a company's TAM slide by a guessed share is a
model inference dressed as a fact, and this agent will not produce one.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import UNKNOWN, FactCategory, Materiality
from ..schemas.fact import Fact, UnresolvedQuestion
from .base import Agent
from .capital_structure import parse_number

log = logging.getLogger(__name__)

_COMPETITOR_RE = re.compile(r"(?i)\bcompetitor\b|\brival\b|\bcompeting\b")
_APPROVED_RE = re.compile(r"(?i)\bapproved\b|\bmarketing application\b|\bphase\s*3\b")

COMPARISON_DIMENSIONS = (
    "product",
    "clinical_efficacy",
    "safety",
    "pricing",
    "sales",
    "approval_status",
    "market_share",
    "cash",
    "valuation",
    "partnership",
    "distribution",
    "technical_superiority",
)

#: Requirement 9 asks for 3-10 peers.  Falling short is reported, not hidden.
MIN_COMPETITORS = 3


class CompetitiveAgent(Agent):
    agent_id = "competitive"
    purpose = "Compare against peers and separate TAM from SAM and SOM"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        competition_facts = list(
            data.facts_in(FactCategory.COMPETITION, FactCategory.COMMERCIAL)
        )
        market_facts = list(data.facts_in(FactCategory.MARKET_SIZE))

        competitors = self._extract_competitors(competition_facts)
        tam, sam, som, market_notes = self._market_sizes(market_facts, out)

        covered = self._dimension_coverage(competition_facts)
        missing_dimensions = [d for d in COMPARISON_DIMENSIONS if d not in covered]

        if len(competitors) < MIN_COMPETITORS:
            out.degraded = True
            out.errors.append(
                f"only {len(competitors)} competitor(s) identified; requirement is at least "
                f"{MIN_COMPETITORS} for a usable comparison"
            )
            out.unresolved.append(
                UnresolvedQuestion(
                    question=(
                        f"Only {len(competitors)} competitor(s) identified; at least "
                        f"{MIN_COMPETITORS} are required for a usable comparison"
                    ),
                    why_it_matters=(
                        "Competitive position cannot be assessed against an unexamined field, "
                        "and the most dangerous competitor is usually the one not on the list."
                    ),
                    blocking=False,
                    category=FactCategory.COMPETITION,
                    raised_by=self.agent_id,
                )
            )
        for dimension in missing_dimensions:
            out.unresolved.append(
                UnresolvedQuestion(
                    question=f"No comparative evidence on: {dimension}",
                    why_it_matters="An unexamined dimension is not a dimension the company wins.",
                    category=FactCategory.COMPETITION,
                    raised_by=self.agent_id,
                )
            )

        for competitor in competitors:
            if competitor.get("ahead"):
                out.risk_flags.append(
                    RiskFlag(
                        flag_id=f"comp_ahead_{competitor['fact_id'][-6:]}",
                        category=FactCategory.COMPETITION,
                        title="A competitor is further advanced",
                        detail=competitor["claim"],
                        severity=Materiality.HIGH,
                        fact_ids=(competitor["fact_id"],),
                        raised_by=self.agent_id,
                    )
                )

        payload: dict[str, Any] = {
            "competitors": competitors,
            "competitor_count": len(competitors),
            "dimensions_covered": sorted(covered),
            "dimensions_missing": missing_dimensions,
            "tam": tam,
            "sam": sam,
            "som": som,
            "market_size_notes": market_notes,
        }
        summary = (
            f"{len(competitors)} competitor(s) identified across "
            f"{len(covered)}/{len(COMPARISON_DIMENSIONS)} comparison dimensions. "
            f"TAM {_fmt(tam)}, SAM {_fmt(sam)}, SOM {_fmt(som)}."
        )
        out.evaluation = self.evaluation(
            Channel.COMPETITIVE, summary, (), payload, self.baseline_from(data)
        )
        return out

    @staticmethod
    def _extract_competitors(facts: list[Fact]) -> list[dict]:
        competitors: list[dict] = []
        for fact in facts:
            if not _COMPETITOR_RE.search(fact.claim):
                continue
            competitors.append(
                {
                    "claim": fact.claim,
                    "fact_id": fact.fact_id,
                    "source_tier": str(fact.source_tier),
                    "ahead": bool(_APPROVED_RE.search(fact.claim)),
                }
            )
        return competitors

    @staticmethod
    def _dimension_coverage(facts: list[Fact]) -> set[str]:
        keywords = {
            "product": ("product", "candidate", "device"),
            "clinical_efficacy": ("efficacy", "response rate", "phase 3", "endpoint"),
            "safety": ("safety", "adverse event", "tolerab"),
            "pricing": ("price", "pricing", "wac", "reimbursement"),
            "sales": ("sales", "revenue", "units"),
            "approval_status": ("approved", "marketing application", "submitted"),
            "market_share": ("market share", "share of"),
            "cash": ("cash", "balance sheet"),
            "valuation": ("market cap", "enterprise value", "valuation"),
            "partnership": ("partner", "collaboration", "license"),
            "distribution": ("distribution", "channel", "specialty pharmacy"),
            "technical_superiority": ("superior", "differentiat", "head-to-head"),
        }
        text = " ".join(f.claim.lower() for f in facts)
        return {name for name, terms in keywords.items() if any(t in text for t in terms)}

    def _market_sizes(
        self, facts: list[Fact], out: AgentOutput
    ) -> tuple[float | None, float | None, float | None, list[str]]:
        """Extract TAM/SAM/SOM without inventing the ones that are absent."""
        tam: float | None = None
        sam: float | None = None
        som: float | None = None
        notes: list[str] = []

        for fact in facts:
            text = fact.claim.lower()
            value = parse_number(fact.value) if fact.value != UNKNOWN else parse_number(fact.claim)
            if value is None:
                continue
            if "addressable market" in text or "tam" in text:
                if tam is None or fact.source_tier.rank < 3:
                    tam = value
                if fact.company_claim:
                    notes.append(
                        f"TAM figure is a company claim from {fact.source_title!r}; "
                        "prevalence x assumed price is an assumption, not a market."
                    )
                    out.risk_flags.append(
                        RiskFlag(
                            flag_id="comp_tam_company_claim",
                            category=FactCategory.MARKET_SIZE,
                            title="TAM rests on a company assumption",
                            detail=fact.claim,
                            severity=Materiality.MEDIUM,
                            fact_ids=(fact.fact_id,),
                            raised_by=self.agent_id,
                        )
                    )
            elif "prevalence" in text or "patients" in text:
                # Diagnosed-and-treated prevalence is the honest basis for SAM,
                # but converting it to revenue needs a price this agent does not
                # have. It is therefore reported in patients, not dollars.
                notes.append(
                    f"Serviceable population evidence: {fact.claim} "
                    f"(tier {fact.source_tier}, independent={not fact.company_claim})"
                )
                if sam is None:
                    sam = value

        if tam is not None and sam is not None:
            notes.append(
                "TAM is expressed in currency and SAM in patients; they are not comparable "
                "without a verified price assumption, which is not in evidence."
            )
        if som is None:
            notes.append(
                "SOM is UNKNOWN: obtainable share requires competitor share, distribution and "
                "payer evidence that is not present."
            )
            out.unresolved.append(
                UnresolvedQuestion(
                    question="What share of the serviceable market is realistically obtainable?",
                    why_it_matters=(
                        "Valuation multiples derived from TAM rather than SOM are the standard "
                        "way a small-cap thesis overstates its own ceiling."
                    ),
                    category=FactCategory.MARKET_SIZE,
                    raised_by=self.agent_id,
                )
            )
        return tam, sam, som, notes


def _fmt(value: float | None) -> str:
    return UNKNOWN if value is None else f"{value:,.0f}"
