"""Agent 4: Capital Structure.

Requirement 4-4: a headline market cap is not a valuation input.

This agent reconstructs the full claim on the equity -- options, RSUs, every
warrant class (pre-funded warrants especially, because they are shares in all
but name), preferred, converts -- and the financing that is *already authorized*
(ATM capacity, effective shelf).  It then computes runway from the actual burn.

Anything it cannot find is listed in ``unknown_fields``.  A diluted share count
assembled from partial data is worse than no diluted share count, so the agent
reports which components are missing and refuses to present the total as
complete.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import UNKNOWN, FactCategory, Materiality
from ..schemas.fact import Fact, UnresolvedQuestion
from .base import Agent

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)")


def parse_number(text: str) -> float | None:
    """Extract the first number from text, honouring million/billion suffixes."""
    if text is None:
        return None
    raw = str(text).replace("$", "").strip()
    match = _NUM_RE.search(raw)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    tail = raw[match.end() : match.end() + 12].lower()
    if "billion" in tail or re.match(r"\s*b\b", tail):
        value *= 1e9
    elif "million" in tail or re.match(r"\s*m\b", tail):
        value *= 1e6
    elif "thousand" in tail or re.match(r"\s*k\b", tail):
        value *= 1e3
    return value


@dataclass
class ShareComponent:
    key: str
    label: str
    patterns: tuple[str, ...]
    dilutive: bool = True


#: Each component is matched against fact claims.  Order matters only for
#: reporting; every component is searched independently.
COMPONENTS: tuple[ShareComponent, ...] = (
    ShareComponent(
        "basic_shares",
        "basic shares outstanding",
        ("basic shares", "shares outstanding", "common stock outstanding"),
        dilutive=False,
    ),
    # Matched most-specific first, and each source fact may satisfy only one
    # component: "Options and RSUs cover 7,100,000 shares" is one number, not two.
    ShareComponent(
        "prefunded_warrants", "pre-funded warrants", ("pre-funded warrant", "prefunded warrant")
    ),
    ShareComponent(
        "public_warrants",
        "public/private warrants",
        ("public warrant", "private warrant", "public and private warrant"),
    ),
    ShareComponent("options", "stock options", ("option",)),
    ShareComponent("rsus", "RSUs", ("rsu", "restricted stock unit")),
    ShareComponent(
        "preferred",
        "preferred stock (as-converted)",
        ("preferred stock", "series a preferred", "convertible preferred"),
    ),
    ShareComponent(
        "convertible_debt",
        "convertible debt (as-converted)",
        ("convertible note", "convertible debt", "convertible senior"),
    ),
)

_ATM_PATTERNS = ("at-the-market", "at the market", "atm program", "atm offering")
_SHELF_PATTERNS = ("shelf registration", "form s-3", "universal shelf")
_GOING_CONCERN_PATTERNS = ("going concern", "substantial doubt")
_REVERSE_SPLIT_PATTERNS = ("reverse split", "reverse stock split")
_LISTING_PATTERNS = (
    "minimum bid price",
    "non-compliance",
    "noncompliance",
    "delisting",
    "listing rule",
)


@dataclass
class CapitalPicture:
    values: dict[str, float | None] = field(default_factory=dict)
    fact_ids: dict[str, str] = field(default_factory=dict)
    unknown_fields: list[str] = field(default_factory=list)

    def get(self, key: str) -> float | None:
        return self.values.get(key)


class CapitalStructureAgent(Agent):
    agent_id = "capital_structure"
    purpose = "Reconstruct fully diluted share count, financing capacity and runway"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        facts = [
            f
            for f in data.facts
            if f.category
            in (
                FactCategory.CAPITAL_STRUCTURE,
                FactCategory.FINANCIAL,
                FactCategory.LIQUIDITY,
                FactCategory.GOVERNANCE,
                FactCategory.LISTING,
                FactCategory.INSIDER,
            )
        ]
        picture = self._extract_components(facts)
        cash, cash_fact = self._find_value(
            facts, ("cash, cash equivalents", "cash and cash equivalents", "cash and equivalents")
        )
        debt, _ = self._find_value(facts, ("total debt", "debt outstanding", "notes payable"))
        burn_six_months, _ = self._find_value(facts, ("net cash used in operating activities",))
        atm_capacity, atm_fact = self._find_value(facts, _ATM_PATTERNS)

        quarterly_burn = burn_six_months / 2.0 if burn_six_months else None
        runway_months = None
        if cash is not None and quarterly_burn:
            runway_months = round((cash / quarterly_burn) * 3.0, 1)

        going_concern = self._flag_present(facts, _GOING_CONCERN_PATTERNS)
        reverse_split = self._flag_present(facts, _REVERSE_SPLIT_PATTERNS)
        listing_issue = self._flag_present(facts, _LISTING_PATTERNS)
        shelf = self._flag_present(facts, _SHELF_PATTERNS)

        basic = picture.get("basic_shares")
        dilutive_total = 0.0
        dilutive_known = False
        missing: list[str] = []
        for component in COMPONENTS:
            if not component.dilutive:
                continue
            value = picture.get(component.key)
            if value is None:
                missing.append(component.label)
            else:
                dilutive_total += value
                dilutive_known = True

        fully_diluted = basic + dilutive_total if basic is not None and dilutive_known else None

        payload: dict[str, Any] = {
            "as_of": self._as_of(facts),
            "basic_shares": basic,
            "fully_diluted_shares": fully_diluted,
            "dilution_components": {c.key: picture.get(c.key) for c in COMPONENTS},
            "dilution_overhang_pct": (round(100.0 * dilutive_total / basic, 1) if basic else None),
            "cash": cash,
            "debt": debt,
            "quarterly_burn": quarterly_burn,
            "runway_months": runway_months,
            "atm_capacity": atm_capacity,
            "shelf_effective": shelf,
            "going_concern": going_concern,
            "reverse_split_history": reverse_split,
            "listing_compliance_issue": listing_issue,
            "unknown_fields": missing,
            "fact_ids": picture.fact_ids,
        }

        # --- risk flags (facts + severity, no verdict) ---------------------
        if going_concern:
            out.risk_flags.append(
                RiskFlag(
                    flag_id="cap_going_concern",
                    category=FactCategory.LIQUIDITY,
                    title="Going concern doubt disclosed",
                    detail="The filing discloses substantial doubt about the ability to continue as a going concern.",
                    severity=Materiality.CRITICAL,
                    fact_ids=tuple(
                        picture.fact_ids.get("going_concern", "")
                        for _ in [0]
                        if picture.fact_ids.get("going_concern")
                    ),
                    raised_by=self.agent_id,
                )
            )
        if runway_months is not None and runway_months < 12:
            out.risk_flags.append(
                RiskFlag(
                    flag_id="cap_short_runway",
                    category=FactCategory.LIQUIDITY,
                    title=f"Cash runway of approximately {runway_months} months",
                    detail=(
                        f"Cash {cash:,.0f} against an implied quarterly burn of "
                        f"{quarterly_burn:,.0f} gives roughly {runway_months} months. "
                        "Financing before the next major catalyst is likely."
                        if cash and quarterly_burn
                        else "Runway is under twelve months."
                    ),
                    severity=Materiality.CRITICAL if runway_months < 9 else Materiality.HIGH,
                    fact_ids=tuple(x for x in [cash_fact] if x),
                    raised_by=self.agent_id,
                )
            )
        if atm_capacity:
            out.risk_flags.append(
                RiskFlag(
                    flag_id="cap_atm_active",
                    category=FactCategory.CAPITAL_STRUCTURE,
                    title="Active at-the-market program",
                    detail=(
                        f"An ATM with roughly {atm_capacity:,.0f} of remaining capacity can be "
                        "drawn without further shareholder action, including into a price spike."
                    ),
                    severity=Materiality.HIGH,
                    fact_ids=tuple(x for x in [atm_fact] if x),
                    raised_by=self.agent_id,
                )
            )
        if listing_issue:
            out.risk_flags.append(
                RiskFlag(
                    flag_id="cap_listing",
                    category=FactCategory.LISTING,
                    title="Listing compliance issue disclosed",
                    detail="An exchange compliance deficiency is disclosed in the evidence set.",
                    severity=Materiality.HIGH,
                    raised_by=self.agent_id,
                )
            )
        if missing:
            out.unresolved.append(
                UnresolvedQuestion(
                    question=(
                        "Fully diluted share count is incomplete; missing components: "
                        + ", ".join(missing)
                    ),
                    why_it_matters=(
                        "Every per-share and market-cap figure below is understated by an "
                        "unknown amount until these are obtained."
                    ),
                    blocking=basic is None,
                    category=FactCategory.CAPITAL_STRUCTURE,
                    raised_by=self.agent_id,
                )
            )
        if basic is None:
            out.degraded = True
            out.errors.append("basic share count not found; market cap cannot be computed")

        summary = (
            f"Basic shares {_fmt(basic)}; fully diluted {_fmt(fully_diluted)} "
            f"({'incomplete: ' + ', '.join(missing) if missing else 'all components found'}). "
            f"Cash {_fmt(cash)}, implied quarterly burn {_fmt(quarterly_burn)}, "
            f"runway {runway_months if runway_months is not None else UNKNOWN} months."
        )
        out.evaluation = self.evaluation(
            Channel.CAPITAL_STRUCTURE, summary, (), payload, self.baseline_from(data)
        )
        out.metrics.update(
            {
                "runway_months": runway_months,
                "fully_diluted_shares": fully_diluted,
                "going_concern": going_concern,
            }
        )
        return out

    # -- extraction helpers ------------------------------------------------
    def _extract_components(self, facts: list[Fact]) -> CapitalPicture:
        """Extract each component, never letting one fact count twice.

        A single disclosure line often names two instrument types ("Options and
        RSUs outstanding cover 7,100,000 shares"). Attributing that number to
        both components silently inflates the diluted share count, so a fact
        that has been consumed is not offered to a later component -- the
        uncovered component is reported UNKNOWN instead.
        """
        picture = CapitalPicture()
        consumed: set[str] = set()
        for component in COMPONENTS:
            value, fact_id = self._find_value(facts, component.patterns, exclude=consumed)
            picture.values[component.key] = value
            if fact_id:
                picture.fact_ids[component.key] = fact_id
                consumed.add(fact_id)
            if value is None:
                picture.unknown_fields.append(component.key)
        return picture

    @staticmethod
    def _find_value(
        facts: list[Fact], patterns: tuple[str, ...], exclude: set[str] | None = None
    ) -> tuple[float | None, str]:
        """First numeric value whose claim matches any pattern.

        Prefers a fact's structured ``value`` over parsing prose, and prefers
        higher-tier sources.
        """
        exclude = exclude or set()
        candidates = [
            f
            for f in facts
            if f.fact_id not in exclude and any(p in f.claim.lower() for p in patterns)
        ]
        candidates.sort(key=lambda f: (f.source_tier.rank, -f.confidence))
        for fact in candidates:
            value = parse_number(fact.value) if fact.value not in (None, UNKNOWN) else None
            if value is None:
                value = parse_number(fact.claim)
            if value is not None:
                return value, fact.fact_id
        return None, ""

    @staticmethod
    def _flag_present(facts: list[Fact], patterns: tuple[str, ...]) -> bool:
        return any(any(p in f.claim.lower() for p in patterns) for f in facts)

    @staticmethod
    def _as_of(facts: list[Fact]) -> str:
        dates = [f.event_date for f in facts if f.event_date != UNKNOWN]
        return max(dates) if dates else UNKNOWN


def _fmt(value: float | None) -> str:
    return UNKNOWN if value is None else f"{value:,.0f}"
