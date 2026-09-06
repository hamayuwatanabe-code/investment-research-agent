"""Independent score dimensions (requirement 7).

There is no aggregate score, and adding one would defeat the system.  A single
number lets a strong Explosive Potential outvote a K5 Regulatory Kill, which is
precisely the arithmetic that produces the failure this system was built after.

Instead: 22 independent dimensions, each with its own confidence, plus a hard
cap applied by the Kill Gate to every investment-quality dimension.
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas.evaluation import (
    INVESTMENT_QUALITY_DIMENSIONS,
    KillGateResult,
    ScoreCard,
)
from .kill_gate import quality_cap

log = logging.getLogger(__name__)

UNSCORED = None


def _clamp(value: float) -> float:
    return round(max(0.0, min(10.0, value)), 1)


def build_scorecard(
    *,
    ticker: str,
    run_id: str,
    channels: dict[str, Any],
    kill_gate: KillGateResult,
    evidence_confidence: float,
    contradiction_count: int,
    catalyst_count: int,
    near_term_catalyst: bool,
) -> ScoreCard:
    """Compute every dimension from evidence-derived inputs, then apply caps."""
    card = ScoreCard(ticker=ticker, run_id=run_id, evidence_confidence=evidence_confidence)
    scores: dict[str, float] = {}
    rationale: dict[str, str] = {}

    regulatory = channels.get("regulatory_findings")
    capital = channels.get("capital_structure")
    science = channels.get("science_findings")
    competitive = channels.get("competitive_findings")
    valuation = channels.get("valuation_math")
    micro = channels.get("microstructure")

    reg_payload = regulatory.payload if regulatory else {}
    cap_payload = capital.payload if capital else {}
    sci_payload = science.payload if science else {}
    comp_payload = competitive.payload if competitive else {}
    val_payload = valuation.payload if valuation else {}
    micro_payload = micro.payload if micro else {}

    # --- evidence & governance -------------------------------------------
    scores["evidence_quality"] = evidence_confidence
    rationale["evidence_quality"] = "Mirrors the evidence confidence breakdown."

    scores["management_quality"] = 5.0
    rationale["management_quality"] = (
        "Neutral by default: management quality was not independently assessed."
    )
    if reg_payload.get("company_framing") and reg_payload.get("endpoint_position") == "REJECTED":
        scores["management_quality"] = 2.0
        rationale["management_quality"] = (
            "Company framed a regulatory interaction favourably while a primary source records "
            "an adverse position."
        )

    # --- asset ------------------------------------------------------------
    design = sci_payload.get("design_quality_score")
    scores["science_technology_quality"] = _clamp(design if design is not None else 5.0)
    rationale["science_technology_quality"] = (
        "From checkable design/technology attributes." if design is not None else "Not assessed."
    )
    scores["clinical_quality"] = _clamp(design if design is not None else 5.0)
    rationale["clinical_quality"] = rationale["science_technology_quality"]

    endpoint_position = reg_payload.get("endpoint_position", "UNKNOWN")
    scores["regulatory_quality"] = {
        "AGREED": 8.5,
        "NOT_APPLICABLE": 6.0,  # no approval gate is a genuine absence of this risk
        "UNKNOWN": 4.0,
        "REJECTED": 1.0,
    }.get(endpoint_position, 4.0)
    rationale["regulatory_quality"] = (
        f"Regulator position on the primary endpoint: {endpoint_position}."
    )

    # --- financial --------------------------------------------------------
    runway = cap_payload.get("runway_months")
    if runway is None:
        scores["financial_strength"] = 3.0
        rationale["financial_strength"] = "Runway unknown; scored low because it is unverified."
    else:
        scores["financial_strength"] = _clamp(min(10.0, runway / 3.0))
        rationale["financial_strength"] = f"About {runway} months of runway."
    if cap_payload.get("going_concern"):
        scores["financial_strength"] = _clamp(min(scores["financial_strength"], 1.5))
        rationale["financial_strength"] += " Going concern doubt disclosed."

    overhang = cap_payload.get("dilution_overhang_pct")
    if overhang is None:
        scores["capital_structure_quality"] = 3.0
        rationale["capital_structure_quality"] = "Dilution overhang could not be computed."
    else:
        scores["capital_structure_quality"] = _clamp(10.0 - overhang / 8.0)
        rationale["capital_structure_quality"] = f"Dilution overhang about {overhang}% over basic."
    if cap_payload.get("atm_capacity"):
        scores["capital_structure_quality"] = _clamp(scores["capital_structure_quality"] - 1.5)
        rationale["capital_structure_quality"] += " Active ATM can add supply at any time."

    # --- market -----------------------------------------------------------
    competitors = comp_payload.get("competitor_count", 0)
    ahead = sum(1 for c in comp_payload.get("competitors", []) if c.get("ahead"))
    scores["competitive_moat"] = _clamp(5.0 - 2.0 * ahead + (0.5 if competitors >= 3 else -1.0))
    rationale["competitive_moat"] = f"{competitors} peer(s) identified, {ahead} further advanced."

    tam = comp_payload.get("tam")
    scores["tam_quality"] = (
        3.0 if tam is None else (4.0 if _tam_is_company_claim(comp_payload) else 7.0)
    )
    rationale["tam_quality"] = (
        "TAM absent." if tam is None else "TAM present; company-sourced TAM is discounted."
    )
    scores["sam_som_quality"] = 2.0 if comp_payload.get("som") is None else 6.0
    rationale["sam_som_quality"] = (
        "SOM is UNKNOWN: obtainable share is not established."
        if comp_payload.get("som") is None
        else "SOM estimated from evidence."
    )

    scores["growth_quality"] = 3.0
    rationale["growth_quality"] = "No verified revenue growth series in evidence."

    # --- valuation --------------------------------------------------------
    unreachable = val_payload.get("unreachable_multiples", [])
    if val_payload.get("fully_diluted_market_cap") is None:
        scores["valuation"] = 2.0
        rationale["valuation"] = "Fully diluted market cap could not be computed."
    else:
        scores["valuation"] = _clamp(6.0 - 1.0 * len(unreachable))
        rationale["valuation"] = (
            f"{len(unreachable)} of the requested multiples require more revenue than the "
            "entire stated addressable market."
        )

    # --- catalysts --------------------------------------------------------
    scores["catalyst_strength"] = _clamp(min(10.0, 3.0 + 1.5 * catalyst_count))
    rationale["catalyst_strength"] = f"{catalyst_count} dated forward catalyst(s)."
    scores["catalyst_timing"] = 7.0 if near_term_catalyst else 4.0
    rationale["catalyst_timing"] = (
        "A catalyst falls inside three months."
        if near_term_catalyst
        else "Nothing within three months."
    )
    scores["market_pricing"] = 5.0
    rationale["market_pricing"] = (
        "UNKNOWN: whether the market has already discounted the catalyst is not in evidence."
    )

    unknown_micro = len(micro_payload.get("unknown_fields", []))
    scores["flow_microstructure"] = _clamp(5.0 - 0.3 * unknown_micro)
    rationale["flow_microstructure"] = f"{unknown_micro} microstructure field(s) unknown."

    # --- outcome dimensions ----------------------------------------------
    # Explosive potential is deliberately NOT capped by the kill gate: a
    # disqualified company can still be capable of a violent move, and hiding
    # that would be its own distortion. What is capped is investment quality.
    explosive = 5.0
    if (
        val_payload.get("fully_diluted_market_cap")
        and val_payload["fully_diluted_market_cap"] < 5e8
    ):
        explosive += 2.0
    if near_term_catalyst:
        explosive += 1.5
    if micro_payload.get("values", {}).get("short_percent_float") not in (None, "UNKNOWN"):
        explosive += 1.0
    scores["explosive_potential"] = _clamp(explosive)
    rationale["explosive_potential"] = (
        "Capacity for a violent move given size, catalyst proximity and positioning. This is "
        "NOT a statement that the move will be upward, and it is not capped by the kill gate."
    )

    scores["long_term_multibagger"] = _clamp(
        (scores["competitive_moat"] + scores["sam_som_quality"] + scores["regulatory_quality"])
        / 3.0
    )
    rationale["long_term_multibagger"] = "Mean of moat, obtainable market and regulatory position."

    max_level = kill_gate.max_level
    downside = 2.0 + 1.5 * max_level.level
    scores["downside_risk"] = _clamp(downside)
    rationale["downside_risk"] = (
        f"Higher is worse. Worst kill level {max_level}; "
        f"{len(kill_gate.major)} category/categories at K3 or above."
    )

    scores["risk_reward"] = _clamp(
        (scores["explosive_potential"] + (10.0 - scores["downside_risk"])) / 2.0
    )
    rationale["risk_reward"] = "Upside capacity against downside severity."

    scores["immediate_buy"] = _clamp(
        (scores["regulatory_quality"] + scores["financial_strength"] + scores["catalyst_timing"])
        / 3.0
    )
    rationale["immediate_buy"] = "Regulatory position, balance sheet and catalyst timing."

    scores["time_efficiency"] = 6.0 if near_term_catalyst else 3.5
    rationale["time_efficiency"] = "How soon the thesis is testable."

    # --- kill gate caps ---------------------------------------------------
    cap = quality_cap(max_level)
    capped: list[str] = []
    for dimension in INVESTMENT_QUALITY_DIMENSIONS:
        if dimension in scores and scores[dimension] > cap:
            scores[dimension] = cap
            capped.append(dimension)
            rationale[dimension] += f" Capped at {cap} by the kill gate (worst level {max_level})."

    card.scores = {k: _clamp(v) for k, v in scores.items()}
    card.rationale = rationale
    card.capped_by_kill_gate = tuple(capped)
    card.per_score_confidence = {
        dimension: _dimension_confidence(dimension, evidence_confidence, channels)
        for dimension in card.scores
    }

    missing = card.missing_dimensions()
    if missing:  # pragma: no cover - guards a future edit that drops a dimension
        log.warning("scorecard is missing dimensions: %s", missing)
    return card


def _tam_is_company_claim(payload: dict[str, Any]) -> bool:
    return any("company claim" in note.lower() for note in payload.get("market_size_notes", []))


def _dimension_confidence(
    dimension: str, evidence_confidence: float, channels: dict[str, Any]
) -> float:
    """Per-dimension confidence: a score computed from a missing channel is not
    as trustworthy as one computed from a populated channel."""
    required = {
        "regulatory_quality": "regulatory_findings",
        "clinical_quality": "science_findings",
        "science_technology_quality": "science_findings",
        "financial_strength": "capital_structure",
        "capital_structure_quality": "capital_structure",
        "valuation": "valuation_math",
        "competitive_moat": "competitive_findings",
        "tam_quality": "competitive_findings",
        "sam_som_quality": "competitive_findings",
        "flow_microstructure": "microstructure",
    }.get(dimension)
    if required and required not in channels:
        return round(min(evidence_confidence, 2.0), 1)
    return round(evidence_confidence, 1)
