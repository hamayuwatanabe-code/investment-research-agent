"""Five-scenario analysis (requirement 8).

Probabilities are given as **ranges**, never point estimates, and every scenario
carries a confidence label.  Requirement 8 is explicit that a weakly-grounded
probability must not be dressed up as precision, so ranges are wide and the
default label is LOW_CONFIDENCE unless the evidence genuinely supports better.
"""

from __future__ import annotations

from typing import Any

from ..schemas.enums import KillLevel, ScenarioName
from ..schemas.evaluation import KillGateResult, Scenario

#: Base probability ranges before kill-gate adjustment.  Deliberately coarse.
_BASE: dict[ScenarioName, tuple[float, float]] = {
    ScenarioName.DISASTER: (0.05, 0.15),
    ScenarioName.BEAR: (0.20, 0.35),
    ScenarioName.BASE: (0.25, 0.40),
    ScenarioName.BULL: (0.10, 0.25),
    ScenarioName.EXTREME_BULL: (0.02, 0.08),
}

#: Price multiples applied to the current price in each scenario.
_PRICE_MULTIPLE: dict[ScenarioName, tuple[float, float]] = {
    ScenarioName.DISASTER: (0.05, 0.25),
    ScenarioName.BEAR: (0.25, 0.60),
    ScenarioName.BASE: (0.60, 1.30),
    ScenarioName.BULL: (1.30, 3.00),
    ScenarioName.EXTREME_BULL: (3.00, 10.00),
}


def build_scenarios(
    *,
    price: float | None,
    fully_diluted_shares: float | None,
    basic_shares: float | None,
    kill_gate: KillGateResult,
    evidence_confidence: float,
    regulatory_position: str,
    runway_months: float | None,
    bear_mechanisms: list[dict[str, Any]],
    bull_points: list[dict[str, Any]],
) -> list[Scenario]:
    max_level = kill_gate.max_level
    shift = _downside_shift(max_level)

    scenarios: list[Scenario] = []
    for name in (
        ScenarioName.DISASTER,
        ScenarioName.BEAR,
        ScenarioName.BASE,
        ScenarioName.BULL,
        ScenarioName.EXTREME_BULL,
    ):
        low, high = _BASE[name]
        if name in (ScenarioName.DISASTER, ScenarioName.BEAR):
            low, high = min(0.95, low + shift), min(0.98, high + shift)
        elif name in (ScenarioName.BULL, ScenarioName.EXTREME_BULL):
            low, high = max(0.005, low - shift * 0.8), max(0.01, high - shift * 0.8)

        price_range: tuple[float, float] | None = (
            (price * _PRICE_MULTIPLE[name][0], price * _PRICE_MULTIPLE[name][1]) if price else None
        )
        price_high = price_range[1] if price_range else None
        shares = fully_diluted_shares or basic_shares
        market_cap = price_high * (basic_shares or 0) if price_high and basic_shares else None
        diluted_cap = price_high * shares if price_high and shares else None

        scenarios.append(
            Scenario(
                name=name,
                probability_range=(round(low, 3), round(high, 3)),
                price_range=price_range,
                market_cap=market_cap,
                fully_diluted_market_cap=diluted_cap,
                time_horizon=_horizon_for(name),
                required_conditions=_required(
                    name, regulatory_position, bull_points, runway_months
                ),
                failure_conditions=_failure(name, bear_mechanisms, max_level),
                confidence=_confidence_label(evidence_confidence),
                notes=(
                    "Probability ranges are coarse by design. Narrower ranges would imply "
                    "precision the evidence does not support."
                ),
            )
        )
    return scenarios


def _downside_shift(level: KillLevel) -> float:
    return {0: 0.0, 1: 0.02, 2: 0.05, 3: 0.12, 4: 0.20, 5: 0.30}[level.level]


def _confidence_label(evidence_confidence: float) -> str:
    if evidence_confidence >= 7.5:
        return "MEDIUM_CONFIDENCE"
    if evidence_confidence >= 5.0:
        return "LOW_CONFIDENCE"
    return "VERY_LOW_CONFIDENCE"


def _horizon_for(name: ScenarioName) -> str:
    return {
        ScenarioName.DISASTER: "0-12 months",
        ScenarioName.BEAR: "6-18 months",
        ScenarioName.BASE: "12-24 months",
        ScenarioName.BULL: "12-36 months",
        ScenarioName.EXTREME_BULL: "3-7 years",
    }[name]


def _required(
    name: ScenarioName,
    regulatory_position: str,
    bull_points: list[dict[str, Any]],
    runway_months: float | None,
) -> tuple[str, ...]:
    if name == ScenarioName.DISASTER:
        return (
            "No further financing available on any terms, or a programme-ending regulatory or "
            "clinical outcome",
        )
    if name == ScenarioName.BEAR:
        return ("Dilutive financing before the catalyst, or a disappointing readout",)
    if name == ScenarioName.BASE:
        return (
            "The programme continues, financing occurs on ordinary terms, and no new "
            "disqualifying fact emerges",
        )
    conditions: list[str] = []
    if regulatory_position != "AGREED":
        conditions.append(
            "The regulator accepts the primary endpoint as adequate to establish effectiveness "
            "-- which is NOT established in the current evidence"
        )
    conditions += [str(p.get("claim", ""))[:180] for p in bull_points[:3]]
    if runway_months is not None and runway_months < 18:
        conditions.append(
            f"Financing is completed without materially diluting holders (runway {runway_months} months)"
        )
    if name == ScenarioName.EXTREME_BULL:
        conditions.append(
            "Approval, launch, and capture of a substantial share of the obtainable market"
        )
    return tuple(c for c in conditions if c)


def _failure(
    name: ScenarioName, bear_mechanisms: list[dict[str, Any]], max_level: KillLevel
) -> tuple[str, ...]:
    mechanisms = tuple(str(m.get("mechanism", ""))[:180] for m in bear_mechanisms[:4])
    if name in (ScenarioName.BULL, ScenarioName.EXTREME_BULL):
        base = mechanisms or ("Any of the identified failure mechanisms occurs",)
        if max_level.level >= 3:
            return (
                f"Worst kill level is {max_level}: this scenario requires the identified "
                "disqualifying issue to be resolved first",
            ) + base
        return base
    return mechanisms or ("No specific failure mechanism identified in the evidence",)
