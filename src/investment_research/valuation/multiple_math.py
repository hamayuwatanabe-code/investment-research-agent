"""Reverse-engineering what each return multiple requires (requirement 10).

The point of this module is to convert "it could 10x" into a statement that can
be falsified: *at a 10x market capitalization, this company must be worth
$X, which requires $Y of revenue at peer multiples, which requires Z% share of a
market whose size is itself uncertain.*

Every figure is computed on the **fully diluted** share count where one exists.
Where the diluted count is incomplete, the result is explicitly labelled a floor
rather than a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas.enums import UNKNOWN
from ..schemas.evaluation import ValuationMath

#: The multiples requirement 10 asks to be reverse-engineered.
MULTIPLES: tuple[float, ...] = (2, 3, 5, 10, 20, 50, 100, 200)

#: Sector-typical EV/Sales used only to translate a required market cap into a
#: required revenue.  Declared as an assumption, never presented as a fact.
DEFAULT_EV_SALES_MULTIPLE = 5.0
#: Typical net margin used for the required-profit translation.
DEFAULT_NET_MARGIN = 0.20


@dataclass
class MultipleRequirement:
    multiple: float
    required_market_cap: float
    required_enterprise_value: float
    required_revenue: float | None
    required_net_income: float | None
    required_market_share_pct: float | None
    notes: list[str]


def compute_valuation(
    *,
    price: float | None,
    basic_shares: float | None,
    fully_diluted_shares: float | None,
    cash: float | None,
    debt: float | None,
    dilution_complete: bool,
    addressable_market: float | None = None,
    ev_sales_multiple: float = DEFAULT_EV_SALES_MULTIPLE,
    net_margin: float = DEFAULT_NET_MARGIN,
) -> ValuationMath:
    assumptions: list[str] = []
    unknowns: list[str] = []

    if price is None:
        unknowns.append("price")
    if basic_shares is None:
        unknowns.append("basic_shares")
    if fully_diluted_shares is None:
        unknowns.append("fully_diluted_shares")
    if cash is None:
        unknowns.append("cash")
    if debt is None:
        unknowns.append("debt")

    basic_cap = price * basic_shares if price is not None and basic_shares else None
    diluted_cap = (
        price * fully_diluted_shares if price is not None and fully_diluted_shares else None
    )
    reference_cap = diluted_cap if diluted_cap is not None else basic_cap

    if not dilution_complete and diluted_cap is not None:
        assumptions.append(
            "Fully diluted share count is incomplete; the diluted market cap below is a "
            "FLOOR, and the true figure is higher by an unknown amount."
        )

    enterprise_value = None
    cash_adjusted_ev = None
    if reference_cap is not None:
        enterprise_value = reference_cap + (debt or 0.0) - (cash or 0.0)
        cash_adjusted_ev = reference_cap - (cash or 0.0)
    if cash is None:
        assumptions.append("Cash unknown; enterprise value treats cash as zero, overstating EV.")
    if debt is None:
        assumptions.append("Debt unknown; enterprise value treats debt as zero, understating EV.")

    multiples: dict[str, dict[str, Any]] = {}
    if reference_cap is not None:
        for multiple in MULTIPLES:
            required_cap = reference_cap * multiple
            required_ev = required_cap - (cash or 0.0) + (debt or 0.0)
            required_revenue = required_ev / ev_sales_multiple if ev_sales_multiple else None
            required_profit = (
                required_revenue * net_margin if required_revenue is not None else None
            )
            required_share = None
            if required_revenue is not None and addressable_market:
                required_share = round(100.0 * required_revenue / addressable_market, 2)
            notes: list[str] = []
            if required_share is not None and required_share > 100:
                notes.append(
                    "Requires more revenue than the entire stated addressable market: this "
                    "multiple is not reachable on the evidence provided."
                )
            elif required_share is not None and required_share > 40:
                notes.append(
                    "Requires a dominant share of the entire addressable market."
                )
            multiples[f"{int(multiple)}x"] = {
                "required_market_cap": required_cap,
                "required_enterprise_value": required_ev,
                "required_revenue": required_revenue,
                "required_net_income": required_profit,
                "required_market_share_pct": required_share,
                "notes": notes,
            }
        assumptions.append(
            f"Required revenue is derived at EV/Sales = {ev_sales_multiple:g}x and required "
            f"profit at a {net_margin:.0%} net margin. Both are stated assumptions, not "
            "evidence about this company."
        )
    else:
        unknowns.append("market_cap (price or share count missing)")

    return ValuationMath(
        price=price,
        basic_shares=basic_shares,
        fully_diluted_shares=fully_diluted_shares,
        basic_market_cap=basic_cap,
        fully_diluted_market_cap=diluted_cap,
        cash=cash,
        debt=debt,
        enterprise_value=enterprise_value,
        cash_adjusted_ev=cash_adjusted_ev,
        multiples=multiples,
        assumptions=tuple(assumptions),
        unknowns=tuple(unknowns),
    )


def required_conditions_text(multiple_key: str, entry: dict[str, Any]) -> str:
    """One-line falsifiable statement of what a multiple requires."""
    parts = [f"{multiple_key} requires a market cap of {_money(entry['required_market_cap'])}"]
    if entry.get("required_revenue") is not None:
        parts.append(f"revenue of about {_money(entry['required_revenue'])}")
    if entry.get("required_net_income") is not None:
        parts.append(f"net income of about {_money(entry['required_net_income'])}")
    if entry.get("required_market_share_pct") is not None:
        parts.append(f"{entry['required_market_share_pct']}% of the stated addressable market")
    return "; ".join(parts) + ("." if not entry.get("notes") else ". " + " ".join(entry["notes"]))


def _money(value: float | None) -> str:
    if value is None:
        return UNKNOWN
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f}{suffix}"
    return f"${value:,.0f}"
