"""Valuation math and capital structure tests (requirements 4 and 10)."""

from __future__ import annotations

import pytest

from investment_research.agents.capital_structure import CapitalStructureAgent, parse_number
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import FactCategory
from investment_research.valuation.multiple_math import MULTIPLES, compute_valuation
from tests.conftest import make_fact


@pytest.mark.parametrize(
    "text,expected",
    [
        ("41,200,000", 41_200_000),
        ("$31,500,000", 31_500_000),
        ("$8 billion", 8e9),
        ("12.5 million", 12.5e6),
        ("$0", 0.0),
        ("no number here", None),
    ],
)
def test_number_parsing(text, expected):
    assert parse_number(text) == expected


# --- capital structure ------------------------------------------------------
def capital_facts():
    return [
        make_fact("Basic shares outstanding were 41,200,000", category=FactCategory.CAPITAL_STRUCTURE, value="41200000"),
        make_fact("Options and RSUs outstanding cover 7,100,000 shares", category=FactCategory.CAPITAL_STRUCTURE, value="7100000"),
        make_fact("Pre-funded warrants for 9,800,000 shares remain outstanding", category=FactCategory.CAPITAL_STRUCTURE, value="9800000"),
        make_fact("Public and private warrants for 5,400,000 shares", category=FactCategory.CAPITAL_STRUCTURE, value="5400000"),
        make_fact("Cash and cash equivalents were $31,500,000", category=FactCategory.FINANCIAL, value="31500000"),
        make_fact("Net cash used in operating activities was $32,800,000 for the six months", category=FactCategory.FINANCIAL, value="32800000"),
        make_fact("Total debt outstanding was $0", category=FactCategory.FINANCIAL, value="0"),
    ]


def run_capital(facts):
    agent = CapitalStructureAgent()
    data = AgentInput(
        agent_id="capital_structure",
        run_id="r",
        ticker="TEST",
        company_name="Test Co",
        facts=tuple(facts),
    )
    output = agent.run(data)
    return output, output.evaluation.payload


def test_one_disclosure_line_is_not_counted_twice():
    """'Options and RSUs cover 7,100,000 shares' is one number, not two."""
    _, payload = run_capital(capital_facts())
    components = payload["dilution_components"]
    assert components["options"] == 7_100_000
    assert components["rsus"] is None
    assert payload["fully_diluted_shares"] == 41_200_000 + 7_100_000 + 9_800_000 + 5_400_000


def test_missing_components_are_reported_not_assumed_zero():
    _, payload = run_capital(capital_facts())
    assert "preferred stock (as-converted)" in payload["unknown_fields"]
    assert "convertible debt (as-converted)" in payload["unknown_fields"]


def test_runway_computed_from_actual_burn():
    _, payload = run_capital(capital_facts())
    # 32.8m over six months -> 16.4m per quarter -> 31.5m / 16.4m * 3 months
    assert payload["quarterly_burn"] == pytest.approx(16_400_000)
    assert payload["runway_months"] == pytest.approx(5.8, abs=0.1)


def test_short_runway_raises_a_critical_flag():
    output, _ = run_capital(capital_facts())
    titles = [f.title for f in output.risk_flags]
    assert any("runway" in t.lower() for t in titles)


def test_missing_share_count_degrades_rather_than_guessing():
    output, payload = run_capital(
        [make_fact("Cash and cash equivalents were $31,500,000", category=FactCategory.FINANCIAL)]
    )
    assert payload["basic_shares"] is None
    assert payload["fully_diluted_shares"] is None
    assert output.degraded


def test_atm_capacity_raises_a_flag():
    facts = capital_facts() + [
        make_fact(
            "An at-the-market offering program with $50,000,000 of remaining capacity is in effect",
            category=FactCategory.CAPITAL_STRUCTURE,
            value="50000000",
        )
    ]
    output, payload = run_capital(facts)
    assert payload["atm_capacity"] == 50_000_000
    assert any("at-the-market" in f.title.lower() for f in output.risk_flags)


# --- valuation math ---------------------------------------------------------
def test_every_requested_multiple_is_reverse_engineered():
    math = compute_valuation(
        price=3.40,
        basic_shares=41_200_000,
        fully_diluted_shares=63_500_000,
        cash=31_500_000,
        debt=0.0,
        dilution_complete=True,
        addressable_market=8e9,
    )
    for multiple in MULTIPLES:
        assert f"{int(multiple)}x" in math.multiples
    entry = math.multiples["10x"]
    assert entry["required_market_cap"] == pytest.approx(3.40 * 63_500_000 * 10)
    assert entry["required_revenue"] is not None
    assert entry["required_market_share_pct"] is not None


def test_diluted_market_cap_uses_diluted_shares():
    math = compute_valuation(
        price=3.40,
        basic_shares=41_200_000,
        fully_diluted_shares=63_500_000,
        cash=31_500_000,
        debt=0.0,
        dilution_complete=True,
    )
    assert math.fully_diluted_market_cap > math.basic_market_cap
    assert math.enterprise_value == pytest.approx(math.fully_diluted_market_cap - 31_500_000)


def test_multiple_needing_more_than_the_whole_market_is_flagged_unreachable():
    """Requirement 10: 'small cap so 10x is easy' must be impossible to state."""
    math = compute_valuation(
        price=3.40,
        basic_shares=41_200_000,
        fully_diluted_shares=63_500_000,
        cash=31_500_000,
        debt=0.0,
        dilution_complete=True,
        addressable_market=8e9,
    )
    assert math.multiples["200x"]["required_market_share_pct"] > 100
    assert math.multiples["200x"]["notes"]


def test_incomplete_dilution_is_declared_a_floor():
    math = compute_valuation(
        price=3.40,
        basic_shares=41_200_000,
        fully_diluted_shares=63_500_000,
        cash=None,
        debt=None,
        dilution_complete=False,
    )
    assert any("FLOOR" in a for a in math.assumptions)
    assert "cash" in math.unknowns and "debt" in math.unknowns


def test_no_price_means_no_market_cap_rather_than_a_guess():
    math = compute_valuation(
        price=None,
        basic_shares=41_200_000,
        fully_diluted_shares=63_500_000,
        cash=1.0,
        debt=0.0,
        dilution_complete=True,
    )
    assert math.basic_market_cap is None
    assert math.multiples == {}
    assert "price" in math.unknowns


def test_multiple_assumptions_are_declared_not_hidden():
    math = compute_valuation(
        price=1.0, basic_shares=1e6, fully_diluted_shares=1e6, cash=0.0, debt=0.0,
        dilution_complete=True,
    )
    assert any("EV/Sales" in a for a in math.assumptions)
