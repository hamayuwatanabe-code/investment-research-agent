"""Portfolio isolation and CLI mode tests (requirements 10 and 21)."""

from __future__ import annotations

from datetime import date

import pytest

from investment_research.agents.portfolio import PortfolioAgent
from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.isolation import POLICIES, Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.reporting.report import MODE_SECTIONS, SECTION_ORDER, render_report
from investment_research.schemas.agent_io import AgentInput, Evaluation
from investment_research.schemas.enums import Action, KillLevel

TODAY = date(2026, 9, 6)
POSITION = {
    "shares": 5000,
    "cost_basis": 4.20,
    "portfolio_value": 300000,
    "stop_loss": "no stop set",
    "tax_notes": "loss would offset gains this year",
}


def run(repo, fixture_dir, ticker="DEMOBIO", mode="standard", preferences=POSITION):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata(ticker)
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    return pipeline.run(
        ticker,
        metadata["company_name"],
        [collector.collect(ticker, metadata["company_name"])],
        mode=mode,
        price=metadata["price"],
        aliases=metadata.get("aliases", ()),
        user_preferences=preferences,
    )


# --- the portfolio step is the only one that sees a position ---------------
def test_only_the_portfolio_policy_sees_preferences():
    seeing = {a for a, p in POLICIES.items() if p.sees_user_preferences}
    assert seeing == {"portfolio"}


def test_portfolio_guidance_is_produced_when_a_position_is_supplied(repo, fixture_dir):
    guidance = run(repo, fixture_dir).portfolio_guidance
    assert guidance["has_position"]
    assert guidance["shares"] == 5000
    assert guidance["portfolio_concentration_pct"] == pytest.approx(5.67, abs=0.01)
    assert guidance["unrealised_pct"] == pytest.approx(-19.0, abs=0.1)


def test_no_guidance_without_a_position(repo, fixture_dir):
    assert run(repo, fixture_dir, preferences=None).portfolio_guidance == {}


def test_the_verdict_is_identical_with_and_without_a_position(repo, fixture_dir):
    """The whole point of the ordering: holdings cannot move the verdict."""
    with_position = run(repo, fixture_dir, preferences=POSITION)
    without_position = run(repo, fixture_dir, preferences=None)

    assert with_position.verdict.action == without_position.verdict.action
    assert with_position.verdict.evidence_confidence == (
        without_position.verdict.evidence_confidence
    )
    assert with_position.verdict.kill_gate.max_level == (
        without_position.verdict.kill_gate.max_level
    )
    assert with_position.scorecard.scores == without_position.scorecard.scores


def test_guidance_says_the_entry_price_is_not_a_reason_to_hold(repo, fixture_dir):
    text = " ".join(run(repo, fixture_dir).portfolio_guidance["guidance"]).lower()
    assert "sunk cost" in text
    assert "not a reason to hold" in text


def test_portfolio_agent_withholds_guidance_without_a_verdict():
    output = PortfolioAgent().run(
        AgentInput(agent_id="portfolio", run_id="r", ticker="X", company_name="X Co")
    )
    assert output.degraded
    assert "no verdict" in output.errors[0]


def test_portfolio_agent_never_revises_the_verdict():
    data = AgentInput(
        agent_id="portfolio",
        run_id="r",
        ticker="X",
        company_name="X Co",
        channels={
            Channel.VERDICT: Evaluation(
                author="blind_judge",
                channel=Channel.VERDICT,
                summary="",
                payload={
                    "action": str(Action.AVOID),
                    "max_kill_level": str(KillLevel.K5),
                    "evidence_confidence": 2.0,
                },
            )
        },
        params={"user_preferences": POSITION, "price": 3.40},
    )
    payload = PortfolioAgent().run(data).evaluation.payload
    assert payload["verdict_action"] == str(Action.AVOID)
    assert payload["verdict_unchanged"] is True


# --- modes ------------------------------------------------------------------
def test_kill_test_mode_does_not_build_a_bull_case(repo, fixture_dir):
    """Requirement 27: in a kill test there is nothing to weigh against."""
    result = run(repo, fixture_dir, mode="kill-test")
    assert Channel.BULL not in result.bus.channels
    assert Channel.BEAR in result.bus.channels


def test_standard_mode_builds_both(repo, fixture_dir):
    result = run(repo, fixture_dir, mode="standard")
    assert Channel.BULL in result.bus.channels
    assert Channel.BEAR in result.bus.channels


@pytest.mark.parametrize("mode", ["kill-test", "catalyst"])
def test_narrow_modes_always_keep_the_disconfirming_sections(mode):
    """Sections 1-4 carry the verdict, confidence, red flags and thesis breakers."""
    assert set(MODE_SECTIONS[mode]) >= {1, 2, 3, 4}


@pytest.mark.parametrize("mode", ["kill-test", "catalyst"])
def test_narrow_mode_reports_render_only_their_sections(repo, fixture_dir, mode):
    report = render_report(run(repo, fixture_dir, mode=mode))
    wanted = set(MODE_SECTIONS[mode])
    for number, title in enumerate(SECTION_ORDER, start=1):
        if number in wanted:
            assert title in report, f"{title} should be present in {mode} mode"
        else:
            assert title not in report, f"{title} should be absent in {mode} mode"


def test_full_report_renders_every_section(repo, fixture_dir):
    report = render_report(run(repo, fixture_dir))
    for title in SECTION_ORDER:
        assert title in report


# --- the clean control company ---------------------------------------------
def test_a_company_with_no_disqualifying_facts_is_not_avoided(repo, fixture_dir):
    """A falsification-first system that always says AVOID would be useless."""
    result = run(repo, fixture_dir, ticker="DEMOTECH", preferences=None)
    assert result.verdict.kill_gate.max_level.level <= 2
    assert not result.verdict.kill_gate.disqualifying
    assert result.verdict.action is not Action.AVOID


def test_endpoint_question_is_not_asked_of_a_non_regulated_business(repo, fixture_dir):
    """Asking it anyway produced a false regulatory risk for a technology company."""
    result = run(repo, fixture_dir, ticker="DEMOTECH", preferences=None)
    payload = result.bus.channels[Channel.REGULATORY].payload
    assert payload["endpoint_position"] == "NOT_APPLICABLE"


def test_endpoint_question_is_still_asked_when_evidence_is_empty(repo):
    """An empty evidence set must not be read as "not regulated"."""
    from investment_research.agents.regulatory import RegulatoryAgent

    output = RegulatoryAgent().run(
        AgentInput(agent_id="regulatory", run_id="r", ticker="X", company_name="X Co")
    )
    assert output.evaluation.payload["endpoint_position"] == "UNKNOWN"
    assert any(q.blocking for q in output.unresolved)
