"""Agent isolation and blind-judge leakage tests.

Requirement 17 names two of these explicitly:

* the Bull Agent's output must not reach the Bear Agent
* the user's holdings must not reach the Blind Judge

Both are tested here against the real isolation guard, in both the direct case
(the channel is present) and the indirect case (the prose was copied into
something that *is* shared).
"""

from __future__ import annotations

import pytest

from investment_research.orchestrator.isolation import (
    POLICIES,
    Channel,
    EvidenceBus,
    IsolationGuard,
    LeakageError,
    derive_fingerprints,
    identity_markers,
    policy_for,
)
from investment_research.schemas.agent_io import Evaluation, RiskFlag
from investment_research.schemas.enums import EvidenceClass, FactCategory, Materiality
from tests.conftest import make_fact

TICKER = "DEMOBIO"
NAME = "Demobio Therapeutics"
ALIASES = ("Demobio",)

BULL_PROSE = (
    "The market misprices the optionality embedded in the second indication because "
    "sell-side coverage lapsed after the reorganisation."
)
BEAR_PROSE = (
    "Financing will occur before the readout and the pre-funded warrant overhang converts "
    "into supply on any rally."
)
KILL_PROSE = "The regulator refused the endpoint in written responses after the meeting."


def build_bus() -> EvidenceBus:
    bus = EvidenceBus()
    bus.add_facts(
        [
            make_fact("Cash was $31.5 million at 2026-06-30", ticker=TICKER),
            make_fact(
                "Demobio Therapeutics filed a Form 10-Q for the quarter ended 2026-06-30",
                ticker=TICKER,
                category=FactCategory.FINANCIAL,
            ),
        ]
    )
    bus.risk_flags.append(
        RiskFlag(
            flag_id="f1",
            category=FactCategory.LIQUIDITY,
            title="Cash runway under twelve months",
            detail="Demobio Therapeutics discloses going concern doubt.",
            severity=Materiality.CRITICAL,
        )
    )
    baseline = [f.claim for f in bus.facts]
    bus.publish(
        Evaluation(
            author="bull_agent",
            channel=Channel.BULL,
            summary=BULL_PROSE,
            fingerprint_tokens=derive_fingerprints([BULL_PROSE], baseline),
        )
    )
    bus.publish(
        Evaluation(
            author="bear_agent",
            channel=Channel.BEAR,
            summary=BEAR_PROSE,
            fingerprint_tokens=derive_fingerprints([BEAR_PROSE], baseline),
        )
    )
    bus.publish(
        Evaluation(
            author="kill_agent",
            channel=Channel.KILL,
            summary=KILL_PROSE,
            payload={"kill_gate": [], "unsearched_categories": []},
            fingerprint_tokens=derive_fingerprints([KILL_PROSE], baseline),
        )
    )
    return bus


def project(agent_id: str, bus: EvidenceBus, **kwargs):
    guard = IsolationGuard(bus)
    return guard.project(
        agent_id, ticker=TICKER, company_name=NAME, aliases=ALIASES, **kwargs
    )


# --- policy sanity ----------------------------------------------------------
def test_every_agent_has_an_explicit_policy():
    """A missing policy must fail closed, never default to permissive."""
    with pytest.raises(LeakageError, match="no isolation policy"):
        policy_for("some_new_agent")


def test_every_policy_states_a_rationale():
    for agent_id, policy in POLICIES.items():
        assert policy.rationale.strip(), f"{agent_id} policy has no stated rationale"


# --- bull / bear mutual blindness ------------------------------------------
def test_bear_agent_cannot_see_the_bull_channel():
    data = project("bear_agent", build_bus())
    assert Channel.BULL not in data.channels
    assert BULL_PROSE not in str(data.channels)


def test_bull_agent_cannot_see_the_bear_channel():
    data = project("bull_agent", build_bus())
    assert Channel.BEAR not in data.channels


def test_bull_agent_cannot_see_the_kill_channel():
    """Requirement 27: the bull case is constructed, not written as a defence."""
    data = project("bull_agent", build_bus())
    assert Channel.KILL not in data.channels


def test_kill_agent_cannot_see_the_bull_channel():
    data = project("kill_agent", build_bus())
    assert Channel.BULL not in data.channels
    assert Channel.BEAR not in data.channels


def test_bull_and_bear_receive_the_same_evidence_set():
    """Requirement 1C: identical verified evidence, no rebuttal loop."""
    bus = build_bus()
    bull = project("bull_agent", bus)
    bear = project("bear_agent", bus)
    assert [f.fact_id for f in bull.facts] == [f.fact_id for f in bear.facts]


def test_bull_prose_laundered_into_a_fact_is_caught():
    """The indirect case: copying the argument into a 'fact' must not work."""
    bus = build_bus()
    bus.add_facts(
        [
            make_fact(
                BULL_PROSE,
                ticker=TICKER,
                evidence_class=EvidenceClass.MODEL_INFERENCE,
                category=FactCategory.OTHER,
            )
        ]
    )
    with pytest.raises(LeakageError, match="bull_case"):
        project("bear_agent", bus)


def test_shared_vocabulary_does_not_false_positive():
    """A bear agent may legitimately use words that appear in the bull case."""
    bus = build_bus()
    bus.add_facts(
        [
            make_fact(
                "The second indication entered the clinic in 2026",
                ticker=TICKER,
                category=FactCategory.CLINICAL,
            )
        ]
    )
    data = project("bear_agent", bus)
    assert data is not None


# --- blind judge -----------------------------------------------------------
def test_blind_judge_receives_no_identity():
    data = project("blind_judge", build_bus())
    assert data.ticker is None
    assert data.company_name is None


def test_blind_judge_pack_contains_no_identity_markers():
    data = project("blind_judge", build_bus())
    blob = str(data.facts) + str(data.channels) + str(data.risk_flags) + str(data.params)
    for marker in identity_markers(TICKER, NAME, ALIASES):
        assert marker.lower() not in blob.lower(), f"identity marker {marker!r} leaked"


def test_blind_judge_source_urls_are_opaque_refs():
    """A sec.gov path names the company as surely as the company name does."""
    data = project("blind_judge", build_bus())
    for fact in data.facts:
        assert fact.source_url.startswith("blindref://")
        assert "sec.gov" not in fact.source_url


def test_source_ref_map_is_kept_out_of_the_pack():
    bus = build_bus()
    guard = IsolationGuard(bus)
    data = guard.project(
        "blind_judge", ticker=TICKER, company_name=NAME, aliases=ALIASES
    )
    assert "_source_ref_map" not in data.params
    ref_map = guard.source_ref_maps["blind_judge"]
    assert ref_map.to_source, "the orchestrator still needs the map to restore citations"
    assert "sec.gov" in str(ref_map.to_source)


def test_user_holdings_never_reach_the_blind_judge():
    """Requirement 10, named explicitly in requirement 17."""
    preferences = {
        "holdings": "I already own 5000 shares at an average cost of 4.20",
        "opinion": "I really like this company and want a 200x",
    }
    data = project("blind_judge", build_bus(), user_preferences=preferences)
    blob = str(data.params) + str(data.channels) + str(data.facts)
    assert "5000 shares" not in blob
    assert "200x" not in blob
    assert "user_preferences" not in data.params


@pytest.mark.parametrize("agent_id", ["fact_collector", "kill_agent", "bear_agent", "blind_judge"])
def test_preferences_are_dropped_for_every_research_agent(agent_id):
    """Requirement 10 names these four agents specifically."""
    preferences = {"holdings": "I already own 5000 shares at an average cost of 4.20"}
    data = project(agent_id, build_bus(), user_preferences=preferences)
    assert "user_preferences" not in data.params
    assert "5000 shares" not in str(data.params)


def test_portfolio_agent_is_the_only_one_that_sees_preferences():
    preferences = {"holdings": "I already own 5000 shares at an average cost of 4.20"}
    data = project("portfolio", build_bus(), user_preferences=preferences)
    assert data.params["user_preferences"] == preferences

    for agent_id, policy in POLICIES.items():
        if agent_id == "portfolio":
            continue
        assert not policy.sees_user_preferences, f"{agent_id} must not see user preferences"


def test_prior_scores_and_rankings_are_stripped_from_the_blind_pack():
    """Requirement 6/14: no anchoring on the system's own previous opinion."""
    data = project(
        "blind_judge",
        build_bus(),
        params={"prior_scores": {"explosive_potential": 9.5}, "prior_rank": 1, "price": 3.4},
    )
    assert "prior_scores" not in data.params
    assert "prior_rank" not in data.params
    assert data.params["price"] == 3.4


# --- fact collector ---------------------------------------------------------
def test_fact_collector_sees_no_analysis_at_all():
    """Requirement 1A: it cannot search selectively for a view it already holds."""
    data = project("fact_collector", build_bus())
    assert data.channels == {}
    assert data.facts == ()
    assert data.risk_flags == ()


# --- fingerprints -----------------------------------------------------------
def test_fingerprints_exclude_shared_baseline_prose():
    shared = "Cash and equivalents were 31.5 million as of the most recent quarter"
    fingerprints = derive_fingerprints([shared + " and the thesis rests on optionality"], [shared])
    assert all("cash and equivalents were" not in f for f in fingerprints)
    assert any("optionality" in f for f in fingerprints)


def test_fingerprints_are_multiword_shingles():
    fingerprints = derive_fingerprints([BULL_PROSE], [])
    assert fingerprints
    assert all(" " in f for f in fingerprints)


def test_identity_markers_include_name_stem_but_not_generic_words():
    markers = [m.lower() for m in identity_markers("CRBP", "Corbus Pharmaceuticals Holdings")]
    assert "crbp" in markers
    assert "corbus" in markers
    assert "pharmaceuticals" not in markers
    assert "holdings" not in markers
