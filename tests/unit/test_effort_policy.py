"""Per-agent LLM reasoning-effort policy (requirement G).

The defect: the interpretive stage consumed its quota before every agent got
a turn, and every LLM agent always ran at the single, uniform --llm-effort
value regardless of how much its own output actually drives the falsification
verdict. This tests the pure policy-resolution logic; an integration test in
tests/integration/test_llm_agents_against_mock_api.py proves the resolved
effort actually reaches the API request.
"""

from __future__ import annotations

from investment_research.llm.effort_policy import (
    DEFAULT_AGENT_EFFORT_POLICY,
    resolve_effort,
)


def test_default_policy_names_falsification_and_judgement_agents_high():
    for agent_id in ("regulatory", "science", "contradiction", "kill_agent", "blind_judge"):
        assert DEFAULT_AGENT_EFFORT_POLICY[agent_id] == "high"


def test_default_policy_names_case_building_agents_medium():
    for agent_id in ("competitive", "bear_agent", "bull_agent"):
        assert DEFAULT_AGENT_EFFORT_POLICY[agent_id] == "medium"


def test_resolve_effort_uses_policy_when_within_ceiling():
    assert resolve_effort("bear_agent", ceiling="high") == "medium"
    assert resolve_effort("regulatory", ceiling="high") == "high"


def test_resolve_effort_never_exceeds_the_ceiling_by_default():
    """CLI --llm-effort is the ceiling. A policy entry above it (e.g. the run
    was invoked with --llm-effort=low) must never be honored unless
    explicitly allowed."""
    assert resolve_effort("regulatory", ceiling="low") == "low"
    assert resolve_effort("kill_agent", ceiling="medium") == "medium"


def test_resolve_effort_can_exceed_ceiling_only_when_explicitly_allowed():
    assert (
        resolve_effort("regulatory", ceiling="low", allow_exceed_ceiling=True) == "high"
    )


def test_resolve_effort_unmapped_agent_uses_the_ceiling():
    assert resolve_effort("catalyst_agent_not_in_policy", ceiling="high") == "high"
    assert resolve_effort("catalyst_agent_not_in_policy", ceiling="low") == "low"


def test_resolve_effort_accepts_a_custom_policy_never_hard_coded():
    """Requirement G: the policy is data a caller can replace wholesale --
    never something an agent decides about itself."""
    custom_policy = {"bear_agent": "xhigh"}
    assert resolve_effort("bear_agent", ceiling="xhigh", policy=custom_policy) == "xhigh"
    # An agent absent from a CUSTOM policy still falls back to the ceiling,
    # not to the default policy's value for it.
    assert resolve_effort("regulatory", ceiling="medium", policy=custom_policy) == "medium"
