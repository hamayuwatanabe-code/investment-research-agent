"""Pipeline wiring for the per-agent effort policy (requirement G).

Proves ``Pipeline._agent_for`` actually resolves and applies a per-agent
effort (via ``llm.effort_policy.resolve_effort``) when constructing an LLM
agent -- not merely that the policy function itself works in isolation
(covered by tests/unit/test_effort_policy.py) or that the resolved effort
reaches the wire (covered by
tests/integration/test_llm_agents_against_mock_api.py).
"""

from __future__ import annotations

from investment_research.agents.bear_agent import BearAgent
from investment_research.agents.llm_agents import LLMRegulatoryAgent
from investment_research.agents.llm_agents2 import LLMBearAgent
from investment_research.agents.regulatory import RegulatoryAgent
from investment_research.collectors.search import NullSearchProvider
from investment_research.llm.client import LLMBudget, LLMClient
from investment_research.orchestrator.pipeline import Pipeline


class _EmptyBus:
    channels: dict = {}


def _pipeline(repo, *, llm_effort: str = "high", agent_effort_policy=None) -> Pipeline:
    budget = LLMBudget(max_total_tokens=1_000_000)
    llm = LLMClient(
        api_key="sk-ant-test", base_url="http://127.0.0.1:1", effort=llm_effort, budget=budget
    )
    return Pipeline(
        repo, NullSearchProvider(), llm=llm, agent_effort_policy=agent_effort_policy
    )


def test_agent_for_resolves_high_effort_for_regulatory(repo):
    pipeline = _pipeline(repo)
    agent = pipeline._agent_for("regulatory", RegulatoryAgent(), LLMRegulatoryAgent, _EmptyBus())
    assert isinstance(agent, LLMRegulatoryAgent)
    assert agent.effort == "high"


def test_agent_for_resolves_medium_effort_for_bear(repo):
    pipeline = _pipeline(repo)
    agent = pipeline._agent_for("bear_agent", BearAgent(), LLMBearAgent, _EmptyBus())
    assert isinstance(agent, LLMBearAgent)
    assert agent.effort == "medium"


def test_agent_for_effort_never_exceeds_the_run_ceiling(repo):
    """If the run was invoked with --llm-effort=low, an agent whose default
    policy entry is 'high' must never silently run above that ceiling."""
    pipeline = _pipeline(repo, llm_effort="low")
    agent = pipeline._agent_for("regulatory", RegulatoryAgent(), LLMRegulatoryAgent, _EmptyBus())
    assert agent.effort == "low"


def test_agent_for_honors_a_custom_agent_effort_policy(repo):
    """Requirement G: the policy is configurable, not hard-coded -- a caller
    can supply its own mapping and the Pipeline must actually use it."""
    pipeline = _pipeline(repo, agent_effort_policy={"bear_agent": "high"})
    agent = pipeline._agent_for("bear_agent", BearAgent(), LLMBearAgent, _EmptyBus())
    assert agent.effort == "high"
