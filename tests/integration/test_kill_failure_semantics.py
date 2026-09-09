"""Regression (requirement B): Kill's failure-state reporting must be accurate
even when an EARLIER stage's budget is already exhausted.

A live run exhausted the "escalation" stage's quota before Kill (Stage 5) ever
ran, and Kill reported "23 mandatory kill queries were not executed (no
search provider configured)" -- collapsing budget starvation from a DIFFERENT
stage into a message that falsely implies no provider was ever configured.

This runs the real ``Pipeline`` with a real, credentialed
``AnthropicWebResearchProvider`` (pointed at a local mock Anthropic API) and
pre-exhausts the "escalation" stage's quota before the run starts, standing
in for a live run where earlier stages spent it all. Because Kill's mandatory
searches now run under their own "discovery" stage (requirement A), Kill must
still see the provider as usable and its own, independently-scoped budget
decides whether it can search -- never reporting NO_PROVIDER for a provider
that plainly is configured.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.llm.client import LLMBudget, LLMClient
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research.anthropic_web import AnthropicWebResearchProvider
from investment_research.schemas.enums import KillSearchFailureReason

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)


class _EmptySearchHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "web_search_tool_result", "content": []}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EmptySearchHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_kill_search_failure_semantics_survive_escalation_exhaustion(
    repo, fixture_dir, mock_server
):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")

    budget = LLMBudget(max_total_tokens=1_000_000)
    # Simulate a live run where earlier stages (2b/3b escalation) already
    # spent escalation's entire quota before Kill (Stage 5) gets a turn.
    escalation_cap = budget.stage_cap_tokens("escalation") or 0
    budget.stage_used["escalation"] = escalation_cap
    budget.stage_exhausted.add("escalation")

    llm = LLMClient(
        api_key="sk-ant-test", base_url=mock_server, max_retries=0, timeout=10, budget=budget
    )
    research = AnthropicWebResearchProvider(llm)
    assert research.available() == (True, "ready")

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, research=research, llm=llm)
    result = pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )

    kill_payload = result.bus.channels[Channel.KILL].payload
    outcomes = kill_payload["query_outcomes"]
    assert outcomes, "the Kill Agent must have attempted its mandatory queries"

    reasons = {o["reason"] for o in outcomes}
    assert KillSearchFailureReason.NO_PROVIDER.value not in reasons, (
        "a genuinely configured, available research provider must never be reported as "
        "NO_PROVIDER"
    )
    assert KillSearchFailureReason.SKIPPED_DUE_TO_BUDGET.value not in reasons, (
        "Kill's own 'discovery' stage budget was never touched by escalation's exhaustion, so "
        "none of its mandatory queries should have been skipped for budget"
    )
    assert reasons <= {
        KillSearchFailureReason.EXECUTED_WITH_RESULTS.value,
        KillSearchFailureReason.EXECUTED_ZERO_RESULTS.value,
    }, f"unexpected kill query failure reasons: {reasons}"

    kill_records = [r for r in result.agent_records if r.agent_id == "kill_agent"]
    assert kill_records
    for record in kill_records:
        assert "no search provider configured" not in " ".join(record.errors)

    assert kill_payload["queries_not_executed"] == []


def test_kill_unexecuted_summary_preserves_the_original_exhaustion_cause(
    repo, fixture_dir, mock_server
):
    """PROVIDER_UNAVAILABLE currently covers both "no credentials configured"
    and "the run's GLOBAL token budget was already exhausted by an earlier
    stage/agent" -- the same enum value either way (LLMClient.available()
    only ever checks the one global flag). The reason CODE alone loses that
    distinction, but the actual cause text (LLMBudget.exhausted_reason,
    which names the offending agent and the used/remaining counts) must
    still survive into Kill's own unexecuted-queries summary, not collapse
    to a bare "23 PROVIDER_UNAVAILABLE" count."""
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")

    budget = LLMBudget(max_total_tokens=1_000_000)
    # Simulate what a live run's post-hoc actual-usage overshoot leaves
    # behind: the budget is GLOBALLY exhausted (not merely one stage), with
    # a descriptive reason naming which agent's call put it over.
    budget.exhausted = True
    budget.exhausted_reason = (
        "actual usage after agent 'science' put the LLM budget over its ceiling "
        "(999999/1000000 tokens used)."
    )

    llm = LLMClient(
        api_key="sk-ant-test", base_url=mock_server, max_retries=0, timeout=10, budget=budget
    )
    research = AnthropicWebResearchProvider(llm)
    assert research.available()[0] is False

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, research=research, llm=llm)
    result = pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )

    kill_payload = result.bus.channels[Channel.KILL].payload
    outcomes = kill_payload["query_outcomes"]
    reasons = {o["reason"] for o in outcomes}
    assert reasons == {KillSearchFailureReason.PROVIDER_UNAVAILABLE.value}

    kill_records = [r for r in result.agent_records if r.agent_id == "kill_agent"]
    assert kill_records
    summary = kill_records[0].errors
    assert "PROVIDER_UNAVAILABLE" in summary
    # The original cause -- WHICH agent's call exhausted the budget, and the
    # actual token counts -- must be readable from the summary, not just a
    # bare reason-code count.
    assert "science" in summary
    assert "999999" in summary
