"""Regression (defect E): the live research provider must reach the Kill Agent.

A second live run reported "23 mandatory kill queries were not executed (no
search provider configured)" despite `--live` wiring up a real
`AnthropicWebResearchProvider`. Root cause: `KillAgent` only ever knew about
the pipeline's unrelated, offline `SearchProvider` (Tavily/Brave/none, built by
`build_search_provider`) -- never the live `ResearchProvider` that adversarial
discovery and escalation already use. So even a fully live, credentialed run
left the Kill Agent's own mandatory search set completely unexecuted.

This test runs the real `Pipeline` end to end (DEMOBIO fixtures for
collection, so this is not a live network call) with a real
`AnthropicWebResearchProvider` pointed at a local mock Anthropic API standing
in for the live one, and proves the Kill Agent's mandatory queries actually
execute against it.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.llm.client import LLMClient
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research.anthropic_web import AnthropicWebResearchProvider
from investment_research.schemas.enums import QueryPurpose

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)

REQUESTS: list[dict] = []


class _EmptySearchHandler(BaseHTTPRequestHandler):
    """Every POST gets an empty, successful web_search_tool_result."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            REQUESTS.append(json.loads(body))
        except json.JSONDecodeError:
            REQUESTS.append({})
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


def test_kill_agent_mandatory_search_uses_the_live_research_provider(
    repo, fixture_dir, mock_server
):
    REQUESTS.clear()
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")

    # Stands in for the real `--live` AnthropicWebResearchProvider: same
    # class, pointed at a local mock instead of the real API. `llm=None` on
    # the Pipeline keeps every OTHER agent on its deterministic path, so this
    # test isolates exactly the thing that broke: Kill Agent <-> the live
    # research provider, not the interpretive LLM agents (covered elsewhere).
    llm = LLMClient(api_key="sk-ant-test", base_url=mock_server, max_retries=0, timeout=10)
    research = AnthropicWebResearchProvider(llm)

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, research=research)
    result = pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )

    assert REQUESTS, "the live research provider must actually have been called"

    kill_records = [r for r in result.agent_records if r.agent_id == "kill_agent"]
    assert kill_records, "kill_agent must have run"
    assert "no search provider configured" not in kill_records[0].errors, (
        "the Kill Agent must not report the offline SearchProvider's failure message when a "
        "usable live research provider was available for this run"
    )

    kill_payload = result.bus.channels[Channel.KILL].payload
    assert kill_payload["queries_executed"], "mandatory kill queries must have actually run"
    assert kill_payload["queries_not_executed"] == [], (
        "every mandatory kill query must execute against the live provider, not go unexecuted"
    )

    # Auditable: every mandatory kill query landed in the DiscoveryLog with a
    # resolvable agent_id and QueryPurpose (never mixed into Bull's context).
    kill_queries = [q for q in result.adversarial.discovery.queries if q.agent_id == "kill_agent"]
    assert kill_queries
    assert all(q.query_purpose is QueryPurpose.BEAR for q in kill_queries)
    assert all(q.executed for q in kill_queries)
