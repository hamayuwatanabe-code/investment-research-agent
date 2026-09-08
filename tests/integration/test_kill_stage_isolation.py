"""Regression (requirement A, item 4): Kill's mandatory web searches must not
run under whatever stage happened to execute immediately before Stage 5.

Before the fix, ``Pipeline`` set ``budget.set_stage("escalation")`` for Stage
3b and never reset it before Stage 5 (Kill) ran -- so Kill's mandatory
searches (issued through the same live research provider used everywhere
else) were charged against, and gated by, the ALREADY-SPENT escalation
quota. This runs the real ``Pipeline`` with a real ``LLMClient``/
``AnthropicWebResearchProvider`` pair against a local mock Anthropic API and
asserts every LLM call recorded for the Kill Agent's mandatory queries was
stamped with an explicit stage other than "escalation".
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
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research.anthropic_web import AnthropicWebResearchProvider

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)


class _EmptySearchHandler(BaseHTTPRequestHandler):
    """Every POST gets an empty, successful web_search_tool_result."""

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


def test_kill_mandatory_search_is_not_stamped_escalation(repo, fixture_dir, mock_server):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")

    budget = LLMBudget(max_total_tokens=1_000_000)
    llm = LLMClient(
        api_key="sk-ant-test", base_url=mock_server, max_retries=0, timeout=10, budget=budget
    )
    research = AnthropicWebResearchProvider(llm)

    # `llm` given to BOTH the Pipeline (so Stage boundaries actually toggle
    # `budget.current_stage`) and the research provider it hands to KillAgent
    # (so KillAgent's searches record against that SAME budget object).
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, research=research, llm=llm)
    pipeline.run(
        "DEMOBIO",
        metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"],
        aliases=metadata.get("aliases", ()),
    )

    kill_calls = [c for c in budget.calls if c.agent_id == "kill_agent"]
    assert kill_calls, "the Kill Agent's mandatory searches must have gone through the LLM budget"
    assert all(c.stage != "escalation" for c in kill_calls), (
        "Kill's mandatory web searches must never be attributed to the 'escalation' stage "
        f"leaked from Stage 3b -- got stages {[c.stage for c in kill_calls]}"
    )
    # Explicit, not merely "not escalation" -- every call landed under a
    # named stage, never the empty/unset ambient default.
    assert all(c.stage for c in kill_calls)
