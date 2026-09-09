"""AnthropicWebResearchProvider.search_batch() against a mock Anthropic API.

Proves the marker-based attribution convention (requirement A), per-intent
source-restriction enforcement when a batch mixes incompatible restrictions
(requirement B/9), and the per-batch diagnostics requirement F asks for --
all without a live network call: a local HTTP server stands in for the real
Anthropic Messages API, and the real SDK talks to it over real sockets.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.llm.client import LLMClient
from investment_research.research.anthropic_web import AnthropicWebResearchProvider
from investment_research.research.batching import ResearchIntent
from investment_research.schemas.enums import ResearchDomain

pytestmark = pytest.mark.integration

RESPONSE: dict = {}
REQUESTS: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            REQUESTS.append(json.loads(body))
        except json.JSONDecodeError:
            REQUESTS.append({})
        payload = json.dumps(RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def provider(base_url):
    REQUESTS.clear()
    llm = LLMClient(api_key="sk-ant-test", base_url=base_url, max_retries=0, timeout=10)
    return AnthropicWebResearchProvider(llm)


def _message(content):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 900, "output_tokens": 150},
    }


def test_search_batch_attributes_each_result_to_its_marked_intent(provider):
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "content": [{"url": "https://www.fda.gov/a", "title": "FDA letter"}],
            },
            {"type": "text", "text": "-- INTENT sci_1 --"},
            {"type": "server_tool_use", "name": "web_search", "input": {"query": "q2"}},
            {"type": "web_search_tool_result", "content": []},
        ]
    )
    intents = [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2"),
    ]
    results, meta = provider.search_batch(intents, agent_id="adversarial_bear")

    assert set(results) == {"reg_1", "sci_1"}
    assert len(results["reg_1"].documents) == 1
    assert results["reg_1"].documents[0].url == "https://www.fda.gov/a"
    assert results["sci_1"].documents == []
    assert results["reg_1"].executed is True
    assert results["sci_1"].executed is True
    assert meta["server_tool_uses"] == 2
    assert meta["prompt_tokens"] == 900
    assert meta["output_tokens"] == 150

    # The request itself must actually list both intents (one call served both).
    assert len(REQUESTS) == 1
    prompt = REQUESTS[0]["messages"][0]["content"]
    assert "INTENT reg_1" in prompt
    assert "INTENT sci_1" in prompt


def test_search_batch_result_before_any_marker_is_never_attributed(provider):
    global RESPONSE
    RESPONSE = _message(
        [
            {
                "type": "web_search_tool_result",
                "content": [{"url": "https://www.fda.gov/unlabelled", "title": "t"}],
            },
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "web_search_tool_result", "content": []},
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert results["reg_1"].documents == [], "the unlabelled result must never be borrowed"


def test_search_batch_never_attributes_an_intent_id_the_batch_never_asked_for(provider):
    global RESPONSE
    RESPONSE = _message([{"type": "text", "text": "-- INTENT not_in_this_batch --"}])
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" not in results, "an intent the model never addressed must stay absent, not guessed"


def test_search_batch_omitted_intent_is_absent_from_results(provider):
    """H3, exercised against the real parser: an intent with no marker at
    all in the response is absent from the returned mapping, never
    silently treated as a zero-result search."""
    global RESPONSE
    RESPONSE = _message([{"type": "text", "text": "-- INTENT reg_1 --"}, {"type": "web_search_tool_result", "content": []}])
    intents = [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2"),
    ]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" in results
    assert "sci_1" not in results


def test_search_batch_shared_source_restriction_applied_at_tool_level(provider):
    global RESPONSE
    RESPONSE = _message([{"type": "text", "text": "-- INTENT a --"}, {"type": "web_search_tool_result", "content": []}])
    intents = [
        ResearchIntent(intent_id="a", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q", source_restrictions=("sec.gov",)),
        ResearchIntent(intent_id="b", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q2", source_restrictions=("sec.gov",)),
    ]
    provider.search_batch(intents, agent_id="adversarial_bear")
    tool = REQUESTS[0]["tools"][0]
    assert tool.get("allowed_domains") == ["sec.gov"]


def test_search_batch_incompatible_restrictions_are_never_blended_at_tool_level(provider):
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT a --"},
            {
                "type": "web_search_tool_result",
                "content": [
                    {"url": "https://www.sec.gov/x", "title": "on-domain for a"},
                    {"url": "https://pubmed.ncbi.nlm.nih.gov/1", "title": "off-domain for a"},
                ],
            },
        ]
    )
    intents = [
        ResearchIntent(intent_id="a", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q", source_restrictions=("sec.gov",)),
        ResearchIntent(
            intent_id="b", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2",
            source_restrictions=("pubmed.ncbi.nlm.nih.gov",),
        ),
    ]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    tool = REQUESTS[0]["tools"][0]
    assert "allowed_domains" not in tool, "two different restrictions must never be merged at the tool level"
    # Intent "a" only keeps documents matching ITS OWN restriction, even
    # though the (unrestricted) tool call surfaced an off-domain document.
    assert [d.url for d in results["a"].documents] == ["https://www.sec.gov/x"]
