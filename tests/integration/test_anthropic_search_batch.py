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

from investment_research.llm.client import LLMBudget, LLMClient
from investment_research.research.anthropic_web import (
    AnthropicWebResearchProvider,
    _affordable_prefix,
)
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
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.fda.gov/a", "title": "FDA letter"}],
            },
            {"type": "text", "text": "-- INTENT sci_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_2", "input": {"query": "q2"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_2", "content": []},
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
            # A search issued before any INTENT marker: current_id is None
            # at this point, so its id is never recorded against any
            # intent -- the matching result below cannot resolve and is
            # dropped as ambiguous, never borrowed by whichever intent
            # happens to be current later.
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_stray", "input": {"query": "stray"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_stray",
                "content": [{"url": "https://www.fda.gov/unlabelled", "title": "t"}],
            },
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
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
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
        ]
    )
    intents = [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2"),
    ]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" in results
    assert "sci_1" not in results


def test_search_batch_shared_source_restriction_applied_at_tool_level(provider):
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT a --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
        ]
    )
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
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
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


# =========================================================================
# Regressions for the c73e51b follow-up fixes:
# 1. a marker alone (no completed search) must never read as executed
# 2. a tool error must propagate per-intent, never be silently discarded
# 3. an unrestricted intent must never be narrowed by a sibling's restriction
# 4. attribution must not fall back to a stale current_id after an unknown
#    marker, and must prefer tool_use_id/id correlation over block order
# =========================================================================


def test_marker_only_with_no_tool_use_or_result_is_incomplete_not_executed(provider):
    """マーカーのみ: a bare marker with no search at all must never become
    EXECUTED_ZERO_RESULTS."""
    global RESPONSE
    RESPONSE = _message([{"type": "text", "text": "-- INTENT reg_1 --"}])
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" not in results


def test_tool_use_with_no_result_block_is_incomplete_not_executed(provider):
    """tool_useのみで結果なし: a search was issued but no result block ever
    arrived (e.g. the response ended first). The intent is reported as
    unresolved (executed=False, an "IncompleteIntent" error) rather than
    silently read as complete success -- never EXECUTED_ZERO_RESULTS."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" in results
    assert results["reg_1"].executed is False
    assert "IncompleteIntent" in results["reg_1"].error
    assert results["reg_1"].documents == []


def test_a_completed_search_with_zero_results_is_executed_zero_results(provider):
    """正常な0件: a genuinely completed search (a real result block) that
    found nothing IS EXECUTED_ZERO_RESULTS -- the one case that must
    actually read as executed with no documents."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" in results
    assert results["reg_1"].documents == []
    assert results["reg_1"].error == ""
    assert results["reg_1"].executed is True


def test_search_tool_error_is_propagated_on_the_intents_result(provider):
    """検索ツールエラー: a tool error block must never be silently
    discarded -- it must appear on the intent's own ResearchResult.error."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": {"error_code": "max_uses_exceeded"},
            },
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" in results
    assert "max_uses_exceeded" in results["reg_1"].error
    assert results["reg_1"].documents == []


def test_same_intent_partial_success_and_failure_keeps_evidence_and_reports_error(provider):
    """同一intentの成功・失敗混在: one intent, two searches -- one succeeds,
    one errors. The successful document must be kept; the failure must
    never be hidden as complete success."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.fda.gov/a", "title": "t"}],
            },
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_2", "input": {"query": "q2"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_2",
                "content": {"error_code": "timeout"},
            },
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    result = results["reg_1"]
    assert [d.url for d in result.documents] == ["https://www.fda.gov/a"], "the successful document must survive"
    assert "timeout" in result.error, "the failure must not be hidden behind the successful document"


def test_response_truncation_leaves_the_unreached_intent_incomplete(provider):
    """応答打ち切りと未完了intent: the response ends after the first intent
    is fully served; the second (only marked, never searched) stays
    INCOMPLETE_RESPONSE."""
    global RESPONSE
    RESPONSE = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
            {"type": "text", "text": "-- INTENT sci_1 --"},
        ],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 900, "output_tokens": 150},
    }
    intents = [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2"),
    ]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert "reg_1" in results
    assert "sci_1" not in results


def test_unknown_marker_after_a_known_one_stops_falling_back_to_the_known_intent(provider):
    """既知マーカーの後に未知マーカー: once an unknown marker appears, a
    result with no tool_use_id of its own must never be re-attached to the
    PRIOR known intent."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.fda.gov/a", "title": "t"}],
            },
            {"type": "text", "text": "-- INTENT unknown_ghost --"},
            # No tool_use_id here -- under the OLD (buggy) order-based-only
            # logic this would have been silently re-attached to reg_1.
            {
                "type": "web_search_tool_result",
                "content": [{"url": "https://www.fda.gov/b", "title": "t2"}],
            },
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert [d.url for d in results["reg_1"].documents] == ["https://www.fda.gov/a"], (
        "a result following an unknown marker must never be borrowed by the prior known intent"
    )


def test_shared_restrictions_never_apply_when_one_intent_is_unrestricted(provider):
    """制限あり＋制限なし: the exact reported bug -- [('sec.gov',), ()] must
    NOT collapse to a shared ('sec.gov',) tool-level restriction."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT a --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.reuters.com/x", "title": "off-sec.gov, but unrestricted intent"}],
            },
        ]
    )
    intents = [
        ResearchIntent(intent_id="a", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q1", source_restrictions=()),
        ResearchIntent(
            intent_id="b", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q2", source_restrictions=("sec.gov",)
        ),
    ]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    tool = REQUESTS[0]["tools"][0]
    assert "allowed_domains" not in tool, (
        "an unrestricted intent must never be narrowed just because a sibling wants sec.gov only"
    )
    # The unrestricted intent's own (post-hoc) filtering keeps the off-domain document.
    assert [d.url for d in results["a"].documents] == ["https://www.reuters.com/x"]


def test_all_intents_sharing_one_restriction_apply_it_at_tool_level(provider):
    """全intentが同一制限: reconfirms the positive case still works after
    the fix -- every intent wanting the exact same non-empty restriction
    still gets it applied once, at the tool level."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT a --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
        ]
    )
    intents = [
        ResearchIntent(intent_id="a", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q", source_restrictions=("sec.gov",)),
        ResearchIntent(intent_id="b", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q2", source_restrictions=("sec.gov",)),
        ResearchIntent(intent_id="c", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q3", source_restrictions=("sec.gov",)),
    ]
    provider.search_batch(intents, agent_id="adversarial_bear")
    tool = REQUESTS[0]["tools"][0]
    assert tool.get("allowed_domains") == ["sec.gov"]


def test_different_restrictions_per_intent_are_enforced_independently(provider):
    """intentごとに異なる制限: each intent's OWN restriction is enforced
    post-hoc, independent of what any other intent in the batch wants."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT a --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    {"url": "https://www.sec.gov/x", "title": "on-domain for a"},
                    {"url": "https://pubmed.ncbi.nlm.nih.gov/1", "title": "on-domain for b, not a"},
                ],
            },
            {"type": "text", "text": "-- INTENT b --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_2", "input": {"query": "q2"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_2",
                "content": [
                    {"url": "https://www.sec.gov/y", "title": "on-domain for a, not b"},
                    {"url": "https://pubmed.ncbi.nlm.nih.gov/2", "title": "on-domain for b"},
                ],
            },
        ]
    )
    intents = [
        ResearchIntent(intent_id="a", domain=ResearchDomain.CAPITAL_STRUCTURE, question="q1", source_restrictions=("sec.gov",)),
        ResearchIntent(
            intent_id="b", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2",
            source_restrictions=("pubmed.ncbi.nlm.nih.gov",),
        ),
    ]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert [d.url for d in results["a"].documents] == ["https://www.sec.gov/x"]
    assert [d.url for d in results["b"].documents] == ["https://pubmed.ncbi.nlm.nih.gov/2"]


def test_out_of_order_tool_results_are_still_correctly_attributed_by_id(provider):
    """複数tool_useの結果順序が入れ替わるケース: both searches are issued
    before either result arrives, so by the time the results come back
    ``current_id`` (order-based tracking) already points at the SECOND
    intent for both -- only id-based correlation attributes them
    correctly."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {"type": "text", "text": "-- INTENT sci_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_2", "input": {"query": "q2"}},
            # Results arrive in the OPPOSITE order from their searches.
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_2",
                "content": [{"url": "https://pubmed.ncbi.nlm.nih.gov/1", "title": "sci"}],
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.fda.gov/a", "title": "reg"}],
            },
        ]
    )
    intents = [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2"),
    ]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    assert [d.url for d in results["reg_1"].documents] == ["https://www.fda.gov/a"]
    assert [d.url for d in results["sci_1"].documents] == ["https://pubmed.ncbi.nlm.nih.gov/1"]


# =========================================================================
# Regressions for the 4fd3c9e follow-up review (this turn):
# 1. an id-less result after two DISTINCT known searches must never be
#    guessed at by falling back to whichever marker was current -- it is
#    genuinely ambiguous and must be dropped, not assigned to either.
# 2. an intent with one resolved search and one issued-but-never-answered
#    search must never read as complete success; its already-obtained
#    documents must still survive.
# 3. an intent whose every issued search actually resolves must still read
#    as ordinary complete success (positive control for #2).
# =========================================================================


def test_search_a_then_search_b_then_id_less_result_is_never_guessed(provider):
    """A検索→B検索→IDなし結果: once two distinct intents have each issued
    their own search, a THIRD result block with no tool_use_id at all
    cannot be attributed to either -- it must be dropped as ambiguous, not
    assigned to whichever intent happens to be "current"."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {"type": "text", "text": "-- INTENT sci_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_2", "input": {"query": "q2"}},
            # An id-less result arrives after BOTH searches were issued --
            # genuinely ambiguous, must not be guessed for either.
            {
                "type": "web_search_tool_result",
                "content": [{"url": "https://www.fda.gov/ambiguous", "title": "t"}],
            },
        ]
    )
    intents = [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2"),
    ]
    results, meta = provider.search_batch(intents, agent_id="adversarial_bear")
    # Neither intent ever got a resolved result -- both stay unresolved
    # (an issued search with nothing attributed to it), never a guessed
    # success and never a silently completed zero-result search.
    assert "https://www.fda.gov/ambiguous" not in [
        d.url for r in results.values() for d in r.documents
    ]
    assert meta["ambiguous_results"] == 1


def test_same_intent_second_result_never_arrives_keeps_evidence_and_is_incomplete(provider):
    """同一intentで1回成功、2回目の結果が未着: the first search for an
    intent resolves with a real document; a second search is issued for
    the SAME intent but its result block never arrives before the response
    ends. The intent must never read as complete success -- but the
    document already obtained must survive."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.fda.gov/a", "title": "t"}],
            },
            # Second search for the SAME intent, issued but never answered.
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_2", "input": {"query": "q1b"}},
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    result = results["reg_1"]
    assert [d.url for d in result.documents] == ["https://www.fda.gov/a"], "the resolved document must survive"
    assert result.executed is False
    assert "IncompleteIntent" in result.error


def test_same_intent_all_searches_resolve_is_ordinary_complete_success(provider):
    """同一intentで全検索が正常終了: two searches for the SAME intent, both
    resolved -- ordinary complete success, positive control for the
    unresolved-second-search case above."""
    global RESPONSE
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.fda.gov/a", "title": "t"}],
            },
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_2", "input": {"query": "q1b"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_2",
                "content": [{"url": "https://www.fda.gov/b", "title": "t2"}],
            },
        ]
    )
    intents = [ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1")]
    results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")
    result = results["reg_1"]
    assert {d.url for d in result.documents} == {"https://www.fda.gov/a", "https://www.fda.gov/b"}
    assert result.executed is True
    assert result.error == ""


# =========================================================================
# Regressions for the v5 discovery-budget-blowout review (this turn):
# a production run showed a single 6-intent batched call spend ~102k actual
# tokens against a 60k discovery-stage quota (the flat, single-search
# preflight reserve never scaled with how many searches the call's own
# max_uses ceiling allowed), starving every other required domain's search
# for the rest of the stage. These prove the split/allocate fix end to end,
# through the real mock HTTP transport, at production-batch scale (six
# required-domain intents) and under a tight, production-shaped budget --
# never by increasing the budget or dropping a required domain.
# =========================================================================


class _EchoIntentsHandler(BaseHTTPRequestHandler):
    """Responds with a real, ATTRIBUTED marker+search+result triple for
    every INTENT id it finds in the request prompt -- whatever subset of
    the original six ends up in the prompt after the split/allocate
    decision, this always answers exactly (and only) that subset, exactly
    as a compliant live model would."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        import re as _re

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            request = {}
        REQUESTS.append(request)
        prompt_text = ""
        for message in request.get("messages", []):
            content = message.get("content", "")
            prompt_text += content if isinstance(content, str) else json.dumps(content)
        intent_ids = _re.findall(r"INTENT (\S+) \(domain:", prompt_text)

        content_blocks = []
        for intent_id in intent_ids:
            tool_use_id = f"toolu_{intent_id}"
            content_blocks.append({"type": "text", "text": f"-- INTENT {intent_id} --"})
            content_blocks.append(
                {"type": "server_tool_use", "name": "web_search", "id": tool_use_id, "input": {"query": intent_id}}
            )
            content_blocks.append(
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"url": f"https://www.fda.gov/{intent_id}", "title": intent_id}],
                }
            )

        payload = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": content_blocks,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 900, "output_tokens": 150},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def echo_base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EchoIntentsHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _six_domain_intents() -> list[ResearchIntent]:
    return [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="regulatory question"),
        ResearchIntent(intent_id="cap_1", domain=ResearchDomain.CAPITAL_STRUCTURE, question="capital structure question"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="science question"),
        ResearchIntent(intent_id="comp_1", domain=ResearchDomain.COMPETITION, question="competition question"),
        ResearchIntent(intent_id="cat_1", domain=ResearchDomain.CATALYST, question="catalyst question"),
        ResearchIntent(intent_id="con_1", domain=ResearchDomain.CONTRADICTION, question="contradiction question"),
    ]


def test_production_scale_batch_under_a_tight_stage_budget_is_served_partially_not_blown(echo_base_url):
    """The exact production shape: six required-domain intents in one wave-1
    batch, under a discovery-stage quota too small to afford all six in one
    call. The batch must be served PARTIALLY -- as many domains as the
    stage can actually afford, sent in a request whose own preflight
    reservation genuinely fits -- never all six gambled into one
    call that then blows the stage's real budget, and never zero domains
    either. The domains left out must be cleanly absent from `results`
    -- excluded explicitly with a SKIPPED_DUE_TO_BUDGET-shaped result, never
    silently marked as searched, and never absent the way a genuinely
    sent-but-omitted intent would be."""
    REQUESTS.clear()
    budget = LLMBudget(max_total_tokens=100_000)
    budget.set_stage("discovery")
    llm = LLMClient(api_key="sk-ant-test", base_url=echo_base_url, max_retries=0, timeout=10, budget=budget)
    provider = AnthropicWebResearchProvider(llm)

    intents = _six_domain_intents()
    expected_included = _affordable_prefix(intents, budget, effort="low", max_uses_per_batch=8)
    assert 0 < len(expected_included) < len(intents), (
        "the test's own budget must actually force a split for this assertion to mean anything"
    )

    results, meta = provider.search_batch(intents, agent_id="adversarial_bear")

    # Exactly one real HTTP call was made -- still cost-safe, never one call
    # per domain -- but it only asked for the affordable subset.
    assert len(REQUESTS) == 1
    prompt = REQUESTS[0]["messages"][0]["content"]
    for intent in expected_included:
        assert f"INTENT {intent.intent_id}" in prompt
    omitted = [i for i in intents if i not in expected_included]
    for intent in omitted:
        assert f"INTENT {intent.intent_id}" not in prompt, (
            "an intent the split/allocate decision excluded must never be sent anyway"
        )

    # The included domains got real, attributed evidence back...
    for intent in expected_included:
        assert intent.intent_id in results
        assert results[intent.intent_id].executed is True
        assert results[intent.intent_id].documents

    # ...and the excluded domains come back with an EXPLICIT, certain
    # SKIPPED_DUE_TO_BUDGET-shaped result -- never absent (absence is
    # reserved for a sent intent whose response never arrived, a genuinely
    # less certain case) and never inferred as a completed, zero-result
    # search.
    for intent in omitted:
        assert intent.intent_id in results
        excluded_result = results[intent.intent_id]
        assert excluded_result.executed is False
        assert excluded_result.documents == []
        assert "BudgetExceeded" in excluded_result.error
        assert "split/allocate" in excluded_result.error

    # The call itself stayed within the stage's real quota -- no post-hoc
    # surprise overshoot of the kind that starved every later stage in the
    # reported production run.
    assert budget.stage_used.get("discovery", 0) <= budget.stage_cap_tokens("discovery")
    assert meta["ambiguous_results"] == 0


def test_production_scale_batch_fits_in_one_call_when_the_stage_budget_is_ample():
    """Positive control: the same six-domain batch, under a normal
    (ample) stage budget, still goes out as ONE call covering all six --
    the split/allocate fix must never fragment a batch that didn't need
    fragmenting."""
    REQUESTS.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EchoIntentsHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        budget = LLMBudget(max_total_tokens=2_000_000)
        budget.set_stage("discovery")
        llm = LLMClient(
            api_key="sk-ant-test",
            base_url=f"http://127.0.0.1:{httpd.server_port}",
            max_retries=0,
            timeout=10,
            budget=budget,
        )
        provider = AnthropicWebResearchProvider(llm)
        intents = _six_domain_intents()

        results, _meta = provider.search_batch(intents, agent_id="adversarial_bear")

        assert len(REQUESTS) == 1
        assert set(results) == {i.intent_id for i in intents}
        for intent in intents:
            assert results[intent.intent_id].executed is True
    finally:
        httpd.shutdown()


def test_never_sent_and_sent_but_unaddressed_are_distinct_within_the_same_call(base_url):
    """The two ways an intent can end up not-executed must never collapse
    into one signal, even when BOTH happen in the same call: an intent this
    method structurally never sent (excluded by split/allocate --
    SKIPPED_DUE_TO_BUDGET, a certain fact) versus an intent it DID send but
    whose response never actually addressed it (INCOMPLETE_RESPONSE, a less
    certain "the model didn't get to it")."""
    global RESPONSE
    REQUESTS.clear()
    # Only reg_1 and sci_1 fit this tight budget (see the affordability
    # assertion below); comp_1 is the one this test's mock response then
    # ALSO fails to address, despite being included in the request --
    # proving that case reads differently from "never sent" even though
    # both are, in the end, not executed.
    RESPONSE = _message(
        [
            {"type": "text", "text": "-- INTENT reg_1 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q1"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"url": "https://www.fda.gov/a", "title": "t"}],
            },
            # sci_1 was included in the request, but the model's response
            # never even mentions it (no marker, no search) -- a genuine
            # sent-but-unaddressed gap.
        ]
    )
    budget = LLMBudget(max_total_tokens=100_000)
    budget.set_stage("discovery")
    llm = LLMClient(api_key="sk-ant-test", base_url=base_url, max_retries=0, timeout=10, budget=budget)
    scoped_provider = AnthropicWebResearchProvider(llm)

    intents = [
        ResearchIntent(intent_id="reg_1", domain=ResearchDomain.REGULATORY, question="q1"),
        ResearchIntent(intent_id="sci_1", domain=ResearchDomain.SCIENCE_TECHNOLOGY, question="q2"),
        ResearchIntent(intent_id="comp_1", domain=ResearchDomain.COMPETITION, question="q3"),
    ]
    included = _affordable_prefix(intents, budget, effort="low", max_uses_per_batch=8)
    assert [i.intent_id for i in included] == ["reg_1", "sci_1"], (
        "this test's own budget must include exactly reg_1 and sci_1, excluding comp_1, for its "
        "assertions to mean what they claim"
    )

    results, _meta = scoped_provider.search_batch(intents, agent_id="adversarial_bear")

    # reg_1: sent, resolved -- ordinary success.
    assert results["reg_1"].executed is True

    # sci_1: SENT (included in the request) but the model's response never
    # addressed it -- absent from `results` entirely; the caller reads that
    # as INCOMPLETE_RESPONSE, never as a certain budget skip.
    assert "sci_1" not in results

    # comp_1: NEVER SENT -- excluded by split/allocate before the request
    # was even built. Present in `results` with an explicit,
    # SKIPPED_DUE_TO_BUDGET-shaped, executed=False result -- never
    # collapsed into the same "absent" signal as sci_1 above.
    assert "comp_1" in results
    assert results["comp_1"].executed is False
    assert "BudgetExceeded" in results["comp_1"].error
    assert "split/allocate" in results["comp_1"].error
