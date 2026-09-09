"""Production-BUDGET-scale behavioral proof for the discovery-batch fixes.

Every test in this file uses the REAL production defaults --
``LLMBudget(max_total_tokens=200_000)`` with the discovery stage at its real
30% (60,000) share -- never the artificially generous 2,000,000-token budget
used elsewhere to prove the split/allocate MECHANISM in isolation. A test
passing under 2,000,000 tokens says nothing about whether six required
domains can complete under the ACTUAL 200,000/60,000 production ceiling (see
``test_production_budget_capacity.py`` for the capacity arithmetic showing
they structurally cannot, by a wide margin) -- this file is the behavioral
counterpart: given that shortfall is real, prove the SYSTEM behaves
correctly around it (accurate status, no double billing, no silent
drops, no lost report), not that the shortfall goes away.

No live API call is made anywhere in this file; a local HTTP server stands
in for the real Anthropic Messages API, same as the rest of this test suite.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.cli import _token_diagnostics
from investment_research.llm.client import LLMBudget, LLMClient
from investment_research.orchestrator.isolation import EvidenceBus
from investment_research.orchestrator.pipeline import ResearchResult
from investment_research.reporting.report import ReportRenderer, _section_cost
from investment_research.research.adversarial import (
    AdversarialOutcome,
    _dedupe_semantically,
    _prioritize_domain_coverage,
    build_plan,
)
from investment_research.research.anthropic_web import (
    AnthropicWebResearchProvider,
    _affordable_prefix,
)
from investment_research.research.batching import (
    ResearchIntent,
    build_research_batches,
    execute_research_batch,
)
from investment_research.research.discovery import DiscoveryLog
from investment_research.research.provider import ResearchQuery
from investment_research.schemas.agent_io import RunContext
from investment_research.schemas.enums import IntentStatus, ResearchDomain, RunStatus

pytestmark = pytest.mark.regression

PRODUCTION_TOTAL_BUDGET = 200_000
TICKER, COMPANY = "TESTCO", "Generic Biotech Holdings"

REQUESTS: list[dict] = []
RESPONSE: dict = {}


def _first_real_bear_batch():
    """The actual first-wave bear batch a production run builds -- six real
    required-domain intents with real question text, not synthetic
    placeholders, so the reservation numbers below match what production
    would really compute."""
    plan = build_plan(TICKER, COMPANY)
    bear_q = _prioritize_domain_coverage(plan.bear)
    bear_q, _ = _dedupe_semantically(bear_q)
    intents = [
        ResearchIntent(intent_id=f"bear_{i}", domain=q.domain, question=q.query, priority=i)
        for i, q in enumerate(bear_q)
    ]
    return build_research_batches(intents)[0]


def _production_budget() -> LLMBudget:
    budget = LLMBudget(max_total_tokens=PRODUCTION_TOTAL_BUDGET)
    budget.set_stage("discovery")
    return budget


class _EchoIntentsHandler(BaseHTTPRequestHandler):
    """Answers with a real, ATTRIBUTED marker+search+result triple for
    every INTENT id found in the request prompt -- whatever subset the
    split/allocate decision actually sent."""

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


class _FixedUsageHandler(BaseHTTPRequestHandler):
    """Always answers successfully but reports a FIXED `usage` regardless of
    the request -- used to simulate actual usage overshooting what a
    conservative preflight reservation assumed."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload = json.dumps(RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def echo_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EchoIntentsHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def fixed_usage_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FixedUsageHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


# === 1: never-sent vs. sent-but-response-omitted, at production scale ======
def test_production_scale_never_sent_intent_is_skipped_due_to_budget_not_incomplete(echo_url):
    REQUESTS.clear()
    budget = _production_budget()
    llm = LLMClient(api_key="sk-ant-test", base_url=echo_url, max_retries=0, timeout=10, budget=budget)
    provider = AnthropicWebResearchProvider(llm)

    batch = _first_real_bear_batch()
    assert len(batch.intents) == 6, "this test's own premise (a full first-wave batch) must hold"
    included = _affordable_prefix(batch.intents, budget, effort="low", max_uses_per_batch=8)
    assert 0 < len(included) < 6, (
        "under REAL 200,000/60,000 production defaults, this realistic six-domain first batch "
        "must NOT all fit in one call -- if it does, the capacity premise this test relies on "
        "has changed and this test needs revisiting"
    )

    results, _meta = provider.search_batch(batch.intents, agent_id="adversarial_bear")
    excluded = [i for i in batch.intents if i not in included]
    for intent in excluded:
        assert intent.intent_id in results, "a never-sent intent must be an EXPLICIT result, never absent"
        assert results[intent.intent_id].executed is False
        assert "split/allocate" in results[intent.intent_id].error


# === 2/3: partial send, remaining-budget retry, no duplicate billing =======
def test_production_scale_unsent_intents_are_retried_and_never_double_billed(echo_url):
    REQUESTS.clear()
    budget = _production_budget()
    llm = LLMClient(api_key="sk-ant-test", base_url=echo_url, max_retries=0, timeout=10, budget=budget)
    provider = AnthropicWebResearchProvider(llm)

    batch = _first_real_bear_batch()
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker=TICKER,
        llm_agent_id="adversarial_bear",
    )

    # Round 1 could only afford part of the batch (production scale); the
    # echo server always answers whatever it's sent, so round 2's
    # re-offered intent(s) succeed once round 1's actual usage frees room.
    assert len(REQUESTS) >= 2, "a genuine retry round must have happened at production scale"
    for intent in batch.intents:
        assert intent.status is IntentStatus.EXECUTED_WITH_EVIDENCE, (
            f"{intent.intent_id} should have resolved across the retry rounds"
        )
    assert diagnostics.retry_rounds >= 1

    # No duplicate billing: every intent recorded into discovery exactly
    # once, whichever round actually sent it.
    assert len(discovery.queries) == len(batch.intents)
    assert len({q.query_id for q in discovery.queries}) == len(batch.intents)
    # And no provider call ever re-sent an intent a previous round already
    # resolved.
    seen_across_calls: set[str] = set()
    for request in REQUESTS:
        prompt = request["messages"][0]["content"]
        import re

        ids_this_call = set(re.findall(r"INTENT (\S+) \(domain:", prompt))
        assert not (ids_this_call & seen_across_calls), (
            "an intent already sent in an earlier call must never be re-sent"
        )
        seen_across_calls |= ids_this_call


# === 4: when budget is genuinely insufficient, every incomplete item and ===
# === its reason survive, with priority intact ===============================
def test_production_scale_exhaustion_leaves_every_unresolved_item_and_reason_visible(echo_url):
    """A batch too large even for retries to fully clear (simulate by
    capping retries at 0) must leave every unresolved intent's own status,
    detail, and priority readable -- never silently dropped."""
    REQUESTS.clear()
    budget = _production_budget()
    llm = LLMClient(api_key="sk-ant-test", base_url=echo_url, max_retries=0, timeout=10, budget=budget)
    provider = AnthropicWebResearchProvider(llm)

    batch = _first_real_bear_batch()
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker=TICKER,
        llm_agent_id="adversarial_bear", max_retry_rounds=0,
    )
    skipped = [i for i in batch.intents if i.status is IntentStatus.SKIPPED_DUE_TO_BUDGET]
    assert skipped, "with retries capped at 0, at least the excluded intent(s) must end up skipped"
    for intent in skipped:
        assert intent.detail, "the specific reason must survive on the intent, never blank"
        assert "split/allocate" in intent.detail
        assert intent.priority is not None  # original priority never reset/lost
        assert intent.intent_id in diagnostics.incomplete_intent_ids


# === 5: actual usage overshooting the reservation stops later calls, ======
# === including a later stage sharing the same budget =======================
def test_actual_overshoot_beyond_reservation_blocks_a_later_stage(fixed_usage_url):
    """A server-side tool's real cost is not known until the response comes
    back (see BudgetExceeded's own docstring) -- when it comes in far above
    what was reserved and pushes the run over its GLOBAL ceiling, every
    later call, in ANY stage (not just the one that overshot), must be
    refused -- exactly the mechanism that turned Kill's mandatory queries
    into PROVIDER_UNAVAILABLE in the original production report."""
    global RESPONSE
    RESPONSE = {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
        "content": [
            {"type": "text", "text": "-- INTENT bear_0 --"},
            {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
        ],
        "stop_reason": "end_turn",
        # Reported actual usage far exceeds ANY reasonable preflight
        # reservation for a single intent, and alone exceeds the entire
        # production global budget.
        "usage": {"input_tokens": 190_000, "output_tokens": 20_000},
    }
    budget = LLMBudget(max_total_tokens=PRODUCTION_TOTAL_BUDGET)
    budget.set_stage("discovery")
    llm = LLMClient(api_key="sk-ant-test", base_url=fixed_usage_url, max_retries=0, timeout=10, budget=budget)
    provider = AnthropicWebResearchProvider(llm)

    one_intent = [ResearchIntent(intent_id="bear_0", domain=ResearchDomain.REGULATORY, question="q")]
    results, _meta = provider.search_batch(one_intent, agent_id="adversarial_bear")
    assert results["bear_0"].executed is True  # the call itself succeeded

    assert budget.used_total > PRODUCTION_TOTAL_BUDGET
    assert budget.exhausted is True, "actual usage overshooting the global ceiling must latch exhaustion"

    # A LATER call under a DIFFERENT stage (e.g. Kill Agent's "discovery"
    # stage reuse, or an interpretive-stage call) must now be refused too --
    # global exhaustion is not scoped to the stage that caused it.
    budget.set_stage("interpretive")
    usable, reason = provider.available()
    assert usable is False
    assert reason  # the actual cause (which agent, used/remaining counts) is preserved
    query = ResearchQuery(query="a later, unrelated query", domain=ResearchDomain.REGULATORY)
    later_result = provider.search(query, agent_id="kill_agent")
    assert later_result.executed is False
    assert "exhausted" in later_result.error.lower() or "budget" in later_result.error.lower()


# === 6: batch/report diagnostics survive from execute_research_batch all ===
# === the way to the rendered report, at production scale ===================
def test_production_scale_diagnostics_reach_both_json_and_rendered_report(echo_url):
    REQUESTS.clear()
    budget = _production_budget()
    llm = LLMClient(api_key="sk-ant-test", base_url=echo_url, max_retries=0, timeout=10, budget=budget)
    provider = AnthropicWebResearchProvider(llm)

    batch = _first_real_bear_batch()
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker=TICKER,
        llm_agent_id="adversarial_bear",
    )
    outcome = AdversarialOutcome(batch_diagnostics=[diagnostics])

    # (a) the --json export path.
    fake_result = type("_R", (), {"llm_budget": budget, "adversarial": outcome, "escalation": None})()
    diag_dict = _token_diagnostics(fake_result)
    assert diag_dict["batches"][0]["batch_id"] == diagnostics.batch_id
    assert diag_dict["batches"][0]["retry_rounds"] == diagnostics.retry_rounds
    assert diag_dict["batches"][0]["completed_intent_ids"] == diagnostics.completed_intent_ids

    # (b) the actual rendered production report text -- not just the JSON
    # export -- must carry the same batch id, tool uses, and retry rounds.
    context = RunContext(run_id="r1", ticker=TICKER, company_name=COMPANY, status=RunStatus.COMPLETE)
    result = ResearchResult(
        context=context, bus=EvidenceBus(), verdict=None, scorecard=None,
        llm_budget=budget, adversarial=outcome,
    )
    section = _section_cost(ReportRenderer(result))
    assert diagnostics.batch_id in section
    assert f"tool uses         : {diagnostics.server_tool_uses}" in section
    if diagnostics.retry_rounds:
        assert f"retry rounds      : {diagnostics.retry_rounds}" in section
