"""LLM agents against a mock Anthropic API speaking the real wire protocol.

This is how the model-backed path is verified without credentials: a local HTTP
server implements ``POST /v1/messages`` and returns real-shaped responses, and
the actual Anthropic SDK talks to it over real sockets. So the request shape,
the tool-use round trip, schema validation, refusal handling, server-tool result
parsing and the degradation contract are all genuinely exercised.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.agents.llm_agents import LLMRegulatoryAgent
from investment_research.agents.llm_agents2 import LLMBearAgent, LLMBullAgent, LLMKillAgent
from investment_research.agents.llm_base import PromptGuard
from investment_research.agents.regulatory import RegulatoryAgent
from investment_research.collectors.documents import Document, chunk_document
from investment_research.llm.client import BudgetExceeded, LLMBudget, LLMClient
from investment_research.orchestrator.isolation import Channel, LeakageError
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import ContentKind, FactCategory, SourceTier
from tests.conftest import make_fact

pytestmark = pytest.mark.integration

#: What the mock returns next. Each test sets this.
RESPONSE: dict = {}
REQUESTS: list[dict] = []


class Handler(BaseHTTPRequestHandler):
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
        status = int(RESPONSE.get("_http_status", 200))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def client(base_url):
    REQUESTS.clear()
    return LLMClient(api_key="sk-ant-test", base_url=base_url, max_retries=0, timeout=10)


def message(content, stop_reason="end_turn", **extra):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 1200, "output_tokens": 300},
        **extra,
    }


def tool_use(name, payload):
    return [{"type": "tool_use", "id": "tu_1", "name": name, "input": payload}]


REGULATORY_PAYLOAD = {
    "endpoint_position": "REJECTED",
    "endpoint_position_reasoning": "The regulator said the endpoint cannot demonstrate efficacy.",
    "agreed": [],
    "not_agreed": [
        {
            "statement": "The regulator advised the primary endpoint is not sufficient to "
            "demonstrate efficacy.",
            "evidence_ids": ["FACT_ID"],
            "severity": "CRITICAL",
        }
    ],
    "unresolved": [],
    "company_framing": [
        {
            "statement": "The company described the meeting as constructive.",
            "evidence_ids": ["FACT_ID"],
            "severity": "HIGH",
        }
    ],
    "procedural_designations": ["Fast Track", "Orphan Drug"],
    "registrational_status_changed": True,
}


def make_input(agent_id="regulatory", ticker="TESTCO", channels=None, params=None):
    fact = make_fact(
        "The regulator advised the primary endpoint is not sufficient to demonstrate efficacy",
        ticker=ticker,
        category=FactCategory.REGULATORY,
    )
    return fact, AgentInput(
        agent_id=agent_id,
        run_id="r1",
        ticker=ticker,
        company_name="Test Company",
        facts=(fact,),
        channels=channels or {},
        params=params or {},
    )


# --- request shape ----------------------------------------------------------
def test_request_uses_current_api_shape(client):
    fact, data = make_input()
    global RESPONSE
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD).replace("FACT_ID", fact.fact_id))
    RESPONSE = message(tool_use("submit_regulatory_analysis", payload))

    LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)

    request = REQUESTS[-1]
    assert request["model"] == "claude-sonnet-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}
    assert request["tools"][0]["strict"] is True
    assert request["tools"][0]["name"] == "submit_regulatory_analysis"
    # Forced tool_choice is rejected on current models; auto is portable.
    assert request["tool_choice"] == {"type": "auto"}


# --- per-agent effort policy (requirement G) ---------------------------------
def test_per_agent_effort_override_reaches_the_actual_api_request(client):
    """Requirement G: a per-agent effort resolved by the caller (e.g. via
    llm.effort_policy.resolve_effort against --llm-effort's ceiling) must
    actually change the API request's output_config.effort -- not merely be
    recorded somewhere and ignored."""
    fact, data = make_input()
    global RESPONSE
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD).replace("FACT_ID", fact.fact_id))
    RESPONSE = message(tool_use("submit_regulatory_analysis", payload))

    output = LLMRegulatoryAgent(
        client, fallback=RegulatoryAgent(), effort="medium"
    ).run(data)

    request = REQUESTS[-1]
    assert request["output_config"] == {"effort": "medium"}
    assert output.metrics["llm_effort"] == "medium"
    assert "preflight_reservation" in output.metrics
    assert output.metrics["prompt_tokens"] > 0
    assert output.metrics["actual_total_tokens"] > 0


def test_no_effort_override_uses_the_client_default(client):
    fact, data = make_input()
    global RESPONSE
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD).replace("FACT_ID", fact.fact_id))
    RESPONSE = message(tool_use("submit_regulatory_analysis", payload))

    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)

    assert REQUESTS[-1]["output_config"] == {"effort": "high"}
    assert output.metrics["llm_effort"] == "high"


def test_evidence_reaches_the_prompt(client):
    fact, data = make_input()
    global RESPONSE
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD).replace("FACT_ID", fact.fact_id))
    RESPONSE = message(tool_use("submit_regulatory_analysis", payload))
    LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)
    prompt = REQUESTS[-1]["messages"][0]["content"]
    assert fact.fact_id in prompt
    assert "not sufficient to demonstrate efficacy" in prompt


# --- interpretation ---------------------------------------------------------
def test_regulatory_agent_interprets_a_valid_response(client):
    fact, data = make_input()
    global RESPONSE
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD).replace("FACT_ID", fact.fact_id))
    RESPONSE = message(tool_use("submit_regulatory_analysis", payload))

    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)

    assert output.metrics["llm_backed"] is True
    assert output.evaluation.payload["endpoint_position"] == "REJECTED"
    assert output.evaluation.payload["registrational_status_changed"] is True
    assert any("no longer described as pivotal" in f.title for f in output.risk_flags)


def test_uncited_claims_are_dropped_not_kept(client):
    """A model claim citing evidence that is not in its pack does not survive."""
    fact, data = make_input()
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD))
    payload["not_agreed"][0]["evidence_ids"] = ["fact_does_not_exist"]
    payload["company_framing"][0]["evidence_ids"] = [fact.fact_id]
    global RESPONSE
    RESPONSE = message(tool_use("submit_regulatory_analysis", payload))

    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)
    assert output.evaluation.payload["not_agreed"] == []
    assert output.metrics["dropped_uncited"] == 1


# --- degradation contract ---------------------------------------------------
def test_schema_violation_falls_back_to_the_deterministic_agent(client):
    fact, data = make_input()
    global RESPONSE
    RESPONSE = message(
        tool_use("submit_regulatory_analysis", {"endpoint_position": "MAYBE", "agreed": []})
    )
    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)

    assert output.metrics["llm_backed"] is False
    assert output.degraded
    assert any("SchemaValidationError" in e for e in output.errors)
    # The deterministic agent still produced a real analysis.
    assert output.evaluation.payload["endpoint_position"] == "REJECTED"


def test_refusal_falls_back_and_is_reported(client):
    fact, data = make_input()
    global RESPONSE
    RESPONSE = message(
        [{"type": "text", "text": "I cannot help with that."}],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "other", "explanation": "declined"},
    )
    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)
    assert output.metrics["llm_backed"] is False
    assert any("declined" in e or "LLMUnavailable" in e for e in output.errors)


def test_prose_instead_of_a_tool_call_falls_back(client):
    fact, data = make_input()
    global RESPONSE
    RESPONSE = message([{"type": "text", "text": "Sorry, I need more information."}])
    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)
    assert output.metrics["llm_backed"] is False


def test_json_in_a_text_block_is_accepted_and_validated(client):
    fact, data = make_input()
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD).replace("FACT_ID", fact.fact_id))
    global RESPONSE
    RESPONSE = message([{"type": "text", "text": json.dumps(payload)}])
    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)
    assert output.metrics["llm_backed"] is True


def test_http_error_falls_back(client):
    fact, data = make_input()
    global RESPONSE
    RESPONSE = {"_http_status": 500, "type": "error", "error": {"message": "boom"}}
    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)
    assert output.metrics["llm_backed"] is False
    assert output.degraded


def test_budget_exhaustion_falls_back_without_calling(client):
    fact, data = make_input()
    client.budget.max_total_tokens = 1
    client.budget.used_input = 1
    REQUESTS.clear()
    output = LLMRegulatoryAgent(client, fallback=RegulatoryAgent()).run(data)
    assert output.metrics["llm_backed"] is False
    assert "budget" in output.metrics["llm_fallback_reason"]
    assert REQUESTS == [], "no request should be made once the budget is exhausted"


# --- isolation, at the prompt --------------------------------------------
def test_a_prompt_carrying_denied_channel_prose_is_never_sent(client):
    """Prompt-level isolation: the request must not leave the process."""
    bull_prose = "the optionality embedded in the second indication is unpriced"
    fact, data = make_input(
        agent_id="bear_agent",
        params={},
    )
    data = AgentInput(
        agent_id="bear_agent",
        run_id="r1",
        ticker="TESTCO",
        company_name="Test Company",
        facts=(make_fact(bull_prose, category=FactCategory.OTHER),),
    )
    guard = PromptGuard(denied_fingerprints={Channel.BULL: ("optionality embedded in the",)})
    REQUESTS.clear()
    from investment_research.agents.bear_agent import BearAgent

    with pytest.raises(LeakageError):
        LLMBearAgent(client, fallback=BearAgent(), guard=guard).run(data)
    assert REQUESTS == [], "a leaking prompt must never be transmitted"


def test_blind_agent_prompt_rejects_identity(client):
    from investment_research.agents.blind_judge import BlindJudgeAgent
    from investment_research.agents.llm_agents2 import LLMBlindJudgeAgent

    data = AgentInput(
        agent_id="blind_judge",
        run_id="r1",
        ticker=None,
        company_name=None,
        facts=(make_fact("Longeveron reported a net loss", ticker="Company X"),),
        params={"evidence_confidence": 3.0, "run_status": "COMPLETE"},
    )
    guard = PromptGuard(forbidden_identity=("Longeveron", "LGVN"))
    REQUESTS.clear()
    with pytest.raises(LeakageError):
        LLMBlindJudgeAgent(client, fallback=BlindJudgeAgent(), guard=guard).run(data)
    assert REQUESTS == []


# --- kill agent proposes, gate disposes -------------------------------------
def test_kill_agent_output_is_proposal_only(client):
    from investment_research.agents.kill_agent import KillAgent
    from investment_research.collectors.search import NullSearchProvider

    fact, data = make_input(agent_id="kill_agent")
    global RESPONSE
    RESPONSE = message(
        tool_use(
            "submit_kill_findings",
            {
                "findings": [
                    {
                        "category": "REGULATORY_KILL",
                        "level": "K5",
                        "title": "Endpoint rejected",
                        "detail": "The regulator rejected the endpoint.",
                        "evidence_ids": [fact.fact_id],
                        "primary_source": True,
                    }
                ],
                "follow_up_queries": ["TESTCO 2024 meeting pivotal designation"],
            },
        )
    )
    output = LLMKillAgent(client, fallback=KillAgent(NullSearchProvider())).run(data)
    payload = output.evaluation.payload
    assert "llm_proposed_findings" in payload
    assert "kill_gate" not in payload, "the model must not emit a binding gate result"
    assert payload["follow_up_queries"] == ["TESTCO 2024 meeting pivotal designation"]


def test_bull_agent_reports_no_case_rather_than_inventing_one(client):
    from investment_research.agents.bull_agent import BullAgent

    fact, data = make_input(agent_id="bull_agent")
    global RESPONSE
    RESPONSE = message(
        tool_use(
            "submit_bull_case",
            {"points": [], "no_case_reason": "No evidence supports undervaluation."},
        )
    )
    output = LLMBullAgent(client, fallback=BullAgent()).run(data)
    assert output.evaluation.payload["points"] == []
    assert "No evidence supports undervaluation." in output.evaluation.summary
    assert output.unresolved


# --- evidence packs ---------------------------------------------------------
def test_agent_reads_only_its_relevant_chunks(client):
    document = Document(
        doc_id="d1",
        url="https://www.sec.gov/x",
        title="10-Q",
        text=(
            "The regulator advised the primary endpoint is not sufficient to demonstrate "
            "efficacy. " + "Unrelated corporate boilerplate about office leases. " * 60
        ),
        content_kind=ContentKind.FULL_DOCUMENT,
        tier=SourceTier.TIER_1,
    )
    chunks = chunk_document(document, target_tokens=60)
    fact, data = make_input()
    payload = json.loads(json.dumps(REGULATORY_PAYLOAD).replace("FACT_ID", chunks[0].chunk_id))
    global RESPONSE
    RESPONSE = message(tool_use("submit_regulatory_analysis", payload))

    agent = LLMRegulatoryAgent(client, fallback=RegulatoryAgent(), chunks=chunks)
    output = agent.run(data)

    assert output.metrics["llm_backed"] is True
    assert output.metrics["pack_chunks"] < len(chunks), "the agent must not read every chunk"
    prompt = REQUESTS[-1]["messages"][0]["content"]
    assert "not sufficient to demonstrate" in prompt


# --- budget enforcement (cost-control fix) ----------------------------------
def test_budget_preflight_blocks_the_call_before_any_request(base_url):
    """A planned request that cannot fit the remaining budget must never reach the API."""
    global RESPONSE
    RESPONSE = message([{"type": "text", "text": "ok"}])
    REQUESTS.clear()
    budget = LLMBudget(max_total_tokens=100)
    tiny_client = LLMClient(
        api_key="sk-ant-test", base_url=base_url, max_retries=0, timeout=10, budget=budget
    )

    with pytest.raises(BudgetExceeded):
        tiny_client.raw_message(
            system="s" * 1000,
            messages=[{"role": "user", "content": "x"}],
            max_tokens=16000,
        )

    assert REQUESTS == [], "a preflight-rejected call must never reach the Anthropic API"
    assert budget.exhausted is True


def test_exhausted_budget_blocks_every_subsequent_call(base_url):
    """Once exhausted, even a call that would otherwise easily fit is refused."""
    global RESPONSE
    RESPONSE = message([{"type": "text", "text": "ok"}])
    REQUESTS.clear()
    budget = LLMBudget(max_total_tokens=1_000_000)
    budget.exhausted = True
    budget.exhausted_reason = "test: pre-exhausted"
    tiny_client = LLMClient(
        api_key="sk-ant-test", base_url=base_url, max_retries=0, timeout=10, budget=budget
    )

    usable, reason = tiny_client.available()
    assert usable is False
    assert "exhausted" in reason.lower()

    with pytest.raises(BudgetExceeded):
        tiny_client.raw_message(
            system="tiny", messages=[{"role": "user", "content": "x"}], max_tokens=10
        )
    assert REQUESTS == [], "no call may reach the API once the budget is exhausted"


def test_actual_usage_crossing_the_budget_is_recorded_but_blocks_the_next_call(base_url):
    """A single call's real usage can overshoot; that must latch exhaustion for the next one."""
    global RESPONSE
    REQUESTS.clear()
    budget = LLMBudget(max_total_tokens=1000)
    tiny_client = LLMClient(
        api_key="sk-ant-test", base_url=base_url, max_retries=0, timeout=10, budget=budget
    )
    # The preflight reservation for this small prompt comfortably fits -- the
    # overshoot below can only be detected after the fact, from real usage.
    RESPONSE = message(
        [{"type": "text", "text": "ok"}],
        usage={"input_tokens": 800, "output_tokens": 500},
    )

    response = tiny_client.raw_message(
        system="s", messages=[{"role": "user", "content": "x"}], max_tokens=50
    )

    assert response is not None
    assert budget.used_total == 1300
    assert budget.exhausted is True, "actual usage over the ceiling must latch exhaustion"

    REQUESTS.clear()
    with pytest.raises(BudgetExceeded):
        tiny_client.raw_message(
            system="s", messages=[{"role": "user", "content": "x"}], max_tokens=1
        )
    assert REQUESTS == [], "no further call may reach the API once exhausted"
