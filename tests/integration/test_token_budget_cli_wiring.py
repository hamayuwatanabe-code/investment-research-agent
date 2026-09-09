"""Regression: verify the proposed next paid-verification command's own
wiring, offline, before it is ever run against the real API.

The previous turn's report proposed `--live --adversarial --research-effort
low --token-budget 40000 --json` as a cheap live-Discovery check. Two
things were wrong with that, both caught here without spending anything:

1. `--live --adversarial` WITHOUT `--llm` never touches
   AnthropicWebResearchProvider at all -- `build_research_stack()` only
   builds it when `args.live and llm is not None`, and `llm` is only ever
   constructed when `--llm` is passed. Omitting `--llm` silently runs
   against `NullResearchProvider` -- a live-API check that in fact makes
   zero requests. `--llm` is required, not optional, for a live Discovery
   check, and this also means the eight interpretive agents WILL run (no
   existing flag decouples Discovery from them).
2. `--token-budget 40000` gives the discovery stage a 12,000-token quota --
   under the ~20,416-token reservation even a single intent needs at low
   effort, so every mandatory query is rejected pre-send. A command whose
   own preflight math guarantees a 100% rejection rate is not a meaningful
   live-API check.

This runs the real `cli.build_parser()`/`cli.run_one()` path against a
local mock Anthropic API (no live network call), proving the CORRECTED
command (`--live --llm --adversarial --research-effort low --llm-effort low
--token-budget 150000`) actually wires the token budget through to the
discovery stage and sends real requests, while the OLD command's premises
(no --llm; --token-budget 40000) are shown to fail exactly as described
above.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research import cli
from investment_research.config import Settings
from investment_research.research.provider import NullResearchProvider
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.integration

REQUESTS: list[dict] = []
CAPTURED_PIPELINE_KWARGS: dict = {}


class _EchoIntentsHandler(BaseHTTPRequestHandler):
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
            content_blocks.append({"type": "text", "text": f"-- INTENT {intent_id} --"})
            content_blocks.append(
                {"type": "server_tool_use", "name": "web_search", "id": f"toolu_{intent_id}", "input": {"query": intent_id}}
            )
            content_blocks.append(
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": f"toolu_{intent_id}",
                    "content": [{"url": f"https://www.fda.gov/{intent_id}", "title": intent_id}],
                }
            )
        payload = json.dumps(
            {
                "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
                "content": content_blocks, "stop_reason": "end_turn",
                "usage": {"input_tokens": 900, "output_tokens": 150},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_anthropic_base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EchoIntentsHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _run(monkeypatch, mock_anthropic_base_url, extra_args: list[str]):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_anthropic_base_url)
    REQUESTS.clear()
    CAPTURED_PIPELINE_KWARGS.clear()

    class _FakePipeline:
        def __init__(self, *args, **kwargs):
            CAPTURED_PIPELINE_KWARGS.update(kwargs)

        def run(self, *args, **kwargs):
            return "STUBBED_RESULT"

    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli, "collect", lambda *a, **k: ([], {}))

    settings = Settings(anthropic_api_key="sk-ant-test")
    args = cli.build_parser().parse_args(["TESTCO", *extra_args])
    repo = Repository(open_db(":memory:"))
    from investment_research.collectors.http import HttpClient

    http = HttpClient(offline=True, cache_dir=None)
    return cli.run_one("TESTCO", args, settings, repo, http)


def test_live_adversarial_without_llm_makes_zero_requests(monkeypatch, mock_anthropic_base_url):
    """The OLD proposal's first flaw: omitting --llm silently no-ops."""
    _run(monkeypatch, mock_anthropic_base_url, ["--live", "--adversarial", "--research-effort", "low"])
    assert REQUESTS == [], "no --llm means no LLMClient, so AnthropicWebResearchProvider is never built"
    research = CAPTURED_PIPELINE_KWARGS.get("research")
    assert isinstance(research, NullResearchProvider), (
        "build_research_stack() must fall back to NullResearchProvider without --llm, even with "
        "--live --adversarial set"
    )


def test_token_budget_40000_rejects_every_mandatory_query_pre_send(monkeypatch, mock_anthropic_base_url):
    """The OLD proposal's second flaw: --token-budget 40000 leaves the
    discovery stage (12,000 tokens) too small to afford even ONE intent, so
    a live run against this command would show a 100% pre-send rejection
    rate -- not a meaningful check of anything."""
    _run(
        monkeypatch, mock_anthropic_base_url,
        ["--live", "--llm", "--adversarial", "--research-effort", "low", "--token-budget", "40000"],
    )
    assert REQUESTS == [], (
        "at --token-budget 40000, the discovery stage cap (12,000) is smaller than even a "
        "single intent's own reservation (~20,416 at low effort) -- nothing should ever be sent"
    )


def test_corrected_token_budget_150000_wires_through_and_sends_real_requests(
    monkeypatch, mock_anthropic_base_url
):
    """The CORRECTED minimal command: --llm present (required), and a
    --token-budget large enough that the discovery stage (45,000 at 30%)
    can actually afford several of the six required-domain intents in its
    first wave -- a live run against this command would exercise the real
    split/allocate path and produce genuine batch diagnostics, not a
    trivial all-rejected or all-fits-in-2M-tokens result."""
    _run(
        monkeypatch, mock_anthropic_base_url,
        [
            "--live", "--llm", "--adversarial",
            "--research-effort", "low", "--llm-effort", "low",
            "--token-budget", "150000",
        ],
    )
    assert REQUESTS, "the corrected command must actually reach the mock API"
    for request in REQUESTS:
        assert request.get("output_config") == {"effort": "low"}, (
            "--research-effort low must still hold at this budget"
        )

    llm = CAPTURED_PIPELINE_KWARGS.get("llm")
    assert llm is not None, "--llm must construct a real LLMClient this time"
    assert llm.budget.max_total_tokens == 150_000, "--token-budget must reach the LLMBudget unchanged"

    adversarial = CAPTURED_PIPELINE_KWARGS.get("adversarial")
    assert adversarial is not None
    assert adversarial.batch_diagnostics, "at least one real discovery batch must have run"
