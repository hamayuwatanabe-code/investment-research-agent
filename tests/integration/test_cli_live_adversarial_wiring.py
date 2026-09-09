"""Regression: `--live --llm --adversarial` (no `--corpus`) must actually search.

DEFECT 1: ``cli.run_one()`` called ``run_adversarial_search(...)`` only inside
the ``if args.corpus:`` branch. So ``python main.py TICKER --live --llm
--adversarial`` built an ``AnthropicWebResearchProvider`` and then never used
it for the adversarial Bear/Bull discovery passes at all -- the exact
production flag combination the task describes.

This test exercises the real ``cli.build_parser()`` / ``cli.run_one()`` code
path end to end. No live network call is made: the Anthropic SDK client is
pointed at a local mock HTTP server via ``ANTHROPIC_BASE_URL`` (the same
mechanism ``ANTHROPIC_BASE_URL`` / ``base_url=`` uses in production, just
aimed at localhost), and the SEC/ClinicalTrials/FDA collector call is stubbed
out since this test is only about the research/adversarial wiring.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research import cli
from investment_research.config import Settings
from investment_research.research.adversarial import (
    BEAR_TEMPLATES,
    BULL_TEMPLATES,
    run_adversarial_search,
)
from investment_research.research.anthropic_web import AnthropicWebResearchProvider
from investment_research.schemas.enums import QueryPurpose
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.integration


class _EmptySearchHandler(BaseHTTPRequestHandler):
    """Every POST gets back an empty, successful web_search_tool_result.

    Empty results mean the follow-up-query generation step (which only calls
    the model again if it already has evidence to reason about) short-circuits
    without needing a second canned response shape -- this test only needs to
    prove the bear/bull passes themselves execute against the live provider.

    A batched request (see research/batching.py + anthropic_web.py's
    ``search_batch``) lists each intent as ``INTENT <id> (domain: ...)`` in
    the prompt; a real model replies with a ``-- INTENT <id> --`` marker
    before searching for it. This mock plays that same part -- one marker
    text block plus an empty result block per intent id found in the
    request -- so a batched call still gets a real (empty but ATTRIBUTED)
    result for every intent, exactly as a compliant live model would.
    """

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
        prompt_text = ""
        for message in request.get("messages", []):
            content = message.get("content", "")
            prompt_text += content if isinstance(content, str) else json.dumps(content)
        intent_ids = _re.findall(r"INTENT (\S+) \(domain:", prompt_text)

        if intent_ids:
            content_blocks = []
            for intent_id in intent_ids:
                tool_use_id = f"toolu_{intent_id}"
                content_blocks.append({"type": "text", "text": f"-- INTENT {intent_id} --"})
                content_blocks.append(
                    {"type": "server_tool_use", "name": "web_search", "id": tool_use_id, "input": {"query": intent_id}}
                )
                content_blocks.append(
                    {"type": "web_search_tool_result", "tool_use_id": tool_use_id, "content": []}
                )
        else:
            content_blocks = [
                {"type": "server_tool_use", "name": "web_search", "id": "toolu_1", "input": {"query": "q"}},
                {"type": "web_search_tool_result", "tool_use_id": "toolu_1", "content": []},
            ]

        payload = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": content_blocks,
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
def mock_anthropic_base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EmptySearchHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_live_non_corpus_adversarial_runs_bear_then_bull_against_live_provider(
    monkeypatch, mock_anthropic_base_url
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_anthropic_base_url)

    calls: list = []
    outcomes: list = []

    def spy_run_adversarial_search(provider, plan, **kwargs):
        calls.append(provider)
        outcome = run_adversarial_search(provider, plan, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(cli, "run_adversarial_search", spy_run_adversarial_search)

    captured_pipeline_kwargs: dict = {}

    class _FakePipeline:
        def __init__(self, *args, **kwargs):
            captured_pipeline_kwargs.update(kwargs)

        def run(self, *args, **kwargs):
            return "STUBBED_RESULT"

    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    # This test is only about research/adversarial wiring, not the SEC/
    # ClinicalTrials/FDA collectors -- stub them out rather than let a --live
    # run reach those hosts.
    monkeypatch.setattr(cli, "collect", lambda *a, **k: ([], {}))

    settings = Settings(anthropic_api_key="sk-ant-test")
    args = cli.build_parser().parse_args(["TESTCO", "--live", "--llm", "--adversarial"])
    repo = Repository(open_db(":memory:"))
    from investment_research.collectors.http import HttpClient

    http = HttpClient(offline=True, cache_dir=None)

    result = cli.run_one("TESTCO", args, settings, repo, http)

    assert result == "STUBBED_RESULT"
    # Ran exactly once -- never once for a corpus pass and again for a live
    # pass when both happen to be configured.
    assert len(calls) == 1
    assert isinstance(calls[0], AnthropicWebResearchProvider), (
        "the live Anthropic research provider must be what adversarial search runs against"
    )

    outcome = outcomes[0]
    total_mandatory_queries = len(BEAR_TEMPLATES) + len(BULL_TEMPLATES)
    assert len(outcome.discovery.queries) == total_mandatory_queries
    assert outcome.executed == total_mandatory_queries, "every mandatory query actually executed"

    # Bear-first, bull-separated: every bear query precedes every bull query,
    # and purposes are never mixed within a stance.
    purposes = [q.query_purpose for q in outcome.discovery.queries]
    bear_count = len(BEAR_TEMPLATES)
    assert all(p is QueryPurpose.BEAR for p in purposes[:bear_count])
    assert all(p is QueryPurpose.BULL for p in purposes[bear_count:])

    # The wired-through result is the same outcome object, not dropped.
    assert captured_pipeline_kwargs["adversarial"] is outcome
    assert captured_pipeline_kwargs["research"] is calls[0]


def test_corpus_and_live_together_still_runs_adversarial_exactly_once(
    monkeypatch, mock_anthropic_base_url, tmp_path
):
    """Both --corpus and --live configured must not double-run adversarial search.

    No captured corpus file exists for this ticker, so the composite provider's
    corpus leg reports unavailable and every query falls through to the live
    leg -- exactly the real behavior of ``CompositeResearchProvider``, exercised
    without needing a real captured-corpus fixture on disk.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_anthropic_base_url)

    calls: list = []

    def spy_run_adversarial_search(provider, plan, **kwargs):
        calls.append(provider)
        return run_adversarial_search(provider, plan, **kwargs)

    monkeypatch.setattr(cli, "run_adversarial_search", spy_run_adversarial_search)

    class _FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return "STUBBED_RESULT"

    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli, "collect", lambda *a, **k: ([], {}))

    settings = Settings(anthropic_api_key="sk-ant-test")
    settings.corpus_dir = tmp_path / "empty-corpus-dir"  # deliberately has no TESTCO.json
    args = cli.build_parser().parse_args(
        ["TESTCO", "--live", "--corpus", "--llm", "--adversarial"]
    )
    repo = Repository(open_db(":memory:"))
    from investment_research.collectors.http import HttpClient

    http = HttpClient(offline=True, cache_dir=None)

    cli.run_one("TESTCO", args, settings, repo, http)

    assert len(calls) == 1, "adversarial search must run exactly once, not once per provider type"
