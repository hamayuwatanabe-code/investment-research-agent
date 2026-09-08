"""Regression: --llm-effort must never leak into web discovery (defect B).

A live run showed two --llm-effort=high adversarial discovery searches alone
consuming 76,891 tokens. Discovery and primary-source fetch/extraction must
default to LOW effort regardless of --llm-effort, and only the explicit
--research-effort flag may raise it. --llm-effort continues to govern the
eight interpretive agents only (Regulatory/Science/Competitive/Contradiction/
Kill/Bear/Bull/Blind Judge), which is unaffected by anything here.

This exercises the real ``cli.build_parser()`` / ``cli.run_one()`` path end to
end against a local mock Anthropic API (no live network call).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research import cli
from investment_research.config import Settings
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.integration

REQUESTS: list[dict] = []


class _RecordingSearchHandler(BaseHTTPRequestHandler):
    """Records every request body; always answers with an empty search result."""

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
def mock_anthropic_base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingSearchHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _run(monkeypatch, mock_anthropic_base_url, extra_args: list[str]):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_anthropic_base_url)
    REQUESTS.clear()

    class _FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return "STUBBED_RESULT"

    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli, "collect", lambda *a, **k: ([], {}))

    settings = Settings(anthropic_api_key="sk-ant-test")
    args = cli.build_parser().parse_args(
        ["TESTCO", "--live", "--llm", "--adversarial", *extra_args]
    )
    repo = Repository(open_db(":memory:"))
    from investment_research.collectors.http import HttpClient

    http = HttpClient(offline=True, cache_dir=None)
    cli.run_one("TESTCO", args, settings, repo, http)


def test_llm_effort_high_does_not_make_discovery_high_by_default(
    monkeypatch, mock_anthropic_base_url
):
    _run(monkeypatch, mock_anthropic_base_url, ["--llm-effort", "high"])

    assert REQUESTS, "the mock API must have received at least one discovery request"
    for request in REQUESTS:
        assert request.get("output_config") == {"effort": "low"}, (
            "--llm-effort high must not leak into web discovery; discovery must stay at "
            "the --research-effort default (low) unless --research-effort is explicit"
        )


def test_research_effort_high_is_explicit_and_overrides_the_default(
    monkeypatch, mock_anthropic_base_url
):
    _run(
        monkeypatch,
        mock_anthropic_base_url,
        ["--llm-effort", "high", "--research-effort", "high"],
    )

    assert REQUESTS
    for request in REQUESTS:
        assert request.get("output_config") == {"effort": "high"}, (
            "an explicit --research-effort high must actually raise discovery effort"
        )
