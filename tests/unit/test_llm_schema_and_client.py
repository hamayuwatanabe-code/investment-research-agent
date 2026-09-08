"""LLM schema validation and client-contract tests (requirement P2)."""

from __future__ import annotations

import pytest

from investment_research.llm.client import (
    DEFAULT_MODEL,
    LLMBudget,
    LLMCallRecord,
    LLMClient,
    LLMUnavailable,
    _first_tool_input,
    _json_from_text,
)
from investment_research.llm.schema import SchemaValidationError, as_strict_tool, validate

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["position", "findings"],
    "properties": {
        "position": {"type": "string", "enum": ["AGREED", "REJECTED", "UNKNOWN"]},
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "count": {"type": "integer", "minimum": 0},
        "flag": {"type": "boolean"},
        "findings": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string", "maxLength": 20}},
            },
        },
    },
}


def test_valid_payload_passes():
    validate({"position": "REJECTED", "score": 9.5, "findings": [{"text": "short"}]}, SCHEMA)


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({"position": "MAYBE", "findings": []}, "not one of"),
        ({"findings": []}, "missing required"),
        ({"position": "AGREED", "findings": [], "extra": 1}, "unexpected properties"),
        ({"position": "AGREED", "score": 11, "findings": []}, "above maximum"),
        ({"position": "AGREED", "score": -1, "findings": []}, "below minimum"),
        ({"position": "AGREED", "count": 1.5, "findings": []}, "expected integer"),
        ({"position": "AGREED", "flag": "yes", "findings": []}, "expected boolean"),
        ({"position": "AGREED", "findings": [{"text": "a"}] * 3}, "exceeds maxItems"),
        ({"position": "AGREED", "findings": [{"text": "x" * 30}]}, "longer than"),
        ({"position": "AGREED", "findings": [{}]}, "missing required"),
        ({"position": "AGREED", "findings": "not-a-list"}, "expected array"),
    ],
)
def test_invalid_payloads_are_rejected(payload, fragment):
    """A malformed model response is a failed run, not a partial result."""
    with pytest.raises(SchemaValidationError, match=fragment):
        validate(payload, SCHEMA)


def test_booleans_are_not_numbers():
    with pytest.raises(SchemaValidationError):
        validate({"position": "AGREED", "score": True, "findings": []}, SCHEMA)


def test_strict_tool_shape():
    tool = as_strict_tool("submit", "desc", SCHEMA)
    assert tool["strict"] is True
    assert tool["name"] == "submit"
    assert tool["input_schema"] is SCHEMA


# --- default model -----------------------------------------------------------
def test_sonnet_5_is_the_normal_default():
    """Verified live: claude-sonnet-5 works. It is now the normal default."""
    assert DEFAULT_MODEL == "claude-sonnet-5"
    assert LLMClient(api_key="sk-ant-not-a-real-key").model == "claude-sonnet-5"


def test_opus_5_remains_explicitly_selectable():
    """Opus 5 is not removed -- it stays available for an explicit red-team run."""
    client = LLMClient(api_key="sk-ant-not-a-real-key", model="claude-opus-5")
    assert client.model == "claude-opus-5"


def test_cli_llm_model_default_is_sonnet_5():
    from investment_research.cli import build_parser

    args = build_parser().parse_args(["ACME"])
    assert args.llm_model == "claude-sonnet-5"


def test_cli_llm_model_can_be_overridden_to_opus_5():
    from investment_research.cli import build_parser

    args = build_parser().parse_args(["ACME", "--llm-model", "claude-opus-5"])
    assert args.llm_model == "claude-opus-5"


# --- client contract --------------------------------------------------------
def test_client_without_credentials_reports_unavailable(monkeypatch):
    """It must not claim readiness and then fail on the first agent."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("pathlib.Path.is_dir", lambda self: False)
    usable, reason = LLMClient().available()
    assert usable is False
    assert "credential" in reason.lower()


def test_client_with_key_reports_available():
    usable, _ = LLMClient(api_key="sk-ant-not-a-real-key").available()
    assert usable is True


def test_calling_an_unavailable_client_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.is_dir", lambda self: False)
    with pytest.raises(LLMUnavailable):
        LLMClient().raw_message(system="s", messages=[{"role": "user", "content": "x"}])


class _Block:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Response:
    def __init__(self, content):
        self.content = content


def test_tool_input_extracted_from_tool_use_block():
    response = _Response([_Block(type="tool_use", name="submit", input={"position": "AGREED"})])
    assert _first_tool_input(response, "submit") == {"position": "AGREED"}


def test_tool_input_ignores_a_different_tool():
    response = _Response([_Block(type="tool_use", name="other", input={"a": 1})])
    assert _first_tool_input(response, "submit") is None


def test_tool_input_parses_a_json_string_payload():
    response = _Response([_Block(type="tool_use", name="submit", input='{"position":"AGREED"}')])
    assert _first_tool_input(response, "submit") == {"position": "AGREED"}


def test_json_fallback_from_a_text_block():
    response = _Response([_Block(type="text", text='Here it is: {"position": "UNKNOWN"} done')])
    assert _json_from_text(response) == {"position": "UNKNOWN"}


def test_json_fallback_returns_none_on_prose():
    response = _Response([_Block(type="text", text="I could not complete this analysis.")])
    assert _json_from_text(response) is None


# --- budget -----------------------------------------------------------------
def test_budget_accounts_per_agent():
    budget = LLMBudget(max_total_tokens=1000)
    budget.record(LLMCallRecord("regulatory", "m", input_tokens=100, output_tokens=50))
    budget.record(LLMCallRecord("regulatory", "m", input_tokens=10, output_tokens=5))
    budget.record(LLMCallRecord("kill_agent", "m", input_tokens=200, output_tokens=20))
    assert budget.used_total == 385
    assert budget.by_agent() == {"regulatory": 165, "kill_agent": 220}
    assert budget.remaining == 615


def test_budget_detects_overrun():
    budget = LLMBudget(max_total_tokens=100)
    budget.record(LLMCallRecord("a", "m", input_tokens=90, output_tokens=0))
    assert budget.would_exceed(50)
    assert not budget.would_exceed(5)
