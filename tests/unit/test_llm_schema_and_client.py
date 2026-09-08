"""LLM schema validation and client-contract tests (requirement P2)."""

from __future__ import annotations

import pytest

from investment_research.llm.client import (
    DEFAULT_MODEL,
    DEFAULT_STAGE_QUOTAS,
    SERVER_TOOL_TOKEN_RESERVE,
    SERVER_TOOL_TOKEN_RESERVE_LOW_EFFORT,
    BudgetExceeded,
    LLMBudget,
    LLMCallRecord,
    LLMClient,
    LLMUnavailable,
    _first_tool_input,
    _json_from_text,
)
from investment_research.llm.schema import (
    ANTHROPIC_UNSUPPORTED_KEYWORDS,
    SchemaValidationError,
    as_strict_tool,
    to_anthropic_schema,
    validate,
)

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
    # The wire schema is a sanitized COPY -- never the original object -- so
    # the caller's schema is never mutated by what gets sent to Anthropic.
    assert tool["input_schema"] is not SCHEMA
    assert "minimum" not in tool["input_schema"]["properties"]["score"]


# --- Anthropic schema sanitization (requirement A) --------------------------
def test_to_anthropic_schema_strips_min_max_items_and_bounds_recursively():
    cleaned = to_anthropic_schema(SCHEMA)

    assert "minimum" not in cleaned["properties"]["score"]
    assert "maximum" not in cleaned["properties"]["score"]
    assert "minimum" not in cleaned["properties"]["count"]
    assert "maxItems" not in cleaned["properties"]["findings"]
    # Nested object/array structure survives the strip.
    assert cleaned["properties"]["findings"]["type"] == "array"
    assert cleaned["properties"]["findings"]["items"]["type"] == "object"
    assert cleaned["properties"]["findings"]["items"]["required"] == ["text"]


def test_to_anthropic_schema_preserves_type_enum_required_properties():
    cleaned = to_anthropic_schema(SCHEMA)
    assert cleaned["type"] == "object"
    assert cleaned["required"] == ["position", "findings"]
    assert cleaned["additionalProperties"] is False
    assert cleaned["properties"]["position"]["enum"] == ["AGREED", "REJECTED", "UNKNOWN"]
    assert cleaned["properties"]["count"]["type"] == "integer"
    assert cleaned["properties"]["findings"]["items"]["properties"]["text"]["maxLength"] == 20


def test_to_anthropic_schema_leaves_no_unsupported_keyword_anywhere():
    def _walk(node):
        if isinstance(node, dict):
            for key in node:
                assert key not in ANTHROPIC_UNSUPPORTED_KEYWORDS, f"{key!r} leaked through"
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(to_anthropic_schema(SCHEMA))


def test_to_anthropic_schema_does_not_mutate_the_original():
    import copy

    original = copy.deepcopy(SCHEMA)
    to_anthropic_schema(SCHEMA)
    assert original == SCHEMA


def test_local_validation_still_enforces_stripped_keywords():
    """The wire schema drops minimum/maximum/maxItems, but validate() -- run
    against the ORIGINAL schema -- must still catch a violation of them.
    Local validation is never weakened by what Anthropic will accept."""
    as_strict_tool("submit", "desc", SCHEMA)  # exercises the sanitized copy
    with pytest.raises(SchemaValidationError, match="above maximum"):
        validate({"position": "AGREED", "score": 999, "findings": []}, SCHEMA)
    with pytest.raises(SchemaValidationError, match="exceeds maxItems"):
        validate({"position": "AGREED", "findings": [{"text": "a"}] * 5}, SCHEMA)


@pytest.mark.parametrize(
    "schema_fragment",
    [
        {"type": "array", "minItems": 1, "items": {"type": "string"}},
        {"type": "array", "maxItems": 5, "items": {"type": "string"}},
        {"type": "integer", "minimum": 0},
        {"type": "integer", "maximum": 100},
        {"type": "number", "minimum": 0.0, "maximum": 1.0},
    ],
)
def test_to_anthropic_schema_handles_each_unsupported_keyword_in_isolation(schema_fragment):
    cleaned = to_anthropic_schema(schema_fragment)
    for keyword in ANTHROPIC_UNSUPPORTED_KEYWORDS:
        assert keyword not in cleaned


def test_to_anthropic_schema_handles_deeply_nested_arrays_of_objects():
    schema = {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {"type": "number", "minimum": 0, "maximum": 10},
                        }
                    },
                },
            }
        },
    }
    cleaned = to_anthropic_schema(schema)
    scores_schema = cleaned["properties"]["groups"]["items"]["properties"]["scores"]
    assert "maxItems" not in cleaned["properties"]["groups"]
    assert "minItems" not in cleaned["properties"]["groups"]
    assert "maxItems" not in scores_schema
    assert "minimum" not in scores_schema["items"]
    assert "maximum" not in scores_schema["items"]
    assert scores_schema["items"]["type"] == "number"


def test_every_production_llm_agent_schema_converts_cleanly():
    """Every one of the eight LLM-backed agents' schemas must produce an
    Anthropic-safe input_schema with none of the unsupported keywords left,
    anywhere in the structure."""
    from investment_research.agents.llm_agents import (
        LLMCompetitiveAgent,
        LLMContradictionAgent,
        LLMRegulatoryAgent,
        LLMScienceAgent,
    )
    from investment_research.agents.llm_agents2 import (
        LLMBearAgent,
        LLMBlindJudgeAgent,
        LLMBullAgent,
        LLMKillAgent,
    )

    def _has_unsupported(node) -> bool:
        if isinstance(node, dict):
            if any(key in ANTHROPIC_UNSUPPORTED_KEYWORDS for key in node):
                return True
            return any(_has_unsupported(v) for v in node.values())
        if isinstance(node, list):
            return any(_has_unsupported(item) for item in node)
        return False

    agent_classes = [
        LLMRegulatoryAgent,
        LLMScienceAgent,
        LLMCompetitiveAgent,
        LLMContradictionAgent,
        LLMKillAgent,
        LLMBearAgent,
        LLMBullAgent,
        LLMBlindJudgeAgent,
    ]
    for agent_cls in agent_classes:
        schema = agent_cls.schema
        # The real production schema must itself use at least one of the
        # keywords being tested, or this assertion proves nothing.
        tool = as_strict_tool(agent_cls.tool_name, agent_cls.tool_description, schema)
        assert not _has_unsupported(tool["input_schema"]), (
            f"{agent_cls.__name__}.schema still has an Anthropic-unsupported "
            "keyword in its wire schema"
        )
        # This test is only meaningful if the original (pre-sanitization)
        # schema actually exercises the fix -- i.e. it really does contain at
        # least one Anthropic-unsupported keyword somewhere in its structure.
        assert _has_unsupported(schema), (
            f"{agent_cls.__name__}.schema does not use any Anthropic-unsupported "
            "keyword, so this test would pass trivially even without the fix"
        )


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


# --- stage-aware budgeting (requirement: discovery must not starve later stages) --
def test_default_stage_quotas_sum_to_a_defensible_allocation():
    """30% discovery / 25% escalation / 30% interpretive / 10% finalization,
    leaving 5% unallocated contingency -- discovery can never again be
    capable of consuming ~99% of the run budget."""
    assert DEFAULT_STAGE_QUOTAS["discovery"] == pytest.approx(0.30)
    assert DEFAULT_STAGE_QUOTAS["escalation"] == pytest.approx(0.25)
    assert DEFAULT_STAGE_QUOTAS["interpretive"] == pytest.approx(0.30)
    assert DEFAULT_STAGE_QUOTAS["finalization"] == pytest.approx(0.10)
    assert sum(DEFAULT_STAGE_QUOTAS.values()) < 1.0, "contingency must be left unallocated"


def test_a_stage_never_set_is_completely_unaffected_by_quotas():
    """Backward compatibility: a caller that never calls set_stage() sees no
    quota enforcement at all -- only the existing global cap applies."""
    budget = LLMBudget(max_total_tokens=1000)
    budget.check(900)  # would consume 90% of the global budget in one call
    budget.record(LLMCallRecord("a", "m", input_tokens=900, output_tokens=0))
    assert budget.exhausted is False
    assert budget.stage_used == {}


def test_stage_quota_blocks_further_calls_in_that_stage_once_hit():
    budget = LLMBudget(max_total_tokens=1000, stage_quotas={"discovery": 0.30})
    budget.set_stage("discovery")
    # Cap is 300 tokens. A single call using 250 fits.
    budget.check(250)
    budget.record(LLMCallRecord("adversarial_bear", "m", input_tokens=200, output_tokens=50))
    assert "discovery" not in budget.stage_exhausted

    # A further reservation that would push discovery over its 300-token cap
    # must be refused, even though the GLOBAL budget has plenty left.
    with pytest.raises(BudgetExceeded, match="discovery"):
        budget.check(200)
    assert "discovery" in budget.stage_exhausted
    assert budget.exhausted is False, "a stage quota hit must not be a global exhaustion"


def test_stage_quota_exhaustion_does_not_block_a_different_stage():
    budget = LLMBudget(
        max_total_tokens=1000, stage_quotas={"discovery": 0.10, "interpretive": 0.30}
    )
    budget.set_stage("discovery")
    budget.record(LLMCallRecord("adversarial_bear", "m", input_tokens=100, output_tokens=0))
    with pytest.raises(BudgetExceeded):
        budget.check(50)  # discovery's 100-token cap is already used up
    assert "discovery" in budget.stage_exhausted

    # Switching to a different stage must not be affected by discovery's cap.
    budget.set_stage("interpretive")
    budget.check(200)  # well within interpretive's 300-token cap
    budget.record(LLMCallRecord("regulatory", "m", input_tokens=150, output_tokens=0))
    assert "interpretive" not in budget.stage_exhausted


def test_stage_actual_usage_overrun_marks_that_stage_exhausted():
    """Mirrors the global-exhaustion case: a stage's ACTUAL usage (not just
    the preflight estimate) can overrun its cap, and that must be caught
    too."""
    budget = LLMBudget(max_total_tokens=1000, stage_quotas={"discovery": 0.10})
    budget.set_stage("discovery")
    budget.check(50)  # preflight looks fine
    # But the real response used far more than reserved.
    budget.record(LLMCallRecord("adversarial_bear", "m", input_tokens=90, output_tokens=90))
    assert "discovery" in budget.stage_exhausted


def test_stage_summary_reports_used_cap_remaining_exhausted():
    budget = LLMBudget(max_total_tokens=1000, stage_quotas={"discovery": 0.30})
    budget.set_stage("discovery")
    budget.record(LLMCallRecord("adversarial_bear", "m", input_tokens=100, output_tokens=0))
    summary = budget.stage_summary()
    assert summary["discovery"] == {
        "used": 100,
        "cap": 300,
        "remaining": 200,
        "exhausted": False,
    }


def test_llm_call_record_is_stamped_with_the_active_stage():
    budget = LLMBudget(max_total_tokens=1000)
    budget.set_stage("escalation")
    record = LLMCallRecord("escalation", "m", input_tokens=10, output_tokens=5)
    budget.record(record)
    assert record.stage == "escalation"


# --- effort-aware server-tool reservation (requirement F) -------------------
_SEARCH_TOOL = [{"type": "web_search_20260318", "name": "web_search", "max_uses": 5}]


def test_low_effort_server_tool_reservation_is_smaller_than_the_default():
    """A low-effort discovery call cannot spend a large internal reasoning
    budget, so its conservative preflight reserve is smaller too -- this is
    what lets a one-query-per-required-domain core pass (six domains) fit a
    30%-of-budget discovery stage quota."""
    client = LLMClient(api_key="sk-ant-test", effort="high")
    low = client._default_reservation(
        system="s", messages=[{"role": "user", "content": "q"}], tools=_SEARCH_TOOL,
        max_tokens=1024, effort="low",
    )
    high = client._default_reservation(
        system="s", messages=[{"role": "user", "content": "q"}], tools=_SEARCH_TOOL,
        max_tokens=1024, effort="high",
    )
    assert low < high
    assert low - (len("s") // 4 + 1024) == SERVER_TOOL_TOKEN_RESERVE_LOW_EFFORT
    assert high - (len("s") // 4 + 1024) == SERVER_TOOL_TOKEN_RESERVE


def test_six_domain_core_pass_reservations_fit_the_default_discovery_quota():
    """Six required-domain core search calls, each at the low-effort reserve,
    must fit inside the default discovery stage quota (30% of a 200000-token
    budget) -- the arithmetic requirement F actually needs to hold."""
    client = LLMClient(api_key="sk-ant-test")
    per_call = client._default_reservation(
        system="s", messages=[{"role": "user", "content": "q"}], tools=_SEARCH_TOOL,
        max_tokens=1024, effort="low",
    )
    discovery_quota = int(200_000 * DEFAULT_STAGE_QUOTAS["discovery"])
    assert per_call * 6 <= discovery_quota


def test_omitting_effort_falls_back_to_the_clients_own_configured_effort():
    """A caller that never overrides effort (every interpretive agent) is
    unaffected -- reservation uses the client's own --llm-effort as before."""
    client = LLMClient(api_key="sk-ant-test", effort="low")
    reservation = client._default_reservation(
        system="s", messages=[{"role": "user", "content": "q"}], tools=_SEARCH_TOOL, max_tokens=1024
    )
    assert reservation - (len("s") // 4 + 1024) == SERVER_TOOL_TOKEN_RESERVE_LOW_EFFORT
