"""Claude client for the LLM-backed agents (requirement P2).

Design constraints that are not negotiable:

1. **The model sees only what its agent saw.** Prompts are built from the
   projected ``AgentInput`` and the agent's evidence pack. Before any request is
   sent, the rendered prompt is passed through the isolation leakage scanner --
   a stricter check than projection alone, because it catches a prompt template
   that accidentally interpolates something the policy denied.
2. **Structured output or failure.** Every agent response comes back through a
   strict tool schema and is validated locally. An unparseable or invalid
   response degrades the agent; it never becomes a partial answer.
3. **No silent fallback.** With no credentials the client reports unavailable and
   the pipeline uses the deterministic agent, saying so in the report.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import SchemaValidationError, as_strict_tool, validate

log = logging.getLogger(__name__)

#: Normal/default model for interpretive agents. Verified against the real
#: Anthropic API (claude-sonnet-5, web_search_20260318, web_fetch_20260318).
#: Opus 5 remains fully supported -- pass it explicitly (LLMClient(model=...)
#: or --llm-model claude-opus-5) for a final red-team/audit run.
DEFAULT_MODEL = "claude-sonnet-5"

#: Models on which `budget_tokens` is rejected and adaptive thinking is the
#: only supported configuration.
_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
    "claude-fable-5-1",
)


class LLMUnavailable(RuntimeError):
    """Raised when an LLM call is attempted with no usable credentials."""


@dataclass
class LLMCallRecord:
    agent_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    duration_ms: int = 0
    attempts: int = 1
    ok: bool = True
    error: str = ""
    stop_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMBudget:
    """Per-run token accounting (requirement P9)."""

    max_total_tokens: int = 2_000_000
    used_input: int = 0
    used_output: int = 0
    calls: list[LLMCallRecord] = field(default_factory=list)

    @property
    def used_total(self) -> int:
        return self.used_input + self.used_output

    @property
    def remaining(self) -> int:
        return max(0, self.max_total_tokens - self.used_total)

    def would_exceed(self, estimated: int) -> bool:
        return self.used_total + estimated > self.max_total_tokens

    def record(self, call: LLMCallRecord) -> None:
        self.used_input += call.input_tokens
        self.used_output += call.output_tokens
        self.calls.append(call)

    def by_agent(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for call in self.calls:
            totals[call.agent_id] = totals.get(call.agent_id, 0) + call.total_tokens
        return totals


class LLMClient:
    """Thin wrapper over the Anthropic Messages API.

    Kept thin on purpose: the deterministic core must remain the system's
    backbone, and this is a component it can run without.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_retries: int = 3,
        timeout: float = 300.0,
        budget: LLMBudget | None = None,
        base_url: str = "",
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_retries = max_retries
        self.timeout = timeout
        self.budget = budget or LLMBudget()
        self.last_usage_tokens = 0
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None
        self._unavailable_reason = ""
        self._init_client()

    # -- lifecycle ---------------------------------------------------------
    def _init_client(self) -> None:
        try:
            import anthropic
        except ImportError:
            self._unavailable_reason = (
                "the 'anthropic' package is not installed; install it to enable LLM agents"
            )
            return
        kwargs: dict[str, Any] = {"timeout": self.timeout, "max_retries": self.max_retries}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        try:
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as exc:  # noqa: BLE001 - covers missing credentials
            self._unavailable_reason = f"{type(exc).__name__}: {exc}"
            self._client = None
            return

        # Constructing the client succeeds even with no credentials -- the SDK
        # only fails when a request is made. Reporting "ready" here and then
        # failing on the first agent would look like an agent bug, so resolve
        # credentials now and say plainly if there are none.
        if not self._has_credentials():
            self._unavailable_reason = (
                "no Anthropic credentials resolvable (set IRA_ANTHROPIC_API_KEY, or the "
                "global ANTHROPIC_API_KEY, or run `ant auth login`); LLM agents disabled, "
                "deterministic agents will run"
            )
            self._client = None

    def _has_credentials(self) -> bool:
        """Whether the SDK will be able to authenticate a request.

        Mirrors the SDK's own resolution order: explicit key -> api_key /
        auth_token on the client -> an on-disk OAuth profile.
        """
        if self._api_key:
            return True
        client = self._client
        if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
            return True
        import os
        from pathlib import Path

        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True
        profile_dir = Path.home() / ".config" / "anthropic"
        return profile_dir.is_dir() and any(profile_dir.iterdir())

    def available(self) -> tuple[bool, str]:
        if self._client is None:
            return False, self._unavailable_reason or "no client"
        return True, "ready"

    def _require(self) -> Any:
        if self._client is None:
            raise LLMUnavailable(self._unavailable_reason or "LLM client unavailable")
        return self._client

    # -- raw ---------------------------------------------------------------
    def raw_message(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 16000,
        agent_id: str = "raw",
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        """One Messages API call, with usage accounting."""
        client = self._require()
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": list(messages),
            "output_config": {"effort": self.effort},
        }
        if any(self.model.startswith(prefix) for prefix in _ADAPTIVE_THINKING_MODELS):
            request["thinking"] = {"type": "adaptive"}
        if tools:
            request["tools"] = list(tools)
        if tool_choice:
            request["tool_choice"] = tool_choice

        started = time.monotonic()
        record = LLMCallRecord(agent_id=agent_id, model=self.model)
        try:
            response = client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            record.duration_ms = int((time.monotonic() - started) * 1000)
            self.budget.record(record)
            raise

        usage = getattr(response, "usage", None)
        record.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        record.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        record.cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        record.stop_reason = str(getattr(response, "stop_reason", "") or "")
        record.duration_ms = int((time.monotonic() - started) * 1000)
        self.budget.record(record)
        self.last_usage_tokens = record.total_tokens
        return response

    # -- structured --------------------------------------------------------
    def structured(
        self,
        *,
        agent_id: str,
        system: str,
        prompt: str,
        tool_name: str,
        tool_description: str,
        schema: dict[str, Any],
        max_tokens: int = 16000,
    ) -> dict[str, Any]:
        """Call the model and return a schema-validated object.

        Raises on failure rather than returning something partial. The caller
        turns that into a degraded agent run, which the report shows.
        """
        tool = as_strict_tool(tool_name, tool_description, schema)
        response = self.raw_message(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            max_tokens=max_tokens,
            agent_id=agent_id,
            # `auto` plus an explicit instruction: forced tool_choice is
            # rejected on some current models, and `auto` is portable.
            tool_choice={"type": "auto"},
        )

        stop_reason = str(getattr(response, "stop_reason", "") or "")
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMUnavailable(f"model declined the request (category={category})")

        payload = _first_tool_input(response, tool_name)
        if payload is None:
            payload = _json_from_text(response)
        if payload is None:
            raise SchemaValidationError(
                f"{agent_id}: model returned no parseable structured output "
                f"(stop_reason={stop_reason!r})"
            )
        validate(payload, schema)
        return payload


def _first_tool_input(response: Any, tool_name: str) -> dict[str, Any] | None:
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != tool_name:
            continue
        payload = getattr(block, "input", None)
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
        if isinstance(payload, dict):
            return payload
    return None


def _json_from_text(response: Any) -> dict[str, Any] | None:
    """Last-resort parse of a JSON object emitted as text.

    Tolerated because a well-formed object in a text block is still verifiable;
    it is validated against the same schema either way.
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        text = (getattr(block, "text", "") or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def build_llm_client(settings, *, budget: LLMBudget | None = None) -> LLMClient:
    return LLMClient(
        api_key=settings.anthropic_api_key,
        model=getattr(settings, "llm_model", DEFAULT_MODEL),
        effort=getattr(settings, "llm_effort", "high"),
        budget=budget,
    )
