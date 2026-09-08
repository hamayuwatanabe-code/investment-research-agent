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


class BudgetExceeded(RuntimeError):
    """Raised when a call cannot be funded by the remaining LLM token budget.

    The guarantee this class is part of is **"no further calls after
    exhaustion, plus a conservative preflight reservation"** -- it is
    deliberately NOT "mathematically guaranteed zero overshoot from a single
    server-side call". Anthropic's server-side tools (``web_search``,
    ``web_fetch``) execute on Anthropic's infrastructure, and their real
    token cost is not knowable to this client until the response's ``usage``
    field comes back. So one in-flight call can still push actual usage past
    ``max_total_tokens`` even though its preflight reservation looked fine --
    see :meth:`LLMBudget.record`. What *is* guaranteed: :meth:`LLMBudget.check`
    never lets a request whose conservative reservation does not fit proceed,
    and once the budget is marked exhausted (whether by a failed preflight
    check or by an actual-usage overrun), every subsequent call is refused.
    """


#: Rough, deliberately conservative chars-per-token ratio for the local
#: preflight estimate below (same heuristic as
#: ``collectors.documents.estimate_tokens``, kept local so this module stays
#: dependency-light).
_CHARS_PER_TOKEN = 4

#: Tool `type` prefixes for Anthropic's server-side web tools. Unlike a local
#: strict-schema tool (whose cost is just the schema plus a JSON object the
#: model writes as part of its own bounded output), these execute remotely
#: and their token cost is unknown until the response comes back.
_SERVER_SIDE_WEB_TOOL_PREFIXES = ("web_search", "web_fetch")

#: Conservative flat pad added to the local preflight estimate when a request
#: declares one of the server-side web tools above. This is a safety buffer,
#: never a prediction of the real cost -- see BudgetExceeded.
SERVER_TOOL_TOKEN_RESERVE = 20_000


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
    """Per-run token accounting AND enforcement (requirement P9).

    Two distinct mechanisms, not one:

    1. **Preflight** (:meth:`check`): refuses a call whose conservative
       reservation would not fit in what remains. This runs before any
       network request, so a rejected call never reaches the Anthropic API.
    2. **Post-hoc exhaustion** (:meth:`record`): inspects *actual* usage
       after a call completes. A server-side tool call's real cost is not
       knowable in advance (see :class:`BudgetExceeded`), so actual usage can
       still push ``used_total`` past ``max_total_tokens`` even when the
       preflight reservation looked fine. When that happens, the budget
       latches ``exhausted`` and every subsequent call is refused for the
       rest of this run -- permanently; there is no reset.

    This class does NOT guarantee that a single in-flight call can never
    overshoot ``max_total_tokens``. The guarantee is: no further calls after
    exhaustion, plus a conservative preflight reservation on every call --
    not a mathematically bounded zero overshoot from one server-side call
    whose cost was unknown before it ran.
    """

    max_total_tokens: int = 2_000_000
    used_input: int = 0
    used_output: int = 0
    calls: list[LLMCallRecord] = field(default_factory=list)
    #: Latched permanently once a preflight check fails or actual usage puts
    #: the budget over its ceiling. Once True, every future call is refused.
    exhausted: bool = False
    exhausted_reason: str = ""

    @property
    def used_total(self) -> int:
        return self.used_input + self.used_output

    @property
    def remaining(self) -> int:
        return max(0, self.max_total_tokens - self.used_total)

    def would_exceed(self, estimated: int) -> bool:
        return self.used_total + estimated > self.max_total_tokens

    def check(self, reserved_tokens: int) -> None:
        """Preflight guard: call before every API request, make none if it raises.

        Raises :class:`BudgetExceeded` -- without making any network call --
        when the budget is already exhausted, or when ``reserved_tokens``
        would not fit in what remains. ``reserved_tokens`` is a conservative
        estimate; it is never treated as an exact prediction, particularly
        for calls that invoke a server-side web tool.
        """
        if self.exhausted:
            raise BudgetExceeded(
                self.exhausted_reason
                or "LLM token budget already exhausted; no further calls this run"
            )
        if self.would_exceed(reserved_tokens):
            self.exhausted = True
            self.exhausted_reason = (
                f"preflight: a planned request reserving ~{reserved_tokens} tokens would "
                f"exceed the LLM budget ({self.used_total}/{self.max_total_tokens} used, "
                f"{self.remaining} remaining) -- refusing to call the API"
            )
            log.warning(self.exhausted_reason)
            raise BudgetExceeded(self.exhausted_reason)

    def record(self, call: LLMCallRecord) -> None:
        self.used_input += call.input_tokens
        self.used_output += call.output_tokens
        self.calls.append(call)
        if not self.exhausted and self.used_total > self.max_total_tokens:
            self.exhausted = True
            self.exhausted_reason = (
                f"actual usage after agent {call.agent_id!r} put the LLM budget over its "
                f"ceiling ({self.used_total}/{self.max_total_tokens} tokens used). A "
                "server-side tool call's real cost is not known until the response comes "
                "back, so this can happen even after a conservative preflight reservation. "
                "No further LLM calls will be made this run."
            )
            log.warning(self.exhausted_reason)

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
        if self.budget.exhausted:
            return False, self.budget.exhausted_reason or "LLM token budget exhausted"
        return True, "ready"

    def _require(self) -> Any:
        if self._client is None:
            raise LLMUnavailable(self._unavailable_reason or "LLM client unavailable")
        return self._client

    def _default_reservation(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
        max_tokens: int,
    ) -> int:
        """Conservative preflight token estimate for one request.

        Deliberately not an exact prediction: it is a local chars/4 estimate of
        the rendered prompt plus the requested output ceiling, padded with
        ``SERVER_TOOL_TOKEN_RESERVE`` when a server-side web tool is declared,
        because that tool's real cost is only known once the response's
        ``usage`` field comes back.
        """
        prompt_chars = len(system) + sum(len(str(m.get("content", ""))) for m in messages)
        reservation = prompt_chars // _CHARS_PER_TOKEN + max_tokens
        if tools and any(
            str(t.get("type", "")).startswith(_SERVER_SIDE_WEB_TOOL_PREFIXES) for t in tools
        ):
            reservation += SERVER_TOOL_TOKEN_RESERVE
        return reservation

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
        reserved_tokens: int | None = None,
    ) -> Any:
        """One Messages API call, with a preflight budget guard and usage accounting.

        Before any network request, ``self.budget.check()`` is given a
        conservative token reservation and may raise :class:`BudgetExceeded`,
        in which case no request is made at all. When the caller does not
        supply ``reserved_tokens`` explicitly, one is derived by
        :meth:`_default_reservation`; a caller with a better estimate of its
        own cost (e.g. a research provider that knows its fetch ceiling)
        should pass ``reserved_tokens`` directly.
        """
        client = self._require()
        if reserved_tokens is None:
            reserved_tokens = self._default_reservation(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
            )
        self.budget.check(reserved_tokens)

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
        reserved_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call the model and return a schema-validated object.

        Raises on failure rather than returning something partial. The caller
        turns that into a degraded agent run, which the report shows.
        ``reserved_tokens`` is forwarded to :meth:`raw_message`'s preflight
        budget guard; omit it to use the conservative default derivation.
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
            reserved_tokens=reserved_tokens,
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
