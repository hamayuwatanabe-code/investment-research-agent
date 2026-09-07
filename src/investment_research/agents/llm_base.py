"""Base class for LLM-backed agents (requirement P2).

The isolation problem an LLM introduces
---------------------------------------
Phase 1 guaranteed that an agent's *input object* contained nothing its policy
denied. Once a model is in the loop there is a second surface: the **rendered
prompt**. A template that interpolates the wrong variable, or an evidence pack
built from the unfiltered bus, would send denied content to the model even
though the ``AgentInput`` was clean.

So every prompt is scanned before it is sent, using the same leakage scanner
that guards projection, seeded with the fingerprints of every denied channel and
(for blind agents) the identity markers. A prompt that fails is never sent.

The degradation contract
------------------------
An LLM agent that cannot run -- no credentials, a refusal, a schema violation,
an exhausted budget -- returns a degraded output and the pipeline falls back to
the deterministic agent for that stage. The report says which agents were
LLM-backed and which were not. It never presents a fallback as an LLM analysis.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..collectors.documents import EvidencePack
from ..llm.client import LLMClient, LLMUnavailable
from ..llm.schema import SchemaValidationError
from ..orchestrator.isolation import LeakageError, LeakageScanner
from ..schemas.agent_io import AgentInput, AgentOutput
from .base import Agent

log = logging.getLogger(__name__)

#: Preamble shared by every analytical agent. It states the system's purpose so
#: the model optimises for the right thing, and it forbids the specific failure
#: modes this system exists to prevent.
COMMON_SYSTEM = """\
You are one agent in an investment research system whose purpose is NOT to find good stocks.
Its purpose is to eliminate wrong investment hypotheses as early as possible, using objective
facts, primary sources and disconfirming evidence.

Rules that apply to you without exception:

1. Use ONLY the evidence provided in this prompt. You have no other knowledge of this company,
   and anything you think you remember about it is inadmissible. If the evidence does not
   establish something, the answer is UNKNOWN.
2. Never invent a URL, a filing, an accession number, a registry identifier, a date, a
   quotation, an analyst rating or a price target.
3. Cite the evidence for every claim you make, by the [chunk_id] or fact_id given to you. A
   claim you cannot cite must be dropped, not softened.
4. A company's own statement is a company claim regardless of how primary the venue it was
   filed in. A company's adjective about a regulator interaction ("constructive", "aligned",
   "productive") is never evidence about what the regulator actually said or decided.
5. Contradictions are an output, not a problem to resolve. Never reconcile conflicting
   evidence toward the more attractive reading.
6. Absence of evidence is not evidence of absence. "We did not find it" and "it is not so"
   are different findings and must not be merged.

The company may be referred to as Company X. Do not speculate about its identity.
"""


@dataclass
class PromptGuard:
    """Isolation context for scanning a rendered prompt.

    Held by the agent, never placed in its ``AgentInput``. Putting the identity
    markers into the input so the agent could scan for them would mean the blind
    judge's own input contained the ticker -- defeating the thing being checked.
    The orchestrator builds this from the bus and hands it over out of band.
    """

    denied_fingerprints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    forbidden_identity: tuple[str, ...] = ()

    def scan(self, agent_id: str, prompt: str) -> list[str]:
        return LeakageScanner(self.denied_fingerprints, self.forbidden_identity).scan(
            agent_id, prompt
        )


@dataclass
class LLMAgentResult:
    payload: dict[str, Any]
    ok: bool = True
    error: str = ""
    tokens: int = 0


@dataclass
class PromptBuildResult:
    prompt: str
    pack: EvidencePack | None = None
    cited_ids: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


class LLMAgent(Agent):
    """An agent whose judgement comes from Claude, under schema and isolation."""

    #: Subclasses set these.
    tool_name: str = "submit_analysis"
    tool_description: str = "Submit the structured result of this analysis."
    schema: dict[str, Any] = {}
    system_extra: str = ""
    max_tokens: int = 12000
    #: Token budget for this agent's evidence pack (requirement P9).
    pack_budget_tokens: int = 12000

    def __init__(
        self,
        llm: LLMClient,
        *,
        fallback: Agent | None = None,
        guard: PromptGuard | None = None,
        chunks: Sequence[Any] = (),
        discovery_summary: str = "",
    ) -> None:
        self.llm = llm
        self.fallback = fallback
        self.guard = guard or PromptGuard()
        # Held out of band for the same reason as the guard: a Chunk carries the
        # document text, and document text carries the company name. Routing
        # chunks through AgentInput.params would put un-anonymised prose into the
        # blind judge's input, which the isolation scanner correctly rejects.
        self.chunks = list(chunks)
        # Purpose-filtered web-search discovery context (requirement M3), built
        # by the orchestrator from purposes_for_agent(self.agent_id) BEFORE
        # construction. Never routed through the shared AgentInput.params dict,
        # which is copied unfiltered into every agent's input -- a Bear-purpose
        # summary sitting there would be one careless read away from reaching
        # the Bull agent.
        self.discovery_summary = discovery_summary

    # -- to implement ------------------------------------------------------
    def build_prompt(self, data: AgentInput) -> PromptBuildResult:  # pragma: no cover
        raise NotImplementedError

    def interpret(
        self, payload: dict[str, Any], data: AgentInput
    ) -> AgentOutput:  # pragma: no cover
        raise NotImplementedError

    # -- isolation ---------------------------------------------------------
    def assert_prompt_is_clean(self, data: AgentInput, prompt: str) -> None:
        """Scan the rendered prompt for anything this agent may not see.

        This is the check that keeps isolation true once a model is involved.
        Projection guards the input object; this guards the string that is
        actually transmitted, and so catches a template that interpolates the
        wrong variable or a pack built from the unfiltered bus.
        """
        violations = self.guard.scan(self.agent_id, prompt)
        if violations:
            log.error("prompt leakage for %s: %s", self.agent_id, violations)
            raise LeakageError(f"refusing to send a prompt containing denied content: {violations}")

    # -- execution ---------------------------------------------------------
    def run(self, data: AgentInput) -> AgentOutput:
        usable, reason = self.llm.available()
        if not usable:
            return self._fallback(data, f"LLM unavailable: {reason}")

        try:
            built = self.build_prompt(data)
        except Exception as exc:  # noqa: BLE001
            return self._fallback(data, f"prompt build failed: {exc}")

        try:
            self.assert_prompt_is_clean(data, built.prompt)
        except LeakageError:
            # Never downgrade a leak into a fallback: it is a correctness
            # failure of the system, not a degraded model call.
            raise

        estimated = len(built.prompt) // 4 + self.max_tokens
        if self.llm.budget.would_exceed(estimated):
            return self._fallback(
                data,
                f"token budget exhausted ({self.llm.budget.used_total}/"
                f"{self.llm.budget.max_total_tokens})",
            )

        try:
            payload = self.llm.structured(
                agent_id=self.agent_id,
                system=COMMON_SYSTEM + ("\n" + self.system_extra if self.system_extra else ""),
                prompt=built.prompt,
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                schema=self.schema,
                max_tokens=self.max_tokens,
            )
        except (LLMUnavailable, SchemaValidationError) as exc:
            return self._fallback(data, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return self._fallback(data, f"LLM call failed: {type(exc).__name__}: {exc}")

        output = self.interpret(payload, data)
        output.metrics["llm_backed"] = True
        output.metrics["llm_model"] = self.llm.model
        if built.pack is not None:
            output.metrics["pack_tokens"] = built.pack.total_tokens
            output.metrics["pack_chunks"] = len(built.pack.chunks)
        return output

    def _fallback(self, data: AgentInput, reason: str) -> AgentOutput:
        log.info("%s falling back to deterministic analysis: %s", self.agent_id, reason)
        if self.fallback is None:
            return AgentOutput(
                agent_id=self.agent_id,
                ok=False,
                degraded=True,
                errors=[f"no LLM and no deterministic fallback: {reason}"],
            )
        output = self.fallback.run(data)
        output.agent_id = self.agent_id
        output.degraded = True
        output.errors.append(f"LLM not used: {reason}")
        output.metrics["llm_backed"] = False
        output.metrics["llm_fallback_reason"] = reason
        return output


def cited_only(
    items: Sequence[dict[str, Any]], valid_ids: set[str], *, key: str = "evidence_ids"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split model claims into cited and uncited.

    Requirement P7: a claim whose citations do not resolve to evidence actually
    in the agent's pack is dropped and counted, never quietly kept.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in items:
        ids = [str(i) for i in (item.get(key) or [])]
        resolved = [i for i in ids if i in valid_ids]
        if resolved:
            item = {**item, key: resolved}
            kept.append(item)
        else:
            dropped.append(item)
    return kept, dropped
