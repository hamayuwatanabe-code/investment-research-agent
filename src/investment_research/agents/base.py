"""Agent base class.

An agent receives only the :class:`AgentInput` the orchestrator projected for
it.  It must not reach into the bus, the database or the network for analysis
inputs -- the isolation tests assert this by giving agents a bus that would fail
if consulted.

Failure policy (requirement 14): an agent that raises does not abort the run.
The orchestrator records the failure, marks the run INCOMPLETE_RESEARCH, and the
final report says which agent failed.  A missing analysis is never treated as an
absence of risk.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ..schemas.agent_io import AgentInput, AgentOutput, Evaluation
from ..orchestrator.isolation import derive_fingerprints

log = logging.getLogger(__name__)


class Agent(ABC):
    #: Must match a key in ``orchestrator.isolation.POLICIES``.
    agent_id: str = "unset"
    #: Human-readable purpose, echoed into logs and the run record.
    purpose: str = ""
    #: Seconds after which the orchestrator considers this agent hung.
    timeout_seconds: float = 120.0

    @abstractmethod
    def run(self, data: AgentInput) -> AgentOutput:  # pragma: no cover - abstract
        ...

    # -- helpers ----------------------------------------------------------
    def evaluation(
        self,
        channel: str,
        summary: str,
        points: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
        baseline_texts: tuple[str, ...] = (),
    ) -> Evaluation:
        """Build a fingerprinted evaluation for publication to ``channel``.

        The fingerprint is computed against the shared evidence baseline so that
        only prose *novel to this evaluation* is treated as diagnostic of a
        leak.
        """
        texts = [summary, *points]
        return Evaluation(
            author=self.agent_id,
            channel=channel,
            summary=summary,
            points=points,
            payload=payload or {},
            fingerprint_tokens=derive_fingerprints(texts, baseline_texts),
        )

    def baseline_from(self, data: AgentInput) -> tuple[str, ...]:
        return tuple(f.claim for f in data.facts)

    def execute(self, data: AgentInput) -> AgentOutput:
        """Run with timing and exception containment."""
        started = time.monotonic()
        try:
            output = self.run(data)
        except Exception as exc:  # noqa: BLE001 - deliberate containment
            log.exception("agent %s failed", self.agent_id)
            output = AgentOutput(
                agent_id=self.agent_id,
                ok=False,
                errors=[f"{type(exc).__name__}: {exc}"],
                degraded=True,
            )
        output.duration_ms = int((time.monotonic() - started) * 1000)
        output.agent_id = self.agent_id
        return output


def inputs_hash(data: AgentInput) -> str:
    payload = "|".join(
        [
            data.agent_id,
            str(data.ticker),
            *sorted(f.fact_id for f in data.facts),
            *sorted(data.channels),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
