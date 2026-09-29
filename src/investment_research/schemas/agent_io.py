"""Agent input/output contracts and the channel-addressed evidence bus.

The central safety idea of this system lives here.

Every agent output is split into two *physically separate* containers:

``facts`` / ``risk_flags`` / ``unresolved`` / ``contradictions``
    Transportable.  Requirement 1D explicitly permits facts, sources, tiers,
    confidences, contradictions, unresolved questions and risk flags to move
    downstream.

``evaluation``
    Quarantined.  Ratings, scores, verdicts, narratives -- anything that
    expresses a *view*.  It is published to a named channel and only agents
    whose isolation policy names that channel can ever see it.

The orchestrator never hands an agent the whole bus.  It hands it a
:class:`AgentInput` built by projecting the bus through the agent's policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .enums import FactCategory, Materiality, RunStatus
from .fact import Contradiction, Fact, Source, UnresolvedQuestion

if TYPE_CHECKING:
    # String-only (PEP 563 deferred annotations, already active above) so
    # this stays a type-checking-only edge: schemas is the foundation layer
    # and nothing here imports scoring at runtime. scoring/program_resolution
    # itself only imports schemas.enums/schemas.fact, so there is no runtime
    # cycle either way -- this import is kept TYPE_CHECKING-only to preserve
    # the existing layering convention (no schemas/*.py imports `..scoring`).
    from ..scoring.program_resolution import ProgramResolution


@dataclass
class RiskFlag:
    """A named risk, without a verdict attached.

    A risk flag says *what is worrying and why*, citing facts.  It never says
    "avoid" or assigns a score -- that is the Kill Gate's and the Judge's job.
    """

    flag_id: str
    category: FactCategory
    title: str
    detail: str
    severity: Materiality
    fact_ids: tuple[str, ...] = ()
    raised_by: str = "unknown"

    def cites(self) -> tuple[str, ...]:
        return self.fact_ids


@dataclass
class Evaluation:
    """Quarantined judgement produced by an agent.

    ``fingerprint_tokens`` are rare tokens unique to this evaluation.  The
    leakage scanner searches downstream payloads for them, so a Bull narrative
    cannot reach the Bear agent even by being copied into a "fact".
    """

    author: str
    channel: str
    summary: str = ""
    points: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    fingerprint_tokens: tuple[str, ...] = ()


@dataclass
class AgentOutput:
    agent_id: str
    ok: bool = True
    facts: list[Fact] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)
    unresolved: list[UnresolvedQuestion] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    evaluation: Evaluation | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    degraded: bool = False
    duration_ms: int = 0

    @property
    def status(self) -> str:
        if not self.ok:
            return "FAILED"
        return "DEGRADED" if self.degraded else "OK"


@dataclass
class AgentInput:
    """What an agent is actually allowed to see.

    Built exclusively by :class:`investment_research.orchestrator.isolation.IsolationGuard`.
    Constructing one by hand inside an agent is a bug -- the isolation tests
    assert that agents only read from the input they were handed.
    """

    agent_id: str
    run_id: str
    ticker: str | None  # None for identity-blind agents (Blind Judge)
    company_name: str | None
    facts: tuple[Fact, ...] = ()
    sources: tuple[Source, ...] = ()
    risk_flags: tuple[RiskFlag, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    unresolved: tuple[UnresolvedQuestion, ...] = ()
    channels: dict[str, Evaluation] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    #: Phase 4.3D: the ONE centrally-computed ProgramResolution for this run,
    #: attached ONLY for the agent_ids IsolationGuard.project() permits
    #: ("science", "kill_agent") -- None for every other agent, including
    #: Blind Judge/Bull/Bear, which must never see which trial was selected
    #: or why. Never read directly by any agent except via this field: no
    #: agent may recompute its own resolution from ``facts``.
    program_resolution: ProgramResolution | None = None

    def channel(self, name: str) -> Evaluation | None:
        return self.channels.get(name)

    def facts_in(self, *categories: FactCategory) -> tuple[Fact, ...]:
        wanted = set(categories)
        return tuple(f for f in self.facts if f.category in wanted)

    def decision_grade_facts(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.is_decision_grade)


@dataclass
class AgentRunRecord:
    run_id: str
    agent_id: str
    status: str
    started_at: str
    finished_at: str
    duration_ms: int
    fact_count: int
    error_count: int
    errors: str = ""
    inputs_hash: str = ""

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RunContext:
    """Immutable per-run identity + mode flags."""

    run_id: str
    ticker: str
    company_name: str
    mode: str = "standard"
    offline: bool = True
    use_fixtures: bool = False
    started_at: str = ""
    status: RunStatus = RunStatus.COMPLETE
    notes: list[str] = field(default_factory=list)
