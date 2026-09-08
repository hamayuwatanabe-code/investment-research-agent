"""The research pipeline.

Stage order is fixed and is itself a requirement (27): the kill test runs before
the bull case, and the blind judgement runs before anything that knows the
ticker or the user's position.

Partial failure policy (requirement 14): a failing agent degrades the run rather
than aborting it, and the run is then marked ``INCOMPLETE_RESEARCH``.  It is
never reported as a successful analysis.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ..agents.base import Agent, inputs_hash
from ..agents.bear_agent import BearAgent
from ..agents.blind_judge import BlindJudgeAgent
from ..agents.bull_agent import BullAgent
from ..agents.capital_structure import CapitalStructureAgent
from ..agents.catalyst import CatalystAgent
from ..agents.competitive import CompetitiveAgent
from ..agents.contradiction import ContradictionAgent
from ..agents.evidence_integrity import EvidenceIntegrityAgent
from ..agents.fact_collector import FactCollectorAgent
from ..agents.kill_agent import KillAgent
from ..agents.microstructure import MicrostructureAgent
from ..agents.portfolio import PortfolioAgent
from ..agents.regulatory import RegulatoryAgent
from ..agents.science import ScienceAgent
from ..agents.valuation import ValuationAgent
from ..collectors.base import CollectionResult
from ..collectors.documents import Chunk
from ..collectors.search import SearchProvider
from ..llm.client import LLMBudget, LLMClient
from ..reporting.traceability import TraceabilityIndex, build_index
from ..research.adversarial import AdversarialOutcome
from ..research.discovery import DiscoveryLog
from ..research.escalation import EscalationReport, escalate, escalate_unresolved_questions
from ..research.provider import NullResearchProvider, ResearchProvider
from ..schemas.agent_io import AgentOutput, AgentRunRecord, RunContext
from ..schemas.enums import (
    UNKNOWN,
    FactCategory,
    Provenance,
    ResearchDomain,
    ResearchStatus,
    RunStatus,
)
from ..schemas.evaluation import KillGateResult, ScoreCard, Verdict
from ..schemas.validation import QuarantinedSource
from ..scoring.completeness import CompletenessResult, assess_completeness
from ..scoring.decision_gate_consistency import blocked_headline, scrub_action_labels
from ..scoring.evidence_confidence import compute_evidence_confidence
from ..scoring.evidence_sufficiency import EvidenceSufficiencyMatrix, assess_evidence_sufficiency
from ..scoring.scenarios import build_scenarios
from ..scoring.scores import build_scorecard
from ..storage.repository import Repository
from ..thesis.versioning import diff_against_previous, snapshot_for
from .isolation import (
    Channel,
    EvidenceBus,
    IsolationGuard,
    LeakageError,
    derive_fingerprints,
)
from .resume import Checkpoint, ResumePlan, stage_index
from .resume import build_plan as build_resume_plan

log = logging.getLogger(__name__)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


@dataclass
class ResearchResult:
    context: RunContext
    bus: EvidenceBus
    verdict: Verdict | None
    scorecard: ScoreCard | None
    scenarios: list = field(default_factory=list)
    catalysts: list = field(default_factory=list)
    agent_records: list[AgentRunRecord] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    confidence_breakdown: Any = None
    thesis_diff: dict[str, Any] = field(default_factory=dict)
    thesis_version: int = 0
    source_ref_map: Any = None
    collection_results: list[CollectionResult] = field(default_factory=list)
    #: Populated only when the caller supplied portfolio information.
    portfolio_guidance: dict[str, Any] = field(default_factory=dict)
    #: Phase 2 additions.
    completeness: CompletenessResult | None = None
    escalation: EscalationReport | None = None
    adversarial: AdversarialOutcome | None = None
    traceability: TraceabilityIndex | None = None
    chunks: list[Chunk] = field(default_factory=list)
    llm_budget: LLMBudget | None = None
    llm_agents_used: list[str] = field(default_factory=list)
    capture_info: dict[str, Any] = field(default_factory=dict)
    resume_plan: ResumePlan | None = None
    #: Sources rejected by schema validation rather than silently persisted
    #: or allowed to crash the run (requirement 14/24).
    quarantined_sources: list[QuarantinedSource] = field(default_factory=list)
    #: Decision-Grade Evidence Gate: whether what was found is actually
    #: verified, as opposed to merely searched-for (requirement DG5).
    evidence_sufficiency: EvidenceSufficiencyMatrix | None = None

    @property
    def blocked(self) -> bool:
        """Whether the completeness gate withheld a verdict (requirement P6)."""
        return bool(self.completeness and self.completeness.blocked)

    @property
    def incomplete(self) -> bool:
        return self.context.status != RunStatus.COMPLETE


class Pipeline:
    """Runs the fourteen agents in the mandated order."""

    def __init__(
        self,
        repository: Repository,
        search: SearchProvider,
        *,
        today: date | None = None,
        strict_isolation: bool = True,
        stale_after_days: int = 400,
        research: ResearchProvider | None = None,
        llm: LLMClient | None = None,
        adversarial: AdversarialOutcome | None = None,
        chunks: Sequence[Chunk] = (),
        capture_info: dict[str, Any] | None = None,
    ) -> None:
        self.repo = repository
        self.search = search
        self.today = today or date.today()
        self.strict_isolation = strict_isolation
        self.stale_after_days = stale_after_days
        self.research = research or NullResearchProvider()
        self.llm = llm
        self.adversarial = adversarial
        self.chunks = list(chunks)
        self.capture_info = capture_info or {}

    def _agent_for(self, stage: str, deterministic: Agent, llm_cls: Any, guard_bus) -> Agent:
        """Return the LLM agent when one is usable, else the deterministic one.

        The deterministic agent is always constructed and always available as the
        fallback, so a missing key or a refused request degrades the run rather
        than failing it -- and the report records which agents were model-backed.
        """
        if self.llm is None or llm_cls is None:
            return deterministic
        usable, _ = self.llm.available()
        if not usable:
            return deterministic
        from ..agents.llm_base import PromptGuard
        from .isolation import policy_for

        policy = policy_for(deterministic.agent_id)
        denied = {
            name: tuple(evaluation.fingerprint_tokens)
            for name, evaluation in guard_bus.channels.items()
            if name not in policy.reads
        }
        forbidden: tuple[str, ...] = ()
        if not policy.sees_identity:
            from .anonymize import ANON_LABEL
            from .isolation import identity_markers

            forbidden = tuple(
                marker
                for marker in identity_markers(
                    self._identity[0], self._identity[1], self._identity[2]
                )
                if marker.strip().lower() != ANON_LABEL.lower()
            )
        # A blind agent gets no raw chunks: chunk text is un-anonymised source
        # prose. It works from its anonymised fact set and the channels it may
        # read, which is what "blind" has to mean once a model is involved.
        chunks = () if not policy.sees_identity else self.chunks
        discovery_summary = "" if not policy.sees_identity else self._discovery_summary_for(
            deterministic.agent_id
        )
        return llm_cls(
            self.llm,
            fallback=deterministic,
            guard=PromptGuard(denied_fingerprints=denied, forbidden_identity=forbidden),
            chunks=chunks,
            discovery_summary=discovery_summary,
        )

    def _discovery_summary_for(self, agent_id: str) -> str:
        """Purpose-filtered web-search context for one agent (requirement M3).

        Built directly from ``self.adversarial.discovery`` and passed to the
        agent out of band (never through the shared ``AgentInput.params`` dict,
        which is copied unfiltered into every agent's input). A Bear-purpose
        hit is never included for an agent whose purposes do not cover BEAR,
        and vice versa for Bull -- see ``purposes_for_agent``.
        """
        if self.adversarial is None:
            return ""
        from ..schemas.enums import purposes_for_agent

        hits = self.adversarial.discovery.hits_for(tuple(purposes_for_agent(agent_id)))
        if not hits:
            return ""
        return "\n".join(
            f"- [{hit.hit_id}] {hit.title} ({hit.url}) -- {hit.snippet[:200]}"
            for hit in hits[:40]
        )

    # -- helpers -----------------------------------------------------------
    def _record(self, output: AgentOutput, run_id: str, started: str) -> AgentRunRecord:
        return AgentRunRecord(
            run_id=run_id,
            agent_id=output.agent_id,
            status=output.status,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_ms=output.duration_ms,
            fact_count=len(output.facts),
            error_count=len(output.errors),
            errors=" | ".join(output.errors)[:2000],
        )

    def _persist_domain_tables(
        self,
        ctx: RunContext,
        bus: EvidenceBus,
        regulatory_payload: dict[str, Any],
        facts: Sequence[Any],
    ) -> None:
        """Write the structured per-domain tables (requirement 11).

        These are derived from facts that are already stored, but keeping them
        in queryable form is what makes "has this trial's design changed since
        last quarter?" answerable without re-parsing prose.
        """
        events = regulatory_payload.get("regulatory_events") or []
        if events:
            self.repo.save_regulatory_events(ctx.run_id, ctx.ticker, events)

        trials = _clinical_trials_from_facts(facts)
        if trials:
            self.repo.save_clinical_trials(ctx.run_id, ctx.ticker, trials)

        trades = _insider_trades_from_facts(facts)
        if trades:
            self.repo.save_insider_trades(ctx.run_id, ctx.ticker, trades)

    @staticmethod
    def _shared_baseline(bus: EvidenceBus, output: AgentOutput) -> list[str]:
        """Text that is shared with every downstream agent by design."""
        texts: list[str] = [f.claim for f in bus.facts]
        texts += [f.claim for f in output.facts]
        for flag in list(bus.risk_flags) + list(output.risk_flags):
            texts += [flag.title, flag.detail]
        for question in list(bus.unresolved) + list(output.unresolved):
            texts += [question.question, question.why_it_matters]
        for contradiction in list(bus.contradictions) + list(output.contradictions):
            texts += [
                contradiction.description,
                contradiction.left_summary,
                contradiction.right_summary,
            ]
        return texts

    def _run_agent(
        self,
        agent: Agent,
        guard: IsolationGuard,
        result: ResearchResult,
        *,
        params: dict[str, Any],
        user_preferences: dict[str, Any] | None = None,
        facts: Sequence | None = None,
    ) -> AgentOutput:
        ctx = result.context
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            agent_input = guard.project(
                agent.agent_id,
                ticker=ctx.ticker,
                company_name=ctx.company_name,
                aliases=params.get("aliases", ()),
                facts=facts,
                params={**params, "run_id": ctx.run_id},
                user_preferences=user_preferences,
            )
        except LeakageError as exc:
            # An isolation breach is never survivable: continuing would produce
            # a contaminated verdict that looks identical to a clean one.
            log.error("isolation failure for %s: %s", agent.agent_id, exc)
            raise

        started_monotonic = time.monotonic()
        output = agent.execute(agent_input)
        elapsed = time.monotonic() - started_monotonic
        if elapsed > agent.timeout_seconds:
            output.degraded = True
            output.errors.append(
                f"agent exceeded its {agent.timeout_seconds}s budget ({elapsed:.1f}s)"
            )

        # Recompute the evaluation fingerprint against everything that is
        # legitimately shared. An agent's risk flags and unresolved questions
        # travel downstream by design (requirement 1D), so prose that also
        # appears there is not evidence of a leak -- only prose unique to the
        # quarantined evaluation is diagnostic.
        if output.evaluation is not None:
            output.evaluation.fingerprint_tokens = derive_fingerprints(
                [output.evaluation.summary, *output.evaluation.points],
                self._shared_baseline(guard.bus, output),
            )

        record = self._record(output, ctx.run_id, started)
        record.inputs_hash = inputs_hash(agent_input)
        result.agent_records.append(record)
        self.repo.save_agent_run(record)

        if not output.ok or output.degraded:
            message = f"{agent.agent_id}: {output.status} :: {'; '.join(output.errors)[:300]}"
            result.failures.append(message)
            log.warning("agent degraded/failed: %s", message)
            if not output.ok:
                ctx.status = RunStatus.INCOMPLETE_RESEARCH

        # publish
        if output.facts:
            guard.bus.add_facts(output.facts)
        if output.sources:
            guard.bus.add_sources(output.sources)
        if output.risk_flags:
            guard.bus.risk_flags.extend(output.risk_flags)
        if output.contradictions:
            guard.bus.contradictions.extend(output.contradictions)
        if output.unresolved:
            guard.bus.unresolved.extend(output.unresolved)
        if output.evaluation:
            guard.bus.publish(output.evaluation)
        return output

    # -- main --------------------------------------------------------------
    def run(
        self,
        ticker: str,
        company_name: str,
        collection_results: Sequence[CollectionResult],
        *,
        mode: str = "standard",
        price: float | None = None,
        aliases: Sequence[str] = (),
        user_preferences: dict[str, Any] | None = None,
        offline: bool = True,
        run_id: str | None = None,
        resume: bool = False,
    ) -> ResearchResult:
        ctx = RunContext(
            run_id=run_id or new_run_id(),
            ticker=ticker.upper(),
            company_name=company_name,
            mode=mode,
            offline=offline,
            use_fixtures=any(r.provenance == Provenance.FIXTURE for r in collection_results),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        bus = EvidenceBus()
        guard = IsolationGuard(bus, strict=self.strict_isolation)
        result = ResearchResult(
            context=ctx,
            bus=bus,
            verdict=None,
            scorecard=None,
            collection_results=list(collection_results),
        )

        # ---- Resume (requirement P8) --------------------------------------
        resume_plan = None
        if resume:
            checkpoints = self.repo.checkpoints(ctx.run_id)
            restored = self.repo.facts_for_resume(ctx.run_id)
            resume_plan = build_resume_plan(ctx.run_id, checkpoints, restored, today=self.today)
            result.resume_plan = resume_plan
            ctx.notes.append(f"resume: {resume_plan.reason}")
            log.info("resume plan for %s: %s", ctx.run_id, resume_plan.reason)

        self.repo.upsert_company(ctx.ticker, company_name, aliases=list(aliases))
        self.repo.start_run(ctx)

        for collection in collection_results:
            for url in collection.attempted_urls:
                self.repo.log_fetch(
                    ctx.run_id,
                    url,
                    collection.collector,
                    str(collection.outcome),
                    detail="; ".join(collection.errors)[:400],
                )

        self._identity = (ctx.ticker, company_name, tuple(aliases))
        result.capture_info = dict(self.capture_info)
        result.adversarial = self.adversarial
        result.chunks = list(self.chunks)
        if self.llm is not None:
            result.llm_budget = self.llm.budget
        if self.adversarial is not None and self.adversarial.discovery.queries:
            # Requirement M3: every query and hit is persisted, purpose intact,
            # separately from facts -- this is what makes the search layer
            # auditable after the run rather than only during it.
            self.repo.save_discovery_log(self.adversarial.discovery)

        params: dict[str, Any] = {
            "price": price,
            "aliases": tuple(aliases),
            "mode": mode,
        }
        def checkpoint(stage: str, payload: dict[str, Any] | None = None) -> None:
            self.repo.save_checkpoint(
                Checkpoint(
                    run_id=ctx.run_id,
                    stage=stage,
                    stage_index=stage_index(stage),
                    status="OK",
                    payload=payload or {},
                    fact_count=len(bus.facts),
                )
            )

        # ---- Stage 1: collect (no evaluation) ---------------------------
        if resume_plan is not None and not resume_plan.should_run("collect"):
            # Collection is the expensive stage; a resumed run reuses the facts
            # it already obtained, and they keep their original run_id and
            # provenance so the report still shows when each was collected.
            log.info(
                "resume: skipping collection, reusing %d fact(s)",
                len(resume_plan.restored_facts),
            )
            bus.add_facts(resume_plan.restored_facts)
            verified_facts = list(resume_plan.restored_facts)
            ctx.notes.append(f"resumed with {len(verified_facts)} previously-collected fact(s)")
            result.failures.append(
                "RESUMED RUN: collection and verification were not re-executed; "
                f"{len(verified_facts)} fact(s) were restored from run {ctx.run_id}"
            )
            collector_output = None
        else:
            collector_output = self._run_agent(
                FactCollectorAgent(collection_results),
                guard,
                result,
                params=params,
                user_preferences=user_preferences,
            )
        raw_facts = list(collector_output.facts) if collector_output else []

        # ---- Stage 2: verify --------------------------------------------
        if collector_output is not None:
            integrity = self._run_agent(
                EvidenceIntegrityAgent(today=self.today, stale_after_days=self.stale_after_days),
                guard,
                result,
                params=params,
                user_preferences=user_preferences,
                facts=raw_facts,
            )
            verified_facts = list(integrity.facts) or raw_facts
        # The verified set replaces the raw set on the bus.
        bus.facts = verified_facts

        # A malformed source is rejected and quarantined, never silently
        # persisted and never allowed to crash the run (requirement 14/24).
        quarantined_sources = self.repo.save_sources(bus.sources)
        if quarantined_sources:
            result.quarantined_sources = quarantined_sources
            ctx.status = RunStatus.INCOMPLETE_RESEARCH
            for q in quarantined_sources:
                result.failures.append(
                    f"quarantined malformed source {q.source_id} ({q.url}): "
                    f"{q.field}={q.value!r} -- {q.error}"
                )
            # Any fact resting only on a quarantined source is evidence this
            # run cannot actually stand behind -- exclude it before it can
            # reach scoring, completeness or the Evidence Sufficiency Matrix,
            # so a domain that depended on it correctly reads as unresolved
            # rather than silently sufficient.
            quarantined_ids = {q.source_id for q in quarantined_sources}
            before = len(verified_facts)
            verified_facts = [f for f in verified_facts if f.source_id not in quarantined_ids]
            bus.facts = verified_facts
            dropped = before - len(verified_facts)
            if dropped:
                result.failures.append(
                    f"{dropped} fact(s) resting only on a quarantined source were excluded "
                    "from evidence"
                )

        for fact in verified_facts:
            try:
                self.repo.save_fact(fact)
            except Exception as exc:  # noqa: BLE001
                result.failures.append(f"fact persistence failed for {fact.fact_id}: {exc}")
                ctx.status = RunStatus.INCOMPLETE_RESEARCH

        checkpoint("collect", {"collectors": [c.collector for c in collection_results]})
        checkpoint("verify", {"verified": len(verified_facts)})

        # ---- Stage 2b: primary-source escalation (requirement P4) --------
        # Stage-aware budgeting (requirement: discovery must not starve later
        # stages): escalation gets its own quota, independent of whatever
        # adversarial discovery already spent.
        if self.llm is not None:
            self.llm.budget.set_stage("escalation")
        usable_research, research_reason = self.research.available()
        if usable_research:
            verified_facts, escalation = escalate(
                verified_facts, self.research, company=company_name
            )
            bus.facts = verified_facts
            result.escalation = escalation
            self.repo.save_escalations(ctx.run_id, escalation.attempts)
            for fact in verified_facts:
                # Escalation rewrites verified_status and confidence, so the
                # updated versions are persisted. Failures here were already
                # reported when the fact was first written.
                with contextlib.suppress(Exception):
                    self.repo.save_fact(fact)
            if escalation.unconfirmed:
                result.failures.append(
                    f"escalation: {len(escalation.unconfirmed)} material claim(s) could not be "
                    "confirmed in a primary source"
                )
        else:
            result.failures.append(f"escalation skipped: {research_reason}")
        checkpoint(
            "escalate", {"attempts": len(result.escalation.attempts) if result.escalation else 0}
        )

        # ---- Stage 3: domain agents (facts only) ------------------------
        # Stage-aware budgeting: everything from here through bear/bull
        # (Stage 6) shares the "interpretive" quota, independent of discovery
        # and escalation's spend.
        if self.llm is not None:
            self.llm.budget.set_stage("interpretive")
        from ..agents.llm_agents import (
            LLMCompetitiveAgent,
            LLMContradictionAgent,
            LLMRegulatoryAgent,
            LLMScienceAgent,
        )
        from ..agents.llm_agents2 import (
            LLMBearAgent,
            LLMBlindJudgeAgent,
            LLMBullAgent,
            LLMKillAgent,
        )

        for deterministic, llm_cls in (
            (RegulatoryAgent(), LLMRegulatoryAgent),
            (CapitalStructureAgent(), None),  # arithmetic stays deterministic
            (ScienceAgent(), LLMScienceAgent),
            (CompetitiveAgent(), LLMCompetitiveAgent),
            (CatalystAgent(today=self.today), None),  # date normalisation is rule-based
            (MicrostructureAgent(), None),  # positioning data is not interpretive
        ):
            agent = self._agent_for(deterministic.agent_id, deterministic, llm_cls, bus)
            output = self._run_agent(
                agent, guard, result, params=params, user_preferences=user_preferences
            )
            if output.metrics.get("llm_backed"):
                result.llm_agents_used.append(deterministic.agent_id)
        checkpoint("domain")

        # ---- Stage 3b: unresolved-question-driven escalation (requirement C) --
        # The Stage 2b escalation pass above only ever looks at Fact objects.
        # A material unresolved question (e.g. "is the primary endpoint
        # acceptable to the regulator?") is only raised by a domain agent
        # HERE, in Stage 3 -- so it could never have been escalated earlier.
        # Same "escalation" budget stage/quota as Stage 2b; this reuses
        # whatever of it remains, it does not get a second allowance.
        if self.llm is not None:
            self.llm.budget.set_stage("escalation")
        usable_research_now, _ = self.research.available()
        new_facts: list = []
        if usable_research_now and bus.unresolved:
            new_facts, question_escalation = escalate_unresolved_questions(
                bus.unresolved, bus.sources, self.research, company=company_name,
                ticker=ctx.ticker, run_id=ctx.run_id, facts=verified_facts,
            )
            if new_facts:
                verified_facts = [*verified_facts, *new_facts]
                bus.add_facts(new_facts)
                for fact in new_facts:
                    with contextlib.suppress(Exception):
                        self.repo.save_fact(fact)
            if result.escalation is not None:
                result.escalation.attempts.extend(question_escalation.attempts)
                result.escalation.fetches_attempted += question_escalation.fetches_attempted
                result.escalation.fetches_failed += question_escalation.fetches_failed
            else:
                result.escalation = question_escalation
            if question_escalation.attempts:
                self.repo.save_escalations(ctx.run_id, question_escalation.attempts)
        checkpoint("escalate_unresolved", {"new_facts": len(new_facts)})

        # ---- Stage 4: contradictions ------------------------------------
        contradiction_output = self._run_agent(
            self._agent_for("contradiction", ContradictionAgent(), LLMContradictionAgent, bus),
            guard,
            result,
            params=params,
            user_preferences=user_preferences,
        )
        if contradiction_output.metrics.get("llm_backed"):
            result.llm_agents_used.append("contradiction")
        self.repo.save_contradictions(bus.contradictions)
        checkpoint("contradiction")

        # ---- Stage 5: KILL BEFORE BULL (requirement 27) ------------------
        # The LLM kill agent proposes findings; the deterministic gate below
        # computes the binding assessment from the same evidence. A model can
        # surface a lethal fact, but it cannot argue a kill level down.
        executed_queries = tuple(
            r.query.query
            for r in (
                [*self.adversarial.bear_results, *self.adversarial.bull_results]
                if self.adversarial
                else []
            )
            if r.executed
        )
        # The live research provider (e.g. AnthropicWebResearchProvider under
        # --live) is passed through so the Kill Agent's mandatory searches
        # actually execute against it, instead of the unrelated offline
        # SearchProvider (Tavily/Brave/none) it used to be limited to
        # (requirement E). Both KillAgent instances below share one
        # DiscoveryLog -- the same one adversarial discovery already writes to
        # when it ran -- so mandatory-kill-query records land in the same
        # auditable, purpose-tagged store rather than a second, disconnected
        # one; exactly one of the two instances ever actually searches.
        kill_discovery = self.adversarial.discovery if self.adversarial is not None else DiscoveryLog()
        llm_kill = self._agent_for(
            "kill_agent",
            KillAgent(
                self.search,
                executed_queries=executed_queries,
                research=self.research,
                discovery=kill_discovery,
            ),
            LLMKillAgent,
            bus,
        )
        kill_output = self._run_agent(
            llm_kill, guard, result, params=params, user_preferences=user_preferences
        )
        if kill_output.metrics.get("llm_backed"):
            result.llm_agents_used.append("kill_agent")
            deterministic_kill = self._run_agent(
                KillAgent(
                    self.search,
                    executed_queries=executed_queries,
                    research=self.research,
                    discovery=kill_discovery,
                ),
                guard,
                result,
                params=params,
                user_preferences=user_preferences,
            )
            gate = _gate_from_output(deterministic_kill)
        else:
            gate = _gate_from_output(kill_output)
        self.repo.save_kill_gate(ctx.run_id, ctx.ticker, gate)
        if kill_discovery.queries:
            self.repo.save_discovery_log(kill_discovery)
            if result.adversarial is None:
                # No adversarial pass ran this time (e.g. --adversarial was not
                # passed), but the Kill Agent's own mandatory queries still
                # produced an auditable discovery log -- surface it the same
                # way regardless, rather than only when adversarial happened
                # to run too.
                result.adversarial = AdversarialOutcome(discovery=kill_discovery)

        # ---- Stage 6: bear and bull, mutually blind ----------------------
        # Order matters only for reproducibility; neither can see the other.
        # In kill-test mode the bull case is not built at all: the question
        # being asked is "is there a reason to discard this", and constructing
        # a case for it would only invite the reader to weigh one against the
        # other, which is exactly the trade this system refuses to make.
        if mode != "catalyst":
            bear = self._agent_for("bear_agent", BearAgent(), LLMBearAgent, bus)
            bear_output = self._run_agent(
                bear, guard, result, params=params, user_preferences=user_preferences
            )
            if bear_output.metrics.get("llm_backed"):
                result.llm_agents_used.append("bear_agent")
        if mode not in ("kill-test", "catalyst"):
            bull = self._agent_for("bull_agent", BullAgent(), LLMBullAgent, bus)
            bull_output = self._run_agent(
                bull, guard, result, params=params, user_preferences=user_preferences
            )
            if bull_output.metrics.get("llm_backed"):
                result.llm_agents_used.append("bull_agent")
        checkpoint("bear_bull")

        # ---- Stage 7: valuation ------------------------------------------
        self._run_agent(
            ValuationAgent(), guard, result, params=params, user_preferences=user_preferences
        )

        # ---- Stage 8: confidence, scores, scenarios ----------------------
        capital_payload = (
            bus.channels[Channel.CAPITAL_STRUCTURE].payload
            if Channel.CAPITAL_STRUCTURE in bus.channels
            else {}
        )
        catalyst_payload = (
            bus.channels[Channel.CATALYSTS].payload if Channel.CATALYSTS in bus.channels else {}
        )
        regulatory_payload = (
            bus.channels[Channel.REGULATORY].payload if Channel.REGULATORY in bus.channels else {}
        )

        collector_failures = sum(1 for c in collection_results if c.degraded)
        breakdown = compute_evidence_confidence(
            verified_facts,
            list(bus.unresolved),
            unsearched_categories=gate.unsearched_categories,
            collector_failures=collector_failures,
            used_fixtures=ctx.use_fixtures,
        )
        result.confidence_breakdown = breakdown

        catalysts = catalyst_payload.get("events", [])
        near_term = any(h in ("T0", "T1", "T2") for h in (e.get("horizon") for e in catalysts))
        result.catalysts = catalysts

        card = build_scorecard(
            ticker=ctx.ticker,
            run_id=ctx.run_id,
            channels=dict(bus.channels),
            kill_gate=gate,
            evidence_confidence=breakdown.score,
            contradiction_count=len(bus.contradictions),
            catalyst_count=len(catalysts),
            near_term_catalyst=near_term,
        )
        result.scorecard = card
        self.repo.save_scores(card)

        scenarios = build_scenarios(
            price=price,
            fully_diluted_shares=capital_payload.get("fully_diluted_shares"),
            basic_shares=capital_payload.get("basic_shares"),
            kill_gate=gate,
            evidence_confidence=breakdown.score,
            regulatory_position=regulatory_payload.get("endpoint_position", UNKNOWN),
            runway_months=capital_payload.get("runway_months"),
            bear_mechanisms=(
                bus.channels[Channel.BEAR].payload.get("mechanisms", [])
                if Channel.BEAR in bus.channels
                else []
            ),
            bull_points=(
                bus.channels[Channel.BULL].payload.get("points", [])
                if Channel.BULL in bus.channels
                else []
            ),
        )
        result.scenarios = scenarios
        self.repo.save_scenarios(ctx.run_id, ctx.ticker, scenarios)

        from ..schemas.agent_io import Evaluation

        bus.publish(
            Evaluation(
                author="pipeline",
                channel=Channel.SCENARIOS,
                summary=f"{len(scenarios)} scenarios computed.",
                payload={"scenarios": [s.to_row() for s in scenarios]},
            )
        )

        if capital_payload:
            self.repo.save_capital_structure(
                ctx.run_id,
                ctx.ticker,
                {
                    key: capital_payload.get(key)
                    for key in (
                        "basic_shares",
                        "fully_diluted_shares",
                        "cash",
                        "debt",
                        "quarterly_burn",
                        "runway_months",
                        "atm_capacity",
                    )
                }
                | {
                    "as_of": capital_payload.get("as_of", UNKNOWN),
                    "going_concern": str(capital_payload.get("going_concern", UNKNOWN)),
                    "unknown_fields": ",".join(capital_payload.get("unknown_fields", [])),
                },
            )

        # ---- Persist the structured domain tables ------------------------
        self._persist_domain_tables(ctx, bus, regulatory_payload, verified_facts)

        # ---- Stage 8b: search completeness gate (requirement P6) ---------
        # Assessed BEFORE the judgement, because whether a verdict may be issued
        # at all is a precondition of judging, not a caveat on the result.
        domain_fact_counts: dict[ResearchDomain, int] = {
            ResearchDomain.REGULATORY: len(
                [f for f in verified_facts if f.category is FactCategory.REGULATORY]
            ),
            ResearchDomain.CAPITAL_STRUCTURE: len(
                [
                    f
                    for f in verified_facts
                    if f.category
                    in (
                        FactCategory.CAPITAL_STRUCTURE,
                        FactCategory.FINANCIAL,
                        FactCategory.LIQUIDITY,
                    )
                ]
            ),
            ResearchDomain.SCIENCE_TECHNOLOGY: len(
                [
                    f
                    for f in verified_facts
                    if f.category
                    in (FactCategory.CLINICAL, FactCategory.SCIENCE, FactCategory.TECHNOLOGY)
                ]
            ),
            ResearchDomain.COMPETITION: len(
                [
                    f
                    for f in verified_facts
                    if f.category in (FactCategory.COMPETITION, FactCategory.MARKET_SIZE)
                ]
            ),
            ResearchDomain.CATALYST: len(catalysts),
            ResearchDomain.CONTRADICTION: len(bus.contradictions),
        }
        search_results = []
        if self.adversarial is not None:
            search_results = [*self.adversarial.bear_results, *self.adversarial.bull_results]
        agents_run = {
            record.agent_id for record in result.agent_records if record.status != "FAILED"
        }
        completeness = assess_completeness(
            search_results=search_results,
            facts_by_domain=domain_fact_counts,
            agents_run=agents_run,
        )
        result.completeness = completeness
        self.repo.save_research_coverage(ctx.run_id, ctx.ticker, completeness.coverage)
        if completeness.blocked:
            ctx.status = RunStatus.INCOMPLETE_RESEARCH
            result.failures.append(f"search completeness gate: {completeness.reason()}")
        checkpoint("score", {"blocked": completeness.blocked})

        # ---- Stage 8c: Decision-Grade Evidence Gate ----------------------
        # Searched and verified are different achievements (requirement DG5).
        # Computed even when the completeness gate already blocks, so the
        # report can say both: an unsearched domain is one failure mode, six
        # fully-searched domains full of unread search summaries is another.
        # Unresolved questions about *which categories were never searched* are
        # already the Search Completeness Gate's job (and, for kill categories
        # specifically, the UNSEARCHED rationale on the assessment itself); only
        # a genuinely unresolved factual question -- one that is not simply
        # "we never ran this query" -- belongs here as a material claim.
        material_claims = tuple(
            uq.question
            for uq in bus.unresolved
            if uq.blocking and "never searched" not in uq.question
        )
        sufficiency = assess_evidence_sufficiency(
            facts=verified_facts,
            completeness=completeness,
            gate=gate,
            unresolved_material_claims=material_claims,
        )
        result.evidence_sufficiency = sufficiency
        self.repo.save_evidence_sufficiency(ctx.run_id, ctx.ticker, sufficiency)

        # ---- Stage 9: blind judgement ------------------------------------
        if self.llm is not None:
            self.llm.budget.set_stage("finalization")
        judge_params = {
            **params,
            "evidence_confidence": breakdown.score,
            "run_status": (
                RunStatus.INCOMPLETE_RESEARCH.value
                if (result.failures or ctx.status != RunStatus.COMPLETE)
                else RunStatus.COMPLETE.value
            ),
        }
        judge_output = self._run_agent(
            self._agent_for("blind_judge", BlindJudgeAgent(), LLMBlindJudgeAgent, bus),
            guard,
            result,
            params=judge_params,
            user_preferences=user_preferences,
        )
        if judge_output.metrics.get("llm_backed"):
            result.llm_agents_used.append("blind_judge")
        verdict = judge_output.metrics.get("_verdict_object")
        result.verdict = verdict
        result.source_ref_map = guard.source_ref_maps.get("blind_judge")

        # Requirement P6/DG1/DG6: an incomplete search OR evidence that has not
        # cleared the decision-grade bar withholds the action label entirely.
        # Not AVOID, not WAIT_FOR_EVENT -- WAIT_FOR_EVENT is itself an Action
        # and must never stand in for insufficient evidence.
        blocking_reasons: list[str] = []
        if completeness.blocked:
            blocking_reasons.append(completeness.reason())
        if not sufficiency.sufficient:
            blocking_reasons.extend(sufficiency.blocking_reasons())

        if verdict is not None:
            if blocking_reasons:
                verdict.blocked = True
                verdict.blocked_reason = " | ".join(blocking_reasons)
                verdict.action = None
                verdict.research_status = ResearchStatus.BLOCKED_PENDING_VERIFICATION
                verdict.blocking_verification_required = tuple(blocking_reasons)
                verdict.caveats = (
                    *verdict.caveats,
                    "RESEARCH_STATUS: BLOCKED_PENDING_VERIFICATION -- FINAL_ACTION: NONE -- "
                    + " | ".join(blocking_reasons),
                )
                # Requirement G: nulling .action is not enough -- the Blind
                # Judge (deterministic or LLM-backed) already baked its
                # internally selected Action into free text (e.g. a headline
                # reading "... action WAIT_FOR_EVENT.") before this was known.
                # No output surface -- headline, reasoning, thesis breakers,
                # red flags, caveats, the report, the JSON export -- may carry
                # an actionable label once BLOCKED_PENDING_VERIFICATION holds.
                verdict.headline = blocked_headline(blocking_reasons)
                verdict.reasoning = tuple(scrub_action_labels(r) for r in verdict.reasoning)
                verdict.thesis_breakers = tuple(
                    scrub_action_labels(b) for b in verdict.thesis_breakers
                )
                verdict.critical_red_flags = tuple(
                    scrub_action_labels(f) for f in verdict.critical_red_flags
                )
                verdict.caveats = tuple(scrub_action_labels(c) for c in verdict.caveats)
            elif result.failures or ctx.status != RunStatus.COMPLETE:
                verdict.research_status = ResearchStatus.INCOMPLETE
            else:
                verdict.research_status = ResearchStatus.COMPLETE
        checkpoint("judge")

        if result.failures and ctx.status == RunStatus.COMPLETE:
            ctx.status = RunStatus.INCOMPLETE_RESEARCH
        if verdict is not None:
            verdict.run_status = ctx.status
            ctx.notes.append(f"action={verdict.action}")

        # ---- Stage 10: portfolio (the ONLY step that sees holdings) -------
        # Deliberately after the verdict is fixed: knowing that a position is
        # held, and at what price, is exactly what turns research into
        # rationalisation. It is admitted only once it can no longer change the
        # judgement.
        if user_preferences:
            portfolio_output = self._run_agent(
                PortfolioAgent(),
                guard,
                result,
                params=params,
                user_preferences=user_preferences,
            )
            if portfolio_output.evaluation:
                result.portfolio_guidance = portfolio_output.evaluation.payload

        # ---- Stage 11: thesis versioning ---------------------------------
        if verdict is not None and card is not None:
            previous = self.repo.latest_thesis(ctx.ticker)
            previous_fact_ids: set[str] = set()
            if previous is not None:
                import json as _json

                try:
                    previous_fact_ids = set(
                        _json.loads(previous["snapshot"] or "{}").get("fact_ids", [])
                    )
                except Exception:  # noqa: BLE001
                    previous_fact_ids = set()
            current_fact_ids = {f.fact_id for f in verified_facts}
            diff = diff_against_previous(
                previous, verdict, card, current_fact_ids, previous_fact_ids
            )
            result.thesis_diff = diff
            result.thesis_version = self.repo.save_thesis_version(
                ctx.ticker,
                ctx.run_id,
                verdict,
                card,
                diff,
                snapshot_for(card, verdict, current_fact_ids),
            )

        if result.catalysts:
            from ..schemas.evaluation import CatalystEvent

            events = [
                CatalystEvent(
                    event_id=e["event_id"],
                    horizon=e["horizon"],
                    date_jst=e["date_jst"],
                    date_confidence=e["date_confidence"],
                    event=e["event"],
                    expected_outcome=e["expected_outcome"],
                    bull_outcome=e["bull_outcome"],
                    bear_outcome=e["bear_outcome"],
                    market_pricing=e["market_pricing"],
                    information_source=e["information_source"],
                    fact_ids=tuple(str(e.get("fact_ids", "")).split(","))
                    if e.get("fact_ids")
                    else (),
                )
                for e in result.catalysts
            ]
            self.repo.save_catalysts(ctx.run_id, ctx.ticker, events)

        # ---- Traceability index (requirement P7) --------------------------
        result.traceability = build_index(verified_facts, bus.sources, self.chunks)

        if self.llm is not None and self.llm.budget.calls:
            self.repo.save_llm_calls(ctx.run_id, self.llm.budget.calls)

        checkpoint("report")
        self.repo.finish_run(ctx)
        return result


_NCT_RE = re.compile(r"\b(NCT[-A-Z0-9]{4,})\b")


def _gate_from_output(output: AgentOutput) -> KillGateResult:
    payload = output.evaluation.payload if output.evaluation else {}
    return KillGateResult.from_channel_payload(payload)


def _clinical_trials_from_facts(facts: Sequence[Any]) -> list[dict[str, Any]]:
    """Group clinical facts by registry id into one row per trial.

    Fields with no evidence stay UNKNOWN rather than being inferred from the
    other fields of the same trial.
    """
    by_nct: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if fact.category is not FactCategory.CLINICAL:
            continue
        match = _NCT_RE.search(fact.claim)
        if not match:
            continue
        nct = match.group(1)
        row = by_nct.setdefault(
            nct,
            {
                "nct_id": nct,
                "title": UNKNOWN,
                "phase": UNKNOWN,
                "status": UNKNOWN,
                "enrollment": None,
                "randomized": UNKNOWN,
                "blinding": UNKNOWN,
                "control_arm": UNKNOWN,
                "primary_endpoint": UNKNOWN,
                "secondary_endpoints": "",
                "primary_completion": UNKNOWN,
                "sponsor": UNKNOWN,
                "fact_ids": [],
            },
        )
        row["fact_ids"].append(fact.fact_id)
        claim = fact.claim.lower()
        value = fact.value if fact.value != UNKNOWN else None

        if "phase is" in claim or "is a phase" in claim:
            phase = re.search(r"phase\s+(?:is\s+)?(?:phase\s*)?([0-4](?:/[0-4])?[ab]?)", claim)
            if phase:
                row["phase"] = phase.group(1)
        if "overall status is" in claim and value:
            row["status"] = str(value)
        if "enrollment is" in claim:
            number = re.search(r"enrollment is (\d+)", claim)
            if number:
                row["enrollment"] = int(number.group(1))
        if "allocation is" in claim:
            allocation = re.search(r"allocation is ([a-z\-]+)", claim)
            masking = re.search(r"masking is ([a-z]+)", claim)
            if allocation:
                row["randomized"] = allocation.group(1).upper()
            if masking:
                row["blinding"] = masking.group(1).upper()
        if "primary outcome measure is" in claim or "primary endpoint" in claim:
            row["primary_endpoint"] = fact.claim.split(":", 1)[-1].strip()[:500]
        if "primary completion date is" in claim and value:
            row["primary_completion"] = str(value)

    for row in by_nct.values():
        row["fact_ids"] = ",".join(row["fact_ids"])
    return list(by_nct.values())


def _insider_trades_from_facts(facts: Sequence[Any]) -> list[dict[str, Any]]:
    """One row per insider-transaction fact.

    Deliberately shallow: without Form 4 XML the individual, the role and the
    transaction code are usually not in the text, and they stay UNKNOWN rather
    than being guessed from context.
    """
    trades: list[dict[str, Any]] = []
    for fact in facts:
        if fact.category is not FactCategory.INSIDER:
            continue
        shares = None
        match = re.search(r"([\d,]{4,})\s+shares", fact.claim)
        if match:
            shares = float(match.group(1).replace(",", ""))
        trades.append(
            {
                "trade_id": "trade_" + hashlib.sha256(fact.fact_id.encode()).hexdigest()[:16],
                "insider": UNKNOWN,
                "role": "officer" if "officer" in fact.claim.lower() else UNKNOWN,
                "transaction_date": fact.event_date,
                "transaction_code": (
                    "S"
                    if "sale" in fact.claim.lower() or "sold" in fact.claim.lower()
                    else ("P" if "purchase" in fact.claim.lower() else UNKNOWN)
                ),
                "shares": shares,
                "price": None,
                "is_10b5_1": ("true" if "10b5-1" in fact.claim.lower() else UNKNOWN),
                "source_url": fact.source_url,
            }
        )
    return trades
