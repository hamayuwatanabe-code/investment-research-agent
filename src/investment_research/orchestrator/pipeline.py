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
from collections.abc import Mapping, Sequence
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
from ..schemas.fact import Fact
from ..schemas.validation import QuarantinedSource
from ..scoring.adaptive_acquisition_plan import (
    AdaptiveAcquisitionPlan,
    build_adaptive_acquisition_plan,
)
from ..scoring.completeness import CompletenessResult, assess_completeness
from ..scoring.decision_gate_consistency import enforce_complete_only_action, sync_verdict_channel
from ..scoring.evidence_confidence import compute_evidence_confidence
from ..scoring.evidence_sufficiency import EvidenceSufficiencyMatrix, assess_evidence_sufficiency
from ..scoring.program_evidence import (
    CompanyIdentityEvidence,
    EvidenceValidationContext,
    LiteratureCandidateEvidence,
    ProgramCandidateEvidence,
)
from ..scoring.program_identity_resolution import (
    LiteratureLinkResolution,
    LiteratureLinkStatus,
    ProgramIdentityResolution,
    ProgramIdentityStatus,
    resolve_literature_link,
    resolve_program_identity,
)
from ..scoring.program_resolution import ProgramResolution, resolve_current_program
from ..scoring.scenarios import build_scenarios
from ..scoring.scores import build_scorecard
from ..storage.repository import (
    CollectSnapshot,
    Repository,
    ResumeSnapshotCorrupted,
    VerifySnapshot,
)
from ..thesis.versioning import diff_against_previous, snapshot_for
from .adaptive_literature_step import run_adaptive_literature_step
from .evidence_integrity_pass import (
    FullIntegrityPassInput,
    IntegrityPassKind,
    run_full_evidence_integrity_pass,
)
from .isolation import (
    Channel,
    EvidenceBus,
    IsolationGuard,
    LeakageError,
    derive_fingerprints,
)
from .resume import Checkpoint, ResumePlan, resume_point, stage_index
from .resume import build_plan as build_resume_plan

log = logging.getLogger(__name__)

#: Must equal ``research.literature_evidence_projection.BRIDGE_COLLECTOR_
#: LABEL`` exactly. Duplicated here (not imported) deliberately, mirroring
#: ``acquisition_executor.MAX_WEB_SEARCH_USES``'s own precedent: this
#: module must never gain an import edge onto the Literature Acquisition
#: path (Phase 4.1A/4.1B's own tests assert ``literature_evidence_
#: projection``/``literature_chunk_projection``/``project_literature_*``
#: never appear in ``pipeline_module.__dict__``) merely to borrow one
#: string constant used for classification only -- this module never calls
#: any acquisition code itself, whatever a run's ``collection_results``
#: happen to contain. A dedicated test asserts the two stay equal.
_LITERATURE_BRIDGE_COLLECTOR_LABEL = "literature_evidence_projection"


def _literature_incomplete_blocking_reasons(
    collection_results: Sequence[CollectionResult],
    direct_acquisition_info: dict[str, Any] | None = None,
) -> list[str]:
    """Phase 4.2A correction 1: see the call site's own comment. Pure and
    read-only -- never imports or calls anything from the Literature
    Acquisition path; reads only ``CollectionResult.collector``/
    ``.degraded``/``.describe()`` (already generic, existing fields) and
    ``direct_acquisition_info`` (an already-sanitized plain ``dict`` the
    caller built via ``literature_pipeline_integration.bundle_diagnostics``
    -- this function invents no new string, no new sanitization, and reads
    no field ``bundle_diagnostics`` does not already expose).

    Two failure shapes the original Phase 4.2A check missed, both meaning
    "an explicitly-requested Literature acquisition did not complete",
    exactly like a degraded ``CollectionResult`` does:

    A. Evidence Projection succeeded (``CollectionResult.degraded`` is
       ``False``, so the loop below finds nothing) but Chunk Projection
       failed -- ``direct_acquisition_info["coverage_complete"]`` is
       ``False``.
    B. The defensive credential re-check inside
       ``run_literature_pipeline_acquisition`` refused before any request
       -- ``direct_acquisition_info["refused"]`` is ``True`` and no
       ``CollectionResult`` was ever appended to ``collection_results`` at
       all (there is nothing for the loop below to see).

    Dedup (never double-report the SAME incompleteness twice): the
    ``direct_acquisition_info`` branch is skipped entirely once the
    collector loop already found a degraded literature ``CollectionResult``
    -- a degraded evidence fetch and an incomplete chunk projection for the
    SAME run are one failure, not two.

    Fires only when ``direct_acquisition_info.get("feature_enabled")`` is
    true -- i.e. never for a run that did not pass
    ``--document-first-literature`` at all (``direct_acquisition_info`` is
    then either absent or ``{"feature_enabled": False, ...}``), and never
    for a clean, complete fetch (``refused`` False, ``coverage_complete``
    True).
    """
    reasons: list[str] = []
    degraded_seen = False
    for result in collection_results:
        if result.collector == _LITERATURE_BRIDGE_COLLECTOR_LABEL and result.degraded:
            degraded_seen = True
            reasons.append(f"Literature Document-First acquisition did not complete: {result.describe()}")

    info = direct_acquisition_info or {}
    if not info.get("feature_enabled"):
        return reasons
    if info.get("refused"):
        reason = info.get("refused_reason") or "refused before any request (no reason recorded)"
        reasons.append(f"Literature Document-First acquisition was refused: {reason}")
    elif info.get("coverage_complete") is False and not degraded_seen:
        unresolved = "; ".join(info.get("unresolved_reasons") or ()) or "no reason recorded"
        reasons.append(
            f"Literature Document-First acquisition did not reach full coverage: {unresolved}"
        )
    return reasons


def _select_bootstrap_company_identity(
    identities: Sequence[CompanyIdentityEvidence],
) -> CompanyIdentityEvidence | None:
    """Phase 4.3G cardinality gate for ``resolve_program_identity()``'s own
    single-``CompanyIdentityEvidence`` contract: ``result.
    company_identity_evidence`` is a LIST (Phase 4.3F flattens it across
    every ``CollectionResult`` a run had), but the resolver takes exactly
    one. Zero or two-or-more records is never resolved by inventing one,
    nor by an arbitrary/first/last pick -- both return ``None`` here, and
    the caller treats that as UNRESOLVED. A Production SEC collector
    today reports at most one, but this boundary enforces the rule
    regardless of how many collectors a future run has."""
    if len(identities) != 1:
        return None
    return identities[0]


def _bootstrap_company_identity_cardinality_rationale(
    identities: Sequence[CompanyIdentityEvidence],
) -> str:
    if not identities:
        return "no CompanyIdentityEvidence was produced this run"
    return (
        f"{len(identities)} CompanyIdentityEvidence records were produced this run; "
        "never resolved by an arbitrary/first/last selection"
    )


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
    #: Phase 4.2A: safe (body/secret-free) diagnostics from an explicit-
    #: reference acquisition path the caller ran before ``Pipeline.run()``
    #: (today, only ``research/literature_pipeline_integration.py``'s
    #: Literature Document-First bridge) -- counts and booleans only, never
    #: a URL, a document/chunk body, or a raw ``ExecutionReport``. Mirrors
    #: ``capture_info``'s own shape/threading exactly (constructor ->
    #: ``self.direct_acquisition_info`` -> ``result.direct_acquisition_info``).
    #: Empty for every run that did not use such a path -- existing callers
    #: are entirely unaffected.
    direct_acquisition_info: dict[str, Any] = field(default_factory=dict)
    #: Phase 4.3C correction 5: non-empty only when a resume genuinely
    #: needed a Stage-1 (collect) or Stage-2 (verify) snapshot and could
    #: not get one intact -- mirrors ``direct_acquisition_info``'s own
    #: pattern (populated early, read once near Blind Judge to extend
    #: ``blocking_reasons``) so this failure withholds ``Verdict.action``
    #: the same way an incomplete Literature acquisition already does,
    #: never silently completing as if nothing were missing.
    resume_snapshot_failures: list[str] = field(default_factory=list)
    #: Phase 4.3F: structured, typed primary-source identity/candidate
    #: metadata (scoring/program_evidence.py), flattened across every
    #: CollectionResult this run had -- populated from a fresh collect or
    #: restored from the collect snapshot on resume (see Stage 1 below),
    #: mirroring quarantined_sources'/escalation's own "visible on the
    #: result object for audit" pattern. This phase does NOT validate,
    #: resolve, or deliver any of this to an Agent, and it plays no role in
    #: Action gating -- purely a lossless, typed carrier.
    company_identity_evidence: list[CompanyIdentityEvidence] = field(default_factory=list)
    program_candidate_evidence: list[ProgramCandidateEvidence] = field(default_factory=list)
    literature_candidate_evidence: list[LiteratureCandidateEvidence] = field(default_factory=list)
    #: Phase 4.3G: the BOOTSTRAP Program Identity / Literature Link
    #: resolution and the resulting Adaptive Acquisition Plan -- computed
    #: once, after Stage 2's full Evidence Integrity pass completes
    #: (quarantine and quarantine-cascade fact exclusion already applied)
    #: and before Stage 2b's primary-source escalation runs (see Pipeline.
    #: run()'s own comment at that exact point). Plain internal diagnostic
    #: fields, exactly like company_identity_evidence/program_candidate_
    #: evidence/literature_candidate_evidence above: never delivered to
    #: any AgentInput (by any path, including params), never folded into
    #: blocking_reasons/RunStatus/Verdict.action, never executed --
    #: scoring/adaptive_acquisition_plan.py computes a PLAN only, and
    #: nothing in this module (or that one) calls an executor, adapter,
    #: HttpClient, or LLM. None only before this computation has run at
    #: all (never reached on any real invocation of Pipeline.run()).
    bootstrap_program_identity: ProgramIdentityResolution | None = None
    bootstrap_literature_link: LiteratureLinkResolution | None = None
    adaptive_acquisition_plan: AdaptiveAcquisitionPlan | None = None
    #: Phase 4.3I: safe (body/secret/URL-free) diagnostics from the
    #: Adaptive Literature Acquisition step -- a SEPARATE typed carrier
    #: from direct_acquisition_info (which belongs exclusively to the
    #: existing EXPLICIT PMID/NCT path and is never mixed with this).
    #: Empty for every run where the step was never attempted or never
    #: applicable, exactly like direct_acquisition_info's own pattern.
    adaptive_literature_diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Non-empty exactly when the Adaptive Literature step's own outcome
    #: must withhold Verdict.action (REFUSED/INCOMPLETE/
    #: AMBIGUOUS_RESUME_STATE/SNAPSHOT_CORRUPTED) -- read once, near Blind
    #: Judge, to extend blocking_reasons, mirroring resume_snapshot_
    #: failures'/the Literature path's own existing pattern exactly.
    adaptive_literature_blocking_reasons: list[str] = field(default_factory=list)

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
        agent_effort_policy: dict[str, str] | None = None,
        direct_acquisition_info: dict[str, Any] | None = None,
        adaptive_http_client: Any | None = None,
        adaptive_env: Mapping[str, str] | None = None,
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
        self.direct_acquisition_info = direct_acquisition_info or {}
        # Phase 4.3I: the Adaptive Literature Acquisition step's own
        # runner injection -- both default None, which is Production's own
        # default and takes the NOT_ATTEMPTED_NO_RUNNER branch
        # unconditionally (see orchestrator/adaptive_literature_step.py).
        # No CLI flag or real transport wires these in this phase; a
        # caller (today, only this repository's own test suite) injects a
        # Fake transport directly.
        self.adaptive_http_client = adaptive_http_client
        self.adaptive_env = adaptive_env
        # Requirement G: configurable per-agent effort policy (never
        # hard-coded into agent logic). None means "use
        # llm.effort_policy.DEFAULT_AGENT_EFFORT_POLICY" -- resolve_effort()
        # itself falls back to that default when this is None.
        self.agent_effort_policy = agent_effort_policy

    def _stage(self, name: str) -> Any:
        """Scoped LLM-budget stage context (requirement A).

        A no-op context manager when no LLM is configured, so every call
        site can wrap its stage-owned work unconditionally rather than
        branching on ``self.llm is None``. When an LLM IS configured, this
        delegates to ``LLMBudget.stage()``, which restores whatever stage
        was active before the ``with`` block on exit -- including when the
        block raises -- so one stage's work can never leak into the next
        stage that runs after it, the way ambient ``set_stage()`` calls used
        to (Stage 3b's "escalation" call was never reset before Contradiction/
        Kill/Bear/Bull, so all four ran misattributed to "escalation").
        """
        if self.llm is None:
            return contextlib.nullcontext()
        return self.llm.budget.stage(name)

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
        # Requirement G: per-agent effort, resolved from the configurable
        # policy (never hard-coded into the agent class itself) against this
        # run's --llm-effort ceiling.
        from ..llm.effort_policy import resolve_effort

        effort = resolve_effort(
            deterministic.agent_id, ceiling=self.llm.effort, policy=self.agent_effort_policy
        )
        return llm_cls(
            self.llm,
            fallback=deterministic,
            guard=PromptGuard(denied_fingerprints=denied, forbidden_identity=forbidden),
            chunks=chunks,
            discovery_summary=discovery_summary,
            effort=effort,
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
        program_resolution: ProgramResolution | None = None,
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
                program_resolution=program_resolution,
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
        # Phase 4.3C correction 5: which snapshot (if any) is needed is
        # decided BEFORE build_plan runs, from checkpoints alone
        # (resume_point) -- collect/pristine when resuming before "verify",
        # verify/verified when resuming after it, none at all for a fresh
        # start. Never facts_for_resume/a MIN- or MAX-version query
        # (removed -- both were proven vulnerable to the SAME cross-run
        # content collision; see ResumePlan's own docstring in
        # orchestrator/resume.py for the empirical reproduction). A
        # snapshot that is needed but missing or corrupted is never
        # silently treated as "nothing to restore": resume_plan.
        # snapshot_unavailable_reason carries the reason forward to a
        # fail-closed outcome (status=INCOMPLETE_RESEARCH, Action=None,
        # a specific blocking reason -- see the Stage 1/Stage 2 handling
        # below and the blocking_reasons computation near Blind Judge).
        resume_plan = None
        collect_snapshot: CollectSnapshot | None = None
        verify_snapshot: VerifySnapshot | None = None
        if resume:
            checkpoints = self.repo.checkpoints(ctx.run_id)
            needed = resume_point(checkpoints)
            snapshot_facts: Sequence[Fact] = ()
            snapshot_missing_reason: str | None = None
            if needed == stage_index("verify"):
                try:
                    collect_snapshot = self.repo.collect_snapshot_for_resume(ctx.run_id)
                except ResumeSnapshotCorrupted as exc:
                    snapshot_missing_reason = str(exc)
                else:
                    if collect_snapshot is None:
                        snapshot_missing_reason = (
                            f"no collect snapshot recorded for run_id={ctx.run_id!r}; "
                            "cannot safely re-evaluate this run's Stage-1 facts without one"
                        )
                    else:
                        snapshot_facts = collect_snapshot.facts
            elif needed > stage_index("verify"):
                try:
                    verify_snapshot = self.repo.verify_snapshot_for_resume(ctx.run_id)
                except ResumeSnapshotCorrupted as exc:
                    snapshot_missing_reason = str(exc)
                else:
                    if verify_snapshot is None:
                        snapshot_missing_reason = (
                            f"no verify snapshot recorded for run_id={ctx.run_id!r}; "
                            "cannot safely reuse this run's already-verified facts without one"
                        )
                    else:
                        snapshot_facts = verify_snapshot.verified_facts
                # Phase 4.3F correction 2: CollectSnapshot is REQUIRED here
                # too -- correction 1's "best-effort" fetch was itself the
                # defect this correction fixes. It is the ONLY source of
                # Sources and structured evidence (company_identity_
                # evidence/program_candidate_evidence/
                # literature_candidate_evidence) for a run resumed past
                # verify, exactly as it already is the only source of
                # Facts+Sources+structured evidence when resuming before
                # verify (the `needed == stage_index("verify")` branch
                # above). A missing or corrupted CollectSnapshot here fails
                # the resume closed through the SAME snapshot_missing_
                # reason -> resume_plan.snapshot_unavailable_reason ->
                # result.resume_snapshot_failures -> ctx.status=
                # INCOMPLETE_RESEARCH/Action=None path VerifySnapshot
                # failure already uses (see the fail-closed handling below
                # and near Blind Judge) -- this never falls back to
                # treating the structured evidence as merely empty, and
                # never reconstructs it from this invocation's (possibly
                # absent, possibly different) fresh collection_results.
                # Only the FIRST failure reason is kept (verify_snapshot's,
                # if it already failed), so a double failure still reports
                # one clear reason rather than overwriting it.
                try:
                    collect_snapshot = self.repo.collect_snapshot_for_resume(ctx.run_id)
                except ResumeSnapshotCorrupted as exc:
                    collect_snapshot = None
                    if snapshot_missing_reason is None:
                        snapshot_missing_reason = str(exc)
                else:
                    if collect_snapshot is None and snapshot_missing_reason is None:
                        snapshot_missing_reason = (
                            f"no collect snapshot recorded for run_id={ctx.run_id!r}; "
                            "cannot safely restore this run's Sources and structured "
                            "evidence without one"
                        )
            # needed == 0 (a fresh start): no snapshot is needed at all.

            resume_plan = build_resume_plan(
                ctx.run_id,
                checkpoints,
                snapshot_facts=snapshot_facts,
                snapshot_missing_reason=snapshot_missing_reason,
                today=self.today,
            )
            result.resume_plan = resume_plan
            ctx.notes.append(f"resume: {resume_plan.reason}")
            log.info("resume plan for %s: %s", ctx.run_id, resume_plan.reason)
            if resume_plan.snapshot_unavailable_reason:
                message = f"RESUME SNAPSHOT UNAVAILABLE: {resume_plan.snapshot_unavailable_reason}"
                result.resume_snapshot_failures.append(message)
                result.failures.append(message)
                log.error(message)

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
        result.direct_acquisition_info = dict(self.direct_acquisition_info)
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
        skip_collect = resume_plan is not None and not resume_plan.should_run("collect")
        snapshot_unavailable = resume_plan is not None and resume_plan.snapshot_unavailable_reason is not None
        if skip_collect:
            assert resume_plan is not None  # narrows for mypy; skip_collect implies this
            # Collection is the expensive stage; a resumed run reuses the facts
            # it already obtained, and they keep their original run_id and
            # provenance so the report still shows when each was collected.
            # Phase 4.3C correction 2: whether these restored facts still
            # need to go through Evidence Integrity, or are already the
            # verified set from an earlier invocation of this run_id, is
            # decided below in Stage 2 from resume_plan.should_run("verify")
            # -- this branch only restores what "collect" itself checkpoints
            # and persists; it does not assume what kind of facts these are.
            log.info(
                "resume: skipping collection, reusing %d fact(s)",
                len(resume_plan.restored_facts),
            )
            bus.add_facts(resume_plan.restored_facts)
            # Phase 4.3C correction 5: Source objects come from THIS SAME
            # snapshot the facts themselves came from when
            # needed==stage_index("verify") (i.e. resume_plan.restored_
            # facts is itself collect_snapshot.facts) -- never the
            # freshly-recollected `collection_results` (Phase 4.3C
            # correction 3's own fix, now superseded): that would mix THIS
            # invocation's newly-fetched Sources with a PRIOR invocation's
            # Facts, which is exactly the "mixing" this correction's own
            # requirements prohibit -- live Source content can differ
            # between fetches even when the fact set does not.
            #
            # Phase 4.3F correction 1/2: when resuming PAST "verify" instead
            # (needed > stage_index("verify")), resume_plan.restored_facts
            # comes from verify_snapshot, never from collect_snapshot -- but
            # collect_snapshot is ALSO fetched at that resume point (the
            # resume-plan-building block above, correction 2: REQUIRED, not
            # best-effort -- its own absence or corruption there already
            # fails the whole resume closed via snapshot_missing_reason),
            # carrying the run's structured evidence and the Sources it
            # references, which were being silently dropped before
            # correction 1. Reaching this line with collect_snapshot still
            # None therefore only happens when it genuinely was not needed
            # (a fresh start, or resuming before "verify" via the other
            # branch above) -- never a swallowed failure. Adding those
            # Sources into bus.sources here is exactly as safe as a FRESH
            # (non-resumed) run already is:
            # bus.sources is populated straight from collection_results at
            # Stage 1 there too, pre-quarantine, un-filtered by Stage 2 --
            # quarantine exclusion has always been the responsibility of a
            # caller consulting result.quarantined_sources separately
            # (confirmed by program_evidence.EvidenceValidationContext's own
            # documented caller contract), never something bus.sources
            # itself guarantees. This branch introduces no new exposure
            # beyond what a fresh run already has.
            if collect_snapshot is not None:
                bus.add_sources(collect_snapshot.sources)
                # Restored from the SAME snapshot the Facts/Sources above
                # came from -- never recomputed from THIS invocation's
                # (possibly absent) collection_results, for the identical
                # reason Sources aren't: a resumed run must never mix a
                # prior invocation's structured evidence with anything this
                # invocation happens to have collected fresh.
                result.company_identity_evidence = list(collect_snapshot.company_identity_evidence)
                result.program_candidate_evidence = list(collect_snapshot.program_candidate_evidence)
                result.literature_candidate_evidence = list(
                    collect_snapshot.literature_candidate_evidence
                )
            ctx.notes.append(
                f"resumed with {len(resume_plan.restored_facts)} previously-collected fact(s)"
            )
            result.failures.append(
                "RESUMED RUN: collection was not re-executed; "
                f"{len(resume_plan.restored_facts)} fact(s) were restored from run {ctx.run_id}"
            )
            collector_output = None
            pre_integrity_facts: list[Fact] = []
        else:
            collector_output = self._run_agent(
                FactCollectorAgent(collection_results),
                guard,
                result,
                params=params,
                user_preferences=user_preferences,
            )
            pre_integrity_facts = list(collector_output.facts)
            # Phase 4.3C correction 2: persisted here, before Evidence
            # Integrity runs, so a crash between collection and verify
            # leaves genuinely pre-Integrity facts recoverable on --resume.
            for fact in pre_integrity_facts:
                with contextlib.suppress(Exception):
                    self.repo.save_fact(fact)
            # Phase 4.3F: flattened across every CollectionResult this run
            # had, exactly like bus.sources already is -- never re-derived
            # later from anything else (a claim string, a note).
            result.company_identity_evidence = [
                c.company_identity_evidence
                for c in collection_results
                if c.company_identity_evidence is not None
            ]
            result.program_candidate_evidence = [
                e for c in collection_results for e in c.program_candidate_evidence
            ]
            result.literature_candidate_evidence = [
                e for c in collection_results for e in c.literature_candidate_evidence
            ]
            # Phase 4.3C correction 5: the run-scoped snapshot resume
            # actually depends on -- see ResumePlan's own docstring in
            # orchestrator/resume.py for why the `facts` table rows just
            # written above (individually, for other purposes -- post-hoc
            # inspection, existing `latest_facts`-based tooling) are NOT
            # what a resumed run reads back.
            self.repo.save_collect_snapshot(
                ctx.run_id,
                pre_integrity_facts,
                list(bus.sources),
                [
                    {
                        "collector": c.collector,
                        "outcome": str(c.outcome),
                        "provenance": str(c.provenance),
                        "errors": list(c.errors),
                        "attempted_urls": list(c.attempted_urls),
                        "notes": list(c.notes),
                        "zero_results": c.zero_results,
                        "raw_fact_count_before_dedup": c.raw_fact_count_before_dedup,
                    }
                    for c in collection_results
                ],
                company_identity_evidence=result.company_identity_evidence,
                program_candidate_evidence=result.program_candidate_evidence,
                literature_candidate_evidence=result.literature_candidate_evidence,
            )

        checkpoint("collect", {"collectors": [c.collector for c in collection_results]})

        # ---- Stage 2: verify (Phase 4.3C / correction 2) ------------------
        # Phase 4.3C correction 2: whether Evidence Integrity re-runs at all
        # is a resume-orchestration decision made HERE, from
        # resume_plan.should_run("verify") -- never inside
        # run_full_evidence_integrity_pass, which (correction 1) always
        # executes the agent for real whenever it is actually called.
        # Resuming after "collect" but before "verify" restores genuinely
        # pre-Integrity facts (Phase 4.3C correction 5: from the collect
        # snapshot, never a MIN(version) query) and this pass IS called on
        # them; resuming after "verify" already completed restores
        # already-verified facts (from the verify snapshot, never a
        # MAX(version) query) and this pass is NOT called again -- zero
        # further EvidenceIntegrityAgent executions for those facts in this
        # invocation.
        skip_verify = resume_plan is not None and not resume_plan.should_run("verify")
        if snapshot_unavailable:
            # Phase 4.3C correction 5: fail closed. A snapshot this resume
            # NEEDED could not be obtained (see the resume-plan-building
            # block above and result.resume_snapshot_failures) -- there is
            # nothing trustworthy to re-evaluate or to reuse, so this never
            # falls back to an empty-but-otherwise-normal pass, and never
            # silently re-collects behind --resume's own back. verified_
            # facts stays empty; ctx.status/blocking_reasons/Action=None
            # are handled once, uniformly, near Blind Judge below (the
            # same place Literature's own incompleteness is handled) from
            # result.resume_snapshot_failures.
            verified_facts: list[Fact] = []
            bus.facts = verified_facts
        elif skip_verify:
            assert resume_plan is not None  # narrows for mypy; skip_verify implies this
            verified_facts = list(resume_plan.restored_facts)
            bus.facts = verified_facts
            ctx.notes.append(
                f"resumed with {len(verified_facts)} already-verified fact(s); "
                "Evidence Integrity was not re-executed"
            )
            result.failures.append(
                "RESUMED RUN: Evidence Integrity was not re-executed; "
                f"{len(verified_facts)} already-verified fact(s) were restored from "
                f"run {ctx.run_id}"
            )
        else:
            if skip_collect:
                assert resume_plan is not None  # narrows for mypy; skip_collect implies this
                # Phase 4.3C correction 5: resume_plan.restored_facts IS
                # already the collect snapshot's own pristine facts here
                # (see the resume-plan-building block above) -- no separate
                # pristine/latest distinction or per-fact fallback remains.
                pre_integrity_facts = list(resume_plan.restored_facts)
            # The "full Evidence Integrity pass" -- EvidenceIntegrityAgent
            # execution, verified_facts determination, Source persistence/
            # quarantine, quarantine-cascade fact exclusion, and the Phase
            # 4.2B-correction-1 Literature completeness-consistency check --
            # extracted to orchestrator.evidence_integrity_pass as one
            # unit, so the identical sequence can also run safely as a
            # SECOND pass in an offline test harness (never in production,
            # never connected here to Adaptive Acquisition). Production
            # runs it at most once per invocation, always as
            # IntegrityPassKind.INITIAL, and (Phase 4.3C correction 1)
            # this call always executes EvidenceIntegrityAgent for real
            # when made -- there is no bypass.
            pass_input = FullIntegrityPassInput(
                ticker=ctx.ticker,
                company_name=company_name,
                run_id=ctx.run_id,
                pre_integrity_facts=tuple(pre_integrity_facts),
                sources=tuple(bus.sources),
                collection_results=tuple(collection_results),
                direct_acquisition_info=result.direct_acquisition_info,
                today=self.today,
                stale_after_days=self.stale_after_days,
                pass_kind=IntegrityPassKind.INITIAL,
            )
            pass_output = run_full_evidence_integrity_pass(pass_input, self.repo)

            verified_facts = list(pass_output.verified_facts)
            bus.facts = verified_facts
            result.quarantined_sources = list(pass_output.quarantined_sources)
            result.failures.extend(pass_output.failures)
            if pass_output.status_incomplete:
                ctx.status = RunStatus.INCOMPLETE_RESEARCH
            result.direct_acquisition_info = dict(pass_output.direct_acquisition_info)
            result.agent_records.append(pass_output.agent_run_record)
            # Phase 4.3C correction 5: the run-scoped snapshot a LATER
            # resume of THIS run_id (past this point) would need.
            self.repo.save_verify_snapshot(
                ctx.run_id, verified_facts, pass_output.quarantined_sources,
                result.direct_acquisition_info,
            )

        if not snapshot_unavailable:
            # Phase 4.3C correction 5: never checkpointed "verify" OK on
            # the fail-closed path -- a snapshot this resume needed was
            # missing/corrupted, so verify genuinely did NOT happen this
            # invocation. Leaving "verify" un-checkpointed (still just
            # "collect") means a LATER --resume of this SAME run_id is
            # correctly judged to still need the collect snapshot (if that
            # one is intact, this is a real recovery path) rather than
            # perpetually re-discovering a missing VERIFY snapshot it can
            # never produce on its own.
            checkpoint("verify", {"verified": len(verified_facts)})

        # ---- Phase 4.3G: bootstrap Program Identity / Literature Link /
        # Adaptive Acquisition Plan ----------------------------------------
        # Computed exactly ONCE per run, HERE -- after Stage 2's full
        # Evidence Integrity pass completes (quarantine and quarantine-
        # cascade fact exclusion already applied to verified_facts/
        # result.quarantined_sources above) and BEFORE Stage 2b's
        # primary-source escalation runs below. Never after Stage 2b:
        # escalation's additional facts have NOT been through a full
        # Evidence Integrity pass (see program_identity_resolution.py's
        # own module docstring on why Stage 3b's escalate_unresolved_
        # questions pattern -- appending facts without a second full pass
        # -- must never become the model for what Program Identity is
        # resolved against). This is deliberately a SEPARATE computation
        # from the central resolve_current_program() call below: that one
        # is unchanged (still runs once, after Stage 2b, for Science/Kill)
        # and resolve_program_identity()/resolve_literature_link() are a
        # different contract entirely -- never confused, never merged.
        #
        # Pure, side-effect-free, no new checkpoint/DB write (Phase 4.3G
        # requirement: resume never needs a NEW snapshot for this -- a
        # resumed run simply recomputes it from whichever already-
        # snapshotted verified_facts/sources/structured-evidence it has
        # at this exact point, identically to a fresh run given the same
        # inputs). On the fail-closed resume path (snapshot_unavailable),
        # verified_facts/bus.sources/result.quarantined_sources/the three
        # structured-evidence lists are already all empty by construction
        # (see the resume-plan-building block and Stage 1 above) -- this
        # naturally yields UNRESOLVED/UNRESOLVED/a no-target, zero-budget
        # plan below, with no special-casing needed here.
        bootstrap_context = EvidenceValidationContext(
            sources_by_id={
                source.source_id: source
                for source in bus.sources
                if source.source_id not in {q.source_id for q in result.quarantined_sources}
            },
            verified_facts_by_id={fact.fact_id: fact for fact in verified_facts},
        )
        chosen_identity = _select_bootstrap_company_identity(result.company_identity_evidence)
        if chosen_identity is None:
            bootstrap_identity = ProgramIdentityResolution(
                status=ProgramIdentityStatus.UNRESOLVED,
                rationale=_bootstrap_company_identity_cardinality_rationale(
                    result.company_identity_evidence
                ),
            )
        else:
            bootstrap_identity = resolve_program_identity(
                chosen_identity, result.program_candidate_evidence, bootstrap_context
            )
        if bootstrap_identity.status is ProgramIdentityStatus.CONFIRMED:
            # Phase 4.3G requirement 6: resolve_literature_link() is called
            # ONLY in this branch. A malformed/non-strict nct_id here would
            # already be unreachable by resolve_program_identity()'s own
            # CONFIRMED contract, and resolve_literature_link() itself
            # degrades safely to UNRESOLVED on one anyway -- the INDEPENDENT
            # defensive re-check lives in build_adaptive_acquisition_plan()
            # below (REFUSED), never duplicated here.
            bootstrap_literature = resolve_literature_link(
                bootstrap_identity.nct_id, result.literature_candidate_evidence, bootstrap_context
            )
        else:
            bootstrap_literature = LiteratureLinkResolution(
                status=LiteratureLinkStatus.UNRESOLVED,
                rationale=(
                    "program identity is not CONFIRMED; literature link resolution was not "
                    "attempted (Phase 4.3G requirement 6)"
                ),
            )
        result.bootstrap_program_identity = bootstrap_identity
        result.bootstrap_literature_link = bootstrap_literature
        result.adaptive_acquisition_plan = build_adaptive_acquisition_plan(
            bootstrap_identity, bootstrap_literature, result.direct_acquisition_info
        )

        # ---- Phase 4.3I: Adaptive Literature Acquisition -----------------
        # Placed HERE -- immediately after the Phase 4.3G plan above is
        # computed, before Stage 2b escalation -- so that (a) Stage 2b's
        # escalate() and the central resolve_current_program() call below
        # both operate on the EXPANDED verified_facts set once this step
        # has run, and (b) this step itself sees the SAME verified_facts/
        # bus.sources Stage 2b would otherwise have seen first. Safe to
        # call on every invocation, fresh or resumed: resume-state is
        # judged first, from persisted checkpoints alone, inside
        # run_adaptive_literature_step() itself -- never gated by
        # resume_plan.should_run(...), which only ever covers "collect"/
        # "verify" (see that module's own docstring for why this step
        # needs its own, independent resume gate).
        adaptive_result = run_adaptive_literature_step(
            repo=self.repo,
            run_id=ctx.run_id,
            ticker=ctx.ticker,
            company_name=company_name,
            plan=result.adaptive_acquisition_plan,
            verified_facts=verified_facts,
            sources=bus.sources,
            base_evidence_available=not snapshot_unavailable,
            http_client=self.adaptive_http_client,
            env=self.adaptive_env,
            today=self.today,
            stale_after_days=self.stale_after_days,
        )
        result.adaptive_literature_diagnostics = dict(adaptive_result.diagnostics)
        result.adaptive_literature_blocking_reasons = list(adaptive_result.blocking_reasons)
        if adaptive_result.verified_facts:
            verified_facts = list(adaptive_result.verified_facts)
            bus.facts = verified_facts
        if adaptive_result.new_chunks:
            self.chunks = [*self.chunks, *adaptive_result.new_chunks]
            # result.chunks was already snapshotted from self.chunks at
            # Stage 1 (before this step could possibly have run) -- kept
            # in sync here so report.py's own "Evidence was chunked into
            # N chunk(s)" banner and any other result.chunks reader see
            # the SAME set Domain Agents/TraceabilityIndex now read from
            # self.chunks, on a fresh run or a resume alike.
            result.chunks = list(self.chunks)
        if adaptive_result.quarantined_sources:
            result.quarantined_sources = [
                *result.quarantined_sources, *adaptive_result.quarantined_sources,
            ]
        result.agent_records.extend(adaptive_result.agent_records)
        for reason in adaptive_result.failures:
            if reason not in result.failures:
                result.failures.append(reason)

        # ---- Stage 2b: primary-source escalation (requirement P4) --------
        # Stage-aware budgeting (requirement: discovery must not starve later
        # stages): escalation gets its own quota, independent of whatever
        # adversarial discovery already spent. Scoped so this stage can never
        # leak into whatever runs after it (requirement A).
        with self._stage("escalation"):
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
                        f"escalation: {len(escalation.unconfirmed)} material claim(s) could not "
                        "be confirmed in a primary source"
                    )
            else:
                result.failures.append(f"escalation skipped: {research_reason}")
            checkpoint(
                "escalate",
                {"attempts": len(result.escalation.attempts) if result.escalation else 0},
            )

        # ---- Central Program Resolution (Phase 4.3D) ----------------------
        # Computed exactly ONCE per run, from the SAME verified_facts Domain
        # Agents are about to receive -- after Stage 2 Evidence Integrity and
        # Stage 2b primary-source escalation both complete, before Stage 3
        # starts. ScienceAgent, LLMScienceAgent and KillAgent used to each
        # independently call resolve_current_program() themselves (relying on
        # both seeing near-identical fact sets to reach the same answer); they
        # no longer do -- see IsolationGuard.project(), the only place this
        # value is attached to an AgentInput, and only for agent_ids "science"
        # and "kill_agent". Stage 3b's unresolved-question-driven escalation
        # runs AFTER Science/Kill and is deliberately out of scope here:
        # splitting an initial vs. a final resolution is a future phase, once
        # Adaptive Acquisition exists to act on newly-escalated facts. A
        # resumed run needs no special-casing: verified_facts above is
        # already either the freshly-verified set or the restored one by this
        # point, so this is naturally "decided once from the restored facts".
        program_resolution = resolve_current_program(list(verified_facts))

        # ---- Stage 3: domain agents (facts only) ------------------------
        # Stage-aware budgeting: everything from here through bear/bull
        # (Stage 6) shares the "interpretive" quota, independent of discovery
        # and escalation's spend.
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

        with self._stage("interpretive"):
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
                    agent,
                    guard,
                    result,
                    params=params,
                    user_preferences=user_preferences,
                    program_resolution=(
                        program_resolution if deterministic.agent_id == "science" else None
                    ),
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
        # whatever of it remains, it does not get a second allowance. Scoped
        # (requirement A) so it restores "interpretive" on exit -- including
        # on an exception -- rather than leaking into Contradiction/Kill/
        # Bear/Bull the way the old ambient set_stage() call did.
        with self._stage("escalation"):
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
        with self._stage("interpretive"):
            contradiction_output = self._run_agent(
                self._agent_for(
                    "contradiction", ContradictionAgent(), LLMContradictionAgent, bus
                ),
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
        # Requirement A: Kill's mandatory web searches get their own explicit
        # "discovery" stage/quota rather than inheriting whatever stage ran
        # immediately before Stage 5 (previously "escalation", leaked from
        # Stage 3b). This covers both places the actual searches can happen:
        # the LLMKillAgent's internal fallback to the deterministic KillAgent
        # (when the LLM call itself fails) and the explicit deterministic_kill
        # run below (when the LLM call succeeds).
        with self._stage("discovery"):
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
                llm_kill,
                guard,
                result,
                params=params,
                user_preferences=user_preferences,
                program_resolution=program_resolution,
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
                    program_resolution=program_resolution,
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
        with self._stage("interpretive"):
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
            collection_results=collection_results,
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
            # Per-domain sufficiency (requirement: a domain must never read
            # SUFFICIENT while it still has its own unresolved BLOCKING
            # question) needs the full, categorized unresolved-question set --
            # `material_claims` above is a flat, domain-agnostic tuple that
            # only ever blocks the matrix's OVERALL `sufficient` property.
            unresolved_questions=bus.unresolved,
        )
        result.evidence_sufficiency = sufficiency
        self.repo.save_evidence_sufficiency(ctx.run_id, ctx.ticker, sufficiency)

        # ---- Stage 9: blind judgement ------------------------------------
        with self._stage("finalization"):
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
        # Phase 4.2A (correction 1): strengthens (never weakens) the gate
        # above. An explicitly-requested Literature Document-First
        # acquisition (see research/literature_pipeline_integration.py)
        # that genuinely did not complete -- a provider failure, BLOCKED,
        # RATE_LIMITED, a real budget exclusion, a Chunk Projection
        # failure, or a pre-send credential refusal -- must never be
        # silently absorbed into a run that still emits an Action; the
        # caller opted into this evidence source explicitly, so its own
        # incompleteness is exactly the same kind of gap completeness.
        # blocked/sufficiency already withhold an Action for. Reads
        # ``self.direct_acquisition_info`` in addition to
        # ``collection_results`` (see ``_literature_incomplete_blocking_
        # reasons``'s own docstring for the two failure shapes a
        # CollectionResult-only check misses); never fires for a clean,
        # complete fetch, and never fires at all when the flag was never
        # used (``direct_acquisition_info`` then reads
        # ``feature_enabled=False``).
        # Phase 4.2A correction 2: kept in its own variable (never inlined
        # into the ``blocking_reasons.extend(...)`` call above) so the
        # RunContext/result-level propagation below can act on Literature's
        # OWN contribution specifically, without re-deriving it or
        # re-running the completeness/sufficiency checks. Held BEFORE the
        # ``if verdict is not None:`` block runs, so ``ctx.status``/
        # ``result.failures`` are already correct by the time
        # ``verdict.research_status``/``verdict.run_status`` are computed
        # from them below -- never a status assigned after the verdict
        # that already read the stale value.
        # Phase 4.2B correction 1: reads ``result.direct_acquisition_info``
        # (never ``self.direct_acquisition_info``) -- ``result``'s own copy
        # is the one the quarantine/Evidence-Integrity consistency check
        # above may have just amended (coverage_complete flipped False,
        # an unresolved reason appended); ``self.direct_acquisition_info``
        # stays the ORIGINAL, upstream-computed, constructor-supplied
        # value for the lifetime of this Pipeline instance and must never
        # be mutated in place (a caller could reuse the same instance).
        literature_blocking_reasons = _literature_incomplete_blocking_reasons(
            collection_results, result.direct_acquisition_info
        )
        blocking_reasons.extend(literature_blocking_reasons)
        if literature_blocking_reasons:
            # Without this, a run whose ONLY incompleteness is Literature's
            # own (completeness.blocked=False, sufficiency.sufficient=True,
            # result.failures otherwise empty) left ctx.status at COMPLETE
            # -- verdict.research_status/action were correctly withheld by
            # the blocking_reasons branch below, but verdict.run_status
            # (line ~1134, copied straight from ctx.status) and result.
            # incomplete/result.context.status all read COMPLETE, producing
            # exactly the contradiction (status=COMPLETE, research_status=
            # BLOCKED_PENDING_VERIFICATION, action=None) an audit found.
            ctx.status = RunStatus.INCOMPLETE_RESEARCH
            for reason in literature_blocking_reasons:
                if reason not in result.failures:  # never register the same reason twice
                    result.failures.append(reason)

        # Phase 4.3C correction 5: an unavailable resume snapshot withholds
        # the Action the same way an incomplete Literature acquisition does
        # -- same pattern, same placement (before verdict.research_status/
        # run_status are computed from ctx.status below), own dedicated
        # result field so this never depends on parsing result.failures'
        # free text to find it again.
        blocking_reasons.extend(result.resume_snapshot_failures)
        if result.resume_snapshot_failures:
            ctx.status = RunStatus.INCOMPLETE_RESEARCH

        # Phase 4.3I: an attempted-but-incomplete/refused Adaptive
        # Literature fetch, or a genuinely ambiguous/corrupted adaptive
        # resume state, withholds the Action the same way an incomplete
        # EXPLICIT Literature acquisition or an unavailable resume
        # snapshot already does -- same pattern, same placement. Never
        # fires for NOT_ATTEMPTED_*/COMPLETE (adaptive_literature_
        # blocking_reasons is empty in both cases).
        blocking_reasons.extend(result.adaptive_literature_blocking_reasons)
        if result.adaptive_literature_blocking_reasons:
            ctx.status = RunStatus.INCOMPLETE_RESEARCH

        if verdict is not None:
            if blocking_reasons:
                verdict.blocked = True
                verdict.blocked_reason = " | ".join(blocking_reasons)
                verdict.research_status = ResearchStatus.BLOCKED_PENDING_VERIFICATION
                verdict.blocking_verification_required = tuple(blocking_reasons)
                verdict.caveats = (
                    *verdict.caveats,
                    "RESEARCH_STATUS: BLOCKED_PENDING_VERIFICATION -- FINAL_ACTION: NONE -- "
                    + " | ".join(blocking_reasons),
                )
            elif result.failures or ctx.status != RunStatus.COMPLETE:
                verdict.research_status = ResearchStatus.INCOMPLETE
            else:
                verdict.research_status = ResearchStatus.COMPLETE

            # ResearchStatus.COMPLETE is the ONLY state that may emit an
            # Action (see ResearchStatus's own docstring: "the single gate
            # that decides whether Verdict.action may be non-None"). This
            # applies uniformly to BLOCKED_PENDING_VERIFICATION AND a plain
            # INCOMPLETE run (e.g. a degraded agent elsewhere) alike -- an
            # Action selected before this was known (including a prior
            # AVOID/BUY/WAIT_FOR_EVENT) must not survive, and a CONFIRMED K5
            # must still be reported prominently rather than hidden.
            enforce_complete_only_action(verdict, blocking_reasons)
            # The Blind Judge already published a (now stale) snapshot of its
            # action/headline/reasoning to the VERDICT channel before this
            # correction ran. The Portfolio agent (Stage 10) reads that
            # channel directly, after the verdict is finalized -- so the
            # published Evaluation must be corrected too, or the pre-
            # correction action leaks straight through to it.
            if judge_output.evaluation is not None:
                sync_verdict_channel(judge_output.evaluation, verdict)
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
