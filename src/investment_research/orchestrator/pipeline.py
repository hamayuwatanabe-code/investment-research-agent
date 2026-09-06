"""The research pipeline.

Stage order is fixed and is itself a requirement (27): the kill test runs before
the bull case, and the blind judgement runs before anything that knows the
ticker or the user's position.

Partial failure policy (requirement 14): a failing agent degrades the run rather
than aborting it, and the run is then marked ``INCOMPLETE_RESEARCH``.  It is
never reported as a successful analysis.
"""

from __future__ import annotations

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
from ..collectors.search import SearchProvider
from ..schemas.agent_io import AgentOutput, AgentRunRecord, RunContext
from ..schemas.enums import (
    UNKNOWN,
    FactCategory,
    KillCategory,
    KillLevel,
    Provenance,
    RunStatus,
)
from ..schemas.evaluation import KillAssessment, KillGateResult, ScoreCard, Verdict
from ..scoring.evidence_confidence import compute_evidence_confidence
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
    ) -> None:
        self.repo = repository
        self.search = search
        self.today = today or date.today()
        self.strict_isolation = strict_isolation
        self.stale_after_days = stale_after_days

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

        params: dict[str, Any] = {
            "price": price,
            "aliases": tuple(aliases),
            "mode": mode,
        }

        # ---- Stage 1: collect (no evaluation) ---------------------------
        collector_output = self._run_agent(
            FactCollectorAgent(collection_results),
            guard,
            result,
            params=params,
            user_preferences=user_preferences,
        )
        raw_facts = list(collector_output.facts)

        # ---- Stage 2: verify --------------------------------------------
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
        self.repo.save_sources(bus.sources)
        for fact in verified_facts:
            try:
                self.repo.save_fact(fact)
            except Exception as exc:  # noqa: BLE001
                result.failures.append(f"fact persistence failed for {fact.fact_id}: {exc}")
                ctx.status = RunStatus.INCOMPLETE_RESEARCH

        # ---- Stage 3: domain agents (facts only) ------------------------
        for agent in (
            RegulatoryAgent(),
            CapitalStructureAgent(),
            ScienceAgent(),
            CompetitiveAgent(),
            CatalystAgent(today=self.today),
            MicrostructureAgent(),
        ):
            self._run_agent(agent, guard, result, params=params, user_preferences=user_preferences)

        # ---- Stage 4: contradictions ------------------------------------
        self._run_agent(
            ContradictionAgent(), guard, result, params=params, user_preferences=user_preferences
        )
        self.repo.save_contradictions(bus.contradictions)

        # ---- Stage 5: KILL BEFORE BULL (requirement 27) ------------------
        kill_output = self._run_agent(
            KillAgent(self.search), guard, result, params=params, user_preferences=user_preferences
        )
        gate = _gate_from_output(kill_output)
        self.repo.save_kill_gate(ctx.run_id, ctx.ticker, gate)

        # ---- Stage 6: bear and bull, mutually blind ----------------------
        # Order matters only for reproducibility; neither can see the other.
        # In kill-test mode the bull case is not built at all: the question
        # being asked is "is there a reason to discard this", and constructing
        # a case for it would only invite the reader to weigh one against the
        # other, which is exactly the trade this system refuses to make.
        if mode != "catalyst":
            self._run_agent(
                BearAgent(), guard, result, params=params, user_preferences=user_preferences
            )
        if mode not in ("kill-test", "catalyst"):
            self._run_agent(
                BullAgent(), guard, result, params=params, user_preferences=user_preferences
            )

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

        # ---- Stage 9: blind judgement ------------------------------------
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
            BlindJudgeAgent(),
            guard,
            result,
            params=judge_params,
            user_preferences=user_preferences,
        )
        verdict = judge_output.metrics.get("_verdict_object")
        result.verdict = verdict
        result.source_ref_map = guard.source_ref_maps.get("blind_judge")

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

        self.repo.finish_run(ctx)
        return result


_NCT_RE = re.compile(r"\b(NCT[-A-Z0-9]{4,})\b")


def _gate_from_output(output: AgentOutput) -> KillGateResult:
    payload = output.evaluation.payload if output.evaluation else {}
    assessments = tuple(
        KillAssessment(
            category=KillCategory(row["category"]),
            level=KillLevel(row["level"]),
            rationale=row.get("rationale", ""),
            evidence_confidence=float(row.get("evidence_confidence", 0.0)),
        )
        for row in payload.get("kill_gate", [])
    )
    unsearched = tuple(KillCategory(c) for c in payload.get("unsearched_categories", []))
    return KillGateResult(assessments=assessments, unsearched_categories=unsearched)


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
