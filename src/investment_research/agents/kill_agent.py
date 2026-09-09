"""Agent 6: Kill Agent.

Its only job is to find reasons to discard the candidate.  It does not weigh,
balance or contextualize, and it never sees a bull case -- there is deliberately
nothing for it to defend (see the isolation policy).

Requirement 13/4-6: it runs the mandatory bear-side search set.  When no search
provider is configured the queries are still enumerated and reported as
**unexecuted**, and the affected kill categories come back UNSEARCHED rather
than K0.  Reporting "no problems found" for a search that never ran is the exact
failure this system exists to prevent.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from ..collectors.search import NullSearchProvider, SearchProvider, kill_queries
from ..collectors.tiering import classify_tier
from ..orchestrator.isolation import Channel
from ..research.discovery import DiscoveryLog, SearchQueryRecord, hits_from_documents, make_query_id
from ..research.provider import ResearchProvider, ResearchQuery
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import (
    MANDATORY_KILL_CATEGORIES,
    UNKNOWN,
    FactCategory,
    FetchOutcome,
    KillCategory,
    KillSearchFailureReason,
    Materiality,
    QueryPurpose,
    ResearchDomain,
)
from ..schemas.fact import UnresolvedQuestion
from ..scoring.kill_gate import evaluate_kill_gate
from ..scoring.program_resolution import resolve_current_program
from .base import Agent

log = logging.getLogger(__name__)

#: Kill category -> the research domain its mandatory queries fall under, for
#: DiscoveryLog bookkeeping only (auditability requirement E). Not the same
#: mapping as QUERY_CATEGORY_MAP's FactCategory routing below.
_KILL_CATEGORY_TO_RESEARCH_DOMAIN: dict[KillCategory, ResearchDomain] = {
    KillCategory.REGULATORY_KILL: ResearchDomain.REGULATORY,
    KillCategory.CLINICAL_KILL: ResearchDomain.SCIENCE_TECHNOLOGY,
    KillCategory.SCIENCE_KILL: ResearchDomain.SCIENCE_TECHNOLOGY,
    KillCategory.CAPITAL_KILL: ResearchDomain.CAPITAL_STRUCTURE,
    KillCategory.LIQUIDITY_KILL: ResearchDomain.CAPITAL_STRUCTURE,
    KillCategory.GOVERNANCE_KILL: ResearchDomain.CONTRADICTION,
    KillCategory.ACCOUNTING_KILL: ResearchDomain.CONTRADICTION,
    KillCategory.COMMERCIAL_KILL: ResearchDomain.COMPETITION,
}

#: Which mandatory query maps to which kill category, so an unexecuted query
#: marks exactly the categories it would have covered as UNSEARCHED.
QUERY_CATEGORY_MAP: dict[str, KillCategory] = {
    "FDA concern": KillCategory.REGULATORY_KILL,
    "regulatory risk": KillCategory.REGULATORY_KILL,
    "endpoint": KillCategory.REGULATORY_KILL,
    "Complete Response Letter": KillCategory.REGULATORY_KILL,
    "failed": KillCategory.CLINICAL_KILL,
    "clinical hold": KillCategory.CLINICAL_KILL,
    "dilution": KillCategory.CAPITAL_KILL,
    "going concern": KillCategory.CAPITAL_KILL,
    "offering priced": KillCategory.CAPITAL_KILL,
    "reverse split": KillCategory.CAPITAL_KILL,
    "delisting": KillCategory.LIQUIDITY_KILL,
    "lawsuit": KillCategory.GOVERNANCE_KILL,
    "insider selling": KillCategory.GOVERNANCE_KILL,
    "SEC investigation": KillCategory.GOVERNANCE_KILL,
    "accounting": KillCategory.ACCOUNTING_KILL,
    "auditor": KillCategory.ACCOUNTING_KILL,
    "restatement": KillCategory.ACCOUNTING_KILL,
    "short thesis": KillCategory.COMMERCIAL_KILL,
}


@dataclass
class KillQueryOutcome:
    """Per-mandatory-query audit record (requirement B).

    One of these is recorded for EVERY mandatory kill query, executed or
    not, so the report can say precisely why -- never collapsing budget
    starvation, a provider that is configured but currently unusable, and a
    genuine absence of any search capability into the same message.
    """

    query: str
    reason: KillSearchFailureReason
    #: The provider's own reported unavailability reason, or
    #: ``ResearchResult.error`` for an executed-but-failed call. Empty for a
    #: successful execution.
    detail: str = ""
    results_count: int = 0


class KillAgent(Agent):
    agent_id = "kill_agent"
    purpose = "Find disqualifying facts; never defend the candidate"

    def __init__(
        self,
        search: SearchProvider,
        *,
        max_hits_per_query: int = 5,
        executed_queries: tuple[str, ...] = (),
        research: ResearchProvider | None = None,
        discovery: DiscoveryLog | None = None,
    ) -> None:
        self.search = search
        self.max_hits = max_hits_per_query
        # Queries already executed by the adversarial search pass. Without this
        # the agent reports categories as UNSEARCHED that were in fact searched
        # through the research provider -- understating the work that was done
        # is as misleading as overstating it.
        self.executed_queries = tuple(executed_queries)
        # The SAME live research provider (e.g. AnthropicWebResearchProvider)
        # used by adversarial discovery and escalation. A live run reported
        # "no search provider configured" for all 23 mandatory kill queries
        # even under --live --llm, because this agent only ever knew about the
        # unrelated, offline `search: SearchProvider` (Tavily/Brave/none) --
        # never the provider `--live` actually wired up. When `research` is
        # given and usable it is preferred; `search` remains the fallback for
        # a run with no live research provider at all (requirement E).
        self.research = research
        self.discovery = discovery if discovery is not None else DiscoveryLog()
        self.queries_skipped_due_to_budget: list[str] = []
        #: One entry per mandatory query attempted this run (requirement B).
        self.query_outcomes: list[KillQueryOutcome] = []

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        ticker = data.ticker or "UNKNOWN"
        company_name = data.company_name or ""
        run_id = data.run_id or ticker

        executed, unexecuted, hits = self._run_searches(ticker, company_name, run_id, out)
        unsearched = self._unsearched_categories(unexecuted)

        capital = data.channel(Channel.CAPITAL_STRUCTURE)
        runway = capital.payload.get("runway_months") if capital else None

        # Same deterministic resolution Science uses, over the same fact set
        # this agent already sees -- so both agree on which trial is current
        # without introducing a new isolation surface (requirement D).
        resolution = resolve_current_program(list(data.facts))
        gate = evaluate_kill_gate(
            list(data.facts),
            list(data.risk_flags),
            unsearched_categories=unsearched,
            runway_months=runway,
            current_program_trial_id=(
                UNKNOWN if resolution.relevance_unresolved else resolution.trial_id
            ),
            program_relevance_unresolved=resolution.relevance_unresolved
            and len(resolution.candidates) > 1,
        )

        for assessment in gate.assessments:
            if assessment.level.level >= 3:
                out.risk_flags.append(
                    RiskFlag(
                        flag_id=f"kill_{assessment.category.value.lower()}",
                        category=_kill_to_fact_category(assessment.category),
                        title=f"{assessment.category} = {assessment.level}",
                        detail=assessment.rationale,
                        severity=(
                            Materiality.CRITICAL
                            if assessment.level.level >= 4
                            else Materiality.HIGH
                        ),
                        fact_ids=tuple(fid for f in assessment.findings for fid in f.fact_ids),
                        raised_by=self.agent_id,
                    )
                )

        if unsearched:
            out.degraded = True
            out.unresolved.append(
                UnresolvedQuestion(
                    question=(
                        "The following kill categories were never searched: "
                        + ", ".join(str(c) for c in unsearched)
                    ),
                    why_it_matters=(
                        "K0 for an unsearched category means 'not examined'. Treating it as "
                        "'no problem found' is the failure mode this system is built to prevent."
                    ),
                    blocking=True,
                    category=FactCategory.OTHER,
                    raised_by=self.agent_id,
                )
            )

        payload = {
            "kill_gate": [a.to_row() for a in gate.assessments],
            "max_level": str(gate.max_level),
            "disqualifying": [str(a.category) for a in gate.disqualifying],
            "major": [str(a.category) for a in gate.major],
            "queries_executed": executed,
            "queries_not_executed": unexecuted,
            "queries_skipped_due_to_budget": list(self.queries_skipped_due_to_budget),
            # Requirement B: per-query failure-state audit, never collapsed
            # into a single "no search provider configured" message.
            "query_outcomes": [
                {
                    "query": o.query,
                    "reason": str(o.reason),
                    "detail": o.detail,
                    "results_count": o.results_count,
                }
                for o in self.query_outcomes
            ],
            "unsearched_categories": [str(c) for c in unsearched],
            "program_resolved": (
                UNKNOWN if resolution.relevance_unresolved else resolution.trial_id
            ),
            "program_relevance_unresolved": resolution.relevance_unresolved,
            "search_hits": hits,
            "findings": [
                {
                    "category": str(f.category),
                    "level": str(f.level),
                    "title": f.title,
                    "detail": f.detail,
                    "fact_ids": list(f.fact_ids),
                    "primary_source": f.evidence_is_primary,
                }
                for a in gate.assessments
                for f in a.findings
            ],
        }
        summary = (
            f"Worst kill level {gate.max_level}. "
            f"{len(gate.disqualifying)} disqualifying, {len(gate.major)} major. "
            f"{len(executed)} queries executed, {len(unexecuted)} not executed."
        )
        out.evaluation = self.evaluation(
            Channel.KILL, summary, (), payload, self.baseline_from(data)
        )
        out.metrics["max_kill_level"] = str(gate.max_level)
        out.metrics["queries_executed"] = len(executed)
        return out

    def _run_searches(
        self, ticker: str, company_name: str, run_id: str, out: AgentOutput
    ) -> tuple[list[str], list[str], list[dict]]:
        executed: list[str] = list(self.executed_queries)
        unexecuted: list[str] = []
        hits: list[dict] = []
        already = {q.lower() for q in self.executed_queries}

        research_usable, research_unavailable_reason = (
            self.research.available() if self.research is not None else (False, "")
        )

        for query in kill_queries(ticker, company_name):
            if _covered_by(query, already):
                executed.append(query)
                continue
            if self.research is not None:
                outcome, query_hits = self._search_via_research(
                    query,
                    ticker=ticker,
                    run_id=run_id,
                    usable=research_usable,
                    unavailable_reason=research_unavailable_reason,
                )
            else:
                outcome, query_hits = self._search_via_legacy(query)
            self.query_outcomes.append(outcome)
            if outcome.reason is KillSearchFailureReason.SKIPPED_DUE_TO_BUDGET:
                self.queries_skipped_due_to_budget.append(query)
            if outcome.reason in (
                KillSearchFailureReason.EXECUTED_WITH_RESULTS,
                KillSearchFailureReason.EXECUTED_ZERO_RESULTS,
            ):
                executed.append(query)
                hits.extend(query_hits)
            else:
                unexecuted.append(query)

        if unexecuted:
            out.errors.append(self._unexecuted_summary(unexecuted))
        return executed, unexecuted, hits

    def _unexecuted_summary(self, unexecuted: list[str]) -> str:
        """Accurate, non-collapsing summary of why queries did not execute.

        Requirement B: never infer "no provider" merely because a query was
        not executed, and never say "no search provider configured" unless
        that is genuinely why -- a per-reason breakdown, built from the
        actual :class:`KillQueryOutcome` recorded for each query, replaces
        the old single hardcoded message.

        A reason CODE alone ("23 PROVIDER_UNAVAILABLE") does not distinguish
        "no client/credentials configured" from "the run's global token
        budget was already exhausted by an earlier stage/agent" -- both
        collapse to the same :class:`KillSearchFailureReason` today (see
        ``LLMClient.available()``), but the underlying cause text (the
        provider's own ``unavailable_reason`` -- e.g. the specific
        ``LLMBudget.exhausted_reason``, which names the offending agent and
        the used/remaining token counts) is preserved per query on
        ``KillQueryOutcome.detail``. The first non-empty detail seen for
        each reason is carried into the summary, so this line still says
        WHY -- auth, connection, budget, or global stop -- not just how
        many.
        """
        unexecuted_set = set(unexecuted)
        matching = [outcome for outcome in self.query_outcomes if outcome.query in unexecuted_set]
        reasons = Counter(outcome.reason for outcome in matching)
        detail_by_reason: dict[KillSearchFailureReason, str] = {}
        for outcome in matching:
            if outcome.detail and outcome.reason not in detail_by_reason:
                detail_by_reason[outcome.reason] = outcome.detail
        parts = []
        for reason, count in sorted(reasons.items(), key=lambda kv: kv[0].value):
            detail = detail_by_reason.get(reason, "")
            part = f"{count} {reason.value}"
            if detail:
                part += f" ({detail[:200]})"
            parts.append(part)
        return f"{len(unexecuted)} mandatory kill queries were not executed ({'; '.join(parts)})"

    def _search_via_research(
        self, query: str, *, ticker: str, run_id: str, usable: bool, unavailable_reason: str
    ) -> tuple[KillQueryOutcome, list[dict]]:
        """Execute one mandatory kill query through the live research provider.

        Recorded into ``self.discovery`` with QueryPurpose.BEAR (mandatory
        kill queries are, by construction, disconfirming/falsification
        searches) and agent_id="kill_agent" so it stays auditable exactly like
        adversarial discovery -- never mixed into the Bull agent's context,
        since Bull's isolation policy never reads BEAR-purpose hits.

        Requirement B: distinguishes PROVIDER_UNAVAILABLE (the provider
        exists but reports itself unusable), SKIPPED_DUE_TO_BUDGET (a
        BudgetExceeded abort caught by the provider), SEARCH_ERROR (any
        other failure, including a partial/malformed response with
        ``executed=True``), and the two successful outcomes -- never
        collapsing any of these into "no provider configured".
        """
        assert self.research is not None
        if not usable:
            return (
                KillQueryOutcome(
                    query=query,
                    reason=KillSearchFailureReason.PROVIDER_UNAVAILABLE,
                    detail=unavailable_reason,
                ),
                [],
            )

        domain = _domain_for_kill_query(query)
        research_query = ResearchQuery(
            query=query,
            domain=domain,
            stance="bear",
            max_results=self.max_hits,
            rationale="mandatory kill query",
        )
        result = self.research.search(research_query, agent_id="kill_agent")
        record = SearchQueryRecord(
            query_id=make_query_id("kill_agent", QueryPurpose.BEAR, query, run_id),
            run_id=run_id,
            ticker=ticker,
            agent_id="kill_agent",
            query_purpose=QueryPurpose.BEAR,
            query_text=query,
            results_count=len(result.documents),
            executed=result.executed,
            provider=getattr(self.research, "name", "unknown"),
            rationale=research_query.rationale,
            outcome=str(result.outcome),
        )
        self.discovery.record_query(record)
        self.discovery.record_hits(
            hits_from_documents(
                result.documents, query=record, provider=record.provider, path=result.path
            )
        )
        if not result.executed:
            error = str(result.error)
            if error.startswith("BudgetExceeded"):
                reason = KillSearchFailureReason.SKIPPED_DUE_TO_BUDGET
            elif result.outcome is FetchOutcome.DISABLED:
                reason = KillSearchFailureReason.PROVIDER_UNAVAILABLE
            else:
                reason = KillSearchFailureReason.SEARCH_ERROR
            return KillQueryOutcome(query=query, reason=reason, detail=error), []

        if result.error:
            # executed=True but the response could not be parsed cleanly --
            # a real failure, never silently treated as "searched, nothing
            # found".
            return (
                KillQueryOutcome(
                    query=query, reason=KillSearchFailureReason.SEARCH_ERROR, detail=result.error
                ),
                [],
            )

        query_hits: list[dict] = []
        for document in result.documents[: self.max_hits]:
            query_hits.append(
                {
                    "query": query,
                    "title": document.title,
                    "url": document.url,
                    "tier": str(document.tier),
                    "snippet": (document.text or "")[:400],
                    "published_date": document.published_date,
                }
            )
        reason = (
            KillSearchFailureReason.EXECUTED_WITH_RESULTS
            if query_hits
            else KillSearchFailureReason.EXECUTED_ZERO_RESULTS
        )
        return KillQueryOutcome(query=query, reason=reason, results_count=len(query_hits)), query_hits

    def _search_via_legacy(self, query: str) -> tuple[KillQueryOutcome, list[dict]]:
        """Fall back to the offline SearchProvider (Tavily/Brave/none).

        Used only when no live research provider was given at all. This is
        the ONLY path that may report NO_PROVIDER -- and only when the
        legacy provider is genuinely ``NullSearchProvider``, i.e. nothing
        was configured for this run whatsoever; a Tavily/Brave provider that
        exists but lacks credentials is PROVIDER_UNAVAILABLE, not NO_PROVIDER.
        """
        if isinstance(self.search, NullSearchProvider):
            return (
                KillQueryOutcome(
                    query=query,
                    reason=KillSearchFailureReason.NO_PROVIDER,
                    detail="no live research provider and no legacy search provider were "
                    "configured for this run",
                ),
                [],
            )

        response = self.search.search(query, limit=self.max_hits)
        if not response.executed:
            return (
                KillQueryOutcome(
                    query=query,
                    reason=KillSearchFailureReason.PROVIDER_UNAVAILABLE,
                    detail=response.error,
                ),
                [],
            )
        query_hits: list[dict] = []
        for hit in response.hits[: self.max_hits]:
            tier = classify_tier(hit.url)
            query_hits.append(
                {
                    "query": query,
                    "title": hit.title,
                    "url": hit.url,
                    "tier": str(tier),
                    "snippet": hit.snippet[:400],
                    "published_date": hit.published_date,
                }
            )
        reason = (
            KillSearchFailureReason.EXECUTED_WITH_RESULTS
            if query_hits
            else KillSearchFailureReason.EXECUTED_ZERO_RESULTS
        )
        return KillQueryOutcome(query=query, reason=reason, results_count=len(query_hits)), query_hits

    @staticmethod
    def _unsearched_categories(unexecuted: list[str]) -> tuple[KillCategory, ...]:
        if not unexecuted:
            return ()
        categories: set[KillCategory] = set()
        for query in unexecuted:
            for token, category in QUERY_CATEGORY_MAP.items():
                if token.lower() in query.lower():
                    categories.add(category)
        # Only mandatory categories are reported, to keep the report readable.
        return tuple(sorted(categories & set(MANDATORY_KILL_CATEGORIES), key=lambda c: c.value))


def _domain_for_kill_query(query: str) -> ResearchDomain:
    """Best-effort ResearchDomain for a mandatory kill query's DiscoveryLog row.

    Purely a diagnostics/audit label (which required domain this query's
    discovery record falls under) -- it plays no role in the kill gate itself,
    which reads facts and risk flags, not this mapping.
    """
    for token, category in QUERY_CATEGORY_MAP.items():
        if token.lower() in query.lower():
            domain = _KILL_CATEGORY_TO_RESEARCH_DOMAIN.get(category)
            if domain is not None:
                return domain
    return ResearchDomain.REGULATORY


def _covered_by(query: str, executed: set[str]) -> bool:
    """Whether an already-executed query covers this mandatory one.

    Matched on the distinctive tail of the template ("going concern",
    "clinical hold"), because the adversarial pass phrases the same question
    with the company name rather than the ticker.
    """
    tail = " ".join(query.split()[1:]).lower().strip()
    if not tail:
        return False
    return any(tail in candidate for candidate in executed)


def _kill_to_fact_category(category: KillCategory) -> FactCategory:
    return {
        KillCategory.REGULATORY_KILL: FactCategory.REGULATORY,
        KillCategory.CLINICAL_KILL: FactCategory.CLINICAL,
        KillCategory.SCIENCE_KILL: FactCategory.SCIENCE,
        KillCategory.CAPITAL_KILL: FactCategory.CAPITAL_STRUCTURE,
        KillCategory.COMMERCIAL_KILL: FactCategory.COMMERCIAL,
        KillCategory.GOVERNANCE_KILL: FactCategory.GOVERNANCE,
        KillCategory.ACCOUNTING_KILL: FactCategory.ACCOUNTING,
        KillCategory.LIQUIDITY_KILL: FactCategory.LISTING,
    }[category]
