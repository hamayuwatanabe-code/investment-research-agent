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

from ..collectors.search import SearchProvider, kill_queries
from ..collectors.tiering import classify_tier
from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import (
    MANDATORY_KILL_CATEGORIES,
    FactCategory,
    KillCategory,
    Materiality,
    SourceTier,
)
from ..schemas.fact import UnresolvedQuestion
from ..scoring.kill_gate import evaluate_kill_gate
from .base import Agent

log = logging.getLogger(__name__)

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


class KillAgent(Agent):
    agent_id = "kill_agent"
    purpose = "Find disqualifying facts; never defend the candidate"

    def __init__(self, search: SearchProvider, *, max_hits_per_query: int = 5) -> None:
        self.search = search
        self.max_hits = max_hits_per_query

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        ticker = data.ticker or "UNKNOWN"
        company_name = data.company_name or ""

        executed, unexecuted, hits = self._run_searches(ticker, company_name, out)
        unsearched = self._unsearched_categories(unexecuted)

        capital = data.channel(Channel.CAPITAL_STRUCTURE)
        runway = capital.payload.get("runway_months") if capital else None

        gate = evaluate_kill_gate(
            list(data.facts),
            list(data.risk_flags),
            unsearched_categories=unsearched,
            runway_months=runway,
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
            "unsearched_categories": [str(c) for c in unsearched],
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
        self, ticker: str, company_name: str, out: AgentOutput
    ) -> tuple[list[str], list[str], list[dict]]:
        executed: list[str] = []
        unexecuted: list[str] = []
        hits: list[dict] = []
        for query in kill_queries(ticker, company_name):
            response = self.search.search(query, limit=self.max_hits)
            if not response.executed:
                unexecuted.append(query)
                continue
            executed.append(query)
            for hit in response.hits[: self.max_hits]:
                tier = classify_tier(hit.url)
                hits.append(
                    {
                        "query": query,
                        "title": hit.title,
                        "url": hit.url,
                        "tier": str(tier),
                        "snippet": hit.snippet[:400],
                        "published_date": hit.published_date,
                    }
                )
                if tier in (SourceTier.TIER_4, SourceTier.TIER_5, SourceTier.UNKNOWN):
                    # Kept for follow-up, never used to establish a kill alone.
                    continue
        if unexecuted:
            out.errors.append(
                f"{len(unexecuted)} mandatory kill queries were not executed "
                "(no search provider configured)"
            )
        return executed, unexecuted, hits

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
