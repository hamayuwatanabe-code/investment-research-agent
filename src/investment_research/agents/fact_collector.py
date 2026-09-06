"""Agent 1: Raw Fact Collector.

Collects and does not evaluate.  Requirement 1A/4-1.

Two structural guarantees rather than a promise:

* Its :class:`AgentInput` contains no facts and no channels at all (see the
  isolation policy), so it cannot search selectively in support of a view it
  has already formed.
* Every claim it emits passes ``assert_evaluation_free``; an evaluative claim
  raises rather than being quietly stored.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..collectors.base import CollectionResult
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import (
    UNKNOWN,
    EvidenceClass,
    FactCategory,
    FetchOutcome,
    Materiality,
    Provenance,
    SourceTier,
    VerifiedStatus,
)
from ..schemas.fact import Fact, RawFact, UnresolvedQuestion
from ..schemas.validation import EvaluationLeak, assert_evaluation_free
from .base import Agent

log = logging.getLogger(__name__)

#: What Agent 1 is required to look for (requirement 4-1).  Anything on this
#: list that produced nothing becomes an explicit gap in the report, not a
#: silent omission.
REQUIRED_COVERAGE: tuple[tuple[str, FactCategory], ...] = (
    ("annual report (10-K)", FactCategory.FINANCIAL),
    ("quarterly report (10-Q)", FactCategory.FINANCIAL),
    ("current reports (8-K)", FactCategory.OTHER),
    ("registration / prospectus (S-1, S-3, 424B)", FactCategory.CAPITAL_STRUCTURE),
    ("proxy statement (DEF 14A)", FactCategory.GOVERNANCE),
    ("beneficial ownership (13D/13G)", FactCategory.GOVERNANCE),
    ("insider transactions (Form 4)", FactCategory.INSIDER),
    ("FDA documents", FactCategory.REGULATORY),
    ("ClinicalTrials.gov registrations", FactCategory.CLINICAL),
    ("trial protocol / design", FactCategory.CLINICAL),
    ("government contracts", FactCategory.CONTRACTS),
    ("peer-reviewed literature", FactCategory.SCIENCE),
    ("company IR materials", FactCategory.OTHER),
    ("earnings call transcript", FactCategory.FINANCIAL),
    ("financing, warrants, convertibles, ATM", FactCategory.CAPITAL_STRUCTURE),
    ("cash, debt, share count, fully diluted securities", FactCategory.CAPITAL_STRUCTURE),
    ("management", FactCategory.MANAGEMENT),
    ("competition", FactCategory.COMPETITION),
    ("market size", FactCategory.MARKET_SIZE),
    ("customer concentration / backlog", FactCategory.COMMERCIAL),
    ("pipeline", FactCategory.CLINICAL),
    ("recent events", FactCategory.CATALYST),
)


class FactCollectorAgent(Agent):
    agent_id = "fact_collector"
    purpose = "Collect observable facts without evaluating them"

    def __init__(self, results: Sequence[CollectionResult]) -> None:
        # Collection I/O happens in the orchestrator; the agent receives the
        # raw results so that it stays deterministic and unit-testable.
        self.results = list(results)

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        ticker = data.ticker or UNKNOWN
        run_id = data.params.get("run_id", UNKNOWN)

        seen: set[str] = set()
        covered: set[FactCategory] = set()

        for result in self.results:
            for note in result.notes:
                out.metrics.setdefault("collector_notes", []).append(
                    f"{result.collector}: {note}"
                )
            if result.degraded:
                out.degraded = True
                out.errors.append(result.describe())
            out.sources.extend(result.sources)

            for raw in result.raw_facts:
                try:
                    assert_evaluation_free(raw.claim, where=f"{result.collector}:raw_fact")
                except EvaluationLeak as exc:
                    # A collector that emits an opinion is a bug in the
                    # collector; drop the record and say so.
                    out.errors.append(str(exc))
                    continue
                fact = self._to_fact(raw, run_id, result.provenance)
                if fact.fact_id in seen:
                    continue
                seen.add(fact.fact_id)
                covered.add(fact.category)
                out.facts.append(fact)

        out.metrics["collectors"] = [r.describe() for r in self.results]
        out.metrics["fact_count"] = len(out.facts)
        out.metrics["fixture_data"] = any(
            r.provenance == Provenance.FIXTURE for r in self.results
        )

        for label, category in REQUIRED_COVERAGE:
            if category not in covered:
                out.unresolved.append(
                    UnresolvedQuestion(
                        question=f"No evidence was obtained for: {label}",
                        why_it_matters=(
                            "This is a required collection target. Absence of evidence here is "
                            "an unexamined area, not a clean result."
                        ),
                        blocking=category
                        in (
                            FactCategory.REGULATORY,
                            FactCategory.CAPITAL_STRUCTURE,
                            FactCategory.LIQUIDITY,
                        ),
                        category=category,
                        raised_by=self.agent_id,
                    )
                )

        if not out.facts:
            out.degraded = True
            out.errors.append(
                f"no facts collected for {ticker}; every downstream conclusion is unsupported"
            )
        return out

    @staticmethod
    def _to_fact(raw: RawFact, run_id: str, provenance: Provenance) -> Fact:
        """Convert a raw observation to an unverified Fact.

        Note the classification: nothing arrives verified.  Agent 1 records what
        a document says; Agent 2 decides what that is worth.
        """
        source = raw.source
        evidence_class = (
            EvidenceClass.COMPANY_CLAIM if raw.company_claim else EvidenceClass.UNVERIFIED_CLAIM
        )
        return Fact(
            fact_id=raw.fact_id(),
            ticker=raw.ticker,
            category=raw.category,
            claim=raw.claim,
            evidence_class=evidence_class,
            source_id=source.source_id,
            source_url=source.url,
            source_title=source.title,
            source_tier=source.tier if isinstance(source.tier, SourceTier) else SourceTier.UNKNOWN,
            publication_date=source.published_date,
            event_date=source.event_date,
            effective_date=source.effective_date,
            filing_date=source.filing_date,
            verified_status=VerifiedStatus.NOT_VERIFIED,
            confidence=0.0,
            company_claim=raw.company_claim,
            materiality=Materiality.INFORMATIONAL,
            value=raw.value,
            unit=raw.unit,
            provenance=provenance if provenance != Provenance.LIVE else source.provenance,
            run_id=run_id,
            notes=f"collected_by={raw.collector}",
        )
