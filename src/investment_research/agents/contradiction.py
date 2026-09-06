"""Agent 13: Contradiction.

Mechanical cross-checking (requirement 13).  It looks for the specific
mismatches that experience says matter, rather than trying to reason in general:

* company characterization vs. regulator's own statement
* company claim vs. statutory filing
* analyst assertion vs. primary source
* old guidance vs. new guidance
* TAM vs. serviceable population
* basic vs. fully diluted share count
* designation-based optimism vs. an adverse endpoint position

Requirement 1F: when facts conflict, the conflict is the output.  This agent
never resolves a contradiction in favour of the more attractive reading.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import UNKNOWN, EvidenceClass, FactCategory, Materiality, SourceTier
from ..schemas.fact import Contradiction, Fact, make_contradiction_id, parse_iso_date
from .base import Agent
from .capital_structure import parse_number

log = logging.getLogger(__name__)

_POSITIVE_FRAMING = re.compile(
    r"(?i)\b(constructive|productive|positive|encouraging|aligned|on\s+track|supportive)\b"
)
_ADVERSE_REGULATOR = re.compile(
    r"(?i)(?:does not|did not|would not)\s+(?:consider|agree|accept)|"
    r"additional[^.]{0,60}(?:trial|study)[^.]{0,60}required|clinical\s+hold|"
    r"complete\s+response\s+letter"
)
_GUIDANCE_RE = re.compile(r"(?i)\bguidance\b|\bexpects?\s+to\b|\bguided\b")


class ContradictionAgent(Agent):
    agent_id = "contradiction"
    purpose = "Find and report inconsistencies without resolving them"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        ticker = data.ticker or UNKNOWN
        facts = list(data.facts)

        found: list[Contradiction] = []
        found += self._company_vs_regulator(ticker, facts)
        found += self._company_vs_filing(ticker, facts)
        found += self._analyst_vs_primary(ticker, facts)
        found += self._guidance_changes(ticker, facts)
        found += self._tam_vs_population(ticker, facts, data)
        found += self._share_count(ticker, facts, data)

        # de-duplicate by id
        unique: dict[str, Contradiction] = {c.contradiction_id: c for c in found}
        out.contradictions = list(unique.values())

        payload = {
            "contradictions": [c.to_row() for c in out.contradictions],
            "count": len(out.contradictions),
            "critical_count": sum(
                1 for c in out.contradictions if c.severity == Materiality.CRITICAL
            ),
            "checks_run": [
                "company_vs_regulator",
                "company_vs_filing",
                "analyst_vs_primary",
                "guidance_change",
                "tam_vs_population",
                "basic_vs_fully_diluted",
            ],
        }
        summary = (
            f"{len(out.contradictions)} contradiction(s) found, "
            f"{payload['critical_count']} critical. Unresolved conflicts are reported as "
            "conflicts, not reconciled into a single story."
        )
        out.evaluation = self.evaluation(
            Channel.CONTRADICTIONS, summary, (), payload, self.baseline_from(data)
        )
        out.metrics["contradiction_count"] = len(out.contradictions)
        return out

    # -- checks ------------------------------------------------------------
    def _company_vs_regulator(self, ticker: str, facts: list[Fact]) -> list[Contradiction]:
        positives = [
            f for f in facts if f.company_claim and _POSITIVE_FRAMING.search(f.claim)
        ]
        adverse = [
            f
            for f in facts
            if _ADVERSE_REGULATOR.search(f.claim) and f.source_tier.rank <= 2
        ]
        results: list[Contradiction] = []
        for positive in positives:
            for negative in adverse:
                results.append(
                    Contradiction(
                        contradiction_id=make_contradiction_id(
                            ticker, "company_vs_regulator", positive.fact_id, negative.fact_id
                        ),
                        ticker=ticker,
                        kind="company_vs_regulator",
                        description=(
                            "The company characterizes the regulatory interaction favourably "
                            "while a primary-source document records an adverse regulator "
                            "position. The company's adjective is not evidence about the "
                            "regulator's view."
                        ),
                        left_fact_id=positive.fact_id,
                        right_fact_id=negative.fact_id,
                        left_summary=positive.claim[:300],
                        right_summary=negative.claim[:300],
                        severity=Materiality.CRITICAL,
                    )
                )
        return results

    def _company_vs_filing(self, ticker: str, facts: list[Fact]) -> list[Contradiction]:
        """Same subject, opposite polarity, company release vs. statutory filing."""
        results: list[Contradiction] = []
        company = [f for f in facts if f.company_claim and f.source_tier == SourceTier.TIER_2]
        filings = [f for f in facts if f.source_tier == SourceTier.TIER_1]
        for release, filing in ((c, f) for c in company for f in filings):
            if release.category != filing.category:
                continue
            if _POSITIVE_FRAMING.search(release.claim) and _ADVERSE_REGULATOR.search(filing.claim):
                results.append(
                    Contradiction(
                        contradiction_id=make_contradiction_id(
                            ticker, "release_vs_filing", release.fact_id, filing.fact_id
                        ),
                        ticker=ticker,
                        kind="release_vs_filing",
                        description=(
                            "A company press release and a statutory filing describe the same "
                            "matter differently. The filing carries liability; the release does not."
                        ),
                        left_fact_id=release.fact_id,
                        right_fact_id=filing.fact_id,
                        left_summary=release.claim[:300],
                        right_summary=filing.claim[:300],
                        severity=Materiality.CRITICAL,
                    )
                )
        return results

    def _analyst_vs_primary(self, ticker: str, facts: list[Fact]) -> list[Contradiction]:
        analysts = [f for f in facts if f.evidence_class == EvidenceClass.ANALYST_OPINION]
        adverse = [f for f in facts if _ADVERSE_REGULATOR.search(f.claim) and f.source_tier.rank <= 2]
        results: list[Contradiction] = []
        for analyst in analysts:
            for negative in adverse:
                results.append(
                    Contradiction(
                        contradiction_id=make_contradiction_id(
                            ticker, "analyst_vs_primary", analyst.fact_id, negative.fact_id
                        ),
                        ticker=ticker,
                        kind="analyst_vs_primary",
                        description=(
                            "A sell-side view coexists with an adverse primary-source fact. "
                            "The primary source governs; the price target is Tier 4 and cannot "
                            "settle the question."
                        ),
                        left_fact_id=analyst.fact_id,
                        right_fact_id=negative.fact_id,
                        left_summary=analyst.claim[:300],
                        right_summary=negative.claim[:300],
                        severity=Materiality.HIGH,
                    )
                )
        return results

    def _guidance_changes(self, ticker: str, facts: list[Fact]) -> list[Contradiction]:
        guidance = [f for f in facts if _GUIDANCE_RE.search(f.claim)]
        results: list[Contradiction] = []
        for left, right in combinations(guidance, 2):
            left_date = parse_iso_date(left.publication_date)
            right_date = parse_iso_date(right.publication_date)
            if not left_date or not right_date or left_date == right_date:
                continue
            left_value = parse_number(left.value) or parse_number(left.claim)
            right_value = parse_number(right.value) or parse_number(right.claim)
            if left_value is None or right_value is None or left_value == right_value:
                continue
            older, newer = (left, right) if left_date < right_date else (right, left)
            results.append(
                Contradiction(
                    contradiction_id=make_contradiction_id(
                        ticker, "guidance_change", older.fact_id, newer.fact_id
                    ),
                    ticker=ticker,
                    kind="guidance_change",
                    description=(
                        "Guidance changed between two dates. The change itself is information; "
                        "only the newer figure should be used, and the revision should be explained."
                    ),
                    left_fact_id=older.fact_id,
                    right_fact_id=newer.fact_id,
                    left_summary=older.claim[:300],
                    right_summary=newer.claim[:300],
                    severity=Materiality.MEDIUM,
                )
            )
        return results

    def _tam_vs_population(self, ticker: str, facts: list[Fact], data) -> list[Contradiction]:
        tam_facts = [f for f in facts if "addressable market" in f.claim.lower()]
        population_facts = [
            f
            for f in facts
            if "prevalence" in f.claim.lower() and not f.company_claim
        ]
        results: list[Contradiction] = []
        for tam in tam_facts:
            tam_value = parse_number(tam.value) or parse_number(tam.claim)
            for population in population_facts:
                pop_value = parse_number(population.value) or parse_number(population.claim)
                if not tam_value or not pop_value:
                    continue
                implied_price = tam_value / pop_value
                if implied_price > 200_000:
                    results.append(
                        Contradiction(
                            contradiction_id=make_contradiction_id(
                                ticker, "tam_vs_population", tam.fact_id, population.fact_id
                            ),
                            ticker=ticker,
                            kind="tam_vs_population",
                            description=(
                                f"The stated addressable market implies about "
                                f"${implied_price:,.0f} of annual revenue per diagnosed patient "
                                "against the independent prevalence estimate. The TAM and the "
                                "serviceable population are not consistent without a price "
                                "assumption that is not in evidence."
                            ),
                            left_fact_id=tam.fact_id,
                            right_fact_id=population.fact_id,
                            left_summary=tam.claim[:300],
                            right_summary=population.claim[:300],
                            severity=Materiality.HIGH,
                        )
                    )
        return results

    def _share_count(self, ticker: str, facts: list[Fact], data) -> list[Contradiction]:
        capital = data.channel(Channel.CAPITAL_STRUCTURE)
        if not capital:
            return []
        basic = capital.payload.get("basic_shares")
        diluted = capital.payload.get("fully_diluted_shares")
        if not basic or not diluted:
            return []
        overhang = (diluted - basic) / basic * 100.0
        if overhang < 20:
            return []
        basic_fact = capital.payload.get("fact_ids", {}).get("basic_shares", "")
        return [
            Contradiction(
                contradiction_id=make_contradiction_id(
                    ticker, "basic_vs_diluted", basic_fact or "basic", "diluted"
                ),
                ticker=ticker,
                kind="basic_vs_fully_diluted",
                description=(
                    f"Fully diluted shares exceed basic shares by about {overhang:.0f}%. Any "
                    "market cap, per-share or upside figure quoted on basic shares overstates "
                    "the return available to a new buyer by roughly that margin."
                ),
                left_fact_id=basic_fact or "UNKNOWN",
                right_fact_id="capital_structure:fully_diluted_shares",
                left_summary=f"basic shares {basic:,.0f}",
                right_summary=f"fully diluted shares {diluted:,.0f}",
                severity=Materiality.HIGH if overhang >= 40 else Materiality.MEDIUM,
            )
        ]
