"""Evidence Confidence (requirement 9).

Reported separately from, and *before*, every other score.  It answers a
different question: not "how good does this look" but "how much of what we just
said is actually established".

"Explosive Potential 9.5 / Evidence Confidence 4.0" is a valid and important
output.  It says: the upside is real if the story is true, and we have not
established that the story is true.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.enums import (
    DECISION_GRADE_CLASSES,
    EvidenceClass,
    KillCategory,
    SourceTier,
    VerifiedStatus,
)
from ..schemas.fact import Fact, UnresolvedQuestion


@dataclass
class ConfidenceBreakdown:
    score: float
    primary_source_ratio: float
    decision_grade_ratio: float
    corroboration_ratio: float
    company_claim_ratio: float
    stale_ratio: float
    blocking_unknowns: int
    unsearched_categories: int
    penalties: tuple[str, ...]
    fact_count: int


def compute_evidence_confidence(
    facts: list[Fact],
    unresolved: list[UnresolvedQuestion],
    *,
    unsearched_categories: tuple[KillCategory, ...] = (),
    collector_failures: int = 0,
    used_fixtures: bool = False,
) -> ConfidenceBreakdown:
    """Score 0-10.  Deliberately harsh: absence of evidence lowers it."""
    total = len(facts)
    if total == 0:
        return ConfidenceBreakdown(
            score=0.0,
            primary_source_ratio=0.0,
            decision_grade_ratio=0.0,
            corroboration_ratio=0.0,
            company_claim_ratio=0.0,
            stale_ratio=0.0,
            blocking_unknowns=len([u for u in unresolved if u.blocking]),
            unsearched_categories=len(unsearched_categories),
            penalties=("no facts collected",),
            fact_count=0,
        )

    primary = sum(1 for f in facts if f.source_tier.is_primary) / total
    decision_grade = sum(1 for f in facts if f.evidence_class in DECISION_GRADE_CLASSES) / total
    corroborated = sum(1 for f in facts if f.independent_confirmation) / total
    company_claims = sum(1 for f in facts if f.company_claim) / total
    stale = sum(1 for f in facts if f.stale) / total
    blocking = len([u for u in unresolved if u.blocking])

    penalties: list[str] = []
    score = 10.0 * (0.35 * primary + 0.30 * decision_grade + 0.20 * corroborated + 0.15)

    if company_claims > 0.6:
        score -= 1.5
        penalties.append(
            f"{company_claims:.0%} of facts are company statements without independent confirmation"
        )
    if stale > 0.3:
        score -= 1.0
        penalties.append(f"{stale:.0%} of facts are stale by event date")
    if blocking:
        score -= min(2.5, 0.8 * blocking)
        penalties.append(f"{blocking} blocking unresolved question(s)")
    if unsearched_categories:
        score -= min(2.0, 0.5 * len(unsearched_categories))
        penalties.append(
            f"{len(unsearched_categories)} kill category/categories were never searched"
        )
    if collector_failures:
        score -= min(1.5, 0.5 * collector_failures)
        penalties.append(f"{collector_failures} collector(s) failed or were unavailable")
    if used_fixtures:
        score = min(score, 3.0)
        penalties.append(
            "run used SYNTHETIC FIXTURE data; evidence confidence is capped and this is not "
            "real research"
        )
    if total < 15:
        score -= 1.0
        penalties.append(f"thin evidence set ({total} facts)")

    return ConfidenceBreakdown(
        score=round(max(0.0, min(10.0, score)), 1),
        primary_source_ratio=round(primary, 3),
        decision_grade_ratio=round(decision_grade, 3),
        corroboration_ratio=round(corroborated, 3),
        company_claim_ratio=round(company_claims, 3),
        stale_ratio=round(stale, 3),
        blocking_unknowns=blocking,
        unsearched_categories=len(unsearched_categories),
        penalties=tuple(penalties),
        fact_count=total,
    )
