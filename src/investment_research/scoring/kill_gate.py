"""Kill Gate: K0-K5 per category (requirement 5).

Deterministic and rule-based on purpose.  This is the one part of the system
that must behave identically every run, be auditable line by line, and never
depend on a model's mood or an LLM's willingness to be discouraging.

Two rules do most of the work:

* A category that was **never searched** is reported as ``UNSEARCHED``, not K0.
  "We found nothing" and "we did not look" are different statements, and
  conflating them is how a thesis survives that should not have.
* ``K4``/``K5`` cap every investment-quality score (requirement 5/18).  A
  disqualifying fact cannot be outvoted by an attractive one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..schemas.agent_io import RiskFlag
from ..schemas.enums import (
    MANDATORY_KILL_CATEGORIES,
    FactCategory,
    KillCategory,
    KillLevel,
    Materiality,
    SourceTier,
)
from ..schemas.evaluation import KillAssessment, KillFinding, KillGateResult
from ..schemas.fact import Fact

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KillRule:
    """One rule: a pattern, the category it kills, and how hard."""

    rule_id: str
    category: KillCategory
    title: str
    pattern: re.Pattern[str]
    level_primary: KillLevel
    level_secondary: KillLevel
    explanation: str


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


#: ``level_primary`` applies when the evidence is Tier 1/2; ``level_secondary``
#: when it rests on weaker sourcing.  Requirement 2: a Tier-4/5 rumour cannot
#: disqualify a company on its own, but it can raise a flag.
KILL_RULES: tuple[KillRule, ...] = (
    KillRule(
        "endpoint_not_accepted",
        KillCategory.REGULATORY_KILL,
        "Regulator does not accept the primary endpoint as adequate to establish effectiveness",
        _rx(
            r"(?:does not|did not|would not)\s+(?:consider|agree|view|regard|accept)[^.]{0,120}"
            r"endpoint[^.]{0,120}(?:appropriate|adequate|acceptable|establish\s+effectiveness|"
            r"establish\s+efficacy)"
        ),
        KillLevel.K5,
        KillLevel.K3,
        "The trial as designed cannot support approval. Every downstream valuation of the "
        "programme assumes an approval path that the regulator has said does not exist.",
    ),
    KillRule(
        "additional_trial_required",
        KillCategory.REGULATORY_KILL,
        "Regulator requires an additional adequate and well-controlled trial",
        _rx(
            r"(?:additional|another|a\s+second)\s+(?:adequate\s+and\s+well[- ]controlled\s+)?"
            r"(?:trial|study|studies)[^.]{0,100}?\b(?:required|necessary|needed)\b"
        ),
        KillLevel.K4,
        KillLevel.K3,
        "Adds years and a financing round before any approval is possible.",
    ),
    KillRule(
        "clinical_hold",
        KillCategory.CLINICAL_KILL,
        "Clinical hold",
        _rx(r"\bclinical\s+hold\b"),
        KillLevel.K5,
        KillLevel.K3,
        "Dosing is stopped by the regulator; timelines and the programme itself are at risk.",
    ),
    KillRule(
        "crl",
        KillCategory.REGULATORY_KILL,
        "Complete Response Letter",
        _rx(r"\bcomplete\s+response\s+letter\b"),
        KillLevel.K4,
        KillLevel.K3,
        "The application was not approved as filed.",
    ),
    KillRule(
        "trial_failed",
        KillCategory.CLINICAL_KILL,
        "Trial missed its primary endpoint",
        _rx(
            r"(?:did not|failed to)\s+(?:meet|achieve)[^.]{0,60}primary\s+endpoint|"
            r"primary\s+endpoint[^.]{0,40}(?:was\s+not\s+met|not\s+met)"
        ),
        KillLevel.K5,
        KillLevel.K3,
        "The central efficacy claim is disproven by the trial itself.",
    ),
    KillRule(
        "trial_terminated",
        KillCategory.CLINICAL_KILL,
        "Trial terminated or withdrawn",
        _rx(r"status is (?:TERMINATED|WITHDRAWN|SUSPENDED)"),
        KillLevel.K4,
        KillLevel.K3,
        "The programme stopped before producing the evidence the thesis needs.",
    ),
    KillRule(
        "surrogate_unvalidated",
        KillCategory.SCIENCE_KILL,
        "Surrogate endpoint contradicted by independent literature",
        _rx(r"(?:did not correlate|no correlation|does not predict|failed to predict)[^.]{0,80}"),
        KillLevel.K3,
        KillLevel.K2,
        "The mechanism story does not connect to clinical benefit in independent evidence.",
    ),
    KillRule(
        "going_concern",
        KillCategory.CAPITAL_KILL,
        "Going concern doubt",
        _rx(r"substantial doubt[^.]{0,80}going concern|going concern"),
        KillLevel.K4,
        KillLevel.K3,
        "The auditor or management states the company may not be able to continue operating. "
        "Financing is not optional and will not be on the holder's terms.",
    ),
    KillRule(
        "reverse_split",
        KillCategory.CAPITAL_KILL,
        "Reverse split",
        _rx(r"reverse\s+(?:stock\s+)?split"),
        KillLevel.K2,
        KillLevel.K1,
        "Historically associated with sustained dilution and price decline.",
    ),
    KillRule(
        "delisting",
        KillCategory.LIQUIDITY_KILL,
        "Delisting or listing non-compliance",
        _rx(r"delisting|minimum bid price|non-?compliance with[^.]{0,40}listing"),
        KillLevel.K3,
        KillLevel.K2,
        "Forced exit of index and institutional holders, and a hard deadline for corrective action.",
    ),
    KillRule(
        "restatement",
        KillCategory.ACCOUNTING_KILL,
        "Restatement or material weakness",
        _rx(r"restat(?:e|ement)|material weakness|non-?reliance"),
        KillLevel.K4,
        KillLevel.K3,
        "The reported numbers underpinning every valuation are not reliable.",
    ),
    KillRule(
        "auditor_change",
        KillCategory.ACCOUNTING_KILL,
        "Auditor resignation or dismissal",
        _rx(r"auditor\s+(?:resign|dismiss|declin)|dismissed\s+its\s+(?:independent\s+)?auditor"),
        KillLevel.K3,
        KillLevel.K2,
        "Auditors rarely resign for neutral reasons.",
    ),
    KillRule(
        "sec_investigation",
        KillCategory.GOVERNANCE_KILL,
        "Regulatory investigation or subpoena",
        _rx(r"SEC\s+(?:investigation|subpoena|enforcement)|formal\s+investigation|wells\s+notice"),
        KillLevel.K4,
        KillLevel.K2,
        "Open enforcement risk with an unbounded timeline.",
    ),
    KillRule(
        "fraud_litigation",
        KillCategory.GOVERNANCE_KILL,
        "Securities fraud litigation",
        _rx(r"securities\s+(?:class\s+action|fraud)|shareholder\s+class\s+action"),
        KillLevel.K3,
        KillLevel.K2,
        "Alleges that prior disclosure was misleading -- directly relevant to whether the "
        "evidence set can be trusted.",
    ),
    KillRule(
        "no_revenue_no_approval",
        KillCategory.COMMERCIAL_KILL,
        "No approved product and no product revenue",
        _rx(r"no approved products|no product revenue"),
        KillLevel.K2,
        KillLevel.K2,
        "All value depends on a future event, so the thesis has no commercial floor.",
    ),
    KillRule(
        "customer_concentration",
        KillCategory.COMMERCIAL_KILL,
        "Extreme customer concentration",
        _rx(r"(?:one|single|a\s+single)\s+customer[^.]{0,40}(?:\d{2,3})\s*(?:%|percent)"),
        KillLevel.K3,
        KillLevel.K2,
        "Revenue is one renewal decision away from disappearing.",
    ),
    KillRule(
        "competitor_ahead",
        KillCategory.COMMERCIAL_KILL,
        "Competitor is ahead with an accepted endpoint or an approved product",
        _rx(
            r"competitor[^.]{0,120}(?:positive\s+phase\s*3|approved|submitted\s+a\s+marketing\s+application)"
        ),
        KillLevel.K3,
        KillLevel.K2,
        "First approval typically takes the standard of care and the reimbursement anchor.",
    ),
)


#: Which risk-flag categories feed which kill category, when a flag exists but
#: no textual rule matched.
_FLAG_CATEGORY_MAP: dict[FactCategory, KillCategory] = {
    FactCategory.REGULATORY: KillCategory.REGULATORY_KILL,
    FactCategory.CLINICAL: KillCategory.CLINICAL_KILL,
    FactCategory.SCIENCE: KillCategory.SCIENCE_KILL,
    FactCategory.TECHNOLOGY: KillCategory.SCIENCE_KILL,
    FactCategory.CAPITAL_STRUCTURE: KillCategory.CAPITAL_KILL,
    FactCategory.LIQUIDITY: KillCategory.CAPITAL_KILL,
    FactCategory.LISTING: KillCategory.LIQUIDITY_KILL,
    FactCategory.GOVERNANCE: KillCategory.GOVERNANCE_KILL,
    FactCategory.ACCOUNTING: KillCategory.ACCOUNTING_KILL,
    FactCategory.LEGAL: KillCategory.GOVERNANCE_KILL,
    FactCategory.COMMERCIAL: KillCategory.COMMERCIAL_KILL,
    FactCategory.COMPETITION: KillCategory.COMMERCIAL_KILL,
    FactCategory.INSIDER: KillCategory.GOVERNANCE_KILL,
}

_SEVERITY_TO_LEVEL = {
    Materiality.CRITICAL: KillLevel.K3,
    Materiality.HIGH: KillLevel.K2,
    Materiality.MEDIUM: KillLevel.K1,
    Materiality.LOW: KillLevel.K0,
    Materiality.INFORMATIONAL: KillLevel.K0,
}


def evaluate_kill_gate(
    facts: list[Fact],
    risk_flags: list[RiskFlag],
    *,
    unsearched_categories: tuple[KillCategory, ...] = (),
    runway_months: float | None = None,
) -> KillGateResult:
    """Run every kill rule over the evidence and return per-category assessments."""
    findings: dict[KillCategory, list[KillFinding]] = {c: [] for c in KillCategory}

    for fact in facts:
        haystack = f"{fact.claim} {fact.value if isinstance(fact.value, str) else ''}"
        for rule in KILL_RULES:
            if not rule.pattern.search(haystack):
                continue
            primary = fact.source_tier in (SourceTier.TIER_1, SourceTier.TIER_2)
            level = rule.level_primary if primary else rule.level_secondary
            findings[rule.category].append(
                KillFinding(
                    category=rule.category,
                    level=level,
                    title=rule.title,
                    detail=f"{rule.explanation} Evidence: {fact.claim[:400]}",
                    fact_ids=(fact.fact_id,),
                    source_urls=(fact.source_url,),
                    evidence_is_primary=primary,
                )
            )

    # Runway is arithmetic, not a phrase, so it gets its own rule.
    if runway_months is not None:
        if runway_months < 6:
            level, label = KillLevel.K4, "under six months"
        elif runway_months < 12:
            level, label = KillLevel.K3, "under twelve months"
        elif runway_months < 18:
            level, label = KillLevel.K2, "under eighteen months"
        else:
            level, label = KillLevel.K0, "over eighteen months"
        if level != KillLevel.K0:
            findings[KillCategory.CAPITAL_KILL].append(
                KillFinding(
                    category=KillCategory.CAPITAL_KILL,
                    level=level,
                    title=f"Cash runway {label} ({runway_months} months)",
                    detail=(
                        "Financing is required before or around the next catalyst, which "
                        "transfers the option value of a good result to new investors."
                    ),
                    evidence_is_primary=True,
                )
            )

    # Risk flags with no matching textual rule still register, one level lower
    # than their severity implies, because a flag is weaker than a quoted fact.
    for flag in risk_flags:
        category = _FLAG_CATEGORY_MAP.get(flag.category)
        if category is None:
            continue
        level = _SEVERITY_TO_LEVEL.get(flag.severity, KillLevel.K0)
        if level == KillLevel.K0:
            continue
        if any(f.title == flag.title for f in findings[category]):
            continue
        findings[category].append(
            KillFinding(
                category=category,
                level=level,
                title=flag.title,
                detail=flag.detail,
                fact_ids=flag.fact_ids,
                evidence_is_primary=False,
            )
        )

    assessments: list[KillAssessment] = []
    categories = sorted(
        set(MANDATORY_KILL_CATEGORIES) | {c for c, items in findings.items() if items},
        key=lambda c: c.value,
    )
    for category in categories:
        items = findings[category]
        if category in unsearched_categories and not items:
            assessments.append(
                KillAssessment(
                    category=category,
                    level=KillLevel.K0,
                    rationale=(
                        "UNSEARCHED: no search was executed for this category, so K0 here means "
                        "'not examined', not 'no problem found'."
                    ),
                    findings=(),
                    evidence_confidence=0.0,
                )
            )
            continue
        if not items:
            assessments.append(
                KillAssessment(
                    category=category,
                    level=KillLevel.K0,
                    rationale="No material concern found in the evidence that was collected.",
                    findings=(),
                    evidence_confidence=0.5,
                )
            )
            continue
        worst = max(items, key=lambda f: f.level.level)
        primary_backed = any(f.evidence_is_primary for f in items)
        assessments.append(
            KillAssessment(
                category=category,
                level=worst.level,
                rationale=worst.title + (" [primary source]" if primary_backed else " [weak sourcing]"),
                findings=tuple(sorted(items, key=lambda f: -f.level.level)),
                evidence_confidence=0.85 if primary_backed else 0.35,
            )
        )

    return KillGateResult(
        assessments=tuple(assessments), unsearched_categories=tuple(unsearched_categories)
    )


#: Score caps by worst kill level (requirement 5 / regression test 18).
#: A K3 red flag means an investment-quality score above 4.5 is not available,
#: whatever the upside looks like.
QUALITY_CAPS: dict[int, float] = {0: 10.0, 1: 9.0, 2: 7.5, 3: 4.5, 4: 2.5, 5: 1.0}


def quality_cap(max_level: KillLevel) -> float:
    return QUALITY_CAPS[max_level.level]
