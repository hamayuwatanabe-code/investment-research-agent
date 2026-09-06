"""Primary-source escalation (requirement P4).

The rule: a **material** claim carried only by Tier 3-5 reporting must not reach
the Final Judge as though it were established. The system attempts to confirm it
in a primary source, and when it cannot, marks it
``UNVERIFIED_MATERIAL_CLAIM`` -- a status barred from decision-grade use.

Why a distinct status rather than NOT_VERIFIED: the two mean different things.
NOT_VERIFIED is "we did not confirm this". UNVERIFIED_MATERIAL_CLAIM is "this
matters, we tried specifically to confirm it in a filing or a regulator
document, and we could not". The second is a much stronger warning, and it is
also a to-do list for a human.

Escalation runs news -> filing -> regulator, in that order, because the cheapest
confirmation is usually a company filing that repeats the claim under liability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from ..collectors.documents import Document
from ..schemas.enums import (
    UNKNOWN,
    EvidenceClass,
    Materiality,
    ResearchDomain,
    SourceTier,
    VerifiedStatus,
)
from ..schemas.fact import Fact
from .provider import ResearchProvider, ResearchQuery

log = logging.getLogger(__name__)

#: Domains searched when escalating, in preference order.
PRIMARY_DOMAINS: tuple[str, ...] = (
    "sec.gov",
    "fda.gov",
    "clinicaltrials.gov",
    "ema.europa.eu",
    "nih.gov",
)

#: A claim is material enough to require escalation if it says one of these
#: things. Deliberately narrow: escalating everything would be noise.
_MATERIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("regulator_position", re.compile(r"(?i)\b(?:fda|ema|pmda|regulator|agency)\b.{0,120}"
                                      r"(?:not sufficient|does not|no longer|advised|required|rejected)")),
    ("endpoint", re.compile(r"(?i)\bendpoint\b.{0,80}(?:not|insufficient|inadequate|cannot)")),
    ("pivotal_status", re.compile(r"(?i)\bno longer\b.{0,40}\b(?:pivotal|registrational)\b")),
    ("going_concern", re.compile(r"(?i)\b(?:going concern|substantial doubt)\b")),
    ("clinical_hold", re.compile(r"(?i)\bclinical hold\b")),
    ("trial_failure", re.compile(r"(?i)\b(?:failed|did not meet|missed)\b.{0,40}\bendpoint\b")),
    ("investigation", re.compile(r"(?i)\b(?:sec investigation|subpoena|wells notice|restatement)\b")),
    ("delisting", re.compile(r"(?i)\b(?:delisting|minimum bid price|non-?compliance)\b")),
)


@dataclass
class EscalationAttempt:
    fact_id: str
    claim: str
    reason: str
    queries: list[str] = field(default_factory=list)
    confirmed: bool = False
    confirming_url: str = ""
    confirming_tier: str = UNKNOWN
    searched: bool = False
    note: str = ""


@dataclass
class EscalationReport:
    attempts: list[EscalationAttempt] = field(default_factory=list)

    @property
    def confirmed(self) -> list[EscalationAttempt]:
        return [a for a in self.attempts if a.confirmed]

    @property
    def unconfirmed(self) -> list[EscalationAttempt]:
        return [a for a in self.attempts if a.searched and not a.confirmed]

    @property
    def not_attempted(self) -> list[EscalationAttempt]:
        return [a for a in self.attempts if not a.searched]


def needs_escalation(fact: Fact) -> tuple[bool, str]:
    """Whether this fact is a material claim resting on weak sourcing."""
    if fact.source_tier in (SourceTier.TIER_1, SourceTier.TIER_2) and fact.independent_confirmation:
        return False, ""
    if fact.source_tier.rank <= 2 and not fact.company_claim:
        # A regulator's or exchange's own document already is the primary source.
        return False, ""
    for name, pattern in _MATERIAL_PATTERNS:
        if pattern.search(fact.claim):
            return True, name
    if fact.materiality == Materiality.CRITICAL:
        return True, "critical_materiality"
    return False, ""


def _confirms(document: Document, fact: Fact) -> bool:
    """Whether a candidate primary document actually supports the claim.

    Deliberately strict: overlapping distinctive terms, not merely being about
    the same company. A 10-K that mentions the FDA does not confirm a specific
    statement about what the FDA said.
    """
    if not document.tier.is_primary:
        return False
    haystack = f"{document.title} {document.text}".lower()
    if not haystack.strip():
        return False
    terms = {
        term
        for term in re.split(r"[^a-z0-9]+", fact.claim.lower())
        if len(term) > 5
    }
    if not terms:
        return False
    overlap = sum(1 for term in terms if term in haystack)
    return overlap >= max(3, len(terms) // 4)


def escalate(
    facts: Sequence[Fact],
    provider: ResearchProvider,
    *,
    company: str,
    max_facts: int = 12,
    max_queries_per_fact: int = 2,
) -> tuple[list[Fact], EscalationReport]:
    """Try to confirm weakly-sourced material claims in primary sources.

    Returns the facts with statuses updated, plus a report of what was tried.
    """
    from dataclasses import replace

    report = EscalationReport()
    usable, reason = provider.available()
    updated: list[Fact] = []
    escalated = 0

    for fact in facts:
        required, why = needs_escalation(fact)
        if not required:
            updated.append(fact)
            continue

        attempt = EscalationAttempt(fact_id=fact.fact_id, claim=fact.claim[:300], reason=why)
        if not usable or escalated >= max_facts:
            attempt.note = (
                reason if not usable else f"escalation budget of {max_facts} facts exhausted"
            )
            report.attempts.append(attempt)
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM,
                    notes=(fact.notes + "; " if fact.notes else "")
                    + f"material claim not escalated to a primary source ({attempt.note})",
                )
            )
            continue

        escalated += 1
        attempt.searched = True
        confirming: Document | None = None
        for domain in PRIMARY_DOMAINS[:max_queries_per_fact]:
            query = ResearchQuery(
                query=f"{company} {_key_terms(fact.claim)}",
                domain=ResearchDomain.REGULATORY,
                stance="verify",
                allowed_domains=(domain,),
                rationale=f"primary-source confirmation of {why}",
            )
            attempt.queries.append(f"{query.query} site:{domain}")
            result = provider.search(query)
            if not result.executed:
                continue
            for document in result.documents:
                if _confirms(document, fact):
                    confirming = document
                    break
            if confirming:
                break

        if confirming is not None:
            attempt.confirmed = True
            attempt.confirming_url = confirming.url
            attempt.confirming_tier = str(confirming.tier)
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.VERIFIED,
                    evidence_class=EvidenceClass.INDEPENDENT_EVIDENCE,
                    independent_confirmation=True,
                    corroborating_source_ids=(*fact.corroborating_source_ids, confirming.doc_id),
                    confidence=min(1.0, fact.confidence + 0.25),
                    notes=(fact.notes + "; " if fact.notes else "")
                    + f"escalated and confirmed in a primary source: {confirming.url}",
                )
            )
        else:
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM,
                    confidence=min(fact.confidence, 0.3),
                    notes=(fact.notes + "; " if fact.notes else "")
                    + "material claim; primary-source confirmation attempted and not found",
                )
            )
        report.attempts.append(attempt)

    return updated, report


def _key_terms(claim: str, limit: int = 10) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9-]+", claim) if len(w) > 4]
    return " ".join(words[:limit])
