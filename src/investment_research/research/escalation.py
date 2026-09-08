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
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..collectors.documents import Document
from ..schemas.enums import (
    UNKNOWN,
    ContentKind,
    EvidenceClass,
    FetchOutcome,
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
    (
        "regulator_position",
        re.compile(
            r"(?i)\b(?:fda|ema|pmda|regulator|agency)\b.{0,120}"
            r"(?:not sufficient|does not|no longer|advised|required|rejected)"
        ),
    ),
    ("endpoint", re.compile(r"(?i)\bendpoint\b.{0,80}(?:not|insufficient|inadequate|cannot)")),
    ("pivotal_status", re.compile(r"(?i)\bno longer\b.{0,40}\b(?:pivotal|registrational)\b")),
    ("going_concern", re.compile(r"(?i)\b(?:going concern|substantial doubt)\b")),
    ("clinical_hold", re.compile(r"(?i)\bclinical hold\b")),
    ("trial_failure", re.compile(r"(?i)\b(?:failed|did not meet|missed)\b.{0,40}\bendpoint\b")),
    (
        "investigation",
        re.compile(r"(?i)\b(?:sec investigation|subpoena|wells notice|restatement)\b"),
    ),
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
    #: Diagnostics (requirement G): how many candidate URLs were actually
    #: fetched, and how many of those fetches came back empty (failed,
    #: raised, or budget-cut) versus produced a body.
    fetches_attempted: int = 0
    fetches_failed: int = 0

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


#: Document types that carry statutory liability or come from the regulator
#: itself. A press release does not qualify, however primary the wire that
#: distributed it.
_PRIMARY_DOC_TYPES = frozenset({"filing", "registry", "regulator", "docket"})


def _confirms(document: Document, fact: Fact) -> bool:
    """Whether a candidate primary document actually supports the claim.

    Two independent bars, both required:

    1. **The document must be able to confirm this.** A company press release is
       not a primary source for what a regulator said -- it is the company's
       account of it, which is the exact conflation this system exists to stop.
       Company-controlled material qualifies only when it is a statutory filing,
       which carries liability that a release does not.
    2. **It must actually say the same thing.** Overlapping distinctive terms,
       not merely being about the same company: a 10-K that mentions the FDA
       does not confirm a specific statement about what the FDA said.
    """
    if not document.tier.is_primary:
        return False
    if document.is_company_ir and document.doc_type not in _PRIMARY_DOC_TYPES:
        return False
    if not document.content_kind.is_primary_text:
        # A search engine's summary about a filing is not the filing.
        return False
    haystack = f"{document.title} {document.text}".lower()
    if not haystack.strip():
        return False
    terms = {term for term in re.split(r"[^a-z0-9]+", fact.claim.lower()) if len(term) > 5}
    if not terms:
        return False
    overlap = sum(1 for term in terms if term in haystack)
    return overlap >= max(3, len(terms) // 4)


def _plausible_primary_candidate(document: Document, domain: str) -> bool:
    """Whether a search hit is worth spending a fetch on.

    A fetch is a real cost (LLM/tool budget), so this is a cheap pre-filter on
    the SearchHit-shaped candidate alone -- it never confirms anything by
    itself. Only ``_confirms()``, run against the FETCHED body, does that. Two
    checks, both required: the hit's own tier classification (derived from its
    URL, same as any other search result) must already look primary, and its
    host must actually be the domain this query was restricted to -- a search
    tool's domain filter is a request, not a guarantee.
    """
    if not document.tier.is_primary:
        return False
    try:
        host = urlparse(document.url).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    domain = domain.lower()
    return host == domain or host.endswith("." + domain)


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
        budget_cut_short = False
        for domain in PRIMARY_DOMAINS[:max_queries_per_fact]:
            query = ResearchQuery(
                query=f"{company} {_key_terms(fact.claim)}",
                domain=ResearchDomain.REGULATORY,
                stance="verify",
                allowed_domains=(domain,),
                rationale=f"primary-source confirmation of {why}",
            )
            attempt.queries.append(f"{query.query} site:{domain}")
            result = provider.search(query, agent_id="escalation")
            if not result.executed:
                # "We did not look" (budget cutoff, provider unavailable, ...),
                # never conflated with "we looked and it isn't there".
                if result.outcome is FetchOutcome.DISABLED:
                    budget_cut_short = True
                continue

            # The search hit is a pointer (METADATA_ONLY): it can tell us a
            # candidate URL is worth fetching, but it can never itself confirm
            # anything -- _confirms() below only runs against a fetched body.
            candidate_urls = [
                document.url
                for document in result.documents
                if _plausible_primary_candidate(document, domain)
            ]
            for url in candidate_urls[:max_queries_per_fact]:
                report.fetches_attempted += 1
                try:
                    fetched = provider.fetch(url, reason=f"confirm: {why}", agent_id="escalation")
                except Exception as exc:  # noqa: BLE001 - a fetch failure never confirms
                    log.warning("escalation fetch failed for %s: %s", url, exc)
                    fetched = None
                if fetched is None:
                    report.fetches_failed += 1
                    # Fetch failure (including a BudgetExceeded abort caught by
                    # the provider) leaves the claim unverified, never silently
                    # confirmed and never treated as a contradiction.
                    continue
                if _confirms(fetched, fact):
                    confirming = fetched
                    break
            if confirming:
                break

        if confirming is not None:
            attempt.confirmed = True
            # Recorded from the FETCHED document only -- never from the search
            # hit that merely pointed at it.
            attempt.confirming_url = confirming.url
            attempt.confirming_tier = str(confirming.tier)
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.VERIFIED,
                    evidence_class=EvidenceClass.INDEPENDENT_EVIDENCE,
                    # The claim was read from an actual document body just now
                    # (the confirming fetch) -- that is what content_kind
                    # records, independent of how the original claim was
                    # sourced (requirement M1).
                    content_kind=ContentKind.FULL_DOCUMENT,
                    independent_confirmation=True,
                    corroborating_source_ids=(*fact.corroborating_source_ids, confirming.doc_id),
                    confidence=min(1.0, fact.confidence + 0.25),
                    notes=(fact.notes + "; " if fact.notes else "")
                    + f"escalated and confirmed in a primary source body: {confirming.url}",
                )
            )
        else:
            note = "material claim; primary-source confirmation attempted and not found"
            if budget_cut_short:
                note = (
                    "material claim; primary-source confirmation incomplete -- the LLM token "
                    "budget was exhausted before every candidate domain could be searched or "
                    "fetched (not the same as searching and finding nothing)"
                )
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM,
                    confidence=min(fact.confidence, 0.3),
                    notes=(fact.notes + "; " if fact.notes else "") + note,
                )
            )
        report.attempts.append(attempt)

    return updated, report


def _key_terms(claim: str, limit: int = 10) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9-]+", claim) if len(w) > 4]
    return " ".join(words[:limit])
