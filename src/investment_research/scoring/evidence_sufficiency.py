"""Evidence Sufficiency Matrix (Decision-Grade Evidence Gate, requirement DG5).

Being SEARCHED and being evidence-*sufficient* are different achievements. The
Search Completeness Gate (:mod:`investment_research.scoring.completeness`)
answers "did we look at this domain at all". This module answers the harder
question: "did what we found actually settle anything, or is it still a
pointer to a document nobody has read".

A run can pass the Search Completeness Gate -- six domains, six executed
queries, documents found in every one -- and still have zero decision-grade
facts, because every document that came back was read only as a search
engine's summary of it. That is exactly the LGVN/CNTB/CRBP case this module
exists to catch: fully searched, not sufficiently verified.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    EvidenceSufficiencyStatus,
    FactCategory,
    KillCategory,
    KillConfirmation,
    Materiality,
    ResearchDomain,
    SearchStatus,
)
from ..schemas.evaluation import KillGateResult
from ..schemas.fact import Fact, UnresolvedQuestion
from .completeness import CompletenessResult, research_domain_for_category

#: Domains whose sufficiency can be judged from Fact evidence directly. The
#: remaining required domains (CATALYST, CONTRADICTION) are not fact-category
#: buckets -- a catalyst is a structured event, a contradiction is a relation
#: between two facts -- so their sufficiency is judged from search status
#: alone; this module does not invent a decision-grade test for structures
#: that were never facts to begin with.
DOMAIN_FACT_CATEGORIES: dict[ResearchDomain, tuple[FactCategory, ...]] = {
    ResearchDomain.REGULATORY: (FactCategory.REGULATORY,),
    ResearchDomain.CAPITAL_STRUCTURE: (
        FactCategory.CAPITAL_STRUCTURE,
        FactCategory.FINANCIAL,
        FactCategory.LIQUIDITY,
    ),
    ResearchDomain.SCIENCE_TECHNOLOGY: (
        FactCategory.CLINICAL,
        FactCategory.SCIENCE,
        FactCategory.TECHNOLOGY,
    ),
    ResearchDomain.COMPETITION: (FactCategory.COMPETITION, FactCategory.MARKET_SIZE),
}

#: A domain is also treated as sufficient when the Kill Gate already holds a
#: CONFIRMED assessment for one of these categories, even if the Fact objects
#: backing it are not individually decision-grade. This is deliberately
#: narrow: it exists for arithmetic disqualifiers (cash runway, share-count
#: maths) that requirement DG6 allows to stand as "a rule-based disqualifier
#: derived from verified structured facts" -- the numbers are structured and
#: cross-checkable even when the underlying filing line item is, by the
#: system's own rule, only ever a COMPANY_CLAIM. It must never be read the
#: other way: a PROVISIONAL kill finding never substitutes for decision-grade
#: evidence (see ``provisional_kill_findings`` below, which still blocks).
DOMAIN_KILL_CATEGORIES: dict[ResearchDomain, tuple[KillCategory, ...]] = {
    ResearchDomain.REGULATORY: (KillCategory.REGULATORY_KILL,),
    ResearchDomain.CAPITAL_STRUCTURE: (
        KillCategory.CAPITAL_KILL,
        KillCategory.LIQUIDITY_KILL,
        KillCategory.ACCOUNTING_KILL,
    ),
    ResearchDomain.SCIENCE_TECHNOLOGY: (KillCategory.CLINICAL_KILL, KillCategory.SCIENCE_KILL),
    ResearchDomain.COMPETITION: (KillCategory.COMMERCIAL_KILL,),
}


@dataclass
class DomainSufficiency:
    domain: ResearchDomain
    search_status: SearchStatus
    evidence_sufficiency_status: EvidenceSufficiencyStatus
    decision_grade_fact_count: int
    total_fact_count: int
    reason: str


@dataclass
class EvidenceSufficiencyMatrix:
    """Per-domain and overall verdict on whether an Action may be issued.

    ``search_status`` and ``evidence_sufficiency_status`` are tracked as two
    separate fields per domain (requirement DG5) precisely so "searched" is
    never read as "verified".
    """

    domains: dict[ResearchDomain, DomainSufficiency] = field(default_factory=dict)
    decision_grade_fact_total: int = 0
    #: K3+ kill findings that are still PROVISIONAL -- open verification work
    #: that blocks a final Action regardless of which domain raised them.
    provisional_kill_findings: tuple[str, ...] = ()
    #: Claims explicitly classified MATERIAL or CRITICAL that remain
    #: unresolved (requirement DG5). Distinct from provisional kill findings:
    #: this covers unresolved questions raised outside the kill gate.
    unresolved_material_claims: tuple[str, ...] = ()

    @property
    def sufficient(self) -> bool:
        """Whether the Evidence Sufficiency Matrix permits a final Action.

        Requirement DG5: zero decision-grade facts always blocks; any
        unresolved MATERIAL/CRITICAL claim blocks; any PROVISIONAL K3+ kill
        finding blocks; and every required domain must be evidence-sufficient
        (or have nothing material pending in it).
        """
        if self.decision_grade_fact_total == 0:
            return False
        if self.unresolved_material_claims:
            return False
        if self.provisional_kill_findings:
            return False
        return all(
            d.evidence_sufficiency_status != EvidenceSufficiencyStatus.INSUFFICIENT
            for d in self.domains.values()
        )

    def blocking_reasons(self) -> list[str]:
        """Human-readable list for ``BLOCKING_VERIFICATION_REQUIRED``."""
        reasons: list[str] = []
        if self.decision_grade_fact_total == 0:
            reasons.append(
                "Zero decision-grade facts were established in this run: every material claim "
                "is still a search summary, an unfetched primary source, or an unverified "
                "material claim. Retrieve and verify the underlying document bodies."
            )
        for finding in self.provisional_kill_findings:
            reasons.append(
                f"Retrieve and verify the primary source behind: {finding}."
            )
        for claim in self.unresolved_material_claims:
            reasons.append(f"Resolve the unresolved material/critical claim: {claim}.")
        for domain, entry in self.domains.items():
            if entry.evidence_sufficiency_status == EvidenceSufficiencyStatus.INSUFFICIENT:
                reasons.append(f"{domain}: {entry.reason}")
        return reasons

    def summary_rows(self) -> list[tuple[str, str, str, int, int, str]]:
        rows: list[tuple[str, str, str, int, int, str]] = []
        for domain in REQUIRED_RESEARCH_DOMAINS:
            entry = self.domains.get(domain)
            if entry is None:
                rows.append((str(domain), "UNKNOWN", "UNKNOWN", 0, 0, "not assessed"))
                continue
            rows.append(
                (
                    str(domain),
                    str(entry.search_status),
                    str(entry.evidence_sufficiency_status),
                    entry.decision_grade_fact_count,
                    entry.total_fact_count,
                    entry.reason,
                )
            )
        return rows


def _apply_blocking_override(
    domain: ResearchDomain,
    status: EvidenceSufficiencyStatus,
    reason: str,
    blocking_by_domain: dict[ResearchDomain, list[UnresolvedQuestion]],
) -> tuple[EvidenceSufficiencyStatus, str]:
    """Never let a domain read SUFFICIENT while it still has its own
    unresolved BLOCKING question, whatever decision-grade evidence or
    confirmed kill finding it otherwise has. Only ever downgrades SUFFICIENT
    to INSUFFICIENT -- a domain that is already UNSEARCHED or INSUFFICIENT
    for some other reason is untouched.
    """
    blocking_here = blocking_by_domain.get(domain, [])
    if status is not EvidenceSufficiencyStatus.SUFFICIENT or not blocking_here:
        return status, reason
    names = "; ".join(uq.question[:150] for uq in blocking_here[:3])
    more = f" (+{len(blocking_here) - 3} more)" if len(blocking_here) > 3 else ""
    new_reason = (
        f"{len(blocking_here)} blocking unresolved question(s) remain in this domain -- "
        f"{names}{more} -- evidence already collected here is never enough on its own while a "
        f"material question about it is still open (previously: {reason})"
    )
    return EvidenceSufficiencyStatus.INSUFFICIENT, new_reason


def assess_evidence_sufficiency(
    *,
    facts: Iterable[Fact],
    completeness: CompletenessResult,
    gate: KillGateResult,
    unresolved_material_claims: tuple[str, ...] = (),
    unresolved_questions: Iterable[UnresolvedQuestion] = (),
) -> EvidenceSufficiencyMatrix:
    """Build the Evidence Sufficiency Matrix from evidence already collected.

    ``unresolved_questions`` (requirement: a domain must never read
    SUFFICIENT while it still has an unresolved BLOCKING question) is
    distinct from ``unresolved_material_claims``: the latter is a flat,
    domain-agnostic tuple that only ever blocks the matrix's OVERALL
    ``sufficient`` property, never any individual domain's own status --
    so a REGULATORY-blocking unresolved question could previously leave
    ``matrix.domains[REGULATORY].evidence_sufficiency_status == SUFFICIENT``
    (because that domain already had a decision-grade fact or a confirmed
    kill finding) even while the run itself was separately blocked. Never
    filling that gap (CLAUDE.md rule 7) means the domain's OWN row must say
    so too, not just the run-wide gate.
    """
    facts = list(facts)
    matrix = EvidenceSufficiencyMatrix()
    matrix.decision_grade_fact_total = sum(1 for f in facts if f.is_decision_grade)

    blocking_by_domain: dict[ResearchDomain, list[UnresolvedQuestion]] = {}
    for uq in unresolved_questions:
        if not uq.blocking:
            continue
        domain = research_domain_for_category(uq.category)
        if domain is None:
            continue
        blocking_by_domain.setdefault(domain, []).append(uq)

    for domain in REQUIRED_RESEARCH_DOMAINS:
        coverage = completeness.coverage.get(domain)
        search_status = coverage.status if coverage else SearchStatus.UNSEARCHED
        categories = DOMAIN_FACT_CATEGORIES.get(domain)

        if search_status in (SearchStatus.UNSEARCHED, SearchStatus.FAILED):
            matrix.domains[domain] = DomainSufficiency(
                domain=domain,
                search_status=search_status,
                evidence_sufficiency_status=EvidenceSufficiencyStatus.UNSEARCHED,
                decision_grade_fact_count=0,
                total_fact_count=0,
                reason="not searched -- the Search Completeness Gate already blocks this run",
            )
            continue

        if categories is None:
            # CATALYST / CONTRADICTION: not a fact-category bucket. Having
            # been searched is the only test this module can meaningfully
            # apply; the Kill Gate's provisional-finding check still catches
            # a contradiction that was raised but never resolved.
            status, reason = _apply_blocking_override(
                domain,
                EvidenceSufficiencyStatus.SUFFICIENT,
                "searched; not a fact-category domain",
                blocking_by_domain,
            )
            matrix.domains[domain] = DomainSufficiency(
                domain=domain,
                search_status=search_status,
                evidence_sufficiency_status=status,
                decision_grade_fact_count=0,
                total_fact_count=0,
                reason=reason,
            )
            continue

        domain_facts = [f for f in facts if f.category in categories]
        domain_decision_grade = [f for f in domain_facts if f.is_decision_grade]
        # Only a MEDIUM+ materiality claim makes a domain "thesis-driving"
        # (requirement DG5). A LOW/INFORMATIONAL claim -- an unverifiable TAM
        # estimate in an investor deck, say -- never needed decision-grade
        # backing in the first place: it was never going to settle anything,
        # and the system's job is to leave it exactly that unpersuasive, not
        # to block the whole run until someone verifies a puffery number.
        material_domain_facts = [
            f for f in domain_facts if f.materiality.rank >= Materiality.MEDIUM.rank
        ]
        kill_confirmed = any(
            (assessment := gate.by_category(kc)) is not None
            and assessment.confirmation is KillConfirmation.CONFIRMED
            and assessment.level.level >= 1
            for kc in DOMAIN_KILL_CATEGORIES.get(domain, ())
        )

        if material_domain_facts and not domain_decision_grade and not kill_confirmed:
            status = EvidenceSufficiencyStatus.INSUFFICIENT
            reason = (
                f"{len(material_domain_facts)} material fact(s) surfaced in this domain, none "
                "of them decision-grade -- search summaries and unfetched primary sources are "
                "discovery evidence, not confirmation."
            )
        elif material_domain_facts and not domain_decision_grade and kill_confirmed:
            status = EvidenceSufficiencyStatus.SUFFICIENT
            reason = (
                "no individually decision-grade fact, but the Kill Gate holds a CONFIRMED "
                "rule-based disqualifier for this domain (requirement DG6)"
            )
        else:
            status = EvidenceSufficiencyStatus.SUFFICIENT
            reason = (
                f"{len(domain_decision_grade)} decision-grade fact(s) of "
                f"{len(domain_facts)} total"
                if domain_facts
                else "searched; no material claim surfaced in this domain"
            )
        status, reason = _apply_blocking_override(domain, status, reason, blocking_by_domain)
        matrix.domains[domain] = DomainSufficiency(
            domain=domain,
            search_status=search_status,
            evidence_sufficiency_status=status,
            decision_grade_fact_count=len(domain_decision_grade),
            total_fact_count=len(domain_facts),
            reason=reason,
        )

    matrix.provisional_kill_findings = tuple(
        f"{a.category} {a.level} ({a.rationale})" for a in gate.provisional_major_or_worse
    )
    matrix.unresolved_material_claims = unresolved_material_claims
    return matrix
