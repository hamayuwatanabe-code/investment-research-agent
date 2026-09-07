"""Agent 2: Evidence Integrity.

Decides what each collected claim is *worth*.  Three jobs:

1. **Classify** every claim into an evidence class (requirement 1B).  A company
   statement stays a COMPANY_CLAIM until something independent confirms it, no
   matter how primary the venue it was filed in.
2. **Corroborate** across sources while refusing to count syndicated reprints or
   same-host republication as independent confirmation (requirement 13).
3. **Date-check**: distinguish publication from event date, flag stale evidence,
   and refuse to treat an old event re-reported today as news (requirement 3).

Confidence is derived mechanically from tier, class, corroboration and recency.
It is not a vibe.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, timedelta

from ..collectors.tiering import is_reprint
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import (
    UNKNOWN,
    EvidenceClass,
    FactCategory,
    Materiality,
    SourceTier,
    VerifiedStatus,
)
from ..schemas.fact import Fact, normalize_claim, parse_iso_date
from .base import Agent

log = logging.getLogger(__name__)

#: Categories whose facts are material to a kill decision by default.
_CRITICAL_CATEGORIES = {
    FactCategory.REGULATORY,
    FactCategory.LIQUIDITY,
    FactCategory.ACCOUNTING,
    FactCategory.LISTING,
}
_HIGH_CATEGORIES = {
    FactCategory.CLINICAL,
    FactCategory.CAPITAL_STRUCTURE,
    FactCategory.GOVERNANCE,
    FactCategory.LEGAL,
    FactCategory.COMPETITION,
}

#: Phrases that make a claim material regardless of its category.
_MATERIAL_PHRASES = (
    "going concern",
    "substantial doubt",
    "clinical hold",
    "complete response letter",
    "does not consider",
    "not appropriate to establish",
    "additional adequate and well-controlled",
    "non-compliance",
    "delisting",
    "restatement",
    "material weakness",
    "resigned",
    "investigation",
    "default",
)


class EvidenceIntegrityAgent(Agent):
    agent_id = "evidence_integrity"
    purpose = "Verify, classify, date-check and corroborate collected facts"

    def __init__(self, *, today: date | None = None, stale_after_days: int = 400) -> None:
        self.today = today or date.today()
        self.stale_after_days = stale_after_days

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        facts = list(data.facts)

        groups = self._group_equivalent(facts)
        for fact in facts:
            group = groups.get(fact.fact_id, [fact])
            out.facts.append(self._assess(fact, group))

        out.metrics["assessed"] = len(out.facts)
        out.metrics["verified"] = sum(
            1 for f in out.facts if f.verified_status == VerifiedStatus.VERIFIED
        )
        out.metrics["company_claims"] = sum(1 for f in out.facts if f.company_claim)
        out.metrics["stale"] = sum(1 for f in out.facts if f.stale)
        out.metrics["tier_histogram"] = {
            str(tier): sum(1 for f in out.facts if f.source_tier == tier) for tier in SourceTier
        }
        return out

    # -- corroboration -----------------------------------------------------
    def _group_equivalent(self, facts: list[Fact]) -> dict[str, list[Fact]]:
        """Group facts making the same claim from genuinely different sources."""
        groups: dict[str, list[Fact]] = {}
        buckets: list[list[Fact]] = []
        for fact in facts:
            placed = False
            for bucket in buckets:
                head = bucket[0]
                if head.category != fact.category:
                    continue
                if normalize_claim(head.claim) == normalize_claim(fact.claim) or is_reprint(
                    head.claim, fact.claim, 0.75
                ):
                    bucket.append(fact)
                    placed = True
                    break
            if not placed:
                buckets.append([fact])
        for bucket in buckets:
            for fact in bucket:
                groups[fact.fact_id] = bucket
        return groups

    @staticmethod
    def _independent_confirmation(fact: Fact, group: list[Fact]) -> tuple[bool, tuple[str, ...]]:
        """Independent confirmation requires a *different, non-company* source.

        Two press releases and their five reprints are one source.  This is the
        check that stops "widely reported" from masquerading as verified.
        """
        others = [
            other
            for other in group
            if other.fact_id != fact.fact_id
            and other.source_id != fact.source_id
            and not other.company_claim
            and other.source_tier.rank <= SourceTier.TIER_3.rank
        ]
        # Same-host republication does not count.
        seen_hosts: set[str] = set()  # same-host republication is not confirmation
        confirming: list[str] = []
        for other in others:
            host = other.source_url.split("/")[2] if "//" in other.source_url else other.source_url
            if host in seen_hosts:
                continue
            seen_hosts.add(host)
            confirming.append(other.source_id)
        return bool(confirming), tuple(confirming)

    # -- per-fact assessment ----------------------------------------------
    def _assess(self, fact: Fact, group: list[Fact]) -> Fact:
        confirmed, corroborating = self._independent_confirmation(fact, group)
        evidence_class = self._classify(fact, confirmed)
        stale, effective = self._staleness(fact)
        status = self._status(fact, evidence_class, confirmed)
        materiality = self._materiality(fact)
        confidence = self._confidence(fact, evidence_class, confirmed, stale, len(corroborating))

        notes = [fact.notes] if fact.notes else []
        if fact.is_search_derived:
            notes.append(
                f"read from a {fact.content_kind}, not the source document body; "
                "not usable as verified evidence"
            )
        if fact.company_claim and not confirmed:
            notes.append("company statement; no independent confirmation found")
        if stale:
            notes.append(f"stale: effective date {effective} older than {self.stale_after_days}d")
        if fact.source_tier in (SourceTier.TIER_4, SourceTier.TIER_5):
            notes.append("tier 4/5 source: cannot settle a material question alone")
        if fact.event_date == UNKNOWN and fact.publication_date != UNKNOWN:
            notes.append("event date unknown; publication date must not be read as the event date")

        return replace(
            fact,
            evidence_class=evidence_class,
            verified_status=status,
            confidence=round(confidence, 3),
            independent_confirmation=confirmed,
            corroborating_source_ids=corroborating,
            materiality=materiality,
            stale=stale,
            notes="; ".join(notes),
        )

    @staticmethod
    def _classify(fact: Fact, confirmed: bool) -> EvidenceClass:
        # Requirement M1: nothing read from a search result can be classified as
        # verified or independent evidence. The tier describes the document; the
        # content kind describes what was actually read, and only the second one
        # can make a claim decision-grade.
        if fact.is_search_derived:
            return (
                EvidenceClass.COMPANY_CLAIM
                if fact.company_claim
                else EvidenceClass.UNVERIFIED_CLAIM
            )
        tier = fact.source_tier
        if fact.company_claim:
            return (
                EvidenceClass.VERIFIED_FACT
                if confirmed and tier.is_primary
                else EvidenceClass.COMPANY_CLAIM
            )
        if tier == SourceTier.TIER_4:
            return EvidenceClass.ANALYST_OPINION
        if tier == SourceTier.TIER_5:
            return EvidenceClass.UNVERIFIED_CLAIM
        if tier == SourceTier.UNKNOWN:
            return EvidenceClass.UNVERIFIED_CLAIM
        if fact.category == FactCategory.MICROSTRUCTURE:
            return EvidenceClass.MARKET_INFERENCE
        if tier == SourceTier.TIER_1:
            # A regulator's or exchange's own statement of its own position.
            return EvidenceClass.VERIFIED_FACT
        return EvidenceClass.INDEPENDENT_EVIDENCE

    def _staleness(self, fact: Fact) -> tuple[bool, str]:
        """Staleness is judged on the event date, never the publication date."""
        for candidate in (fact.event_date, fact.effective_date, fact.filing_date):
            parsed = parse_iso_date(candidate)
            if parsed:
                return (self.today - parsed) > timedelta(days=self.stale_after_days), candidate
        parsed_pub = parse_iso_date(fact.publication_date)
        if parsed_pub:
            # Publication-only dating is itself a weakness; flag conservatively.
            return (self.today - parsed_pub) > timedelta(
                days=self.stale_after_days
            ), fact.publication_date
        return False, UNKNOWN

    @staticmethod
    def _status(fact: Fact, evidence_class: EvidenceClass, confirmed: bool) -> VerifiedStatus:
        # Requirement M1: a search-derived claim keeps a discovery status. It is
        # not "unverified because we did not get round to it" -- it is
        # unverifiable from what we hold, because we hold a summary.
        if fact.is_search_derived:
            return (
                VerifiedStatus.PRIMARY_SOURCE_IDENTIFIED_BUT_NOT_FETCHED
                if fact.primary_source_url
                else VerifiedStatus.SEARCH_EVIDENCE
            )
        if evidence_class == EvidenceClass.VERIFIED_FACT:
            return VerifiedStatus.VERIFIED
        if evidence_class == EvidenceClass.INDEPENDENT_EVIDENCE:
            return VerifiedStatus.PARTIALLY_VERIFIED
        if evidence_class == EvidenceClass.COMPANY_CLAIM:
            return VerifiedStatus.PARTIALLY_VERIFIED if confirmed else VerifiedStatus.NOT_VERIFIED
        if evidence_class in (EvidenceClass.ANALYST_OPINION, EvidenceClass.UNVERIFIED_CLAIM):
            return VerifiedStatus.INSUFFICIENT_EVIDENCE
        return VerifiedStatus.NOT_VERIFIED

    @staticmethod
    def _materiality(fact: Fact) -> Materiality:
        text = fact.claim.lower()
        if any(phrase in text for phrase in _MATERIAL_PHRASES):
            return Materiality.CRITICAL
        if fact.category in _CRITICAL_CATEGORIES:
            return Materiality.HIGH
        if fact.category in _HIGH_CATEGORIES:
            return Materiality.MEDIUM
        return Materiality.LOW

    @staticmethod
    def _confidence(
        fact: Fact,
        evidence_class: EvidenceClass,
        confirmed: bool,
        stale: bool,
        corroborating: int,
    ) -> float:
        base = {
            SourceTier.TIER_1: 0.80,
            SourceTier.TIER_2: 0.65,
            SourceTier.TIER_3: 0.45,
            SourceTier.TIER_4: 0.20,
            SourceTier.TIER_5: 0.10,
            SourceTier.UNKNOWN: 0.05,
        }[fact.source_tier]
        if fact.company_claim and not confirmed:
            base -= 0.20
        if evidence_class == EvidenceClass.ANALYST_OPINION:
            base = min(base, 0.20)
        base += min(0.10 * corroborating, 0.15)
        if stale:
            base -= 0.10
        if fact.event_date == UNKNOWN:
            base -= 0.05
        # Requirement M1: scale by how much of the document was actually read.
        base *= fact.content_kind.confidence_multiplier
        return max(0.0, min(1.0, base))
