"""Evidence records: sources, raw facts, verified facts, contradictions.

Design notes
------------
* Facts are **append-only and versioned** (requirement 11/12).  A revised fact
  never overwrites its predecessor; it gets ``version = n+1`` and the old row is
  marked ``superseded_by``.
* ``fact_id`` is a deterministic content hash so the same claim from the same
  source in two different runs collapses to one identity (duplicate detection).
* There is deliberately **no score, rating, or recommendation field** on
  :class:`Fact`.  Evaluations cannot ride downstream on a fact (requirement 1D).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from typing import Any

from .enums import (
    UNKNOWN,
    DateKind,
    EvidenceClass,
    FactCategory,
    Materiality,
    Provenance,
    SourceTier,
    VerifiedStatus,
)

_WS = re.compile(r"\s+")


def normalize_claim(text: str) -> str:
    """Normalize claim text for hashing / duplicate detection."""
    return _WS.sub(" ", (text or "").strip().lower())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso_date(value: str | None) -> date | None:
    """Parse an ISO-8601 date, tolerating datetimes and ``UNKNOWN``.

    Returns ``None`` rather than guessing.  Requirement 1G: never impute a date;
    an unparseable date is an unknown date.
    """
    if not value or value == UNKNOWN:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d %b %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class Source:
    """A single retrievable document.

    ``content_hash`` lets the Evidence Integrity agent detect syndicated
    reprints, which requirement 13 forbids counting as independent sources.
    """

    source_id: str
    url: str
    title: str
    tier: SourceTier
    publisher: str = UNKNOWN
    published_date: str = UNKNOWN
    event_date: str = UNKNOWN
    effective_date: str = UNKNOWN
    filing_date: str = UNKNOWN
    accession: str = UNKNOWN
    retrieved_at: str = field(default_factory=utc_now_iso)
    provenance: Provenance = Provenance.LIVE
    content_hash: str = UNKNOWN
    syndicated_from: str | None = None
    excerpt: str = ""

    @property
    def is_reprint(self) -> bool:
        return self.syndicated_from is not None

    def best_date(self) -> tuple[str, DateKind] | tuple[None, None]:
        """The date that should drive recency reasoning.

        Requirement 3: recency is judged by when the *event* happened, not when
        an article about it was published.
        """
        for value, kind in (
            (self.event_date, DateKind.EVENT_DATE),
            (self.effective_date, DateKind.EFFECTIVE_DATE),
            (self.filing_date, DateKind.FILING_DATE),
            (self.published_date, DateKind.PUBLISHED_DATE),
        ):
            if value and value != UNKNOWN:
                return value, kind
        return None, None

    def to_row(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "tier": str(self.tier),
            "publisher": self.publisher,
            "published_date": self.published_date,
            "event_date": self.event_date,
            "effective_date": self.effective_date,
            "filing_date": self.filing_date,
            "accession": self.accession,
            "retrieved_at": self.retrieved_at,
            "provenance": str(self.provenance),
            "content_hash": self.content_hash,
            "syndicated_from": self.syndicated_from,
            "excerpt": self.excerpt[:2000],
        }


def make_source_id(url: str, title: str = "") -> str:
    digest = hashlib.sha256(f"{url}|{normalize_claim(title)}".encode()).hexdigest()
    return f"src_{digest[:16]}"


def make_fact_id(
    ticker: str, category: FactCategory | str, claim: str, source_url: str, event_date: str
) -> str:
    payload = "|".join(
        [ticker.upper(), str(category), normalize_claim(claim), source_url, str(event_date)]
    )
    return f"fact_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


@dataclass
class RawFact:
    """Output of Agent 1.  Evaluation-free by construction.

    The Fact Collector may only observe.  It has no field in which to place a
    view, and the orchestrator rejects any raw fact whose claim contains a
    recommendation verb (see ``validation.assert_evaluation_free``).
    """

    ticker: str
    category: FactCategory
    claim: str
    source: Source
    value: Any = UNKNOWN
    unit: str = UNKNOWN
    company_claim: bool = False
    collector: str = UNKNOWN
    raw_payload_ref: str | None = None

    def fact_id(self) -> str:
        return make_fact_id(
            self.ticker, self.category, self.claim, self.source.url, self.source.event_date
        )


@dataclass
class Fact:
    """A verified (or explicitly unverified) evidence record.

    This is the *only* structure that crosses agent boundaries in bulk.  It
    carries facts, provenance and doubt -- never an evaluation.
    """

    fact_id: str
    ticker: str
    category: FactCategory
    claim: str
    evidence_class: EvidenceClass
    source_id: str
    source_url: str
    source_title: str
    source_tier: SourceTier
    publication_date: str = UNKNOWN
    event_date: str = UNKNOWN
    effective_date: str = UNKNOWN
    filing_date: str = UNKNOWN
    verified_status: VerifiedStatus = VerifiedStatus.NOT_VERIFIED
    confidence: float = 0.0
    company_claim: bool = False
    independent_confirmation: bool = False
    corroborating_source_ids: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    materiality: Materiality = Materiality.INFORMATIONAL
    value: Any = UNKNOWN
    unit: str = UNKNOWN
    provenance: Provenance = Provenance.LIVE
    stale: bool = False
    version: int = 1
    superseded_by: str | None = None
    run_id: str = UNKNOWN
    notes: str = ""
    tags: tuple[str, ...] = ()

    # -- derived -----------------------------------------------------------
    @property
    def is_decision_grade(self) -> bool:
        """May this fact, alone, settle a material question? (requirement 2)"""
        from .enums import DECISION_GRADE_CLASSES, NON_DECISIVE_TIERS

        return (
            self.evidence_class in DECISION_GRADE_CLASSES
            and self.source_tier not in NON_DECISIVE_TIERS
            and self.verified_status
            in (VerifiedStatus.VERIFIED, VerifiedStatus.PARTIALLY_VERIFIED)
        )

    def next_version(self, **changes: Any) -> Fact:
        """Produce the successor version of this fact (requirement 12)."""
        return replace(self, version=self.version + 1, superseded_by=None, **changes)

    def to_row(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "ticker": self.ticker,
            "category": str(self.category),
            "claim": self.claim,
            "evidence_class": str(self.evidence_class),
            "source_id": self.source_id,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "source_tier": str(self.source_tier),
            "publication_date": self.publication_date,
            "event_date": self.event_date,
            "effective_date": self.effective_date,
            "filing_date": self.filing_date,
            "verified_status": str(self.verified_status),
            "confidence": float(self.confidence),
            "company_claim": int(self.company_claim),
            "independent_confirmation": int(self.independent_confirmation),
            "corroborating_source_ids": ",".join(self.corroborating_source_ids),
            "contradicting_evidence": ",".join(self.contradicting_evidence),
            "materiality": str(self.materiality),
            "value": UNKNOWN if self.value is None else str(self.value),
            "unit": self.unit,
            "provenance": str(self.provenance),
            "stale": int(self.stale),
            "version": self.version,
            "superseded_by": self.superseded_by,
            "run_id": self.run_id,
            "notes": self.notes,
            "tags": ",".join(self.tags),
        }


@dataclass
class Contradiction:
    """A machine-detected inconsistency (Agent 13, requirement 13)."""

    contradiction_id: str
    ticker: str
    kind: str
    description: str
    left_fact_id: str
    right_fact_id: str
    left_summary: str = ""
    right_summary: str = ""
    severity: Materiality = Materiality.MEDIUM
    resolved: bool = False
    resolution_note: str = ""
    run_id: str = UNKNOWN

    def to_row(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "ticker": self.ticker,
            "kind": self.kind,
            "description": self.description,
            "left_fact_id": self.left_fact_id,
            "right_fact_id": self.right_fact_id,
            "left_summary": self.left_summary,
            "right_summary": self.right_summary,
            "severity": str(self.severity),
            "resolved": int(self.resolved),
            "resolution_note": self.resolution_note,
            "run_id": self.run_id,
        }


def make_contradiction_id(ticker: str, kind: str, left: str, right: str) -> str:
    payload = "|".join([ticker.upper(), kind, *sorted([left, right])])
    return f"con_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


@dataclass
class UnresolvedQuestion:
    """An explicit hole in the evidence (requirement 1F/1G)."""

    question: str
    why_it_matters: str
    blocking: bool = False
    category: FactCategory = FactCategory.OTHER
    raised_by: str = UNKNOWN
