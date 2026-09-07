"""Shared fixtures."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from investment_research.schemas.enums import (  # noqa: E402
    ContentKind,
    EvidenceClass,
    FactCategory,
    Materiality,
    Provenance,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Fact, Source, make_fact_id  # noqa: E402
from investment_research.storage.db import open_db  # noqa: E402
from investment_research.storage.repository import Repository  # noqa: E402

TODAY = date(2026, 9, 6)


@pytest.fixture
def repo():
    conn = open_db(":memory:")
    try:
        yield Repository(conn)
    finally:
        conn.close()


@pytest.fixture
def fixture_dir() -> Path:
    return ROOT / "data" / "fixtures"


def make_fact(
    claim: str,
    *,
    ticker: str = "TEST",
    category: FactCategory = FactCategory.OTHER,
    evidence_class: EvidenceClass = EvidenceClass.VERIFIED_FACT,
    tier: SourceTier = SourceTier.TIER_1,
    url: str = "https://www.sec.gov/example",
    title: str = "Form 10-Q",
    event_date: str = "2026-06-30",
    publication_date: str = "2026-08-07",
    company_claim: bool = False,
    verified: VerifiedStatus = VerifiedStatus.VERIFIED,
    materiality: Materiality = Materiality.MEDIUM,
    value: str = "UNKNOWN",
    confidence: float = 0.8,
    independent_confirmation: bool = False,
    provenance: Provenance = Provenance.LIVE,
    run_id: str = "test-run",
    content_kind: ContentKind = ContentKind.FULL_DOCUMENT,
    primary_source_url: str | None = None,
) -> Fact:
    from investment_research.schemas.fact import make_source_id

    return Fact(
        fact_id=make_fact_id(ticker, category, claim, url, event_date),
        ticker=ticker,
        category=category,
        claim=claim,
        evidence_class=evidence_class,
        # Derived from the url: two facts from different documents must not
        # share a source id, or corroboration checks silently collapse.
        source_id=make_source_id(url, title),
        source_url=url,
        source_title=title,
        source_tier=tier,
        publication_date=publication_date,
        event_date=event_date,
        verified_status=verified,
        confidence=confidence,
        company_claim=company_claim,
        independent_confirmation=independent_confirmation,
        materiality=materiality,
        value=value,
        provenance=provenance,
        run_id=run_id,
        content_kind=content_kind,
        primary_source_url=primary_source_url,
    )


def make_source(
    url: str = "https://www.sec.gov/example",
    *,
    tier: SourceTier = SourceTier.TIER_1,
    title: str = "Form 10-Q",
    event_date: str = "2026-06-30",
    published_date: str = "2026-08-07",
    provenance: Provenance = Provenance.LIVE,
) -> Source:
    from investment_research.schemas.fact import make_source_id

    return Source(
        source_id=make_source_id(url, title),
        url=url,
        title=title,
        tier=tier,
        event_date=event_date,
        published_date=published_date,
        provenance=provenance,
    )
