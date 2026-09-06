"""Fixture (mock) data provider.

Requirement 24: mock and production data are kept unmistakably separate.

Three mechanisms enforce that here:

1. Every fixture source URL uses the ``fixture://`` scheme, so a fixture link
   can never be mistaken for -- or clicked as -- a real citation.
2. Every fixture source carries ``Provenance.FIXTURE``.
3. The reporter refuses to render a run containing fixture provenance without a
   prominent SYNTHETIC DATA banner, and such a run is never marked COMPLETE.

The fixture companies are **synthetic**.  They are not real issuers and their
facts are not claims about any real company.  ``DEMOBIO`` deliberately
reproduces the *shape* of the failure this system was built to prevent: an
attractive small-cap clinical story whose one disqualifying fact is that the
regulator does not accept the primary endpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..schemas.enums import UNKNOWN, FactCategory, FetchOutcome, Provenance, SourceTier
from ..schemas.fact import RawFact, Source, make_source_id
from .base import CollectionResult

log = logging.getLogger(__name__)

FIXTURE_SCHEME = "fixture://"


class FixtureCollector:
    """Serves raw facts from ``data/fixtures/<TICKER>.json``."""

    name = "fixtures"

    def __init__(self, fixture_dir: str | Path) -> None:
        self.fixture_dir = Path(fixture_dir)

    def available(self) -> list[str]:
        return sorted(p.stem.upper() for p in self.fixture_dir.glob("*.json"))

    def load(self, ticker: str) -> dict[str, Any] | None:
        path = self.fixture_dir / f"{ticker.upper()}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.error("malformed fixture %s: %s", path, exc)
            return None

    def collect(self, ticker: str, company_name: str = UNKNOWN) -> CollectionResult:
        out = CollectionResult(
            collector=self.name, provenance=Provenance.FIXTURE
        )
        payload = self.load(ticker)
        if payload is None:
            out.outcome = FetchOutcome.NOT_FOUND
            out.errors.append(f"no fixture for {ticker}; available: {self.available()}")
            return out

        out.notes.append(
            "SYNTHETIC FIXTURE DATA -- not real research, not a claim about any real issuer"
        )
        for entry in payload.get("facts", []):
            source_spec = entry.get("source", {})
            url = source_spec.get("url", "")
            if not url.startswith(FIXTURE_SCHEME):
                out.errors.append(
                    f"fixture source url must use the {FIXTURE_SCHEME} scheme: {url!r}"
                )
                continue
            source = Source(
                source_id=make_source_id(url, source_spec.get("title", "")),
                url=url,
                title=source_spec.get("title", UNKNOWN),
                tier=SourceTier(source_spec.get("tier", "UNKNOWN")),
                publisher=source_spec.get("publisher", UNKNOWN),
                published_date=source_spec.get("published_date", UNKNOWN),
                event_date=source_spec.get("event_date", UNKNOWN),
                effective_date=source_spec.get("effective_date", UNKNOWN),
                filing_date=source_spec.get("filing_date", UNKNOWN),
                accession=source_spec.get("accession", UNKNOWN),
                provenance=Provenance.FIXTURE,
                excerpt=source_spec.get("excerpt", ""),
                content_hash=source_spec.get("content_hash", UNKNOWN),
            )
            out.sources.append(source)
            out.raw_facts.append(
                RawFact(
                    ticker=ticker.upper(),
                    category=FactCategory(entry.get("category", "OTHER")),
                    claim=entry["claim"],
                    source=source,
                    value=entry.get("value", UNKNOWN),
                    unit=entry.get("unit", UNKNOWN),
                    company_claim=bool(entry.get("company_claim", False)),
                    collector=self.name,
                )
            )
        return out

    def metadata(self, ticker: str) -> dict[str, Any]:
        payload = self.load(ticker) or {}
        return payload.get("company", {})
