"""Collector protocol and the collection report.

Every collector returns a :class:`CollectionResult` that states, per source,
whether the data was obtained.  Requirement 14/24: a partial failure must be
visible in the final report, never smoothed over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..schemas.enums import FetchOutcome, Provenance
from ..schemas.fact import RawFact, Source


@dataclass
class CollectionResult:
    collector: str
    raw_facts: list[RawFact] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    outcome: FetchOutcome = FetchOutcome.OK
    provenance: Provenance = Provenance.LIVE
    errors: list[str] = field(default_factory=list)
    attempted_urls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome == FetchOutcome.OK and not self.errors

    @property
    def degraded(self) -> bool:
        return self.outcome != FetchOutcome.OK

    def describe(self) -> str:
        return (
            f"{self.collector}: {self.outcome} "
            f"({len(self.raw_facts)} facts, {len(self.sources)} sources)"
            + (f" errors={self.errors[:2]}" if self.errors else "")
        )


class Collector(Protocol):
    name: str

    def collect(self, ticker: str, company_name: str) -> CollectionResult:  # pragma: no cover
        ...
