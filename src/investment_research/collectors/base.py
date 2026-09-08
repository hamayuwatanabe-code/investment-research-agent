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
    #: True only when a specific collector's own semantic boundary has
    #: determined that a query executed successfully and definitively found
    #: nothing -- e.g. Drugs@FDA returning no applications for a sponsor. This
    #: is never inferred generically from an HTTP-layer outcome (a 404/
    #: NOT_FOUND is not, by itself, evidence of a clean zero-result search):
    #: only the collector that actually knows what its API's 404 means may
    #: set this, and in doing so it must also set ``outcome`` to ``OK`` --
    #: this flag exists purely to make that determination explicit and
    #: auditable, not to change what "degraded" means generically.
    zero_results: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome == FetchOutcome.OK and not self.errors

    @property
    def degraded(self) -> bool:
        """Whether this collection represents a failure worth flagging.

        Generic and collector-agnostic: any outcome other than ``OK`` is
        degraded, full stop. ``NOT_FOUND`` (including an HTTP 404) is NOT
        globally reinterpreted as a successful zero-result search here --
        that would silently paper over a collector that genuinely could not
        resolve its query. A collector that knows its own API well enough to
        tell "zero results" apart from "not found" (see ``zero_results``)
        must say so explicitly by returning ``outcome=OK``; this property
        does not guess on any collector's behalf.
        """
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
