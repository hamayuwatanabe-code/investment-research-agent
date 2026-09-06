"""Citation validation (requirement 15: hallucination protection).

Every URL and fact reference that reaches the reader must resolve to a source
that was actually retrieved during the run.  A reference that does not resolve
is not silently dropped -- it is replaced with an explicit ``[UNVERIFIED
CITATION]`` marker, because a quietly-removed fabrication looks exactly like
prose that was always clean.

Also checks the shapes of identifiers that are easy to invent convincingly:
SEC accession numbers and NCT registry ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas.fact import Fact, Source

_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+|fixture://[^\s\)\]\"'<>]+")
_ACCESSION_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
_NCT_RE = re.compile(r"\bNCT\d{8}\b")
_FACT_REF_RE = re.compile(r"\bfact_[0-9a-f]{20}\b")

UNVERIFIED_MARKER = "[UNVERIFIED CITATION]"


@dataclass
class CitationReport:
    checked_urls: int = 0
    unknown_urls: list[str] = field(default_factory=list)
    unknown_fact_ids: list[str] = field(default_factory=list)
    malformed_accessions: list[str] = field(default_factory=list)
    malformed_nct: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (
            self.unknown_urls
            or self.unknown_fact_ids
            or self.malformed_accessions
            or self.malformed_nct
        )


class CitationValidator:
    """Validates references against the sources actually retrieved."""

    def __init__(self, sources: list[Source], facts: list[Fact]) -> None:
        self.known_urls = {s.url for s in sources} | {f.source_url for f in facts}
        self.known_fact_ids = {f.fact_id for f in facts}
        self.known_accessions = {
            s.accession for s in sources if s.accession and s.accession != "UNKNOWN"
        }
        self.known_nct = {
            match
            for f in facts
            for match in _NCT_RE.findall(f.claim)
        }

    def check(self, text: str) -> CitationReport:
        report = CitationReport()
        for url in _URL_RE.findall(text or ""):
            report.checked_urls += 1
            if url.rstrip(".,;") not in self.known_urls and url not in self.known_urls:
                report.unknown_urls.append(url)
        for fact_id in _FACT_REF_RE.findall(text or ""):
            if fact_id not in self.known_fact_ids:
                report.unknown_fact_ids.append(fact_id)
        for accession in _ACCESSION_RE.findall(text or ""):
            if self.known_accessions and accession not in self.known_accessions:
                report.malformed_accessions.append(accession)
        for nct in _NCT_RE.findall(text or ""):
            if self.known_nct and nct not in self.known_nct:
                report.malformed_nct.append(nct)
        return report

    def sanitize(self, text: str) -> tuple[str, CitationReport]:
        """Replace unresolvable references with a visible marker."""
        report = self.check(text)
        out = text
        for url in set(report.unknown_urls):
            out = out.replace(url, f"{UNVERIFIED_MARKER}({url})")
        for fact_id in set(report.unknown_fact_ids):
            out = out.replace(fact_id, f"{UNVERIFIED_MARKER}({fact_id})")
        return out, report
