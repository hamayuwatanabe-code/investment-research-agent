"""Citation traceability (requirement P7).

Every material claim in the final report must be traceable:

    claim -> fact_id -> source_id -> exact URL -> publication_date -> event_date

and the report carries the index that makes the chain walkable.

The second half is the enforcement. Sentences that *attribute* something -- "the
FDA said", "an analyst expects", "the study showed" -- are found in the narrative
text, and any that carries no resolvable citation is a violation. That phrasing
is precisely how an unsourced assertion acquires the authority of a source, so
it is the phrasing the validator hunts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..collectors.documents import Chunk
from ..schemas.enums import UNKNOWN
from ..schemas.fact import Fact, Source

log = logging.getLogger(__name__)

_FACT_REF = re.compile(r"\bfact_[0-9a-f]{20}\b")
_CHUNK_REF = re.compile(r"\[[^\]\s]+#c\d+\]")
_DOC_REF = re.compile(r"\[([A-Za-z0-9_\-]{4,})\]")
_URL_REF = re.compile(r"https?://[^\s\)\]\"'<>]+")

#: Verbs and phrasings that assert something on someone else's authority.
ATTRIBUTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("regulator", re.compile(
        r"(?i)\b(?:FDA|EMA|PMDA|MHRA|the agency|the regulator)\b\s+"
        r"(?:has |had |have )?(?:said|stated|advised|told|indicated|noted|determined|"
        r"concluded|agreed|refused|rejected|recommended|requires?|required)")),
    ("analyst", re.compile(
        r"(?i)\b(?:analysts?|the street|sell-side|broker)\b\s+"
        r"(?:said|expects?|estimates?|believes?|rates?|maintains?|forecasts?)")),
    ("study", re.compile(
        r"(?i)\b(?:the )?(?:study|trial|paper|research|data|results?)\b\s+"
        r"(?:showed|shows|demonstrated|demonstrates|found|indicated|suggests?|reported)")),
    ("company", re.compile(
        r"(?i)\bthe (?:company|management|issuer)\b\s+"
        r"(?:said|stated|disclosed|reported|announced|guided|expects?)")),
    ("filing", re.compile(
        r"(?i)\b(?:the )?(?:filing|10-K|10-Q|8-K|prospectus|registration statement)\b\s+"
        r"(?:said|states?|discloses?|reported|shows?)")),
)


@dataclass
class TraceLink:
    """One walkable claim -> source chain."""

    claim: str
    fact_id: str
    source_id: str
    url: str
    publication_date: str
    event_date: str
    source_tier: str
    evidence_class: str
    verified_status: str
    content_kind: str = UNKNOWN

    def as_row(self) -> tuple[str, ...]:
        return (
            self.fact_id,
            self.source_id,
            self.url,
            self.publication_date,
            self.event_date,
            self.source_tier,
            self.verified_status,
        )


@dataclass
class TraceabilityIndex:
    """The report's citation index."""

    links: dict[str, TraceLink] = field(default_factory=dict)
    chunk_to_fact: dict[str, str] = field(default_factory=dict)
    source_urls: dict[str, str] = field(default_factory=dict)

    def resolve(self, reference: str) -> TraceLink | None:
        if reference in self.links:
            return self.links[reference]
        fact_id = self.chunk_to_fact.get(reference)
        return self.links.get(fact_id) if fact_id else None

    def known_references(self) -> set[str]:
        return set(self.links) | set(self.chunk_to_fact) | set(self.source_urls)

    def rows(self) -> list[TraceLink]:
        return sorted(self.links.values(), key=lambda link: (link.source_tier, link.fact_id))


def build_index(
    facts: Sequence[Fact], sources: Sequence[Source] = (), chunks: Sequence[Chunk] = ()
) -> TraceabilityIndex:
    index = TraceabilityIndex()
    by_source = {source.source_id: source for source in sources}

    for fact in facts:
        source = by_source.get(fact.source_id)
        index.links[fact.fact_id] = TraceLink(
            claim=fact.claim,
            fact_id=fact.fact_id,
            source_id=fact.source_id,
            url=fact.source_url,
            publication_date=fact.publication_date,
            event_date=fact.event_date,
            source_tier=str(fact.source_tier),
            evidence_class=str(fact.evidence_class),
            verified_status=str(fact.verified_status),
            content_kind=str(getattr(source, "content_kind", UNKNOWN)) if source else UNKNOWN,
        )
        index.source_urls[fact.source_id] = fact.source_url

    # A chunk reference resolves to any fact extracted from the same document.
    doc_to_fact: dict[str, str] = {}
    for fact in facts:
        doc_to_fact.setdefault(fact.source_id, fact.fact_id)
    for chunk in chunks:
        fact_id = doc_to_fact.get(chunk.doc_id)
        if fact_id:
            index.chunk_to_fact[chunk.chunk_id] = fact_id
        index.source_urls.setdefault(chunk.doc_id, chunk.document.url)

    return index


@dataclass
class AttributionViolation:
    kind: str
    sentence: str
    section: str

    def describe(self) -> str:
        return f"[{self.section}] unattributed {self.kind} claim: {self.sentence[:220]}"


@dataclass
class TraceabilityReport:
    violations: list[AttributionViolation] = field(default_factory=list)
    checked_sentences: int = 0
    attributed_sentences: int = 0
    resolved_references: int = 0
    unresolved_references: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.violations and not self.unresolved_references


def _has_citation(sentence: str, index: TraceabilityIndex) -> bool:
    if _URL_REF.search(sentence):
        return True
    for reference in _FACT_REF.findall(sentence):
        if index.resolve(reference):
            return True
    for reference in _CHUNK_REF.findall(sentence):
        if index.resolve(reference.strip("[]")):
            return True
    for reference in _DOC_REF.findall(sentence):
        if index.resolve(reference) or reference in index.source_urls:
            return True
    return False


def check_attribution(
    text: str, index: TraceabilityIndex, *, section: str = "report"
) -> TraceabilityReport:
    """Find attributed sentences with no resolvable citation."""
    from ..collectors.documents import split_sentences

    report = TraceabilityReport()
    for sentence in split_sentences(text):
        report.checked_sentences += 1
        matched: str | None = None
        for kind, pattern in ATTRIBUTION_PATTERNS:
            if pattern.search(sentence):
                matched = kind
                break
        if matched is None:
            continue
        report.attributed_sentences += 1
        if _has_citation(sentence, index):
            report.resolved_references += 1
        else:
            report.violations.append(
                AttributionViolation(kind=matched, sentence=sentence.strip(), section=section)
            )
    return report


def merge(reports: Iterable[TraceabilityReport]) -> TraceabilityReport:
    merged = TraceabilityReport()
    for report in reports:
        merged.violations.extend(report.violations)
        merged.checked_sentences += report.checked_sentences
        merged.attributed_sentences += report.attributed_sentences
        merged.resolved_references += report.resolved_references
        merged.unresolved_references.extend(report.unresolved_references)
    return merged
