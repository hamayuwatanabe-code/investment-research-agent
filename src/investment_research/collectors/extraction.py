"""Deterministic fact extraction from documents.

This is the rule-based half of the collection layer. It pulls out the things
that can be recognised reliably by pattern -- dated regulatory positions, cash
figures, runway statements, designations, endpoint language -- and attributes
each to the exact document and chunk it came from.

An LLM extractor can add semantic coverage on top of this, but this layer runs
first and always, because it is deterministic and free, and because the facts it
finds are the ones the Kill Gate depends on.

Every extracted fact carries the ``content_kind`` of its source document, so a
claim drawn from a search summary is visibly weaker than the same claim drawn
from the filing itself.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..schemas.enums import (
    UNKNOWN,
    FactCategory,
    Provenance,
)
from ..schemas.fact import RawFact, Source
from .base import CollectionResult
from .documents import Chunk, Document, chunk_document, split_sentences

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionRule:
    """One recognisable statement type."""

    key: str
    category: FactCategory
    pattern: re.Pattern[str]
    #: Materiality hint used downstream; the extractor never assigns a verdict.
    company_claim_default: bool = False
    unit: str = UNKNOWN


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


#: Ordered most-specific first. A sentence may match several rules; each match
#: becomes its own fact so the evidence set records every distinct statement.
RULES: tuple[ExtractionRule, ...] = (
    ExtractionRule(
        "endpoint_not_sufficient",
        FactCategory.REGULATORY,
        _rx(
            r"\b(?:endpoint|rvef|right ventricular ejection fraction)\b[^.]*?"
            r"(?:is not sufficient|not sufficient to demonstrate|does not (?:consider|agree)|"
            r"cannot (?:prove|demonstrate)|not (?:adequate|appropriate) to establish)"
        ),
    ),
    ExtractionRule(
        "no_longer_pivotal",
        FactCategory.REGULATORY,
        _rx(r"\bno longer\b[^.]*\b(?:pivotal|registrational)\b"),
    ),
    ExtractionRule(
        "regulator_recommends_endpoints",
        FactCategory.REGULATORY,
        _rx(r"\brecommend(?:ed|s)?\b[^.]*\b(?:mortality|survival|MACE|objective measures)\b"),
    ),
    ExtractionRule(
        "additional_trial_required",
        FactCategory.REGULATORY,
        _rx(
            r"\b(?:additional|another|second)\b[^.]{0,80}\b(?:trial|study)\b[^.]{0,100}?\b(?:required|necessary|needed)\b"
        ),
    ),
    ExtractionRule(
        "regulator_meeting",
        FactCategory.REGULATORY,
        _rx(r"\btype\s+[abcd]\s+meeting\b"),
    ),
    ExtractionRule(
        "designation",
        FactCategory.REGULATORY,
        _rx(
            r"\b(?:orphan drug|fast track|rare pediatric disease|breakthrough therapy|"
            r"RMAT|regenerative medicine advanced therapy|priority review)\b[^.]*\bdesignation\b"
        ),
        company_claim_default=True,
    ),
    ExtractionRule(
        "clinical_hold",
        FactCategory.REGULATORY,
        _rx(r"\bclinical hold\b"),
    ),
    ExtractionRule(
        "going_concern",
        FactCategory.LIQUIDITY,
        _rx(r"\b(?:substantial doubt|going concern)\b"),
        company_claim_default=True,
    ),
    ExtractionRule(
        "cash_position",
        FactCategory.FINANCIAL,
        _rx(
            r"\bcash(?: and cash equivalents| and equivalents)?\b[^.]*\$[\d,.]+ ?(?:million|billion|m|bn)?"
        ),
        company_claim_default=True,
        unit="USD",
    ),
    ExtractionRule(
        "runway",
        FactCategory.LIQUIDITY,
        _rx(r"\b(?:runway|fund operations|fund its operations)\b"),
        company_claim_default=True,
    ),
    ExtractionRule(
        "financing",
        FactCategory.CAPITAL_STRUCTURE,
        _rx(r"\b(?:private placement|public offering|at-the-market|registered direct|PIPE)\b"),
        company_claim_default=True,
        unit="USD",
    ),
    ExtractionRule(
        "dilution",
        FactCategory.CAPITAL_STRUCTURE,
        _rx(r"\bdilution\b"),
    ),
    ExtractionRule(
        "net_loss",
        FactCategory.FINANCIAL,
        _rx(r"\bnet loss\b"),
        company_claim_default=True,
        unit="USD",
    ),
    ExtractionRule(
        "dmc",
        FactCategory.CLINICAL,
        _rx(r"\b(?:data monitoring committee|DMC|DSMB)\b"),
        company_claim_default=True,
    ),
    ExtractionRule(
        "trial_design",
        FactCategory.CLINICAL,
        _rx(r"\b(?:randomi[sz]ed|double-blind|placebo-controlled|dose-ranging)\b"),
        company_claim_default=True,
    ),
    ExtractionRule(
        "enrollment",
        FactCategory.CLINICAL,
        _rx(r"\b(?:enrolled|enrollment|participants|patients)\b[^.]*\b\d{2,5}\b"),
        company_claim_default=True,
    ),
    ExtractionRule(
        "readout_timing",
        FactCategory.CATALYST,
        _rx(r"\b(?:topline|top-line|data (?:are|is) (?:expected|anticipated)|readout)\b"),
        company_claim_default=True,
    ),
    ExtractionRule(
        "nct_id",
        FactCategory.CLINICAL,
        _rx(r"\bNCT\d{8}\b"),
    ),
    ExtractionRule(
        "safety_class_history",
        FactCategory.SCIENCE,
        _rx(r"\b(?:neuropsychiatric|adverse event|abandoned|safety (?:signal|concern))\b"),
    ),
    ExtractionRule(
        "competitor",
        FactCategory.COMPETITION,
        _rx(r"\b(?:competitor|rival|compared to|versus|monlunabant|GLP-1)\b"),
    ),
    ExtractionRule(
        "listing_compliance",
        FactCategory.LISTING,
        _rx(r"\b(?:minimum bid price|non-?compliance|delisting|listing rule)\b"),
        company_claim_default=True,
    ),
    ExtractionRule(
        "pay_cut",
        FactCategory.GOVERNANCE,
        _rx(r"\bpay cuts?\b"),
    ),
    ExtractionRule(
        "equity_grant",
        FactCategory.GOVERNANCE,
        _rx(r"\b(?:restricted stock units?|RSUs?|equity grants?)\b"),
    ),
    ExtractionRule(
        "company_framing",
        FactCategory.REGULATORY,
        _rx(
            r"\b(?:constructive|productive|encouraging|collaborative|aligned|supportive|"
            r"favou?rable|on track)\b"
        ),
        company_claim_default=True,
        unit="company_characterization",
    ),
    ExtractionRule(
        "pipeline_asset",
        FactCategory.SCIENCE,
        _rx(
            r"\b(?:pipeline includes|pipeline comprises|next-generation|antibody drug conjugate)\b"
        ),
        company_claim_default=True,
    ),
    ExtractionRule(
        "programme_code",
        FactCategory.CLINICAL,
        _rx(r"\b[A-Z]{2,4}-\d{3}\b"),
    ),
    ExtractionRule(
        "market_size",
        FactCategory.MARKET_SIZE,
        _rx(r"\b(?:addressable market|prevalence|patients (?:in|with))\b"),
    ),
    ExtractionRule(
        "milestone_payment",
        FactCategory.COMMERCIAL,
        _rx(r"\b(?:milestone payments?|royalt(?:y|ies))\b"),
        company_claim_default=True,
    ),
)


def _source_for(document: Document) -> Source:
    return Source(
        source_id=document.doc_id,
        url=document.url,
        title=document.title,
        tier=document.tier,
        publisher=document.publisher,
        published_date=document.published_date,
        event_date=document.event_date,
        filing_date=document.filing_date,
        retrieved_at=document.retrieved_at,
        provenance=document.provenance,
        content_hash=document.content_hash(),
        excerpt=document.text[:2000],
        content_kind=document.content_kind,
    )


def extract_from_chunk(chunk: Chunk, ticker: str) -> list[RawFact]:
    """Apply every rule to each sentence of one chunk.

    Rules match whole sentences rather than running a ``[^.]*`` window over raw
    text: that window stops at the first period, so "a Type C meeting with the
    U.S. FDA" would be captured as "a Type C meeting with the U." -- losing the
    part of the statement that carries the meaning.
    """
    document = chunk.document
    source = _source_for(document)
    facts: list[RawFact] = []
    seen: set[tuple[str, str]] = set()

    for raw_sentence in split_sentences(chunk.text):
        sentence = " ".join(raw_sentence.split()).strip()
        if len(sentence) < 25:
            continue
        for rule in RULES:
            if not rule.pattern.search(sentence):
                continue
            key = (rule.key, sentence.lower())
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                RawFact(
                    ticker=ticker.upper(),
                    category=rule.category,
                    claim=sentence[:1200],
                    source=source,
                    value=UNKNOWN,
                    unit=rule.unit,
                    company_claim=document.is_company_ir or rule.company_claim_default,
                    collector=f"extraction:{rule.key}",
                    raw_payload_ref=chunk.chunk_id,
                )
            )
    return facts


def extract_documents(
    documents: Sequence[Document], ticker: str, *, target_tokens: int = 400
) -> tuple[list[RawFact], list[Chunk]]:
    """Chunk each document and extract facts from every chunk."""
    all_facts: list[RawFact] = []
    all_chunks: list[Chunk] = []
    for document in documents:
        chunks = chunk_document(document, target_tokens=target_tokens)
        all_chunks.extend(chunks)
        for chunk in chunks:
            all_facts.extend(extract_from_chunk(chunk, ticker))
    return _deduplicate(all_facts), all_chunks


def _deduplicate(facts: Iterable[RawFact]) -> list[RawFact]:
    seen: set[str] = set()
    out: list[RawFact] = []
    for fact in facts:
        fact_id = fact.fact_id()
        if fact_id in seen:
            continue
        seen.add(fact_id)
        out.append(fact)
    return out


class DocumentCollector:
    """Adapts a set of documents to the collector interface."""

    name = "documents"

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        provenance: Provenance = Provenance.LIVE,
        note: str = "",
    ) -> None:
        self.documents = list(documents)
        self.provenance = provenance
        self.note = note
        self.chunks: list[Chunk] = []

    def collect(self, ticker: str, company_name: str = UNKNOWN) -> CollectionResult:
        out = CollectionResult(collector=self.name, provenance=self.provenance)
        if self.note:
            out.notes.append(self.note)
        if not self.documents:
            from ..schemas.enums import FetchOutcome

            out.outcome = FetchOutcome.NOT_FOUND
            out.errors.append(f"no documents available for {ticker}")
            return out

        facts, chunks = extract_documents(self.documents, ticker)
        self.chunks = chunks
        out.raw_facts.extend(facts)
        out.sources.extend(_source_for(document) for document in self.documents)

        summary_only = [
            document.doc_id
            for document in self.documents
            if not document.content_kind.is_primary_text
        ]
        if summary_only:
            out.notes.append(
                f"{len(summary_only)}/{len(self.documents)} documents are summaries or metadata "
                "rather than the source text; claims drawn from them are marked for "
                "primary-source escalation"
            )
        out.notes.append(f"chunked into {len(chunks)} chunks; {len(facts)} candidate facts")
        return out
