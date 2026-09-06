"""Document model, chunking and per-agent relevance selection (requirement P9).

The cost problem this solves: a 10-K is ~150k tokens. Fourteen agents each
reading it whole is both ruinously expensive and worse analysis -- an agent
asked to find one regulatory sentence in 150k tokens does it less reliably than
one handed the twelve relevant chunks.

So the flow is:

    document -> chunk -> score each chunk for one agent -> evidence pack

Selection is deterministic and rule-based (keyword and category scoring), not a
model call, so building a pack costs nothing and is reproducible. The LLM spends
its tokens on judgement, not on retrieval.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..schemas.enums import (
    UNKNOWN,
    ContentKind,
    Provenance,
    ResearchPath,
    SourceTier,
)

log = logging.getLogger(__name__)

#: Rough token estimate. The real tokenizer lives behind the API; for budgeting
#: a chunker, 4 characters per token is close enough and costs nothing.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN)


@dataclass
class Document:
    """One retrieved (or captured) source document."""

    doc_id: str
    url: str
    title: str
    publisher: str = UNKNOWN
    published_date: str = UNKNOWN
    event_date: str = UNKNOWN
    filing_date: str = UNKNOWN
    doc_type: str = "unknown"
    is_company_ir: bool = False
    text: str = ""
    content_kind: ContentKind = ContentKind.FULL_DOCUMENT
    provenance: Provenance = Provenance.LIVE
    research_path: ResearchPath = ResearchPath.NONE
    retrieved_at: str = UNKNOWN
    tier: SourceTier = SourceTier.UNKNOWN

    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:32]


@dataclass
class Chunk:
    """A slice of a document, carrying its provenance forward."""

    chunk_id: str
    doc_id: str
    index: int
    text: str
    document: Document

    def token_estimate(self) -> int:
        return estimate_tokens(self.text)


#: Abbreviations whose trailing period is not a sentence end. Splitting
#: "a Type C meeting with the U.S. FDA" after "U." truncates the statement and
#: can hide the very sentence the analysis turns on.
_ABBREVIATIONS = (
    "U.S.A", "Ph.D", "U.S", "U.K", "e.g", "i.e", "Inc", "Corp", "Ltd", "Co",
    "Dr", "Mr", "Ms", "St", "No", "vs", "approx", "etc", "Fig", "al", "Jr", "Sr",
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_PERIOD_SENTINEL = "\u0000P\u0000"


def split_sentences(text: str) -> list[str]:
    """Split into sentences, protecting abbreviation periods first.

    Python's ``re`` has no variable-width lookbehind, so rather than encode the
    abbreviation list into the boundary pattern, the periods inside known
    abbreviations are swapped for a sentinel, the split runs, and the sentinel is
    restored.
    """
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = re.sub(
            rf"\b{re.escape(abbreviation)}\.",
            abbreviation + _PERIOD_SENTINEL,
            protected,
        )
    parts = _SENTENCE_END.split(protected)
    return [p.replace(_PERIOD_SENTINEL, ".").strip() for p in parts if p.strip()]


def chunk_document(
    document: Document, *, target_tokens: int = 400, overlap_sentences: int = 1
) -> list[Chunk]:
    """Split on sentence boundaries into ~``target_tokens`` chunks.

    Sentence boundaries matter here: a regulatory finding cut in half mid-clause
    ("the FDA does not consider the primary endpoint / appropriate to establish
    effectiveness") can defeat both the pattern matcher and the model.
    """
    text = (document.text or "").strip()
    if not text:
        return []
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = " ".join(current).strip()
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#c{index}",
                doc_id=document.doc_id,
                index=index,
                text=body,
                document=document,
            )
        )
        current = current[-overlap_sentences:] if overlap_sentences else []
        current_tokens = sum(estimate_tokens(s) for s in current)

    for sentence in sentences:
        tokens = estimate_tokens(sentence)
        if current and current_tokens + tokens > target_tokens:
            flush()
        current.append(sentence)
        current_tokens += tokens
    # final flush must not re-seed overlap
    if current:
        body = " ".join(current).strip()
        if not chunks or chunks[-1].text != body:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#c{len(chunks)}",
                    doc_id=document.doc_id,
                    index=len(chunks),
                    text=body,
                    document=document,
                )
            )
    return chunks


#: Per-agent relevance vocabulary. Weighted: a term in the first list is worth
#: more than one in the second, so a chunk that actually states a regulator
#: position outranks one that merely mentions the FDA.
AGENT_KEYWORDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "regulatory": (
        (
            "does not consider",
            "not sufficient to demonstrate",
            "no longer refers",
            "not appropriate to establish",
            "clinical hold",
            "complete response letter",
            "refuse to file",
            "special protocol assessment",
            "type a meeting",
            "type b meeting",
            "type c meeting",
            "endpoint",
            "pivotal",
            "accelerated approval",
            "surrogate",
        ),
        ("fda", "ema", "pmda", "regulator", "agency", "designation", "approval", "ind", "bla", "nda"),
    ),
    "science": (
        (
            "primary endpoint",
            "primary outcome",
            "randomized",
            "double-blind",
            "placebo-controlled",
            "sample size",
            "statistical power",
            "effect size",
            "interim analysis",
            "brain penetration",
            "adverse event",
        ),
        ("trial", "study", "phase", "cohort", "dose", "patients", "efficacy", "safety", "mechanism"),
    ),
    "capital_structure": (
        (
            "going concern",
            "substantial doubt",
            "cash runway",
            "private placement",
            "at-the-market",
            "shelf registration",
            "pre-funded warrant",
            "fully diluted",
            "reverse split",
            "dilution",
        ),
        ("cash", "shares", "warrant", "financing", "offering", "burn", "debt", "equity", "rsu"),
    ),
    "competitive": (
        ("market share", "competitor", "standard of care", "head-to-head", "addressable market"),
        ("compete", "rival", "class", "approved", "launch", "pricing", "prevalence"),
    ),
    "kill_agent": (
        (
            "going concern",
            "clinical hold",
            "does not consider",
            "not sufficient to demonstrate",
            "delisting",
            "restatement",
            "investigation",
            "lawsuit",
            "material weakness",
            "failed",
            "terminated",
        ),
        ("risk", "concern", "doubt", "decline", "loss", "dilution", "auditor", "short"),
    ),
    "bear_agent": (
        ("going concern", "dilution", "competitor", "not sufficient", "failed", "risk factor"),
        ("cash", "burn", "endpoint", "delay", "safety", "runway"),
    ),
    "bull_agent": (
        ("designation", "positive", "met the primary", "granted", "milestone", "partnership"),
        ("efficacy", "growth", "revenue", "approval", "cash", "pipeline"),
    ),
    "contradiction": (
        ("constructive", "pivotal", "no longer", "does not consider", "however", "previously"),
        ("fda", "endpoint", "guidance", "expects", "agreed"),
    ),
    "catalyst": (
        ("topline", "readout", "data expected", "primary completion", "pdufa", "anticipated in"),
        ("quarter", "2026", "2027", "september", "august", "milestone"),
    ),
}

#: Terms that make a chunk important to every agent.
_UNIVERSAL_SIGNALS = (
    "going concern",
    "does not consider",
    "not sufficient to demonstrate",
    "clinical hold",
    "no longer refers",
)


def score_chunk(chunk: Chunk, agent_id: str) -> float:
    """Relevance of one chunk to one agent. Deterministic, no model call."""
    strong, weak = AGENT_KEYWORDS.get(agent_id, ((), ()))
    text = chunk.text.lower()
    score = 0.0
    for term in strong:
        if term in text:
            score += 3.0
    for term in weak:
        if term in text:
            score += 1.0
    for term in _UNIVERSAL_SIGNALS:
        if term in text:
            score += 4.0
    # Primary sources and primary text outrank paraphrase at equal keyword score.
    score *= 1.0 + (0.3 if chunk.document.tier.is_primary else 0.0)
    score *= chunk.document.content_kind.confidence_multiplier
    return round(score, 3)


@dataclass
class EvidencePack:
    """The bounded slice of the corpus one agent actually reads."""

    agent_id: str
    chunks: list[Chunk] = field(default_factory=list)
    total_tokens: int = 0
    considered_chunks: int = 0
    dropped_chunks: int = 0
    budget_tokens: int = 0

    def render(self, max_chars_per_chunk: int = 2000) -> str:
        """Render the pack for a prompt, every chunk carrying its citation."""
        parts: list[str] = []
        for chunk in self.chunks:
            document = chunk.document
            parts.append(
                f"[{chunk.chunk_id}] source_id={document.doc_id} "
                f"tier={document.tier} content_kind={document.content_kind} "
                f"published={document.published_date} event={document.event_date}\n"
                f"title: {document.title}\n"
                f"url: {document.url}\n"
                f"text: {chunk.text[:max_chars_per_chunk]}"
            )
        return "\n\n---\n\n".join(parts)

    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.doc_id not in seen:
                seen.append(chunk.doc_id)
        return seen


def build_evidence_pack(
    chunks: Sequence[Chunk],
    agent_id: str,
    *,
    budget_tokens: int = 12000,
    min_score: float = 1.0,
) -> EvidencePack:
    """Select the highest-scoring chunks for one agent within a token budget.

    Ties and near-ties are broken toward primary sources, and a chunk carrying a
    universal signal (a going-concern disclosure, an adverse regulator position)
    is effectively always selected because its score dominates.
    """
    scored = [(score_chunk(chunk, agent_id), chunk) for chunk in chunks]
    scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))

    pack = EvidencePack(
        agent_id=agent_id, considered_chunks=len(chunks), budget_tokens=budget_tokens
    )
    for score, chunk in scored:
        if score < min_score:
            pack.dropped_chunks += 1
            continue
        tokens = chunk.token_estimate()
        if pack.total_tokens + tokens > budget_tokens:
            pack.dropped_chunks += 1
            continue
        pack.chunks.append(chunk)
        pack.total_tokens += tokens
    # Restore document order so the agent reads a coherent narrative.
    pack.chunks.sort(key=lambda c: (c.doc_id, c.index))
    return pack


def chunk_corpus(documents: Iterable[Document], **kwargs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, **kwargs))
    return chunks
