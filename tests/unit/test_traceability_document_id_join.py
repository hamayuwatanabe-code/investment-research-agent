"""Phase 4.1B (traceability fix): ``build_index``'s chunk-to-fact join must
key on ``Fact.document_id`` when present, never on ``Fact.source_id`` alone.

The bug this pins down: Phase 4.1A's Literature bridge sets
``Source.source_id = make_source_id(document.url, document.title)`` (a
hash) while ``Fact.document_id`` carries the real ``Document.doc_id`` --
the SAME identity ``Chunk.doc_id`` carries (``chunk_document()`` always
sets ``doc_id=document.doc_id``). ``build_index``'s old
``doc_to_fact.setdefault(fact.source_id, fact.fact_id)`` could never
resolve a Literature chunk to its fact, since the keys never matched.

No network, no LLM, no AcquisitionExecutor -- direct construction against
the same ``Document``/``Chunk``/``Fact``/``Source`` types the real bridges
use, isolating this fix from everything else in the pipeline.
"""

from __future__ import annotations

from investment_research.collectors.documents import Document, chunk_document
from investment_research.reporting.traceability import build_index
from investment_research.schemas.enums import (
    ContentKind,
    EvidenceClass,
    FactCategory,
    SourceTier,
)
from investment_research.schemas.fact import Fact, make_source_id


def _document(doc_id: str, *, url: str, title: str, text: str) -> Document:
    return Document(
        doc_id=doc_id,
        url=url,
        title=title,
        text=text,
        content_kind=ContentKind.EXCERPT,
    )


def _fact(*, fact_id: str, source_id: str, document_id: str | None) -> Fact:
    return Fact(
        fact_id=fact_id,
        ticker="DEMOBIO",
        category=FactCategory.SCIENCE,
        claim="a claim drawn from the chunked document",
        evidence_class=EvidenceClass.INDEPENDENT_EVIDENCE,
        source_id=source_id,
        source_url="https://example.org/irrelevant-to-the-join",
        source_title="irrelevant",
        source_tier=SourceTier.TIER_2,
        document_id=document_id,
    )


# --- (a) source_id != document_id still resolves correctly -------------------
def test_chunk_resolves_via_document_id_when_source_id_differs():
    document_id = "doc_lit_pubmed_aaaa"
    document = _document(
        document_id, url="https://pubmed.ncbi.nlm.nih.gov/90000001/", title="A Trial", text="Sentence one. Sentence two."
    )
    chunks = chunk_document(document, target_tokens=100)
    assert chunks

    hashed_source_id = make_source_id(document.url, document.title)
    assert hashed_source_id != document_id  # the exact divergence this fix addresses

    fact = _fact(fact_id="fact_pubmed_1", source_id=hashed_source_id, document_id=document_id)

    index = build_index([fact], [], chunks)
    for chunk in chunks:
        link = index.resolve(chunk.chunk_id)
        assert link is not None
        assert link.fact_id == fact.fact_id


# --- (b) never cross-connects to a different Document's Fact -----------------
def test_chunk_never_cross_connects_to_a_different_documents_fact():
    doc_a_id = "doc_lit_pubmed_aaaa"
    doc_b_id = "doc_lit_epmc_search_bbbb"
    document_a = _document(
        doc_a_id, url="https://pubmed.ncbi.nlm.nih.gov/90000001/", title="Doc A", text="A one. A two."
    )
    document_b = _document(
        doc_b_id, url="https://www.ebi.ac.uk/europepmc/search", title="Doc B", text="B one. B two."
    )
    chunks_a = chunk_document(document_a, target_tokens=100)
    chunks_b = chunk_document(document_b, target_tokens=100)

    fact_a = _fact(
        fact_id="fact_a", source_id=make_source_id(document_a.url, document_a.title), document_id=doc_a_id
    )
    fact_b = _fact(
        fact_id="fact_b", source_id=make_source_id(document_b.url, document_b.title), document_id=doc_b_id
    )

    index = build_index([fact_a, fact_b], [], [*chunks_a, *chunks_b])
    for chunk in chunks_a:
        assert index.resolve(chunk.chunk_id).fact_id == fact_a.fact_id
    for chunk in chunks_b:
        assert index.resolve(chunk.chunk_id).fact_id == fact_b.fact_id


# --- (c) legacy Fact with document_id=None keeps existing behavior -----------
def test_legacy_fact_with_no_document_id_keeps_source_id_fallback():
    """Mirrors ``collectors/extraction.py``'s own convention, where
    ``source_id == document.doc_id`` directly (never a hash) -- the
    pre-existing path this fix must not disturb."""
    document_id = "doc_extraction_legacy_1"
    document = _document(
        document_id, url="https://www.sec.gov/legacy-filing", title="Legacy Filing", text="Old one. Old two."
    )
    chunks = chunk_document(document, target_tokens=100)

    legacy_fact = _fact(fact_id="fact_legacy", source_id=document_id, document_id=None)

    index = build_index([legacy_fact], [], chunks)
    for chunk in chunks:
        link = index.resolve(chunk.chunk_id)
        assert link is not None
        assert link.fact_id == legacy_fact.fact_id


# --- (d) Literature's PubMed/search/fulltext lineage is never mixed ----------
def test_literature_three_document_lineage_never_mixed_in_chunk_resolution():
    pubmed_id = "doc_lit_pubmed_9999"
    search_id = "doc_lit_epmc_search_9999"
    fulltext_id = "doc_lit_epmc_fulltext_9999"

    pubmed_doc = _document(
        pubmed_id, url="https://pubmed.ncbi.nlm.nih.gov/90000008/", title="PubMed Doc",
        text="Pubmed sentence one. Pubmed sentence two.",
    )
    fulltext_doc = _document(
        fulltext_id, url="https://www.ebi.ac.uk/europepmc/fullTextXML/PMC9990008",
        title="Fulltext Doc", text="Fulltext sentence one. Fulltext sentence two.",
    )
    # The Europe PMC search-response document is METADATA_ONLY -- never
    # chunked (Phase 4.1B's own exclusion rule) -- included here only to
    # prove its Fact still resolves correctly via source_urls/no chunks,
    # and is never confused with the other two documents' chunks.
    search_doc = Document(
        doc_id=search_id,
        url="https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        title="Search Doc",
        text="",
        content_kind=ContentKind.METADATA_ONLY,
    )

    pubmed_chunks = chunk_document(pubmed_doc, target_tokens=100)
    fulltext_chunks = chunk_document(fulltext_doc, target_tokens=100)

    pubmed_fact = _fact(
        fact_id="fact_pubmed",
        source_id=make_source_id(pubmed_doc.url, pubmed_doc.title),
        document_id=pubmed_id,
    )
    search_fact = _fact(
        fact_id="fact_search",
        source_id=make_source_id(search_doc.url, search_doc.title),
        document_id=search_id,
    )
    fulltext_fact = _fact(
        fact_id="fact_fulltext",
        source_id=make_source_id(fulltext_doc.url, fulltext_doc.title),
        document_id=fulltext_id,
    )

    index = build_index(
        [pubmed_fact, search_fact, fulltext_fact], [], [*pubmed_chunks, *fulltext_chunks]
    )
    for chunk in pubmed_chunks:
        assert index.resolve(chunk.chunk_id).fact_id == pubmed_fact.fact_id
    for chunk in fulltext_chunks:
        assert index.resolve(chunk.chunk_id).fact_id == fulltext_fact.fact_id
    # No chunk from either document ever resolves to the search fact.
    all_chunk_ids = {c.chunk_id for c in [*pubmed_chunks, *fulltext_chunks]}
    assert all(index.resolve(cid).fact_id != search_fact.fact_id for cid in all_chunk_ids)
