"""Phase 3B requirement 8: Fact/RawFact dedup diagnostics.

A sentence matched by several extraction rules, or matched twice via
overlapping chunks, must collapse to a bounded set of facts -- never grow
without limit -- and the before/after count must be visible in diagnostics
rather than something a caller has to infer. Distinct claims must never be
coalesced just because they share some fields.
"""

from __future__ import annotations

from investment_research.collectors.documents import Document, chunk_document
from investment_research.collectors.extraction import (
    DocumentCollector,
    extract_documents,
    extract_documents_with_dedup_counts,
    extract_from_chunk,
)
from investment_research.schemas.enums import ContentKind, DocumentAuthority, Provenance

_MULTI_RULE_TEXT = (
    "TESTCO Holdings, Inc. today announced that, following a recent meeting, the "
    "regulatory authority indicated that it did not consider the Company's proposed "
    "primary endpoint appropriate to establish the effectiveness of the therapy. "
    "As a result, the ongoing study can no longer be characterized as pivotal or "
    "registrational."
)


def _document(doc_id: str = "doc_multi_rule") -> Document:
    return Document(
        doc_id=doc_id,
        url="https://www.sec.gov/Archives/testco/ex99-1.htm",
        title="Exhibit 99.1",
        text=_MULTI_RULE_TEXT,
        content_kind=ContentKind.FULL_DOCUMENT,
        authority=DocumentAuthority.STATUTORY_FILING,
        is_company_ir=True,
    )


def test_the_same_sentence_matched_by_two_rules_produces_two_distinct_category_facts_not_unbounded_duplicates():
    document = _document()
    chunks = chunk_document(document)
    raw_facts = extract_from_chunk(chunks[0], "TESTCO")
    matched_rules = {raw.collector for raw in raw_facts}
    assert "extraction:regulator_endpoint_not_accepted" in matched_rules
    assert "extraction:no_longer_pivotal" in matched_rules
    # Two distinct rules matched -- bounded (exactly one RawFact per rule
    # match on this text), never an unbounded/duplicated blow-up.
    assert len(raw_facts) == len(matched_rules)


def test_extracting_the_same_document_twice_deduplicates_to_the_same_fact_ids():
    """Deterministic re-extraction (e.g. the same chunk processed again due
    to overlap) must never double-count -- ``fact_id()`` is stable and
    ``_deduplicate`` collapses exact repeats."""
    document = _document()
    facts_first, _chunks, before_first, after_first = extract_documents_with_dedup_counts([document], "TESTCO")
    facts_second, _chunks2, before_second, after_second = extract_documents_with_dedup_counts(
        [document, document], "TESTCO"
    )
    assert before_first == after_first  # a single document has no cross-copy repeats to remove
    assert before_second == 2 * before_first  # extraction ran over the document twice
    assert after_second == after_first  # but dedup collapsed the repeat back down
    assert {f.fact_id() for f in facts_first} == {f.fact_id() for f in facts_second}


def test_document_collector_surfaces_before_and_after_dedup_counts():
    document = _document()
    collector = DocumentCollector([document, document], provenance=Provenance.FIXTURE)
    result = collector.collect("TESTCO", "Generic Biotech Holdings")
    assert result.raw_fact_count_before_dedup == 2 * len(result.raw_facts)
    assert any("before dedup" in note for note in result.notes)


def test_distinct_claims_about_different_programs_are_never_coalesced():
    """Two RawFacts about DIFFERENT drug programs, sharing category/date/
    source URL, must never collapse into one just because those fields
    match -- the literal claim/sentence text is part of ``fact_id()``, so a
    genuinely different program's sentence (naming a different program code)
    never dedups against another program's.

    Known limitation (reported, not silently worked around): ``RawFact``/
    ``Fact`` have no dedicated ``program_scope`` field yet -- this test
    demonstrates that the CURRENT claim-text-based dedup key already keeps
    distinct programs apart in practice, not that a first-class field
    enforces it. Adding an explicit ``program_scope`` field is a Phase 3C
    candidate (see the Phase 3B final report's unresolved items).
    """
    from investment_research.schemas.enums import FactCategory, SourceTier
    from investment_research.schemas.fact import RawFact, Source

    source = Source(source_id="src_1", url="https://example.test/doc", title="Doc", tier=SourceTier.TIER_1, event_date="2026-01-01")
    fact_a = RawFact(
        ticker="TESTCO", category=FactCategory.CLINICAL,
        claim="Program GBH-101 met its primary endpoint in the Phase 2 study.",
        source=source, company_claim=True, document_id="doc_1",
    )
    fact_b = RawFact(
        ticker="TESTCO", category=FactCategory.CLINICAL,
        claim="Program GBH-202 did not meet its primary endpoint in the Phase 2 study.",
        source=source, company_claim=True, document_id="doc_1",
    )
    assert fact_a.fact_id() != fact_b.fact_id()


def test_extract_documents_backward_compatible_signature_matches_the_counted_variant():
    document = _document()
    facts_plain, chunks_plain = extract_documents([document], "TESTCO")
    facts_counted, chunks_counted, _before, after = extract_documents_with_dedup_counts([document], "TESTCO")
    assert [f.fact_id() for f in facts_plain] == [f.fact_id() for f in facts_counted]
    assert len(chunks_plain) == len(chunks_counted)
    assert after == len(facts_counted)


# --- required fields survive extraction (Phase 3B requirement 8's list) ----
def test_required_fields_survive_from_chunk_to_raw_fact():
    document = _document()
    chunks = chunk_document(document)
    raw_facts = extract_from_chunk(chunks[0], "TESTCO")
    assert raw_facts
    for raw in raw_facts:
        # document_id
        assert raw.document_id == document.doc_id
        # chunk_id (carried as raw_payload_ref -- see RawFact's own docstring)
        assert raw.raw_payload_ref == chunks[0].chunk_id
        # supporting sentence/quote
        assert raw.claim
        # rule/category
        assert raw.collector.startswith("extraction:")
        assert raw.category is not None
        # company_claim (authority itself is only reachable via the
        # Document/DocumentStore, not carried on RawFact directly -- see
        # test_phase3a_regulatory_rejection_fixture.py's
        # test_authority_is_traceable_via_the_document_store)
        assert raw.company_claim is True
