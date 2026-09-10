"""Phase 1: Document.doc_id -> Chunk -> RawFact.document_id -> Fact.document_id.

Runs the real, already-wired collection path (``DocumentCollector`` ->
``extract_documents``/``extract_from_chunk`` -> ``FactCollectorAgent``) end to
end over a synthetic ("TESTCO") document, and asserts the document identity
survives every hop -- the exact lineage gap this Phase 1 change closes.
"""

from __future__ import annotations

from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.documents import Document, chunk_document
from investment_research.collectors.extraction import DocumentCollector, extract_from_chunk
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import ContentKind, DocumentAuthority, Provenance

_TEXT = (
    "TESTCO reported cash and cash equivalents of $42.0 million as of quarter end. "
    "The company said the going concern doubt raised in the prior filing has been resolved."
)


def _testco_document() -> Document:
    return Document(
        doc_id="doc_testco_10q_2026q2",
        url="https://www.sec.gov/Archives/testco/10q.htm",
        title="TESTCO 10-Q",
        text=_TEXT,
        content_kind=ContentKind.FULL_DOCUMENT,
        authority=DocumentAuthority.STATUTORY_FILING,
        is_company_ir=True,
    )


def test_raw_fact_carries_the_chunk_document_id():
    document = _testco_document()
    chunks = chunk_document(document)
    assert chunks, "synthetic fixture text must actually produce chunks"

    raw_facts = extract_from_chunk(chunks[0], "TESTCO")
    assert raw_facts, "synthetic fixture text must actually match an extraction rule"
    for raw in raw_facts:
        assert raw.document_id == document.doc_id


def test_document_id_survives_through_fact_collector_agent_to_fact():
    document = _testco_document()
    collector = DocumentCollector([document], provenance=Provenance.FIXTURE)
    result = collector.collect("TESTCO", "Generic Biotech Holdings")

    assert result.raw_facts, "extraction must produce at least one raw fact for this test"
    assert all(raw.document_id == document.doc_id for raw in result.raw_facts)

    agent = FactCollectorAgent(results=[result])
    agent_input = AgentInput(
        agent_id="fact_collector",
        run_id="test_run_1",
        ticker="TESTCO",
        company_name="Generic Biotech Holdings",
    )
    output = agent.run(agent_input)

    assert output.facts, "FactCollectorAgent must emit at least one Fact for this test"
    for fact in output.facts:
        assert fact.document_id == document.doc_id, (
            "Fact.document_id must equal the originating Document.doc_id: the "
            "RawFact -> Fact conversion in FactCollectorAgent._to_fact must not "
            "drop the document reference the way raw_payload_ref historically was"
        )


def test_document_id_is_none_when_no_document_was_involved():
    """A RawFact built without going through extract_from_chunk (e.g. a
    structured-collector fact built straight from an API payload) must never
    have a document_id invented for it."""
    from investment_research.schemas.enums import UNKNOWN, FactCategory, Provenance, SourceTier
    from investment_research.schemas.fact import RawFact, Source

    raw = RawFact(
        ticker="TESTCO",
        category=FactCategory.CLINICAL,
        claim="Trial registered as Phase 2, randomized, double-blind.",
        source=Source(
            source_id="src_structured_1",
            url="https://clinicaltrials.gov/study/NCT00000000",
            title="Structured registry record",
            tier=SourceTier.TIER_1,
        ),
        value=UNKNOWN,
        collector="clinicaltrials",
    )
    assert raw.document_id is None

    fact = FactCollectorAgent._to_fact(raw, run_id="r1", provenance=Provenance.LIVE)
    assert fact.document_id is None
