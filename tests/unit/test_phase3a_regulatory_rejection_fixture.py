"""Phase 3A requirement 8: a synthetic (non-real-ticker) SEC exhibit fixture
whose content, in substance, mirrors the exact failure this system exists to
catch -- a regulator declining to accept the proposed primary endpoint as
sufficient to establish effectiveness, with a previously-anticipated pivotal/
registrational status changing as a result.

Explicitly forbidden and NOT present anywhere in this file: LGVN, RVEF,
Longeveron, "Type C" meeting, HLHS, ELPIS/ELPIS II, real dates, real NCT
numbers. "TESTCO" mirrors the synthetic-company convention already used by
``test_fact_document_id_lineage.py``.

Runs the same real, already-wired path that file uses (chunk_document ->
extract_from_chunk/DocumentCollector -> FactCollectorAgent) end to end, and
confirms every Phase 3A item 8 checkpoint.
"""

from __future__ import annotations

from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.documents import Document, chunk_document
from investment_research.collectors.extraction import DocumentCollector, extract_from_chunk
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import ContentKind, DocumentAuthority, Provenance
from investment_research.schemas.fact import EvidenceClass

_EXHIBIT_TEXT = (
    "TESTCO Holdings, Inc. (the \"Company\") today announced an update "
    "regarding its ongoing engagement with the applicable regulatory "
    "authority for its lead clinical program. "
    "Following a recent meeting, the regulatory authority indicated that it "
    "did not consider the Company's proposed primary endpoint appropriate to "
    "establish the effectiveness of the therapy for the target indication. "
    "As a result, the Company states that the ongoing study can no longer be "
    "characterized as pivotal or registrational until the primary endpoint is "
    "revised in further discussion with the regulatory authority."
)

#: Deliberately distinct from every date field below, and deliberately never
#: reused as one -- see ``test_retrieved_at_is_never_used_as_an_event_date``.
_RETRIEVED_AT = "2026-03-02T00:00:00Z"


def _exhibit_document() -> Document:
    return Document(
        doc_id="doc_testco_ex99_1_regulatory_update",
        url="https://www.sec.gov/Archives/edgar/data/9999999999/000999999925000123/ex99-1.htm",
        title="Exhibit 99.1 -- Corporate Update on Regulatory Discussions and Clinical Program Status",
        publisher="SEC EDGAR",
        # Deliberately UNKNOWN: this fixture does not fabricate a specific
        # meeting date, filing date, or event date (CLAUDE.md rule 7/10 --
        # never fill a gap, never fabricate a date).
        text=_EXHIBIT_TEXT,
        doc_type="exhibit",
        is_company_ir=True,
        content_kind=ContentKind.FULL_DOCUMENT,
        provenance=Provenance.FIXTURE,
        retrieved_at=_RETRIEVED_AT,
        authority=DocumentAuthority.STATUTORY_FILING,
    )


def test_chunking_produces_at_least_one_chunk():
    document = _exhibit_document()
    chunks = chunk_document(document)
    assert chunks, "synthetic exhibit fixture text must actually produce chunks"


def test_the_regulatory_rejection_sentence_is_extracted():
    document = _exhibit_document()
    chunks = chunk_document(document)
    raw_facts = extract_from_chunk(chunks[0], "TESTCO")
    matched_rules = {raw.collector for raw in raw_facts}
    assert "extraction:regulator_endpoint_not_accepted" in matched_rules
    rejection_facts = [
        raw for raw in raw_facts if raw.collector == "extraction:regulator_endpoint_not_accepted"
    ]
    assert any("did not consider" in raw.claim.lower() for raw in rejection_facts)


def test_the_pivotal_status_change_sentence_is_also_extracted():
    document = _exhibit_document()
    chunks = chunk_document(document)
    raw_facts = extract_from_chunk(chunks[0], "TESTCO")
    matched_rules = {raw.collector for raw in raw_facts}
    assert "extraction:no_longer_pivotal" in matched_rules


def test_raw_facts_carry_the_document_id():
    document = _exhibit_document()
    chunks = chunk_document(document)
    raw_facts = extract_from_chunk(chunks[0], "TESTCO")
    assert raw_facts
    assert all(raw.document_id == document.doc_id for raw in raw_facts)


def test_no_search_summary_content_kind_was_used():
    document = _exhibit_document()
    assert document.content_kind is ContentKind.FULL_DOCUMENT
    assert document.content_kind.is_primary_text


def test_retrieved_at_is_never_used_as_an_event_date():
    document = _exhibit_document()
    assert document.retrieved_at == _RETRIEVED_AT
    for field_value in (document.event_date, document.published_date, document.filing_date, document.effective_date):
        assert field_value != _RETRIEVED_AT


def test_authority_is_traceable_via_the_document_store():
    """STATUTORY_FILING is recoverable via the Document this Fact resolves
    to through the DocumentStore -- never a value invented at the Fact
    layer."""
    document = _exhibit_document()
    store = DocumentStore()
    stored = store.put(
        document,
        document_id=document.doc_id,
        accession="0009999999-25-000123",
        filename="ex99-1.htm",
        document_role=DocumentRole.EXHIBIT,
    )
    assert stored.authority is DocumentAuthority.STATUTORY_FILING
    resolved = store.get(document.doc_id)
    assert resolved is not None
    assert resolved.authority is DocumentAuthority.STATUTORY_FILING
    assert resolved.document_role is DocumentRole.EXHIBIT


def test_document_id_survives_through_fact_collector_agent_to_fact():
    document = _exhibit_document()
    collector = DocumentCollector([document], provenance=Provenance.FIXTURE)
    result = collector.collect("TESTCO", "Generic Biotech Holdings")
    assert result.raw_facts

    agent = FactCollectorAgent(results=[result])
    agent_input = AgentInput(
        agent_id="fact_collector",
        run_id="test_phase3a_run",
        ticker="TESTCO",
        company_name="Generic Biotech Holdings",
    )
    output = agent.run(agent_input)
    assert output.facts

    rejection_facts = [f for f in output.facts if "did not consider" in f.claim.lower()]
    assert rejection_facts, "the regulatory-rejection Fact must survive into FactCollectorAgent output"
    for fact in rejection_facts:
        # Fact.document_id lineage (CLAUDE.md invariant, Phase 1).
        assert fact.document_id == document.doc_id
        # Item 8: company_claim=True (the issuer disclosed this in its own
        # filed exhibit) -- but never independent regulator confirmation.
        assert fact.company_claim is True
        assert fact.evidence_class is EvidenceClass.COMPANY_CLAIM
        assert fact.independent_confirmation is False
        # Never a search snippet.
        assert fact.content_kind is ContentKind.FULL_DOCUMENT
        assert fact.content_kind.is_primary_text
