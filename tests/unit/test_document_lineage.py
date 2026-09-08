"""Document authority and date lineage through web fetch (requirement B).

The defect: AnthropicWebResearchProvider.fetch() used Anthropic's own
"retrieved_at" marker (when THEIR server fetched the page) as the fetched
Document's published_date, and hardcoded doc_type="web"/is_company_ir=False
regardless of the document's real nature. Escalation then built facts
straight from that: a historical filing fetched today could end up dated
today and marked independently confirmed, and a company press release could
become decision-grade merely because its body was fetched.

Four things are tested here:
  1. retrieved_at is retrieval time only -- never copied into any date field.
  2. known Source/SearchHit metadata is carried forward onto the fetched
     Document wherever it is actually known.
  3. DocumentAuthority is derived from the URL (or an explicit is_company_ir
     override), never invented, and drives evidence_class/company_claim/
     independent_confirmation for any fact built from the fetched body.
  4. A statutory issuer filing is distinguishable from genuine regulator
     confirmation, and a press release never becomes decision-grade merely
     because its body was fetched.
"""

from __future__ import annotations

import types

from investment_research.collectors.documents import Document
from investment_research.research.anthropic_web import AnthropicWebResearchProvider
from investment_research.research.escalation import (
    _evidence_for_authority,
    _fact_from_answered_question,
)
from investment_research.schemas.enums import (
    UNKNOWN,
    DocumentAuthority,
    EvidenceClass,
    FactCategory,
    SourceTier,
)
from investment_research.schemas.fact import Source, UnresolvedQuestion, make_source_id

TODAY_RETRIEVED_AT = "2026-09-08T00:00:00+00:00"

_QUESTION = UnresolvedQuestion(
    question="Does the regulator consider the primary endpoint adequate to establish "
    "effectiveness for the intended indication?",
    why_it_matters="A rejected endpoint invalidates the registrational path.",
    blocking=True,
    category=FactCategory.REGULATORY,
)


class _FakeLLM:
    """Minimal LLMClient double: records the request, returns a canned reply."""

    def __init__(self, model: str, response):
        self.model = model
        self.last_usage_tokens = 0
        self._response = response

    def available(self):
        return True, "ready"

    def raw_message(self, *, system, messages, tools=None, max_tokens=16000, **_kw):
        return self._response


def _fetch_response(text: str, title: str, retrieved_at: str = TODAY_RETRIEVED_AT):
    fetch_block = types.SimpleNamespace(
        type="web_fetch_tool_result",
        content={
            "content": {"source": {"data": text}, "title": title},
            "retrieved_at": retrieved_at,
        },
    )
    return types.SimpleNamespace(content=[fetch_block])


# --- Case 1: historical statutory filing --------------------------------
def test_case1_historical_statutory_filing_keeps_its_may_dates_not_september():
    known_source = Source(
        source_id=make_source_id("https://www.sec.gov/Archives/x/8-K.htm", "Form 8-K"),
        url="https://www.sec.gov/Archives/x/8-K.htm",
        title="Form 8-K",
        tier=SourceTier.TIER_1,
        publisher="SEC",
        published_date="2026-05-08",
        event_date="2026-05-08",
        filing_date="2026-05-08",
        accession="0001234567-26-000123",
    )
    llm = _FakeLLM(
        "claude-sonnet-5",
        _fetch_response(
            "The agency advised the company that it does not agree the primary endpoint is "
            "adequate, and that the endpoint is not sufficient to establish effectiveness for "
            "the intended indication.",
            "Form 8-K",
        ),
    )
    provider = AnthropicWebResearchProvider(llm)

    document = provider.fetch(known_source.url, known=known_source)

    assert document is not None
    # Retained May metadata -- never invented, never overwritten.
    assert document.published_date == "2026-05-08"
    assert document.event_date == "2026-05-08"
    assert document.filing_date == "2026-05-08"
    assert document.accession == "0001234567-26-000123"
    # retrieved_at is real (September) and kept SEPARATE from those dates.
    assert document.retrieved_at.startswith("2026-09")
    assert document.published_date != document.retrieved_at
    assert document.event_date != document.retrieved_at
    assert document.authority is DocumentAuthority.STATUTORY_FILING

    fact = _fact_from_answered_question(
        document.text, document, _QUESTION, ticker="TESTCO", run_id="r1"
    )
    # Statutory filing disclosure is distinguishable from regulator
    # confirmation: verified that the ISSUER disclosed this, not that a
    # regulator independently confirmed it.
    assert fact.company_claim is True
    assert fact.independent_confirmation is False
    assert fact.evidence_class is EvidenceClass.VERIFIED_FACT
    assert fact.event_date == "2026-05-08"
    assert fact.publication_date == "2026-05-08"
    assert fact.event_date != TODAY_RETRIEVED_AT[:10]


# --- Case 2: genuine regulator-issued document ---------------------------
def test_case2_regulator_issued_document_keeps_its_own_date_and_is_independent():
    known_source = Source(
        source_id=make_source_id("https://www.fda.gov/letters/x", "FDA correspondence"),
        url="https://www.fda.gov/letters/x",
        title="FDA correspondence",
        tier=SourceTier.TIER_1,
        publisher="FDA",
        published_date="2026-04-15",
        event_date="2026-04-15",
    )
    llm = _FakeLLM(
        "claude-sonnet-5",
        _fetch_response(
            "The agency stated it does not consider the primary endpoint adequate to "
            "establish effectiveness for the intended indication.",
            "FDA correspondence",
        ),
    )
    provider = AnthropicWebResearchProvider(llm)

    document = provider.fetch(known_source.url, known=known_source)

    assert document is not None
    assert document.published_date == "2026-04-15"
    assert document.event_date == "2026-04-15"
    assert document.retrieved_at.startswith("2026-09")
    assert document.authority is DocumentAuthority.REGULATOR

    fact = _fact_from_answered_question(
        document.text, document, _QUESTION, ticker="TESTCO", run_id="r1"
    )
    assert fact.independent_confirmation is True
    assert fact.evidence_class is EvidenceClass.INDEPENDENT_EVIDENCE
    assert fact.event_date == "2026-04-15"


# --- Case 3: ordinary company press release -------------------------------
def test_case3_company_press_release_never_becomes_decision_grade_from_fetch():
    known_source = Source(
        source_id=make_source_id("https://www.globenewswire.com/news/x", "Press release"),
        url="https://www.globenewswire.com/news/x",
        title="Press release",
        tier=SourceTier.TIER_2,
        publisher="GlobeNewswire",
        published_date="2026-08-01",
    )
    llm = _FakeLLM(
        "claude-sonnet-5",
        _fetch_response(
            "The company said its recent regulatory meeting was constructive and it remains "
            "aligned with the agency on the path forward.",
            "Press release",
        ),
    )
    provider = AnthropicWebResearchProvider(llm)

    document = provider.fetch(known_source.url, known=known_source)

    assert document is not None
    assert document.authority is DocumentAuthority.COMPANY_IR
    assert document.is_company_ir is True

    fact = _fact_from_answered_question(
        document.text, document, _QUESTION, ticker="TESTCO", run_id="r1"
    )
    assert fact.evidence_class is EvidenceClass.COMPANY_CLAIM
    assert fact.independent_confirmation is False
    assert not fact.is_decision_grade, (
        "a company press release must never become decision-grade merely because its body "
        "was fetched"
    )


# --- Case 4: SearchHit with no real date ----------------------------------
def test_case4_search_hit_with_no_known_date_stays_unknown_not_retrieval_date():
    hit = Document(
        doc_id="hit1",
        url="https://www.sec.gov/Archives/x/8-K-2.htm",
        title="Form 8-K",
        tier=SourceTier.TIER_1,
        # No published_date/event_date/filing_date known from the hit at all.
    )
    llm = _FakeLLM(
        "claude-sonnet-5", _fetch_response("Some filing text with no known date.", "Form 8-K")
    )
    provider = AnthropicWebResearchProvider(llm)

    document = provider.fetch(hit.url, known=hit)

    assert document is not None
    assert document.published_date == UNKNOWN
    assert document.event_date == UNKNOWN
    assert document.filing_date == UNKNOWN
    assert document.published_date != document.retrieved_at
    assert document.retrieved_at.startswith("2026-09")


# --- fetch() with no known seed at all -------------------------------------
def test_fetch_with_no_known_seed_leaves_dates_unknown_never_retrieval_time():
    llm = _FakeLLM(
        "claude-sonnet-5", _fetch_response("Cold fetch, no prior pointer.", "Untitled")
    )
    provider = AnthropicWebResearchProvider(llm)

    document = provider.fetch("https://www.reuters.com/business/x")

    assert document is not None
    assert document.published_date == UNKNOWN
    assert document.event_date == UNKNOWN
    assert document.filing_date == UNKNOWN
    assert document.effective_date == UNKNOWN
    assert document.accession == UNKNOWN
    assert document.authority is DocumentAuthority.INDEPENDENT


# --- _evidence_for_authority mapping table ----------------------------------
def test_evidence_for_authority_regulator_is_independent_confirmation():
    evidence_class, company_claim, independent = _evidence_for_authority(
        DocumentAuthority.REGULATOR
    )
    assert evidence_class is EvidenceClass.INDEPENDENT_EVIDENCE
    assert company_claim is False
    assert independent is True


def test_evidence_for_authority_statutory_filing_is_verified_but_not_independent():
    evidence_class, company_claim, independent = _evidence_for_authority(
        DocumentAuthority.STATUTORY_FILING
    )
    assert evidence_class is EvidenceClass.VERIFIED_FACT
    assert company_claim is True
    assert independent is False


def test_evidence_for_authority_company_ir_is_company_claim_never_decision_grade():
    evidence_class, company_claim, independent = _evidence_for_authority(
        DocumentAuthority.COMPANY_IR
    )
    assert evidence_class is EvidenceClass.COMPANY_CLAIM
    assert company_claim is True
    assert independent is False
    assert evidence_class not in (EvidenceClass.VERIFIED_FACT, EvidenceClass.INDEPENDENT_EVIDENCE)


def test_evidence_for_authority_unknown_is_conservative():
    evidence_class, company_claim, independent = _evidence_for_authority(
        DocumentAuthority.UNKNOWN
    )
    assert evidence_class is EvidenceClass.UNVERIFIED_CLAIM
    assert independent is False


# --- Document field defaults -------------------------------------------------
def test_document_new_fields_default_safely():
    document = Document(doc_id="d1", url="https://example.com/x", title="t")
    assert document.effective_date == UNKNOWN
    assert document.accession == UNKNOWN
    assert document.authority is DocumentAuthority.UNKNOWN
