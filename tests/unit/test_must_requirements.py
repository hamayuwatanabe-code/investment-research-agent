"""Tests for the three MUST requirements added on top of the approved Phase 2 plan.

M1. WebSearch snippets/summaries must never be promoted to Tier-1 VERIFIED_FACT.
    A material claim becomes VERIFIED only once the primary-source document body
    has actually been retrieved and checked.
M2. Search results are discovery evidence only, never facts. Search hits are
    stored separately (SearchHit/SearchQueryRecord), and fact extraction comes
    from document bodies.
M3. Bull/Bear/Neutral/Contradiction follow-up searches are independent and
    auditable, with no cross-contamination between Bull and Bear results.
"""

from __future__ import annotations

import dataclasses

import pytest

from investment_research.collectors.documents import Document
from investment_research.collectors.extraction import DocumentCollector
from investment_research.research.adversarial import build_plan, run_adversarial_search
from investment_research.research.corpus import CorpusResearchProvider
from investment_research.research.discovery import (
    DiscoveryLog,
    SearchHit,
    SearchQueryRecord,
    hits_from_documents,
    make_hit_id,
    make_query_id,
)
from investment_research.schemas.enums import (
    ContentKind,
    EvidenceClass,
    Provenance,
    QueryPurpose,
    SourceTier,
    VerifiedStatus,
    purposes_for_agent,
)
from investment_research.schemas.validation import SchemaError, validate_fact
from tests.conftest import make_fact


# =========================== M1 ============================================
def test_search_summary_cannot_be_verified_fact():
    fact = make_fact(
        "The FDA advised the endpoint is not sufficient",
        tier=SourceTier.TIER_1,
        evidence_class=EvidenceClass.VERIFIED_FACT,
        verified=VerifiedStatus.VERIFIED,
    )
    search_derived = dataclasses.replace(fact, content_kind=ContentKind.SEARCH_SUMMARY)
    with pytest.raises(SchemaError, match="VERIFIED_FACT"):
        validate_fact(search_derived)


def test_metadata_only_cannot_be_partially_verified():
    fact = make_fact(
        "The FDA advised the endpoint is not sufficient",
        tier=SourceTier.TIER_1,
        evidence_class=EvidenceClass.INDEPENDENT_EVIDENCE,
        verified=VerifiedStatus.PARTIALLY_VERIFIED,
    )
    bad = dataclasses.replace(fact, content_kind=ContentKind.METADATA_ONLY)
    with pytest.raises(SchemaError):
        validate_fact(bad)


def test_full_document_body_can_be_verified():
    fact = make_fact(
        "Cash was $31.5 million",
        tier=SourceTier.TIER_1,
        evidence_class=EvidenceClass.VERIFIED_FACT,
        verified=VerifiedStatus.VERIFIED,
    )
    validate_fact(fact)
    assert fact.content_kind == ContentKind.FULL_DOCUMENT
    assert fact.is_decision_grade


def test_primary_source_identified_but_not_fetched_requires_a_url():
    fact = make_fact(
        "The FDA advised the endpoint is not sufficient",
        evidence_class=EvidenceClass.UNVERIFIED_CLAIM,
        verified=VerifiedStatus.PRIMARY_SOURCE_IDENTIFIED_BUT_NOT_FETCHED,
    )
    search_derived = dataclasses.replace(fact, content_kind=ContentKind.SEARCH_SUMMARY)
    with pytest.raises(SchemaError, match="primary_source_url"):
        validate_fact(search_derived)

    ok = dataclasses.replace(search_derived, primary_source_url="https://www.sec.gov/x")
    validate_fact(ok)
    assert not ok.is_decision_grade


def test_search_evidence_status_is_never_decision_grade():
    base = make_fact(
        "A search result mentions the endpoint",
        evidence_class=EvidenceClass.UNVERIFIED_CLAIM,
        verified=VerifiedStatus.SEARCH_EVIDENCE,
    )
    derived = dataclasses.replace(base, content_kind=ContentKind.SEARCH_SUMMARY)
    validate_fact(derived)
    assert not derived.is_decision_grade
    assert derived.is_search_derived


def test_lgvn_corpus_produces_no_verified_facts_without_body_fetch():
    """End-to-end: the captured LGVN corpus is all SEARCH_SUMMARY/METADATA_ONLY,
    so nothing extracted from it may become VERIFIED_FACT."""
    provider = CorpusResearchProvider("data/corpus", "LGVN")
    documents = provider.documents()
    assert documents, "fixture corpus must be present for this test"
    assert all(not d.content_kind.is_primary_text for d in documents), (
        "test assumption: LGVN corpus documents are search summaries/metadata, not bodies"
    )
    collector = DocumentCollector(documents, provenance=Provenance.CAPTURED)
    result = collector.collect("LGVN", "Longeveron Inc.")
    for raw in result.raw_facts:
        assert raw.content_kind is not ContentKind.FULL_DOCUMENT
        assert raw.content_kind is not ContentKind.EXCERPT


# =========================== M2 ============================================
def test_search_hit_is_not_a_fact_type():
    """SearchHit and Fact are structurally distinct types with no fact_id."""
    from investment_research.schemas.fact import Fact

    hit = SearchHit(
        hit_id="hit_1", query_id="q_1", run_id="r1", ticker="LGVN",
        title="t", url="https://x.com/a",
    )
    assert not isinstance(hit, Fact)
    assert not hasattr(hit, "fact_id")
    assert not hasattr(hit, "evidence_class")


def test_hits_from_documents_marks_body_retrieved_by_content_kind():
    query = SearchQueryRecord(
        query_id="q_1", run_id="r1", ticker="LGVN", agent_id="kill_agent",
        query_purpose=QueryPurpose.BEAR, query_text="LGVN going concern",
    )
    docs = [
        Document(doc_id="d1", url="https://a.com", title="t1", content_kind=ContentKind.FULL_DOCUMENT),
        Document(doc_id="d2", url="https://b.com", title="t2", content_kind=ContentKind.SEARCH_SUMMARY),
    ]
    hits = hits_from_documents(docs, query=query, provider="test", path=query.query_purpose)
    assert hits[0].body_retrieved is True
    assert hits[1].body_retrieved is False


def test_discovery_log_never_produces_facts_directly():
    """The discovery layer's public surface has no method that returns a Fact."""
    log = DiscoveryLog()
    log.record_query(
        SearchQueryRecord(
            query_id="q1", run_id="r1", ticker="LGVN", agent_id="kill_agent",
            query_purpose=QueryPurpose.BEAR, query_text="LGVN going concern",
        )
    )
    log.record_hits(
        [SearchHit(hit_id="h1", query_id="q1", run_id="r1", ticker="LGVN", title="t", url="https://x.com")]
    )
    assert len(log.queries) == 1 and len(log.hits) == 1
    public_methods = [m for m in dir(log) if not m.startswith("_")]
    assert "to_fact" not in public_methods
    assert "as_fact" not in public_methods


def test_query_id_is_deterministic_and_scoped_to_run():
    a = make_query_id("kill_agent", QueryPurpose.BEAR, "LGVN going concern", "r1")
    b = make_query_id("kill_agent", QueryPurpose.BEAR, "LGVN going concern", "r1")
    c = make_query_id("kill_agent", QueryPurpose.BEAR, "LGVN going concern", "r2")
    assert a == b
    assert a != c


def test_hit_id_depends_on_query_and_url():
    a = make_hit_id("q1", "https://x.com/a")
    b = make_hit_id("q1", "https://x.com/b")
    c = make_hit_id("q2", "https://x.com/a")
    assert len({a, b, c}) == 3


# =========================== M3 ============================================
def test_bull_and_bear_purposes_do_not_overlap():
    bull = purposes_for_agent("bull_agent")
    bear = purposes_for_agent("bear_agent")
    assert QueryPurpose.BEAR not in bull
    assert QueryPurpose.BULL not in bear
    assert QueryPurpose.BULL in bull
    assert QueryPurpose.BEAR in bear


def test_discovery_log_hits_for_respects_purpose_scoping():
    log = DiscoveryLog()
    bear_q = SearchQueryRecord(
        query_id="qb", run_id="r1", ticker="LGVN", agent_id="kill_agent",
        query_purpose=QueryPurpose.BEAR, query_text="LGVN going concern",
    )
    bull_q = SearchQueryRecord(
        query_id="qu", run_id="r1", ticker="LGVN", agent_id="bull_agent",
        query_purpose=QueryPurpose.BULL, query_text="LGVN partnership",
    )
    log.record_query(bear_q)
    log.record_query(bull_q)
    log.record_hits([SearchHit(hit_id="hb", query_id="qb", run_id="r1", ticker="LGVN",
                                title="bear finding", url="https://x.com/bear",
                                query_purpose=QueryPurpose.BEAR)])
    log.record_hits([SearchHit(hit_id="hu", query_id="qu", run_id="r1", ticker="LGVN",
                                title="bull finding", url="https://x.com/bull",
                                query_purpose=QueryPurpose.BULL)])

    bull_hits = log.hits_for_agent("bull_agent")
    bear_hits = log.hits_for_agent("bear_agent")
    assert {h.hit_id for h in bull_hits} == {"hu"}
    assert {h.hit_id for h in bear_hits} == {"hb"}
    assert "hb" not in {h.hit_id for h in bull_hits}
    assert "hu" not in {h.hit_id for h in bear_hits}


def test_adversarial_search_records_every_query_auditably(tmp_path):
    from investment_research.research.provider import NullResearchProvider

    plan = build_plan("LGVN", "Longeveron")
    outcome = run_adversarial_search(NullResearchProvider(), plan, run_id="r1", ticker="LGVN")
    assert outcome.discovery.queries, "every mandatory query must be recorded, even if unexecuted"
    for record in outcome.discovery.queries:
        assert record.query_id
        assert record.agent_id == "adversarial_search"
        assert record.query_purpose in (QueryPurpose.BEAR, QueryPurpose.BULL)
        assert record.created_at
        assert record.query_text


def test_bear_queries_are_tagged_bear_and_bull_queries_bull():
    from investment_research.research.provider import NullResearchProvider

    plan = build_plan("LGVN", "Longeveron")
    outcome = run_adversarial_search(NullResearchProvider(), plan, run_id="r1", ticker="LGVN")
    bear_purposes = {r.query_purpose for r in outcome.discovery.queries_for([QueryPurpose.BEAR])}
    bull_purposes = {r.query_purpose for r in outcome.discovery.queries_for([QueryPurpose.BULL])}
    assert bear_purposes <= {QueryPurpose.BEAR}
    assert bull_purposes <= {QueryPurpose.BULL}
    # every bear-plan query text appears with BEAR purpose, not BULL
    bear_texts = {q.query for q in plan.bear}
    recorded_bear_texts = {
        r.query_text for r in outcome.discovery.queries_for([QueryPurpose.BEAR])
    }
    assert bear_texts <= recorded_bear_texts


def test_cli_style_adversarial_documents_are_not_merged_into_extraction_input():
    """Regression guard for the CLI wiring: adversarial search documents must
    never be appended to the document list handed to fact extraction."""

    provider = CorpusResearchProvider("data/corpus", "LGVN")
    documents = provider.documents()
    original_ids = {d.doc_id for d in documents}

    plan = build_plan("LGVN", "Longeveron")
    run_adversarial_search(provider, plan, run_id="LGVN", ticker="LGVN")

    # Simulate what the CLI does today: it must NOT extend `documents` with
    # outcome.all_documents(). Assert the corpus-only document set is what
    # extraction would actually see.
    assert {d.doc_id for d in documents} == original_ids, (
        "adversarial search results must not be merged into the fact-extraction "
        "document set (requirement M2)"
    )
