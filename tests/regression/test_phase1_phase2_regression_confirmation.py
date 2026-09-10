"""Phase 2.5 requirement 9: explicit re-confirmation that Phase 1/2 guarantees
still hold after this turn's changes. Each assertion here also has its own
dedicated test elsewhere in the suite; this file exists as one place that
states the full checklist and fails loudly if any of it regresses.

Synthetic ("TESTCO") fixtures only. No network.
"""

from __future__ import annotations

import dataclasses

from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.documents import Document, chunk_document
from investment_research.collectors.extraction import DocumentCollector, extract_from_chunk
from investment_research.orchestrator.anonymize import anonymize_facts
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.research.legacy_catalog import (
    LEGACY_CATALOG,
    collapsed_count,
    group_legacy_needs,
)
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    ContentKind,
    DocumentAuthority,
    Provenance,
)
from investment_research.schemas.fact import Source
from investment_research.scoring.kill_gate import KILL_RULES
from investment_research.storage.db import migrate, open_db
from investment_research.storage.repository import Repository
from tests.conftest import make_fact


# --- RawFact/Fact.document_id lineage ---------------------------------------
def test_document_id_lineage_survives_chunk_extraction_and_fact_collection():
    document = Document(
        doc_id="doc_testco_lineage_check",
        url="https://www.sec.gov/Archives/testco/8k.htm",
        title="TESTCO 8-K",
        text="TESTCO disclosed a going concern doubt in its most recent quarterly filing.",
        content_kind=ContentKind.FULL_DOCUMENT,
        authority=DocumentAuthority.STATUTORY_FILING,
        is_company_ir=True,
    )
    chunks = chunk_document(document)
    raw_facts = extract_from_chunk(chunks[0], "TESTCO")
    assert raw_facts and all(r.document_id == document.doc_id for r in raw_facts)

    collector = DocumentCollector([document], provenance=Provenance.FIXTURE)
    result = collector.collect("TESTCO", "Generic Biotech Holdings")
    agent = FactCollectorAgent(results=[result])
    output = agent.run(
        AgentInput(agent_id="fact_collector", run_id="r1", ticker="TESTCO", company_name="Generic Biotech Holdings")
    )
    assert output.facts and all(f.document_id == document.doc_id for f in output.facts)


# --- SQLite additive migration -----------------------------------------------
def test_facts_document_id_column_is_additive_and_nondestructive():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id TEXT NOT NULL, ticker TEXT NOT NULL, category TEXT NOT NULL,
            claim TEXT NOT NULL, evidence_class TEXT NOT NULL, source_url TEXT NOT NULL,
            source_tier TEXT NOT NULL, verified_status TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1, run_id TEXT NOT NULL,
            PRIMARY KEY (fact_id, version)
        );
        """
    )
    conn.execute(
        """INSERT INTO facts(fact_id, ticker, category, claim, evidence_class, source_url,
                              source_tier, verified_status, version, run_id)
           VALUES('fact_pre_existing', 'TESTCO', 'REGULATORY', 'pre-existing claim',
                  'VERIFIED_FACT', 'https://www.sec.gov/x', 'TIER_1', 'VERIFIED', 1, 'run-old')"""
    )
    conn.commit()
    migrate(conn)
    row = conn.execute("SELECT * FROM facts WHERE fact_id = 'fact_pre_existing'").fetchone()
    assert row is not None
    assert row["claim"] == "pre-existing claim"
    assert row["document_id"] is None


def test_fact_document_id_round_trips_through_repository():
    conn = open_db(":memory:")
    repo = Repository(conn)
    fact = dataclasses.replace(
        make_fact("Cash position stated in the filing.", ticker="TESTCO"),
        document_id="doc_testco_roundtrip",
    )
    repo.save_source(Source(source_id=fact.source_id, url=fact.source_url, title=fact.source_title, tier=fact.source_tier))
    repo.save_fact(fact)
    row = repo.latest_fact_row(fact.fact_id)
    assert row is not None
    assert row["document_id"] == "doc_testco_roundtrip"


# --- 49 / 31 / 18, and the COMPANY/PROGRAM separation -----------------------
def test_catalog_counts_are_49_31_18():
    assert len(LEGACY_CATALOG) == 49
    groups = group_legacy_needs()
    assert len(groups) == 31
    assert collapsed_count() == 18


def test_company_and_program_scopes_are_kept_separate():
    scopes = {need.subject_scope for need in LEGACY_CATALOG}
    assert scopes == {SubjectScope.COMPANY, SubjectScope.PROGRAM}
    program_needs = [n for n in LEGACY_CATALOG if n.subject_scope is SubjectScope.PROGRAM]
    assert len(program_needs) == 4  # bear_2, bear_15, bear_16, bull_4


# --- untouched aggregate gates ------------------------------------------------
def test_kill_rules_and_required_domains_still_19_and_6():
    assert len(KILL_RULES) == 19
    assert len(REQUIRED_RESEARCH_DOMAINS) == 6


# --- DocumentStore multi-index / versioning ----------------------------------
def test_document_store_multi_index_and_versioning_still_hold():
    store = DocumentStore()
    v1 = store.put(
        Document(
            doc_id="d1", url="https://ir.testco.example/press/x", text="v1 text", title="p",
            authority=DocumentAuthority.COMPANY_IR,
        ),
        document_id="d1",
        accession="0001-26-000555",
        filename="primary.htm",
        document_role=DocumentRole.PRIMARY_DOCUMENT,
        canonical_url="https://ir.testco.example/press/x",
    )
    v2 = store.put(
        Document(
            doc_id="d2", url="https://ir.testco.example/press/x", text="v2 text (revised)", title="p",
            authority=DocumentAuthority.COMPANY_IR,
        ),
        document_id="d2",
        canonical_url="https://ir.testco.example/press/x",
    )
    assert v2.previous_version_id == v1.document_id
    assert store.get(v1.document_id).document.text == "v1 text"
    assert isinstance(store.resolve_by_accession("0001-26-000555"), list)


# --- Blind Judge never receives document_id ----------------------------------
def test_document_id_never_leaks_into_the_blind_judge_pack():
    fact = dataclasses.replace(
        make_fact("TESTCO disclosed a filing detail.", ticker="TESTCO"),
        document_id="testco_typec_pr_20260508",  # ticker-embedding doc_id style
    )
    anonymized, _ref_map = anonymize_facts((fact,), ticker="TESTCO", company_name="Generic Biotech Holdings")
    assert anonymized[0].document_id is None
