"""Database migration safety (requirement DG10).

``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already exists
under an older shape -- it neither adds nor removes a column. A database
created before the Decision-Grade Evidence Gate (no ``confirmation`` on
``kill_assessments``, no ``research_status``/``blocking_verification_required``
on ``thesis_versions``, and -- from the earlier MUST-1 work -- no
``content_kind``/``primary_source_url`` on ``facts``) must keep opening, keep
its history readable, and pick up the new columns with safe defaults rather
than fail or silently drop data. These tests build exactly that kind of
pre-existing database by hand (the real historical DDL, not today's
schema.sql) and then run today's :func:`migrate` over it.
"""

from __future__ import annotations

import sqlite3

import pytest

from investment_research.storage.db import migrate

#: The `facts` table as it existed before requirement M1 added content_kind
#: and primary_source_url -- a deliberately trimmed but representative slice
#: of the pre-Decision-Grade-Gate schema (only the columns relevant to these
#: tests; the historical table had more, but ALTER TABLE ADD COLUMN does not
#: care how many other columns already exist).
_OLD_FACTS_DDL = """
CREATE TABLE facts (
    fact_id             TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    category            TEXT NOT NULL,
    claim               TEXT NOT NULL,
    evidence_class      TEXT NOT NULL,
    source_url          TEXT NOT NULL,
    source_tier         TEXT NOT NULL,
    verified_status     TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    run_id              TEXT NOT NULL,
    PRIMARY KEY (fact_id, version)
);
"""

_OLD_KILL_ASSESSMENTS_DDL = """
CREATE TABLE kill_assessments (
    run_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    category            TEXT NOT NULL,
    level               TEXT NOT NULL,
    rationale           TEXT DEFAULT '',
    evidence_confidence REAL DEFAULT 0.0,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker, category)
);
"""

_OLD_THESIS_VERSIONS_DDL = """
CREATE TABLE thesis_versions (
    ticker              TEXT NOT NULL,
    version             INTEGER NOT NULL,
    run_id              TEXT NOT NULL,
    action              TEXT NOT NULL,
    evidence_confidence REAL NOT NULL,
    headline            TEXT DEFAULT '',
    max_kill_level      TEXT DEFAULT 'K0',
    created_at          TEXT NOT NULL,
    PRIMARY KEY (ticker, version)
);
"""


@pytest.fixture
def pre_gate_db() -> sqlite3.Connection:
    """An in-memory database in the shape it had before this gate existed."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_OLD_FACTS_DDL + _OLD_KILL_ASSESSMENTS_DDL + _OLD_THESIS_VERSIONS_DDL)
    conn.execute(
        """INSERT INTO facts(fact_id, ticker, category, claim, evidence_class, source_url,
                              source_tier, verified_status, version, run_id)
           VALUES('fact_old1', 'LEGACY', 'REGULATORY', 'A pre-existing claim', 'VERIFIED_FACT',
                  'https://www.sec.gov/legacy', 'TIER_1', 'VERIFIED', 1, 'run-legacy')"""
    )
    conn.execute(
        """INSERT INTO kill_assessments(run_id, ticker, category, level, rationale,
                                        evidence_confidence, created_at)
           VALUES('run-legacy', 'LEGACY', 'REGULATORY_KILL', 'K5', 'Pre-existing finding',
                  0.85, '2026-01-01T00:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO thesis_versions(ticker, version, run_id, action, evidence_confidence,
                                       headline, max_kill_level, created_at)
           VALUES('LEGACY', 1, 'run-legacy', 'AVOID', 8.5, 'Old headline', 'K5',
                  '2026-01-01T00:00:00+00:00')"""
    )
    conn.commit()
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# --- the migration adds every new column --------------------------------
def test_migration_adds_content_kind_and_primary_source_url_to_facts(pre_gate_db):
    assert "content_kind" not in _columns(pre_gate_db, "facts")
    migrate(pre_gate_db)
    columns = _columns(pre_gate_db, "facts")
    assert "content_kind" in columns
    assert "primary_source_url" in columns


def test_migration_adds_confirmation_to_kill_assessments(pre_gate_db):
    assert "confirmation" not in _columns(pre_gate_db, "kill_assessments")
    migrate(pre_gate_db)
    assert "confirmation" in _columns(pre_gate_db, "kill_assessments")


def test_migration_adds_research_status_columns_to_thesis_versions(pre_gate_db):
    columns_before = _columns(pre_gate_db, "thesis_versions")
    assert "research_status" not in columns_before
    assert "blocking_verification_required" not in columns_before
    migrate(pre_gate_db)
    columns_after = _columns(pre_gate_db, "thesis_versions")
    assert "research_status" in columns_after
    assert "blocking_verification_required" in columns_after


def test_migration_creates_evidence_sufficiency_table(pre_gate_db):
    tables_before = {
        row[0] for row in pre_gate_db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "evidence_sufficiency" not in tables_before
    migrate(pre_gate_db)
    tables_after = {
        row[0] for row in pre_gate_db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "evidence_sufficiency" in tables_after


# --- existing history survives, with safe defaults ------------------------
def test_existing_fact_history_survives_migration_with_a_safe_default(pre_gate_db):
    migrate(pre_gate_db)
    row = pre_gate_db.execute("SELECT * FROM facts WHERE fact_id = 'fact_old1'").fetchone()
    assert row is not None
    assert row["claim"] == "A pre-existing claim"
    assert row["source_url"] == "https://www.sec.gov/legacy"
    # A fact recorded before content_kind existed predates the very
    # distinction the column encodes; FULL_DOCUMENT is the pre-M1 assumption
    # (everything was read from the body) and it is not silently upgraded to
    # VERIFIED/decision-grade by the migration itself -- only the migrated
    # value is asserted here, not a claim about the fact's evidentiary status.
    assert row["content_kind"] == "FULL_DOCUMENT"
    assert row["primary_source_url"] is None


def test_existing_kill_assessment_survives_migration_as_provisional_by_default(pre_gate_db):
    migrate(pre_gate_db)
    row = pre_gate_db.execute(
        "SELECT * FROM kill_assessments WHERE run_id = 'run-legacy'"
    ).fetchone()
    assert row is not None
    assert row["level"] == "K5"
    assert row["rationale"] == "Pre-existing finding"
    # Requirement DG2: a pre-gate row carries no evidence about whether it was
    # decision-grade-backed, so the safe default is PROVISIONAL, never
    # CONFIRMED-by-migration -- confirmation is earned, not assumed.
    assert row["confirmation"] == "PROVISIONAL"


def test_existing_thesis_version_survives_migration_as_complete_by_default(pre_gate_db):
    migrate(pre_gate_db)
    row = pre_gate_db.execute(
        "SELECT * FROM thesis_versions WHERE ticker = 'LEGACY' AND version = 1"
    ).fetchone()
    assert row is not None
    assert row["action"] == "AVOID"
    assert row["headline"] == "Old headline"
    # A run recorded before this gate existed is not retroactively re-judged;
    # its stored research_status defaults to COMPLETE so old history renders
    # the same as it always did, and only *new* runs are subject to the gate.
    assert row["research_status"] == "COMPLETE"
    assert row["blocking_verification_required"] == ""


def test_migration_is_idempotent(pre_gate_db):
    """Running migrate() twice must not fail or duplicate columns."""
    migrate(pre_gate_db)
    migrate(pre_gate_db)  # must not raise "duplicate column name"
    row = pre_gate_db.execute("SELECT * FROM facts WHERE fact_id = 'fact_old1'").fetchone()
    assert row["content_kind"] == "FULL_DOCUMENT"


def test_fresh_database_needs_no_additive_migration():
    """A brand-new database gets every current column straight from schema.sql."""
    from investment_research.storage.db import open_db

    conn = open_db(":memory:")
    for table, column in (
        ("facts", "content_kind"),
        ("facts", "primary_source_url"),
        ("facts", "document_id"),
        ("kill_assessments", "confirmation"),
        ("thesis_versions", "research_status"),
        ("thesis_versions", "blocking_verification_required"),
    ):
        assert column in _columns(conn, table), f"{table}.{column} missing on a fresh database"


# --- Phase 2: document_id additive migration -------------------------------
def test_migration_adds_document_id_to_facts(pre_gate_db):
    assert "document_id" not in _columns(pre_gate_db, "facts")
    migrate(pre_gate_db)
    assert "document_id" in _columns(pre_gate_db, "facts")


def test_existing_fact_history_survives_migration_with_document_id_null(pre_gate_db):
    """A fact recorded before document_id existed carries no document
    reference at all; the migration must never invent one -- NULL, not a
    guessed or empty-string placeholder."""
    migrate(pre_gate_db)
    row = pre_gate_db.execute("SELECT * FROM facts WHERE fact_id = 'fact_old1'").fetchone()
    assert row is not None
    assert row["claim"] == "A pre-existing claim"
    assert row["document_id"] is None
