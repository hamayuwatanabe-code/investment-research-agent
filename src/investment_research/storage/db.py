"""SQLite connection management and migration.

``CREATE TABLE IF NOT EXISTS`` alone is not a migration: it creates tables
that do not yet exist, but it does nothing to a table that already exists
under an older shape. A database from before the Decision-Grade Evidence Gate
(no ``confirmation`` column on ``kill_assessments``, no ``evidence_sufficiency``
table) must keep opening and keep its history readable after this module is
upgraded -- schema evolution here is therefore two steps: run the DDL to
create anything missing, then walk a declared list of additive columns and
``ALTER TABLE ... ADD COLUMN`` in any that an existing table is missing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Columns added to a table after its first release. Each entry is applied
#: with ``ALTER TABLE <table> ADD COLUMN <name> <ddl>`` when a table already
#: exists but is missing it -- e.g. a database created before requirement M1
#: (content_kind/primary_source_url) or before the Decision-Grade Evidence
#: Gate (confirmation, evidence_sufficiency's columns, thesis_versions'
#: research_status). New tables need no entry here: ``CREATE TABLE IF NOT
#: EXISTS`` in schema.sql already creates them with every column they need.
ADDITIVE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "facts": (
        ("content_kind", "TEXT NOT NULL DEFAULT 'FULL_DOCUMENT'"),
        ("primary_source_url", "TEXT"),
        # Phase 2: Document identity a fact was extracted from. Nullable, so
        # every existing row (which has no document_id at all) reads back as
        # NULL rather than a guessed or invented value.
        ("document_id", "TEXT"),
    ),
    "kill_assessments": (("confirmation", "TEXT NOT NULL DEFAULT 'PROVISIONAL'"),),
    "thesis_versions": (
        ("research_status", "TEXT NOT NULL DEFAULT 'COMPLETE'"),
        ("blocking_verification_required", "TEXT DEFAULT ''"),
    ),
}


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _apply_additive_columns(conn: sqlite3.Connection) -> None:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, columns in ADDITIVE_COLUMNS.items():
        if table not in tables:
            # A brand-new database: schema.sql's CREATE TABLE already declared
            # every current column, so there is nothing additive to apply.
            continue
        present = _existing_columns(conn, table)
        for name, ddl in columns:
            if name in present:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _apply_additive_columns(conn)
    conn.commit()


def open_db(db_path: str | Path) -> sqlite3.Connection:
    conn = connect(db_path)
    migrate(conn)
    return conn
