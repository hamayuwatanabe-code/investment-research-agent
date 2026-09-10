"""Phase 2: Fact.document_id persists through SQLite round-trip, and (within
one process/run) resolves back to Document.authority through a live
DocumentStore.

Offline only: an in-memory sqlite3 database via storage.db.open_db, no
network. Uses the synthetic "TESTCO" placeholder throughout.
"""

from __future__ import annotations

import dataclasses

from investment_research.collectors.documents import Document
from investment_research.research.document_store import DocumentStore
from investment_research.schemas.enums import DocumentAuthority
from investment_research.schemas.fact import Source
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository
from tests.conftest import make_fact


def test_fact_document_id_round_trips_through_save_and_read():
    conn = open_db(":memory:")
    repo = Repository(conn)
    repo.upsert_company("TESTCO", "Generic Biotech Holdings")

    fact = dataclasses.replace(
        make_fact("Cash and cash equivalents of $42.0 million.", ticker="TESTCO"),
        document_id="doc_testco_10q_2026q2",
    )
    repo.save_source(
        Source(
            source_id=fact.source_id, url=fact.source_url, title=fact.source_title, tier=fact.source_tier
        )
    )
    repo.save_fact(fact)

    row = repo.latest_fact_row(fact.fact_id)
    assert row is not None
    assert row["document_id"] == "doc_testco_10q_2026q2"


def test_fact_with_no_document_id_persists_as_null_not_a_guess():
    conn = open_db(":memory:")
    repo = Repository(conn)
    fact = make_fact("A claim with no document backing.", ticker="TESTCO")
    assert fact.document_id is None
    repo.save_fact(fact)

    row = repo.latest_fact_row(fact.fact_id)
    assert row is not None
    assert row["document_id"] is None


def test_document_id_resolves_to_authority_via_documentstore_within_one_run():
    """The full chain: Fact.document_id (read back from a saved row) ->
    DocumentStore.get() -> Document.authority, when the same DocumentStore
    instance that produced the document is still available."""
    conn = open_db(":memory:")
    repo = Repository(conn)

    store = DocumentStore()
    stored = store.put(
        Document(
            doc_id="doc_testco_fda_letter",
            url="https://www.fda.gov/letters/testco-1",
            title="FDA correspondence",
            text="The agency's correspondence text.",
            authority=DocumentAuthority.REGULATOR,
        ),
        document_id="doc_testco_fda_letter",
    )

    fact = dataclasses.replace(
        make_fact("The agency stated a position on the endpoint.", ticker="TESTCO"),
        document_id=stored.document_id,
    )
    repo.save_fact(fact)

    row = repo.latest_fact_row(fact.fact_id)
    assert row is not None
    recovered = store.get(row["document_id"])
    assert recovered is not None
    assert recovered.authority is DocumentAuthority.REGULATOR


def test_document_id_cannot_recover_authority_without_a_live_documentstore():
    """The known, reported limitation: a bare Fact row (as read back from
    SQLite) has no way to reconstruct Document.authority on its own -- there
    is no DocumentStore table, so this only works within the process/run
    that built the DocumentStore, never from a fresh process holding only
    the database file."""
    conn = open_db(":memory:")
    repo = Repository(conn)
    fact = dataclasses.replace(
        make_fact("Some claim.", ticker="TESTCO"),
        document_id="doc_from_a_run_whose_documentstore_is_gone",
    )
    repo.save_fact(fact)

    row = repo.latest_fact_row(fact.fact_id)
    assert row is not None
    # A fresh, empty DocumentStore -- standing in for "a new process with only
    # the database file, no in-memory store from the original run".
    fresh_store = DocumentStore()
    assert fresh_store.get(row["document_id"]) is None
