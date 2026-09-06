"""Duplicate detection and append-only versioning (requirements 11, 14)."""

from __future__ import annotations

import pytest

from investment_research.schemas.enums import FactCategory
from investment_research.schemas.fact import make_fact_id, normalize_claim
from tests.conftest import make_fact


def test_fact_id_is_deterministic():
    args = ("TEST", FactCategory.FINANCIAL, "Cash was $31.5 million", "https://x/1", "2026-06-30")
    assert make_fact_id(*args) == make_fact_id(*args)


def test_fact_id_ignores_whitespace_and_case_in_the_claim():
    a = make_fact_id("T", FactCategory.OTHER, "Cash  was   $31.5m", "https://x/1", "2026-06-30")
    b = make_fact_id("T", FactCategory.OTHER, "cash was $31.5m", "https://x/1", "2026-06-30")
    assert a == b


def test_fact_id_differs_by_source():
    a = make_fact_id("T", FactCategory.OTHER, "same claim", "https://x/1", "2026-06-30")
    b = make_fact_id("T", FactCategory.OTHER, "same claim", "https://y/2", "2026-06-30")
    assert a != b


def test_identical_fact_saved_twice_is_one_row(repo):
    fact = make_fact("Cash was $31.5 million")
    assert repo.save_fact(fact) == (fact.fact_id, 1)
    assert repo.save_fact(fact) == (fact.fact_id, 1)
    assert len(repo.fact_versions(fact.fact_id)) == 1


def test_changed_fact_creates_a_new_version_and_supersedes_the_old(repo):
    """Requirement 11/12: history is never overwritten."""
    fact = make_fact("Cash was $31.5 million", value="31500000")
    repo.save_fact(fact)
    revised = fact.next_version(claim="Cash was $28.1 million", value="28100000")
    _, version = repo.save_fact(revised)

    assert version == 2
    versions = repo.fact_versions(fact.fact_id)
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["claim"] == "Cash was $31.5 million"
    assert versions[0]["superseded_by"] == f"{fact.fact_id}#v2"
    assert versions[1]["superseded_by"] is None


def test_latest_facts_returns_only_the_newest_version(repo):
    fact = make_fact("Cash was $31.5 million", value="31500000")
    repo.save_fact(fact)
    repo.save_fact(fact.next_version(claim="Cash was $28.1 million", value="28100000"))
    latest = repo.latest_facts("TEST")
    assert len(latest) == 1
    assert latest[0]["claim"] == "Cash was $28.1 million"


def test_direct_update_of_a_stored_fact_is_refused(repo):
    """A database-level trigger, so a future code path cannot quietly overwrite."""
    fact = make_fact("Cash was $31.5 million")
    repo.save_fact(fact)
    with pytest.raises(Exception, match="append-only"):
        repo.conn.execute(
            "UPDATE facts SET claim = ? WHERE fact_id = ?", ("rewritten", fact.fact_id)
        )


def test_bus_deduplicates_facts_by_id():
    from investment_research.orchestrator.isolation import EvidenceBus

    bus = EvidenceBus()
    fact = make_fact("Cash was $31.5 million")
    bus.add_facts([fact, fact, make_fact("A different claim")])
    assert len(bus.facts) == 2


def test_normalize_claim():
    assert normalize_claim("  Cash   WAS  $31.5m \n") == "cash was $31.5m"
    assert normalize_claim("") == ""
