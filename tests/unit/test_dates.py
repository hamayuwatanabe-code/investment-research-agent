"""Date integrity tests (requirement 3)."""

from __future__ import annotations

from datetime import date

import pytest

from investment_research.agents.catalyst import horizon_for, to_jst_date
from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.schemas.enums import UNKNOWN, DateKind, Horizon
from investment_research.schemas.fact import parse_iso_date
from tests.conftest import TODAY, make_fact, make_source


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-01-31", date(2026, 1, 31)),
        ("2026/01/31", date(2026, 1, 31)),
        ("20260131", date(2026, 1, 31)),
        ("2026-01-31T10:00:00Z", date(2026, 1, 31)),
        ("Jan 31, 2026", date(2026, 1, 31)),
    ],
)
def test_dates_parsed(value, expected):
    assert parse_iso_date(value) == expected


@pytest.mark.parametrize("value", [None, "", UNKNOWN, "soon", "next quarter", "garbage"])
def test_unparseable_dates_return_none_not_a_guess(value):
    assert parse_iso_date(value) is None


def test_event_date_preferred_over_publication_date():
    """An article published today about a meeting in May is May evidence."""
    source = make_source(event_date="2026-05-19", published_date="2026-08-07")
    best, kind = source.best_date()
    assert best == "2026-05-19"
    assert kind is DateKind.EVENT_DATE


def test_publication_date_used_only_as_last_resort():
    source = make_source(event_date=UNKNOWN, published_date="2026-08-07")
    best, kind = source.best_date()
    assert best == "2026-08-07"
    assert kind is DateKind.PUBLISHED_DATE


def test_staleness_judged_on_event_date_not_publication_date():
    """An old event republished today is still old."""
    agent = EvidenceIntegrityAgent(today=TODAY, stale_after_days=400)
    old_event_new_article = make_fact(
        "The agency issued guidance", event_date="2024-01-05", publication_date="2026-09-01"
    )
    stale, used = agent._staleness(old_event_new_article)
    assert stale is True
    assert used == "2024-01-05"


def test_recent_event_is_not_stale():
    agent = EvidenceIntegrityAgent(today=TODAY, stale_after_days=400)
    stale, used = agent._staleness(make_fact("Cash was reported", event_date="2026-06-30"))
    assert stale is False
    assert used == "2026-06-30"


@pytest.mark.parametrize(
    "text,expected_date,expected_confidence",
    [
        ("2026-12-15", "2026-12-15", "HIGH"),
        ("Q4 2026", "2026-12-31", "LOW"),
        ("the fourth quarter of 2026", "2026-12-31", "LOW"),
        ("first half of 2027", "2027-06-30", "LOW"),
        ("March 2027", "2027-03-31", "MEDIUM"),
    ],
)
def test_jst_normalisation_keeps_vagueness_visible(text, expected_date, expected_confidence):
    """Guidance is widened to the end of its window and labelled, never sharpened."""
    parsed, confidence = to_jst_date(text)
    assert parsed == expected_date
    assert confidence == expected_confidence


@pytest.mark.parametrize("text", ["soon", "in due course", "", UNKNOWN])
def test_undatable_guidance_is_unknown(text):
    assert to_jst_date(text) == (UNKNOWN, "UNKNOWN")


@pytest.mark.parametrize(
    "target,expected",
    [
        ("2026-09-06", Horizon.T0),
        ("2026-09-15", Horizon.T1),
        ("2026-11-01", Horizon.T2),
        ("2027-06-01", Horizon.T3),
        ("2031-01-01", Horizon.T4),
    ],
)
def test_horizon_bucketing(target, expected):
    assert horizon_for(target, TODAY) is expected
