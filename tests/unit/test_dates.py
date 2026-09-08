"""Date integrity tests (requirement 3)."""

from __future__ import annotations

from datetime import date

import pytest

from investment_research.agents.catalyst import horizon_for, to_jst_date
from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.schemas.enums import UNKNOWN, DateKind, Horizon
from investment_research.schemas.fact import parse_date_bounds, parse_iso_date
from investment_research.schemas.validation import SchemaError, validate_date_field
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


# --- partial-date precision (requirement 1G extended) -----------------------
# Never impute a fake day: "2023-12" is not "2023-12-01". A crash on a
# legitimate partial-precision date (SchemaError on "2023-12") was a
# production defect; these tests pin the fix down at every layer.
@pytest.mark.parametrize(
    "value,earliest,latest",
    [
        ("2023", date(2023, 1, 1), date(2023, 12, 31)),
        ("2023-12", date(2023, 12, 1), date(2023, 12, 31)),
        ("2023-12-31", date(2023, 12, 31), date(2023, 12, 31)),
        ("2023-12-31T10:00:00Z", date(2023, 12, 31), date(2023, 12, 31)),
        ("2024-02", date(2024, 2, 1), date(2024, 2, 29)),  # leap year
    ],
)
def test_parse_date_bounds_at_every_supported_precision(value, earliest, latest):
    assert parse_date_bounds(value) == (earliest, latest)


@pytest.mark.parametrize("value", [None, "", UNKNOWN, "soon", "next quarter", "garbage", "2023-13"])
def test_parse_date_bounds_returns_none_not_a_guess(value):
    assert parse_date_bounds(value) is None


def test_parse_date_bounds_never_upgrades_precision():
    """A month-precision value's bounds must not collapse to a single fake day."""
    earliest, latest = parse_date_bounds("2023-12")
    assert earliest != latest
    assert earliest == date(2023, 12, 1)
    assert latest == date(2023, 12, 31)


@pytest.mark.parametrize(
    "value",
    ["2023", "2023-12", "2023-12-31", "2023-12-31T10:00:00Z", UNKNOWN],
)
def test_validate_date_field_accepts_every_supported_precision(value):
    validate_date_field(value, "source.event_date")  # must not raise


@pytest.mark.parametrize("value", ["soon", "garbage", "2023-13", "not-a-date", "2023/13/40"])
def test_validate_date_field_still_rejects_genuinely_invalid_values(value):
    with pytest.raises(SchemaError, match="event_date"):
        validate_date_field(value, "source.event_date")


def test_validate_date_field_error_carries_field_and_value():
    try:
        validate_date_field("garbage", "source.event_date")
    except SchemaError as exc:
        assert exc.field == "source.event_date"
        assert exc.value == "garbage"
    else:
        pytest.fail("expected SchemaError")


# --- staleness with partial precision ---------------------------------------
def test_month_precision_event_date_is_stale_when_latest_possible_day_is_old():
    """A month whose LAST day is already past the cutoff must read as stale."""
    agent = EvidenceIntegrityAgent(today=TODAY, stale_after_days=400)
    # TODAY is 2026-09-06; 400 days back is roughly 2025-08-02. A claim dated
    # only to "2024-01" -- latest possible day 2024-01-31 -- is unambiguously
    # older than the cutoff no matter which day in January it actually was.
    fact = make_fact("The agency issued guidance", event_date="2024-01")
    stale, used = agent._staleness(fact)
    assert stale is True
    assert used == "2024-01"  # original precision preserved, never padded


def test_month_precision_event_date_is_not_marked_fresh_just_because_incomplete():
    """The latest-possible-day rule must not flip to 'assume fresh' either:
    a month whose latest day is still within the cutoff stays not-stale, but
    only because that latest day genuinely is recent -- not because precision
    is incomplete."""
    agent = EvidenceIntegrityAgent(today=TODAY, stale_after_days=400)
    fact = make_fact("The agency issued guidance", event_date="2026-08")
    stale, used = agent._staleness(fact)
    assert stale is False
    assert used == "2026-08"


def test_year_precision_event_date_uses_latest_possible_bound():
    """A bare year far in the past must be stale even though the exact day is
    unknown -- December 31 of that year is still well past the cutoff."""
    agent = EvidenceIntegrityAgent(today=TODAY, stale_after_days=400)
    fact = make_fact("The agency issued guidance", event_date="2020")
    stale, used = agent._staleness(fact)
    assert stale is True
    assert used == "2020"
