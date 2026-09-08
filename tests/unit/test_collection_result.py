"""CollectionResult.degraded semantics (requirement: openFDA zero-result fix).

A collector that executed a valid query and got a definitive, error-free
zero-result answer is not degraded. Any outcome carrying an error, or any
non-OK outcome other than a clean NOT_FOUND, is.
"""

from __future__ import annotations

import pytest

from investment_research.collectors.base import CollectionResult
from investment_research.schemas.enums import FetchOutcome


def test_ok_is_never_degraded():
    assert CollectionResult(collector="x", outcome=FetchOutcome.OK).degraded is False


def test_clean_not_found_is_not_degraded():
    """A valid query that definitively found nothing -- SEARCH_EXECUTED_ZERO_RESULTS."""
    result = CollectionResult(collector="x", outcome=FetchOutcome.NOT_FOUND)
    result.notes.append("no applications listed")
    assert result.errors == []
    assert result.degraded is False


def test_not_found_with_an_error_is_still_degraded():
    """The query itself could not be meaningfully attempted -- not a clean answer."""
    result = CollectionResult(collector="x", outcome=FetchOutcome.NOT_FOUND)
    result.errors.append("no company name available; search skipped")
    assert result.degraded is True


@pytest.mark.parametrize(
    "outcome",
    [
        FetchOutcome.BLOCKED,
        FetchOutcome.RATE_LIMITED,
        FetchOutcome.TIMEOUT,
        FetchOutcome.ERROR,
        FetchOutcome.DISABLED,
    ],
)
def test_real_failures_are_always_degraded(outcome):
    """A real connectivity/permission/server failure must still degrade the run."""
    assert CollectionResult(collector="x", outcome=outcome).degraded is True


def test_ok_with_an_error_is_degraded():
    """Errors always win, regardless of the outcome label."""
    result = CollectionResult(collector="x", outcome=FetchOutcome.OK)
    result.errors.append("partial parse failure")
    assert result.degraded is True
