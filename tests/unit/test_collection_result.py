"""CollectionResult.degraded semantics: generic and collector-agnostic.

NOT_FOUND (including an HTTP 404) is never globally reinterpreted as a
successful zero-result search. Only a specific collector's own semantic
boundary may make that determination (see FdaCollector), and it does so by
returning outcome=OK with zero_results=True -- not by NOT_FOUND itself
becoming "not degraded" as a general rule.
"""

from __future__ import annotations

import pytest

from investment_research.collectors.base import CollectionResult
from investment_research.schemas.enums import FetchOutcome


def test_ok_is_never_degraded():
    assert CollectionResult(collector="x", outcome=FetchOutcome.OK).degraded is False


def test_not_found_is_degraded_even_with_no_errors():
    """Generic semantics: NOT_FOUND is degraded by default. A collector that
    wants to say "this 404 means a clean zero-result search" must say so
    explicitly by returning outcome=OK (see FdaCollector), not rely on
    NOT_FOUND itself being treated as success."""
    result = CollectionResult(collector="x", outcome=FetchOutcome.NOT_FOUND)
    result.notes.append("no applications listed")
    assert result.errors == []
    assert result.degraded is True


def test_not_found_with_an_error_is_degraded():
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


def test_zero_results_flag_defaults_false_and_does_not_affect_degraded_by_itself():
    """zero_results is purely a structured marker a collector sets alongside
    outcome=OK; setting it without also setting outcome=OK must not, by
    itself, make an otherwise-degraded result look clean."""
    default = CollectionResult(collector="x")
    assert default.zero_results is False

    still_not_found = CollectionResult(collector="x", outcome=FetchOutcome.NOT_FOUND)
    still_not_found.zero_results = True
    assert still_not_found.degraded is True, (
        "zero_results is not a backdoor around outcome -- only outcome=OK is not-degraded"
    )
