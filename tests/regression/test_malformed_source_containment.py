"""One malformed source must not destroy the entire research run.

Production failure this guards against: a single Source with an unparseable
``event_date`` raised an uncaught SchemaError inside the bulk
``Repository.save_sources()`` call, crashing the whole pipeline run before a
report could ever be produced. Schema integrity must stay strict -- the bad
source is still rejected, never silently persisted -- but persistence of one
malformed record must not abort research on every other, valid record.
"""

from __future__ import annotations

import pytest

from investment_research.collectors.base import CollectionResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.reporting.report import render_report
from investment_research.schemas.enums import Action, FactCategory, RunStatus, SourceTier
from investment_research.schemas.fact import RawFact, Source, make_source_id
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.regression


def _repo() -> Repository:
    return Repository(open_db(":memory:"))


def test_one_malformed_source_cannot_crash_the_run_or_produce_a_confident_action():
    good_source = Source(
        source_id=make_source_id("https://www.sec.gov/good-filing", "Annual report"),
        url="https://www.sec.gov/good-filing",
        title="Annual report",
        tier=SourceTier.TIER_1,
        event_date="2026-06-30",
        published_date="2026-08-07",
    )
    good_raw = RawFact(
        ticker="GENCO",
        category=FactCategory.CAPITAL_STRUCTURE,
        claim="Cash and cash equivalents were reported at the end of the period",
        source=good_source,
        value="reported",
        unit="text",
        company_claim=True,
        collector="unit_test",
    )

    # The malformed record: an unparseable event_date on an otherwise
    # well-formed Tier 1 source, backing the only regulatory evidence in
    # this run.
    bad_source = Source(
        source_id=make_source_id("https://www.fda.gov/bad-correspondence", "Correspondence"),
        url="https://www.fda.gov/bad-correspondence",
        title="Correspondence",
        tier=SourceTier.TIER_1,
        event_date="not-a-real-date",
        published_date="2026-08-07",
    )
    bad_raw = RawFact(
        ticker="GENCO",
        category=FactCategory.REGULATORY,
        claim="The regulator addressed the primary endpoint in correspondence",
        source=bad_source,
        value="addressed",
        unit="text",
        company_claim=False,
        collector="unit_test",
    )

    collection = CollectionResult(
        collector="unit_test", raw_facts=[good_raw, bad_raw], sources=[good_source, bad_source]
    )
    pipeline = Pipeline(_repo(), NullSearchProvider())

    # Must not raise.
    result = pipeline.run("GENCO", "Generic Biotech Holdings", [collection], price=5.0)

    # Still generates a report.
    report = render_report(result)
    assert report

    # The malformed source is visible, not silently dropped.
    assert len(result.quarantined_sources) == 1
    quarantined = result.quarantined_sources[0]
    assert quarantined.source_id == bad_source.source_id
    assert quarantined.url == bad_source.url
    assert "event_date" in quarantined.field
    assert quarantined.value == "not-a-real-date"
    assert quarantined.error

    # The run is marked incomplete, and the reason is visible in failures.
    assert result.context.status == RunStatus.INCOMPLETE_RESEARCH
    assert any("quarantined malformed source" in f for f in result.failures)
    assert any(bad_source.source_id in f for f in result.failures)

    # The good source's fact was still processed -- other valid sources
    # continue where safe.
    assert any(f.source_id == good_source.source_id for f in result.bus.facts)
    assert not any(f.source_id == bad_source.source_id for f in result.bus.facts)

    # The malformed source backed the only regulatory evidence: no confident
    # Action may be issued over an unexamined/corrupted required domain.
    assert result.verdict is not None
    assert result.verdict.action is None
    assert result.verdict.action is not Action.AVOID
    assert result.verdict.action is not Action.BUY

    # The quarantine is also visible in the machine-readable diagnostics.
    from investment_research.cli import result_to_json

    payload = result_to_json(result)
    assert payload["quarantined_sources"] == [
        {
            "source_id": bad_source.source_id,
            "url": bad_source.url,
            "field": "source.event_date",
            "value": "not-a-real-date",
        }
    ]
