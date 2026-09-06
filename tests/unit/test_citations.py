"""Hallucinated citation tests (requirement 15/17)."""

from __future__ import annotations

from investment_research.reporting.citations import UNVERIFIED_MARKER, CitationValidator
from tests.conftest import make_fact, make_source


def validator():
    source = make_source("https://www.sec.gov/Archives/real-filing.htm", title="Form 10-Q")
    facts = [
        make_fact(
            "NCT12345678 enrolled 84 participants",
            url="https://www.sec.gov/Archives/real-filing.htm",
        )
    ]
    return CitationValidator([source], facts), facts[0]


def test_known_url_passes():
    validate, _ = validator()
    report = validate.check("See https://www.sec.gov/Archives/real-filing.htm for detail.")
    assert report.clean
    assert report.checked_urls == 1


def test_invented_url_is_flagged_and_marked_not_silently_dropped():
    """A quietly-removed fabrication looks exactly like clean prose."""
    validate, _ = validator()
    text = "According to https://www.sec.gov/Archives/does-not-exist.htm the endpoint was agreed."
    out, report = validate.sanitize(text)
    assert not report.clean
    assert "https://www.sec.gov/Archives/does-not-exist.htm" in report.unknown_urls
    assert UNVERIFIED_MARKER in out
    assert "does-not-exist" in out, "the fabricated reference stays visible"


def test_invented_fact_id_is_flagged():
    validate, _ = validator()
    out, report = validate.sanitize("This rests on fact_" + "0" * 20 + ".")
    assert report.unknown_fact_ids
    assert UNVERIFIED_MARKER in out


def test_real_fact_id_passes():
    validate, fact = validator()
    report = validate.check(f"This rests on {fact.fact_id}.")
    assert not report.unknown_fact_ids


def test_invented_nct_number_is_flagged():
    validate, _ = validator()
    report = validate.check("Results from NCT99999999 support the mechanism.")
    assert report.malformed_nct == ["NCT99999999"]


def test_real_nct_number_passes():
    validate, _ = validator()
    report = validate.check("Results from NCT12345678 support the mechanism.")
    assert not report.malformed_nct


def test_invented_sec_accession_is_flagged():
    source = make_source("https://www.sec.gov/a")
    source = source.__class__(**{**vars(source), "accession": "0001234567-26-000001"})
    validate = CitationValidator([source], [])
    report = validate.check("See accession 0009999999-26-000009.")
    assert report.malformed_accessions == ["0009999999-26-000009"]


def test_fixture_urls_are_recognised_when_retrieved():
    source = make_source("fixture://sec/DEMOBIO/10-Q", title="Form 10-Q")
    validate = CitationValidator([source], [])
    assert validate.check("See fixture://sec/DEMOBIO/10-Q").clean


def test_report_renderer_surfaces_citation_warnings():
    """The renderer must not swallow a failed citation check."""
    from investment_research.reporting.report import ReportRenderer

    class _Stub:
        pass

    result = _Stub()
    result.bus = _Stub()
    result.bus.sources = [make_source("https://www.sec.gov/real")]
    result.bus.facts = []
    renderer = ReportRenderer.__new__(ReportRenderer)
    renderer.result = result
    renderer.bus = result.bus
    renderer.validator = CitationValidator(result.bus.sources, [])
    renderer.citation_issues = []

    renderer._safe("cite https://www.invented.example/doc here")
    assert renderer.citation_issues
