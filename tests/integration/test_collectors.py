"""Collector integration tests against a local server serving API-shaped payloads.

The payloads mirror the real response shapes of the SEC submissions API and the
ClinicalTrials.gov v2 API, so the parsing code is exercised for real without
depending on network access to those services.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.collectors import clinicaltrials, fda, sec_edgar
from investment_research.collectors.clinicaltrials import ClinicalTrialsCollector, parse_study
from investment_research.collectors.fda import FdaCollector
from investment_research.collectors.http import HttpClient
from investment_research.collectors.sec_edgar import SecEdgarCollector, extract_xbrl_metric
from investment_research.schemas.enums import UNKNOWN, FactCategory, FetchOutcome, SourceTier
from investment_research.scoring.program_evidence import (
    EvidenceValidationContext,
    EvidenceValidationOutcome,
    validate_company_identity_evidence,
    validate_program_candidate_evidence,
)

DRUGSFDA_WITH_RESULTS = {
    "results": [
        {
            "application_number": "NDA123456",
            "products": [{"brand_name": "Examplinib", "marketing_status": "Prescription"}],
        }
    ]
}

STUDIES_EMPTY = {"studies": []}

pytestmark = pytest.mark.integration

TICKER_MAP = {
    "0": {"cik_str": 1595097, "ticker": "TESTCO", "title": "Test Company Holdings Inc"},
    "1": {"cik_str": 320193, "ticker": "OTHER", "title": "Other Inc"},
}

SUBMISSIONS = {
    "cik": "1595097",
    "name": "Test Company Holdings Inc",
    "filings": {
        "recent": {
            "form": ["10-Q", "8-K", "424B5", "4", "S-3", "SC 13G"],
            "filingDate": [
                "2026-08-07",
                "2026-07-02",
                "2026-03-14",
                "2026-06-30",
                "2026-02-01",
                "2026-01-15",
            ],
            "reportDate": ["2026-06-30", "2026-06-29", "", "2026-06-24", "", ""],
            "accessionNumber": [
                "0001595097-26-000021",
                "0001595097-26-000018",
                "0001595097-26-000009",
                "0001595097-26-000015",
                "0001595097-26-000004",
                "0001595097-26-000002",
            ],
            "primaryDocument": [
                "q2.htm",
                "8k.htm",
                "424b5.htm",
                "form4.xml",
                "s3.htm",
                "sc13g.htm",
            ],
            "primaryDocDescription": ["10-Q", "8-K", "424B5", "FORM 4", "S-3", "SC 13G"],
            "items": ["", "3.01", "", "", "", ""],
        }
    },
}

STUDIES = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT01234567", "briefTitle": "A Phase 2b Study"},
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "primaryCompletionDateStruct": {"date": "2026-11-30", "type": "ESTIMATED"},
                    "lastUpdatePostDateStruct": {"date": "2026-04-02"},
                    "startDateStruct": {"date": "2025-06-01"},
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Test Company Holdings Inc"}
                },
                "designModule": {
                    "phases": ["PHASE2"],
                    "enrollmentInfo": {"count": 84, "type": "ACTUAL"},
                    "designInfo": {
                        "allocation": "RANDOMIZED",
                        "primaryPurpose": "TREATMENT",
                        "interventionModel": "PARALLEL",
                        "maskingInfo": {"masking": "DOUBLE"},
                    },
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {
                            "measure": "Change from baseline in a biomarker composite",
                            "timeFrame": "24 weeks",
                        }
                    ],
                    "secondaryOutcomes": [{"measure": "Overall survival"}],
                },
                "armsInterventionsModule": {
                    "armGroups": [{"label": "Active"}, {"label": "Placebo"}]
                },
            }
        }
    ]
}

COMPANY_FACTS = {
    "facts": {
        "us-gaap": {
            "CashAndCashEquivalentsAtCarryingValue": {
                "units": {
                    "USD": [
                        {"end": "2026-03-31", "val": 42_000_000},
                        {"end": "2026-06-30", "val": 31_500_000},
                    ]
                }
            }
        }
    }
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        routes = {
            "/files/company_tickers.json": TICKER_MAP,
            "/submissions/CIK0001595097.json": SUBMISSIONS,
            "/api/v2/studies": STUDIES,
            "/api/xbrl/companyfacts/CIK0001595097.json": COMPANY_FACTS,
            "/drug/drugsfda_with_results.json": DRUGSFDA_WITH_RESULTS,
            "/api/v2/studies_empty": STUDIES_EMPTY,
        }
        if path in routes:
            body = json.dumps(routes[path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture(scope="module")
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def patched_endpoints(base_url, monkeypatch):
    monkeypatch.setattr(sec_edgar, "TICKER_MAP_URL", f"{base_url}/files/company_tickers.json")
    monkeypatch.setattr(sec_edgar, "SUBMISSIONS_URL", base_url + "/submissions/CIK{cik:010d}.json")
    monkeypatch.setattr(
        sec_edgar, "COMPANY_FACTS_URL", base_url + "/api/xbrl/companyfacts/CIK{cik:010d}.json"
    )
    monkeypatch.setattr(clinicaltrials, "STUDIES_URL", f"{base_url}/api/v2/studies")
    return base_url


@pytest.fixture
def client(tmp_path):
    return HttpClient(timeout=2.0, max_retries=2, rate_limit_rps=0, cache_dir=tmp_path / "c")


# --- SEC --------------------------------------------------------------------
def test_cik_resolution(patched_endpoints, client):
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("TESTCO")
    assert cik == 1595097
    assert name == "Test Company Holdings Inc"
    assert outcome is FetchOutcome.OK


def test_unknown_ticker_is_not_found_not_a_guess(patched_endpoints, client):
    cik, _, outcome = SecEdgarCollector(client).resolve_cik("NOSUCH")
    assert cik is None
    assert outcome is FetchOutcome.NOT_FOUND


def test_filings_become_tier1_facts_with_correct_categories(patched_endpoints, client):
    result = SecEdgarCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert result.outcome is FetchOutcome.OK
    categories = {f.category for f in result.raw_facts}
    assert FactCategory.FINANCIAL in categories  # 10-Q
    assert FactCategory.CAPITAL_STRUCTURE in categories  # 424B5, S-3
    assert FactCategory.INSIDER in categories  # Form 4
    assert FactCategory.GOVERNANCE in categories  # SC 13G
    assert all(s.tier is SourceTier.TIER_1 for s in result.sources)
    assert all(f.company_claim for f in result.raw_facts), "a filer authored its own filing"


def test_filing_dates_and_report_dates_are_kept_apart(patched_endpoints, client):
    result = SecEdgarCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    quarterly = next(s for s in result.sources if s.title.startswith("10-Q"))
    assert quarterly.filing_date == "2026-08-07"
    assert quarterly.event_date == "2026-06-30"
    assert quarterly.accession == "0001595097-26-000021"


def test_collector_failure_is_reported_not_swallowed(client, monkeypatch):
    monkeypatch.setattr(sec_edgar, "TICKER_MAP_URL", "http://127.0.0.1:1/nothing")
    result = SecEdgarCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert result.degraded
    assert result.errors
    assert result.raw_facts == []


# --- Phase 4.3F: structured evidence on CollectionResult -------------------
def test_collect_populates_company_identity_evidence(patched_endpoints, client):
    result = SecEdgarCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    evidence = result.company_identity_evidence
    assert evidence is not None
    assert evidence.ticker == "TESTCO"
    assert evidence.cik == 1595097
    assert evidence.sec_official_name == "Test Company Holdings Inc"
    assert evidence.source_tier is SourceTier.TIER_1
    assert evidence.content_hash != UNKNOWN


def test_company_identity_evidence_source_is_registered_and_validates(patched_endpoints, client):
    """The ticker-map Source this evidence references is actually present
    in result.sources (never a dangling reference), and the evidence
    validates cleanly against a context built from exactly that list --
    proving source_tier/content_hash/retrieved_at genuinely match, not
    merely happen to be present."""
    result = SecEdgarCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    evidence = result.company_identity_evidence
    assert evidence is not None
    assert evidence.source_id in {s.source_id for s in result.sources}
    context = EvidenceValidationContext(sources_by_id={s.source_id: s for s in result.sources})
    outcome = validate_company_identity_evidence(evidence, context)
    assert outcome.outcome == EvidenceValidationOutcome.VALID


def test_unresolved_ticker_never_populates_company_identity_evidence(patched_endpoints, client):
    result = SecEdgarCollector(client).collect("NOSUCH", "Nobody Inc")
    assert result.company_identity_evidence is None


def test_collector_failure_never_populates_company_identity_evidence(client, monkeypatch):
    monkeypatch.setattr(sec_edgar, "TICKER_MAP_URL", "http://127.0.0.1:1/nothing")
    result = SecEdgarCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert result.company_identity_evidence is None


def test_resolve_cik_return_signature_is_unchanged_by_the_richer_collect_path(
    patched_endpoints, client
):
    """Phase 4.3F refactored collect()'s internals but must not change
    resolve_cik()'s own public 3-tuple contract -- existing callers unpack
    exactly (cik, name, outcome)."""
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("TESTCO")
    assert cik == 1595097
    assert name == "Test Company Holdings Inc"
    assert outcome is FetchOutcome.OK


def test_collect_populates_program_candidate_evidence(patched_endpoints, client):
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert len(result.program_candidate_evidence) == 1
    evidence = result.program_candidate_evidence[0]
    assert evidence.nct_id == "NCT01234567"
    assert evidence.lead_sponsor == "Test Company Holdings Inc"
    assert evidence.overall_status == "RECRUITING"
    assert evidence.phases == ("PHASE2",)
    assert evidence.source_tier is SourceTier.TIER_1


def test_program_candidate_evidence_source_and_facts_are_registered_and_validate(
    patched_endpoints, client
):
    """The candidate's source_id resolves in result.sources, its
    supporting_fact_ids resolve to the SAME fact_id RawFact.fact_id()
    computes (the exact formula FactCollectorAgent later uses for
    Fact.fact_id), and the whole record validates cleanly against a
    context built from real Fact objects at those exact ids."""
    from investment_research.schemas.enums import EvidenceClass, Materiality, VerifiedStatus
    from investment_research.schemas.fact import Fact

    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    evidence = result.program_candidate_evidence[0]
    assert evidence.source_id in {s.source_id for s in result.sources}
    assert evidence.supporting_fact_ids
    assert set(evidence.supporting_fact_ids) == {rf.fact_id() for rf in result.raw_facts}

    verified_facts = {
        rf.fact_id(): Fact(
            fact_id=rf.fact_id(), ticker=rf.ticker, category=rf.category, claim=rf.claim,
            evidence_class=EvidenceClass.VERIFIED_FACT, source_id=rf.source.source_id,
            source_url=rf.source.url, source_title=rf.source.title, source_tier=rf.source.tier,
            verified_status=VerifiedStatus.VERIFIED, materiality=Materiality.MEDIUM,
        )
        for rf in result.raw_facts
    }
    context = EvidenceValidationContext(
        sources_by_id={s.source_id: s for s in result.sources},
        verified_facts_by_id=verified_facts,
    )
    outcome = validate_program_candidate_evidence(evidence, context)
    assert outcome.outcome == EvidenceValidationOutcome.VALID


def test_no_studies_found_yields_empty_program_candidate_evidence(
    patched_endpoints, monkeypatch, client
):
    monkeypatch.setattr(clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_empty")
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Nobody Sponsors This Inc")
    assert result.program_candidate_evidence == ()


def test_xbrl_metric_takes_the_latest_period(patched_endpoints, client):
    cik, _, _ = SecEdgarCollector(client).resolve_cik("TESTCO")
    facts, outcome = SecEdgarCollector(client).company_facts(cik)
    assert outcome is FetchOutcome.OK
    value, as_of = extract_xbrl_metric(facts, "CashAndCashEquivalentsAtCarryingValue")
    assert value == 31_500_000
    assert as_of == "2026-06-30"


def test_missing_xbrl_tag_returns_unknown_not_zero(patched_endpoints, client):
    cik, _, _ = SecEdgarCollector(client).resolve_cik("TESTCO")
    facts, _ = SecEdgarCollector(client).company_facts(cik)
    value, as_of = extract_xbrl_metric(facts, "NoSuchTag")
    assert value is None
    assert as_of == UNKNOWN


# --- ClinicalTrials.gov -----------------------------------------------------
def test_study_parsing_extracts_design_not_just_existence():
    parsed = parse_study(STUDIES["studies"][0])
    assert parsed["nct_id"] == "NCT01234567"
    assert parsed["phase"] == "PHASE2"
    assert parsed["enrollment"] == 84
    assert parsed["allocation"] == "RANDOMIZED"
    assert parsed["masking"] == "DOUBLE"
    assert parsed["primary_endpoint"].startswith("Change from baseline")
    assert parsed["primary_completion"] == "2026-11-30"
    assert parsed["primary_completion_type"] == "ESTIMATED"


def test_trial_collection_emits_endpoint_and_design_facts(patched_endpoints, client):
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert result.outcome is FetchOutcome.OK
    claims = [f.claim for f in result.raw_facts]
    assert any("primary outcome measure" in c for c in claims)
    assert any("allocation is RANDOMIZED" in c for c in claims)
    assert any("enrollment is 84" in c for c in claims)
    assert all(f.category is FactCategory.CLINICAL for f in result.raw_facts)


def test_no_company_name_skips_rather_than_guessing(client):
    result = ClinicalTrialsCollector(client).collect("TESTCO", UNKNOWN)
    assert result.outcome is FetchOutcome.NOT_FOUND
    assert "skipped rather than guessed" in result.errors[0]
    assert result.degraded is True


def test_clinicaltrials_zero_studies_found_is_still_degraded(patched_endpoints, monkeypatch, client):
    """Generic NOT_FOUND semantics are restored: unlike FdaCollector, this
    collector makes no explicit zero-result determination, so a clean,
    error-free "no registered studies" answer is NOT globally reinterpreted
    as success -- it stays degraded, exactly as it did before any FDA-specific
    exemption existed."""
    monkeypatch.setattr(clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_empty")
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Nobody Sponsors This Inc")
    assert result.outcome is FetchOutcome.NOT_FOUND
    assert result.errors == []
    assert any("no registered studies found" in n for n in result.notes)
    assert result.degraded is True, (
        "a non-FDA collector's NOT_FOUND must never be silently treated as success"
    )


# --- FDA: zero-result 404 vs a real failure (requirement D) -----------------
def test_fda_zero_result_404_is_explicit_ok_zero_results_not_degraded(base_url, monkeypatch, client):
    """A valid Drugs@FDA query with no matching applications is openFDA's own
    normal 404 response. FdaCollector -- and only FdaCollector -- translates
    that into an explicit outcome=OK, zero_results=True; the 404 itself is
    never globally reinterpreted as success."""
    monkeypatch.setattr(fda, "DRUGSFDA_URL", f"{base_url}/drug/drugsfda_zero_results.json")
    result = FdaCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert result.outcome is FetchOutcome.OK
    assert result.zero_results is True
    assert result.errors == []
    assert result.degraded is False
    assert any("ZERO_RESULTS" in n and "no Drugs@FDA applications listed" in n for n in result.notes)


def test_fda_zero_results_note_does_not_resolve_other_regulatory_questions(
    base_url, monkeypatch, client
):
    monkeypatch.setattr(fda, "DRUGSFDA_URL", f"{base_url}/drug/drugsfda_zero_results.json")
    result = FdaCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    note = " ".join(result.notes)
    for phrase in (
        "Type A/B/C",
        "endpoint acceptability",
        "Special Protocol Assessment",
        "CMC",
        "clinical hold",
        "unresolved",
    ):
        assert phrase in note, f"zero-result note must not imply {phrase!r} is settled"


def test_fda_query_with_results_is_not_degraded(base_url, monkeypatch, client):
    monkeypatch.setattr(fda, "DRUGSFDA_URL", f"{base_url}/drug/drugsfda_with_results.json")
    result = FdaCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert result.outcome is FetchOutcome.OK
    assert result.degraded is False
    assert result.raw_facts
    assert result.sources and result.sources[0].tier is SourceTier.TIER_1


def test_fda_real_connectivity_failure_still_degrades(monkeypatch, client):
    """A genuine connectivity/server failure -- not a 404 -- must still degrade."""
    monkeypatch.setattr(fda, "DRUGSFDA_URL", "http://127.0.0.1:1/nothing")
    result = FdaCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert result.degraded is True
    assert result.errors
    assert result.outcome is not FetchOutcome.NOT_FOUND


def test_fda_no_company_name_is_still_degraded():
    """Distinct from the zero-result case: the query was never meaningfully
    attempted at all, so this must still count as degraded."""
    result = FdaCollector(HttpClient(offline=True, cache_dir=None)).collect("TESTCO", UNKNOWN)
    assert result.outcome is FetchOutcome.NOT_FOUND
    assert result.errors
    assert result.degraded is True
