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
                    "studyFirstPostDateStruct": {"date": "2025-05-15", "type": "ACTUAL"},
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

def _single_study(
    nct_id: str,
    *,
    sponsor: str | None = "Test Company Holdings Inc",
    organization: str | None = None,
    collaborators: list[str] | None = None,
    status: str = "RECRUITING",
) -> dict:
    """A minimal single-study payload, shaped like ``STUDIES``'s own entry,
    varying the ``nctId``/sponsor/organization/collaborators/status -- for
    Phase 4.3F correction 1/2's collector-level gate tests.

    ``sponsor=None`` omits ``sponsorCollaboratorsModule.leadSponsor``
    entirely (``parse_study()`` then reads it back as ``UNKNOWN``);
    ``sponsor=""`` sets an explicit empty name."""
    sponsor_module: dict = {}
    if sponsor is not None:
        sponsor_module["leadSponsor"] = {"name": sponsor}
    if collaborators:
        sponsor_module["collaborators"] = [{"name": c} for c in collaborators]
    ident: dict = {"nctId": nct_id, "briefTitle": "An Edge-Case Study"}
    if organization is not None:
        ident["organization"] = {"fullName": organization}
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": ident,
                    "statusModule": {
                        "overallStatus": status,
                        "primaryCompletionDateStruct": {"date": "2026-11-30", "type": "ESTIMATED"},
                        "lastUpdatePostDateStruct": {"date": "2026-04-02"},
                        "startDateStruct": {"date": "2025-06-01"},
                    },
                    "sponsorCollaboratorsModule": sponsor_module,
                    "designModule": {"phases": ["PHASE2"]},
                }
            }
        ]
    }


def _studies_payload(*studies_payloads: dict) -> dict:
    """Combine multiple ``_single_study``-shaped single-entry payloads into
    one multi-study payload -- Phase 4.3F correction 2 Finding 3's
    duplicate-NCT-id tests."""
    return {"studies": [p["studies"][0] for p in studies_payloads]}


#: Non-empty but strictly-invalid (7 digits, not 8) NCT id -- Phase 4.3F
#: correction 1 Finding B: must skip structured candidate evidence, never
#: skip the study's RawFacts.
STUDIES_INVALID_NCT = _single_study("NCT1234567")

#: Strict-valid but non-canonically-cased/whitespace-padded NCT id -- Phase
#: 4.3F correction 1 Finding B: the resulting candidate's nct_id must be the
#: validator's canonical (upper-cased, stripped) form.
STUDIES_LOWERCASE_NCT = _single_study(" nct01234567 ")

#: Lead sponsor missing (UNKNOWN) even though organization/collaborator ARE
#: present -- Phase 4.3F correction 2 Finding 2: must never be substituted
#: for lead_sponsor, so no candidate is produced.
STUDIES_MISSING_SPONSOR = _single_study(
    "NCT55555555",
    sponsor=None,
    organization="Some Research Organization",
    collaborators=["Some Collaborator University"],
)

#: Two study records for the SAME NCT id whose structured content agrees in
#: full -- Phase 4.3F correction 2 Finding 3: must dedup to one candidate.
STUDIES_DUPLICATE_IDENTICAL = _studies_payload(
    _single_study("NCT77777777"),
    _single_study("NCT77777777"),
)

#: Two study records for the SAME NCT id whose overall_status disagrees --
#: Phase 4.3F correction 2 Finding 3: must yield zero candidates for this
#: NCT id (never first-wins/last-wins).
STUDIES_DUPLICATE_CONFLICTING = _studies_payload(
    _single_study("NCT88888888", status="RECRUITING"),
    _single_study("NCT88888888", status="COMPLETED"),
)

#: A ticker map with malformed records mixed in with well-formed ones --
#: Phase 4.3F correction 2 Finding 4. Entry "0" is a non-dict entry placed
#: FIRST so every lookup below scans past it. "NOCIK"/"BADCIK"/"BOOLCIK"
#: have an unusable cik_str (missing, non-numeric, bool); "NOTITLE"/
#: "UNKNOWNTITLE" have a valid cik_str but no usable official title;
#: "GOODCIK" is well-formed, to prove the malformed entries never break
#: resolution of a later, valid one.
TICKER_MAP_MALFORMED = {
    "0": "not-a-dict-entry",
    "1": {"ticker": "NOCIK", "title": "Missing Cik Str Inc"},
    "2": {"cik_str": "not-a-number", "ticker": "BADCIK", "title": "Bad Cik Str Inc"},
    "3": {"cik_str": True, "ticker": "BOOLCIK", "title": "Bool Cik Str Inc"},
    "4": {"cik_str": 1595097, "ticker": "NOTITLE", "title": ""},
    "5": {"cik_str": 1234567, "ticker": "UNKNOWNTITLE", "title": "UNKNOWN"},
    "6": {"cik_str": 9999999, "ticker": "GOODCIK", "title": "Good Cik Inc"},
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
            "/files/company_tickers_malformed.json": TICKER_MAP_MALFORMED,
            "/submissions/CIK0001595097.json": SUBMISSIONS,
            "/api/v2/studies": STUDIES,
            "/api/xbrl/companyfacts/CIK0001595097.json": COMPANY_FACTS,
            "/drug/drugsfda_with_results.json": DRUGSFDA_WITH_RESULTS,
            "/api/v2/studies_empty": STUDIES_EMPTY,
            "/api/v2/studies_invalid_nct": STUDIES_INVALID_NCT,
            "/api/v2/studies_lowercase_nct": STUDIES_LOWERCASE_NCT,
            "/api/v2/studies_missing_sponsor": STUDIES_MISSING_SPONSOR,
            "/api/v2/studies_duplicate_identical": STUDIES_DUPLICATE_IDENTICAL,
            "/api/v2/studies_duplicate_conflicting": STUDIES_DUPLICATE_CONFLICTING,
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


def _malformed_ticker_map(patched_endpoints, monkeypatch):
    monkeypatch.setattr(
        sec_edgar, "TICKER_MAP_URL", f"{patched_endpoints}/files/company_tickers_malformed.json"
    )


def test_missing_cik_str_does_not_raise(patched_endpoints, monkeypatch, client):
    """Phase 4.3F correction 2 Finding 4: a matching ticker-map entry with
    no cik_str at all is skipped, never raises, and never guesses a CIK."""
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("NOCIK")
    assert cik is None
    assert name == UNKNOWN
    assert outcome is FetchOutcome.NOT_FOUND


def test_non_numeric_cik_str_does_not_raise(patched_endpoints, monkeypatch, client):
    """A cik_str that cannot be converted to an int is skipped, never
    raises ValueError."""
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("BADCIK")
    assert cik is None
    assert name == UNKNOWN
    assert outcome is FetchOutcome.NOT_FOUND


def test_bool_cik_str_is_never_treated_as_a_valid_cik(patched_endpoints, monkeypatch, client):
    """A cik_str of ``True`` would silently become 1 under a bare int()
    call -- normalize_cik() rejects bools explicitly, so this must never
    resolve to a (wrong) valid-looking CIK."""
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("BOOLCIK")
    assert cik is None
    assert name == UNKNOWN
    assert outcome is FetchOutcome.NOT_FOUND


def test_empty_title_resolves_cik_but_name_stays_unknown(patched_endpoints, monkeypatch, client):
    """A valid cik_str with an empty official title still resolves the
    CIK (it IS valid) but the name is reported as UNKNOWN -- never left as
    an empty string, never guessed."""
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("NOTITLE")
    assert cik == 1595097
    assert name == UNKNOWN
    assert outcome is FetchOutcome.OK


def test_explicit_unknown_title_stays_unknown(patched_endpoints, monkeypatch, client):
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("UNKNOWNTITLE")
    assert cik == 1234567
    assert name == UNKNOWN
    assert outcome is FetchOutcome.OK


def test_malformed_entries_do_not_break_resolution_of_a_later_well_formed_one(
    patched_endpoints, monkeypatch, client
):
    """A non-dict entry and several malformed dict entries appear BEFORE
    the well-formed "GOODCIK" entry in the payload -- the scan must pass
    over all of them without raising and still resolve the good one."""
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    cik, name, outcome = SecEdgarCollector(client).resolve_cik("GOODCIK")
    assert cik == 9999999
    assert name == "Good Cik Inc"
    assert outcome is FetchOutcome.OK


def test_malformed_cik_and_missing_title_never_raise_and_never_populate_identity_evidence(
    patched_endpoints, monkeypatch, client
):
    """Phase 4.3F correction 2 Finding 4's own required coverage: neither a
    malformed cik_str nor a missing/empty/UNKNOWN official title ever
    raises an exception through the full collect() path, and neither ever
    produces a CompanyIdentityEvidence record."""
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    collector = SecEdgarCollector(client)

    malformed_cik = collector.collect("BADCIK", "User Supplied Name Inc")
    assert malformed_cik.company_identity_evidence is None
    assert malformed_cik.outcome is FetchOutcome.NOT_FOUND

    missing_title = collector.collect("NOTITLE", "User Supplied Name Inc")
    assert missing_title.company_identity_evidence is None
    assert any("official title" in note for note in missing_title.notes)


def test_sec_collect_request_count_is_unaffected_by_malformed_entries(
    patched_endpoints, monkeypatch, client
):
    """Phase 4.3F correction 2 Finding 4: validating cik_str/title more
    strictly adds no additional HTTP request -- still exactly one
    ticker-map fetch and one submissions fetch, the same as a well-formed
    entry."""
    _malformed_ticker_map(patched_endpoints, monkeypatch)
    result = SecEdgarCollector(client).collect("NOTITLE", "User Supplied Name Inc")
    assert len(result.attempted_urls) == 2


def test_clinicaltrials_collect_request_count_is_unaffected_by_dedup_and_gates(
    patched_endpoints, monkeypatch, client
):
    """Phase 4.3F correction 2 Findings 2/3: sponsor-gating and duplicate-
    NCT dedup/conflict resolution are pure post-processing of the SAME
    single already-fetched payload -- never an additional request."""
    monkeypatch.setattr(
        clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_duplicate_conflicting"
    )
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert len(result.attempted_urls) == 1


def test_collect_populates_program_candidate_evidence(patched_endpoints, client):
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")
    assert len(result.program_candidate_evidence) == 1
    evidence = result.program_candidate_evidence[0]
    assert evidence.nct_id == "NCT01234567"
    assert evidence.lead_sponsor == "Test Company Holdings Inc"
    assert evidence.overall_status == "RECRUITING"
    assert evidence.phases == ("PHASE2",)
    assert evidence.source_tier is SourceTier.TIER_1
    # Phase 4.3F correction 1 Finding C: studyFirstPostDateStruct now flows
    # through end-to-end, never left at UNKNOWN when CT.gov provided it.
    assert evidence.first_posted_date == "2025-05-15"


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


def test_strictly_invalid_nct_id_skips_only_the_structured_candidate(
    patched_endpoints, monkeypatch, client
):
    """Phase 4.3F correction 1 Finding B: a non-empty but strictly-invalid
    NCT id (here, 7 digits, not 8) must never produce a
    ProgramCandidateEvidence -- but the study's RawFacts are still built
    normally, and the collector's own outcome is unaffected. Optional
    structured evidence being unavailable is never grounds to change
    Action/status."""
    monkeypatch.setattr(
        clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_invalid_nct"
    )
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")

    assert result.program_candidate_evidence == ()
    assert result.outcome is FetchOutcome.OK
    assert result.raw_facts
    assert any("NCT1234567" in f.claim for f in result.raw_facts)
    assert any("strict validation" in note for note in result.notes)


def test_lowercase_whitespace_nct_id_is_canonicalized_not_rejected(
    patched_endpoints, monkeypatch, client
):
    """Phase 4.3F correction 1 Finding B: a strict-valid NCT id that is not
    already in canonical form (here, lowercase with surrounding whitespace)
    still produces a candidate, whose nct_id is always the validator's
    canonical NCT+8-digit form -- never the raw, un-normalized value."""
    monkeypatch.setattr(
        clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_lowercase_nct"
    )
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")

    assert len(result.program_candidate_evidence) == 1
    assert result.program_candidate_evidence[0].nct_id == "NCT01234567"


def test_missing_sponsor_skips_only_the_structured_candidate(patched_endpoints, monkeypatch, client):
    """Phase 4.3F correction 2 Finding 2: a strict-valid NCT id with no lead
    sponsor never produces a ProgramCandidateEvidence -- but RawFacts and
    the collector's own outcome are unaffected, and a note is recorded."""
    monkeypatch.setattr(
        clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_missing_sponsor"
    )
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")

    assert result.program_candidate_evidence == ()
    assert result.outcome is FetchOutcome.OK
    assert result.raw_facts
    assert any("lead sponsor" in note for note in result.notes)


def test_missing_sponsor_is_never_substituted_from_organization_or_collaborator(
    patched_endpoints, monkeypatch, client
):
    """Even when organization and collaborator names ARE present in the
    payload, neither is ever used as a stand-in for a missing lead
    sponsor."""
    monkeypatch.setattr(
        clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_missing_sponsor"
    )
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")

    assert result.program_candidate_evidence == ()


def test_identical_duplicate_nct_dedups_to_one_candidate(patched_endpoints, monkeypatch, client):
    """Phase 4.3F correction 2 Finding 3: two study records for the same
    NCT id with agreeing structured content collapse to exactly one
    candidate, never two."""
    monkeypatch.setattr(
        clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_duplicate_identical"
    )
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")

    assert len(result.program_candidate_evidence) == 1
    assert result.program_candidate_evidence[0].nct_id == "NCT77777777"
    # RawFacts are unaffected by the dedup -- one full set of facts per
    # study record (two records here), even though they collapse to one
    # candidate.
    facts_per_study = sum(1 for f in result.raw_facts if "NCT77777777" in f.claim) // 2
    assert facts_per_study > 0
    assert sum(1 for f in result.raw_facts if "NCT77777777" in f.claim) == 2 * facts_per_study


def test_conflicting_duplicate_nct_yields_zero_candidates(patched_endpoints, monkeypatch, client):
    """Phase 4.3F correction 2 Finding 3: two study records for the same
    NCT id with CONFLICTING structured content (here, overall_status)
    yield zero candidates for that NCT id -- never an arbitrary
    first-wins/last-wins pick -- and a note is recorded. RawFacts for both
    study records are still built."""
    monkeypatch.setattr(
        clinicaltrials, "STUDIES_URL", f"{patched_endpoints}/api/v2/studies_duplicate_conflicting"
    )
    result = ClinicalTrialsCollector(client).collect("TESTCO", "Test Company Holdings Inc")

    assert result.program_candidate_evidence == ()
    assert result.outcome is FetchOutcome.OK
    assert any("NCT88888888" in note and "conflicting" in note for note in result.notes)
    assert sum(1 for f in result.raw_facts if "NCT88888888" in f.claim) > 0


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
    # Phase 4.3F correction 1 Finding C: studyFirstPostDateStruct.
    assert parsed["first_posted_date"] == "2025-05-15"
    assert parsed["first_posted_date_type"] == "ACTUAL"


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
