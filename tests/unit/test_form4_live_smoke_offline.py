"""Phase 3E.2: Form4 Live Smoke, entirely offline -- every test drives
``form4_live_smoke.run_live_smoke``/``analyze_capture`` against
``FakeHttpClient``/``_PoisonHttpClient`` (never a real socket) or a temp
directory of already-saved captures. No real network call anywhere in
this file -- enforced structurally by ``forbid_external_network_autouse``
(Phase 3E.2.1), not merely by test authoring discipline: every test in
this module runs with real socket connections to any non-loopback host
blocked, so a test that (by a future bug) let a real ``AllowlistedHttpClient``
reach the real network would fail loudly with ``AssertionError`` instead
of silently making a real SEC request.
"""

from __future__ import annotations

import json

from investment_research.research import form4_live_smoke as live_smoke
from investment_research.research.capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    CaptureManifest,
    write_manifest,
)

from . import _form4_fixture_support as fx
from ._network_guard import forbid_external_network_autouse  # noqa: F401

USER_AGENT = "test-agent contact@example.com"


class _PoisonHttpClient:
    """A fake transport whose ``.get()`` must never be called -- proves a
    refused/pre-network-failure run makes ZERO requests, not merely
    "few" (mirrors ``sec_live_smoke_offline``'s own ``_PoisonHttpClient``)."""

    def get(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"must never be called -- attempted GET {url}")


class _PoisonHttpClientFactory:
    """A ``main()``-level ``http_client_factory`` whose construction must
    never even be INVOKED when a run refuses before reaching the client-
    construction step (e.g. missing user agent, marker guard)."""

    def __init__(self) -> None:
        self.called = False

    def __call__(self, user_agent: str, out_dir, max_requests: int) -> object:
        self.called = True
        return _PoisonHttpClient()


def _run(http, **kwargs):
    kwargs.setdefault("as_of_date", "2026-03-15")
    kwargs.setdefault("lookback_days", 180)
    kwargs.setdefault("max_filings", 3)
    return live_smoke.run_live_smoke(
        fx.ISSUER_CIK_INT, user_agent=USER_AGENT, http_client=http, **kwargs
    )


# --- input / refusal ---------------------------------------------------
def test_refuses_with_zero_requests_when_user_agent_unset():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = live_smoke.run_live_smoke(
        fx.ISSUER_CIK_INT, user_agent=None, http_client=http,
    )
    assert report.refused
    assert http.requested_urls == []
    assert report.anthropic_api_calls == 0
    assert report.web_search_calls == 0
    assert report.external_llm_tokens == 0


def test_no_accession_or_primary_document_input_accepted_by_run_live_smoke():
    """run_live_smoke's own signature never accepts an accession, a
    primary document filename, or a reporting-owner CIK -- issuer_cik is
    the only identifying input."""
    import inspect

    params = set(inspect.signature(live_smoke.run_live_smoke).parameters)
    assert "accession" not in params
    assert "primary_document" not in params
    assert "reporting_owner_cik" not in params
    assert "issuer_cik" in params


# --- discovery: happy path, form3/5 exclusion, candidates_found/form4_count
def test_issuer_cik_alone_discovers_candidates_and_excludes_form3_5():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = _run(http)
    assert report.status == "COMPLETED"
    # Every _SCENARIOS entry becomes a candidate except "metadata_says_form3"
    # (its own submissions-metadata `form` value is "3", filtered by
    # VALID_FORM4_DOCUMENT_TYPES before any body is ever fetched).
    assert report.candidates_found_total == len(fx._SCENARIOS) - 1
    assert report.form4_count_total + report.form4a_count_total == report.candidates_found_total
    # Form 3 exists in the RAW submissions data (proves the filter did
    # something, never merely asserted) but zero Form-3 candidates survive.
    assert report.raw_submissions_forms_seen.get("3", 0) >= 1
    assert all(c["form"] in ("4", "4/A") for c in report.discovery_candidates)


def test_archive_cik_reported_per_candidate():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = _run(http)
    assert report.discovery_candidates
    for candidate in report.discovery_candidates:
        assert candidate["archive_cik_used"] == fx.cik_for_archives(fx.ISSUER_CIK_INT)


def test_continuation_page_candidate_is_discovered():
    http = fx.FakeHttpClient(responses=fx.default_responses(with_continuation_pointer=True))
    report = _run(http, lookback_days=3000)  # wide enough to include the 2020 continuation-page filing
    accessions = {c["accession"] for c in report.discovery_candidates}
    assert fx.continuation_accession() in accessions
    assert report.submissions_pages_fetched == 1
    assert report.submissions_pages_excluded == 0


# --- DISCOVERY_NOT_SUPPORTED --------------------------------------------
def test_zero_form4_filings_reports_discovery_not_supported_never_fallback():
    payload = {
        "cik": fx.ISSUER_CIK_INT, "name": fx.ISSUER_NAME, "tickers": [fx.ISSUER_TICKER],
        "filings": {"recent": {"accessionNumber": [], "form": [], "primaryDocument": [], "filingDate": [], "reportDate": []}, "files": []},
    }
    http = fx.FakeHttpClient(responses={fx.submissions_url(): fx.ok(json.dumps(payload))})
    report = _run(http)
    assert report.status == "DISCOVERY_NOT_SUPPORTED"
    assert report.fetched == 0
    assert not report.parsed_documents


def test_submissions_fetch_failure_is_locate_failed_not_discovery_not_supported():
    http = fx.FakeHttpClient(responses={fx.submissions_url(): fx.not_found()})
    report = _run(http)
    assert report.status == "LOCATE_FAILED"


# --- lookback window / cap ----------------------------------------------
def test_no_candidates_in_lookback_window_is_distinct_from_discovery_not_supported():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = _run(http, as_of_date="1999-01-01", lookback_days=1)
    assert report.status == "NO_CANDIDATES_IN_LOOKBACK_WINDOW"
    assert report.candidates_found_total > 0  # discovery genuinely worked
    assert report.candidates_found == 0


def test_not_fetched_due_to_cap_is_nonzero_when_more_candidates_than_max_filings():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = _run(http, max_filings=2)
    assert report.fetched == 2
    assert report.not_fetched_due_to_cap == report.candidates_found - 2
    assert report.not_fetched_due_to_cap > 0


def test_never_overwrites_original_candidates_with_fetched_subset():
    """discovery_candidates (the full LOCATE result) stays the full set even
    when fetched is capped smaller -- never silently replaced/shrunk."""
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = _run(http, max_filings=1)
    assert len(report.discovery_candidates) == report.candidates_found_total
    assert report.fetched == 1
    assert len(report.parsed_documents) == 1


# --- CIK mismatch ---------------------------------------------------------
def test_cik_mismatch_candidate_excluded_from_parsed_documents():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = _run(http, max_filings=50, lookback_days=3650)
    parsed_accessions = {d["accession"] for d in report.parsed_documents}
    assert fx.accession("cik_mismatch") not in parsed_accessions
    assert any(fx.accession("cik_mismatch") in f["accession"] for f in report.fetch_failures)


# --- Rule 10b5-1 / dateOfOriginalSubmission diagnostics -------------------
def _single_candidate_client(fixture_name: str, *, accession_suffix: str = "000099"):
    accession = f"{fx.ISSUER_CIK}-26-{accession_suffix}"
    accession_nodash = accession.replace("-", "")
    submissions_payload = {
        "cik": fx.ISSUER_CIK_INT, "name": fx.ISSUER_NAME, "tickers": [fx.ISSUER_TICKER],
        "filings": {
            "recent": {
                "accessionNumber": [accession], "form": ["4"], "primaryDocument": [fx.PRIMARY_DOCUMENT],
                "filingDate": ["2026-03-01"], "reportDate": ["2026-03-01"],
            },
            "files": [],
        },
    }
    from investment_research.collectors.sec_edgar import FILING_INDEX_URL, cik_for_archives

    directory_url = FILING_INDEX_URL.format(cik=cik_for_archives(fx.ISSUER_CIK_INT), accession_nodash=accession_nodash, document="index.json")
    document_url = FILING_INDEX_URL.format(cik=cik_for_archives(fx.ISSUER_CIK_INT), accession_nodash=accession_nodash, document=fx.PRIMARY_DOCUMENT)
    responses = {
        fx.submissions_url(): fx.ok(json.dumps(submissions_payload)),
        directory_url: fx.ok(json.dumps({"directory": {"item": [{"name": fx.PRIMARY_DOCUMENT, "type": "4"}]}})),
        document_url: fx.ok(fx.fixture_text(fixture_name)),
    }
    return fx.FakeHttpClient(responses=responses), accession


def test_checkbox_and_date_present_scanned_independently_of_production_guess():
    http, accession = _single_candidate_client("checkbox_and_date_present.xml")
    report = _run(http)
    assert report.status == "COMPLETED"
    doc = next(d for d in report.parsed_documents if d["accession"] == accession)
    diag = doc["structure_diagnostics"]
    assert diag["well_formed"] is True
    assert diag["document_type"] == "4"
    assert diag["issuer_cik"] == fx.ISSUER_CIK
    checkbox_matches = diag["rule_10b5_1_candidate_elements"]
    assert any(m["tag"] == "aff10b5One" and m["raw_value"] == "1" for m in checkbox_matches)
    assert any(m["is_direct_child_of_ownership_document"] for m in checkbox_matches)
    assert diag["date_of_original_submission"]["found"] is True
    assert diag["date_of_original_submission"]["raw_value"] == "2026-02-15"
    # production's own PARSE-stage guess should also agree for this fixture.
    assert doc["parsed_ten_b5_1_checkbox"] is True


def test_checkbox_and_date_absent_report_unknown_never_guessed():
    http, accession = _single_candidate_client("normal_market_purchase.xml")
    report = _run(http)
    doc = next(d for d in report.parsed_documents if d["accession"] == accession)
    diag = doc["structure_diagnostics"]
    assert diag["rule_10b5_1_candidate_elements"] == []
    assert diag["date_of_original_submission"] == {"found": False, "tag": None, "raw_value": None}
    assert doc["parsed_ten_b5_1_checkbox"] is None  # never a guessed False


def test_scan_functions_never_raise_on_malformed_xml():
    assert live_smoke.scan_rule_10b5_1_elements("<not><closed>") == []
    result = live_smoke.scan_date_of_original_submission("<not><closed>")
    assert result == {"found": False, "tag": None, "raw_value": None}
    diag = live_smoke.structure_diagnostics("<not><closed>")
    assert diag["well_formed"] is False


# --- malformed / 404 / 429 / 5xx / timeout fetch failures ------------------
def test_malformed_xml_candidate_reported_as_fetch_failure_not_parsed():
    http, accession = _single_candidate_client("malformed_xml.xml")
    report = _run(http)
    assert accession not in {d["accession"] for d in report.parsed_documents}
    assert any(accession in f["accession"] for f in report.fetch_failures)


def test_document_body_404_reported_distinctly():
    accession = f"{fx.ISSUER_CIK}-26-000099"
    accession_nodash = accession.replace("-", "")
    from investment_research.collectors.sec_edgar import FILING_INDEX_URL, cik_for_archives

    submissions_payload = {
        "cik": fx.ISSUER_CIK_INT, "name": fx.ISSUER_NAME, "tickers": [fx.ISSUER_TICKER],
        "filings": {"recent": {
            "accessionNumber": [accession], "form": ["4"], "primaryDocument": [fx.PRIMARY_DOCUMENT],
            "filingDate": ["2026-03-01"], "reportDate": ["2026-03-01"],
        }, "files": []},
    }
    directory_url = FILING_INDEX_URL.format(cik=cik_for_archives(fx.ISSUER_CIK_INT), accession_nodash=accession_nodash, document="index.json")
    document_url = FILING_INDEX_URL.format(cik=cik_for_archives(fx.ISSUER_CIK_INT), accession_nodash=accession_nodash, document=fx.PRIMARY_DOCUMENT)
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(submissions_payload)),
        directory_url: fx.ok(json.dumps({"directory": {"item": [{"name": fx.PRIMARY_DOCUMENT, "type": "4"}]}})),
        document_url: fx.not_found("404"),
    })
    report = _run(http)
    assert any(accession in f["accession"] and "404" in f["reason"] for f in report.fetch_failures)


# --- request cap formula ---------------------------------------------------
def test_compute_max_live_gets_formula():
    from investment_research.research.form4_acquisition_adapter import MAX_SUBMISSIONS_PAGES

    assert live_smoke.compute_max_live_gets(3) == 1 + MAX_SUBMISSIONS_PAGES + 3 * 2
    assert live_smoke.compute_max_live_gets(1) == 1 + MAX_SUBMISSIONS_PAGES + 1 * 2


def test_zero_anthropic_web_search_llm_calls_always():
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = _run(http)
    assert report.anthropic_api_calls == 0
    assert report.web_search_calls == 0
    assert report.external_llm_tokens == 0


# --- Capture Manifest: save + offline re-analysis --------------------------
def test_capture_manifest_saved_and_analyze_capture_reconstructs_it(tmp_path):
    """This module's real capture-saving path IS
    ``sec_live_smoke.AllowlistedHttpClient._save_response`` (already
    proven, for ``source="form4"``, by ``test_capture_manifest.py``'s own
    generic tests and by ``test_form4_live_smoke_transport.py``'s
    loopback-server test). Here we only prove ``analyze_capture``'s own
    reconstruction logic against a manifest written the same way that
    code path writes one."""
    import hashlib

    body = b"body"
    digest = hashlib.sha256(fx.submissions_url().encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_bytes(body)
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION, source="form4",
        requested_url=fx.submissions_url(), final_url=fx.submissions_url(),
        http_status=200, capture_retrieved_at="2026-03-15T00:00:00Z",
        content_hash=hashlib.sha256(body).hexdigest(), content_length=len(body),
    )
    write_manifest(tmp_path, digest, manifest)

    result = live_smoke.analyze_capture(tmp_path)
    assert result["captures_found"] == 1
    entry = result["entries"][0]
    assert entry["capture_manifest_status"] == "VERIFIED"
    assert entry["requested_url"] == fx.submissions_url()
    assert entry["category"] == "submissions"
    assert result["evidence_integrity_failures"] == []


def test_analyze_capture_reports_hash_mismatch_as_evidence_integrity_failure(tmp_path):
    import hashlib

    digest = hashlib.sha256(fx.submissions_url().encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_bytes(b"actual-body")
    bad_manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION, source="form4",
        requested_url=fx.submissions_url(), final_url=fx.submissions_url(),
        http_status=200, capture_retrieved_at="2026-03-15T00:00:00Z",
        content_hash="f" * 64, content_length=len(b"actual-body"),
    )
    write_manifest(tmp_path, digest, bad_manifest)
    result = live_smoke.analyze_capture(tmp_path)
    entry = result["entries"][0]
    assert entry["capture_manifest_status"] == "HASH_MISMATCH"
    assert result["evidence_integrity_failures"]
    # never usable as a "verified" capture_retrieved_at.
    assert entry["capture_retrieved_at"] == "UNKNOWN"


def test_analyze_capture_body_missing_is_evidence_integrity_failure(tmp_path):
    import hashlib

    digest = hashlib.sha256(fx.submissions_url().encode()).hexdigest()[:24]
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION, source="form4",
        requested_url=fx.submissions_url(), final_url=fx.submissions_url(),
        http_status=200, capture_retrieved_at="2026-03-15T00:00:00Z",
        content_hash="a" * 64, content_length=1,
    )
    write_manifest(tmp_path, digest, manifest)
    result = live_smoke.analyze_capture(tmp_path)
    entry = result["entries"][0]
    assert entry["capture_manifest_status"] == "BODY_MISSING"
    assert result["evidence_integrity_failures"]


def test_analyze_capture_empty_directory_reports_zero_captures(tmp_path):
    result = live_smoke.analyze_capture(tmp_path)
    assert result["captures_found"] == 0
    assert result["entries"] == []
    assert result["evidence_integrity_failures"] == []


# --- one-time marker semantics ---------------------------------------------
#
# Phase 3E.2.1: the ORIGINAL version of this test called main() with no
# environment isolation at all, implicitly assuming IRA_SEC_USER_AGENT was
# unset in whatever shell ran the suite. On a Mac where it was genuinely
# and validly set, main() resolved a REAL user agent, constructed a REAL
# AllowlistedHttpClient (main() had no client-injection seam), and made a
# real GET against data.sec.gov for synthetic CIK 1112223 -- which
# correctly 404'd, and correctly wrote the one-time marker (a real
# communication attempt DID begin; that marker behavior is production-
# correct and must never be "fixed" by suppressing the marker write on a
# 404). The bug was test isolation, not marker semantics: this test now
# injects env={} (never relies on monkeypatching/ambient os.environ) and a
# poison http_client_factory (never even invoked, proving refusal happens
# before any client is constructed) so it is hermetically isolated from
# whatever the real ambient environment happens to contain.
def test_marker_not_written_on_missing_user_agent_refusal(tmp_path):
    marker_path = tmp_path / "LAST_RUN.json"
    poison_factory = _PoisonHttpClientFactory()
    exit_code = live_smoke.main(
        ["--issuer-cik", str(fx.ISSUER_CIK_INT), "--marker-path", str(marker_path)],
        env={},  # explicitly empty -- never falls through to a real ambient IRA_SEC_USER_AGENT
        http_client_factory=poison_factory,
    )
    assert exit_code == 1
    assert not marker_path.is_file()
    assert poison_factory.called is False  # refusal happens before any client would be constructed


def test_marker_not_written_when_run_live_smoke_refuses(monkeypatch):
    """The same guarantee at the run_live_smoke() level directly -- the
    Poison client proves zero requests, independent of main()'s own
    marker-writing code."""
    report = live_smoke.run_live_smoke(
        fx.ISSUER_CIK_INT, user_agent=None, http_client=_PoisonHttpClient(),
    )
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_main_writes_marker_only_after_real_communication_begins(tmp_path):
    """The positive case: when the user agent genuinely resolves (via
    injected env, never ambient) and the run proceeds, the marker IS
    written -- proving this phase did not weaken marker semantics, only
    isolated the test that exercises the refusal path."""
    marker_path = tmp_path / "LAST_RUN.json"
    out_dir = tmp_path / "captures"
    http = fx.FakeHttpClient(responses=fx.default_responses())
    exit_code = live_smoke.main(
        [
            "--issuer-cik", str(fx.ISSUER_CIK_INT), "--marker-path", str(marker_path),
            "--out-dir", str(out_dir), "--as-of", "2026-03-15",
        ],
        env={"IRA_SEC_USER_AGENT": USER_AGENT},
        http_client_factory=lambda user_agent, out_dir, max_requests: http,
    )
    assert exit_code == 0
    assert marker_path.is_file()
    assert http.requested_urls  # real communication (against the fake) genuinely began


# --- ambient environment variable isolation (regression) -------------------
#
# Mirrors sec_live_smoke_offline.py's own established battery exactly
# (same sentinel semantics: only OMITTING user_agent/env resolves from the
# ambient environment; an explicit None/""/whitespace/placeholder value
# must never silently fall back to it, however valid the ambient value
# is) -- unified here for form4_live_smoke (Phase 3E.2.1 requirement 2).
_AMBIENT_VALID_AGENT = "investment-research-agent ambient-mac-contact@example.test"


def test_explicit_none_refuses_even_with_valid_ambient_env(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live_smoke.run_live_smoke(fx.ISSUER_CIK_INT, user_agent=None, http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_explicit_placeholder_refuses_even_with_valid_ambient_env(monkeypatch):
    from investment_research.research.sec_live_smoke import PLACEHOLDER_SEC_USER_AGENT

    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live_smoke.run_live_smoke(
        fx.ISSUER_CIK_INT, user_agent=PLACEHOLDER_SEC_USER_AGENT, http_client=_PoisonHttpClient(),
    )
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_explicit_empty_string_refuses_even_with_valid_ambient_env(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live_smoke.run_live_smoke(fx.ISSUER_CIK_INT, user_agent="", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_explicit_whitespace_refuses_even_with_valid_ambient_env(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live_smoke.run_live_smoke(fx.ISSUER_CIK_INT, user_agent="   ", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_omitted_user_agent_resolves_from_ambient_env(monkeypatch):
    """Only omitting the parameter entirely (the shape main() uses after
    it has already resolved+validated the value itself) may resolve from
    the real environment."""
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    http = fx.FakeHttpClient(responses=fx.default_responses())
    report = live_smoke.run_live_smoke(
        fx.ISSUER_CIK_INT, as_of_date="2026-03-15", http_client=http,
    )
    assert not report.refused
    assert report.request_count > 0


def test_main_env_injection_isolates_from_real_ambient_environment(monkeypatch):
    """The main()-level analogue: even when the REAL ambient environment
    (via monkeypatch, standing in for a genuinely-set shell variable) has
    a valid IRA_SEC_USER_AGENT, an explicitly injected env={} still
    refuses -- proving main()'s env parameter, not the ambient process
    environment, is what actually governs resolution when supplied."""
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    poison_factory = _PoisonHttpClientFactory()
    exit_code = live_smoke.main(
        ["--issuer-cik", str(fx.ISSUER_CIK_INT)], env={}, http_client_factory=poison_factory,
    )
    assert exit_code == 1
    assert poison_factory.called is False


def test_main_omitted_env_resolves_from_real_ambient_ira_sec_user_agent(monkeypatch, tmp_path):
    """Confirms main()'s DEFAULT (env omitted -- what the real CLI always
    does) genuinely reads the real process environment, via monkeypatch
    rather than trusting whatever the suite's own ambient value is."""
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    http = fx.FakeHttpClient(responses=fx.default_responses())
    marker_path = tmp_path / "LAST_RUN.json"
    exit_code = live_smoke.main(
        [
            "--issuer-cik", str(fx.ISSUER_CIK_INT), "--marker-path", str(marker_path),
            "--out-dir", str(tmp_path / "captures"), "--as-of", "2026-03-15",
        ],
        http_client_factory=lambda user_agent, out_dir, max_requests: http,
    )
    assert exit_code == 0
    assert http.requested_urls
