"""Phase 3C: sec_live_smoke.py exercised entirely offline.

No real network call anywhere in this file. The orchestration tests inject
a ``FakeHttpClient`` (reusing ``_sec_fixture_support``'s real-SEC-format
fixtures, exactly like the Phase 3B adapter tests) in place of
``AllowlistedHttpClient``; the transport-layer tests (host allowlist,
redirect refusal, request cap) live in
``tests/integration/test_sec_live_smoke_transport.py`` against a local
loopback server, never the internet.
"""

from __future__ import annotations

import hashlib
import json
import re

from investment_research.config import Settings
from investment_research.research import sec_live_smoke as live
from investment_research.research.capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    CaptureManifest,
    ManifestReadStatus,
    compute_content_hash,
    is_evidence_integrity_failure,
    write_manifest,
)
from investment_research.research.document_store import DocumentRole
from investment_research.research.source_routing import (
    ImplementationStatus,
    SourceRoutingGraph,
    TargetAcquisitionOutcome,
)
from investment_research.schemas.enums import UNKNOWN

from . import _sec_fixture_support as fx
from ._network_guard import forbid_external_network_autouse  # noqa: F401
from ._sec_fixture_support import ACCESSION, CIK, PRIMARY_DOCUMENT, FakeHttpClient

_REAL_SECRET_LOOKING_AGENT = "investment-research-agent smoke-test-contact@example.test"


# --- env var / placeholder gating -------------------------------------------
def test_resolve_user_agent_returns_none_when_unset():
    assert live.resolve_user_agent(env={}) is None


def test_resolve_user_agent_returns_none_when_blank():
    assert live.resolve_user_agent(env={live.SEC_USER_AGENT_ENV_VAR: "   "}) is None


def test_resolve_user_agent_returns_none_when_placeholder():
    assert live.resolve_user_agent(env={live.SEC_USER_AGENT_ENV_VAR: live.PLACEHOLDER_SEC_USER_AGENT}) is None


def test_resolve_user_agent_returns_value_when_genuinely_set():
    assert live.resolve_user_agent(env={live.SEC_USER_AGENT_ENV_VAR: _REAL_SECRET_LOOKING_AGENT}) == _REAL_SECRET_LOOKING_AGENT


def test_placeholder_matches_config_default():
    """The placeholder this module refuses on must never silently drift
    away from config.Settings' own default -- both are asserted equal
    here, against a clean environment."""
    default_factory = Settings.__dataclass_fields__["sec_user_agent"].default_factory
    assert default_factory is not None
    import os

    original = os.environ.pop(live.SEC_USER_AGENT_ENV_VAR, None)
    try:
        assert default_factory() == live.PLACEHOLDER_SEC_USER_AGENT
    finally:
        if original is not None:
            os.environ[live.SEC_USER_AGENT_ENV_VAR] = original


class _PoisonHttpClient:
    """A fake transport whose .get() must never be called -- used to prove
    that a refused run makes ZERO requests, not merely "few"."""

    def get(self, url, **kwargs):
        raise AssertionError(f"must never be called -- attempted GET {url}")


def test_run_live_smoke_makes_zero_requests_when_user_agent_unset(monkeypatch):
    """user_agent=None (the production default) resolves from the real
    environment -- hermetically forced empty here via monkeypatch, rather
    than relying on this session's ambient environment being unset."""
    monkeypatch.delenv("IRA_SEC_USER_AGENT", raising=False)
    report = live.run_live_smoke(CIK, user_agent=None, http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0


def test_run_live_smoke_refuses_with_explicit_missing_user_agent():
    report = live.run_live_smoke(CIK, user_agent="", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_run_live_smoke_refuses_with_explicit_placeholder_user_agent():
    """run_live_smoke's user_agent parameter is the ALREADY-RESOLVED value;
    resolve_user_agent() itself (tested above) is what maps the placeholder
    to None -- confirm main()'s actual call path (resolve then pass) yields
    the same zero-request refusal when wired together."""
    resolved = live.resolve_user_agent(env={live.SEC_USER_AGENT_ENV_VAR: live.PLACEHOLDER_SEC_USER_AGENT})
    report = live.run_live_smoke(CIK, user_agent=resolved, http_client=_PoisonHttpClient())
    assert resolved is None
    assert report.refused
    assert report.request_count == 0


def test_no_user_agent_value_ever_appears_in_printed_report():
    report = live.run_live_smoke(CIK, user_agent="", http_client=_PoisonHttpClient())
    printed = live.format_report_for_print(report, secrets=[_REAL_SECRET_LOOKING_AGENT])
    assert _REAL_SECRET_LOOKING_AGENT not in printed
    assert "contact@example.test" not in printed


def test_secret_values_includes_sec_user_agent_for_defense_in_depth(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _REAL_SECRET_LOOKING_AGENT)
    settings = Settings()
    assert _REAL_SECRET_LOOKING_AGENT in settings.secret_values()


# --- explicit user_agent vs. ambient environment (regression) ---------------
#
# Reproduces a real failure: on a machine where IRA_SEC_USER_AGENT is
# genuinely and validly set (e.g. a developer's own Mac), a test that passes
# user_agent=None/""/placeholder EXPLICITLY must still refuse -- it must
# never silently fall back to that valid ambient value. Only OMITTING the
# parameter entirely may resolve from the environment. Every test below
# monkeypatches a valid ambient value on purpose, so these fail loudly if
# the distinction between "omitted" and "explicitly invalid" ever collapses
# again, regardless of what is or isn't set in whatever environment runs
# the suite.
_AMBIENT_VALID_AGENT = "investment-research-agent ambient-mac-contact@example.test"


def test_explicit_none_refuses_even_with_valid_ambient_env(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live.run_live_smoke(CIK, user_agent=None, http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_explicit_placeholder_refuses_even_with_valid_ambient_env(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live.run_live_smoke(CIK, user_agent=live.PLACEHOLDER_SEC_USER_AGENT, http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_explicit_empty_string_refuses_even_with_valid_ambient_env(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live.run_live_smoke(CIK, user_agent="", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_explicit_whitespace_refuses_even_with_valid_ambient_env(monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live.run_live_smoke(CIK, user_agent="   ", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_omitted_user_agent_resolves_from_ambient_env(monkeypatch):
    """Only omitting the parameter entirely -- the shape ``main()`` uses
    after it has already validated the ambient value itself -- may resolve
    from the real environment. Uses ``FakeHttpClient`` (never the real
    transport) to prove the resolved value actually let the run past the
    refusal gate."""
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, http_client=http)
    assert not report.refused
    assert report.request_count > 0


def test_explicit_valid_value_takes_priority_over_ambient_env(monkeypatch):
    """An explicitly-passed valid value must win over a DIFFERENT ambient
    value -- proven by setting the ambient env to the placeholder itself:
    if the explicit value were ever ignored in favour of the ambient one,
    this would refuse instead of proceeding."""
    monkeypatch.setenv("IRA_SEC_USER_AGENT", live.PLACEHOLDER_SEC_USER_AGENT)
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)
    assert not report.refused
    assert report.request_count > 0


def test_no_leak_of_ambient_user_agent_when_explicit_value_refuses(monkeypatch):
    """Even though a real, valid ambient value exists, it must never appear
    in the printed report of a run that was refused because of a
    DIFFERENT, explicitly-passed invalid value."""
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _AMBIENT_VALID_AGENT)
    report = live.run_live_smoke(CIK, user_agent=None, http_client=_PoisonHttpClient())
    printed = live.format_report_for_print(report, secrets=[_AMBIENT_VALID_AGENT, _REAL_SECRET_LOOKING_AGENT])
    assert _AMBIENT_VALID_AGENT not in printed
    assert "ambient-mac-contact@example.test" not in printed


# --- full offline orchestration (FakeHttpClient, real-format fixtures) -----
def test_full_orchestration_against_fake_transport_mirrors_phase3b_fixtures():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    assert not report.refused
    assert report.errors == []
    assert report.request_count > 0
    assert report.primary_document_id is not None
    assert report.exhibit_document_id is not None
    # zero external LLM/AI activity, always
    assert report.external_llm_tokens == 0
    assert report.anthropic_api_calls == 0
    assert report.web_search_calls == 0
    # execution diagnostics agree
    assert report.execution_report is not None
    assert report.execution_report.diagnostics.external_llm_tokens == 0


def test_dedup_and_cache_diagnostics_are_populated():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)
    assert report.execution_report is not None
    assert report.execution_report.diagnostics.duplicate_acquisition_avoided >= 1  # shared directory index.json
    assert report.cache_hit_count >= 0  # never negative; populated from the client


def test_no_form_10k_in_submissions_is_reported_never_silently_acquired():
    responses = dict(fx.default_responses())
    payload = json.loads(fx.fixture_text("submissions_testco.json"))
    payload["filings"]["recent"]["form"] = ["8-K", "S-1", "S-1"]  # no 10-K anywhere
    responses[fx.submissions_url()] = fx.ok(json.dumps(payload))
    http = FakeHttpClient(responses=responses)
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)
    assert not report.refused
    assert report.errors
    assert report.primary_document_id is None


def test_failed_primary_fetch_is_never_reported_as_acquired():
    responses = dict(fx.default_responses())
    responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.not_found("503")
    http = FakeHttpClient(responses=responses)
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)
    assert not report.refused
    assert report.execution_report is not None
    primary_outcome = report.execution_report.outcome_for("target_primary")
    assert primary_outcome is not TargetAcquisitionOutcome.ACQUIRED
    assert report.primary_document_id is None


def test_issuer_authority_and_claim_flags_are_never_altered_by_this_script():
    """This script never constructs a Document/Fact itself -- it only wires
    the SAME Phase 3A/3B adapters that already guarantee authority=
    STATUTORY_FILING/company_claim=True/independent_confirmation=False.
    Confirmed here via the DocumentStore this run actually populated."""
    from investment_research.research.acquisition_executor import AcquisitionExecutor
    from investment_research.research.document_store import DocumentStore
    from investment_research.research.sec_acquisition_adapters import (
        SecExhibitAdapter,
        SecExhibitSelectionRequest,
        SecFilingReference,
        SecPrimaryDocumentAdapter,
    )
    from investment_research.schemas.enums import DocumentAuthority

    http = FakeHttpClient(responses=fx.default_responses())
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={
            "sec_primary_document_adapter": SecPrimaryDocumentAdapter(http),
            "sec_exhibit_enumeration": SecExhibitAdapter(http),
        },
        document_store=store,
    )
    graph = live._build_graph(("target_primary", "target_exhibit"))
    executor.run(
        graph,
        filing_references={
            "target_primary": SecFilingReference(cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT, form="10-K"),
        },
        exhibit_selectors={
            "target_exhibit": SecExhibitSelectionRequest(
                cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT,
                priority_keywords=live.DEFAULT_EXHIBIT_PRIORITY_KEYWORDS,
            ),
        },
    )
    for stored in store.resolve_by_accession(ACCESSION):
        assert stored.document.authority is DocumentAuthority.STATUTORY_FILING
        assert stored.document.is_company_ir is True
        assert stored.document_role in (DocumentRole.PRIMARY_DOCUMENT, DocumentRole.EXHIBIT)


def test_fixture_diff_reports_structural_comparison_not_content():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)
    assert "keys_in_both" in report.fixture_diff
    assert report.fixture_diff["keys_in_both"] > 0


def test_live_verified_candidates_are_diagnostic_only_never_a_code_change():
    """Running this smoke test must never touch source_routing_catalog.py
    or promote any step's ImplementationStatus in code -- confirmed by the
    fact that the real catalog's build function is never even imported by
    this module."""
    import investment_research.research.sec_live_smoke as module

    assert "source_routing_catalog" not in module.__dict__
    assert not hasattr(module, "build_source_routing_graph")


# --- Phase 3C.1: post-live reconciliation regressions ------------------------
#
# Reproduces the real Mac live run's own finding: this script's OWN
# DEFAULT_EXHIBIT_PRIORITY_KEYWORDS (smoke-only, ranks certification/EX-31/
# EX-32 above EX-99/material agreement so the exhibit FETCH/PARSE/
# DocumentStore path gets exercised against a real control filing that only
# carries SOX certifications) picked a CERTIFICATION exhibit -- never a
# production adapter fallback: sec_acquisition_adapters.DEFAULT_EXHIBIT_
# PRIORITY does not include certification keywords at all.
def _without_ex99_and_ex10(directory_json_text: str, filing_detail_html_text: str) -> tuple[str, str]:
    """A variant of Phase 3B's fixture set with NO EX-99/EX-10 candidate
    anywhere in the accession -- proves the "genuinely absent" case, not
    merely "present but outranked by this script's own priority order"."""
    payload = json.loads(directory_json_text)
    payload["directory"]["item"] = [
        item for item in payload["directory"]["item"]
        if "ex99-1" not in item["name"] and "ex10-1" not in item["name"]
    ]
    directory_json_text = json.dumps(payload)
    filing_detail_html_text = re.sub(
        r"<tr>\s*<td>\d+</td>.*?(ex99-1|ex10-1).*?</tr>", "", filing_detail_html_text, flags=re.S,
    )
    return directory_json_text, filing_detail_html_text


def test_ex99_and_ex10_absent_certification_never_satisfies_the_requirement():
    directory_text, detail_text = _without_ex99_and_ex10(
        fx.fixture_text("directory_index_10k.json"), fx.fixture_text("filing_detail_10k.htm"),
    )
    responses = dict(fx.default_responses())
    responses[fx.directory_index_url()] = fx.ok(directory_text)
    responses[fx.filing_detail_url()] = fx.ok(detail_text)
    del responses[fx.document_url("testco-20251231_ex99-1.htm")]
    del responses[fx.document_url("testco-20251231_ex10-1.htm")]
    http = FakeHttpClient(responses=responses)
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    assert not report.refused
    audit = report.exhibit_selection_audit
    assert audit["selected_category"] == "CERTIFICATION"
    assert audit["would_production_default_select_this_category"] is False
    # No EX-99/EX-10 candidate matched even this script's own (broader)
    # priority list either -- confirms genuine absence, not merely a lost
    # priority race (other certifications, e.g. EX-31.2/EX-32.1, are still
    # deferred here since fetch_limit=1, which is expected and fine).
    assert not any(
        "ex99" in name or "ex10-1" in name for name in audit["not_yet_fetched_other_matches"]
    )
    assert not any(c.startswith("target_exhibit:") for c in report.live_verified_candidates)
    assert "target_exhibit:ep" in report.structurally_completed_not_requirement_relevant


def test_smoke_priority_still_selects_certification_when_ex99_and_ex10_are_present():
    """The default fixture set DOES carry EX-99/EX-10 candidates -- this
    reproduces exactly what the real Mac run observed: certification is
    still selected, because THIS script's own priority list (not
    production's) ranks it first."""
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    audit = report.exhibit_selection_audit
    assert audit["selected_category"] == "CERTIFICATION"
    assert audit["would_production_default_select_this_category"] is False
    # EX-99 (and other certifications) genuinely exist in this accession,
    # just outranked by this script's own priority order.
    assert any("ex99" in name for name in audit["not_yet_fetched_other_matches"])


def test_certification_fetch_success_is_distinct_from_target_completion_relevance():
    """A CERTIFICATION exhibit fetching and parsing successfully is a real,
    confirmed transport/parser fact -- but it must never be reported as
    though it satisfies EX-99/EX-10 selection or a real EvidenceRequirement."""
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    by_name = {c["capability"]: c for c in report.capability_confirmations}
    assert by_name["certification body parser"]["confirmed"] is True
    assert by_name["generic exhibit body fetch"]["confirmed"] is True
    assert by_name["EX-99 selection"]["confirmed"] is False
    assert by_name["EX-10 selection"]["confirmed"] is False
    assert by_name["Requirement適合判定 (acquired evidence satisfies a real investment-research EvidenceRequirement)"]["confirmed"] is False


def test_live_verified_candidate_requires_completion_condition_not_just_any_progress():
    """A target whose required steps never all reached their completion
    condition must never appear in live_verified_candidates -- confirmed
    here on the SAME failed-primary-fetch scenario already used to prove
    outcome_for() is not ACQUIRED."""
    responses = dict(fx.default_responses())
    responses[fx.document_url(PRIMARY_DOCUMENT)] = fx.not_found("503")
    http = FakeHttpClient(responses=responses)
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    assert not any(c.startswith("target_primary:") for c in report.live_verified_candidates)
    assert not any(c.startswith("target_primary:") for c in report.structurally_completed_not_requirement_relevant)


def test_physical_request_categories_sum_to_physical_http_requests():
    """Invariant (Phase 3C.1 section 2): the physical request breakdown's
    4 categories must always sum to physical_http_requests exactly -- never
    approximately, never adjusted after the fact."""
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    rb = report.request_breakdown
    assert rb is not None
    assert rb.categorized_total == rb.physical_http_requests
    assert rb.physical_http_requests == len(report.requested_urls)
    # Every physical GET in this graph is also reported by exactly one
    # step's own StepExecutionResult -- the two independent measurements
    # agree, now that acquisition_executor._tally() no longer discards one
    # counter based on the step's acquisition_method.
    assert rb.logical_direct_api_steps + rb.logical_direct_http_steps == rb.physical_http_requests


def test_request_breakdown_categorizes_submissions_directory_detail_and_bodies():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    rb = report.request_breakdown
    assert rb.submissions_requests == 1
    assert rb.directory_index_requests == 1
    assert rb.filing_detail_requests == 1
    assert rb.document_body_requests == 2  # primary body + exhibit body


def test_fixture_comparisons_report_five_artifacts_separately():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    assert set(report.fixture_comparisons.keys()) == {
        "submissions_json", "directory_index_json", "filing_detail_html",
        "primary_ixbrl_document", "exhibit_document",
    }
    allowed_statuses = {"MATCH", "COMPATIBLE_VARIATION", "INCOMPATIBLE", "NOT_OBSERVED"}
    for artifact, comparison in report.fixture_comparisons.items():
        assert comparison["status"] in allowed_statuses, artifact


def test_fixture_comparisons_report_not_observed_when_never_fetched():
    """A refused run (zero network) must report every artifact as
    NOT_OBSERVED, never guess a status for something never fetched."""
    comparisons = live._run_all_artifact_comparisons({})
    assert {c.status for c in comparisons} == {"NOT_OBSERVED"}


def test_document_diagnostics_report_authority_and_is_company_ir():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)

    assert report.document_diagnostics
    for doc in report.document_diagnostics:
        assert doc["authority"] == "STATUTORY_FILING"
        assert doc["is_company_ir"] is True
        assert doc["content_kind"] == "FULL_DOCUMENT"
        assert doc["url"].startswith("https://www.sec.gov/")


def test_printed_report_documents_independent_confirmation_is_a_fact_level_field():
    """Never fabricate a company_claim/independent_confirmation field on a
    Document -- the printed report must say plainly that this script never
    constructs a Fact, so it cannot report either value directly."""
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)
    printed = live.format_report_for_print(report, secrets=[])
    assert "independent_confirmation" in printed
    assert "company_claim" in printed


def test_no_user_agent_leak_in_new_phase3c1_diagnostics_fields():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(CIK, user_agent=_REAL_SECRET_LOOKING_AGENT, http_client=http)
    printed = live.format_report_for_print(report, secrets=[_REAL_SECRET_LOOKING_AGENT])
    assert _REAL_SECRET_LOOKING_AGENT not in printed
    assert "contact@example.test" not in printed


# --- analyze_capture: offline re-analysis of already-saved bodies -----------
# analyze_capture() never accepts an HTTP client at all -- there is no
# injection point for one, so "makes no network access" is true by
# construction rather than something to mock and assert on.
def test_analyze_capture_makes_no_network_access_and_reads_only_local_files(tmp_path):
    submissions_text = fx.fixture_text("submissions_testco.json")
    directory_text = fx.fixture_text("directory_index_10k.json")
    detail_text = fx.fixture_text("filing_detail_10k.htm")
    primary_text = fx.fixture_text("primary_10k_ixbrl.htm")
    exhibit_text = fx.fixture_text("exhibit_31_1_certification.htm")

    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, "testco-20251231_ex31-1.htm")
    for category, text, suffix in (
        ("submissions", submissions_text, ".json"),
        ("directory_index", directory_text, ".json"),
        ("filing_detail", detail_text, ".htm"),
        ("primary_document", primary_text, ".htm"),
        ("exhibit_document", exhibit_text, ".htm"),
    ):
        digest = hashlib.sha256(urls[category].encode()).hexdigest()[:24]
        (tmp_path / f"{digest}{suffix}").write_text(text, encoding="utf-8")

    result = live.analyze_capture(
        tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT,
        exhibit_filename="testco-20251231_ex31-1.htm", exhibit_type="EX-31.1",
    )
    assert result["categories_missing"] == []
    assert set(result["categories_found"]) == {
        "submissions", "directory_index", "filing_detail", "primary_document", "exhibit_document",
    }
    statuses = {c["artifact"]: c["status"] for c in result["comparisons"]}
    assert statuses["submissions_json"] == "MATCH"
    assert statuses["exhibit_document"] in {"MATCH", "COMPATIBLE_VARIATION"}
    assert result["exhibit_category_guess"] == "CERTIFICATION"
    # --- Phase 3D.4 backward compatibility: none of these bodies has a
    # manifest (they were written directly by this test, mirroring a
    # capture from before Phase 3D.4) -- never a crash, never a guessed
    # capture_retrieved_at.
    assert set(result["capture_manifests"]) == set(result["categories_found"])
    for diag in result["capture_manifests"].values():
        assert diag["capture_manifest_status"] == "MISSING"
        assert diag["capture_manifest_schema_version"] is None
        assert diag["capture_manifest_error"] is None
        assert diag["capture_retrieved_at"] == UNKNOWN


# --- Phase 3D.4: Capture Manifest consumption by analyze_capture() ----------
def test_analyze_capture_reports_manifest_capture_retrieved_at_when_hash_verified(tmp_path):
    submissions_text = fx.fixture_text("submissions_testco.json")
    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, None)
    content = submissions_text.encode("utf-8")
    digest = hashlib.sha256(urls["submissions"].encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_bytes(content)
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        source="sec",
        requested_url=urls["submissions"],
        final_url=urls["submissions"],
        http_status=200,
        capture_retrieved_at="2026-08-01T00:00:00Z",
        content_hash=compute_content_hash(content),
        content_length=len(content),
    )
    write_manifest(tmp_path, digest, manifest)

    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    diag = result["capture_manifests"]["submissions"]
    assert diag["capture_manifest_status"] == "VERIFIED"
    assert diag["capture_manifest_schema_version"] == CAPTURE_MANIFEST_SCHEMA_VERSION
    assert diag["capture_manifest_error"] is None
    assert diag["capture_retrieved_at"] == "2026-08-01T00:00:00Z"


def test_analyze_capture_manifest_hash_mismatch_withholds_capture_retrieved_at(tmp_path):
    """A manifest whose content_hash no longer matches the body actually on
    disk -- but whose content_length still happens to agree (a same-size
    corruption) -- must never be treated as a normal, verified capture
    (Phase 3D.4 requirement 9): capture_retrieved_at is withheld (UNKNOWN),
    not reported from an untrustworthy manifest."""
    submissions_text = fx.fixture_text("submissions_testco.json")
    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, None)
    actual_content = submissions_text.encode("utf-8")
    digest = hashlib.sha256(urls["submissions"].encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_bytes(actual_content)
    corrupted_same_length = actual_content[:-1] + (b"X" if actual_content[-1:] != b"X" else b"Y")
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        source="sec",
        requested_url=urls["submissions"],
        final_url=urls["submissions"],
        http_status=200,
        capture_retrieved_at="2026-08-01T00:00:00Z",
        content_hash=compute_content_hash(corrupted_same_length),
        content_length=len(corrupted_same_length),
    )
    write_manifest(tmp_path, digest, manifest)

    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    diag = result["capture_manifests"]["submissions"]
    assert diag["capture_manifest_status"] == "HASH_MISMATCH"
    assert diag["capture_manifest_error"] is not None
    assert diag["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_content_length_mismatch_takes_priority(tmp_path):
    """When BOTH content_length and content_hash disagree, content_length
    is reported (Phase 3D.4.1 requirement 3's explicit priority)."""
    submissions_text = fx.fixture_text("submissions_testco.json")
    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, None)
    actual_content = submissions_text.encode("utf-8")
    digest = hashlib.sha256(urls["submissions"].encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_bytes(actual_content)
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        source="sec",
        requested_url=urls["submissions"],
        final_url=urls["submissions"],
        http_status=200,
        capture_retrieved_at="2026-08-01T00:00:00Z",
        content_hash=compute_content_hash(b"not the real body"),
        content_length=len(b"not the real body"),
    )
    write_manifest(tmp_path, digest, manifest)

    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    diag = result["capture_manifests"]["submissions"]
    assert diag["capture_manifest_status"] == "CONTENT_LENGTH_MISMATCH"
    assert diag["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_malformed_json_is_an_integrity_failure_not_missing(tmp_path):
    """Corrupt manifest JSON must never be treated the same as MISSING
    (Phase 3D.4.1 requirement 6)."""
    submissions_text = fx.fixture_text("submissions_testco.json")
    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, None)
    digest = hashlib.sha256(urls["submissions"].encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(submissions_text, encoding="utf-8")
    (tmp_path / f"{digest}.manifest.json").write_text("{not valid json", encoding="utf-8")

    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    diag = result["capture_manifests"]["submissions"]
    assert diag["capture_manifest_status"] == "MALFORMED"
    assert diag["capture_manifest_status"] != "MISSING"
    assert diag["capture_manifest_error"] is not None
    assert diag["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_unsupported_schema_is_an_integrity_failure(tmp_path):
    submissions_text = fx.fixture_text("submissions_testco.json")
    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, None)
    digest = hashlib.sha256(urls["submissions"].encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(submissions_text, encoding="utf-8")
    (tmp_path / f"{digest}.manifest.json").write_text(
        json.dumps({"totally": "unrecognized", "fields": True}), encoding="utf-8",
    )

    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    diag = result["capture_manifests"]["submissions"]
    assert diag["capture_manifest_status"] == "UNSUPPORTED_SCHEMA"
    assert diag["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_io_error_is_an_integrity_failure(tmp_path, monkeypatch):
    from pathlib import Path as _Path

    submissions_text = fx.fixture_text("submissions_testco.json")
    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, None)
    digest = hashlib.sha256(urls["submissions"].encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(submissions_text, encoding="utf-8")
    manifest_path = tmp_path / f"{digest}.manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    original_read_text = _Path.read_text

    def boom(self, *args, **kwargs):
        if self == manifest_path:
            raise OSError("simulated IO error")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "read_text", boom)

    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    diag = result["capture_manifests"]["submissions"]
    assert diag["capture_manifest_status"] == "IO_ERROR"
    assert diag["capture_retrieved_at"] == UNKNOWN


def test_analyze_capture_manifest_body_missing_is_an_integrity_failure_distinct_from_missing(tmp_path):
    """A manifest exists but its body file does not -- an Evidence
    Integrity failure (Phase 3D.4.1.1 correction: the manifest claims a
    capture was made for an artifact that cannot be found to re-verify or
    analyze, which is unsubstantiated evidence, not a harmless legacy
    case), and explicitly distinct from MISSING (no manifest at all,
    which stays a non-failure legacy capture, Phase 3D.4.1 requirement 5
    unchanged)."""
    urls = live._category_urls(CIK, ACCESSION, PRIMARY_DOCUMENT, None)
    digest = hashlib.sha256(urls["submissions"].encode()).hexdigest()[:24]
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        source="sec",
        requested_url=urls["submissions"],
        final_url=urls["submissions"],
        http_status=200,
        capture_retrieved_at="2026-08-01T00:00:00Z",
        content_hash=compute_content_hash(b"whatever was captured"),
        content_length=len(b"whatever was captured"),
    )
    write_manifest(tmp_path, digest, manifest)
    # No body file written at all for "submissions".

    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    assert "submissions" not in result["categories_found"]
    diag = result["capture_manifests"]["submissions"]
    assert diag["capture_manifest_status"] == "BODY_MISSING"
    assert diag["capture_manifest_status"] != "MISSING"
    assert is_evidence_integrity_failure(ManifestReadStatus.BODY_MISSING)
    assert not is_evidence_integrity_failure(ManifestReadStatus.MISSING)
    assert diag["capture_manifest_error"] is not None
    # error_reason is a static, factual description -- never a secret.
    assert "user_agent" not in diag["capture_manifest_error"].lower()
    assert "@" not in diag["capture_manifest_error"]
    assert diag["capture_retrieved_at"] == UNKNOWN
    # Never treated as VERIFIED/acquisition-complete.
    assert diag["capture_manifest_status"] != "VERIFIED"
    assert "live_verified_candidates" not in result


def test_analyze_capture_manifest_never_included_in_live_verified_candidates(tmp_path):
    """analyze_capture() never sets/returns a live_verified_candidates key
    at all, regardless of manifest status -- offline re-analysis must never
    be conflated with a live run's own LIVE_VERIFIED diagnostic (Phase
    3D.4.1 requirement 5)."""
    result = live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    assert "live_verified_candidates" not in result


def test_analyze_capture_reports_missing_categories_never_guesses(tmp_path):
    result = live.analyze_capture(
        tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT,
    )
    assert set(result["categories_missing"]) == {"submissions", "directory_index", "filing_detail", "primary_document"}
    assert result["categories_found"] == []
    statuses = {c["artifact"]: c["status"] for c in result["comparisons"]}
    assert statuses["submissions_json"] == "NOT_OBSERVED"
    assert statuses["primary_ixbrl_document"] == "NOT_OBSERVED"


def test_analyze_capture_never_touches_the_one_time_marker(tmp_path):
    marker = tmp_path / "marker.json"
    assert not marker.exists()
    live.analyze_capture(tmp_path, cik=CIK, accession=ACCESSION, primary_document=PRIMARY_DOCUMENT)
    assert not marker.exists()


def test_main_analyze_capture_mode_requires_no_user_agent(tmp_path, monkeypatch):
    """--analyze-capture must work even with IRA_SEC_USER_AGENT completely
    unset -- it is a pure local file re-analysis mode, never a live
    acquisition attempt."""
    monkeypatch.delenv("IRA_SEC_USER_AGENT", raising=False)
    exit_code = live.main([
        "--analyze-capture", str(tmp_path), "--accession", ACCESSION, "--primary-document", PRIMARY_DOCUMENT,
    ])
    assert exit_code == 1  # nothing captured in an empty tmp_path -- reported, not a crash


# --- one-time marker guard (main()) -----------------------------------------
def test_main_refuses_without_network_when_user_agent_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("IRA_SEC_USER_AGENT", raising=False)
    marker = tmp_path / "marker.json"
    exit_code = live.main(["--marker-path", str(marker)])
    assert exit_code == 1
    assert not marker.exists()  # a refused, zero-network run never claims the one-time slot


def test_main_marker_guard_blocks_a_second_invocation(tmp_path, monkeypatch):
    monkeypatch.setenv("IRA_SEC_USER_AGENT", _REAL_SECRET_LOOKING_AGENT)
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"attempted_at": "2020-01-01T00:00:00Z", "refused": False}))
    # No http_client injection point in main() -- but the marker check
    # happens BEFORE any client would be constructed, so this call must
    # return before ever touching the network regardless of the (missing)
    # real submissions endpoint.
    exit_code = live.main(["--marker-path", str(marker)])
    assert exit_code == 2


def test_main_force_rerun_bypasses_the_marker(tmp_path, monkeypatch):
    """--force-rerun bypasses the marker guard, but this test must not
    actually reach the network -- confirmed by leaving the user agent unset
    so main() refuses at the very next gate instead."""
    monkeypatch.delenv("IRA_SEC_USER_AGENT", raising=False)
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"attempted_at": "2020-01-01T00:00:00Z", "refused": False}))
    exit_code = live.main(["--marker-path", str(marker), "--force-rerun"])
    assert exit_code == 1  # past the marker gate, refused at the user-agent gate instead


# --- plan is fully printable before any request -----------------------------
def test_build_plan_never_makes_a_request_and_shows_max_six():
    plan = live.build_plan(CIK)
    assert plan.max_requests == live.MAX_LIVE_GETS
    assert live.MAX_LIVE_GETS <= 6
    assert plan.allowed_hosts == tuple(sorted(live.ALLOWED_HOSTS))
    assert set(plan.allowed_hosts) == {"data.sec.gov", "www.sec.gov"}
    printed = live.format_plan_for_print(plan)
    assert "nothing sent yet" in printed


# --- helper graph shape ------------------------------------------------------
def test_build_graph_uses_offline_verified_status_never_a_higher_one():
    graph = live._build_graph(("target_primary", "target_exhibit"))
    assert isinstance(graph, SourceRoutingGraph)
    for step in graph.steps:
        assert step.implementation_status is ImplementationStatus.OFFLINE_VERIFIED
        assert not step.implementation_status.is_live_verified


def test_apple_cik_is_not_present_in_the_production_catalog():
    """Phase 3C requirement 18: the control CIK must never leak into
    research/source_routing_catalog.py's production data."""
    import inspect

    import investment_research.research.source_routing_catalog as catalog_module

    source = inspect.getsource(catalog_module)
    assert str(live.DEFAULT_APPLE_CIK) not in source
