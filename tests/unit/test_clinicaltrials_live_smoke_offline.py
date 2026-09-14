"""Phase 3D: clinicaltrials_live_smoke.py exercised entirely offline.

No real network call anywhere in this file. The orchestration tests inject
a ``FakeHttpClient`` (reusing ``_clinicaltrials_fixture_support``'s
real-API-v2-format fixtures) in place of ``AllowlistedHttpClient``.
"""

from __future__ import annotations

import json

from investment_research.research import clinicaltrials_live_smoke as live

from . import _clinicaltrials_fixture_support as fx
from ._clinicaltrials_fixture_support import FakeHttpClient

_REAL_LOOKING_AGENT = "clinicaltrials-live-smoke-test-agent/9.9"


class _PoisonHttpClient:
    def get(self, url, **kwargs):
        raise AssertionError(f"must never be called -- attempted GET {url}")


# --- User-Agent: optional, never a refusal gate, never logged --------------
def test_resolve_user_agent_returns_default_when_unset():
    assert live.resolve_user_agent(env={}) == live.DEFAULT_CT_USER_AGENT


def test_resolve_user_agent_returns_configured_value_when_set():
    assert live.resolve_user_agent(env={live.CT_USER_AGENT_ENV_VAR: _REAL_LOOKING_AGENT}) == _REAL_LOOKING_AGENT


# --- NCT ID validation: zero network on malformed input ---------------------
def test_run_live_smoke_refuses_malformed_nct_id_with_zero_requests():
    report = live.run_live_smoke("not-an-nct-id", http_client=_PoisonHttpClient())
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_main_refuses_without_nct_id():
    assert live.main([]) == 1


# --- full offline orchestration (FakeHttpClient, real-format fixtures) -----
def test_full_orchestration_against_fake_transport():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)

    assert not report.refused
    assert report.errors == []
    assert report.request_count == 1
    assert report.document_id is not None
    assert report.parsed_fields.get("status") == "RECRUITING"
    assert report.document_diagnostics["authority"] == "REGISTRY"
    assert report.document_diagnostics["is_company_ir"] is True
    # zero external LLM/AI activity, always
    assert report.external_llm_tokens == 0
    assert report.anthropic_api_calls == 0
    assert report.web_search_calls == 0
    assert report.execution_report is not None
    assert report.execution_report.diagnostics.external_llm_tokens == 0
    assert report.live_verified_candidates  # structurally completed this run (diagnostic only)


def test_404_is_reported_never_acquired():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID_404): fx.not_found()})
    report = live.run_live_smoke(fx.NCT_ID_404, http_client=http)
    assert not report.refused
    assert report.errors
    assert report.document_id is None
    assert report.live_verified_candidates == []


def test_malformed_json_is_reported_never_acquired():
    http = FakeHttpClient(responses={fx.study_url(fx.NCT_ID_MALFORMED): fx.ok(fx.fixture_text("study_malformed.txt"))})
    report = live.run_live_smoke(fx.NCT_ID_MALFORMED, http_client=http)
    assert not report.refused
    assert report.errors
    assert report.document_id is None


def test_fixture_diff_reports_structural_comparison():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)
    assert "keys_in_both" in report.fixture_diff
    assert report.fixture_diff["keys_in_both"] > 0


def test_no_user_agent_value_ever_appears_in_printed_report():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, user_agent=_REAL_LOOKING_AGENT, http_client=http)
    printed = live.format_report_for_print(report, secrets=[_REAL_LOOKING_AGENT])
    assert _REAL_LOOKING_AGENT not in printed


def test_printed_report_documents_independent_confirmation_is_a_fact_level_field():
    http = FakeHttpClient(responses=fx.default_responses())
    report = live.run_live_smoke(fx.NCT_ID, http_client=http)
    printed = live.format_report_for_print(report, secrets=[])
    assert "independent_confirmation" in printed
    assert "company_claim" in printed


# --- one-time marker guard (main()) -----------------------------------------
def test_main_marker_guard_blocks_a_second_invocation(tmp_path):
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"attempted_at": "2020-01-01T00:00:00Z", "refused": False}))
    exit_code = live.main(["--nct-id", fx.NCT_ID, "--marker-path", str(marker)])
    assert exit_code == 2


def test_main_never_writes_marker_on_malformed_nct_id(tmp_path):
    marker = tmp_path / "marker.json"
    exit_code = live.main(["--nct-id", "garbage", "--marker-path", str(marker)])
    assert exit_code == 1
    assert not marker.exists()


# --- analyze_capture: offline re-analysis of already-saved bodies -----------
def test_analyze_capture_reads_saved_body_offline(tmp_path):
    import hashlib

    url = fx.study_url(fx.NCT_ID)
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.json").write_text(fx.fixture_text("study_recruiting_interventional.json"), encoding="utf-8")

    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID)
    assert result["found"] is True
    assert result["schema_ok"] is True
    assert result["parsed_fields"]["status"] == "RECRUITING"
    assert result["fixture_diff"]["keys_in_both"] > 0


def test_analyze_capture_reports_not_found_never_guesses(tmp_path):
    result = live.analyze_capture(tmp_path, nct_id=fx.NCT_ID)
    assert result["found"] is False


def test_analyze_capture_never_touches_the_one_time_marker(tmp_path):
    marker = tmp_path / "marker.json"
    live.analyze_capture(tmp_path, nct_id=fx.NCT_ID)
    assert not marker.exists()


def test_main_analyze_capture_mode_requires_no_marker_or_network(tmp_path):
    exit_code = live.main(["--nct-id", fx.NCT_ID, "--analyze-capture", str(tmp_path)])
    assert exit_code == 1  # nothing captured in an empty tmp_path -- reported, not a crash


# --- transport wiring: allowlisted to clinicaltrials.gov only --------------
def test_allowed_hosts_is_scoped_to_clinicaltrials_gov_only():
    assert frozenset({"clinicaltrials.gov"}) == live.ALLOWED_HOSTS


def test_build_plan_shows_max_two_requests_never_makes_one():
    plan = live.build_plan(fx.NCT_ID)
    assert plan.max_requests == live.MAX_LIVE_GETS
    assert live.MAX_LIVE_GETS <= 2
    printed = live.format_plan_for_print(plan)
    assert "nothing sent yet" in printed


def test_no_nct_id_is_hardcoded_as_a_default_anywhere():
    """An NCT ID may appear in --help example text (documentation), but
    never as a default value/constant this module would use if the caller
    didn't supply one -- run_live_smoke/build_plan/main all require an
    explicit nct_id with no fallback."""
    import inspect

    assert "nct_id" in inspect.signature(live.run_live_smoke).parameters
    assert inspect.signature(live.run_live_smoke).parameters["nct_id"].default is inspect.Parameter.empty
    assert inspect.signature(live.build_plan).parameters["nct_id"].default is inspect.Parameter.empty
    assert live.main([]) == 1  # no --nct-id supplied -> refuses, never guesses one
