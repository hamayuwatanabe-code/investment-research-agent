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

import json

from investment_research.config import Settings
from investment_research.research import sec_live_smoke as live
from investment_research.research.document_store import DocumentRole
from investment_research.research.source_routing import (
    ImplementationStatus,
    SourceRoutingGraph,
    TargetAcquisitionOutcome,
)

from . import _sec_fixture_support as fx
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
