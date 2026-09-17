"""Phase 3F.1: literature_live_smoke.py exercised entirely offline.

No real network call anywhere in this file (enforced by
``_network_guard.forbid_external_network_autouse``). Orchestration tests
inject a ``FakeHttpClient`` (reusing ``_literature_fixture_support``'s
real-format fixtures) in place of ``AllowlistedHttpClient`` -- exactly like
``test_clinicaltrials_live_smoke_offline.py``'s own precedent. Real-socket
tests (secret non-leakage over the wire, disallowed-redirect hit counts,
Capture Manifest write-then-replay) live in
``tests/integration/test_literature_live_smoke_transport.py`` instead,
since ``AllowlistedHttpClient``/``FakeHttpClient`` are the only two
transport types this suite ever uses (Phase 3F.1 requirement 8).
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

from investment_research.research import literature_live_smoke as live
from investment_research.research.capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    CaptureManifest,
    ManifestReadStatus,
    compute_content_hash,
    is_evidence_integrity_failure,
    write_manifest,
)
from investment_research.research.literature_acquisition_adapter import (
    MAX_PMIDS_PER_BATCH,
    NCBI_EFETCH_URL,
    NCBI_ESEARCH_URL,
)
from investment_research.schemas.enums import FetchOutcome

from . import _literature_fixture_support as fx
from ._literature_fixture_support import FakeHttpClient, RaisingHttpClient
from ._network_guard import forbid_external_network_autouse  # noqa: F401

_ENV = {"IRA_NCBI_TOOL": "test-tool", "IRA_NCBI_EMAIL": "test@example.test"}
_ENV_WITH_KEY = {**_ENV, "IRA_NCBI_API_KEY": "test-api-key-999"}


class _PoisonHttpClient:
    def get(self, url, **kwargs):
        raise AssertionError(f"must never be called -- attempted GET {url}")


def _esearch_wire_url(term: str, retmax: int = MAX_PMIDS_PER_BATCH, env: dict[str, str] = _ENV) -> str:
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(retmax)}
    for key, param in (("IRA_NCBI_TOOL", "tool"), ("IRA_NCBI_EMAIL", "email"), ("IRA_NCBI_API_KEY", "api_key")):
        if env.get(key):
            params[param] = env[key]
    return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"


def _efetch_wire_url(pmids: list[str], env: dict[str, str] = _ENV) -> str:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    for key, param in (("IRA_NCBI_TOOL", "tool"), ("IRA_NCBI_EMAIL", "email"), ("IRA_NCBI_API_KEY", "api_key")):
        if env.get(key):
            params[param] = env[key]
    return f"{NCBI_EFETCH_URL}?{urlencode(params)}"


def _epmc_search_url(pmid: str) -> str:
    return fx.europepmc_search_url(pmid)


def _epmc_fulltext_url(pmcid: str) -> str:
    return fx.europepmc_fulltext_url(pmcid)


def _discovery_responses(nct_id: str, pmids: list[str], efetch_fixture: str, *, env=_ENV) -> dict:
    return {
        _esearch_wire_url(f"{nct_id}[si]", env=env): fx.ok(fx.esearch_response(pmids)),
        _efetch_wire_url(pmids, env=env): fx.ok(fx.fixture_text(efetch_fixture)),
    }


# --- credential gating (Phase 3F.1 requirement 3) ---------------------------
def test_resolve_ncbi_credentials_all_missing_reports_tool_and_email():
    credentials, status, missing = live.resolve_ncbi_credentials(env={})
    assert missing == [live.NCBI_TOOL_ENV_VAR, live.NCBI_EMAIL_ENV_VAR]
    assert status.tool_configured is False
    assert status.email_configured is False
    assert status.api_key_configured is False


def test_resolve_ncbi_credentials_api_key_never_required():
    _credentials, status, missing = live.resolve_ncbi_credentials(env=_ENV)
    assert missing == []
    assert status.api_key_configured is False  # not configured, but never in `missing`


def test_resolve_ncbi_credentials_api_key_configured_when_set():
    _credentials, status, missing = live.resolve_ncbi_credentials(env=_ENV_WITH_KEY)
    assert missing == []
    assert status.api_key_configured is True


def test_run_live_smoke_refuses_when_tool_missing_zero_requests():
    env = {"IRA_NCBI_EMAIL": "test@example.test"}
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990001", http_client=_PoisonHttpClient(), env=env)
    assert report.refused
    assert "IRA_NCBI_TOOL" in report.refused_reason
    assert report.request_count == 0


def test_run_live_smoke_refuses_when_email_missing_zero_requests():
    env = {"IRA_NCBI_TOOL": "test-tool"}
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=_PoisonHttpClient(), env=env)
    assert report.refused
    assert "IRA_NCBI_EMAIL" in report.refused_reason
    assert report.request_count == 0


def test_run_live_smoke_does_not_refuse_solely_for_missing_api_key():
    http = FakeHttpClient(responses=_discovery_responses("NCT09990001", ["90000007"], "nct_id_present.xml"))
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990001", http_client=http, env=_ENV)
    assert not report.refused


def test_credential_values_never_appear_in_the_printed_plan():
    _credentials, status, _missing = live.resolve_ncbi_credentials(env=_ENV_WITH_KEY)
    plan = live.build_plan(
        mode="discovery", nct_id="NCT09990001", pmid=None, max_articles=3, max_fulltext_fetches=1, status=status,
    )
    printed = live.format_plan_for_print(plan)
    assert "test-tool" not in printed
    assert "test@example.test" not in printed
    assert "test-api-key-999" not in printed
    assert "ncbi_tool_configured: True" in printed
    assert "ncbi_email_configured: True" in printed
    assert "ncbi_api_key_configured: True" in printed


def test_collect_transport_diagnostics_scrubs_all_credentials_even_from_an_unsanitized_fake():
    """_collect_transport_diagnostics applies BOTH _strip_ncbi_tool_param
    (param-name-based) AND _scrub_credentials (value-based, raw and
    percent-encoded) to every URL it writes into the report -- regardless
    of transport. This holds even against a FakeHttpClient that (unlike
    the real AllowlistedHttpClient) never sanitizes requested_urls at all
    (Phase 3F.1 correction 2 requirement 2: the report itself must be safe,
    not merely whatever a well-behaved transport happens to hand it)."""
    http = FakeHttpClient(responses=_discovery_responses("NCT09990001", ["90000007"], "nct_id_present.xml"))
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990001", http_client=http, env=_ENV)
    joined = "".join(report.requested_urls)
    assert "tool=" not in joined
    assert "test-tool" not in joined
    assert "test@example.test" not in joined
    assert "email=test%40example.test" not in joined
    assert "test%40example.test" not in joined


def test_adapter_level_errors_never_contain_a_raw_email_regardless_of_transport():
    """literature_acquisition_adapter.py's own _sanitize_text (Phase 3F.0.2)
    redacts email/api_key (raw AND percent-encoded) from every failure
    message it constructs, using the SAME NcbiCredentials object passed as
    `settings` -- this holds even for a RaisingHttpClient that never
    sanitizes anything on its own, since the sanitization here happens at
    the adapter layer, not the transport layer. `tool` is deliberately
    NOT included in that adapter-level redaction (matching
    config.Settings.secret_values()'s own exclusion) -- see the next test
    for how THIS module's stricter no-tool-value policy is enforced at
    print time instead."""
    http = RaisingHttpClient()
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    assert report.errors
    assert "test@example.test" not in " ".join(report.errors)


def test_report_errors_never_contain_the_tool_value_from_an_adapter_error_message():
    """A RaisingHttpClient's exception message embeds the adapter's own
    public_url substitution, which (unlike literature_acquisition_adapter.py's
    own email/api_key-only _sanitize_text) still includes `tool=<value>`
    coming straight from the transport layer. This must now be caught at
    the REPORT level -- report.errors is scrubbed at the point of
    insertion in run_live_smoke() -- rather than relying solely on
    format_report_for_print's print-time _safe() call (Phase 3F.1
    correction 2 requirement 2: the report object itself must never carry
    the raw value, not merely whatever gets printed from it)."""
    http = RaisingHttpClient()
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    assert report.errors
    assert "tool=test-tool" not in " ".join(report.errors)
    assert "test-tool" not in " ".join(report.errors)
    printed = live.format_report_for_print(report, secrets=["test-tool", "test@example.test"])
    assert "test-tool" not in printed


def test_format_report_for_print_never_needed_for_the_real_transport():
    """Documents the actual production boundary: run_live_smoke's default
    (no injected http_client) always constructs a real AllowlistedHttpClient,
    whose .requested_urls are already sanitized BEFORE format_report_for_print
    ever sees them -- the _safe() pass above is defense in depth, not the
    primary control."""
    import inspect

    source = inspect.getsource(live.run_live_smoke)
    assert "AllowlistedHttpClient" in source


def test_env_parameter_overrides_ambient_environment(monkeypatch):
    """An explicit env={} passed to resolve_ncbi_credentials/run_live_smoke
    must never fall through to a real ambient IRA_NCBI_* the test-runner's
    own machine happens to have set."""
    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, "ambient-tool-value")
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, "ambient@example.test")
    _credentials, _status, missing = live.resolve_ncbi_credentials(env={})
    assert missing == [live.NCBI_TOOL_ENV_VAR, live.NCBI_EMAIL_ENV_VAR]


def test_main_refuses_without_nct_id_or_pmid():
    assert live.main([]) == 1


def test_main_refuses_when_both_nct_id_and_pmid_given():
    assert live.main(["--nct-id", "NCT09990001", "--pmid", "90000001"]) == 1


def test_main_never_writes_marker_on_missing_credentials(tmp_path, monkeypatch):
    for var in (live.NCBI_TOOL_ENV_VAR, live.NCBI_EMAIL_ENV_VAR, live.NCBI_API_KEY_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    marker = tmp_path / "marker.json"
    exit_code = live.main(["--nct-id", "NCT09990001", "--marker-path", str(marker)])
    assert exit_code == 2
    assert not marker.exists()


def test_main_never_writes_marker_on_malformed_nct_id(tmp_path, monkeypatch):
    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, "test-tool")
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, "test@example.test")
    marker = tmp_path / "marker.json"
    exit_code = live.main(["--nct-id", "garbage", "--marker-path", str(marker)])
    assert exit_code == 1
    assert not marker.exists()


# --- ID validation: zero network on malformed input -------------------------
def test_discovery_refuses_malformed_nct_id_with_zero_requests():
    report = live.run_live_smoke(mode="discovery", nct_id="not-an-nct-id", http_client=_PoisonHttpClient(), env=_ENV)
    assert report.refused
    assert report.request_count == 0
    assert report.requested_urls == []


def test_targeted_refuses_malformed_pmid_with_zero_requests():
    report = live.run_live_smoke(mode="targeted", pmid="not-a-pmid", http_client=_PoisonHttpClient(), env=_ENV)
    assert report.refused
    assert report.request_count == 0


def test_run_live_smoke_rejects_unknown_mode():
    import pytest

    with pytest.raises(ValueError):
        live.run_live_smoke(mode="bogus", env=_ENV)


# --- discovery mode: NCT-ID search, full orchestration ----------------------
def test_discovery_full_orchestration_against_fake_transport():
    http = FakeHttpClient(
        responses={
            **_discovery_responses("NCT09990001", ["90000008"], "ids_complete.xml"),
            _epmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            _epmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report = live.run_live_smoke(
        mode="discovery", nct_id="NCT09990001", max_articles=3, max_fulltext_fetches=1, http_client=http, env=_ENV,
    )
    assert not report.refused
    assert report.errors == []
    assert report.pmids_located == ["90000008"]
    assert len(report.articles) == 1
    article = report.articles[0]
    assert article["pmid"] == "90000008"
    assert article["publication_stage"] == "JOURNAL_ARTICLE"
    assert article["peer_review_status"] == "UNKNOWN"
    assert article["europepmc_is_open_access"] is True
    assert article["full_text_acquired"] is True
    assert article["document_authority"] == "BIOMEDICAL_LITERATURE"
    assert report.coverage_complete is True
    # zero external LLM/AI activity, always
    assert report.anthropic_api_calls == 0
    assert report.web_search_calls == 0
    assert report.external_llm_tokens == 0
    assert report.execution_report is not None
    assert report.execution_report.diagnostics.external_llm_tokens == 0


def test_discovery_matches_requested_nct_id_true():
    http = FakeHttpClient(
        responses=_discovery_responses("NCT09990001", ["90000007"], "nct_id_present.xml")
        | {_epmc_search_url("90000007"): fx.not_found()}
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990001", http_client=http, env=_ENV)
    assert report.articles[0]["nct_ids"] == ["NCT09990001"]
    assert report.articles[0]["matches_requested_nct_id"] is True


def test_discovery_matches_requested_nct_id_false_when_article_carries_no_match():
    http = FakeHttpClient(
        responses=_discovery_responses("NCT09990002", ["90000001"], "normal_abstract.xml")
        | {_epmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json"))}
    )
    # NCT09990002 is a different (but still validly-shaped) NCT id than the
    # one embedded in the ESearch term below -- what matters is the
    # article itself (normal_abstract.xml) carries no NCT id at all.
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990002", http_client=http, env=_ENV)
    assert report.articles[0]["nct_ids"] == []
    assert report.articles[0]["matches_requested_nct_id"] is False


def test_discovery_zero_results_reported_never_as_a_crash():
    http = FakeHttpClient(responses={_esearch_wire_url("NCT09990099[si]"): fx.ok(fx.esearch_response([]))})
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990099", http_client=http, env=_ENV)
    assert not report.refused
    assert report.pmids_located == []
    assert report.errors  # ZERO_RESULTS surfaced as an error entry, never silently empty


# --- targeted mode: single PMID, no ESearch ----------------------------------
def test_targeted_full_orchestration_skips_esearch_entirely():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            _epmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            _epmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000008", http_client=http, env=_ENV)
    assert not report.refused
    assert not any("esearch.fcgi" in url for url in report.requested_urls)
    assert report.articles[0]["pmid"] == "90000008"
    assert report.articles[0]["full_text_acquired"] is True


def test_targeted_fulltext_cap_enforced_even_with_a_larger_flag():
    """Requesting --max-fulltext-fetches larger than the targeted-mode hard
    cap must still be clamped (Phase 3F.1 requirement 5)."""
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            _epmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            _epmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report = live.run_live_smoke(
        mode="targeted", pmid="90000008", max_fulltext_fetches=99, http_client=http, env=_ENV,
    )
    assert report.plan.max_fulltext_fetches == live.TARGETED_MAX_FULLTEXT_FETCHES_CAP
    assert not report.refused


def test_targeted_smaller_requested_fulltext_cap_is_still_honored():
    http = FakeHttpClient(responses={_efetch_wire_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report = live.run_live_smoke(
        mode="targeted", pmid="90000001", max_fulltext_fetches=0, http_client=http, env=_ENV,
    )
    assert report.plan.max_fulltext_fetches == 0


# --- batch EFetch (never one PMID at a time) --------------------------------
def test_batch_efetch_fetches_multiple_pmids_in_exactly_one_call():
    http = FakeHttpClient(
        responses={
            _esearch_wire_url("NCT09990003[si]"): fx.ok(fx.esearch_response(["90000015", "90000016"])),
            _efetch_wire_url(["90000015", "90000016"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
            _epmc_search_url("90000015"): fx.not_found(),
            _epmc_search_url("90000016"): fx.not_found(),
        }
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990003", max_articles=3, http_client=http, env=_ENV)
    assert not report.refused
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1  # ONE batched call, never one PMID at a time
    assert {a["pmid"] for a in report.articles} == {"90000015", "90000016"}


# --- OA / non-OA / preprint --------------------------------------------------
def test_open_access_article_full_text_is_acquired():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            _epmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            _epmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000008", http_client=http, env=_ENV)
    a = report.articles[0]
    assert a["europepmc_is_open_access"] is True
    assert a["europepmc_in_epmc"] is True
    assert a["full_text_acquired"] is True
    assert a["document_content_kind"] in ("FULL_DOCUMENT", "EXCERPT")


def test_non_open_access_article_full_text_never_acquired():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            _epmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    a = report.articles[0]
    assert a["europepmc_is_open_access"] is False
    assert a["full_text_acquired"] is False


def test_preprint_marks_not_peer_reviewed_and_never_treated_as_confirmed():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000022"]): fx.ok(fx.fixture_text("preprint.xml")),
            _epmc_search_url("90000022"): fx.ok(fx.fixture_text("europepmc_search_preprint.json")),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000022", http_client=http, env=_ENV)
    a = report.articles[0]
    assert a["publication_stage"] == "PREPRINT"
    assert a["peer_review_status"] == "NOT_PEER_REVIEWED"
    printed = live.format_report_for_print(report, secrets=[])
    assert "never means peer-reviewed" in printed.lower() or "never peer-reviewed" in printed.lower()


# --- malformed XML/JSON -------------------------------------------------------
def test_malformed_efetch_xml_reported_as_error_never_a_crash():
    http = FakeHttpClient(
        responses={
            _esearch_wire_url("NCT09990004[si]"): fx.ok(fx.esearch_response(["90000012"])),
            _efetch_wire_url(["90000012"]): fx.ok(fx.fixture_text("malformed.xml")),
        }
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990004", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors
    assert report.articles == []


def test_malformed_esearch_json_reported_as_error_never_a_crash():
    http = FakeHttpClient(responses={_esearch_wire_url("NCT09990005[si]"): fx.ok("{not valid json at all")})
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990005", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors
    assert report.pmids_located == []


def test_html_error_page_instead_of_xml_reported_as_error():
    http = FakeHttpClient(
        responses={
            _esearch_wire_url("NCT09990006[si]"): fx.ok(fx.esearch_response(["90000013"])),
            _efetch_wire_url(["90000013"]): fx.ok(fx.fixture_text("html_error_page.htm")),
        }
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990006", http_client=http, env=_ENV)
    assert report.errors
    assert report.articles == []


# --- 404 / 429 / 5xx / timeout -----------------------------------------------
def test_esearch_404_reported_never_a_crash():
    http = FakeHttpClient(responses={_esearch_wire_url("NCT09990007[si]"): fx.not_found("404")})
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990007", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors


def test_esearch_429_reported_never_a_crash():
    http = FakeHttpClient(
        responses={_esearch_wire_url("NCT09990008[si]"): fx.failed(FetchOutcome.RATE_LIMITED, "429")}
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990008", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors
    assert any("429" in e or "RATE_LIMITED" in e for e in report.errors)


def test_esearch_5xx_reported_never_a_crash():
    http = FakeHttpClient(
        responses={_esearch_wire_url("NCT09990009[si]"): fx.failed(FetchOutcome.ERROR, "500")}
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990009", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors


def test_esearch_timeout_reported_never_a_crash():
    http = FakeHttpClient(
        responses={_esearch_wire_url("NCT09990010[si]"): fx.failed(FetchOutcome.TIMEOUT, "timed out")}
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990010", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors


def test_transport_raising_an_exception_never_crashes_and_never_leaks_the_url():
    """RaisingHttpClient's .get() raises instead of returning -- proves
    run_live_smoke degrades to a reported error, never an uncaught
    exception, exactly as literature_acquisition_adapter.py's own
    _safe_get already guarantees at a lower layer."""
    http = RaisingHttpClient()
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990011", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors


# --- request cap / fulltext cap / coverage_complete (never NOT_FOUND) -------
def test_article_budget_exceeded_reports_coverage_incomplete_never_not_found():
    http = FakeHttpClient(
        responses={
            _esearch_wire_url("NCT09990012[si]"): fx.ok(fx.esearch_response(["90000015", "90000016"])),
            _efetch_wire_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
            _epmc_search_url("90000015"): fx.not_found(),
        }
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990012", max_articles=1, http_client=http, env=_ENV)
    assert not report.refused
    assert report.coverage_complete is False
    assert not any(e.startswith("l1: StepStatus.NOT_FOUND") for e in report.errors)
    assert not any(e.startswith("f: StepStatus.NOT_FOUND") for e in report.errors)


#: A minimal, synthetic 2-article PubmedArticleSet -- mirrors
#: ids_complete.xml's own real-format shape exactly (see that fixture),
#: just with TWO articles, each carrying its own PMCID -- needed because
#: no existing fixture has two PMCID-bearing articles in one batch (the
#: existing batch_articleset.xml deliberately carries no PMCID at all).
#: Never a real PMID/PMCID/journal name -- same fictional-identifier
#: convention as every fixture in tests/fixtures/literature_real_format/.
_TWO_OA_ARTICLES_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
<MedlineCitation Status="MEDLINE" Owner="NLM">
<PMID Version="1">90000030</PMID>
<Article PubModel="Print-Electronic">
<Journal><Title>Fictional Journal Batch OA A</Title></Journal>
<ArticleTitle>Fictional batch OA article A</ArticleTitle>
<PublicationTypeList><PublicationType UI="D016428">Journal Article</PublicationType></PublicationTypeList>
</Article>
</MedlineCitation>
<PubmedData>
<ArticleIdList>
<ArticleId IdType="pubmed">90000030</ArticleId>
<ArticleId IdType="pmc">PMC9990030</ArticleId>
</ArticleIdList>
</PubmedData>
</PubmedArticle>
<PubmedArticle>
<MedlineCitation Status="MEDLINE" Owner="NLM">
<PMID Version="1">90000031</PMID>
<Article PubModel="Print-Electronic">
<Journal><Title>Fictional Journal Batch OA B</Title></Journal>
<ArticleTitle>Fictional batch OA article B</ArticleTitle>
<PublicationTypeList><PublicationType UI="D016428">Journal Article</PublicationType></PublicationTypeList>
</Article>
</MedlineCitation>
<PubmedData>
<ArticleIdList>
<ArticleId IdType="pubmed">90000031</ArticleId>
<ArticleId IdType="pmc">PMC9990031</ArticleId>
</ArticleIdList>
</PubmedData>
</PubmedArticle>
</PubmedArticleSet>"""


def _oa_search_response(pmid: str, pmcid: str) -> str:
    return json.dumps(
        {"resultList": {"result": [{"pmid": pmid, "pmcid": pmcid, "isOpenAccess": "Y", "inEPMC": "Y", "source": "MED"}]}}
    )


def test_fulltext_budget_skip_reported_never_not_found():
    """Two OA PMIDs but max_fulltext_fetches=1 -- the SECOND full-text fetch
    is budget-skipped, reported via budget_skipped_fulltext_pmcids and
    coverage_complete=False, never as a 404/NOT_FOUND."""
    http = FakeHttpClient(
        responses={
            _esearch_wire_url("NCT09990013[si]"): fx.ok(fx.esearch_response(["90000030", "90000031"])),
            _efetch_wire_url(["90000030", "90000031"]): fx.ok(_TWO_OA_ARTICLES_XML),
            _epmc_search_url("90000030"): fx.ok(_oa_search_response("90000030", "PMC9990030")),
            _epmc_search_url("90000031"): fx.ok(_oa_search_response("90000031", "PMC9990031")),
            fx.europepmc_fulltext_url("PMC9990030"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
            fx.europepmc_fulltext_url("PMC9990031"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report = live.run_live_smoke(
        mode="discovery", nct_id="NCT09990013", max_articles=3, max_fulltext_fetches=1, http_client=http, env=_ENV,
    )
    assert not report.refused
    assert report.coverage_complete is False
    assert report.budget_skipped_fulltext_pmcids == ["PMC9990031"]
    assert not any("NOT_FOUND" in e for e in report.errors)
    full_text_flags = {a["pmid"]: a["full_text_acquired"] for a in report.articles}
    assert full_text_flags["90000030"] is True
    assert full_text_flags["90000031"] is False
    assert report.budget_skipped_fulltext_pmcids
    assert not any("NOT_FOUND" in e for e in report.errors)


def test_max_requests_cap_matches_the_printed_formula():
    plan = live.build_plan(
        mode="discovery", nct_id="NCT09990001", pmid=None, max_articles=3, max_fulltext_fetches=1,
        status=live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False),
    )
    assert plan.max_requests == 1 + 1 + 3 + 1
    assert str(plan.max_requests) in plan.max_requests_formula


def test_no_url_is_requested_more_than_once_within_a_single_run():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            _epmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            _epmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    live.run_live_smoke(mode="targeted", pmid="90000008", http_client=http, env=_ENV)
    assert len(http.requested_urls) == len(set(http.requested_urls))


# --- output: never prints abstract/full-text body ----------------------------
def test_printed_report_never_contains_abstract_or_fulltext_body_text():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000002"]): fx.ok(fx.fixture_text("structured_abstract.xml")),
            _epmc_search_url("90000002"): fx.not_found(),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000002", http_client=http, env=_ENV)
    printed = live.format_report_for_print(report, secrets=[])
    # structured_abstract.xml's own BACKGROUND/METHODS/RESULTS body text
    # must never appear -- only structural metadata is ever printed.
    raw_body = fx.fixture_text("structured_abstract.xml")
    for marker in ("BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"):
        if f">{marker}<" in raw_body:
            continue  # label-only occurrences aren't body text
    assert "abstract_sections" not in printed
    jsonable = live.report_to_jsonable(report)
    assert "abstract_sections" not in json.dumps(jsonable)


def test_json_output_is_curated_never_a_raw_dataclass_asdict_of_execution_report():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            _epmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            _epmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000008", http_client=http, env=_ENV)
    jsonable = live.report_to_jsonable(report)
    dumped = json.dumps(jsonable, default=str)
    assert "execution_report" not in jsonable
    # The full-text section body must never leak into the json output either.
    for section_marker in ("<sec>", "<title>", "<p>"):
        assert section_marker not in dumped


def test_report_to_jsonable_round_trips_through_json_dumps():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            _epmc_search_url("90000001"): fx.not_found(),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    json.dumps(live.report_to_jsonable(report), default=str)  # must not raise


# --- retraction / correction -------------------------------------------------
def test_retracted_article_still_generates_diagnostics_never_hidden():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000011"]): fx.ok(fx.fixture_text("retracted.xml")),
            _epmc_search_url("90000011"): fx.not_found(),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000011", http_client=http, env=_ENV)
    assert report.articles[0]["retracted"] is True


# --- one-time marker guard (main()) -----------------------------------------
def test_main_marker_guard_blocks_a_second_invocation(tmp_path, monkeypatch):
    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, "test-tool")
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, "test@example.test")
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"attempted_at": "2020-01-01T00:00:00Z", "refused": False}))
    exit_code = live.main(["--nct-id", "NCT09990001", "--marker-path", str(marker)])
    assert exit_code == 2


def test_marker_isolation_across_different_marker_paths(tmp_path, monkeypatch):
    """A LAST_RUN marker recorded at one path must never block a run using
    a DIFFERENT --marker-path -- marker state is scoped strictly to the
    exact path given, never ambient/global."""
    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, "test-tool")
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, "test@example.test")
    other_marker = tmp_path / "other_marker.json"
    other_marker.write_text(json.dumps({"attempted_at": "2020-01-01T00:00:00Z", "refused": False}))
    this_marker = tmp_path / "this_marker.json"
    assert live._marker_state(this_marker) is None
    assert live._marker_state(other_marker) is not None


# --- analyze_capture: entirely offline, zero credentials required ----------
def _write_capture(
    tmp_path: Path, *, requested_url: str, final_url: str, body: bytes, source: str = "literature",
) -> str:
    import hashlib

    digest = hashlib.sha256(requested_url.encode()).hexdigest()[:24]
    (tmp_path / f"{digest}.bin").write_bytes(body)
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION, source=source, requested_url=requested_url,
        final_url=final_url, http_status=200, capture_retrieved_at="2026-08-01T00:00:00Z",
        content_hash=compute_content_hash(body), content_length=len(body),
    )
    write_manifest(tmp_path, digest, manifest)
    return digest


def test_analyze_capture_makes_zero_network_calls_and_requires_no_credentials(tmp_path, monkeypatch):
    for var in (live.NCBI_TOOL_ENV_VAR, live.NCBI_EMAIL_ENV_VAR, live.NCBI_API_KEY_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    esearch_public_url = f"{NCBI_ESEARCH_URL}?db=pubmed&term=NCT09990001%5Bsi%5D&retmode=json&retmax=3"
    body = fx.esearch_response(["90000008"]).encode("utf-8")
    _write_capture(tmp_path, requested_url=esearch_public_url, final_url=esearch_public_url, body=body)

    result = live.analyze_capture(tmp_path)  # no exception -- no network attempted, no env needed
    assert result["found"] is True
    assert result["esearch"]["pmids_found"] == ["90000008"]


def test_analyze_capture_reports_not_found_on_an_empty_directory(tmp_path):
    result = live.analyze_capture(tmp_path)
    assert result["found"] is False
    assert result["manifest_count"] == 0


def test_analyze_capture_reports_not_found_on_a_missing_directory():
    result = live.analyze_capture(Path("/nonexistent/definitely/not/here"))
    assert result["found"] is False
    assert "error" in result


def test_analyze_capture_classifies_esearch_efetch_europepmc_search_and_fulltext(tmp_path):
    esearch_url = f"{NCBI_ESEARCH_URL}?db=pubmed&term=NCT09990001%5Bsi%5D&retmode=json&retmax=3"
    efetch_url = f"{NCBI_EFETCH_URL}?db=pubmed&id=90000008&retmode=xml&rettype=abstract"
    epmc_search_url_ = fx.europepmc_search_url("90000008")
    epmc_fulltext_url_ = fx.europepmc_fulltext_url("PMC9990008")

    _write_capture(tmp_path, requested_url=esearch_url, final_url=esearch_url, body=fx.esearch_response(["90000008"]).encode())
    _write_capture(tmp_path, requested_url=efetch_url, final_url=efetch_url, body=fx.fixture_text("ids_complete.xml").encode())
    _write_capture(
        tmp_path, requested_url=epmc_search_url_, final_url=epmc_search_url_,
        body=fx.fixture_text("europepmc_search_oa.json").encode(),
    )
    _write_capture(
        tmp_path, requested_url=epmc_fulltext_url_, final_url=epmc_fulltext_url_,
        body=fx.fixture_text("europepmc_fulltext_oa.xml").encode(),
    )

    result = live.analyze_capture(tmp_path)
    assert result["found"] is True
    assert result["esearch"]["pmids_found"] == ["90000008"]
    assert len(result["efetch"]) == 1
    assert result["efetch"][0]["articles"][0]["pmid"] == "90000008"
    assert len(result["europepmc_search"]) == 1
    assert result["europepmc_search"][0]["results"][0]["is_open_access"] is True
    assert len(result["europepmc_fulltext"]) == 1
    assert result["europepmc_fulltext"][0]["section_count"] > 0
    # The full-text SECTION TEXT itself must never appear in the result.
    assert "section_text" not in json.dumps(result)
    assert "sections" not in result["europepmc_fulltext"][0]


def test_analyze_capture_malformed_efetch_body_is_reported_never_a_crash(tmp_path):
    efetch_url = f"{NCBI_EFETCH_URL}?db=pubmed&id=90000012&retmode=xml&rettype=abstract"
    _write_capture(tmp_path, requested_url=efetch_url, final_url=efetch_url, body=fx.fixture_text("malformed.xml").encode())
    result = live.analyze_capture(tmp_path)
    assert result["efetch"][0]["parsed_ok"] is False
    assert result["efetch"][0]["shape_error"] is not None


def test_analyze_capture_manifest_hash_mismatch_is_an_evidence_integrity_failure(tmp_path):
    esearch_url = f"{NCBI_ESEARCH_URL}?db=pubmed&term=NCT09990001%5Bsi%5D&retmode=json&retmax=3"
    digest = _write_capture(
        tmp_path, requested_url=esearch_url, final_url=esearch_url, body=fx.esearch_response(["90000008"]).encode(),
    )
    # Corrupt the body on disk AFTER the manifest was written, same length.
    body_path = tmp_path / f"{digest}.bin"
    original = body_path.read_bytes()
    corrupted = original[:-1] + (b"X" if original[-1:] != b"X" else b"Y")
    body_path.write_bytes(corrupted)

    result = live.analyze_capture(tmp_path)
    statuses = {f["status"] for f in result["evidence_integrity_failures"]}
    assert "HASH_MISMATCH" in statuses
    assert is_evidence_integrity_failure(ManifestReadStatus.HASH_MISMATCH)


def test_analyze_capture_manifest_missing_is_not_an_evidence_integrity_failure(tmp_path):
    """A body saved with no manifest at all is unremarkable -- it is simply
    never classified/parsed (it cannot be, since classification depends on
    the manifest's own requested_url), and MUST NOT be reported as an
    Evidence Integrity failure."""
    (tmp_path / "deadbeefdeadbeefdeadbeef.bin").write_bytes(b"orphan body, no manifest")
    result = live.analyze_capture(tmp_path)
    assert result["evidence_integrity_failures"] == []
    assert result["found"] is False


def test_analyze_capture_manifest_body_missing_is_an_evidence_integrity_failure(tmp_path):
    esearch_url = f"{NCBI_ESEARCH_URL}?db=pubmed&term=NCT09990001%5Bsi%5D&retmode=json&retmax=3"
    import hashlib

    digest = hashlib.sha256(esearch_url.encode()).hexdigest()[:24]
    body = fx.esearch_response(["90000008"]).encode()
    manifest = CaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION, source="literature", requested_url=esearch_url,
        final_url=esearch_url, http_status=200, capture_retrieved_at="2026-08-01T00:00:00Z",
        content_hash=compute_content_hash(body), content_length=len(body),
    )
    write_manifest(tmp_path, digest, manifest)
    # No body file written at all.

    result = live.analyze_capture(tmp_path)
    statuses = {f["status"] for f in result["evidence_integrity_failures"]}
    assert "BODY_MISSING" in statuses
    assert is_evidence_integrity_failure(ManifestReadStatus.BODY_MISSING)


def test_analyze_capture_never_touches_the_one_time_marker(tmp_path):
    marker = tmp_path / "marker.json"
    live.analyze_capture(tmp_path)
    assert not marker.exists()


def test_main_analyze_capture_mode_requires_no_marker_or_network(tmp_path):
    exit_code = live.main(["--analyze-capture", str(tmp_path)])
    assert exit_code == 1  # nothing captured in an empty tmp_path -- reported, not a crash


def test_analyze_capture_signature_carries_no_http_client_or_credentials_parameter():
    """Structural guarantee, mirroring clinicaltrials_live_smoke.py's own
    precedent: there is no live-transport or credential code path
    analyze_capture could reach even in error."""
    import inspect

    params = inspect.signature(live.analyze_capture).parameters
    assert "http_client" not in params
    assert "env" not in params


# --- Pipeline non-connection / catalog non-modification ---------------------
def test_module_never_imports_pipeline_or_source_routing_catalog():
    assert "pipeline" not in live.__dict__
    assert "source_routing_catalog" not in live.__dict__
    assert not hasattr(live, "build_source_routing_graph")


def test_live_verified_candidates_are_diagnostic_only_never_promoted():
    """This phase never promotes anything to LIVE_VERIFIED -- confirmed
    structurally by the fact this module's own internal graph
    (_build_graph) marks every step OFFLINE_VERIFIED, never LIVE_VERIFIED,
    and never touches the production catalog at all."""
    from investment_research.research.source_routing import ImplementationStatus

    graph = live._build_graph()
    assert all(step.implementation_status is ImplementationStatus.OFFLINE_VERIFIED for step in graph.steps)
    assert not any(step.implementation_status is ImplementationStatus.LIVE_VERIFIED for step in graph.steps)


def test_no_id_is_hardcoded_as_a_default_anywhere():
    import inspect

    sig = inspect.signature(live.run_live_smoke)
    assert sig.parameters["nct_id"].default is None
    assert sig.parameters["pmid"].default is None


# --- Phase 3F.1 correction requirement 4: CLI communication boundary -------
def test_plan_text_states_live_mode_will_perform_real_communication():
    status = live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False)
    plan = live.build_plan(
        mode="discovery", nct_id="NCT09990001", pmid=None, max_articles=3, max_fulltext_fetches=1, status=status,
    )
    printed = live.format_plan_for_print(plan)
    assert "nothing sent yet" in printed
    assert "OFFLINE this phase" not in printed
    assert "LIVE mode" in printed
    assert "will make real" in printed.lower() or "will perform real" in printed.lower()


def test_module_docstring_never_claims_live_mode_itself_is_offline():
    assert "OFFLINE this phase" not in (live.__doc__ or "")
    assert "LIVE mode" in (live.__doc__ or "")


def test_cli_help_states_live_mode_will_perform_real_communication(capsys):
    import pytest

    with pytest.raises(SystemExit):
        live.main(["--help"])
    # argparse wraps the description across lines -- normalize before
    # substring-checking so wrap points never cause a false failure.
    printed = capsys.readouterr().out.replace("\n", " ")
    assert "LIVE" in printed
    assert "WILL make real" in printed
    assert "network requests" in printed


def test_analyze_capture_help_still_says_fully_offline(capsys):
    import pytest

    with pytest.raises(SystemExit):
        live.main(["--help"])
    printed = capsys.readouterr().out.replace("\n", " ")
    assert "no credentials" in printed.lower() or "no network" in printed.lower()


# --- Phase 3F.1 correction requirement 5: marker exception-safety ----------
def test_pre_send_refusal_never_calls_the_transport_and_never_completes():
    report = live.run_live_smoke(mode="discovery", nct_id="not-an-nct-id", http_client=_PoisonHttpClient(), env=_ENV)
    assert report.refused
    assert report.smoke_run_completed is False


def test_sent_then_failed_is_never_a_refusal():
    http = RaisingHttpClient()
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    assert not report.refused
    assert report.smoke_run_completed is True
    assert report.status is live.LiveSmokeStatus.FAILED
    assert report.errors


def test_main_never_writes_marker_when_run_live_smoke_raises_before_any_attempt(tmp_path, monkeypatch):
    """Phase 3F.1 correction 2 requirement 1: main() no longer wraps
    run_live_smoke() in a try/except that writes the marker on ANY
    exception. If run_live_smoke() itself never got a chance to invoke the
    on_first_attempt callback (e.g. it raised immediately, before even
    constructing a transport), no marker is written -- the OLD behavior
    (a marker written for any exception, even one before any real
    communication) was exactly the bug this correction fixes."""
    import pytest

    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, "test-tool")
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, "test@example.test")
    marker = tmp_path / "marker.json"

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure before any attempt")

    monkeypatch.setattr(live, "run_live_smoke", _boom)
    with pytest.raises(RuntimeError):
        live.main(["--nct-id", "NCT09990001", "--marker-path", str(marker)])
    assert not marker.exists()


def test_main_writes_marker_via_on_first_attempt_callback_before_any_get_call(tmp_path, monkeypatch):
    """The marker is written by main()'s own _mark_first_attempt callback,
    which run_live_smoke() threads into the client as on_first_attempt --
    proven here by monkeypatching run_live_smoke() to invoke the callback
    it was given (standing in for "a real attempt started") without ever
    touching a transport, and confirming the marker exists afterward."""
    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, "test-tool")
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, "test@example.test")
    marker = tmp_path / "marker.json"

    def _fake_run_live_smoke(*, on_first_attempt=None, **kwargs):
        assert on_first_attempt is not None
        on_first_attempt()
        return live.LiveSmokeReport(
            plan=live.build_plan(
                mode="discovery", nct_id="NCT09990001", pmid=None, max_articles=3, max_fulltext_fetches=1,
                status=live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False),
            ),
        )

    monkeypatch.setattr(live, "run_live_smoke", _fake_run_live_smoke)
    exit_code = live.main(["--nct-id", "NCT09990001", "--marker-path", str(marker)])
    assert exit_code == 0
    assert marker.is_file()
    state = live._marker_state(marker)
    assert state is not None
    assert state["refused"] is False


def test_main_raises_and_writes_no_marker_when_the_marker_write_itself_fails(tmp_path, monkeypatch):
    """Phase 3F.1 correction 2 requirement 1: if writing the marker itself
    fails, HTTP communication must never begin. Simulated here by pointing
    --marker-path at a location _write_marker cannot create (a path
    component that is a FILE, not a directory), and confirming main()
    propagates the failure rather than swallowing it."""
    import pytest

    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, "test-tool")
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, "test@example.test")
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("occupied")
    marker = blocking_file / "marker.json"
    calls: list[str] = []

    def _fake_run_live_smoke(*, on_first_attempt=None, **kwargs):
        assert on_first_attempt is not None
        on_first_attempt()  # raises NotADirectoryError -- must propagate uncaught
        calls.append("never reached")
        return live.LiveSmokeReport(plan=live.build_plan(
            mode="discovery", nct_id="NCT09990001", pmid=None, max_articles=3, max_fulltext_fetches=1,
            status=live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False),
        ))

    monkeypatch.setattr(live, "run_live_smoke", _fake_run_live_smoke)
    with pytest.raises((NotADirectoryError, OSError)):
        live.main(["--nct-id", "NCT09990001", "--marker-path", str(marker)])
    assert calls == []
    assert not marker.exists()


def test_main_still_never_writes_marker_on_missing_credentials_even_with_the_new_try_except(tmp_path, monkeypatch):
    for var in (live.NCBI_TOOL_ENV_VAR, live.NCBI_EMAIL_ENV_VAR, live.NCBI_API_KEY_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    marker = tmp_path / "marker.json"
    exit_code = live.main(["--nct-id", "NCT09990001", "--marker-path", str(marker)])
    assert exit_code == 2
    assert not marker.exists()


# --- Phase 3F.1 correction requirement 6: output contract -------------------
def test_report_status_completed_on_a_clean_run():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            _epmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            _epmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000008", http_client=http, env=_ENV)
    assert report.status is live.LiveSmokeStatus.COMPLETED
    assert report.smoke_run_completed is True


def test_report_status_failed_when_a_transport_exception_occurs():
    http = RaisingHttpClient()
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    assert report.status is live.LiveSmokeStatus.FAILED


def test_report_status_refused_when_pre_send_rejected():
    report = live.run_live_smoke(mode="discovery", nct_id="garbage", http_client=_PoisonHttpClient(), env=_ENV)
    assert report.status is live.LiveSmokeStatus.REFUSED


def test_report_status_rate_limited_when_a_429_occurs():
    http = FakeHttpClient(responses={_esearch_wire_url("NCT09990021[si]"): fx.failed(FetchOutcome.RATE_LIMITED, "429")})
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990021", http_client=http, env=_ENV)
    assert report.status is live.LiveSmokeStatus.RATE_LIMITED


def _blank_plan() -> live.LiveSmokePlan:
    return live.build_plan(
        mode="targeted", nct_id=None, pmid="90000001", max_articles=3, max_fulltext_fetches=1,
        status=live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False),
    )


def test_status_is_never_misclassified_by_an_error_message_that_merely_mentions_blocked():
    """Phase 3F.1 correction 2 requirement 3: status is derived from
    STRUCTURED transport_outcomes, never from pattern-matching errors'
    free text. A generic failure whose message happens to contain the word
    "BLOCKED" (for a reason unrelated to any actual host-block/redirect-
    refusal/egress-denial) must be reported as FAILED, not BLOCKED --
    report.transport_outcomes is empty here, so there is no structured
    evidence of an actual BLOCKED outcome."""
    report = live.LiveSmokeReport(
        plan=_blank_plan(),
        errors=["f: StepStatus.FAILED -- transport raised BLOCKED somehow"],
    )
    assert report.transport_outcomes == []
    assert report.status is live.LiveSmokeStatus.FAILED


def test_status_priority_blocked_over_rate_limited_over_failed():
    """When transport_outcomes carries BOTH BLOCKED and RATE_LIMITED (a run
    that hit one of each), BLOCKED -- the more severe classification --
    wins, per LiveSmokeStatus's own documented priority order."""
    report = live.LiveSmokeReport(
        plan=_blank_plan(),
        transport_outcomes=[FetchOutcome.RATE_LIMITED.value, FetchOutcome.BLOCKED.value, FetchOutcome.OK.value],
        errors=["some step failed"],
    )
    assert report.status is live.LiveSmokeStatus.BLOCKED


def test_status_rate_limited_wins_over_generic_failed():
    report = live.LiveSmokeReport(
        plan=_blank_plan(),
        transport_outcomes=[FetchOutcome.RATE_LIMITED.value, FetchOutcome.OK.value],
        errors=["some step failed"],
    )
    assert report.status is live.LiveSmokeStatus.RATE_LIMITED


def test_status_refused_wins_over_everything_even_a_blocked_transport_outcome():
    report = live.LiveSmokeReport(
        plan=_blank_plan(),
        refused_reason="missing required environment variable(s): IRA_NCBI_TOOL",
        transport_outcomes=[FetchOutcome.BLOCKED.value],
    )
    assert report.status is live.LiveSmokeStatus.REFUSED


def test_output_contract_fields_all_present_in_text_and_json():
    http = FakeHttpClient(
        responses={
            _efetch_wire_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            _epmc_search_url("90000001"): fx.not_found(),
        }
    )
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    printed = live.format_report_for_print(report, secrets=[])
    for label in (
        "status:", "smoke_run_completed:", "coverage_complete:", "logical_request_count:", "physical_attempt_count:",
    ):
        assert label in printed, f"missing {label!r} in printed report"
    jsonable = live.report_to_jsonable(report)
    for key in (
        "status", "smoke_run_completed", "coverage_complete", "logical_request_count", "physical_attempt_count",
    ):
        assert key in jsonable, f"missing {key!r} in report_to_jsonable output"
    assert jsonable["status"] == report.status.value
    assert isinstance(jsonable["status"], str)


def test_budget_skip_never_reported_as_not_found_status():
    """A budget-induced skip must never surface as StepStatus.NOT_FOUND --
    exercised here via the article-budget-exceeded scenario (already
    proven not to emit a NOT_FOUND error string elsewhere); this test
    additionally confirms the overall status still reflects COMPLETED
    (never FAILED) when the only "error-shaped" entries are budget skips
    reported through coverage_complete, not report.errors."""
    http = FakeHttpClient(
        responses={
            _esearch_wire_url("NCT09990022[si]"): fx.ok(fx.esearch_response(["90000015", "90000016"])),
            _efetch_wire_url(["90000015"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
            _epmc_search_url("90000015"): fx.not_found(),
        }
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990022", max_articles=1, http_client=http, env=_ENV)
    assert report.coverage_complete is False
    assert not any("NOT_FOUND" in e for e in report.errors)


# --- Phase 3F.1 correction requirement 7: targeted-mode plan consistency ---
def test_targeted_mode_build_plan_clamps_the_field_itself_not_just_the_formula():
    status = live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False)
    plan = live.build_plan(
        mode="targeted", nct_id=None, pmid="90000001", max_articles=3, max_fulltext_fetches=99, status=status,
    )
    assert plan.max_fulltext_fetches == live.TARGETED_MAX_FULLTEXT_FETCHES_CAP
    assert str(live.TARGETED_MAX_FULLTEXT_FETCHES_CAP) in plan.max_requests_formula


def test_displayed_plan_matches_the_plan_run_live_smoke_actually_executes():
    """The plan main() prints BEFORE running and the plan attached to the
    final report (built inside run_live_smoke) must show the SAME clamped
    max_fulltext_fetches -- this was the exact bug the Phase 3F.1
    correction fixes: main() used to build its own upfront plan from the
    raw, un-clamped CLI value while run_live_smoke built a separately-
    clamped one."""
    status = live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False)
    upfront_plan = live.build_plan(
        mode="targeted", nct_id=None, pmid="90000001", max_articles=3, max_fulltext_fetches=99, status=status,
    )
    http = FakeHttpClient(responses={_efetch_wire_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report = live.run_live_smoke(mode="targeted", pmid="90000001", max_fulltext_fetches=99, http_client=http, env=_ENV)
    assert upfront_plan.max_fulltext_fetches == report.plan.max_fulltext_fetches
    assert report.plan.max_fulltext_fetches == live.TARGETED_MAX_FULLTEXT_FETCHES_CAP


def test_discovery_mode_never_clamps_max_fulltext_fetches():
    status = live.CredentialStatus(tool_configured=True, email_configured=True, api_key_configured=False)
    plan = live.build_plan(
        mode="discovery", nct_id="NCT09990001", pmid=None, max_articles=3, max_fulltext_fetches=99, status=status,
    )
    assert plan.max_fulltext_fetches == 99


# --- Phase 3F.1 correction 2 requirement 1: on_first_attempt marker timing --
def test_on_first_attempt_never_fires_when_no_get_call_is_ever_made():
    """A refused run (missing credential/malformed ID) never constructs a
    client at all, so a caller-supplied on_first_attempt is never invoked --
    the callback is the ONLY thing that would write a marker in main(), and
    a refusal must never trigger it."""
    calls: list[str] = []
    report = live.run_live_smoke(
        mode="discovery", nct_id="not-an-nct-id", http_client=_PoisonHttpClient(), env=_ENV,
        on_first_attempt=lambda: calls.append("fired"),
    )
    assert report.refused
    assert calls == []


def test_on_first_attempt_fires_exactly_once_before_the_first_get_call():
    """FakeHttpClient.on_first_attempt (mirroring AllowlistedHttpClient's
    own contract) fires exactly once, before its FIRST .get() call --
    proven here via a FakeHttpClient constructed directly with the
    callback (the pattern an offline test uses in place of a real socket)."""
    calls: list[str] = []
    http = FakeHttpClient(
        responses=_discovery_responses("NCT09990001", ["90000007"], "nct_id_present.xml"),
        on_first_attempt=lambda: calls.append("fired"),
    )
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990001", http_client=http, env=_ENV)
    assert not report.refused
    assert calls == ["fired"]
    assert http.requested_urls  # at least one real .get() call happened


def test_on_first_attempt_fires_even_when_the_transport_then_fails():
    """A RaisingHttpClient's on_first_attempt fires BEFORE it raises --
    proving the marker-writing signal survives a subsequent transport
    failure (Phase 3F.1 correction 2 requirement 1: "sent, then failed"
    must still preserve the marker)."""
    calls: list[str] = []
    http = RaisingHttpClient(on_first_attempt=lambda: calls.append("fired"))
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    assert not report.refused
    assert report.errors
    assert calls == ["fired"]


def test_on_first_attempt_raising_prevents_any_get_call_from_completing():
    """If the callback itself raises (standing in for "the marker write
    failed"), no request may complete. literature_acquisition_adapter.py's
    own ``_safe_get`` catches ANY exception a transport's ``.get()`` raises
    (so a raw wire_url embedded in a transport exception can never
    propagate outward unsanitized) -- so this surfaces as a normal,
    reported FAILED step, never an uncaught crash. What actually proves
    "no HTTP communication began" is that FakeHttpClient's own
    requested_urls stays empty: the callback runs (and raises) BEFORE
    FakeHttpClient.get() ever appends the URL or looks up a response."""
    def _boom() -> None:
        raise OSError("simulated marker write failure")

    http = FakeHttpClient(responses={}, on_first_attempt=_boom)
    report = live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=_ENV)
    assert http.requested_urls == []
    assert report.errors
    assert any("simulated marker write failure" in e for e in report.errors)


def test_run_live_smoke_threads_on_first_attempt_into_the_default_client(monkeypatch):
    """run_live_smoke(), when NOT given an http_client, threads
    on_first_attempt straight into the internally-constructed
    AllowlistedHttpClient -- this is what lets main() tie its marker to
    the real transport's first physical attempt without injecting a fake
    one (Phase 3F.1 correction 2 requirement 1)."""
    captured: dict[str, object] = {}
    real_client_cls = live.AllowlistedHttpClient

    class _RecordingClient(real_client_cls):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            captured["on_first_attempt"] = kwargs.get("on_first_attempt")
            super().__init__(*args, **kwargs)

        def get(self, url, **kwargs):
            raise ConnectionError("no real network in this test")

    monkeypatch.setattr(live, "AllowlistedHttpClient", _RecordingClient)
    sentinel = lambda: None  # noqa: E731
    report = live.run_live_smoke(mode="targeted", pmid="90000001", env=_ENV, on_first_attempt=sentinel)
    assert captured["on_first_attempt"] is sentinel
    assert not report.refused  # the transport failure is reported, not raised, past this point


# --- Phase 3F.1 correction 2 requirement 2: report-level secret sweep ------
def _report_with_every_secret_leak_vector(*, credentials_env: dict[str, str]) -> live.LiveSmokeReport:
    """Drives a real orchestration (LOCATE -> FETCH, with FETCH failing)
    through an UNSANITIZED RaisingHttpClient, whose exception message
    embeds the tool value raw (see
    test_report_errors_never_contain_the_tool_value_from_an_adapter_error_message)
    -- the worst case for every secret-sweep assertion below."""
    http = RaisingHttpClient()
    return live.run_live_smoke(mode="targeted", pmid="90000001", http_client=http, env=credentials_env)


def _all_secret_forms(env: dict[str, str]) -> list[str]:
    from urllib.parse import quote as _quote

    forms: list[str] = []
    for key in ("IRA_NCBI_TOOL", "IRA_NCBI_EMAIL", "IRA_NCBI_API_KEY"):
        value = env.get(key)
        if value:
            forms.append(value)
            forms.append(_quote(value, safe=""))
    return forms


def test_report_object_itself_carries_no_secret_after_an_unsanitized_transport_failure():
    report = _report_with_every_secret_leak_vector(credentials_env=_ENV_WITH_KEY)
    haystacks = [
        " ".join(report.requested_urls),
        " ".join(str(v) for v in report.http_statuses),
        " ".join(report.errors),
        report.refused_reason or "",
    ]
    for secret in _all_secret_forms(_ENV_WITH_KEY):
        for haystack in haystacks:
            assert secret not in haystack, f"{secret!r} leaked into {haystack!r}"


def test_report_to_jsonable_standalone_carries_no_secret_bypassing_main():
    """report_to_jsonable(report), dumped directly via json.dumps -- i.e.
    bypassing main()'s own defense-in-depth _safe() call entirely -- must
    already be free of every secret value, raw or percent-encoded (Phase
    3F.1 correction 2 requirement 2)."""
    report = _report_with_every_secret_leak_vector(credentials_env=_ENV_WITH_KEY)
    dumped = json.dumps(live.report_to_jsonable(report), default=str)
    for secret in _all_secret_forms(_ENV_WITH_KEY):
        assert secret not in dumped, f"{secret!r} leaked into report_to_jsonable() output"


def test_repr_of_report_carries_no_secret():
    """repr(report) must never expose a credential value -- in particular,
    it must never recurse into report.execution_report's own nested
    StepExecutionResult.failure_reason/payload fields, which this module
    does not scrub in place (LiveSmokeReport.execution_report is declared
    repr=False for exactly this reason)."""
    report = _report_with_every_secret_leak_vector(credentials_env=_ENV_WITH_KEY)
    rendered = repr(report)
    assert "execution_report" not in rendered  # excluded from repr entirely, not merely scrubbed
    for secret in _all_secret_forms(_ENV_WITH_KEY):
        assert secret not in rendered, f"{secret!r} leaked into repr(report)"


def test_format_report_for_print_carries_no_secret_text_output():
    report = _report_with_every_secret_leak_vector(credentials_env=_ENV_WITH_KEY)
    secrets = [v for v in (_ENV_WITH_KEY["IRA_NCBI_TOOL"], _ENV_WITH_KEY["IRA_NCBI_EMAIL"], _ENV_WITH_KEY["IRA_NCBI_API_KEY"]) if v]
    printed = live.format_report_for_print(report, secrets=secrets)
    for secret in _all_secret_forms(_ENV_WITH_KEY):
        assert secret not in printed, f"{secret!r} leaked into format_report_for_print() output"


def test_json_output_via_main_carries_no_secret(tmp_path, monkeypatch, capsys):
    """Drives the REAL main()/run_live_smoke() end-to-end, with the
    internally-constructed client replaced by an unsanitized
    RaisingHttpClient (standing in for "the transport does no sanitization
    of its own") -- proving the --json output stays free of every secret
    value purely because of the report-level scrub, not because of
    whatever the real AllowlistedHttpClient would have done anyway."""
    monkeypatch.setenv(live.NCBI_TOOL_ENV_VAR, _ENV_WITH_KEY["IRA_NCBI_TOOL"])
    monkeypatch.setenv(live.NCBI_EMAIL_ENV_VAR, _ENV_WITH_KEY["IRA_NCBI_EMAIL"])
    monkeypatch.setenv(live.NCBI_API_KEY_ENV_VAR, _ENV_WITH_KEY["IRA_NCBI_API_KEY"])
    marker = tmp_path / "marker.json"

    def _unsanitized_client_factory(*args, **kwargs):
        return RaisingHttpClient(on_first_attempt=kwargs.get("on_first_attempt"))

    monkeypatch.setattr(live, "AllowlistedHttpClient", _unsanitized_client_factory)
    live.main(["--pmid", "90000001", "--marker-path", str(marker), "--json"])
    printed = capsys.readouterr().out
    for secret in _all_secret_forms(_ENV_WITH_KEY):
        assert secret not in printed, f"{secret!r} leaked into --json stdout"
    assert marker.is_file()  # on_first_attempt still fired via the real client construction path
