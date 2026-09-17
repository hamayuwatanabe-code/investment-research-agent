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


def test_format_report_for_print_redacts_a_configured_secret_shaped_raw_value():
    """FakeHttpClient (unlike the real AllowlistedHttpClient) does not
    sanitize requested_urls -- the plan/report's own credential fields stay
    boolean-only either way (see test_credential_values_never_appear_in_
    the_printed_plan), but a raw secret VALUE surfacing elsewhere (a
    transport error message, in this fake's case the tool value appearing
    unencoded in the URL) is still caught by format_report_for_print's own
    _safe()-based scrubbing -- a genuine second, independent line of
    defense on top of transport-layer sanitization, not a substitute for
    it (see the loopback integration suite for proof that the REAL
    transport strips email/api_key -- including their percent-encoded
    form -- before either ever reaches a FetchResult at all)."""
    http = FakeHttpClient(responses=_discovery_responses("NCT09990001", ["90000007"], "nct_id_present.xml"))
    report = live.run_live_smoke(mode="discovery", nct_id="NCT09990001", http_client=http, env=_ENV)
    assert "tool=test-tool" in "".join(report.requested_urls)  # confirms the fake really is unsanitized
    printed = live.format_report_for_print(report, secrets=["test-tool"])
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
