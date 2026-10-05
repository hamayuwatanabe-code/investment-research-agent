"""Phase 4.3H: ``research/adaptive_literature_acquisition.py``'s offline
bridge from a Phase 4.3G ``AdaptiveAcquisitionPlan`` to the existing
Literature Document-First acquisition path.

Offline end-to-end, driven against the SAME real-format fixtures/
FakeHttpClient double every other literature test in this repository
uses (``tests/unit/_literature_fixture_support.py``) -- no real NCBI/
Europe PMC communication, no socket of any kind, anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlencode

import pytest

import investment_research.research.adaptive_literature_acquisition as ala
from investment_research.research.adaptive_literature_acquisition import (
    AdaptiveExecutionStatus,
    execute_adaptive_literature_plan,
)
from investment_research.research.literature_acquisition_adapter import (
    NCBI_EFETCH_URL,
    NCBI_ESEARCH_URL,
)
from investment_research.research.literature_pipeline_integration import (
    LiteraturePipelineBundle,
    LiteratureReferenceMode,
)
from investment_research.schemas.enums import FetchOutcome
from investment_research.scoring.adaptive_acquisition_plan import (
    READY_MAX_REQUESTS,
    REFERENCE_MODE_NCT_ID,
    AcquisitionPlanStatus,
    AdaptiveAcquisitionPlan,
)

from . import _literature_fixture_support as fx
from ._network_guard import forbid_external_network_autouse  # noqa: F401

NCT_ID = "NCT09990777"
PMIDS = ["90000015", "90000016"]
TICKER = "P43H"
_ENV = {"IRA_NCBI_TOOL": "test-tool", "IRA_NCBI_EMAIL": "test@example.com"}
_EPMC_EMPTY = '{"resultList": {"result": []}}'


def _efetch_url(pmids: list[str], *, tool: str = "test-tool", email: str = "test@example.com") -> str:
    """Mirrors ``tests/unit/test_literature_pipeline_integration.py``'s own
    local helper exactly: ``run_literature_pipeline_acquisition`` always
    constructs ``PubMedLiteratureAdapter`` with REAL ``NcbiCredentials``,
    so the wire URL genuinely carries tool/email -- unlike ``_literature_
    fixture_support.efetch_url``, which omits them for adapter tests
    constructed differently."""
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email
    return f"{NCBI_EFETCH_URL}?{urlencode(params)}"


def _esearch_url(
    term: str, max_pmids: int, *, tool: str = "test-tool", email: str = "test@example.com"
) -> str:
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids)}
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email
    return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"


def _ready_plan(**overrides) -> AdaptiveAcquisitionPlan:
    kwargs = {
        "status": AcquisitionPlanStatus.READY,
        "reference_mode": REFERENCE_MODE_NCT_ID,
        "nct_id": NCT_ID,
        "pmid": "UNKNOWN",
        "rationale": "ready for test",
        "max_requests": READY_MAX_REQUESTS,
        "requires_external_communication": True,
    }
    kwargs.update(overrides)
    return AdaptiveAcquisitionPlan(**kwargs)


def _success_fake_http() -> fx.FakeHttpClient:
    """NCT discovery mode, two abstract-bearing PMIDs via one batched
    EFetch, both with an empty (non-open-access) Europe PMC search --
    mirrors ``test_nct_discovery_success_via_pubmed_batch_efetch`` in
    ``tests/unit/test_literature_pipeline_integration.py`` exactly."""
    term = f"{NCT_ID}[si]"
    return fx.FakeHttpClient(
        responses={
            _esearch_url(term, 20): fx.ok(fx.esearch_response(PMIDS)),
            _efetch_url(PMIDS): fx.ok(fx.fixture_text("batch_articleset.xml")),
            fx.europepmc_search_url(PMIDS[0]): fx.ok(_EPMC_EMPTY),
            fx.europepmc_search_url(PMIDS[1]): fx.ok(_EPMC_EMPTY),
        }
    )


# =============================================================================
# 1/2. READY -> real ESearch/EFetch/Europe PMC -> Projection -> COMPLETE
# =============================================================================
def test_ready_plan_reaches_full_offline_acquisition_and_completes():
    http = _success_fake_http()
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=http, env=_ENV,
    )

    assert execution.status is AdaptiveExecutionStatus.COMPLETE
    assert execution.bundle is not None
    bundle = execution.bundle
    assert bundle.refused_reason is None
    assert bundle.coverage_complete is True
    assert bundle.raw_fact_count > 0
    assert bundle.chunk_count > 0
    assert bundle.document_count >= 2
    assert bundle.logical_request_count <= READY_MAX_REQUESTS

    # Every expected call actually happened: ESearch, one batched EFetch,
    # and one Europe PMC search per document.
    assert any("esearch.fcgi" in u for u in http.requested_urls)
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1
    assert sum("ebi.ac.uk" in u for u in http.requested_urls) == 2


def test_complete_execution_never_rebuilds_the_bundle():
    http = _success_fake_http()
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=http, env=_ENV,
    )
    assert isinstance(execution.bundle, LiteraturePipelineBundle)
    assert execution.bundle.reference_mode == LiteratureReferenceMode.NCT_ID
    assert execution.bundle.nct_id == NCT_ID
    assert execution.bundle.current_program_confirmed is False
    assert execution.bundle.anthropic_api_calls == 0
    assert execution.bundle.web_search_calls == 0
    assert execution.bundle.external_llm_tokens == 0


# =============================================================================
# 3. max_articles=3, full-text fetch<=1, physical retries vs logical requests
# =============================================================================
def test_bridge_always_requests_the_fixed_article_and_fulltext_caps(monkeypatch):
    captured: dict = {}
    original = ala.validate_literature_pipeline_request

    def _capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(ala, "validate_literature_pipeline_request", _capture)
    execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=_success_fake_http(), env=_ENV,
    )

    assert captured["max_articles"] == 3
    assert captured["max_fulltext_fetches"] == 1
    assert captured["live"] is True
    assert captured["use_fixtures"] is False
    assert captured["use_corpus"] is False
    assert captured["pmid"] is None
    assert captured["nct_id"] == NCT_ID
    assert captured["enabled"] is True
    # A copy, never the caller's own mapping object.
    assert captured["env"] is not _ENV
    assert captured["env"] == _ENV


@dataclass
class _RetryingFakeHttpClient:
    """A fake transport that simulates exactly one physical retry for one
    chosen URL -- proving ``logical_request_count``/``physical_attempt_
    count`` are tracked as genuinely separate counters all the way
    through this bridge's output, not collapsed into one."""

    responses: dict
    retry_once_urls: frozenset[str] = field(default_factory=frozenset)
    requested_urls: list = field(default_factory=list)
    requests_made: int = 0
    attempts_made: int = 0
    cache_hits: int = 0

    def get(self, url: str, **_kwargs: object):
        self.requested_urls.append(url)
        self.requests_made += 1
        self.attempts_made += 2 if url in self.retry_once_urls else 1
        return self.responses.get(url, fx.not_found("no fake response registered"))


def test_physical_retries_and_logical_requests_are_distinct_counters():
    term = f"{NCT_ID}[si]"
    esearch_url = _esearch_url(term, 20)
    http = _RetryingFakeHttpClient(
        responses={
            esearch_url: fx.ok(fx.esearch_response(PMIDS)),
            _efetch_url(PMIDS): fx.ok(fx.fixture_text("batch_articleset.xml")),
            fx.europepmc_search_url(PMIDS[0]): fx.ok(_EPMC_EMPTY),
            fx.europepmc_search_url(PMIDS[1]): fx.ok(_EPMC_EMPTY),
        },
        retry_once_urls=frozenset({esearch_url}),
    )
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=http, env=_ENV,
    )
    assert execution.status is AdaptiveExecutionStatus.COMPLETE
    bundle = execution.bundle
    assert bundle.logical_request_count == 4  # ESearch + EFetch + 2x EuropePMC search
    assert bundle.physical_attempt_count == 5  # one extra attempt for the retried ESearch
    assert bundle.physical_attempt_count > bundle.logical_request_count
    assert bundle.logical_request_count <= READY_MAX_REQUESTS


# =============================================================================
# 4. current_program_confirmed False, zero Anthropic/Web Search/LLM tokens
# =============================================================================
def test_current_program_confirmed_and_external_llm_usage_always_zero():
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=_success_fake_http(), env=_ENV,
    )
    bundle = execution.bundle
    assert bundle.current_program_confirmed is False
    assert bundle.anthropic_api_calls == 0
    assert bundle.web_search_calls == 0
    assert bundle.external_llm_tokens == 0


# =============================================================================
# 5. Every non-READY plan is SKIPPED, with the acquisition path never called
# =============================================================================
@pytest.mark.parametrize(
    "status",
    [
        AcquisitionPlanStatus.NO_ACTION,
        AcquisitionPlanStatus.UNRESOLVED,
        AcquisitionPlanStatus.CONFLICTED,
        AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE,
        AcquisitionPlanStatus.REFUSED,
    ],
)
def test_non_ready_plan_is_skipped_without_touching_the_acquisition_path(status, monkeypatch):
    def _explode(*_a, **_k):
        raise AssertionError("the acquisition path must never be reached for a non-READY plan")

    monkeypatch.setattr(ala, "validate_literature_pipeline_request", _explode)
    monkeypatch.setattr(ala, "run_literature_pipeline_acquisition", _explode)

    plan = AdaptiveAcquisitionPlan(status=status, rationale="not ready")
    http = fx.FakeHttpClient(responses={})
    execution = execute_adaptive_literature_plan(plan, ticker=TICKER, http_client=http, env=_ENV)

    assert execution.status is AdaptiveExecutionStatus.SKIPPED
    assert execution.bundle is None
    assert http.requested_urls == []


@pytest.mark.parametrize(
    "status",
    [
        AcquisitionPlanStatus.NO_ACTION,
        AcquisitionPlanStatus.UNRESOLVED,
        AcquisitionPlanStatus.CONFLICTED,
        AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE,
        AcquisitionPlanStatus.REFUSED,
    ],
)
@pytest.mark.parametrize(
    "http_client_present,env_present",
    [
        (False, True),   # http_client=None, env injected
        (True, False),   # http_client injected, env=None
        (False, False),  # both None
    ],
    ids=["http_client_none", "env_none", "both_none"],
)
def test_non_ready_plan_is_skipped_even_with_missing_injection(
    status, http_client_present, env_present, monkeypatch
):
    """Correction 1's final priority contract: plan.status is judged
    BEFORE http_client/env are even inspected. A non-READY plan is
    SKIPPED no matter which (or both) of http_client/env is missing --
    never REFUSED, and the request validator/acquisition function are
    never reached either way."""

    def _explode(*_a, **_k):
        raise AssertionError(
            "the acquisition path must never be reached for a non-READY plan, "
            "regardless of http_client/env injection"
        )

    monkeypatch.setattr(ala, "validate_literature_pipeline_request", _explode)
    monkeypatch.setattr(ala, "run_literature_pipeline_acquisition", _explode)

    plan = AdaptiveAcquisitionPlan(status=status, rationale="not ready")
    http = fx.FakeHttpClient(responses={}) if http_client_present else None
    env = _ENV if env_present else None

    execution = execute_adaptive_literature_plan(plan, ticker=TICKER, http_client=http, env=env)

    assert execution.status is AdaptiveExecutionStatus.SKIPPED
    assert execution.bundle is None
    if http is not None:
        assert http.requested_urls == []


# =============================================================================
# 6. Each individually-broken READY field is REFUSED, zero HTTP
# =============================================================================
def _assert_refused_with_no_http(plan: AdaptiveAcquisitionPlan) -> None:
    http = fx.FakeHttpClient(responses={})
    execution = execute_adaptive_literature_plan(plan, ticker=TICKER, http_client=http, env=_ENV)
    assert execution.status is AdaptiveExecutionStatus.REFUSED
    assert execution.bundle is None
    assert http.requested_urls == []


def test_ready_without_requires_external_communication_is_refused():
    _assert_refused_with_no_http(_ready_plan(requires_external_communication=False))


def test_ready_with_wrong_reference_mode_is_refused():
    _assert_refused_with_no_http(_ready_plan(reference_mode="PMID"))


def test_ready_with_malformed_nct_id_is_refused():
    _assert_refused_with_no_http(_ready_plan(nct_id="NCT1234567"))


def test_ready_with_lowercase_non_canonical_nct_id_is_refused():
    """Strict-VALID but not already canonical -- this bridge never
    silently re-canonicalizes a plan it did not build itself."""
    _assert_refused_with_no_http(_ready_plan(nct_id="nct09990777"))


def test_ready_with_a_populated_pmid_is_refused():
    _assert_refused_with_no_http(_ready_plan(pmid="12345678"))


def test_ready_with_wrong_max_requests_is_refused():
    _assert_refused_with_no_http(_ready_plan(max_requests=5))


def test_ready_with_bool_max_requests_is_refused():
    _assert_refused_with_no_http(_ready_plan(max_requests=True))


def test_ready_max_requests_constant_is_six():
    assert READY_MAX_REQUESTS == 6


# =============================================================================
# 7. Missing required injection / missing credential
# =============================================================================
def test_none_http_client_is_refused_with_no_crash():
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=None, env=_ENV,
    )
    assert execution.status is AdaptiveExecutionStatus.REFUSED
    assert execution.bundle is None


def test_none_env_is_refused_with_no_crash():
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=fx.FakeHttpClient(responses={}), env=None,
    )
    assert execution.status is AdaptiveExecutionStatus.REFUSED
    assert execution.bundle is None


def test_both_none_is_refused_with_no_crash():
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=None, env=None,
    )
    assert execution.status is AdaptiveExecutionStatus.REFUSED
    assert execution.bundle is None


@pytest.mark.parametrize("missing_key", ["IRA_NCBI_TOOL", "IRA_NCBI_EMAIL"])
def test_missing_credential_is_refused_with_zero_http(missing_key):
    env = {k: v for k, v in _ENV.items() if k != missing_key}
    http = fx.FakeHttpClient(responses={})
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=http, env=env,
    )
    assert execution.status is AdaptiveExecutionStatus.REFUSED
    assert execution.bundle is None
    assert http.requested_urls == []


# =============================================================================
# 8. Provider failure (e.g. Europe PMC 503) -> INCOMPLETE
# =============================================================================
def test_europepmc_failure_is_incomplete_with_coverage_false_and_a_reason():
    term = f"{NCT_ID}[si]"
    http = fx.FakeHttpClient(
        responses={
            _esearch_url(term, 20): fx.ok(fx.esearch_response(PMIDS)),
            _efetch_url(PMIDS): fx.ok(fx.fixture_text("batch_articleset.xml")),
            fx.europepmc_search_url(PMIDS[0]): fx.failed(FetchOutcome.ERROR, "503"),
            fx.europepmc_search_url(PMIDS[1]): fx.failed(FetchOutcome.ERROR, "503"),
        }
    )
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=http, env=_ENV,
    )
    assert execution.status is AdaptiveExecutionStatus.INCOMPLETE
    assert execution.bundle is not None
    assert execution.bundle.coverage_complete is False
    assert len(execution.bundle.unresolved_reasons) > 0
    assert execution.bundle.refused_reason is None


# =============================================================================
# 9. A refused bundle is reported REFUSED and the original bundle is kept
# =============================================================================
def test_a_refused_bundle_from_the_acquisition_itself_is_kept_not_discarded(monkeypatch):
    canned_bundle = LiteraturePipelineBundle(
        collection_result=None,
        chunks=(),
        document_count=0,
        source_count=0,
        raw_fact_count=0,
        chunk_count=0,
        logical_request_count=0,
        physical_attempt_count=0,
        cache_hit_count=0,
        coverage_complete=False,
        unresolved_reasons=("simulated pre-send refusal for this test",),
        reference_mode=REFERENCE_MODE_NCT_ID,
        pmid=None,
        nct_id=NCT_ID,
        refused_reason="simulated pre-send refusal for this test",
    )

    def _fake_run(*_a, **_k):
        return canned_bundle

    monkeypatch.setattr(ala, "run_literature_pipeline_acquisition", _fake_run)
    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=fx.FakeHttpClient(responses={}), env=_ENV,
    )
    assert execution.status is AdaptiveExecutionStatus.REFUSED
    assert execution.bundle is canned_bundle


# =============================================================================
# 10. Real-network/LLM constructors never touched; Fake acquisition still works
# =============================================================================
def test_real_network_and_llm_constructors_are_never_touched(monkeypatch):
    import socket

    import investment_research.collectors.http as http_module
    import investment_research.llm.client as llm_client_module
    import investment_research.research.anthropic_web as anthropic_web_module
    import investment_research.research.sec_live_smoke as sec_live_smoke_module

    def _explode(*_a, **_k):
        raise AssertionError("Phase 4.3H must never reach a real network/LLM construction path")

    monkeypatch.setattr(http_module.HttpClient, "__init__", _explode)
    monkeypatch.setattr(sec_live_smoke_module.AllowlistedHttpClient, "__init__", _explode)
    monkeypatch.setattr(llm_client_module.LLMClient, "__init__", _explode)
    monkeypatch.setattr(anthropic_web_module.AnthropicWebResearchProvider, "__init__", _explode)
    monkeypatch.setattr(socket.socket, "connect", _explode)

    execution = execute_adaptive_literature_plan(
        _ready_plan(), ticker=TICKER, http_client=_success_fake_http(), env=_ENV,
    )
    assert execution.status is AdaptiveExecutionStatus.COMPLETE


# =============================================================================
# 11. pipeline.py/cli.py never import or call this module
# =============================================================================
def test_pipeline_and_cli_never_reference_this_module():
    from pathlib import Path

    repo_src = Path(__file__).resolve().parent.parent.parent / "src" / "investment_research"
    for relative in ("orchestrator/pipeline.py", "cli.py"):
        text = (repo_src / relative).read_text(encoding="utf-8")
        assert "adaptive_literature_acquisition" not in text
        assert "execute_adaptive_literature_plan" not in text
