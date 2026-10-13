"""Phase 4.3J: ``--adaptive-literature`` CLI opt-in connection.

Verifies the CLI-level wiring added on top of Phase 4.3I's
``Pipeline(adaptive_http_client=..., adaptive_env=...)`` contract:
startup flag-combination validation (exit 2 before any resource is
built), per-ticker transport/env construction, the
``adaptive_literature_info()`` diagnostics gate in ``--json``/the
report, and non-interference with the explicit PMID/NCT path and
``--resume``.

Entirely offline: every test here drives either a monkeypatched
``cli.Pipeline`` (plumbing-only assertions) or the REAL ``Pipeline``
against ``tests/unit/_literature_fixture_support.FakeHttpClient``/a
poison client that raises on any ``.get()`` call. No real HTTP, LLM, or
Web Search of any kind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from investment_research import cli
from investment_research.collectors.base import CollectionResult
from investment_research.config import Settings
from investment_research.reporting.report import adaptive_literature_info, render_report
from investment_research.research.ncbi_credentials import NCBI_EMAIL_ENV_VAR, NCBI_TOOL_ENV_VAR
from investment_research.schemas.enums import FactCategory, Provenance, SourceTier
from investment_research.schemas.fact import RawFact, Source
from investment_research.scoring.adaptive_acquisition_plan import AcquisitionPlanStatus
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
    ProgramCandidateEvidence,
)
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unit"))
import _literature_fixture_support as fx  # noqa: E402

pytestmark = pytest.mark.integration

TICKER = "P43J"
COMPANY = "Phase Fourthreej Biotherapeutics Inc"
NCT_ID = "NCT77778888"
_ENV = {NCBI_TOOL_ENV_VAR: "test-tool", NCBI_EMAIL_ENV_VAR: "test@example.com"}
#: >=8 chars so config.Settings.secret_values()' own length gate would
#: redact it if it reached a log line -- used by the "never logged"
#: test to prove it never even gets that far.
_DUMMY_EMAIL = "dummy-credential-value@example.com"


def _settings(tmp_path: Path) -> Settings:
    fixture_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fixtures"
    return Settings(
        fixture_dir=fixture_dir, corpus_dir=tmp_path, db_path=tmp_path / "x.db",
        cache_dir=tmp_path, log_dir=tmp_path,
    )


def _repo() -> Repository:
    return Repository(open_db(":memory:"))


def _source(source_id: str, url: str) -> Source:
    return Source(
        source_id=source_id, url=url, title="Doc", tier=SourceTier.TIER_1,
        publisher="SEC", published_date="2026-09-01",
    )


def _confirmed_scenario() -> list[CollectionResult]:
    """A hand-built CollectionResult whose Program Identity resolves to
    CONFIRMED with literature UNRESOLVED -- i.e. a READY Adaptive
    Acquisition Plan -- mirrors
    tests/integration/test_phase_4_3i_pipeline_integration.py's own
    ``_confirmed_scenario_collection_results()`` exactly, duplicated here
    (not imported across test modules, per this repo's own convention)
    so this file carries no dependency on that one's internals."""
    source = _source("src_sec", "https://www.sec.gov/Archives/x/8-K.htm")
    fact = RawFact(
        ticker=TICKER, category=FactCategory.CLINICAL, claim="A registered claim.",
        source=source, company_claim=True, collector="phase43j_fixture",
    )
    identity = CompanyIdentityEvidence(
        ticker=TICKER, cik=1234567, sec_official_name=COMPANY,
        source_id=source.source_id, source_tier=source.tier,
        retrieved_at=source.retrieved_at, content_hash=source.content_hash,
    )
    candidate = ProgramCandidateEvidence(
        nct_id=NCT_ID, lead_sponsor=COMPANY, overall_status="RECRUITING",
        source_id=source.source_id, source_tier=source.tier,
        retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        supporting_fact_ids=(fact.fact_id(),),
    )
    return [
        CollectionResult(
            collector="phase43j_fixture", raw_facts=[fact], sources=[source],
            provenance=Provenance.FIXTURE,
            company_identity_evidence=identity,
            program_candidate_evidence=(candidate,),
        )
    ]


def _args(extra_argv: list[str]) -> object:
    """Always through build_parser() -- never a hand-built
    argparse.Namespace -- so every field (including
    --adaptive-literature's own default) is populated exactly as a real
    invocation would get it."""
    return cli.build_parser().parse_args([TICKER, *extra_argv])


class _PoisonHttpClient:
    """Raises on any ``.get()`` -- proves a code path makes ZERO
    Adaptive Literature HTTP requests."""

    def get(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"must not fetch via the Adaptive Literature transport: {url}")


class _CapturingPipeline:
    """Stands in for ``cli.Pipeline`` to capture constructor kwargs
    without running the real 14-agent pipeline -- mirrors
    tests/integration/test_token_budget_cli_wiring.py's own
    ``_FakePipeline`` convention exactly."""

    last_kwargs: dict = {}
    instances: list = []

    def __init__(self, *args, **kwargs):
        type(self).last_kwargs = kwargs
        type(self).instances.append(self)

    def run(self, *args, **kwargs):
        return "STUBBED_RESULT"


# =============================================================================
# 1. Flag OFF: zero transport/env injected, zero change to existing
#    --json/report output (DEMOBIO --fixtures)
# =============================================================================
def test_flag_off_injects_no_adaptive_transport_or_env(monkeypatch):
    _CapturingPipeline.last_kwargs = {}
    monkeypatch.setattr(cli, "Pipeline", _CapturingPipeline)
    monkeypatch.setattr(cli, "collect", lambda *a, **k: ([], {}))
    args = _args(["--fixtures"])
    settings = Settings(fixture_dir=Path("data/fixtures"))
    cli.run_one(TICKER, args, settings, _repo(), http=None)
    assert _CapturingPipeline.last_kwargs["adaptive_http_client"] is None
    assert _CapturingPipeline.last_kwargs["adaptive_env"] is None


def test_flag_off_demobio_fixtures_output_has_no_adaptive_keys_or_sections(capsys):
    assert cli.main(["DEMOBIO", "--fixtures", "--json"]) in (0, 1)
    payload = json.loads(capsys.readouterr().out)
    assert "adaptive_literature_info" not in payload[0]

    assert cli.main(["DEMOBIO", "--fixtures"]) in (0, 1)
    report_text = capsys.readouterr().out
    assert "Adaptive Literature Acquisition" not in report_text


# =============================================================================
# 2. Invalid flag combination: exit 2 before settings/DB/HTTP/any ticker
# =============================================================================
@pytest.mark.parametrize(
    "extra_argv",
    [
        ["--adaptive-literature"],  # no --live
        ["--adaptive-literature", "--fixtures"],  # no --live either, fixtures too
        ["--adaptive-literature", "--live", "--fixtures"],
        ["--adaptive-literature", "--live", "--corpus"],
    ],
    ids=["no-live", "fixtures-no-live", "live-and-fixtures", "live-and-corpus"],
)
def test_invalid_adaptive_flag_combo_exits_2_before_any_resource(monkeypatch, capsys, extra_argv):
    def _explode(*a, **k):
        raise AssertionError("must not be called for a rejected flag combination")

    monkeypatch.setattr(cli, "get_settings", _explode)
    monkeypatch.setattr(cli, "open_db", _explode)
    monkeypatch.setattr(cli, "HttpClient", _explode)

    exit_code = cli.main([TICKER, *extra_argv])
    assert exit_code == 2
    assert "--adaptive-literature" in capsys.readouterr().err


def test_valid_adaptive_flag_combo_never_rejected_at_startup():
    """live=True, fixtures/corpus=False is accepted regardless of NCBI
    credential configuration -- that check happens only at runtime, per
    ticker, if/when a plan actually reaches READY (see
    test_ready_plan_missing_credentials_is_refused_* below)."""
    cli.validate_adaptive_literature_flags(
        enabled=True, live=True, use_fixtures=False, use_corpus=False,
    )  # must not raise


def test_flag_off_validator_is_a_no_op_for_every_combination():
    for live in (True, False):
        for use_fixtures in (True, False):
            for use_corpus in (True, False):
                cli.validate_adaptive_literature_flags(
                    enabled=False, live=live, use_fixtures=use_fixtures, use_corpus=use_corpus,
                )  # must never raise when the flag itself is off


# =============================================================================
# 3. Flag ON + --live: independent transport per ticker, minimal env
# =============================================================================
def test_flag_on_per_ticker_transport_instances_are_distinct(monkeypatch):
    captured_clients = []

    class _CapturingPipeline2:
        def __init__(self, *a, **kwargs):
            captured_clients.append(kwargs.get("adaptive_http_client"))

        def run(self, *a, **k):
            return "STUBBED_RESULT"

    monkeypatch.setattr(cli, "Pipeline", _CapturingPipeline2)
    monkeypatch.setattr(cli, "collect", lambda *a, **k: ([], {}))
    args = _args(["--adaptive-literature", "--live"])
    settings = _settings(Path("/tmp"))

    cli.run_one(TICKER, args, settings, _repo(), http=None)
    cli.run_one(TICKER, args, settings, _repo(), http=None)

    assert len(captured_clients) == 2
    assert captured_clients[0] is not None
    assert captured_clients[1] is not None
    assert captured_clients[0] is not captured_clients[1]


def test_flag_on_env_contains_only_the_three_ncbi_keys(monkeypatch):
    captured_envs = []

    class _CapturingPipeline3:
        def __init__(self, *a, **kwargs):
            captured_envs.append(kwargs.get("adaptive_env"))

        def run(self, *a, **k):
            return "STUBBED_RESULT"

    monkeypatch.setattr(cli, "Pipeline", _CapturingPipeline3)
    monkeypatch.setattr(cli, "collect", lambda *a, **k: ([], {}))
    monkeypatch.setattr(
        cli.os, "environ",
        {
            **_ENV,
            "IRA_ANTHROPIC_API_KEY": "sk-unrelated-secret",
            "PATH": "/usr/bin",
        },
    )
    args = _args(["--adaptive-literature", "--live"])
    settings = _settings(Path("/tmp"))
    cli.run_one(TICKER, args, settings, _repo(), http=None)

    assert captured_envs[0] == dict(_ENV)


# =============================================================================
# 4. Explicit PMID/NCT override in effect: Adaptive makes ZERO HTTP
# =============================================================================
def test_explicit_literature_override_means_zero_adaptive_http(monkeypatch, tmp_path):
    from investment_research.research.literature_pipeline_integration import (
        run_literature_pipeline_acquisition,
        validate_literature_pipeline_request,
    )

    def fake_collect(*_a, **_k):
        return _confirmed_scenario(), {}

    def fake_literature(request, *, ticker, **_kw):
        http = fx.FakeHttpClient(responses={})
        return run_literature_pipeline_acquisition(request, ticker=ticker, http_client=http, env=_ENV)

    monkeypatch.setattr(cli, "collect", fake_collect)
    monkeypatch.setattr(cli, "run_literature_pipeline_acquisition", fake_literature)
    monkeypatch.setattr(cli, "build_adaptive_http_client", lambda: _PoisonHttpClient())

    args = cli.build_parser().parse_args([
        TICKER, "--adaptive-literature", "--live",
        "--document-first-literature", "--literature-nct-id", NCT_ID,
    ])
    settings = _settings(tmp_path)
    request = validate_literature_pipeline_request(
        enabled=True, pmid=None, nct_id=NCT_ID, live=True,
        use_fixtures=False, use_corpus=False, env=_ENV,
    )
    result = cli.run_one(TICKER, args, settings, _repo(), http=None, literature_request=request)

    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    assert result.adaptive_literature_diagnostics["outcome"] == "NOT_ATTEMPTED_NON_READY"
    assert result.adaptive_literature_blocking_reasons == []


# =============================================================================
# 5. READY plan + missing NCBI credentials: Adaptive REFUSED, zero HTTP,
#    INCOMPLETE_RESEARCH, Action=None -- never "zero HTTP for the whole run"
# =============================================================================
def test_ready_plan_missing_credentials_is_refused_with_zero_adaptive_http(monkeypatch, tmp_path):
    def fake_collect(*_a, **_k):
        return _confirmed_scenario(), {}

    monkeypatch.setattr(cli, "collect", fake_collect)
    monkeypatch.setattr(cli, "build_adaptive_http_client", lambda: _PoisonHttpClient())
    monkeypatch.setattr(cli.os, "environ", {})  # no NCBI credentials configured at all

    args = cli.build_parser().parse_args([TICKER, "--adaptive-literature", "--live"])
    settings = _settings(tmp_path)
    result = cli.run_one(TICKER, args, settings, _repo(), http=None)

    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.READY
    assert result.adaptive_literature_diagnostics["outcome"] == "REFUSED"
    assert result.adaptive_literature_blocking_reasons != ()
    assert result.context.status.value == "INCOMPLETE_RESEARCH"
    if result.verdict is not None:
        assert result.verdict.action is None


# =============================================================================
# 6. COMPLETE outcome: visible in --json/report; NOT_ATTEMPTED: invisible
# =============================================================================
def test_complete_outcome_shown_in_json_and_report_diagnostics(monkeypatch, tmp_path):
    from urllib.parse import urlencode

    from investment_research.research.literature_acquisition_adapter import (
        NCBI_EFETCH_URL,
        NCBI_ESEARCH_URL,
    )

    pmids = ["90000015", "90000016"]

    def _esearch_url(term, max_pmids):
        params = {
            "db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids),
            "tool": "test-tool", "email": "test@example.com",
        }
        return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"

    def _efetch_url(ids):
        params = {
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract",
            "tool": "test-tool", "email": "test@example.com",
        }
        return f"{NCBI_EFETCH_URL}?{urlencode(params)}"

    term = f"{NCT_ID}[si]"
    fake_http = fx.FakeHttpClient(responses={
        _esearch_url(term, 20): fx.ok(fx.esearch_response(pmids)),
        _efetch_url(pmids): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url(pmids[0]): fx.ok('{"resultList": {"result": []}}'),
        fx.europepmc_search_url(pmids[1]): fx.ok('{"resultList": {"result": []}}'),
    })

    def fake_collect(*_a, **_k):
        return _confirmed_scenario(), {}

    monkeypatch.setattr(cli, "collect", fake_collect)
    monkeypatch.setattr(cli, "build_adaptive_http_client", lambda: fake_http)

    args = cli.build_parser().parse_args([TICKER, "--adaptive-literature", "--live"])
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli.os, "environ", dict(_ENV))
    result = cli.run_one(TICKER, args, settings, _repo(), http=None)

    assert result.adaptive_literature_diagnostics["outcome"] == "COMPLETE"
    info = adaptive_literature_info(result)
    assert info["outcome"] == "COMPLETE"
    assert info["nct_id"] == NCT_ID
    assert info["blocking_reasons"] == []

    payload = cli.result_to_json(result)
    assert payload["adaptive_literature_info"]["outcome"] == "COMPLETE"

    report_text = render_report(result)
    assert "Adaptive Literature Acquisition" in report_text
    assert "outcome: COMPLETE" in report_text


def test_not_attempted_no_runner_outcome_hidden_from_json_and_report(monkeypatch, tmp_path):
    def fake_collect(*_a, **_k):
        return _confirmed_scenario(), {}

    monkeypatch.setattr(cli, "collect", fake_collect)
    args = cli.build_parser().parse_args([TICKER, "--live"])  # no --adaptive-literature
    settings = _settings(tmp_path)
    result = cli.run_one(TICKER, args, settings, _repo(), http=None)

    assert result.adaptive_literature_diagnostics["outcome"] == "NOT_ATTEMPTED_NO_RUNNER"
    assert adaptive_literature_info(result) == {}

    payload = cli.result_to_json(result)
    assert "adaptive_literature_info" not in payload

    report_text = render_report(result)
    assert "Adaptive Literature Acquisition" not in report_text


# =============================================================================
# 7. --resume with an existing Adaptive snapshot: zero Adaptive HTTP even
#    with this invocation's own flag OFF; diagnostics still shown.
#    "Zero re-fetch" below is an Adaptive-channel-only claim -- it says
#    nothing about cli.py's own structured-collector re-execution on
#    --resume (collect() is monkeypatched to a no-HTTP fake here, same
#    as every other test in this file, precisely so this test cannot be
#    misread as "the whole CLI makes zero requests on resume").
# =============================================================================
def test_resume_with_existing_adaptive_snapshot_makes_zero_adaptive_http_even_flag_off(
    monkeypatch, tmp_path,
):
    from urllib.parse import urlencode

    from investment_research.research.literature_acquisition_adapter import (
        NCBI_EFETCH_URL,
        NCBI_ESEARCH_URL,
    )

    pmids = ["90000015", "90000016"]

    def _esearch_url(term, max_pmids):
        params = {
            "db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids),
            "tool": "test-tool", "email": "test@example.com",
        }
        return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"

    def _efetch_url(ids):
        params = {
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract",
            "tool": "test-tool", "email": "test@example.com",
        }
        return f"{NCBI_EFETCH_URL}?{urlencode(params)}"

    term = f"{NCT_ID}[si]"
    fake_http = fx.FakeHttpClient(responses={
        _esearch_url(term, 20): fx.ok(fx.esearch_response(pmids)),
        _efetch_url(pmids): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url(pmids[0]): fx.ok('{"resultList": {"result": []}}'),
        fx.europepmc_search_url(pmids[1]): fx.ok('{"resultList": {"result": []}}'),
    })

    def fake_collect(*_a, **_k):
        return _confirmed_scenario(), {}

    repo = _repo()
    monkeypatch.setattr(cli, "collect", fake_collect)
    monkeypatch.setattr(cli, "build_adaptive_http_client", lambda: fake_http)
    monkeypatch.setattr(cli.os, "environ", dict(_ENV))

    first_args = cli.build_parser().parse_args([TICKER, "--adaptive-literature", "--live"])
    settings = _settings(tmp_path)
    first = cli.run_one(TICKER, first_args, settings, repo, http=None)
    assert first.adaptive_literature_diagnostics["outcome"] == "COMPLETE"
    run_id = first.context.run_id

    # Second invocation: --resume, flag OFF this time, and the Adaptive
    # transport builder poisoned -- if it were ever called, the test
    # fails loudly instead of silently restoring by luck.
    monkeypatch.setattr(cli, "build_adaptive_http_client", lambda: _PoisonHttpClient())
    second_args = cli.build_parser().parse_args([TICKER, "--live", "--resume", run_id])
    second = cli.run_one(TICKER, second_args, settings, repo, http=None)

    assert second.adaptive_literature_diagnostics["outcome"] == "COMPLETE"
    assert second.adaptive_literature_diagnostics.get("restored_from") == "adaptive_verify_snapshot"
    info = adaptive_literature_info(second)
    assert info["outcome"] == "COMPLETE"  # diagnostics not hidden despite this invocation's flag being off


# =============================================================================
# 8. A dummy credential value never reaches --json or the report
# =============================================================================
def test_dummy_credential_value_never_appears_in_json_or_report(monkeypatch, tmp_path):
    from urllib.parse import urlencode

    from investment_research.research.literature_acquisition_adapter import (
        NCBI_EFETCH_URL,
        NCBI_ESEARCH_URL,
    )

    pmids = ["90000015", "90000016"]
    dummy_env = {NCBI_TOOL_ENV_VAR: "test-tool", NCBI_EMAIL_ENV_VAR: _DUMMY_EMAIL}

    def _esearch_url(term, max_pmids):
        params = {
            "db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids),
            "tool": "test-tool", "email": _DUMMY_EMAIL,
        }
        return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"

    def _efetch_url(ids):
        params = {
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract",
            "tool": "test-tool", "email": _DUMMY_EMAIL,
        }
        return f"{NCBI_EFETCH_URL}?{urlencode(params)}"

    term = f"{NCT_ID}[si]"
    fake_http = fx.FakeHttpClient(responses={
        _esearch_url(term, 20): fx.ok(fx.esearch_response(pmids)),
        _efetch_url(pmids): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url(pmids[0]): fx.ok('{"resultList": {"result": []}}'),
        fx.europepmc_search_url(pmids[1]): fx.ok('{"resultList": {"result": []}}'),
    })

    def fake_collect(*_a, **_k):
        return _confirmed_scenario(), {}

    monkeypatch.setattr(cli, "collect", fake_collect)
    monkeypatch.setattr(cli, "build_adaptive_http_client", lambda: fake_http)
    monkeypatch.setattr(cli.os, "environ", dict(dummy_env))

    args = cli.build_parser().parse_args([TICKER, "--adaptive-literature", "--live"])
    settings = _settings(tmp_path)
    result = cli.run_one(TICKER, args, settings, _repo(), http=None)
    assert result.adaptive_literature_diagnostics["outcome"] == "COMPLETE"

    payload = cli.result_to_json(result)
    serialized = json.dumps(payload, default=str)
    assert _DUMMY_EMAIL not in serialized

    report_text = render_report(result)
    assert _DUMMY_EMAIL not in report_text


# =============================================================================
# 9. cli.py <-> reporting/report.py: no circular import
# =============================================================================
def test_result_to_json_and_report_import_without_circularity():
    import importlib

    import investment_research.reporting.report as report_module

    importlib.reload(report_module)
    importlib.reload(cli)
    assert hasattr(cli, "adaptive_literature_info")
    assert cli.adaptive_literature_info is report_module.adaptive_literature_info


# =============================================================================
# 10. --resume of a run_id whose base already holds an EXPLICIT PMID/NCT
#     acquisition, with the explicit flags omitted this time: Adaptive must
#     not acquire the same literature a second time on top of the restored
#     copy (duplicate requests, and the same facts counted twice).
# =============================================================================
def _complete_literature_http() -> fx.FakeHttpClient:
    from urllib.parse import urlencode

    from investment_research.research.literature_acquisition_adapter import (
        NCBI_EFETCH_URL,
        NCBI_ESEARCH_URL,
    )

    pmids = ["90000015", "90000016"]
    courtesy = {"tool": "test-tool", "email": "test@example.com"}
    esearch = urlencode({
        "db": "pubmed", "term": f"{NCT_ID}[si]", "retmode": "json", "retmax": "20", **courtesy,
    })
    efetch = urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract", **courtesy,
    })
    return fx.FakeHttpClient(responses={
        f"{NCBI_ESEARCH_URL}?{esearch}": fx.ok(fx.esearch_response(pmids)),
        f"{NCBI_EFETCH_URL}?{efetch}": fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url(pmids[0]): fx.ok('{"resultList": {"result": []}}'),
        fx.europepmc_search_url(pmids[1]): fx.ok('{"resultList": {"result": []}}'),
    })


@pytest.mark.parametrize("restored_facts_fresh", [True, False], ids=["restored", "re-collected"])
def test_resume_of_explicit_override_run_without_explicit_flags(
    monkeypatch, tmp_path, restored_facts_fresh,
):
    from investment_research.orchestrator import resume as resume_module
    from investment_research.research.literature_pipeline_integration import (
        run_literature_pipeline_acquisition,
        validate_literature_pipeline_request,
    )

    explicit_http = _complete_literature_http()
    adaptive_http = _complete_literature_http()

    def fake_literature(request, *, ticker, **_kw):
        return run_literature_pipeline_acquisition(
            request, ticker=ticker, http_client=explicit_http, env=_ENV,
        )

    monkeypatch.setattr(cli, "collect", lambda *_a, **_k: (_confirmed_scenario(), {}))
    monkeypatch.setattr(cli, "run_literature_pipeline_acquisition", fake_literature)
    monkeypatch.setattr(cli, "build_adaptive_http_client", lambda: adaptive_http)
    monkeypatch.setattr(cli.os, "environ", dict(_ENV))
    if restored_facts_fresh:
        # The fixture articles are older than their category's freshness
        # threshold, which would make the resume re-collect from the start
        # (the "re-collected" case below). Hold them fresh so the resume
        # really restores the first invocation's base evidence.
        monkeypatch.setattr(resume_module, "is_stale", lambda _fact, _today: False)

    repo = _repo()
    settings = _settings(tmp_path)
    first_args = cli.build_parser().parse_args([
        TICKER, "--adaptive-literature", "--live",
        "--document-first-literature", "--literature-nct-id", NCT_ID,
    ])
    request = validate_literature_pipeline_request(
        enabled=True, pmid=None, nct_id=NCT_ID, live=True,
        use_fixtures=False, use_corpus=False, env=_ENV,
    )
    first = cli.run_one(TICKER, first_args, settings, repo, http=None, literature_request=request)
    assert first.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    assert explicit_http.requested_urls
    assert adaptive_http.requested_urls == []

    second_args = cli.build_parser().parse_args([
        TICKER, "--adaptive-literature", "--live", "--resume", first.context.run_id,
    ])
    second = cli.run_one(TICKER, second_args, settings, repo, http=None)
    fact_ids = [fact.fact_id for fact in second.bus.facts]
    assert len(fact_ids) == len(set(fact_ids))

    if restored_facts_fresh:
        assert second.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
        assert second.adaptive_literature_diagnostics["outcome"] == "NOT_ATTEMPTED_NON_READY"
        assert adaptive_http.requested_urls == []
        assert len(second.bus.facts) == len(first.bus.facts)
    else:
        # Re-collected from the start: the explicit evidence is no longer in
        # the base, so this invocation plans exactly like a fresh run with
        # its own flags -- the Adaptive acquisition is the only literature.
        assert second.adaptive_acquisition_plan.status is AcquisitionPlanStatus.READY
        assert second.adaptive_literature_diagnostics["outcome"] == "COMPLETE"
        assert adaptive_http.requested_urls
