"""Phase 4.3K: --adaptive-literature diagnostics in --json and the report.

When the flag is given, ``adaptive_literature_info`` always appears and
carries the plan status/rationale, the Adaptive outcome, and THIS
invocation's own logical and actual GET counts -- a non-READY plan included
(zero GETs, and why). A flag-off run keeps the pre-4.3K shape exactly.

Drives ``cli.main()`` with the REAL ``build_adaptive_http_client()``
(``AllowlistedHttpClient``, budget 6). Only the lowest layer is faked:
``urllib.request.HTTPSHandler.https_open`` returns canned responses, so the
client's own budget reservation, retries, error processing and secret
stripping all still run. No socket is ever opened (guarded below).
"""

from __future__ import annotations

import io
import json
import socket
import sqlite3
import sys
import urllib.parse
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path

import pytest

from investment_research import cli
from investment_research.collectors.base import CollectionResult
from investment_research.config import Settings
from investment_research.schemas.enums import FactCategory, Provenance, SourceTier
from investment_research.schemas.fact import RawFact, Source
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
    ProgramCandidateEvidence,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unit"))
import _literature_fixture_support as fx  # noqa: E402

pytestmark = pytest.mark.integration

TICKER = "P43K"
COMPANY = "Phase Fourthreek Biotherapeutics Inc"
NCT_ID = "NCT77779999"
PMIDS = ["90000015", "90000016"]
_TOOL = "p43k-dummy-tool-value"
_EMAIL = "p43k-dummy-contact@example.org"
_API_KEY = "p43kDummyApiKeyValue0123456789"
_NEW_KEYS = (
    "plan_status", "plan_rationale",
    "this_invocation_logical_get_count", "this_invocation_actual_get_count",
)


# --- scenario ---------------------------------------------------------------
def _scenario(*, with_candidate: bool = True) -> list[CollectionResult]:
    source = Source(
        source_id="src_p43k_sec", url="https://www.sec.gov/Archives/x/8-K.htm", title="Doc",
        tier=SourceTier.TIER_1, publisher="SEC", published_date="2026-09-01",
    )
    fact = RawFact(
        ticker=TICKER, category=FactCategory.CLINICAL, claim="A registered claim.",
        source=source, company_claim=True, collector="p43k_fixture",
    )
    identity = CompanyIdentityEvidence(
        ticker=TICKER, cik=1234567, sec_official_name=COMPANY, source_id=source.source_id,
        source_tier=source.tier, retrieved_at=source.retrieved_at,
        content_hash=source.content_hash,
    )
    candidates = (
        ProgramCandidateEvidence(
            nct_id=NCT_ID, lead_sponsor=COMPANY, overall_status="RECRUITING",
            source_id=source.source_id, source_tier=source.tier,
            retrieved_at=source.retrieved_at, content_hash=source.content_hash,
            supporting_fact_ids=(fact.fact_id(),),
        ),
    ) if with_candidate else ()
    return [
        CollectionResult(
            collector="p43k_fixture", raw_facts=[fact], sources=[source],
            provenance=Provenance.FIXTURE, company_identity_evidence=identity,
            program_candidate_evidence=candidates,
        )
    ]


# --- lowest-layer fake wire ---------------------------------------------------
class _FakeWire:
    """Stands in for the network under the client's budgeted HTTPS handler.
    ``fail`` maps a URL path to an HTTP status to return instead of 200."""

    def __init__(self, fail: dict[str, int] | None = None) -> None:
        self.sent: list[str] = []
        self.fail = dict(fail or {})

    def __call__(self, _handler, req):
        url = req.full_url
        self.sent.append(url)
        parsed = urllib.parse.urlparse(url)
        status = 200
        if "esearch" in parsed.path:
            body = fx.esearch_response(PMIDS)
        elif "efetch" in parsed.path:
            body = fx.fixture_text("batch_articleset.xml")
        else:
            query = urllib.parse.parse_qs(parsed.query).get("query", [""])[0]
            key = f"{parsed.path}?{query}"
            status = self.fail.get(key, 200)
            body = '{"resultList": {"result": []}}'
        headers = Message()
        response = urllib.response.addinfourl(io.BytesIO(body.encode()), headers, url, status)
        response.msg = "OK" if status == 200 else "Error"
        return response


@pytest.fixture
def harness(monkeypatch, tmp_path):
    def no_socket(*_a, **_k):
        raise AssertionError("a real socket connection was attempted")

    monkeypatch.setattr(socket, "create_connection", no_socket)
    monkeypatch.setattr(socket.socket, "connect", no_socket)
    for name in ("IRA_NCBI_TOOL", "IRA_NCBI_EMAIL", "IRA_NCBI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    fixture_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fixtures"
    state = {"settings": None}

    def settings():
        if state["settings"] is None:
            state["settings"] = Settings(
                fixture_dir=fixture_dir, corpus_dir=tmp_path, db_path=tmp_path / "r.db",
                cache_dir=tmp_path / "cache", log_dir=tmp_path / "logs",
            )
        return state["settings"]

    monkeypatch.setattr(cli, "get_settings", settings)
    monkeypatch.setattr(cli, "collect", lambda *a, **k: (_scenario(), {}))
    wire = _FakeWire()
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", lambda h, req: wire(h, req))
    return {"wire": wire, "tmp": tmp_path, "monkeypatch": monkeypatch}


def _credentials(monkeypatch):
    monkeypatch.setenv("IRA_NCBI_TOOL", _TOOL)
    monkeypatch.setenv("IRA_NCBI_EMAIL", _EMAIL)
    monkeypatch.setenv("IRA_NCBI_API_KEY", _API_KEY)


def _main(capsys, argv: list[str]) -> tuple[int, dict | None, str]:
    code = cli.main([TICKER, *argv, "--json"])
    payload = json.loads(capsys.readouterr().out)[0]
    report_code = cli.main([TICKER, *argv])
    report = capsys.readouterr().out
    assert code == report_code
    return code, payload, report


def _json_and_report(capsys, argv):
    _, payload, report = _main(capsys, argv)
    return payload, report


# --- READY -> COMPLETE ---------------------------------------------------------
def test_ready_complete_shows_plan_outcome_and_exact_get_counts(harness, capsys):
    _credentials(harness["monkeypatch"])
    payload, report = _json_and_report(capsys, ["--live", "--adaptive-literature"])
    info = payload["adaptive_literature_info"]

    assert info["outcome"] == "COMPLETE"
    assert info["plan_status"] == "READY"
    assert NCT_ID in info["plan_rationale"]
    # ESearch + EFetch + Europe PMC x2, per invocation (two invocations ran).
    assert info["this_invocation_logical_get_count"] == 4
    assert info["this_invocation_actual_get_count"] == 4
    assert len(harness["wire"].sent) == 8
    assert "not_attempted_reason" not in info

    assert "plan_status: READY" in report
    assert "this_invocation_logical_get_count: 4" in report
    assert "this_invocation_actual_get_count: 4" in report


# --- READY -> INCOMPLETE (a retried failure: logical != actual) -------------
def test_incomplete_with_retries_reports_logical_and_actual_separately(harness, capsys):
    _credentials(harness["monkeypatch"])
    harness["wire"].fail = {
        f"/europepmc/webservices/rest/search?ext_id:{PMIDS[1]} AND src:med": 503,
    }
    payload, report = _json_and_report(capsys, ["--live", "--adaptive-literature"])
    info = payload["adaptive_literature_info"]

    assert info["outcome"] == "INCOMPLETE"
    assert info["plan_status"] == "READY"
    assert info["blocking_reasons"]
    assert info["this_invocation_logical_get_count"] == 4
    assert info["this_invocation_actual_get_count"] == 5  # one retry, max_retries=1
    assert info["this_invocation_actual_get_count"] <= 6
    assert payload["action"] is None
    assert "outcome: INCOMPLETE" in report


# --- READY -> REFUSED (no NCBI credentials): zero GETs -------------------------
def test_refused_reports_zero_gets(harness, capsys):
    payload, report = _json_and_report(capsys, ["--live", "--adaptive-literature"])
    info = payload["adaptive_literature_info"]

    assert info["outcome"] == "REFUSED"
    assert info["plan_status"] == "READY"
    assert info["this_invocation_logical_get_count"] == 0
    assert info["this_invocation_actual_get_count"] == 0
    assert harness["wire"].sent == []
    assert payload["action"] is None
    assert "this_invocation_actual_get_count: 0" in report


# --- non-READY plan: zero GETs, and why ----------------------------------------
def test_non_ready_plan_reports_zero_gets_and_the_reason(harness, capsys):
    _credentials(harness["monkeypatch"])
    harness["monkeypatch"].setattr(
        cli, "collect", lambda *a, **k: (_scenario(with_candidate=False), {}),
    )
    payload, report = _json_and_report(capsys, ["--live", "--adaptive-literature"])
    info = payload["adaptive_literature_info"]

    assert info["outcome"] == "NOT_ATTEMPTED_NON_READY"
    assert info["plan_status"] == "UNRESOLVED"
    assert info["plan_rationale"]
    assert "not READY" in info["not_attempted_reason"]
    assert info["this_invocation_logical_get_count"] == 0
    assert info["this_invocation_actual_get_count"] == 0
    assert info["blocking_reasons"] == []
    assert harness["wire"].sent == []
    assert "plan_status: UNRESOLVED" in report
    assert "not_attempted_reason:" in report


def test_explicit_reference_this_invocation_shows_skipped_plan_and_zero_adaptive_gets(
    harness, capsys,
):
    from investment_research.research.literature_pipeline_integration import (
        run_literature_pipeline_acquisition,
    )

    _credentials(harness["monkeypatch"])
    explicit = fx.FakeHttpClient(responses={})
    harness["monkeypatch"].setattr(
        cli, "run_literature_pipeline_acquisition",
        lambda request, *, ticker, **_k: run_literature_pipeline_acquisition(
            request, ticker=ticker, http_client=explicit,
            env={"IRA_NCBI_TOOL": _TOOL, "IRA_NCBI_EMAIL": _EMAIL},
        ),
    )
    payload, _ = _json_and_report(capsys, [
        "--live", "--adaptive-literature",
        "--document-first-literature", "--literature-nct-id", NCT_ID,
    ])
    info = payload["adaptive_literature_info"]
    assert info["plan_status"] == "SKIPPED_EXPLICIT_OVERRIDE"
    assert info["outcome"] == "NOT_ATTEMPTED_NON_READY"
    assert info["this_invocation_actual_get_count"] == 0
    assert harness["wire"].sent == []  # the Adaptive transport sent nothing
    assert explicit.requested_urls  # the explicit path did run


# --- resume: never shows a past invocation's GETs as this invocation's ------
def test_resume_reports_this_invocations_zero_gets_not_the_restored_runs(harness, capsys):
    _credentials(harness["monkeypatch"])
    code = cli.main([TICKER, "--live", "--adaptive-literature", "--json"])
    first = json.loads(capsys.readouterr().out)[0]
    assert first["adaptive_literature_info"]["this_invocation_actual_get_count"] == 4
    sent_before = len(harness["wire"].sent)

    for argv in (["--live", "--adaptive-literature"], ["--live"]):
        assert cli.main([TICKER, *argv, "--resume", first["run_id"], "--json"]) == code
        info = json.loads(capsys.readouterr().out)[0]["adaptive_literature_info"]
        assert info["outcome"] == "COMPLETE"
        assert info["restored_from"] == "adaptive_verify_snapshot"
        if "--adaptive-literature" in argv:
            assert info["this_invocation_logical_get_count"] == 0
            assert info["this_invocation_actual_get_count"] == 0
            assert info["plan_status"] == "READY"
        else:
            # Flag off: the pre-4.3K shape, no invocation keys at all.
            assert not any(key in info for key in _NEW_KEYS)
    assert len(harness["wire"].sent) == sent_before


# --- flag off: no new key anywhere ---------------------------------------------
def test_flag_off_live_run_has_no_adaptive_info_or_block(harness, capsys):
    _credentials(harness["monkeypatch"])
    payload, report = _json_and_report(capsys, ["--live"])
    assert "adaptive_literature_info" not in payload
    assert "Adaptive Literature Acquisition" not in report
    assert harness["wire"].sent == []


# --- no credential value or wire URL anywhere it could be stored ---------------
def test_no_credential_or_wire_url_in_json_report_logs_db_or_snapshots(harness, capsys):
    _credentials(harness["monkeypatch"])
    harness["wire"].fail = {
        f"/europepmc/webservices/rest/search?ext_id:{PMIDS[1]} AND src:med": 503,
    }
    _, payload, report = _main(capsys, ["--live", "--adaptive-literature"])
    assert any("email=" in u for u in harness["wire"].sent)  # it WAS on the wire

    tmp = harness["tmp"]
    blobs = {"json": json.dumps(payload, default=str), "report": report}
    for path in (tmp / "logs").rglob("*"):
        if path.is_file():
            blobs[f"log:{path.name}"] = path.read_text(errors="ignore")
    con = sqlite3.connect(tmp / "r.db")
    blobs["db"] = "\n".join(con.iterdump())
    con.close()
    for name, text in blobs.items():
        for needle in (_TOOL, _EMAIL, _API_KEY, urllib.parse.quote(_EMAIL, safe=""),
                       "email=", "api_key=", "tool="):
            assert needle not in text, f"{needle!r} leaked into {name}"


def test_transport_without_counters_reports_unknown_never_a_guess(monkeypatch):
    """A transport that exposes no requests_made/attempts_made (e.g. a test
    double) yields UNKNOWN, never a count inferred from something else."""
    from investment_research.orchestrator.pipeline import _transport_count

    assert _transport_count(fx.FakeHttpClient(responses={}), "attempts_made") == "UNKNOWN"
    assert _transport_count(object(), "requests_made") == "UNKNOWN"
