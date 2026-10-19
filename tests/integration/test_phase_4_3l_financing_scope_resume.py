"""Phase 4.3L Correction 6: (a) a company-reported financing condition tied to
a regulatory step stops the Action on its own, with no invented kill level;
(b) the collect snapshot records WHICH 8-K scope was read, and a resume --
from the COLLECT snapshot or from the VERIFY snapshot -- refuses a record
written under another (or no) scope instead of reusing it or re-fetching.

Non-vacuity: every resume appends a "RESUMED RUN" failure, so Action=None /
INCOMPLETE alone proves nothing. The discriminating signal is the specific
blocking reason, measured against a control with the CURRENT scope that adds
none. The noise-free harness (copied, per this repo's no-cross-test-import
convention) removes the DEMOBIO fixture's unrelated gates, so the baseline run
genuinely reaches COMPLETE with an Action.

Offline: Fake transport only. All 8-K wording is SYNTHETIC.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import investment_research.orchestrator.pipeline as pipeline_module
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator import resume as resume_module
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research import sec_material_8k as m8k
from investment_research.research.escalation import EscalationReport
from investment_research.research.provider import DomainCoverage, NullResearchProvider
from investment_research.schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    FetchOutcome,
    ResearchStatus,
    RunStatus,
    SearchStatus,
)
from investment_research.scoring.completeness import CompletenessResult
from investment_research.scoring.evidence_sufficiency import EvidenceSufficiencyMatrix
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

pytestmark = pytest.mark.integration

TODAY = date(2026, 10, 16)
TICKER = "DEMOBIO"
FINANCING_PREFIX = "FINANCING CONDITION REQUIRES VERIFICATION:"
CIK = 9900001
ACC = "0009900001-26-000123"
ACC_ND = ACC.replace("-", "")
BASE = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{ACC_ND}"
PRIMARY = "synth-8k.htm"
EX991 = "synth-ex99-1.htm"
FILLER = (
    " This synthetic paragraph exists only to give the test document a realistic length so "
    "that body validation treats it as a real filing rather than an index or error page."
)
PRIMARY_BODY = (
    "<html><body><p>FORM 8-K CURRENT REPORT</p><p>Date of Report (Date of earliest event "
    "reported): September 16, 2026</p><p>Synthetica Therapeutics Inc.</p><p>Item 8.01 Other "
    "Events." + FILLER + "</p></body></html>"
)
FINANCING = (
    "The Second Closing Trigger Date shall occur upon satisfaction or waiver of the Milestone set "
    "forth in the Purchase Agreement, being the Company's receipt of written confirmation from the "
    "FDA that a statistically significant improvement in the primary endpoint of the trial would "
    "be sufficient to support a marketing application, and the Investors shall purchase the "
    "second tranche of Shares at the Second Closing."
)
UNCONDITIONAL = "The Company closed a private placement of $10 million in gross proceeds."
SPECULATION = (
    "There is a risk that the second tranche could be conditioned on FDA agreement on the "
    "primary endpoint."
)


def _ex99(sentences: str) -> str:
    return (
        "<html><body><p>SOUTH CITY, Sept. 16, 2026 (SYNTHETIC WIRE) -- Synthetica Therapeutics "
        f"Inc. today reported. {sentences}</p>" + "<p>" + FILLER + "</p></body></html>"
    )


class FakeSec:
    def __init__(self, routes):
        self.routes = routes
        self.requested: list[str] = []

    def get(self, url):
        from investment_research.collectors.http import FetchResult

        self.requested.append(url)
        route = self.routes.get(url)
        if route is None:
            return FetchResult(url=url, outcome=FetchOutcome.NOT_FOUND, status=404, error="HTTP 404")
        return FetchResult(url=url, outcome=FetchOutcome.OK, status=200, body=route.encode())


def _material(text: str | None) -> CollectionResult:
    """The material-8-K collection for one synthetic 8-K whose EX-99 says
    ``text`` (``None``: no 8-K at all in scope)."""
    if text is None:
        recent = {"form": [], "filingDate": [], "items": [], "accessionNumber": [],
                  "primaryDocument": []}
        return m8k.collect_material_8k_disclosures(
            TICKER, cik=CIK, recent=recent, http=FakeSec({}), today=TODAY,
        )
    index = json.dumps({"directory": {"item": [
        {"name": f"{ACC}-index.htm", "type": "text.gif"},
        {"name": PRIMARY, "type": "8-K"}, {"name": EX991, "type": "EX-99.1"},
    ]}})
    routes = {
        f"{BASE}/index.json": index, f"{BASE}/{PRIMARY}": PRIMARY_BODY,
        f"{BASE}/{EX991}": _ex99(text),
    }
    recent = {"form": ["8-K"], "filingDate": ["2026-09-23"], "items": ["1.01,3.02,9.01"],
              "accessionNumber": [ACC], "primaryDocument": [PRIMARY]}
    return m8k.collect_material_8k_disclosures(
        TICKER, cik=CIK, recent=recent, http=FakeSec(routes), today=TODAY,
    )


class _AvailableNullResearchProvider(NullResearchProvider):
    def available(self):
        return True, ""


def _install_noise_free_pipeline_harness(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_module, "assess_completeness",
        lambda **_k: CompletenessResult(coverage={
            d: DomainCoverage(domain=d, status=SearchStatus.SEARCHED)
            for d in REQUIRED_RESEARCH_DOMAINS
        }),
    )
    monkeypatch.setattr(
        pipeline_module, "assess_evidence_sufficiency",
        lambda **_k: EvidenceSufficiencyMatrix(decision_grade_fact_total=1),
    )
    monkeypatch.setattr(
        pipeline_module, "escalate", lambda facts, _p, *, company, **_k: (list(facts), EscalationReport()),
    )
    real_run_agent = Pipeline._run_agent
    noisy = {"competitive", "microstructure", "kill_agent"}

    def patched(self, agent, guard, result, **kwargs):
        before_status, before_len = result.context.status, len(result.failures)
        output = real_run_agent(self, agent, guard, result, **kwargs)
        if agent.agent_id in noisy:
            del result.failures[before_len:]
            result.context.status = before_status
        return output

    monkeypatch.setattr(Pipeline, "_run_agent", patched)


def _base() -> tuple[str, float, tuple, CollectionResult]:
    fixtures = FixtureCollector(Path(__file__).resolve().parent.parent.parent / "data" / "fixtures")
    metadata = fixtures.metadata(TICKER)
    company = metadata["company_name"]
    return company, metadata["price"], tuple(metadata.get("aliases", ())), fixtures.collect(TICKER, company)


def _pipeline(repo) -> Pipeline:
    return Pipeline(repo, NullSearchProvider(), today=TODAY, research=_AvailableNullResearchProvider())


def _run(monkeypatch, material: CollectionResult, *, run_id="c6-run", repo=None, resume=False):
    _install_noise_free_pipeline_harness(monkeypatch)
    monkeypatch.setattr(resume_module, "is_stale", lambda _f, _t: False)
    repo = repo or Repository(open_db(":memory:"))
    company, price, aliases, base = _base()
    result = _pipeline(repo).run(
        TICKER, company, [base, material], price=price, aliases=aliases, run_id=run_id, resume=resume,
    )
    return result, repo


def _reasons(result, prefix: str) -> list[str]:
    return [r for r in result.verdict.blocking_verification_required if r.startswith(prefix)]


# ---- (a) the financing condition blocks on its own -----------------------------------
def test_c6_baseline_with_no_8k_reaches_a_genuine_action(monkeypatch):
    result, _ = _run(monkeypatch, _material(None))
    assert result.context.status is RunStatus.COMPLETE and result.failures == []
    assert result.verdict.research_status is ResearchStatus.COMPLETE
    assert result.verdict.action is not None


def test_c6_a_financing_condition_alone_stops_the_action_with_an_explicit_reason(monkeypatch):
    material = _material(FINANCING)
    assert [f.claim for f in material.raw_facts] == [FINANCING] and material.errors == []
    result, _ = _run(monkeypatch, material)

    (reason,) = _reasons(result, FINANCING_PREFIX)
    assert f"{BASE}/{EX991}" in reason and "filed 2026-09-23" in reason and "event 2026-09-16" in reason
    assert "COMPANY_CLAIM" in reason and FINANCING[:60] in reason
    # nothing else is wrong with this run: only the financing condition blocks
    assert not _reasons(result, m8k.UNVERIFIED_PREFIX)
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    # no kill level was invented for it
    assert all(f.title != "Financing condition" for a in result.verdict.kill_gate.assessments
               for f in a.findings)


@pytest.mark.parametrize("text", [UNCONDITIONAL, SPECULATION])
def test_c6_ordinary_financing_or_speculation_does_not_block(monkeypatch, text):
    material = _material(text)
    assert material.raw_facts == []
    result, _ = _run(monkeypatch, material)
    assert not _reasons(result, FINANCING_PREFIX)
    assert result.context.status is RunStatus.COMPLETE and result.verdict.action is not None


# ---- (b) scope ID is saved, and both resume paths refuse another scope ---------------
def _first_run(monkeypatch, material, *, crash_before_verify: bool, expect_complete: bool = True):
    _install_noise_free_pipeline_harness(monkeypatch)
    monkeypatch.setattr(resume_module, "is_stale", lambda _f, _t: False)
    repo = Repository(open_db(":memory:"))
    company, price, aliases, base = _base()
    run_id = "c6-scope"
    kwargs = {"price": price, "aliases": aliases, "run_id": run_id}
    if crash_before_verify:
        def boom(*_a, **_k):
            raise RuntimeError("simulated crash")

        with monkeypatch.context() as m:
            m.setattr(pipeline_module, "run_full_evidence_integrity_pass", boom)
            with pytest.raises(RuntimeError, match="simulated crash"):
                _pipeline(repo).run(TICKER, company, [base, material], **kwargs)
        stages = {c.stage for c in repo.checkpoints(run_id)}
        assert "collect" in stages and "verify" not in stages
    else:
        first = _pipeline(repo).run(TICKER, company, [base, material], **kwargs)
        assert (first.context.status is RunStatus.COMPLETE) is expect_complete
    return repo, run_id


def _edit_material_summary(repo, run_id, edit) -> None:
    row = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
        (run_id,),
    ).fetchone()
    payload = json.loads(row["payload"])
    summary = next(c for c in payload["collectors"] if c["collector"] == m8k.MATERIAL_8K_COLLECTOR)
    edit(summary)
    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = ? AND stage = '_collect_snapshot'",
        (json.dumps(payload), run_id),
    )
    repo.conn.commit()


def _resume(monkeypatch, repo, run_id):
    fresh_http = FakeSec({})  # a resume must never fetch: any GET would be recorded here
    company, price, aliases, base = _base()
    # this invocation's own collection is deliberately CLEAN and carries the CURRENT scope
    result = _pipeline(repo).run(
        TICKER, company, [base, _material(None)], price=price, aliases=aliases,
        run_id=run_id, resume=True,
    )
    assert fresh_http.requested == []
    return result


PATHS = pytest.mark.parametrize("crash_before_verify", [True, False], ids=["collect-path", "verify-path"])


@PATHS
def test_c6_the_collect_snapshot_records_a_structured_scope_id(monkeypatch, crash_before_verify):
    repo, run_id = _first_run(monkeypatch, _material(None), crash_before_verify=crash_before_verify)
    row = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
        (run_id,),
    ).fetchone()
    summary = next(c for c in json.loads(row["payload"])["collectors"]
                   if c["collector"] == m8k.MATERIAL_8K_COLLECTOR)
    scope = summary["scope"]
    assert scope["id"] == m8k.MATERIAL_8K_SCOPE_ID
    assert scope["items"] == sorted(m8k.MATERIAL_8K_ITEMS)
    assert {"1.01", "2.03", "3.02"} <= set(scope["items"])
    assert scope["forms"] == sorted(m8k.MATERIAL_8K_FORMS)
    assert scope["max_filings"] == m8k.MAX_FILINGS == 12 and scope["max_outbound_gets"] == 48
    assert scope["lookback_days"] == 365


@PATHS
def test_c6_resume_with_the_current_scope_adds_no_scope_reason(monkeypatch, crash_before_verify):
    repo, run_id = _first_run(monkeypatch, _material(None), crash_before_verify=crash_before_verify)
    result = _resume(monkeypatch, repo, run_id)
    assert not _reasons(result, m8k.UNVERIFIED_PREFIX)
    assert result.resume_plan.resume_from_index > 0


@PATHS
@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.pop("scope", None),                                  # legacy record: no scope at all
        lambda s: s.__setitem__("scope", {"id": "material-8k-scope/v1"}),  # an older rule
        lambda s: s.__setitem__("scope", {}),                            # empty
        lambda s: s.__setitem__("scope", "material-8k-scope/v1"),        # malformed
        lambda s: s["scope"].__setitem__("id", m8k.MATERIAL_8K_SCOPE_ID.replace("1.01,", "")),  # items drifted
    ],
    ids=["missing", "older-rule", "empty", "malformed", "item-set-differs"],
)
def test_c6_resume_refuses_a_record_written_under_another_scope(
    monkeypatch, crash_before_verify, edit,
):
    repo, run_id = _first_run(monkeypatch, _material(None), crash_before_verify=crash_before_verify)
    _edit_material_summary(repo, run_id, edit)
    result = _resume(monkeypatch, repo, run_id)

    reasons = _reasons(result, m8k.UNVERIFIED_PREFIX)
    assert len(reasons) == 1
    assert "scope" in reasons[0] and m8k.MATERIAL_8K_SCOPE_ID in reasons[0]
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION


@PATHS
def test_c6_a_restored_record_is_not_mixed_with_this_invocations_clean_read(
    monkeypatch, crash_before_verify,
):
    """Run 1 read an 8-K that turned out unread-able; run 2's fresh read is
    clean and current. The restored (run 1) record still decides."""
    unread = _material(FINANCING)
    unread.errors.append(f"{m8k.UNVERIFIED_PREFIX} 8-K accession {ACC} filed 2026-09-23 -- synthetic")
    repo, run_id = _first_run(
        monkeypatch, unread, crash_before_verify=crash_before_verify, expect_complete=False,
    )
    result = _resume(monkeypatch, repo, run_id)
    assert any(ACC in r for r in _reasons(result, m8k.UNVERIFIED_PREFIX))
    # the restored financing fact still decides too (judged from the restored evidence)
    assert _reasons(result, FINANCING_PREFIX)
    assert result.verdict.action is None
