"""Resume must not lose an earlier invocation's explicit Literature
Document-First incompleteness.

Run 1 makes an explicit NCT acquisition (``--document-first-literature
--literature-nct-id``). Run 2 resumes the same run_id WITHOUT the explicit
flags, so its own ``direct_acquisition_info`` is empty and its own
``collection_results`` carry no Literature result. The restored base
evidence is still run 1's, and run 1's completeness record has to reach the
final Literature completeness judgement:

* resume from the VERIFY snapshot (which persists ``direct_acquisition_
  info``): an incomplete run-1 acquisition must stay a blocking reason;
  a complete one must not invent one;
* resume from the COLLECT snapshot, re-running Integrity (no
  ``direct_acquisition_info`` is persisted there): coverage cannot be
  proven, so it is never presumed complete -- fail closed.

Both resume paths must also make zero Adaptive Literature requests (commit
803cad3's own guard), against a plan that WOULD be READY without it: the
base carries a confirmed DEMOBIO program identity.

Non-vacuity: every resume appends a "RESUMED RUN" entry to
``result.failures``, which on its own already yields RunStatus=
INCOMPLETE_RESEARCH and Action=None. Those two assertions therefore cannot
by themselves prove anything about Literature. The discriminating signal is
``research_status=BLOCKED_PENDING_VERIFICATION`` with the Literature reason
in ``blocking_verification_required``, measured against the complete-
acquisition control below, which resumes to plain INCOMPLETE with no
blocking reason at all. The noise-free harness (copied from
tests/unit/test_literature_pipeline_integration.py, per this repo's
no-cross-test-import convention) removes the fixture's unrelated gates.

Entirely offline: FakeHttpClient only, plus the socket-level guard.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

import investment_research.orchestrator.pipeline as pipeline_module
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator import resume as resume_module
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research.escalation import EscalationReport
from investment_research.research.literature_pipeline_integration import (
    bundle_diagnostics,
    run_literature_pipeline_acquisition,
    validate_literature_pipeline_request,
)
from investment_research.research.provider import DomainCoverage, NullResearchProvider
from investment_research.schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    FactCategory,
    Provenance,
    ResearchStatus,
    RunStatus,
    SearchStatus,
    SourceTier,
)
from investment_research.schemas.fact import RawFact, Source
from investment_research.scoring.adaptive_acquisition_plan import AcquisitionPlanStatus
from investment_research.scoring.completeness import CompletenessResult
from investment_research.scoring.evidence_sufficiency import EvidenceSufficiencyMatrix
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
    ProgramCandidateEvidence,
)
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unit"))
import _literature_fixture_support as fx  # noqa: E402
from _network_guard import forbid_external_network_autouse  # noqa: E402, F401

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)
TICKER = "DEMOBIO"
NCT_ID = "NCT77778888"
PMIDS = ["90000015", "90000016"]
_ENV = {"IRA_NCBI_TOOL": "test-tool", "IRA_NCBI_EMAIL": "test@example.com"}
_EPMC_EMPTY = '{"resultList": {"result": []}}'
_LITERATURE_REASON_MARKER = "Literature Document-First"


def _with_courtesy(url: str) -> str:
    return f"{url}&tool=test-tool&email=test%40example.com"


def _literature_http(*, complete: bool) -> fx.FakeHttpClient:
    """``complete=False`` leaves the second PMID's Europe PMC search
    unanswered (FakeHttpClient treats it as a 404) -- a genuinely
    incomplete acquisition, not a hand-edited diagnostics dict."""
    responses = {
        _with_courtesy(fx.esearch_url(f"{NCT_ID}[si]", 20)): fx.ok(fx.esearch_response(PMIDS)),
        _with_courtesy(fx.efetch_url(PMIDS)): fx.ok(fx.fixture_text("batch_articleset.xml")),
        fx.europepmc_search_url(PMIDS[0]): fx.ok(_EPMC_EMPTY),
    }
    if complete:
        responses[fx.europepmc_search_url(PMIDS[1])] = fx.ok(_EPMC_EMPTY)
    return fx.FakeHttpClient(responses=responses)


class _AvailableNullResearchProvider(NullResearchProvider):
    def available(self):
        return True, ""


def _install_noise_free_pipeline_harness(monkeypatch) -> None:
    def fake_assess_completeness(**_kwargs):
        return CompletenessResult(
            coverage={
                domain: DomainCoverage(domain=domain, status=SearchStatus.SEARCHED)
                for domain in REQUIRED_RESEARCH_DOMAINS
            }
        )

    def fake_assess_evidence_sufficiency(**_kwargs):
        return EvidenceSufficiencyMatrix(decision_grade_fact_total=1)

    def fake_escalate(facts, _provider, *, company, **_kwargs):
        return list(facts), EscalationReport()

    monkeypatch.setattr(pipeline_module, "assess_completeness", fake_assess_completeness)
    monkeypatch.setattr(
        pipeline_module, "assess_evidence_sufficiency", fake_assess_evidence_sufficiency
    )
    monkeypatch.setattr(pipeline_module, "escalate", fake_escalate)

    real_run_agent = Pipeline._run_agent
    noisy_agent_ids = {"competitive", "microstructure", "kill_agent"}

    def patched_run_agent(self, agent, guard, result, **kwargs):
        before_status = result.context.status
        before_len = len(result.failures)
        output = real_run_agent(self, agent, guard, result, **kwargs)
        if agent.agent_id in noisy_agent_ids:
            del result.failures[before_len:]
            result.context.status = before_status
        return output

    monkeypatch.setattr(Pipeline, "_run_agent", patched_run_agent)


def _base_collection_results() -> tuple[str, float, tuple, list[CollectionResult]]:
    fixture_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fixtures"
    fixtures = FixtureCollector(fixture_dir)
    metadata = fixtures.metadata(TICKER)
    company = metadata["company_name"]
    source = Source(
        source_id="src_resume_lit_sec", url="https://www.sec.gov/Archives/x/8-K.htm",
        title="Doc", tier=SourceTier.TIER_1, publisher="SEC", published_date="2026-09-01",
    )
    fact = RawFact(
        ticker=TICKER, category=FactCategory.CLINICAL, claim="A registered claim.",
        source=source, company_claim=True, collector="resume_lit_identity_fixture",
    )
    identity = CollectionResult(
        collector="resume_lit_identity_fixture", raw_facts=[fact], sources=[source],
        provenance=Provenance.FIXTURE,
        company_identity_evidence=CompanyIdentityEvidence(
            ticker=TICKER, cik=1234567, sec_official_name=company,
            source_id=source.source_id, source_tier=source.tier,
            retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        ),
        program_candidate_evidence=(
            ProgramCandidateEvidence(
                nct_id=NCT_ID, lead_sponsor=company, overall_status="RECRUITING",
                source_id=source.source_id, source_tier=source.tier,
                retrieved_at=source.retrieved_at, content_hash=source.content_hash,
                supporting_fact_ids=(fact.fact_id(),),
            ),
        ),
    )
    base = fixtures.collect(TICKER, company)
    return company, metadata["price"], tuple(metadata.get("aliases", ())), [base, identity]


def _pipeline(repo, **kwargs) -> Pipeline:
    return Pipeline(
        repo, NullSearchProvider(), today=TODAY, research=_AvailableNullResearchProvider(),
        **kwargs,
    )


def _literature_reasons(result) -> list[str]:
    return [
        r for r in result.verdict.blocking_verification_required
        if _LITERATURE_REASON_MARKER in r
    ]


def _explicit_bundle(kind: str):
    """``complete``/``incomplete`` are real acquisitions over FakeHttpClient;
    ``refused`` is the defensive credential re-check refusing before any
    request (no CollectionResult at all)."""
    request = validate_literature_pipeline_request(
        enabled=True, pmid=None, nct_id=NCT_ID, live=True,
        use_fixtures=False, use_corpus=False, env=_ENV,
    )
    if kind == "refused":
        bundle = run_literature_pipeline_acquisition(
            request, ticker=TICKER, http_client=_literature_http(complete=True), env={},
        )
        assert bundle.refused and bundle.collection_result is None
        return bundle
    bundle = run_literature_pipeline_acquisition(
        request, ticker=TICKER, http_client=_literature_http(complete=kind == "complete"), env=_ENV,
    )
    assert bundle.coverage_complete is (kind == "complete")
    return bundle


def _explicit_pipeline_inputs(bundle) -> tuple[dict, list]:
    kwargs = {
        "chunks": list(bundle.chunks),
        "direct_acquisition_info": bundle_diagnostics(bundle, enabled=True),
    }
    results = [bundle.collection_result] if bundle.collection_result is not None else []
    return kwargs, results


def _run_explicit_then_resume(
    monkeypatch, *, first: str, crash_before_verify: bool = False,
    second_explicit: str | None = None, drop_verify_snapshot: bool = False,
):
    """Run 1 with an explicit NCT acquisition of kind ``first``; run 2
    resumes the same run_id with a READY-capable Adaptive transport and,
    unless ``second_explicit`` is given, no explicit flags at all."""
    _install_noise_free_pipeline_harness(monkeypatch)
    # The fixture articles are older than their category's freshness
    # threshold relative to TODAY, which would make the resume re-collect
    # from scratch. Hold them fresh so the resume really restores run 1.
    monkeypatch.setattr(resume_module, "is_stale", lambda _fact, _today: False)
    repo = Repository(open_db(":memory:"))
    company, price, aliases, base = _base_collection_results()
    run_id = "resume-explicit-lit-run"

    first_kwargs, first_extra = _explicit_pipeline_inputs(_explicit_bundle(first))
    first_adaptive = _literature_http(complete=True)
    first_pipeline = _pipeline(
        repo, adaptive_http_client=first_adaptive, adaptive_env=dict(_ENV), **first_kwargs,
    )
    first_args = (TICKER, company, [*base, *first_extra])
    first_run_kwargs = {"price": price, "aliases": aliases, "run_id": run_id}
    if crash_before_verify:
        # Simulate a crash after the "collect" checkpoint and snapshot are
        # durably written but before Stage 2 completes, so the resume
        # restores the COLLECT snapshot and re-runs Integrity.
        def crash(*_a, **_k):
            raise RuntimeError("simulated crash before verify")

        with monkeypatch.context() as m:
            m.setattr(pipeline_module, "run_full_evidence_integrity_pass", crash)
            with pytest.raises(RuntimeError, match="simulated crash"):
                first_pipeline.run(*first_args, **first_run_kwargs)
        stages = {c.stage for c in repo.checkpoints(run_id)}
        assert "collect" in stages and "verify" not in stages
    else:
        first_result = first_pipeline.run(*first_args, **first_run_kwargs)
        assert first_result.adaptive_acquisition_plan.status is (
            AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
        )
        assert bool(_literature_reasons(first_result)) is (first != "complete")
    assert first_adaptive.requested_urls == []

    if drop_verify_snapshot:
        repo.conn.execute(
            "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = '_verify_snapshot'",
            (run_id,),
        )
        repo.conn.commit()

    second_kwargs: dict = {}
    second_extra: list = []
    if second_explicit is not None:
        second_kwargs, second_extra = _explicit_pipeline_inputs(_explicit_bundle(second_explicit))
    second_adaptive = _literature_http(complete=True)
    second = _pipeline(
        repo, adaptive_http_client=second_adaptive, adaptive_env=dict(_ENV), **second_kwargs,
    )
    result = second.run(
        TICKER, company, [*base, *second_extra], price=price, aliases=aliases,
        run_id=run_id, resume=True,
    )
    assert second_adaptive.requested_urls == []
    if second_explicit is None:
        assert result.direct_acquisition_info == {}  # no explicit flags this invocation
    return result


def _assert_blocked_by_literature(result) -> list[str]:
    reasons = _literature_reasons(result)
    assert reasons, result.verdict.blocking_verification_required
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.action is None
    assert all(r in result.failures for r in reasons)
    return reasons


# =============================================================================
# Resume from the VERIFY snapshot (persists direct_acquisition_info)
# =============================================================================
def test_resume_from_verify_snapshot_keeps_earlier_incomplete_explicit_literature_blocking(
    monkeypatch,
):
    result = _run_explicit_then_resume(monkeypatch, first="incomplete")
    assert result.resume_plan.resume_from_index > resume_module.stage_index("verify")
    assert not result.resume_snapshot_failures
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    reasons = _assert_blocked_by_literature(result)
    assert any("coverage" in r.lower() for r in reasons)


def test_resume_from_verify_snapshot_of_complete_explicit_literature_adds_no_blocking(
    monkeypatch,
):
    """Control: proves the resume itself contributes no blocking reason, so
    the incomplete case above is BLOCKED because of Literature alone."""
    result = _run_explicit_then_resume(monkeypatch, first="complete")
    assert result.resume_plan.resume_from_index > resume_module.stage_index("verify")
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    assert result.verdict.blocking_verification_required == ()
    assert result.verdict.research_status is ResearchStatus.INCOMPLETE  # RESUMED RUN only
    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.action is None
    assert not any(_LITERATURE_REASON_MARKER in f for f in result.failures)


def test_resume_from_verify_snapshot_keeps_earlier_refused_explicit_literature_blocking(
    monkeypatch,
):
    """A refusal leaves no Literature collector in the collect snapshot at
    all; only the verify snapshot's persisted direct_acquisition_info
    records that the explicit acquisition was requested and refused."""
    result = _run_explicit_then_resume(monkeypatch, first="refused")
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    reasons = _assert_blocked_by_literature(result)
    assert any("refused" in r for r in reasons)


def test_this_invocations_explicit_acquisition_keeps_precedence_without_masking_restored_gap(
    monkeypatch,
):
    """THIS invocation's explicit acquisition stays the direct_acquisition_
    info on the result, but its completeness never masks the restored base
    evidence's own gap -- the restored facts are the ones being judged."""
    result = _run_explicit_then_resume(monkeypatch, first="incomplete", second_explicit="complete")
    assert result.direct_acquisition_info["feature_enabled"] is True
    assert result.direct_acquisition_info["coverage_complete"] is True
    reasons = _assert_blocked_by_literature(result)
    assert all(r.startswith("Restored from an earlier invocation") for r in reasons)


def test_missing_verify_snapshot_keeps_the_existing_fail_closed_path(monkeypatch):
    result = _run_explicit_then_resume(monkeypatch, first="complete", drop_verify_snapshot=True)
    assert result.resume_snapshot_failures
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.verdict.action is None
    # Unchanged: the snapshot failure is the blocking reason; no restored-
    # Literature judgement is layered on top of an unavailable base.
    assert not any(
        "Restored from an earlier invocation" in r or "cannot be shown complete" in r
        for r in result.verdict.blocking_verification_required
    )


# =============================================================================
# Resume from the COLLECT snapshot, re-running Integrity (no persisted
# direct_acquisition_info): never presumed complete
# =============================================================================
@pytest.mark.parametrize("first", ["incomplete", "complete"])
def test_resume_from_collect_snapshot_fails_closed_on_unprovable_literature_coverage(
    monkeypatch, first,
):
    result = _run_explicit_then_resume(monkeypatch, first=first, crash_before_verify=True)
    assert result.resume_plan.resume_from_index == resume_module.stage_index("verify")
    assert not result.resume_snapshot_failures
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    reasons = _assert_blocked_by_literature(result)
    if first == "incomplete":
        # The degraded Literature collector outcome IS persisted in the
        # collect snapshot -- a concrete reason, reported as such.
        assert any("did not complete: collector outcome was" in r for r in reasons)
    else:
        # Nothing persisted on this path can prove coverage: fail closed.
        assert reasons == [
            r for r in result.verdict.blocking_verification_required
            if "cannot be shown complete" in r
        ]
