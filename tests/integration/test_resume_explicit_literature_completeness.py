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

import json
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
from investment_research.storage.repository import Repository, ResumeSnapshotCorrupted

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
    if bundle is None:  # no explicit flags on that invocation
        return {}, []
    kwargs = {
        "chunks": list(bundle.chunks),
        "direct_acquisition_info": bundle_diagnostics(bundle, enabled=True),
    }
    results = [bundle.collection_result] if bundle.collection_result is not None else []
    return kwargs, results


def _run_explicit_then_resume(
    monkeypatch, *, first: str, crash_before_verify: bool = False,
    second_explicit: str | None = None, drop_verify_snapshot: bool = False,
    edit_collect_snapshot=None, before_second=None, expect_adaptive_http: bool = False,
    return_repo: bool = False,
):
    """Run 1 with an explicit NCT acquisition of kind ``first`` (``none``:
    no explicit flags at all); run 2 resumes the same run_id with a
    READY-capable Adaptive transport and, unless ``second_explicit`` is
    given, no explicit flags at all. ``edit_collect_snapshot`` rewrites the
    stored collect-snapshot payload dict between the two runs."""
    _install_noise_free_pipeline_harness(monkeypatch)
    # The fixture articles are older than their category's freshness
    # threshold relative to TODAY, which would make the resume re-collect
    # from scratch. Hold them fresh so the resume really restores run 1.
    monkeypatch.setattr(resume_module, "is_stale", lambda _fact, _today: False)
    repo = Repository(open_db(":memory:"))
    company, price, aliases, base = _base_collection_results()
    run_id = "resume-explicit-lit-run"

    first_kwargs, first_extra = _explicit_pipeline_inputs(
        None if first == "none" else _explicit_bundle(first)
    )
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
        with monkeypatch.context() as m:
            m.setattr(pipeline_module, "run_full_evidence_integrity_pass", _raise_crash)
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

    if edit_collect_snapshot is not None:
        row = repo.conn.execute(
            "SELECT payload FROM run_checkpoints WHERE run_id = ? AND stage = '_collect_snapshot'",
            (run_id,),
        ).fetchone()
        payload = json.loads(row["payload"])
        edit_collect_snapshot(payload)
        repo.conn.execute(
            "UPDATE run_checkpoints SET payload = ? WHERE run_id = ? AND stage = '_collect_snapshot'",
            (json.dumps(payload), run_id),
        )
        repo.conn.commit()

    if drop_verify_snapshot:
        repo.conn.execute(
            "DELETE FROM run_checkpoints WHERE run_id = ? AND stage = '_verify_snapshot'",
            (run_id,),
        )
        repo.conn.commit()

    if before_second is not None:
        before_second(monkeypatch)
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
    assert bool(second_adaptive.requested_urls) is expect_adaptive_http
    if second_explicit is None:
        assert result.direct_acquisition_info == {}  # no explicit flags this invocation
    return (result, repo) if return_repo else result


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
# Resume from the COLLECT snapshot, re-running Integrity. Stage 1 now
# records its own direct_acquisition_info there; a snapshot without that
# record (legacy format) fails closed.
# =============================================================================
_LEGACY_MARKER = "does not record direct_acquisition_info"


def _assert_collect_path(result) -> None:
    assert result.resume_plan.resume_from_index == resume_module.stage_index("verify")
    assert not result.resume_snapshot_failures


def test_collect_snapshot_resume_of_refused_explicit_literature_skips_adaptive_and_blocks(
    monkeypatch,
):
    """Regression: a refusal leaves no Literature collector entry, and the
    run stopped before the verify snapshot existed -- only the collect
    snapshot's Stage-1 record can carry it. Before that record existed this
    resume planned READY and made Adaptive requests."""
    result = _run_explicit_then_resume(monkeypatch, first="refused", crash_before_verify=True)
    _assert_collect_path(result)
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    reasons = _assert_blocked_by_literature(result)
    assert any("refused" in r for r in reasons)


def test_collect_snapshot_resume_of_incomplete_explicit_literature_blocks(monkeypatch):
    result = _run_explicit_then_resume(monkeypatch, first="incomplete", crash_before_verify=True)
    _assert_collect_path(result)
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    reasons = _assert_blocked_by_literature(result)
    assert any("coverage" in r.lower() for r in reasons)


def test_collect_snapshot_resume_of_complete_explicit_literature_adds_no_blocking(monkeypatch):
    """The Stage-1 record now proves coverage, so nothing is invented --
    the same outcome as the verify-path control."""
    result = _run_explicit_then_resume(monkeypatch, first="complete", crash_before_verify=True)
    _assert_collect_path(result)
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    assert result.verdict.blocking_verification_required == ()
    assert result.verdict.research_status is ResearchStatus.INCOMPLETE  # RESUMED RUN only
    assert result.verdict.action is None


def test_collect_snapshot_resume_reapplies_integrity_consistency_to_the_restored_record(
    monkeypatch,
):
    """A Stage-1 record of a complete acquisition is pre-Integrity. If this
    resume's own Integrity pass drops restored Literature facts, the record
    is complete no longer -- exactly Integrity item 8's rule for a fresh run."""
    import dataclasses

    def drop_literature_facts(m):
        real = pipeline_module.run_full_evidence_integrity_pass

        def wrapped(pass_input, repo):
            out = real(pass_input, repo)
            kept = tuple(
                f for f in out.verified_facts if not f.notes.startswith("collected_by=literature_")
            )
            assert len(kept) < len(out.verified_facts)  # non-vacuous: something was dropped
            return dataclasses.replace(out, verified_facts=kept)

        m.setattr(pipeline_module, "run_full_evidence_integrity_pass", wrapped)

    result = _run_explicit_then_resume(
        monkeypatch, first="complete", crash_before_verify=True, before_second=drop_literature_facts,
    )
    _assert_collect_path(result)
    reasons = _assert_blocked_by_literature(result)
    assert any("did not survive Evidence Integrity" in r for r in reasons)


def test_collect_snapshot_resume_with_recorded_no_explicit_literature_plans_like_fresh(
    monkeypatch,
):
    """A new-format snapshot that RECORDS "no explicit acquisition" ({}) is
    not the legacy case: the plan is READY and the Adaptive acquisition runs,
    exactly as a fresh run with these flags would."""
    result = _run_explicit_then_resume(
        monkeypatch, first="none", crash_before_verify=True, expect_adaptive_http=True,
    )
    _assert_collect_path(result)
    assert result.adaptive_acquisition_plan.status is AcquisitionPlanStatus.READY
    assert result.adaptive_literature_diagnostics["outcome"] == "COMPLETE"
    assert _literature_reasons(result) == []
    assert result.verdict.blocking_verification_required == ()


@pytest.mark.parametrize("first", ["none", "refused", "complete"])
def test_legacy_collect_snapshot_without_the_record_fails_closed(monkeypatch, first):
    """A collect snapshot written before the field existed: never read as
    "no explicit acquisition", whatever run 1 actually did."""
    result, repo = _run_explicit_then_resume(
        monkeypatch, first=first, crash_before_verify=True,
        edit_collect_snapshot=lambda payload: payload.pop("direct_acquisition_info"),
        return_repo=True,
    )
    assert any(_LEGACY_MARKER in f for f in result.resume_snapshot_failures)
    assert any(_LEGACY_MARKER in r for r in result.verdict.blocking_verification_required)
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.context.status is RunStatus.INCOMPLETE_RESEARCH
    assert result.verdict.action is None
    assert result.adaptive_literature_diagnostics["outcome"] == (
        "NOT_ATTEMPTED_BASE_EVIDENCE_UNAVAILABLE"
    )
    # No "verify" checkpoint: a later resume hits the same fail-closed path
    # instead of inheriting a verify snapshot built on a guess.
    assert "verify" not in {c.stage for c in repo.checkpoints(result.context.run_id)}


def test_collect_path_resume_record_survives_into_a_later_verify_path_resume(monkeypatch):
    """Run 2 (collect-path resume) writes the verify snapshot with the
    COLLECTING invocation's record, not its own empty one -- so run 3
    (verify-path resume) still sees the refusal."""
    result, repo = _run_explicit_then_resume(
        monkeypatch, first="refused", crash_before_verify=True, return_repo=True,
    )
    assert repo.verify_snapshot_for_resume(result.context.run_id).direct_acquisition_info[
        "refused"
    ] is True

    company, price, aliases, base = _base_collection_results()
    third_adaptive = _literature_http(complete=True)
    third = _pipeline(repo, adaptive_http_client=third_adaptive, adaptive_env=dict(_ENV))
    third_result = third.run(
        TICKER, company, base, price=price, aliases=aliases,
        run_id=result.context.run_id, resume=True,
    )
    assert third_result.resume_plan.resume_from_index > resume_module.stage_index("verify")
    assert third_adaptive.requested_urls == []
    assert third_result.direct_acquisition_info == {}
    assert third_result.adaptive_acquisition_plan.status is (
        AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    )
    assert any("refused" in r for r in _assert_blocked_by_literature(third_result))


def test_collect_path_this_invocations_explicit_acquisition_keeps_precedence(monkeypatch):
    result = _run_explicit_then_resume(
        monkeypatch, first="incomplete", crash_before_verify=True, second_explicit="complete",
    )
    _assert_collect_path(result)
    assert result.direct_acquisition_info["feature_enabled"] is True
    assert result.direct_acquisition_info["refused"] is False
    reasons = _assert_blocked_by_literature(result)
    assert any(r.startswith("Restored from an earlier invocation") for r in reasons)


# =============================================================================
# The collect snapshot contract itself
# =============================================================================
def test_collect_snapshot_records_exactly_the_sanitized_stage1_info_and_no_secret(monkeypatch):
    _install_noise_free_pipeline_harness(monkeypatch)
    repo = Repository(open_db(":memory:"))
    company, price, aliases, base = _base_collection_results()
    bundle = _explicit_bundle("incomplete")
    kwargs, extra = _explicit_pipeline_inputs(bundle)
    with monkeypatch.context() as m:
        m.setattr(pipeline_module, "run_full_evidence_integrity_pass", _raise_crash)
        with pytest.raises(RuntimeError):
            _pipeline(repo, **kwargs).run(
                TICKER, company, [*base, *extra], price=price, aliases=aliases, run_id="r1",
            )
    snapshot = repo.collect_snapshot_for_resume("r1")
    assert snapshot.direct_acquisition_info == json.loads(
        json.dumps(bundle_diagnostics(bundle, enabled=True), default=str)
    )
    raw = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = 'r1' AND stage = '_collect_snapshot'"
    ).fetchone()["payload"]
    stored = json.dumps(json.loads(raw)["direct_acquisition_info"])
    for secret in (*_ENV.values(), "email=", "tool=", "api_key="):
        assert secret not in stored


def _raise_crash(*_a, **_k):
    raise RuntimeError("simulated crash before verify")


def test_collect_snapshot_round_trip_distinguishes_recorded_empty_from_absent():
    repo = Repository(open_db(":memory:"))
    repo.save_collect_snapshot("rec", [], [], [], direct_acquisition_info={})
    repo.save_collect_snapshot("old", [], [], [])
    assert repo.collect_snapshot_for_resume("rec").direct_acquisition_info == {}
    assert repo.collect_snapshot_for_resume("old").direct_acquisition_info is None

    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = 'rec' AND stage = '_collect_snapshot'",
        (json.dumps({"facts": [], "sources": [], "collectors": [], "direct_acquisition_info": []}),),
    )
    repo.conn.commit()
    with pytest.raises(ResumeSnapshotCorrupted):
        repo.collect_snapshot_for_resume("rec")


def test_literature_raw_facts_carry_the_collector_names_pipeline_recognises():
    """The collect-path resume identifies restored Literature facts by the
    FactCollector notes stamp; every RawFact the acquisition emits must keep
    a ``literature_*`` collector name for that to hold."""
    bundle = _explicit_bundle("complete")
    assert bundle.collection_result.raw_facts
    prefix = pipeline_module._LITERATURE_RAW_FACT_NOTES_PREFIX
    assert all(
        f"collected_by={raw.collector}".startswith(prefix)
        for raw in bundle.collection_result.raw_facts
    )
