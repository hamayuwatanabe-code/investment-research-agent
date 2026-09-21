"""Phase 4.3C / Phase 4.3C Correction 1: the "full Evidence Integrity
pass" (``orchestrator/evidence_integrity_pass.py``), extracted from
``Pipeline.run()``'s Stage 2, and an offline 2-pass harness proving the
identical pass can safely run TWICE over a growing evidence set -- never
connected to Pipeline/CLI/HTTP/LLM in this phase.

Section map:
1. Single-pass behavioral spec (direct correctness of the extracted
   function -- the "current single-pass behavior" the refactor must
   preserve; the FULL existing test suite -- especially
   test_literature_pipeline_integration.py's Correction-1/2 tests and the
   DEMOBIO --fixtures regression -- is the primary before/after proof;
   this section adds focused, direct tests of the function's own contract).
2. Offline 2-pass harness: new facts always pass through Evidence
   Integrity, cross-pass full-set re-evaluation, second-pass quarantine,
   idempotence, Literature completeness cascade in pass 2.
3. Phase 4.3C Correction 1: no-bypass guarantee (the agent genuinely
   executes exactly once per call, twice across a 2-pass harness) and
   same-run_id audit-record identification (initial vs. adaptive).
4. Isolation: the new module, and pipeline.py's use of it, never import
   any Phase 4.3B Program Identity/Program Resolution module, HTTP client,
   or LLM.
"""

from __future__ import annotations

from pathlib import Path

from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.base import CollectionResult
from investment_research.orchestrator.evidence_integrity_pass import (
    LITERATURE_BRIDGE_COLLECTOR_LABEL,
    FullIntegrityPassInput,
    run_full_evidence_integrity_pass,
)
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import (
    EvidenceClass,
    FactCategory,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import RawFact, Source

RUN_ID = "test-run"
TICKER = "DEMOBIO"
COMPANY = "Demo Biotherapeutics Inc"


# =============================================================================
# Helpers
# =============================================================================
def _source(
    source_id: str, *, url: str, tier: SourceTier = SourceTier.TIER_1, title: str = "Doc",
    published_date: str = "2026-01-01",
) -> Source:
    return Source(source_id=source_id, url=url, title=title, tier=tier, published_date=published_date)


def _raw_fact(
    claim: str, source: Source, *, category: FactCategory = FactCategory.CLINICAL,
    company_claim: bool = True,
) -> RawFact:
    return RawFact(
        ticker=TICKER, category=category, claim=claim, source=source,
        company_claim=company_claim,
    )


def _stage1_facts(raw_facts: list[RawFact], *, run_id: str = RUN_ID) -> list:
    """Real FactCollectorAgent execution -- the genuine "Stage-1 Facts"
    shape (baseline COMPANY_CLAIM/UNVERIFIED_CLAIM, NOT_VERIFIED) this
    pass must run Evidence Integrity over. Returns ``Fact`` objects, never
    ``RawFact`` -- the RawFact -> Fact conversion is FactCollectorAgent's
    own job, done here for real, never faked or skipped."""
    sources = [rf.source for rf in raw_facts]
    collection_result = CollectionResult(collector="test_collector", raw_facts=raw_facts, sources=sources)
    agent = FactCollectorAgent([collection_result])
    agent_input = AgentInput(agent_id=agent.agent_id, run_id=run_id, ticker=TICKER, company_name=COMPANY)
    output = agent.execute(agent_input)
    return list(output.facts)


def _pin(**overrides) -> FullIntegrityPassInput:
    kwargs = {
        "ticker": TICKER, "company_name": COMPANY, "run_id": RUN_ID,
        "pre_integrity_facts": (), "sources": (),
    }
    kwargs.update(overrides)
    return FullIntegrityPassInput(**kwargs)


def _count_agent_runs(monkeypatch) -> list:
    """Wraps EvidenceIntegrityAgent.run itself (never Agent.execute, and
    never a call counter living inside evidence_integrity_pass.py) so the
    count is a genuine, independent observation of how many times the
    agent's own logic actually executed -- returns the list of AgentInput
    objects it was called with, in order."""
    calls: list = []
    original_run = EvidenceIntegrityAgent.run

    def counting_run(self, data):
        calls.append(data)
        return original_run(self, data)

    monkeypatch.setattr(EvidenceIntegrityAgent, "run", counting_run)
    return calls


# =============================================================================
# 1. Single-pass behavioral spec
# =============================================================================
def test_clean_facts_are_verified_no_quarantine_no_failures(repo):
    source = _source("src_1", url="https://www.sec.gov/example", tier=SourceTier.TIER_1)
    raw = [_raw_fact("The company filed a 10-Q.", source, company_claim=True)]
    stage1 = _stage1_facts(raw)
    assert stage1[0].evidence_class == EvidenceClass.COMPANY_CLAIM  # pre-Integrity baseline

    out = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1), sources=(source,)), repo,
    )
    assert len(out.verified_facts) == 1
    assert out.quarantined_sources == ()
    assert out.failures == ()
    assert out.status_incomplete is False
    assert out.agent_run_record is not None
    # Phase 4.3C correction 1: production's default identity, unchanged
    # from before this correction.
    assert out.agent_run_record.agent_id == "evidence_integrity"


def test_malformed_source_is_quarantined_and_dependent_fact_excluded(repo):
    good_source = _source("src_good", url="https://www.sec.gov/good")
    bad_source = _source("src_bad", url="not-a-well-formed-url")
    raw = [
        _raw_fact("Good claim.", good_source, company_claim=True),
        _raw_fact("Bad claim.", bad_source, company_claim=True),
    ]
    stage1 = _stage1_facts(raw)

    out = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1), sources=(good_source, bad_source)), repo,
    )
    assert len(out.quarantined_sources) == 1
    assert out.quarantined_sources[0].source_id == "src_bad"
    assert out.status_incomplete is True
    assert any("quarantined malformed source" in f for f in out.failures)
    assert any("resting only on a quarantined source" in f for f in out.failures)
    surviving_claims = {f.claim for f in out.verified_facts}
    assert "Good claim." in surviving_claims
    assert "Bad claim." not in surviving_claims


def test_literature_consistency_flips_stale_coverage_complete_in_one_pass(repo):
    """The Phase 4.2B correction 1 logic, exercised at the pass level
    directly: a Literature Source that quarantines must flip
    direct_acquisition_info.coverage_complete, add exactly one
    unresolved_reasons entry."""
    lit_source = _source("src_lit", url="not-a-well-formed-url", tier=SourceTier.UNKNOWN)
    raw = RawFact(
        ticker=TICKER, category=FactCategory.SCIENCE, claim="A PubMed article exists.",
        source=lit_source, company_claim=False,
    )
    literature_collection_result = CollectionResult(
        collector=LITERATURE_BRIDGE_COLLECTOR_LABEL, raw_facts=[raw], sources=[lit_source],
    )
    stage1 = _stage1_facts([raw])

    out = run_full_evidence_integrity_pass(
        _pin(
            pre_integrity_facts=tuple(stage1), sources=(lit_source,),
            collection_results=(literature_collection_result,),
            direct_acquisition_info={"feature_enabled": True, "coverage_complete": True},
        ),
        repo,
    )
    assert out.quarantined_sources
    assert out.direct_acquisition_info["coverage_complete"] is False
    unresolved = out.direct_acquisition_info.get("unresolved_reasons", [])
    assert len(unresolved) == 1
    assert "quarantined" in unresolved[0].lower()


def test_literature_bridge_collector_label_stays_in_sync_with_pipeline_and_research_module():
    import investment_research.orchestrator.pipeline as pipeline_module
    from investment_research.research.literature_evidence_projection import BRIDGE_COLLECTOR_LABEL

    assert LITERATURE_BRIDGE_COLLECTOR_LABEL == BRIDGE_COLLECTOR_LABEL
    assert pipeline_module._LITERATURE_BRIDGE_COLLECTOR_LABEL == LITERATURE_BRIDGE_COLLECTOR_LABEL


# =============================================================================
# 2. Offline 2-pass harness
# =============================================================================
def test_pass_2_new_facts_are_never_left_unevaluated(repo):
    """A brand-new RawFact, run through FactCollectorAgent then through
    pass 2, must never keep its pre-Integrity baseline (UNVERIFIED_CLAIM/
    NOT_VERIFIED) -- EvidenceClass, verified_status, and is_decision_grade
    must all reflect a genuine Evidence Integrity assessment."""
    source_1 = _source("src_1", url="https://www.sec.gov/one")
    raw_1 = [_raw_fact("Initial claim.", source_1, company_claim=False)]
    stage1_initial = _stage1_facts(raw_1, run_id="pass1")

    pass1 = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1_initial), sources=(source_1,)), repo,
    )
    assert pass1.verified_facts[0].evidence_class == EvidenceClass.VERIFIED_FACT  # Tier-1, non-company

    # New RawFact, simulating an Adaptive-Acquisition-style addition --
    # run through FactCollectorAgent for real, never hand-built as an
    # already-"verified" Fact.
    source_2 = _source("src_2", url="https://www.sec.gov/two")
    raw_2 = [_raw_fact("A brand-new claim from pass 2.", source_2, company_claim=False)]
    stage1_new = _stage1_facts(raw_2, run_id="pass2")
    assert stage1_new[0].evidence_class == EvidenceClass.UNVERIFIED_CLAIM  # baseline, pre-Integrity
    assert stage1_new[0].verified_status == VerifiedStatus.NOT_VERIFIED
    assert stage1_new[0].is_decision_grade is False

    combined_raw = tuple(stage1_initial) + tuple(stage1_new)
    combined_sources = (source_1, source_2)
    pass2 = run_full_evidence_integrity_pass(
        _pin(
            pre_integrity_facts=combined_raw, sources=combined_sources,
            audit_agent_id="evidence_integrity_adaptive",
        ),
        repo,
    )
    new_fact_after = next(f for f in pass2.verified_facts if f.claim == "A brand-new claim from pass 2.")
    assert new_fact_after.evidence_class == EvidenceClass.VERIFIED_FACT  # no longer UNVERIFIED_CLAIM
    assert new_fact_after.verified_status == VerifiedStatus.VERIFIED  # no longer NOT_VERIFIED
    assert new_fact_after.is_decision_grade is True


def test_cross_pass_full_set_re_evaluation_changes_pass_1_fact_classification(repo):
    """Cross-pass full-set re-evaluation (never a simple append): a
    pass-1 company-claim fact with NO corroboration
    (independent_confirmation=False, COMPANY_CLAIM) must be RE-ASSESSED,
    not merely carried over unchanged, once pass 2 introduces a new,
    independent, non-company Tier-3 source making the SAME claim -- its
    evidence_class/verified_status/independent_confirmation must change
    for the SAME fact_id, proving the full combined set was genuinely
    re-evaluated. A simple append (Stage 3b's pattern) would leave the
    original Fact object's already-computed fields untouched.

    This is deliberately NOT a logical-contradiction test:
    EvidenceIntegrityAgent has no contradiction detector of its own (that
    is ContradictionAgent's job, an unrelated, later Stage 4 agent this
    phase does not touch or move) -- corroboration re-assessment is the
    real, correctly-scoped mechanism by which re-running Integrity over a
    combined set can change an EXISTING fact's own computed fields, and is
    what this test demonstrates and names precisely."""
    claim_text = "Primary endpoint was not met in the Phase 3 trial."
    company_source = _source("src_company", url="https://www.sec.gov/8k", tier=SourceTier.TIER_1)
    raw_1 = [_raw_fact(claim_text, company_source, category=FactCategory.CLINICAL, company_claim=True)]
    stage1_initial = _stage1_facts(raw_1, run_id="pass1")

    pass1 = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1_initial), sources=(company_source,)), repo,
    )
    original_fact_id = pass1.verified_facts[0].fact_id
    assert pass1.verified_facts[0].independent_confirmation is False
    assert pass1.verified_facts[0].evidence_class == EvidenceClass.COMPANY_CLAIM
    assert pass1.verified_facts[0].verified_status == VerifiedStatus.NOT_VERIFIED

    press_source = _source("src_press", url="https://www.example-press.test/article", tier=SourceTier.TIER_3)
    raw_2 = [
        _raw_fact(claim_text, press_source, category=FactCategory.CLINICAL, company_claim=False)
    ]
    stage1_new = _stage1_facts(raw_2, run_id="pass2")

    combined_raw = tuple(stage1_initial) + tuple(stage1_new)
    combined_sources = (company_source, press_source)
    pass2 = run_full_evidence_integrity_pass(
        _pin(
            pre_integrity_facts=combined_raw, sources=combined_sources,
            audit_agent_id="evidence_integrity_adaptive",
        ),
        repo,
    )

    reassessed = next(f for f in pass2.verified_facts if f.fact_id == original_fact_id)
    assert reassessed.independent_confirmation is True  # flipped by the new corroborating fact
    assert reassessed.evidence_class == EvidenceClass.VERIFIED_FACT  # upgraded from COMPANY_CLAIM
    assert reassessed.verified_status == VerifiedStatus.VERIFIED  # upgraded from NOT_VERIFIED
    # Never a simple append: appending would have kept the pass-1 output's
    # OWN fact object (independent_confirmation=False) as the final state.
    assert reassessed.independent_confirmation != pass1.verified_facts[0].independent_confirmation


def test_pass_2_quarantines_a_new_malformed_source_and_keeps_initial_valid_facts(repo):
    source_1 = _source("src_1", url="https://www.sec.gov/one")
    raw_1 = [_raw_fact("Initial valid claim.", source_1, company_claim=False)]
    stage1_initial = _stage1_facts(raw_1, run_id="pass1")

    pass1 = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1_initial), sources=(source_1,)), repo,
    )
    assert pass1.quarantined_sources == ()
    initial_fact_id = pass1.verified_facts[0].fact_id

    bad_source = _source("src_bad", url="not-a-well-formed-url")
    raw_2 = [_raw_fact("A new claim resting on a malformed source.", bad_source, company_claim=False)]
    stage1_new = _stage1_facts(raw_2, run_id="pass2")

    combined_raw = tuple(stage1_initial) + tuple(stage1_new)
    combined_sources = (source_1, bad_source)
    pass2 = run_full_evidence_integrity_pass(
        _pin(
            pre_integrity_facts=combined_raw, sources=combined_sources,
            audit_agent_id="evidence_integrity_adaptive",
        ),
        repo,
    )

    assert len(pass2.quarantined_sources) == 1
    assert pass2.quarantined_sources[0].source_id == "src_bad"
    assert pass2.status_incomplete is True
    surviving_ids = {f.fact_id for f in pass2.verified_facts}
    assert initial_fact_id in surviving_ids  # the initial valid fact survives
    assert not any(f.claim == "A new claim resting on a malformed source." for f in pass2.verified_facts)
    assert any("quarantined malformed source" in f for f in pass2.failures)


def test_idempotent_rerun_of_the_identical_pass_never_duplicates(repo):
    """Calling the SAME pass twice with IDENTICAL input against the SAME
    repo must never double facts/failures/quarantined_sources, and
    repository state stays stable (save_fact/save_sources are themselves
    idempotent for unchanged content)."""
    good_source = _source("src_good", url="https://www.sec.gov/good")
    bad_source = _source("src_bad", url="not-a-well-formed-url")
    raw = [
        _raw_fact("Idempotence claim.", good_source, company_claim=False),
        _raw_fact("Bad claim.", bad_source, company_claim=False),
    ]
    stage1 = tuple(_stage1_facts(raw, run_id="idempotent"))
    pin = _pin(pre_integrity_facts=stage1, sources=(good_source, bad_source))

    first = run_full_evidence_integrity_pass(pin, repo)
    second = run_full_evidence_integrity_pass(pin, repo)

    assert len(first.verified_facts) == len(second.verified_facts) == 1
    assert {f.fact_id for f in first.verified_facts} == {f.fact_id for f in second.verified_facts}
    assert len(first.quarantined_sources) == len(second.quarantined_sources) == 1
    assert len(first.failures) == len(second.failures)

    # Repository state itself does not grow: the surviving fact stays at
    # version 1 (save_fact is a no-op for unchanged content).
    fact_id = next(iter({f.fact_id for f in second.verified_facts}))
    row = repo.latest_fact_row(fact_id)
    assert int(row["version"]) == 1


def test_pass_2_literature_cascade_flips_coverage_complete(repo):
    """A Literature Fact introduced/excluded specifically in pass 2
    (never present in pass 1's own evidence) must still trigger the
    coverage_complete/unresolved_reasons cascade when the combined pass-2
    run detects it missing from verified_facts. Action Gate itself
    (blocking_verification_required/Verdict.action) is Pipeline-level and
    is explicitly NOT touched by this phase."""
    source_1 = _source("src_1", url="https://www.sec.gov/one")
    raw_1 = [_raw_fact("Initial claim.", source_1, company_claim=False)]
    stage1_initial = _stage1_facts(raw_1, run_id="pass1")
    pass1 = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1_initial), sources=(source_1,)), repo,
    )
    assert pass1.direct_acquisition_info == {}

    lit_source = _source("src_lit_bad", url="not-a-well-formed-url", tier=SourceTier.UNKNOWN)
    lit_raw = RawFact(
        ticker=TICKER, category=FactCategory.SCIENCE, claim="A new Literature claim in pass 2.",
        source=lit_source, company_claim=False,
    )
    literature_collection_result = CollectionResult(
        collector=LITERATURE_BRIDGE_COLLECTOR_LABEL, raw_facts=[lit_raw], sources=[lit_source],
    )
    stage1_new = _stage1_facts([lit_raw], run_id="pass2")

    combined_raw = tuple(stage1_initial) + tuple(stage1_new)
    combined_sources = (source_1, lit_source)
    pass2 = run_full_evidence_integrity_pass(
        _pin(
            pre_integrity_facts=combined_raw, sources=combined_sources,
            collection_results=(literature_collection_result,),
            direct_acquisition_info={"feature_enabled": True, "coverage_complete": True},
            audit_agent_id="evidence_integrity_adaptive",
        ),
        repo,
    )
    assert pass2.direct_acquisition_info["coverage_complete"] is False
    unresolved = pass2.direct_acquisition_info.get("unresolved_reasons", [])
    assert len(unresolved) == 1


# =============================================================================
# 3. Phase 4.3C Correction 1: no-bypass guarantee + same-run_id audit
#    identity
# =============================================================================
def test_single_call_executes_the_agent_exactly_once(repo, monkeypatch):
    calls = _count_agent_runs(monkeypatch)

    source = _source("src_1", url="https://www.sec.gov/example")
    raw = [_raw_fact("Claim.", source, company_claim=False)]
    stage1 = _stage1_facts(raw)
    out = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1), sources=(source,)), repo,
    )
    assert len(calls) == 1
    assert out.agent_run_record is not None


def test_two_pass_harness_executes_the_agent_exactly_twice_over_the_full_combined_set(repo, monkeypatch):
    calls = _count_agent_runs(monkeypatch)

    source_1 = _source("src_1", url="https://www.sec.gov/one")
    raw_1 = [_raw_fact("Initial claim.", source_1, company_claim=False)]
    stage1_initial = _stage1_facts(raw_1, run_id="pass1")
    pass1 = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1_initial), sources=(source_1,)), repo,
    )
    assert len(calls) == 1
    assert len(calls[0].facts) == 1
    assert pass1.agent_run_record is not None

    source_2 = _source("src_2", url="https://www.sec.gov/two")
    raw_2 = [_raw_fact("New claim.", source_2, company_claim=False)]
    stage1_new = _stage1_facts(raw_2, run_id="pass2")
    combined_raw = tuple(stage1_initial) + tuple(stage1_new)
    pass2 = run_full_evidence_integrity_pass(
        _pin(
            pre_integrity_facts=combined_raw, sources=(source_1, source_2),
            audit_agent_id="evidence_integrity_adaptive",
        ),
        repo,
    )
    assert len(calls) == 2
    assert pass2.agent_run_record is not None
    # Pass 2 received the FULL combined set (initial UNION new), never
    # just the new facts alone.
    second_call_fact_ids = {f.fact_id for f in calls[1].facts}
    assert second_call_fact_ids == {f.fact_id for f in combined_raw}
    assert len(second_call_fact_ids) == 2


def test_no_bypass_argument_or_branch_remains_in_source():
    module_text = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "investment_research" / "orchestrator" / "evidence_integrity_pass.py"
    ).read_text(encoding="utf-8")
    pipeline_text = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "investment_research" / "orchestrator" / "pipeline.py"
    ).read_text(encoding="utf-8")
    assert "already_verified" not in module_text
    assert "already_verified" not in pipeline_text


def test_initial_and_adaptive_agent_run_records_both_survive_under_the_same_run_id(repo):
    """agent_runs's own PRIMARY KEY (run_id, agent_id) means two records
    sharing both would silently overwrite each other -- proves that using
    a distinct audit_agent_id for the second call keeps BOTH queryable."""
    shared_run_id = "shared-run"
    source_1 = _source("src_1", url="https://www.sec.gov/one")
    raw_1 = [_raw_fact("Initial claim.", source_1, company_claim=False)]
    stage1_initial = _stage1_facts(raw_1, run_id=shared_run_id)

    initial = run_full_evidence_integrity_pass(
        _pin(run_id=shared_run_id, pre_integrity_facts=tuple(stage1_initial), sources=(source_1,)),
        repo,
    )
    assert initial.agent_run_record.agent_id == "evidence_integrity"

    source_2 = _source("src_2", url="https://www.sec.gov/two")
    raw_2 = [_raw_fact("New claim.", source_2, company_claim=False)]
    stage1_new = _stage1_facts(raw_2, run_id=shared_run_id)
    combined_raw = tuple(stage1_initial) + tuple(stage1_new)

    adaptive = run_full_evidence_integrity_pass(
        _pin(
            run_id=shared_run_id, pre_integrity_facts=combined_raw, sources=(source_1, source_2),
            audit_agent_id="evidence_integrity_adaptive",
        ),
        repo,
    )
    assert adaptive.agent_run_record.agent_id == "evidence_integrity_adaptive"

    rows = repo.conn.execute(
        "SELECT agent_id, run_id, fact_count FROM agent_runs WHERE run_id = ? ORDER BY agent_id",
        (shared_run_id,),
    ).fetchall()
    agent_ids = {row["agent_id"] for row in rows}
    assert agent_ids == {"evidence_integrity", "evidence_integrity_adaptive"}
    by_agent_id = {row["agent_id"]: row for row in rows}
    assert by_agent_id["evidence_integrity"]["fact_count"] == 1
    assert by_agent_id["evidence_integrity_adaptive"]["fact_count"] == 2


def test_default_audit_agent_id_matches_pre_correction_1_production_identity(repo):
    """A caller that never sets audit_agent_id (production, today) gets
    EXACTLY the same AgentRunRecord.agent_id Phase 4.3C's own single-pass
    production call produced before this correction."""
    source = _source("src_1", url="https://www.sec.gov/example")
    raw = [_raw_fact("Claim.", source, company_claim=False)]
    stage1 = _stage1_facts(raw)
    out = run_full_evidence_integrity_pass(
        _pin(pre_integrity_facts=tuple(stage1), sources=(source,)), repo,
    )
    assert out.agent_run_record.agent_id == "evidence_integrity"


# =============================================================================
# 4. Isolation (Phase 4.3C requirement D)
# =============================================================================
REPO_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "investment_research"
_FORBIDDEN_MODULE_NAMES = ("program_evidence", "identifier_validation", "program_identity_resolution")


def test_new_module_never_references_program_identity_modules():
    text = (REPO_SRC / "orchestrator" / "evidence_integrity_pass.py").read_text(encoding="utf-8")
    for name in _FORBIDDEN_MODULE_NAMES:
        assert name not in text


def test_pipeline_still_never_references_program_identity_modules():
    text = (REPO_SRC / "orchestrator" / "pipeline.py").read_text(encoding="utf-8")
    for name in _FORBIDDEN_MODULE_NAMES:
        assert name not in text


def test_new_module_never_imports_http_llm_or_web_search():
    text = (REPO_SRC / "orchestrator" / "evidence_integrity_pass.py").read_text(encoding="utf-8")
    for token in ("HttpClient", "AcquisitionExecutor", "ExecutionContext", "LLMClient", "WebSearch"):
        assert token not in text


def test_new_module_import_graph_carries_no_llm_agent_classes():
    import investment_research.orchestrator.evidence_integrity_pass as module

    assert not any(name.startswith("LLM") for name in module.__dict__)
