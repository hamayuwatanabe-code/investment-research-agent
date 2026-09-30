"""Phase 4.3D: central Program Resolution, wired into Pipeline.run() end to
end.

``ScienceAgent``, ``LLMScienceAgent`` and ``KillAgent`` used to each
independently call ``resolve_current_program()`` over their own (near-
identical) fact sets. ``Pipeline.run()`` now calls it exactly once, after
Stage 2 Evidence Integrity and Stage 2b escalation complete and before Stage
3 Domain Agents begin, and threads the SAME ``ProgramResolution`` only to the
agents permitted to see it.

Section map (matches the task's own lettering):
A. Single call across the whole Pipeline.run().
B. Deterministic consistency: Science and Kill agree because they share one
   value, not because they happened to compute the same thing twice.
C. Ambiguous resolution: neither agent guesses, neither promotes a specific
   trial to a company-level conclusion.
F. Resume: the central resolution is (re)computed exactly once per
   Pipeline.run() call, from whichever facts that call actually has, and a
   resumed run's not-yet-executed Science/Kill still receive it.
Section 9: non-leakage to Blind Judge / Bull / Bear / other forbidden
   agents, proven structurally (AgentInput.program_resolution is None for
   every one of them) within the SAME Pipeline.run() call that proves
   delivery to Science/Kill.

Test G (default DEMOBIO/regression output unchanged) is covered by the
existing tests/regression/test_program_and_escalation_acceptance.py, which
runs Pipeline.run() end to end over a multi-trial fixture and already
asserts on Channel.SCIENCE's program_resolved and the KILL channel's
DIFFERENT PROGRAMME downgrade -- unchanged by this phase's refactor since
the resolution value at that point in the run is identical to what each
agent used to compute for itself.
"""

from __future__ import annotations

from datetime import date

import investment_research.orchestrator.pipeline as pipeline_module
from investment_research.agents.base import inputs_hash
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.isolation import Channel, IsolationGuard
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.schemas.enums import FactCategory, Provenance, SourceTier
from investment_research.schemas.fact import RawFact, Source, make_source_id
from investment_research.scoring.program_resolution import (
    canonical_program_resolution,
    canonical_program_resolution_fingerprint,
)

TODAY = date(2026, 9, 6)
TICKER = "PROGRES"
# Deliberately avoids the word "Program"/"Programme": that word also appears
# in this fixture's own PROGRAM_RELEVANCE_UNRESOLVED unresolved-question text
# (which legitimately reaches Blind Judge -- unresolved questions travel
# downstream by design), and identity_markers() would otherwise treat the
# company name's head word as a forbidden identity marker, producing a false
# leakage failure that has nothing to do with this phase's actual isolation
# guarantee.
NAME = "Resolvix Biotherapeutics Holdings"

CURRENT_TRIAL = "NCT40000004"
OLD_TRIAL = "NCT40000005"
CAND_A = "NCT40000006"
CAND_B = "NCT40000007"


def _source(url: str, event: str, tier: SourceTier = SourceTier.TIER_1) -> Source:
    return Source(
        source_id=make_source_id(url, "Registry record"),
        url=url,
        title="Registry record",
        tier=tier,
        event_date=event,
        published_date=event,
        filing_date=event,
        provenance=Provenance.FIXTURE,
    )


def _raw(claim: str, source: Source, *, company_claim: bool = False) -> RawFact:
    return RawFact(
        ticker=TICKER,
        category=FactCategory.CLINICAL,
        claim=claim,
        source=source,
        company_claim=company_claim,
        collector="central_resolution_fixture",
    )


def _collection(raw_facts: list[RawFact]) -> CollectionResult:
    sources = {f.source.source_id: f.source for f in raw_facts}
    return CollectionResult(
        collector="central_resolution_fixture",
        raw_facts=raw_facts,
        sources=list(sources.values()),
        provenance=Provenance.FIXTURE,
    )


def _resolvable_facts() -> list[RawFact]:
    """A single, explicitly-marked current/lead trial plus an old, unrelated
    one -- resolves uniquely (Case A)."""
    current_source = _source(f"fixture://ctgov/{CURRENT_TRIAL}", "2026-06-01")
    old_source = _source(f"fixture://ctgov/{OLD_TRIAL}", "2019-06-01")
    company_source = _source("fixture://ir/PROGRES/pr", "2026-06-01")
    return [
        _raw(f"{OLD_TRIAL} overall status is TERMINATED", old_source),
        _raw(f"{CURRENT_TRIAL} overall status is RECRUITING", current_source),
        _raw(
            f"{NAME} describes {CURRENT_TRIAL} as its current lead registrational programme",
            company_source,
            company_claim=True,
        ),
    ]


def _ambiguous_facts() -> list[RawFact]:
    """Two candidates, no lead marker, identical recency -- resolves to
    neither (Case B). TERMINATED (not merely ACTIVE_NOT_RECRUITING) so each
    also trips a real KILL_RULE, which is what makes the PROGRAM_RELEVANCE_
    UNRESOLVED provisional-downgrade path in evaluate_kill_gate observable."""
    a = _source(f"fixture://ctgov/{CAND_A}", "2026-01-01")
    b = _source(f"fixture://ctgov/{CAND_B}", "2026-01-01")
    return [
        _raw(f"{CAND_A} overall status is TERMINATED", a),
        _raw(f"{CAND_B} overall status is TERMINATED", b),
    ]


def _count_resolver_calls(monkeypatch) -> list:
    calls: list = []
    original = pipeline_module.resolve_current_program

    def counting(facts):
        calls.append(list(facts))
        return original(facts)

    monkeypatch.setattr(pipeline_module, "resolve_current_program", counting)
    return calls


def _capture_projections(monkeypatch) -> list[tuple[str, object]]:
    """Records (agent_id, resulting AgentInput) for every guard.project()
    call in one Pipeline.run() -- a direct, structural view of what each
    agent actually received, independent of anything the agent itself does
    with it."""
    captured: list[tuple[str, object]] = []
    original = IsolationGuard.project

    def wrapped(self, agent_id, **kwargs):
        agent_input = original(self, agent_id, **kwargs)
        captured.append((agent_id, agent_input))
        return agent_input

    monkeypatch.setattr(IsolationGuard, "project", wrapped)
    return captured


# --- A: single call ----------------------------------------------------------
def test_resolve_current_program_is_called_exactly_once_per_run(repo, monkeypatch):
    calls = _count_resolver_calls(monkeypatch)
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)

    pipeline.run(TICKER, NAME, [_collection(_resolvable_facts())], price=2.0)

    assert len(calls) == 1


# --- B: deterministic consistency --------------------------------------------
def test_science_and_kill_use_the_same_resolved_trial(repo):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run(TICKER, NAME, [_collection(_resolvable_facts())], price=2.0)

    science_payload = result.bus.channels[Channel.SCIENCE].payload
    kill_payload = result.bus.channels[Channel.KILL].payload
    assert science_payload["program_resolved"] == CURRENT_TRIAL
    assert kill_payload["program_resolved"] == CURRENT_TRIAL
    assert science_payload["program_relevance_unresolved"] is False
    assert kill_payload["program_relevance_unresolved"] is False

    old_trial_findings = [f for f in kill_payload["findings"] if OLD_TRIAL in f["detail"]]
    assert old_trial_findings
    assert all("DIFFERENT PROGRAMME" in f["detail"] for f in old_trial_findings)


# --- C: ambiguous --------------------------------------------------------------
def test_ambiguous_resolution_is_never_guessed_by_either_agent(repo):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run(TICKER, NAME, [_collection(_ambiguous_facts())], price=2.0)

    science_payload = result.bus.channels[Channel.SCIENCE].payload
    kill_payload = result.bus.channels[Channel.KILL].payload
    assert science_payload["program_resolved"] == "UNKNOWN"
    assert science_payload["program_relevance_unresolved"] is True
    assert kill_payload["program_resolved"] == "UNKNOWN"
    assert kill_payload["program_relevance_unresolved"] is True

    # Neither candidate was promoted to a company-level (CONFIRMED, non-
    # provisional) kill on its own -- both stay explicitly provisional.
    per_trial_findings = [
        f for f in kill_payload["findings"] if CAND_A in f["detail"] or CAND_B in f["detail"]
    ]
    assert per_trial_findings
    assert all("PROGRAM_RELEVANCE_UNRESOLVED" in f["detail"] for f in per_trial_findings)

    science_questions = [q for q in result.bus.unresolved if q.raised_by == "science"]
    assert any("PROGRAM_RELEVANCE_UNRESOLVED" in q.question for q in science_questions)


# --- Section 9: non-leakage ---------------------------------------------------
def test_program_resolution_never_reaches_forbidden_agents(repo, monkeypatch):
    captured = _capture_projections(monkeypatch)
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    pipeline.run(TICKER, NAME, [_collection(_resolvable_facts())], price=2.0)

    by_agent: dict[str, list] = {}
    for agent_id, agent_input in captured:
        by_agent.setdefault(agent_id, []).append(agent_input)

    science_inputs = by_agent.get("science", [])
    kill_inputs = by_agent.get("kill_agent", [])
    assert science_inputs and all(ai.program_resolution is not None for ai in science_inputs)
    assert kill_inputs and all(ai.program_resolution is not None for ai in kill_inputs)
    # Every kill_agent AgentInput this run built carries the identical
    # object -- one shared value, never a per-call recomputation.
    resolutions = {id(ai.program_resolution) for ai in science_inputs + kill_inputs}
    assert len(resolutions) == 1

    for forbidden in ("blind_judge", "bull_agent", "bear_agent", "regulatory", "contradiction"):
        forbidden_inputs = by_agent.get(forbidden, [])
        assert forbidden_inputs, f"expected {forbidden!r} to have run in this fixture"
        assert all(ai.program_resolution is None for ai in forbidden_inputs), (
            f"{forbidden!r} must never receive the central ProgramResolution"
        )


# --- F: resume -----------------------------------------------------------------
def _clinical_resume_kwargs() -> dict:
    current_source = _source(f"fixture://ctgov/{CURRENT_TRIAL}", "2026-06-01")
    company_source = _source("fixture://ir/PROGRES/pr", "2026-06-01")
    raw_facts = [
        _raw(f"{CURRENT_TRIAL} overall status is RECRUITING", current_source),
        _raw(
            f"{NAME} describes {CURRENT_TRIAL} as its current lead registrational programme",
            company_source,
            company_claim=True,
        ),
    ]
    return {
        "ticker": TICKER,
        "company_name": NAME,
        "collection_results": [_collection(raw_facts)],
        "price": 2.0,
    }


def test_resumed_run_computes_the_resolution_once_and_delivers_it(repo, monkeypatch):
    """A run that crashes after 'verify' but before Science/Kill run must,
    on resume, still compute the central resolution exactly once (for that
    resumed Pipeline.run() call) and deliver the SAME value to Science and
    Kill -- never leaked to Blind Judge either."""
    import investment_research.storage.repository as repository_module

    run_id = "central-resolution-resume"

    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_verify(self, checkpoint):
        if checkpoint.stage not in ("collect", "verify"):
            raise RuntimeError("simulated crash after verify checkpoint")
        return original_save_checkpoint(self, checkpoint)

    import pytest

    with monkeypatch.context() as m:
        m.setattr(repository_module.Repository, "save_checkpoint", _crash_after_verify)
        pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
        with pytest.raises(RuntimeError, match="simulated crash after verify checkpoint"):
            pipeline.run(**_clinical_resume_kwargs(), run_id=run_id)

    calls = _count_resolver_calls(monkeypatch)
    captured = _capture_projections(monkeypatch)
    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_clinical_resume_kwargs(), run_id=run_id, resume=True)

    assert resumed.resume_plan is not None
    assert len(calls) == 1  # computed exactly once for this resumed call

    science_payload = resumed.bus.channels[Channel.SCIENCE].payload
    kill_payload = resumed.bus.channels[Channel.KILL].payload
    assert science_payload["program_resolved"] == CURRENT_TRIAL
    assert kill_payload["program_resolved"] == CURRENT_TRIAL

    by_agent: dict[str, list] = {}
    for agent_id, agent_input in captured:
        by_agent.setdefault(agent_id, []).append(agent_input)
    blind_judge_inputs = by_agent.get("blind_judge", [])
    assert blind_judge_inputs
    assert all(ai.program_resolution is None for ai in blind_judge_inputs)


# =============================================================================
# Phase 4.3D correction 1: ProgramResolution-aware AgentRunRecord fingerprint
# =============================================================================
def _agent_run_record(result, agent_id: str):
    matches = [r for r in result.agent_records if r.agent_id == agent_id]
    assert matches, f"no AgentRunRecord for {agent_id!r}"
    return matches[-1]  # the deterministic pass, when both an LLM and a fallback ran


# --- D: AgentRunRecord --------------------------------------------------------
def test_saved_inputs_hash_matches_the_canonical_input_science_and_kill_share(repo, monkeypatch):
    captured = _capture_projections(monkeypatch)
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run(TICKER, NAME, [_collection(_resolvable_facts())], price=2.0)

    by_agent: dict[str, list] = {}
    for agent_id, agent_input in captured:
        by_agent.setdefault(agent_id, []).append(agent_input)

    science_input = by_agent["science"][-1]
    kill_input = by_agent["kill_agent"][-1]
    assert science_input.program_resolution is not None
    assert kill_input.program_resolution is not None
    # Both agents' central resolution is the SAME value (identity, not just
    # equal content) -- Test B's own guarantee, re-asserted here as the
    # premise Test D's hash comparison depends on.
    assert science_input.program_resolution is kill_input.program_resolution
    assert canonical_program_resolution(science_input.program_resolution) == (
        canonical_program_resolution(kill_input.program_resolution)
    )

    science_record = _agent_run_record(result, "science")
    kill_record = _agent_run_record(result, "kill_agent")
    # The SAVED inputs_hash equals inputs_hash() recomputed, independently,
    # from the actual AgentInput IsolationGuard.project() built for that
    # call -- proving the persisted audit value genuinely reflects the
    # central resolution, not a stale or partial view of it.
    assert science_record.inputs_hash == inputs_hash(science_input)
    assert kill_record.inputs_hash == inputs_hash(kill_input)
    # Science and Kill see different facts/channels (their own isolation
    # policies differ), so their overall hashes need not, and generally do
    # not, match each other -- only the ProgramResolution component they
    # share is required to be identical, already proven above.
    assert science_record.inputs_hash != kill_record.inputs_hash


# --- E: resume / audit --------------------------------------------------------
CURRENT_TRIAL_2 = "NCT40000008"


def _clinical_resume_kwargs_2() -> dict:
    """A second, DIFFERENT fixture (different trial id) for cross-run
    fingerprint-sensitivity comparison."""
    current_source = _source(f"fixture://ctgov/{CURRENT_TRIAL_2}", "2026-06-01")
    company_source = _source("fixture://ir/PROGRES2/pr", "2026-06-01")
    raw_facts = [
        _raw(f"{CURRENT_TRIAL_2} overall status is RECRUITING", current_source),
        _raw(
            f"{NAME} describes {CURRENT_TRIAL_2} as its current lead registrational programme",
            company_source,
            company_claim=True,
        ),
    ]
    return {
        "ticker": TICKER,
        "company_name": NAME,
        "collection_results": [_collection(raw_facts)],
        "price": 2.0,
    }


def test_resumed_run_fingerprint_matches_an_equivalent_fresh_run_never_a_stale_one(
    repo, monkeypatch
):
    """A resumed run's Science AgentRunRecord.inputs_hash must equal a
    completely fresh (never-crashed) run's over IDENTICAL fixture content --
    same central resolution, same facts, same fingerprint, regardless of
    which code path (resume vs. single-shot) produced it. And it must
    differ from a run over a fixture that resolves to a DIFFERENT trial, so
    a stale AgentRunRecord (from before content changed) is never mistaken
    for describing the same effective input as a fresh one."""
    import investment_research.storage.repository as repository_module
    from investment_research.storage.db import open_db
    from investment_research.storage.repository import Repository

    original_save_checkpoint = repository_module.Repository.save_checkpoint

    def _crash_after_verify(self, checkpoint):
        if checkpoint.stage not in ("collect", "verify"):
            raise RuntimeError("simulated crash after verify checkpoint")
        return original_save_checkpoint(self, checkpoint)

    import pytest

    run_id = "central-resolution-resume-fingerprint"
    with monkeypatch.context() as m:
        m.setattr(repository_module.Repository, "save_checkpoint", _crash_after_verify)
        pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
        with pytest.raises(RuntimeError, match="simulated crash after verify checkpoint"):
            pipeline.run(**_clinical_resume_kwargs(), run_id=run_id)

    fresh_pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    resumed = fresh_pipeline.run(**_clinical_resume_kwargs(), run_id=run_id, resume=True)
    resumed_science_hash = _agent_run_record(resumed, "science").inputs_hash

    # A brand-new, never-crashed, never-resumed run over the identical
    # fixture content, in a separate repository (so nothing about run_id or
    # database state can be the reason two hashes happen to match).
    control_repo = Repository(open_db(":memory:"))
    control_pipeline = Pipeline(control_repo, NullSearchProvider(), today=TODAY)
    control = control_pipeline.run(
        **_clinical_resume_kwargs(), run_id="central-resolution-control"
    )
    control_science_hash = _agent_run_record(control, "science").inputs_hash
    assert resumed_science_hash == control_science_hash, (
        "resume must not change the fingerprint for identical effective content"
    )

    # A run over fixture content that resolves to a DIFFERENT trial must
    # produce a genuinely different fingerprint -- never collapsed onto the
    # same hash as the two runs above.
    other_repo = Repository(open_db(":memory:"))
    other_pipeline = Pipeline(other_repo, NullSearchProvider(), today=TODAY)
    other = other_pipeline.run(**_clinical_resume_kwargs_2(), run_id="central-resolution-other")
    other_science_hash = _agent_run_record(other, "science").inputs_hash
    assert other_science_hash != resumed_science_hash, (
        "a different central resolution must not be mistaken for the same input"
    )


# --- F: no-leakage at the fingerprint level ------------------------------------
def test_blind_judge_fingerprint_input_carries_no_program_resolution_content(repo, monkeypatch):
    captured = _capture_projections(monkeypatch)
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY)
    result = pipeline.run(TICKER, NAME, [_collection(_resolvable_facts())], price=2.0)

    by_agent: dict[str, list] = {}
    for agent_id, agent_input in captured:
        by_agent.setdefault(agent_id, []).append(agent_input)

    for forbidden in ("blind_judge", "bull_agent", "bear_agent", "regulatory"):
        for agent_input in by_agent.get(forbidden, []):
            assert agent_input.program_resolution is None

    judge_record = _agent_run_record(result, "blind_judge")
    # Recomputed independently, from the actual (program_resolution=None)
    # input Blind Judge received -- matches the stored value.
    judge_input = by_agent["blind_judge"][-1]
    assert judge_record.inputs_hash == inputs_hash(judge_input)
    # The structural guarantee this correction actually needs: the
    # ProgramResolution OBJECT never reaches this agent's input (so
    # inputs_hash's fingerprint contribution for it is always the fixed
    # empty string -- see test_program_resolution_fingerprint.py's own
    # direct proof of that). NOT tested here: whether any TEXT that happens
    # to overlap with the resolver's own rationale phrasing appears
    # anywhere in judge_input -- ScienceAgent legitimately publishes
    # program_resolved/program_relevance_unresolved/
    # program_resolution_rationale on its OWN Channel.SCIENCE evaluation
    # (Phase 4.3D), and blind_judge's policy has always permitted reading
    # Channel.SCIENCE; that is Science's own finding travelling through an
    # explicitly allowed channel, not a leak of the central object this
    # correction concerns itself with.
    assert judge_input.program_resolution is None
    assert canonical_program_resolution_fingerprint(judge_input.program_resolution) == ""
