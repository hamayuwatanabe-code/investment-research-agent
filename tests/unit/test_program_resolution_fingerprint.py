"""Phase 4.3D correction 1: ProgramResolution-aware AgentRunRecord input
fingerprint.

Before this correction, ``agents.base.inputs_hash`` never read
``AgentInput.program_resolution`` at all -- it simply was not one of the
fields the function's hardcoded list touched (``agent_id``, ``ticker``,
sorted fact_ids, sorted channel names). So a Science or Kill run over a
DIFFERENT central resolution left an identical, misleading audit
fingerprint: the stored ``AgentRunRecord.inputs_hash`` said "this input" when
the actual programme-identity content behind it could have been completely
different.

``scoring.program_resolution.canonical_program_resolution`` /
``canonical_program_resolution_fingerprint`` (the ONE place this projection
is defined) now give ``inputs_hash`` a complete, JSON-safe, order-stable view
of the resolution, folded in as one more element of the hashed payload.
Delivery scoping itself is unchanged (see test_program_resolution_isolation.
py): a forbidden agent's ``AgentInput.program_resolution`` is still always
``None``, so its fingerprint contribution is still always the fixed empty
string, regardless of which central resolution the run actually computed.
"""

from __future__ import annotations

from investment_research.agents.base import inputs_hash
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import UNKNOWN, FactCategory
from investment_research.scoring.program_resolution import (
    ProgramCandidate,
    ProgramResolution,
    canonical_program_resolution,
    canonical_program_resolution_fingerprint,
    resolve_current_program,
)
from tests.conftest import make_fact

TRIAL_A = "NCT60000001"
TRIAL_B = "NCT60000002"


def _facts_resolving_to(trial: str, *, event_date: str = "2026-06-01"):
    return [
        make_fact(
            f"{trial} overall status is RECRUITING",
            category=FactCategory.CLINICAL,
            value="RECRUITING",
            event_date=event_date,
        )
    ]


def _agent_input(agent_id: str, resolution: ProgramResolution | None) -> AgentInput:
    return AgentInput(
        agent_id=agent_id,
        run_id="r1",
        ticker="TESTCO",
        company_name="Test Co",
        facts=(),
        program_resolution=resolution,
    )


# --- A: canonical stability --------------------------------------------------
def test_canonical_projection_is_identical_across_separately_built_equal_instances():
    facts = _facts_resolving_to(TRIAL_A)
    first = resolve_current_program(list(facts))
    second = resolve_current_program(list(facts))  # a second, independent call
    assert first is not second  # genuinely different instances
    assert canonical_program_resolution(first) == canonical_program_resolution(second)
    assert canonical_program_resolution_fingerprint(
        first
    ) == canonical_program_resolution_fingerprint(second)


def test_canonical_projection_is_identical_for_manually_constructed_equal_values():
    """Not just resolve_current_program's own output -- any two
    ProgramResolution VALUES with the same semantic content, however built,
    project identically."""
    candidate = ProgramCandidate(
        trial_id=TRIAL_A,
        status="RECRUITING",
        phase="PHASE3",
        has_lead_marker=True,
        most_recent_date="2026-06-01",
        fact_ids=("fact_b", "fact_a"),  # deliberately out of sorted order
    )
    same_candidate_different_order = ProgramCandidate(
        trial_id=TRIAL_A,
        status="RECRUITING",
        phase="PHASE3",
        has_lead_marker=True,
        most_recent_date="2026-06-01",
        fact_ids=("fact_a", "fact_b"),  # same set, different tuple order
    )
    a = ProgramResolution(
        trial_id=TRIAL_A,
        status="RECRUITING",
        phase="PHASE3",
        relevance_unresolved=False,
        rationale="only one clinical programme in evidence",
        candidates=(candidate,),
    )
    b = ProgramResolution(
        trial_id=TRIAL_A,
        status="RECRUITING",
        phase="PHASE3",
        relevance_unresolved=False,
        rationale="only one clinical programme in evidence",
        candidates=(same_candidate_different_order,),
    )
    # fact_ids order within a candidate carries no meaning of its own (which
    # facts mention this trial, not a ranking) -- canonicalization sorts it,
    # so these two otherwise-identical values project identically.
    assert canonical_program_resolution(a) == canonical_program_resolution(b)
    assert canonical_program_resolution_fingerprint(a) == canonical_program_resolution_fingerprint(b)


def test_canonical_projection_explicitly_includes_unknown_rather_than_omitting_it():
    resolution = ProgramResolution()  # every field at its UNKNOWN/default value
    projection = canonical_program_resolution(resolution)
    assert projection["trial_id"] == UNKNOWN
    assert projection["status"] == UNKNOWN
    assert projection["phase"] == UNKNOWN
    assert projection["relevance_unresolved"] is True
    assert projection["candidates"] == []
    assert set(projection) == {
        "trial_id", "status", "phase", "relevance_unresolved", "rationale", "candidates",
    }


def test_fingerprint_of_none_is_the_fixed_empty_string():
    assert canonical_program_resolution_fingerprint(None) == ""


# --- B: allowed-agent sensitivity --------------------------------------------
def test_hash_changes_when_selected_trial_id_changes():
    a = resolve_current_program(_facts_resolving_to(TRIAL_A))
    b = resolve_current_program(_facts_resolving_to(TRIAL_B))
    assert a.trial_id != b.trial_id
    for agent_id in ("science", "kill_agent"):
        assert inputs_hash(_agent_input(agent_id, a)) != inputs_hash(_agent_input(agent_id, b))


def test_hash_changes_when_relevance_unresolved_changes():
    resolved = ProgramResolution(
        trial_id=TRIAL_A, relevance_unresolved=False, rationale="resolved",
        candidates=(ProgramCandidate(trial_id=TRIAL_A),),
    )
    unresolved = ProgramResolution(
        trial_id=UNKNOWN, relevance_unresolved=True, rationale="resolved",
        candidates=(ProgramCandidate(trial_id=TRIAL_A),),
    )
    for agent_id in ("science", "kill_agent"):
        assert inputs_hash(_agent_input(agent_id, resolved)) != inputs_hash(
            _agent_input(agent_id, unresolved)
        )


def test_hash_changes_when_rationale_alone_changes():
    a = ProgramResolution(
        trial_id=TRIAL_A, relevance_unresolved=False, rationale="rationale one",
        candidates=(ProgramCandidate(trial_id=TRIAL_A),),
    )
    b = ProgramResolution(
        trial_id=TRIAL_A, relevance_unresolved=False, rationale="rationale two",
        candidates=(ProgramCandidate(trial_id=TRIAL_A),),
    )
    for agent_id in ("science", "kill_agent"):
        assert inputs_hash(_agent_input(agent_id, a)) != inputs_hash(_agent_input(agent_id, b))


def test_hash_changes_when_a_candidates_status_phase_or_date_changes():
    base_candidate = ProgramCandidate(
        trial_id=TRIAL_A, status="RECRUITING", phase="PHASE2", most_recent_date="2026-01-01",
    )
    for changed_candidate, what in (
        (ProgramCandidate(trial_id=TRIAL_A, status="TERMINATED", phase="PHASE2", most_recent_date="2026-01-01"), "status"),
        (ProgramCandidate(trial_id=TRIAL_A, status="RECRUITING", phase="PHASE3", most_recent_date="2026-01-01"), "phase"),
        (ProgramCandidate(trial_id=TRIAL_A, status="RECRUITING", phase="PHASE2", most_recent_date="2026-06-01"), "most_recent_date"),
        (ProgramCandidate(trial_id=TRIAL_A, status="RECRUITING", phase="PHASE2", most_recent_date="2026-01-01", has_lead_marker=True), "has_lead_marker"),
    ):
        base = ProgramResolution(trial_id=TRIAL_A, relevance_unresolved=False, candidates=(base_candidate,))
        changed = ProgramResolution(trial_id=TRIAL_A, relevance_unresolved=False, candidates=(changed_candidate,))
        for agent_id in ("science", "kill_agent"):
            assert inputs_hash(_agent_input(agent_id, base)) != inputs_hash(
                _agent_input(agent_id, changed)
            ), f"candidate.{what} change did not move {agent_id}'s inputs_hash"


# --- C: non-allowed isolation --------------------------------------------------
def test_forbidden_agent_hash_never_moves_between_resolution_a_and_b():
    """Drives the REAL IsolationGuard.project() (not a hand-built AgentInput)
    with two genuinely different resolutions, A and B, and shows a forbidden
    agent's inputs_hash is identical either way -- the guard's own scoping
    and inputs_hash's own new sensitivity are proven TOGETHER here, on top of
    (not instead of) each being independently tested elsewhere."""
    from investment_research.orchestrator.isolation import EvidenceBus, IsolationGuard

    a = resolve_current_program(_facts_resolving_to(TRIAL_A))
    b = resolve_current_program(_facts_resolving_to(TRIAL_B))
    assert a.trial_id != b.trial_id  # sanity: genuinely different resolutions

    for agent_id in (
        "blind_judge", "bull_agent", "bear_agent", "regulatory", "capital_structure",
        "competitive", "catalyst", "microstructure", "contradiction", "valuation", "portfolio",
    ):
        guard = IsolationGuard(EvidenceBus())
        input_with_a = guard.project(
            agent_id, ticker="TESTCO", company_name="Test Co", program_resolution=a
        )
        input_with_b = guard.project(
            agent_id, ticker="TESTCO", company_name="Test Co", program_resolution=b
        )
        assert input_with_a.program_resolution is None
        assert input_with_b.program_resolution is None
        assert inputs_hash(input_with_a) == inputs_hash(input_with_b)
    # And the fingerprint text itself never carries trial/candidate content
    # for either resolution once it is (correctly) not delivered.
    assert canonical_program_resolution_fingerprint(None) == ""
