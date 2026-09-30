"""Phase 4.3D: scoped delivery of the central ProgramResolution.

``IsolationGuard.project()`` is the SINGLE place a ProgramResolution is ever
attached to an ``AgentInput`` -- and only for the fixed recipient set
(``science``, ``kill_agent``), regardless of what a caller passes in. These
tests exercise that guarantee directly, at the isolation layer, independent
of the full Pipeline (see tests/integration/test_central_program_resolution_
pipeline.py for the end-to-end proof).
"""

from __future__ import annotations

from investment_research.agents.base import inputs_hash
from investment_research.orchestrator.isolation import EvidenceBus, IsolationGuard
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import FactCategory
from investment_research.scoring.program_resolution import resolve_current_program
from tests.conftest import make_fact

TRIAL = "NCT30000003"


def _resolution():
    facts = [make_fact(f"{TRIAL} phase is Phase 2", category=FactCategory.CLINICAL, value="Phase 2")]
    resolution = resolve_current_program(facts)
    assert resolution.trial_id == TRIAL  # sanity
    return resolution


def _guard() -> IsolationGuard:
    return IsolationGuard(EvidenceBus())


def test_science_receives_the_supplied_resolution_by_identity():
    resolution = _resolution()
    agent_input = _guard().project(
        "science",
        ticker="TESTCO",
        company_name="Test Co",
        program_resolution=resolution,
    )
    assert agent_input.program_resolution is resolution


def test_kill_agent_receives_the_supplied_resolution_by_identity():
    resolution = _resolution()
    agent_input = _guard().project(
        "kill_agent",
        ticker="TESTCO",
        company_name="Test Co",
        program_resolution=resolution,
    )
    assert agent_input.program_resolution is resolution


def test_science_and_kill_agent_get_the_same_object_from_one_shared_call():
    """Test A's identity requirement at the isolation layer: passing the
    SAME resolution object to two different agent_ids' project() calls
    yields the same object on both resulting AgentInputs -- never a copy,
    never a per-agent recomputation."""
    resolution = _resolution()
    guard = _guard()
    science_input = guard.project(
        "science", ticker="TESTCO", company_name="Test Co", program_resolution=resolution
    )
    kill_input = guard.project(
        "kill_agent", ticker="TESTCO", company_name="Test Co", program_resolution=resolution
    )
    assert science_input.program_resolution is kill_input.program_resolution is resolution


def test_forbidden_agents_never_receive_it_even_if_a_caller_passes_it():
    """The enforcement is structural, not caller discipline: project() drops
    the value for any agent_id outside the fixed recipient set, regardless
    of what is passed in -- so a stray Pipeline-side pass-through for e.g.
    blind_judge can never leak it."""
    resolution = _resolution()
    guard = _guard()
    for agent_id in ("blind_judge", "bull_agent", "bear_agent", "regulatory", "contradiction"):
        agent_input = guard.project(
            agent_id,
            ticker="TESTCO",
            company_name="Test Co",
            aliases=(),
            facts=(),
            params={},
            user_preferences=None,
            program_resolution=resolution,
        )
        assert agent_input.program_resolution is None, (
            f"{agent_id!r} must never receive the central ProgramResolution"
        )


def test_default_project_call_with_no_resolution_argument_leaves_it_none():
    agent_input = _guard().project("science", ticker="TESTCO", company_name="Test Co")
    assert agent_input.program_resolution is None


def test_inputs_hash_now_reflects_program_resolution_for_a_permitted_agent():
    """Superseded by Phase 4.3D correction 1: inputs_hash used to ignore
    AgentInput.program_resolution entirely (it simply was not one of the
    fields the function read), so a Science/Kill run over a DIFFERENT
    central resolution left an identical, misleading audit fingerprint.
    Presence-or-absence must now change the hash for a permitted agent; the
    fingerprint's actual content-sensitivity (different trial_id / different
    candidate fields) is covered by tests/unit/
    test_program_resolution_fingerprint.py, next to the canonical
    projection this hash is built from."""
    resolution = _resolution()
    base = AgentInput(
        agent_id="science", run_id="r1", ticker="TESTCO", company_name="Test Co", facts=()
    )
    with_resolution = AgentInput(
        agent_id="science",
        run_id="r1",
        ticker="TESTCO",
        company_name="Test Co",
        facts=(),
        program_resolution=resolution,
    )
    assert inputs_hash(base) != inputs_hash(with_resolution)


def test_inputs_hash_stays_unaffected_for_a_forbidden_agent():
    """The flip side of the above: since IsolationGuard.project() never
    attaches program_resolution to a forbidden agent_id in the first place
    (see the tests above), that agent's inputs_hash cannot move regardless
    of which central resolution a run computed -- the field it would be
    folded from is always None for it, by construction, not by convention."""
    base = AgentInput(
        agent_id="blind_judge", run_id="r1", ticker=None, company_name=None, facts=()
    )
    still_none = AgentInput(
        agent_id="blind_judge",
        run_id="r1",
        ticker=None,
        company_name=None,
        facts=(),
        program_resolution=None,  # exactly what project() would produce for this agent_id
    )
    assert inputs_hash(base) == inputs_hash(still_none)
