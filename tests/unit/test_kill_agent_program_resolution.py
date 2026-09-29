"""Phase 4.3D: KillAgent reads the central ProgramResolution; it never
recomputes its own (requirement D / Sections 4-5 of the centralization task).

Before this phase, ScienceAgent and KillAgent each independently called
``resolve_current_program()`` over the same fact set, relying on implicit
agreement rather than a single shared value. Now KillAgent reads ONLY
``data.program_resolution``, supplied once by ``Pipeline.run()`` in
production; a standalone unit test must pass it explicitly.
"""

from __future__ import annotations

from investment_research.agents.kill_agent import KillAgent
from investment_research.collectors.search import NullSearchProvider
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import UNKNOWN, FactCategory, KillLevel, SourceTier
from investment_research.scoring.program_resolution import resolve_current_program
from tests.conftest import make_fact

OLD_TRIAL = "NCT10000001"
CURRENT_TRIAL = "NCT20000002"


def _input(facts, program_resolution=None):
    return AgentInput(
        agent_id="kill_agent",
        run_id="r1",
        ticker="TESTCO",
        company_name="Generic Biotech Holdings",
        facts=tuple(facts),
        program_resolution=program_resolution,
    )


def _agent() -> KillAgent:
    return KillAgent(NullSearchProvider())


def test_kill_agent_uses_the_supplied_central_resolution_not_its_own():
    facts = [
        make_fact(
            f"{OLD_TRIAL} overall status is TERMINATED",
            category=FactCategory.CLINICAL,
            value="TERMINATED",
            tier=SourceTier.TIER_1,
            event_date="2019-06-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} overall status is RECRUITING",
            category=FactCategory.CLINICAL,
            value="RECRUITING",
            tier=SourceTier.TIER_1,
            event_date="2026-06-01",
        ),
        make_fact(
            f"Generic Biotech Holdings describes {CURRENT_TRIAL} as its current lead "
            "registrational programme",
            category=FactCategory.CLINICAL,
            company_claim=True,
            event_date="2026-06-01",
        ),
    ]
    resolution = resolve_current_program(facts)
    assert resolution.trial_id == CURRENT_TRIAL  # sanity: the fixture resolves cleanly

    output = _agent().run(_input(facts, program_resolution=resolution))
    payload = output.evaluation.payload
    assert payload["program_resolved"] == CURRENT_TRIAL
    assert payload["program_relevance_unresolved"] is False
    old_trial_findings = [f for f in payload["findings"] if OLD_TRIAL in f["detail"]]
    assert old_trial_findings, "the old trial's status must still be recorded as a finding"
    assert all("DIFFERENT PROGRAMME" in f["detail"] for f in old_trial_findings)
    assert all(KillLevel(f["level"]).level <= KillLevel.K1.level for f in old_trial_findings)


def test_missing_central_resolution_fails_closed_never_recomputes():
    """Section 5 / Test D: no central Resolution supplied at all (e.g. this
    agent run standalone) -- KillAgent must not recompute one from facts,
    must pass UNKNOWN as the current trial, and must keep every per-trial
    finding provisional rather than letting an old/different trial drive a
    company-level kill."""
    facts = [
        make_fact(
            f"{OLD_TRIAL} overall status is TERMINATED",
            category=FactCategory.CLINICAL,
            value="TERMINATED",
            tier=SourceTier.TIER_1,
            event_date="2019-06-01",
        ),
    ]
    output = _agent().run(_input(facts, program_resolution=None))
    payload = output.evaluation.payload
    assert payload["program_resolved"] == UNKNOWN
    assert payload["program_relevance_unresolved"] is True
    old_trial_findings = [f for f in payload["findings"] if OLD_TRIAL in f["detail"]]
    assert old_trial_findings
    assert all("PROGRAM_RELEVANCE_UNRESOLVED" in f["detail"] for f in old_trial_findings)
    questions = [q.question for q in output.unresolved]
    assert any("no central programme resolution was supplied" in q for q in questions)
