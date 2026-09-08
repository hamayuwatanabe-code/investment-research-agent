"""Science must evaluate the thesis-relevant CURRENT programme (requirement D).

A live run picked an unrelated historical Phase 2 trial for Science's design
assessment instead of the company's current lead programme. ScienceAgent must
scope its trial-design assessment to the resolved current programme and state
which programme/trial it evaluated.
"""

from __future__ import annotations

from investment_research.agents.science import ScienceAgent
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import FactCategory
from tests.conftest import make_fact

OLD_TRIAL = "NCT10000001"
CURRENT_TRIAL = "NCT20000002"


def _input(facts):
    return AgentInput(
        agent_id="science",
        run_id="r1",
        ticker="TESTCO",
        company_name="Generic Biotech Holdings",
        facts=tuple(facts),
    )


def test_science_evaluates_the_current_lead_programme_not_an_old_unrelated_one():
    facts = [
        # Old, unrelated, terminated trial in a different indication -- a
        # small, open-label design that would otherwise drag the score down.
        make_fact(
            f"{OLD_TRIAL} phase is Phase 2",
            category=FactCategory.CLINICAL,
            value="Phase 2",
            event_date="2019-01-01",
        ),
        make_fact(
            f"{OLD_TRIAL} overall status is TERMINATED",
            category=FactCategory.CLINICAL,
            value="TERMINATED",
            event_date="2019-06-01",
        ),
        make_fact(
            f"{OLD_TRIAL} allocation is Non-Randomized, masking is NONE",
            category=FactCategory.CLINICAL,
            event_date="2019-01-01",
        ),
        # Current lead programme: randomized, double-blind, large, clinical
        # outcome endpoint, explicitly described as current/lead.
        make_fact(
            f"{CURRENT_TRIAL} phase is Phase 3",
            category=FactCategory.CLINICAL,
            value="Phase 3",
            event_date="2026-01-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} overall status is RECRUITING",
            category=FactCategory.CLINICAL,
            value="RECRUITING",
            event_date="2026-06-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} allocation is Randomized, masking is DOUBLE",
            category=FactCategory.CLINICAL,
            event_date="2026-01-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} enrollment is 400 (ESTIMATED)",
            category=FactCategory.CLINICAL,
            event_date="2026-01-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} primary outcome measure is: overall survival",
            category=FactCategory.CLINICAL,
            event_date="2026-01-01",
        ),
        make_fact(
            f"Generic Biotech Holdings describes {CURRENT_TRIAL} as its current lead "
            "registrational programme",
            category=FactCategory.CLINICAL,
            company_claim=True,
            event_date="2026-06-01",
        ),
    ]
    output = ScienceAgent().run(_input(facts))
    payload = output.evaluation.payload
    assert payload["program_resolved"] == CURRENT_TRIAL
    assert payload["program_relevance_unresolved"] is False
    # The old trial's design (non-randomized, open-label, small n) must not
    # leak into the assessed design -- the current lead trial's design wins.
    assert payload["randomized"] is True
    assert payload["masked"] is True
    assert payload["endpoint_type"] == "CLINICAL_OUTCOME"
    assert CURRENT_TRIAL in output.evaluation.summary


def test_ambiguous_programmes_are_flagged_program_relevance_unresolved():
    facts = [
        make_fact(
            f"{OLD_TRIAL} overall status is ACTIVE_NOT_RECRUITING",
            category=FactCategory.CLINICAL,
            value="ACTIVE_NOT_RECRUITING",
            event_date="2026-01-01",
        ),
        make_fact(
            f"{CURRENT_TRIAL} overall status is ACTIVE_NOT_RECRUITING",
            category=FactCategory.CLINICAL,
            value="ACTIVE_NOT_RECRUITING",
            event_date="2026-01-01",
        ),
    ]
    output = ScienceAgent().run(_input(facts))
    payload = output.evaluation.payload
    assert payload["program_relevance_unresolved"] is True
    questions = [u.question for u in output.unresolved]
    assert any("PROGRAM_RELEVANCE_UNRESOLVED" in q for q in questions)
