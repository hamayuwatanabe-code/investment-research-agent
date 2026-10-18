"""Phase 4.3L Correction 2: the Kill Gate's regulator rules and the 8-K
extractor share one vocabulary (scoring/regulator_language.py). Positive
wording must fire the rule; benign wording must not (a rule that fires on
benign text destroys the gate's usefulness). All sentences are SYNTHETIC."""

from __future__ import annotations

import pytest

from investment_research.schemas.enums import FactCategory, KillCategory, KillLevel
from investment_research.scoring.kill_gate import evaluate_kill_gate
from tests.conftest import make_fact


def _regulatory_level(claim: str) -> KillLevel:
    gate = evaluate_kill_gate([make_fact(claim, category=FactCategory.REGULATORY)], [])
    return next(a for a in gate.assessments if a.category == KillCategory.REGULATORY_KILL).level


@pytest.mark.parametrize(
    "claim",
    [
        "The FDA advised that RVEF is not an appropriate endpoint to demonstrate efficacy.",
        "The FDA stated that RVEF is not an appropriate primary endpoint to demonstrate efficacy.",
        "The FDA informed the Company that the endpoint is not sufficient to demonstrate efficacy.",
        "The FDA informed the Company that RVEF is not appropriate.",
        "The FDA indicated it does not consider RVEF an appropriate endpoint.",
        "The FDA does not consider the primary endpoint appropriate to establish effectiveness.",
        "The FDA no longer considers the trial to be pivotal.",
        "The FDA does not consider the trial to be a registrational trial.",
    ],
)
def test_regulator_wording_fires_k5(claim):
    assert _regulatory_level(claim) is KillLevel.K5


@pytest.mark.parametrize(
    "claim",
    [
        "The endpoint is not appropriate for the pediatric subgroup analysis described in Table 3.",
        "The Company believes the primary endpoint is appropriate to demonstrate efficacy.",
        "RVEF is measured by echocardiography at week 52.",
        "The Company does not consider the cost of the study material.",
    ],
)
def test_benign_wording_does_not_fire(claim):
    assert _regulatory_level(claim) < KillLevel.K3
