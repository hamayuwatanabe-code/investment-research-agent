"""collectors/extraction.py: the ``regulator_endpoint_not_accepted`` rule.

Phase 3A requirement 8 / Phase 3B requirement 7: the existing
``endpoint_not_sufficient`` rule only fires when the negation follows
"endpoint" in the sentence ("the endpoint ... is not sufficient to
demonstrate ..."). It misses the equally natural, and equally generic, "did
not consider the endpoint appropriate to establish effectiveness" phrasing
(negation precedes "endpoint"), and the plain "would not support approval"
construction (no appropriate/adequate/sufficient adjective at all) -- exactly
the class of statement the synthetic exhibit fixture in
``test_phase3a_regulatory_rejection_fixture.py`` needs to trigger. No drug,
company, or endpoint name appears anywhere in this test.

This file also proves two exclusions hold: third-party analyst commentary
and a hypothetical/subjunctive framing must never fire this rule, since it
exists to capture an actual company/regulator statement, never a guess about
one. Detecting a QUOTATION of separately-dated prior text, and telling a
CURRENT-program statement apart from an unrelated historical program's, are
explicitly NOT this rule's job -- see the rule's own comment in
``extraction.py`` and the Phase 3B final report's known limitations
(program-scope discrimination is handled downstream by
``scoring/program_resolution.py``).
"""

from __future__ import annotations

from investment_research.collectors.extraction import RULES

_RULE = next(r for r in RULES if r.key == "regulator_endpoint_not_accepted")


def _fires(sentence: str) -> bool:
    return bool(_RULE.pattern.search(sentence))


# --- positive: user's exact examples ----------------------------------------
def test_fires_regulator_did_not_consider_endpoint_appropriate():
    assert _fires(
        "During the meeting, the agency stated that it did not consider the "
        "proposed primary endpoint appropriate to establish the effectiveness "
        "of the therapy in this indication."
    )


def test_fires_endpoint_not_considered_adequate_to_demonstrate_efficacy():
    assert _fires(
        "The regulator stated that the endpoint was not considered adequate "
        "to demonstrate efficacy in the target population."
    )


def test_fires_endpoint_not_considered_sufficient_to_demonstrate_efficacy():
    assert _fires(
        "The regulator stated that the endpoint was not considered sufficient "
        "to demonstrate efficacy in the target population."
    )


def test_fires_agency_advised_endpoint_would_not_support_approval():
    assert _fires(
        "The agency advised the Company that the endpoint would not support "
        "approval of the product candidate."
    )


def test_fires_when_negation_follows_endpoint():
    assert _fires("The agency concluded the endpoint was not adequate to establish efficacy in the reviewed population.")


# --- negative: user's exact examples ----------------------------------------
def test_does_not_fire_on_benign_endpoint_discussion():
    """The negative case matters most: a rule that fires on ordinary,
    non-adverse endpoint language destroys the gate's usefulness."""
    assert not _fires(
        "The company believes its endpoint selection is appropriate and "
        "sufficient to establish efficacy once additional data mature."
    )


def test_does_not_fire_when_regulator_previously_agreed_the_endpoint_is_adequate():
    assert not _fires(
        "The agency previously agreed the endpoint was adequate to establish "
        "effectiveness, and no further changes are anticipated."
    )


def test_does_not_fire_without_any_endpoint_mention():
    assert not _fires("The company did not consider the current cash position sufficient to establish a two-year runway.")


def test_does_not_fire_on_negated_negative_regulator_has_not_stated_insufficient():
    assert not _fires("The regulator has not stated that the endpoint is insufficient for the intended purpose.")


def test_does_not_fire_on_absence_of_evidence_statement():
    assert not _fires("There is no evidence that the FDA rejected the proposed endpoint for this program.")


def test_does_not_fire_on_planned_future_discussion():
    assert not _fires("A discussion of the endpoint with the regulator is planned for later this year.")


def test_does_not_fire_on_hypothetical_conditional_framing():
    assert not _fires(
        "If the regulator were ever to conclude the endpoint was not adequate "
        "to establish effectiveness, the Company would revise its plan."
    )


def test_does_not_fire_on_analyst_opinion():
    assert not _fires("An analyst noted that, in her opinion, the endpoint might not be adequate to establish effectiveness.")


def test_does_not_fire_on_general_statement_with_no_negation():
    assert not _fires("The endpoint remains appropriate and the company continues to believe it is sufficient to establish efficacy.")


def test_does_not_fire_on_double_negative_supportive_statement():
    assert not _fires("The trial's endpoint was sufficient to establish efficacy, not merely as supportive evidence.")


def test_does_not_fire_when_endpoint_fully_supports_approval():
    assert not _fires("The endpoint fully supports approval and the company does not anticipate any issues.")


def test_does_not_fire_on_unrelated_management_decision_about_the_endpoint():
    assert not _fires("Management does not support changing the endpoint given it continues to establish effectiveness reliably.")
