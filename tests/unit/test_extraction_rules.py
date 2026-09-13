"""collectors/extraction.py: the ``regulator_endpoint_not_accepted`` rule.

Phase 3A requirement 8: the existing ``endpoint_not_sufficient`` rule only
fires when the negation follows "endpoint" in the sentence ("the endpoint ...
is not sufficient to demonstrate ..."). It misses the equally natural, and
equally generic, "did not consider the endpoint appropriate to establish
effectiveness" phrasing, where the negation precedes "endpoint" -- exactly the
class of statement the synthetic exhibit fixture in
``test_phase3a_regulatory_rejection_fixture.py`` needs to trigger. No drug,
company, or endpoint name appears anywhere in this test.
"""

from __future__ import annotations

from investment_research.collectors.extraction import RULES

_RULE = next(r for r in RULES if r.key == "regulator_endpoint_not_accepted")


def test_fires_when_negation_precedes_endpoint():
    sentence = (
        "During the meeting, the agency stated that it did not consider the "
        "proposed primary endpoint appropriate to establish the effectiveness "
        "of the therapy in this indication."
    )
    assert _RULE.pattern.search(sentence)


def test_fires_when_negation_follows_endpoint():
    sentence = "The agency concluded the endpoint was not adequate to establish efficacy in the reviewed population."
    assert _RULE.pattern.search(sentence)


def test_does_not_fire_on_benign_endpoint_discussion():
    """The negative case matters most: a rule that fires on ordinary,
    non-adverse endpoint language destroys the gate's usefulness."""
    sentence = (
        "The company believes its endpoint selection is appropriate and "
        "sufficient to establish efficacy once additional data mature."
    )
    assert not _RULE.pattern.search(sentence)


def test_does_not_fire_when_regulator_previously_agreed_the_endpoint_is_adequate():
    sentence = (
        "The agency previously agreed the endpoint was adequate to establish "
        "effectiveness, and no further changes are anticipated."
    )
    assert not _RULE.pattern.search(sentence)


def test_does_not_fire_without_any_endpoint_mention():
    sentence = "The company did not consider the current cash position sufficient to establish a two-year runway."
    assert not _RULE.pattern.search(sentence)
