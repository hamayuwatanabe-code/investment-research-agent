"""Scoped LLM-budget stage context (requirement A).

The defect: ``Pipeline`` set the budget's current stage ambiently via
``budget.set_stage("escalation")`` for Stage 3b (unresolved-question
escalation) and never reset it before Contradiction/Kill/Bear/Bull ran --
those stages then executed, and had their token usage attributed to, whatever
stage happened to run immediately before them. A live run's Kill Agent
searches were starved because "escalation" was already exhausted, and the
report attributed Contradiction/Kill/Bear/Bull spend to the wrong bucket
entirely.

``LLMBudget.stage()`` is the fix: a context manager that saves and restores
``current_stage`` around a ``with`` block, on every exit path including an
exception, so one stage's work can never bleed into whatever runs after it.
"""

from __future__ import annotations

from investment_research.llm.client import LLMBudget, LLMCallRecord


def _call(agent_id: str, tokens: int = 100) -> LLMCallRecord:
    return LLMCallRecord(agent_id=agent_id, model="claude-sonnet-5", input_tokens=tokens)


def test_escalation_then_contradiction_restores_interpretive():
    budget = LLMBudget(max_total_tokens=1_000_000)
    budget.set_stage("interpretive")
    budget.record(_call("regulatory"))  # domain agents, Stage 3

    with budget.stage("escalation"):
        budget.record(_call("escalation_question"))  # Stage 3b
        assert budget.current_stage == "escalation"

    # Back to "interpretive" -- not "escalation", and not "" -- exactly as if
    # Stage 3b had never run, which is what lets Contradiction (Stage 4) be
    # correctly attributed without the pipeline having to re-declare
    # "interpretive" itself.
    assert budget.current_stage == "interpretive"
    budget.record(_call("contradiction"))
    assert budget.calls[-1].stage == "interpretive"


def test_escalation_budget_exhausted_does_not_block_later_interpretive_calls():
    budget = LLMBudget(
        max_total_tokens=1_000_000,
        stage_quotas={"escalation": 0.0001, "interpretive": 0.5},
    )
    # Actual usage that overruns escalation's tiny quota -- escalation becomes
    # stage-exhausted (its own, LOCAL exhaustion; not the global `exhausted`).
    with budget.stage("escalation"):
        budget.record(_call("escalation_question", tokens=50_000))
        assert "escalation" in budget.stage_exhausted
        assert not budget.exhausted  # local, not global

    assert budget.current_stage == ""  # nothing was active before this `with` above

    with budget.stage("interpretive"):
        # Must not raise: interpretive has its own quota/accounting, entirely
        # independent of escalation's exhaustion.
        budget.check(1000)
        budget.record(_call("science", tokens=1000))
        assert "interpretive" not in budget.stage_exhausted


def test_exception_inside_stage_context_still_restores_prior_stage():
    budget = LLMBudget(max_total_tokens=1_000_000)
    budget.set_stage("interpretive")

    class _Boom(Exception):
        pass

    try:
        with budget.stage("escalation"):
            assert budget.current_stage == "escalation"
            raise _Boom("simulated failure mid-escalation")
    except _Boom:
        pass

    assert budget.current_stage == "interpretive", (
        "a stage context must restore the previous stage even when the block raises, so a "
        "crash mid-escalation can never leave later stages misattributed"
    )


def test_nested_stage_contexts_restore_correctly():
    budget = LLMBudget(max_total_tokens=1_000_000)
    with budget.stage("discovery"):
        assert budget.current_stage == "discovery"
        with budget.stage("escalation"):
            assert budget.current_stage == "escalation"
        assert budget.current_stage == "discovery"
    assert budget.current_stage == ""
