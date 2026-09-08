"""Token diagnostics (requirement G): where the tokens went, visible up front.

Uses lightweight fakes rather than a full ResearchResult -- _token_diagnostics
only reads .llm_budget / .adversarial / .escalation off whatever it is given.
"""

from __future__ import annotations

import types

from investment_research.cli import _token_diagnostics
from investment_research.llm.client import LLMBudget, LLMCallRecord
from investment_research.research.adversarial import AdversarialOutcome
from investment_research.research.escalation import EscalationReport


def _fake_result(*, budget=None, adversarial=None, escalation=None):
    return types.SimpleNamespace(llm_budget=budget, adversarial=adversarial, escalation=escalation)


def test_token_diagnostics_reports_remaining_by_stage_and_by_agent():
    budget = LLMBudget(max_total_tokens=1000, stage_quotas={"discovery": 0.30})
    budget.set_stage("discovery")
    budget.record(LLMCallRecord("adversarial_bear", "m", input_tokens=100, output_tokens=0))

    diagnostics = _token_diagnostics(_fake_result(budget=budget))

    assert diagnostics["remaining"] == 900
    assert diagnostics["by_stage"]["discovery"]["used"] == 100
    assert diagnostics["by_agent"] == {"adversarial_bear": 100}


def test_token_diagnostics_reports_search_and_fetch_counts():
    adversarial = AdversarialOutcome(
        executed=10,
        unexecuted=["q1", "q2"],
        unexecuted_due_to_budget=["q1"],
        deduplicated=["q3"],
    )
    escalation = EscalationReport(fetches_attempted=4, fetches_failed=1)

    diagnostics = _token_diagnostics(_fake_result(adversarial=adversarial, escalation=escalation))

    assert diagnostics["search"] == {
        "executed": 10,
        "unexecuted": 2,
        "unexecuted_due_to_budget": 1,
        "deduplicated": 1,
        "skipped_due_to_direct_coverage": 0,
    }
    assert diagnostics["fetch"] == {"attempted": 4, "failed": 1, "audit_entries": 0}


def test_token_diagnostics_handles_a_run_with_no_llm_activity_at_all():
    """Deterministic-only runs (no --llm) must not crash the diagnostics."""
    diagnostics = _token_diagnostics(_fake_result())
    assert diagnostics["remaining"] is None
    assert diagnostics["by_stage"] == {}
    assert diagnostics["by_agent"] == {}
    assert diagnostics["search"]["executed"] == 0
    assert diagnostics["fetch"]["attempted"] == 0
