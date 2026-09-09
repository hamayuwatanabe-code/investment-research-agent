"""Section E (ANALYSIS PROVENANCE AND COST) must carry the same diagnostic
detail a unit-level "smoke" test already proves is buildable
(``cli._token_diagnostics``) through into the actual rendered PRODUCTION
report -- batch id, intents, tool uses, actual consumption, completion
status, unattributed-result counts, why an LLM-backed agent fell back to
its deterministic counterpart, and the pre-send-vs-sent-then-failed split
on fetch attempts. Before this, none of that reached ``render_report``'s
output text at all (it only reached the ``--json`` export path via
``_token_diagnostics``).
"""

from __future__ import annotations

from investment_research.llm.client import LLMBudget, LLMCallRecord
from investment_research.orchestrator.isolation import EvidenceBus
from investment_research.orchestrator.pipeline import ResearchResult
from investment_research.reporting.report import ReportRenderer, _section_cost
from investment_research.research.adversarial import AdversarialOutcome
from investment_research.research.batching import BatchDiagnostics
from investment_research.research.escalation import EscalationReport
from investment_research.schemas.agent_io import AgentRunRecord, RunContext
from investment_research.schemas.enums import RunStatus


def _context() -> RunContext:
    return RunContext(
        run_id="r1", ticker="TESTCO", company_name="Generic Biotech Holdings", status=RunStatus.COMPLETE
    )


def _result(**kwargs) -> ResearchResult:
    return ResearchResult(context=_context(), bus=EvidenceBus(), verdict=None, scorecard=None, **kwargs)


def test_cost_section_renders_batch_diagnostics_for_every_batch():
    adversarial = AdversarialOutcome(
        batch_diagnostics=[
            BatchDiagnostics(
                batch_id="batch_1",
                intent_ids=["reg_1", "cap_1", "comp_1"],
                domains=["REGULATORY", "CAPITAL_STRUCTURE", "COMPETITION"],
                server_tool_uses=4,
                actual_total_tokens=47238,
                completed_intent_ids=["reg_1", "cap_1"],
                incomplete_intent_ids=["comp_1"],
                ambiguous_results=1,
                retry_rounds=2,
            )
        ]
    )
    result = _result(adversarial=adversarial)
    section = _section_cost(ReportRenderer(result))

    assert "batch_1" in section
    assert "reg_1, cap_1, comp_1" in section
    assert "tool uses         : 4" in section
    assert "47,238" in section
    assert "completed intents : reg_1, cap_1" in section
    assert "incomplete intents: comp_1" in section
    assert "unattributed results: 1" in section
    assert "retry rounds      : 2" in section


def test_cost_section_explains_a_spent_but_discarded_agent_call():
    """The exact discrepancy the bug report describes: tokens recorded for
    an agent_id that never made it into llm_agents_used. Section E must
    print the actual reason (from the agent's own AgentRunRecord), not
    just leave "science" silently absent from the LLM-backed list."""
    budget = LLMBudget(max_total_tokens=200_000)
    budget.set_stage("interpretive")
    budget.record(LLMCallRecord(agent_id="science", model="m", input_tokens=20000, output_tokens=11131))
    result = _result(
        llm_budget=budget,
        llm_agents_used=["regulatory"],
        agent_records=[
            AgentRunRecord(
                run_id="r1",
                agent_id="science",
                status="degraded",
                started_at="",
                finished_at="",
                duration_ms=0,
                fact_count=0,
                error_count=1,
                errors="LLM not used: LLM call failed: BudgetExceeded: stage 'interpretive' budget exhausted",
            )
        ],
    )
    section = _section_cost(ReportRenderer(result))

    assert "science" in section
    assert "fell back to deterministic" in section
    assert "BudgetExceeded" in section
    # The agent that DID succeed must never get a spurious fallback line.
    regulatory_line_index = section.index("regulatory")
    following = section[regulatory_line_index : regulatory_line_index + 200]
    assert "fell back to deterministic" not in following.split("science")[0]


def test_cost_section_never_prints_a_fallback_line_when_none_was_recorded():
    budget = LLMBudget(max_total_tokens=200_000)
    budget.record(LLMCallRecord(agent_id="regulatory", model="m", input_tokens=100, output_tokens=100))
    result = _result(llm_budget=budget, llm_agents_used=["regulatory"], agent_records=[])
    section = _section_cost(ReportRenderer(result))
    assert "fell back to deterministic" not in section


def test_cost_section_distinguishes_not_sent_from_failed_fetches():
    escalation = EscalationReport(fetches_attempted=9, fetches_failed=9, fetches_not_sent=9)
    result = _result(escalation=escalation)
    section = _section_cost(ReportRenderer(result))
    assert "fetches attempted : 9" in section
    assert "fetches failed    : 9" in section
    assert "fetches not sent  : 9" in section
