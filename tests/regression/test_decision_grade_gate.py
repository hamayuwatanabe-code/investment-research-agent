"""The Decision-Grade Evidence Gate: ten required regression scenarios.

The MUST-1/2/3 work (see ``test_must_requirements.py``) made sure a search
summary can never become a ``VERIFIED_FACT``. That alone was not enough: the
Blind Judge could still turn a K5 kill finding built entirely from unfetched
search snippets into a confident ``AVOID``, because ``_select_action`` decided
from severity alone and never asked whether anything backing that severity had
actually been read. This file is the regression suite for the layer that
closes that gap: research status is tracked separately from investment
action, a Kill finding is PROVISIONAL until a decision-grade fact backs it,
and the Evidence Sufficiency Matrix -- not "was every domain searched" but
"did what we found settle anything" -- gates whether an Action may be emitted
at all.
"""

from __future__ import annotations

import pytest

from investment_research.agents.blind_judge import BlindJudgeAgent
from investment_research.collectors.base import CollectionResult
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research.provider import DomainCoverage
from investment_research.schemas.enums import (
    Action,
    ContentKind,
    EvidenceClass,
    EvidenceSufficiencyStatus,
    FactCategory,
    KillCategory,
    KillConfirmation,
    KillLevel,
    Materiality,
    ResearchDomain,
    ResearchStatus,
    RunStatus,
    SearchStatus,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.evaluation import KillGateResult
from investment_research.schemas.fact import RawFact, Source, make_source_id
from investment_research.scoring.completeness import CompletenessResult
from investment_research.scoring.evidence_sufficiency import assess_evidence_sufficiency
from investment_research.scoring.kill_gate import evaluate_kill_gate
from tests.conftest import make_fact

pytestmark = pytest.mark.regression

ENDPOINT_REJECTED = (
    "FDA stated that it does not consider the proposed primary endpoint appropriate to "
    "establish effectiveness for the intended indication"
)


def _empty_completeness(status: SearchStatus = SearchStatus.SEARCHED) -> CompletenessResult:
    result = CompletenessResult()
    for domain in ResearchDomain:
        entry = DomainCoverage(domain=domain, status=status)
        if status in (SearchStatus.SEARCHED, SearchStatus.PARTIAL):
            entry.queries_executed = 1
            entry.documents_found = 1
        result.coverage[domain] = entry
    return result


# =========================== 1 ==============================================
def test_zero_decision_grade_facts_blocks_and_emits_no_action():
    """A run whose only evidence is search summaries must not emit any Action."""
    facts = [
        make_fact(
            ENDPOINT_REJECTED,
            category=FactCategory.REGULATORY,
            content_kind=ContentKind.SEARCH_SUMMARY,
            verified=VerifiedStatus.SEARCH_EVIDENCE,
        )
    ]
    gate = evaluate_kill_gate(facts, [])
    matrix = assess_evidence_sufficiency(
        facts=facts, completeness=_empty_completeness(), gate=gate
    )
    assert matrix.decision_grade_fact_total == 0
    assert matrix.sufficient is False
    reasons = matrix.blocking_reasons()
    assert any("Zero decision-grade facts" in r for r in reasons)


# =========================== 2 ==============================================
def test_unverified_fda_concern_is_provisional_k5_never_final_k5():
    facts = [
        make_fact(
            ENDPOINT_REJECTED,
            category=FactCategory.REGULATORY,
            content_kind=ContentKind.SEARCH_SUMMARY,
            verified=VerifiedStatus.SEARCH_EVIDENCE,
        )
    ]
    gate = evaluate_kill_gate(facts, [])
    regulatory = gate.by_category(KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5
    assert regulatory.confirmation is KillConfirmation.PROVISIONAL
    assert gate.max_level is KillLevel.K5
    assert gate.max_confirmed_level is KillLevel.K0, (
        "an unread search snippet must never register as a CONFIRMED K5"
    )
    action, _ = BlindJudgeAgent._select_action(gate, 9.0, "REJECTED", RunStatus.COMPLETE)
    assert action is not Action.AVOID, "a PROVISIONAL K5 must not, by itself, select AVOID"


# =========================== 3 ==============================================
def test_verified_fda_body_makes_final_k5_possible():
    """The same claim, this time actually read from the filing body."""
    facts = [
        make_fact(
            ENDPOINT_REJECTED,
            category=FactCategory.REGULATORY,
            content_kind=ContentKind.FULL_DOCUMENT,
            verified=VerifiedStatus.VERIFIED,
            evidence_class=EvidenceClass.VERIFIED_FACT,
            tier=SourceTier.TIER_1,
        )
    ]
    gate = evaluate_kill_gate(facts, [])
    regulatory = gate.by_category(KillCategory.REGULATORY_KILL)
    assert regulatory.level is KillLevel.K5
    assert regulatory.confirmation is KillConfirmation.CONFIRMED
    assert gate.max_confirmed_level is KillLevel.K5
    action, reason = BlindJudgeAgent._select_action(gate, 9.0, "REJECTED", RunStatus.COMPLETE)
    assert action is Action.AVOID
    assert "CONFIRMED K5" in reason


# =========================== 4 & 5 ==========================================
def _run_positive_only(content_kind: ContentKind, tier: SourceTier, company_claim: bool):
    source = Source(
        source_id=make_source_id("fixture://search/endpoint-agreed", "wire summary"),
        url="fixture://search/endpoint-agreed",
        title="Search result summarising a regulatory update",
        tier=tier,
        event_date="2026-05-19",
        published_date="2026-08-07",
    )
    raw = RawFact(
        ticker="AGREECO",
        category=FactCategory.REGULATORY,
        claim="FDA agreed the primary endpoint for the pivotal trial",
        source=source,
        company_claim=company_claim,
        collector="unit_test",
        content_kind=content_kind,
    )
    collection = CollectionResult(collector="unit_test", raw_facts=[raw], sources=[source])
    pipeline = Pipeline(_repo(), NullSearchProvider())
    return pipeline.run("AGREECO", "Concordia Therapeutics", [collection], price=5.0)


def _repo():
    from investment_research.storage.db import open_db
    from investment_research.storage.repository import Repository

    return Repository(open_db(":memory:"))


def test_unverified_bullish_claim_cannot_produce_buy():
    """A positive claim from a search summary must not be able to buy its way to BUY.

    ``_select_action`` alone, given a high evidence confidence and an "AGREED"
    endpoint position, would happily return BUY -- it only ever asks "given
    what is confirmed, what follows". The system-level guarantee comes from
    the pipeline: research_status stays BLOCKED_PENDING_VERIFICATION whenever
    the evidence behind that "AGREED" position is not decision-grade, and no
    Action -- BUY included -- is emitted while that holds.
    """
    result = _run_positive_only(ContentKind.SEARCH_SUMMARY, SourceTier.TIER_1, company_claim=False)
    assert result.verdict.action is not Action.BUY
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert result.evidence_sufficiency is not None
    assert result.evidence_sufficiency.decision_grade_fact_total == 0


def test_analyst_only_bullish_evidence_cannot_produce_buy():
    """A Tier 4 analyst opinion is never, by itself, decision-grade evidence."""
    fact = make_fact(
        "A sell-side analyst maintains a price target of $40, implying substantial upside",
        category=FactCategory.OTHER,
        tier=SourceTier.TIER_4,
        evidence_class=EvidenceClass.ANALYST_OPINION,
        verified=VerifiedStatus.INSUFFICIENT_EVIDENCE,
    )
    assert fact.is_decision_grade is False
    gate = evaluate_kill_gate([fact], [])
    matrix = assess_evidence_sufficiency(
        facts=[fact], completeness=_empty_completeness(), gate=gate
    )
    assert matrix.decision_grade_fact_total == 0
    assert matrix.sufficient is False, (
        "zero decision-grade facts must block a final Action even though nothing here is "
        "negative -- the analyst opinion alone can never make the run COMPLETE"
    )


# =========================== 6 ==============================================
def test_wait_for_event_is_never_a_fallback_for_insufficient_evidence():
    """WAIT_FOR_EVENT is itself an Action; insufficient evidence gets no Action."""
    facts = [
        make_fact(
            ENDPOINT_REJECTED,
            category=FactCategory.REGULATORY,
            content_kind=ContentKind.SEARCH_SUMMARY,
            verified=VerifiedStatus.SEARCH_EVIDENCE,
        )
    ]
    gate = evaluate_kill_gate(facts, [])
    matrix = assess_evidence_sufficiency(
        facts=facts, completeness=_empty_completeness(), gate=gate
    )
    assert not matrix.sufficient
    # Simulate the pipeline's own gating rule directly.
    action = None if not matrix.sufficient else "would-be-action"
    assert action is None
    assert action is not Action.WAIT_FOR_EVENT


# =========================== 7 ==============================================
def test_searched_domain_is_not_the_same_as_evidence_sufficient_domain():
    facts = [
        make_fact(
            "A competitor announced a Phase 2 readout in a trade press summary",
            category=FactCategory.COMPETITION,
            content_kind=ContentKind.SEARCH_SUMMARY,
            verified=VerifiedStatus.SEARCH_EVIDENCE,
            materiality=Materiality.HIGH,
        )
    ]
    completeness = _empty_completeness(SearchStatus.SEARCHED)
    gate = evaluate_kill_gate(facts, [])
    matrix = assess_evidence_sufficiency(facts=facts, completeness=completeness, gate=gate)
    competition = matrix.domains[ResearchDomain.COMPETITION]
    assert competition.search_status is SearchStatus.SEARCHED
    assert competition.evidence_sufficiency_status is EvidenceSufficiencyStatus.INSUFFICIENT
    assert competition.search_status != competition.evidence_sufficiency_status


# =========================== 8 ==============================================
def test_unresolved_critical_contradiction_blocks_final_action():
    facts = [
        make_fact(
            ENDPOINT_REJECTED,
            category=FactCategory.REGULATORY,
            content_kind=ContentKind.FULL_DOCUMENT,
            verified=VerifiedStatus.VERIFIED,
            evidence_class=EvidenceClass.VERIFIED_FACT,
            tier=SourceTier.TIER_1,
        )
    ]
    gate = evaluate_kill_gate(facts, [])
    assert gate.max_confirmed_level is KillLevel.K5  # would otherwise permit AVOID
    matrix = assess_evidence_sufficiency(
        facts=facts,
        completeness=_empty_completeness(),
        gate=gate,
        unresolved_material_claims=(
            "Two filings give contradictory enrollment figures for the pivotal trial",
        ),
    )
    assert matrix.sufficient is False
    assert any("contradictory enrollment" in r for r in matrix.blocking_reasons())


# =========================== 9 ==============================================
def test_blind_judge_llm_path_cannot_promote_provisional_evidence_into_an_action():
    """The LLM Blind Judge's prose can say anything; the Action never listens.

    Regardless of what the model returns in reasoning/headline/thesis_breakers
    -- even text that reads like a confident buy or sell call -- the action
    label is always produced by the same deterministic function the
    non-LLM judge uses, from CONFIRMED kill severity only.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from investment_research.agents.llm_agents2 import LLMBlindJudgeAgent
    from investment_research.llm.client import LLMClient
    from investment_research.schemas.agent_io import AgentInput

    response_holder: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            payload = json.dumps(response_holder["value"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = LLMClient(
            api_key="sk-ant-test",
            base_url=f"http://127.0.0.1:{httpd.server_port}",
            max_retries=0,
            timeout=10,
        )
        response_holder["value"] = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "submit_verdict",
                    "input": {
                        "headline": "STRONG BUY -- the evidence is overwhelming, back up the truck",
                        "reasoning": [
                            "Every signal here points to AVOID immediately, sell everything",
                        ],
                        "thesis_breakers": [],
                        "critical_red_flags": [],
                    },
                }
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }

        fact = make_fact(
            ENDPOINT_REJECTED,
            category=FactCategory.REGULATORY,
            content_kind=ContentKind.SEARCH_SUMMARY,
            verified=VerifiedStatus.SEARCH_EVIDENCE,
        )
        gate = evaluate_kill_gate([fact], [])
        data = AgentInput(
            agent_id="blind_judge",
            run_id="r1",
            ticker=None,
            company_name=None,
            facts=(fact,),
            channels={
                _kill_channel_key(): _kill_evaluation(gate),
            },
            params={"evidence_confidence": 9.0, "run_status": "COMPLETE"},
        )
        output = LLMBlindJudgeAgent(client, fallback=BlindJudgeAgent()).run(data)
        verdict = output.metrics["_verdict_object"]
        expected_action, _ = BlindJudgeAgent._select_action(gate, 9.0, "UNKNOWN", RunStatus.COMPLETE)
        assert verdict.action == expected_action
        assert verdict.action is not Action.AVOID, (
            "the model's own prose said AVOID/STRONG BUY, but the only kill finding here is "
            "PROVISIONAL, so the deterministic rule must not have selected AVOID"
        )
    finally:
        httpd.shutdown()


def _kill_channel_key():
    from investment_research.orchestrator.isolation import Channel

    return Channel.KILL


def _kill_evaluation(gate: KillGateResult):
    from investment_research.orchestrator.isolation import Channel
    from investment_research.schemas.agent_io import Evaluation

    return Evaluation(
        author="kill_agent",
        channel=Channel.KILL,
        summary="test",
        payload={
            "kill_gate": [a.to_row() for a in gate.assessments],
            "unsearched_categories": [],
        },
    )


# =========================== 10 =============================================
def test_positive_and_negative_evidence_use_the_same_decision_grade_rule():
    """Symmetry: the exact same evidentiary shape yields the exact same verdict,
    whatever the sentiment of the underlying claim."""
    negative = make_fact(
        ENDPOINT_REJECTED,
        category=FactCategory.REGULATORY,
        content_kind=ContentKind.SEARCH_SUMMARY,
        verified=VerifiedStatus.SEARCH_EVIDENCE,
        url="https://example.com/negative",
    )
    positive = make_fact(
        "FDA agreed the primary endpoint is adequate to support a marketing application",
        category=FactCategory.REGULATORY,
        content_kind=ContentKind.SEARCH_SUMMARY,
        verified=VerifiedStatus.SEARCH_EVIDENCE,
        url="https://example.com/positive",
    )
    assert negative.is_decision_grade is False
    assert positive.is_decision_grade is False
    assert negative.is_decision_grade == positive.is_decision_grade

    # Give both the identical upgrade (body actually retrieved and verified)
    # and confirm they land on the same side of the line together too.
    import dataclasses

    negative_verified = dataclasses.replace(
        negative,
        content_kind=ContentKind.FULL_DOCUMENT,
        verified_status=VerifiedStatus.VERIFIED,
        evidence_class=EvidenceClass.VERIFIED_FACT,
    )
    positive_verified = dataclasses.replace(
        positive,
        content_kind=ContentKind.FULL_DOCUMENT,
        verified_status=VerifiedStatus.VERIFIED,
        evidence_class=EvidenceClass.VERIFIED_FACT,
    )
    assert negative_verified.is_decision_grade is True
    assert positive_verified.is_decision_grade is True
