"""Synthetic acceptance regressions for the cost-safe batched required-domain
research strategy (requirement H, this turn).

The problem this fixes: a live run spent 66,731 of a 60,000-token discovery
quota on three searches (~22k actual tokens per Anthropic web-search call
even at low effort), skipped twenty more mandatory queries to budget, and
left required domains unsearched. Six independent expensive web-search calls
per candidate simply do not fit. The fix separates the research INTENT
(what question, what domain) from the underlying API CALL that answers it,
so several intents can share one call while still keeping fully independent
per-intent completion status.

Numbered 1-10 to match the acceptance list verbatim. No real ticker,
company, endpoint, meeting type, or indication is named anywhere here.
"""

from __future__ import annotations

from investment_research.collectors.base import CollectionResult
from investment_research.collectors.documents import Document
from investment_research.research.batching import (
    ResearchBatch,
    ResearchIntent,
    build_research_batches,
    execute_research_batch,
    intents_from_unresolved_questions,
    run_research_batches,
)
from investment_research.research.discovery import DiscoveryLog
from investment_research.research.provider import ResearchResult
from investment_research.schemas.enums import (
    ContentKind,
    FactCategory,
    FetchOutcome,
    IntentStatus,
    Provenance,
    ResearchDomain,
    ResearchPath,
    SearchStatus,
    SourceTier,
)
from investment_research.schemas.fact import UnresolvedQuestion
from investment_research.scoring.completeness import assess_completeness


def _intent(domain: ResearchDomain, *, intent_id: str, critical: bool = False,
            priority: int = 50, source_restrictions: tuple[str, ...] = ()) -> ResearchIntent:
    return ResearchIntent(
        intent_id=intent_id, domain=domain, question=f"question for {intent_id}",
        source_restrictions=source_restrictions, priority=priority, critical=critical,
    )


def _doc(url: str) -> Document:
    return Document(
        doc_id=url, url=url, title="t", content_kind=ContentKind.METADATA_ONLY,
        provenance=Provenance.LIVE, tier=SourceTier.TIER_1,
    )


class _BatchingProvider:
    name = "fake_batching"
    path = ResearchPath.ANTHROPIC_WEB

    def __init__(self, documents_by_intent: dict[str, list[Document]] | None = None) -> None:
        self.documents_by_intent = documents_by_intent or {}
        self.batch_calls: list[list[str]] = []

    def available(self):
        return True, "ready"

    def search_batch(self, intents, *, agent_id="research"):
        self.batch_calls.append([i.intent_id for i in intents])
        results = {}
        for intent in intents:
            if intent.intent_id not in self.documents_by_intent:
                continue
            results[intent.intent_id] = ResearchResult(
                query=None, documents=self.documents_by_intent[intent.intent_id],
                path=self.path, executed=True,
            )
        return results, {"server_tool_uses": len(intents), "prompt_tokens": 400,
                          "output_tokens": 80, "actual_total_tokens": 480}


# --- 1. six uncovered required-domain intents fit into <=2 core batches ----
def test_1_six_uncovered_required_domain_intents_fit_in_at_most_two_batches():
    intents = [_intent(domain, intent_id=f"gap_{domain.value.lower()}") for domain in ResearchDomain]
    batches = build_research_batches(intents)
    assert len(batches) <= 2


# --- 2. all six intents retain independent audit identities ---------------
def test_2_all_six_intents_retain_independent_audit_identities():
    intents = [_intent(domain, intent_id=f"gap_{domain.value.lower()}") for domain in ResearchDomain]
    provider = _BatchingProvider({intent.intent_id: [_doc(f"https://www.sec.gov/{intent.intent_id}")] for intent in intents})
    discovery = DiscoveryLog()
    batches = build_research_batches(intents)
    run_research_batches(
        batches, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO",
    )
    assert len(discovery.queries) == 6
    query_ids = {record.query_id for record in discovery.queries}
    assert len(query_ids) == 6, "each intent must have its own distinct, auditable query record"
    domains_recorded = {record.query_text for record in discovery.queries}
    assert len(domains_recorded) == 6


# --- 3. one omitted intent remains UNSEARCHED/PARTIAL when the other five succeed
def test_3_one_omitted_intent_stays_unsearched_when_five_others_succeed():
    intents = [_intent(ResearchDomain.REGULATORY, intent_id=f"i{i}") for i in range(6)]
    documents = {f"i{i}": [_doc(f"https://www.fda.gov/{i}")] for i in range(5)}  # i5 omitted
    provider = _BatchingProvider(documents)
    discovery = DiscoveryLog()
    batch = ResearchBatch(batch_id="b1", intents=intents)
    execute_research_batch(batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO")
    for i in range(5):
        assert intents[i].status is IntentStatus.EXECUTED_WITH_EVIDENCE
    assert intents[5].status is IntentStatus.INCOMPLETE_RESPONSE
    result = intents[5].to_research_result()
    assert result.executed is False, "an omitted intent must never read as searched"


# --- 4. a critical regulatory unresolved question is placed before generic
#        competition/catalyst work ------------------------------------------
def test_4_critical_regulatory_question_placed_before_generic_domain_work():
    question = UnresolvedQuestion(
        question="Does the regulator consider the primary endpoint appropriate to establish "
        "effectiveness for the intended indication?",
        why_it_matters="A rejected endpoint invalidates the registrational path.",
        blocking=True,
        category=FactCategory.REGULATORY,
    )
    critical_intents = intents_from_unresolved_questions([question])
    generic_intents = [
        _intent(ResearchDomain.COMPETITION, intent_id="competition_gap", priority=0),
        _intent(ResearchDomain.CATALYST, intent_id="catalyst_gap", priority=1),
    ]
    batches = build_research_batches([*generic_intents, *critical_intents])
    first_batch_order = batches[0].intent_ids
    assert critical_intents[0].intent_id in first_batch_order
    assert first_batch_order.index(critical_intents[0].intent_id) < first_batch_order.index("competition_gap")
    assert first_batch_order.index(critical_intents[0].intent_id) < first_batch_order.index("catalyst_gap")


# --- 5. direct collector PARTIAL coverage never suppresses an unresolved intent
def test_5_direct_collector_partial_coverage_never_suppresses_an_unresolved_intent():
    # A collector that only TOUCHES REGULATORY (PARTIAL, never sufficient --
    # see scoring.completeness) must never cause a material regulatory
    # unresolved question to be dropped from scheduling.
    collection_results = [CollectionResult(collector="fda", outcome=FetchOutcome.OK, zero_results=True)]
    completeness = assess_completeness(
        search_results=[], facts_by_domain={}, agents_run=set(), collection_results=collection_results,
    )
    assert completeness.coverage[ResearchDomain.REGULATORY].status is SearchStatus.PARTIAL

    question = UnresolvedQuestion(
        question="Does the regulator consider the primary endpoint appropriate?",
        why_it_matters="Determines the registrational path.",
        blocking=True,
        category=FactCategory.REGULATORY,
    )
    # intents_from_unresolved_questions never consults collector coverage at
    # all -- a blocking question always becomes an intent, regardless.
    intents = intents_from_unresolved_questions([question])
    assert len(intents) == 1
    assert intents[0].critical is True


# --- 6. search summaries remain non-decision-grade --------------------------
def test_6_search_summaries_remain_non_decision_grade():
    intent = _intent(ResearchDomain.REGULATORY, intent_id="r1")
    intent.status = IntentStatus.EXECUTED_WITH_EVIDENCE
    intent.documents = [_doc("https://www.fda.gov/x")]  # METADATA_ONLY by construction
    result = intent.to_research_result()
    assert result.documents[0].content_kind is ContentKind.METADATA_ONLY
    assert not result.documents[0].content_kind.is_primary_text, (
        "batching must never promote a search pointer to a fetched, decision-grade body"
    )


# --- 7. batch budget exhaustion leaves later pipeline stages usable ---------
def test_7_batch_budget_exhaustion_leaves_later_pipeline_stages_usable():
    class _BudgetExceededProvider:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def available(self):
            return True, "ready"

        def search(self, query, *, agent_id="research"):
            return ResearchResult(
                query=query, outcome=FetchOutcome.DISABLED, path=self.path,
                executed=False, error="BudgetExceeded: discovery quota spent",
            )

    exhausted_provider = _BudgetExceededProvider()
    batch = ResearchBatch(batch_id="b1", intents=[_intent(ResearchDomain.REGULATORY, intent_id="r1")])
    discovery = DiscoveryLog()
    # Must not raise -- a later, independently-scoped stage using a WORKING
    # provider must still be able to run normally afterward.
    run_research_batches(
        [batch], exhausted_provider, agent_id="adversarial_search", discovery=discovery,
        run_id="r1", ticker="TESTCO",
    )

    working_provider = _BatchingProvider({"e1": [_doc("https://www.sec.gov/x")]})
    escalation_batch = ResearchBatch(batch_id="b2", intents=[_intent(ResearchDomain.CAPITAL_STRUCTURE, intent_id="e1")])
    later_diagnostics = run_research_batches(
        [escalation_batch], working_provider, agent_id="escalation", discovery=DiscoveryLog(),
        run_id="r1", ticker="TESTCO",
    )
    assert later_diagnostics[0].completed_intent_ids == ["e1"]


# --- 8. completed intents from a partially exhausted batch are preserved ---
def test_8_completed_intents_from_a_partially_exhausted_batch_are_preserved():
    batch1 = ResearchBatch(
        batch_id="b1",
        intents=[_intent(ResearchDomain.REGULATORY, intent_id="r1"), _intent(ResearchDomain.SCIENCE_TECHNOLOGY, intent_id="s1")],
    )
    batch2 = ResearchBatch(batch_id="b2", intents=[_intent(ResearchDomain.COMPETITION, intent_id="c1")])
    # r1 succeeds, s1 is omitted (INCOMPLETE, not budget) -- NOT a full wipeout.
    provider = _BatchingProvider({"r1": [_doc("https://www.fda.gov/x")], "c1": [_doc("https://www.example.com/y")]})
    discovery = DiscoveryLog()
    run_research_batches(
        [batch1, batch2], provider, agent_id="adversarial_search", discovery=discovery,
        run_id="r1", ticker="TESTCO",
    )
    assert batch1.intents[0].status is IntentStatus.EXECUTED_WITH_EVIDENCE, "completed intent must survive"
    assert batch1.intents[1].status is IntentStatus.INCOMPLETE_RESPONSE
    assert batch2.intents[0].status is IntentStatus.EXECUTED_WITH_EVIDENCE, "batch 2 must still have run"


# --- 9. source restrictions remain intent-specific inside a batch ----------
def test_9_source_restrictions_remain_intent_specific_inside_a_batch():
    from investment_research.research.anthropic_web import (
        _host_matches,
        _shared_source_restrictions,
    )

    sec_only = _intent(ResearchDomain.CAPITAL_STRUCTURE, intent_id="a", source_restrictions=("sec.gov",))
    pubmed_only = _intent(ResearchDomain.SCIENCE_TECHNOLOGY, intent_id="b", source_restrictions=("pubmed.ncbi.nlm.nih.gov",))
    assert _shared_source_restrictions([sec_only, pubmed_only]) == (), (
        "two intents with different restrictions must never be merged into one shared filter"
    )
    assert _host_matches("https://www.sec.gov/x", sec_only.source_restrictions) is True
    assert _host_matches("https://www.sec.gov/x", pubmed_only.source_restrictions) is False


# --- 10. the completeness gate works from per-intent outcomes -------------
def test_10_completeness_gate_reads_per_intent_outcomes_not_batch_execution():
    """A batch 'running' is not the same as a domain being searched: a
    domain whose only intent came back INCOMPLETE_RESPONSE must not read
    SEARCHED just because SOME batch containing it executed."""
    incomplete_intent = _intent(ResearchDomain.CATALYST, intent_id="cat1")
    incomplete_intent.status = IntentStatus.INCOMPLETE_RESPONSE
    incomplete_intent.detail = "omitted from the batch response"

    evidenced_intent = _intent(ResearchDomain.REGULATORY, intent_id="reg1")
    evidenced_intent.status = IntentStatus.EXECUTED_WITH_EVIDENCE
    evidenced_intent.documents = [_doc("https://www.fda.gov/x")]

    search_results = [incomplete_intent.to_research_result(), evidenced_intent.to_research_result()]
    result = assess_completeness(search_results=search_results, facts_by_domain={}, agents_run=set())

    assert result.coverage[ResearchDomain.CATALYST].status is not SearchStatus.SEARCHED
    assert result.coverage[ResearchDomain.REGULATORY].status is SearchStatus.SEARCHED
