"""Cost-safe batched required-domain research (the v4 token-starvation fix).

Unit-level coverage of research/batching.py: intent/batch construction,
priority waves, per-intent completion semantics under both a batching-
capable provider and a provider that can only serve one intent at a time,
and clean budget-exhaustion handling.
"""

from __future__ import annotations

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
    SourceTier,
)
from investment_research.schemas.fact import UnresolvedQuestion


def _intent(domain: ResearchDomain, *, intent_id: str, critical: bool = False, priority: int = 50,
            source_restrictions: tuple[str, ...] = ()) -> ResearchIntent:
    return ResearchIntent(
        intent_id=intent_id, domain=domain, question=f"question for {intent_id}",
        source_restrictions=source_restrictions, priority=priority, critical=critical,
    )


def _doc(url: str, *, content_kind: ContentKind = ContentKind.METADATA_ONLY) -> Document:
    return Document(
        doc_id=url, url=url, title="t", content_kind=content_kind,
        provenance=Provenance.LIVE, tier=SourceTier.TIER_1,
    )


class _FakeBatchingProvider:
    """A provider WITH search_batch -- serves the whole batch in one call."""

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
                continue  # simulate the model omitting this intent
            results[intent.intent_id] = ResearchResult(
                query=None, documents=self.documents_by_intent[intent.intent_id],
                path=self.path, executed=True,
            )
        return results, {
            "server_tool_uses": len(intents), "prompt_tokens": 500,
            "output_tokens": 100, "actual_total_tokens": 600,
        }

    def search(self, query, *, agent_id="research"):
        raise AssertionError("search() must not be called when search_batch is available")


class _FakePerIntentProvider:
    """A provider WITHOUT search_batch -- one .search() call per intent."""

    name = "fake_per_intent"
    path = ResearchPath.CORPUS

    def __init__(self, documents_by_question: dict[str, list[Document]] | None = None) -> None:
        self.documents_by_question = documents_by_question or {}
        self.calls: list[str] = []

    def available(self):
        return True, "ready"

    def search(self, query, *, agent_id="research"):
        self.calls.append(query.query)
        documents = self.documents_by_question.get(query.query, [])
        return ResearchResult(query=query, documents=documents, path=self.path, executed=True)


# --- build_research_batches(): waves and per-domain coverage guarantee -----
def test_six_domain_intents_fit_in_at_most_two_batches():
    intents = [_intent(domain, intent_id=f"gap_{domain.value.lower()}") for domain in ResearchDomain]
    batches = build_research_batches(intents)
    assert len(batches) <= 2
    all_ids = {i for batch in batches for i in batch.intent_ids}
    assert all_ids == {intent.intent_id for intent in intents}


def test_every_domain_gets_a_first_wave_slot_even_under_max_batch_size():
    """Six template gaps -- one per domain -- must all land in the FIRST
    wave, so a single executed batch alone guarantees full domain coverage
    even if no later batch is ever reached."""
    intents = [_intent(domain, intent_id=f"gap_{domain.value.lower()}") for domain in ResearchDomain]
    batches = build_research_batches(intents)
    first_batch_domains = {intent.domain for intent in batches[0].intents}
    assert first_batch_domains == set(ResearchDomain)


def test_critical_intents_always_land_in_the_first_batch():
    generic = [_intent(ResearchDomain.COMPETITION, intent_id="comp_1", priority=1)]
    critical = _intent(ResearchDomain.CAPITAL_STRUCTURE, intent_id="critical_1", critical=True, priority=99)
    batches = build_research_batches([*generic, critical])
    assert critical.intent_id in batches[0].intent_ids


def test_regulatory_and_contradiction_rank_ahead_of_other_domains_within_a_wave():
    intents = [
        _intent(ResearchDomain.COMPETITION, intent_id="competition", priority=0),
        _intent(ResearchDomain.REGULATORY, intent_id="regulatory", priority=1),
        _intent(ResearchDomain.CATALYST, intent_id="catalyst", priority=2),
        _intent(ResearchDomain.CONTRADICTION, intent_id="contradiction", priority=3),
    ]
    batches = build_research_batches(intents)
    order = batches[0].intent_ids
    assert order.index("regulatory") < order.index("competition")
    assert order.index("contradiction") < order.index("catalyst")


def test_overflow_beyond_max_batch_size_spills_into_an_additional_batch():
    intents = [_intent(domain, intent_id=f"gap_{i}") for i, domain in enumerate(list(ResearchDomain) * 2)]
    batches = build_research_batches(intents, max_intents_per_batch=3)
    assert all(len(batch.intents) <= 3 for batch in batches)
    assert sum(len(batch.intents) for batch in batches) == len(intents)


# --- execute_research_batch(): batching-capable vs per-intent fallback ------
def test_execute_batch_uses_search_batch_when_available():
    intents = [_intent(ResearchDomain.REGULATORY, intent_id="reg_1"), _intent(ResearchDomain.SCIENCE_TECHNOLOGY, intent_id="sci_1")]
    batch = ResearchBatch(batch_id="b1", intents=intents)
    provider = _FakeBatchingProvider({"reg_1": [_doc("https://www.fda.gov/x")], "sci_1": []})
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO",
    )
    assert provider.batch_calls == [["reg_1", "sci_1"]]
    assert intents[0].status is IntentStatus.EXECUTED_WITH_EVIDENCE
    assert intents[1].status is IntentStatus.EXECUTED_ZERO_RESULTS
    assert diagnostics.completed_intent_ids == ["reg_1", "sci_1"]
    assert diagnostics.server_tool_uses == 2
    assert len(discovery.queries) == 2


def test_execute_batch_falls_back_to_per_intent_search_without_search_batch():
    intents = [_intent(ResearchDomain.REGULATORY, intent_id="reg_1")]
    intents[0].question = "TESTCO FDA concern"
    batch = ResearchBatch(batch_id="b1", intents=intents)
    provider = _FakePerIntentProvider({"TESTCO FDA concern": [_doc("https://www.fda.gov/x")]})
    discovery = DiscoveryLog()
    execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO",
    )
    assert provider.calls == ["TESTCO FDA concern"]
    assert intents[0].status is IntentStatus.EXECUTED_WITH_EVIDENCE


def test_execute_batch_error_result_keeps_documents_but_marks_error():
    """A ResearchResult carrying BOTH documents and an error (a same-intent
    partial success/failure, as AnthropicWebResearchProvider.search_batch()
    now reports it) must land as ERROR while still keeping the evidence
    that was found -- never silently promoted to full success."""
    intents = [_intent(ResearchDomain.REGULATORY, intent_id="reg_1")]
    batch = ResearchBatch(batch_id="b1", intents=intents)

    class _MixedResultProvider:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def available(self):
            return True, "ready"

        def search_batch(self, batch_intents, *, agent_id="research"):
            return (
                {
                    "reg_1": ResearchResult(
                        query=None, documents=[_doc("https://www.fda.gov/a")],
                        path=self.path, executed=True, error="timeout on a second search",
                    )
                },
                {"server_tool_uses": 2, "prompt_tokens": 0, "output_tokens": 0, "actual_total_tokens": 0},
            )

    discovery = DiscoveryLog()
    execute_research_batch(
        batch, _MixedResultProvider(), agent_id="adversarial_search", discovery=discovery,
        run_id="r1", ticker="TESTCO",
    )
    assert intents[0].status is IntentStatus.ERROR
    assert intents[0].detail == "timeout on a second search"
    assert len(intents[0].documents) == 1
    assert intents[0].documents[0].url == "https://www.fda.gov/a"


def test_omitted_intent_is_incomplete_response_not_zero_results():
    """H3: the model addressing five of six intents must not silently read
    as the sixth having been searched and found empty."""
    intents = [_intent(ResearchDomain.REGULATORY, intent_id=f"i{i}") for i in range(6)]
    batch = ResearchBatch(batch_id="b1", intents=intents)
    documents = {f"i{i}": [_doc(f"https://www.fda.gov/{i}")] for i in range(5)}  # i5 omitted
    provider = _FakeBatchingProvider(documents)
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO",
    )
    for i in range(5):
        assert intents[i].status is IntentStatus.EXECUTED_WITH_EVIDENCE
    assert intents[5].status is IntentStatus.INCOMPLETE_RESPONSE
    assert "i5" in diagnostics.incomplete_intent_ids
    assert set(diagnostics.completed_intent_ids) == {f"i{i}" for i in range(5)}


# --- run_research_batches(): clean budget-exhaustion handling ---------------
def test_run_batches_stops_after_a_fully_exhausted_batch():
    batch1 = ResearchBatch(batch_id="b1", intents=[_intent(ResearchDomain.REGULATORY, intent_id="r1")])
    batch2 = ResearchBatch(batch_id="b2", intents=[_intent(ResearchDomain.COMPETITION, intent_id="c1")])

    class _AlwaysBudgetExceededProvider:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def available(self):
            return True, "ready"

        def search(self, query, *, agent_id="research"):
            return ResearchResult(
                query=query, outcome=FetchOutcome.DISABLED, path=self.path,
                executed=False, error="BudgetExceeded: discovery quota spent",
            )

    provider = _AlwaysBudgetExceededProvider()
    discovery = DiscoveryLog()
    run_research_batches(
        [batch1, batch2], provider, agent_id="adversarial_search", discovery=discovery,
        run_id="r1", ticker="TESTCO",
    )
    assert batch1.intents[0].status is IntentStatus.SKIPPED_DUE_TO_BUDGET
    assert batch2.intents[0].status is IntentStatus.SKIPPED_DUE_TO_BUDGET
    assert "already exhausted by an earlier batch" in batch2.intents[0].detail


def test_run_batches_partial_exhaustion_does_not_stop_the_next_batch():
    """H8: a batch that only PARTIALLY succeeded must not be treated as a
    total wipeout -- the next batch still runs normally."""
    batch1 = ResearchBatch(
        batch_id="b1",
        intents=[_intent(ResearchDomain.REGULATORY, intent_id="r1"), _intent(ResearchDomain.SCIENCE_TECHNOLOGY, intent_id="s1")],
    )
    batch2 = ResearchBatch(batch_id="b2", intents=[_intent(ResearchDomain.COMPETITION, intent_id="c1")])
    provider = _FakeBatchingProvider({"r1": [_doc("https://www.fda.gov/x")]})  # s1 omitted -> INCOMPLETE, not budget
    discovery = DiscoveryLog()
    run_research_batches(
        [batch1, batch2], provider, agent_id="adversarial_search", discovery=discovery,
        run_id="r1", ticker="TESTCO",
    )
    assert batch1.intents[0].status is IntentStatus.EXECUTED_WITH_EVIDENCE
    assert batch1.intents[1].status is IntentStatus.INCOMPLETE_RESPONSE
    # batch2 still ran -- not skipped due to a false "exhausted" signal.
    assert provider.batch_calls == [["r1", "s1"], ["c1"]]


def test_run_batches_never_raises_even_when_execute_raises():
    class _RaisingProvider:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def available(self):
            return True, "ready"

        def search(self, query, *, agent_id="research"):
            raise RuntimeError("provider blew up")

    batch = ResearchBatch(batch_id="b1", intents=[_intent(ResearchDomain.REGULATORY, intent_id="r1")])
    discovery = DiscoveryLog()
    diagnostics = run_research_batches(
        [batch], _RaisingProvider(), agent_id="adversarial_search", discovery=discovery,
        run_id="r1", ticker="TESTCO",
    )
    assert batch.intents[0].status is IntentStatus.ERROR
    assert diagnostics[0].incomplete_intent_ids == ["r1"]


# --- intents_from_unresolved_questions() ------------------------------------
def test_intents_from_unresolved_questions_are_always_critical():
    question = UnresolvedQuestion(
        question="Does the regulator consider the primary endpoint appropriate?",
        why_it_matters="Determines the registrational path.",
        blocking=True,
        category=FactCategory.REGULATORY,
    )
    intents = intents_from_unresolved_questions([question])
    assert len(intents) == 1
    assert intents[0].critical is True
    assert intents[0].domain is ResearchDomain.REGULATORY
    assert "sec.gov" in intents[0].source_restrictions or "fda.gov" in intents[0].source_restrictions


def test_intents_from_unresolved_questions_ignores_non_blocking():
    question = UnresolvedQuestion(
        question="What is the pre-specified statistical power?",
        why_it_matters="Nice to know.",
        blocking=False,
        category=FactCategory.SCIENCE,
    )
    assert intents_from_unresolved_questions([question]) == []


# --- to_research_result() / decision-grade isolation ------------------------
def test_to_research_result_preserves_metadata_only_content_kind():
    """H6: batching must never promote a search result to decision-grade
    evidence -- documents stay whatever ContentKind they arrived with."""
    intent = _intent(ResearchDomain.REGULATORY, intent_id="r1")
    intent.status = IntentStatus.EXECUTED_WITH_EVIDENCE
    intent.documents = [_doc("https://www.fda.gov/x", content_kind=ContentKind.METADATA_ONLY)]
    result = intent.to_research_result()
    assert result.documents[0].content_kind is ContentKind.METADATA_ONLY
    assert result.executed is True


def test_incomplete_response_never_becomes_a_research_result_saying_executed():
    intent = _intent(ResearchDomain.REGULATORY, intent_id="r1")
    intent.status = IntentStatus.INCOMPLETE_RESPONSE
    result = intent.to_research_result()
    assert result.executed is False
