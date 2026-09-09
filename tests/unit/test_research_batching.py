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


def test_execute_batch_incomplete_intent_keeps_documents_but_is_never_success():
    """同一intentで1回成功、2回目の結果が未着 (end-to-end through
    execute_research_batch): a provider reporting an intent as
    ``executed=False`` with an "IncompleteIntent" error -- exactly what
    AnthropicWebResearchProvider.search_batch() now returns for an intent
    with an unresolved sub-search -- must land as INCOMPLETE_RESPONSE, not
    EXECUTED_WITH_EVIDENCE, while the document already obtained from the
    resolved search still survives on the intent."""
    intents = [_intent(ResearchDomain.REGULATORY, intent_id="reg_1")]
    batch = ResearchBatch(batch_id="b1", intents=intents)

    class _IncompleteIntentProvider:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def available(self):
            return True, "ready"

        def search_batch(self, batch_intents, *, agent_id="research"):
            return (
                {
                    "reg_1": ResearchResult(
                        query=None, documents=[_doc("https://www.fda.gov/a")],
                        path=self.path, executed=False,
                        error="IncompleteIntent: 1 unresolved web_search call(s)",
                    )
                },
                {"server_tool_uses": 2, "prompt_tokens": 0, "output_tokens": 0, "actual_total_tokens": 0},
            )

    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, _IncompleteIntentProvider(), agent_id="adversarial_search", discovery=discovery,
        run_id="r1", ticker="TESTCO",
    )
    assert intents[0].status is IntentStatus.INCOMPLETE_RESPONSE
    assert "IncompleteIntent" in intents[0].detail
    assert len(intents[0].documents) == 1
    assert intents[0].documents[0].url == "https://www.fda.gov/a"
    assert intents[0].executed is False
    assert "reg_1" in diagnostics.incomplete_intent_ids
    assert "reg_1" not in diagnostics.completed_intent_ids
    # The intent's own ResearchResult-shaped view keeps the evidence too --
    # never silently dropped by the completion-determination projection.
    view = intents[0].to_research_result()
    assert view.executed is False
    assert [d.url for d in view.documents] == ["https://www.fda.gov/a"]


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


# --- execute_research_batch(): pending/retry for split/allocate exclusions --
class _RetryingBudgetProvider:
    """Simulates a provider whose split/allocate decision can only afford a
    LIMITED number of intents on any one call -- exactly the shape
    AnthropicWebResearchProvider.search_batch() now returns: an intent
    beyond that limit comes back with the certain, never-sent "excluded ...
    by split/allocate" result, never silently absent. ``capacity_by_call``
    gives the limit for call 1, 2, 3, ...; the last value repeats for any
    further call.
    """

    name = "fake_retry"
    path = ResearchPath.ANTHROPIC_WEB

    def __init__(self, capacity_by_call: list[int], documents_by_intent=None):
        self.capacity_by_call = capacity_by_call
        self.documents_by_intent = documents_by_intent or {}
        self.batch_calls: list[list[str]] = []

    def available(self):
        return True, "ready"

    def search_batch(self, intents, *, agent_id="research"):
        self.batch_calls.append([i.intent_id for i in intents])
        call_index = len(self.batch_calls) - 1
        capacity = self.capacity_by_call[min(call_index, len(self.capacity_by_call) - 1)]
        included, excluded = intents[:capacity], intents[capacity:]
        results = {}
        for intent in included:
            results[intent.intent_id] = ResearchResult(
                query=None,
                documents=self.documents_by_intent.get(intent.intent_id, []),
                path=self.path,
                executed=True,
            )
        for intent in excluded:
            results[intent.intent_id] = ResearchResult(
                query=None,
                path=self.path,
                executed=False,
                error=(
                    "BudgetExceeded: excluded from this call by split/allocate -- did not fit "
                    "the remaining budget at dispatch time (0 global token(s) / 0 stage "
                    "token(s) left)."
                ),
            )
        return results, {
            "server_tool_uses": len(included), "prompt_tokens": 100,
            "output_tokens": 50, "actual_total_tokens": 150,
        }


def test_execute_batch_retries_split_allocate_exclusions_against_remaining_budget():
    """同一intentで1回成功、残予算で再評価: intents excluded purely by the
    provider's split/allocate decision are kept PENDING, not finalized --
    once the round that DID send something records its actual usage, the
    same still-pending intents are re-offered and this time succeed."""
    intents = [_intent(ResearchDomain.REGULATORY, intent_id=f"i{i}") for i in range(4)]
    batch = ResearchBatch(batch_id="b1", intents=intents)
    provider = _RetryingBudgetProvider(
        capacity_by_call=[2],  # round 1 affords only 2; every later call affords everything given
        documents_by_intent={f"i{i}": [_doc(f"https://www.fda.gov/{i}")] for i in range(4)},
    )
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO",
    )

    assert provider.batch_calls == [["i0", "i1", "i2", "i3"], ["i2", "i3"]], (
        "the second round must re-offer ONLY the still-pending intents, never resend i0/i1"
    )
    for intent in intents:
        assert intent.status is IntentStatus.EXECUTED_WITH_EVIDENCE, (
            f"{intent.intent_id} should have succeeded once re-offered with remaining budget"
        )
    assert diagnostics.retry_rounds == 1
    assert set(diagnostics.completed_intent_ids) == {"i0", "i1", "i2", "i3"}
    # No duplicate billing/audit record for any intent across the two rounds.
    assert len(discovery.queries) == 4
    assert len({q.query_id for q in discovery.queries}) == 4


def test_execute_batch_stops_retrying_the_moment_a_round_makes_no_progress():
    """予算が尽きた場合の停止条件: a round that excludes every intent it was
    given AGAIN, completely unchanged, means no actual usage was recorded --
    retrying would repeat the identical rejection forever, so this must stop
    immediately rather than spending every remaining retry round."""
    intents = [_intent(ResearchDomain.REGULATORY, intent_id=f"i{i}") for i in range(3)]
    batch = ResearchBatch(batch_id="b1", intents=intents)
    provider = _RetryingBudgetProvider(capacity_by_call=[0])  # never affords anything, ever
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO",
    )
    assert len(provider.batch_calls) == 1, "must not keep retrying once a round makes zero progress"
    for intent in intents:
        assert intent.status is IntentStatus.SKIPPED_DUE_TO_BUDGET
        # The specific reason and this intent's own priority both survive --
        # never a silent drop.
        assert "split/allocate" in intent.detail
    assert diagnostics.retry_rounds == 0
    assert set(diagnostics.incomplete_intent_ids) == {"i0", "i1", "i2"}


def test_execute_batch_retry_rounds_are_bounded_and_remainder_is_finalized():
    """試行回数の上限: with progress every round but never enough to finish,
    retrying must still stop at max_retry_rounds -- whatever is left after
    that is finalized as SKIPPED_DUE_TO_BUDGET (priority/reason preserved),
    never retried forever."""
    intents = [_intent(ResearchDomain.REGULATORY, intent_id=f"i{i}", priority=i) for i in range(10)]
    batch = ResearchBatch(batch_id="b1", intents=intents)
    # Call 1 affords 1 of what it's given, call 2 affords 2, call 3 affords 3
    # -- always SOME progress, never enough to clear 10 intents in 3 rounds.
    provider = _RetryingBudgetProvider(
        capacity_by_call=[1, 2, 3],
        documents_by_intent={f"i{i}": [_doc(f"https://www.fda.gov/{i}")] for i in range(10)},
    )
    discovery = DiscoveryLog()
    diagnostics = execute_research_batch(
        batch, provider, agent_id="adversarial_search", discovery=discovery, run_id="r1", ticker="TESTCO",
        max_retry_rounds=3,
    )
    assert len(provider.batch_calls) == 3, "must stop at max_retry_rounds even though progress never stalled"
    succeeded = [i for i in intents if i.status is IntentStatus.EXECUTED_WITH_EVIDENCE]
    skipped = [i for i in intents if i.status is IntentStatus.SKIPPED_DUE_TO_BUDGET]
    assert len(succeeded) == 6  # 1 + 2 + 3 sent across the three rounds
    assert len(skipped) == 4
    for intent in skipped:
        assert "split/allocate" in intent.detail
        assert intent.priority in {6, 7, 8, 9}  # the lowest-priority intents, never reordered
    assert diagnostics.retry_rounds == 2
    # Every intent still recorded exactly once, success or final skip alike.
    assert len(discovery.queries) == 10


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
