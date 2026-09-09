"""Unit tests for the Phase 2 research layer."""

from __future__ import annotations

import pytest

from investment_research.collectors.documents import (
    Document,
    build_evidence_pack,
    chunk_document,
    estimate_tokens,
    score_chunk,
    split_sentences,
)
from investment_research.llm.client import BudgetExceeded, LLMBudget
from investment_research.research.adversarial import (
    BEAR_TEMPLATES,
    BULL_TEMPLATES,
    SearchPlan,
    _dedupe_semantically,
    _prioritize_domain_coverage,
    build_plan,
    run_adversarial_search,
)
from investment_research.research.anthropic_web import (
    SEARCH_MAX_TOKENS,
    WEB_FETCH_TOOL,
    WEB_SEARCH_TOOL,
    AnthropicWebResearchProvider,
    parse_search_response,
    tool_types_for,
)
from investment_research.research.corpus import CorpusResearchProvider
from investment_research.research.escalation import (
    domains_for_category,
    escalate,
    escalate_unresolved_questions,
    needs_escalation,
)
from investment_research.research.provider import (
    CompositeResearchProvider,
    NullResearchProvider,
    ResearchQuery,
    ResearchResult,
)
from investment_research.schemas.enums import (
    ContentKind,
    FactCategory,
    FetchOutcome,
    Provenance,
    ResearchDomain,
    ResearchPath,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Source, UnresolvedQuestion, make_source_id
from tests.conftest import make_fact


# --- content kind -----------------------------------------------------------
def test_a_search_summary_is_not_primary_text():
    """The distinction the whole capture story rests on."""
    assert ContentKind.FULL_DOCUMENT.is_primary_text
    assert ContentKind.EXCERPT.is_primary_text
    assert not ContentKind.SEARCH_SUMMARY.is_primary_text
    assert not ContentKind.METADATA_ONLY.is_primary_text
    assert (
        ContentKind.SEARCH_SUMMARY.confidence_multiplier
        < ContentKind.FULL_DOCUMENT.confidence_multiplier
    )


# --- sentence splitting -----------------------------------------------------
@pytest.mark.parametrize(
    "text,expected_first",
    [
        (
            "Longeveron met the U.S. FDA in March. The endpoint failed.",
            "Longeveron met the U.S. FDA in March.",
        ),
        ("Corbus Inc. reported results. Cash was low.", "Corbus Inc. reported results."),
        ("Revenue rose 4%. Costs fell.", "Revenue rose 4%."),
    ],
)
def test_abbreviations_do_not_split_sentences(text, expected_first):
    assert split_sentences(text)[0] == expected_first


def test_split_handles_empty_text():
    assert split_sentences("") == []


# --- chunking and packs -----------------------------------------------------
def _doc(text: str, **kwargs) -> Document:
    return Document(doc_id="d1", url="https://www.sec.gov/x", title="10-Q", text=text, **kwargs)


def test_chunking_respects_a_token_target():
    document = _doc("This is a sentence about the company. " * 80)
    chunks = chunk_document(document, target_tokens=50)
    assert len(chunks) > 1
    assert all(c.token_estimate() <= 120 for c in chunks)


def test_chunk_ids_are_stable_and_ordered():
    chunks = chunk_document(_doc("One. Two. Three. Four. Five. " * 20), target_tokens=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id.startswith("d1#c") for c in chunks)


def test_relevance_scoring_prefers_the_right_agent():
    regulatory = chunk_document(
        _doc("The FDA advised that the primary endpoint is not sufficient."), target_tokens=200
    )[0]
    capital = chunk_document(
        _doc("Cash runway extends into Q4 and a private placement closed."), target_tokens=200
    )[0]
    assert score_chunk(regulatory, "regulatory") > score_chunk(capital, "regulatory")
    assert score_chunk(capital, "capital_structure") > score_chunk(regulatory, "capital_structure")


def test_evidence_pack_respects_its_budget():
    chunks = chunk_document(_doc("The FDA endpoint concern is material. " * 200), target_tokens=40)
    pack = build_evidence_pack(chunks, "regulatory", budget_tokens=200)
    assert pack.total_tokens <= 200
    assert pack.dropped_chunks > 0
    assert len(pack.chunks) < len(chunks)


def test_universal_signals_survive_a_tight_budget():
    """A going-concern disclosure must not be squeezed out by a token budget."""
    filler = "Routine corporate description of office facilities. " * 40
    document = _doc(
        filler + "The company discloses substantial doubt about its ability to continue as a "
        "going concern. " + filler
    )
    chunks = chunk_document(document, target_tokens=40)
    pack = build_evidence_pack(chunks, "bull_agent", budget_tokens=120)
    assert any("going concern" in c.text for c in pack.chunks)


def test_pack_render_carries_citations():
    chunks = chunk_document(
        _doc("The FDA advised the endpoint is not sufficient."), target_tokens=200
    )
    rendered = build_evidence_pack(chunks, "regulatory", budget_tokens=500).render()
    assert "d1#c0" in rendered
    assert "https://www.sec.gov/x" in rendered
    assert "tier=TIER_1" not in rendered or "tier=" in rendered


def test_token_estimate_is_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# --- anthropic web parsing --------------------------------------------------
def test_tool_variants_by_model():
    assert tool_types_for("claude-opus-5")[0] == "web_search_20260318"
    assert tool_types_for("claude-sonnet-5")[0] == "web_search_20260318"
    assert tool_types_for("claude-haiku-4-5")[0] == "web_search_20250305"


def test_search_results_become_documents():
    documents, error = parse_search_response(
        {
            "content": [
                {
                    "type": "web_search_tool_result",
                    "content": [
                        {"url": "https://www.sec.gov/a", "title": "10-Q", "page_age": "2026-08-07"},
                        {"url": "https://seekingalpha.com/b", "title": "Opinion"},
                    ],
                }
            ]
        }
    )
    assert error == ""
    assert [d.tier for d in documents] == [SourceTier.TIER_1, SourceTier.TIER_5]
    assert all(d.content_kind is ContentKind.METADATA_ONLY for d in documents)


def test_a_failed_search_is_an_error_not_an_empty_result():
    """The distinction that keeps "we did not look" out of "nothing found"."""
    documents, error = parse_search_response(
        {
            "content": [
                {"type": "web_search_tool_result", "content": {"error_code": "max_uses_exceeded"}}
            ]
        }
    )
    assert documents == []
    assert "max_uses_exceeded" in error


def test_no_search_block_is_neither_documents_nor_error():
    documents, error = parse_search_response({"content": [{"type": "text", "text": "hello"}]})
    assert documents == [] and error == ""


def test_missing_result_content_is_never_a_normal_zero_result():
    """content欠落/None: a result block with no ``content`` key at all (or an
    explicit ``None``) is a malformed/truncated response, never a genuine
    "searched, found nothing" outcome."""
    documents, error = parse_search_response(
        {"content": [{"type": "web_search_tool_result", "content": None}]}
    )
    assert documents == []
    assert error != ""
    assert "missing" in error


def test_malformed_result_content_is_never_a_normal_zero_result():
    """不正構造: a ``content`` shape that is neither a list of hits nor a
    recognizable error object must still be reported as an error, not as
    zero results."""
    documents, error = parse_search_response(
        {"content": [{"type": "web_search_tool_result", "content": 12345}]}
    )
    assert documents == []
    assert error != ""


def test_genuine_empty_list_content_is_a_normal_zero_result():
    """content=[]: a well-formed, empty list IS a genuine zero-result
    search -- must stay indistinguishable in outcome from any other clean
    search that simply found nothing, and distinct from the missing/
    malformed cases above."""
    documents, error = parse_search_response(
        {"content": [{"type": "web_search_tool_result", "content": []}]}
    )
    assert documents == []
    assert error == ""


class _FakeLLM:
    """Records the request `raw_message` was called with; returns a canned reply."""

    def __init__(self, model: str, response: dict, *, raises: Exception | None = None):
        self.model = model
        self.last_usage_tokens = 0
        self._response = response
        self._raises = raises
        self.last_request: dict | None = None

    def available(self):
        return True, "ready"

    def raw_message(self, *, system, messages, tools=None, max_tokens=16000, effort=None, **_kw):
        self.last_request = {
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "effort": effort,
        }
        if self._raises is not None:
            raise self._raises
        return self._response


def test_current_web_tools_selected_and_direct_caller_marked_for_sonnet_5():
    """Sonnet 5 must get the current (20260318) dated tools, called directly."""
    llm = _FakeLLM("claude-sonnet-5", {"content": []})
    provider = AnthropicWebResearchProvider(llm)

    provider.search(ResearchQuery(query="q", domain=ResearchDomain.REGULATORY))
    search_tool = llm.last_request["tools"][0]
    assert search_tool["type"] == WEB_SEARCH_TOOL
    assert search_tool["allowed_callers"] == ["direct"]

    provider.fetch("https://www.sec.gov/x", reason="verify")
    fetch_tool = llm.last_request["tools"][0]
    assert fetch_tool["type"] == WEB_FETCH_TOOL
    assert fetch_tool["allowed_callers"] == ["direct"]


def test_older_basic_tool_variant_is_not_given_allowed_callers():
    """The compatibility fallback for older models doesn't carry the new param."""
    llm = _FakeLLM("claude-haiku-4-5", {"content": []})
    provider = AnthropicWebResearchProvider(llm)

    provider.search(ResearchQuery(query="q", domain=ResearchDomain.REGULATORY))
    search_tool = llm.last_request["tools"][0]
    assert search_tool["type"] == "web_search_20250305"
    assert "allowed_callers" not in search_tool


def test_provider_keeps_search_hits_separate_from_fetched_document_bodies():
    """Evidence integrity: a search hit is a pointer, a fetch is the document.

    The Decision-Grade Evidence Gate depends on these never being conflated --
    a SEARCH_SUMMARY-shaped result must stay METADATA_ONLY even after the
    provider has also fetched a FULL_DOCUMENT for a different URL.
    """
    search_llm = _FakeLLM(
        "claude-sonnet-5",
        {
            "content": [
                {
                    "type": "web_search_tool_result",
                    "content": [{"url": "https://www.sec.gov/a", "title": "10-Q"}],
                }
            ]
        },
    )
    search_provider = AnthropicWebResearchProvider(search_llm)
    search_result = search_provider.search(
        ResearchQuery(query="q", domain=ResearchDomain.REGULATORY)
    )
    assert len(search_result.documents) == 1
    assert search_result.documents[0].content_kind is ContentKind.METADATA_ONLY

    import types

    fetch_block = types.SimpleNamespace(
        type="web_fetch_tool_result",
        content={
            "content": {"source": {"data": "full filing text"}, "title": "10-Q"},
            "retrieved_at": "2026-09-08T00:00:00+00:00",
        },
    )
    fetch_llm = _FakeLLM("claude-sonnet-5", types.SimpleNamespace(content=[fetch_block]))
    fetch_provider = AnthropicWebResearchProvider(fetch_llm)
    fetched = fetch_provider.fetch("https://www.sec.gov/a", reason="verify")
    assert fetched is not None
    assert fetched.content_kind is ContentKind.FULL_DOCUMENT
    assert fetched.text == "full filing text"

    # The two are still distinct records -- fetching one URL never upgrades or
    # mutates a search hit recorded separately.
    assert search_result.documents[0].content_kind is ContentKind.METADATA_ONLY


def _search_result_response(count: int) -> dict:
    return {
        "content": [
            {
                "type": "web_search_tool_result",
                "content": [
                    {"url": f"https://www.sec.gov/{i}", "title": f"Doc {i}"}
                    for i in range(count)
                ],
            }
        ]
    }


def test_search_enforces_query_max_results():
    """A provider returning 10 hits for a max_results=3 query must yield exactly 3."""
    llm = _FakeLLM("claude-sonnet-5", _search_result_response(10))
    provider = AnthropicWebResearchProvider(llm)

    result = provider.search(
        ResearchQuery(query="q", domain=ResearchDomain.REGULATORY, max_results=3)
    )

    assert len(result.documents) == 3
    # Ordering preserved: the first 3 of the original 10, in order.
    assert [d.url for d in result.documents] == [
        "https://www.sec.gov/0",
        "https://www.sec.gov/1",
        "https://www.sec.gov/2",
    ]
    # Evidence integrity untouched by the cap.
    assert all(d.content_kind is ContentKind.METADATA_ONLY for d in result.documents)


def test_search_max_results_never_exceeded_even_when_fewer_returned():
    llm = _FakeLLM("claude-sonnet-5", _search_result_response(2))
    provider = AnthropicWebResearchProvider(llm)

    result = provider.search(
        ResearchQuery(query="q", domain=ResearchDomain.REGULATORY, max_results=3)
    )
    assert len(result.documents) == 2


def test_search_output_ceiling_is_reduced_for_discovery():
    """Discovery-only search calls must not need a long model answer (cost control)."""
    assert SEARCH_MAX_TOKENS < 2000, "search is discovery-only; it never needed 8000 tokens"

    llm = _FakeLLM("claude-sonnet-5", {"content": []})
    provider = AnthropicWebResearchProvider(llm)
    provider.search(ResearchQuery(query="q", domain=ResearchDomain.REGULATORY))
    assert llm.last_request["max_tokens"] == SEARCH_MAX_TOKENS


# --- research effort decoupled from interpretive (--llm-effort) effort ------
def test_research_provider_defaults_to_low_effort_search_and_fetch():
    """The provider must not inherit whatever the client's own .effort is
    (that governs the interpretive agents via --llm-effort) -- it has its own
    independently-controlled research_effort, defaulting to low."""
    llm = _FakeLLM("claude-sonnet-5", {"content": []})
    provider = AnthropicWebResearchProvider(llm)
    assert provider.research_effort == "low"

    provider.search(ResearchQuery(query="q", domain=ResearchDomain.REGULATORY))
    assert llm.last_request["effort"] == "low"

    llm2 = _FakeLLM(
        "claude-sonnet-5",
        types_namespace_fetch_response(),
    )
    provider2 = AnthropicWebResearchProvider(llm2)
    provider2.fetch("https://www.sec.gov/x", reason="verify")
    assert llm2.last_request["effort"] == "low"


def test_research_provider_uses_explicit_research_effort_not_client_effort():
    """Even when the shared client's own .effort is 'high' (--llm-effort high,
    used by the interpretive agents), --research-effort controls discovery
    independently: a client configured for high-effort interpretation must not
    silently make discovery high-effort too."""
    llm = _FakeLLM("claude-sonnet-5", {"content": []})
    llm.effort = "high"  # what --llm-effort high would set on the shared client
    provider = AnthropicWebResearchProvider(llm, research_effort="low")

    provider.search(ResearchQuery(query="q", domain=ResearchDomain.REGULATORY))
    assert llm.last_request["effort"] == "low", (
        "discovery must use --research-effort, never leak the client's --llm-effort"
    )


def types_namespace_fetch_response():
    import types

    return types.SimpleNamespace(
        content=[
            types.SimpleNamespace(
                type="web_fetch_tool_result",
                content={
                    "content": {"source": {"data": "text"}, "title": "t"},
                    "retrieved_at": "2026-09-08T00:00:00+00:00",
                },
            )
        ]
    )


def test_follow_up_generation_uses_research_effort():
    """Follow-up query proposal is discovery work, not interpretation -- it
    must honor --research-effort, not the client's configured --llm-effort."""
    from investment_research.research.adversarial import _generate_follow_ups

    class _EffortCapturingLLM:
        def __init__(self):
            self.structured_kwargs: dict | None = None

        def available(self):
            return True, "ready"

        def structured(self, **kwargs):
            self.structured_kwargs = kwargs
            return {"queries": []}

    outcome = AdversarialOutcomeForTest()
    llm = _EffortCapturingLLM()
    _generate_follow_ups(llm, outcome, limit=5, research_effort="low")
    assert llm.structured_kwargs is not None
    assert llm.structured_kwargs["effort"] == "low"


class AdversarialOutcomeForTest:
    """Minimal stand-in exposing just what _generate_follow_ups reads."""

    def bear_documents(self):
        from investment_research.collectors.documents import Document
        from investment_research.schemas.enums import ContentKind, Provenance

        return [
            Document(
                doc_id="d1",
                url="https://www.sec.gov/a",
                title="10-Q",
                publisher="sec.gov",
                published_date="2026-01-01",
                doc_type="filing",
                text="a material disclosure",
                content_kind=ContentKind.FULL_DOCUMENT,
                provenance=Provenance.LIVE,
            )
        ]


def test_search_translates_budget_exceeded_into_unsearched_not_empty_result():
    """A budget cutoff is 'we did not look', not 'we looked and found nothing'.

    executed=False is what lets the Search Completeness Gate mark this domain
    FAILED/UNSEARCHED rather than silently SEARCHED with zero documents --
    the exact distinction CLAUDE.md rule 8 exists to protect.
    """
    llm = _FakeLLM(
        "claude-sonnet-5", {"content": []}, raises=BudgetExceeded("budget exhausted")
    )
    provider = AnthropicWebResearchProvider(llm)

    result = provider.search(ResearchQuery(query="q", domain=ResearchDomain.REGULATORY))

    assert result.executed is False
    assert result.documents == []
    assert result.outcome is FetchOutcome.DISABLED
    assert "BudgetExceeded" in result.error


def test_fetch_returns_none_on_budget_exceeded():
    llm = _FakeLLM(
        "claude-sonnet-5", {"content": []}, raises=BudgetExceeded("budget exhausted")
    )
    provider = AnthropicWebResearchProvider(llm)
    assert provider.fetch("https://www.sec.gov/x", reason="verify") is None


# --- providers --------------------------------------------------------------
def test_null_provider_reports_not_executed():
    result = NullResearchProvider().search(
        ResearchQuery(query="q", domain=ResearchDomain.REGULATORY)
    )
    assert result.executed is False
    assert result.outcome is FetchOutcome.DISABLED


def test_composite_falls_through_to_an_available_provider(fixture_dir):
    corpus = CorpusResearchProvider(fixture_dir.parent / "corpus", "LGVN")
    composite = CompositeResearchProvider([NullResearchProvider(), corpus])
    usable, _ = composite.available()
    assert usable
    result = composite.search(
        ResearchQuery(query="Longeveron FDA endpoint pivotal", domain=ResearchDomain.REGULATORY)
    )
    assert result.documents
    assert result.path is ResearchPath.CORPUS


def test_corpus_provider_marks_provenance_and_capture(fixture_dir):
    corpus = CorpusResearchProvider(fixture_dir.parent / "corpus", "LGVN")
    info = corpus.capture_info()
    assert info["captured_via"] == "anthropic_websearch"
    documents = corpus.documents()
    assert documents
    assert all(d.provenance is Provenance.CAPTURED for d in documents)
    assert all(d.url.startswith("https://") for d in documents)


def test_corpus_rejects_a_non_http_document(tmp_path):
    import json

    (tmp_path / "FAKE.json").write_text(
        json.dumps(
            {
                "ticker": "FAKE",
                "captured_at": "2026-01-01T00:00:00+00:00",
                "documents": [
                    {"doc_id": "d", "url": "fixture://not-real", "title": "t", "text": "x"}
                ],
            }
        )
    )
    assert CorpusResearchProvider(tmp_path, "FAKE").documents() == []


# --- adversarial ------------------------------------------------------------
def test_mandatory_bear_queries_are_all_present():
    plan = build_plan("LGVN", "Longeveron", ["laromestrocel"])
    queries = " | ".join(q.query.lower() for q in plan.bear)
    for required in (
        "fda concern",
        "regulatory risk",
        "endpoint concern",
        "failed trial",
        "dilution",
        "going concern",
        "warrant",
        "reverse split",
        "delisting",
        "lawsuit",
        "auditor",
        "insider selling",
        "criticism",
        "short thesis",
        "competitor superiority",
        "safety concern",
    ):
        assert required in queries, f"mandatory bear query missing: {required}"


def test_bear_and_bull_plans_are_separate():
    plan = build_plan("LGVN", "Longeveron")
    assert len(plan.bear) == len(BEAR_TEMPLATES)
    assert len(plan.bull) == len(BULL_TEMPLATES)
    assert all(q.stance == "bear" for q in plan.bear)
    assert all(q.stance == "bull" for q in plan.bull)
    assert not {q.query for q in plan.bear} & {q.query for q in plan.bull}


# --- adaptive adversarial search (requirement F) -----------------------------
def test_prioritize_domain_coverage_runs_every_domains_first_query_before_seconds():
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    domains_before = [q.domain for q in plan.bear]
    # The raw template list has at least one domain (CAPITAL_STRUCTURE) with
    # several queries clustered together -- proving the reorder actually moves
    # something, not merely confirming an already-sorted list.
    assert domains_before.count(ResearchDomain.CAPITAL_STRUCTURE) > 1

    reordered = _prioritize_domain_coverage(plan.bear)
    assert {q.query for q in reordered} == {q.query for q in plan.bear}, "no query lost or added"

    # After exactly one query per distinct domain, every domain must already
    # have appeared once -- a budget cutoff right there still covered them all.
    all_domains = {q.domain for q in plan.bear}
    first_n = reordered[: len(all_domains)]
    assert {q.domain for q in first_n} == all_domains


def test_dedupe_semantically_drops_near_duplicate_queries():
    queries = [
        ResearchQuery(query="Generic Biotech Holdings dilution risk", domain=ResearchDomain.CAPITAL_STRUCTURE),
        # Same keywords, reworded -- the "same question asked two different
        # ways" case this heuristic exists to catch.
        ResearchQuery(query="dilution risk Generic Biotech Holdings", domain=ResearchDomain.CAPITAL_STRUCTURE),
        ResearchQuery(query="Generic Biotech Holdings FDA endpoint concern", domain=ResearchDomain.REGULATORY),
    ]
    kept, dropped = _dedupe_semantically(queries, threshold=0.85)
    assert len(kept) == 2
    assert dropped == ["dilution risk Generic Biotech Holdings"]


def test_dedupe_semantically_does_not_drop_the_standard_mandatory_templates():
    """The real mandatory bear/bull templates must survive dedup untouched --
    they are deliberately distinct topics, not near-duplicates of each other."""
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    kept, dropped = _dedupe_semantically(plan.bear)
    assert dropped == []
    assert len(kept) == len(plan.bear)


def test_dedupe_semantically_respects_seed_tokens_from_an_earlier_batch():
    already_asked = [_query_tokens_for_test("Generic Biotech Holdings dilution risk")]
    follow_ups = [
        ResearchQuery(query="Generic Biotech Holdings dilution risk", domain=ResearchDomain.CAPITAL_STRUCTURE),
        ResearchQuery(query="Generic Biotech Holdings going concern doubt", domain=ResearchDomain.CAPITAL_STRUCTURE),
    ]
    kept, dropped = _dedupe_semantically(follow_ups, seed_tokens=already_asked)
    assert [q.query for q in kept] == ["Generic Biotech Holdings going concern doubt"]
    assert dropped == ["Generic Biotech Holdings dilution risk"]


def _query_tokens_for_test(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


class _BudgetCuttingProvider:
    """A ResearchProvider double: the first `allow` queries execute, then
    every further one is refused exactly as AnthropicWebResearchProvider
    refuses one on a BudgetExceeded abort."""

    name = "fake_discovery_provider"
    path = ResearchPath.ANTHROPIC_WEB

    def __init__(self, allow: int):
        self.allow = allow
        self.calls = 0
        self.agent_ids: list[str] = []

    def available(self):
        return True, "ready"

    def search(self, query, *, agent_id: str = "research"):
        self.calls += 1
        self.agent_ids.append(agent_id)
        if self.calls > self.allow:
            return ResearchResult(
                query=query,
                outcome=FetchOutcome.DISABLED,
                path=self.path,
                executed=False,
                error="BudgetExceeded: LLM token budget already exhausted",
            )
        return ResearchResult(query=query, documents=[], outcome=FetchOutcome.NOT_FOUND, path=self.path)

    def fetch(self, url, *, reason="", agent_id: str = "research", known=None):
        return None


def test_queries_cut_off_by_budget_are_recorded_distinctly_not_as_searched():
    provider = _BudgetCuttingProvider(allow=3)
    plan = SearchPlan(
        bear=[
            ResearchQuery(query=f"query {i}", domain=domain)
            for i, domain in enumerate(ResearchDomain)
        ],
        bull=[],
    )
    outcome = run_adversarial_search(provider, plan, run_id="r1", ticker="TESTCO")

    assert len(outcome.unexecuted_due_to_budget) == len(plan.bear) - 3
    assert set(outcome.unexecuted_due_to_budget) <= set(outcome.unexecuted)
    # Every skipped query is still on the record -- the completeness gate can
    # see it -- just tagged executed=False, never "searched, nothing found".
    skipped_records = [q for q in outcome.discovery.queries if not q.executed]
    assert len(skipped_records) == len(outcome.unexecuted_due_to_budget)


# --- required-domain core discovery must fit the discovery budget (F) ------
def test_bear_core_pass_alone_covers_every_required_domain_before_any_second_query():
    """When the discovery budget can only fund one query per required domain,
    that must be exactly what gets funded -- not five domains plus a second
    query for one of them, leaving a sixth domain completely unsearched (the
    live-run defect: only 2 of 6 required domains got searched before the
    budget ran out). The bear pass's own first-pass core must cover all six
    domains on its own, without depending on the (possibly never-reached)
    bull pass."""
    provider = _BudgetCuttingProvider(allow=len(ResearchDomain))
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    outcome = run_adversarial_search(provider, plan, run_id="r1", ticker="TESTCO")

    executed_bear_domains = {r.query.domain for r in outcome.bear_results if r.executed}
    assert executed_bear_domains == set(ResearchDomain), (
        f"expected all {len(ResearchDomain)} required domains covered by the bear core "
        f"pass alone, got {executed_bear_domains}"
    )
    # And it never pretends the 7th+ (second-round) queries were searched.
    assert outcome.unexecuted_due_to_budget, "queries past the core pass must be recorded cut"


def test_bear_bull_and_followup_calls_are_tagged_with_distinct_agent_ids():
    provider = _BudgetCuttingProvider(allow=1000)
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    run_adversarial_search(provider, plan, run_id="r1", ticker="TESTCO")
    assert "adversarial_bear" in provider.agent_ids
    assert "adversarial_bull" in provider.agent_ids


class _FakeLLMForFollowUps:
    def __init__(self, *, budget: LLMBudget):
        self.budget = budget
        self.structured_calls = 0

    def available(self):
        return True, "ready"

    def structured(self, **_kw):
        self.structured_calls += 1
        return {"queries": []}


def test_follow_ups_are_never_generated_once_discovery_quota_already_exhausted():
    provider = _BudgetCuttingProvider(allow=1000)
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    budget = LLMBudget(max_total_tokens=1000)
    budget.set_stage("discovery")
    budget.stage_exhausted.add("discovery")  # simulate the bear pass having used it all
    llm = _FakeLLMForFollowUps(budget=budget)

    run_adversarial_search(provider, plan, llm=llm, run_id="r1", ticker="TESTCO")

    assert llm.structured_calls == 0, "no point spending tokens proposing searches we can't run"


def test_discovery_stage_is_set_on_the_llm_budget_during_the_run():
    """Requirement A: the stage is scoped, not ambient.

    ``run_adversarial_search`` sets "discovery" for the duration of its own
    work (every query it issues must be attributed to it) and restores
    whatever was active before on return, so it can never leak into
    whatever the caller runs next -- unlike the old ambient
    ``budget.set_stage("discovery")`` call this replaced, which left
    "discovery" active indefinitely after the function returned.
    """
    provider = _BudgetCuttingProvider(allow=1000)
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    budget = LLMBudget(max_total_tokens=1000)
    llm = _FakeLLMForFollowUps(budget=budget)

    stages_seen: list[str] = []
    original_search = provider.search

    def _tracking_search(query, *, agent_id: str = "research"):
        stages_seen.append(budget.current_stage)
        return original_search(query, agent_id=agent_id)

    provider.search = _tracking_search  # type: ignore[method-assign]

    run_adversarial_search(provider, plan, llm=llm, run_id="r1", ticker="TESTCO")

    assert stages_seen, "the provider must have been called at least once"
    assert all(stage == "discovery" for stage in stages_seen), (
        "every query issued while the adversarial pass is running must be attributed to the "
        f"'discovery' stage, got {stages_seen}"
    )
    assert budget.current_stage == "", (
        "the stage must be restored once the pass returns, not left dangling"
    )


# --- escalation -------------------------------------------------------------
def test_a_primary_source_claim_needs_no_escalation():
    fact = make_fact("The FDA advised the endpoint is not sufficient", tier=SourceTier.TIER_1)
    assert needs_escalation(fact) == (False, "")


def test_a_material_news_claim_needs_escalation():
    fact = make_fact(
        "The FDA advised the endpoint is not sufficient",
        tier=SourceTier.TIER_3,
        url="https://www.reuters.com/a",
    )
    required, reason = needs_escalation(fact)
    assert required and reason == "regulator_position"


def test_a_routine_claim_does_not():
    fact = make_fact(
        "The company opened an office in Boston",
        tier=SourceTier.TIER_3,
        url="https://www.reuters.com/a",
    )
    assert needs_escalation(fact)[0] is False


def test_unconfirmed_material_claims_are_marked_and_barred():
    fact = make_fact(
        "The FDA advised the endpoint is not sufficient",
        tier=SourceTier.TIER_3,
        url="https://www.reuters.com/a",
    )
    facts, report = escalate([fact], NullResearchProvider(), company="Test Co")
    assert facts[0].verified_status is VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM
    assert not facts[0].is_decision_grade
    assert report.not_attempted


# --- unresolved-question-driven escalation (requirement C) ------------------
class _FakeProviderReturningBody:
    """A ResearchProvider double: fetch() always returns a canned document
    body; search() should never be reached when a Tier 1/2 source is already
    in hand (the point of this requirement)."""

    def __init__(self, body_text: str, *, doc_type: str = "filing", is_company_ir: bool = False):
        self.body_text = body_text
        self.doc_type = doc_type
        self.is_company_ir = is_company_ir
        self.search_calls = 0
        self.fetch_calls = 0

    def available(self):
        return True, "ready"

    def search(self, query, *, agent_id: str = "research"):
        self.search_calls += 1
        return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=ResearchPath.ANTHROPIC_WEB)

    def fetch(self, url, *, reason="", agent_id: str = "research", known=None):
        self.fetch_calls += 1
        from investment_research.collectors.documents import Document
        from investment_research.collectors.tiering import classify_authority

        return Document(
            doc_id="d1",
            url=url,
            title="Form 8-K",
            publisher="sec.gov",
            published_date="2026-08-01",
            doc_type=self.doc_type,
            is_company_ir=self.is_company_ir,
            text=self.body_text,
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            tier=SourceTier.TIER_1,
            authority=classify_authority(url, is_company_ir=self.is_company_ir),
        )


_REGULATOR_QUESTION = UnresolvedQuestion(
    question="Does the regulator consider the primary endpoint appropriate to establish "
    "effectiveness for the intended indication?",
    why_it_matters="A rejected endpoint invalidates the registrational path this thesis "
    "assumes exists.",
    blocking=True,
    category=FactCategory.REGULATORY,
)

_ALREADY_COLLECTED_FILING = Source(
    source_id=make_source_id("https://www.sec.gov/filing/8-K", "Form 8-K"),
    url="https://www.sec.gov/filing/8-K",
    title="Form 8-K",
    tier=SourceTier.TIER_1,
)


def test_escalation_resolves_a_material_unresolved_question_from_an_already_collected_filing():
    """Positive case (requirement C): the collector already found a Tier 1
    filing pointer; the material question is unresolved; escalation fetches
    the body; adverse regulator language appears; the resulting fact becomes
    decision-grade; no search snippet is promoted, and no redundant search
    is issued when a plausible primary source is already in hand."""
    provider = _FakeProviderReturningBody(
        "In written responses, the agency stated that it does not consider the primary "
        "endpoint appropriate to establish effectiveness for the intended indication."
    )
    facts, report = escalate_unresolved_questions(
        [_REGULATOR_QUESTION],
        [_ALREADY_COLLECTED_FILING],
        provider,
        company="Generic Biotech Holdings",
        ticker="TESTCO",
        run_id="r1",
    )
    assert provider.search_calls == 0, (
        "an already-collected Tier 1 source must not force a redundant web search"
    )
    assert provider.fetch_calls == 1
    assert len(facts) == 1
    assert facts[0].is_decision_grade
    assert facts[0].content_kind is ContentKind.FULL_DOCUMENT
    assert "endpoint" in facts[0].claim.lower()
    assert report.confirmed


def test_escalation_leaves_the_question_unresolved_when_the_body_does_not_answer_it():
    """Inverse case (requirement C): the filing exists and is fetched, but its
    body does not actually address the question -- it must stay unresolved,
    and no fact may be fabricated from an unrelated body."""
    provider = _FakeProviderReturningBody(
        "The company also announced a new manufacturing facility lease in North Carolina "
        "and reiterated its quarterly guidance for the coming fiscal year."
    )
    facts, report = escalate_unresolved_questions(
        [_REGULATOR_QUESTION],
        [_ALREADY_COLLECTED_FILING],
        provider,
        company="Generic Biotech Holdings",
        ticker="TESTCO",
        run_id="r1",
    )
    assert facts == []
    assert not report.confirmed
    assert report.attempts[0].note != ""


def test_escalation_falls_back_to_search_when_nothing_already_collected_answers():
    provider = _FakeProviderReturningBody(
        "In written responses, the agency stated that it does not consider the primary "
        "endpoint appropriate to establish effectiveness for the intended indication."
    )
    # No already-collected sources at all: escalation must fall through to a
    # search-then-fetch path rather than simply giving up.
    facts, report = escalate_unresolved_questions(
        [_REGULATOR_QUESTION],
        [],
        provider,
        company="Generic Biotech Holdings",
        ticker="TESTCO",
        run_id="r1",
    )
    assert report.attempts[0].searched is True


def test_non_material_unresolved_questions_are_never_escalated():
    non_material = UnresolvedQuestion(
        question="What is the pre-specified statistical power?",
        why_it_matters="Nice to know.",
        blocking=False,
        category=FactCategory.CLINICAL,
    )
    provider = _FakeProviderReturningBody("irrelevant")
    facts, report = escalate_unresolved_questions(
        [non_material], [_ALREADY_COLLECTED_FILING], provider, company="Generic Biotech Holdings"
    )
    assert facts == []
    assert report.attempts == []
    assert provider.fetch_calls == 0


# --- question-aware domain routing (requirement C) ---------------------------
def test_domains_for_category_capital_structure_is_sec_only():
    domains = domains_for_category(FactCategory.CAPITAL_STRUCTURE)
    assert domains == ("sec.gov",)
    assert "fda.gov" not in domains
    assert "clinicaltrials.gov" not in domains


def test_domains_for_category_regulatory_includes_fda_and_sec():
    domains = domains_for_category(FactCategory.REGULATORY)
    assert "fda.gov" in domains
    assert "sec.gov" in domains


def test_domains_for_category_science_never_queries_sec():
    domains = domains_for_category(FactCategory.SCIENCE)
    assert "sec.gov" not in domains
    assert "clinicaltrials.gov" in domains


class _DomainRecordingProvider:
    """A ResearchProvider double that records every ``allowed_domains`` a
    search was restricted to, without ever answering from a fetched body --
    forces escalation past the already-collected-sources step and into an
    actual search, so the domain routing on the search itself is exercised."""

    def __init__(self) -> None:
        self.searched_domains: list[str] = []

    def available(self):
        return True, "ready"

    def search(self, query, *, agent_id: str = "research"):
        self.searched_domains.extend(query.allowed_domains)
        return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=ResearchPath.ANTHROPIC_WEB)

    def fetch(self, url, *, reason="", agent_id: str = "research", known=None):
        return None


def test_unrelated_capital_structure_question_never_queries_fda_gov():
    """Requirement C / H4: the exact defect the v3 report showed -- a fully
    diluted share count question searched fda.gov, which cannot possibly
    answer it."""
    question = UnresolvedQuestion(
        question="What is the fully diluted share count including all outstanding warrants?",
        why_it_matters="Understates dilution if wrong.",
        blocking=True,
        category=FactCategory.CAPITAL_STRUCTURE,
    )
    provider = _DomainRecordingProvider()
    escalate_unresolved_questions(
        [question], [], provider, company="Generic Biotech Holdings", ticker="TESTCO", run_id="r1"
    )
    assert provider.searched_domains, "the question must have actually been searched"
    assert "fda.gov" not in provider.searched_domains
    assert "clinicaltrials.gov" not in provider.searched_domains
    assert all(d == "sec.gov" for d in provider.searched_domains)


def test_science_question_never_queries_sec_gov():
    question = UnresolvedQuestion(
        question="Was the trial designed as a randomized, double-blind study?",
        why_it_matters="Trial design determines evidentiary weight.",
        blocking=True,
        category=FactCategory.SCIENCE,
    )
    provider = _DomainRecordingProvider()
    escalate_unresolved_questions(
        [question], [], provider, company="Generic Biotech Holdings", ticker="TESTCO", run_id="r1"
    )
    assert provider.searched_domains
    assert "sec.gov" not in provider.searched_domains
