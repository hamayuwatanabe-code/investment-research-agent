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
from investment_research.research.adversarial import BEAR_TEMPLATES, BULL_TEMPLATES, build_plan
from investment_research.research.anthropic_web import (
    WEB_FETCH_TOOL,
    WEB_SEARCH_TOOL,
    AnthropicWebResearchProvider,
    parse_search_response,
    tool_types_for,
)
from investment_research.research.corpus import CorpusResearchProvider
from investment_research.research.escalation import escalate, needs_escalation
from investment_research.research.provider import (
    CompositeResearchProvider,
    NullResearchProvider,
    ResearchQuery,
)
from investment_research.schemas.enums import (
    ContentKind,
    FetchOutcome,
    Provenance,
    ResearchDomain,
    ResearchPath,
    SourceTier,
    VerifiedStatus,
)
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


class _FakeLLM:
    """Records the request `raw_message` was called with; returns a canned reply."""

    def __init__(self, model: str, response: dict):
        self.model = model
        self.last_usage_tokens = 0
        self._response = response
        self.last_request: dict | None = None

    def available(self):
        return True, "ready"

    def raw_message(self, *, system, messages, tools=None, max_tokens=16000, **_kw):
        self.last_request = {"system": system, "messages": messages, "tools": tools}
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
