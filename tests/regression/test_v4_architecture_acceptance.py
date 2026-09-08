"""Synthetic end-to-end acceptance regressions for the LGVN v3 live-run
architectural fixes (requirement H).

Six properties, each proven by a synthetic scenario using generic
placeholders only -- no real ticker, company, endpoint, meeting type, or
indication is named anywhere in this file:

1. collector direct coverage + targeted web search can satisfy required-
   domain coverage without six independent web-search calls.
2. a material regulatory unresolved question with many generic SEC filings
   results in targeted candidate selection and <=2 initial fetches.
3. an adverse statement in a fetched statutory filing/exhibit becomes
   decision-grade as issuer-disclosed statutory evidence.
4. an unrelated capital-structure question never queries fda.gov.
5. stage exhaustion in escalation does not poison Contradiction/Kill/Bear/
   Bull.
6. provider-configured Kill failure never reports NO_PROVIDER incorrectly.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from investment_research.collectors.base import CollectionResult
from investment_research.collectors.documents import Document
from investment_research.collectors.fixtures import FixtureCollector
from investment_research.collectors.search import NullSearchProvider
from investment_research.llm.client import LLMBudget, LLMClient
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.research.adversarial import build_plan
from investment_research.research.anthropic_web import AnthropicWebResearchProvider
from investment_research.research.escalation import escalate_unresolved_questions
from investment_research.research.provider import ResearchResult
from investment_research.schemas.enums import (
    ContentKind,
    DocumentAuthority,
    FactCategory,
    FetchOutcome,
    KillSearchFailureReason,
    Provenance,
    ResearchDomain,
    ResearchPath,
    SourceTier,
)
from investment_research.schemas.fact import Source, UnresolvedQuestion, make_source_id
from investment_research.scoring.completeness import direct_collector_coverage

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 6)


class _EmptySearchHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "web_search_tool_result", "content": []}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _EmptySearchHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


# --- 1. collector coverage reduces required-domain web-search calls ---------
def test_acceptance_1_collector_coverage_avoids_six_independent_web_searches():
    collection_results = [
        CollectionResult(collector="sec_edgar", outcome=FetchOutcome.OK),
        CollectionResult(collector="clinicaltrials", outcome=FetchOutcome.OK),
        CollectionResult(collector="fda", outcome=FetchOutcome.OK),
    ]
    strong, _weak = direct_collector_coverage(collection_results)
    plan = build_plan("TESTCO", "Generic Biotech Holdings", already_covered_domains=strong)

    core_domains_queried = {q.domain for q in plan.bear}
    # REGULATORY, CAPITAL_STRUCTURE and SCIENCE_TECHNOLOGY are directly
    # covered by the three collectors above -- they must not need their own
    # web-search calls; only the domains no collector can ever cover remain.
    assert core_domains_queried.isdisjoint(
        {ResearchDomain.REGULATORY, ResearchDomain.CAPITAL_STRUCTURE, ResearchDomain.SCIENCE_TECHNOLOGY}
    )
    assert len(core_domains_queried) < 6
    assert plan.skipped_due_to_direct_coverage


# --- 2. targeted candidate selection, <=2 initial fetches -------------------
def test_acceptance_2_material_question_many_generic_filings_targets_and_caps_fetches():
    filings = [
        Source(
            source_id=make_source_id(f"https://www.sec.gov/Archives/x/8-K-{i}.htm", "8-K"),
            url=f"https://www.sec.gov/Archives/x/8-K-{i}.htm",
            title="8-K CURRENT REPORT",
            tier=SourceTier.TIER_1,
        )
        for i in range(8)
    ]
    question = UnresolvedQuestion(
        question="Does the regulator consider the primary endpoint appropriate to establish "
        "effectiveness for the intended indication?",
        why_it_matters="A rejected endpoint invalidates the registrational path.",
        blocking=True,
        category=FactCategory.REGULATORY,
    )

    class _Provider:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def __init__(self):
            self.fetch_calls: list[str] = []

        def available(self):
            return True, "ready"

        def search(self, query, *, agent_id="research"):
            return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=self.path)

        def fetch(self, url, *, reason="", agent_id="research", known=None):
            self.fetch_calls.append(url)
            return Document(
                doc_id=url,
                url=url,
                title="8-K",
                doc_type="filing",
                text="Unrelated boilerplate.",
                content_kind=ContentKind.FULL_DOCUMENT,
                provenance=Provenance.LIVE,
                tier=SourceTier.TIER_1,
                authority=DocumentAuthority.STATUTORY_FILING,
            )

    provider = _Provider()
    _, report = escalate_unresolved_questions(
        [question], filings, provider, company="Generic Biotech Holdings",
        ticker="TESTCO", run_id="r1",
    )
    # The INITIAL fetch batch is capped at 2, whatever else exists (never
    # "three fetches simply because three -- let alone eight -- Tier-1
    # sources exist"). Every candidate's rank is recorded, so the first
    # phase is directly inspectable in the audit trail.
    initial_phase = [e for e in report.fetch_log if e.candidate_rank in (-1, 1, 2)]
    assert len(initial_phase) <= 2
    # Only a bounded expansion beyond that -- never "fetch every collected
    # source" (8 filings here).
    assert len(provider.fetch_calls) < len(filings)


# --- 3. exhibit statement becomes decision-grade statutory evidence ---------
def test_acceptance_3_adverse_exhibit_statement_becomes_decision_grade_statutory_evidence():
    filing_url = "https://www.sec.gov/Archives/x/8-K.htm"
    exhibit_url = "https://www.sec.gov/Archives/x/ex99-1.htm"
    filing = Source(
        source_id=make_source_id(filing_url, "8-K"), url=filing_url, title="8-K",
        tier=SourceTier.TIER_1, accession="0001234567-26-000777",
    )
    exhibit = Source(
        source_id=make_source_id(exhibit_url, "Exhibit 99.1"), url=exhibit_url,
        title="Exhibit 99.1", tier=SourceTier.TIER_1, accession="0001234567-26-000777",
    )
    question = UnresolvedQuestion(
        question="Does the regulator consider the primary endpoint appropriate to establish "
        "effectiveness for the intended indication?",
        why_it_matters="A rejected endpoint invalidates the registrational path.",
        blocking=True,
        category=FactCategory.REGULATORY,
    )

    class _ExhibitProvider:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def available(self):
            return True, "ready"

        def search(self, query, *, agent_id="research"):
            return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=self.path)

        def fetch(self, url, *, reason="", agent_id="research", known=None):
            if url == filing_url:
                return Document(
                    doc_id="filing", url=url, title="8-K", doc_type="filing",
                    accession="0001234567-26-000777",
                    published_date="2026-08-05",
                    text="See Exhibit 99.1 for further detail.",
                    content_kind=ContentKind.FULL_DOCUMENT, provenance=Provenance.LIVE,
                    tier=SourceTier.TIER_1, authority=DocumentAuthority.STATUTORY_FILING,
                )
            if url == exhibit_url:
                return Document(
                    doc_id="exhibit", url=url, title="Exhibit 99.1", doc_type="filing",
                    published_date="2026-08-05",
                    text="The agency stated in writing that it does not consider the primary "
                    "endpoint appropriate to establish effectiveness for the intended "
                    "indication.",
                    content_kind=ContentKind.FULL_DOCUMENT, provenance=Provenance.LIVE,
                    tier=SourceTier.TIER_1, authority=DocumentAuthority.STATUTORY_FILING,
                )
            return None

    facts, _ = escalate_unresolved_questions(
        [question], [filing, exhibit], _ExhibitProvider(), company="Generic Biotech Holdings",
        ticker="TESTCO", run_id="r1",
    )
    assert facts, "the adverse exhibit statement must have produced a fact"
    fact = facts[0]
    assert fact.source_url == exhibit_url
    assert fact.company_claim is True, "statutory filing disclosure, not independent confirmation"
    assert fact.independent_confirmation is False
    assert fact.is_decision_grade, (
        "an adverse statement in a fetched statutory filing/exhibit must become decision-grade "
        "as issuer-disclosed statutory evidence"
    )


# --- 4. unrelated capital-structure question never queries fda.gov ---------
def test_acceptance_4_capital_structure_question_never_queries_fda_gov():
    question = UnresolvedQuestion(
        question="What is the fully diluted share count including all outstanding warrants "
        "and convertible notes?",
        why_it_matters="Materially understates dilution if wrong.",
        blocking=True,
        category=FactCategory.CAPITAL_STRUCTURE,
    )

    class _Recorder:
        name = "fake"
        path = ResearchPath.ANTHROPIC_WEB

        def __init__(self):
            self.domains: list[str] = []

        def available(self):
            return True, "ready"

        def search(self, query, *, agent_id="research"):
            self.domains.extend(query.allowed_domains)
            return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=self.path)

        def fetch(self, url, *, reason="", agent_id="research", known=None):
            return None

    provider = _Recorder()
    escalate_unresolved_questions(
        [question], [], provider, company="Generic Biotech Holdings", ticker="TESTCO", run_id="r1",
    )
    assert provider.domains
    assert "fda.gov" not in provider.domains
    assert "clinicaltrials.gov" not in provider.domains


# --- 5. stage exhaustion in escalation does not poison downstream agents ---
def test_acceptance_5_escalation_stage_exhaustion_does_not_poison_downstream_agents(
    repo, fixture_dir, mock_server
):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")

    budget = LLMBudget(max_total_tokens=1_000_000)
    escalation_cap = budget.stage_cap_tokens("escalation") or 0
    budget.stage_used["escalation"] = escalation_cap
    budget.stage_exhausted.add("escalation")

    llm = LLMClient(
        api_key="sk-ant-test", base_url=mock_server, max_retries=0, timeout=10, budget=budget
    )
    research = AnthropicWebResearchProvider(llm)

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, research=research, llm=llm)
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )

    for agent_id in ("contradiction", "kill_agent", "bear_agent", "bull_agent"):
        records = [r for r in result.agent_records if r.agent_id == agent_id]
        assert records, f"{agent_id} must have run despite escalation's stage exhaustion"
        assert records[0].status != "FAILED", (
            f"{agent_id} must not fail merely because a DIFFERENT stage (escalation) was "
            "exhausted"
        )
    # None of these calls were charged against the ALREADY-exhausted
    # "escalation" stage.
    poisoned = [c for c in budget.calls if c.stage == "escalation" and c.agent_id in (
        "contradiction", "kill_agent", "bear_agent", "bull_agent",
    )]
    assert not poisoned


# --- 6. provider-configured Kill failure never reports NO_PROVIDER ---------
def test_acceptance_6_provider_configured_kill_never_reports_no_provider(
    repo, fixture_dir, mock_server
):
    collector = FixtureCollector(fixture_dir)
    metadata = collector.metadata("DEMOBIO")

    llm = LLMClient(api_key="sk-ant-test", base_url=mock_server, max_retries=0, timeout=10)
    research = AnthropicWebResearchProvider(llm)
    assert research.available() == (True, "ready")

    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, research=research, llm=llm)
    result = pipeline.run(
        "DEMOBIO", metadata["company_name"],
        [collector.collect("DEMOBIO", metadata["company_name"])],
        price=metadata["price"], aliases=metadata.get("aliases", ()),
    )

    kill_payload = result.bus.channels[Channel.KILL].payload
    reasons = {entry["reason"] for entry in kill_payload["query_outcomes"]}
    assert KillSearchFailureReason.NO_PROVIDER.value not in reasons
    kill_records = [r for r in result.agent_records if r.agent_id == "kill_agent"]
    assert kill_records
    assert all("no search provider configured" not in " ".join(r.errors) for r in kill_records)
