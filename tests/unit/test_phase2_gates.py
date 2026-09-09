"""Completeness gate, traceability and resume tests (P6, P7, P8)."""

from __future__ import annotations

from datetime import date

import pytest

from investment_research.collectors.documents import Document, chunk_document
from investment_research.orchestrator.resume import (
    STAGES,
    Checkpoint,
    build_plan,
    freshness_days,
    is_stale,
)
from investment_research.reporting.traceability import (
    build_index,
    check_attribution,
)
from investment_research.research.provider import ResearchQuery, ResearchResult
from investment_research.schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    FactCategory,
    ResearchDomain,
    ResearchPath,
    SearchStatus,
)
from investment_research.scoring.completeness import assess_completeness
from tests.conftest import make_fact

TODAY = date(2026, 9, 6)


def _result(domain, executed=True, docs=1):
    document = Document(doc_id="d", url="https://www.sec.gov/x", title="t")
    return ResearchResult(
        query=ResearchQuery(query="q", domain=domain),
        documents=[document] * docs,
        path=ResearchPath.CORPUS,
        executed=executed,
    )


# --- completeness gate (P6) -------------------------------------------------
def test_all_six_domains_are_required():
    assert len(REQUIRED_RESEARCH_DOMAINS) == 6
    names = {str(d) for d in REQUIRED_RESEARCH_DOMAINS}
    assert names == {
        "REGULATORY",
        "CAPITAL_STRUCTURE",
        "SCIENCE_TECHNOLOGY",
        "COMPETITION",
        "CATALYST",
        "CONTRADICTION",
    }


def test_a_missing_domain_blocks_the_verdict():
    result = assess_completeness(search_results=[_result(ResearchDomain.REGULATORY)])
    assert result.blocked
    assert ResearchDomain.CATALYST in result.missing
    assert "were not searched" in result.reason()


def test_all_domains_searched_permits_a_verdict():
    result = assess_completeness(
        search_results=[_result(domain) for domain in REQUIRED_RESEARCH_DOMAINS]
    )
    assert not result.blocked
    assert result.searched_count == 6
    assert result.reason() == ""


def test_an_attempted_but_unexecuted_query_is_failed_not_searched():
    result = assess_completeness(
        search_results=[_result(ResearchDomain.REGULATORY, executed=False)]
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.FAILED
    assert not entry.searched


def test_agent_analysis_without_a_query_counts_as_partial():
    """Analysing the collected corpus is real examination; it is not a full
    search, and (per the v4 completeness-gate fix) PARTIAL must NOT satisfy
    the final required-domain gate -- it is visible in reports, but a
    verdict may not be issued on PARTIAL coverage alone."""
    result = assess_completeness(
        search_results=[],
        facts_by_domain={ResearchDomain.REGULATORY: 4},
        agents_run={"regulatory"},
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.PARTIAL
    assert not entry.searched


def test_an_agent_that_ran_but_found_nothing_is_unsearched():
    result = assess_completeness(
        search_results=[],
        facts_by_domain={ResearchDomain.REGULATORY: 0},
        agents_run={"regulatory"},
    )
    assert result.coverage[ResearchDomain.REGULATORY].status is SearchStatus.UNSEARCHED


def test_summary_rows_cover_every_domain():
    rows = assess_completeness(search_results=[]).summary_rows()
    assert len(rows) == 6


# --- traceability (P7) ------------------------------------------------------
@pytest.fixture
def index():
    fact = make_fact(
        "The FDA advised the endpoint is not sufficient",
        category=FactCategory.REGULATORY,
        url="https://www.globenewswire.com/x",
    )
    document = Document(doc_id=fact.source_id, url=fact.source_url, title="PR", text="body text")
    return build_index([fact], [], chunk_document(document, target_tokens=100)), fact


def test_every_link_walks_the_full_chain(index):
    idx, fact = index
    link = idx.resolve(fact.fact_id)
    assert link.fact_id == fact.fact_id
    assert link.source_id == fact.source_id
    assert link.url.startswith("https://")
    assert link.publication_date and link.event_date
    assert link.source_tier and link.verified_status


def test_a_chunk_reference_resolves_to_its_fact(index):
    idx, fact = index
    chunk_id = next(iter(idx.chunk_to_fact))
    assert idx.resolve(chunk_id).fact_id == fact.fact_id


def test_an_unknown_reference_does_not_resolve(index):
    idx, _ = index
    assert idx.resolve("fact_" + "0" * 20) is None


@pytest.mark.parametrize(
    "sentence,kind",
    [
        ("The FDA advised that the endpoint is not sufficient.", "regulator"),
        ("Analysts expect approval next year.", "analyst"),
        ("The study showed a meaningful benefit.", "study"),
        ("The company disclosed a going concern doubt.", "company"),
        ("The 10-Q discloses substantial doubt.", "filing"),
    ],
)
def test_attributed_sentences_without_citations_are_violations(index, sentence, kind):
    idx, _ = index
    report = check_attribution(sentence, idx)
    assert len(report.violations) == 1
    assert report.violations[0].kind == kind


def test_an_attributed_sentence_with_a_citation_passes(index):
    idx, fact = index
    report = check_attribution(
        f"The FDA advised the endpoint is not sufficient [{fact.fact_id}].", idx
    )
    assert report.clean


def test_a_url_counts_as_a_citation(index):
    idx, _ = index
    report = check_attribution(
        "The FDA advised the endpoint is not sufficient, see https://www.sec.gov/a for detail.",
        idx,
    )
    assert not report.violations


def test_unattributed_prose_is_not_a_violation(index):
    idx, _ = index
    report = check_attribution("Cash fell during the quarter and the burn rate rose.", idx)
    assert report.clean
    assert report.attributed_sentences == 0


# --- resume (P8) ------------------------------------------------------------
def test_stage_order_is_collection_first_report_last():
    assert STAGES[0] == "collect"
    assert STAGES[-1] == "report"
    assert "judge" in STAGES


def test_regulatory_evidence_goes_stale_faster_than_clinical():
    assert freshness_days(FactCategory.REGULATORY) < freshness_days(FactCategory.CLINICAL)
    assert freshness_days(FactCategory.MICROSTRUCTURE) < freshness_days(FactCategory.REGULATORY)


def test_a_recent_fact_is_fresh_and_an_old_one_is_not():
    fresh = make_fact("a", category=FactCategory.REGULATORY, event_date="2026-09-01")
    old = make_fact("b", category=FactCategory.REGULATORY, event_date="2026-01-01")
    assert not is_stale(fresh, TODAY)
    assert is_stale(old, TODAY)


def test_an_undated_fact_is_treated_as_stale():
    """A fact that cannot be shown to be current is not assumed to be."""
    from investment_research.schemas.enums import UNKNOWN

    undated = make_fact("c", event_date=UNKNOWN, publication_date=UNKNOWN)
    assert is_stale(undated, TODAY)


def test_resume_restarts_after_the_last_completed_stage():
    checkpoints = [Checkpoint("r1", stage, i) for i, stage in enumerate(STAGES[:5])]
    fact = make_fact("a", category=FactCategory.REGULATORY, event_date="2026-09-01")
    plan = build_plan("r1", checkpoints, [fact], today=TODAY)
    assert plan.resume_from_stage == STAGES[5]
    assert not plan.should_run(STAGES[0])
    assert plan.should_run(STAGES[5])
    assert plan.restored_facts == [fact]


def test_stale_evidence_forces_a_full_recollection():
    checkpoints = [Checkpoint("r1", stage, i) for i, stage in enumerate(STAGES[:8])]
    stale = make_fact("a", category=FactCategory.REGULATORY, event_date="2026-01-01")
    plan = build_plan("r1", checkpoints, [stale], today=TODAY)
    assert plan.resume_from_stage == "collect"
    assert plan.is_fresh_start
    assert stale.fact_id in plan.stale_fact_ids
    assert "stale" in plan.reason


def test_no_checkpoints_means_a_fresh_start():
    plan = build_plan("r1", [], [], today=TODAY)
    assert plan.is_fresh_start
    assert plan.should_run("collect")
