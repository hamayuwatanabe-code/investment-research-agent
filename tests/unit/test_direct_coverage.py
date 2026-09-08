"""Direct-collector coverage semantics and discovery-call reduction
(requirement F).

The defect: ~22k actual tokens per Anthropic web-search call even at low
research effort meant six independent required-domain web searches alone
could exhaust a 60k discovery quota before COMPETITION/CONTRADICTION/CATALYST
-- the domains no structured collector can ever cover -- got a turn. And the
completeness gate could not tell "a structured collector genuinely,
auditably covered this domain" apart from "facts of the right category
happen to exist" -- both looked like PARTIAL.
"""

from __future__ import annotations

from investment_research.collectors.base import CollectionResult
from investment_research.research.adversarial import BEAR_TEMPLATES, build_plan
from investment_research.research.provider import ResearchQuery, ResearchResult
from investment_research.schemas.enums import (
    FetchOutcome,
    ResearchDomain,
    ResearchPath,
    SearchStatus,
)
from investment_research.scoring.completeness import assess_completeness, direct_collector_coverage


def _ok_collection(collector: str) -> CollectionResult:
    return CollectionResult(collector=collector, outcome=FetchOutcome.OK)


def _failed_collection(collector: str) -> CollectionResult:
    return CollectionResult(collector=collector, outcome=FetchOutcome.ERROR, errors=["boom"])


# --- direct_collector_coverage() ---------------------------------------------
def test_direct_collector_coverage_maps_strong_domains():
    strong, _weak = direct_collector_coverage(
        [_ok_collection("sec_edgar"), _ok_collection("clinicaltrials"), _ok_collection("fda")]
    )
    assert ResearchDomain.CAPITAL_STRUCTURE in strong
    assert ResearchDomain.SCIENCE_TECHNOLOGY in strong
    assert ResearchDomain.REGULATORY in strong


def test_direct_collector_coverage_ignores_a_failed_collector():
    strong, weak = direct_collector_coverage([_failed_collection("sec_edgar")])
    assert not strong
    assert not weak


def test_direct_collector_coverage_sec_edgar_is_only_weak_for_regulatory():
    strong, weak = direct_collector_coverage([_ok_collection("sec_edgar")])
    assert ResearchDomain.CAPITAL_STRUCTURE in strong
    assert ResearchDomain.REGULATORY not in strong
    assert ResearchDomain.REGULATORY in weak


# --- assess_completeness(): DIRECTLY_RESEARCHED vs PARTIAL vs UNSEARCHED ----
def test_assess_completeness_marks_directly_researched_from_collector_alone():
    """A structured collector executed but NO facts and NO agent evidence
    exist yet -- must still be DIRECTLY_RESEARCHED, never UNSEARCHED, because
    there IS an actual collector execution record."""
    result = assess_completeness(
        search_results=[],
        facts_by_domain={},
        agents_run=set(),
        collection_results=[_ok_collection("fda")],
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.DIRECTLY_RESEARCHED
    assert entry.searched is True
    assert ResearchPath.DIRECT_API in entry.paths


def test_assess_completeness_never_calls_it_searched_from_facts_alone():
    """Facts existing for a domain, with NO query and NO collector
    execution record, must never read DIRECTLY_RESEARCHED or SEARCHED --
    only PARTIAL (agent ran) or UNSEARCHED can result from facts alone."""
    result = assess_completeness(
        search_results=[],
        facts_by_domain={ResearchDomain.REGULATORY: 3},
        agents_run=set(),  # the regulatory agent did NOT run
        collection_results=[],
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.UNSEARCHED
    assert entry.status is not SearchStatus.DIRECTLY_RESEARCHED


def test_assess_completeness_domain_with_no_collector_and_no_search_is_unsearched():
    result = assess_completeness(
        search_results=[],
        facts_by_domain={},
        agents_run=set(),
        collection_results=[_ok_collection("sec_edgar"), _ok_collection("fda")],
    )
    # COMPETITION has no structured collector coverage in this system at all.
    entry = result.coverage[ResearchDomain.COMPETITION]
    assert entry.status is SearchStatus.UNSEARCHED


def test_assess_completeness_web_search_still_wins_over_direct_when_both_present():
    query = ResearchQuery(query="q", domain=ResearchDomain.REGULATORY)
    search_results = [ResearchResult(query=query, executed=True, documents=[__import__(
        "investment_research.collectors.documents", fromlist=["Document"]
    ).Document(doc_id="d1", url="https://www.fda.gov/x", title="t")])]
    result = assess_completeness(
        search_results=search_results,
        facts_by_domain={},
        agents_run=set(),
        collection_results=[_ok_collection("fda")],
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.SEARCHED


# --- build_plan(): required-domain discovery skips collector-covered domains
def test_build_plan_skips_templates_for_directly_covered_domains():
    covered = frozenset({ResearchDomain.REGULATORY, ResearchDomain.CAPITAL_STRUCTURE})
    plan = build_plan("TESTCO", "Generic Biotech Holdings", already_covered_domains=covered)
    remaining_domains = {q.domain for q in plan.bear}
    assert ResearchDomain.REGULATORY not in remaining_domains
    assert ResearchDomain.CAPITAL_STRUCTURE not in remaining_domains
    assert plan.skipped_due_to_direct_coverage, "skipped queries must be recorded, not silent"


def test_build_plan_with_no_coverage_keeps_every_bear_domain():
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    domains = {domain for _template, domain in BEAR_TEMPLATES}
    plan_domains = {q.domain for q in plan.bear}
    assert domains <= plan_domains
    assert plan.skipped_due_to_direct_coverage == []


def test_build_plan_with_full_collector_coverage_reserves_search_for_uncovered_domains():
    """Requirement F acceptance: collector coverage of REGULATORY/
    CAPITAL_STRUCTURE/SCIENCE_TECHNOLOGY leaves only COMPETITION/CATALYST/
    CONTRADICTION queries in the bear plan -- not all six domains' worth of
    expensive web-search calls."""
    strong, _weak = direct_collector_coverage(
        [_ok_collection("sec_edgar"), _ok_collection("clinicaltrials"), _ok_collection("fda")]
    )
    plan = build_plan("TESTCO", "Generic Biotech Holdings", already_covered_domains=strong)
    remaining_domains = {q.domain for q in plan.bear}
    assert remaining_domains <= {
        ResearchDomain.COMPETITION,
        ResearchDomain.CATALYST,
        ResearchDomain.CONTRADICTION,
    }
    assert remaining_domains, "at least the uncovered domains must still be searched"
