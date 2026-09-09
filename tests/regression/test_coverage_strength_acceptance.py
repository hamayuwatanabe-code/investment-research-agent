"""Synthetic acceptance regressions for the coverage-strength fix
(requirement H, this turn).

The regression fixed: a successful-but-EMPTY Drugs@FDA search (or a
successful ClinicalTrials.gov sponsor search, or SEC EDGAR submissions
metadata) was read as the ENTIRE ResearchDomain being complete
(``SearchStatus.DIRECTLY_RESEARCHED``), silently dropping every mandatory
adversarial query for that domain -- recreating exactly the class of
failure this system exists to prevent (REGULATORY marked "done" without
ever asking whether FDA accepted the endpoint).

Numbered 1-9 to match the acceptance list verbatim. No real ticker,
company, endpoint, meeting type, or indication is named anywhere here.
"""

from __future__ import annotations

from investment_research.collectors.base import CollectionResult
from investment_research.research.adversarial import build_plan
from investment_research.schemas.enums import (
    FactCategory,
    FetchOutcome,
    ResearchDomain,
    SearchStatus,
    SourceTier,
)
from investment_research.schemas.fact import RawFact, Source, UnresolvedQuestion, make_source_id
from investment_research.scoring.completeness import assess_completeness


def _ok(collector: str, raw_facts=()) -> CollectionResult:
    return CollectionResult(collector=collector, outcome=FetchOutcome.OK, raw_facts=list(raw_facts))


def _raw_fact(unit: str, *, category=FactCategory.CAPITAL_STRUCTURE, collector="sec_edgar") -> RawFact:
    url = f"https://www.sec.gov/x/{unit}"
    return RawFact(
        ticker="TESTCO",
        category=category,
        claim=f"{unit} extracted",
        source=Source(source_id=make_source_id(url, unit), url=url, title=unit, tier=SourceTier.TIER_1),
        value="1",
        unit=unit,
        collector=collector,
    )


# 1. openFDA zero-result cannot satisfy REGULATORY completeness.
def test_1_openfda_zero_result_cannot_satisfy_regulatory_completeness():
    zero_result = CollectionResult(collector="fda", outcome=FetchOutcome.OK, zero_results=True)
    result = assess_completeness(
        search_results=[], facts_by_domain={}, agents_run=set(), collection_results=[zero_result]
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is not SearchStatus.DIRECTLY_RESEARCHED
    assert entry.searched is False


# 2. openFDA positive application result cannot satisfy endpoint-acceptability coverage.
def test_2_openfda_positive_result_cannot_satisfy_endpoint_acceptability_coverage():
    with_applications = _ok(
        "fda", raw_facts=[_raw_fact("marketing_status", category=FactCategory.REGULATORY, collector="fda")]
    )
    result = assess_completeness(
        search_results=[], facts_by_domain={}, agents_run=set(),
        collection_results=[with_applications],
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.PARTIAL
    assert entry.searched is False
    # And the mandatory endpoint-acceptability query stays scheduled.
    plan = build_plan("TESTCO", "Generic Biotech Holdings", collection_results=[with_applications])
    assert "Generic Biotech Holdings endpoint concern" in {q.query for q in plan.bear}


# 3. ClinicalTrials successful sponsor search is PARTIAL science coverage.
def test_3_clinicaltrials_successful_search_is_partial_science_coverage():
    result = assess_completeness(
        search_results=[], facts_by_domain={}, agents_run=set(),
        collection_results=[_ok("clinicaltrials")],
    )
    entry = result.coverage[ResearchDomain.SCIENCE_TECHNOLOGY]
    assert entry.status is SearchStatus.PARTIAL
    assert entry.searched is False


# 4. SEC submissions metadata is PARTIAL capital coverage.
def test_4_sec_submissions_metadata_is_partial_capital_coverage():
    metadata_only = _ok("sec_edgar", raw_facts=[_raw_fact("form_type")])
    result = assess_completeness(
        search_results=[], facts_by_domain={}, agents_run=set(),
        collection_results=[metadata_only],
    )
    entry = result.coverage[ResearchDomain.CAPITAL_STRUCTURE]
    assert entry.status is SearchStatus.PARTIAL
    assert entry.searched is False


# 5. PARTIAL never satisfies the final required-domain gate.
def test_5_partial_never_satisfies_the_final_required_domain_gate():
    result = assess_completeness(
        search_results=[],
        facts_by_domain=dict.fromkeys(ResearchDomain, 1),
        agents_run={"regulatory", "capital_structure", "science", "competitive", "catalyst", "contradiction"},
    )
    for domain in ResearchDomain:
        entry = result.coverage[domain]
        assert entry.status is SearchStatus.PARTIAL
        assert entry.searched is False
    assert result.blocked, "an all-PARTIAL run must still block the final verdict"


# 6. exact already-covered query intent may be skipped.
def test_6_exact_already_covered_query_intent_may_be_skipped():
    redundant_intents = {"clinicaltrials": frozenset({"science.failed_trial"})}
    plan = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[_ok("clinicaltrials")],
        redundant_intents=redundant_intents,
    )
    bear_texts = {q.query for q in plan.bear}
    assert "Generic Biotech Holdings failed trial" not in bear_texts
    assert plan.skipped_due_to_direct_coverage == ["Generic Biotech Holdings failed trial"]


# 7. another unresolved intent in the SAME domain still gets searched.
def test_7_another_unresolved_intent_in_the_same_domain_still_gets_searched():
    redundant_intents = {"clinicaltrials": frozenset({"science.failed_trial"})}
    plan = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[_ok("clinicaltrials")],
        redundant_intents=redundant_intents,
    )
    bear_texts = {q.query for q in plan.bear}
    # "failed trial" (skipped) and "safety concern" are both SCIENCE_TECHNOLOGY.
    assert "Generic Biotech Holdings safety concern" in bear_texts
    science_domain_texts = {
        q.query for q in plan.bear if q.domain is ResearchDomain.SCIENCE_TECHNOLOGY
    }
    assert len(science_domain_texts) >= 1


# 8. a regulatory endpoint-acceptability unresolved question remains scheduled
#    even after openFDA + ClinicalTrials + SEC collectors all execute successfully.
def test_8_regulatory_endpoint_question_remains_scheduled_after_all_collectors_run():
    question = UnresolvedQuestion(
        question="Does the regulator consider the primary endpoint appropriate to establish "
        "effectiveness for the intended indication?",
        why_it_matters="A rejected endpoint invalidates the registrational path.",
        blocking=True,
        category=FactCategory.REGULATORY,
    )
    plan = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[_ok("fda"), _ok("clinicaltrials"), _ok("sec_edgar")],
        unresolved_questions=[question],
    )
    bear_texts = {q.query for q in plan.bear}
    assert "Generic Biotech Holdings endpoint concern" in bear_texts
    assert "TESTCO FDA concern" in bear_texts
    assert "Generic Biotech Holdings regulatory risk" in bear_texts


# 9. cost-saving behavior still avoids truly redundant searches.
def test_9_cost_saving_behavior_still_avoids_truly_redundant_searches():
    redundant_intents = {"clinicaltrials": frozenset({"science.failed_trial"})}
    plan_without_collector = build_plan(
        "TESTCO", "Generic Biotech Holdings", redundant_intents=redundant_intents
    )
    plan_with_collector = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[_ok("clinicaltrials")],
        redundant_intents=redundant_intents,
    )
    # Exactly the one genuinely-redundant query is saved -- no more, no less.
    assert len(plan_with_collector.bear) == len(plan_without_collector.bear) - 1
    assert plan_with_collector.skipped_due_to_direct_coverage == [
        "Generic Biotech Holdings failed trial"
    ]
    without_texts = {q.query for q in plan_without_collector.bear}
    with_texts = {q.query for q in plan_with_collector.bear}
    assert without_texts - with_texts == {"Generic Biotech Holdings failed trial"}
