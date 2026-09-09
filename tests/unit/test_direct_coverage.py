"""Direct-collector coverage semantics, corrected (requirement A/B/C/D/F).

The regression this replaces: a successful-but-EMPTY Drugs@FDA search (or a
successful ClinicalTrials.gov sponsor search, or SEC EDGAR submissions
metadata) was read as the ENTIRE ResearchDomain being
``SearchStatus.DIRECTLY_RESEARCHED`` -- "complete, no web search needed" --
recreating exactly the class of failure this system exists to prevent
(REGULATORY "done" without ever asking whether FDA accepted the endpoint).

Corrected model: a collector's successful execution is auditable coverage
(PARTIAL) but never, by itself, completeness. Only a domain-specific
sufficiency checklist -- passable for CAPITAL_STRUCTURE, structurally
never passable for REGULATORY or SCIENCE_TECHNOLOGY with today's
collectors -- may promote a domain to DIRECTLY_RESEARCHED. And PARTIAL
itself must not satisfy the final required-domain gate.
"""

from __future__ import annotations

from investment_research.collectors.base import CollectionResult
from investment_research.research.adversarial import BEAR_TEMPLATES, build_plan
from investment_research.research.provider import ResearchQuery, ResearchResult
from investment_research.schemas.enums import (
    FactCategory,
    FetchOutcome,
    ResearchDomain,
    ResearchPath,
    SearchStatus,
    SourceTier,
)
from investment_research.schemas.fact import RawFact, Source, make_source_id
from investment_research.scoring.completeness import (
    CAPITAL_STRUCTURE_REQUIRED_FIELDS,
    assess_completeness,
    direct_collector_coverage,
)


def _ok_collection(collector: str, raw_facts=()) -> CollectionResult:
    return CollectionResult(collector=collector, outcome=FetchOutcome.OK, raw_facts=list(raw_facts))


def _failed_collection(collector: str) -> CollectionResult:
    return CollectionResult(collector=collector, outcome=FetchOutcome.ERROR, errors=["boom"])


def _capital_raw_fact(unit: str) -> RawFact:
    url = f"https://www.sec.gov/x/{unit}"
    source = Source(source_id=make_source_id(url, unit), url=url, title=unit, tier=SourceTier.TIER_1)
    return RawFact(
        ticker="TESTCO",
        category=FactCategory.CAPITAL_STRUCTURE,
        claim=f"{unit} extracted from filing body",
        source=source,
        value="1",
        unit=unit,
        collector="sec_edgar",
    )


# --- direct_collector_coverage(): touched (auditable) vs sufficient (complete)
def test_a_successful_collector_only_touches_its_domain_never_sufficient_alone():
    sufficient, touched = direct_collector_coverage(
        [_ok_collection("sec_edgar"), _ok_collection("clinicaltrials"), _ok_collection("fda")]
    )
    assert ResearchDomain.CAPITAL_STRUCTURE in touched
    assert ResearchDomain.SCIENCE_TECHNOLOGY in touched
    assert ResearchDomain.REGULATORY in touched
    # None of these are complete merely because the collector ran.
    assert ResearchDomain.CAPITAL_STRUCTURE not in sufficient
    assert ResearchDomain.SCIENCE_TECHNOLOGY not in sufficient
    assert ResearchDomain.REGULATORY not in sufficient


def test_direct_collector_coverage_ignores_a_failed_collector():
    sufficient, touched = direct_collector_coverage([_failed_collection("sec_edgar")])
    assert not sufficient
    assert not touched


def test_sec_edgar_touches_both_capital_structure_and_regulatory():
    _sufficient, touched = direct_collector_coverage([_ok_collection("sec_edgar")])
    assert ResearchDomain.CAPITAL_STRUCTURE in touched
    assert ResearchDomain.REGULATORY in touched


# --- B: openFDA can never by itself complete REGULATORY ---------------------
def test_openfda_zero_result_is_partial_not_sufficient():
    """A clean, successful, zero-result Drugs@FDA search (out.outcome=OK,
    out.zero_results=True, no raw_facts) still only TOUCHES REGULATORY."""
    zero_result = CollectionResult(collector="fda", outcome=FetchOutcome.OK, zero_results=True)
    sufficient, touched = direct_collector_coverage([zero_result])
    assert ResearchDomain.REGULATORY in touched
    assert ResearchDomain.REGULATORY not in sufficient


def test_openfda_with_matching_applications_is_still_partial():
    """A Drugs@FDA search that DOES find applications is still only
    PARTIAL -- it establishes marketing-status facts, never meeting
    outcome, endpoint acceptability, an SPA, CMC resolution or CRL history."""
    with_apps = _ok_collection("fda", raw_facts=[_capital_raw_fact("marketing_status")])
    sufficient, touched = direct_collector_coverage([with_apps])
    assert ResearchDomain.REGULATORY in touched
    assert ResearchDomain.REGULATORY not in sufficient


def test_regulatory_can_never_reach_sufficient_via_any_collector_combination():
    """Structural, not incidental: no combination of today's collectors may
    ever mark REGULATORY sufficient, however much data they return."""
    sufficient, _touched = direct_collector_coverage(
        [
            _ok_collection("fda", raw_facts=[_capital_raw_fact("marketing_status")] * 20),
            _ok_collection("sec_edgar"),
            _ok_collection("clinicaltrials"),
        ]
    )
    assert ResearchDomain.REGULATORY not in sufficient


# --- C: ClinicalTrials.gov is registry-scope only for SCIENCE_TECHNOLOGY ----
def test_clinicaltrials_successful_search_is_partial_science_coverage():
    sufficient, touched = direct_collector_coverage([_ok_collection("clinicaltrials")])
    assert ResearchDomain.SCIENCE_TECHNOLOGY in touched
    assert ResearchDomain.SCIENCE_TECHNOLOGY not in sufficient


# --- D: SEC submission metadata is not complete capital-structure research -
def test_sec_submissions_metadata_alone_is_partial_capital_coverage():
    """SecEdgarCollector.collect() only ever records filing metadata
    (unit='form_type') -- never the required capital fields themselves."""
    metadata_only = _ok_collection(
        "sec_edgar",
        raw_facts=[
            RawFact(
                ticker="TESTCO",
                category=FactCategory.CAPITAL_STRUCTURE,
                claim="S-1 filed 2026-01-01",
                source=Source(
                    source_id=make_source_id("https://www.sec.gov/x/s1", "S-1"),
                    url="https://www.sec.gov/x/s1",
                    title="S-1",
                    tier=SourceTier.TIER_1,
                ),
                value="S-1",
                unit="form_type",
                collector="sec_edgar",
            )
        ],
    )
    sufficient, touched = direct_collector_coverage([metadata_only])
    assert ResearchDomain.CAPITAL_STRUCTURE in touched
    assert ResearchDomain.CAPITAL_STRUCTURE not in sufficient


def test_capital_structure_reaches_sufficient_only_with_the_full_field_checklist():
    """The mechanism is real, not permanently disabled: genuinely extracting
    every required field (from filing bodies/XBRL, per requirement D)
    DOES promote CAPITAL_STRUCTURE to sufficient."""
    full_extraction = _ok_collection(
        "sec_edgar", raw_facts=[_capital_raw_fact(unit) for unit in CAPITAL_STRUCTURE_REQUIRED_FIELDS]
    )
    sufficient, touched = direct_collector_coverage([full_extraction])
    assert ResearchDomain.CAPITAL_STRUCTURE in touched
    assert ResearchDomain.CAPITAL_STRUCTURE in sufficient


def test_capital_structure_partial_field_extraction_is_not_yet_sufficient():
    partial_extraction = _ok_collection(
        "sec_edgar", raw_facts=[_capital_raw_fact("basic_shares_outstanding")]
    )
    sufficient, _touched = direct_collector_coverage([partial_extraction])
    assert ResearchDomain.CAPITAL_STRUCTURE not in sufficient


# --- assess_completeness(): PARTIAL vs DIRECTLY_RESEARCHED vs UNSEARCHED ----
def test_assess_completeness_marks_partial_from_a_touching_collector_alone():
    result = assess_completeness(
        search_results=[], facts_by_domain={}, agents_run=set(),
        collection_results=[_ok_collection("fda")],
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.PARTIAL
    assert entry.searched is False, "PARTIAL must not satisfy the final required-domain gate"


def test_assess_completeness_marks_directly_researched_only_when_sufficient():
    full_extraction = _ok_collection(
        "sec_edgar", raw_facts=[_capital_raw_fact(unit) for unit in CAPITAL_STRUCTURE_REQUIRED_FIELDS]
    )
    result = assess_completeness(
        search_results=[], facts_by_domain={}, agents_run=set(),
        collection_results=[full_extraction],
    )
    entry = result.coverage[ResearchDomain.CAPITAL_STRUCTURE]
    assert entry.status is SearchStatus.DIRECTLY_RESEARCHED
    assert entry.searched is True
    assert ResearchPath.DIRECT_API in entry.paths


def test_assess_completeness_never_calls_it_searched_from_facts_alone():
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
        search_results=[], facts_by_domain={}, agents_run=set(),
        collection_results=[_ok_collection("sec_edgar"), _ok_collection("fda")],
    )
    # COMPETITION has no structured collector coverage in this system at all.
    entry = result.coverage[ResearchDomain.COMPETITION]
    assert entry.status is SearchStatus.UNSEARCHED


def test_assess_completeness_web_search_still_wins_over_partial_when_both_present():
    from investment_research.collectors.documents import Document

    query = ResearchQuery(query="q", domain=ResearchDomain.REGULATORY)
    search_results = [
        ResearchResult(
            query=query, executed=True,
            documents=[Document(doc_id="d1", url="https://www.fda.gov/x", title="t")],
        )
    ]
    result = assess_completeness(
        search_results=search_results, facts_by_domain={}, agents_run=set(),
        collection_results=[_ok_collection("fda")],
    )
    entry = result.coverage[ResearchDomain.REGULATORY]
    assert entry.status is SearchStatus.SEARCHED


# --- E/G: build_plan() skips only genuinely redundant QUERY INTENTS --------
def test_build_plan_never_skips_a_whole_domain_from_mere_collector_touch():
    """The false-completeness bug this replaces: a collector merely
    touching REGULATORY/CAPITAL_STRUCTURE/SCIENCE_TECHNOLOGY must not drop
    every adversarial query in that domain -- none of today's mandatory
    templates are genuinely redundant with any real collector."""
    plan = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[
            _ok_collection("sec_edgar"), _ok_collection("clinicaltrials"), _ok_collection("fda"),
        ],
    )
    remaining_domains = {q.domain for q in plan.bear}
    all_domains = {domain for _template, domain in BEAR_TEMPLATES}
    assert remaining_domains == all_domains, (
        "no domain's mandatory queries may be dropped merely because a collector touched it"
    )
    assert plan.skipped_due_to_direct_coverage == []


def test_build_plan_with_no_coverage_keeps_every_bear_domain():
    plan = build_plan("TESTCO", "Generic Biotech Holdings")
    domains = {domain for _template, domain in BEAR_TEMPLATES}
    plan_domains = {q.domain for q in plan.bear}
    assert domains <= plan_domains
    assert plan.skipped_due_to_direct_coverage == []


def test_build_plan_can_skip_a_genuinely_redundant_intent_when_one_exists():
    """The mechanism is real and generic (requirement E/G/H6/H9): given an
    explicit override naming a genuinely redundant intent, ONLY that exact
    query is skipped -- every other query, including other queries in the
    SAME domain, is untouched (H7)."""
    redundant_intents = {"clinicaltrials": frozenset({"science.failed_trial"})}
    plan = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[_ok_collection("clinicaltrials")],
        redundant_intents=redundant_intents,
    )
    bear_texts = {q.query for q in plan.bear}
    assert "Generic Biotech Holdings failed trial" not in bear_texts
    assert plan.skipped_due_to_direct_coverage == ["Generic Biotech Holdings failed trial"]
    # A sibling SCIENCE_TECHNOLOGY query (safety concern) is untouched.
    assert "Generic Biotech Holdings safety concern" in bear_texts


def test_build_plan_material_unresolved_question_overrides_redundant_intent():
    """H8: even with a (hypothetical) genuinely-redundant mapping, a
    blocking unresolved question in the SAME domain forces the query to
    stay scheduled."""
    from investment_research.schemas.fact import UnresolvedQuestion

    redundant_intents = {"clinicaltrials": frozenset({"science.failed_trial"})}
    question = UnresolvedQuestion(
        question="Has the trial's primary endpoint been reached?",
        why_it_matters="Determines whether the programme remains viable.",
        blocking=True,
        category=FactCategory.SCIENCE,
    )
    plan = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[_ok_collection("clinicaltrials")],
        unresolved_questions=[question],
        redundant_intents=redundant_intents,
    )
    bear_texts = {q.query for q in plan.bear}
    assert "Generic Biotech Holdings failed trial" in bear_texts
    assert plan.skipped_due_to_direct_coverage == []


def test_h8_regulatory_endpoint_question_stays_scheduled_after_all_collectors_run():
    """H8 acceptance: a regulatory endpoint-acceptability query intent
    remains scheduled even after openFDA + ClinicalTrials + SEC all execute
    successfully -- no collector-redundancy mapping in this system ever
    covers it (requirement B)."""
    plan = build_plan(
        "TESTCO", "Generic Biotech Holdings",
        collection_results=[
            _ok_collection("fda"), _ok_collection("clinicaltrials"), _ok_collection("sec_edgar"),
        ],
    )
    bear_texts = {q.query for q in plan.bear}
    assert "Generic Biotech Holdings endpoint concern" in bear_texts
