"""The 49-row legacy acquisition catalog and its exact-match dedup.

Locks the counts: 49 raw rows, 31 groups after exact
(equivalence_key, domain, subject_scope) dedup, 18 collapsed -- unchanged
across the Phase 2 COMPANY/PROGRAM correction to equivalence_key (see
legacy_catalog.py's module docstring for why the count did not move). Also
locks that KILL_RULES and REQUIRED_RESEARCH_DOMAINS -- both explicitly
untouched by this change -- still report their pre-existing counts.
"""

from __future__ import annotations

from investment_research.research.checks import AcquisitionOrigin, LegacyResearchNeed, SubjectScope
from investment_research.research.legacy_catalog import (
    LEGACY_CATALOG,
    collapsed_count,
    equivalence_key,
    group_legacy_needs,
    subject_scope_for_template,
)
from investment_research.schemas.enums import REQUIRED_RESEARCH_DOMAINS, ResearchDomain
from investment_research.scoring.kill_gate import KILL_RULES


def test_catalog_has_exactly_49_rows():
    assert len(LEGACY_CATALOG) == 49


def test_every_row_has_a_unique_individually_addressable_id():
    ids = [need.legacy_need_id for need in LEGACY_CATALOG]
    assert len(ids) == len(set(ids)) == 49
    # Spot-check a few from each origin family are directly retrievable.
    by_id = {need.legacy_need_id: need for need in LEGACY_CATALOG}
    assert by_id["bear_0"].original_template == "{t} FDA concern"
    assert by_id["bull_5"].original_template == "{c} upcoming catalyst readout"
    assert by_id["kill_17"].original_template == "{t} offering priced"
    assert by_id["kill_25"].original_template == "{t} delisting"


def test_origin_breakdown_is_17_bear_6_bull_26_kill():
    counts: dict[AcquisitionOrigin, int] = {}
    for need in LEGACY_CATALOG:
        counts[need.origin] = counts.get(need.origin, 0) + 1
    assert counts[AcquisitionOrigin.BEAR] == 17
    assert counts[AcquisitionOrigin.BULL] == 6
    assert counts[AcquisitionOrigin.KILL] == 26


def test_grouping_produces_exactly_31_groups_and_18_collapsed():
    """Unchanged from the original (pre-COMPANY/PROGRAM-fix) Phase 1 count.

    Every Phase 1 merge already paired a {c}-template with a {t}-template
    (both COMPANY-scoped under the corrected key too); none of the four
    {d}-templates ever shared a suffix with another {d}-template, so no
    PROGRAM-scoped merge existed to lose.
    """
    groups = group_legacy_needs()
    assert len(groups) == 31
    assert collapsed_count() == 18
    assert sum(len(ids) for ids in groups.values()) == 49


def test_legacy_research_need_carries_no_investment_judgment_status_field():
    need = LEGACY_CATALOG[0]
    forbidden = {"status", "verdict", "rating", "evidenced", "contradicted", "score"}
    field_names = set(need.__dataclass_fields__)
    assert not (field_names & forbidden)


# --- exact-match-only dedup: no fuzzy / token-overlap merging --------------
def test_similar_but_not_identical_keys_never_merge():
    """bear_2 ('{d} endpoint concern') and kill_3 ('{t} endpoint') both fall
    under ResearchDomain.REGULATORY but are NOT the same equivalence_key (and
    are in different subject scopes) -- they must stay in separate groups
    despite the obvious topical overlap a fuzzy/token-overlap comparison
    would collapse."""
    by_id = {need.legacy_need_id: need for need in LEGACY_CATALOG}
    bear_2 = by_id["bear_2"]
    kill_3 = by_id["kill_3"]
    assert bear_2.domain == kill_3.domain == ResearchDomain.REGULATORY
    assert bear_2.equivalence_key != kill_3.equivalence_key
    assert bear_2.subject_scope is SubjectScope.PROGRAM
    assert kill_3.subject_scope is SubjectScope.COMPANY

    groups = group_legacy_needs()
    group_of = {need_id: key for key, ids in groups.items() for need_id in ids}
    assert group_of["bear_2"] != group_of["kill_3"]


def test_same_text_different_domain_never_merges():
    """Grouping keys on domain too, not equivalence_key alone: two needs with
    byte-identical keys but different domains must never collapse into one
    group, even though nothing in the real 49-row catalog happens to produce
    this case today."""
    a = LegacyResearchNeed(
        legacy_need_id="synthetic_a",
        origin=AcquisitionOrigin.BEAR,
        original_template="{c} pricing pressure",
        domain=ResearchDomain.COMPETITION,
        subject_scope=subject_scope_for_template("{c} pricing pressure"),
        equivalence_key=equivalence_key("{c} pricing pressure"),
    )
    b = LegacyResearchNeed(
        legacy_need_id="synthetic_b",
        origin=AcquisitionOrigin.KILL,
        original_template="{t} pricing pressure",
        domain=ResearchDomain.CAPITAL_STRUCTURE,
        subject_scope=subject_scope_for_template("{t} pricing pressure"),
        equivalence_key=equivalence_key("{t} pricing pressure"),
    )
    assert a.equivalence_key == b.equivalence_key
    assert a.domain != b.domain

    groups = group_legacy_needs((a, b))
    assert len(groups) == 2


# --- Phase 2 required tests: COMPANY/PROGRAM subject-scope correctness -----
def test_ticker_and_company_name_templates_merge_as_the_same_company_subject():
    """{t} failed and {c} failed must merge: ticker and company name refer to
    the same issuer and may share acquisition."""
    a = LegacyResearchNeed(
        legacy_need_id="synthetic_t_failed",
        origin=AcquisitionOrigin.KILL,
        original_template="{t} failed",
        domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        subject_scope=subject_scope_for_template("{t} failed"),
        equivalence_key=equivalence_key("{t} failed"),
    )
    b = LegacyResearchNeed(
        legacy_need_id="synthetic_c_failed",
        origin=AcquisitionOrigin.BEAR,
        original_template="{c} failed",
        domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        subject_scope=subject_scope_for_template("{c} failed"),
        equivalence_key=equivalence_key("{c} failed"),
    )
    assert a.subject_scope is b.subject_scope is SubjectScope.COMPANY
    assert a.equivalence_key == b.equivalence_key

    groups = group_legacy_needs((a, b))
    assert len(groups) == 1
    assert set(next(iter(groups.values()))) == {"synthetic_t_failed", "synthetic_c_failed"}


def test_company_and_program_templates_never_merge_even_with_identical_wording():
    """{c} failed and {d} failed must NOT merge: a company-level question and
    a programme-level question are different subjects even when every other
    word is identical."""
    a = LegacyResearchNeed(
        legacy_need_id="synthetic_c_failed_2",
        origin=AcquisitionOrigin.BEAR,
        original_template="{c} failed",
        domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        subject_scope=subject_scope_for_template("{c} failed"),
        equivalence_key=equivalence_key("{c} failed"),
    )
    b = LegacyResearchNeed(
        legacy_need_id="synthetic_d_failed",
        origin=AcquisitionOrigin.BEAR,
        original_template="{d} failed",
        domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        subject_scope=subject_scope_for_template("{d} failed"),
        equivalence_key=equivalence_key("{d} failed"),
    )
    assert a.subject_scope is SubjectScope.COMPANY
    assert b.subject_scope is SubjectScope.PROGRAM
    assert a.equivalence_key != b.equivalence_key  # "company failed" vs "program failed"

    groups = group_legacy_needs((a, b))
    assert len(groups) == 2


def test_same_text_same_domain_different_subject_scope_never_merges():
    """Even if a hand-built equivalence_key were engineered to collide, the
    explicit subject_scope component of the grouping key still separates a
    COMPANY need from a PROGRAM need -- defense in depth beyond the key
    string alone."""
    a = LegacyResearchNeed(
        legacy_need_id="synthetic_forced_company",
        origin=AcquisitionOrigin.BEAR,
        original_template="{c} safety concern",
        domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        subject_scope=SubjectScope.COMPANY,
        equivalence_key="forced identical key",
    )
    b = LegacyResearchNeed(
        legacy_need_id="synthetic_forced_program",
        origin=AcquisitionOrigin.BEAR,
        original_template="{d} safety concern",
        domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        subject_scope=SubjectScope.PROGRAM,
        equivalence_key="forced identical key",
    )
    assert a.equivalence_key == b.equivalence_key
    assert a.domain == b.domain
    assert a.subject_scope != b.subject_scope

    groups = group_legacy_needs((a, b))
    assert len(groups) == 2


def test_subject_scope_for_template_rejects_ambiguous_or_missing_placeholders():
    import pytest

    with pytest.raises(ValueError):
        subject_scope_for_template("no placeholder at all")
    with pytest.raises(ValueError):
        subject_scope_for_template("{t} and {d} both present")


def test_equivalence_key_is_placeholder_and_case_and_whitespace_insensitive():
    assert equivalence_key("{c} going concern") == equivalence_key("{t}  going   concern")
    assert equivalence_key("{d} Safety Concern") == "program safety concern"
    assert equivalence_key("{c} Safety Concern") == "company safety concern"


# --- untouched aggregate gates ----------------------------------------------
def test_kill_rules_count_is_unchanged_at_19():
    assert len(KILL_RULES) == 19


def test_required_research_domains_unchanged_at_6():
    assert len(REQUIRED_RESEARCH_DOMAINS) == 6
    assert set(REQUIRED_RESEARCH_DOMAINS) == set(ResearchDomain)
