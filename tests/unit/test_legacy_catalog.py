"""Phase 1: the 49-row legacy acquisition catalog and its exact-match dedup.

Locks the counts the Phase 1 authorization made an absolute condition: 49 raw
rows, 31 groups after exact (equivalence_key, domain) dedup, 18 collapsed.
Also locks that KILL_RULES and REQUIRED_RESEARCH_DOMAINS -- both explicitly
untouched by this change -- still report their pre-existing counts.
"""

from __future__ import annotations

from investment_research.research.checks import AcquisitionOrigin, LegacyResearchNeed
from investment_research.research.legacy_catalog import (
    LEGACY_CATALOG,
    collapsed_count,
    equivalence_key,
    group_legacy_needs,
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
    under ResearchDomain.REGULATORY but are NOT the same equivalence_key --
    they must stay in separate groups despite the obvious topical overlap a
    fuzzy/token-overlap comparison would collapse."""
    by_id = {need.legacy_need_id: need for need in LEGACY_CATALOG}
    bear_2 = by_id["bear_2"]
    kill_3 = by_id["kill_3"]
    assert bear_2.domain == kill_3.domain == ResearchDomain.REGULATORY
    assert bear_2.equivalence_key != kill_3.equivalence_key

    groups = group_legacy_needs()
    group_of = {need_id: key for key, ids in groups.items() for need_id in ids}
    assert group_of["bear_2"] != group_of["kill_3"]


def test_same_text_different_domain_never_merges():
    """Grouping is (equivalence_key, domain), not equivalence_key alone: two
    needs with byte-identical keys but different domains must never collapse
    into one group, even though nothing in the real 49-row catalog happens to
    produce this case today."""
    a = LegacyResearchNeed(
        legacy_need_id="synthetic_a",
        origin=AcquisitionOrigin.BEAR,
        original_template="{c} pricing pressure",
        domain=ResearchDomain.COMPETITION,
        equivalence_key=equivalence_key("{c} pricing pressure"),
    )
    b = LegacyResearchNeed(
        legacy_need_id="synthetic_b",
        origin=AcquisitionOrigin.KILL,
        original_template="{t} pricing pressure",
        domain=ResearchDomain.CAPITAL_STRUCTURE,
        equivalence_key=equivalence_key("{t} pricing pressure"),
    )
    assert a.equivalence_key == b.equivalence_key
    assert a.domain != b.domain

    groups = group_legacy_needs((a, b))
    assert len(groups) == 2


def test_equivalence_key_is_placeholder_and_case_and_whitespace_insensitive():
    assert equivalence_key("{c} going concern") == equivalence_key("{t}  going   concern")
    assert equivalence_key("{d} Safety Concern") == "entity safety concern"


# --- untouched aggregate gates ----------------------------------------------
def test_kill_rules_count_is_unchanged_at_19():
    assert len(KILL_RULES) == 19


def test_required_research_domains_unchanged_at_6():
    assert len(REQUIRED_RESEARCH_DOMAINS) == 6
    assert set(REQUIRED_RESEARCH_DOMAINS) == set(ResearchDomain)
