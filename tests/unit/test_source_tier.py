"""Source tier and syndication tests (requirements 2 and 13)."""

from __future__ import annotations

import pytest

from investment_research.collectors.tiering import (
    classify_tier,
    content_signature,
    independent_source_count,
    is_company_controlled,
    is_reprint,
    similarity,
)
from investment_research.schemas.enums import NON_DECISIVE_TIERS, SourceTier


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.sec.gov/Archives/edgar/data/1/x.htm", SourceTier.TIER_1),
        ("https://data.sec.gov/submissions/CIK0000320193.json", SourceTier.TIER_1),
        ("https://clinicaltrials.gov/study/NCT00000000", SourceTier.TIER_1),
        ("https://www.fda.gov/news/x", SourceTier.TIER_1),
        ("https://www.jpx.co.jp/x", SourceTier.TIER_1),
        ("https://pubmed.ncbi.nlm.nih.gov/12345678/", SourceTier.TIER_2),
        ("https://www.nejm.org/doi/full/10.1056/x", SourceTier.TIER_2),
        ("https://www.reuters.com/business/x", SourceTier.TIER_3),
        ("https://www.nikkei.com/article/x", SourceTier.TIER_3),
        ("https://www.tipranks.com/stocks/x", SourceTier.TIER_4),
        ("https://seekingalpha.com/article/x", SourceTier.TIER_5),
        ("https://www.reddit.com/r/x", SourceTier.TIER_5),
        ("https://unknown-blog.example/post", SourceTier.UNKNOWN),
    ],
)
def test_tier_classification(url, expected):
    assert classify_tier(url) is expected


def test_pubmed_is_tier2_despite_nih_suffix():
    """Longest-suffix matching: literature under nih.gov must not become Tier 1."""
    assert classify_tier("https://pubmed.ncbi.nlm.nih.gov/1/") is SourceTier.TIER_2
    assert classify_tier("https://www.nih.gov/news/1") is SourceTier.TIER_1


def test_press_release_wires_are_company_controlled():
    url = "https://www.globenewswire.com/news-release/x"
    assert is_company_controlled(url)
    assert classify_tier(url) is SourceTier.TIER_2


def test_company_ir_flag_overrides_host():
    assert classify_tier("https://ir.example.com/press", is_company_ir=True) is SourceTier.TIER_2


def test_unknown_tier_is_not_decisive():
    assert SourceTier.UNKNOWN in NON_DECISIVE_TIERS
    assert SourceTier.TIER_4 in NON_DECISIVE_TIERS
    assert SourceTier.TIER_5 in NON_DECISIVE_TIERS
    assert SourceTier.TIER_1 not in NON_DECISIVE_TIERS


# --- syndication ------------------------------------------------------------
WIRE = (
    "Demobio announced today that the independent data monitoring committee completed its "
    "planned interim review and recommended that the study continue without modification."
)
REPRINT = (
    "PRESS RELEASE: Demobio announced today that the independent data monitoring committee "
    "completed its planned interim review and recommended that the study continue without "
    "modification. Copyright 2026 Wire Service."
)
DISTINCT = (
    "The company disclosed a registered direct offering of twelve million shares priced at "
    "the market under Nasdaq rules, with warrants attached."
)


def test_reprint_detected_despite_different_headline():
    assert is_reprint(WIRE, REPRINT)
    assert similarity(WIRE, REPRINT) > 0.6


def test_distinct_articles_are_not_reprints():
    assert not is_reprint(WIRE, DISTINCT)


def test_syndicated_copies_count_as_one_source():
    """Requirement 13: a wire story on five sites is one source, not five."""
    count, representatives = independent_source_count(
        [
            ("s1", "https://www.reuters.com/a", WIRE),
            ("s2", "https://finance.yahoo.com/a", REPRINT),
            ("s3", "https://www.marketbeat.com/a", REPRINT),
            ("s4", "https://www.sec.gov/filing", DISTINCT),
        ]
    )
    assert count == 2
    assert representatives == ["s1", "s4"]


def test_same_host_counts_once():
    count, _ = independent_source_count(
        [
            ("s1", "https://www.reuters.com/a", "one story about the balance sheet"),
            ("s2", "https://www.reuters.com/b", "a totally different story about hiring"),
        ]
    )
    assert count == 1


def test_content_signature_stable_and_distinct():
    assert content_signature(WIRE) == content_signature(WIRE)
    assert content_signature(WIRE) != content_signature(DISTINCT)
    assert content_signature("") == "UNKNOWN"
