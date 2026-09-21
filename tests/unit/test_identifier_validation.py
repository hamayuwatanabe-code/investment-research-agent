"""Phase 4.3B requirement 4/6: strict identifier validation and
deterministic company-name normalization (``scoring/identifier_validation.py``).
"""

from __future__ import annotations

from investment_research.scoring.identifier_validation import (
    normalize_company_name,
    validate_strict_nct_id,
    validate_strict_pmid,
)
from investment_research.scoring.program_resolution import NCT_RE as LOOSE_NCT_RE


# =============================================================================
# NCT id: strict ^NCT\d{8}$
# =============================================================================
def test_valid_nct_id_normalized_uppercase():
    assert validate_strict_nct_id("nct12345678") == "NCT12345678"


def test_valid_nct_id_strips_whitespace():
    assert validate_strict_nct_id("  NCT12345678  ") == "NCT12345678"


def test_nct_id_too_few_digits_rejected():
    assert validate_strict_nct_id("NCT1234567") is None


def test_nct_id_too_many_digits_rejected():
    assert validate_strict_nct_id("NCT123456789") is None


def test_nct_id_non_digit_suffix_rejected():
    assert validate_strict_nct_id("NCT1234567X") is None


def test_nct_id_missing_prefix_rejected():
    assert validate_strict_nct_id("12345678") is None


def test_nct_id_hyphenated_shape_rejected():
    """The exact loose shape scoring.program_resolution.NCT_RE accepts
    (``[-A-Z0-9]{4,}``) but strict validation must not."""
    assert validate_strict_nct_id("NCT-ABCD") is None


def test_nct_id_none_and_empty_rejected():
    assert validate_strict_nct_id(None) is None
    assert validate_strict_nct_id("") is None
    assert validate_strict_nct_id("   ") is None


def test_loose_nct_re_accepts_strings_strict_validator_rejects():
    """Proves the two regexes are genuinely different in strictness --
    guards against ever silently reusing the loose one for outbound
    validation (Phase 4.3A-correction-1's own finding)."""
    loose_only_examples = ["NCT-ABC1", "NCT123A", "NCTXXXX"]
    for example in loose_only_examples:
        assert LOOSE_NCT_RE.search(f"trial {example} is ongoing") is not None, example
        assert validate_strict_nct_id(example) is None, example


# =============================================================================
# PMID: strict ^\d{1,9}$ after stripping, no locale/case concerns
# =============================================================================
def test_valid_pmid_passthrough():
    assert validate_strict_pmid("33378609") == "33378609"


def test_valid_pmid_strips_whitespace():
    assert validate_strict_pmid("  33378609  ") == "33378609"


def test_pmid_nine_digits_is_the_generous_upper_bound():
    assert validate_strict_pmid("123456789") == "123456789"


def test_pmid_ten_digits_rejected():
    assert validate_strict_pmid("1234567890") is None


def test_pmid_non_numeric_rejected():
    assert validate_strict_pmid("PMID123") is None


def test_pmid_none_and_empty_rejected():
    assert validate_strict_pmid(None) is None
    assert validate_strict_pmid("") is None


def test_pmid_ambiguous_mixed_content_rejected():
    assert validate_strict_pmid("123 or 456") is None


# =============================================================================
# Company name normalization -- deterministic, never fuzzy
# =============================================================================
def test_normalize_strips_legal_suffix_and_punctuation():
    assert normalize_company_name("Moderna, Inc.") == "MODERNA"


def test_normalize_case_insensitive():
    assert normalize_company_name("moderna inc") == normalize_company_name("MODERNA INC")


def test_normalize_collapses_whitespace():
    assert normalize_company_name("Demo   Biotherapeutics   Inc") == "DEMO BIOTHERAPEUTICS"


def test_normalize_never_strips_industry_descriptor_words():
    """Deliberate: stripping 'Therapeutics'/'Pharmaceuticals'/'Biosciences'
    would risk merging two DIFFERENT companies' identities -- the exact
    misjoin risk Phase 4.3A-correction-1 flagged."""
    assert normalize_company_name("Acme Therapeutics Inc") == "ACME THERAPEUTICS"
    assert normalize_company_name("Acme Diagnostics Inc") == "ACME DIAGNOSTICS"
    assert normalize_company_name("Acme Therapeutics Inc") != normalize_company_name(
        "Acme Diagnostics Inc"
    )


def test_normalize_unknown_and_empty_yield_empty_string():
    assert normalize_company_name("UNKNOWN") == ""
    assert normalize_company_name("") == ""


def test_normalize_never_fuzzy_matches_similar_but_different_names():
    """No edit-distance/phonetic matching anywhere -- a genuinely different
    company name never normalizes to the same string as a similar one."""
    assert normalize_company_name("Moderna Inc") != normalize_company_name("Modrna Inc")
    assert normalize_company_name("Moderna Inc") != normalize_company_name("Moderna Therapeutics Inc")


def test_normalize_only_strips_pure_legal_form_tokens_not_holdings():
    """"Holdings" is deliberately NOT stripped (see identifier_validation.py's
    own docstring: it can be part of what distinguishes one corporate
    entity from another) -- only the trailing pure legal-form token is."""
    assert normalize_company_name("Demo Holdings LLC") == "DEMO HOLDINGS"


def test_normalize_multiple_trailing_legal_suffixes_all_stripped():
    assert normalize_company_name("Demo Pharma Corp Inc") == "DEMO PHARMA"
