"""Phase 3E: collectors/form4.py exercised entirely offline against the
real-format fixtures in tests/fixtures/form4_real_format/.

No real network call anywhere in this file -- pure local fixture text.
"""

from __future__ import annotations

from investment_research.collectors.form4 import (
    VALID_FORM4_DOCUMENT_TYPES,
    XmlShapeError,
    check_ownership_xml_shape,
    parse_ownership_document,
    raw_facts_from_form4,
)
from investment_research.schemas.enums import UNKNOWN, SourceTier
from investment_research.schemas.fact import Source, make_source_id

from . import _form4_fixture_support as fx


def _source() -> Source:
    return Source(source_id=make_source_id("http://x", "t"), url="http://x", title="t", tier=SourceTier.TIER_1)


# --- shape gate ---------------------------------------------------------
def test_valid_form4_types_are_exactly_4_and_4a():
    assert frozenset({"4", "4/A"}) == VALID_FORM4_DOCUMENT_TYPES


def test_check_shape_accepts_a_normal_form4():
    error, document_type = check_ownership_xml_shape(fx.fixture_xml("normal_market_purchase"))
    assert error is None
    assert document_type == "4"


def test_check_shape_accepts_form4a_as_its_own_distinct_type():
    error, document_type = check_ownership_xml_shape(fx.fixture_xml("form4a_amendment"))
    assert error is None
    assert document_type == "4/A"
    assert document_type != "4"


def test_check_shape_rejects_malformed_xml():
    error, reason = check_ownership_xml_shape(fx.fixture_xml("malformed_xml"))
    assert error is XmlShapeError.MALFORMED
    assert "malformed" in reason.lower()


def test_check_shape_rejects_missing_ownership_document_root():
    error, reason = check_ownership_xml_shape(fx.fixture_xml("missing_ownership_document"))
    assert error is XmlShapeError.MISSING_OWNERSHIP_DOCUMENT
    assert "ownershipDocument" in reason


def test_check_shape_rejects_form3_masquerading_as_form4():
    error, reason = check_ownership_xml_shape(fx.fixture_xml("body_says_form3"))
    assert error is XmlShapeError.WRONG_DOCUMENT_TYPE
    assert "3" in reason


def test_parse_returns_none_for_any_shape_failure():
    assert parse_ownership_document(fx.fixture_xml("malformed_xml")) is None
    assert parse_ownership_document(fx.fixture_xml("missing_ownership_document")) is None
    assert parse_ownership_document(fx.fixture_xml("body_says_form3")) is None
    assert parse_ownership_document("") is None


# --- transaction codes kept verbatim, never interpreted ---------------------
def test_market_purchase_code_and_direction_kept_literal():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    t = parsed["non_derivative_transactions"][0]
    assert t["transaction_code"] == "P"
    assert t["acquired_or_disposed"] == "A"
    # Never translated into English words anywhere in the parsed record.
    flat = " ".join(str(v) for v in t.values())
    assert "bought" not in flat.lower()
    assert "purchase" not in flat.lower()


def test_market_sale_code_and_direction_kept_literal():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_sale"))
    t = parsed["non_derivative_transactions"][0]
    assert t["transaction_code"] == "S"
    assert t["acquired_or_disposed"] == "D"


def test_tax_withholding_code_never_confused_with_market_sale():
    parsed = parse_ownership_document(fx.fixture_xml("tax_withholding"))
    t = parsed["non_derivative_transactions"][0]
    assert t["transaction_code"] == "F"
    assert t["transaction_code"] != "S"


def test_grant_award_code_never_confused_with_market_purchase():
    parsed = parse_ownership_document(fx.fixture_xml("grant_award"))
    t = parsed["non_derivative_transactions"][0]
    assert t["transaction_code"] == "A"
    assert t["transaction_code"] != "P"


def test_gift_code_kept_distinct():
    parsed = parse_ownership_document(fx.fixture_xml("gift"))
    t = parsed["non_derivative_transactions"][0]
    assert t["transaction_code"] == "G"


def test_option_exercise_is_a_derivative_transaction_never_non_derivative():
    parsed = parse_ownership_document(fx.fixture_xml("option_exercise"))
    assert len(parsed["derivative_transactions"]) == 1
    assert parsed["non_derivative_transactions"] == []
    t = parsed["derivative_transactions"][0]
    assert t["transaction_code"] == "M"
    assert t["is_derivative"] is True


def test_derivative_transaction_carries_derivative_only_fields():
    parsed = parse_ownership_document(fx.fixture_xml("derivative_transaction"))
    t = parsed["derivative_transactions"][0]
    assert t["exercise_price"] == "20.00"
    assert t["exercise_date"] == "2027-01-01"
    assert t["expiration_date"] == "2031-06-01"
    assert t["underlying_security_title"] == "Common Stock"
    assert t["underlying_security_shares"] == "200"
    # Non-derivative transactions never carry these as anything but UNKNOWN.
    purchase = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    nd = purchase["non_derivative_transactions"][0]
    assert nd["exercise_price"] == UNKNOWN
    assert nd["underlying_security_title"] == UNKNOWN


def test_indirect_ownership_nature_preserved():
    parsed = parse_ownership_document(fx.fixture_xml("indirect_ownership"))
    t = parsed["non_derivative_transactions"][0]
    assert t["direct_or_indirect_ownership"] == "I"
    assert t["nature_of_indirect_ownership"] == "By Family Trust"


def test_multiple_transactions_never_merged():
    parsed = parse_ownership_document(fx.fixture_xml("multiple_transactions"))
    transactions = parsed["non_derivative_transactions"]
    assert len(transactions) == 2
    assert transactions[0]["transaction_code"] == "P"
    assert transactions[1]["transaction_code"] == "S"
    assert transactions[0]["transaction_date"] != transactions[1]["transaction_date"]
    # Never summed into one shares figure.
    assert {transactions[0]["transaction_shares"], transactions[1]["transaction_shares"]} == {"400", "150"}


# --- Rule 10b5-1: footnote-explicit legacy/fallback signal -----------------
def test_10b5_1_plan_footnote_fallback_set_true_only_on_explicit_footnote_reference():
    parsed = parse_ownership_document(fx.fixture_xml("footnote_10b5_1"))
    t = parsed["non_derivative_transactions"][0]
    assert t["footnote_ids"] == ("F1",)
    assert t["is_10b5_1_plan_footnote_fallback"] is True


def test_10b5_1_plan_footnote_fallback_never_guessed_when_no_footnote_referenced():
    for scenario in ("normal_market_purchase", "normal_market_sale", "tax_withholding", "grant_award", "gift"):
        parsed = parse_ownership_document(fx.fixture_xml(scenario))
        for t in parsed["non_derivative_transactions"] + parsed["derivative_transactions"]:
            assert t["is_10b5_1_plan_footnote_fallback"] is None, (
                f"{scenario} transaction wrongly guessed a 10b5-1 plan"
            )


# --- Rule 10b5-1: document-level checkbox (Phase 3E.1) --------------------
def test_ten_b5_1_checkbox_absent_is_unknown_never_false():
    """Every existing fixture has no checkbox element at all -- absence
    must resolve to None (UNKNOWN), never a guessed False (Phase 3E.1
    requirement 2)."""
    for scenario in ("normal_market_purchase", "footnote_10b5_1", "multiple_transactions"):
        parsed = parse_ownership_document(fx.fixture_xml(scenario))
        assert parsed["ten_b5_1_checkbox"] is None


def test_ten_b5_1_checkbox_true_when_element_present():
    import xml.etree.ElementTree as ET

    from investment_research.collectors.form4 import _read_checkbox

    root = ET.fromstring("<ownershipDocument><aff10b5One><value>1</value></aff10b5One></ownershipDocument>")
    assert _read_checkbox(root, ("aff10b5One",)) is True


def test_ten_b5_1_checkbox_false_when_element_present_and_unchecked():
    import xml.etree.ElementTree as ET

    from investment_research.collectors.form4 import _read_checkbox

    root = ET.fromstring("<ownershipDocument><aff10b5One><value>0</value></aff10b5One></ownershipDocument>")
    assert _read_checkbox(root, ("aff10b5One",)) is False


def test_ten_b5_1_checkbox_never_auto_attributed_to_individual_transactions():
    """The document-level checkbox field lives only at the top level of
    the parsed record -- never copied down onto individual transactions,
    which carry only their own separate legacy/fallback field (Phase
    3E.1 requirement 2)."""
    parsed = parse_ownership_document(fx.fixture_xml("footnote_10b5_1"))
    assert "ten_b5_1_checkbox" in parsed
    for t in parsed["non_derivative_transactions"] + parsed["derivative_transactions"]:
        assert "ten_b5_1_checkbox" not in t
        assert "is_10b5_1_plan_footnote_fallback" in t


def test_ten_b5_1_plan_adoption_date_only_from_explicit_text():
    parsed = parse_ownership_document(fx.fixture_xml("footnote_10b5_1"))
    # This fixture's footnote mentions 10b5-1 but never an adoption date.
    assert parsed["ten_b5_1_plan_adoption_date"] == UNKNOWN


def test_ten_b5_1_plan_adoption_date_extracted_when_explicitly_stated():
    from investment_research.collectors.form4 import _extract_plan_adoption_date

    remarks = "Effected pursuant to a Rule 10b5-1 trading plan adopted on March 2, 2026."
    assert _extract_plan_adoption_date(remarks, {}) == "March 2, 2026"


def test_ten_b5_1_plan_adoption_date_never_guessed_without_10b5_1_mention():
    from investment_research.collectors.form4 import _extract_plan_adoption_date

    remarks = "Adopted on March 2, 2026."  # no "10b5-1" anywhere
    assert _extract_plan_adoption_date(remarks, {}) == UNKNOWN


# --- amendment cross-reference extraction (Phase 3E.1 requirement 3) ------
def test_extract_referenced_accession_finds_an_embedded_accession():
    from investment_research.collectors.form4 import extract_referenced_accession

    text = "This Form 4/A amends the Form 4 filed with accession number 0005556667-26-000001."
    assert extract_referenced_accession(text) == "0005556667-26-000001"


def test_extract_referenced_accession_unknown_when_absent():
    from investment_research.collectors.form4 import extract_referenced_accession

    assert extract_referenced_accession("No accession mentioned here.") == UNKNOWN
    assert extract_referenced_accession("") == UNKNOWN


def test_accession_in_text_pattern_stays_in_sync_with_sec_accession_pattern():
    """collectors/form4.py's module docstring claims this module's own
    accession-shaped-text pattern is kept in sync with sec_acquisition_
    adapters._ACCESSION_RE -- this test is that sync check."""
    from investment_research.collectors.form4 import _ACCESSION_IN_TEXT_RE
    from investment_research.research.sec_acquisition_adapters import _ACCESSION_RE

    core_in_text = _ACCESSION_IN_TEXT_RE.pattern.replace(r"\b", "")
    core_standalone = _ACCESSION_RE.pattern.replace("^", "").replace("$", "")
    assert core_in_text == core_standalone


# --- Form 4/A amendment -------------------------------------------------
def test_form4a_amendment_document_type_kept_distinct():
    parsed = parse_ownership_document(fx.fixture_xml("form4a_amendment"))
    assert parsed["document_type"] == "4/A"
    assert parsed["document_type"] != "4"


# --- missing fields: UNKNOWN, never guessed ---------------------------------
def test_missing_optional_fields_are_unknown_never_guessed():
    parsed = parse_ownership_document(fx.fixture_xml("missing_fields"))
    assert parsed["issuer_ticker"] == UNKNOWN
    assert parsed["reporting_owners"][0]["officer_title"] == UNKNOWN
    # Required fields are still present.
    assert parsed["issuer_cik"] != UNKNOWN
    assert parsed["reporting_owners"]


def test_missing_required_field_leaves_issuer_cik_unknown():
    parsed = parse_ownership_document(fx.fixture_xml("missing_required_fields"))
    assert parsed is not None  # shape itself is still valid XML
    assert parsed["issuer_cik"] == UNKNOWN


# --- issuer/reporting-owner CIK never confused ------------------------------
def test_issuer_cik_and_reporting_owner_cik_are_distinct_fields():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    assert parsed["issuer_cik"] != parsed["reporting_owners"][0]["reporting_owner_cik"]
    assert parsed["issuer_cik"] == fx.ISSUER_CIK
    assert parsed["reporting_owners"][0]["reporting_owner_cik"] == fx.OWNER_CIK_PADDED


# --- raw facts: never merged, company_claim=False, verbatim codes ----------
def test_raw_facts_never_set_company_claim_true():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    facts = raw_facts_from_form4("SAMPB", parsed, _source())
    assert facts
    assert all(f.company_claim is False for f in facts)


def test_raw_facts_one_per_transaction_never_aggregated():
    parsed = parse_ownership_document(fx.fixture_xml("multiple_transactions"))
    facts = raw_facts_from_form4("SAMPB", parsed, _source())
    transaction_facts = [f for f in facts if f.unit == "non_derivative_transaction"]
    assert len(transaction_facts) == 2
    assert {f.value for f in transaction_facts} == {"P", "S"}


def test_raw_facts_carry_document_id_when_supplied():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    facts = raw_facts_from_form4("SAMPB", parsed, _source(), document_id="doc_form4_abc123")
    assert all(f.document_id == "doc_form4_abc123" for f in facts)


def test_raw_facts_document_id_none_when_not_supplied():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    facts = raw_facts_from_form4("SAMPB", parsed, _source())
    assert all(f.document_id is None for f in facts)


def test_raw_facts_relationship_fact_present_per_reporting_owner():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    facts = raw_facts_from_form4("SAMPB", parsed, _source())
    relationship_facts = [f for f in facts if f.unit == "reporting_owner_relationship"]
    assert len(relationship_facts) == len(parsed["reporting_owners"])


def test_raw_facts_never_contain_buy_sell_words():
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    facts = raw_facts_from_form4("SAMPB", parsed, _source())
    for f in facts:
        lowered = f.claim.lower()
        assert "bought" not in lowered
        assert "sold" not in lowered
        assert "bullish" not in lowered
        assert "bearish" not in lowered


# =============================================================================
# Phase 3E.4: relationship field inconsistency (real Mac Live Smoke finding)
# =============================================================================
def test_relationship_fields_inconsistent_true_when_officer_false_but_title_present():
    """A real capture found isOfficer=False with a genuinely non-empty
    officerTitle -- flagged, never 'corrected' in either direction."""
    parsed = parse_ownership_document(fx.fixture_text("officer_title_inconsistent.xml"))
    owner = parsed["reporting_owners"][0]
    assert owner["is_officer"] is False
    assert owner["officer_title"] == "Chief Financial Officer"
    assert owner["relationship_fields_inconsistent"] is True


def test_relationship_fields_inconsistent_false_on_ordinary_fixtures():
    for scenario in ("normal_market_purchase", "footnote_10b5_1", "multiple_transactions"):
        parsed = parse_ownership_document(fx.fixture_xml(scenario))
        for owner in parsed["reporting_owners"]:
            assert owner["relationship_fields_inconsistent"] is False


def test_relationship_inconsistency_never_promoted_to_a_transaction_or_confirmation_fact():
    """The inconsistency flag lives only on the parsed reporting_owners
    record -- raw_facts_from_form4 never emits a fact about it, and never
    sets company_claim=True because of it (requirement 17)."""
    parsed = parse_ownership_document(fx.fixture_text("officer_title_inconsistent.xml"))
    facts = raw_facts_from_form4("SAMPB", parsed, _source())
    assert facts
    for f in facts:
        assert "inconsistent" not in f.claim.lower()
        assert f.company_claim is False


# =============================================================================
# Phase 3E.4: dateOfOriginalSubmission promoted to a real parsed field
# =============================================================================
def test_date_of_original_submission_unknown_when_absent():
    for scenario in ("normal_market_purchase", "form4a_amendment", "reconciled_amendment"):
        parsed = parse_ownership_document(fx.fixture_xml(scenario))
        assert parsed["date_of_original_submission"] == UNKNOWN


def test_date_of_original_submission_read_when_genuinely_present():
    parsed = parse_ownership_document(fx.fixture_text("amendment_with_date_of_original_submission.xml"))
    assert parsed["date_of_original_submission"] == "2026-02-15"
    assert parsed["document_type"] == "4/A"


def test_date_of_original_submission_never_used_to_infer_original_accession():
    """Requirement 19/20: a genuinely present dateOfOriginalSubmission,
    with no accession named in remarks, still leaves
    remarks_referenced_accession UNKNOWN -- the date is never consulted
    as a substitute (this module's own extract_referenced_accession is
    remarks-text-only and has no date parameter at all)."""
    parsed = parse_ownership_document(fx.fixture_text("amendment_with_date_of_original_submission.xml"))
    assert parsed["date_of_original_submission"] == "2026-02-15"
    assert parsed["remarks_referenced_accession"] == UNKNOWN
