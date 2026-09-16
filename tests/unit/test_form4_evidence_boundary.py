"""Phase 3E.4: the REPORTING_PERSON_STATUTORY_ASSERTION EvidenceClass /
REPORTING_PERSON_FILING DocumentAuthority boundary -- audited across every
consumer requirement 12 names (evidence_integrity.py, escalation.py,
fact_collector.py, validation.py, storage round-trip), proving no path
promotes a Form 4 fact to VERIFIED_FACT or independent_confirmation=True,
and that ordinary (non-INSIDER) company-filing/regulator evidence is
completely unaffected.

Entirely offline: no network, no fixtures beyond synthetic in-memory
constructions and the existing Form4 real-format XML fixtures.
"""

from __future__ import annotations

import pytest

from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.form4 import parse_ownership_document, raw_facts_from_form4
from investment_research.research.escalation import _EVIDENCE_FOR_AUTHORITY, _evidence_for_authority
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import (
    DECISION_GRADE_CLASSES,
    DocumentAuthority,
    EvidenceClass,
    FactCategory,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Fact, Source, make_source_id
from investment_research.schemas.validation import SchemaError, validate_fact
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository
from tests.conftest import TODAY, make_fact

from . import _form4_fixture_support as fx


# --- schemas/enums.py: the new values exist and are correctly excluded -----
def test_reporting_person_statutory_assertion_excluded_from_decision_grade():
    assert EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION not in DECISION_GRADE_CLASSES
    assert EvidenceClass.VERIFIED_FACT in DECISION_GRADE_CLASSES
    assert EvidenceClass.INDEPENDENT_EVIDENCE in DECISION_GRADE_CLASSES


def test_reporting_person_filing_is_distinct_from_statutory_filing():
    assert DocumentAuthority.REPORTING_PERSON_FILING != DocumentAuthority.STATUTORY_FILING


# --- research/escalation.py: complete mapping, correct triple --------------
def test_escalation_evidence_for_authority_covers_every_documentauthority_value():
    """Every DocumentAuthority member must have an entry in
    _EVIDENCE_FOR_AUTHORITY, or _evidence_for_authority raises KeyError --
    this test fails loudly the moment a future enum addition forgets to
    update the mapping (Phase 3E.4 requirement 12)."""
    for authority in DocumentAuthority:
        assert authority in _EVIDENCE_FOR_AUTHORITY, f"{authority} has no _EVIDENCE_FOR_AUTHORITY entry"
        _evidence_for_authority(authority)  # must not raise


def test_escalation_reporting_person_filing_maps_to_the_dedicated_class():
    evidence_class, company_claim, independent_confirmation = _evidence_for_authority(
        DocumentAuthority.REPORTING_PERSON_FILING
    )
    assert evidence_class is EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION
    assert company_claim is False
    assert independent_confirmation is False


def test_escalation_statutory_filing_and_regulator_mappings_unchanged():
    """Requirement 13: ordinary company SEC filings and REGULATOR
    authority keep their existing, pre-Phase-3E.4 treatment exactly."""
    assert _evidence_for_authority(DocumentAuthority.STATUTORY_FILING) == (
        EvidenceClass.VERIFIED_FACT, True, False,
    )
    assert _evidence_for_authority(DocumentAuthority.REGULATOR) == (
        EvidenceClass.INDEPENDENT_EVIDENCE, False, True,
    )
    assert _evidence_for_authority(DocumentAuthority.COMPANY_IR) == (
        EvidenceClass.COMPANY_CLAIM, True, False,
    )


# --- agents/fact_collector.py: safe initial placeholder --------------------
def test_fact_collector_initial_classification_of_a_form4_fact_is_never_verified():
    """Agent 1's own placeholder classification (before Agent 2 ever runs)
    must never itself be VERIFIED_FACT or the new class -- 'nothing
    arrives verified' (requirement 12)."""
    parsed = parse_ownership_document(fx.fixture_xml("normal_market_purchase"))
    source = Source(
        source_id=make_source_id("https://example.test/form4", "Form 4"),
        url="https://example.test/form4", title="Form 4", tier=SourceTier.TIER_1,
    )
    raw_facts = raw_facts_from_form4("SAMPB", parsed, source, document_id="doc_form4_test")
    assert raw_facts
    for raw in raw_facts:
        fact = FactCollectorAgent._to_fact(raw, run_id="test_run", provenance=raw.source.provenance)
        assert fact.evidence_class is EvidenceClass.UNVERIFIED_CLAIM
        assert fact.evidence_class is not EvidenceClass.VERIFIED_FACT
        assert fact.evidence_class is not EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION
        assert fact.company_claim is False


# --- agents/evidence_integrity.py: coincidental-grouping override ----------
def _form4_fact(claim: str, *, source_url: str, event_date: str = "2026-02-27") -> Fact:
    return Fact(
        fact_id=f"fact_{abs(hash((claim, source_url)))}",
        ticker="SAMPB",
        category=FactCategory.INSIDER,
        claim=claim,
        evidence_class=EvidenceClass.UNVERIFIED_CLAIM,
        source_id=make_source_id(source_url, "Form 4"),
        source_url=source_url,
        source_title="Form 4",
        source_tier=SourceTier.TIER_1,
        event_date=event_date,
        company_claim=False,
    )


def test_two_form4_facts_with_identical_claim_text_never_confirm_each_other():
    """The generic corroboration path (_independent_confirmation) only
    checks company_claim/tier, not category -- two Form 4 facts (from two
    different accessions/reporting persons) whose claim text happens to
    match exactly must still NEVER confirm each other (Phase 3E.4
    requirement 10)."""
    claim = "Form 4 non_derivative transaction for Sample Biotech Holdings, Inc.: identical text"
    fact_a = _form4_fact(claim, source_url="https://example.test/form4/a")
    fact_b = _form4_fact(claim, source_url="https://example.test/form4/b")

    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=(fact_a, fact_b),
    )
    assessed = list(agent.run(agent_input).facts)
    assert len(assessed) == 2
    for fact in assessed:
        assert fact.independent_confirmation is False
        assert fact.corroborating_source_ids == ()
        assert fact.evidence_class is EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION
        assert fact.verified_status is VerifiedStatus.NOT_VERIFIED


def test_non_insider_company_claim_confirmation_path_is_unaffected():
    """Requirement 13: an ordinary (non-INSIDER) company claim, confirmed
    by an independent Tier-1 source, still upgrades to VERIFIED_FACT
    exactly as before -- the Phase 3E.4 override is scoped to
    REPORTING_PERSON_STATUTORY_ASSERTION only."""
    claim = "The company completed a $50 million private placement"
    company = make_fact(claim, category=FactCategory.CAPITAL_STRUCTURE, company_claim=True, tier=SourceTier.TIER_2)
    confirming = make_fact(
        claim, category=FactCategory.CAPITAL_STRUCTURE, company_claim=False, tier=SourceTier.TIER_1,
        url="https://www.sec.gov/other-filing",
    )
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="TEST", company_name="Test Co",
        facts=(company, confirming),
    )
    assessed = {f.source_url: f for f in agent.run(agent_input).facts}
    assert assessed[company.source_url].independent_confirmation is True
    assert assessed[company.source_url].evidence_class is EvidenceClass.VERIFIED_FACT


# --- schemas/validation.py: defense-in-depth schema guard -------------------
def test_validate_fact_rejects_reporting_person_assertion_with_independent_confirmation_true():
    fact = make_fact(
        "Form 4 non_derivative transaction", category=FactCategory.INSIDER,
        evidence_class=EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION,
        company_claim=False, independent_confirmation=True, tier=SourceTier.TIER_1,
    )
    with pytest.raises(SchemaError, match="independent_confirmation=True"):
        validate_fact(fact)


def test_validate_fact_accepts_reporting_person_assertion_with_independent_confirmation_false():
    fact = make_fact(
        "Form 4 non_derivative transaction", category=FactCategory.INSIDER,
        evidence_class=EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION,
        company_claim=False, independent_confirmation=False, tier=SourceTier.TIER_1,
        verified=VerifiedStatus.NOT_VERIFIED,
    )
    validate_fact(fact)  # must not raise


# --- storage round-trip: the new enum value persists and reads back --------
def test_reporting_person_statutory_assertion_round_trips_through_sqlite():
    conn = open_db(":memory:")
    repo = Repository(conn)
    repo.upsert_company("SAMPB", "Sample Biotech Holdings, Inc.")
    fact = make_fact(
        "Form 4 non_derivative transaction", ticker="SAMPB", category=FactCategory.INSIDER,
        evidence_class=EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION,
        company_claim=False, independent_confirmation=False, tier=SourceTier.TIER_1,
        verified=VerifiedStatus.NOT_VERIFIED,
    )
    repo.save_source(
        Source(source_id=fact.source_id, url=fact.source_url, title=fact.source_title, tier=fact.source_tier)
    )
    repo.save_fact(fact)

    row = repo.latest_fact_row(fact.fact_id)
    assert row is not None
    assert row["evidence_class"] == "REPORTING_PERSON_STATUTORY_ASSERTION"
    assert EvidenceClass(row["evidence_class"]) is EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION
