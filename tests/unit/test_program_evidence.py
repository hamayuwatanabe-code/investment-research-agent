"""Phase 4.3B: immutable structured evidence types and their structural
validation (``scoring/program_evidence.py``)."""

from __future__ import annotations

import dataclasses

import pytest

from investment_research.schemas.enums import SourceTier
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
    EvidenceValidationOutcome,
    LiteratureCandidateEvidence,
    ProgramCandidateEvidence,
    build_program_candidate_evidence_from_parsed_study,
    validate_company_identity_evidence,
    validate_literature_candidate_evidence,
    validate_program_candidate_evidence,
)

RETRIEVED_AT = "2026-01-01T00:00:00+00:00"
REAL_FACT_ID = "fact_" + "a" * 20  # matches make_fact_id's own shape exactly


# =============================================================================
# Immutability (Phase 4.3B requirement 2)
# =============================================================================
def test_company_identity_evidence_is_frozen():
    evidence = CompanyIdentityEvidence(ticker="DEMOBIO")
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.ticker = "OTHER"  # type: ignore[misc]


def test_program_candidate_evidence_is_frozen():
    evidence = ProgramCandidateEvidence(nct_id="NCT12345678")
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.nct_id = "NCT00000000"  # type: ignore[misc]


def test_literature_candidate_evidence_is_frozen():
    evidence = LiteratureCandidateEvidence(pmid="123")
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.pmid = "456"  # type: ignore[misc]


def test_collection_fields_are_tuples_never_lists_or_dicts():
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678",
        collaborators=("A", "B"),
        interventions=("Drug X",),
        conditions=("Condition Y",),
        phases=("PHASE2",),
        supporting_fact_ids=(REAL_FACT_ID,),
    )
    for field_name in ("collaborators", "interventions", "conditions", "phases", "supporting_fact_ids"):
        value = getattr(evidence, field_name)
        assert isinstance(value, tuple), f"{field_name} is {type(value)}, not a tuple"

    company = CompanyIdentityEvidence(ticker="DEMOBIO", explicitly_verified_aliases=("Old Name",))
    assert isinstance(company.explicitly_verified_aliases, tuple)

    lit = LiteratureCandidateEvidence(pmid="123", nct_ids=("NCT12345678",))
    assert isinstance(lit.nct_ids, tuple)


def test_no_field_default_is_a_mutable_container():
    """Dataclass field defaults themselves must never be list/dict (which
    would be silently SHARED across every instance) -- every default here
    is either an immutable literal or absent."""
    for cls in (CompanyIdentityEvidence, ProgramCandidateEvidence, LiteratureCandidateEvidence):
        for f in dataclasses.fields(cls):
            assert not isinstance(f.default, (list, dict)), f"{cls.__name__}.{f.name}"
            assert f.default_factory in (dataclasses.MISSING,) or f.default_factory is tuple, (
                f"{cls.__name__}.{f.name} uses a non-tuple default_factory"
            )


# =============================================================================
# supporting_fact_ids never invented (Phase 4.3B requirement 2)
# =============================================================================
def test_real_shaped_fact_id_is_accepted():
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_1",
        retrieved_at=RETRIEVED_AT, supporting_fact_ids=(REAL_FACT_ID,),
    )
    result = validate_program_candidate_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.VALID


def test_invented_fact_id_shape_is_rejected():
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_1",
        retrieved_at=RETRIEVED_AT, supporting_fact_ids=("fact_totally_made_up",),
    )
    result = validate_program_candidate_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("supporting_fact_ids" in r for r in result.reasons)


def test_no_supporting_facts_is_not_itself_a_validation_failure():
    """Absence is fine -- an invented id is not."""
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_1",
        retrieved_at=RETRIEVED_AT, supporting_fact_ids=(),
    )
    result = validate_program_candidate_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.VALID


# =============================================================================
# CompanyIdentityEvidence validation
# =============================================================================
def test_company_identity_valid():
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=123456, sec_official_name="Demo Biotherapeutics Inc",
        source_id="src_sec_1", retrieved_at=RETRIEVED_AT,
    )
    assert validate_company_identity_evidence(evidence).outcome == EvidenceValidationOutcome.VALID


def test_company_identity_incomplete_when_cik_not_yet_resolved():
    evidence = CompanyIdentityEvidence(ticker="DEMOBIO", cik=None, source_id="src_sec_1", retrieved_at=RETRIEVED_AT)
    result = validate_company_identity_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.INCOMPLETE


def test_company_identity_incomplete_when_official_name_unknown():
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=123456, source_id="src_sec_1", retrieved_at=RETRIEVED_AT,
    )
    result = validate_company_identity_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.INCOMPLETE


def test_company_identity_invalid_empty_ticker():
    evidence = CompanyIdentityEvidence(ticker="", cik=1, source_id="src_1", retrieved_at=RETRIEVED_AT)
    assert validate_company_identity_evidence(evidence).outcome == EvidenceValidationOutcome.INVALID


def test_company_identity_invalid_negative_cik():
    evidence = CompanyIdentityEvidence(ticker="DEMOBIO", cik=-5, source_id="src_1", retrieved_at=RETRIEVED_AT)
    assert validate_company_identity_evidence(evidence).outcome == EvidenceValidationOutcome.INVALID


def test_company_identity_invalid_missing_source_id():
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=1, sec_official_name="Demo Inc", retrieved_at=RETRIEVED_AT,
    )
    assert validate_company_identity_evidence(evidence).outcome == EvidenceValidationOutcome.INVALID


# =============================================================================
# ProgramCandidateEvidence validation -- strict NCT / SourceTier preserved
# =============================================================================
def test_program_candidate_invalid_malformed_nct():
    evidence = ProgramCandidateEvidence(
        nct_id="NCT123", lead_sponsor="Demo Inc", source_id="src_1", retrieved_at=RETRIEVED_AT,
    )
    result = validate_program_candidate_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("nct_id" in r for r in result.reasons)


def test_program_candidate_invalid_loose_nct_shape_rejected():
    """The exact shape scoring.program_resolution.NCT_RE would accept but
    strict validation must not: a short alphanumeric id, never a real
    8-digit NCT id."""
    evidence = ProgramCandidateEvidence(
        nct_id="NCT-ABC1", lead_sponsor="Demo Inc", source_id="src_1", retrieved_at=RETRIEVED_AT,
    )
    assert validate_program_candidate_evidence(evidence).outcome == EvidenceValidationOutcome.INVALID


def test_program_candidate_incomplete_when_sponsor_unknown():
    evidence = ProgramCandidateEvidence(nct_id="NCT12345678", source_id="src_1", retrieved_at=RETRIEVED_AT)
    assert validate_program_candidate_evidence(evidence).outcome == EvidenceValidationOutcome.INCOMPLETE


def test_program_candidate_valid_preserves_source_tier_unmodified():
    for tier in (SourceTier.TIER_1, SourceTier.TIER_2, SourceTier.UNKNOWN):
        evidence = ProgramCandidateEvidence(
            nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_1",
            source_tier=tier, retrieved_at=RETRIEVED_AT,
        )
        assert validate_program_candidate_evidence(evidence).outcome == EvidenceValidationOutcome.VALID
        assert evidence.source_tier is tier  # never rewritten by validation


# =============================================================================
# LiteratureCandidateEvidence validation -- strict PMID/NCT, tier preserved
# =============================================================================
def test_literature_candidate_invalid_malformed_pmid():
    evidence = LiteratureCandidateEvidence(pmid="not-a-pmid", source_id="src_1", retrieved_at=RETRIEVED_AT)
    result = validate_literature_candidate_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("pmid" in r for r in result.reasons)


def test_literature_candidate_invalid_malformed_embedded_nct():
    evidence = LiteratureCandidateEvidence(
        pmid="33378609", nct_ids=("NCT-BAD",), source_id="src_1", retrieved_at=RETRIEVED_AT,
    )
    assert validate_literature_candidate_evidence(evidence).outcome == EvidenceValidationOutcome.INVALID


def test_literature_candidate_valid_with_unknown_tier_never_upgraded():
    """Phase 4.3A-correction-1's finding: every real literature Source this
    repository produces today carries SourceTier.UNKNOWN. This must remain
    a VALID, unmodified outcome -- never rejected, never silently
    promoted."""
    evidence = LiteratureCandidateEvidence(
        pmid="33378609", nct_ids=("NCT12345678",), source_id="src_lit_1",
        source_tier=SourceTier.UNKNOWN, retrieved_at=RETRIEVED_AT,
    )
    result = validate_literature_candidate_evidence(evidence)
    assert result.outcome == EvidenceValidationOutcome.VALID
    assert evidence.source_tier is SourceTier.UNKNOWN


def test_literature_candidate_incomplete_missing_retrieved_at():
    evidence = LiteratureCandidateEvidence(pmid="33378609", source_id="src_1")
    assert validate_literature_candidate_evidence(evidence).outcome == EvidenceValidationOutcome.INCOMPLETE


# =============================================================================
# build_program_candidate_evidence_from_parsed_study (Phase 4.3B req. 7)
# =============================================================================
def _real_shaped_parsed_study() -> dict:
    """Mirrors collectors.clinicaltrials.parse_study()'s exact output
    dict shape -- not imported (this test stays independent of that
    collector), but keeps the same keys deliberately."""
    return {
        "nct_id": "NCT12345678",
        "title": "A fictional demo study",
        "sponsor": "Demo Biotherapeutics, Inc.",
        "collaborators": ["Demo University", ""],
        "status": "RECRUITING",
        "phases": ["PHASE2"],
        "phase": "PHASE2",
        "interventions": ["Demo Compound X", ""],
        "conditions": ["Demo Fictional Indication"],
        "primary_completion": "2026-06-30",
        "primary_completion_type": "ESTIMATED",
        "completion_date": "2026-12-31",
        "completion_date_type": "ESTIMATED",
        "study_first_submit_date": "2024-01-15",
        "has_results": False,
    }


def test_build_from_parsed_study_reads_structured_fields_only():
    evidence = build_program_candidate_evidence_from_parsed_study(
        _real_shaped_parsed_study(),
        source_id="src_ct_1", source_tier=SourceTier.TIER_1, retrieved_at=RETRIEVED_AT,
    )
    assert evidence.nct_id == "NCT12345678"
    assert evidence.lead_sponsor == "Demo Biotherapeutics, Inc."
    assert evidence.collaborators == ("Demo University",)  # empty string dropped, never fabricated
    assert evidence.interventions == ("Demo Compound X",)
    assert evidence.conditions == ("Demo Fictional Indication",)
    assert evidence.overall_status == "RECRUITING"
    assert evidence.phases == ("PHASE2",)
    assert evidence.primary_completion_date == "2026-06-30"
    assert evidence.completion_date == "2026-12-31"
    assert evidence.source_id == "src_ct_1"
    assert evidence.source_tier is SourceTier.TIER_1
    assert evidence.retrieved_at == RETRIEVED_AT


def test_build_from_parsed_study_never_populates_first_posted_date():
    """parse_study() does not read studyFirstPostDateStruct today -- this
    constructor must never repurpose study_first_submit_date (a DIFFERENT
    date) for it."""
    evidence = build_program_candidate_evidence_from_parsed_study(
        _real_shaped_parsed_study(),
        source_id="src_ct_1", source_tier=SourceTier.TIER_1, retrieved_at=RETRIEVED_AT,
    )
    assert evidence.first_posted_date == "UNKNOWN"


def test_build_from_parsed_study_carries_supporting_fact_ids_verbatim():
    evidence = build_program_candidate_evidence_from_parsed_study(
        _real_shaped_parsed_study(),
        source_id="src_ct_1", source_tier=SourceTier.TIER_1, retrieved_at=RETRIEVED_AT,
        supporting_fact_ids=[REAL_FACT_ID],
    )
    assert evidence.supporting_fact_ids == (REAL_FACT_ID,)


def test_build_from_parsed_study_missing_optional_fields_stays_unknown():
    sparse = {"nct_id": "NCT99999999"}
    evidence = build_program_candidate_evidence_from_parsed_study(
        sparse, source_id="src_ct_2", source_tier=SourceTier.TIER_1, retrieved_at=RETRIEVED_AT,
    )
    assert evidence.lead_sponsor == "UNKNOWN"
    assert evidence.collaborators == ()
    assert evidence.conditions == ()
