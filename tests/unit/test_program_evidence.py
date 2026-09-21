"""Phase 4.3B / Phase 4.3B Correction 1: immutable structured evidence
types and their FORMAT + REFERENTIAL validation (``scoring/
program_evidence.py``).

Correction 1's root fix: Phase 4.3B only checked that ``supporting_fact_ids``/
``source_id`` were the right SHAPE. This file's positive tests build REAL
``Source``/``Fact`` objects and prove VALID is reached only when the
evidence record's own claimed ``source_tier``/``content_hash``/
``retrieved_at`` genuinely match that real record -- and its negative
tests prove a format-valid-but-fabricated/mismatched/absent reference is
rejected, never silently accepted.
"""

from __future__ import annotations

import dataclasses

import pytest

from investment_research.schemas.enums import (
    EvidenceClass,
    FactCategory,
    Materiality,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Fact, Source
from investment_research.scoring.program_evidence import (
    EMPTY_VALIDATION_CONTEXT,
    CompanyIdentityEvidence,
    EvidenceValidationContext,
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
OTHER_FACT_ID = "fact_" + "b" * 20


def _source(
    source_id="src_1", tier=SourceTier.TIER_1, content_hash="hash_1", retrieved_at=RETRIEVED_AT,
    url="https://example.test/demo", title="Demo Source",
) -> Source:
    return Source(
        source_id=source_id, url=url, title=title, tier=tier,
        retrieved_at=retrieved_at, content_hash=content_hash,
    )


def _fact(fact_id=REAL_FACT_ID, source_id="src_1") -> Fact:
    return Fact(
        fact_id=fact_id, ticker="DEMOBIO", category=FactCategory.CLINICAL,
        claim="demo claim", evidence_class=EvidenceClass.VERIFIED_FACT,
        source_id=source_id, source_url="https://example.test/demo",
        source_title="Demo Source", source_tier=SourceTier.TIER_1,
        verified_status=VerifiedStatus.VERIFIED, materiality=Materiality.MEDIUM,
    )


def _context(sources=(), facts=()) -> EvidenceValidationContext:
    return EvidenceValidationContext(
        sources_by_id={s.source_id: s for s in sources},
        verified_facts_by_id={f.fact_id: f for f in facts},
    )


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


def test_validation_context_is_frozen_and_immutable():
    ctx = _context(sources=[_source()])
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.sources_by_id = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        ctx.sources_by_id["new"] = _source(source_id="new")  # type: ignore[index]


def test_validation_context_copies_input_never_shares_the_caller_mutable_dict():
    mutable = {"src_1": _source()}
    ctx = EvidenceValidationContext(sources_by_id=mutable)
    mutable["src_2"] = _source(source_id="src_2")  # mutate the ORIGINAL dict after construction
    assert "src_2" not in ctx.sources_by_id  # context is unaffected


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


# =============================================================================
# supporting_fact_ids -- format vs. referential (Phase 4.3B Correction 1,
# requirement 3)
# =============================================================================
def test_positive_real_fact_backing_the_same_source_is_valid():
    source = _source()
    fact = _fact(source_id=source.source_id)
    context = _context(sources=[source], facts=[fact])
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id=source.source_id,
        source_tier=source.tier, retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        supporting_fact_ids=(REAL_FACT_ID,),
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.VALID


def test_negative_format_valid_but_nonexistent_fact_id_is_invalid():
    source = _source()
    context = _context(sources=[source])  # no facts at all
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id=source.source_id,
        source_tier=source.tier, retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        supporting_fact_ids=(REAL_FACT_ID,),
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("does not exist" in r for r in result.reasons)


def test_negative_invented_fact_id_shape_is_still_rejected_before_referential_check():
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_1",
        retrieved_at=RETRIEVED_AT, supporting_fact_ids=("fact_totally_made_up",),
    )
    result = validate_program_candidate_evidence(evidence, EMPTY_VALIDATION_CONTEXT)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("supporting_fact_ids" in r for r in result.reasons)


def test_negative_fact_source_id_mismatch_is_invalid():
    """The Fact exists, but is attributed to a DIFFERENT source than the
    candidate claims -- must never be accepted as support."""
    source_a = _source(source_id="src_a")
    source_b = _source(source_id="src_b")
    fact = _fact(source_id="src_b")  # backs source_b, not source_a
    context = _context(sources=[source_a, source_b], facts=[fact])
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_a",
        source_tier=source_a.tier, retrieved_at=source_a.retrieved_at,
        content_hash=source_a.content_hash, supporting_fact_ids=(REAL_FACT_ID,),
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("does not match" in r and "source_id" in r for r in result.reasons)


def test_negative_fact_resting_on_a_quarantined_source_is_invalid():
    """The Fact's OWN source_id is not present in the context (simulating
    quarantine/exclusion) -- the fact can never support a candidate."""
    candidate_source = _source(source_id="src_candidate")
    fact = _fact(source_id="src_quarantined")  # this source is NOT in the context
    context = _context(sources=[candidate_source], facts=[fact])  # src_quarantined absent
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_candidate",
        source_tier=candidate_source.tier, retrieved_at=candidate_source.retrieved_at,
        content_hash=candidate_source.content_hash, supporting_fact_ids=(REAL_FACT_ID,),
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("not present in the validation context" in r for r in result.reasons)


def test_no_supporting_facts_is_allowed_and_marked_metadata_only():
    """Phase 4.3B Correction 1's own decision (requirement 3): empty
    supporting_fact_ids is ALLOWED -- a structured-Source-only metadata
    record, never reported as 'confirmed by a Fact'."""
    source = _source()
    context = _context(sources=[source])
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id=source.source_id,
        source_tier=source.tier, retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        supporting_fact_ids=(),
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.VALID
    assert any("structured Source metadata only" in r for r in result.reasons)


def test_multiple_supporting_facts_all_checked_one_bad_one_fails_the_whole_record():
    source = _source()
    good_fact = _fact(fact_id=REAL_FACT_ID, source_id=source.source_id)
    context = _context(sources=[source], facts=[good_fact])  # OTHER_FACT_ID never added
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id=source.source_id,
        source_tier=source.tier, retrieved_at=source.retrieved_at, content_hash=source.content_hash,
        supporting_fact_ids=(REAL_FACT_ID, OTHER_FACT_ID),
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID


# =============================================================================
# Source referential integrity -- CompanyIdentityEvidence
# =============================================================================
def test_positive_company_identity_valid_when_source_matches_exactly():
    source = _source(source_id="src_sec_1", tier=SourceTier.TIER_1, content_hash="sechash")
    context = _context(sources=[source])
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=123456, sec_official_name="Demo Biotherapeutics Inc",
        source_id="src_sec_1", source_tier=source.tier, retrieved_at=source.retrieved_at,
        content_hash=source.content_hash,
    )
    result = validate_company_identity_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.VALID


def test_negative_company_identity_source_not_in_context_is_incomplete():
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=123456, sec_official_name="Demo Biotherapeutics Inc",
        source_id="src_sec_1", retrieved_at=RETRIEVED_AT,
    )
    result = validate_company_identity_evidence(evidence, EMPTY_VALIDATION_CONTEXT)
    assert result.outcome == EvidenceValidationOutcome.INCOMPLETE
    assert any("not found in the validation context" in r for r in result.reasons)


def test_negative_company_identity_tier_mismatch_is_invalid():
    source = _source(source_id="src_sec_1", tier=SourceTier.TIER_1)
    context = _context(sources=[source])
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=123456, sec_official_name="Demo Biotherapeutics Inc",
        source_id="src_sec_1", source_tier=SourceTier.TIER_2,  # claims TIER_2, real Source is TIER_1
        retrieved_at=source.retrieved_at, content_hash=source.content_hash,
    )
    result = validate_company_identity_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("source_tier" in r for r in result.reasons)


def test_negative_company_identity_content_hash_mismatch_is_invalid():
    source = _source(source_id="src_sec_1", content_hash="real_hash")
    context = _context(sources=[source])
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=123456, sec_official_name="Demo Biotherapeutics Inc",
        source_id="src_sec_1", source_tier=source.tier, retrieved_at=source.retrieved_at,
        content_hash="wrong_hash",
    )
    result = validate_company_identity_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("content_hash" in r for r in result.reasons)


def test_negative_company_identity_retrieved_at_mismatch_is_invalid():
    source = _source(source_id="src_sec_1", retrieved_at="2026-01-01T00:00:00+00:00")
    context = _context(sources=[source])
    evidence = CompanyIdentityEvidence(
        ticker="DEMOBIO", cik=123456, sec_official_name="Demo Biotherapeutics Inc",
        source_id="src_sec_1", source_tier=source.tier, content_hash=source.content_hash,
        retrieved_at="2030-01-01T00:00:00+00:00",
    )
    result = validate_company_identity_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("retrieved_at" in r for r in result.reasons)


# =============================================================================
# Source referential integrity -- ProgramCandidateEvidence
# =============================================================================
def test_positive_program_candidate_valid_when_source_matches_exactly():
    source = _source(source_id="src_ct_1")
    context = _context(sources=[source])
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_ct_1",
        source_tier=source.tier, retrieved_at=source.retrieved_at, content_hash=source.content_hash,
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.VALID


def test_negative_program_candidate_source_not_in_context_is_incomplete():
    """Simulates a quarantined/excluded Source -- the caller contract says
    sources_by_id must never include a quarantined Source, so its absence
    here is exactly the correct signal."""
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_ct_1",
        retrieved_at=RETRIEVED_AT,
    )
    result = validate_program_candidate_evidence(evidence, EMPTY_VALIDATION_CONTEXT)
    assert result.outcome == EvidenceValidationOutcome.INCOMPLETE


def test_negative_program_candidate_tier_mismatch_is_invalid():
    source = _source(source_id="src_ct_1", tier=SourceTier.TIER_1)
    context = _context(sources=[source])
    evidence = ProgramCandidateEvidence(
        nct_id="NCT12345678", lead_sponsor="Demo Inc", source_id="src_ct_1",
        source_tier=SourceTier.UNKNOWN, retrieved_at=source.retrieved_at,
        content_hash=source.content_hash,
    )
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID


def test_negative_program_candidate_malformed_nct_never_reaches_referential_check():
    evidence = ProgramCandidateEvidence(nct_id="NCT123", lead_sponsor="Demo Inc", source_id="src_1")
    result = validate_program_candidate_evidence(evidence, EMPTY_VALIDATION_CONTEXT)
    assert result.outcome == EvidenceValidationOutcome.INVALID
    assert any("nct_id" in r for r in result.reasons)


# =============================================================================
# Source referential integrity -- LiteratureCandidateEvidence, tier UNKNOWN
# =============================================================================
def test_positive_literature_candidate_valid_when_source_matches_exactly_tier_unknown():
    """Phase 4.3A-correction-1's finding: every real literature Source
    carries SourceTier.UNKNOWN. UNKNOWN == UNKNOWN compares equal, never
    special-cased, never upgraded."""
    source = _source(source_id="src_lit_1", tier=SourceTier.UNKNOWN)
    context = _context(sources=[source])
    evidence = LiteratureCandidateEvidence(
        pmid="90000099", nct_ids=("NCT12345678",), source_id="src_lit_1",
        source_tier=SourceTier.UNKNOWN, retrieved_at=source.retrieved_at,
        content_hash=source.content_hash,
    )
    result = validate_literature_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.VALID
    assert evidence.source_tier is SourceTier.UNKNOWN  # never upgraded


def test_negative_literature_candidate_claims_tier_never_actually_upgraded_by_evidence():
    """A record that LIES about its tier (claims TIER_1 while the real
    Source is UNKNOWN) is rejected -- this is the exact mechanism that
    prevents a literature record from silently claiming a higher tier
    than its Source actually has."""
    source = _source(source_id="src_lit_1", tier=SourceTier.UNKNOWN)
    context = _context(sources=[source])
    evidence = LiteratureCandidateEvidence(
        pmid="90000099", nct_ids=("NCT12345678",), source_id="src_lit_1",
        source_tier=SourceTier.TIER_1,  # claims TIER_1, real Source is UNKNOWN
        retrieved_at=source.retrieved_at, content_hash=source.content_hash,
    )
    result = validate_literature_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.INVALID


def test_negative_literature_candidate_source_not_in_context_is_incomplete():
    evidence = LiteratureCandidateEvidence(pmid="90000099", source_id="src_lit_1")
    result = validate_literature_candidate_evidence(evidence, EMPTY_VALIDATION_CONTEXT)
    assert result.outcome == EvidenceValidationOutcome.INCOMPLETE


def test_negative_literature_candidate_malformed_pmid_never_reaches_referential_check():
    evidence = LiteratureCandidateEvidence(pmid="not-a-pmid", source_id="src_1")
    result = validate_literature_candidate_evidence(evidence, EMPTY_VALIDATION_CONTEXT)
    assert result.outcome == EvidenceValidationOutcome.INVALID


# =============================================================================
# build_program_candidate_evidence_from_parsed_study (Phase 4.3B req. 7)
# =============================================================================
def _real_shaped_parsed_study() -> dict:
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
    assert evidence.collaborators == ("Demo University",)
    assert evidence.source_id == "src_ct_1"
    assert evidence.source_tier is SourceTier.TIER_1
    assert evidence.retrieved_at == RETRIEVED_AT


def test_build_from_parsed_study_output_passes_referential_validation_when_source_matches():
    """The pure constructor's contract (module docstring): a caller who
    passes the SAME Source's own tier/retrieved_at/content_hash gets a
    record that validates cleanly by construction."""
    source = _source(source_id="src_ct_1", content_hash="realhash")
    evidence = build_program_candidate_evidence_from_parsed_study(
        _real_shaped_parsed_study(),
        source_id=source.source_id, source_tier=source.tier, retrieved_at=source.retrieved_at,
        content_hash=source.content_hash,
    )
    context = _context(sources=[source])
    result = validate_program_candidate_evidence(evidence, context)
    assert result.outcome == EvidenceValidationOutcome.VALID


def test_build_from_parsed_study_never_populates_first_posted_date():
    evidence = build_program_candidate_evidence_from_parsed_study(
        _real_shaped_parsed_study(),
        source_id="src_ct_1", source_tier=SourceTier.TIER_1, retrieved_at=RETRIEVED_AT,
    )
    assert evidence.first_posted_date == "UNKNOWN"


def test_build_from_parsed_study_missing_optional_fields_stays_unknown():
    sparse = {"nct_id": "NCT99999999"}
    evidence = build_program_candidate_evidence_from_parsed_study(
        sparse, source_id="src_ct_2", source_tier=SourceTier.TIER_1, retrieved_at=RETRIEVED_AT,
    )
    assert evidence.lead_sponsor == "UNKNOWN"
    assert evidence.collaborators == ()
