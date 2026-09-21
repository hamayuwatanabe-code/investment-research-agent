"""Phase 4.3B / Phase 4.3B Correction 1: pure deterministic Program
Identity / Literature Link resolvers (``scoring/program_identity_resolution.py``).

Every test here drives ``resolve_program_identity``/``resolve_literature_
link`` directly with synthetic, offline evidence PLUS a real
``EvidenceValidationContext`` built from real ``Source``/``Fact`` objects
-- no Pipeline, no HTTP, no LLM. Positive cases prove the resolvers reach
CONFIRMED/LINKED only when every reference genuinely resolves; negative/
ambiguous cases prove a long list of plausible-looking-but-insufficient
signals never wrongly produce CONFIRMED/LINKED, including the Correction-1
referential-integrity gaps (fabricated fact ids, mismatched Source fields,
unverified aliases, mixed valid/invalid evidence for the same identifier).
"""

from __future__ import annotations

from investment_research.schemas.enums import (
    EvidenceClass,
    FactCategory,
    Materiality,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Fact, Source
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
    EvidenceValidationContext,
    LiteratureCandidateEvidence,
    ProgramCandidateEvidence,
)
from investment_research.scoring.program_identity_resolution import (
    LiteratureLinkStatus,
    ProgramIdentityStatus,
    resolve_literature_link,
    resolve_program_identity,
)

RETRIEVED_AT = "2026-01-01T00:00:00+00:00"
NCT_A = "NCT12345678"
NCT_B = "NCT87654321"
SEC_HASH = "sec_hash_1"
CT_HASH = "ct_hash_1"
LIT_HASH = "lit_hash_1"
FACT_ID = "fact_" + "a" * 20


def _sec_source(**overrides) -> Source:
    kwargs = {
        "source_id": "src_sec_1", "url": "https://www.sec.gov/files/company_tickers.json",
        "title": "SEC ticker map", "tier": SourceTier.TIER_1,
        "retrieved_at": RETRIEVED_AT, "content_hash": SEC_HASH,
    }
    kwargs.update(overrides)
    return Source(**kwargs)


def _ct_source(**overrides) -> Source:
    kwargs = {
        "source_id": "src_ct_1", "url": "https://clinicaltrials.gov/study/NCT12345678",
        "title": "Demo study", "tier": SourceTier.TIER_1,
        "retrieved_at": RETRIEVED_AT, "content_hash": CT_HASH,
    }
    kwargs.update(overrides)
    return Source(**kwargs)


def _lit_source(**overrides) -> Source:
    kwargs = {
        "source_id": "src_lit_1", "url": "https://pubmed.ncbi.nlm.nih.gov/90000099",
        "title": "Demo PubMed article", "tier": SourceTier.UNKNOWN,
        "retrieved_at": RETRIEVED_AT, "content_hash": LIT_HASH,
    }
    kwargs.update(overrides)
    return Source(**kwargs)


def _fact(**overrides) -> Fact:
    kwargs = {
        "fact_id": FACT_ID, "ticker": "DEMOBIO", "category": FactCategory.CLINICAL,
        "claim": "demo claim", "evidence_class": EvidenceClass.VERIFIED_FACT,
        "source_id": "src_ct_1", "source_url": "https://clinicaltrials.gov/study/NCT12345678",
        "source_title": "Demo study", "source_tier": SourceTier.TIER_1,
        "verified_status": VerifiedStatus.VERIFIED, "materiality": Materiality.MEDIUM,
    }
    kwargs.update(overrides)
    return Fact(**kwargs)


def _context(sources=(), facts=()) -> EvidenceValidationContext:
    return EvidenceValidationContext(
        sources_by_id={s.source_id: s for s in sources},
        verified_facts_by_id={f.fact_id: f for f in facts},
    )


def _company(**overrides) -> CompanyIdentityEvidence:
    kwargs = {
        "ticker": "DEMOBIO", "cik": 1234567, "sec_official_name": "Demo Biotherapeutics Inc",
        "source_id": "src_sec_1", "source_tier": SourceTier.TIER_1,
        "retrieved_at": RETRIEVED_AT, "content_hash": SEC_HASH,
    }
    kwargs.update(overrides)
    return CompanyIdentityEvidence(**kwargs)


def _candidate(**overrides) -> ProgramCandidateEvidence:
    kwargs = {
        "nct_id": NCT_A, "lead_sponsor": "Demo Biotherapeutics, Inc.",
        "source_id": "src_ct_1", "source_tier": SourceTier.TIER_1,
        "retrieved_at": RETRIEVED_AT, "content_hash": CT_HASH,
    }
    kwargs.update(overrides)
    return ProgramCandidateEvidence(**kwargs)


def _literature(**overrides) -> LiteratureCandidateEvidence:
    kwargs = {
        "pmid": "90000099", "nct_ids": (NCT_A,), "source_id": "src_lit_1",
        "source_tier": SourceTier.UNKNOWN, "retrieved_at": RETRIEVED_AT, "content_hash": LIT_HASH,
    }
    kwargs.update(overrides)
    return LiteratureCandidateEvidence(**kwargs)


#: The default, fully-consistent context every "happy path" test uses --
#: real Source objects whose own tier/hash/retrieved_at match what
#: _company()/_candidate()/_literature() claim by default.
DEFAULT_CONTEXT = _context(sources=[_sec_source(), _ct_source(), _lit_source()])


# =============================================================================
# Positive: Program Identity
# =============================================================================
def test_positive_sec_name_matches_lead_sponsor_single_candidate_confirms():
    resolution = resolve_program_identity(_company(), [_candidate()], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_A
    assert resolution.evaluated_candidate_count == 1
    assert resolution.excluded_evidence_reasons == ()


def test_positive_pmid_never_required_for_program_identity_confirmation():
    resolution = resolve_program_identity(_company(), [_candidate()], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    # resolve_program_identity's own signature has no literature parameter.


def test_positive_old_trial_same_company_still_confirms():
    candidate = _candidate(primary_completion_date="1999-01-01", completion_date="1999-06-01")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.CONFIRMED


def test_positive_explicit_lead_marker_breaks_a_multi_candidate_tie():
    ct_source_b = _ct_source(source_id="src_ct_2", url="https://clinicaltrials.gov/study/NCT87654321")
    context = _context(sources=[_sec_source(), _ct_source(), ct_source_b])
    candidates = [_candidate(nct_id=NCT_A), _candidate(nct_id=NCT_B, source_id="src_ct_2")]
    resolution = resolve_program_identity(
        _company(), candidates, context, explicit_lead_trial_ids=frozenset({NCT_B}),
    )
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_B


# =============================================================================
# Positive: Literature Link
# =============================================================================
def test_positive_nct_explicit_in_pubmed_metadata_links():
    resolution = resolve_literature_link(NCT_A, [_literature()], DEFAULT_CONTEXT)
    assert resolution.status is LiteratureLinkStatus.LINKED
    assert resolution.pmids == ("90000099",)


def test_positive_multiple_genuinely_matching_pmids_all_returned():
    lit_source_2 = _lit_source(source_id="src_lit_2", url="https://pubmed.ncbi.nlm.nih.gov/22222222")
    context = _context(sources=[_sec_source(), _ct_source(), _lit_source(), lit_source_2])
    candidates = [
        _literature(pmid="11111111"),
        _literature(pmid="22222222", source_id="src_lit_2"),
    ]
    resolution = resolve_literature_link(NCT_A, candidates, context)
    assert resolution.status is LiteratureLinkStatus.LINKED
    assert resolution.pmids == ("11111111", "22222222")


def test_positive_literature_source_tier_unknown_never_blocks_a_genuine_link():
    lit = _literature(source_tier=SourceTier.UNKNOWN)
    resolution = resolve_literature_link(NCT_A, [lit], DEFAULT_CONTEXT)
    assert resolution.status is LiteratureLinkStatus.LINKED
    assert lit.source_tier is SourceTier.UNKNOWN


# =============================================================================
# Negative/ambiguous: Program Identity -- must never wrongly CONFIRM
# =============================================================================
def test_negative_malformed_nct_never_confirms():
    resolution = resolve_program_identity(_company(), [_candidate(nct_id="NCT123")], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_sponsor_mismatch_never_confirms():
    resolution = resolve_program_identity(
        _company(), [_candidate(lead_sponsor="Totally Unrelated Corp")], DEFAULT_CONTEXT,
    )
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_collaborator_only_match_never_confirms():
    candidate = _candidate(
        lead_sponsor="Some Other University", collaborators=("Demo Biotherapeutics, Inc.",),
    )
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_same_drug_name_different_company_never_confirms():
    candidate = _candidate(
        lead_sponsor="A Completely Different Sponsor Inc", interventions=("Demo Compound X",),
    )
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_different_indication_alone_never_confirms_without_sponsor_match():
    candidate = _candidate(
        lead_sponsor="Unrelated Sponsor Inc", conditions=("A Completely Different Indication",),
    )
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_multiple_sponsor_matched_candidates_without_lead_marker_unresolved():
    ct_source_b = _ct_source(source_id="src_ct_2", url="https://clinicaltrials.gov/study/NCT87654321")
    context = _context(sources=[_sec_source(), _ct_source(), ct_source_b])
    candidates = [_candidate(nct_id=NCT_A), _candidate(nct_id=NCT_B, source_id="src_ct_2")]
    resolution = resolve_program_identity(_company(), candidates, context)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED
    assert "recency" in resolution.rationale.lower() or "arbitrary" in resolution.rationale.lower()


def test_negative_unconfirmed_acquisition_relationship_never_confirms():
    candidate = _candidate(lead_sponsor="Acquired Target Biotech Inc")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_unconfirmed_license_out_relationship_never_confirms():
    candidate = _candidate(lead_sponsor="Licensee Pharma Corp")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_recency_alone_never_substitutes_for_identity_confirmation():
    candidate = _candidate(lead_sponsor="Unrelated Sponsor Inc", primary_completion_date="2030-01-01")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_company_identity_incomplete_never_confirms():
    company = _company(sec_official_name="UNKNOWN")
    resolution = resolve_program_identity(company, [_candidate()], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_no_candidates_never_confirms():
    resolution = resolve_program_identity(_company(), [], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


# =============================================================================
# Negative: Correction 1's own referential-integrity gaps
# =============================================================================
def test_negative_company_source_not_in_context_never_confirms():
    resolution = resolve_program_identity(
        _company(), [_candidate()], _context(sources=[_ct_source()]),  # no SEC source at all
    )
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_candidate_source_not_in_context_never_confirms():
    resolution = resolve_program_identity(
        _company(), [_candidate()], _context(sources=[_sec_source()]),  # no CT source at all
    )
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_candidate_tier_mismatch_never_confirms():
    candidate = _candidate(source_tier=SourceTier.TIER_2)  # real CT source is TIER_1
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_candidate_content_hash_mismatch_never_confirms():
    candidate = _candidate(content_hash="wrong_hash")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_candidate_retrieved_at_mismatch_never_confirms():
    candidate = _candidate(retrieved_at="2099-01-01T00:00:00+00:00")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_fabricated_supporting_fact_id_never_confirms():
    candidate = _candidate(supporting_fact_ids=(FACT_ID,))  # FACT_ID never added to context
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED
    assert any("does not exist" in r for r in resolution.excluded_evidence_reasons)


def test_negative_fact_source_id_mismatch_never_confirms():
    other_source = _ct_source(source_id="src_ct_other", url="https://clinicaltrials.gov/study/other")
    mismatched_fact = _fact(source_id="src_ct_other")  # not src_ct_1
    context = _context(sources=[_sec_source(), _ct_source(), other_source], facts=[mismatched_fact])
    candidate = _candidate(supporting_fact_ids=(FACT_ID,))
    resolution = resolve_program_identity(_company(), [candidate], context)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_fact_on_quarantined_source_never_confirms():
    fact_on_missing_source = _fact(source_id="src_quarantined")
    context = _context(sources=[_sec_source(), _ct_source()], facts=[fact_on_missing_source])
    candidate = _candidate(supporting_fact_ids=(FACT_ID,))
    resolution = resolve_program_identity(_company(), [candidate], context)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


# =============================================================================
# Negative: alias handling (Correction 1, requirement 4 -- option B)
# =============================================================================
def test_negative_unverified_alias_never_confirms():
    """explicitly_verified_aliases is not read at all in this phase --
    even a matching alias never confirms."""
    company = _company(sec_official_name="Demo Bio Holdings Inc", explicitly_verified_aliases=("Demo Biotherapeutics, Inc.",))
    resolution = resolve_program_identity(company, [_candidate()], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_fixture_style_alias_never_confirms():
    """A caller passing fixture-metadata-shaped aliases gets the same
    UNRESOLVED outcome -- there is no code path where aliases matter."""
    company = _company(sec_official_name="Something Else Entirely Inc", explicitly_verified_aliases=("Demo Biotherapeutics, Inc.", "Demo Bio", "DEMOBIO"))
    resolution = resolve_program_identity(company, [_candidate()], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_sec_former_name_alone_without_explicit_verification_never_confirms():
    candidate = _candidate(lead_sponsor="Old Co Name Inc")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


# =============================================================================
# Negative: explicit_lead_trial_ids validation (Correction 1, requirement 5)
# =============================================================================
def test_negative_nonexistent_explicit_lead_nct_never_confirms():
    ct_source_b = _ct_source(source_id="src_ct_2", url="https://clinicaltrials.gov/study/NCT87654321")
    context = _context(sources=[_sec_source(), _ct_source(), ct_source_b])
    candidates = [_candidate(nct_id=NCT_A), _candidate(nct_id=NCT_B, source_id="src_ct_2")]
    resolution = resolve_program_identity(
        _company(), candidates, context, explicit_lead_trial_ids=frozenset({"NCT00000099"}),
    )
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_malformed_explicit_lead_nct_never_confirms():
    ct_source_b = _ct_source(source_id="src_ct_2", url="https://clinicaltrials.gov/study/NCT87654321")
    context = _context(sources=[_sec_source(), _ct_source(), ct_source_b])
    candidates = [_candidate(nct_id=NCT_A), _candidate(nct_id=NCT_B, source_id="src_ct_2")]
    resolution = resolve_program_identity(
        _company(), candidates, context, explicit_lead_trial_ids=frozenset({"NCT-BOGUS"}),
    )
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_explicit_lead_nct_for_a_different_non_sponsor_matched_candidate_never_wins():
    """Two candidates share the SEC-matched sponsor (so both are eligible
    lead-marker targets); the explicit lead id names one of them -- the
    OTHER, unmarked one must never be selected merely because it exists."""
    ct_source_b = _ct_source(source_id="src_ct_2", url="https://clinicaltrials.gov/study/NCT87654321")
    context = _context(sources=[_sec_source(), _ct_source(), ct_source_b])
    candidates = [
        _candidate(nct_id=NCT_A),
        _candidate(nct_id=NCT_B, source_id="src_ct_2"),  # same matching sponsor
    ]
    resolution = resolve_program_identity(
        _company(), candidates, context, explicit_lead_trial_ids=frozenset({NCT_B}),
    )
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_B  # the marked one, never NCT_A


def test_negative_explicit_lead_nct_naming_a_non_matched_sponsor_candidate_is_irrelevant():
    """An explicit lead id naming a candidate whose sponsor never matched
    at all has no effect -- the genuinely single sponsor-matched candidate
    still confirms on its own, and the unrelated trial is never selected."""
    ct_source_b = _ct_source(source_id="src_ct_2", url="https://clinicaltrials.gov/study/NCT87654321")
    context = _context(sources=[_sec_source(), _ct_source(), ct_source_b])
    candidates = [
        _candidate(nct_id=NCT_A),
        _candidate(nct_id=NCT_B, source_id="src_ct_2", lead_sponsor="Totally Different Sponsor Inc"),
    ]
    resolution = resolve_program_identity(
        _company(), candidates, context, explicit_lead_trial_ids=frozenset({NCT_B}),
    )
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_A  # never NCT_B, whose sponsor never matched


# =============================================================================
# CONFLICTED -- only an explicit, evidence-internal contradiction
# =============================================================================
def test_conflicted_same_nct_contradictory_sponsors():
    candidates = [
        _candidate(nct_id=NCT_A, lead_sponsor="Demo Biotherapeutics, Inc."),
        _candidate(nct_id=NCT_A, lead_sponsor="A Totally Different Sponsor Corp"),
    ]
    resolution = resolve_program_identity(_company(), candidates, DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.CONFLICTED
    assert NCT_A in resolution.conflicting_nct_ids


def test_pre_listing_trial_alone_is_never_conflicted():
    candidate = _candidate(primary_completion_date="1999-01-01")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status != ProgramIdentityStatus.CONFLICTED


def test_collaborator_relationship_alone_is_never_conflicted():
    candidate = _candidate(collaborators=("Some Collaborator Inc",))
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status != ProgramIdentityStatus.CONFLICTED


def test_former_name_alone_is_never_conflicted():
    candidate = _candidate(lead_sponsor="Old Co Name Inc")
    resolution = resolve_program_identity(_company(), [candidate], DEFAULT_CONTEXT)
    assert resolution.status != ProgramIdentityStatus.CONFLICTED


# =============================================================================
# Mixed VALID/INVALID evidence for the SAME identifier (Correction 1,
# requirement 6/7)
# =============================================================================
def test_negative_valid_and_invalid_evidence_for_same_nct_never_confirms():
    """Two candidates for the SAME nct_id: one VALID (matches the real
    Source exactly), one INVALID (tier mismatch). Must not silently
    confirm using only the valid one."""
    good = _candidate(nct_id=NCT_A)
    bad = _candidate(nct_id=NCT_A, source_tier=SourceTier.TIER_2)  # invalid: tier mismatch
    resolution = resolve_program_identity(_company(), [good, bad], DEFAULT_CONTEXT)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED
    assert any("excluded" in r for r in resolution.excluded_evidence_reasons)
    assert any("also have a VALID candidate record" in r for r in resolution.excluded_evidence_reasons)


def test_negative_invalid_candidate_for_an_unrelated_nct_does_not_block_a_clean_confirmation():
    """An invalid record for a DIFFERENT, unrelated nct_id must not taint
    a genuinely clean confirmation for a different identifier."""
    good = _candidate(nct_id=NCT_A)
    unrelated_ct_source = _ct_source(source_id="src_ct_bad", url="https://clinicaltrials.gov/study/NCT87654321")
    context = _context(sources=[_sec_source(), _ct_source(), unrelated_ct_source])
    bad = _candidate(nct_id=NCT_B, source_id="src_ct_bad", source_tier=SourceTier.TIER_2)
    resolution = resolve_program_identity(_company(), [good, bad], context)
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_A
    assert any("excluded" in r for r in resolution.excluded_evidence_reasons)  # still recorded


# =============================================================================
# Negative/ambiguous: Literature Link -- must never wrongly LINK
# =============================================================================
def test_negative_malformed_pmid_never_links():
    resolution = resolve_literature_link(NCT_A, [_literature(pmid="not-a-pmid")], DEFAULT_CONTEXT)
    assert resolution.status is not LiteratureLinkStatus.LINKED


def test_negative_pmid_without_direct_nct_reference_not_found():
    resolution = resolve_literature_link(NCT_A, [_literature(nct_ids=())], DEFAULT_CONTEXT)
    assert resolution.status is LiteratureLinkStatus.NOT_FOUND


def test_negative_multiple_pmids_none_referencing_target_not_found():
    lit_source_2 = _lit_source(source_id="src_lit_2", url="https://pubmed.ncbi.nlm.nih.gov/22222222")
    context = _context(sources=[_sec_source(), _ct_source(), _lit_source(), lit_source_2])
    candidates = [
        _literature(pmid="11111111", nct_ids=()),
        _literature(pmid="22222222", nct_ids=(NCT_B,), source_id="src_lit_2"),
    ]
    resolution = resolve_literature_link(NCT_A, candidates, context)
    assert resolution.status is LiteratureLinkStatus.NOT_FOUND
    assert resolution.pmids == ()


def test_negative_no_confirmed_nct_never_links():
    resolution = resolve_literature_link(None, [_literature()], DEFAULT_CONTEXT)
    assert resolution.status is LiteratureLinkStatus.UNRESOLVED


def test_negative_malformed_confirmed_nct_never_links():
    resolution = resolve_literature_link("NCT123", [_literature(nct_ids=("NCT123",))], DEFAULT_CONTEXT)
    assert resolution.status is LiteratureLinkStatus.UNRESOLVED


def test_negative_empty_candidate_list_is_unresolved_not_not_found():
    resolution = resolve_literature_link(NCT_A, [], DEFAULT_CONTEXT)
    assert resolution.status is LiteratureLinkStatus.UNRESOLVED


def test_negative_literature_source_not_in_context_never_links():
    resolution = resolve_literature_link(NCT_A, [_literature()], _context(sources=[]))
    assert resolution.status is not LiteratureLinkStatus.LINKED


def test_negative_literature_tier_claim_mismatch_never_links():
    lit = _literature(source_tier=SourceTier.TIER_1)  # real lit source is UNKNOWN
    resolution = resolve_literature_link(NCT_A, [lit], DEFAULT_CONTEXT)
    assert resolution.status is not LiteratureLinkStatus.LINKED


def test_conflicted_same_pmid_contradictory_nct_ids():
    candidates = [
        _literature(pmid="90000099", nct_ids=(NCT_A,)),
        _literature(pmid="90000099", nct_ids=(NCT_B,)),
    ]
    resolution = resolve_literature_link(NCT_A, candidates, DEFAULT_CONTEXT)
    assert resolution.status is LiteratureLinkStatus.CONFLICTED
