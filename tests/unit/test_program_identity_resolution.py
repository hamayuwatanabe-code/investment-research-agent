"""Phase 4.3B: pure deterministic Program Identity / Literature Link
resolvers (``scoring/program_identity_resolution.py``).

Every test here drives ``resolve_program_identity``/``resolve_literature_
link`` directly with synthetic, offline evidence -- no Pipeline, no HTTP,
no LLM, no Fact/RawFact objects at all. Positive cases prove the resolvers
reach CONFIRMED/LINKED when they genuinely should; negative/ambiguous cases
(Phase 4.3B requirement 9) prove a long list of plausible-looking-but-
insufficient signals never wrongly produce CONFIRMED/LINKED.
"""

from __future__ import annotations

from investment_research.schemas.enums import SourceTier
from investment_research.scoring.program_evidence import (
    CompanyIdentityEvidence,
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


def _company(**overrides) -> CompanyIdentityEvidence:
    kwargs = {
        "ticker": "DEMOBIO", "cik": 1234567, "sec_official_name": "Demo Biotherapeutics Inc",
        "source_id": "src_sec_1", "source_tier": SourceTier.TIER_1, "retrieved_at": RETRIEVED_AT,
    }
    kwargs.update(overrides)
    return CompanyIdentityEvidence(**kwargs)


def _candidate(**overrides) -> ProgramCandidateEvidence:
    kwargs = {
        "nct_id": NCT_A, "lead_sponsor": "Demo Biotherapeutics, Inc.",
        "source_id": "src_ct_1", "source_tier": SourceTier.TIER_1, "retrieved_at": RETRIEVED_AT,
    }
    kwargs.update(overrides)
    return ProgramCandidateEvidence(**kwargs)


def _literature(**overrides) -> LiteratureCandidateEvidence:
    kwargs = {
        "pmid": "33378609", "nct_ids": (NCT_A,), "source_id": "src_lit_1",
        "source_tier": SourceTier.UNKNOWN, "retrieved_at": RETRIEVED_AT,
    }
    kwargs.update(overrides)
    return LiteratureCandidateEvidence(**kwargs)


# =============================================================================
# Positive: Program Identity
# =============================================================================
def test_positive_sec_name_matches_lead_sponsor_single_candidate_confirms():
    resolution = resolve_program_identity(_company(), [_candidate()])
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_A
    assert resolution.evaluated_candidate_count == 1


def test_positive_strict_valid_nct_required_for_confirmation():
    resolution = resolve_program_identity(_company(), [_candidate(nct_id=NCT_A)])
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_A


def test_positive_pmid_never_required_for_program_identity_confirmation():
    """resolve_program_identity takes no literature evidence at all --
    structurally proves a PMID is never a precondition (Phase 4.3B
    requirement 1)."""
    resolution = resolve_program_identity(_company(), [_candidate()])
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    # No literature evidence was ever passed to this call, and the
    # resolver's own signature has no such parameter -- CONFIRMED here
    # proves it can be reached with zero PMIDs in hand.


def test_positive_verified_alias_matches_when_official_name_does_not():
    company = _company(sec_official_name="Demo Bio Holdings Inc", explicitly_verified_aliases=("Demo Biotherapeutics Inc",))
    resolution = resolve_program_identity(company, [_candidate(lead_sponsor="Demo Biotherapeutics Inc")])
    assert resolution.status is ProgramIdentityStatus.CONFIRMED


def test_positive_old_trial_same_company_still_confirms():
    """A genuinely pre-listing-dated trial belonging to the SAME company
    must not be penalized merely for its date (Phase 4.3B requirement 5:
    'that alone' must never cause CONFLICTED, and here it must not even
    prevent CONFIRMED)."""
    candidate = _candidate(primary_completion_date="1999-01-01", completion_date="1999-06-01")
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status is ProgramIdentityStatus.CONFIRMED


def test_positive_explicit_lead_marker_breaks_a_multi_candidate_tie():
    candidates = [_candidate(nct_id=NCT_A), _candidate(nct_id=NCT_B)]
    resolution = resolve_program_identity(
        _company(), candidates, explicit_lead_trial_ids=frozenset({NCT_B}),
    )
    assert resolution.status is ProgramIdentityStatus.CONFIRMED
    assert resolution.nct_id == NCT_B


# =============================================================================
# Positive: Literature Link
# =============================================================================
def test_positive_nct_explicit_in_pubmed_metadata_links():
    resolution = resolve_literature_link(NCT_A, [_literature()])
    assert resolution.status is LiteratureLinkStatus.LINKED
    assert resolution.pmids == ("33378609",)


def test_positive_multiple_genuinely_matching_pmids_all_returned():
    candidates = [_literature(pmid="11111111"), _literature(pmid="22222222")]
    resolution = resolve_literature_link(NCT_A, candidates)
    assert resolution.status is LiteratureLinkStatus.LINKED
    assert resolution.pmids == ("11111111", "22222222")


def test_positive_literature_source_tier_unknown_never_blocks_a_genuine_link():
    lit = _literature(source_tier=SourceTier.UNKNOWN)
    resolution = resolve_literature_link(NCT_A, [lit])
    assert resolution.status is LiteratureLinkStatus.LINKED
    assert lit.source_tier is SourceTier.UNKNOWN  # never upgraded


# =============================================================================
# Negative/ambiguous: Program Identity -- must never wrongly CONFIRM
# =============================================================================
def test_negative_malformed_nct_never_confirms():
    resolution = resolve_program_identity(_company(), [_candidate(nct_id="NCT123")])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_sponsor_mismatch_never_confirms():
    resolution = resolve_program_identity(_company(), [_candidate(lead_sponsor="Totally Unrelated Corp")])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_collaborator_only_match_never_confirms():
    """A matching name in collaborators, with a non-matching lead_sponsor,
    must never confirm -- only lead_sponsor is ever checked."""
    candidate = _candidate(
        lead_sponsor="Some Other University", collaborators=("Demo Biotherapeutics, Inc.",),
    )
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_same_drug_name_different_company_never_confirms():
    """Intervention name matching a well-known compound is never itself
    checked, and a genuine sponsor mismatch is never overridden by it."""
    candidate = _candidate(
        lead_sponsor="A Completely Different Sponsor Inc", interventions=("Demo Compound X",),
    )
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_different_indication_alone_never_confirms_without_sponsor_match():
    candidate = _candidate(
        lead_sponsor="Unrelated Sponsor Inc", conditions=("A Completely Different Indication",),
    )
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_multiple_sponsor_matched_candidates_without_lead_marker_unresolved():
    candidates = [_candidate(nct_id=NCT_A), _candidate(nct_id=NCT_B)]
    resolution = resolve_program_identity(_company(), candidates)
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED
    assert "recency" in resolution.rationale.lower() or "arbitrary" in resolution.rationale.lower()


def test_negative_unconfirmed_acquisition_relationship_never_confirms():
    """A pre-acquisition sponsor name that was never added to
    explicitly_verified_aliases must not match, however plausible."""
    candidate = _candidate(lead_sponsor="Acquired Target Biotech Inc")
    resolution = resolve_program_identity(_company(explicitly_verified_aliases=()), [candidate])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_unconfirmed_license_out_relationship_never_confirms():
    """A licensee's name, unverified, must not match the licensor ticker's
    identity."""
    candidate = _candidate(lead_sponsor="Licensee Pharma Corp")
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_sec_former_name_alone_without_explicit_verification_never_confirms():
    """A sponsor name that HAPPENS to be a real former SEC name is never
    auto-matched unless the caller placed it in
    explicitly_verified_aliases -- this module has no formerNames lookup
    of its own (Phase 4.3B requirement 8: no such wiring exists)."""
    candidate = _candidate(lead_sponsor="Old Co Name Inc")  # a hypothetical former name
    resolution = resolve_program_identity(_company(explicitly_verified_aliases=()), [candidate])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_recency_alone_never_substitutes_for_identity_confirmation():
    """A candidate with a very recent date but a mismatched sponsor must
    never confirm merely for being the newest-dated evidence -- recency is
    never read by this resolver at all."""
    candidate = _candidate(
        lead_sponsor="Unrelated Sponsor Inc", primary_completion_date="2030-01-01",
    )
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_company_identity_incomplete_never_confirms():
    company = _company(sec_official_name="UNKNOWN")
    resolution = resolve_program_identity(company, [_candidate()])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


def test_negative_no_candidates_never_confirms():
    resolution = resolve_program_identity(_company(), [])
    assert resolution.status is ProgramIdentityStatus.UNRESOLVED


# =============================================================================
# CONFLICTED -- only an explicit, evidence-internal contradiction
# =============================================================================
def test_conflicted_same_nct_contradictory_sponsors():
    candidates = [
        _candidate(nct_id=NCT_A, lead_sponsor="Demo Biotherapeutics, Inc."),
        _candidate(nct_id=NCT_A, lead_sponsor="A Totally Different Sponsor Corp"),
    ]
    resolution = resolve_program_identity(_company(), candidates)
    assert resolution.status is ProgramIdentityStatus.CONFLICTED
    assert NCT_A in resolution.conflicting_nct_ids


def test_pre_listing_trial_alone_is_never_conflicted():
    """Phase 4.3B requirement 5: a pre-listing-dated trial, by itself, must
    never be CONFLICTED."""
    candidate = _candidate(primary_completion_date="1999-01-01")
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status != ProgramIdentityStatus.CONFLICTED


def test_collaborator_relationship_alone_is_never_conflicted():
    candidate = _candidate(collaborators=("Some Collaborator Inc",))
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status != ProgramIdentityStatus.CONFLICTED


def test_former_name_alone_is_never_conflicted():
    candidate = _candidate(lead_sponsor="Old Co Name Inc")
    resolution = resolve_program_identity(_company(), [candidate])
    assert resolution.status != ProgramIdentityStatus.CONFLICTED


# =============================================================================
# Negative/ambiguous: Literature Link -- must never wrongly LINK
# =============================================================================
def test_negative_malformed_pmid_never_links():
    resolution = resolve_literature_link(NCT_A, [_literature(pmid="not-a-pmid")])
    assert resolution.status is not LiteratureLinkStatus.LINKED


def test_negative_pmid_without_direct_nct_reference_not_found():
    resolution = resolve_literature_link(NCT_A, [_literature(nct_ids=())])
    assert resolution.status is LiteratureLinkStatus.NOT_FOUND


def test_negative_multiple_pmids_none_referencing_target_not_found():
    candidates = [
        _literature(pmid="11111111", nct_ids=()),
        _literature(pmid="22222222", nct_ids=(NCT_B,)),  # a different trial entirely
    ]
    resolution = resolve_literature_link(NCT_A, candidates)
    assert resolution.status is LiteratureLinkStatus.NOT_FOUND
    assert resolution.pmids == ()


def test_negative_no_confirmed_nct_never_links():
    resolution = resolve_literature_link(None, [_literature()])
    assert resolution.status is LiteratureLinkStatus.UNRESOLVED


def test_negative_malformed_confirmed_nct_never_links():
    resolution = resolve_literature_link("NCT123", [_literature(nct_ids=("NCT123",))])
    assert resolution.status is LiteratureLinkStatus.UNRESOLVED


def test_negative_empty_candidate_list_is_unresolved_not_not_found():
    """CLAUDE.md rule 8, applied to literature linking: never searched is
    not the same as searched-and-found-nothing."""
    resolution = resolve_literature_link(NCT_A, [])
    assert resolution.status is LiteratureLinkStatus.UNRESOLVED


def test_conflicted_same_pmid_contradictory_nct_ids():
    candidates = [
        _literature(pmid="33378609", nct_ids=(NCT_A,)),
        _literature(pmid="33378609", nct_ids=(NCT_B,)),
    ]
    resolution = resolve_literature_link(NCT_A, candidates)
    assert resolution.status is LiteratureLinkStatus.CONFLICTED
