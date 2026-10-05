"""Phase 4.3G: ``scoring/adaptive_acquisition_plan.py``'s pure plan
builder, tested in isolation from Pipeline -- no Repository, no
Pipeline.run(), no collectors. Mirrors ``tests/unit/
test_program_identity_resolution.py``'s own style: hand-built resolution
values in, a plan out, nothing executed."""

from __future__ import annotations

import pytest

from investment_research.scoring.adaptive_acquisition_plan import (
    READY_MAX_REQUESTS,
    REFERENCE_MODE_NCT_ID,
    AcquisitionPlanStatus,
    build_adaptive_acquisition_plan,
)
from investment_research.scoring.program_identity_resolution import (
    LiteratureLinkResolution,
    LiteratureLinkStatus,
    ProgramIdentityResolution,
    ProgramIdentityStatus,
)

NCT_ID = "NCT01234567"


def _identity(status: ProgramIdentityStatus, nct_id: str = NCT_ID) -> ProgramIdentityResolution:
    return ProgramIdentityResolution(status=status, nct_id=nct_id, lead_sponsor="Demo Biotherapeutics")


def _literature(status: LiteratureLinkStatus, pmids: tuple[str, ...] = ()) -> LiteratureLinkResolution:
    return LiteratureLinkResolution(status=status, pmids=pmids)


# --- Required test 1: CONFIRMED + unresolved literature link -> READY -----
@pytest.mark.parametrize("literature_status", [LiteratureLinkStatus.UNRESOLVED, LiteratureLinkStatus.NOT_FOUND])
def test_confirmed_with_unresolved_or_not_found_literature_is_ready(literature_status):
    plan = build_adaptive_acquisition_plan(
        _identity(ProgramIdentityStatus.CONFIRMED), _literature(literature_status),
    )
    assert plan.status is AcquisitionPlanStatus.READY
    assert plan.reference_mode == REFERENCE_MODE_NCT_ID
    assert plan.nct_id == NCT_ID
    assert plan.max_requests == 6
    assert plan.max_requests == READY_MAX_REQUESTS
    assert plan.requires_external_communication is True


# --- Required test 2: CONFIRMED + LINKED -> NO_ACTION ----------------------
def test_confirmed_with_linked_literature_is_no_action():
    plan = build_adaptive_acquisition_plan(
        _identity(ProgramIdentityStatus.CONFIRMED), _literature(LiteratureLinkStatus.LINKED, ("12345678",)),
    )
    assert plan.status is AcquisitionPlanStatus.NO_ACTION
    assert plan.nct_id == "UNKNOWN"
    assert plan.max_requests == 0
    assert plan.requires_external_communication is False


# --- Required test 3: Program UNRESOLVED -> no target, budget=0 -----------
def test_program_unresolved_has_no_target_and_zero_budget():
    plan = build_adaptive_acquisition_plan(
        _identity(ProgramIdentityStatus.UNRESOLVED, nct_id="UNKNOWN"),
        _literature(LiteratureLinkStatus.UNRESOLVED),
    )
    assert plan.status is AcquisitionPlanStatus.UNRESOLVED
    assert plan.nct_id == "UNKNOWN"
    assert plan.pmid == "UNKNOWN"
    assert plan.max_requests == 0
    assert plan.requires_external_communication is False


# --- Required test 4: Program CONFLICTED -> no target, budget=0 -----------
def test_program_conflicted_has_no_target_and_zero_budget():
    plan = build_adaptive_acquisition_plan(
        _identity(ProgramIdentityStatus.CONFLICTED, nct_id="UNKNOWN"),
        _literature(LiteratureLinkStatus.UNRESOLVED),
    )
    assert plan.status is AcquisitionPlanStatus.CONFLICTED
    assert plan.nct_id == "UNKNOWN"
    assert plan.max_requests == 0
    assert plan.requires_external_communication is False


def test_literature_conflicted_under_a_confirmed_identity_is_also_conflicted():
    """Section 7's contract covers Program Identity CONFLICTED explicitly;
    a self-contradictory LITERATURE link under an otherwise-CONFIRMED
    identity must also never be planned against."""
    plan = build_adaptive_acquisition_plan(
        _identity(ProgramIdentityStatus.CONFIRMED), _literature(LiteratureLinkStatus.CONFLICTED),
    )
    assert plan.status is AcquisitionPlanStatus.CONFLICTED
    assert plan.nct_id == "UNKNOWN"
    assert plan.max_requests == 0
    assert plan.requires_external_communication is False


# --- Required test 5: explicit override already used -----------------------
@pytest.mark.parametrize(
    "identity_status",
    [
        ProgramIdentityStatus.CONFIRMED,
        ProgramIdentityStatus.UNRESOLVED,
        ProgramIdentityStatus.CONFLICTED,
    ],
)
def test_explicit_override_enabled_skips_regardless_of_identity_status(identity_status):
    plan = build_adaptive_acquisition_plan(
        _identity(identity_status),
        _literature(LiteratureLinkStatus.UNRESOLVED),
        direct_acquisition_info={"feature_enabled": True, "pmid": "99999999", "nct_id": None},
    )
    assert plan.status is AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE
    assert plan.nct_id == "UNKNOWN"
    assert plan.pmid == "UNKNOWN"
    assert plan.max_requests == 0
    assert plan.requires_external_communication is False


def test_explicit_override_disabled_or_absent_does_not_skip():
    for info in (None, {}, {"feature_enabled": False}):
        plan = build_adaptive_acquisition_plan(
            _identity(ProgramIdentityStatus.CONFIRMED),
            _literature(LiteratureLinkStatus.NOT_FOUND),
            direct_acquisition_info=info,
        )
        assert plan.status is AcquisitionPlanStatus.READY


# --- Required test 6: malformed NCT -> REFUSED, zero communication --------
@pytest.mark.parametrize("bad_nct", ["NCT1234567", "NOTANNCT", "", "UNKNOWN"])
def test_confirmed_with_malformed_nct_is_refused(bad_nct):
    plan = build_adaptive_acquisition_plan(
        _identity(ProgramIdentityStatus.CONFIRMED, nct_id=bad_nct),
        _literature(LiteratureLinkStatus.UNRESOLVED),
    )
    assert plan.status is AcquisitionPlanStatus.REFUSED
    assert plan.nct_id == "UNKNOWN"
    assert plan.max_requests == 0
    assert plan.requires_external_communication is False


# --- Structural: requires_external_communication is True ONLY for READY --
def test_confirmed_with_non_canonical_case_nct_is_still_ready_canonicalized():
    """A strict-VALID but non-canonical-case NCT id (lowercase) is not
    "malformed" -- validate_strict_nct_id() canonicalizes it, exactly like
    the ClinicalTrials collector's own gate does -- so this must reach
    READY with the canonical form, never REFUSED."""
    plan = build_adaptive_acquisition_plan(
        _identity(ProgramIdentityStatus.CONFIRMED, nct_id="nct01234567"),
        _literature(LiteratureLinkStatus.UNRESOLVED),
    )
    assert plan.status is AcquisitionPlanStatus.READY
    assert plan.nct_id == NCT_ID


def test_requires_external_communication_is_true_only_for_ready():
    all_plans = [
        build_adaptive_acquisition_plan(
            _identity(ProgramIdentityStatus.CONFIRMED), _literature(LiteratureLinkStatus.NOT_FOUND),
        ),
        build_adaptive_acquisition_plan(
            _identity(ProgramIdentityStatus.CONFIRMED), _literature(LiteratureLinkStatus.LINKED, ("1",)),
        ),
        build_adaptive_acquisition_plan(
            _identity(ProgramIdentityStatus.UNRESOLVED), _literature(LiteratureLinkStatus.UNRESOLVED),
        ),
        build_adaptive_acquisition_plan(
            _identity(ProgramIdentityStatus.CONFLICTED), _literature(LiteratureLinkStatus.UNRESOLVED),
        ),
        build_adaptive_acquisition_plan(
            _identity(ProgramIdentityStatus.CONFIRMED, nct_id="BAD"), _literature(LiteratureLinkStatus.UNRESOLVED),
        ),
        build_adaptive_acquisition_plan(
            _identity(ProgramIdentityStatus.CONFIRMED),
            _literature(LiteratureLinkStatus.UNRESOLVED),
            direct_acquisition_info={"feature_enabled": True},
        ),
    ]
    for plan in all_plans:
        if plan.status is AcquisitionPlanStatus.READY:
            assert plan.requires_external_communication is True
            assert plan.max_requests == 6
        else:
            assert plan.requires_external_communication is False
            assert plan.max_requests == 0


def test_plan_status_enum_is_closed_and_covers_exactly_six_members():
    assert {member.value for member in AcquisitionPlanStatus} == {
        "READY", "NO_ACTION", "UNRESOLVED", "CONFLICTED",
        "SKIPPED_EXPLICIT_OVERRIDE", "REFUSED",
    }
