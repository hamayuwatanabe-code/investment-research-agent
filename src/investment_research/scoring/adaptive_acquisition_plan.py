"""Phase 4.3G: a pure, offline, side-effect-free Adaptive Acquisition PLAN.

This module computes what literature acquisition request (if any) would be
worth making next, given an already-resolved ``ProgramIdentityResolution``
and ``LiteratureLinkResolution`` (both from ``program_identity_resolution.py``
-- reused, never re-implemented here). It never executes anything: no
acquisition-executing code of any kind, no adapter, no HTTP client of any
kind, no LLM, no Web Search. The result is a plan to act on in a LATER
phase, not an action.

Kept as its own module, separate from ``program_identity_resolution.py``,
because resolution (what do we currently know?) and planning (what would be
worth fetching next, and at what budget?) are different questions with
different callers in mind -- a consumer reads ``AdaptiveAcquisitionPlan``
without ever needing to re-derive it from the two resolutions itself,
mirroring ``program_evidence.py``/``program_identity_resolution.py``'s own
layering. Phase 4.3H (``research/adaptive_literature_acquisition.py``) is
that consumer, offline-verified only -- it independently re-verifies a
``READY`` plan before turning it into one Literature Document-First
acquisition request; it is still not connected to
``orchestrator/pipeline.py``/``cli.py``/the Action Gate.

Budget contract (Phase 4.3G correction of ``program_identity_resolution.py``'s
own now-superseded "5 requests" note): a ``READY`` plan carries
``max_requests=6``, covering (per ticker, if executed in a future phase)::

    PubMed ESearch                 <= 1
    PubMed EFetch                  <= 1
    Europe PMC search              <= 3
    full-text fetch                <= 1
    --------------------------------------
    = 6 requests (maximum)

ClinicalTrials confirmation is deliberately NOT counted here (unlike the
superseded 5-request note): ``ProgramCandidateEvidence`` -- the structured
trial-identity evidence this plan's own NCT id comes from -- was already
obtained by the ORIGINAL ClinicalTrials collector during Stage 1 collection,
so a future executor acting on this plan would never need a second
ClinicalTrials fetch merely to re-confirm the same NCT id.

This module imports ``identifier_validation`` directly (the same offline,
dependency-free validator ``program_identity_resolution.py`` already uses)
to independently re-check ``ProgramIdentityResolution.nct_id``'s strict
validity before planning against it -- defensive-in-depth, since a
CONFIRMED resolution's own contract already guarantees this, but this
module never simply trusts that invariant without checking it itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..schemas.enums import UNKNOWN, StrEnum
from .identifier_validation import validate_strict_nct_id
from .program_identity_resolution import (
    LiteratureLinkResolution,
    LiteratureLinkStatus,
    ProgramIdentityResolution,
    ProgramIdentityStatus,
)

#: See the module docstring's budget contract.
READY_MAX_REQUESTS = 6

#: Mirrors (but never imports) ``research.literature_pipeline_integration.
#: LiteratureReferenceMode.NCT_ID``'s own string value -- this module keeps
#: zero import edge onto that HTTP-adjacent module (Phase 4.3G forbids
#: connecting to the Literature acquisition adapter entirely), so the
#: value is a self-contained literal, not a shared constant.
REFERENCE_MODE_NCT_ID = "NCT_ID"


class AcquisitionPlanStatus(StrEnum):
    """A closed set of plan outcomes -- never a bare bool, mirroring
    ``ProgramIdentityStatus``/``LiteratureLinkStatus``'s own convention."""

    #: A literature acquisition attempt against a confirmed NCT id would be
    #: worth making -- never executed by this module, only planned.
    READY = "READY"
    #: Program identity is CONFIRMED and literature is already LINKED --
    #: nothing left to acquire.
    NO_ACTION = "NO_ACTION"
    #: Program identity itself could not be confirmed -- never plan an
    #: acquisition against an unresolved identity.
    UNRESOLVED = "UNRESOLVED"
    #: Program identity (or, in principle, the literature link) is
    #: self-contradictory -- never plan an acquisition against conflicting
    #: evidence.
    CONFLICTED = "CONFLICTED"
    #: An explicit PMID/NCT Literature Document-First override already ran
    #: this invocation (``direct_acquisition_info["feature_enabled"]`` is
    #: true) -- an automatic plan must never second-guess or override an
    #: explicit user-supplied reference.
    SKIPPED_EXPLICIT_OVERRIDE = "SKIPPED_EXPLICIT_OVERRIDE"
    #: Program identity claims CONFIRMED but its own ``nct_id`` fails this
    #: module's independent, defensive strict-NCT re-check -- refuses to
    #: plan against a malformed identifier rather than silently proceeding.
    REFUSED = "REFUSED"


@dataclass(frozen=True)
class AdaptiveAcquisitionPlan:
    """What a future Adaptive Acquisition executor would do next -- purely
    descriptive. ``requires_external_communication`` is True only for
    ``READY``, and even then this module (Phase 4.3G) never acts on it:
    the field records what a LATER phase's executor would need to do,
    nothing here calls it.

    ``nct_id``/``pmid`` are the plan's own proposed target, never a copy
    of an explicit override's identifiers (``SKIPPED_EXPLICIT_OVERRIDE``
    carries no target at all, by construction -- see ``build_
    adaptive_acquisition_plan``). ``pmid`` is never populated by this
    phase: a plan's only ever-populated target identifier is an NCT id
    (``reference_mode="NCT_ID"``), since the whole point of a READY plan
    is that NO pmid is known yet.
    """

    status: AcquisitionPlanStatus
    reference_mode: str = UNKNOWN
    nct_id: str = UNKNOWN
    pmid: str = UNKNOWN
    rationale: str = ""
    max_requests: int = 0
    requires_external_communication: bool = False


def build_adaptive_acquisition_plan(
    identity: ProgramIdentityResolution,
    literature: LiteratureLinkResolution,
    direct_acquisition_info: Mapping[str, Any] | None = None,
) -> AdaptiveAcquisitionPlan:
    """Pure function: three already-computed values in, one plan out. No
    I/O, no randomness, no clock reads -- the SAME inputs always produce
    the SAME plan (Phase 4.3G's fresh-run/resume equivalence requirement
    depends on this).

    Evaluation order (first match wins):

    1. An explicit PMID/NCT override already ran this invocation
       (``direct_acquisition_info.get("feature_enabled")`` is true) --
       ``SKIPPED_EXPLICIT_OVERRIDE``, regardless of what ``identity``/
       ``literature`` say. An automatic plan must never second-guess or
       override an explicit, already-used reference.
    2. ``identity.status`` is ``CONFLICTED`` -- ``CONFLICTED``, no target.
    3. ``identity.status`` is ``UNRESOLVED`` -- ``UNRESOLVED``, no target.
    4. ``identity.status`` is ``CONFIRMED`` but ``identity.nct_id`` fails
       this module's own independent strict-NCT re-check -- ``REFUSED``
       (defensive-in-depth; ``resolve_program_identity``'s own contract
       should make this unreachable in practice, but this module never
       simply trusts that without checking).
    5. ``identity.status`` is ``CONFIRMED`` with a strict-valid NCT id,
       and ``literature.status`` is ``LINKED`` -- ``NO_ACTION``, no
       target (nothing left to acquire).
    6. ``identity.status`` is ``CONFIRMED`` with a strict-valid NCT id,
       and ``literature.status`` is ``UNRESOLVED``/``NOT_FOUND`` --
       ``READY``, targeting the confirmed NCT id, budget 6.
    7. ``identity.status`` is ``CONFIRMED`` with a strict-valid NCT id,
       and ``literature.status`` is ``CONFLICTED`` -- ``CONFLICTED``, no
       target (a self-contradictory literature link is never planned
       against).
    """
    info = direct_acquisition_info or {}
    if info.get("feature_enabled"):
        return AdaptiveAcquisitionPlan(
            status=AcquisitionPlanStatus.SKIPPED_EXPLICIT_OVERRIDE,
            rationale=(
                "an explicit PMID/NCT Literature Document-First override already ran this "
                "invocation; an automatic plan never overrides an explicit reference"
            ),
        )

    if identity.status is ProgramIdentityStatus.CONFLICTED:
        return AdaptiveAcquisitionPlan(
            status=AcquisitionPlanStatus.CONFLICTED,
            rationale=identity.rationale or "program identity evidence is self-contradictory",
        )

    if identity.status is ProgramIdentityStatus.UNRESOLVED:
        return AdaptiveAcquisitionPlan(
            status=AcquisitionPlanStatus.UNRESOLVED,
            rationale=identity.rationale or "program identity could not be confirmed",
        )

    # identity.status is CONFIRMED from here on.
    canonical_nct = validate_strict_nct_id(identity.nct_id)
    if canonical_nct is None:
        return AdaptiveAcquisitionPlan(
            status=AcquisitionPlanStatus.REFUSED,
            rationale=(
                f"program identity is CONFIRMED but its nct_id {identity.nct_id!r} failed "
                "this module's own independent strict-NCT re-check; refusing to plan an "
                "acquisition against a malformed identifier"
            ),
        )

    if literature.status is LiteratureLinkStatus.LINKED:
        return AdaptiveAcquisitionPlan(
            status=AcquisitionPlanStatus.NO_ACTION,
            rationale=(
                f"program identity CONFIRMED for {canonical_nct}; literature is already "
                "LINKED -- nothing left to acquire"
            ),
        )

    if literature.status is LiteratureLinkStatus.CONFLICTED:
        return AdaptiveAcquisitionPlan(
            status=AcquisitionPlanStatus.CONFLICTED,
            rationale=literature.rationale or "literature link evidence is self-contradictory",
        )

    # literature.status is UNRESOLVED or NOT_FOUND.
    return AdaptiveAcquisitionPlan(
        status=AcquisitionPlanStatus.READY,
        reference_mode=REFERENCE_MODE_NCT_ID,
        nct_id=canonical_nct,
        rationale=(
            f"program identity CONFIRMED for {canonical_nct}; literature link is "
            f"{literature.status} -- a literature acquisition attempt would be worth planning"
        ),
        max_requests=READY_MAX_REQUESTS,
        requires_external_communication=True,
    )


__all__ = [
    "READY_MAX_REQUESTS",
    "REFERENCE_MODE_NCT_ID",
    "AcquisitionPlanStatus",
    "AdaptiveAcquisitionPlan",
    "build_adaptive_acquisition_plan",
]
