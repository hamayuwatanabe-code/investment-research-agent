"""Phase 4.3B: pure, deterministic Program Identity and Literature Link
resolvers -- the offline contract layer's decision logic.

**Not connected to Pipeline/CLI/any Agent.** ``resolve_program_identity``/
``resolve_literature_link`` take only the frozen evidence types from
``program_evidence.py`` (plus plain ``str``/``frozenset`` inputs and an
``EvidenceValidationContext``) and return a plain, frozen result -- no
``AgentInput``, no network, no LLM. A dedicated test
(``tests/unit/test_program_identity_offline_isolation.py``) asserts this
module's name never appears in ``cli.py``/``pipeline.py``/any
``agents/*.py`` source text.

Two states, kept structurally separate (Phase 4.3B requirement 1):

* ``ProgramIdentityStatus`` -- company/ticker/CIK <-> sponsor <-> NCT.
  ``CONFIRMED`` requires a matching sponsor identity; it is reachable with
  ZERO literature evidence at all. A PMID is never required for
  ``ProgramIdentityStatus.CONFIRMED``.
* ``LiteratureLinkStatus`` -- NCT <-> PMID, evaluated entirely separately,
  and only ever relative to an already-resolved NCT id. Program Identity
  confirmation is never gated on Literature Link status, and Literature
  Link resolution is never used to backfill or override Program Identity.

---------------------------------------------------------------------------
Phase 4.3B Correction 1: what changed
---------------------------------------------------------------------------

1. Both resolvers now REQUIRE an ``EvidenceValidationContext`` (a breaking
   signature change from Phase 4.3B) and validate every candidate against
   it via ``program_evidence.py``'s now-referential ``validate_*_
   evidence`` functions -- a format-valid but non-existent ``source_id``/
   ``supporting_fact_ids`` entry can no longer reach CONFIRMED/LINKED. See
   ``program_evidence.py``'s own module docstring for the full format-vs-
   referential distinction.
2. Alias-based CONFIRMED is DISABLED in this phase (requirement 4, option
   B): ``resolve_program_identity`` compares a candidate's
   ``lead_sponsor`` only against ``company.sec_official_name`` --
   ``CompanyIdentityEvidence.explicitly_verified_aliases`` is never read.
   See ``program_evidence.CompanyIdentityEvidence``'s own docstring for
   why option A (a referentially-checked ``CompanyAliasEvidence`` type)
   was deferred instead.
3. ``explicit_lead_trial_ids`` entries are now normalized through
   ``validate_strict_nct_id`` before use -- a malformed entry can never
   coincidentally "match" a candidate (requirement 5).
4. INVALID/INCOMPLETE evidence mixed with VALID evidence for the SAME
   identifier (nct_id / pmid) is no longer silently dropped: it taints
   that identifier out of CONFIRMED/LINKED eligibility, and the exclusion
   is recorded in the resolution's own ``excluded_evidence_reasons``
   field -- never a silent skip (requirement 6/7).

---------------------------------------------------------------------------
Future (NOT implemented in Phase 4.3B) Pipeline connection
---------------------------------------------------------------------------

The 2-pass structure Phase 4.3A-correction-1's audit specified, for a
LATER phase to implement::

    initial collect
      -> FactCollector
      -> full Evidence Integrity pass #1
      -> Program Identity Resolution        (bootstrap resolution)
      -> Adaptive Acquisition Plan
      -> new Fact/Chunk projection
      -> full Evidence Integrity pass #2
      -> final Program Resolution           (post-Integrity-#2 resolution)
      -> Domain Agents (Kill/Science/...)

"full Evidence Integrity pass" means the ENTIRE existing
``orchestrator.pipeline.Pipeline.run()`` Stage-2 sequence, not merely an
``EvidenceIntegrityAgent.run()`` call in isolation: Source persistence
(``Repository.save_sources``), quarantine, quarantine-cascade fact
exclusion, ``bus`` update, and ``result.failures``/``ctx.status``
recording -- never just the agent invocation on its own. Stage 3b's
existing pattern (``escalate_unresolved_questions`` appending new facts
straight to ``verified_facts`` without a second Evidence Integrity pass)
is explicitly NOT the model for Adaptive Acquisition's new facts: a future
implementation must run the full pass above again over
``verified_facts_1 UNION new_facts``, never append new facts directly.
The resulting ``verified_facts``/``sources`` are exactly what a future
caller would use to build the ``EvidenceValidationContext`` this
correction now requires.

Bootstrap resolution (before Adaptive Acquisition ever runs) and final
resolution (after Evidence Integrity pass #2) are two DIFFERENT calls to
``resolve_program_identity``, over two different evidence sets AND two
different ``EvidenceValidationContext`` values, and must be kept as two
separate, separately-inspectable ``ProgramIdentityResolution`` values --
never silently collapsed into one. If they disagree, a future caller must
treat that as ``ProgramIdentityStatus.CONFLICTED``/``UNRESOLVED``, and
stop -- never silently prefer the later (or earlier) resolution.

A future caller must never place a ``ProgramIdentityResolution``/
``LiteratureLinkResolution`` (or the raw company/program/NCT identity they
carry) into the Blind Judge's input, by any path, including the existing
``AgentInput.params`` dict every agent already reads from. More generally,
threading either resolution into the pre-existing, already-shared
``Pipeline.run()`` ``params: dict[str, Any]`` unconditionally is exactly
what a future implementation must NOT do without an explicit,
agent-by-agent allowlist decision.

---------------------------------------------------------------------------
Future (NOT implemented in Phase 4.3B) request-budget contract
---------------------------------------------------------------------------

For a future Adaptive Acquisition step that fetches literature to attempt
a Literature Link, per ticker, including at most one full-text fetch::

    ClinicalTrials confirmation  1
    + PubMed ESearch             1
    + PubMed EFetch              1
    + Europe PMC search          1
    + full-text fetch (<=1)      1
    ----------------------------------
    = 5 requests/ticker (maximum)

Phase 4.3B implements zero network communication of any kind; this
contract is recorded here for a future phase to hold itself to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..schemas.enums import UNKNOWN, StrEnum
from .identifier_validation import normalize_company_name, validate_strict_nct_id
from .program_evidence import (
    CompanyIdentityEvidence,
    EvidenceValidationContext,
    EvidenceValidationOutcome,
    LiteratureCandidateEvidence,
    ProgramCandidateEvidence,
    validate_company_identity_evidence,
    validate_literature_candidate_evidence,
    validate_program_candidate_evidence,
)


class ProgramIdentityStatus(StrEnum):
    """company/ticker/CIK <-> sponsor <-> NCT (Phase 4.3B requirement 1A).
    A PMID is never a precondition for CONFIRMED."""

    #: Every REQUIRED condition (see ``resolve_program_identity``'s own
    #: docstring) is met, with no explicit contradiction, and every
    #: referenced Source/Fact resolved cleanly in the supplied context.
    CONFIRMED = "CONFIRMED"
    #: Insufficient evidence to confirm -- never guessed into CONFIRMED.
    #: The default/safe outcome whenever evidence merely falls short.
    UNRESOLVED = "UNRESOLVED"
    #: A primary-source record explicitly contradicts another (e.g. the
    #: SAME nct_id reported with two different, non-matching lead sponsors
    #: across two evidence records) -- never inferred from mere absence of
    #: a match.
    CONFLICTED = "CONFLICTED"


class LiteratureLinkStatus(StrEnum):
    """NCT <-> PMID (Phase 4.3B requirement 1B), evaluated independently of
    ``ProgramIdentityStatus``."""

    #: At least one strict-valid PMID's OWN structured metadata explicitly
    #: names the confirmed NCT id, and its evidence resolved cleanly.
    LINKED = "LINKED"
    #: Literature evidence was genuinely evaluated and none of it
    #: references the confirmed NCT id -- distinct from UNRESOLVED (never
    #: evaluated at all).
    NOT_FOUND = "NOT_FOUND"
    #: No confirmed NCT id to link against, no literature candidates
    #: supplied, or every supplied candidate failed validation.
    UNRESOLVED = "UNRESOLVED"
    #: The SAME pmid's evidence contradicts itself (two records for one
    #: PMID naming different ``nct_ids`` sets).
    CONFLICTED = "CONFLICTED"


@dataclass(frozen=True)
class ProgramIdentityResolution:
    status: ProgramIdentityStatus
    nct_id: str = UNKNOWN
    lead_sponsor: str = UNKNOWN
    rationale: str = ""
    #: Every candidate this resolution considered (valid or not), for
    #: audit -- never hidden.
    evaluated_candidate_count: int = 0
    conflicting_nct_ids: tuple[str, ...] = ()
    #: Phase 4.3B Correction 1, requirement 6/7: one entry per candidate
    #: (or company/context reference) excluded from consideration because
    #: it was INVALID/INCOMPLETE, or because it shared an identifier with
    #: an excluded record -- NEVER a silent drop. Empty when every input
    #: was VALID.
    excluded_evidence_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiteratureLinkResolution:
    status: LiteratureLinkStatus
    #: Every PMID whose own structured metadata explicitly names the
    #: confirmed NCT id -- ALL of them, never silently narrowed to one
    #: (Phase 4.3B requirement 6).
    pmids: tuple[str, ...] = ()
    rationale: str = ""
    excluded_evidence_reasons: tuple[str, ...] = ()


def resolve_program_identity(
    company: CompanyIdentityEvidence,
    candidates: Sequence[ProgramCandidateEvidence],
    context: EvidenceValidationContext,
    *,
    explicit_lead_trial_ids: frozenset[str] = frozenset(),
) -> ProgramIdentityResolution:
    """CONFIRMED requires ALL of:

    1. A candidate's ``nct_id`` passes strict ``^NCT\\d{8}$`` validation.
    2. That candidate's own validation outcome (format AND referential --
       see ``program_evidence.validate_program_candidate_evidence``) is
       VALID against the supplied ``context``.
    3. ``company``'s own validation outcome (format AND referential) is
       VALID against ``context``, AND its ``sec_official_name`` is
       resolved (not UNKNOWN).
    4. The candidate's ``lead_sponsor``, after
       ``normalize_company_name()``, equals ``company.sec_official_name``
       after the SAME normalization -- exact match only, never fuzzy,
       never LLM-assisted, and (Correction 1) NEVER against
       ``company.explicitly_verified_aliases``, which this function does
       not read at all in this phase (see this module's own top
       docstring, item 2).
    5. No explicit conflict (see CONFLICTED below), and no identifier
       (``nct_id``) this candidate shares is "tainted" by a sibling
       INVALID/INCOMPLETE record for the same ``nct_id`` (Correction 1,
       requirement 6/7 -- a mix of valid and invalid evidence for the
       SAME identifier is never silently resolved using only the valid
       half).

    NEVER sufficient for CONFIRMED, by construction: a name appearing only
    in ``collaborators``; an ``interventions``/``conditions`` match alone;
    recency of any date field; a PMID/literature match of any kind; a
    verified/unverified alias of any kind (disabled entirely this phase).

    Multiple sponsor-matched candidates: if exactly one is named in
    ``explicit_lead_trial_ids`` (normalized through ``validate_strict_
    nct_id`` first -- a malformed entry is simply dropped, never used),
    that one is CONFIRMED. Otherwise: UNRESOLVED. Never resolved by
    recency, and never by picking an arbitrary one.

    CONFLICTED fires ONLY when the evidence ITSELF is self-contradictory:
    the same ``nct_id`` appears across two or more (valid) candidates with
    different, non-UNKNOWN, normalized ``lead_sponsor`` values.
    """
    excluded_reasons: list[str] = []

    company_result = validate_company_identity_evidence(company, context)
    if company_result.outcome != EvidenceValidationOutcome.VALID:
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.UNRESOLVED,
            rationale=(
                f"company identity evidence is {company_result.outcome}: "
                f"{'; '.join(company_result.reasons) or 'no reason recorded'}"
            ),
        )

    valid_candidates: list[ProgramCandidateEvidence] = []
    tainted_nct_ids: set[str] = set()
    for candidate in candidates:
        result = validate_program_candidate_evidence(candidate, context)
        if result.outcome == EvidenceValidationOutcome.VALID:
            valid_candidates.append(candidate)
        else:
            tainted_nct_ids.add(candidate.nct_id)
            excluded_reasons.append(
                f"candidate {candidate.nct_id!r} excluded ({result.outcome}): "
                f"{'; '.join(result.reasons) or 'no reason recorded'}"
            )

    if tainted_nct_ids:
        # A VALID candidate sharing an nct_id with an excluded record is
        # itself no longer eligible: mixed valid/invalid evidence for the
        # SAME identifier is never resolved using only the valid half
        # (Correction 1, requirement 6/7).
        newly_tainted = {c.nct_id for c in valid_candidates if c.nct_id in tainted_nct_ids}
        if newly_tainted:
            excluded_reasons.append(
                f"NCT id(s) {sorted(newly_tainted)} also have a VALID candidate record, but "
                "are excluded anyway because an INVALID/INCOMPLETE record exists for the same id"
            )
        valid_candidates = [c for c in valid_candidates if c.nct_id not in tainted_nct_ids]

    # Explicit conflict: the SAME nct_id, contradictory lead_sponsor values
    # across independently-valid evidence records.
    by_nct: dict[str, list[ProgramCandidateEvidence]] = {}
    for candidate in valid_candidates:
        by_nct.setdefault(candidate.nct_id, []).append(candidate)
    conflicting_nct_ids = tuple(
        sorted(
            nct_id
            for nct_id, group in by_nct.items()
            if len(
                {
                    normalize_company_name(g.lead_sponsor)
                    for g in group
                    if g.lead_sponsor != UNKNOWN
                }
            )
            > 1
        )
    )
    if conflicting_nct_ids:
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.CONFLICTED,
            rationale=(
                f"{len(conflicting_nct_ids)} NCT id(s) have contradictory lead_sponsor "
                f"values across evidence records: {', '.join(conflicting_nct_ids)}"
            ),
            evaluated_candidate_count=len(candidates),
            conflicting_nct_ids=conflicting_nct_ids,
            excluded_evidence_reasons=tuple(excluded_reasons),
        )

    # Correction 1, requirement 4 option B: alias matching is disabled --
    # only the SEC official name is ever an acceptable identity.
    acceptable_identity = normalize_company_name(company.sec_official_name)

    sponsor_matched = [
        c
        for c in valid_candidates
        if c.lead_sponsor != UNKNOWN
        and acceptable_identity
        and normalize_company_name(c.lead_sponsor) == acceptable_identity
    ]

    if not sponsor_matched:
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.UNRESOLVED,
            rationale=(
                "no VALID ProgramCandidateEvidence's lead_sponsor matched the SEC official "
                "company name (alias matching is disabled in this phase)"
            ),
            evaluated_candidate_count=len(candidates),
            excluded_evidence_reasons=tuple(excluded_reasons),
        )

    if len(sponsor_matched) == 1:
        chosen = sponsor_matched[0]
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.CONFIRMED,
            nct_id=chosen.nct_id,
            lead_sponsor=chosen.lead_sponsor,
            rationale=f"exactly one sponsor-matched, VALID NCT candidate: {chosen.nct_id}",
            evaluated_candidate_count=len(candidates),
            excluded_evidence_reasons=tuple(excluded_reasons),
        )

    normalized_lead_ids = frozenset(
        normalized
        for raw in explicit_lead_trial_ids
        if (normalized := validate_strict_nct_id(raw)) is not None
    )
    marked = [c for c in sponsor_matched if c.nct_id in normalized_lead_ids]
    if len(marked) == 1:
        chosen = marked[0]
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.CONFIRMED,
            nct_id=chosen.nct_id,
            lead_sponsor=chosen.lead_sponsor,
            rationale=(
                f"{chosen.nct_id} is the one sponsor-matched candidate carrying an "
                "explicit lead/current marker"
            ),
            evaluated_candidate_count=len(candidates),
            excluded_evidence_reasons=tuple(excluded_reasons),
        )

    return ProgramIdentityResolution(
        status=ProgramIdentityStatus.UNRESOLVED,
        rationale=(
            f"{len(sponsor_matched)} sponsor-matched candidates with no unambiguous lead "
            "marker -- never resolved by recency or arbitrary choice alone"
        ),
        evaluated_candidate_count=len(candidates),
        excluded_evidence_reasons=tuple(excluded_reasons),
    )


def resolve_literature_link(
    confirmed_nct_id: str | None,
    candidates: Sequence[LiteratureCandidateEvidence],
    context: EvidenceValidationContext,
) -> LiteratureLinkResolution:
    """LINKED requires a strict-valid PMID whose OWN structured metadata
    (``LiteratureCandidateEvidence.nct_ids``) explicitly names
    ``confirmed_nct_id``, AND whose validation outcome (format AND
    referential) is VALID against ``context`` -- never a drug-name/
    company-name match in the article's text, which this function has no
    access to at all (Phase 4.3B requirement 6).

    A PMID with a mix of VALID and INVALID/INCOMPLETE evidence records is
    excluded from LINKED eligibility entirely (Correction 1, requirement
    6/7), recorded in ``excluded_evidence_reasons``.

    ``SourceTier`` is never read or upgraded here -- literature evidence
    stays whatever tier it already carries.
    """
    excluded_reasons: list[str] = []

    normalized_target = validate_strict_nct_id(confirmed_nct_id) if confirmed_nct_id else None
    if normalized_target is None:
        return LiteratureLinkResolution(
            status=LiteratureLinkStatus.UNRESOLVED,
            rationale="no strict-valid confirmed NCT id was supplied to link literature evidence against",
        )
    if not candidates:
        return LiteratureLinkResolution(
            status=LiteratureLinkStatus.UNRESOLVED,
            rationale=(
                "no literature candidate evidence was supplied -- never searched, not the "
                "same as searched-and-not-found"
            ),
        )

    valid_candidates: list[LiteratureCandidateEvidence] = []
    tainted_pmids: set[str] = set()
    for candidate in candidates:
        result = validate_literature_candidate_evidence(candidate, context)
        if result.outcome == EvidenceValidationOutcome.VALID:
            valid_candidates.append(candidate)
        else:
            tainted_pmids.add(candidate.pmid)
            excluded_reasons.append(
                f"PMID {candidate.pmid!r} excluded ({result.outcome}): "
                f"{'; '.join(result.reasons) or 'no reason recorded'}"
            )

    if tainted_pmids:
        newly_tainted = {c.pmid for c in valid_candidates if c.pmid in tainted_pmids}
        if newly_tainted:
            excluded_reasons.append(
                f"PMID(s) {sorted(newly_tainted)} also have a VALID candidate record, but are "
                "excluded anyway because an INVALID/INCOMPLETE record exists for the same PMID"
            )
        valid_candidates = [c for c in valid_candidates if c.pmid not in tainted_pmids]

    if not valid_candidates:
        return LiteratureLinkResolution(
            status=LiteratureLinkStatus.UNRESOLVED,
            rationale="no literature candidate evidence passed validation",
            excluded_evidence_reasons=tuple(excluded_reasons),
        )

    by_pmid: dict[str, list[LiteratureCandidateEvidence]] = {}
    for candidate in valid_candidates:
        by_pmid.setdefault(candidate.pmid, []).append(candidate)
    for pmid, group in by_pmid.items():
        if len({g.nct_ids for g in group}) > 1:
            return LiteratureLinkResolution(
                status=LiteratureLinkStatus.CONFLICTED,
                rationale=f"PMID {pmid} has contradictory nct_ids across evidence records",
                excluded_evidence_reasons=tuple(excluded_reasons),
            )

    matching = tuple(sorted({c.pmid for c in valid_candidates if normalized_target in c.nct_ids}))
    if matching:
        return LiteratureLinkResolution(
            status=LiteratureLinkStatus.LINKED,
            pmids=matching,
            rationale=(
                f"{len(matching)} PMID(s) explicitly reference {normalized_target} in "
                "their own structured metadata"
            ),
            excluded_evidence_reasons=tuple(excluded_reasons),
        )
    return LiteratureLinkResolution(
        status=LiteratureLinkStatus.NOT_FOUND,
        rationale=(
            f"{len(valid_candidates)} literature candidate(s) evaluated; none explicitly "
            f"reference {normalized_target}"
        ),
        excluded_evidence_reasons=tuple(excluded_reasons),
    )


__all__ = [
    "LiteratureLinkResolution",
    "LiteratureLinkStatus",
    "ProgramIdentityResolution",
    "ProgramIdentityStatus",
    "resolve_literature_link",
    "resolve_program_identity",
]
