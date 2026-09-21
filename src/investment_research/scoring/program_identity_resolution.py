"""Phase 4.3B: pure, deterministic Program Identity and Literature Link
resolvers -- the offline contract layer's decision logic.

**Not connected to Pipeline/CLI/any Agent.** ``resolve_program_identity``/
``resolve_literature_link`` take only the frozen evidence types from
``program_evidence.py`` (plus plain ``str``/``frozenset`` inputs) and
return a plain, frozen result -- no ``Fact``, no ``AgentInput``, no
network, no LLM. A dedicated test
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
recording (see ``pipeline.py`` lines ~589-694 as of this phase's base
commit) -- never just the agent invocation on its own. Stage 3b's existing
pattern (``escalate_unresolved_questions`` appending new facts straight to
``verified_facts`` without a second Evidence Integrity pass) is explicitly
NOT the model for Adaptive Acquisition's new facts: a future
implementation must run the full pass above again over
``verified_facts_1 UNION new_facts``, never append new facts directly.

Bootstrap resolution (before Adaptive Acquisition ever runs) and final
resolution (after Evidence Integrity pass #2) are two DIFFERENT calls to
``resolve_program_identity``, over two different evidence sets, and must
be kept as two separate, separately-inspectable ``ProgramIdentityResolution``
values -- never silently collapsed into one. If they disagree (e.g. new
Adaptive evidence surfaces a same-NCT sponsor contradiction the bootstrap
pass never saw), a future caller must treat that as
``ProgramIdentityStatus.CONFLICTED``/``UNRESOLVED``, and stop -- never
silently prefer the later (or earlier) resolution.

A future caller must never place a ``ProgramIdentityResolution``/
``LiteratureLinkResolution`` (or the raw company/program/NCT identity they
carry) into the Blind Judge's input, by any path, including the existing
``AgentInput.params`` dict every agent already reads from: the Blind
Judge's whole purpose is evaluating an ANONYMIZED evidence pack, and
ticker/sponsor/NCT identity is exactly the identity information
``orchestrator/anonymize.py`` exists to withhold from it. More generally,
threading either resolution into the pre-existing, already-shared
``Pipeline.run()`` ``params: dict[str, Any]`` unconditionally (the way
``price``/``aliases``/``mode`` are today) is exactly what a future
implementation must NOT do without an explicit, agent-by-agent allowlist
decision -- "every agent already reads params" must never become "every
agent silently gets company identity", including agents that should never
see it.

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
contract is recorded here for a future phase to hold itself to, mirroring
``research/literature_pipeline_integration.py``'s own existing
``_targeted_request_budget``/``_discovery_request_budget`` precedent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..schemas.enums import UNKNOWN, StrEnum
from .identifier_validation import normalize_company_name, validate_strict_nct_id
from .program_evidence import (
    CompanyIdentityEvidence,
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
    #: docstring) is met, with no explicit contradiction.
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
    #: names the confirmed NCT id.
    LINKED = "LINKED"
    #: Literature evidence was genuinely evaluated and none of it
    #: references the confirmed NCT id -- distinct from UNRESOLVED (never
    #: evaluated at all): CLAUDE.md's "unsearched is not the same as
    #: searched and found nothing", applied to this link.
    NOT_FOUND = "NOT_FOUND"
    #: No confirmed NCT id to link against, no literature candidates
    #: supplied, or every supplied candidate failed structural validation.
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
    #: audit -- never hidden, mirroring ``ProgramResolution.candidates``'s
    #: own existing precedent in ``scoring/program_resolution.py``.
    evaluated_candidate_count: int = 0
    conflicting_nct_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiteratureLinkResolution:
    status: LiteratureLinkStatus
    #: Every PMID whose own structured metadata explicitly names the
    #: confirmed NCT id -- ALL of them, never silently narrowed to one
    #: (Phase 4.3B requirement 6).
    pmids: tuple[str, ...] = ()
    rationale: str = ""


def resolve_program_identity(
    company: CompanyIdentityEvidence,
    candidates: Sequence[ProgramCandidateEvidence],
    *,
    explicit_lead_trial_ids: frozenset[str] = frozenset(),
) -> ProgramIdentityResolution:
    """CONFIRMED requires ALL of:

    1. A candidate's ``nct_id`` passes strict ``^NCT\\d{8}$`` validation
       (checked by ``validate_program_candidate_evidence``).
    2. That candidate's own structural validation outcome is VALID (a
       ``source_id`` is present, ``supporting_fact_ids`` -- if any -- are
       all fact_id-shaped).
    3. ``company``'s own structural validation outcome is VALID AND its
       ``sec_official_name`` is resolved (not UNKNOWN) -- i.e. ticker ->
       CIK -> SEC official company name is actually established.
    4. The candidate's ``lead_sponsor``, after
       ``normalize_company_name()``, equals ``company.sec_official_name``
       OR one of ``company.explicitly_verified_aliases``, after the SAME
       normalization -- exact match only, never fuzzy, never LLM-assisted.
    5. No explicit conflict (see CONFLICTED below).

    NEVER sufficient for CONFIRMED, by construction (none of these are
    ever read by the matching logic below):

    * a name appearing only in ``collaborators`` (never checked -- only
      ``lead_sponsor`` is);
    * an ``interventions``/``conditions`` match alone (never checked at
      all -- this function has no drug-name/indication matching logic);
    * recency of any date field (never checked -- no date field is ever
      read here);
    * a PMID/literature match of any kind (this function never receives
      literature evidence);
    * ``explicitly_verified_aliases`` containing a value the CALLER never
      actually verified (this function trusts its input's own field name
      and cannot detect a caller who mislabels an unverified guess as
      "explicitly verified" -- that discipline belongs to the caller, and
      production callers must never source this field from fixture
      metadata; see this module's own top docstring).

    Multiple sponsor-matched candidates (Phase 4.3B requirement 5's
    ambiguity rule): if exactly one is named in ``explicit_lead_trial_ids``
    (meant to carry whatever a future caller determined, from VERIFIED
    Facts, has an explicit lead/current marker -- see
    ``scoring/program_resolution.py``'s own ``_LEAD_MARKER_RE`` for the
    precedent this module deliberately does not reimplement, to stay
    Fact-independent), that one is CONFIRMED. Otherwise: UNRESOLVED. Never
    resolved by recency, and never by picking an arbitrary one.

    CONFLICTED fires ONLY when the evidence ITSELF is self-contradictory:
    the same ``nct_id`` appears across two or more (valid) candidates with
    different, non-UNKNOWN, normalized ``lead_sponsor`` values. A
    pre-listing trial, a collaborator relationship, a former company name
    that was never independently verified, or simply "no candidate's
    sponsor matched" are NEVER, by themselves, CONFLICTED -- they resolve
    to UNRESOLVED (Phase 4.3B requirement 5).
    """
    company_result = validate_company_identity_evidence(company)
    if company_result.outcome != EvidenceValidationOutcome.VALID:
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.UNRESOLVED,
            rationale=(
                f"company identity evidence is {company_result.outcome}: "
                f"{'; '.join(company_result.reasons) or 'no reason recorded'}"
            ),
        )

    valid_candidates: list[ProgramCandidateEvidence] = []
    for candidate in candidates:
        if validate_program_candidate_evidence(candidate).outcome == EvidenceValidationOutcome.VALID:
            valid_candidates.append(candidate)

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
        )

    acceptable_identities = {normalize_company_name(company.sec_official_name)} | {
        normalize_company_name(alias)
        for alias in company.explicitly_verified_aliases
        if alias and alias != UNKNOWN
    }
    acceptable_identities.discard("")

    sponsor_matched = [
        c
        for c in valid_candidates
        if c.lead_sponsor != UNKNOWN and normalize_company_name(c.lead_sponsor) in acceptable_identities
    ]

    if not sponsor_matched:
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.UNRESOLVED,
            rationale=(
                "no strict-valid ProgramCandidateEvidence's lead_sponsor matched the SEC "
                "official company name or an explicitly verified alias"
            ),
            evaluated_candidate_count=len(candidates),
        )

    if len(sponsor_matched) == 1:
        chosen = sponsor_matched[0]
        return ProgramIdentityResolution(
            status=ProgramIdentityStatus.CONFIRMED,
            nct_id=chosen.nct_id,
            lead_sponsor=chosen.lead_sponsor,
            rationale=f"exactly one sponsor-matched, strict-valid NCT candidate: {chosen.nct_id}",
            evaluated_candidate_count=len(candidates),
        )

    marked = [c for c in sponsor_matched if c.nct_id in explicit_lead_trial_ids]
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
        )

    return ProgramIdentityResolution(
        status=ProgramIdentityStatus.UNRESOLVED,
        rationale=(
            f"{len(sponsor_matched)} sponsor-matched candidates with no unambiguous lead "
            "marker -- never resolved by recency or arbitrary choice alone"
        ),
        evaluated_candidate_count=len(candidates),
    )


def resolve_literature_link(
    confirmed_nct_id: str | None,
    candidates: Sequence[LiteratureCandidateEvidence],
) -> LiteratureLinkResolution:
    """LINKED requires a strict-valid PMID whose OWN structured metadata
    (``LiteratureCandidateEvidence.nct_ids``) explicitly names
    ``confirmed_nct_id`` -- never a drug-name/company-name match in the
    article's text, which this function has no access to at all (Phase
    4.3B requirement 6).

    ``SourceTier`` is never read or upgraded here -- literature evidence
    stays whatever tier it already carries (``SourceTier.UNKNOWN`` for
    every literature record this repository can currently produce; see
    ``program_evidence.LiteratureCandidateEvidence``'s own docstring).
    """
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

    valid_candidates = [
        c
        for c in candidates
        if validate_literature_candidate_evidence(c).outcome == EvidenceValidationOutcome.VALID
    ]
    if not valid_candidates:
        return LiteratureLinkResolution(
            status=LiteratureLinkStatus.UNRESOLVED,
            rationale="no literature candidate evidence passed structural validation",
        )

    by_pmid: dict[str, list[LiteratureCandidateEvidence]] = {}
    for candidate in valid_candidates:
        by_pmid.setdefault(candidate.pmid, []).append(candidate)
    for pmid, group in by_pmid.items():
        if len({g.nct_ids for g in group}) > 1:
            return LiteratureLinkResolution(
                status=LiteratureLinkStatus.CONFLICTED,
                rationale=f"PMID {pmid} has contradictory nct_ids across evidence records",
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
        )
    return LiteratureLinkResolution(
        status=LiteratureLinkStatus.NOT_FOUND,
        rationale=(
            f"{len(valid_candidates)} literature candidate(s) evaluated; none explicitly "
            f"reference {normalized_target}"
        ),
    )


__all__ = [
    "LiteratureLinkResolution",
    "LiteratureLinkStatus",
    "ProgramIdentityResolution",
    "ProgramIdentityStatus",
    "resolve_literature_link",
    "resolve_program_identity",
]
