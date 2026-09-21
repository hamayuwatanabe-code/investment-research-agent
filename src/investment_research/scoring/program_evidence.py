"""Phase 4.3B: immutable, structured clinical-program/company-identity
evidence records -- the offline contract layer for Ticker-only Program
Resolution (Phase 4.3A/4.3A-correction-1's audits).

This module is deliberately **not** Fact/RawFact. Sponsor, intervention,
condition, and company-identity metadata are structured, multi-field
registry data (ClinicalTrials.gov's own API v2 shape, SEC's own
ticker/CIK/company-title shape) -- collapsing them into a single free-text
``Fact.claim`` string and later regex-re-extracting them (as
``scoring/program_resolution.py``'s own ``NCT_RE`` does, deliberately, for
a different purpose: disambiguating among trial ids ALREADY present in
verified Fact claims) would lose structure and risk misattribution. See
this module's own docstring precedent: Phase 4.2B's date-normalization
work established "never misattribute a date meant for one field to a
different field" -- the same principle applies here to sponsor/
intervention/condition/date fields, which is why ``ProgramCandidateEvidence``
keeps each CT.gov date in its own explicitly named field (never a generic
``dict[str, str]``) and never round-trips through a claim string.

Every dataclass here is frozen, and every collection-valued field is a
``tuple``, never a ``list``/``dict`` -- Phase 4.3B requirement 2.

---------------------------------------------------------------------------
Phase 4.3B Correction 1: format validation vs. referential validation
---------------------------------------------------------------------------

Phase 4.3B's own first cut only checked that ``supporting_fact_ids``/
``source_id`` were the right SHAPE (``fact_`` + 20 hex characters, a
non-empty string) -- never that they actually named a record that exists.
A format-valid but entirely invented fact id or source id passed
validation. This correction adds a second, separate validation layer:
**referential** validation against an explicitly-supplied
:class:`EvidenceValidationContext`, checked by
``validate_company_identity_evidence``/``validate_program_candidate_
evidence``/``validate_literature_candidate_evidence``, all three of which
now REQUIRE a context argument (a breaking signature change from Phase
4.3B, confined to this module and ``program_identity_resolution.py`` --
see that module's own Correction-1 changes).

* **Format validation** (unchanged from Phase 4.3B): is this string
  shaped like a real identifier/hash/timestamp? Answerable from the
  record alone, no external data needed.
* **Referential validation** (new): does the id actually resolve to a
  real record in the caller-supplied context, and do the evidence
  record's OWN claimed ``source_tier``/``content_hash``/``retrieved_at``
  values match what that real record actually says? A format-valid
  ``source_id`` naming a ``Source`` that does not exist in the context is
  now **INCOMPLETE** (data not yet available to confirm -- the same
  precedent as an unresolved CIK); a ``source_id`` that DOES resolve but
  whose tier/hash/retrieved_at the evidence record misreports is now
  **INVALID** (the record is not merely incomplete, it is wrong).
  ``supporting_fact_ids`` follows a DIFFERENT rule (see
  ``ProgramCandidateEvidence``'s own docstring below): a format-valid id
  that does not resolve in the context is **INVALID**, not INCOMPLETE --
  unlike a Source, which a caller may legitimately not have fetched yet,
  a claimed supporting Fact either already exists (Evidence Integrity has
  already run over it) or the claim itself is simply false.

``EvidenceValidationContext`` is a plain, frozen, explicitly-passed
container -- never global state, never a live ``Repository``/DB handle,
never a network call. See its own docstring for the exact contract a
caller must uphold (which this module cannot enforce in code, only
document, exactly like the pre-existing "never pass a raw, pre-Integrity
Fact" contract already implicit in Phase 4.3B).

``SourceTier`` is always copied verbatim from wherever a record's evidence
came from (a ``schemas.fact.Source.tier``) -- never inferred here from
``DocumentAuthority``, open-access status, peer-review status, or
``ContentKind``. Phase 4.3A-correction-1 documented, with evidence, that an
earlier version of ``research/literature_evidence_projection.py`` got this
exact inference wrong (hard-coded ``SourceTier.TIER_2`` for "peer-reviewed"
literature) and was corrected back to copying ``Document.tier`` verbatim,
which for every PubMed/Europe PMC ``Document`` this repository's adapters
build today is ``SourceTier.UNKNOWN``. This module's referential tier
check (``evidence.source_tier == source.tier``) never upgrades or
special-cases ``UNKNOWN`` -- two ``UNKNOWN`` values compare equal, exactly
as any other tier pair does, so a genuinely ``UNKNOWN``-tier literature
Source still validates cleanly without this module ever promoting it.

Phase 4.3B (and this correction) implement ONLY this offline contract
layer: the frozen evidence types, their structural AND referential
validation, and a pure constructor from ``collectors.clinicaltrials.
parse_study()``'s existing output shape. No Pipeline/CLI/Agent wiring, no
HTTP, no LLM. See ``scoring/program_identity_resolution.py``'s module
docstring for the future (not-yet-implemented) 2-pass Pipeline design
these types exist to support.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ..schemas.enums import UNKNOWN, SourceTier, StrEnum
from ..schemas.fact import Fact, Source
from .identifier_validation import validate_strict_nct_id, validate_strict_pmid

#: The exact shape ``schemas.fact.make_fact_id`` produces (``fact_`` +
#: 20 lowercase hex characters) -- the FORMAT this module requires before
#: even attempting a referential lookup. Passing this check alone is never
#: sufficient (see module docstring): a format-valid but non-existent id
#: is INVALID once checked against an ``EvidenceValidationContext``.
_FACT_ID_RE = re.compile(r"^fact_[0-9a-f]{20}$")


class EvidenceValidationOutcome(StrEnum):
    """Phase 4.3B requirement 3: a structured evidence record's own
    schema-validation result -- never a bare bool. ``VALID`` is the only
    outcome a resolver (``program_identity_resolution.py``) may act on;
    ``INVALID``/``INCOMPLETE`` records are excluded before resolution, same
    as a quarantined ``Source`` is excluded before Evidence Integrity
    (Phase 4.2B's precedent)."""

    #: Every required field is present, well-formed, AND (Correction 1)
    #: every reference it makes into the supplied ``EvidenceValidationContext``
    #: resolves to a real record whose own tier/hash/retrieved_at agree.
    VALID = "VALID"
    #: A field is present but self-contradictory/malformed (format), OR
    #: (Correction 1) a reference resolves to a real record whose own
    #: tier/content_hash/retrieved_at CONTRADICTS what the evidence record
    #: claims, OR a ``supporting_fact_ids`` entry is format-valid but does
    #: not exist in ``verified_facts_by_id`` -- this is not "missing data",
    #: it is wrong or fabricated data.
    INVALID = "INVALID"
    #: Structurally well-formed but a REQUIRED field is still ``UNKNOWN``/
    #: unset, OR (Correction 1) a format-valid ``source_id`` does not (yet)
    #: resolve in the supplied context -- distinct from INVALID exactly as
    #: CLAUDE.md distinguishes "not searched" from "searched and found
    #: nothing": this record is not wrong, it is simply not yet backed by
    #: data the caller has actually supplied.
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class EvidenceValidationResult:
    outcome: EvidenceValidationOutcome
    reasons: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.outcome is EvidenceValidationOutcome.VALID


@dataclass(frozen=True)
class EvidenceValidationContext:
    """A pure, immutable, EXPLICITLY-supplied read-only view onto the
    Source/Fact records a caller has already obtained through the real
    pipeline (``Repository``, ``EvidenceIntegrityAgent``, etc.) -- never a
    live ``Repository``/DB handle, never a network call, never global
    state of any kind. A resolver call passes one of these in by value;
    nothing in this module (or ``program_identity_resolution.py``) ever
    reaches out to fetch, refresh, or re-derive anything from it.

    Both mappings are copied into an immutable ``MappingProxyType`` in
    ``__post_init__`` -- mutating the dict a caller originally passed in
    after construction has no effect on this context, and the context
    itself exposes no mutation API.

    **Caller contract (documented here because this module has no way to
    enforce it in code -- it never reads a Repository or a live run at
    all):**

    * ``sources_by_id`` must contain ONLY ``Source`` records the caller
      has confirmed are actually persisted and NOT quarantined.
      ``orchestrator.pipeline.Pipeline.run()``'s own quarantine step
      (``Repository.save_sources`` returning a ``QuarantinedSource`` list,
      Phase 4.2B's precedent) already names which sources were rejected --
      a real caller must exclude those before constructing this context.
      A ``source_id`` this module cannot find here is treated as "does not
      exist" (INCOMPLETE), which is also the CORRECT outcome for a
      quarantined source a caller correctly excluded.
    * ``verified_facts_by_id`` must contain ONLY ``Fact`` records that
      have actually been assessed by a real ``EvidenceIntegrityAgent``
      pass (``Fact.verified_status``/``evidence_class`` actually set by
      ``_assess()``, not a raw, pre-Integrity ``Fact`` -- ``FactCollector
      Agent._to_fact``'s own output is NOT sufficient on its own). A raw
      Fact and an Integrity-assessed Fact are the same Python type
      (``schemas.fact.Fact``), so this module cannot detect the
      difference by inspecting a Fact object alone -- this is a pure
      caller obligation, exactly like the Source-quarantine contract
      above.

    A future Pipeline-connected caller (Phase 4.3C+, not this phase)
    would build one of these from, e.g., ``{s.source_id: s for s in
    result.bus.sources}``/``{f.fact_id: f for f in result.bus.facts}``
    taken AFTER Evidence Integrity and quarantine have already run --
    never before.
    """

    sources_by_id: Mapping[str, Source] = field(default_factory=dict)
    verified_facts_by_id: Mapping[str, Fact] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources_by_id", MappingProxyType(dict(self.sources_by_id)))
        object.__setattr__(
            self, "verified_facts_by_id", MappingProxyType(dict(self.verified_facts_by_id))
        )


#: A context with nothing in it -- every referential lookup against this
#: resolves to "not found", so every record checked against it lands at
#: best on INCOMPLETE (a source_id that cannot be confirmed) and never
#: VALID. Provided as a named constant so a caller who genuinely has no
#: Source/Fact catalog yet (e.g. constructing evidence before any
#: acquisition has run) can be explicit about it rather than passing an
#: ad hoc empty context each time.
EMPTY_VALIDATION_CONTEXT = EvidenceValidationContext()


@dataclass(frozen=True)
class CompanyIdentityEvidence:
    """A ticker's SEC-confirmed identity. ``cik``/``sec_official_name`` are
    ``None``/``UNKNOWN`` until a (future, Phase 4.3C+) caller has actually
    resolved them via SEC's own ``company_tickers.json`` -- mirroring
    ``collectors.sec_edgar.SecEdgarCollector.resolve_cik()``'s own
    ``(cik: int | None, resolved_name: str, outcome)`` contract exactly, so
    a future caller can construct this record directly from that
    function's return value without any additional guessing.

    ``explicitly_verified_aliases`` (Phase 4.3B Correction 1, requirement
    4, option B): retained as a field for a future phase, but
    ``resolve_program_identity`` in this correction NO LONGER reads it at
    all -- alias-based CONFIRMED is disabled in this phase. Building a
    referentially-checked ``CompanyAliasEvidence`` type (option A) was
    considered and deliberately deferred: this repository has no real
    alias source implemented yet (SEC's ``formerNames`` field is not
    currently fetched by anything in ``collectors/sec_edgar.py`` --
    confirmed in Phase 4.3A-correction-1's audit), so a referentially-
    checked alias type would have nothing genuine to validate against in
    this phase, and a plain ``tuple[str, ...]`` "verified" by naming
    convention alone is exactly the unenforced-trust gap this correction
    exists to close. CONFIRMED in this phase requires an EXACT match
    against ``sec_official_name`` only.
    """

    ticker: str
    cik: int | None = None
    sec_official_name: str = UNKNOWN
    explicitly_verified_aliases: tuple[str, ...] = ()
    source_id: str = UNKNOWN
    source_tier: SourceTier = SourceTier.UNKNOWN
    retrieved_at: str = UNKNOWN
    content_hash: str = UNKNOWN


@dataclass(frozen=True)
class ProgramCandidateEvidence:
    """One ClinicalTrials.gov study record's structured identity/sponsor/
    intervention/condition/date metadata -- built directly from
    ``collectors.clinicaltrials.parse_study()``'s own output dict (see
    ``build_program_candidate_evidence_from_parsed_study`` below), never
    from re-parsing a ``Fact.claim`` string a collector already turned that
    same data into.

    Each CT.gov date this module cares about is its own explicitly named
    field -- never a ``dict[str, str]`` -- so a future caller can never
    misattribute one date's meaning to a different field.

    ``first_posted_date`` is always ``UNKNOWN`` when built by
    ``build_program_candidate_evidence_from_parsed_study`` today:
    ``collectors.clinicaltrials.parse_study()`` does not currently read
    CT.gov API v2's ``statusModule.studyFirstPostDateStruct`` field.

    ---------------------------------------------------------------------
    ``supporting_fact_ids`` -- Phase 4.3B Correction 1, requirement 3
    ---------------------------------------------------------------------

    **Empty is explicitly ALLOWED.** A ``ProgramCandidateEvidence`` with
    ``supporting_fact_ids == ()`` is a valid, structured-Source-only
    metadata record: it is exactly as trustworthy as the ``Source`` it
    references (checked referentially -- see ``validate_program_candidate_
    evidence``), but it is NOT additionally confirmed by any Fact, and
    nothing in this module or ``program_identity_resolution.py`` ever
    reports it as "confirmed by a Fact" -- ``resolve_program_identity``'s
    own rationale strings never claim Fact-backing at all, empty or not.
    This choice (rather than treating empty as INCOMPLETE) was made
    because ``ProgramCandidateEvidence`` is built directly from
    ClinicalTrials.gov's own structured, Tier-1 API response -- it does
    not need a separately-derived Fact to be trustworthy, and requiring
    one would create a chicken-and-egg ordering problem for a future
    caller that resolves Program Identity before Facts have necessarily
    been derived from the same structured payload.

    When ``supporting_fact_ids`` IS non-empty, EVERY entry must (Correction
    1): (a) be format-valid (``fact_`` + 20 hex chars -- unchanged from
    Phase 4.3B), (b) actually exist in the caller-supplied
    ``EvidenceValidationContext.verified_facts_by_id``, (c) have
    ``Fact.source_id == ProgramCandidateEvidence.source_id`` (a Fact
    attributed to a DIFFERENT source can never "support" this candidate),
    and (d) itself rest on a Source present in
    ``EvidenceValidationContext.sources_by_id`` (i.e. not quarantined).
    Any failure among these makes the WHOLE record INVALID -- never a
    partial/best-effort acceptance of only the entries that happen to
    check out.
    """

    nct_id: str
    lead_sponsor: str = UNKNOWN
    collaborators: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    overall_status: str = UNKNOWN
    phases: tuple[str, ...] = ()
    primary_completion_date: str = UNKNOWN
    completion_date: str = UNKNOWN
    first_posted_date: str = UNKNOWN
    source_id: str = UNKNOWN
    source_tier: SourceTier = SourceTier.UNKNOWN
    retrieved_at: str = UNKNOWN
    content_hash: str = UNKNOWN
    supporting_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiteratureCandidateEvidence:
    """One PubMed/Europe PMC record's PMID and whatever NCT id(s) its OWN
    structured metadata explicitly names (e.g. PubMed's
    ``DataBankList/DataBank[DataBankName="ClinicalTrials.gov"]/
    AccessionNumber``, already extracted by
    ``collectors.literature._parse_nct_ids`` -- this module does not
    re-implement that extraction, only holds its result).

    ``nct_ids`` is deliberately NOT "every NCT id this article's text
    happens to mention" -- only ones the source's own structured metadata
    asserts. A drug/company name appearing in an abstract is never
    sufficient to populate this field (Phase 4.3B requirement 6).

    ``source_tier`` is preserved exactly as given -- for every literature
    record this repository can currently produce, that is
    ``SourceTier.UNKNOWN`` (Phase 4.3A-correction-1's finding). This
    module never upgrades it -- including in the Correction-1 referential
    check, where ``UNKNOWN == UNKNOWN`` compares equal like any other tier
    pair, so a genuinely UNKNOWN-tier record still validates cleanly.
    """

    pmid: str
    nct_ids: tuple[str, ...] = ()
    source_id: str = UNKNOWN
    source_tier: SourceTier = SourceTier.UNKNOWN
    retrieved_at: str = UNKNOWN
    content_hash: str = UNKNOWN


def _strict_nct_ok(value: str) -> bool:
    return validate_strict_nct_id(value) == value


def _strict_pmid_ok(value: str) -> bool:
    return validate_strict_pmid(value) == value


def _referential_source_check(
    *, source_id: str, claimed_tier: SourceTier, claimed_hash: str, claimed_retrieved_at: str,
    context: EvidenceValidationContext,
) -> tuple[bool, bool, list[str]]:
    """Shared by all three ``validate_*_evidence`` functions below.
    Returns ``(invalid, incomplete, reasons)`` for the referential Source
    check ONLY -- callers merge this with their own format-check results.
    """
    source = context.sources_by_id.get(source_id)
    if source is None:
        return False, True, [
            f"source_id {source_id!r} was not found in the validation context "
            "(not yet available, or quarantined)"
        ]
    reasons: list[str] = []
    if claimed_tier != source.tier:
        reasons.append(
            f"source_tier {claimed_tier} does not match Source {source_id!r}'s own tier {source.tier}"
        )
    if claimed_hash != source.content_hash:
        reasons.append(
            f"content_hash does not match Source {source_id!r}'s own content_hash"
        )
    if claimed_retrieved_at != source.retrieved_at:
        reasons.append(
            f"retrieved_at does not match Source {source_id!r}'s own retrieved_at"
        )
    return bool(reasons), False, reasons


def validate_company_identity_evidence(
    evidence: CompanyIdentityEvidence, context: EvidenceValidationContext,
) -> EvidenceValidationResult:
    """Format validation (unchanged from Phase 4.3B) plus (Correction 1)
    referential validation of ``source_id``/``source_tier``/
    ``content_hash``/``retrieved_at`` against ``context``."""
    reasons: list[str] = []
    invalid = False
    incomplete = False

    if not evidence.ticker or not evidence.ticker.strip():
        reasons.append("ticker is empty")
        invalid = True
    if evidence.cik is not None and evidence.cik <= 0:
        reasons.append(f"cik must be a positive integer, got {evidence.cik!r}")
        invalid = True
    if not evidence.source_id or evidence.source_id == UNKNOWN:
        reasons.append("source_id is missing")
        invalid = True

    if evidence.source_id and evidence.source_id != UNKNOWN:
        ref_invalid, ref_incomplete, ref_reasons = _referential_source_check(
            source_id=evidence.source_id, claimed_tier=evidence.source_tier,
            claimed_hash=evidence.content_hash, claimed_retrieved_at=evidence.retrieved_at,
            context=context,
        )
        reasons.extend(ref_reasons)
        invalid = invalid or ref_invalid
        incomplete = incomplete or ref_incomplete

    if invalid:
        return EvidenceValidationResult(EvidenceValidationOutcome.INVALID, tuple(reasons))

    if evidence.cik is None:
        reasons.append("cik not yet resolved")
        incomplete = True
    if evidence.sec_official_name == UNKNOWN:
        reasons.append("sec_official_name not yet resolved")
        incomplete = True
    if evidence.retrieved_at == UNKNOWN:
        reasons.append("retrieved_at is missing")
        incomplete = True
    if incomplete:
        return EvidenceValidationResult(EvidenceValidationOutcome.INCOMPLETE, tuple(reasons))

    return EvidenceValidationResult(EvidenceValidationOutcome.VALID)


def validate_program_candidate_evidence(
    evidence: ProgramCandidateEvidence, context: EvidenceValidationContext,
) -> EvidenceValidationResult:
    """Format validation (unchanged from Phase 4.3B) plus (Correction 1):
    referential validation of ``source_id``/``source_tier``/
    ``content_hash``/``retrieved_at``, and (when non-empty) referential
    validation of every ``supporting_fact_ids`` entry -- see the class's
    own docstring for the exact contract."""
    reasons: list[str] = []
    invalid = False
    incomplete = False

    if not evidence.nct_id or not _strict_nct_ok(evidence.nct_id):
        reasons.append(f"nct_id fails strict NCT validation: {evidence.nct_id!r}")
        invalid = True
    if not evidence.source_id or evidence.source_id == UNKNOWN:
        reasons.append("source_id is missing")
        invalid = True
    bad_fact_ids = tuple(fid for fid in evidence.supporting_fact_ids if not _FACT_ID_RE.match(fid))
    if bad_fact_ids:
        reasons.append(f"supporting_fact_ids contains non-fact_id-shaped value(s): {bad_fact_ids}")
        invalid = True

    if evidence.source_id and evidence.source_id != UNKNOWN:
        ref_invalid, ref_incomplete, ref_reasons = _referential_source_check(
            source_id=evidence.source_id, claimed_tier=evidence.source_tier,
            claimed_hash=evidence.content_hash, claimed_retrieved_at=evidence.retrieved_at,
            context=context,
        )
        reasons.extend(ref_reasons)
        invalid = invalid or ref_invalid
        incomplete = incomplete or ref_incomplete

    # supporting_fact_ids referential check -- empty is allowed (see class
    # docstring); only format-valid, non-empty entries are checked here
    # (a format-invalid entry already made the record INVALID above).
    format_valid_fact_ids = tuple(fid for fid in evidence.supporting_fact_ids if _FACT_ID_RE.match(fid))
    for fid in format_valid_fact_ids:
        fact = context.verified_facts_by_id.get(fid)
        if fact is None:
            reasons.append(f"supporting_fact_ids entry {fid!r} does not exist in verified_facts_by_id")
            invalid = True
            continue
        if fact.source_id != evidence.source_id:
            reasons.append(
                f"Fact {fid!r}.source_id ({fact.source_id!r}) does not match "
                f"candidate.source_id ({evidence.source_id!r})"
            )
            invalid = True
        if fact.source_id not in context.sources_by_id:
            reasons.append(
                f"Fact {fid!r} rests on Source {fact.source_id!r}, which is not present in "
                "the validation context (quarantined or unknown)"
            )
            invalid = True

    if invalid:
        return EvidenceValidationResult(EvidenceValidationOutcome.INVALID, tuple(reasons))

    if evidence.lead_sponsor == UNKNOWN:
        reasons.append("lead_sponsor not yet resolved")
        incomplete = True
    if evidence.retrieved_at == UNKNOWN:
        reasons.append("retrieved_at is missing")
        incomplete = True
    if incomplete:
        return EvidenceValidationResult(EvidenceValidationOutcome.INCOMPLETE, tuple(reasons))

    if not evidence.supporting_fact_ids:
        return EvidenceValidationResult(
            EvidenceValidationOutcome.VALID,
            ("structured Source metadata only -- not independently confirmed by a Fact",),
        )
    return EvidenceValidationResult(EvidenceValidationOutcome.VALID)


def validate_literature_candidate_evidence(
    evidence: LiteratureCandidateEvidence, context: EvidenceValidationContext,
) -> EvidenceValidationResult:
    """Format validation (unchanged from Phase 4.3B) plus (Correction 1)
    referential validation of ``source_id``/``source_tier``/
    ``content_hash``/``retrieved_at`` against ``context``. Tier is never
    upgraded: ``SourceTier.UNKNOWN`` on both sides compares equal like any
    other value."""
    reasons: list[str] = []
    invalid = False
    incomplete = False

    if not evidence.pmid or not _strict_pmid_ok(evidence.pmid):
        reasons.append(f"pmid fails strict PMID validation: {evidence.pmid!r}")
        invalid = True
    if not evidence.source_id or evidence.source_id == UNKNOWN:
        reasons.append("source_id is missing")
        invalid = True
    bad_ncts = tuple(n for n in evidence.nct_ids if not _strict_nct_ok(n))
    if bad_ncts:
        reasons.append(f"nct_ids contains value(s) failing strict NCT validation: {bad_ncts}")
        invalid = True

    if evidence.source_id and evidence.source_id != UNKNOWN:
        ref_invalid, ref_incomplete, ref_reasons = _referential_source_check(
            source_id=evidence.source_id, claimed_tier=evidence.source_tier,
            claimed_hash=evidence.content_hash, claimed_retrieved_at=evidence.retrieved_at,
            context=context,
        )
        reasons.extend(ref_reasons)
        invalid = invalid or ref_invalid
        incomplete = incomplete or ref_incomplete

    if invalid:
        return EvidenceValidationResult(EvidenceValidationOutcome.INVALID, tuple(reasons))

    if evidence.retrieved_at == UNKNOWN:
        reasons.append("retrieved_at is missing")
        incomplete = True
    if incomplete:
        return EvidenceValidationResult(EvidenceValidationOutcome.INCOMPLETE, tuple(reasons))
    return EvidenceValidationResult(EvidenceValidationOutcome.VALID)


def build_program_candidate_evidence_from_parsed_study(
    parsed: Mapping[str, object],
    *,
    source_id: str,
    source_tier: SourceTier,
    retrieved_at: str,
    content_hash: str = UNKNOWN,
    supporting_fact_ids: Sequence[str] = (),
) -> ProgramCandidateEvidence:
    """Pure constructor from ``collectors.clinicaltrials.parse_study()``'s
    existing output dict shape (Phase 4.3B requirement 7). Reads ONLY that
    dict's own keys -- never a ``Fact.claim`` string, never a regex
    re-extraction of anything ``_study_claims``/``raw_facts_from_study``
    already turned into a Fact. ``collectors/clinicaltrials.py`` itself is
    not imported here.

    ``source_id``/``source_tier``/``retrieved_at``/``content_hash`` are
    supplied by the caller (from the SAME real ``Source`` object a future
    caller would also place into an ``EvidenceValidationContext``), so a
    correctly-constructed record here already satisfies the referential
    check in ``validate_program_candidate_evidence`` by construction --
    this function never invents any of these four values itself.
    """

    def _tuple_of_str(value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(v) for v in value if v)

    return ProgramCandidateEvidence(
        nct_id=str(parsed.get("nct_id") or UNKNOWN),
        lead_sponsor=str(parsed.get("sponsor") or UNKNOWN),
        collaborators=_tuple_of_str(parsed.get("collaborators")),
        interventions=_tuple_of_str(parsed.get("interventions")),
        conditions=_tuple_of_str(parsed.get("conditions")),
        overall_status=str(parsed.get("status") or UNKNOWN),
        phases=_tuple_of_str(parsed.get("phases")),
        primary_completion_date=str(parsed.get("primary_completion") or UNKNOWN),
        completion_date=str(parsed.get("completion_date") or UNKNOWN),
        # See the class docstring: parse_study() does not currently read
        # studyFirstPostDateStruct, so this is never populated here.
        first_posted_date=UNKNOWN,
        source_id=source_id,
        source_tier=source_tier,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        supporting_fact_ids=tuple(supporting_fact_ids),
    )


__all__ = [
    "CompanyIdentityEvidence",
    "EMPTY_VALIDATION_CONTEXT",
    "EvidenceValidationContext",
    "EvidenceValidationOutcome",
    "EvidenceValidationResult",
    "LiteratureCandidateEvidence",
    "ProgramCandidateEvidence",
    "build_program_candidate_evidence_from_parsed_study",
    "validate_company_identity_evidence",
    "validate_literature_candidate_evidence",
    "validate_program_candidate_evidence",
]
