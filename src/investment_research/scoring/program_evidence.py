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
``tuple``, never a ``list``/``dict`` -- Phase 4.3B requirement 2 ("frozen
dataclass内にmutableなdict/listを置かない"). ``ProgramCandidateEvidence.
supporting_fact_ids`` may only ever name Facts that genuinely exist
(enforced structurally here by requiring each id to match
``schemas.fact.make_fact_id``'s own output shape, ``fact_`` + 20 lowercase
hex characters -- never invented).

``SourceTier`` is always copied verbatim from wherever a record's evidence
came from (a ``schemas.fact.Source.tier``) -- never inferred here from
``DocumentAuthority``, open-access status, peer-review status, or
``ContentKind``. Phase 4.3A-correction-1 documented, with evidence, that an
earlier version of ``research/literature_evidence_projection.py`` got this
exact inference wrong (hard-coded ``SourceTier.TIER_2`` for "peer-reviewed"
literature) and was corrected back to copying ``Document.tier`` verbatim,
which for every PubMed/Europe PMC ``Document`` this repository's adapters
build today is ``SourceTier.UNKNOWN``. This module reproduces that same
discipline for its own evidence types: nothing here ever assigns or
upgrades a ``SourceTier`` -- callers construct records with whatever tier
their own upstream Source already carries.

Phase 4.3B implements ONLY this offline contract layer: the frozen
evidence types, their structural validation, and a pure constructor from
``collectors.clinicaltrials.parse_study()``'s existing output shape. No
Pipeline/CLI/Agent wiring, no HTTP, no LLM. See
``scoring/program_identity_resolution.py``'s module docstring for the
future (not-yet-implemented) 2-pass Pipeline design these types exist to
support.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..schemas.enums import UNKNOWN, SourceTier, StrEnum
from .identifier_validation import validate_strict_nct_id, validate_strict_pmid

#: The exact shape ``schemas.fact.make_fact_id`` produces (``fact_`` +
#: 20 lowercase hex characters) -- the only way this module will accept a
#: ``supporting_fact_ids`` entry as structurally plausible. This does NOT
#: prove the id refers to a Fact that actually exists in a given run (this
#: module never reads a Fact database or bus) -- only that it is not an
#: obviously invented string. A real existence check is a Pipeline-
#: connected caller's job (Phase 4.3C+, not this phase).
_FACT_ID_RE = re.compile(r"^fact_[0-9a-f]{20}$")


class EvidenceValidationOutcome(StrEnum):
    """Phase 4.3B requirement 3: a structured evidence record's own
    schema-validation result -- never a bare bool. ``VALID`` is the only
    outcome a resolver (``program_identity_resolution.py``) may act on;
    ``INVALID``/``INCOMPLETE`` records are excluded before resolution, same
    as a quarantined ``Source`` is excluded before Evidence Integrity
    (Phase 4.2B's precedent)."""

    #: Every required field is present and well-formed.
    VALID = "VALID"
    #: A field is present but self-contradictory/malformed (e.g. a
    #: ``nct_id`` that does not match strict ``NCT\\d{8}`` shape, an empty
    #: ``source_id``) -- this is not "missing data", it is wrong data.
    INVALID = "INVALID"
    #: Structurally well-formed but a REQUIRED field is still ``UNKNOWN``/
    #: unset (e.g. a CIK not yet resolved, a sponsor not yet known) --
    #: distinct from INVALID exactly as CLAUDE.md distinguishes "not
    #: searched" from "searched and found nothing": this record is not
    #: wrong, it is simply not yet ready to support a CONFIRMED resolution.
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class EvidenceValidationResult:
    outcome: EvidenceValidationOutcome
    reasons: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.outcome is EvidenceValidationOutcome.VALID


@dataclass(frozen=True)
class CompanyIdentityEvidence:
    """A ticker's SEC-confirmed identity. ``cik``/``sec_official_name`` are
    ``None``/``UNKNOWN`` until a (future, Phase 4.3C+) caller has actually
    resolved them via SEC's own ``company_tickers.json`` -- mirroring
    ``collectors.sec_edgar.SecEdgarCollector.resolve_cik()``'s own
    ``(cik: int | None, resolved_name: str, outcome)`` contract exactly, so
    a future caller can construct this record directly from that
    function's return value without any additional guessing.

    ``explicitly_verified_aliases`` is not a plausibility list -- every
    entry must have been independently confirmed against a primary source
    (e.g. a future SEC ``formerNames`` lookup, once implemented) by
    whatever produced this record. This module has no way to enforce that
    provenance itself (it is not a source of aliases, only a container),
    so it is the CALLER's obligation. See ``program_identity_resolution.
    py``'s module docstring: fixture metadata aliases must never be passed
    here in a production context.
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
    misattribute one date's meaning to a different field by writing
    ``dates["primary_completion"]`` where ``dates["completion"]`` was
    meant, the same class of bug Phase 4.2B fixed for PubMed dates.

    ``first_posted_date`` is always ``UNKNOWN`` when built by
    ``build_program_candidate_evidence_from_parsed_study`` today:
    ``collectors.clinicaltrials.parse_study()`` does not currently read
    CT.gov API v2's ``statusModule.studyFirstPostDateStruct`` field (it
    reads ``studyFirstSubmitDate`` instead, a DIFFERENT date -- when a
    study became publicly visible vs. when the sponsor submitted it -- so
    this module never repurposes one for the other). Populating this field
    for real requires extending ``parse_study()`` itself, which Phase 4.3B
    explicitly does not do (see this repository's ``CLAUDE.md``: "never
    misattribute a date meant for one field... to a different field").
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
    #: Fact ids (already-persisted, Evidence-Integrity-checked Facts) this
    #: candidate's sponsor/intervention/condition claims are backed by, if
    #: any exist for this run. Empty when no Fact exists yet for this
    #: candidate -- never populated with an invented id (Phase 4.3B
    #: requirement 2). Structurally checked (``_FACT_ID_RE``) by
    #: ``validate_program_candidate_evidence`` below, but NOT existence-
    #: checked against a real Fact store, since this module never reads
    #: one.
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
    sufficient to populate this field (Phase 4.3B requirement 6: literature
    linking never happens on a drug-name match alone).

    ``source_tier`` is preserved exactly as given -- for every literature
    record this repository can currently produce, that is
    ``SourceTier.UNKNOWN`` (Phase 4.3A-correction-1's finding, confirmed
    against ``research/literature_evidence_projection.py`` and its own
    tests). This module never upgrades it.
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


def validate_company_identity_evidence(evidence: CompanyIdentityEvidence) -> EvidenceValidationResult:
    """Never lets a resolver see a record that is missing what it needs, or
    that is internally malformed, without saying which."""
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


def validate_program_candidate_evidence(evidence: ProgramCandidateEvidence) -> EvidenceValidationResult:
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

    return EvidenceValidationResult(EvidenceValidationOutcome.VALID)


def validate_literature_candidate_evidence(
    evidence: LiteratureCandidateEvidence,
) -> EvidenceValidationResult:
    reasons: list[str] = []
    invalid = False

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
    if invalid:
        return EvidenceValidationResult(EvidenceValidationOutcome.INVALID, tuple(reasons))

    if evidence.retrieved_at == UNKNOWN:
        return EvidenceValidationResult(
            EvidenceValidationOutcome.INCOMPLETE, ("retrieved_at is missing",)
        )
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
    not imported here (only this function's caller, in a future phase,
    would import ``parse_study`` directly) -- this keeps this module's own
    import graph free of any collector/HTTP-capable code, matching Phase
    4.3B requirement 8's isolation goal.

    ``source_id``/``source_tier``/``retrieved_at``/``content_hash`` are
    supplied by the caller (from the ``Source`` object
    ``ClinicalTrialsCollector.collect()``/``clinicaltrials_acquisition_
    adapter.py`` already built for this same study) rather than derived
    here, so this constructor never invents provenance ``parse_study()``'s
    own dict does not carry.
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
    "EvidenceValidationOutcome",
    "EvidenceValidationResult",
    "LiteratureCandidateEvidence",
    "ProgramCandidateEvidence",
    "build_program_candidate_evidence_from_parsed_study",
    "validate_company_identity_evidence",
    "validate_literature_candidate_evidence",
    "validate_program_candidate_evidence",
]
