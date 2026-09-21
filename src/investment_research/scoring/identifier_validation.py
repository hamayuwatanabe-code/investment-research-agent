"""Phase 4.3B requirement 4/6: strict identifier validation and
deterministic company-name normalization for the offline Ticker-only
Program Resolution contract layer.

**Never confuse this module's regexes with ``scoring.program_resolution.
NCT_RE``.** That regex (``r"\\b(NCT[-A-Z0-9]{4,})\\b"``) is deliberately
LOOSE: it exists only to find an NCT-id-shaped substring already sitting
inside an already-verified ``Fact.claim`` string, for the unrelated purpose
of disambiguating which of several trial ids ALREADY present in evidence is
the thesis-relevant one. It is not a validator, and Phase 4.3A-correction-1
found it would be actively dangerous if reused to gate a live NCT ID lookup
or PubMed/ClinicalTrials.gov identifier passed into a future adapter --
``[-A-Z0-9]{4,}`` accepts far more than a real NCT id's fixed
``NCT\\d{8}`` shape. The functions below are the ONLY validators this
module considers safe to gate identifiers a future (Phase 4.3C+) caller
would send to NCBI/ClinicalTrials.gov -- self-contained (not imported from
``research/clinicaltrials_acquisition_adapter.py``/``research/
literature_pipeline_integration.py``, even though the regexes are
identical to ``NCT_ID_RE``/``PMID_RE`` there) so that this offline contract
layer never gains an import edge onto any HTTP-capable module, mirroring
``scoring/program_resolution.py``'s own "deliberately deterministic and
dependency-free" precedent.

No locale-dependent parsing, no fuzzy matching, anywhere in this module.
"""

from __future__ import annotations

import re

from ..schemas.enums import UNKNOWN

#: Mirrors ``research.clinicaltrials_acquisition_adapter.NCT_ID_RE``
#: exactly (``^NCT\\d{8}$``) -- duplicated, not imported; see module
#: docstring.
_STRICT_NCT_RE = re.compile(r"^NCT\d{8}$")

#: Mirrors ``research.literature_pipeline_integration.PMID_RE`` exactly
#: (``^\\d{1,9}$``) -- duplicated, not imported; see module docstring.
_STRICT_PMID_RE = re.compile(r"^\d{1,9}$")


def validate_strict_nct_id(value: str | None) -> str | None:
    """The normalized (upper-cased, stripped) NCT id, or ``None`` when
    ``value`` is missing, empty, or does not match ``^NCT\\d{8}$`` exactly
    after normalization -- never a guess, never a partial match. This is
    the ONLY form a future adapter may send to ClinicalTrials.gov/NCBI; a
    caller getting ``None`` back must stop (zero requests), never fall
    back to a looser interpretation."""
    if not value:
        return None
    candidate = value.strip().upper()
    return candidate if _STRICT_NCT_RE.match(candidate) else None


def validate_strict_pmid(value: str | None) -> str | None:
    """The normalized (stripped) PMID, or ``None`` when ``value`` is
    missing, empty, or not ``^\\d{1,9}$`` after stripping. PMIDs carry no
    letter casing to normalize (all-digit), and no locale-dependent
    parsing is ever applied."""
    if not value:
        return None
    candidate = value.strip()
    return candidate if _STRICT_PMID_RE.match(candidate) else None


#: A fixed allowlist of PURE legal-entity-form tokens -- deliberately never
#: includes an industry/sector descriptor like "PHARMACEUTICALS",
#: "THERAPEUTICS", "BIOSCIENCES", or "HOLDINGS": those words are frequently
#: part of what actually DISTINGUISHES one biotech company's name from
#: another's (e.g. "Acme Therapeutics" vs. "Acme Diagnostics" are two
#: different companies), so stripping them would increase, not decrease,
#: the misjoin risk Phase 4.3A-correction-1's audit flagged. Only a token
#: that names nothing but the entity's legal FORM is ever stripped.
_LEGAL_SUFFIX_TOKENS: frozenset[str] = frozenset(
    {
        "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
        "LTD", "LIMITED", "LLC", "LLP", "LP", "PLC", "AG", "SA", "NV", "SE", "GMBH", "KK",
    }
)

_PUNCTUATION_RE = re.compile(r"[.,]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_company_name(value: str) -> str:
    """Deterministic-only company-name normalization: upper-cases, strips
    periods/commas, collapses whitespace, and drops a TRAILING run of pure
    legal-entity-form tokens (see ``_LEGAL_SUFFIX_TOKENS``). Returns ``""``
    for ``UNKNOWN``/empty input.

    Never fuzzy (no edit distance, no phonetic matching), never
    LLM-assisted, never locale-dependent. Two names that normalize to the
    same string are treated as the same identity; two that do not are
    treated as different, full stop -- there is no partial-credit
    "probably the same company" outcome anywhere in this function. This is
    intentionally conservative: a false UNRESOLVED (two spellings of the
    same real company failing to match) is the safe failure direction
    (CLAUDE.md rule 7); a false match across two different companies is
    not.
    """
    if not value or value == UNKNOWN:
        return ""
    text = _PUNCTUATION_RE.sub("", value.upper())
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = text.split(" ") if text else []
    while tokens and tokens[-1] in _LEGAL_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


__all__ = [
    "normalize_company_name",
    "validate_strict_nct_id",
    "validate_strict_pmid",
]
