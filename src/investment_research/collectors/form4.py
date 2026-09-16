"""SEC Form 4 (Section 16 ownership) XML parsing -- pure functions only.

Parses the real ``ownershipDocument`` XML schema SEC EDGAR's Form 4/4-A
filings use (the same schema underlies Form 3 and Form 5, distinguished by
``documentType``). Nothing here makes a network call; see
``research/form4_acquisition_adapter.py`` for the LOCATE/FETCH/PARSE
adapter that acquires the XML this module parses.

Real structure modeled (``xsd`` element names, unabbreviated):

    ownershipDocument
        schemaVersion, documentType, periodOfReport
        issuer (issuerCik, issuerName, issuerTradingSymbol)
        reportingOwner+ (one per co-filer -- NEVER assume exactly one)
            reportingOwnerId (rptOwnerCik, rptOwnerName)
            reportingOwnerRelationship (isDirector, isOfficer, officerTitle,
                                         isTenPercentOwner, isOther)
        nonDerivativeTable / nonDerivativeTransaction*
        derivativeTable / derivativeTransaction*
            each transaction: securityTitle, transactionDate,
            transactionCoding (transactionFormType, transactionCode,
            equitySwapInvolved), transactionAmounts (transactionShares,
            transactionPricePerShare, transactionAcquiredDisposedCode),
            postTransactionAmounts (sharesOwnedFollowingTransaction),
            ownershipNature (directOrIndirectOwnership, natureOfOwnership);
            derivative rows additionally carry conversionOrExercisePrice,
            exerciseDate, expirationDate, underlyingSecurity
            (underlyingSecurityTitle, underlyingSecurityShares)
        footnotes / footnote (id, text) -- referenced by a ``footnoteId``
            child element wherever a value may carry one
        remarks
        ownerSignature+ (signatureName, signatureDate)

Almost every leaf value is wrapped in a ``<value>`` child (SEC's schema
does this uniformly so a ``footnoteId`` can sit alongside it) -- this
module reads through that wrapper via ``_value()``, never assumes the
wrapper is present via ``_text()`` for the handful of fields that are NOT
value-wrapped (``documentType``, ``periodOfReport``, CIKs, names, dates on
issuer/reportingOwner/signature elements).

Evidence Integrity (Phase 3E requirement 6, CONFIRMED and given its own
EvidenceClass in Phase 3E.4): Form 4 is a REPORTING OWNER's own statutory
filing (``DocumentAuthority.REPORTING_PERSON_FILING`` -- a dedicated
value, distinct from the issuer's own ``STATUTORY_FILING``) -- it is
NEVER a company claim (the issuer did not author or file it), never an
independent third-party confirmation, and never converted into either.
Every Fact derived from a Form 4/4-A carries
``EvidenceClass.REPORTING_PERSON_STATUTORY_ASSERTION``,
``company_claim=False``, ``independent_confirmation=False`` -- enforced
in ``agents/evidence_integrity.py`` (classification) and
``research/escalation.py`` (the same mapping for a document fetched as a
would-be "confirming source"), and never in
``schemas.enums.DECISION_GRADE_CLASSES``, so a Form 4 fact can never
alone be decision-grade. What a successful parse confirms is limited to
what the reporting owner itself asserted on the form: a transaction
occurred as coded, on the stated date, at the stated price/share count,
changing beneficial ownership as stated. It confirms NONE of: the
filer's trading intent, whether a transaction was "bullish" or "bearish",
or anything about the issuer's business, regulatory standing, or
clinical/financial condition (Phase 3E requirement 5's last point).

Transaction semantics (Phase 3E requirement 5) -- all enforced by what this
module does NOT do:

* ``transaction_code`` is kept EXACTLY as SEC's own single/double-letter
  code (``P``, ``S``, ``A``, ``M``, ``F``, ``G``, ...) -- never translated
  into "buy"/"sell"/"bullish"/"bearish". A caller wanting that translation
  must do it explicitly, elsewhere, with its own documented mapping.
* ``acquired_or_disposed`` is kept as SEC's own literal ``"A"``/``"D"``
  code -- never treated as a synonym for "bought"/"sold" (an option
  EXERCISE, a tax-withholding disposition, and a gift can each carry
  either code without being a market purchase/sale at all).
* Derivative and non-derivative transactions are returned in two SEPARATE
  lists, mirroring the two separate XML tables -- never merged into one
  undifferentiated list.
* Multiple transactions in one filing are returned as multiple list
  entries, one each -- never summed, averaged, or otherwise collapsed
  into a single aggregate record.
* ``is_10b5_1_plan_footnote_fallback`` is ``True`` ONLY when a footnote
  this transaction actually references contains the literal substring
  ``"10b5-1"`` -- ``None`` (never a guessed ``False``) whenever no such
  footnote reference exists. Never inferred from the transaction code,
  the filer's role, or anything else. This is a LEGACY/FALLBACK signal,
  kept structurally separate from ``ten_b5_1_checkbox`` (Phase 3E.1
  requirement 2) -- see that field's own docstring below for why, and for
  the honest uncertainty about the real SEC schema element name.

Rule 10b5-1 checkbox (Phase 3E.1 requirement 2, CONFIRMED in Phase 3E.3):
SEC's Insider Trading Arrangements rule (Release No. 33-11138, effective
2023) added a checkbox indicating whether a transaction was made under a
Rule 10b5-1(c) trading arrangement. **``aff10b5One`` is now a CONFIRMED
real, document-level element name** -- a real Mac Live Smoke run (Phase
3E.3) against a real issuer observed ``aff10b5One`` as a direct child of
``ownershipDocument`` in 3 real, normal (non-amendment) Form 4 documents,
all Capture-Manifest-``VERIFIED`` with zero Evidence Integrity failures,
with a real ``0`` raw value parsing correctly to ``False``. This
confirmation covers NORMAL Form 4 only -- it has not been independently
re-confirmed against a real Form 4/A instance, though the two document
types share the identical ``ownershipDocument`` schema and there is no
structural reason to expect the checkbox to behave differently there.
``rule10b51PlanChecked`` remains in
``_DOCUMENT_LEVEL_10B5_1_CHECKBOX_PATHS`` only as a defensive secondary
candidate -- it has never been observed in any real capture and may be
removed in a future phase once that is itself confirmed one way or the
other. ``_read_checkbox`` still returns ``None`` (UNKNOWN) whenever
neither candidate element is present -- an absent checkbox is NEVER read
as ``False`` (a checkbox that cannot be found is unknown, not negative).
``ten_b5_1_plan_adoption_date`` is extracted ONLY from an explicit date
following "adopted ... on" wording in ``remarks`` or a referenced
footnote whose text also mentions "10b5-1" -- never inferred otherwise.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..schemas.enums import UNKNOWN, FactCategory
from ..schemas.fact import RawFact, Source

#: Document-level Rule 10b5-1 checkbox element names, checked in order.
#: ``aff10b5One`` is CONFIRMED against a real SEC capture (Phase 3E.3 --
#: see the module docstring); ``rule10b51PlanChecked`` remains only as an
#: unconfirmed defensive secondary candidate, never observed for real.
_DOCUMENT_LEVEL_10B5_1_CHECKBOX_PATHS: tuple[str, ...] = ("aff10b5One", "rule10b51PlanChecked")

_PLAN_ADOPTION_DATE_RE = re.compile(
    r"adopt(?:ed|ion)?(?:\s+the\s+plan)?\s+(?:on\s+)?"
    r"([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

#: Distinguishes a genuine Form 4/4-A from a Form 3 or Form 5 -- all three
#: share the identical ``ownershipDocument`` XML schema, differing only in
#: ``documentType`` (Phase 3E requirement 3: never conflate them). ``"4/A"``
#: is kept as its own distinct value throughout this module -- an amendment
#: is never silently treated as if it were the original Form 4.
VALID_FORM4_DOCUMENT_TYPES: frozenset[str] = frozenset({"4", "4/A"})


class XmlShapeError(str, Enum):
    """Distinguishes WHY a response body is not usable Form 4 XML -- never
    collapsed into one generic failure (Phase 3E requirement 10)."""

    MALFORMED = "MALFORMED"
    MISSING_OWNERSHIP_DOCUMENT = "MISSING_OWNERSHIP_DOCUMENT"
    WRONG_DOCUMENT_TYPE = "WRONG_DOCUMENT_TYPE"


def check_ownership_xml_shape(xml_text: str) -> tuple[XmlShapeError | None, str]:
    """A cheap, non-extracting gate: is this body well-formed XML, rooted
    at ``<ownershipDocument>``, declaring ``documentType`` "4" or "4/A"?

    Returns ``(None, document_type)`` on success, or
    ``(XmlShapeError, human_readable_reason)`` on any failure -- never
    raises. Used by the FETCH step so a malformed body, a non-ownership
    document, and a Form 3/5 masquerading as this adapter's target are
    each reported as their own distinct, diagnosable failure rather than
    one opaque "shape mismatch" (Phase 3E requirement 10).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return XmlShapeError.MALFORMED, f"malformed XML: {exc}"
    if root.tag != "ownershipDocument":
        return (
            XmlShapeError.MISSING_OWNERSHIP_DOCUMENT,
            f"root element is <{root.tag}>, not <ownershipDocument> -- not a Section 16 ownership filing body",
        )
    document_type = (root.findtext("documentType") or "").strip()
    if document_type not in VALID_FORM4_DOCUMENT_TYPES:
        return (
            XmlShapeError.WRONG_DOCUMENT_TYPE,
            f"documentType is {document_type!r}, not Form 4/4-A -- refusing to treat a Form "
            f"{document_type or 'UNKNOWN'} filing as a Form 4",
        )
    return None, document_type


#: Phase 3E.2.2: SEC serves an ownership filing's raw XML under a SECOND,
#: XSLT-styled display path -- observed against a real issuer (CIK
#: 1721484) as ``primaryDocument`` values shaped
#: ``xslF345X06/marketforms-73885.xml`` -- which renders as an HTML page
#: (the raw XML transformed client-side by an XSL stylesheet SEC serves
#: alongside it), NOT the raw XML bytes, even though it lives at a real,
#: SEC-served URL. The general ``sec_acquisition_adapters._validate_filename``
#: (a bare-filename-only check used by every OTHER SEC target) correctly
#: refuses this shape outright since it contains a ``/`` -- and is
#: deliberately NOT relaxed to accommodate it. This function is a
#: SEPARATE, Form4-ownership-specific normalizer: it recognizes ONLY the
#: two real shapes SEC's own submissions metadata uses for an ownership
#: filing's primaryDocument (a bare ``safe_basename.xml``, or
#: ``xslF345X<2 digits>/safe_basename.xml``), and returns the raw XML's
#: own basename separately from the XSL wrapper directory -- the basename
#: is what ``research/form4_acquisition_adapter.py`` cross-checks against
#: the accession's own ``index.json`` and uses to build the RAW XML URL
#: (never the XSL-wrapped path, which would fetch an HTML rendering, not
#: the ownership XML this system actually needs to parse).
class OwnershipPrimaryDocumentRejection(str, Enum):
    """Every distinguishable reason a ``primaryDocument`` value is refused
    -- never collapsed into one generic failure, mirroring
    ``XmlShapeError``'s own precedent (Phase 3E requirement 10)."""

    EMPTY = "EMPTY"
    CONTROL_CHARACTERS = "CONTROL_CHARACTERS"
    PERCENT_ENCODED = "PERCENT_ENCODED"
    BACKSLASH = "BACKSLASH"
    QUERY_OR_FRAGMENT = "QUERY_OR_FRAGMENT"
    SCHEME_OR_NETLOC = "SCHEME_OR_NETLOC"
    ABSOLUTE_PATH = "ABSOLUTE_PATH"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    CURRENT_DIR_SEGMENT = "CURRENT_DIR_SEGMENT"
    TOO_MANY_PATH_SEGMENTS = "TOO_MANY_PATH_SEGMENTS"
    UNRECOGNIZED_XSL_WRAPPER = "UNRECOGNIZED_XSL_WRAPPER"
    EMPTY_BASENAME = "EMPTY_BASENAME"
    NOT_XML = "NOT_XML"


@dataclass(frozen=True)
class NormalizedOwnershipPrimaryDocument:
    """The three pieces kept SEPARATE, never re-merged (Phase 3E.2.2
    requirement 1) -- ``original_primary_document`` is what SEC's own
    submissions metadata literally said (kept for diagnostics/audit only,
    NEVER used again to build a URL); ``xsl_wrapper_path`` is the XSL
    display-path's directory component, ``UNKNOWN`` for the bare-filename
    shape; ``normalized_xml_filename`` is the raw XML's own basename --
    the ONLY piece ever used to (a) cross-check against a real accession
    directory listing, and (b) build the raw XML URL."""

    original_primary_document: str
    xsl_wrapper_path: str
    normalized_xml_filename: str


#: Basename charset: letters/digits plus ``_-.``, must end in a literal
#: ``.xml`` -- deliberately conservative (real SEC ownership filenames
#: never need anything wider), checked only AFTER every categorical
#: rejection above already ruled out control characters, ``%``, ``\``,
#: ``?``/``#``, ``:``, and a leading ``.`` in the basename character class
#: would be ambiguous with the ``.`` path-segment check, so both are kept.
_OWNERSHIP_XML_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.xml$")
#: SEC's own XSLT-display-path convention -- "F345" names the combined
#: Form 3/4/5 ownership schema the stylesheet renders, "X" plus a 2-digit
#: version suffix. Exactly this shape, never a looser "starts with xsl"
#: guess.
_XSL_WRAPPER_DIR_RE = re.compile(r"^xslF345X\d{2}$")


def normalize_ownership_primary_document(
    primary_document: str,
) -> tuple[NormalizedOwnershipPrimaryDocument | None, OwnershipPrimaryDocumentRejection | None]:
    """Recognizes ONLY ``safe_basename.xml`` or
    ``xslF345X<2 digits>/safe_basename.xml`` -- anything else is refused
    with a specific, diagnosable reason, never guessed into one of the two
    shapes. Returns ``(normalized, None)`` on success or
    ``(None, rejection)`` on refusal; exactly one of the pair is
    non-``None``. Pure function -- no network, no filesystem, matching
    this module's own "pure functions only" scope. Does NOT relax
    ``research/sec_acquisition_adapters._validate_filename`` (the general
    bare-filename check every other SEC target still uses unchanged) --
    this is a separate, Form4-ownership-specific normalizer.
    """
    if not primary_document:
        return None, OwnershipPrimaryDocumentRejection.EMPTY
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in primary_document):
        return None, OwnershipPrimaryDocumentRejection.CONTROL_CHARACTERS
    if "%" in primary_document:
        # Catches every percent-encoded slash/backslash/traversal variant
        # (%2e%2e, %2f, %5c, ...) in one conservative rule -- a real SEC
        # ownership primaryDocument value never contains a percent sign.
        return None, OwnershipPrimaryDocumentRejection.PERCENT_ENCODED
    if "\\" in primary_document:
        return None, OwnershipPrimaryDocumentRejection.BACKSLASH
    if "?" in primary_document or "#" in primary_document:
        return None, OwnershipPrimaryDocumentRejection.QUERY_OR_FRAGMENT
    if ":" in primary_document:
        # Covers "scheme://", a bare "scheme:", and a Windows drive letter
        # alike -- a real ownership primaryDocument value never contains one.
        return None, OwnershipPrimaryDocumentRejection.SCHEME_OR_NETLOC
    if primary_document.startswith("/"):
        return None, OwnershipPrimaryDocumentRejection.ABSOLUTE_PATH
    if ".." in primary_document:
        return None, OwnershipPrimaryDocumentRejection.PATH_TRAVERSAL

    segments = primary_document.split("/")
    if any(segment == "." for segment in segments):
        return None, OwnershipPrimaryDocumentRejection.CURRENT_DIR_SEGMENT
    if len(segments) > 2:
        return None, OwnershipPrimaryDocumentRejection.TOO_MANY_PATH_SEGMENTS

    if len(segments) == 1:
        xsl_wrapper_path, basename = UNKNOWN, segments[0]
    else:
        wrapper_dir, basename = segments
        if not _XSL_WRAPPER_DIR_RE.match(wrapper_dir):
            return None, OwnershipPrimaryDocumentRejection.UNRECOGNIZED_XSL_WRAPPER
        xsl_wrapper_path = wrapper_dir

    if not basename:
        return None, OwnershipPrimaryDocumentRejection.EMPTY_BASENAME
    if not _OWNERSHIP_XML_BASENAME_RE.match(basename):
        return None, OwnershipPrimaryDocumentRejection.NOT_XML

    return (
        NormalizedOwnershipPrimaryDocument(
            original_primary_document=primary_document,
            xsl_wrapper_path=xsl_wrapper_path,
            normalized_xml_filename=basename,
        ),
        None,
    )


def _text(el: ET.Element | None, path: str, default: str = UNKNOWN) -> str:
    if el is None:
        return default
    found = el.find(path)
    if found is None or found.text is None:
        return default
    stripped = found.text.strip()
    return stripped or default


def _value(el: ET.Element | None, path: str, default: str = UNKNOWN) -> str:
    return _text(el, f"{path}/value", default)


def _bool01(el: ET.Element | None, path: str) -> bool | None:
    """SEC's ``0``/``1`` boolean convention -- ``None`` (never guessed
    ``False``) when the element itself is absent."""
    if el is None:
        return None
    found = el.find(path)
    if found is None or found.text is None:
        return None
    return found.text.strip() == "1"


def _footnote_ids(t_el: ET.Element) -> tuple[str, ...]:
    ids = {fid.get("id") for fid in t_el.iter("footnoteId") if fid.get("id")}
    return tuple(sorted(i for i in ids if i is not None))


def _read_checkbox(el: ET.Element, paths: tuple[str, ...]) -> bool | None:
    """SEC's 0/1 checkbox convention, tried against each candidate element
    name in turn. ``None`` -- never a guessed ``False`` -- when none of
    them are present (Phase 3E.1 requirement 2)."""
    for path in paths:
        found = el.find(path)
        if found is None:
            continue
        value_el = found.find("value")
        text = value_el.text if value_el is not None else found.text
        if text is None:
            continue
        text = text.strip()
        if text == "1":
            return True
        if text == "0":
            return False
    return None


def _relationship_fields_inconsistent(is_officer: bool | None, officer_title: str) -> bool:
    """Phase 3E.4: a real Mac Live Smoke capture found a reporting owner
    with ``isOfficer=0`` (False) yet a genuinely non-empty
    ``officerTitle``. Both raw values are kept EXACTLY as SEC reported
    them -- ``is_officer`` is never "corrected" to ``True`` by inferring
    it from ``officer_title`` being present, in either direction. This
    function only NAMES the disagreement as a diagnostic flag; it is
    never treated as a transaction fact, never used to set
    ``independent_confirmation``, and never fed back into either raw
    field (requirement 17)."""
    return is_officer is False and officer_title not in (UNKNOWN, "")


def _extract_plan_adoption_date(remarks: str, footnotes: dict[str, str]) -> str:
    """Only from text that BOTH mentions "10b5-1" AND states an explicit
    adoption date -- never inferred from a transaction date, a filing
    date, or anything else (Phase 3E.1 requirement 2)."""
    for text in (remarks, *footnotes.values()):
        if "10b5-1" not in text.lower():
            continue
        match = _PLAN_ADOPTION_DATE_RE.search(text)
        if match:
            return match.group(1)
    return UNKNOWN


#: An SEC accession number, as it would appear embedded in free-text
#: remarks (e.g. "This Form 4/A amends the Form 4 filed with accession
#: number 0001112223-26-000001."). Same shape as sec_acquisition_
#: adapters._ACCESSION_RE, duplicated here deliberately: this module must
#: stay import-independent of research/ code (a pure parsing module), and
#: a dedicated test keeps the two patterns in sync.
_ACCESSION_IN_TEXT_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")


def extract_referenced_accession(text: str) -> str:
    """The FIRST SEC-accession-shaped substring in free text (typically a
    4/A's own ``remarks``) -- ``UNKNOWN`` if none is present. Purely a
    textual pattern match: this is evidence an amendment's remarks name a
    specific prior accession, never proof the two are actually related
    (Phase 3E.1 requirement 3) -- research/form4_acquisition_adapter.py's
    reconciliation logic still requires that referenced accession to
    genuinely be among the OTHER filings discovered for the same issuer
    before treating it as reconciled."""
    match = _ACCESSION_IN_TEXT_RE.search(text or "")
    return match.group(0) if match else UNKNOWN


def _parse_transaction(
    t_el: ET.Element, *, is_derivative: bool, footnotes: dict[str, str]
) -> dict[str, Any]:
    coding = t_el.find("transactionCoding")
    amounts = t_el.find("transactionAmounts")
    post = t_el.find("postTransactionAmounts")
    nature = t_el.find("ownershipNature")
    footnote_ids = _footnote_ids(t_el)

    transaction: dict[str, Any] = {
        "security_title": _value(t_el, "securityTitle"),
        "transaction_date": _value(t_el, "transactionDate"),
        "transaction_form_type": _text(coding, "transactionFormType"),
        # Kept EXACTLY as SEC's own code -- never translated (module
        # docstring / Phase 3E requirement 5).
        "transaction_code": _text(coding, "transactionCode"),
        "equity_swap_involved": _bool01(coding, "equitySwapInvolved"),
        "transaction_shares": _value(amounts, "transactionShares"),
        "transaction_price_per_share": _value(amounts, "transactionPricePerShare"),
        # Literal "A"/"D" -- never read as "bought"/"sold".
        "acquired_or_disposed": _value(amounts, "transactionAcquiredDisposedCode"),
        "shares_owned_following_transaction": _value(post, "sharesOwnedFollowingTransaction"),
        "direct_or_indirect_ownership": _value(nature, "directOrIndirectOwnership"),
        "nature_of_indirect_ownership": _value(nature, "natureOfOwnership"),
        "is_derivative": is_derivative,
        "footnote_ids": footnote_ids,
        "exercise_price": UNKNOWN,
        "exercise_date": UNKNOWN,
        "expiration_date": UNKNOWN,
        "underlying_security_title": UNKNOWN,
        "underlying_security_shares": UNKNOWN,
    }
    if is_derivative:
        underlying = t_el.find("underlyingSecurity")
        transaction["exercise_price"] = _value(t_el, "conversionOrExercisePrice")
        transaction["exercise_date"] = _value(t_el, "exerciseDate")
        transaction["expiration_date"] = _value(t_el, "expirationDate")
        transaction["underlying_security_title"] = _value(underlying, "underlyingSecurityTitle")
        transaction["underlying_security_shares"] = _value(underlying, "underlyingSecurityShares")

    # Rule 10b5-1 LEGACY/FALLBACK signal: True ONLY on an explicit
    # footnote match -- never a guessed False, never inferred from the
    # transaction code. Structurally separate from the document-level
    # ten_b5_1_checkbox field (Phase 3E.1 requirement 2) -- never
    # conflated with it.
    transaction["is_10b5_1_plan_footnote_fallback"] = (
        any("10b5-1" in footnotes.get(fid, "").lower() for fid in footnote_ids) if footnote_ids else None
    )
    return transaction


def parse_ownership_document(xml_text: str) -> dict[str, Any] | None:
    """Full field extraction. Returns ``None`` if the body is not a
    well-formed ``<ownershipDocument>`` declaring ``documentType`` "4" or
    "4/A" -- callers that already ran ``check_ownership_xml_shape`` should
    not normally see ``None`` here, but this function never trusts that
    check blindly; it re-validates independently.

    Every field defaults to ``UNKNOWN`` (or ``None`` for a genuinely
    absent SEC 0/1 boolean) when the source XML omits it -- never guessed
    (Phase 3E requirement 4).
    """
    shape_error, document_type_or_reason = check_ownership_xml_shape(xml_text)
    if shape_error is not None:
        return None
    document_type = document_type_or_reason

    root = ET.fromstring(xml_text)  # already proven well-formed above
    issuer = root.find("issuer")
    issuer_cik = _text(issuer, "issuerCik")
    issuer_name = _text(issuer, "issuerName")
    issuer_ticker = _text(issuer, "issuerTradingSymbol")

    reporting_owners: list[dict[str, Any]] = []
    for ro in root.findall("reportingOwner"):
        owner_id = ro.find("reportingOwnerId")
        rel = ro.find("reportingOwnerRelationship")
        is_officer = _bool01(rel, "isOfficer")
        officer_title = _text(rel, "officerTitle")
        reporting_owners.append(
            {
                "reporting_owner_cik": _text(owner_id, "rptOwnerCik"),
                "reporting_owner_name": _text(owner_id, "rptOwnerName"),
                "is_director": _bool01(rel, "isDirector"),
                # Kept EXACTLY as SEC reported -- never "corrected" from
                # officer_title (Phase 3E.4 requirement 15).
                "is_officer": is_officer,
                "officer_title": officer_title,
                "is_ten_percent_owner": _bool01(rel, "isTenPercentOwner"),
                "is_other": _bool01(rel, "isOther"),
                # Diagnostic only -- never a transaction fact, never used
                # to set independent_confirmation (requirement 16/17).
                "relationship_fields_inconsistent": _relationship_fields_inconsistent(is_officer, officer_title),
            }
        )

    footnotes: dict[str, str] = {
        footnote_id: (fn.text or "").strip()
        for fn in root.findall("footnotes/footnote")
        if (footnote_id := fn.get("id")) is not None
    }

    non_derivative = [
        _parse_transaction(t, is_derivative=False, footnotes=footnotes)
        for t in root.findall("nonDerivativeTable/nonDerivativeTransaction")
    ]
    derivative = [
        _parse_transaction(t, is_derivative=True, footnotes=footnotes)
        for t in root.findall("derivativeTable/derivativeTransaction")
    ]

    signatures = [
        {"signature_name": _text(sig, "signatureName"), "signature_date": _text(sig, "signatureDate")}
        for sig in root.findall("ownerSignature")
    ]

    remarks = _text(root, "remarks")
    return {
        "document_type": document_type,
        "period_of_report": _text(root, "periodOfReport"),
        "issuer_name": issuer_name,
        "issuer_cik": issuer_cik,
        "issuer_ticker": issuer_ticker,
        "reporting_owners": reporting_owners,
        "non_derivative_transactions": non_derivative,
        "derivative_transactions": derivative,
        "footnotes": footnotes,
        "remarks": remarks,
        "signatures": signatures,
        # Flat convenience fields for the (common) single-signer case --
        # UNKNOWN, never a guess, when there is more than one or none.
        "signature_name": signatures[0]["signature_name"] if len(signatures) == 1 else UNKNOWN,
        "signature_date": signatures[0]["signature_date"] if len(signatures) == 1 else UNKNOWN,
        # Document-level Rule 10b5-1 checkbox -- see module docstring for
        # the honest uncertainty about the real SEC element name. None
        # (UNKNOWN) whenever absent, never a guessed False.
        "ten_b5_1_checkbox": _read_checkbox(root, _DOCUMENT_LEVEL_10B5_1_CHECKBOX_PATHS),
        "ten_b5_1_plan_adoption_date": _extract_plan_adoption_date(remarks, footnotes),
        # A 4/A's own remarks naming a prior accession -- a textual clue
        # only, never proof of relatedness on its own (Phase 3E.1
        # requirement 3; see research/form4_acquisition_adapter.py's
        # reconciliation logic for how this is actually used).
        "remarks_referenced_accession": extract_referenced_accession(remarks),
        # Phase 3E.4: CONFIRMED against a real Form 4/A capture (a real
        # document-level <dateOfOriginalSubmission><value>...</value>
        # element, direct child of ownershipDocument) -- read directly,
        # UNKNOWN when genuinely absent (a normal Form 4 need not carry
        # it at all). Never used, on its own, to determine an amendment's
        # original accession (requirement 19) -- that determination stays
        # exclusively remarks-based, in
        # research/form4_acquisition_adapter.py's own reconciliation
        # logic, which this field does not feed.
        "date_of_original_submission": _value(root, "dateOfOriginalSubmission"),
    }


def _relationship_claim(document_type: str, issuer_name: str, owner: dict[str, Any]) -> str:
    return (
        f"Form {document_type} reporting owner {owner['reporting_owner_name']} "
        f"relationship to {issuer_name}: is_director={owner['is_director']}, "
        f"is_officer={owner['is_officer']} (officer_title={owner['officer_title']}), "
        f"is_ten_percent_owner={owner['is_ten_percent_owner']}, is_other={owner['is_other']}"
    )


def _transaction_claim(document_type: str, issuer_name: str, kind: str, t: dict[str, Any]) -> str:
    # A plain factual statement only -- the claim text itself carries no
    # interpretation or disclaimer about what the codes mean (that
    # explanation belongs in this module's docstring/comments, never in a
    # RawFact.claim, which is Fact-Collector output and must stay
    # evaluation-free by construction).
    return (
        f"Form {document_type} {kind} transaction for {issuer_name}: "
        f"security_title={t['security_title']}, transaction_date={t['transaction_date']}, "
        f"transaction_code={t['transaction_code']}, acquired_or_disposed={t['acquired_or_disposed']}, "
        f"transaction_shares={t['transaction_shares']}, "
        f"transaction_price_per_share={t['transaction_price_per_share']}, "
        f"shares_owned_following_transaction={t['shares_owned_following_transaction']}, "
        f"direct_or_indirect_ownership={t['direct_or_indirect_ownership']}, "
        f"is_10b5_1_plan_footnote_fallback={t['is_10b5_1_plan_footnote_fallback']}"
    )


def raw_facts_from_form4(
    ticker: str,
    parsed: dict[str, Any],
    source: Source,
    *,
    collector: str = "form4",
    document_id: str | None = None,
) -> list[RawFact]:
    """One RawFact per reporting-owner relationship, and one RawFact per
    transaction -- never merged, never aggregated (Phase 3E requirement 5).

    ``company_claim`` is always ``False``: a Form 4 is the REPORTING
    OWNER's own statutory filing, never a statement the issuer itself
    made, however primary the venue it was filed in (Phase 3E requirement
    6; CLAUDE.md rule 2's "fact carries an EvidenceClass by TYPE, not
    convention" -- this is the ``company_claim`` half of that split for a
    filer who is emphatically not the company).
    """
    issuer_name = parsed["issuer_name"]
    document_type = parsed["document_type"]
    facts: list[RawFact] = []

    for owner in parsed["reporting_owners"]:
        facts.append(
            RawFact(
                ticker=ticker.upper(),
                category=FactCategory.INSIDER,
                claim=_relationship_claim(document_type, issuer_name, owner),
                source=source,
                value=owner["reporting_owner_name"],
                unit="reporting_owner_relationship",
                company_claim=False,
                collector=collector,
                document_id=document_id,
            )
        )

    for kind, transactions in (
        ("non_derivative", parsed["non_derivative_transactions"]),
        ("derivative", parsed["derivative_transactions"]),
    ):
        for t in transactions:
            facts.append(
                RawFact(
                    ticker=ticker.upper(),
                    category=FactCategory.INSIDER,
                    claim=_transaction_claim(document_type, issuer_name, kind, t),
                    source=source,
                    value=t["transaction_code"],
                    unit=f"{kind}_transaction",
                    company_claim=False,
                    collector=collector,
                    document_id=document_id,
                )
            )
    return facts


__all__ = [
    "VALID_FORM4_DOCUMENT_TYPES",
    "NormalizedOwnershipPrimaryDocument",
    "OwnershipPrimaryDocumentRejection",
    "XmlShapeError",
    "check_ownership_xml_shape",
    "extract_referenced_accession",
    "normalize_ownership_primary_document",
    "parse_ownership_document",
    "raw_facts_from_form4",
]
