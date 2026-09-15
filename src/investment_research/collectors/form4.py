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

Evidence Integrity (Phase 3E requirement 6): Form 4 is a REPORTING OWNER's
own statutory filing (``DocumentAuthority.STATUTORY_FILING``) -- it is
NEVER a company claim (the issuer did not author or file it), never an
independent third-party confirmation, and never converted into either.
What a successful parse confirms is limited to what the reporting owner
itself asserted on the form: a transaction occurred as coded, on the
stated date, at the stated price/share count, changing beneficial
ownership as stated. It confirms NONE of: the filer's trading intent,
whether a transaction was "bullish" or "bearish", or anything about the
issuer's business, regulatory standing, or clinical/financial condition
(Phase 3E requirement 5's last point).

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
* ``is_10b5_1_plan`` is ``True`` ONLY when a footnote this transaction
  actually references contains the literal substring ``"10b5-1"`` --
  ``None`` (never a guessed ``False``) whenever no such footnote reference
  exists. Never inferred from the transaction code, the filer's role, or
  anything else.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from enum import Enum
from typing import Any

from ..schemas.enums import UNKNOWN, FactCategory
from ..schemas.fact import RawFact, Source

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

    # Rule 10b5-1: True ONLY on an explicit footnote match -- never a
    # guessed False, never inferred from the transaction code (module
    # docstring / Phase 3E requirement 5).
    transaction["is_10b5_1_plan"] = (
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
        reporting_owners.append(
            {
                "reporting_owner_cik": _text(owner_id, "rptOwnerCik"),
                "reporting_owner_name": _text(owner_id, "rptOwnerName"),
                "is_director": _bool01(rel, "isDirector"),
                "is_officer": _bool01(rel, "isOfficer"),
                "officer_title": _text(rel, "officerTitle"),
                "is_ten_percent_owner": _bool01(rel, "isTenPercentOwner"),
                "is_other": _bool01(rel, "isOther"),
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
        "remarks": _text(root, "remarks"),
        "signatures": signatures,
        # Flat convenience fields for the (common) single-signer case --
        # UNKNOWN, never a guess, when there is more than one or none.
        "signature_name": signatures[0]["signature_name"] if len(signatures) == 1 else UNKNOWN,
        "signature_date": signatures[0]["signature_date"] if len(signatures) == 1 else UNKNOWN,
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
        f"is_10b5_1_plan={t['is_10b5_1_plan']}"
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
    "XmlShapeError",
    "check_ownership_xml_shape",
    "parse_ownership_document",
    "raw_facts_from_form4",
]
