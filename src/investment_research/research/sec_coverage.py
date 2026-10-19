"""Per-accession coverage ledger for SEC primary documents (Phase 4.3M).

The question this module answers is not "what did the extractor find" but
"which recent SEC filings were READ, which were NOT, and why" -- one entry
per accession -- so that an unread important filing can never hide inside a
collection note.

Two parts:

* the **item policy**: what each 8-K item code is, whether this system can
  interpret it, and therefore whether leaving a filing of that kind unread
  withholds the Action;
* the **ledger entry** and its statuses.

Item policy (decided from the 8-K item *definitions*, never from any
company's actual disclosures -- no filing text is assumed):

========================  =======================================================
class                     rule
========================  =======================================================
MATERIAL                  body is read; an unread / partly read filing BLOCKS.
                          Items whose content the extractors can interpret:
                          1.01, 1.02, 2.03, 2.05, 3.01, 3.02, 4.01, 4.02, 7.01,
                          8.01 -- and a filing that states no item at all.
PERIODIC_RESULTS (2.02)   earnings releases. The NEWEST ``PERIODIC_RESULTS_READ``
                          are read and BLOCK if unread (they carry the current
                          financial position and pipeline commentary); older
                          ones are recorded NOT_READ_POLICY (superseded by the
                          newer releases and the periodic reports).
ADMINISTRATIVE            5.02, 5.03, 5.04, 5.05, 5.07, 5.08, 9.01: officers,
                          bylaws, votes, exhibits-only. Not read; recorded per
                          accession as NOT_READ_POLICY and reported as NOT
                          EXAMINED -- they do not block, and the report says so.
UNSUPPORTED               every other code (1.03, 1.04, 1.05, 2.01, 2.04, 2.06,
                          3.03, 5.01, 5.06, 6.xx, ...) and any code this table
                          does not know. No extractor exists for them, so
                          silence about them is not evidence of absence: the
                          filing is NOT interpreted and BLOCKS.
========================  =======================================================

A filing with several items takes the strictest class: UNSUPPORTED >
MATERIAL > PERIODIC_RESULTS > ADMINISTRATIVE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

MATERIAL_ITEMS = frozenset(
    {"1.01", "1.02", "2.03", "2.05", "3.01", "3.02", "4.01", "4.02", "7.01", "8.01"}
)
PERIODIC_RESULTS_ITEMS = frozenset({"2.02"})
ADMINISTRATIVE_ITEMS = frozenset({"5.02", "5.03", "5.04", "5.05", "5.07", "5.08", "9.01"})
#: How many of the newest 2.02 earnings releases are read (and block if unread).
PERIODIC_RESULTS_READ = 2


class PolicyClass(str, Enum):
    MATERIAL = "MATERIAL"
    PERIODIC_RESULTS = "PERIODIC_RESULTS"
    ADMINISTRATIVE = "ADMINISTRATIVE"
    UNSUPPORTED = "UNSUPPORTED"
    #: a 10-K / 10-Q selected by sec_periodic_reports
    PERIODIC_REPORT = "PERIODIC_REPORT"


def classify_items(items: tuple[str, ...]) -> PolicyClass:
    """The strictest class among a filing's items; a filing that states no
    item cannot be judged and is MATERIAL (it is read)."""
    if not items:
        return PolicyClass.MATERIAL
    classes = set()
    for item in items:
        if item in MATERIAL_ITEMS:
            classes.add(PolicyClass.MATERIAL)
        elif item in PERIODIC_RESULTS_ITEMS:
            classes.add(PolicyClass.PERIODIC_RESULTS)
        elif item in ADMINISTRATIVE_ITEMS:
            classes.add(PolicyClass.ADMINISTRATIVE)
        else:
            classes.add(PolicyClass.UNSUPPORTED)
    for strictest in (
        PolicyClass.UNSUPPORTED, PolicyClass.MATERIAL, PolicyClass.PERIODIC_RESULTS,
    ):
        if strictest in classes:
            return strictest
    return PolicyClass.ADMINISTRATIVE


class CoverageStatus(str, Enum):
    #: every document the policy asks for was read and classified
    READ = "READ"
    #: at least one document was read but something is unverified (an exhibit
    #: beyond the per-filing cap, an unclassifiable statement, a failed read)
    READ_PARTIAL = "READ_PARTIAL"
    UNREAD_BEYOND_CAP = "UNREAD_BEYOND_CAP"
    UNREAD_BUDGET = "UNREAD_BUDGET"
    UNREAD_FETCH_FAILED = "UNREAD_FETCH_FAILED"
    UNREAD_INVALID = "UNREAD_INVALID"
    UNREAD_NO_CLIENT = "UNREAD_NO_CLIENT"
    #: policy: not read, does not block (ADMINISTRATIVE; older PERIODIC_RESULTS)
    NOT_READ_POLICY = "NOT_READ_POLICY"
    #: policy: an item no extractor interprets -- not read, BLOCKS
    NOT_READ_UNSUPPORTED = "NOT_READ_UNSUPPORTED"
    #: a required periodic report is absent from the filing listing
    NOT_FOUND = "NOT_FOUND"


#: statuses that do not withhold the Action
NON_BLOCKING_STATUSES = frozenset({CoverageStatus.READ, CoverageStatus.NOT_READ_POLICY})


@dataclass(frozen=True)
class CoverageEntry:
    accession: str
    form: str
    filing_date: str
    items: tuple[str, ...]
    policy: PolicyClass
    status: CoverageStatus
    #: why (fixed, secret-free wording; never filing text)
    reason: str = ""
    documents_read: tuple[str, ...] = ()
    #: documents deliberately not read, e.g. ``EX-10.1 (agreement exhibits are not read)``
    documents_not_read: tuple[str, ...] = ()
    fact_count: int = 0
    #: sentences found but not turned into facts (periodic reports only)
    unclassified_statements: int = 0
    #: read order under the policy (1 = first); 0 = not a read candidate
    rank: int = 0

    @property
    def blocking(self) -> bool:
        return self.status not in NON_BLOCKING_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession, "form": self.form, "filing_date": self.filing_date,
            "items": list(self.items), "policy": self.policy.value, "status": self.status.value,
            "blocking": self.blocking, "reason": self.reason,
            "documents_read": list(self.documents_read),
            "documents_not_read": list(self.documents_not_read),
            "fact_count": self.fact_count,
            "unclassified_statements": self.unclassified_statements, "rank": self.rank,
        }


def entry_from_dict(raw: Any) -> CoverageEntry | None:
    """Inverse of :meth:`CoverageEntry.to_dict`; ``None`` for anything that is
    not a well-formed entry (a malformed snapshot record is never repaired)."""
    try:
        return CoverageEntry(
            accession=str(raw["accession"]), form=str(raw["form"]),
            filing_date=str(raw["filing_date"]), items=tuple(str(i) for i in raw["items"]),
            policy=PolicyClass(raw["policy"]), status=CoverageStatus(raw["status"]),
            reason=str(raw.get("reason", "")),
            documents_read=tuple(str(d) for d in raw.get("documents_read", ())),
            documents_not_read=tuple(str(d) for d in raw.get("documents_not_read", ())),
            fact_count=int(raw.get("fact_count", 0)),
            unclassified_statements=int(raw.get("unclassified_statements", 0)),
            rank=int(raw.get("rank", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


#: What clears a FINANCING CONDITION REQUIRES VERIFICATION blocker. Nothing in
#: code clears it: the blocker is withdrawn only by a person recording these
#: checks against the primary documents, and the run is re-done.
FINANCING_RELEASE_CONDITIONS = (
    "RELEASE ONLY by a person who has (1) read the executed financing agreement itself "
    "(the securities purchase / subscription agreement filed by the issuer on EDGAR, normally "
    "an EX-10.x exhibit to the same 8-K, which this system does not read) and recorded the "
    "exact closing condition, any waiver right and the amount at stake, AND (2) established "
    "from a later issuer SEC filing (8-K item 3.02/1.01, or a 10-Q/10-K subsequent-events or "
    "liquidity note) whether that closing occurred, AND (3) for a condition that depends on "
    "the FDA, read the FDA's own communication or a regulator-published record -- a statement "
    "by the issuer about the FDA remains a COMPANY_CLAIM and never satisfies this. "
    "No later issuer statement, no re-run and no resume removes this blocker automatically."
)


def ledger_consistency_reasons(
    summary: Any, *, prefix: str,
) -> list[str]:
    """Blocking reasons derived from a collector summary's LEDGER, so the
    Action Gate never rests on the error list alone: a malformed ledger row,
    or a blocking row that no error names, is itself a blocker. (A fresh run's
    ledger and errors agree by construction; this protects a restored or
    edited snapshot.)"""
    reasons: list[str] = []
    errors = [str(e) for e in (summary.get("errors") or [])]
    for raw in summary.get("coverage") or []:
        entry = entry_from_dict(raw)
        if entry is None:
            reasons.append(
                f"{prefix} a coverage ledger row of the restored record is malformed, so which "
                "filings it covered cannot be established"
            )
            continue
        # a NOT_FOUND row has no accession; its reason names the form instead
        key = entry.accession if entry.accession != "NONE" else f"{entry.form} --"
        if entry.blocking and not any(key in e for e in errors):
            reasons.append(
                f"{prefix} {entry.form} accession {entry.accession} filed {entry.filing_date} "
                f"is recorded {entry.status.value} ({entry.reason or 'no reason recorded'}) but "
                "no matching unverified reason exists; treated as UNVERIFIED"
            )
    return reasons
