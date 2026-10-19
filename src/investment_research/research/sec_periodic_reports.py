"""The latest 10-K and 10-Q bodies (Phase 4.3M).

8-Ks report events; the periodic reports carry the standing picture -- the
going-concern assessment, the liquidity note, listing status, subsequent
events. Before this module no 10-K / 10-Q body was ever read: the SEC
collector turned each into a metadata fact ("10-K filed <date>") and the
Action could be issued with the issuer's own annual and quarterly disclosures
unread. That was never "no problem found"; it was "not looked at".

Scope (``periodic-report-scope/v1``), decided from filing mechanics, not from
any company's content:

* the NEWEST ``10-K`` filed within ``FORM_WINDOWS['10-K']`` days (15 months),
  and the NEWEST ``10-Q`` filed AFTER that 10-K within ``FORM_WINDOWS['10-Q']``
  days -- required once more than ``TENQ_DUE_AFTER_10K_DAYS`` days have passed
  since the 10-K (a quarterly report is then due). A required report that the
  SEC listing does not show is NOT_FOUND and blocks: absence from the listing
  is not evidence that nothing was filed;
* an amendment (10-K/A, 10-Q/A) filed after the selected report is recorded
  NOT_READ_POLICY (not read, NOT EXAMINED -- it can restate) and does not block;
* the primary document and the index only (``GETS_PER_REPORT`` actual GETs per
  report, retries and redirects included); a body over ``MAX_BODY_BYTES`` is
  refused by the client, never truncated -- an unread report blocks;
* NOT covered: 20-F / 40-F / 6-K (foreign private issuers), 10-KT, registration
  statements and prospectuses (S-1/S-3/424B*), proxy statements, Form 4, SC 13D/G.
  For a filer whose annual report is not a 10-K the 10-K is NOT_FOUND and the
  run blocks -- unsupported, not "clean".

Extraction is the same deterministic extractor as for 8-Ks, but over a long
document only DEFINITE statements become facts (a trial result, a regulator's
reported view, a reported listing notice or going-concern doubt, an auditor
change, a financing condition). A sentence the extractor finds but cannot
classify is COUNTED in the ledger entry (``unclassified_statements``) and does
not block: a 10-K contains many such sentences and blocking on them would make
every annual report a permanent blocker. That is a residual risk, stated in
the README, not a claim that those sentences are fine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from ..collectors.base import CollectionResult
from ..schemas.enums import UNKNOWN, DocumentAuthority, FetchOutcome, Provenance
from .sec_acquisition_adapters import (
    _content_is_valid_body,
    _directory_index_json_url,
    _html_to_text,
    _resolve_document_url,
    _validate_accession,
    _validate_filename,
)
from .sec_coverage import CoverageEntry, CoverageStatus, PolicyClass
from .sec_live_smoke import MaxRequestsExceededError
from .sec_material_8k import (
    AMBIGUOUS_KINDS,
    FACT_UNIT_PERIODIC,
    FilingRead,
    _get,
    _parse_iso,
    extract_statements,
    filing_source,
    get_failure_status,
    statement_fact,
)

PERIODIC_COLLECTOR = "sec_periodic_reports"
UNVERIFIED_PERIODIC_PREFIX = "UNVERIFIED PERIODIC REPORT:"
PERIODIC_SCOPE_RULE = "periodic-report-scope/v1"
FORM_WINDOWS = {"10-K": 460, "10-Q": 200}
MAX_REPORTS = 2
#: index + primary document, each allowed one retry
GETS_PER_REPORT = 4
MAX_OUTBOUND_GETS = MAX_REPORTS * GETS_PER_REPORT
#: A quarterly report is due once this long after the annual report.
TENQ_DUE_AFTER_10K_DAYS = 110
#: 16 MiB: a report larger than this is refused by the client (never truncated).
MAX_BODY_BYTES = 16 * 1024 * 1024
_ORIGINAL_FORMS = ("10-K", "10-Q")
_AMENDMENT_FORMS = {"10-K": "10-K/A", "10-Q": "10-Q/A"}


def _scope_id() -> str:
    windows = ",".join(f"{form}:{days}" for form, days in sorted(FORM_WINDOWS.items()))
    return (
        f"{PERIODIC_SCOPE_RULE}|windows={windows}|max_reports={MAX_REPORTS}"
        f"|tenq_due_after_10k_days={TENQ_DUE_AFTER_10K_DAYS}|max_body_bytes={MAX_BODY_BYTES}"
        "|amendments=not_read|unclassified=counted"
    )


PERIODIC_SCOPE_ID = _scope_id()


def periodic_scope() -> dict[str, Any]:
    return {
        "id": PERIODIC_SCOPE_ID,
        "rule": PERIODIC_SCOPE_RULE,
        "forms": list(_ORIGINAL_FORMS),
        "windows_days": dict(FORM_WINDOWS),
        "max_reports": MAX_REPORTS,
        "max_outbound_gets": MAX_OUTBOUND_GETS,
        "max_body_bytes": MAX_BODY_BYTES,
        "tenq_due_after_10k_days": TENQ_DUE_AFTER_10K_DAYS,
        "not_covered": [
            "20-F / 40-F / 6-K / 10-KT filers (their annual report is not a 10-K)",
            "S-1 / S-3 / 424B* prospectuses, DEF 14A, Form 4, SC 13D/G bodies",
            "10-K/A and 10-Q/A amendments (recorded, not read)",
            "unclassifiable sentences inside a report (counted, not blocking)",
        ],
    }


@dataclass(frozen=True)
class ReportCandidate:
    accession: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str


@dataclass(frozen=True)
class PeriodicSelection:
    selected: list[ReportCandidate]
    #: dispositions decided without reading: NOT_FOUND (blocking) and amendments
    entries: list[CoverageEntry]


def select_periodic_reports(recent: Mapping[str, Any], *, today: date) -> PeriodicSelection:
    forms = recent.get("form") or []

    def at(key: str, index: int) -> str:
        values = recent.get(key) or []
        return str(values[index]) if index < len(values) and values[index] is not None else ""

    rows: dict[str, list[ReportCandidate]] = {}
    for index, form in enumerate(forms):
        form = str(form)
        if form not in (*_ORIGINAL_FORMS, *_AMENDMENT_FORMS.values()):
            continue
        filed = _parse_iso(at("filingDate", index))
        rows.setdefault(form, []).append(ReportCandidate(
            accession=at("accessionNumber", index) or UNKNOWN, form=form,
            filing_date=filed.isoformat() if filed else UNKNOWN,
            report_date=at("reportDate", index) or UNKNOWN,
            primary_document=at("primaryDocument", index),
        ))

    def newest_within(form: str, days: int, *, after: str | None = None) -> ReportCandidate | None:
        floor = today - timedelta(days=days)
        pool = [
            c for c in rows.get(form, [])
            if (d := _parse_iso(c.filing_date)) is not None and floor <= d <= today
            and (after is None or c.filing_date > after)
        ]
        return max(pool, key=lambda c: (c.filing_date, c.accession), default=None)

    selected: list[ReportCandidate] = []
    entries: list[CoverageEntry] = []

    def missing(form: str, why: str) -> None:
        entries.append(CoverageEntry(
            accession="NONE", form=form, filing_date=UNKNOWN, items=(),
            policy=PolicyClass.PERIODIC_REPORT, status=CoverageStatus.NOT_FOUND, reason=why,
        ))

    tenk = newest_within("10-K", FORM_WINDOWS["10-K"])
    if tenk is None:
        missing("10-K", (
            f"no 10-K filed within the last {FORM_WINDOWS['10-K']} days is listed (a recent "
            "listing, a foreign filer or a delinquent filer are indistinguishable here)"
        ))
    else:
        selected.append(tenk)
    tenq = newest_within("10-Q", FORM_WINDOWS["10-Q"], after=tenk.filing_date if tenk else None)
    tenk_filed = _parse_iso(tenk.filing_date) if tenk else None
    tenk_age = (today - tenk_filed).days if tenk_filed else None
    if tenq is not None:
        selected.append(tenq)
    elif tenk is not None and tenk_age is not None and tenk_age > TENQ_DUE_AFTER_10K_DAYS:
        missing("10-Q", (
            f"no 10-Q filed after the latest 10-K ({tenk.filing_date}) within the last "
            f"{FORM_WINDOWS['10-Q']} days is listed although one is due"
        ))
    for chosen in selected:
        for amendment in rows.get(_AMENDMENT_FORMS[chosen.form], []):
            if amendment.filing_date > chosen.filing_date:
                entries.append(CoverageEntry(
                    accession=amendment.accession, form=amendment.form,
                    filing_date=amendment.filing_date, items=(),
                    policy=PolicyClass.PERIODIC_REPORT, status=CoverageStatus.NOT_READ_POLICY,
                    reason="amendment filed after the selected report: not read and NOT EXAMINED "
                    "(an amendment can restate)",
                ))
    return PeriodicSelection(selected, entries)


def _unverified(c: ReportCandidate | None, form: str, why: str) -> str:
    who = f"{c.form} accession {c.accession} filed {c.filing_date}" if c else f"{form}"
    return (
        f"{UNVERIFIED_PERIODIC_PREFIX} {who} -- {why}; its content is UNVERIFIED and no action "
        "label can rest on the evidence without it"
    )


def _read_report(ticker: str, cik: int, c: ReportCandidate, http: Any) -> FilingRead:
    def fail(status: CoverageStatus, why: str) -> FilingRead:
        return FilingRead([], [_unverified(c, c.form, why)], [], status, why, [], [])

    if not _validate_accession(c.accession):
        return fail(CoverageStatus.UNREAD_INVALID, "the accession number is malformed")
    if not _validate_filename(c.primary_document):
        return fail(CoverageStatus.UNREAD_INVALID, "the primary document name is missing or unsafe")
    nodash = c.accession.replace("-", "")
    index, failure = _get(http, _directory_index_json_url(cik, nodash))
    if index is None:
        return fail(get_failure_status(failure), f"the filing index could not be read: {failure}")
    listing = ((index.json() or {}).get("directory") or {}).get("item") or []
    if c.primary_document not in {str(i.get("name", "")) for i in listing if isinstance(i, Mapping)}:
        return fail(CoverageStatus.UNREAD_INVALID, "the filing index does not list the primary document")
    url = _resolve_document_url(cik, nodash, c.primary_document)
    if url is None:
        return fail(CoverageStatus.UNREAD_INVALID, f"{c.primary_document!r} has no safe URL")
    fetched, failure = _get(http, url)
    if fetched is None:
        return fail(get_failure_status(failure), f"the report body was not read: {failure}")
    text = _html_to_text(fetched.text)
    if not _content_is_valid_body(text):
        return fail(
            CoverageStatus.UNREAD_INVALID,
            "the report body failed validation (too short, an index page, or an "
            "access-denial/rate-limit page)",
        )
    source = filing_source(
        form=c.form, accession=c.accession, filing_date=c.filing_date, doc_type=c.form,
        name=c.primary_document, url=fetched.url,
        event_date=c.report_date if c.report_date != UNKNOWN else c.filing_date,
    )
    facts = []
    unclassified = 0
    for statement in extract_statements(text):
        if statement.kind in AMBIGUOUS_KINDS:
            unclassified += 1
            continue
        facts.append(statement_fact(
            ticker, statement, source, DocumentAuthority.STATUTORY_FILING,
            collector=PERIODIC_COLLECTOR, unit=FACT_UNIT_PERIODIC,
        ))
    notes = [
        f"{c.form} {c.accession} filed {c.filing_date}: read {c.primary_document}; "
        f"{len(facts)} definite statement(s) extracted; {unclassified} unclassified sentence(s) "
        "counted, not blocking"
    ]
    return FilingRead(
        facts, [], notes, CoverageStatus.READ, "", [f"{c.form} {c.primary_document}"], [],
        unclassified,
    )


def collect_periodic_reports(
    ticker: str,
    *,
    cik: int | None,
    recent: Mapping[str, Any] | None,
    http: Any | None,
    today: date,
    unavailable_reason: str = "",
    older_listing_pages: int = 0,
) -> CollectionResult:
    """One ``CollectionResult`` for the latest 10-K / 10-Q. ``http`` is
    ``None`` exactly when no request may be made; a listing that was never
    obtained makes the whole scope UNVERIFIED."""
    out = CollectionResult(
        collector=PERIODIC_COLLECTOR, provenance=Provenance.LIVE, scope=periodic_scope(),
    )
    if cik is None or recent is None:
        out.outcome = FetchOutcome.ERROR
        out.errors.append(_unverified(None, "10-K / 10-Q", (
            "the SEC filing list was not obtained, so which annual and quarterly reports exist "
            "cannot be determined" + (f" ({unavailable_reason})" if unavailable_reason else "")
        )))
        return out
    selection = select_periodic_reports(recent, today=today)
    entries = list(selection.entries)
    out.notes.append(
        f"periodic report scope: newest 10-K (<= {FORM_WINDOWS['10-K']} days) and newest 10-Q after "
        f"it (<= {FORM_WINDOWS['10-Q']} days); {len(selection.selected)} selected. NOT covered: "
        "20-F/40-F/6-K filers, prospectuses, proxy statements, Form 4, amendments"
    )
    for entry in selection.entries:
        if entry.status is CoverageStatus.NOT_FOUND:
            out.errors.append(_unverified(None, entry.form, entry.reason))
    if older_listing_pages > 0:
        dates = [d for d in (_parse_iso(x) for x in (recent.get("filingDate") or [])) if d]
        floor = today - timedelta(days=FORM_WINDOWS["10-K"])
        if dates and min(dates) > floor:
            out.errors.append(_unverified(None, "10-K / 10-Q", (
                f"the SEC filing list reaches back only to {min(dates)} but the annual-report "
                f"window starts {floor}; {older_listing_pages} older listing page(s) are not "
                "fetched"
            )))
    for rank, c in enumerate(selection.selected, start=1):
        if http is None:
            why = unavailable_reason or "no SEC client available"
            out.errors.append(_unverified(c, c.form, why))
            entries.append(CoverageEntry(
                c.accession, c.form, c.filing_date, (), PolicyClass.PERIODIC_REPORT,
                CoverageStatus.UNREAD_NO_CLIENT, why, rank=rank,
            ))
            continue
        try:
            read = _read_report(ticker, cik, c, http)
        except MaxRequestsExceededError:  # pragma: no cover - _get already maps it
            read = FilingRead(
                [], [_unverified(c, c.form, "the GET budget was exhausted")], [],
                CoverageStatus.UNREAD_BUDGET, "the GET budget was exhausted", [], [],
            )
        out.raw_facts.extend(read.facts)
        out.sources.extend({f.source.source_id: f.source for f in read.facts}.values())
        out.errors.extend(read.reasons)
        out.notes.extend(read.notes)
        entries.append(CoverageEntry(
            c.accession, c.form, c.filing_date, (), PolicyClass.PERIODIC_REPORT, read.status,
            read.cause, tuple(read.documents_read), tuple(read.documents_not_read),
            len(read.facts), read.unclassified, rank,
        ))
    out.coverage = [e.to_dict() for e in sorted(
        entries, key=lambda e: (e.filing_date, e.accession), reverse=True,
    )]
    if out.errors:
        out.outcome = FetchOutcome.ERROR
    return out


def periodic_scope_blocking_reasons(
    collector_summaries: Sequence[Mapping[str, Any]],
) -> list[str]:
    """For a RESTORED collect snapshot: its periodic-report record is only
    usable if it was written under the CURRENT periodic scope. A missing,
    unreadable or different scope is refused, never reused or repaired (no
    request is made)."""
    reasons: list[str] = []
    for summary in collector_summaries:
        if summary.get("collector") != PERIODIC_COLLECTOR:
            continue
        scope = summary.get("scope")
        recorded = scope.get("id") if isinstance(scope, Mapping) else None
        if recorded == PERIODIC_SCOPE_ID:
            continue
        if not isinstance(scope, Mapping) or not scope:
            found = "no scope recorded"
        else:
            found = f"scope {recorded!r}" if isinstance(recorded, str) else "an unreadable scope"
        reasons.append(_unverified(None, "10-K / 10-Q", (
            f"the restored collection's periodic-report record was written under a different "
            f"selection scope ({found}; current scope {PERIODIC_SCOPE_ID}), so which annual and "
            "quarterly reports it covered cannot be established and it is not reused; no "
            "request was made to fill the gap"
        )))
    return reasons
