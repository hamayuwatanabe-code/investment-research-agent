"""Phase 3E.2: a limited, one-time Live Smoke tool that verifies Phase
3E.1's UNVERIFIED assumptions about Form 4/4-A's real SEC structure --
NOT a promotion of anything to ``LIVE_VERIFIED``, and NOT a production
acquisition path.

Three Phase 3E.1 assumptions were flagged as unverified because this
repository has no live network access:

1. An issuer's own ``data.sec.gov/submissions/CIK##########.json``
   genuinely includes Form 3/4/5 entries (a well-documented EDGAR
   behavior, never actually checked against a real response here).
2. The Rule 10b5-1 checkbox's real XML element name --
   ``collectors/form4.py`` currently guesses between two candidate names
   and has never seen a real capture.
3. Whether ``dateOfOriginalSubmission`` exists at all in a real filing,
   and under what element name.

This module exists ONLY to check those three things against a real
issuer's real submissions.json and real ownership XML, using the SAME
``Form4Adapter``/``AllowlistedHttpClient``/Capture Manifest machinery
Phase 3E.1/3D.4 already built and tested offline -- it adds no new
production code path, and nothing it does can change
``research/source_routing_catalog.py``'s ``ImplementationStatus`` values
(that catalog is never imported here).

**NEVER imported by ``cli.py``/``pipeline.py``/any production code path.**
Running this module makes REAL network calls; every test in
``tests/unit/test_form4_live_smoke_offline.py`` /
``tests/integration/test_form4_live_smoke_transport.py`` exercises this
module's logic against a fake transport or a local loopback test server
-- never the real internet.

Invoke via ``scripts/form4_live_smoke.py``. Requirements enforced here,
not just documented (mirroring ``sec_live_smoke.py`` exactly, since this
hits the same SEC EDGAR hosts under the same Fair Access policy):

* Only ``--issuer-cik`` is required. ``--accession``/``--primary-document``/
  a reporting-owner CIK are never accepted as input -- discovery is
  entirely issuer-driven, exactly like ``Form4Adapter`` itself (Phase
  3E.2 requirement 1).
* ``IRA_SEC_USER_AGENT`` is read from the environment at call time via
  ``sec_live_smoke.resolve_user_agent`` -- reused, never re-implemented.
  If it is unset, blank, or the placeholder, this module makes ZERO
  network calls.
* ``ALLOWED_HOSTS``/``AllowlistedHttpClient``/host-allowlist redirect
  re-validation are reused from ``sec_live_smoke.py`` verbatim -- this
  module hits the identical two hosts.
* A hard cap on real HTTP GETs, computed from ``--max-filings`` by
  ``compute_max_live_gets`` (see that function for the exact formula) --
  enforced by ``AllowlistedHttpClient`` itself.
* If the issuer's own submissions produce NO Form 4/4-A filings at all
  (before any lookback filtering), this is reported as
  ``DISCOVERY_NOT_SUPPORTED`` and the run stops -- never treated as a
  successful run with zero results, and NEVER falls back to guessing a
  different CIK (Phase 3E.2 requirement 2).
* The Rule 10b5-1 checkbox and ``dateOfOriginalSubmission`` diagnostics
  (``scan_rule_10b5_1_elements``/``scan_date_of_original_submission``)
  are independent, raw-XML tree walks -- they do NOT reuse
  ``collectors/form4.py``'s own two-candidate-name guess, and never
  substitute a footnote-derived value for an unconfirmed checkbox
  element (Phase 3E.2 requirement 3).
* ``candidates_found``/``fetched``/``not_fetched_due_to_cap`` are always
  reported as three separate numbers; when the last is nonzero, nothing
  in this module's output claims coverage is complete (Phase 3E.2
  requirement 4).
* Every real response is saved via the SAME ``AllowlistedHttpClient``/
  Capture Manifest ``sec_live_smoke.py`` already uses (``source="form4"``)
  -- ``analyze_capture`` re-analyzes a prior capture directory fully
  offline, never trusting a manifest integrity failure as evidence of
  anything (Phase 3E.2 requirement 5).
* No Anthropic API call, no LLM, no Web Search, no ``Pipeline.run()`` --
  this module imports none of them; ``anthropic_api_calls``/
  ``web_search_calls``/``external_llm_tokens`` are always 0 in the
  returned report.
* A single-run guard identical to ``sec_live_smoke.py``'s: a marker file
  is written ONLY after a real communication attempt began (never on a
  before-any-request refusal), and a second invocation refuses unless
  ``--force-rerun`` is passed.

Evidence semantics (Phase 3E.2 requirement 7): this module changes
NOTHING about how a Form 4 fact is classified -- see this module's own
report / the Phase 3E.2 completion report for the documented interim
treatment (``FactCategory.INSIDER`` -> ``EvidenceClass.COMPANY_CLAIM`` in
``agents/evidence_integrity.py``) and the deferred proposal for a
dedicated ``REPORTING_PERSON_STATUTORY_ASSERTION``-style EvidenceClass.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ..collectors.form4 import VALID_FORM4_DOCUMENT_TYPES, check_ownership_xml_shape
from ..collectors.sec_edgar import SUBMISSIONS_URL, cik_for_archives, normalize_cik
from ..schemas.enums import UNKNOWN, ResearchDomain
from ..schemas.fact import parse_iso_date
from .acquisition_executor import ExecutionContext
from .acquisition_planning import AcquisitionMethod
from .capture_manifest import (
    ManifestReadStatus,
    is_evidence_integrity_failure,
    read_manifest,
)
from .checks import SubjectScope
from .document_store import DocumentStore
from .form4_acquisition_adapter import (
    FORM4_ADAPTER_ID,
    MAX_SUBMISSIONS_PAGES,
    Form4Adapter,
    Form4IssuerReference,
)
from .sec_live_smoke import (
    _UNSET,
    ALLOWED_HOSTS,
    AllowlistedHttpClient,
    _classify_sec_url,
    _marker_state,
    _safe,
    _Unset,
    _valid_user_agent_or_none,
    _write_marker,
    resolve_user_agent,
)
from .source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    ImplementationStatus,
    StepKind,
    StepStatus,
    TargetKind,
)

#: LOCATE's own candidate-selection cap is set generously high for THIS
#: module's discovery pass -- ``Form4Adapter._locate`` still bounds the
#: real work done (pagination is capped at ``MAX_SUBMISSIONS_PAGES``
#: regardless of this value); the lookback-window filtering and the real
#: ``--max-filings`` fetch cap are both applied by THIS module, after
#: LOCATE returns, never inside the adapter (Phase 3E.2 requirement 4 --
#: keeping this filtering out of ``form4_acquisition_adapter.py`` means
#: its own Phase 3E.1 tests stay untouched).
DISCOVERY_CANDIDATE_CEILING = 1000

#: Default reporting window -- deliberately conservative (recent activity
#: only) since this is a one-time structure-verification run, not a
#: research pass. Always overridable via ``--lookback-days``.
DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_MAX_FILINGS = 3


def compute_max_live_gets(max_filings: int) -> int:
    """The hard live-GET cap for a given ``--max-filings``: 1 submissions
    GET, up to ``MAX_SUBMISSIONS_PAGES`` ``filings.files`` continuation
    pages (worst case -- an issuer whose ``files`` index happens to have
    that many entries), plus 2 GETs per fetched candidate (one directory
    index.json, one ownership XML body -- exactly what
    ``Form4Adapter._fetch_one_candidate`` issues per candidate on a cold
    cache). Never a soft estimate: this is the exact number passed as
    ``AllowlistedHttpClient.max_requests``, so a run that would exceed it
    raises ``MaxRequestsExceededError`` rather than silently making more
    requests.
    """
    return 1 + MAX_SUBMISSIONS_PAGES + max(1, max_filings) * 2


def _build_dummy_target() -> AcquisitionTarget:
    """A structurally valid, but never-inspected-by-Form4Adapter,
    ``AcquisitionTarget`` -- ``ExecutionContext.target`` is typed as
    required, but neither ``Form4Adapter`` nor this module's own
    orchestration ever reads it (confirmed by inspection of
    ``form4_acquisition_adapter.py``: every method reads only
    ``context.form4_reference_for``/``context.payload_for``/
    ``context.request_cache``/``context.document_store``)."""
    requirement = EvidenceRequirement(
        requirement_id="req_form4_live_smoke", serves_legacy_need_ids=("form4_live_smoke",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    return AcquisitionTarget(
        target_id="target_form4_live_smoke", target_kind=TargetKind.FORM4_FILING,
        required_step_ids=("l1", "f", "p"), serves_requirement_ids=(requirement.requirement_id,),
    )


def _step(step_id: str, kind: StepKind, *, depends_on: tuple[str, ...] = ()) -> AcquisitionStep:
    condition = {
        StepKind.LOCATE: StepStatus.URL_RESOLVED,
        StepKind.FETCH: StepStatus.BODY_FETCHED,
        StepKind.PARSE: StepStatus.PARSED,
    }[kind]
    return AcquisitionStep(
        step_id=step_id, target_id="target_form4_live_smoke", step_kind=kind,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER if kind is StepKind.LOCATE else AcquisitionMethod.KNOWN_URL_HTTP,
        adapter_id=FORM4_ADAPTER_ID, depends_on_step_ids=depends_on,
        completion_condition=condition, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )


# --------------------------------------------------------------------------
# Raw-XML diagnostic scanners -- independent of collectors/form4.py's own
# two-candidate-name checkbox guess (Phase 3E.2 requirement 3). Never used
# to decide any PARSE-stage field; diagnostic-only.
# --------------------------------------------------------------------------


def _walk_with_parent(root: ET.Element) -> Iterator[tuple[ET.Element, str | None]]:
    stack: list[tuple[ET.Element, str | None]] = [(root, None)]
    while stack:
        el, parent_tag = stack.pop()
        yield el, parent_tag
        for child in el:
            stack.append((child, el.tag))


def scan_rule_10b5_1_elements(xml_text: str) -> list[dict[str, Any]]:
    """Every element ANYWHERE in the real XML tree whose own tag name
    contains ``10b5`` or ``10b-5`` (case-insensitive) -- the real tag
    name and raw value, verbatim, plus its immediate parent tag (so a
    reviewer can see for themselves whether it sits at document level,
    directly under ``ownershipDocument``, or nested inside a transaction
    -- never decided by this function). Returns ``[]`` (never a guess)
    if the body is not well-formed XML or nothing matches."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    matches: list[dict[str, Any]] = []
    for el, parent_tag in _walk_with_parent(root):
        tag_lower = el.tag.lower()
        if "10b5" not in tag_lower and "10b-5" not in tag_lower:
            continue
        value_el = el.find("value")
        raw_value = (value_el.text if value_el is not None else el.text) or None
        matches.append({
            "tag": el.tag,
            "parent_tag": parent_tag,
            "is_direct_child_of_ownership_document": parent_tag == "ownershipDocument",
            "raw_value": raw_value.strip() if raw_value else None,
        })
    return matches


def scan_date_of_original_submission(xml_text: str) -> dict[str, Any]:
    """Whether a real ``dateOfOriginalSubmission``-tagged element exists
    anywhere in the tree, and its raw value if so -- ``found=False`` (never
    a guessed date) when absent."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {"found": False, "tag": None, "raw_value": None}
    for el, parent_tag in _walk_with_parent(root):
        if el.tag.lower() != "dateoforiginalsubmission":
            continue
        value_el = el.find("value")
        raw_value = (value_el.text if value_el is not None else el.text) or None
        return {
            "found": True, "tag": el.tag, "parent_tag": parent_tag,
            "raw_value": raw_value.strip() if raw_value else None,
        }
    return {"found": False, "tag": None, "raw_value": None}


def structure_diagnostics(xml_text: str) -> dict[str, Any]:
    """Confirms ``ownershipDocument``/``documentType``/``issuerCik`` are
    genuinely present in the REAL body (Phase 3E.2 requirement 3), plus
    the two raw-XML scans above. Never raises: an unparseable body reports
    ``well_formed=False`` and every other field as ``UNKNOWN``/empty."""
    shape_error, type_or_reason = check_ownership_xml_shape(xml_text)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {
            "well_formed": False, "root_tag": None, "document_type": UNKNOWN, "issuer_cik": UNKNOWN,
            "rule_10b5_1_candidate_elements": [], "date_of_original_submission": scan_date_of_original_submission(""),
        }
    issuer = root.find("issuer")
    issuer_cik = (issuer.findtext("issuerCik") or UNKNOWN).strip() if issuer is not None else UNKNOWN
    return {
        "well_formed": True,
        "root_tag": root.tag,
        "document_type": (type_or_reason if shape_error is None else UNKNOWN),
        "shape_check_error": shape_error.value if shape_error is not None else None,
        "issuer_cik": issuer_cik or UNKNOWN,
        "rule_10b5_1_candidate_elements": scan_rule_10b5_1_elements(xml_text),
        "date_of_original_submission": scan_date_of_original_submission(xml_text),
    }


# --------------------------------------------------------------------------
# Plan / report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSmokePlan:
    allowed_hosts: tuple[str, ...]
    max_requests: int
    issuer_cik: int
    as_of_date: str
    lookback_days: int
    max_filings: int
    submissions_url: str


@dataclass
class Form4LiveSmokeReport:
    plan: LiveSmokePlan
    #: One of: REFUSED / DISCOVERY_NOT_SUPPORTED / LOCATE_FAILED /
    #: NO_CANDIDATES_IN_LOOKBACK_WINDOW / COMPLETED -- always set, never
    #: left for a reader to infer from other fields.
    status: str = "REFUSED"
    refused_reason: str | None = None
    requested_urls: list[str] = field(default_factory=list)
    request_count: int = 0
    cache_hit_count: int = 0
    #: Every distinct SEC `form` value seen in the issuer's RAW
    #: submissions.json (before any Form 3/4/5 filtering) -- a neutral
    #: count, never phrased as a claim about what it proves (Phase 3E.2.1
    #: requirement 6: the previous wording asserted "Form 3/5 are present"
    #: even for an issuer whose raw data has none).
    raw_submissions_forms_seen: dict[str, int] = field(default_factory=dict)
    #: Subset of ``raw_submissions_forms_seen`` restricted to form values
    #: OTHER than "4"/"4-A" -- populated (and only ever printed) when at
    #: least one non-Form-4 filing genuinely appears in the raw data, so a
    #: reader can see for themselves whether the exclusion filter had
    #: anything real to exclude for THIS issuer, never asserted generically
    #: (Phase 3E.2.1 requirement 6).
    excluded_non_form4_form_counts: dict[str, int] = field(default_factory=dict)
    candidates_found_total: int = 0  # all Form 4/4-A candidates LOCATE discovered, before lookback filtering
    form4_count_total: int = 0
    form4a_count_total: int = 0
    submissions_pages_fetched: int = 0
    submissions_pages_excluded: int = 0
    discovery_candidates: list[dict[str, Any]] = field(default_factory=list)  # accession/form/filingDate/primaryDocument/archive_cik
    excluded_unparseable_filing_date: int = 0
    candidates_found: int = 0  # within the lookback window
    fetched: int = 0
    not_fetched_due_to_cap: int = 0
    fetch_failures: list[dict[str, Any]] = field(default_factory=list)
    parsed_documents: list[dict[str, Any]] = field(default_factory=list)  # accession/form/reconciliation/diagnostics/structure
    parse_failures: list[dict[str, Any]] = field(default_factory=list)
    capture_manifest_failures: list[str] = field(default_factory=list)
    anthropic_api_calls: int = 0
    web_search_calls: int = 0
    external_llm_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.status == "REFUSED"


def build_plan(
    issuer_cik: int, *, as_of_date: str, lookback_days: int, max_filings: int,
) -> LiveSmokePlan:
    return LiveSmokePlan(
        allowed_hosts=tuple(sorted(ALLOWED_HOSTS)),
        max_requests=compute_max_live_gets(max_filings),
        issuer_cik=issuer_cik,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        max_filings=max_filings,
        submissions_url=SUBMISSIONS_URL.format(cik=issuer_cik),
    )


def _within_lookback(filing_date: str, *, as_of: date, lookback_days: int) -> bool | None:
    """``True``/``False`` for a parseable date; ``None`` (never guessed
    into or out of the window) for one that is not."""
    parsed = parse_iso_date(filing_date)
    if parsed is None:
        return None
    return (as_of - timedelta(days=lookback_days)) <= parsed <= as_of


def run_live_smoke(
    issuer_cik: int,
    *,
    as_of_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_filings: int = DEFAULT_MAX_FILINGS,
    user_agent: str | None | _Unset = _UNSET,
    http_client: Any | None = None,
    out_dir: Path | None = None,
) -> Form4LiveSmokeReport:
    """Run the Form 4 structure-verification smoke test. ``http_client`` is
    injectable for offline testing only. See the module docstring for the
    full set of enforced constraints.
    """
    resolved_as_of = as_of_date or time.strftime("%Y-%m-%d", time.gmtime())
    plan = build_plan(issuer_cik, as_of_date=resolved_as_of, lookback_days=lookback_days, max_filings=max_filings)
    report = Form4LiveSmokeReport(plan=plan)

    if isinstance(user_agent, _Unset):
        resolved_user_agent = resolve_user_agent()
    else:
        resolved_user_agent = _valid_user_agent_or_none(user_agent)
    if resolved_user_agent is None:
        report.refused_reason = (
            "IRA_SEC_USER_AGENT is unset, blank, or still the built-in placeholder -- "
            "refusing to make any SEC request"
        )
        return report

    client = http_client if http_client is not None else AllowlistedHttpClient(
        user_agent=resolved_user_agent, out_dir=out_dir, source="form4",
        max_requests=plan.max_requests,
    )

    # A diagnostic pre-fetch of the raw submissions payload -- served from
    # AllowlistedHttpClient's own URL cache (not a second live GET) when
    # Form4Adapter._locate fetches the identical URL moments later via the
    # composed SecPrimaryDocumentAdapter (see module docstring).
    submissions_result = client.get(plan.submissions_url)
    if not submissions_result.ok:
        report.status = "LOCATE_FAILED"
        report.errors.append(f"submissions fetch failed: {submissions_result.outcome} {submissions_result.error}")
        _collect_diagnostics(client, report)
        return report
    submissions_payload = submissions_result.json() or {}
    recent_forms = (((submissions_payload or {}).get("filings") or {}).get("recent") or {}).get("form") or []
    forms_seen: dict[str, int] = {}
    for form in recent_forms:
        forms_seen[form] = forms_seen.get(form, 0) + 1
    report.raw_submissions_forms_seen = forms_seen
    report.excluded_non_form4_form_counts = {
        form: count for form, count in forms_seen.items() if form not in VALID_FORM4_DOCUMENT_TYPES
    }

    store = DocumentStore()
    adapter = Form4Adapter(client)
    ref = Form4IssuerReference(issuer_cik=issuer_cik, max_candidates=DISCOVERY_CANDIDATE_CEILING)
    context = ExecutionContext(
        target=_build_dummy_target(), document_store=store,
        form4_references={"target_form4_live_smoke": ref},
    )

    step_locate = _step("l1", StepKind.LOCATE)
    locate_result = adapter.execute(step_locate, context)
    context.payloads[step_locate.step_id] = locate_result.payload
    _collect_diagnostics(client, report)

    if locate_result.status is StepStatus.ZERO_RESULTS:
        report.status = "DISCOVERY_NOT_SUPPORTED"
        report.errors.append(locate_result.failure_reason)
        return report
    if locate_result.status is not StepStatus.URL_RESOLVED:
        report.status = "LOCATE_FAILED"
        report.errors.append(locate_result.failure_reason or "LOCATE did not resolve")
        return report

    raw_candidates = list(locate_result.payload.get("candidates") or [])
    report.candidates_found_total = len(raw_candidates)
    report.form4_count_total = sum(1 for c in raw_candidates if c.get("form") == "4")
    report.form4a_count_total = sum(1 for c in raw_candidates if c.get("form") == "4/A")
    report.submissions_pages_fetched = locate_result.payload.get("submissions_pages_fetched", 0)
    report.submissions_pages_excluded = locate_result.payload.get("submissions_pages_excluded", 0)
    report.discovery_candidates = [
        {
            "accession": c.get("accession"), "form": c.get("form"),
            "filing_date": c.get("filing_date"), "primary_document": c.get("primary_document"),
            "archive_cik_used": cik_for_archives(issuer_cik),
        }
        for c in raw_candidates
    ]

    as_of = parse_iso_date(resolved_as_of)
    assert as_of is not None, "resolved_as_of is either the caller's own validated value or today's UTC date"
    within_window: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        verdict = _within_lookback(candidate.get("filing_date", UNKNOWN), as_of=as_of, lookback_days=lookback_days)
        if verdict is None:
            report.excluded_unparseable_filing_date += 1
            continue
        if verdict:
            within_window.append(candidate)

    report.candidates_found = len(within_window)
    if not within_window:
        report.status = "NO_CANDIDATES_IN_LOOKBACK_WINDOW"
        return report

    fetch_planned = within_window[:max_filings]
    report.fetched = len(fetch_planned)
    report.not_fetched_due_to_cap = max(0, len(within_window) - len(fetch_planned))

    # Never simply add/overwrite onto the raw LOCATE payload's candidate
    # list -- a NEW payload dict, scoped to only what THIS module decided
    # to actually fetch (Phase 3E.2 requirement 4).
    context.payloads[step_locate.step_id] = {**locate_result.payload, "candidates": fetch_planned}

    step_fetch = _step("f", StepKind.FETCH, depends_on=(step_locate.step_id,))
    fetch_result = adapter.execute(step_fetch, context)
    context.payloads[step_fetch.step_id] = fetch_result.payload
    _collect_diagnostics(client, report)
    report.fetch_failures = list(fetch_result.payload.get("fetch_failures") or [])

    if fetch_result.status is not StepStatus.BODY_FETCHED:
        report.status = "LOCATE_FAILED"
        report.errors.append(fetch_result.failure_reason or "FETCH did not fetch any candidate")
        return report

    step_parse = _step("p", StepKind.PARSE, depends_on=(step_fetch.step_id,))
    parse_result = adapter.execute(step_parse, context)
    _collect_diagnostics(client, report)
    report.parse_failures = list(parse_result.payload.get("parse_failures") or [])

    if parse_result.status is not StepStatus.PARSED:
        report.status = "LOCATE_FAILED"
        report.errors.append(parse_result.failure_reason or "PARSE did not parse any candidate")
        return report

    fetched_by_accession = {e["accession"]: e for e in fetch_result.payload.get("fetched") or []}
    for doc in parse_result.payload.get("parsed_documents") or []:
        accession = doc["accession"]
        xml_text = fetched_by_accession.get(accession, {}).get("ownership_xml_text", "")
        report.parsed_documents.append({
            "accession": accession,
            "form": doc["form"],
            "document_id": doc["document_id"],
            "reconciliation": {
                "status": doc["reconciliation"].status,
                "original_accession": doc["reconciliation"].original_accession,
                "reason": doc["reconciliation"].reason,
            },
            "parsed_ten_b5_1_checkbox": doc["parsed"].get("ten_b5_1_checkbox"),
            "parsed_ten_b5_1_plan_adoption_date": doc["parsed"].get("ten_b5_1_plan_adoption_date"),
            "structure_diagnostics": structure_diagnostics(xml_text) if xml_text else None,
        })

    report.status = "COMPLETED"
    return report


def _collect_diagnostics(client: Any, report: Form4LiveSmokeReport) -> None:
    seen = list(getattr(client, "requested_urls", report.requested_urls))
    report.requested_urls = list(dict.fromkeys(seen))
    report.request_count = getattr(client, "requests_made", len(report.requested_urls))
    report.cache_hit_count = getattr(client, "cache_hits", 0)


def format_report_for_print(report: Form4LiveSmokeReport, *, secrets: list[str]) -> str:
    lines: list[str] = []
    lines.append("=== Form4 Live Smoke (Phase 3E.2) -- structure verification, not an investment conclusion ===")
    lines.append(f"allowed_hosts: {report.plan.allowed_hosts}")
    lines.append(f"max_requests (computed from --max-filings): {report.plan.max_requests}")
    lines.append(f"issuer_cik: {report.plan.issuer_cik}")
    lines.append(f"as_of_date: {report.plan.as_of_date}  lookback_days: {report.plan.lookback_days}  max_filings: {report.plan.max_filings}")
    lines.append(f"status: {report.status}")
    if report.refused:
        lines.append(f"REFUSED: {report.refused_reason}")
        return "\n".join(_safe(line, secrets) for line in lines)

    lines.append(f"requested_urls ({len(report.requested_urls)}): {report.requested_urls}")
    lines.append(f"request_count: {report.request_count}  cache_hit_count: {report.cache_hit_count}")
    lines.append(f"raw_submissions_forms_seen: {report.raw_submissions_forms_seen}")
    if report.excluded_non_form4_form_counts:
        lines.append(f"excluded_non_form4_form_counts (genuinely present in this issuer's raw data): {report.excluded_non_form4_form_counts}")
    if report.status == "DISCOVERY_NOT_SUPPORTED":
        lines.append(f"errors: {report.errors}")
        return "\n".join(_safe(line, secrets) for line in lines)
    if report.status == "LOCATE_FAILED":
        lines.append(f"errors: {report.errors}")
        return "\n".join(_safe(line, secrets) for line in lines)

    lines.append(
        f"discovery: candidates_found_total={report.candidates_found_total} "
        f"form4_count_total={report.form4_count_total} form4a_count_total={report.form4a_count_total} "
        f"submissions_pages_fetched={report.submissions_pages_fetched} "
        f"submissions_pages_excluded={report.submissions_pages_excluded}"
    )
    for candidate in report.discovery_candidates:
        lines.append(f"  candidate: {candidate}")
    if report.status == "NO_CANDIDATES_IN_LOOKBACK_WINDOW":
        lines.append(
            f"NO_CANDIDATES_IN_LOOKBACK_WINDOW: {report.candidates_found_total} candidate(s) exist for this "
            f"issuer, but none fall within the requested lookback window (never treated as DISCOVERY_NOT_SUPPORTED)."
        )
        return "\n".join(_safe(line, secrets) for line in lines)

    lines.append(
        f"period_and_cap: candidates_found={report.candidates_found} (within lookback) "
        f"excluded_unparseable_filing_date={report.excluded_unparseable_filing_date} "
        f"fetched={report.fetched} not_fetched_due_to_cap={report.not_fetched_due_to_cap}"
    )
    if report.not_fetched_due_to_cap:
        lines.append(
            "  NOTE: not_fetched_due_to_cap > 0 -- coverage for this lookback window is NOT complete."
        )
    lines.append(f"fetch_failures: {report.fetch_failures}")
    lines.append(f"parse_failures: {report.parse_failures}")
    lines.append("parsed_documents (real XML structure diagnostics per candidate):")
    for doc in report.parsed_documents:
        lines.append(f"  {doc}")
    lines.append(f"capture_manifest_failures: {report.capture_manifest_failures}")
    lines.append(f"anthropic_api_calls: {report.anthropic_api_calls}  web_search_calls: {report.web_search_calls}  external_llm_tokens: {report.external_llm_tokens}")
    if report.errors:
        lines.append(f"errors: {report.errors}")
    return "\n".join(_safe(line, secrets) for line in lines)


def format_plan_for_print(plan: LiveSmokePlan) -> str:
    return (
        "=== Form4 Live Smoke plan (nothing sent yet) ===\n"
        f"allowed_hosts: {plan.allowed_hosts}\n"
        f"max_requests: {plan.max_requests} (= 1 submissions + {MAX_SUBMISSIONS_PAGES} continuation pages "
        f"+ {plan.max_filings} candidates * 2 GETs each)\n"
        f"issuer_cik: {plan.issuer_cik}\n"
        f"as_of_date: {plan.as_of_date}  lookback_days: {plan.lookback_days}  max_filings: {plan.max_filings}\n"
        f"GET 1 of at most {plan.max_requests} (submissions metadata): {plan.submissions_url}\n"
        "GET 2..N (continuation pages / directory index.json / ownership XML bodies) are resolved from the "
        "submissions response and are not known until then; the request count is hard-capped in code regardless."
    )


# --------------------------------------------------------------------------
# Offline re-analysis of a prior capture directory
# --------------------------------------------------------------------------


def analyze_capture(capture_dir: Path) -> dict[str, Any]:
    """Re-analyze every already-saved capture under ``capture_dir``,
    entirely offline. Makes ZERO network calls, touches NO marker file.

    Unlike ``sec_live_smoke.analyze_capture``, this needs no
    accession/primary-document arguments: every saved response already
    has its own Capture Manifest recording its own ``requested_url``, so
    this simply discovers every ``*.manifest.json`` file under
    ``capture_dir`` and re-verifies each one against its body -- a
    manifest integrity failure is reported as such and is NEVER treated
    as a usable LIVE verification candidate (Phase 3E.2 requirement 5).
    """
    manifest_files = sorted(capture_dir.glob("*.manifest.json"))
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for manifest_file in manifest_files:
        digest = manifest_file.name.removesuffix(".manifest.json")
        body_path: Path | None = None
        for suffix in (".json", ".htm", ".bin"):
            candidate = capture_dir / f"{digest}{suffix}"
            if candidate.is_file():
                body_path = candidate
                break
        body = body_path.read_bytes() if body_path is not None else None
        result = read_manifest(capture_dir, digest, body=body)
        entry: dict[str, Any] = {
            "digest": digest,
            "capture_manifest_status": result.status.value,
            "capture_manifest_error": result.error_reason,
        }
        if result.status is ManifestReadStatus.VERIFIED:
            assert result.manifest is not None
            entry["requested_url"] = result.manifest.requested_url
            entry["final_url"] = result.manifest.final_url
            entry["http_status"] = result.manifest.http_status
            entry["capture_retrieved_at"] = result.manifest.capture_retrieved_at
            entry["category"] = _classify_sec_url(result.manifest.requested_url)
            if body is not None:
                text = body.decode("utf-8", errors="replace")
                shape_error, _ = check_ownership_xml_shape(text)
                if shape_error is None:
                    entry["structure_diagnostics"] = structure_diagnostics(text)
        else:
            entry["requested_url"] = result.manifest.requested_url if result.manifest is not None else UNKNOWN
            entry["capture_retrieved_at"] = UNKNOWN
            if is_evidence_integrity_failure(result.status):
                failures.append(f"{digest}: {result.status.value} -- {result.error_reason}")
        entries.append(entry)

    return {
        "capture_dir": str(capture_dir),
        "captures_found": len(entries),
        "entries": entries,
        "evidence_integrity_failures": failures,
    }


def main(
    argv: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    http_client_factory: Callable[[str, Path, int], Any] | None = None,
) -> int:
    """CLI entry point -- see ``scripts/form4_live_smoke.py``. Never
    imported by cli.py/pipeline.py.

    ``env``/``http_client_factory`` are test-only dependency-injection
    seams (Phase 3E.2.1 requirement 3) -- production callers (``scripts/
    form4_live_smoke.py``) never pass either, so the real CLI's behavior
    is unchanged: ``env=None`` resolves ``IRA_SEC_USER_AGENT`` from the
    real ``os.environ`` exactly as before (via
    ``sec_live_smoke.resolve_user_agent``, called here exactly ONCE --
    never re-read from the environment a second time by ``run_live_smoke``,
    since the already-resolved string is always passed through explicitly),
    and ``http_client_factory=None`` lets ``run_live_smoke`` construct its
    own real ``AllowlistedHttpClient`` exactly as it always has. A test
    that passes an explicit ``env`` mapping (e.g. ``{}``) is therefore
    hermetically isolated from whatever ``IRA_SEC_USER_AGENT`` value
    happens to be set in the real shell running the test suite -- the bug
    this seam fixes (Phase 3E.2.1) is a test that implicitly depended on
    the ambient environment being unset and, on a machine where it
    genuinely was set, made a real SEC request during an "offline" test.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="form4_live_smoke",
        description=(
            "Phase 3E.2: a one-time Form 4/4-A structure-verification Live Smoke test "
            "against real SEC EDGAR data.gov endpoints for one real issuer CIK. "
            "Read-only GETs to data.sec.gov/www.sec.gov only. Requires IRA_SEC_USER_AGENT "
            "to already be set. Never promotes anything to LIVE_VERIFIED."
        ),
    )
    parser.add_argument("--issuer-cik", type=int, default=None, help="SEC CIK of the issuer to query (required unless --analyze-capture).")
    parser.add_argument("--as-of", type=str, default=None, help="Reference date (YYYY-MM-DD) for the lookback window. Default: today (UTC).")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help=f"Lookback window in days. Default: {DEFAULT_LOOKBACK_DAYS}.")
    parser.add_argument("--max-filings", type=int, default=DEFAULT_MAX_FILINGS, help=f"Maximum candidates to actually fetch. Default: {DEFAULT_MAX_FILINGS}.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to save raw response bodies + Capture Manifests. Default: data/live_smoke/form4/<UTC timestamp>/ (gitignored).")
    parser.add_argument("--force-rerun", action="store_true", help="Bypass the one-time marker guard and run again anyway.")
    parser.add_argument("--marker-path", type=Path, default=None, help="Override the one-time marker file path (mainly for testing).")
    parser.add_argument("--analyze-capture", type=Path, default=None, metavar="DIR", help="Offline-only mode: re-analyze already-saved captures under DIR. Makes NO network call.")
    args = parser.parse_args(argv)

    if args.analyze_capture is not None:
        result = analyze_capture(args.analyze_capture)
        print(json.dumps(result, indent=2, default=str))
        return 0 if not result["evidence_integrity_failures"] else 1

    if args.issuer_cik is None:
        print("REFUSED: --issuer-cik is required (unless --analyze-capture is used).")
        return 1
    if normalize_cik(args.issuer_cik) is None:
        print(f"REFUSED: malformed --issuer-cik: {args.issuer_cik!r}")
        return 1

    repo_root = Path(__file__).resolve().parents[3]
    marker_path = args.marker_path or (repo_root / "data" / "live_smoke" / "form4" / "LAST_RUN.json")
    out_dir = args.out_dir or (
        repo_root / "data" / "live_smoke" / "form4" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )

    plan = build_plan(
        args.issuer_cik, as_of_date=args.as_of or time.strftime("%Y-%m-%d", time.gmtime()),
        lookback_days=args.lookback_days, max_filings=args.max_filings,
    )
    print(format_plan_for_print(plan))

    previous = _marker_state(marker_path)
    if previous is not None and not args.force_rerun:
        print(
            f"\nREFUSED: a previous live smoke run is already recorded at {marker_path} "
            f"(attempted_at={previous.get('attempted_at')}). Pass --force-rerun to override deliberately."
        )
        return 2

    # Resolved exactly once -- env=None reads the real os.environ (unchanged
    # production behavior); a test-injected env mapping is hermetically
    # isolated from the real ambient environment. The resolved value is
    # then passed EXPLICITLY to run_live_smoke below, which never re-reads
    # the environment itself when given an already-resolved value.
    user_agent = resolve_user_agent(env)
    secrets = [user_agent] if user_agent else []
    if user_agent is None:
        print(
            "\nREFUSED: IRA_SEC_USER_AGENT is unset, blank, or still the built-in placeholder. "
            "No SEC request was made."
        )
        # Deliberately NOT writing the one-time marker -- zero network
        # activity occurred (mirrors sec_live_smoke.py exactly).
        return 1

    # None (the production default) lets run_live_smoke construct its own
    # real AllowlistedHttpClient exactly as before; a test-injected factory
    # (e.g. a client whose .get() always raises) makes that construction
    # observable/preventable from the test, without changing what the real
    # CLI does.
    client = http_client_factory(user_agent, out_dir, plan.max_requests) if http_client_factory is not None else None

    report = run_live_smoke(
        args.issuer_cik, as_of_date=args.as_of, lookback_days=args.lookback_days,
        max_filings=args.max_filings, user_agent=user_agent, http_client=client, out_dir=out_dir,
    )
    print()
    print(format_report_for_print(report, secrets=secrets))
    _write_marker(marker_path, refused=report.refused)
    return 0 if report.status == "COMPLETED" and not report.errors else 1
