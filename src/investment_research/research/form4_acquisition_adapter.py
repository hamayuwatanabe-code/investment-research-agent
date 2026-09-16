"""SEC Form 4 (Section 16 ownership) Direct Adapter (Phase 3E, hardened in
Phase 3E.1).

Phase 3E.1 replaced Phase 3E's caller-must-already-know-the-accession
design with genuine ISSUER-DRIVEN DISCOVERY: a caller supplies only an
issuer CIK or ticker, and LOCATE discovers Form 4/4-A candidates itself
from SEC's own submissions metadata -- never a pre-specified accession,
primary-document filename, or reporting-owner CIK (Phase 3E.1
requirement 1). This is possible because SEC's own
``data.sec.gov/submissions/CIK##########.json`` for an ISSUER includes
every Section 16 (Form 3/4/5) filing made against it, in the exact same
``filings.recent`` parallel-array structure other filing types use (this
mirrors documented EDGAR "company filing history" behavior -- the same
concept ``browse-edgar?action=getcompany&type=4`` exposes over HTML) --
see the module-level note below for the explicit caveat that this has
not been checked against a real captured submissions.json in this
offline session.

Renamed from Phase 3E's ``filer_cik``/``Form4FilingReference`` to
``issuer_cik``/``Form4IssuerReference`` throughout (Phase 3E.1
requirement 1's last point): the ambiguous "filer" framing conflated
"whose CIK do we query and resolve the Archives directory against" with
an unrelated, never-actually-needed third concept. There is now exactly
one CIK this adapter accepts from a caller -- the issuer's -- and it is
used for both the submissions.json fetch and the Archives directory
resolution (SEC serves one accession's documents under every CIK
associated with it, issuer included).

Fetches every LOCATEd candidate's ownership XML (bounded by
``max_candidates``), reusing ``sec_acquisition_adapters.
SecPrimaryDocumentAdapter``'s own submissions/directory fetch-and-cache
methods by composition (not duplication) so a Form 4 target and an SEC
primary-document target sharing the same issuer CIK within one
``AcquisitionExecutor.run()`` never issue a second real request for the
same submissions listing or directory index.

Registered with an ``AcquisitionExecutor`` by the caller -- like every
other adapter in this repository, this is NOT called from the production
pipeline (see CLAUDE.md's Phase 3A-3E forbidden-changes lists), and every
test drives it against a fake HTTP double loaded from
``tests/fixtures/form4_real_format/``. ``ImplementationStatus`` for the
steps this adapter backs therefore tops out at OFFLINE_VERIFIED, never
LIVE_VERIFIED: nothing here has ever been checked against the real SEC
EDGAR site, and this Phase deliberately does not add a Form 4 Live Smoke
entry point (explicit instruction: no Live communication this phase).

**CONFIRMED against a real capture (Phase 3E.3):** a real Mac Live Smoke
run (``research/form4_live_smoke.py``, itself never invoked from this
adapter or from Pipeline) confirmed that a real issuer's own
submissions.json genuinely includes Form 3/4/5 entries, that a real
ownership XML's ``primaryDocument`` can be an XSL display path (handled
by ``collectors/form4.normalize_ownership_primary_document`` since Phase
3E.2.2), and that ``ownershipDocument``/``documentType``/``issuerCik``
all resolve exactly as this adapter assumes -- for 3 real, normal
(non-amendment) Form 4 documents, all Capture-Manifest-``VERIFIED`` with
zero Evidence Integrity failures. This confirmation is scoped to NORMAL
Form 4 only; no real Form 4/A instance has yet been observed end-to-end
this way (see this module's own docstring note on the Live-verification
scope split, and ``research/form4_live_smoke.py``'s targeted mode, added
in Phase 3E.3, for verifying one specific accession -- including a
Form 4/A -- directly).

Safe identification of the acquisition target (Phase 3E/3E.1 requirement
3): Form 3, Form 4, and Form 5 share the identical ``ownershipDocument``
XML schema -- distinguished ONLY by ``documentType``; ``4/A`` is kept
distinct from ``4`` throughout, and a 4/A is a genuinely SEPARATE
``DocumentStore`` document (its own accession/URL), never a "version" of
the original; the XML's OWN ``issuerCik`` is cross-checked against the
REQUESTED issuer CIK at FETCH time -- a mismatch excludes that candidate
from being stored as ACQUIRED evidence, never silently accepted (Phase
3E.1 requirement 1's CIK-mismatch guard).

Evidence Integrity (Phase 3E/3E.1 requirement 4/6): Form 4 is the
REPORTING PERSON's own statutory filing (``DocumentAuthority.
STATUTORY_FILING``, ``Document.is_company_ir=False``). See
``collectors/form4.py``'s module docstring for the full transaction-
semantics boundary and the Rule 10b5-1 checkbox's real, CONFIRMED (Phase
3E.3) document-level element name. Nothing here generates an investment
Action, promotes a domain to SUFFICIENT, or interprets a transaction as a
judgment about the
issuer -- and, new in Phase 3E.1, nothing here ever computes a net
insider-buying/selling figure across an original filing and its
amendment(s): amendment reconciliation is reported as a STATUS
(RECONCILED/UNRESOLVED/NOT_APPLICABLE), never as a merged or summed
transaction record (requirement 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..collectors.documents import Document
from ..collectors.form4 import (
    VALID_FORM4_DOCUMENT_TYPES,
    check_ownership_xml_shape,
    normalize_ownership_primary_document,
    parse_ownership_document,
)
from ..collectors.sec_edgar import TICKER_MAP_URL, normalize_cik
from ..schemas.enums import UNKNOWN, ContentKind, DocumentAuthority, Provenance
from ..schemas.fact import utc_now_iso
from .acquisition_executor import ExecutionContext, StepExecutionResult
from .document_store import DocumentRole, derive_document_id
from .sec_acquisition_adapters import (
    SecPrimaryDocumentAdapter,
    _resolve_document_url,
    _validate_accession,
)
from .source_routing import AcquisitionStep, StepKind, StepStatus

log = logging.getLogger(__name__)

#: Matches source_routing_catalog.py's ``form4`` archetype's own
#: ``direct_adapter="form4_xml_parser"`` string exactly.
FORM4_ADAPTER_ID = "form4_xml_parser"

#: An ownership XML record is terse structured data, not HTML prose --
#: this bar exists only to refuse a suspiciously empty/near-empty body,
#: never to demand prose length (mirrors ClinicalTrials' MIN_BODY_CHARS
#: reasoning, not SEC HTML's much higher one).
MIN_BODY_CHARS = 60

#: Hard bound on how many Form 4/4-A candidates one LOCATE will ever
#: return, and how many ``filings.files`` continuation pages it will ever
#: fetch -- excess candidates/pages are reported as excluded, never
#: silently fetched without limit (Phase 3E.1 requirement 5's "budget/
#: limit exclusion").
DEFAULT_MAX_CANDIDATES = 5
MAX_SUBMISSIONS_PAGES = 3


@dataclass(frozen=True)
class Form4IssuerReference:
    """The concrete issuer identity a caller supplies -- and NOTHING
    else: no accession, no primary-document filename, no reporting-owner
    CIK (Phase 3E.1 requirement 1). Exactly one of ``issuer_cik``/
    ``issuer_ticker`` must resolve to a usable CIK; ``issuer_cik`` wins if
    both are given. ``issuer_ticker`` is resolved via SEC's own ticker
    map (one cached HTTP GET, shared across every target in a run that
    needs it)."""

    issuer_cik: int | str | None = None
    issuer_ticker: str | None = None
    max_candidates: int = DEFAULT_MAX_CANDIDATES


@dataclass(frozen=True)
class Form4Candidate:
    accession: str
    form: str
    filing_date: str
    primary_document: str


@dataclass(frozen=True)
class Form4Reconciliation:
    """Whether a Form 4/A amendment's relationship to an original Form 4
    could be confirmed -- never inferred from mere coincidence (matching
    period/owner), only from an explicit accession reference in the
    amendment's own remarks that genuinely matches another candidate
    discovered for the SAME issuer (Phase 3E.1 requirement 3)."""

    status: str  # "RECONCILED" | "UNRESOLVED" | "NOT_APPLICABLE"
    original_accession: str
    reason: str


def _upstream_payload(step: AcquisitionStep, context: ExecutionContext) -> dict[str, Any]:
    for dep_id in step.depends_on_step_ids:
        payload = context.payload_for(dep_id)
        if payload:
            return dict(payload)
    return {}


def _candidates_from_recent(recent: dict[str, Any]) -> list[Form4Candidate]:
    accessions = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    documents = recent.get("primaryDocument") or []
    filing_dates = recent.get("filingDate") or []
    candidates: list[Form4Candidate] = []
    for index, form in enumerate(forms):
        if form not in VALID_FORM4_DOCUMENT_TYPES:
            continue  # Never Form 3/5 (Phase 3E.1 requirement 1/3).
        accession = accessions[index] if index < len(accessions) else UNKNOWN
        document = documents[index] if index < len(documents) else ""
        filing_date = filing_dates[index] if index < len(filing_dates) else UNKNOWN
        if accession == UNKNOWN or not document:
            continue
        candidates.append(
            Form4Candidate(accession=accession, form=form, filing_date=filing_date, primary_document=document)
        )
    return candidates


class Form4Adapter:
    """LOCATE (resolve the issuer CIK, discover Form 4/4-A candidates from
    SEC's own submissions metadata across ``filings.recent`` and, up to a
    hard cap, ``filings.files`` continuation pages) -> FETCH (fetch every
    candidate's ownership XML, up to ``max_candidates``, cross-checking
    each one's own ``issuerCik`` against the requested issuer) -> PARSE
    (extract ownership fields for every successfully fetched candidate,
    reconcile 4/A amendments against their referenced original where
    possible, and surface the Phase 3E.1 diagnostics). Registered under
    adapter_id ``form4_xml_parser``.
    """

    def __init__(self, http: Any) -> None:
        self.http = http
        #: Composition, not duplication: this adapter's LOCATE step needs
        #: EXACTLY the same submissions/directory fetch-and-cache
        #: behavior ``SecPrimaryDocumentAdapter`` already implements
        #: (including its exact ``request_cache`` key conventions), so
        #: the two adapters share one real HTTP call for the same issuer
        #: CIK/accession within one run (Phase 3E requirement 2).
        self._sec = SecPrimaryDocumentAdapter(http)

    def execute(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        if step.step_kind is StepKind.LOCATE:
            return self._locate(step, context)
        if step.step_kind is StepKind.FETCH:
            return self._fetch(step, context)
        if step.step_kind is StepKind.PARSE:
            return self._parse(step, context)
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.FAILED,
            failure_reason=f"Form4Adapter cannot handle step_kind={step.step_kind}",
        )

    # -- LOCATE: issuer-driven discovery --------------------------------
    def _resolve_ticker(self, ticker: str, context: ExecutionContext) -> tuple[int | None, int, bool, str | None]:
        cache_key = "sec_ticker_map"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.URL_RESOLVED:
                return None, 0, True, cached.failure_reason or "cached ticker map fetch failed"
            payload = cached.payload.get("ticker_map", {})
        else:
            fetch = self.http.get(TICKER_MAP_URL)
            if not fetch.ok:
                failure = f"ticker map fetch failed: {fetch.outcome} {fetch.error}"
                context.request_cache[cache_key] = StepExecutionResult(
                    step_id="_cache_sec_ticker_map", status=StepStatus.FAILED,
                    http_requests_made=1, failure_reason=failure,
                )
                return None, 1, False, failure
            payload = fetch.json() or {}
            context.request_cache[cache_key] = StepExecutionResult(
                step_id="_cache_sec_ticker_map", status=StepStatus.URL_RESOLVED,
                payload={"ticker_map": payload}, http_requests_made=1,
            )

        wanted = ticker.strip().upper()
        for entry in (payload or {}).values():
            if isinstance(entry, dict) and str(entry.get("ticker", "")).upper() == wanted:
                cik_str_value = entry.get("cik_str")
                cik = normalize_cik(cik_str_value) if cik_str_value is not None else None
                if cik is not None:
                    return cik, (0 if cached is not None else 1), cached is not None, None
        return None, (0 if cached is not None else 1), cached is not None, f"ticker {ticker!r} not found in SEC's ticker map"

    def _fetch_continuation_page(self, name: str, context: ExecutionContext) -> tuple[dict[str, Any], int, bool, str | None]:
        from ..collectors.sec_edgar import SUBMISSIONS_PAGE_URL

        url = SUBMISSIONS_PAGE_URL.format(name=name)
        cache_key = f"http_get:{url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.URL_RESOLVED:
                return {}, 0, True, cached.failure_reason or "cached continuation page fetch failed"
            return cached.payload.get("page", {}), 0, True, None
        fetch = self.http.get(url)
        if not fetch.ok:
            failure = f"continuation page fetch failed for {name}: {fetch.outcome} {fetch.error}"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_form4_page_{name}", status=StepStatus.FAILED,
                http_requests_made=1, failure_reason=failure,
            )
            return {}, 1, False, failure
        page = fetch.json() or {}
        context.request_cache[cache_key] = StepExecutionResult(
            step_id=f"_cache_form4_page_{name}", status=StepStatus.URL_RESOLVED,
            payload={"page": page}, http_requests_made=1,
        )
        return page, 1, False, None

    def _locate(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        ref = context.form4_reference_for(step.target_id)
        if ref is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no Form4IssuerReference supplied for this target",
            )

        api_requests_made = 0
        http_requests_made = 0
        cache_hit = True
        issuer_cik: int | None = None

        if ref.issuer_cik is not None:
            issuer_cik = normalize_cik(ref.issuer_cik)
            if issuer_cik is None:
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.FAILED,
                    failure_reason=f"malformed issuer_cik: {ref.issuer_cik!r}",
                )
        elif ref.issuer_ticker:
            issuer_cik, ticker_reqs, ticker_cache_hit, ticker_failure = self._resolve_ticker(ref.issuer_ticker, context)
            http_requests_made += ticker_reqs
            cache_hit = cache_hit and ticker_cache_hit
            if issuer_cik is None:
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.NOT_FOUND,
                    http_requests_made=http_requests_made, cache_hit=cache_hit,
                    failure_reason=ticker_failure or f"could not resolve ticker {ref.issuer_ticker!r} to a CIK",
                )
        else:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="Form4IssuerReference supplies neither issuer_cik nor issuer_ticker",
            )

        submissions_payload, sub_api_reqs, sub_cache_hit, failure = self._sec._fetch_submissions(issuer_cik, context)
        api_requests_made += sub_api_reqs
        cache_hit = cache_hit and sub_cache_hit
        if failure is not None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, http_requests_made=http_requests_made, cache_hit=cache_hit,
                failure_reason=failure,
            )

        recent = ((submissions_payload or {}).get("filings") or {}).get("recent") or {}
        candidates = _candidates_from_recent(recent)

        files_index = (((submissions_payload or {}).get("filings") or {}).get("files")) or []
        pages_fetched = 0
        for page_ref in files_index[:MAX_SUBMISSIONS_PAGES]:
            name = page_ref.get("name") if isinstance(page_ref, dict) else None
            if not name:
                continue
            page, page_reqs, page_cache_hit, page_failure = self._fetch_continuation_page(name, context)
            http_requests_made += page_reqs
            cache_hit = cache_hit and page_cache_hit
            pages_fetched += 1
            if page_failure is not None:
                continue  # a bad continuation page never fails the whole LOCATE -- reported implicitly by fewer candidates
            candidates.extend(_candidates_from_recent(page))
        pages_excluded = max(0, len(files_index) - MAX_SUBMISSIONS_PAGES)

        if not candidates:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.ZERO_RESULTS,
                api_requests_made=api_requests_made, http_requests_made=http_requests_made, cache_hit=cache_hit,
                failure_reason=f"no Form 4/4-A filings found for issuer CIK {issuer_cik} (checked filings.recent"
                f"{f' + {pages_fetched} continuation page(s)' if pages_fetched else ''})",
                payload={"issuer_cik": issuer_cik},
            )

        # Newest first -- an unparseable/UNKNOWN filing date sorts last,
        # never guessed into a position it wasn't actually filed at.
        candidates.sort(key=lambda c: (c.filing_date == UNKNOWN, c.filing_date), reverse=False)
        candidates.sort(key=lambda c: c.filing_date, reverse=True)
        max_candidates = max(1, ref.max_candidates)
        selected = candidates[:max_candidates]
        excluded_candidates_count = max(0, len(candidates) - max_candidates)

        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.URL_RESOLVED,
            api_requests_made=api_requests_made, http_requests_made=http_requests_made, cache_hit=cache_hit,
            payload={
                "issuer_cik": issuer_cik,
                "issuer_ticker": ref.issuer_ticker or UNKNOWN,
                "candidates": [
                    {
                        "accession": c.accession, "form": c.form,
                        "filing_date": c.filing_date, "primary_document": c.primary_document,
                    }
                    for c in selected
                ],
                "excluded_candidates_count": excluded_candidates_count,
                "submissions_pages_fetched": pages_fetched,
                "submissions_pages_excluded": pages_excluded,
            },
        )

    # -- FETCH: every discovered candidate, cross-checked ----------------
    def _fetch_one_candidate(
        self, candidate: dict[str, Any], issuer_cik: int, context: ExecutionContext,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
        """Returns ``(fetched_entry, failure_entry, http_requests_made)`` --
        exactly one of the first two is non-``None``."""
        accession = candidate["accession"]
        primary_document = candidate["primary_document"]
        if not _validate_accession(accession):
            return None, {"accession": accession, "reason": f"malformed accession number: {accession!r}"}, 0

        # Phase 3E.2.2: SEC's own ownership-filing primaryDocument can be an
        # XSLT display path ("xslF345X06/marketforms-73885.xml"), which the
        # GENERAL sec_acquisition_adapters._validate_filename correctly
        # refuses outright (it contains "/") -- that check is deliberately
        # left unchanged for every other SEC target. This Form4-specific
        # normalizer recognizes ONLY the two real shapes SEC uses and
        # separates the raw XML's own basename from the XSL wrapper
        # directory; the wrapper directory is NEVER used to build a URL --
        # only the basename is (see below).
        normalized, rejection = normalize_ownership_primary_document(primary_document)
        if normalized is None:
            assert rejection is not None
            return None, {
                "accession": accession,
                "original_primary_document": primary_document,
                "reason": f"unsafe or unrecognized ownership primaryDocument shape: {primary_document!r} ({rejection.value})",
            }, 0

        accession_nodash = accession.replace("-", "")
        directory, http_reqs, _cache_hit, dir_failure = self._sec._fetch_directory(
            issuer_cik, accession, accession_nodash, context,
        )
        if dir_failure is not None:
            return None, {
                "accession": accession, "original_primary_document": primary_document,
                "normalized_xml_filename": normalized.normalized_xml_filename,
                "xsl_wrapper_path": normalized.xsl_wrapper_path, "directory_index_verified": False,
                "reason": dir_failure,
            }, http_reqs

        # The XSL display path is NEVER trusted at face value -- the raw
        # XML's own basename must appear, verbatim, as a real item in the
        # accession's OWN directory index.json before any URL is built
        # from it (Phase 3E.2.2 requirement 3). A basename absent from the
        # directory is refused, never guessed at.
        directory_filenames = {item.get("name", "") for item in (directory.get("item") or [])}
        directory_index_verified = normalized.normalized_xml_filename in directory_filenames
        if not directory_index_verified:
            return None, {
                "accession": accession, "original_primary_document": primary_document,
                "normalized_xml_filename": normalized.normalized_xml_filename,
                "xsl_wrapper_path": normalized.xsl_wrapper_path, "directory_index_verified": False,
                "reason": (
                    f"normalized ownership XML filename {normalized.normalized_xml_filename!r} "
                    f"(from primaryDocument {primary_document!r}) not present in the accession "
                    "directory listing"
                ),
            }, http_reqs

        # The RAW XML URL is built from the accession root + the verified
        # basename -- NEVER from the XSL wrapper path (which serves an
        # XSLT-transformed HTML rendering of the same file, not the raw
        # XML bytes this system needs to parse).
        url = _resolve_document_url(issuer_cik, accession_nodash, normalized.normalized_xml_filename)
        if url is None:
            return None, {
                "accession": accession, "original_primary_document": primary_document,
                "normalized_xml_filename": normalized.normalized_xml_filename,
                "xsl_wrapper_path": normalized.xsl_wrapper_path, "directory_index_verified": True,
                "reason": "could not safely resolve a document URL",
            }, http_reqs

        def _failure(reason: str, *, ownership_document_verified: bool = False, issuer_cik_verified: bool = False) -> dict[str, Any]:
            """Every failure from here on has already passed directory-index
            verification and URL resolution -- this carries those confirmed
            diagnostics forward (Phase 3E.2.2 requirement 4) rather than
            dropping them at the first later failure."""
            return {
                "accession": accession, "original_primary_document": primary_document,
                "normalized_xml_filename": normalized.normalized_xml_filename,
                "xsl_wrapper_path": normalized.xsl_wrapper_path, "directory_index_verified": True,
                "resolved_raw_xml_url": url, "ownership_document_verified": ownership_document_verified,
                "issuer_cik_verified": issuer_cik_verified, "reason": reason,
            }

        cache_key = f"http_get:{url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.BODY_FETCHED:
                return None, _failure(cached.failure_reason or "cached fetch failed"), http_reqs
            entry = dict(cached.payload)
            return entry, None, http_reqs

        fetch = self.http.get(url)
        http_reqs += 1
        if not fetch.ok:
            failure = f"GET {url} failed: {fetch.outcome} {fetch.error}"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_form4_body_{accession}", status=StepStatus.FAILED,
                http_requests_made=1, failure_reason=failure,
            )
            return None, _failure(failure), http_reqs

        text = fetch.text
        if not text.strip():
            failure = "response body was empty"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_form4_body_{accession}", status=StepStatus.FAILED,
                http_requests_made=1, failure_reason=failure,
            )
            return None, _failure(failure), http_reqs

        # The shape check is what actually refuses an XSL-transformed HTML
        # rendering (or any other non-ownership-XML body) -- an
        # ``<ownershipDocument>``-rooted well-formed-XML requirement that
        # HTML can never satisfy (Phase 3E.2.2 requirement 3: never
        # ACQUIRED from an HTML conversion result).
        shape_error, reason_or_type = check_ownership_xml_shape(text)
        if shape_error is not None:
            failure = f"{shape_error.value}: {reason_or_type}"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_form4_body_{accession}", status=StepStatus.FAILED,
                http_requests_made=1, failure_reason=failure,
            )
            return None, _failure(failure), http_reqs
        document_type = reason_or_type

        if len(text.strip()) < MIN_BODY_CHARS:
            failure = "response body too short to be a real ownership document"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_form4_body_{accession}", status=StepStatus.FAILED,
                http_requests_made=1, failure_reason=failure,
            )
            return None, _failure(failure, ownership_document_verified=True), http_reqs

        # Cross-check: the XML's OWN issuerCik must match the REQUESTED
        # issuer -- a mismatch is never silently accepted as evidence for
        # this issuer (Phase 3E.1 requirement 1).
        parsed_for_check = parse_ownership_document(text)
        xml_issuer_cik = normalize_cik(parsed_for_check["issuer_cik"]) if parsed_for_check else None
        if xml_issuer_cik != issuer_cik:
            failure = (
                f"issuer CIK mismatch: requested {issuer_cik}, ownership XML declares "
                f"issuerCik={parsed_for_check['issuer_cik'] if parsed_for_check else UNKNOWN!r}"
            )
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_form4_body_{accession}", status=StepStatus.FAILED,
                http_requests_made=1, failure_reason=failure,
            )
            return None, _failure(failure, ownership_document_verified=True), http_reqs

        doc_id = derive_document_id("form4", url, text)
        document = Document(
            doc_id=doc_id,
            url=url,
            title=f"Form {document_type} ownership filing ({accession})",
            publisher="SEC EDGAR",
            filing_date=candidate.get("filing_date", UNKNOWN),
            accession=accession,
            doc_type="form4_ownership_xml",
            is_company_ir=False,
            text=text,
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            retrieved_at=utc_now_iso(),
            authority=DocumentAuthority.STATUTORY_FILING,
        )
        stored = context.document_store.put(
            document, document_id=doc_id, accession=accession,
            filename=normalized.normalized_xml_filename, document_role=DocumentRole.PRIMARY_DOCUMENT,
        )
        entry = {
            "accession": accession, "form": document_type, "document_id": stored.document_id,
            "ownership_xml_text": text, "primary_document": primary_document,
            # Phase 3E.2.2 requirement 4 diagnostics -- kept on every
            # successfully-fetched entry, not just on failures.
            "original_primary_document": primary_document,
            "normalized_xml_filename": normalized.normalized_xml_filename,
            "xsl_wrapper_path": normalized.xsl_wrapper_path,
            "directory_index_verified": True,
            "resolved_raw_xml_url": url,
            "ownership_document_verified": True,
            "issuer_cik_verified": True,
        }
        result = StepExecutionResult(
            step_id=f"_cache_form4_body_{accession}", status=StepStatus.BODY_FETCHED,
            document_id=stored.document_id, http_requests_made=1, payload=entry,
        )
        context.request_cache[cache_key] = result
        return entry, None, http_reqs

    def _fetch(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        candidates = upstream.get("candidates") or []
        issuer_cik = upstream.get("issuer_cik")
        if not candidates or issuer_cik is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no discovered candidates from the LOCATE step",
            )

        fetched: list[dict[str, Any]] = []
        fetch_failures: list[dict[str, Any]] = []
        total_http_requests = 0
        for candidate in candidates:
            entry, failure_entry, http_reqs = self._fetch_one_candidate(candidate, issuer_cik, context)
            total_http_requests += http_reqs
            if entry is not None:
                fetched.append(entry)
            else:
                assert failure_entry is not None
                fetch_failures.append(failure_entry)

        if not fetched:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=total_http_requests,
                failure_reason=f"no candidate could be fetched/verified: {fetch_failures}",
                payload={**upstream, "fetch_failures": fetch_failures},
            )

        # A single document_id "headline" pointer (the most recently
        # filed successfully-fetched candidate) for callers/diagnostics
        # that expect one -- the FULL set is always in "fetched" below.
        headline_document_id = fetched[0]["document_id"]
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.BODY_FETCHED, document_id=headline_document_id,
            http_requests_made=total_http_requests,
            payload={**upstream, "fetched": fetched, "fetch_failures": fetch_failures},
        )

    # -- PARSE: field extraction + amendment reconciliation + diagnostics
    def _reconcile(self, parsed_documents: list[dict[str, Any]]) -> dict[str, Form4Reconciliation]:
        by_accession = {d["accession"] for d in parsed_documents}
        reconciliation: dict[str, Form4Reconciliation] = {}
        for doc in parsed_documents:
            if doc["form"] != "4/A":
                reconciliation[doc["accession"]] = Form4Reconciliation(
                    status="NOT_APPLICABLE", original_accession=UNKNOWN,
                    reason="not an amendment",
                )
                continue
            referenced = doc["parsed"].get("remarks_referenced_accession", UNKNOWN)
            if referenced != UNKNOWN and referenced in by_accession and referenced != doc["accession"]:
                reconciliation[doc["accession"]] = Form4Reconciliation(
                    status="RECONCILED", original_accession=referenced,
                    reason="remarks explicitly reference a co-discovered accession",
                )
            else:
                # Never inferred from matching period/owner alone (Phase
                # 3E.1 requirement 3) -- no explicit, verifiable reference
                # means UNRESOLVED, full stop.
                reconciliation[doc["accession"]] = Form4Reconciliation(
                    status="UNRESOLVED", original_accession=UNKNOWN,
                    reason="no explicit, co-discovered original accession reference in remarks",
                )
        return reconciliation

    def _parse(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        fetched = upstream.get("fetched") or []
        if not fetched:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no fetched ownership XML bodies available to parse",
            )

        parsed_documents: list[dict[str, Any]] = []
        parse_failures: list[dict[str, Any]] = []
        for entry in fetched:
            text = entry.get("ownership_xml_text")
            parsed = parse_ownership_document(text) if text else None
            if parsed is None:
                parse_failures.append({"accession": entry["accession"], "reason": "did not match the expected Form 4/4-A schema on re-parse"})
                continue
            if not parsed.get("issuer_cik") or parsed["issuer_cik"] == UNKNOWN:
                parse_failures.append({"accession": entry["accession"], "reason": "issuer CIK is UNKNOWN in the ownership document"})
                continue
            if not parsed.get("reporting_owners"):
                parse_failures.append({"accession": entry["accession"], "reason": "no reportingOwner recorded"})
                continue
            parsed_documents.append({
                "accession": entry["accession"], "form": entry["form"],
                "document_id": entry["document_id"], "parsed": parsed,
            })

        if not parsed_documents:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"no fetched candidate parsed successfully: {parse_failures}",
                payload={**upstream, "parse_failures": parse_failures},
            )

        reconciliation = self._reconcile(parsed_documents)
        for doc in parsed_documents:
            doc["reconciliation"] = reconciliation[doc["accession"]]

        diagnostics = self._compute_diagnostics(parsed_documents)
        headline_document_id = parsed_documents[0]["document_id"]
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.PARSED, document_id=headline_document_id,
            payload={
                **upstream,
                "parsed_documents": parsed_documents,
                "parse_failures": parse_failures,
                "diagnostics": diagnostics,
            },
        )

    @staticmethod
    def _compute_diagnostics(parsed_documents: list[dict[str, Any]]) -> dict[str, int]:
        diagnostics = {
            "parsed_non_derivative_transactions": 0,
            "parsed_derivative_transactions": 0,
            "parsed_reporting_owners": 0,
            "form4_amendments_found": 0,
            "amendments_reconciled": 0,
            "amendments_unresolved": 0,
            "ten_b5_1_checkbox_true": 0,
            "ten_b5_1_checkbox_false": 0,
            "ten_b5_1_checkbox_unknown": 0,
        }
        for doc in parsed_documents:
            parsed = doc["parsed"]
            diagnostics["parsed_non_derivative_transactions"] += len(parsed["non_derivative_transactions"])
            diagnostics["parsed_derivative_transactions"] += len(parsed["derivative_transactions"])
            diagnostics["parsed_reporting_owners"] += len(parsed["reporting_owners"])
            checkbox = parsed.get("ten_b5_1_checkbox")
            if checkbox is True:
                diagnostics["ten_b5_1_checkbox_true"] += 1
            elif checkbox is False:
                diagnostics["ten_b5_1_checkbox_false"] += 1
            else:
                diagnostics["ten_b5_1_checkbox_unknown"] += 1
            if doc["form"] == "4/A":
                diagnostics["form4_amendments_found"] += 1
                status = doc["reconciliation"].status
                if status == "RECONCILED":
                    diagnostics["amendments_reconciled"] += 1
                elif status == "UNRESOLVED":
                    diagnostics["amendments_unresolved"] += 1
        return diagnostics
