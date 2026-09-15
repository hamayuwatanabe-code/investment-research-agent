"""SEC Form 4 (Section 16 ownership) Direct Adapter (Phase 3E).

Fetches ONE Form 4/4-A ownership XML document from SEC EDGAR when the
accession/primary-document filename is already known -- never by ticker or
company-name search, never via Web Search, never via the Anthropic API.
Mirrors ``sec_acquisition_adapters.SecPrimaryDocumentAdapter`` closely
(LOCATE cross-checks the caller-supplied reference against SEC's own
submissions metadata and accession directory listing before ever
resolving a URL; the ``sec_chain`` catalog archetype fits this adapter
without modification), reusing its own submissions/directory fetch-and-
cache methods directly (composition, not duplication) so a Form 4 target
and an SEC primary-document target sharing the same filer CIK/accession
within one ``AcquisitionExecutor.run()`` call never issue a second real
request for the same submissions listing or directory index (Phase 3E
requirement 2).

Registered with an ``AcquisitionExecutor`` by the caller -- like every
other adapter in this repository, this is NOT called from the production
pipeline (see CLAUDE.md's Phase 3A-3D forbidden-changes lists), and every
test drives it against a fake HTTP double loaded from
``tests/fixtures/form4_real_format/``. ``ImplementationStatus`` for the
steps this adapter backs therefore tops out at OFFLINE_VERIFIED, never
LIVE_VERIFIED: nothing here has ever been checked against the real SEC
EDGAR site.

Safe identification of the acquisition target (Phase 3E requirement 3):

* Form 3, Form 4, and Form 5 share the identical ``ownershipDocument`` XML
  schema -- distinguished ONLY by ``documentType``. This adapter cross-
  checks BOTH the caller-supplied reference's expectation and SEC's own
  submissions-metadata ``form`` value against
  ``form4.VALID_FORM4_DOCUMENT_TYPES`` at LOCATE time, and re-checks the
  fetched XML's own ``documentType`` at FETCH time -- a Form 3/5 (or
  anything else) accession is refused at whichever point the mismatch is
  first detectable, never silently accepted because "some SEC filing" was
  found.
* ``4/A`` is kept distinct from ``4`` throughout -- an amendment is never
  read as if it were the original filing. Since ``DocumentStore`` versions
  by URL (see ``document_store.derive_document_id``'s docstring) and a
  4/A is filed under its OWN accession/URL, it is naturally stored as a
  SEPARATE document, never a new "version" of the original Form 4.
* ``Form4FilingReference.filer_cik`` (whose submissions.json + Archives
  directory this accession actually lives under) is kept structurally
  separate from ``issuer_cik`` and ``reporting_owner_cik`` (carried
  through only as filing metadata) -- this adapter never assumes the
  issuer's CIK is also the filer CIK for Section 16 purposes, and never
  guesses one from the other.
* The XML filename is resolved from SEC's own submissions metadata
  (``primaryDocument``) and cross-checked against the accession's
  directory listing -- never guessed from a ticker or company name.

Evidence Integrity (Phase 3E requirement 6): Form 4 is the REPORTING
OWNER's own statutory filing (``DocumentAuthority.STATUTORY_FILING``,
``Document.is_company_ir=False`` -- the issuer neither authored nor
controls this channel). See ``collectors/form4.py``'s module docstring for
the full transaction-semantics boundary (codes kept verbatim, acquired/
disposed never read as buy/sell, derivative/non-derivative never merged,
10b5-1 only on explicit footnote match). Nothing here generates an
investment Action, promotes a domain to SUFFICIENT, or interprets a
transaction as a judgment about the issuer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..collectors.documents import Document
from ..collectors.form4 import (
    VALID_FORM4_DOCUMENT_TYPES,
    check_ownership_xml_shape,
    parse_ownership_document,
)
from ..schemas.enums import UNKNOWN, ContentKind, DocumentAuthority, FetchOutcome, Provenance
from ..schemas.fact import utc_now_iso
from .acquisition_executor import ExecutionContext, StepExecutionResult
from .document_store import DocumentRole, derive_document_id
from .sec_acquisition_adapters import (
    SecPrimaryDocumentAdapter,
    _resolve_document_url,
    _submissions_primary_document,
    _validate_accession,
    _validate_filename,
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


@dataclass(frozen=True)
class Form4FilingReference:
    """The concrete filing identity a caller supplies for one Form 4
    target -- see the module docstring for why ``filer_cik`` is kept
    separate from ``issuer_cik``/``reporting_owner_cik``. Every field
    beyond ``filer_cik``/``accession``/``primary_document`` is CALLER-
    SUPPLIED METADATA, cross-checked where SEC's own submissions payload
    can confirm it (form type, primary document filename) and passed
    through as-is otherwise -- never re-derived or guessed by this
    adapter.
    """

    #: The CIK whose submissions.json + Archives accession directory this
    #: filing actually lives under (commonly the reporting owner's own
    #: CIK, but this adapter makes no assumption -- the caller supplies
    #: whichever CIK is actually correct for this accession).
    filer_cik: int
    accession: str
    primary_document: str
    issuer_cik: str = UNKNOWN
    issuer_name: str = UNKNOWN
    issuer_ticker: str = UNKNOWN
    reporting_owner_cik: str = UNKNOWN
    reporting_owner_name: str = UNKNOWN
    filing_date: str = UNKNOWN
    period_of_report: str = UNKNOWN


def _upstream_payload(step: AcquisitionStep, context: ExecutionContext) -> dict[str, Any]:
    for dep_id in step.depends_on_step_ids:
        payload = context.payload_for(dep_id)
        if payload:
            return dict(payload)
    return {}


def _submissions_form(payload: Any, accession: str) -> str | None:
    """The ``form`` value SEC's own submissions payload records for
    ``accession`` -- the authoritative value, mirroring
    ``sec_acquisition_adapters._submissions_primary_document``'s pattern
    exactly. ``None`` if the accession is not present at all."""
    recent = ((payload or {}).get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    if accession not in accessions:
        return None
    index = accessions.index(accession)
    forms = recent.get("form") or []
    return forms[index] if index < len(forms) else None


class Form4Adapter:
    """LOCATE (submissions metadata + directory listing cross-check,
    reusing ``SecPrimaryDocumentAdapter``'s own cached fetch methods) ->
    FETCH (HTTP GET, validate the body is well-formed Form 4/4-A
    ownership XML, store a PRIMARY_DOCUMENT Document) -> PARSE (extract
    ownership fields via ``collectors.form4.parse_ownership_document`` and
    confirm the minimum required fields are present). Registered under
    adapter_id ``form4_xml_parser``.
    """

    def __init__(self, http: Any) -> None:
        self.http = http
        #: Composition, not duplication: this adapter's LOCATE step needs
        #: EXACTLY the same submissions/directory fetch-and-cache behavior
        #: ``SecPrimaryDocumentAdapter`` already implements (including its
        #: exact ``request_cache`` key conventions), so the two adapters
        #: share one real HTTP call for the same CIK/accession within one
        #: run (Phase 3E requirement 2) -- reusing the code, not merely
        #: matching its shape by hand.
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

    def _locate(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        ref = context.form4_reference_for(step.target_id)
        if ref is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no Form4FilingReference supplied for this target",
            )
        if not _validate_accession(ref.accession):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"malformed accession number: {ref.accession!r}",
            )
        if not ref.primary_document or not _validate_filename(ref.primary_document):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"unsafe or empty primaryDocument filename: {ref.primary_document!r}",
            )

        submissions_payload, api_requests_made, cache_hit_submissions, failure = (
            self._sec._fetch_submissions(ref.filer_cik, context)
        )
        if failure is not None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=failure,
            )

        authoritative_form = _submissions_form(submissions_payload, ref.accession)
        if authoritative_form is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=f"accession {ref.accession} not found in CIK {ref.filer_cik}'s submissions listing",
            )
        if authoritative_form not in VALID_FORM4_DOCUMENT_TYPES:
            # Requirement 3: never treat a Form 3/5 (or anything else) as
            # a Form 4 merely because SOME filing was found at this
            # accession.
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=(
                    f"accession {ref.accession} is form {authoritative_form!r} per SEC's own "
                    "submissions metadata, not Form 4/4-A -- refusing to treat it as one"
                ),
            )

        authoritative_document = _submissions_primary_document(submissions_payload, ref.accession)
        if authoritative_document is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=f"accession {ref.accession} not found in CIK {ref.filer_cik}'s submissions listing",
            )
        if authoritative_document.lower() != ref.primary_document.lower():
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=(
                    f"primaryDocument mismatch: expected {ref.primary_document!r}, "
                    f"submissions metadata records {authoritative_document!r}"
                ),
            )

        accession_nodash = ref.accession.replace("-", "")
        directory, http_requests_made, cache_hit_directory, dir_failure = self._sec._fetch_directory(
            ref.filer_cik, ref.accession, accession_nodash, context
        )
        if dir_failure is not None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, http_requests_made=http_requests_made,
                cache_hit=cache_hit_submissions or cache_hit_directory,
                failure_reason=dir_failure,
            )
        directory_filenames = {item.get("name", "").lower() for item in (directory.get("item") or [])}
        if ref.primary_document.lower() not in directory_filenames:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND,
                api_requests_made=api_requests_made, http_requests_made=http_requests_made,
                cache_hit=cache_hit_submissions or cache_hit_directory,
                failure_reason=(
                    f"primaryDocument {ref.primary_document!r} is not present in the accession "
                    "directory listing -- submissions metadata and the directory disagree"
                ),
            )

        url = _resolve_document_url(ref.filer_cik, accession_nodash, ref.primary_document)
        if url is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, http_requests_made=http_requests_made,
                failure_reason="could not safely resolve a document URL",
            )
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.URL_RESOLVED,
            api_requests_made=api_requests_made, http_requests_made=http_requests_made,
            cache_hit=cache_hit_submissions and cache_hit_directory,
            payload={
                "url": url, "accession": ref.accession, "primary_document": ref.primary_document,
                "filer_cik": ref.filer_cik, "form": authoritative_form,
                "issuer_cik": ref.issuer_cik, "issuer_name": ref.issuer_name, "issuer_ticker": ref.issuer_ticker,
                "reporting_owner_cik": ref.reporting_owner_cik, "reporting_owner_name": ref.reporting_owner_name,
                "filing_date": ref.filing_date, "period_of_report": ref.period_of_report,
            },
        )

    def _fetch(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        url = upstream.get("url")
        if not url:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no resolved URL from the LOCATE step",
            )

        cache_key = f"http_get:{url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            return StepExecutionResult(
                step_id=step.step_id, status=cached.status, document_id=cached.document_id,
                payload=cached.payload, cache_hit=True, failure_reason=cached.failure_reason,
            )

        fetch = self.http.get(url)
        if not fetch.ok:
            status = StepStatus.NOT_FOUND if fetch.outcome is FetchOutcome.NOT_FOUND else StepStatus.FAILED
            result = StepExecutionResult(
                step_id=step.step_id, status=status, http_requests_made=1,
                failure_reason=f"GET {url} failed: {fetch.outcome} {fetch.error}",
            )
            context.request_cache[cache_key] = result
            return result

        text = fetch.text
        if not text.strip():
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason="response body was empty",
            )
            context.request_cache[cache_key] = result
            return result

        shape_error, reason_or_type = check_ownership_xml_shape(text)
        if shape_error is not None:
            # Requirement 10: malformed XML, a missing <ownershipDocument>
            # root, and a Form 3/5 wrong-document-type are each their own
            # distinguishable failure -- never collapsed into one message.
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=f"{shape_error.value}: {reason_or_type}",
            )
            context.request_cache[cache_key] = result
            return result
        document_type = reason_or_type

        if len(text.strip()) < MIN_BODY_CHARS:
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason="response body too short to be a real ownership document",
            )
            context.request_cache[cache_key] = result
            return result

        # Content-dependent, not URL-only -- see document_store.
        # derive_document_id()'s docstring (Phase 3D.1) for why a URL-only
        # id risks a self-referencing version-chain cycle if this exact
        # URL is ever re-fetched with changed content. A 4/A amendment is
        # filed under its OWN accession/URL, so it naturally derives its
        # own id and is stored as a separate document, never a "version"
        # of the original Form 4 (Phase 3E requirement 8).
        doc_id = derive_document_id("form4", url, text)
        document = Document(
            doc_id=doc_id,
            url=url,
            title=f"Form {document_type} ownership filing ({upstream.get('accession', UNKNOWN)})",
            publisher="SEC EDGAR",
            filing_date=upstream.get("filing_date", UNKNOWN),
            event_date=upstream.get("period_of_report", UNKNOWN) or UNKNOWN,
            accession=str(upstream.get("accession", UNKNOWN)),
            doc_type="form4_ownership_xml",
            # A Form 4 is filed BY the reporting owner, THROUGH SEC EDGAR
            # -- the issuer does not control this channel and did not
            # author it (Phase 3E requirement 6 / module docstring).
            is_company_ir=False,
            text=text,
            # A complete ownership XML record was retrieved whole -- never
            # merely because HTTP returned 200 (the shape/length checks
            # above already ran).
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            # Retrieval time is retrieval time only -- never assigned into
            # filing_date/period_of_report/transaction dates.
            retrieved_at=utc_now_iso(),
            authority=DocumentAuthority.STATUTORY_FILING,
        )
        stored = context.document_store.put(
            document,
            document_id=doc_id,
            accession=document.accession,
            filename=str(upstream.get("primary_document", "")),
            document_role=DocumentRole.PRIMARY_DOCUMENT,
        )
        payload = {**upstream, "document_id": stored.document_id, "document_type": document_type, "ownership_xml_text": text}
        result = StepExecutionResult(
            step_id=step.step_id, status=StepStatus.BODY_FETCHED, document_id=stored.document_id,
            http_requests_made=1, payload=payload,
        )
        context.request_cache[cache_key] = result
        return result

    def _parse(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        document_id = upstream.get("document_id")
        text = upstream.get("ownership_xml_text")
        if not document_id or not text:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no fetched ownership XML body available to parse",
            )
        parsed = parse_ownership_document(text)
        if parsed is None:
            # FETCH already gated shape; PARSE never trusts that blindly
            # and re-validates independently.
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, document_id=document_id,
                failure_reason="ownership XML did not match the expected Form 4/4-A schema on re-parse",
            )
        if not parsed.get("issuer_cik") or parsed["issuer_cik"] == UNKNOWN:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, document_id=document_id,
                payload={**upstream, "parsed": parsed},
                failure_reason="issuer CIK is UNKNOWN in the ownership document -- required fields not present",
            )
        if not parsed.get("reporting_owners"):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, document_id=document_id,
                payload={**upstream, "parsed": parsed},
                failure_reason="no reportingOwner recorded in the ownership document -- required fields not present",
            )
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.PARSED,
            document_id=document_id, payload={**upstream, "parsed": parsed},
        )
