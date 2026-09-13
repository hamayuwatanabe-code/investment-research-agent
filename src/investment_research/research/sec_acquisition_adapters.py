"""SEC primary-document and exhibit adapters (Phase 3A requirements 5/6).

Both adapters implement ``acquisition_executor.AdapterProtocol`` and are
registered with an ``AcquisitionExecutor`` by the caller -- neither is called
from the production pipeline in this phase (see CLAUDE.md's Phase 3A
forbidden-changes list), and every test in this repository drives them
against a fake ``HttpClient`` double. ``ImplementationStatus`` for the steps
these adapters back therefore tops out at OFFLINE_VERIFIED, never
LIVE_VERIFIED: nothing here has been checked against the real SEC site.

Both adapters need a concrete filing/exhibit reference the abstract
Source Routing Graph has no field for (an ``AcquisitionTarget`` only carries
``issuer_identifier``/``program_identifier`` strings, since the catalog
models 31 abstract equivalence-group archetypes, not one concrete filing).
Phase 3A does not wire a real pipeline that would resolve this
automatically, so a caller supplies it explicitly via
``ExecutionContext.filing_references``/``exhibit_selectors``, keyed by
``target_id`` -- see ``SecFilingReference``/``SecExhibitSelectionRequest``.

Dedup (Phase 3A requirement 10) is implemented here, not in the executor:
only the adapter knows what "the same request" means for its source (a CIK's
submissions listing; a specific document URL). Every HTTP GET and API
request is checked against ``ExecutionContext.request_cache`` (a run-scoped
dict shared across every target in one ``AcquisitionExecutor.run()`` call)
before being made.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from ..collectors.documents import Document
from ..collectors.sec_edgar import FILING_INDEX_URL, SUBMISSIONS_URL
from ..schemas.enums import UNKNOWN, ContentKind, DocumentAuthority, Provenance
from ..schemas.fact import utc_now_iso
from .acquisition_executor import ExecutionContext, StepExecutionResult
from .document_store import DocumentRole
from .source_routing import AcquisitionStep, StepKind, StepStatus

log = logging.getLogger(__name__)

#: Below this many characters of extracted text, a "successful" fetch is
#: treated as too short to be a real filing/exhibit body -- an index page, an
#: error page, or a near-empty placeholder all fail this bar (Phase 3A
#: requirement 5: "apply minimum body/content validation").
MIN_BODY_CHARS = 200

#: Default selection priority for exhibits when a caller does not override
#: it -- Phase 3A requirement 6's own examples, most decision-relevant first.
DEFAULT_EXHIBIT_PRIORITY: tuple[str, ...] = (
    "EX-99",
    "material agreement",
    "press release",
    "regulatory correspondence",
    "clinical results",
)


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text: strips tags/scripts/styles, keeps whitespace-
    joined visible text. Not a full HTML parser -- good enough to turn an
    SEC filing's HTML body into plain text for chunking, using only the
    standard library (matching ``collectors/http.py``'s no-new-dependency
    approach)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


def _html_to_text(body: str) -> str:
    if "<" not in body:
        # Plain text already (a fixture, or a non-HTML exhibit) -- nothing
        # to strip.
        return body.strip()
    extractor = _TextExtractor()
    try:
        extractor.feed(body)
    except Exception as exc:  # pragma: no cover - malformed HTML is rare
        log.warning("HTML parse failed, falling back to raw body: %s", exc)
        return body.strip()
    return " ".join(extractor.parts)


_INDEX_PAGE_MARKERS = (
    "edgar filing documents",
    "document format files",
    "index of /archives",
)


def _looks_like_index_page(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INDEX_PAGE_MARKERS)


def _content_is_valid_body(text: str) -> bool:
    """Phase 3A requirement 5: never treat an index page, an HTML error
    page, or a too-short body as a successful fetch."""
    stripped = text.strip()
    if len(stripped) < MIN_BODY_CHARS:
        return False
    return not _looks_like_index_page(stripped)


@dataclass(frozen=True)
class SecFilingReference:
    """The concrete filing identity a caller supplies for one
    SEC-primary-document target -- see the module docstring for why the
    abstract graph cannot carry this itself."""

    cik: int
    accession: str
    primary_document: str
    form: str = UNKNOWN
    filing_date: str = UNKNOWN
    report_date: str = UNKNOWN
    company_name: str = UNKNOWN


@dataclass(frozen=True)
class SecExhibitCandidate:
    filename: str
    exhibit_type: str
    sequence: int = 0
    description: str = UNKNOWN


@dataclass(frozen=True)
class SecExhibitSelectionRequest:
    """The concrete accession + selection policy a caller supplies for one
    SEC-exhibit target. ``fetch_limit`` bounds how many candidates this
    adapter will actually fetch across the whole run for one accession;
    candidates beyond it are reported, never silently dropped (Phase 3A
    requirement 6: "not yet fetched", never "does not exist")."""

    cik: int
    accession: str
    priority_keywords: tuple[str, ...] = DEFAULT_EXHIBIT_PRIORITY
    fetch_limit: int = 1


def _upstream_payload(step: AcquisitionStep, context: ExecutionContext) -> dict[str, Any]:
    """The payload of whichever dependency step this step names -- used by
    FETCH to read its LOCATE step's resolved URL, and by PARSE to read its
    FETCH step's document_id. Both adapters share this: it is about the
    step-DAG's dependency wiring, not about either source specifically."""
    for dep_id in step.depends_on_step_ids:
        payload = context.payload_for(dep_id)
        if payload:
            return dict(payload)
    return {}


def _accession_present(submissions_payload: Any, accession: str) -> bool:
    recent = ((submissions_payload or {}).get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    return accession in accessions


def _select_exhibit(
    candidates: list[SecExhibitCandidate], priority_keywords: tuple[str, ...], fetch_limit: int
) -> tuple[list[SecExhibitCandidate], list[SecExhibitCandidate]]:
    """Rank candidates by ``priority_keywords`` order (first match wins),
    lowest sequence first as a tiebreaker. Returns (within_limit, deferred).
    Never fetches everything unconditionally (requirement 6)."""

    def rank(candidate: SecExhibitCandidate) -> tuple[int, int]:
        haystack = f"{candidate.exhibit_type} {candidate.description}".lower()
        for index, keyword in enumerate(priority_keywords):
            if keyword.lower() in haystack:
                return (index, candidate.sequence)
        return (len(priority_keywords), candidate.sequence)

    ranked = sorted(candidates, key=rank)
    matched = [c for c in ranked if rank(c)[0] < len(priority_keywords)]
    return matched[:fetch_limit], matched[fetch_limit:]


class SecPrimaryDocumentAdapter:
    """LOCATE (submissions metadata -> resolve primaryDocument URL) ->
    FETCH (HTTP GET, validate, build+store a PRIMARY_DOCUMENT Document) ->
    PARSE (confirm the stored document is chunk-ready). Registered under
    adapter_id ``sec_primary_document_adapter``.
    """

    def __init__(self, http: Any) -> None:
        self.http = http

    def execute(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        if step.step_kind is StepKind.LOCATE:
            return self._locate(step, context)
        if step.step_kind is StepKind.FETCH:
            return self._fetch(step, context)
        if step.step_kind is StepKind.PARSE:
            return self._parse(step, context)
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.FAILED,
            failure_reason=f"SecPrimaryDocumentAdapter cannot handle step_kind={step.step_kind}",
        )

    def _locate(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        ref = context.filing_reference_for(step.target_id)
        if ref is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no SecFilingReference supplied for this target",
            )
        if not ref.primary_document:
            # Requirement 5: never treat an empty primaryDocument as a body,
            # and never even attempt to resolve a document URL for one --
            # the deterministic URL for an empty document is the accession's
            # INDEX page, not a document.
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND,
                failure_reason="primaryDocument is empty; refusing to treat the accession index page as a document",
            )

        submissions_cache_key = f"sec_submissions:{ref.cik}"
        cached_submissions = context.request_cache.get(submissions_cache_key)
        api_requests_made = 0
        cache_hit = False
        if cached_submissions is not None:
            cache_hit = True
            if cached_submissions.status is not StepStatus.URL_RESOLVED:
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.FAILED, cache_hit=True,
                    failure_reason=cached_submissions.failure_reason,
                )
            submissions_payload = cached_submissions.payload.get("submissions", {})
        else:
            fetch = self.http.get(SUBMISSIONS_URL.format(cik=ref.cik))
            api_requests_made = 1
            if not fetch.ok:
                result = StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.FAILED, api_requests_made=1,
                    failure_reason=f"submissions fetch failed: {fetch.outcome} {fetch.error}",
                )
                context.request_cache[submissions_cache_key] = result
                return result
            submissions_payload = fetch.json() or {}
            context.request_cache[submissions_cache_key] = StepExecutionResult(
                step_id=f"_cache_sec_submissions_{ref.cik}", status=StepStatus.URL_RESOLVED,
                payload={"submissions": submissions_payload}, api_requests_made=1,
            )

        if not _accession_present(submissions_payload, ref.accession):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND,
                api_requests_made=api_requests_made, cache_hit=cache_hit,
                failure_reason=f"accession {ref.accession} not found in CIK {ref.cik}'s submissions listing",
            )

        url = FILING_INDEX_URL.format(
            cik=ref.cik, accession_nodash=ref.accession.replace("-", ""), document=ref.primary_document
        )
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.URL_RESOLVED,
            api_requests_made=api_requests_made, cache_hit=cache_hit,
            payload={
                "url": url, "accession": ref.accession, "primary_document": ref.primary_document,
                "cik": ref.cik, "form": ref.form, "filing_date": ref.filing_date,
                "report_date": ref.report_date, "company_name": ref.company_name,
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
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=f"GET {url} failed: {fetch.outcome} {fetch.error}",
            )
            context.request_cache[cache_key] = result
            return result

        text = _html_to_text(fetch.text)
        if not _content_is_valid_body(text):
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=(
                    "fetched body failed minimum content validation -- looked like an "
                    "index/error page, or was shorter than the minimum body length"
                ),
            )
            context.request_cache[cache_key] = result
            return result

        doc_id = f"doc_sec_primary_{hashlib.sha256(url.encode()).hexdigest()[:24]}"
        document = Document(
            doc_id=doc_id,
            url=url,
            title=f"{upstream.get('form', UNKNOWN)} primary document ({upstream.get('accession', UNKNOWN)})",
            publisher="SEC EDGAR",
            filing_date=upstream.get("filing_date", UNKNOWN),
            event_date=upstream.get("report_date", UNKNOWN) or UNKNOWN,
            accession=str(upstream.get("accession", UNKNOWN)),
            doc_type="primary_document",
            is_company_ir=True,
            text=text,
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            # Retrieval time is retrieval time only -- never assigned into
            # filing_date/event_date/published_date (Phase 3A requirement 5).
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
        payload = {**upstream, "document_id": stored.document_id}
        result = StepExecutionResult(
            step_id=step.step_id, status=StepStatus.BODY_FETCHED, document_id=stored.document_id,
            http_requests_made=1, payload=payload,
        )
        context.request_cache[cache_key] = result
        return result

    def _parse(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        document_id = upstream.get("document_id")
        if not document_id:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no fetched body available to parse",
            )
        stored = context.document_store.get(document_id)
        if stored is None or not stored.document.text.strip():
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="fetched document has no extractable text",
            )
        return StepExecutionResult(step_id=step.step_id, status=StepStatus.PARSED, document_id=document_id, payload=upstream)


class SecExhibitAdapter:
    """LOCATE (enumerate the accession's exhibit index, apply the selection
    policy) -> FETCH (HTTP GET the selected exhibit body) -> PARSE (confirm
    the stored EXHIBIT document is chunk-ready). Registered under adapter_id
    ``sec_exhibit_enumeration``.

    Consumes a small, documented JSON shape for the exhibit index --
    ``{"items": [{"filename", "type", "sequence", "description"}, ...]}`` --
    rather than SEC's real ``index.json`` wire format, which does not itself
    carry a clean exhibit-type/description field; nothing here claims
    LIVE_VERIFIED fidelity to the real endpoint (see module docstring).
    """

    def __init__(self, http: Any) -> None:
        self.http = http

    def execute(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        if step.step_kind is StepKind.LOCATE:
            return self._locate(step, context)
        if step.step_kind is StepKind.FETCH:
            return self._fetch(step, context)
        if step.step_kind is StepKind.PARSE:
            return self._parse(step, context)
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.FAILED,
            failure_reason=f"SecExhibitAdapter cannot handle step_kind={step.step_kind}",
        )

    def _locate(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        selection = context.exhibit_selector_for(step.target_id)
        if selection is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no SecExhibitSelectionRequest supplied for this target",
            )

        index_url = _accession_index_url(selection.cik, selection.accession)
        cache_key = f"http_get:{index_url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            index_payload = cached.payload.get("index", {})
            http_requests_made = 0
            cache_hit = True
        else:
            fetch = self.http.get(index_url)
            http_requests_made = 1
            cache_hit = False
            if not fetch.ok:
                result = StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                    failure_reason=f"exhibit index fetch failed: {fetch.outcome} {fetch.error}",
                )
                context.request_cache[cache_key] = result
                return result
            index_payload = fetch.json() or {}
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_sec_index_{selection.accession}", status=StepStatus.URL_RESOLVED,
                payload={"index": index_payload}, http_requests_made=1,
            )

        candidates = [
            SecExhibitCandidate(
                filename=item.get("filename", ""), exhibit_type=item.get("type", UNKNOWN),
                sequence=int(item.get("sequence", 0) or 0), description=item.get("description", UNKNOWN),
            )
            for item in (index_payload.get("items") or [])
            if item.get("filename")
        ]
        if not candidates:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND, http_requests_made=http_requests_made,
                cache_hit=cache_hit, failure_reason="accession index listed no exhibits",
            )

        selected, deferred = _select_exhibit(candidates, selection.priority_keywords, selection.fetch_limit)
        if not selected:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND, http_requests_made=http_requests_made,
                cache_hit=cache_hit,
                failure_reason="no exhibit in the index matched the selection priority keywords",
            )

        chosen = selected[0]
        url = FILING_INDEX_URL.format(
            cik=selection.cik, accession_nodash=selection.accession.replace("-", ""), document=chosen.filename
        )
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.URL_RESOLVED,
            http_requests_made=http_requests_made, cache_hit=cache_hit,
            payload={
                "url": url, "accession": selection.accession, "filename": chosen.filename,
                "exhibit_type": chosen.exhibit_type, "cik": selection.cik,
                # Requirement 6: exhibits beyond the fetch limit are recorded
                # as not-yet-fetched, never as nonexistent.
                "not_yet_fetched": [c.filename for c in deferred],
            },
        )

    def _fetch(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        url = upstream.get("url")
        if not url:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no resolved exhibit URL from the LOCATE step",
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
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=f"GET {url} failed: {fetch.outcome} {fetch.error}",
            )
            context.request_cache[cache_key] = result
            return result

        text = _html_to_text(fetch.text)
        if not _content_is_valid_body(text):
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason="fetched exhibit body failed minimum content validation",
            )
            context.request_cache[cache_key] = result
            return result

        doc_id = f"doc_sec_exhibit_{hashlib.sha256(url.encode()).hexdigest()[:24]}"
        document = Document(
            doc_id=doc_id,
            url=url,
            title=f"Exhibit {upstream.get('exhibit_type', UNKNOWN)} ({upstream.get('accession', UNKNOWN)})",
            publisher="SEC EDGAR",
            accession=str(upstream.get("accession", UNKNOWN)),
            doc_type="exhibit",
            is_company_ir=True,
            text=text,
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            retrieved_at=utc_now_iso(),
            authority=DocumentAuthority.STATUTORY_FILING,
        )
        stored = context.document_store.put(
            document,
            document_id=doc_id,
            accession=document.accession,
            # (accession, filename, document_role) distinguishes this from
            # the primary document even within the same accession (Phase 3A
            # requirement 6).
            filename=str(upstream.get("filename", "")),
            document_role=DocumentRole.EXHIBIT,
        )
        payload = {**upstream, "document_id": stored.document_id}
        result = StepExecutionResult(
            step_id=step.step_id, status=StepStatus.BODY_FETCHED, document_id=stored.document_id,
            http_requests_made=1, payload=payload,
        )
        context.request_cache[cache_key] = result
        return result

    def _parse(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        document_id = upstream.get("document_id")
        if not document_id:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no fetched exhibit body available to parse",
            )
        stored = context.document_store.get(document_id)
        if stored is None or not stored.document.text.strip():
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="fetched exhibit has no extractable text",
            )
        return StepExecutionResult(step_id=step.step_id, status=StepStatus.PARSED, document_id=document_id, payload=upstream)


def _accession_index_url(cik: int, accession: str) -> str:
    return FILING_INDEX_URL.format(cik=cik, accession_nodash=accession.replace("-", ""), document="index.json")
