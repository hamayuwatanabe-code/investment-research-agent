"""SEC primary-document and exhibit adapters (Phase 3A requirements 5/6;
Phase 3B: real SEC EDGAR response-shape fidelity).

Both adapters implement ``acquisition_executor.AdapterProtocol`` and are
registered with an ``AcquisitionExecutor`` by the caller -- neither is called
from the production pipeline in this phase (see CLAUDE.md's Phase 3A/3B
forbidden-changes lists), and every test in this repository drives them
against a fake ``HttpClient`` double loaded from
``tests/fixtures/sec_edgar_real_format/`` (see that directory's MANIFEST.md
for provenance). ``ImplementationStatus`` for the steps these adapters back
therefore tops out at OFFLINE_VERIFIED, never LIVE_VERIFIED: nothing here has
been checked against the real SEC site.

Phase 3B corrects Phase 3A's simplified, non-real exhibit-index JSON shape.
Real SEC EDGAR exposes (see ``tests/fixtures/sec_edgar_real_format/
MANIFEST.md`` for the exact schemas modeled):

* ``.../submissions/CIK{cik:010d}.json`` -- filing history, including
  ``isInlineXBRL`` per filing.
* ``.../{accession-nodash}/index.json`` -- a directory listing of
  filename/type/size/last-modified. ``type`` is populated for most exhibits
  but is UNRELIABLE for XBRL viewer artifacts (``R1.htm``, ...), and this
  endpoint carries NO Sequence or Description at all.
* ``.../{accession-nodash}/{accession}-index.htm`` -- the filing detail page,
  whose "Document Format Files" table is the only reliable source of
  Sequence + Description together. This adapter fetches it to enrich
  ``index.json``'s bare listing; if it is unreachable, exhibit selection
  degrades to Type-only classification and says so explicitly
  (``detail_html_unavailable`` in the LOCATE payload) rather than silently
  proceeding as if nothing were missing.

Both adapters need a concrete filing/exhibit reference the abstract
Source Routing Graph has no field for (an ``AcquisitionTarget`` only carries
``issuer_identifier``/``program_identifier`` strings, since the catalog
models 31 abstract equivalence-group archetypes, not one concrete filing).
Phase 3A/3B do not wire a real pipeline that would resolve this
automatically, so a caller supplies it explicitly via
``ExecutionContext.filing_references``/``exhibit_selectors``, keyed by
``target_id`` -- see ``SecFilingReference``/``SecExhibitSelectionRequest``.

Dedup (Phase 3A requirement 10) is implemented here, not in the executor:
only the adapter knows what "the same request" means for its source (a CIK's
submissions listing; a specific document URL). Every HTTP GET and API
request is checked against ``ExecutionContext.request_cache`` (a run-scoped
dict shared across every target in one ``AcquisitionExecutor.run()`` call)
before being made -- including the directory ``index.json``, which both
adapters need for the SAME accession and must therefore share, never
re-fetch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
from typing import Any

from ..collectors.documents import Document
from ..collectors.sec_edgar import FILING_INDEX_URL, SUBMISSIONS_URL
from ..schemas.enums import UNKNOWN, ContentKind, DocumentAuthority, Provenance
from ..schemas.fact import utc_now_iso
from .acquisition_executor import ExecutionContext, StepExecutionResult
from .document_store import DocumentRole, derive_document_id
from .source_routing import AcquisitionStep, StepKind, StepStatus

log = logging.getLogger(__name__)

#: Below this many characters of extracted text, a "successful" fetch is
#: treated as too short to be a real filing/exhibit body -- an index page, an
#: error page, or a near-empty placeholder all fail this bar (Phase 3A
#: requirement 5: "apply minimum body/content validation").
MIN_BODY_CHARS = 200

#: Default selection priority for narrative exhibits when a caller does not
#: override it -- Phase 3A requirement 6's own examples, most
#: decision-relevant first. Never consulted for a candidate this module has
#: already classified as PRIMARY_DOCUMENT/GRAPHIC/XBRL_TECHNICAL/DUPLICATE
#: (Phase 3B requirement 4: never select by filename/extension alone, and
#: never re-offer the primary document or a technical/graphic artifact as if
#: it were narrative evidence).
DEFAULT_EXHIBIT_PRIORITY: tuple[str, ...] = (
    "EX-99",
    "material agreement",
    "press release",
    "regulatory correspondence",
    "clinical results",
)

#: SEC's real accession-number format: 10-digit filer CIK, 2-digit year,
#: 6-digit sequence, dash-separated (Phase 3B requirement 3).
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


def _validate_accession(accession: str) -> bool:
    return bool(_ACCESSION_RE.match(accession or ""))


def _validate_filename(name: str) -> bool:
    """A bare filename within the accession directory -- never a path
    (``/``, ``..``) and never something that is already a full URL. Phase 3B
    requirement 3: a malformed or unsafe name is refused, never "helpfully"
    joined into a URL that could point outside the accession directory."""
    if not name:
        return False
    if "/" in name or "\\" in name or ".." in name:
        return False
    return not (name.lower().startswith("http://") or name.lower().startswith("https://"))


def _directory_index_json_url(cik: int, accession_nodash: str) -> str:
    return FILING_INDEX_URL.format(cik=cik, accession_nodash=accession_nodash, document="index.json")


def _filing_detail_html_url(cik: int, accession: str, accession_nodash: str) -> str:
    return FILING_INDEX_URL.format(
        cik=cik, accession_nodash=accession_nodash, document=f"{accession}-index.htm"
    )


def _resolve_document_url(cik: int, accession_nodash: str, name_or_href: str) -> str | None:
    """Canonicalize a document reference into an absolute URL under the
    accession's Archives directory. Handles all three shapes SEC pages
    actually use (Phase 3B requirement 3):

    * a bare filename (the common case, from ``index.json`` or a filing
      detail table's ``<a href>``) -- resolved relative to the accession
      directory;
    * an absolute path (``/Archives/edgar/data/...``) -- prefixed with the
      SEC host;
    * an already-absolute URL -- returned unchanged, but only if it is
      actually an ``https://www.sec.gov/...`` URL (never trusted blindly:
      an off-host absolute URL is refused, never followed).

    Returns ``None`` for anything unsafe -- an empty name, a path-traversal
    attempt, or an absolute URL pointing off the SEC host -- rather than
    guessing.
    """
    if not name_or_href:
        return None
    if name_or_href.startswith("https://www.sec.gov/"):
        return name_or_href
    if name_or_href.startswith("http://") or name_or_href.startswith("https://"):
        return None  # an absolute URL to somewhere else -- never followed
    if name_or_href.startswith("/"):
        if ".." in name_or_href:
            return None
        return f"https://www.sec.gov{name_or_href}"
    if not _validate_filename(name_or_href):
        return None
    return FILING_INDEX_URL.format(cik=cik, accession_nodash=accession_nodash, document=name_or_href)


class _TextExtractor(HTMLParser):
    """HTML/Inline-XBRL-to-text: strips tags/scripts/styles/navigation and
    everything visually hidden, keeps whitespace-joined visible text.

    Phase 3B additions over Phase 3A's minimal stripper:

    * ``ix:header``/``ix:hidden`` (inline XBRL's own non-visible metadata
      block) is skipped entirely -- the *visible* ``ix:nonFraction``/
      ``ix:nonNumeric`` tags outside that block are ordinary inline elements
      and their text passes through normally, since that is exactly the
      number/fact a reader sees on the rendered page.
    * Any element carrying ``style="display:none"`` (with or without the
      space after the colon) is skipped -- a common way real filings hide
      duplicate/legacy content from the rendered page.
    * ``nav``/``header``/``footer`` chrome is skipped, for pages that carry
      site navigation alongside the filing body.

    Not a full HTML/CSS engine -- good enough to turn an SEC filing's HTML
    body into plain text for chunking, using only the standard library
    (matching ``collectors/http.py``'s no-new-dependency approach).
    """

    _SKIP_TAGS = ("script", "style", "title", "ix:header", "nav", "header", "footer")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_stack: list[str] = []
        self.parts: list[str] = []

    def _is_hidden(self, attrs: list[tuple[str, str | None]]) -> bool:
        for name, value in attrs:
            if name == "style" and value and re.search(r"display\s*:\s*none", value, re.I):
                return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_stack:
            self._skip_stack.append(tag)
            return
        if tag in self._SKIP_TAGS or self._is_hidden(attrs):
            self._skip_stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing tags (e.g. <br/>) never open a skip scope.
        return

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack:
            self._skip_stack.pop()

    def handle_data(self, data: str) -> None:
        if not self._skip_stack and data.strip():
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


#: A bare Apache/EDGAR-style directory listing -- never the filing itself.
_INDEX_PAGE_MARKERS = (
    "edgar filing documents",
    "document format files",
    "index of /archives",
)

#: SEC's own access-denial wording, and generic rate-limit/error pages.
#: Phase 3B requirement 5: none of these may ever be accepted as a body,
#: however "successful" the HTTP status looked.
_REJECTION_PAGE_MARKERS = (
    "undeclared automated tool",
    "too many requests",
    "request denied",
    "service unavailable",
    "429 too many requests",
    "503 service unavailable",
)


def _looks_like_index_page(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INDEX_PAGE_MARKERS)


def _looks_like_rejection_page(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REJECTION_PAGE_MARKERS)


def _content_is_valid_body(text: str) -> bool:
    """Phase 3A/3B requirement 5: never treat an index page, an access-
    denial/rate-limit page, or a too-short body as a successful fetch.
    ``FULL_DOCUMENT`` is set by the caller only when this returns True --
    HTTP 200 alone is never sufficient (Phase 3B requirement 5)."""
    stripped = text.strip()
    if len(stripped) < MIN_BODY_CHARS:
        return False
    if _looks_like_index_page(stripped):
        return False
    return not _looks_like_rejection_page(stripped)


@dataclass(frozen=True)
class SecFilingReference:
    """The concrete filing identity a caller supplies for one
    SEC-primary-document target -- see the module docstring for why the
    abstract graph cannot carry this itself. ``primary_document`` is treated
    as an EXPECTED value, cross-checked against the submissions payload's
    own record for ``accession`` -- never trusted blindly (Phase 3B
    requirement 3)."""

    cik: int
    accession: str
    primary_document: str
    form: str = UNKNOWN
    filing_date: str = UNKNOWN
    report_date: str = UNKNOWN
    company_name: str = UNKNOWN


class ExhibitCategory(str, Enum):
    """What KIND of accession-directory entry this is -- decided from
    Type/Sequence/Description/filename together, never from filename or
    extension alone (Phase 3B requirement 4). Only ``PRESS_RELEASE``,
    ``MATERIAL_AGREEMENT``, ``CERTIFICATION`` and ``OTHER_NARRATIVE`` are
    ever eligible to be fetched as exhibit content; the rest are structurally
    excluded from selection regardless of priority keywords."""

    PRIMARY_DOCUMENT = "PRIMARY_DOCUMENT"
    PRESS_RELEASE = "PRESS_RELEASE"
    MATERIAL_AGREEMENT = "MATERIAL_AGREEMENT"
    CERTIFICATION = "CERTIFICATION"
    XBRL_TECHNICAL = "XBRL_TECHNICAL"
    GRAPHIC = "GRAPHIC"
    OTHER_NARRATIVE = "OTHER_NARRATIVE"
    UNKNOWN = "UNKNOWN"

    @property
    def is_fetchable_as_narrative_content(self) -> bool:
        return self in (
            ExhibitCategory.PRESS_RELEASE,
            ExhibitCategory.MATERIAL_AGREEMENT,
            ExhibitCategory.CERTIFICATION,
            ExhibitCategory.OTHER_NARRATIVE,
        )


_GRAPHIC_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff")
_XBRL_EXTENSIONS = (".xsd", ".cal", ".def", ".lab", ".pre")


def classify_exhibit_entry(
    *, filename: str, exhibit_type: str, description: str, primary_document: str
) -> ExhibitCategory:
    """Phase 3B requirement 4: classify by Type + Description + filename
    together -- never by filename/extension alone, though extension is one
    input among several (a real ``index.json`` entry for an XBRL linkbase
    carries an ``EX-101.*`` type, but a defensive extension check catches
    the rare case where Type was blank, e.g. the ``R1.htm``/``R2.htm``
    XBRL-viewer report pages this module's own fixture reproduces)."""
    name = (filename or "").lower()
    kind = (exhibit_type or "").upper()
    desc = (description or "").lower()

    if filename and primary_document and filename.lower() == primary_document.lower():
        return ExhibitCategory.PRIMARY_DOCUMENT
    if kind.startswith("EX-101") or any(name.endswith(ext) for ext in _XBRL_EXTENSIONS):
        return ExhibitCategory.XBRL_TECHNICAL
    if kind == "GRAPHIC" or any(name.endswith(ext) for ext in _GRAPHIC_EXTENSIONS):
        return ExhibitCategory.GRAPHIC
    if re.match(r"^EX-3[12]", kind) or "certification" in desc:
        return ExhibitCategory.CERTIFICATION
    if kind.startswith("EX-99") or "press release" in desc:
        return ExhibitCategory.PRESS_RELEASE
    if kind.startswith("EX-10") or "agreement" in desc:
        return ExhibitCategory.MATERIAL_AGREEMENT
    # An auto-generated XBRL viewer report page (real EDGAR: R1.htm, R2.htm,
    # ...) carries no informative Type at all -- never offer it as narrative
    # evidence merely because nothing else matched.
    if re.match(r"^R\d+\.htm$", filename or "", re.I) and not kind:
        return ExhibitCategory.XBRL_TECHNICAL
    if not kind and not desc:
        return ExhibitCategory.UNKNOWN
    return ExhibitCategory.OTHER_NARRATIVE


@dataclass(frozen=True)
class SecExhibitCandidate:
    filename: str
    exhibit_type: str = UNKNOWN
    sequence: int = 0
    description: str = UNKNOWN
    category: ExhibitCategory = ExhibitCategory.UNKNOWN


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
    #: The accession's own primary document filename, when known -- used
    #: only to exclude it from exhibit candidates (Phase 3B requirement 4:
    #: the primary document is never re-offered as if it were an exhibit).
    #: Leaving this UNKNOWN never blocks exhibit selection; it only means
    #: this one exclusion cannot be applied.
    primary_document: str = UNKNOWN


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


def _submissions_primary_document(submissions_payload: Any, accession: str) -> str | None:
    """The primaryDocument SEC's own submissions payload records for
    ``accession`` -- the authoritative value. Returns ``None`` if the
    accession is not present at all."""
    recent = ((submissions_payload or {}).get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    if accession not in accessions:
        return None
    index = accessions.index(accession)
    documents = recent.get("primaryDocument") or []
    return documents[index] if index < len(documents) else None


def _select_narrative_exhibit(
    candidates: list[SecExhibitCandidate], priority_keywords: tuple[str, ...], fetch_limit: int
) -> tuple[list[SecExhibitCandidate], list[SecExhibitCandidate]]:
    """Rank NARRATIVE-eligible candidates by ``priority_keywords`` order
    (first match wins), lowest sequence first as a tiebreaker. Returns
    (within_limit, deferred). Never fetches everything unconditionally
    (Phase 3A requirement 6), and never ranks a PRIMARY_DOCUMENT/GRAPHIC/
    XBRL_TECHNICAL/UNKNOWN entry at all (Phase 3B requirement 4)."""
    narrative = [c for c in candidates if c.category.is_fetchable_as_narrative_content]

    def rank(candidate: SecExhibitCandidate) -> tuple[int, int]:
        haystack = f"{candidate.exhibit_type} {candidate.description} {candidate.category.value}".lower()
        for index, keyword in enumerate(priority_keywords):
            if keyword.lower() in haystack:
                return (index, candidate.sequence)
        return (len(priority_keywords), candidate.sequence)

    ranked = sorted(narrative, key=rank)
    matched = [c for c in ranked if rank(c)[0] < len(priority_keywords)]
    return matched[:fetch_limit], matched[fetch_limit:]


class _DirectoryTableParser(HTMLParser):
    """Parses every ``<table>``'s rows into lists of (cell text, first
    ``<a href>`` inside the cell, if any) -- used to read a filing detail
    page's "Document Format Files"/"Data Files" tables (Phase 3B requirement
    1). Header rows (``<th>``) are collected but not returned as data rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._table_depth = 0
        self._row: list[tuple[str, str | None]] | None = None
        self._cell_text: list[str] = []
        self._cell_href: str | None = None
        self._in_cell = False
        self._row_is_header = False
        self.rows: list[list[tuple[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
        elif tag == "tr" and self._table_depth:
            self._row = []
            self._row_is_header = False
        elif tag in ("td", "th") and self._row is not None:
            self._in_cell = True
            self._cell_text = []
            self._cell_href = None
            if tag == "th":
                self._row_is_header = True
        elif tag == "a" and self._in_cell:
            for name, value in attrs:
                if name == "href":
                    self._cell_href = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._table_depth:
            self._table_depth -= 1
        elif tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            assert self._row is not None
            self._row.append((" ".join(self._cell_text).strip(), self._cell_href))
        elif tag == "tr" and self._row is not None:
            if self._row and not self._row_is_header:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and data.strip():
            self._cell_text.append(data.strip())


def _parse_filing_detail_table(html: str) -> list[SecExhibitCandidate]:
    """Parse a filing detail (``-index.htm``) page's document tables into
    candidates carrying Sequence + Description + Type together -- the
    combination ``index.json`` alone cannot provide (Phase 3B requirement 1
    -- see the module docstring's documented limitation)."""
    parser = _DirectoryTableParser()
    try:
        parser.feed(html)
    except Exception as exc:  # pragma: no cover - malformed HTML is rare
        log.warning("filing detail table parse failed: %s", exc)
        return []

    candidates: list[SecExhibitCandidate] = []
    for row in parser.rows:
        # Real layout: Seq | Description | Document | Type | Size (5 cells).
        # Anything shorter is not a document row (e.g. a stray table on the
        # same page) and is skipped, never guessed at.
        if len(row) < 5:
            continue
        seq_text, description_text, document_cell, type_text = row[0][0], row[1][0], row[2], row[3][0]
        filename = document_cell[1] or document_cell[0]
        if not filename or "/" in filename.strip("/"):
            filename = filename.rsplit("/", 1)[-1] if filename else ""
        if not filename:
            continue
        try:
            sequence = int(seq_text)
        except ValueError:
            sequence = 0
        candidates.append(
            SecExhibitCandidate(
                filename=filename, exhibit_type=type_text or UNKNOWN,
                sequence=sequence, description=description_text or UNKNOWN,
            )
        )
    return candidates


def _merge_candidates(
    from_index_json: list[SecExhibitCandidate], from_detail_html: list[SecExhibitCandidate]
) -> list[SecExhibitCandidate]:
    """Merge by filename: the filing detail page's Sequence+Description win
    when both sources list the same file (it is the only reliable source for
    those two fields); ``index.json``-only entries (e.g. auto-generated
    XBRL-viewer report pages the detail page's tables do not list) are kept
    as-is. Never silently drops an entry either source names -- that would
    be exactly the "duplicate entry" confusion Phase 3B requirement 4 warns
    against, just applied to loss instead of duplication."""
    by_filename: dict[str, SecExhibitCandidate] = {c.filename.lower(): c for c in from_index_json}
    for detailed in from_detail_html:
        key = detailed.filename.lower()
        existing = by_filename.get(key)
        exhibit_type = detailed.exhibit_type if detailed.exhibit_type != UNKNOWN else (existing.exhibit_type if existing else UNKNOWN)
        by_filename[key] = SecExhibitCandidate(
            filename=detailed.filename, exhibit_type=exhibit_type,
            sequence=detailed.sequence, description=detailed.description,
        )
    return list(by_filename.values())


class SecPrimaryDocumentAdapter:
    """LOCATE (submissions metadata -> cross-check against the directory
    listing -> resolve primaryDocument URL) -> FETCH (HTTP GET, validate,
    build+store a PRIMARY_DOCUMENT Document) -> PARSE (confirm the stored
    document is chunk-ready). Registered under adapter_id
    ``sec_primary_document_adapter``.
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
        if not _validate_accession(ref.accession):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"malformed accession number: {ref.accession!r}",
            )
        if not _validate_filename(ref.primary_document):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"unsafe primaryDocument filename: {ref.primary_document!r}",
            )

        api_requests_made = 0
        cache_hit_submissions = False
        submissions_payload, api_requests_made, cache_hit_submissions, failure = self._fetch_submissions(ref.cik, context)
        if failure is not None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=failure,
            )

        authoritative_document = _submissions_primary_document(submissions_payload, ref.accession)
        if authoritative_document is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=f"accession {ref.accession} not found in CIK {ref.cik}'s submissions listing",
            )
        if authoritative_document.lower() != ref.primary_document.lower():
            # Requirement 3: submissions metadata and the caller's expected
            # value disagree -- never silently prefer one; report the
            # mismatch as a failure rather than guessing which is right.
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                api_requests_made=api_requests_made, cache_hit=cache_hit_submissions,
                failure_reason=(
                    f"primaryDocument mismatch: expected {ref.primary_document!r}, "
                    f"submissions metadata records {authoritative_document!r}"
                ),
            )

        accession_nodash = ref.accession.replace("-", "")
        directory, http_requests_made, cache_hit_directory, dir_failure = self._fetch_directory(
            ref.cik, ref.accession, accession_nodash, context
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
            # Requirement 3: submissions says this filename exists, but the
            # accession's own directory listing disagrees -- never resolve a
            # URL for a file the directory does not actually contain.
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND,
                api_requests_made=api_requests_made, http_requests_made=http_requests_made,
                cache_hit=cache_hit_submissions or cache_hit_directory,
                failure_reason=(
                    f"primaryDocument {ref.primary_document!r} is not present in the accession "
                    "directory listing -- submissions metadata and the directory disagree"
                ),
            )

        url = _resolve_document_url(ref.cik, accession_nodash, ref.primary_document)
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
                "cik": ref.cik, "form": ref.form, "filing_date": ref.filing_date,
                "report_date": ref.report_date, "company_name": ref.company_name,
            },
        )

    def _fetch_submissions(
        self, cik: int, context: ExecutionContext
    ) -> tuple[dict[str, Any], int, bool, str | None]:
        cache_key = f"sec_submissions:{cik}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.URL_RESOLVED:
                return {}, 0, True, cached.failure_reason or "cached submissions fetch failed"
            return cached.payload.get("submissions", {}), 0, True, None

        fetch = self.http.get(SUBMISSIONS_URL.format(cik=cik))
        if not fetch.ok:
            failure = f"submissions fetch failed: {fetch.outcome} {fetch.error}"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_sec_submissions_{cik}", status=StepStatus.FAILED,
                api_requests_made=1, failure_reason=failure,
            )
            return {}, 1, False, failure
        payload = fetch.json() or {}
        context.request_cache[cache_key] = StepExecutionResult(
            step_id=f"_cache_sec_submissions_{cik}", status=StepStatus.URL_RESOLVED,
            payload={"submissions": payload}, api_requests_made=1,
        )
        return payload, 1, False, None

    def _fetch_directory(
        self, cik: int, accession: str, accession_nodash: str, context: ExecutionContext
    ) -> tuple[dict[str, Any], int, bool, str | None]:
        url = _directory_index_json_url(cik, accession_nodash)
        cache_key = f"http_get:{url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.URL_RESOLVED:
                return {}, 0, True, cached.failure_reason or "cached directory fetch failed"
            return cached.payload.get("directory", {}), 0, True, None

        fetch = self.http.get(url)
        if not fetch.ok:
            failure = f"directory index fetch failed for accession {accession}: {fetch.outcome} {fetch.error}"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_sec_directory_{accession}", status=StepStatus.FAILED,
                http_requests_made=1, failure_reason=failure,
            )
            return {}, 1, False, failure
        directory = (fetch.json() or {}).get("directory", {})
        context.request_cache[cache_key] = StepExecutionResult(
            step_id=f"_cache_sec_directory_{accession}", status=StepStatus.URL_RESOLVED,
            payload={"directory": directory}, http_requests_made=1,
        )
        return directory, 1, False, None

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
                    "index/access-denial page, or was shorter than the minimum body length"
                ),
            )
            context.request_cache[cache_key] = result
            return result

        # Content-dependent, not URL-only (Phase 3D.1): see
        # document_store.derive_document_id()'s docstring for why a
        # URL-only id risks a self-referencing version-chain cycle when
        # this URL is re-fetched with changed content.
        doc_id = derive_document_id("sec_primary", url, text)
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
            # Set only because ``_content_is_valid_body`` actually confirmed
            # extractable, non-rejected, non-index-page text -- never merely
            # because the HTTP status was 200 (Phase 3B requirement 5).
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
    """LOCATE (fetch the accession's directory ``index.json``, best-effort
    enrich with the filing detail HTML's Sequence+Description, classify and
    select) -> FETCH (HTTP GET the selected exhibit body) -> PARSE (confirm
    the stored EXHIBIT document is chunk-ready). Registered under adapter_id
    ``sec_exhibit_enumeration``.
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
        if not _validate_accession(selection.accession):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"malformed accession number: {selection.accession!r}",
            )
        accession_nodash = selection.accession.replace("-", "")

        index_url = _directory_index_json_url(selection.cik, accession_nodash)
        index_cache_key = f"http_get:{index_url}"
        cached_index = context.request_cache.get(index_cache_key)
        http_requests_made = 0
        cache_hit = False
        if cached_index is not None:
            cache_hit = True
            if cached_index.status is not StepStatus.URL_RESOLVED:
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.FAILED, cache_hit=True,
                    failure_reason=cached_index.failure_reason or "cached directory fetch failed",
                )
            directory_payload = cached_index.payload.get("directory", {})
        else:
            fetch = self.http.get(index_url)
            http_requests_made += 1
            if not fetch.ok:
                failure = f"exhibit index fetch failed: {fetch.outcome} {fetch.error}"
                context.request_cache[index_cache_key] = StepExecutionResult(
                    step_id=f"_cache_sec_directory_{selection.accession}", status=StepStatus.FAILED,
                    http_requests_made=1, failure_reason=failure,
                )
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                    failure_reason=failure,
                )
            directory_payload = (fetch.json() or {}).get("directory", {})
            # Same cache key AND payload shape ``SecPrimaryDocumentAdapter``
            # uses for this exact URL -- the directory index.json fetch is
            # shared across both adapters for the same accession, never
            # re-fetched (Phase 3A requirement 10).
            context.request_cache[index_cache_key] = StepExecutionResult(
                step_id=f"_cache_sec_directory_{selection.accession}", status=StepStatus.URL_RESOLVED,
                payload={"directory": directory_payload}, http_requests_made=1,
            )

        from_index_json = [
            SecExhibitCandidate(
                # Deliberately NOT substituting UNKNOWN for a blank "type" --
                # index.json genuinely leaves it blank for auto-generated
                # XBRL-viewer report pages (R1.htm, ...), and
                # ``classify_exhibit_entry`` treats "truly blank" as a
                # meaningful signal (Phase 3B requirement 1's documented
                # limitation), distinct from a caller-facing UNKNOWN label.
                filename=item.get("name", ""), exhibit_type=item.get("type", ""),
            )
            for item in (directory_payload.get("item") or [])
            if item.get("name") and not item["name"].endswith("-index.htm")
        ]

        detail_html_unavailable = False
        detail_url = _filing_detail_html_url(selection.cik, selection.accession, accession_nodash)
        detail_cache_key = f"http_get:{detail_url}"
        cached_detail = context.request_cache.get(detail_cache_key)
        if cached_detail is not None and cached_detail.status is StepStatus.URL_RESOLVED:
            cache_hit = cache_hit and True
            from_detail_html = cached_detail.payload.get("candidates", [])
        elif cached_detail is not None:
            detail_html_unavailable = True
            from_detail_html = []
        else:
            detail_fetch = self.http.get(detail_url)
            http_requests_made += 1
            if detail_fetch.ok:
                from_detail_html = _parse_filing_detail_table(detail_fetch.text)
                context.request_cache[detail_cache_key] = StepExecutionResult(
                    step_id=f"_cache_sec_detail_{selection.accession}", status=StepStatus.URL_RESOLVED,
                    payload={"candidates": from_detail_html}, http_requests_made=1,
                )
            else:
                detail_html_unavailable = True
                from_detail_html = []
                context.request_cache[detail_cache_key] = StepExecutionResult(
                    step_id=f"_cache_sec_detail_{selection.accession}", status=StepStatus.FAILED,
                    http_requests_made=1,
                    failure_reason=f"filing detail fetch failed: {detail_fetch.outcome} {detail_fetch.error}",
                )

        merged = _merge_candidates(from_index_json, from_detail_html)
        primary_document = selection.primary_document if selection.primary_document != UNKNOWN else ""
        classified = [
            SecExhibitCandidate(
                filename=c.filename, exhibit_type=c.exhibit_type, sequence=c.sequence,
                description=c.description,
                category=classify_exhibit_entry(
                    filename=c.filename, exhibit_type=c.exhibit_type,
                    description=c.description, primary_document=primary_document,
                ),
            )
            for c in merged
        ]
        if not classified:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND, http_requests_made=http_requests_made,
                cache_hit=cache_hit, failure_reason="accession directory listed no exhibits",
            )

        selected, deferred = _select_narrative_exhibit(classified, selection.priority_keywords, selection.fetch_limit)
        if not selected:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.NOT_FOUND, http_requests_made=http_requests_made,
                cache_hit=cache_hit,
                failure_reason="no narrative exhibit in the accession matched the selection priority keywords",
            )

        chosen = selected[0]
        url = _resolve_document_url(selection.cik, accession_nodash, chosen.filename)
        if url is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=http_requests_made,
                failure_reason="could not safely resolve the selected exhibit's URL",
            )
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.URL_RESOLVED,
            http_requests_made=http_requests_made, cache_hit=cache_hit,
            payload={
                "url": url, "accession": selection.accession, "filename": chosen.filename,
                "exhibit_type": chosen.exhibit_type, "cik": selection.cik,
                "category": chosen.category.value,
                # Requirement 6: exhibits beyond the fetch limit are recorded
                # as not-yet-fetched, never as nonexistent.
                "not_yet_fetched": [c.filename for c in deferred],
                "detail_html_unavailable": detail_html_unavailable,
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

        # Content-dependent, not URL-only -- see derive_document_id()'s
        # docstring (document_store.py, Phase 3D.1).
        doc_id = derive_document_id("sec_exhibit", url, text)
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
