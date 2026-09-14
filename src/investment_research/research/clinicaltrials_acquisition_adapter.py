"""ClinicalTrials.gov Direct Adapter (Phase 3D).

Fetches ONE study record from the real ClinicalTrials.gov API v2
single-study endpoint (``GET /api/v2/studies/{nctId}``) when the NCT ID is
already known -- never by sponsor-name search, never via Web Search. This
is deliberately narrower than ``collectors/clinicaltrials.py``'s existing
``ClinicalTrialsCollector`` (which searches by sponsor name and never has a
Document): here the caller already knows exactly which study it wants, so
there is nothing to search or locate beyond formatting a deterministic URL.

Registered with an ``AcquisitionExecutor`` by the caller -- like the SEC
adapters, this is NOT called from the production pipeline in this phase
(see CLAUDE.md's Phase 3A/3B/3C/3D forbidden-changes lists), and every test
in this repository drives it against a fake HTTP double loaded from
``tests/fixtures/clinicaltrials_v2_real_format/``. ``ImplementationStatus``
for the steps this adapter backs therefore tops out at OFFLINE_VERIFIED,
never LIVE_VERIFIED: nothing here has ever been checked against the real
ClinicalTrials.gov site.

Evidence Integrity (Phase 3D requirement 4): ClinicalTrials.gov is a
REGISTRY (``DocumentAuthority.REGISTRY``), sponsor-authored. What a
successful fetch DIRECTLY confirms is limited to what the registry record
itself asserts -- registered status, phase, enrollment, dates, design,
endpoint wording, sponsor, and whether a results section was ever
submitted. It confirms NONE of: efficacy, that an endpoint was accepted by
a regulator, approval likelihood, peer-reviewed validity, mechanism
plausibility, reproducibility, or that a company's own characterization of
the trial is accurate. In particular:

* ``hasResults``/``resultsSection`` PRESENCE is recorded; its CONTENT is
  never parsed, and its ABSENCE is never read as "the trial failed" -- many
  genuinely ongoing or recently-completed trials have not posted results
  yet, for entirely mundane reasons (the 12-month reporting window, sponsor
  administrative delay).
* ``overallStatus == "COMPLETED"`` is a registry status, never a synonym
  for "succeeded" -- a completed trial can have failed its primary
  endpoint, and this adapter has no opinion on which happened.
* ``overallStatus in ("TERMINATED", "WITHDRAWN")`` is likewise never
  auto-propagated as a Kill finding for the company's CURRENT lead
  programme -- ``scoring/program_resolution.py`` already exists precisely
  to stop a terminated trial in a DIFFERENT programme from driving a
  company-level disqualification, and this adapter changes nothing about
  that: it only fetches and stores one registry record, it does not decide
  what that record means for any thesis.

Nothing here ever generates an investment Action, promotes a domain to
SUFFICIENT, or interprets registered data as a judgment.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from ..collectors.clinicaltrials import STUDY_DETAIL_URL, parse_study
from ..collectors.documents import Document
from ..schemas.enums import UNKNOWN, ContentKind, DocumentAuthority, FetchOutcome, Provenance
from ..schemas.fact import utc_now_iso
from .acquisition_executor import ExecutionContext, StepExecutionResult
from .document_store import DocumentRole, derive_document_id
from .source_routing import AcquisitionStep, StepKind, StepStatus

log = logging.getLogger(__name__)

#: adapter_id this adapter is meant to be registered under for LOCATE/FETCH
#: -- matches source_routing_catalog.py's ``_clinicaltrials_web`` archetype
#: (``direct_adapter="clinicaltrials_api"``) exactly. That archetype's
#: structured-path PARSE step uses the generic ``"local_parser"`` id
#: instead (shared with every other structured-or-web archetype); this
#: adapter answers to either id, dispatching purely on ``step.step_kind``.
CLINICALTRIALS_ADAPTER_ID = "clinicaltrials_api"

NCT_ID_RE = re.compile(r"^NCT\d{8}$")

#: A structured JSON record has no HTML boilerplate to strip, so this bar
#: is far lower than SEC's HTML MIN_BODY_CHARS -- it only exists to refuse
#: a suspiciously empty/near-empty body, never to demand prose length from
#: a terse but complete record.
MIN_BODY_CHARS = 40


def _valid_nct_id(nct_id: str) -> bool:
    return bool(NCT_ID_RE.match(nct_id or ""))


@dataclass(frozen=True)
class ClinicalTrialsStudyReference:
    """The one concrete input this adapter needs per target: an EXPLICIT,
    already-known NCT ID, supplied by the caller before ``AcquisitionExecutor
    .run()`` -- never resolved by this module itself. A caller that only has
    a company name and no NCT ID cannot use this adapter at all; that is
    exactly the intended boundary (Phase 3D requirement 3: no guessing a
    programme from a company name here -- see ``scoring/program_resolution.py``
    for how a specific NCT ID is supposed to be chosen from evidence)."""

    nct_id: str


def _upstream_payload(step: AcquisitionStep, context: ExecutionContext) -> dict[str, Any]:
    for dep_id in step.depends_on_step_ids:
        payload = context.payload_for(dep_id)
        if payload:
            return dict(payload)
    return {}


def _resolve_url_and_nct_id(step: AcquisitionStep, context: ExecutionContext) -> tuple[str | None, str | None]:
    """The URL/NCT ID to FETCH -- from an upstream LOCATE step's payload if
    this graph has one (a bespoke LOCATE+FETCH+PARSE graph, e.g. the Live
    Smoke entry point's own graph), or built fresh from the caller-supplied
    ``ClinicalTrialsStudyReference`` otherwise. The production catalog's
    structured-fetch archetype (``_clinicaltrials_web`` in
    source_routing_catalog.py) has NO separate LOCATE step at all: the URL
    is fully deterministic from an already-known NCT ID, so there is
    nothing a LOCATE step would add for that path."""
    upstream = _upstream_payload(step, context)
    if upstream.get("url") and upstream.get("nct_id"):
        return upstream["url"], upstream["nct_id"]
    ref = context.study_reference_for(step.target_id)
    if ref is None:
        return None, None
    return STUDY_DETAIL_URL.format(nct_id=ref.nct_id), ref.nct_id


class ClinicalTrialsStudyAdapter:
    """LOCATE (validate the caller-supplied NCT ID, build the deterministic
    single-study URL -- ZERO network calls, and NEVER Web Search) -> FETCH
    (HTTP GET, validate the response is well-formed JSON matching the
    expected CT.gov API v2 study shape, store a ``STRUCTURED_API_RECORD``
    Document) -> PARSE (extract registered fields via ``parse_study`` and
    confirm the minimum required fields are actually present).
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
            failure_reason=f"ClinicalTrialsStudyAdapter cannot handle step_kind={step.step_kind}",
        )

    def _locate(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        ref = context.study_reference_for(step.target_id)
        if ref is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no ClinicalTrialsStudyReference supplied for this target",
            )
        if not _valid_nct_id(ref.nct_id):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"malformed NCT ID: {ref.nct_id!r}",
            )
        url = STUDY_DETAIL_URL.format(nct_id=ref.nct_id)
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.URL_RESOLVED,
            payload={"url": url, "nct_id": ref.nct_id},
        )

    def _fetch(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        url, nct_id = _resolve_url_and_nct_id(step, context)
        if url is None or nct_id is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=(
                    "no NCT ID/URL available -- no ClinicalTrialsStudyReference supplied and "
                    "no upstream LOCATE payload"
                ),
            )
        if not _valid_nct_id(nct_id):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason=f"malformed NCT ID: {nct_id!r}",
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
            # A 404 means "no such study" -- a definitive negative, never
            # ACQUIRED. Everything else (429/5xx/timeout/blocked) is a
            # transient or policy-level transport failure, reported as
            # FAILED with the specific outcome preserved in the message so
            # the two remain diagnosable apart without inventing a new
            # StepStatus value per HTTP status (Phase 3D requirement 6).
            status = StepStatus.NOT_FOUND if fetch.outcome is FetchOutcome.NOT_FOUND else StepStatus.FAILED
            result = StepExecutionResult(
                step_id=step.step_id, status=status, http_requests_made=1,
                failure_reason=f"GET {url} failed: {fetch.outcome} {fetch.error}",
            )
            context.request_cache[cache_key] = result
            return result

        payload = fetch.json()
        if payload is None:
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=f"response body was not valid JSON: {fetch.error}",
            )
            context.request_cache[cache_key] = result
            return result
        if not isinstance(payload, dict) or "protocolSection" not in payload:
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason="response did not match the expected CT.gov API v2 study shape (no protocolSection)",
            )
            context.request_cache[cache_key] = result
            return result

        text = json.dumps(payload, sort_keys=True)
        if len(text.strip()) < MIN_BODY_CHARS:
            result = StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=1,
                failure_reason="response body too short to be a real study record",
            )
            context.request_cache[cache_key] = result
            return result

        ident = (payload.get("protocolSection") or {}).get("identificationModule") or {}
        # Content-dependent, not URL-only -- see document_store.
        # derive_document_id()'s docstring (Phase 3D.1) for why a URL-only
        # id risks a self-referencing version-chain cycle when this NCT
        # ID's record is re-fetched with changed content.
        doc_id = derive_document_id("clinicaltrials", url, text)
        document = Document(
            doc_id=doc_id,
            url=url,
            title=ident.get("briefTitle", UNKNOWN) or nct_id,
            publisher="ClinicalTrials.gov",
            doc_type="clinical_trial_registry_record",
            # ClinicalTrials.gov is a GOVERNMENT-OPERATED REGISTRY, never a
            # company IR channel -- is_company_ir specifically means "the
            # issuing company controls this host/channel" (see
            # collectors/tiering.py's classify_authority/classify_tier,
            # which is exactly the distinction DocumentAuthority.REGISTRY
            # vs. DocumentAuthority.COMPANY_IR exists to keep separate).
            # That the registry's CONTENT is sponsor-submitted is a
            # SEPARATE fact, already captured correctly at the RawFact
            # level (company_claim=True, in raw_facts_from_study()) --
            # conflating "who submitted the claim" with "who controls the
            # channel" would misclassify a NIH-operated registry page as
            # company-controlled content, which it is not (Phase 3D.1
            # requirement 1).
            is_company_ir=False,
            text=text,
            # A complete, single-study JSON record was retrieved whole --
            # never merely because HTTP returned 200 (the shape/length
            # checks above already ran).
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            # Retrieval time is retrieval time only -- never assigned into
            # any of the study's own registered dates (Phase 3D requirement 5).
            retrieved_at=utc_now_iso(),
            authority=DocumentAuthority.REGISTRY,
        )
        stored = context.document_store.put(
            document,
            document_id=doc_id,
            # NCT ID stands in for "accession" here -- CT.gov has no
            # accession concept, but the NCT ID plays the identical role:
            # a stable, unique identifier for exactly one registry record.
            accession=nct_id,
            filename=nct_id,
            document_role=DocumentRole.STRUCTURED_API_RECORD,
            canonical_url=url,
        )
        result_payload = {"url": url, "nct_id": nct_id, "document_id": stored.document_id, "study": payload}
        result = StepExecutionResult(
            step_id=step.step_id, status=StepStatus.STRUCTURED_RECORD_RETRIEVED,
            document_id=stored.document_id, http_requests_made=1, payload=result_payload,
        )
        context.request_cache[cache_key] = result
        return result

    def _parse(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        study = upstream.get("study")
        document_id = upstream.get("document_id")
        if not document_id or study is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no fetched study record available to parse",
            )
        parsed = parse_study(study)
        if not parsed.get("nct_id") or not _valid_nct_id(parsed["nct_id"]):
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, document_id=document_id,
                failure_reason="parsed record has no valid NCT ID",
            )
        if parsed.get("status", UNKNOWN) == UNKNOWN:
            # The minimum bar for REQUIRED_FIELDS_PARSED: at least the
            # registered overall status must be present. Never invented
            # when CT.gov itself omitted it -- this step is honest about
            # how thin that leaves the parse, rather than declaring success
            # over a record missing its single most basic field.
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, document_id=document_id,
                payload={**upstream, "parsed": parsed},
                failure_reason="overall status is UNKNOWN in the registry record -- required fields not present",
            )
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.REQUIRED_FIELDS_PARSED,
            document_id=document_id, payload={**upstream, "parsed": parsed},
        )
