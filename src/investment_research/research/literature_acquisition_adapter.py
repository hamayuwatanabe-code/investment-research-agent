"""PubMed (NCBI E-utilities) / Europe PMC Direct Adapters (Phase 3F).

A peer-reviewed-literature acquisition path ClinicalTrials.gov alone cannot
provide: a ClinicalTrials.gov record is the sponsor's own structured trial
registration; it never contains a published paper's actual reported result,
methodology critique, or independent replication. This phase goes only as
far as OFFLINE_VERIFIED -- no Live communication is ever made (see
``ALLOWED_HOSTS``/the module's HTTP-safety section below for what a FUTURE
Live Smoke phase would need to respect), and this module is never called
from ``Pipeline.run()`` (CLAUDE.md's standing prohibition, unchanged).

Official specs modeled: NCBI E-utilities
(https://www.ncbi.nlm.nih.gov/books/NBK25497/) and the Europe PMC RESTful
Web Service (https://europepmc.org/RestfulWebService).

Acquisition priority order (Phase 3F requirement 3), tried in this exact
sequence per ``LiteratureReference``:

    A. ``ct_pmid`` -- a PMID already present on a ClinicalTrials.gov study
       record's own ``references`` (``collectors/clinicaltrials.py``'s
       ``{"pmid": ..., "citation": ...}`` shape) -- EFetch it directly, no
       ESearch needed.
    B. ``nct_id`` -- ESearch with an EXACT NCT-ID match
       (``term=<NCT_ID>[si]``, NCBI's own Secondary Source ID field tag).
    C. ``alias``/``condition`` -- a free-text discovery search. Results are
       marked ``confirmed_same_trial=False`` in every payload they produce
       and are NEVER promoted to "the same trial/program" by anything in
       this module -- a caller wanting that confirmation must establish it
       independently, elsewhere.

Registered with an ``AcquisitionExecutor`` by the caller under adapter_id
``"pubmed_europepmc"`` -- like every other adapter in this repository, NOT
called from the production pipeline. Every test drives it against a fake
HTTP double loaded from ``tests/fixtures/literature_real_format/``, so
``ImplementationStatus`` for the steps this adapter backs tops out at
OFFLINE_VERIFIED, never LIVE_VERIFIED (mirrors
``research/form4_acquisition_adapter.py``'s own precedent exactly).

Structure (Phase 3F requirement 3): ``LiteratureReference``,
``LiteratureSearchQuery``, ``PubMedLiteratureAdapter`` (registered adapter;
LOCATE=ESearch, FETCH=batched EFetch, PARSE=structured field extraction --
never a RawFact, mirroring ``Form4Adapter``'s own precedent exactly: RawFact
generation is a separate, independently-tested pure function,
``collectors/literature.raw_facts_from_pubmed_article``/
``raw_facts_from_europepmc_fulltext``, never wired into this adapter),
``EuropePmcFullTextAdapter`` (composed by ``PubMedLiteratureAdapter``,
exactly as ``Form4Adapter`` composes ``SecPrimaryDocumentAdapter`` -- never
registered under its own adapter_id, since the source-routing catalog's
``literature_web`` target has exactly one direct adapter slot).

Evidence boundary (Phase 3F requirement 6 -- CRITICAL, see
``collectors/literature.py``'s own module docstring for the full
reasoning): nothing here ever sets ``independent_confirmation=True``,
promotes a fact to ``VERIFIED_FACT``, or claims efficacy/safety/endpoint
acceptance was confirmed. A paper existing is never "the FDA accepted this
endpoint"; a paper reporting a result is never "the result is true". Every
RawFact ``collectors/literature.py``'s builder functions produce from this
module's PARSE output carries ``EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION``
once ``agents/evidence_integrity.py`` classifies it (via the ``unit`` prefix
``collectors/literature.py`` applies), which ``schemas/enums.py``'s
``DECISION_GRADE_CLASSES`` excludes.

HTTP & safety (Phase 3F requirement 4, hardened in Phase 3F.0.2):
``ALLOWED_HOSTS`` is the complete set this module will ever build a request
URL against -- ``eutils.ncbi.nlm.nih.gov``, ``www.ebi.ac.uk``,
``europepmc.org`` -- checked defensively by ``_require_allowed_host`` even
though no live call is ever issued this phase. ``MAX_PMIDS_PER_BATCH``/
``MAX_REQUESTS_PER_RUN`` bound how much even a future Live Smoke run could
ever do; NCBI's own documented limit without an API key is 3 requests/
second -- this module's own ``DEFAULT_RATE_LIMIT_RPS`` is deliberately more
conservative. Europe PMC full text is fetched ONLY when a search result's
own ``isOpenAccess``/``inEPMC`` flags are true; there is no paywall-bypass
or PDF-scraping code path anywhere in this module, by construction (the
only full-text source this module reads is Europe PMC's own
``fullTextXML`` REST endpoint for a confirmed-OA PMCID).
``IRA_NCBI_TOOL``/``IRA_NCBI_EMAIL``/``IRA_NCBI_API_KEY`` (``config.
Settings``) are future-use only this phase -- this module never configures
or uses a real value for either.

wire_url / public_url (Phase 3F.0.2 requirement 1): a "wire_url" is a URL
carrying email/api_key when ``Settings`` supplies them, and it is used for
EXACTLY ONE THING -- the single ``http.get(wire_url)`` call each request
site makes, always via ``_safe_get`` (the only function in this module
that touches a wire_url). Every other consumer -- cache keys, failure/
exception text, ``StepExecutionResult.payload``, ``Document.url``, and (in
a future Live Smoke phase) a Capture Manifest's ``requested_url``/
``final_url`` -- sees only the "public_url" ``_public_request_url``
derives from it (query-parameter stripping) immediately after
construction, PLUS a second, value-based redaction pass
(``_sanitize_text``/``_secret_values``) that catches a secret leaking into
free text that isn't shaped like a URL at all, in both its raw and
percent-encoded form. ``_safe_get`` also catches and sanitizes any
exception the transport itself raises, so a raw wire_url embedded in a
``urllib`` exception message can never propagate outward unsanitized. It
is fine -- necessary -- for email/api_key to reach the real transport as
part of a wire_url; it is never fine for a wire_url to be RETAINED
anywhere else in this module's own diagnostics.

Request cap (Phase 3F.0.1 requirement 4, made run-scoped in Phase 3F.0.2):
``RequestBudgetLimits`` (immutable caps, held on the adapter instance) and
``RequestBudgetUsage`` (mutable per-run counters, held in
``ExecutionContext.adapter_state`` -- see ``_budget_usage_for``) bound
PubMed + Europe PMC combined, shared across every literature target WITHIN
one ``AcquisitionExecutor.run()`` call but reset to zero for a NEW
``run()`` call even when the same adapter instance is reused --
``max_search_requests`` (ESearch + Europe PMC ``/search`` combined),
``max_articles`` (distinct NEW PMIDs actually processed; already-cached
PMIDs are free and never count against it), ``max_fulltext_fetches``
(Europe PMC ``fullTextXML`` GETs), and ``max_total_requests`` (an overall
ceiling across all of the above). A candidate PMID list larger than the
budget allows is never silently fetched in full, and excess is never
reported as NOT_FOUND: excluded PMIDs are surfaced explicitly in
FETCH/PARSE's own payload (``budget_excluded_pmids``/
``budget_skipped_europepmc_search_pmids``/
``budget_skipped_fulltext_pmcids``) and ``coverage_complete=False`` is set
on the PARSE result whenever any budget skip occurred for this target --
``True`` only when every candidate was genuinely processed.
Not thread-safe: see ``RequestBudgetUsage``'s own docstring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from ..collectors.documents import Document
from ..collectors.literature import (
    EuropePmcSearchResult,
    ParsedPubmedArticle,
    check_pubmed_xml_shape,
    parse_europepmc_fulltext_xml,
    parse_europepmc_search_response,
    parse_pubmed_articleset,
)
from ..schemas.enums import UNKNOWN, ContentKind, DocumentAuthority, Provenance
from ..schemas.fact import utc_now_iso
from .acquisition_executor import ExecutionContext, StepExecutionResult
from .document_store import DocumentRole, derive_document_id
from .source_routing import AcquisitionStep, StepKind, StepStatus

log = logging.getLogger(__name__)

#: Matches source_routing_catalog.py's own ``_LITERATURE_ADAPTER_ID``
#: string exactly.
LITERATURE_ADAPTER_ID = "pubmed_europepmc"

#: The complete set of hosts this module will ever build a request against
#: (Phase 3F requirement 4). Any URL construction that would leave this set
#: is a programming error, never a runtime fallback.
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {"eutils.ncbi.nlm.nih.gov", "www.ebi.ac.uk", "europepmc.org"}
)

NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{pmcid}/fullTextXML"

#: NCBI's own documented cap without an API key is 3 req/s -- this module's
#: own default is deliberately more conservative (Phase 3F requirement 4).
DEFAULT_RATE_LIMIT_RPS = 2.0
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RETRIES = 2
#: Never fetch a large PMID set one at a time -- but a single EFetch call
#: is still bounded, never unlimited.
MAX_PMIDS_PER_BATCH = 20
#: Hard bound on how many real HTTP requests one adapter run may ever issue
#: -- a budget/limit exclusion (mirrors Form4's own precedent), never a
#: silent unlimited fetch loop.
MAX_REQUESTS_PER_RUN = 40

#: Phase 3F.0.1 requirement 4: PubMed + Europe PMC combined caps, enforced
#: by ``RequestBudgetLimits``/``RequestBudgetUsage`` across a whole
#: ``AcquisitionExecutor.run()`` -- never per-target (a caller that wants a
#: tighter/looser budget passes its own values to
#: ``PubMedLiteratureAdapter.__init__``).
DEFAULT_MAX_SEARCH_REQUESTS = 10
DEFAULT_MAX_ARTICLES = 20
DEFAULT_MAX_FULLTEXT_FETCHES = 10
DEFAULT_MAX_TOTAL_REQUESTS = MAX_REQUESTS_PER_RUN

#: Prefix marking a failure string as "budget exhausted", never "genuinely
#: not found" -- callers check this prefix to choose
#: ``StepStatus.SKIPPED_DUE_TO_BUDGET`` over ``StepStatus.FAILED``/
#: ``NOT_FOUND`` (Phase 3F.0.1 requirement 4).
BUDGET_EXCEEDED_PREFIX = "BUDGET_EXCEEDED: "

MIN_BODY_CHARS = 40


@dataclass(frozen=True)
class RequestBudgetLimits:
    """The CAPS only -- immutable configuration, safe to hold for the
    lifetime of a ``PubMedLiteratureAdapter`` instance (Phase 3F.0.2
    requirement 2). Never carries any usage counter; see
    ``RequestBudgetUsage`` for the run-scoped mutable half."""

    max_search_requests: int = DEFAULT_MAX_SEARCH_REQUESTS
    max_articles: int = DEFAULT_MAX_ARTICLES
    max_fulltext_fetches: int = DEFAULT_MAX_FULLTEXT_FETCHES
    max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS


@dataclass
class RequestBudgetUsage:
    """PubMed + Europe PMC combined request USAGE -- mutable, and deliberately
    never stored on ``self`` by either adapter class (Phase 3F.0.2
    requirement 2). Instead, one instance is created per
    ``AcquisitionExecutor.run()`` call and stashed in
    ``ExecutionContext.adapter_state`` (see ``_budget_usage_for`` below), so
    it is: shared across every literature target WITHIN one run (that
    dict is threaded, by the same object reference, through every
    per-target ``ExecutionContext`` the executor builds); and reset to zero
    for a brand new run, even when the exact same
    ``PubMedLiteratureAdapter``/``EuropePmcFullTextAdapter`` instances are
    reused across two separate ``.run()`` calls, and even when the first
    run ended in FAILED/an exception (the executor's own ``adapter_state``
    dict is a fresh local variable each call, never carried over).

    Not thread-safe: ``AcquisitionExecutor.run()`` itself processes targets
    sequentially in a single thread, and this class assumes the same --
    concurrent ``.execute()`` calls against the same
    ``ExecutionContext.adapter_state`` from multiple threads (e.g. two
    ``run()`` calls sharing one adapter instance concurrently, or a future
    parallelized executor) are NOT supported and would race on these plain
    ``int`` counters.
    """

    search_requests_made: int = 0
    fulltext_fetches_made: int = 0
    articles_processed: int = 0
    total_requests_made: int = 0

    def cap_articles(self, limits: RequestBudgetLimits, pmids: list[str]) -> tuple[list[str], list[str]]:
        """``(allowed, excluded)`` -- bounds how many NEW pmids may be
        processed this run (never fan-out into Europe PMC search/fullText
        without limit merely because a candidate list is large). Reserves
        the allowed slots immediately, so two calls in the same run never
        double-count."""
        remaining = max(0, limits.max_articles - self.articles_processed)
        allowed, excluded = pmids[:remaining], pmids[remaining:]
        self.articles_processed += len(allowed)
        return allowed, excluded

    def reserve_search(self, limits: RequestBudgetLimits) -> bool:
        """Call immediately before issuing an ESearch OR Europe PMC
        ``/search`` GET (they share this one budget) -- never after."""
        if self.search_requests_made >= limits.max_search_requests:
            return False
        if self.total_requests_made >= limits.max_total_requests:
            return False
        self.search_requests_made += 1
        self.total_requests_made += 1
        return True

    def reserve_fulltext(self, limits: RequestBudgetLimits) -> bool:
        """Call immediately before issuing a Europe PMC ``fullTextXML``
        GET -- never after."""
        if self.fulltext_fetches_made >= limits.max_fulltext_fetches:
            return False
        if self.total_requests_made >= limits.max_total_requests:
            return False
        self.fulltext_fetches_made += 1
        self.total_requests_made += 1
        return True

    def reserve_other(self, limits: RequestBudgetLimits) -> bool:
        """Call immediately before issuing an EFetch GET -- EFetch has no
        dedicated per-kind cap (one call already covers a whole batch), but
        it still counts toward ``max_total_requests``."""
        if self.total_requests_made >= limits.max_total_requests:
            return False
        self.total_requests_made += 1
        return True


#: Private key this module uses to stash its own ``RequestBudgetUsage`` in
#: ``ExecutionContext.adapter_state`` -- private to this module by
#: convention (a leading underscore plus the module's own adapter_id), so
#: it can never collide with another adapter's own state.
_BUDGET_USAGE_STATE_KEY = "_pubmed_europepmc_request_budget_usage"


def _budget_usage_for(context: ExecutionContext) -> RequestBudgetUsage:
    """The current run's shared ``RequestBudgetUsage``, creating one (and
    only one, for the whole run) the first time any literature step in
    this run asks for it (Phase 3F.0.2 requirement 2)."""
    usage = context.adapter_state.get(_BUDGET_USAGE_STATE_KEY)
    if usage is None:
        usage = RequestBudgetUsage()
        context.adapter_state[_BUDGET_USAGE_STATE_KEY] = usage
    return usage


def _require_allowed_host(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing to build a request against a non-allow-listed host: {host!r}")


#: wire_url / public_url boundary (Phase 3F.0.2 requirement 1). A "wire_url"
#: is a URL actually handed to ``http.get()`` -- the ONLY place it may ever
#: be used. A "public_url" (built from a wire_url by ``_public_request_url``
#: below, immediately after construction) is what every other consumer
#: sees: cache keys, failure/exception text, ``StepExecutionResult.payload``,
#: ``Document.url``, and (in a future Live Smoke phase) a Capture Manifest's
#: ``requested_url``/``final_url``. It is fine -- necessary -- for
#: email/api_key to reach the real transport as part of a wire_url; it is
#: never fine for a wire_url itself to be retained anywhere else.
#: ``_secret_values``/``_sanitize_text`` below add a second, independent
#: layer that redacts by VALUE (raw and percent-encoded) rather than by
#: query-parameter name, so a secret that leaked into free text -- an
#: exception message, a future retry diagnostic -- that isn't shaped like a
#: URL at all is still caught.
def _public_request_url(url: str) -> str:
    """Strips any ``email``/``api_key`` query parameter -- never ``tool``,
    which carries no personal information and is not treated as a secret
    (see ``config.Settings.secret_values``, which likewise excludes it) --
    before this URL is ever placed in a diagnostic, log line, cache key, or
    Capture-Manifest-shaped payload."""
    if "email=" not in url and "api_key=" not in url:
        return url
    prefix, _, query = url.partition("?")
    kept = [
        pair
        for pair in query.split("&")
        if pair and not pair.startswith("email=") and not pair.startswith("api_key=")
    ]
    return prefix + ("?" + "&".join(kept) if kept else "")


def _secret_values(settings: Any) -> tuple[str, ...]:
    """The real secret VALUES (never ``tool``) this module might send this
    run, for the value-based redaction pass below."""
    if settings is None:
        return ()
    values = [
        getattr(settings, attr, "") for attr in ("ncbi_email", "ncbi_api_key")
    ]
    return tuple(v for v in values if v)


def _sanitize_text(text: str, *, settings: Any = None, wire_url: str = "", public_url: str = "") -> str:
    """Belt-and-braces text sanitizer (Phase 3F.0.2 requirement 1): first
    replaces a known wire_url substring with its already-computed
    public_url (covers a transport exception that embeds the full request
    URL verbatim), then replaces every known secret VALUE -- both raw and
    percent-encoded, so this catches a secret whether or not it was
    URL-encoded at the point of leakage -- with a fixed redaction marker.
    Applied to every string this module returns, raises, caches, or logs
    that did not already go through ``_public_request_url`` on a
    freshly-built URL.
    """
    result = text
    if wire_url and public_url and wire_url != public_url:
        result = result.replace(wire_url, public_url)
    for secret in _secret_values(settings):
        result = result.replace(secret, "[REDACTED]")
        result = result.replace(quote(secret, safe=""), "[REDACTED]")
    return result


def _eutils_params(base: dict[str, str], settings: Any) -> dict[str, str]:
    """NCBI's own courtesy-identification parameters -- added only when a
    caller's ``Settings`` actually supplies them (this phase never
    configures a real value; see module docstring)."""
    params = dict(base)
    tool = getattr(settings, "ncbi_tool", "") if settings is not None else ""
    email = getattr(settings, "ncbi_email", "") if settings is not None else ""
    api_key = getattr(settings, "ncbi_api_key", "") if settings is not None else ""
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def _safe_get(http: Any, wire_url: str, public_url: str, settings: Any) -> tuple[Any, str | None]:
    """The ONLY call site in this module that may ever pass a wire_url to
    the transport. Calls ``http.get(wire_url)``, catching and sanitizing
    ANY exception the transport raises before it could propagate a raw
    wire_url or secret value outward (Phase 3F.0.2 requirement 1: "even if
    an exception from urllib etc. contains a raw URL, sanitize it before
    returning it outward"). Returns ``(result, None)`` on a call that
    completed (including a transport-reported non-OK result, which the
    caller inspects via ``result.ok`` as before) or
    ``(None, sanitized_failure_reason)`` if the transport itself raised.
    """
    try:
        return http.get(wire_url), None
    except Exception as exc:  # noqa: BLE001 - must never leak wire_url/secrets outward
        sanitized = _sanitize_text(str(exc), settings=settings, wire_url=wire_url, public_url=public_url)
        return None, f"transport raised {type(exc).__name__}: {sanitized}"


@dataclass(frozen=True)
class LiteratureReference:
    """The concrete acquisition input a caller supplies for one literature
    target -- Priority A/B/C, tried in that exact order (module docstring).
    Never a pre-specified PMID list to blindly trust as "the answer"; even
    Priority A's ``ct_pmid`` still goes through the full FETCH/PARSE
    validation (shape check, PMID cross-check) below."""

    ct_pmid: str | None = None
    nct_id: str | None = None
    alias: str | None = None
    condition: str | None = None
    max_pmids: int = MAX_PMIDS_PER_BATCH


@dataclass(frozen=True)
class LiteratureSearchQuery:
    """One ESearch/Europe-PMC-search query actually issued -- kept for
    diagnostics/audit, mirroring ``research/escalation.py``'s own
    ``EscalationAttempt.queries`` precedent. ``term`` is the literal query
    string; secrets are never part of it (courtesy params are added only at
    request-build time, never stored here)."""

    term: str
    priority: str  # "A" | "B" | "C"
    confirmed_same_trial: bool  # Priority C is ALWAYS False -- see module docstring


def _status_for_failure(failure: str) -> StepStatus:
    """Phase 3F.0.1 requirement 4: a budget exhaustion is never reported as
    a generic FAILED (which reads as "we tried and it broke") or as
    NOT_FOUND/ZERO_RESULTS (which reads as "we looked and there is
    nothing") -- it is its own, honest SKIPPED_DUE_TO_BUDGET."""
    if failure.startswith(BUDGET_EXCEEDED_PREFIX):
        return StepStatus.SKIPPED_DUE_TO_BUDGET
    return StepStatus.FAILED


def _upstream_payload(step: AcquisitionStep, context: ExecutionContext) -> dict[str, Any]:
    for dep_id in step.depends_on_step_ids:
        payload = context.payload_for(dep_id)
        if payload:
            return dict(payload)
    return {}


class EuropePmcFullTextAdapter:
    """Fetches Europe PMC's own OA full text -- and ONLY that. No paywall
    bypass, no PDF scraping: the only body this class ever reads is the
    ``fullTextXML`` REST response for a PMCID whose OWN search-result entry
    already reports ``isOpenAccess=Y``/``inEPMC=Y`` (Phase 3F requirement
    3/6). Composed by ``PubMedLiteratureAdapter`` -- never registered under
    its own adapter_id (see module docstring)."""

    def __init__(self, http: Any, settings: Any = None, *, limits: RequestBudgetLimits | None = None) -> None:
        self.http = http
        self.settings = settings
        #: Immutable caps only -- safe to share with the composing
        #: ``PubMedLiteratureAdapter`` across the instance's whole lifetime
        #: (Phase 3F.0.2 requirement 2). USAGE lives in
        #: ``ExecutionContext.adapter_state`` instead (``_budget_usage_for``),
        #: never on ``self``.
        self.limits = limits if limits is not None else RequestBudgetLimits()

    def search(self, pmid: str, context: ExecutionContext) -> tuple[EuropePmcSearchResult | None, int, str | None]:
        """Returns ``(result, http_requests_made, failure_reason)``. Never
        raises -- a malformed/failed search leaves OA status simply unknown
        (never guessed ``True``). ``failure_reason`` is prefixed with
        ``BUDGET_EXCEEDED_PREFIX`` when the search budget, not a real
        transport failure, is why this did not run."""
        query = urlencode({"query": f"ext_id:{pmid} AND src:med", "format": "json"})
        wire_url = f"{EUROPEPMC_SEARCH_URL}?{query}"
        _require_allowed_host(wire_url)
        public_url = _public_request_url(wire_url)  # Europe PMC never carries email/api_key; kept for consistency
        cache_key = f"http_get:{public_url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.URL_RESOLVED:
                return None, 0, cached.failure_reason or "cached Europe PMC search failed"
            results = parse_europepmc_search_response(cached.payload.get("body", ""))
            return (results[0] if results else None), 0, None

        usage = _budget_usage_for(context)
        if not usage.reserve_search(self.limits):
            return None, 0, f"{BUDGET_EXCEEDED_PREFIX}Europe PMC search budget exhausted for PMID {pmid}"

        fetch, transport_error = _safe_get(self.http, wire_url, public_url, self.settings)
        if transport_error is not None:
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_epmc_search_{pmid}", status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=transport_error,
            )
            return None, 1, transport_error
        if not fetch.ok:
            failure = _sanitize_text(
                f"Europe PMC search failed for PMID {pmid}: {fetch.outcome} {fetch.error}",
                settings=self.settings, wire_url=wire_url, public_url=public_url,
            )
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_epmc_search_{pmid}", status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=failure,
            )
            return None, 1, failure

        results = parse_europepmc_search_response(fetch.text)
        context.request_cache[cache_key] = StepExecutionResult(
            step_id=f"_cache_epmc_search_{pmid}", status=StepStatus.URL_RESOLVED,
            payload={"body": fetch.text}, http_requests_made=1,
        )
        if results is None:
            return None, 1, f"malformed Europe PMC search response for PMID {pmid}"
        return (results[0] if results else None), 1, None

    def fetch_fulltext(
        self, pmcid: str, context: ExecutionContext
    ) -> tuple[Document | None, int, str | None]:
        """Only ever called by the caller after confirming OA/inEPMC --
        this method itself does not re-check that (single responsibility;
        the caller in ``PubMedLiteratureAdapter._fetch`` is the one place
        that decision is made, so it cannot be bypassed by a second code
        path)."""
        wire_url = EUROPEPMC_FULLTEXT_URL.format(source="PMC", pmcid=pmcid)
        _require_allowed_host(wire_url)
        public_url = _public_request_url(wire_url)
        cache_key = f"http_get:{public_url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.BODY_FETCHED:
                return None, 0, cached.failure_reason or "cached Europe PMC full-text fetch failed"
            text = cached.payload.get("text", "")
        else:
            usage = _budget_usage_for(context)
            if not usage.reserve_fulltext(self.limits):
                return None, 0, (
                    f"{BUDGET_EXCEEDED_PREFIX}Europe PMC full-text fetch budget exhausted for PMCID {pmcid}"
                )
            fetch, transport_error = _safe_get(self.http, wire_url, public_url, self.settings)
            if transport_error is not None:
                context.request_cache[cache_key] = StepExecutionResult(
                    step_id=f"_cache_epmc_fulltext_{pmcid}", status=StepStatus.FAILED,
                    http_requests_made=1, failure_reason=transport_error,
                )
                return None, 1, transport_error
            if not fetch.ok:
                failure = _sanitize_text(
                    f"Europe PMC full-text fetch failed for {pmcid}: {fetch.outcome} {fetch.error}",
                    settings=self.settings, wire_url=wire_url, public_url=public_url,
                )
                context.request_cache[cache_key] = StepExecutionResult(
                    step_id=f"_cache_epmc_fulltext_{pmcid}", status=StepStatus.FAILED,
                    http_requests_made=1, failure_reason=failure,
                )
                return None, 1, failure
            text = fetch.text
            if len(text.strip()) < MIN_BODY_CHARS:
                failure = "Europe PMC full-text response body too short to be real full text"
                context.request_cache[cache_key] = StepExecutionResult(
                    step_id=f"_cache_epmc_fulltext_{pmcid}", status=StepStatus.FAILED,
                    http_requests_made=1, failure_reason=failure,
                )
                return None, 1, failure
            context.request_cache[cache_key] = StepExecutionResult(
                step_id=f"_cache_epmc_fulltext_{pmcid}", status=StepStatus.BODY_FETCHED,
                payload={"text": text}, http_requests_made=1,
            )

        parsed = parse_europepmc_fulltext_xml(text)
        if parsed is None or not parsed.sections:
            return None, (0 if cached is not None else 1), (
                "Europe PMC full-text body did not match the expected JATS-like <article> shape"
            )
        # public_url only -- this is a genuine Europe PMC URL, never a wire
        # URL carrying NCBI courtesy params (Europe PMC calls never do), but
        # routed through the same public_url variable for consistency and
        # auditability (grep for "wire_url" finds every real secret-bearing
        # use site; this one is a documented no-op).
        doc_id = derive_document_id("literature_europepmc_fulltext", public_url, text)
        document = Document(
            doc_id=doc_id, url=public_url, title=f"Europe PMC open-access full text (PMCID {pmcid})",
            publisher="Europe PMC", is_company_ir=False, text=parsed.full_text,
            content_kind=ContentKind.FULL_DOCUMENT, provenance=Provenance.LIVE,
            retrieved_at=utc_now_iso(), authority=DocumentAuthority.BIOMEDICAL_LITERATURE,
        )
        return document, (0 if cached is not None else 1), None


class PubMedLiteratureAdapter:
    """LOCATE (ESearch, Priority A/B/C) -> FETCH (batched EFetch of every
    LOCATEd PMID, plus a composed Europe PMC OA full-text attempt per
    PMCID) -> PARSE (field extraction + RawFact generation). Registered
    under adapter_id ``"pubmed_europepmc"``."""

    def __init__(
        self,
        http: Any,
        settings: Any = None,
        *,
        max_search_requests: int = DEFAULT_MAX_SEARCH_REQUESTS,
        max_articles: int = DEFAULT_MAX_ARTICLES,
        max_fulltext_fetches: int = DEFAULT_MAX_FULLTEXT_FETCHES,
        max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS,
    ) -> None:
        self.http = http
        self.settings = settings
        #: Immutable caps, shared with ``self.europepmc`` -- safe to keep on
        #: ``self`` for the instance's whole lifetime, since it carries no
        #: usage counter (Phase 3F.0.2 requirement 2). Actual usage lives in
        #: ``ExecutionContext.adapter_state`` per run; see
        #: ``_budget_usage_for``.
        self.limits = RequestBudgetLimits(
            max_search_requests=max_search_requests, max_articles=max_articles,
            max_fulltext_fetches=max_fulltext_fetches, max_total_requests=max_total_requests,
        )
        self.europepmc = EuropePmcFullTextAdapter(http, settings, limits=self.limits)

    def execute(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        if step.step_kind is StepKind.LOCATE:
            return self._locate(step, context)
        if step.step_kind is StepKind.FETCH:
            return self._fetch(step, context)
        if step.step_kind is StepKind.PARSE:
            return self._parse(step, context)
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.FAILED,
            failure_reason=f"PubMedLiteratureAdapter cannot handle step_kind={step.step_kind}",
        )

    # -- LOCATE: Priority A/B/C -------------------------------------------
    def _esearch(
        self, term: str, max_pmids: int, context: ExecutionContext
    ) -> tuple[list[str] | None, int, str | None]:
        params = _eutils_params(
            {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids)},
            self.settings,
        )
        wire_url = f"{NCBI_ESEARCH_URL}?{urlencode(params)}"
        _require_allowed_host(wire_url)
        public_url = _public_request_url(wire_url)
        cache_key = f"http_get:{public_url}"
        cached = context.request_cache.get(cache_key)
        if cached is not None:
            if cached.status is not StepStatus.URL_RESOLVED:
                return None, 0, cached.failure_reason or "cached ESearch failed"
            return list(cached.payload.get("pmids", [])), 0, None

        usage = _budget_usage_for(context)
        if not usage.reserve_search(self.limits):
            return None, 0, f"{BUDGET_EXCEEDED_PREFIX}ESearch budget exhausted for term: {term}"

        fetch, transport_error = _safe_get(self.http, wire_url, public_url, self.settings)
        if transport_error is not None:
            context.request_cache[cache_key] = StepExecutionResult(
                step_id="_cache_esearch", status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=transport_error,
            )
            return None, 1, transport_error
        if not fetch.ok:
            failure = _sanitize_text(
                f"ESearch failed: {fetch.outcome} {fetch.error} (url={public_url})",
                settings=self.settings, wire_url=wire_url, public_url=public_url,
            )
            context.request_cache[cache_key] = StepExecutionResult(
                step_id="_cache_esearch", status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=failure,
            )
            return None, 1, failure

        # ``fetch.json()`` returns ``None`` for malformed JSON (see
        # ``collectors/http.py``'s real ``HttpResponse.json()``) -- never
        # conflated with a well-formed response whose ``idlist`` is
        # genuinely empty (handled by the caller as ZERO_RESULTS).
        body = fetch.json()
        pmids: list[str] | None
        if not isinstance(body, dict):
            pmids = None
        else:
            esearchresult = body.get("esearchresult")
            pmids = esearchresult.get("idlist") if isinstance(esearchresult, dict) else None
        if pmids is None:
            failure = f"malformed ESearch response (url={public_url})"
            context.request_cache[cache_key] = StepExecutionResult(
                step_id="_cache_esearch", status=StepStatus.FAILED, http_requests_made=1,
                failure_reason=failure,
            )
            return None, 1, failure

        context.request_cache[cache_key] = StepExecutionResult(
            step_id="_cache_esearch", status=StepStatus.URL_RESOLVED,
            payload={"pmids": pmids}, http_requests_made=1,
        )
        return list(pmids), 1, None

    def _locate(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        ref: LiteratureReference | None = context.literature_reference_for(step.target_id)
        if ref is None:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no LiteratureReference supplied for this target",
            )

        # Priority A: a PMID already known from a ClinicalTrials.gov
        # reference -- no ESearch needed at all.
        if ref.ct_pmid:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.URL_RESOLVED,
                payload={
                    "pmids": [ref.ct_pmid], "priority": "A", "confirmed_same_trial": True,
                    "query_log": [], "coverage_complete": True,
                },
            )

        if ref.nct_id:
            term = f"{ref.nct_id}[si]"
            pmids, http_reqs, failure = self._esearch(term, ref.max_pmids, context)
            query_log = [LiteratureSearchQuery(term=term, priority="B", confirmed_same_trial=True).__dict__]
            if failure is not None:
                return StepExecutionResult(
                    step_id=step.step_id, status=_status_for_failure(failure), http_requests_made=http_reqs,
                    failure_reason=failure,
                    payload={"query_log": query_log, "coverage_complete": False},
                )
            if not pmids:
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.ZERO_RESULTS, http_requests_made=http_reqs,
                    failure_reason=f"no PubMed results for exact NCT-ID search: {term}",
                    payload={"query_log": query_log},
                )
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.URL_RESOLVED, http_requests_made=http_reqs,
                payload={
                    "pmids": pmids[: ref.max_pmids], "priority": "B", "confirmed_same_trial": True,
                    "query_log": query_log, "coverage_complete": True,
                },
            )

        if ref.alias or ref.condition:
            term = " AND ".join(part for part in (ref.alias, ref.condition) if part)
            pmids, http_reqs, failure = self._esearch(term, ref.max_pmids, context)
            query_log = [LiteratureSearchQuery(term=term, priority="C", confirmed_same_trial=False).__dict__]
            if failure is not None:
                return StepExecutionResult(
                    step_id=step.step_id, status=_status_for_failure(failure), http_requests_made=http_reqs,
                    failure_reason=failure,
                    payload={"query_log": query_log, "coverage_complete": False},
                )
            if not pmids:
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.ZERO_RESULTS, http_requests_made=http_reqs,
                    failure_reason=f"no PubMed results for alias/condition discovery search: {term}",
                    payload={"query_log": query_log},
                )
            # Priority C candidates are NEVER auto-confirmed as the same
            # trial/program -- carried forward explicitly on every payload
            # this LOCATE produces (module docstring / Phase 3F requirement
            # 3's "treat ONLY as a discovery candidate").
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.URL_RESOLVED, http_requests_made=http_reqs,
                payload={
                    "pmids": pmids[: ref.max_pmids], "priority": "C", "confirmed_same_trial": False,
                    "query_log": query_log, "coverage_complete": True,
                },
            )

        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.FAILED,
            failure_reason="LiteratureReference supplies neither ct_pmid, nct_id, nor alias/condition",
        )

    # -- FETCH: batched EFetch + composed Europe PMC OA full text --------
    def _efetch_batch(
        self, pmids: list[str], context: ExecutionContext
    ) -> tuple[dict[str, ParsedPubmedArticle] | None, int, str | None, list[str]]:
        """Fetches every PMID not already cached, in exactly ONE EFetch
        call (never one at a time -- Phase 3F requirement). Returns
        ``pmid -> ParsedPubmedArticle`` for every PMID successfully
        resolved (from cache or this call), plus the list of PMIDs excluded
        by ``RequestBudgetUsage.cap_articles`` (Phase 3F.0.1 requirement 4) --
        never fetched, never cached as failed, simply not yet processed."""
        resolved: dict[str, ParsedPubmedArticle] = {}
        to_fetch: list[str] = []
        for pmid in pmids:
            cached = context.request_cache.get(f"literature_pmid:{pmid}")
            if cached is not None and cached.status is StepStatus.BODY_FETCHED:
                article = cached.payload.get("article")
                if article is not None:
                    resolved[pmid] = article
            elif cached is None:
                to_fetch.append(pmid)
            # a cached FAILED entry for this pmid is simply left unresolved

        # Phase 3F.0.1 requirement 4: only genuinely NEW PMIDs consume the
        # article budget -- already-cached PMIDs above never reach this
        # point at all, so the same-PMID-fan-out dedup guarantee is
        # unaffected by this cap.
        usage = _budget_usage_for(context)
        to_fetch, budget_excluded = usage.cap_articles(self.limits, to_fetch)
        if not to_fetch:
            return resolved, 0, None, budget_excluded

        if not usage.reserve_other(self.limits):
            return resolved, 0, None, budget_excluded + to_fetch

        params = _eutils_params(
            {"db": "pubmed", "id": ",".join(to_fetch), "retmode": "xml", "rettype": "abstract"},
            self.settings,
        )
        wire_url = f"{NCBI_EFETCH_URL}?{urlencode(params)}"
        _require_allowed_host(wire_url)
        public_url = _public_request_url(wire_url)
        fetch, transport_error = _safe_get(self.http, wire_url, public_url, self.settings)
        if transport_error is not None:
            for pmid in to_fetch:
                context.request_cache[f"literature_pmid:{pmid}"] = StepExecutionResult(
                    step_id=f"_cache_literature_pmid_{pmid}", status=StepStatus.FAILED,
                    failure_reason=transport_error,
                )
            return (resolved if resolved else None), 1, transport_error, budget_excluded
        if not fetch.ok:
            failure = _sanitize_text(
                f"EFetch failed: {fetch.outcome} {fetch.error} (url={public_url})",
                settings=self.settings, wire_url=wire_url, public_url=public_url,
            )
            for pmid in to_fetch:
                context.request_cache[f"literature_pmid:{pmid}"] = StepExecutionResult(
                    step_id=f"_cache_literature_pmid_{pmid}", status=StepStatus.FAILED,
                    failure_reason=failure,
                )
            return (resolved if resolved else None), 1, failure, budget_excluded

        text = fetch.text
        shape_error, reason = check_pubmed_xml_shape(text)
        if shape_error is not None:
            failure = f"{shape_error.value}: {reason}"
            for pmid in to_fetch:
                context.request_cache[f"literature_pmid:{pmid}"] = StepExecutionResult(
                    step_id=f"_cache_literature_pmid_{pmid}", status=StepStatus.FAILED,
                    failure_reason=failure,
                )
            return (resolved if resolved else None), 1, failure, budget_excluded

        articles = parse_pubmed_articleset(text) or []
        returned_pmids = {a.pmid for a in articles}
        for article in articles:
            context.request_cache[f"literature_pmid:{article.pmid}"] = StepExecutionResult(
                step_id=f"_cache_literature_pmid_{article.pmid}", status=StepStatus.BODY_FETCHED,
                payload={"article": article, "raw_xml": text}, http_requests_made=0,
            )
            if article.pmid in to_fetch:
                resolved[article.pmid] = article

        # PMID mismatch: a requested PMID absent from the response body --
        # never silently accepted as if it had been fetched (Phase 3F
        # requirement: "PMID mismatch -> FAILED").
        missing = [pmid for pmid in to_fetch if pmid not in returned_pmids]
        for pmid in missing:
            failure = f"PMID mismatch: {pmid} was requested but not present in the EFetch response"
            context.request_cache[f"literature_pmid:{pmid}"] = StepExecutionResult(
                step_id=f"_cache_literature_pmid_{pmid}", status=StepStatus.FAILED,
                failure_reason=failure,
            )

        return (
            (resolved if resolved else None), 1,
            (None if resolved else "no requested PMID resolved"), budget_excluded,
        )

    def _fetch(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        upstream = _upstream_payload(step, context)
        pmids = upstream.get("pmids") or []
        if not pmids:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no PMIDs from the LOCATE step",
            )

        articles, http_reqs, failure, budget_excluded_pmids = self._efetch_batch(pmids, context)
        if not articles:
            if budget_excluded_pmids and not failure:
                return StepExecutionResult(
                    step_id=step.step_id, status=StepStatus.SKIPPED_DUE_TO_BUDGET, http_requests_made=http_reqs,
                    failure_reason=f"{BUDGET_EXCEEDED_PREFIX}article budget exhausted before any PMID "
                    f"could be fetched: {budget_excluded_pmids}",
                    payload={**upstream, "budget_excluded_pmids": budget_excluded_pmids, "coverage_complete": False},
                )
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED, http_requests_made=http_reqs,
                failure_reason=failure or "no PMID could be fetched",
                payload={**upstream},
            )

        # Deliberately rebuilt WITHOUT ``_eutils_params`` (no tool/email/
        # api_key) -- this is a public_url by construction, never a
        # wire_url, since it identifies the document for Document.url/
        # derive_document_id, never a real outbound request (the actual
        # EFetch request already happened above, via the batched wire_url).
        pubmed_public_url_by_pmid = {
            pmid: _public_request_url(
                f"{NCBI_EFETCH_URL}?{urlencode({'db': 'pubmed', 'id': pmid, 'retmode': 'xml'})}"
            )
            for pmid in articles
        }
        # Composed Europe PMC OA full-text attempt, per PMCID -- only ever
        # when the search result's own flags confirm OA + inEPMC (Phase 3F
        # requirement 3/6). A non-OA or missing-PMCID article still counts
        # as successfully FETCHED (its PubMed abstract/metadata), it simply
        # carries no full-text document. Budget-skipped Europe PMC calls
        # (Phase 3F.0.1 requirement 4) are recorded, never silently dropped.
        europepmc_by_pmid: dict[str, dict[str, Any]] = {}
        total_epmc_reqs = 0
        budget_skipped_europepmc_search_pmids: list[str] = []
        budget_skipped_fulltext_pmcids: list[str] = []
        for pmid, article in articles.items():
            search_result, search_reqs, search_failure = self.europepmc.search(pmid, context)
            total_epmc_reqs += search_reqs
            if search_result is None:
                if search_failure and search_failure.startswith(BUDGET_EXCEEDED_PREFIX):
                    budget_skipped_europepmc_search_pmids.append(pmid)
                continue
            fulltext_document = None
            if article.pmcid != UNKNOWN and search_result.is_open_access and search_result.in_epmc:
                fulltext_document, fetch_reqs, fetch_failure = self.europepmc.fetch_fulltext(
                    article.pmcid, context
                )
                total_epmc_reqs += fetch_reqs
                if fulltext_document is None and fetch_failure and fetch_failure.startswith(BUDGET_EXCEEDED_PREFIX):
                    budget_skipped_fulltext_pmcids.append(article.pmcid)
            europepmc_by_pmid[pmid] = {
                "search_result": search_result, "fulltext_document": fulltext_document,
            }

        # Store the PubMed Document(s) now (PARSE only re-reads text already
        # in hand; DocumentStore.put is idempotent by URL+content, so a
        # later target reusing the same PMID never creates a second GET).
        document_ids: dict[str, str] = {}
        for pmid, article in articles.items():
            cached = context.request_cache.get(f"literature_pmid:{pmid}")
            raw_xml = cached.payload.get("raw_xml", "") if cached is not None else ""
            public_url = pubmed_public_url_by_pmid[pmid]
            doc_id = derive_document_id("literature_pubmed", public_url, raw_xml or article.pmid)
            document = Document(
                doc_id=doc_id, url=public_url,
                title=article.article_title if article.article_title != UNKNOWN else f"PMID {pmid}",
                publisher=article.journal_title if article.journal_title != UNKNOWN else "PubMed",
                published_date=article.publication_date,
                doc_type="pubmed_article",
                is_company_ir=False, text=raw_xml,
                content_kind=ContentKind.EXCERPT if article.has_abstract else ContentKind.METADATA_ONLY,
                provenance=Provenance.LIVE, retrieved_at=utc_now_iso(),
                authority=DocumentAuthority.BIOMEDICAL_LITERATURE,
            )
            stored = context.document_store.put(
                document, document_id=doc_id, document_role=DocumentRole.JOURNAL_ARTICLE,
            )
            document_ids[pmid] = stored.document_id

        for epmc in europepmc_by_pmid.values():
            fulltext_document = epmc.get("fulltext_document")
            if fulltext_document is None:
                continue
            stored = context.document_store.put(
                fulltext_document, document_id=fulltext_document.doc_id,
                document_role=DocumentRole.JOURNAL_ARTICLE,
            )
            epmc["fulltext_document_id"] = stored.document_id

        coverage_complete = (
            upstream.get("coverage_complete", True)
            and not budget_excluded_pmids
            and not budget_skipped_europepmc_search_pmids
            and not budget_skipped_fulltext_pmcids
        )
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.BODY_FETCHED,
            http_requests_made=http_reqs + total_epmc_reqs,
            payload={
                **upstream,
                "fetched_pmids": list(articles.keys()),
                "document_ids": document_ids,
                "budget_excluded_pmids": budget_excluded_pmids,
                "budget_skipped_europepmc_search_pmids": budget_skipped_europepmc_search_pmids,
                "budget_skipped_fulltext_pmcids": budget_skipped_fulltext_pmcids,
                "coverage_complete": coverage_complete,
                "europepmc": {
                    pmid: {
                        "is_open_access": v["search_result"].is_open_access,
                        "in_epmc": v["search_result"].in_epmc,
                        "license": v["search_result"].license,
                        "fulltext_document_id": v.get("fulltext_document_id"),
                    }
                    for pmid, v in europepmc_by_pmid.items()
                },
            },
        )

    # -- PARSE: structured field extraction + diagnostics -----------------
    def _parse(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        """Mirrors ``Form4Adapter._parse``'s own precedent exactly: this
        step returns fully-parsed, structured per-article data (never a
        RawFact) -- RawFact generation is a SEPARATE, independently-tested
        pure function (``collectors/literature.raw_facts_from_pubmed_article``
        / ``raw_facts_from_europepmc_fulltext``), not wired into this
        adapter, exactly as ``collectors/form4.raw_facts_from_form4`` exists
        but is never called from ``Form4Adapter`` -- RawFact/Fact-Collector
        wiring is Pipeline territory, which this module never touches
        (module docstring)."""
        upstream = _upstream_payload(step, context)
        fetched_pmids = upstream.get("fetched_pmids") or []
        if not fetched_pmids:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no fetched PMIDs available to parse",
            )
        document_ids = upstream.get("document_ids") or {}
        europepmc = upstream.get("europepmc") or {}

        parsed_documents: list[dict[str, Any]] = []
        for pmid in fetched_pmids:
            cached = context.request_cache.get(f"literature_pmid:{pmid}")
            article: ParsedPubmedArticle | None = cached.payload.get("article") if cached is not None else None
            if article is None:
                continue
            epmc_entry = europepmc.get(pmid) or {}
            parsed_documents.append(
                {
                    "pmid": pmid, "document_id": document_ids.get(pmid), "article": article,
                    "retracted": article.is_retracted,
                    "is_correction_or_erratum": article.is_correction_or_erratum,
                    "nct_ids": list(article.nct_ids),
                    "europepmc_is_open_access": epmc_entry.get("is_open_access", False),
                    "europepmc_in_epmc": epmc_entry.get("in_epmc", False),
                    "europepmc_license": epmc_entry.get("license", ""),
                    "europepmc_fulltext_document_id": epmc_entry.get("fulltext_document_id"),
                    "full_text_acquired": bool(epmc_entry.get("fulltext_document_id")),
                }
            )

        if not parsed_documents:
            return StepExecutionResult(
                step_id=step.step_id, status=StepStatus.FAILED,
                failure_reason="no cached article payload available at PARSE time",
            )

        headline_document_id = parsed_documents[0]["document_id"]
        return StepExecutionResult(
            step_id=step.step_id, status=StepStatus.PARSED, document_id=headline_document_id,
            payload={**upstream, "parsed_documents": parsed_documents},
        )


__all__ = [
    "ALLOWED_HOSTS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RATE_LIMIT_RPS",
    "DEFAULT_TIMEOUT_SECONDS",
    "EUROPEPMC_FULLTEXT_URL",
    "EUROPEPMC_SEARCH_URL",
    "LITERATURE_ADAPTER_ID",
    "MAX_PMIDS_PER_BATCH",
    "MAX_REQUESTS_PER_RUN",
    "NCBI_EFETCH_URL",
    "NCBI_ESEARCH_URL",
    "EuropePmcFullTextAdapter",
    "LiteratureReference",
    "LiteratureSearchQuery",
    "PubMedLiteratureAdapter",
    "RequestBudgetLimits",
    "RequestBudgetUsage",
]
