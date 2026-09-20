"""Phase 4.2A: Literature Pipeline Integration.

The ONLY module in this repository that connects the Literature
Document-First acquisition path --

    AcquisitionExecutor -> PubMedLiteratureAdapter -> DocumentStore
    -> Literature Evidence Projection (Phase 4.1A)
    -> Literature Chunk Projection (Phase 4.1B)

-- to the real production orchestration path
(``cli.py::run_one()`` -> ``Pipeline.run()``). Everything upstream of this
module (``research/acquisition_executor.py``,
``research/literature_acquisition_adapter.py``,
``research/literature_evidence_projection.py``,
``research/literature_chunk_projection.py``) is reused exactly as Phase
3F/3F.0.x/3F.1/3F.2/4.1A/4.1B built and tested it -- this module adds no
new PubMed/Europe PMC parsing, no new chunking rule, no new Evidence
Integrity classification. It only decides WHEN those existing pieces run
in a real invocation, and carries their output into ``Pipeline.run()``'s
existing ``collection_results``/``chunks`` parameters.

User-authorized scope for this phase, and nothing beyond it:

* Literature ONLY. SEC/ClinicalTrials/Form4's own new Acquisition Adapters
  remain fully unconnected; ``collectors/fda.py`` (openFDA) is untouched;
  no Web Search fallback is enabled; no Master Registry is implemented.
  The pre-existing SEC EDGAR/ClinicalTrials.gov/openFDA structured
  collectors keep running exactly as before, unconditionally, in
  ``cli.py::collect()``.
* Reachable ONLY when BOTH a default-OFF CLI flag
  (``--document-first-literature``) AND EXACTLY ONE explicit reference
  (``--literature-pmid`` XOR ``--literature-nct-id``) are given, AND
  ``--live`` is also given. Network reachability is genuinely gated on
  this: ``run_literature_pipeline_acquisition`` -- the one function that
  can ever touch a network -- is called only when
  ``validate_literature_pipeline_request`` returned a non-``None``
  request, which itself requires the flag. This is NOT the same claim as
  "every other invocation is byte-identical to before this phase" --
  it is not, in two narrow, documented ways, and neither is a behavior
  change beyond a diagnostics-surface addition:

  1. ``cli.py`` imports this module unconditionally (so
     ``validate_literature_pipeline_request`` can run before any
     collector/DB/HTTP object is constructed) -- this is a real, new
     import that did not exist before Phase 4.2A, even on a flag-OFF run.
  2. ``ResearchResult.direct_acquisition_info``/``Pipeline.__init__``'s
     ``direct_acquisition_info`` parameter are new fields that exist on
     every run's result object, flag-OFF included (see
     ``bundle_diagnostics(None, enabled=False)`` returning ``{}`` and
     ``cli.py::result_to_json`` omitting the ``direct_acquisition_info``
     key when it is empty -- Phase 4.2A correction 1 -- so the flag-OFF
     ``--json``/console output ITSELF is unchanged, even though the
     Python object underneath it now carries one additional, empty
     field).

  Everything a flag-OFF run actually DOES -- which collectors run, what
  facts/chunks/verdict/report/JSON keys it produces -- is unchanged; only
  the import graph and this one always-present-but-usually-empty
  diagnostics field are new.
* No real NCBI/Europe PMC communication happens anywhere in this
  repository's own test suite or CI -- every test here drives this module
  against a fake HTTP double or a local loopback server, exactly like
  every other Acquisition Adapter test in this repository.
* ``--force-rerun``, ``ImplementationStatus.LIVE_VERIFIED`` promotion, a
  Web Search fallback, and any scheduler/Routine wiring are NOT part of
  this phase, and this module implements none of them. Nothing here writes
  a Live Smoke LAST_RUN marker or touches a capture directory --
  ``AllowlistedHttpClient`` is used WITHOUT ``out_dir`` unless a caller
  explicitly supplies one, and this module never calls
  ``literature_live_smoke.py`` at all (see that module's own "NEVER
  imported by cli.py" boundary, preserved here by construction: this file
  never imports it).
* An explicit PMID/NCT ID reference is never promoted to "the current
  program". ``LiteraturePipelineBundle.current_program_confirmed`` is
  always ``False`` -- an NCT-ID match or a PMID fetch is evidence a paper
  exists/references that identifier, never a confirmation that it concerns
  the SAME clinical program a ticker's other evidence is about. Program
  resolution / adaptive, ticker-driven acquisition is explicitly future
  work (Phase 4.3+), not attempted here.

Credential boundary: ``IRA_NCBI_TOOL``/``IRA_NCBI_EMAIL`` are read via
``ncbi_credentials.resolve_ncbi_credentials`` -- directly from the process
environment (or an injected ``env`` mapping, for offline testing), never
through ``config.Settings`` (whose ``ncbi_tool`` carries a silent
non-empty default that must never be mistaken for "the environment is
configured"). Both are checked TWICE, independently: once in
``validate_literature_pipeline_request`` (so a CLI invocation can fail
fast, before ``Pipeline`` or any collector is even constructed) and again,
defensively, at the top of ``run_literature_pipeline_acquisition`` itself
(never assuming a caller already validated -- the same "never trust
silently" pattern Phase 4.1A/4.1B use throughout this codebase). Neither
credential VALUE is ever placed in a request, a report, an exception, a
``CollectionResult``, a diagnostics dict, a cache key, a ``Document``, a
``Chunk``, a log line, a ``repr()``, JSON output, a URL, or a manifest --
this module never constructs a string containing a credential value at
all; only ``CredentialStatus``-shaped booleans (configured/not) ever cross
into a diagnostic this module builds. Every wire-URL-touching call goes
through the SAME ``AllowlistedHttpClient``
(``research/sec_live_smoke.py``) and ``PubMedLiteratureAdapter``
(``research/literature_acquisition_adapter.py``) every other Acquisition
Adapter in this repository already uses -- this module builds no HTTP
client and no redirect validator of its own.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..collectors.base import CollectionResult
from ..collectors.documents import Chunk
from ..collectors.http import DEFAULT_MAX_REDIRECTS
from .acquisition_executor import AcquisitionExecutor
from .clinicaltrials_acquisition_adapter import NCT_ID_RE
from .document_store import DocumentStore
from .literature_acquisition_adapter import (
    ALLOWED_HOSTS,
    LITERATURE_ADAPTER_ID,
    MAX_PMIDS_PER_BATCH,
    LiteratureReference,
    PubMedLiteratureAdapter,
    RequestBudgetLimits,
    build_single_target_literature_graph,
)
from .literature_chunk_projection import project_literature_chunks
from .literature_evidence_projection import project_literature_target_reports
from .ncbi_credentials import NCBI_API_KEY_ENV_VAR, resolve_ncbi_credentials
from .sec_live_smoke import AllowlistedHttpClient
from .source_routing import TargetAcquisitionOutcome

log = logging.getLogger(__name__)

#: The single target id this module ever drives one acquisition through.
#: Never reused across two different calls within the SAME run (this
#: module only ever runs ONE literature target per invocation, per the
#: "exactly one explicit reference" input rule), and never one of the
#: 31-group master catalog's own target ids.
TARGET_ID = "target_literature_pipeline"

#: Never a secret -- a fixed, generic software identifier, unrelated to
#: NCBI's own courtesy-identification query parameters (tool/email/
#: api_key), which this module gates separately via
#: ``ncbi_credentials.py``.
LITERATURE_PIPELINE_USER_AGENT = "investment-research-agent-literature-pipeline/1.0"

DEFAULT_TIMEOUT_SECONDS = 20.0
#: Deliberately far below NCBI's own documented 3 req/s no-API-key cap,
#: and kept the SAME regardless of whether an API key is configured --
#: mirrors Literature Live Smoke's own, equally conservative choice
#: (Phase 3F.1).
DEFAULT_RATE_LIMIT_RPS = 0.5
MAX_RETRIES = 1
MAX_REDIRECTS = DEFAULT_MAX_REDIRECTS

#: Section 6's explicit request-count formulas (mirrors
#: ``literature_live_smoke.py``'s own ``_discovery_budget``/
#: ``_targeted_budget``/``_effective_max_fulltext_fetches`` exactly, in
#: shape and in every numeric default). Duplicated here rather than
#: imported -- ``literature_live_smoke.py`` is declared "NEVER imported by
#: cli.py/pipeline.py/any production code path" in its own module
#: docstring, and this module IS imported by ``cli.py``, so importing
#: from it here would silently break that module's own stated boundary.
#: A dedicated test asserts this module's formula/caps stay numerically
#: identical to Literature Live Smoke's.
TARGETED_MAX_FULLTEXT_FETCHES_CAP = 1
DEFAULT_MAX_ARTICLES = 3
DEFAULT_MAX_FULLTEXT_FETCHES = 1

#: NCBI PMIDs are plain decimal integers; generous upper bound on digit
#: count, never a guess at a real PMID's current maximum length. Mirrors
#: ``literature_live_smoke.PMID_RE`` exactly (duplicated for the same
#: import-boundary reason as the budget formula above).
PMID_RE = re.compile(r"^\d{1,9}$")


class LiteraturePipelineRequestError(ValueError):
    """Raised by ``validate_literature_pipeline_request`` for every
    rejected input combination this phase's own input rules define --
    distinguishable from a generic ``ValueError`` so a caller (``cli.py``)
    can catch it and print/exit(2) without a traceback. Every raise site
    fires strictly BEFORE any network-capable object (``DocumentStore``,
    ``AcquisitionExecutor``, ``AllowlistedHttpClient``) is even
    constructed -- HTTP attempt count is always 0 for a rejected request.
    """


class LiteratureReferenceMode:
    """Which explicit reference kind this request carries -- a plain
    namespace of string constants (not an ``Enum``) so
    ``LiteraturePipelineBundle.reference_mode`` serializes to JSON/prints
    without a ``.value`` indirection at every call site."""

    PMID = "PMID"
    NCT_ID = "NCT_ID"


@dataclass(frozen=True)
class LiteraturePipelineRequest:
    """A validated, ready-to-execute Literature Document-First acquisition
    request. Constructing one always means every CLI-input rule this
    phase defines already passed -- never build one by hand; only
    ``validate_literature_pipeline_request`` produces one."""

    reference_mode: str  # LiteratureReferenceMode.PMID | .NCT_ID
    pmid: str | None
    nct_id: str | None
    #: Already clamped to ``TARGETED_MAX_FULLTEXT_FETCHES_CAP`` when
    #: ``reference_mode`` is PMID -- see ``_effective_max_fulltext_fetches``
    #: below. What a caller displays and what actually executes can never
    #: silently disagree (mirrors Literature Live Smoke's own correction on
    #: exactly this point).
    max_articles: int
    max_fulltext_fetches: int


def _effective_max_fulltext_fetches(reference_mode: str, max_fulltext_fetches: int) -> int:
    if reference_mode == LiteratureReferenceMode.PMID:
        return min(max_fulltext_fetches, TARGETED_MAX_FULLTEXT_FETCHES_CAP)
    return max_fulltext_fetches


def _discovery_request_budget(
    max_articles: int, max_fulltext_fetches: int
) -> tuple[RequestBudgetLimits, int, str]:
    """NCT discovery mode: ``ESearch(1) + EFetch(1, batched) + EuropePMC
    search(1 x max_articles) + EuropePMC fulltext(max_fulltext_fetches)``.
    ``max_search_requests`` covers BOTH ESearch and Europe PMC search,
    since ``RequestBudgetUsage.reserve_search`` shares one budget between
    them."""
    max_search_requests = 1 + max_articles
    max_total_requests = 1 + 1 + max_articles + max_fulltext_fetches
    formula = (
        f"ESearch(1) + EFetch(1, batched) + EuropePMC search(1 x max_articles={max_articles}) "
        f"+ EuropePMC fulltext(max_fulltext_fetches={max_fulltext_fetches}) = {max_total_requests}"
    )
    limits = RequestBudgetLimits(
        max_search_requests=max_search_requests,
        max_articles=max_articles,
        max_fulltext_fetches=max_fulltext_fetches,
        max_total_requests=max_total_requests,
    )
    return limits, max_total_requests, formula


def _targeted_request_budget(max_fulltext_fetches: int) -> tuple[RequestBudgetLimits, int, str]:
    """Targeted PMID mode: a single PMID (Priority A ``ct_pmid`` direct
    path) skips ESearch entirely -- ``EFetch(1) + EuropePMC search(1) +
    EuropePMC fulltext(<=1)``. ``max_fulltext_fetches`` is clamped to
    ``TARGETED_MAX_FULLTEXT_FETCHES_CAP`` regardless of what the caller
    requested."""
    capped = min(max_fulltext_fetches, TARGETED_MAX_FULLTEXT_FETCHES_CAP)
    max_search_requests = 1  # Europe PMC search only -- no ESearch in targeted mode.
    max_total_requests = 1 + 1 + capped  # EFetch(1) + EuropePMC search(1) + fulltext(<=capped)
    formula = (
        f"EFetch(1) + EuropePMC search(1) + EuropePMC fulltext(max_fulltext_fetches={capped}, "
        f"hard cap {TARGETED_MAX_FULLTEXT_FETCHES_CAP} for targeted mode) = {max_total_requests}"
    )
    limits = RequestBudgetLimits(
        max_search_requests=max_search_requests,
        max_articles=1,
        max_fulltext_fetches=capped,
        max_total_requests=max_total_requests,
    )
    return limits, max_total_requests, formula


def validate_literature_pipeline_request(
    *,
    enabled: bool,
    pmid: str | None,
    nct_id: str | None,
    live: bool,
    use_fixtures: bool,
    use_corpus: bool,
    max_articles: int = DEFAULT_MAX_ARTICLES,
    max_fulltext_fetches: int = DEFAULT_MAX_FULLTEXT_FETCHES,
    env: dict[str, str] | None = None,
) -> LiteraturePipelineRequest | None:
    """Validate one CLI invocation's Literature Document-First inputs,
    BEFORE any collector, ``DocumentStore``, or network-capable object is
    constructed. Returns ``None`` only for the fully-default case (the
    flag was never passed and no identifier was given either) -- in that
    case, ``run_literature_pipeline_acquisition`` is never called and no
    acquisition-related code runs at all for this invocation (see the
    module docstring's own note on the narrow, documented ways a flag-OFF
    run is NOT literally byte-identical to before this phase -- an
    unconditional import and an always-present-but-empty diagnostics
    field -- neither of which this function's own return value is
    responsible for). EVERY other combination either returns a
    fully-validated ``LiteraturePipelineRequest`` or raises
    ``LiteraturePipelineRequestError`` -- there is no silent partial
    acceptance.

    Order of checks (each one is a distinct, explicitly-named rejection
    reason a caller can print verbatim):

    1. An identifier given WITHOUT the flag -- rejected, never silently
       ignored.
    2. Flag on, no identifier -- rejected (exactly one is required).
    3. Flag on, both identifiers -- rejected (mutually exclusive).
    4. The one given identifier is malformed -- rejected before any
       network-shaped object exists.
    5. ``--fixtures``/``--corpus`` combined with the flag -- rejected (out
       of scope this phase).
    6. ``--live`` not given -- rejected (this path can never reach the
       network without it).
    7. A required NCBI credential (``IRA_NCBI_TOOL``/``IRA_NCBI_EMAIL``) is
       unconfigured -- rejected. ``IRA_NCBI_API_KEY`` is never required.

    ``env`` is injectable for offline testing only -- omit it in
    production to read the real process environment.
    """
    normalized_pmid = (pmid or "").strip() or None
    normalized_nct_id = (nct_id or "").strip() or None

    if not enabled:
        if normalized_pmid or normalized_nct_id:
            raise LiteraturePipelineRequestError(
                "--literature-pmid/--literature-nct-id was given without "
                "--document-first-literature -- refusing rather than silently ignoring the "
                "identifier"
            )
        return None  # the fully-default case: pre-existing behavior, untouched.

    if normalized_pmid and normalized_nct_id:
        raise LiteraturePipelineRequestError(
            "--literature-pmid and --literature-nct-id are mutually exclusive -- "
            "--document-first-literature requires exactly ONE explicit reference"
        )
    if not normalized_pmid and not normalized_nct_id:
        raise LiteraturePipelineRequestError(
            "--document-first-literature requires exactly one of --literature-pmid/"
            "--literature-nct-id"
        )

    if normalized_pmid is not None:
        if not PMID_RE.match(normalized_pmid):
            raise LiteraturePipelineRequestError(f"malformed PMID: {normalized_pmid!r}")
        reference_mode = LiteratureReferenceMode.PMID
    else:
        assert normalized_nct_id is not None  # guaranteed by the two checks above
        normalized_nct_id = normalized_nct_id.upper()
        if not NCT_ID_RE.match(normalized_nct_id):
            raise LiteraturePipelineRequestError(f"malformed NCT ID: {nct_id!r}")
        reference_mode = LiteratureReferenceMode.NCT_ID

    if use_fixtures or use_corpus:
        raise LiteraturePipelineRequestError(
            "--document-first-literature cannot be combined with --fixtures/--corpus in this "
            "phase"
        )
    if not live:
        raise LiteraturePipelineRequestError(
            "--document-first-literature requires --live -- refusing before any request rather "
            "than accepting a flag combination that can never reach the network"
        )
    if max_articles < 1:
        raise LiteraturePipelineRequestError(
            f"--literature-max-articles must be >= 1, got {max_articles}"
        )
    if max_fulltext_fetches < 0:
        raise LiteraturePipelineRequestError(
            f"--literature-max-fulltext-fetches must be >= 0, got {max_fulltext_fetches}"
        )

    _credentials, _status, missing = resolve_ncbi_credentials(env)
    if missing:
        raise LiteraturePipelineRequestError(
            f"missing required environment variable(s): {', '.join(missing)} -- refusing before "
            f"any request ({NCBI_API_KEY_ENV_VAR} remains optional)"
        )

    return LiteraturePipelineRequest(
        reference_mode=reference_mode,
        pmid=normalized_pmid,
        nct_id=normalized_nct_id,
        max_articles=max_articles,
        max_fulltext_fetches=_effective_max_fulltext_fetches(reference_mode, max_fulltext_fetches),
    )


@dataclass(frozen=True)
class LiteraturePipelineBundle:
    """Everything ``cli.run_one()`` needs from one Literature Document-
    First acquisition call -- and nothing more. Deliberately excludes the
    raw ``ExecutionReport``, ``ParsedPubmedArticle``, ``EuropePmcSearch
    Result``, and any Document/Chunk BODY duplicated into a diagnostic
    field: ``collection_result``/``chunks`` below ARE the canonical
    objects (never a copy) that feed ``Pipeline.run()``'s existing
    ``collection_results``/``chunks`` parameters; every other field here
    is a count, a boolean, or a short structured reason string.
    """

    #: ``None`` only when this request was refused before any acquisition
    #: attempt (see ``refused_reason``) -- never a placeholder empty
    #: ``CollectionResult``, so a caller can tell "we tried and got
    #: nothing" apart from "we never tried" (CLAUDE.md rule 8's spirit).
    collection_result: CollectionResult | None
    chunks: tuple[Chunk, ...]
    document_count: int
    source_count: int
    raw_fact_count: int
    chunk_count: int
    #: One per distinct URL actually dispatched (cache misses past the
    #: host allowlist), regardless of physical retries -- see
    #: ``physical_attempt_count`` for the retry-inclusive count.
    logical_request_count: int
    physical_attempt_count: int
    cache_hit_count: int
    coverage_complete: bool
    unresolved_reasons: tuple[str, ...]
    reference_mode: str
    pmid: str | None
    nct_id: str | None
    #: The single target's own ``TargetAcquisitionOutcome`` (e.g.
    #: "ACQUIRED"/"ACQUISITION_INCOMPLETE"), or ``None`` when refused
    #: before an executor run ever happened.
    target_outcome: str | None = None
    duplicate_document_count: int = 0
    excluded_non_primary_document_count: int = 0
    #: Always False -- an explicit PMID/NCT ID reference is never promoted
    #: to "the current program" by this module. See module docstring.
    current_program_confirmed: bool = False
    #: Set only for a pre-send refusal (missing credential this call
    #: re-checked defensively, or -- in principle -- a caller that skipped
    #: ``validate_literature_pipeline_request``). ``None`` for every run
    #: that reached the executor, whatever its outcome there.
    refused_reason: str | None = None
    anthropic_api_calls: int = 0
    web_search_calls: int = 0
    external_llm_tokens: int = 0

    @property
    def refused(self) -> bool:
        return self.refused_reason is not None


def _refused_bundle(request: LiteraturePipelineRequest, reason: str) -> LiteraturePipelineBundle:
    return LiteraturePipelineBundle(
        collection_result=None,
        chunks=(),
        document_count=0,
        source_count=0,
        raw_fact_count=0,
        chunk_count=0,
        logical_request_count=0,
        physical_attempt_count=0,
        cache_hit_count=0,
        coverage_complete=False,
        unresolved_reasons=(reason,),
        reference_mode=request.reference_mode,
        pmid=request.pmid,
        nct_id=request.nct_id,
        target_outcome=None,
        refused_reason=reason,
    )


def run_literature_pipeline_acquisition(
    request: LiteraturePipelineRequest,
    *,
    ticker: str = "",
    http_client: Any | None = None,
    env: dict[str, str] | None = None,
    out_dir: Path | None = None,
    on_first_attempt: Callable[[], None] | None = None,
) -> LiteraturePipelineBundle:
    """Run ONE Literature Document-First acquisition for ``request`` and
    project it all the way through to a canonical ``CollectionResult`` plus
    a canonical ``Chunk`` tuple, ready for ``Pipeline.run()``.

    ``http_client``/``env`` are injectable for offline testing only; a
    production caller leaves both ``None``, in which case this constructs
    a real, host-allowlisted ``AllowlistedHttpClient`` (the SAME class
    every other live-communication-capable module in this repository
    uses -- see module docstring) and WILL make real outbound NCBI/Europe
    PMC requests if credentials are configured. Never called from anywhere
    except ``cli.py::run_one()``, and only when
    ``validate_literature_pipeline_request`` returned a non-``None``
    request for this invocation.

    Never raises for a request/network-level failure: a provider failure,
    a BLOCKED/RATE_LIMITED outcome, a budget exclusion, or a missing
    credential all become an explicit, structured field on the returned
    bundle (``refused_reason``, ``coverage_complete=False``,
    ``unresolved_reasons``) -- never a crash, and never a silently
    "successful" empty result. ``CollectionResult.degraded`` on the
    returned bundle's ``collection_result`` is what
    ``Pipeline.run()``'s own blocking-propagation check (Phase 4.2A) reads
    to withhold an Action on a genuinely incomplete fetch.
    """
    credentials, _status, missing = resolve_ncbi_credentials(env)
    if missing:
        reason = (
            f"missing required environment variable(s): {', '.join(missing)} -- refusing before "
            f"any request ({NCBI_API_KEY_ENV_VAR} remains optional)"
        )
        return _refused_bundle(request, reason)

    if request.reference_mode == LiteratureReferenceMode.PMID:
        limits, max_requests, _formula = _targeted_request_budget(request.max_fulltext_fetches)
        reference = LiteratureReference(ct_pmid=request.pmid, max_pmids=1)
    else:
        limits, max_requests, _formula = _discovery_request_budget(
            request.max_articles, request.max_fulltext_fetches
        )
        reference = LiteratureReference(nct_id=request.nct_id, max_pmids=MAX_PMIDS_PER_BATCH)

    client = (
        http_client
        if http_client is not None
        else AllowlistedHttpClient(
            user_agent=LITERATURE_PIPELINE_USER_AGENT,
            allowed_hosts=ALLOWED_HOSTS,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            rate_limit_rps=DEFAULT_RATE_LIMIT_RPS,
            max_retries=MAX_RETRIES,
            max_requests=max_requests,
            max_redirects=MAX_REDIRECTS,
            out_dir=out_dir,
            source="literature_pipeline_integration",
            # Mirrors Literature Live Smoke's own stricter policy: this
            # module never displays or stores the configured
            # IRA_NCBI_TOOL value either, even though `tool` is not
            # secret-shaped the way email/api_key are.
            additional_secret_query_params=frozenset({"tool"}),
            on_first_attempt=on_first_attempt,
            save_failed_responses=True,
        )
    )

    graph = build_single_target_literature_graph(TARGET_ID)
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(
        client,
        credentials,
        max_search_requests=limits.max_search_requests,
        max_articles=limits.max_articles,
        max_fulltext_fetches=limits.max_fulltext_fetches,
        max_total_requests=limits.max_total_requests,
    )
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    execution_report = executor.run(graph, literature_references={TARGET_ID: reference})
    target_outcome: TargetAcquisitionOutcome | None = execution_report.outcome_for(TARGET_ID)

    evidence = project_literature_target_reports(ticker, execution_report.target_reports, store)
    chunk_projection = project_literature_chunks(evidence.document_ids, store)

    # Mirrors literature_live_smoke.py's own _collect_transport_diagnostics
    # fallback exactly: a duck-typed test double (e.g. FakeHttpClient) that
    # exposes no .requests_made/.attempts_made/.cache_hits still yields a
    # meaningful logical_request_count from its .requested_urls list,
    # rather than silently reading as 0 requests made.
    deduped_urls = list(dict.fromkeys(getattr(client, "requested_urls", [])))
    logical_request_count = getattr(client, "requests_made", len(deduped_urls))
    physical_attempt_count = getattr(client, "attempts_made", logical_request_count)
    cache_hit_count = getattr(client, "cache_hits", 0)

    coverage_complete = evidence.coverage_complete and chunk_projection.coverage_complete
    unresolved_reasons = tuple(evidence.unresolved_reasons) + tuple(chunk_projection.unresolved_reasons)

    return LiteraturePipelineBundle(
        collection_result=evidence.collection_result,
        chunks=tuple(chunk_projection.chunks),
        document_count=len(evidence.document_ids),
        source_count=len(evidence.collection_result.sources),
        raw_fact_count=len(evidence.collection_result.raw_facts),
        chunk_count=len(chunk_projection.chunks),
        logical_request_count=logical_request_count,
        physical_attempt_count=physical_attempt_count,
        cache_hit_count=cache_hit_count,
        coverage_complete=coverage_complete,
        unresolved_reasons=unresolved_reasons,
        reference_mode=request.reference_mode,
        pmid=request.pmid,
        nct_id=request.nct_id,
        target_outcome=target_outcome.value if target_outcome is not None else None,
        duplicate_document_count=chunk_projection.duplicate_document_count,
        excluded_non_primary_document_count=len(chunk_projection.excluded_non_primary_document_ids),
    )


def bundle_diagnostics(bundle: LiteraturePipelineBundle | None, *, enabled: bool) -> dict[str, Any]:
    """A safe (body/secret/URL-free), JSON-serializable diagnostics dict --
    the ``direct_acquisition_info`` payload ``cli.py`` passes to
    ``Pipeline(...)``. ``bundle is None`` covers both "the feature was
    never enabled" and "no ticker used it this run" -- Phase 4.2A
    correction 1: returns an EMPTY dict in that case (never a dict with
    ``feature_enabled``/token-count keys), so ``cli.py``'s
    ``result_to_json`` can tell "the feature was used" apart from "it
    wasn't" by simple truthiness and omit the ``direct_acquisition_info``
    key entirely for a run that never passed
    ``--document-first-literature`` -- the default-OFF ``--json`` output
    contract this repository had before Phase 4.2A existed. ``enabled`` is
    accepted (and still required, for the non-``None`` shape below) but
    intentionally unused when ``bundle`` is ``None``: in ``cli.py``'s own
    call site, ``bundle is None`` implies ``enabled is False`` always
    (``run_literature_pipeline_acquisition`` never returns ``None``), so
    there is no meaningful ``feature_enabled=True``-with-no-bundle state
    to record."""
    if bundle is None:
        return {}
    return {
        "feature_enabled": enabled,
        "reference_mode": bundle.reference_mode,
        "pmid": bundle.pmid,
        "nct_id": bundle.nct_id,
        "current_program_confirmed": bundle.current_program_confirmed,
        "target_outcome": bundle.target_outcome,
        "refused": bundle.refused,
        "refused_reason": bundle.refused_reason,
        "coverage_complete": bundle.coverage_complete,
        "logical_request_count": bundle.logical_request_count,
        "physical_attempt_count": bundle.physical_attempt_count,
        "cache_hit_count": bundle.cache_hit_count,
        "document_count": bundle.document_count,
        "source_count": bundle.source_count,
        "raw_fact_count": bundle.raw_fact_count,
        "chunk_count": bundle.chunk_count,
        "duplicate_document_count": bundle.duplicate_document_count,
        "excluded_non_primary_document_count": bundle.excluded_non_primary_document_count,
        "unresolved_reasons": list(bundle.unresolved_reasons),
        "anthropic_api_calls": bundle.anthropic_api_calls,
        "web_search_calls": bundle.web_search_calls,
        "external_llm_tokens": bundle.external_llm_tokens,
    }


__all__ = [
    "DEFAULT_MAX_ARTICLES",
    "DEFAULT_MAX_FULLTEXT_FETCHES",
    "TARGETED_MAX_FULLTEXT_FETCHES_CAP",
    "TARGET_ID",
    "LiteraturePipelineBundle",
    "LiteraturePipelineRequest",
    "LiteraturePipelineRequestError",
    "LiteratureReferenceMode",
    "bundle_diagnostics",
    "run_literature_pipeline_acquisition",
    "validate_literature_pipeline_request",
]
