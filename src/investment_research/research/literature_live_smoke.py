"""Phase 3F.1: a manual, human-run Literature (PubMed/Europe PMC) Live Smoke
tool.

**``--nct-id``/``--pmid`` are a LIVE mode**: running either against a real
environment (real ``IRA_NCBI_TOOL``/``IRA_NCBI_EMAIL`` configured, real
network access) WILL make real outbound GET requests to NCBI E-utilities/
Europe PMC. Nothing in this module's code disables that -- there is no
offline/dry-run switch on the live path itself. What IS offline, and
exhaustively so (Phase 3F.1 correction requirement 4, correcting this
docstring's own earlier, misleading wording): this module's automated test
suite (every test in ``tests/unit/test_literature_live_smoke_offline.py``
exercises a ``FakeHttpClient``; every test in
``tests/integration/test_literature_live_smoke_transport.py`` exercises a
local loopback server -- never the real internet, and no test in this
repository has ever invoked the real NCBI/Europe PMC hosts), and
``--analyze-capture`` (below), which makes zero network calls by
construction, always, with no exception. This phase's own author never ran
this tool against the real internet either; that is a fact about how this
code was developed and tested, not a property the CODE itself enforces on
a future operator who runs it for real. Nothing here promotes any step to
``ImplementationStatus.LIVE_VERIFIED`` -- that stays a manual/reporting
decision outside this module, regardless of whether a real run ever
happens.

Deliberately reuses ``sec_live_smoke.AllowlistedHttpClient`` (a purpose-
built, read-only, host-allowlisted GET client with Phase 3F.0.4's
pre-connect redirect validation, an explicit redirect-count cap, a hard
request cap, retry/rate limiting, response caching, and Capture Manifest
writing) rather than duplicating that transport, and reuses
``literature_acquisition_adapter.PubMedLiteratureAdapter``/
``EuropePmcFullTextAdapter`` (Phase 3F/3F.0.1/3F.0.2's own LOCATE/FETCH/
PARSE adapter and request-budget machinery) rather than reimplementing
ESearch/EFetch/Europe-PMC-search/full-text acquisition logic a second
time -- only the CLI surface, credential gating, plan/report shape, and
Capture Manifest replay are new here.

**NEVER imported by ``cli.py``/``pipeline.py``/any production code path.**

Credentials (Phase 3F.1 requirement 3): ``IRA_NCBI_TOOL`` and
``IRA_NCBI_EMAIL`` are REQUIRED for this tool -- unlike
``config.Settings.ncbi_tool``'s own silent non-empty default (used by the
general-purpose ``literature_acquisition_adapter.py``, which this phase
never runs against a live host), this tool refuses outright if either is
unset or blank, mirroring ``sec_live_smoke.py``'s own
``IRA_SEC_USER_AGENT`` placeholder-refusal precedent. ``IRA_NCBI_API_KEY``
is always optional and never causes a refusal. The three VALUES are never
placed in a plan, a report, an exception message, a cache key, a
``Document``, or a Capture Manifest anywhere in this module -- only
whether each is CONFIGURED (a plain boolean) is ever displayed. This is
enforced at the REPORT level, not merely at print time (Phase 3F.1
correction 2 requirement 2): ``_scrub_credentials`` replaces every
configured credential value -- raw and percent-encoded alike -- wherever
one could otherwise land (``requested_urls``, ``http_statuses`` keys,
``errors``, ``refused_reason``), so ``report_to_jsonable(report)`` and
``repr(report)`` stay safe even if a caller injects a transport (e.g. a
test double) that never sanitizes a URL itself; ``format_report_for_print``'s
own ``_safe()`` call remains as defense in depth on top of that, not as
the only line of defense. A refusal (missing credential, malformed ID) or
anything before the first real request is attempted never writes the
one-time LAST_RUN marker.

Request budget (Phase 3F.1 requirement 5): discovery mode's request cap is
an explicit, printed formula -- ``ESearch(1) + EFetch(1, batched) +
EuropePMC search(1 x max_articles) + EuropePMC fulltext
(max_fulltext_fetches)`` -- with defaults ``max_articles=3``,
``max_fulltext_fetches=1``. Targeted (single-PMID) mode skips ESearch
entirely (the adapter's own Priority A ``ct_pmid`` direct path) and additionally
enforces its own small, fixed ceiling on full-text fetches
(``TARGETED_MAX_FULLTEXT_FETCHES_CAP``) independent of whatever
``--max-fulltext-fetches`` a caller passes. A budget overrun is reported by
the adapter's own ``coverage_complete``/``budget_excluded_pmids``/
``budget_skipped_*`` fields -- never as ``NOT_FOUND`` (CLAUDE.md rule 8's
"unsearched is not K0" spirit applied to a request budget: "we didn't get
to it" and "we looked and found nothing" are different statements).

Output (Phase 3F.1 requirement 6): PMID, publication stage, peer-review
status, publication types, NCT-ID match against the requested discovery
ID, Europe PMC OA/full-text-acquired status, and the stored ``Document``'s
own authority/content_kind are all reported. The abstract body and any
acquired full text are NEVER printed to the console or included in
``--json`` output -- only structural metadata. ``ContentKind.FULL_DOCUMENT``
on a full-text-acquired article means only that the text was retrieved,
never that it is peer-reviewed (see ``collectors/literature.py``'s own
module docstring); every RawFact this data COULD build (this module never
builds one -- see ``PubMedLiteratureAdapter._parse``'s own docstring) would
carry ``EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION``, excluded from
``DECISION_GRADE_CLASSES``, with ``independent_confirmation`` staying
``False``. ``anthropic_api_calls``/``web_search_calls``/
``external_llm_tokens`` are always ``0`` -- this module makes none of
those calls.

Capture Manifest replay (Phase 3F.1 requirement 7, corrected in the Phase
3F.1 correction pass): ``AllowlistedHttpClient._save_response`` now names
each saved file's digest from the SANITIZED public URL
(``sha256(public_url)[:24]``), never the wire URL that actually carried
``email``/``api_key`` -- so, unlike before that correction,
``analyze_capture`` COULD in principle recompute an expected digest
directly. It still scans every ``*.manifest.json`` actually present in the
capture directory instead, because that remains simpler and more robust
than reconstructing an exact query-parameter set/order to match against
(and because ``analyze_capture`` must work with ZERO credentials
configured, which a from-scratch URL reconstruction would only
coincidentally need anyway now that the digest excludes secrets).
Classification is by each manifest's own (already-sanitized)
``requested_url`` shape. Every manifest is resolved through the SHARED
``capture_manifest.read_manifest`` (never a reimplementation), so
``VERIFIED``/``MISSING``/``BODY_MISSING``/``MALFORMED``/
``UNSUPPORTED_SCHEMA``/``HASH_MISMATCH``/``CONTENT_LENGTH_MISMATCH``/
``IO_ERROR`` are resolved identically to every other Live Smoke tool, and
every non-``VERIFIED``, non-``MISSING`` status is surfaced explicitly as an
Evidence Integrity failure.

Loopback verification scope (Phase 3F.1 correction requirement 8): a real
loopback-server test drives ``run_live_smoke()``'s FULL orchestration
(LOCATE via ESearch -> FETCH via EFetch + Europe PMC search/full-text ->
PARSE) end-to-end, by monkeypatching ``literature_acquisition_adapter``'s
module-level URL/``ALLOWED_HOSTS`` constants for the duration of that one
test -- Python resolves a module global at call time, so this requires no
production code change or DI seam in that module, which still hardcodes
its real NCBI/Europe PMC hostnames for every other caller. Every other
loopback test in this suite exercises ``AllowlistedHttpClient`` and the
Capture Manifest write/replay cycle directly, without going through
``PubMedLiteratureAdapter`` at all.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..collectors.http import DEFAULT_MAX_REDIRECTS
from ..collectors.literature import (
    ParsedPubmedArticle,
    check_pubmed_xml_shape,
    parse_europepmc_fulltext_xml,
    parse_europepmc_search_response,
    parse_pubmed_articleset,
)
from ..schemas.enums import FetchOutcome
from . import ncbi_credentials as _ncbi_credentials
from .acquisition_executor import AcquisitionExecutor, ExecutionDiagnostics
from .capture_manifest import ManifestReadStatus, read_manifest
from .clinicaltrials_acquisition_adapter import NCT_ID_RE
from .clinicaltrials_live_smoke import normalize_nct_id
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
from .sec_live_smoke import AllowlistedHttpClient, _safe
from .source_routing import SourceRoutingGraph, StepStatus

DEFAULT_TIMEOUT_SECONDS = 20.0
#: Deliberately far below NCBI's own documented 3 req/s no-API-key cap, and
#: kept the SAME regardless of whether an API key is configured (Phase
#: 3F.1 requirement 4: an API key raises NCBI's OWN cap to 10 req/s -- this
#: tool never uses that headroom; it keeps its own low rate limit either
#: way).
DEFAULT_RATE_LIMIT_RPS = 0.5
MAX_RETRIES = 1
#: Phase 3F.0.4's pre-connect redirect validation cap, reused as-is.
MAX_REDIRECTS = DEFAULT_MAX_REDIRECTS

DEFAULT_MAX_ARTICLES = 3
DEFAULT_MAX_FULLTEXT_FETCHES = 1
#: Targeted (single-PMID) mode's own hard ceiling on full-text fetches,
#: independent of whatever ``--max-fulltext-fetches`` a caller passes
#: (Phase 3F.1 requirement 5: "targeted PMID also carries a small, fixed
#: cap"). A caller may ask for FEWER (a smaller --max-fulltext-fetches
#: still applies); never more.
TARGETED_MAX_FULLTEXT_FETCHES_CAP = 1

#: Never a secret -- a fixed, generic software identifier for this tool's
#: own HTTP User-Agent header, unrelated to NCBI's own courtesy-
#: identification query parameters (tool/email/api_key) this module
#: separately gates via IRA_NCBI_TOOL/IRA_NCBI_EMAIL/IRA_NCBI_API_KEY.
LITERATURE_LIVE_SMOKE_USER_AGENT = "investment-research-agent-literature-live-smoke/1.0"

#: NCBI PMIDs are plain decimal integers; generous upper bound on digit
#: count, never a guess at a real PMID's current maximum length.
PMID_RE = re.compile(r"^\d{1,9}$")

#: Phase 4.2A extraction: credential resolution moved to
#: ``ncbi_credentials.py`` (shared with the production Literature Pipeline
#: Integration) -- re-exported here under their original names so every
#: existing caller/test of THIS module (``live.resolve_ncbi_credentials``,
#: ``live.NcbiCredentials``, etc.) keeps working unchanged. Behavior is
#: byte-identical to before the extraction; only the implementation moved.
NCBI_TOOL_ENV_VAR = _ncbi_credentials.NCBI_TOOL_ENV_VAR
NCBI_EMAIL_ENV_VAR = _ncbi_credentials.NCBI_EMAIL_ENV_VAR
NCBI_API_KEY_ENV_VAR = _ncbi_credentials.NCBI_API_KEY_ENV_VAR
NcbiCredentials = _ncbi_credentials.NcbiCredentials
CredentialStatus = _ncbi_credentials.CredentialStatus
resolve_ncbi_credentials = _ncbi_credentials.resolve_ncbi_credentials


def _discovery_budget(max_articles: int, max_fulltext_fetches: int) -> tuple[RequestBudgetLimits, int, str]:
    """Phase 3F.1 requirement 5's explicit formula: ``ESearch(1) +
    EFetch(1, batched) + EuropePMC search(1 x max_articles) + EuropePMC
    fulltext(max_fulltext_fetches)``. ``max_search_requests`` covers BOTH
    ESearch and Europe PMC search, since ``RequestBudgetUsage.reserve_search``
    shares one budget between them (see ``literature_acquisition_adapter.py``)."""
    max_search_requests = 1 + max_articles
    max_total_requests = 1 + 1 + max_articles + max_fulltext_fetches
    formula = (
        f"ESearch(1) + EFetch(1, batched) + EuropePMC search(1 x max_articles={max_articles}) "
        f"+ EuropePMC fulltext(max_fulltext_fetches={max_fulltext_fetches}) = {max_total_requests}"
    )
    limits = RequestBudgetLimits(
        max_search_requests=max_search_requests, max_articles=max_articles,
        max_fulltext_fetches=max_fulltext_fetches, max_total_requests=max_total_requests,
    )
    return limits, max_total_requests, formula


def _effective_max_fulltext_fetches(mode: str, max_fulltext_fetches: int) -> int:
    """The ACTUAL ``max_fulltext_fetches`` value that will govern
    execution -- the single source of truth every caller (``build_plan``,
    ``run_live_smoke``) must derive from, so a caller-displayed plan and
    the plan actually executed can never silently disagree (Phase 3F.1
    correction requirement 7). Targeted mode clamps to
    ``TARGETED_MAX_FULLTEXT_FETCHES_CAP`` regardless of what was requested;
    discovery mode never clamps."""
    if mode == "targeted":
        return min(max_fulltext_fetches, TARGETED_MAX_FULLTEXT_FETCHES_CAP)
    return max_fulltext_fetches


def _targeted_budget(max_fulltext_fetches: int) -> tuple[RequestBudgetLimits, int, str]:
    """A single PMID (Priority A ``ct_pmid`` direct path) skips ESearch
    entirely. ``max_fulltext_fetches`` is clamped to
    ``TARGETED_MAX_FULLTEXT_FETCHES_CAP`` regardless of what the caller
    requested (Phase 3F.1 requirement 5)."""
    capped = min(max_fulltext_fetches, TARGETED_MAX_FULLTEXT_FETCHES_CAP)
    max_search_requests = 1  # Europe PMC search only -- no ESearch in targeted mode.
    max_total_requests = 1 + 1 + capped  # EFetch(1) + EuropePMC search(1) + fulltext(<=capped)
    formula = (
        f"EFetch(1) + EuropePMC search(1) + EuropePMC fulltext(max_fulltext_fetches={capped}, "
        f"hard cap {TARGETED_MAX_FULLTEXT_FETCHES_CAP} for targeted mode) = {max_total_requests}"
    )
    limits = RequestBudgetLimits(
        max_search_requests=max_search_requests, max_articles=1,
        max_fulltext_fetches=capped, max_total_requests=max_total_requests,
    )
    return limits, max_total_requests, formula


@dataclass(frozen=True)
class LiveSmokePlan:
    """What this run intends to do -- printed BEFORE any network call."""

    mode: str  # "discovery" | "targeted"
    allowed_hosts: tuple[str, ...]
    max_requests: int
    max_requests_formula: str
    timeout_seconds: float
    max_retries: int
    max_redirects: int
    rate_limit_rps: float
    nct_id: str | None
    pmid: str | None
    max_articles: int
    max_fulltext_fetches: int
    #: Booleans ONLY -- never the underlying credential values.
    ncbi_tool_configured: bool
    ncbi_email_configured: bool
    ncbi_api_key_configured: bool


class LiveSmokeStatus(str, Enum):
    """Overall run outcome -- mirrors ``schemas.enums.FetchOutcome``'s own
    naming convention (BLOCKED/RATE_LIMITED/...) at the whole-run level
    rather than the single-request level, since no existing enum in this
    repository already models "how did an entire Live Smoke run go"
    (Phase 3F.1 correction requirement 6). Priority when computing this
    from a report, most severe first: REFUSED > BLOCKED > RATE_LIMITED >
    FAILED > COMPLETED."""

    #: The run reached the point of making (or attempting) real requests,
    #: and every step that ran completed without a FAILED/NOT_FOUND/
    #: ZERO_RESULTS/SKIPPED_DUE_TO_BUDGET outcome. Note this is compatible
    #: with coverage_complete=False from a budget skip that never itself
    #: produced a step-level failure entry.
    COMPLETED = "COMPLETED"
    #: At least one request in this run was rate-limited (HTTP 429) --
    #: never auto-retried (see AllowlistedHttpClient), reported here as its
    #: own status rather than folded into the generic FAILED.
    RATE_LIMITED = "RATE_LIMITED"
    #: At least one request in this run was blocked -- a disallowed host,
    #: a rejected redirect, or an egress policy denial (403/407).
    BLOCKED = "BLOCKED"
    #: The run made (or attempted) real requests, and at least one step
    #: failed for a reason other than rate-limiting/blocking.
    FAILED = "FAILED"
    #: Refused before any request was ever attempted (missing credential
    #: or a malformed NCT ID/PMID). Distinct from FAILED: zero network
    #: activity occurred.
    REFUSED = "REFUSED"


@dataclass
class LiveSmokeReport:
    """Everything to report after the run -- diagnostics, never an
    investment conclusion, and never an abstract/full-text body."""

    plan: LiveSmokePlan
    refused_reason: str | None = None
    requested_urls: list[str] = field(default_factory=list)
    http_statuses: dict[str, int | None] = field(default_factory=dict)
    #: LOGICAL request count -- one per distinct URL actually dispatched
    #: (cache misses past the host-allowlist check), regardless of how many
    #: physical retries each one took. See ``attempt_count`` for the
    #: physical count (Phase 3F.1 correction requirement 3/6).
    request_count: int = 0
    #: PHYSICAL GET-attempt count -- every real socket-level attempt,
    #: including every retry of every logical request. This is the number
    #: ``AllowlistedHttpClient.max_requests`` actually bounds.
    attempt_count: int = 0
    cache_hit_count: int = 0
    #: Structured per-request outcomes (``FetchOutcome`` values, e.g.
    #: "BLOCKED"/"RATE_LIMITED"/"OK") collected from the transport's own
    #: cache -- what ``status`` (below) is actually computed from (Phase
    #: 3F.1 correction 2 requirement 3). Never inferred from matching a
    #: substring against ``errors``' free text, which a provider's own error
    #: message could coincidentally contain (e.g. an EFetch error whose
    #: text happens to mention "blocked" for an unrelated reason).
    transport_outcomes: list[str] = field(default_factory=list)
    #: True once run_live_smoke reached the point of actually building an
    #: execution report (i.e. NOT refused before the first attempt) --
    #: distinct from ``status == COMPLETED``, which additionally requires
    #: no errors. A plain STORED field, not a ``execution_report is not
    #: None`` computed property (Phase 3F.1 correction 3 requirement 1):
    #: the raw ``ExecutionReport`` itself is never held anywhere on this
    #: object at all, so there is nothing left for that check to read.
    #: ``run_live_smoke`` sets this exactly once, right after
    #: ``executor.run()`` returns successfully; it never resets to
    #: ``False`` afterward on any code path.
    smoke_run_completed: bool = False
    #: The run's own diagnostics, curated down to ONLY
    #: ``acquisition_executor.ExecutionDiagnostics`` -- an all-``int``
    #: dataclass with no string fields, so it can never carry a credential
    #: value or document body regardless of what the raw ``ExecutionReport``
    #: it was read from contained. This is deliberately the ONLY thing this
    #: module keeps from that raw ``ExecutionReport`` -- everything else
    #: (``StepExecutionResult.failure_reason``/``.payload``, ``Document``,
    #: abstract/full-text bodies, ``EuropePmcSearchResult``) is read from a
    #: LOCAL variable inside ``run_live_smoke`` and extracted into this
    #: report's other, already-curated fields (``errors``, ``articles``,
    #: ``pmids_located``, ...) -- never stored on this object itself (Phase
    #: 3F.1 correction 3 requirement 1: the raw ``ExecutionReport`` must not
    #: merely be hidden from ``repr()``, it must not be held here at all).
    diagnostics: ExecutionDiagnostics | None = None
    pmids_located: list[str] = field(default_factory=list)
    #: One curated, JSON-safe dict per article -- metadata/diagnostics
    #: only; abstract/full-text body text is never included here.
    articles: list[dict[str, Any]] = field(default_factory=list)
    coverage_complete: bool = True
    budget_excluded_pmids: list[str] = field(default_factory=list)
    budget_skipped_europepmc_search_pmids: list[str] = field(default_factory=list)
    budget_skipped_fulltext_pmcids: list[str] = field(default_factory=list)
    #: A GENUINE (non-budget) Europe PMC search/fulltext-fetch/fulltext-
    #: parse failure -- structured (never free text a caller must
    #: substring-match), sanitized, one entry per failure. Distinct from
    #: the two ``budget_skipped_*`` fields above: those mean "we chose not
    #: to ask"; this means "we asked and the provider said no" (Phase
    #: 3F.2 correction requirement 3 -- a real Live Smoke run against PMID
    #: 20668659 found this exact case: Europe PMC search succeeded
    #: (isOpenAccess=true, inEPMC=true) but the fullTextXML fetch 404'd,
    #: and that 404 was previously never recorded anywhere in this report
    #: at all). Each entry: ``{"pmid": str, "pmcid": str | None, "stage":
    #: "search" | "fulltext_fetch" | "fulltext_parse", "kind": str,
    #: "reason": str}`` -- ``kind`` is a bare ``FetchOutcome`` value or one
    #: of ``literature_acquisition_adapter.FAILURE_KIND_*``, never inferred
    #: from ``reason``'s free text.
    europepmc_failures: list[dict[str, Any]] = field(default_factory=list)
    #: Every entry reflects a step that ACTUALLY made (or, for a targeted
    #: ct_pmid PMID, genuinely did not need to make) a real network call
    #: this run -- never inferred from a target reaching ACQUIRED alone
    #: (Phase 3F.2 correction requirement 4: targeted mode's own LOCATE
    #: resolves synthetically, zero requests, and must never appear here;
    #: a Europe PMC fulltext 404 -- this run's own real-world finding --
    #: must never appear as a fulltext-success candidate either). Still
    #: diagnostic only: this phase never promotes anything, and this
    #: module's own internal graph (``_build_graph``) marks every step
    #: OFFLINE_VERIFIED, never LIVE_VERIFIED, regardless of what this list
    #: contains.
    live_verified_candidates: list[str] = field(default_factory=list)
    anthropic_api_calls: int = 0
    web_search_calls: int = 0
    external_llm_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.refused_reason is not None

    @property
    def status(self) -> LiveSmokeStatus:
        """Computed on every access from ``refused_reason``/``errors`` --
        never a separately-stored field a code path could forget to set
        (Phase 3F.1 correction requirement 6). See ``_compute_status``."""
        return _compute_status(self)


def build_plan(
    *,
    mode: str,
    nct_id: str | None,
    pmid: str | None,
    max_articles: int,
    max_fulltext_fetches: int,
    status: CredentialStatus,
) -> LiveSmokePlan:
    """``max_fulltext_fetches`` is clamped HERE via
    ``_effective_max_fulltext_fetches`` before it ever reaches either the
    budget formula or the returned plan's own field -- so a caller (e.g.
    ``main()``, printing a plan before running) that passes the raw,
    un-clamped CLI value always gets back a plan whose displayed
    ``max_fulltext_fetches`` already matches what ``run_live_smoke`` will
    actually execute (Phase 3F.1 correction requirement 7)."""
    effective_max_fulltext_fetches = _effective_max_fulltext_fetches(mode, max_fulltext_fetches)
    if mode == "discovery":
        _limits, max_requests, formula = _discovery_budget(max_articles, effective_max_fulltext_fetches)
    else:
        _limits, max_requests, formula = _targeted_budget(effective_max_fulltext_fetches)
    return LiveSmokePlan(
        mode=mode,
        allowed_hosts=tuple(sorted(ALLOWED_HOSTS)),
        max_requests=max_requests,
        max_requests_formula=formula,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        max_redirects=MAX_REDIRECTS,
        rate_limit_rps=DEFAULT_RATE_LIMIT_RPS,
        nct_id=nct_id,
        pmid=pmid,
        max_articles=max_articles,
        max_fulltext_fetches=effective_max_fulltext_fetches,
        ncbi_tool_configured=status.tool_configured,
        ncbi_email_configured=status.email_configured,
        ncbi_api_key_configured=status.api_key_configured,
    )


def _build_graph(target_id: str = "target_lit_smoke") -> SourceRoutingGraph:
    """Delegates to the ONE shared single-target LOCATE->FETCH->PARSE graph
    definition (Phase 4.2A extraction) -- see
    ``literature_acquisition_adapter.build_single_target_literature_graph``'s
    own docstring. ``requirement_id``/``legacy_need_id`` are pinned to this
    function's own pre-existing values so this remains a byte-identical
    graph to the one this module built before the extraction."""
    return build_single_target_literature_graph(
        target_id, requirement_id="req_lit_smoke", legacy_need_id="live_smoke_literature"
    )


def _strip_ncbi_tool_param(url: str) -> str:
    """This Live Smoke tool's OWN, STRICTER policy than the shared
    ``AllowlistedHttpClient``/``_sanitize_url`` convention: that shared
    convention deliberately keeps a ``tool=`` query parameter (it is a
    non-secret software identifier -- see ``config.Settings.secret_values``'s
    own exclusion of it), because most callers have no reason to hide it.
    Literature Live Smoke is stricter regardless: it never displays or
    stores the configured ``IRA_NCBI_TOOL`` VALUE anywhere, even though it
    is "not as secret as an API key" (Phase 3F.1 correction requirement 2).
    A no-op for a URL that carries no ``tool=`` parameter."""
    if "tool=" not in url:
        return url
    prefix, _, query = url.partition("?")
    kept = [pair for pair in query.split("&") if pair and not pair.startswith("tool=")]
    return prefix + ("?" + "&".join(kept) if kept else "")


#: Placeholder substituted for a matched credential value -- distinct from
#: sec_live_smoke.py's generic REDACTED marker so a grep for either can
#: tell which layer caught it, though both mean the same thing: a secret
#: was found and replaced before this text left this module.
_CREDENTIAL_REDACTED = "[REDACTED-NCBI-CREDENTIAL]"


def _credential_secret_values(credentials: NcbiCredentials) -> list[str]:
    """Every non-empty credential VALUE, plus its percent-encoded form (the
    two shapes a secret can appear in inside a URL or an exception message
    that embedded one) -- the input ``_scrub_credentials`` scrubs against.
    An unset ``ncbi_api_key`` (empty string) is skipped entirely: an empty
    "secret" would otherwise match (and mangle) every string, since ``""
    in text`` is always true (Phase 3F.1 correction 2 requirement 2)."""
    values: list[str] = []
    for raw in (credentials.ncbi_tool, credentials.ncbi_email, credentials.ncbi_api_key):
        if not raw:
            continue
        values.append(raw)
        encoded = quote(raw, safe="")
        if encoded != raw:
            values.append(encoded)
    return values


def _scrub_credentials(text: str, secret_values: list[str]) -> str:
    """Value-based redaction -- catches a credential VALUE (raw or
    percent-encoded) wherever it appears in ``text``, regardless of which
    query parameter NAME carried it or whether the transport that produced
    ``text`` sanitized anything itself. This is what closes the leak
    ``_strip_ncbi_tool_param`` (a param-NAME-based strip, tool-only) could
    never close: an unsanitized test double's raw wire URL, or an
    adapter's own exception-message text embedding one (Phase 3F.1
    correction 2 requirement 2)."""
    for value in secret_values:
        if value and value in text:
            text = text.replace(value, _CREDENTIAL_REDACTED)
    return text


def _collect_transport_diagnostics(client: Any, report: LiveSmokeReport, secret_values: list[str]) -> None:
    """Same generic, getattr-based approach as sec_live_smoke.py/
    clinicaltrials_live_smoke.py's own helper -- degrades cleanly for a
    minimal duck-typed fake transport. ``client.requested_urls`` is
    expected to already be the sanitized public-URL form (Phase 3F.0.3)
    with respect to email/api_key for the REAL ``AllowlistedHttpClient``,
    but this function must never ASSUME that of every possible duck-typed
    transport a caller might inject (Phase 3F.1 correction 2 requirement
    2): ``_strip_ncbi_tool_param`` (param-name-based) and
    ``_scrub_credentials`` (value-based, covers email/api_key/tool alike,
    raw and percent-encoded) are BOTH applied to every URL this function
    ever writes into the report, regardless of transport.

    Also collects ``report.transport_outcomes`` -- each cached
    ``FetchResult``'s OWN ``.outcome`` value (Phase 3F.1 correction 2
    requirement 3), which ``_compute_status`` uses instead of pattern-
    matching ``report.errors``' free text.
    """

    def sanitize(url: str) -> str:
        return _scrub_credentials(_strip_ncbi_tool_param(url), secret_values)

    seen = [sanitize(u) for u in getattr(client, "requested_urls", report.requested_urls)]
    deduped = list(dict.fromkeys(seen))
    report.requested_urls = deduped
    report.request_count = getattr(client, "requests_made", len(deduped))
    report.attempt_count = getattr(client, "attempts_made", report.request_count)
    report.cache_hit_count = getattr(client, "cache_hits", 0)
    cache = getattr(client, "_cache", None)
    outcomes: list[str] = []
    if isinstance(cache, dict):
        for result in cache.values():
            result_url = getattr(result, "url", None)
            if isinstance(result_url, str):
                report.http_statuses[sanitize(result_url)] = getattr(result, "status", None)
            outcome = getattr(result, "outcome", None)
            if outcome is not None:
                outcomes.append(getattr(outcome, "value", str(outcome)))
    report.transport_outcomes = outcomes
    for url in report.requested_urls:
        report.http_statuses.setdefault(url, None)


def _compute_status(report: LiveSmokeReport) -> LiveSmokeStatus:
    """Derives the overall run status from STRUCTURED transport outcomes,
    never from pattern-matching ``report.errors``' free text (Phase 3F.1
    correction 2 requirement 3): a provider failure message that happens
    to contain the word "blocked" or "rate limited" for an unrelated
    reason must never misclassify the run. The specific provider-failure
    reason stays exactly where it always was -- ``report.errors`` -- this
    function only ever reads ``report.transport_outcomes``."""
    if report.refused:
        return LiveSmokeStatus.REFUSED
    if FetchOutcome.BLOCKED.value in report.transport_outcomes:
        return LiveSmokeStatus.BLOCKED
    if FetchOutcome.RATE_LIMITED.value in report.transport_outcomes:
        return LiveSmokeStatus.RATE_LIMITED
    if report.errors:
        return LiveSmokeStatus.FAILED
    return LiveSmokeStatus.COMPLETED


def run_live_smoke(
    *,
    mode: str,
    nct_id: str | None = None,
    pmid: str | None = None,
    max_articles: int = DEFAULT_MAX_ARTICLES,
    max_fulltext_fetches: int = DEFAULT_MAX_FULLTEXT_FETCHES,
    http_client: Any | None = None,
    out_dir: Path | None = None,
    env: dict[str, str] | None = None,
    on_first_attempt: Callable[[], None] | None = None,
) -> LiveSmokeReport:
    """Run one Literature Live Smoke pass. ``http_client``/``env`` are
    injectable for offline testing only -- production callers (``scripts/
    literature_live_smoke.py``) leave both as ``None``, in which case this
    resolves the real environment and constructs a real, host-allowlisted
    ``AllowlistedHttpClient`` and WILL make real outbound NCBI/Europe PMC
    requests (see module docstring's LIVE mode notice) if credentials are
    configured and no refusal applies.

    ``on_first_attempt``, when given, is threaded straight into the
    internally-constructed ``AllowlistedHttpClient`` (see that class's own
    docstring) -- it fires exactly once, immediately before this run's
    FIRST real socket-level attempt, never for a refusal or any other
    pre-send failure. ``main()`` uses this to tie the one-time LAST_RUN
    marker to the moment real communication actually begins, rather than
    inferring it from a broad try/except around this whole function (Phase
    3F.1 correction 2 requirement 1). Ignored when ``http_client`` is
    injected -- a caller providing its own transport also controls that
    transport's own ``on_first_attempt`` directly, if it has one.

    Never called from ``Pipeline.run()``; never makes an Anthropic/LLM/Web
    Search call.
    """
    if mode not in ("discovery", "targeted"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'discovery' or 'targeted')")

    credentials, credential_status, missing = resolve_ncbi_credentials(env)
    secret_values = _credential_secret_values(credentials)
    normalized_nct_id = normalize_nct_id(nct_id) if nct_id else None
    normalized_pmid = (pmid or "").strip()
    effective_max_fulltext_fetches = _effective_max_fulltext_fetches(mode, max_fulltext_fetches)
    plan = build_plan(
        mode=mode, nct_id=normalized_nct_id, pmid=normalized_pmid or None, max_articles=max_articles,
        max_fulltext_fetches=effective_max_fulltext_fetches, status=credential_status,
    )
    report = LiveSmokeReport(plan=plan)

    # Credential check FIRST -- before validating the ID -- so a missing
    # env var is never masked by an unrelated ID-format refusal (Phase
    # 3F.1 requirement 3). Nothing has been constructed yet that could ever
    # make a request.
    if missing:
        report.refused_reason = _scrub_credentials(
            f"missing required environment variable(s): {', '.join(missing)} -- refusing to make "
            f"any request ({NCBI_API_KEY_ENV_VAR} remains optional)",
            secret_values,
        )
        return report

    if mode == "discovery":
        if not normalized_nct_id or not NCT_ID_RE.match(normalized_nct_id):
            report.refused_reason = _scrub_credentials(
                f"malformed or missing NCT ID for discovery mode: {nct_id!r}", secret_values
            )
            return report
        limits, _, _ = _discovery_budget(max_articles, effective_max_fulltext_fetches)
        # Deliberately decoupled from max_articles: ESearch's own retmax is
        # a generous discovery ceiling (a single ESearch call's cost is the
        # same regardless of retmax -- the printed request-count formula
        # only ever counts it once), so a search finding MORE candidates
        # than the processing budget allows surfaces as a genuine
        # coverage_complete=False / budget_excluded_pmids report, never as
        # a silently-truncated "only N results" ESearch response (Phase
        # 3F.1 requirement 5).
        reference = LiteratureReference(nct_id=normalized_nct_id, max_pmids=MAX_PMIDS_PER_BATCH)
    else:
        if not normalized_pmid or not PMID_RE.match(normalized_pmid):
            report.refused_reason = _scrub_credentials(
                f"malformed or missing PMID for targeted mode: {pmid!r}", secret_values
            )
            return report
        limits, _, _ = _targeted_budget(effective_max_fulltext_fetches)
        reference = LiteratureReference(ct_pmid=normalized_pmid, max_pmids=1)

    client = http_client if http_client is not None else AllowlistedHttpClient(
        user_agent=LITERATURE_LIVE_SMOKE_USER_AGENT, allowed_hosts=ALLOWED_HOSTS,
        timeout=DEFAULT_TIMEOUT_SECONDS, rate_limit_rps=DEFAULT_RATE_LIMIT_RPS, max_retries=MAX_RETRIES,
        max_requests=plan.max_requests, max_redirects=MAX_REDIRECTS, out_dir=out_dir, source="literature",
        # This module's own stricter policy: never display or store the
        # configured IRA_NCBI_TOOL value either, even though `tool` is not
        # as secret as an API key and AllowlistedHttpClient's shared
        # _sanitize_url deliberately keeps it for other callers (Phase
        # 3F.1 correction requirement 2). This is what keeps `tool=...`
        # out of the persisted Capture Manifest itself, not just this
        # module's own in-memory diagnostics.
        additional_secret_query_params=frozenset({"tool"}),
        on_first_attempt=on_first_attempt,
        # Literature's own opt-in (Phase 3F.2 correction requirement 5):
        # SEC/ClinicalTrials/Form4 all keep the shared client's default
        # (False) unchanged. A failed-but-responded Europe PMC/NCBI
        # request (404/429/5xx) is still worth an offline-auditable
        # Capture Manifest -- this run's own real finding (a genuine
        # fullTextXML 404) would otherwise leave no on-disk trace at all.
        save_failed_responses=True,
    )

    graph = _build_graph()
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(
        client, credentials,
        max_search_requests=limits.max_search_requests, max_articles=limits.max_articles,
        max_fulltext_fetches=limits.max_fulltext_fetches, max_total_requests=limits.max_total_requests,
    )
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    # execution_report is a LOCAL variable ONLY, for the rest of this
    # function's own curation logic below -- it is never assigned onto
    # `report` (Phase 3F.1 correction 3 requirement 1). Everything this
    # report needs FROM it is extracted into report's own, already-curated
    # fields (errors/articles/pmids_located/... below, diagnostics here);
    # the raw StepExecutionResult/Document/EuropePmcSearchResult objects
    # it carries never survive past this function's own stack frame.
    execution_report = executor.run(graph, literature_references={"target_lit_smoke": reference})
    report.smoke_run_completed = True
    report.diagnostics = execution_report.diagnostics
    _collect_transport_diagnostics(client, report, secret_values)

    for target_report in execution_report.target_reports:
        for step_result in target_report.step_results:
            payload = step_result.payload or {}
            if step_result.step_id == "l1":
                report.pmids_located = list(payload.get("pmids", []))
            if "budget_excluded_pmids" in payload:
                report.budget_excluded_pmids = list(payload["budget_excluded_pmids"])
            if "budget_skipped_europepmc_search_pmids" in payload:
                report.budget_skipped_europepmc_search_pmids = list(
                    payload["budget_skipped_europepmc_search_pmids"]
                )
            if "budget_skipped_fulltext_pmcids" in payload:
                report.budget_skipped_fulltext_pmcids = list(payload["budget_skipped_fulltext_pmcids"])
            if "coverage_complete" in payload:
                report.coverage_complete = payload["coverage_complete"]
            if step_result.step_id == "p":
                for doc in payload.get("parsed_documents", []):
                    article: ParsedPubmedArticle = doc["article"]
                    doc_id = doc.get("document_id")
                    stored = store.get(doc_id) if doc_id else None
                    doc_nct_ids = list(doc.get("nct_ids", []))
                    report.articles.append(
                        {
                            "pmid": doc["pmid"],
                            "publication_stage": article.publication_stage.value,
                            "peer_review_status": article.peer_review_status.value,
                            "publication_types": list(article.publication_types),
                            "nct_ids": doc_nct_ids,
                            "matches_requested_nct_id": (
                                normalized_nct_id in doc_nct_ids if normalized_nct_id else None
                            ),
                            "retracted": bool(doc.get("retracted", False)),
                            "is_correction_or_erratum": bool(doc.get("is_correction_or_erratum", False)),
                            "europepmc_search_succeeded": bool(doc.get("europepmc_search_succeeded", False)),
                            "europepmc_is_open_access": bool(doc.get("europepmc_is_open_access", False)),
                            "europepmc_in_epmc": bool(doc.get("europepmc_in_epmc", False)),
                            "full_text_acquired": bool(doc.get("full_text_acquired", False)),
                            "document_id": doc_id,
                            "document_authority": stored.document.authority.value if stored else None,
                            "document_content_kind": stored.document.content_kind.value if stored else None,
                        }
                    )
                # Only ever processed for "p" (the final step): "f"'s own
                # payload carries the SAME two dicts (via **upstream
                # passthrough), so extracting them at every step_result
                # would double-report each failure. Every reachable state
                # where these dicts are non-empty already implies "f"
                # reached BODY_FETCHED (a genuine PubMed EFetch success --
                # these Europe PMC calls only ever happen AFTER that), so
                # "p" always runs too (Phase 3F.2 correction requirement 3).
                for pmid, info in (payload.get("europepmc_search_failures") or {}).items():
                    reason = _scrub_credentials(str(info.get("reason", "")), secret_values)
                    report.europepmc_failures.append(
                        {
                            "pmid": pmid, "pmcid": None, "stage": info.get("stage", "search"),
                            "kind": info.get("kind", "UNKNOWN"), "reason": reason,
                        }
                    )
                    report.errors.append(
                        _scrub_credentials(f"europepmc search failed for PMID {pmid}: {reason}", secret_values)
                    )
                for pmid, info in (payload.get("europepmc_fulltext_failures") or {}).items():
                    reason = _scrub_credentials(str(info.get("reason", "")), secret_values)
                    stage = info.get("stage", "fulltext_fetch")
                    pmcid = info.get("pmcid")
                    report.europepmc_failures.append(
                        {"pmid": pmid, "pmcid": pmcid, "stage": stage, "kind": info.get("kind", "UNKNOWN"), "reason": reason}
                    )
                    report.errors.append(
                        _scrub_credentials(
                            f"europepmc {stage} failed for PMID {pmid} (PMCID {pmcid}): {reason}", secret_values
                        )
                    )
            if step_result.status in (
                StepStatus.FAILED, StepStatus.NOT_FOUND, StepStatus.ZERO_RESULTS, StepStatus.SKIPPED_DUE_TO_BUDGET,
            ):
                report.errors.append(
                    _scrub_credentials(
                        f"{step_result.step_id}: {step_result.status} -- {step_result.failure_reason}",
                        secret_values,
                    )
                )

    # Every candidate below reflects a step that ACTUALLY made a real
    # network call this run (Phase 3F.2 correction requirement 4) -- never
    # derived from a target's overall ACQUIRED outcome, which is agnostic
    # to whether a given step needed the network at all (targeted mode's
    # own LOCATE) or whether a bundled sub-operation inside FETCH/PARSE
    # (Europe PMC search/fulltext) genuinely succeeded.
    for target_report in execution_report.target_reports:
        step_by_id = {r.step_id: r for r in target_report.step_results}
        l1 = step_by_id.get("l1")
        if l1 is not None and l1.status is StepStatus.URL_RESOLVED and l1.http_requests_made > 0:
            # A real NCBI ESearch call (discovery/alias mode, priority B/C)
            # -- never targeted ct_pmid mode's priority A, which resolves
            # synthetically with zero requests.
            report.live_verified_candidates.append(f"{target_report.target_id}:l1:pubmed_esearch")
        f = step_by_id.get("f")
        if f is not None and f.status is StepStatus.BODY_FETCHED:
            # BODY_FETCHED requires the batched PubMed EFetch to have
            # succeeded -- true regardless of Europe PMC's own outcome, so
            # this candidate is PubMed-scoped only; it is never extended to
            # imply Europe PMC search/fulltext also succeeded.
            report.live_verified_candidates.append(f"{target_report.target_id}:f:pubmed_efetch")
        p = step_by_id.get("p")
        if p is not None and p.status is StepStatus.PARSED:
            report.live_verified_candidates.append(f"{target_report.target_id}:p:pubmed_parse")

    target_id = "target_lit_smoke"
    for article_entry in report.articles:
        pmid = article_entry.get("pmid")
        if article_entry.get("europepmc_search_succeeded"):
            report.live_verified_candidates.append(f"{target_id}:europepmc_search:{pmid}")
        if article_entry.get("full_text_acquired"):
            # Only ever True when fetch_fulltext_document_id was actually
            # set -- i.e. Europe PMC's fullTextXML fetch AND parse both
            # genuinely succeeded this run. A 404/429/5xx/timeout/
            # connection-error/malformed-body result (this run's own real
            # finding, PMID 20668659 -> PMC2910600 -> 404) never reaches
            # this branch (Phase 3F.2 correction requirement 4).
            report.live_verified_candidates.append(f"{target_id}:europepmc_fulltext:{pmid}")
    return report


def format_plan_for_print(plan: LiveSmokePlan) -> str:
    return "\n".join(
        [
            "=== Literature (PubMed/Europe PMC) Live Smoke plan (nothing sent yet) ===",
            f"mode: {plan.mode} -- LIVE mode: running this will make real NCBI/Europe PMC network "
            "requests once credentials are configured and no refusal applies (--analyze-capture is "
            "the only fully offline mode).",
            f"allowed_hosts: {plan.allowed_hosts}",
            f"max_requests: {plan.max_requests} [{plan.max_requests_formula}]",
            f"timeout_seconds: {plan.timeout_seconds}",
            f"max_retries: {plan.max_retries}",
            f"max_redirects: {plan.max_redirects}",
            f"rate_limit_rps: {plan.rate_limit_rps}",
            f"nct_id: {plan.nct_id}",
            f"pmid: {plan.pmid}",
            f"max_articles: {plan.max_articles}",
            f"max_fulltext_fetches: {plan.max_fulltext_fetches}",
            f"ncbi_tool_configured: {plan.ncbi_tool_configured}",
            f"ncbi_email_configured: {plan.ncbi_email_configured}",
            f"ncbi_api_key_configured: {plan.ncbi_api_key_configured} (optional)",
            "  note: IRA_NCBI_TOOL/IRA_NCBI_EMAIL/IRA_NCBI_API_KEY VALUES are never printed here or "
            "anywhere else this tool prints -- only whether each is configured.",
        ]
    )


def format_report_for_print(report: LiveSmokeReport, *, secrets: list[str]) -> str:
    lines = [format_plan_for_print(report.plan), ""]
    lines.append(f"status: {report.status.value}")
    lines.append(f"smoke_run_completed: {report.smoke_run_completed}")
    if report.refused:
        lines.append(f"REFUSED: {report.refused_reason}")
        lines.append("requested_urls: []")
        lines.append("logical_request_count: 0")
        lines.append("physical_attempt_count: 0")
        return "\n".join(_safe(line, secrets) for line in lines)

    lines.append(f"requested_urls ({len(report.requested_urls)}):")
    for url in report.requested_urls:
        lines.append(f"  {url} -> status={report.http_statuses.get(url)}")
    lines.append(f"logical_request_count: {report.request_count}")
    lines.append(f"physical_attempt_count: {report.attempt_count}")
    lines.append(f"cache_hit_count: {report.cache_hit_count}")
    lines.append(f"transport_outcomes: {report.transport_outcomes}")
    lines.append(f"pmids_located: {report.pmids_located}")
    lines.append(f"coverage_complete: {report.coverage_complete}")
    lines.append(f"budget_excluded_pmids: {report.budget_excluded_pmids}")
    lines.append(f"budget_skipped_europepmc_search_pmids: {report.budget_skipped_europepmc_search_pmids}")
    lines.append(f"budget_skipped_fulltext_pmcids: {report.budget_skipped_fulltext_pmcids}")
    lines.append(f"europepmc_failures: {report.europepmc_failures}")
    lines.append(f"articles ({len(report.articles)}) -- structural metadata only, never abstract/full-text body:")
    for a in report.articles:
        lines.append(f"  {json.dumps(a)}")
    lines.append(
        "  note: publication_stage/peer_review_status are structural signals only. "
        "content_kind=FULL_DOCUMENT (full text acquired) NEVER means peer-reviewed. Any RawFact this "
        "data could build (this tool builds none) would carry "
        "EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION, excluded from DECISION_GRADE_CLASSES; "
        "independent_confirmation stays False."
    )
    lines.append(f"live_verified_candidates (diagnostic only, no code was changed): {report.live_verified_candidates}")
    lines.append(f"anthropic_api_calls: {report.anthropic_api_calls}")
    lines.append(f"web_search_calls: {report.web_search_calls}")
    lines.append(f"external_llm_tokens: {report.external_llm_tokens}")
    if report.diagnostics is not None:
        lines.append(f"diagnostics: {report.diagnostics}")
    if report.errors:
        lines.append(f"errors: {report.errors}")
    return "\n".join(_safe(line, secrets) for line in lines)


def report_to_jsonable(report: LiveSmokeReport) -> dict[str, Any]:
    """A curated, JSON-safe dict. ``LiveSmokeReport`` itself never holds a
    raw ``ExecutionReport`` at all (Phase 3F.1 correction 3 requirement 1)
    -- the step payloads that (for the FETCH step) would hold real
    ``Document``/``EuropePmcSearchResult`` objects, and a ``Document``
    carrying the FULL fetched text, are read from a local variable inside
    ``run_live_smoke`` and never stored on the report object this function
    reads. Every field below is one already curated by ``run_live_smoke``
    to be metadata-only (Phase 3F.1 requirement 6); this still builds the
    dict field-by-field, rather than ``dataclasses.asdict(report)``, only
    because ``status``/``refused`` are computed properties ``asdict``
    would not see."""
    return {
        "plan": asdict(report.plan),
        "status": report.status.value,
        "smoke_run_completed": report.smoke_run_completed,
        "refused_reason": report.refused_reason,
        "requested_urls": report.requested_urls,
        "http_statuses": report.http_statuses,
        "logical_request_count": report.request_count,
        "physical_attempt_count": report.attempt_count,
        "cache_hit_count": report.cache_hit_count,
        "transport_outcomes": report.transport_outcomes,
        "pmids_located": report.pmids_located,
        "articles": report.articles,
        "coverage_complete": report.coverage_complete,
        "budget_excluded_pmids": report.budget_excluded_pmids,
        "budget_skipped_europepmc_search_pmids": report.budget_skipped_europepmc_search_pmids,
        "budget_skipped_fulltext_pmcids": report.budget_skipped_fulltext_pmcids,
        "europepmc_failures": report.europepmc_failures,
        "live_verified_candidates": report.live_verified_candidates,
        "anthropic_api_calls": report.anthropic_api_calls,
        "web_search_calls": report.web_search_calls,
        "external_llm_tokens": report.external_llm_tokens,
        "diagnostics": (asdict(report.diagnostics) if report.diagnostics is not None else None),
        "errors": report.errors,
    }


# -- offline Capture Manifest replay (Phase 3F.1 requirement 7) -------------


def _classify_capture(requested_url: str) -> str:
    """Classifies a captured request by its (already-sanitized)
    ``requested_url`` shape -- never by a recomputed filename digest (see
    module docstring: scanning by manifest content stays simpler and more
    robust than reconstructing an exact query string, even though the
    digest itself is now derived from the public URL, not a secret-bearing
    one)."""
    if "esearch.fcgi" in requested_url:
        return "esearch"
    if "efetch.fcgi" in requested_url:
        return "efetch"
    if "fullTextXML" in requested_url:
        return "europepmc_fulltext"
    if "/search" in requested_url and "europepmc" in requested_url:
        return "europepmc_search"
    return "unknown"


def analyze_capture(capture_dir: Path) -> dict[str, Any]:
    """Re-analyzes an already-saved Literature Live Smoke capture directory
    ENTIRELY OFFLINE. Makes ZERO network calls, touches NO marker file, is
    safe to run any number of times, and requires NO credentials --
    ``IRA_NCBI_*`` env vars are read only by ``run_live_smoke``'s live
    path; this function never reads them.

    Never prints an abstract or full-text BODY -- only structural
    metadata and (for full text) a section count (Phase 3F.1 requirement
    6/7).
    """
    if not capture_dir.is_dir():
        return {"capture_dir": str(capture_dir), "found": False, "error": "capture directory does not exist"}

    results: dict[str, Any] = {
        "capture_dir": str(capture_dir),
        "found": False,
        "esearch": None,
        "efetch": [],
        "europepmc_search": [],
        "europepmc_fulltext": [],
        "unclassified": [],
        "evidence_integrity_failures": [],
        "manifest_count": 0,
    }
    manifest_paths = sorted(capture_dir.glob("*.manifest.json"))
    results["manifest_count"] = len(manifest_paths)

    for manifest_path in manifest_paths:
        digest = manifest_path.name[: -len(".manifest.json")]
        body_path = next(
            (capture_dir / f"{digest}{suffix}" for suffix in (".bin", ".json", ".htm") if (capture_dir / f"{digest}{suffix}").is_file()),
            None,
        )
        raw_body = body_path.read_bytes() if body_path is not None else None
        manifest_result = read_manifest(capture_dir, digest, body=raw_body)
        if manifest_result.status is not ManifestReadStatus.VERIFIED:
            results["evidence_integrity_failures"].append(
                {"digest": digest, "status": manifest_result.status.value, "reason": manifest_result.error_reason}
            )
            if manifest_result.status is ManifestReadStatus.MISSING or manifest_result.manifest is None:
                continue
            # A non-VERIFIED-but-parseable manifest (e.g. a body that no
            # longer matches its own recorded hash/length) is still
            # reported above as an integrity failure, but is not trusted
            # enough to parse further below.
            continue

        manifest = manifest_result.manifest
        assert manifest is not None
        results["found"] = True
        kind = _classify_capture(manifest.requested_url)
        text = raw_body.decode("utf-8", errors="replace") if raw_body is not None else ""
        entry_common = {
            "requested_url": manifest.requested_url,
            "final_url": manifest.final_url,
            "http_status": manifest.http_status,
            "capture_retrieved_at": manifest.capture_retrieved_at,
            "capture_manifest_status": manifest_result.status.value,
            # Explicit and structural (HTTP-status-based), never inferred
            # from parsed_ok alone -- distinguishes "the manifest/body pair
            # is trustworthy" (capture_manifest_status == VERIFIED, an
            # Evidence Integrity question) from "the acquisition itself
            # succeeded" (this field, an entirely different question). A
            # VERIFIED manifest for a 404 capture is completely normal and
            # is NOT an Evidence Integrity failure (Phase 3F.2 correction
            # requirement 6) -- it just also has acquisition_failed=True.
            "acquisition_failed": manifest.http_status is None or manifest.http_status >= 400,
        }

        if kind == "esearch":
            body: Any = None
            try:
                body = json.loads(text) if text else None
            except json.JSONDecodeError:
                body = None
            pmids: list[str] = []
            if isinstance(body, dict):
                esearchresult = body.get("esearchresult")
                if isinstance(esearchresult, dict):
                    pmids = list(esearchresult.get("idlist") or [])
            results["esearch"] = {**entry_common, "parsed_ok": body is not None, "pmids_found": pmids}
        elif kind == "efetch":
            shape_error, reason = check_pubmed_xml_shape(text)
            articles = parse_pubmed_articleset(text) if shape_error is None else None
            results["efetch"].append(
                {
                    **entry_common,
                    "parsed_ok": articles is not None,
                    "shape_error": shape_error.value if shape_error is not None else None,
                    "shape_error_reason": reason,
                    "articles": [
                        {
                            "pmid": a.pmid,
                            "publication_stage": a.publication_stage.value,
                            "peer_review_status": a.peer_review_status.value,
                            "publication_types": list(a.publication_types),
                            "nct_ids": list(a.nct_ids),
                            "retracted": a.is_retracted,
                            "is_correction_or_erratum": a.is_correction_or_erratum,
                        }
                        for a in (articles or [])
                    ],
                }
            )
        elif kind == "europepmc_search":
            search_results = parse_europepmc_search_response(text)
            results["europepmc_search"].append(
                {
                    **entry_common,
                    "parsed_ok": search_results is not None,
                    "results": [
                        {
                            "pmid": r.pmid, "pmcid": r.pmcid, "is_open_access": r.is_open_access,
                            "in_epmc": r.in_epmc, "publication_stage": r.publication_stage.value,
                            "peer_review_status": r.peer_review_status.value,
                        }
                        for r in (search_results or [])
                    ],
                }
            )
        elif kind == "europepmc_fulltext":
            parsed = parse_europepmc_fulltext_xml(text)
            results["europepmc_fulltext"].append(
                {
                    **entry_common,
                    "parsed_ok": parsed is not None,
                    # Section COUNT only -- never the section text itself.
                    "section_count": len(parsed.sections) if parsed is not None else 0,
                }
            )
        else:
            results["unclassified"].append(entry_common)

    return results


# -- one-time marker (Phase 3F.1 requirement 3: never written on refusal;
# Phase 3F.1 correction 2 requirement 1: tied to the FIRST physical HTTP
# attempt via AllowlistedHttpClient.on_first_attempt/main()'s
# _mark_first_attempt, never inferred from run_live_smoke() raising) ------


def _marker_state(marker_path: Path) -> dict[str, Any] | None:
    if not marker_path.is_file():
        return None
    try:
        return json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_marker(marker_path: Path, *, refused: bool) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {"attempted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "refused": refused}, indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point -- see ``scripts/literature_live_smoke.py`` for the
    exact command a human runs. Never imported by cli.py/pipeline.py."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="literature_live_smoke",
        description=(
            "Phase 3F.1: Literature (PubMed/Europe PMC) Live Smoke. --nct-id (discovery) and --pmid "
            "(targeted) are a LIVE mode: running either WILL make real NCBI/Europe PMC network "
            "requests once IRA_NCBI_TOOL/IRA_NCBI_EMAIL are configured and no refusal applies. "
            "--analyze-capture is the only fully offline mode (zero network calls, no credentials "
            "required). Exactly one of --nct-id or --pmid is required, unless --analyze-capture is "
            "given."
        ),
    )
    parser.add_argument("--nct-id", type=str, default=None, help="Discovery mode: NCT ID to search PubMed for.")
    parser.add_argument("--pmid", type=str, default=None, help="Targeted mode: a specific PMID to fetch.")
    parser.add_argument(
        "--analyze-capture", type=Path, default=None, metavar="DIR",
        help="Offline-only mode: re-analyze an already-saved capture directory. Makes NO network "
        "call, touches no marker, and requires no credentials.",
    )
    parser.add_argument(
        "--max-articles", type=int, default=DEFAULT_MAX_ARTICLES,
        help=f"Discovery mode's ESearch retmax / article budget (default {DEFAULT_MAX_ARTICLES}).",
    )
    parser.add_argument(
        "--max-fulltext-fetches", type=int, default=DEFAULT_MAX_FULLTEXT_FETCHES,
        help=f"Europe PMC full-text fetch budget (default {DEFAULT_MAX_FULLTEXT_FETCHES}; targeted "
        f"mode additionally hard-caps this at {TARGETED_MAX_FULLTEXT_FETCHES_CAP}).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to save captured response bodies.")
    parser.add_argument("--force-rerun", action="store_true", help="Bypass the one-time marker guard.")
    parser.add_argument("--marker-path", type=Path, default=None, help="Override the one-time marker file path.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON instead of text.")
    args = parser.parse_args(argv)

    if args.analyze_capture is not None:
        result = analyze_capture(args.analyze_capture)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("found") else 1

    if bool(args.nct_id) == bool(args.pmid):
        print("REFUSED: exactly one of --nct-id (discovery mode) or --pmid (targeted mode) is required.")
        return 1
    mode = "discovery" if args.nct_id else "targeted"

    repo_root = Path(__file__).resolve().parents[3]
    marker_path = args.marker_path or (repo_root / "data" / "live_smoke" / "literature" / "LAST_RUN.json")
    out_dir = args.out_dir or (
        repo_root / "data" / "live_smoke" / "literature" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )

    credentials, status, missing = resolve_ncbi_credentials()
    normalized_nct_id = normalize_nct_id(args.nct_id) if args.nct_id else None

    # Displayed BEFORE any network call and BEFORE the marker/credential
    # checks below, using the SAME values run_live_smoke() will actually
    # use -- the plan shown must never silently differ from what executes.
    plan = build_plan(
        mode=mode, nct_id=normalized_nct_id, pmid=args.pmid, max_articles=args.max_articles,
        max_fulltext_fetches=args.max_fulltext_fetches, status=status,
    )
    print(format_plan_for_print(plan))

    if missing:
        print(
            f"\nREFUSED: missing required environment variable(s): {', '.join(missing)} -- refusing "
            f"to make any request ({NCBI_API_KEY_ENV_VAR} remains optional). No marker written."
        )
        return 2

    previous = _marker_state(marker_path)
    if previous is not None and not args.force_rerun:
        print(
            f"\nREFUSED: a previous live smoke run is already recorded at {marker_path} "
            f"(attempted_at={previous.get('attempted_at')}). Pass --force-rerun to override deliberately."
        )
        return 2

    secrets = [v for v in (credentials.ncbi_tool, credentials.ncbi_email, credentials.ncbi_api_key) if v]

    def _mark_first_attempt() -> None:
        # Fires exactly once, immediately before the very first real
        # outbound HTTP attempt this run makes (see
        # AllowlistedHttpClient.on_first_attempt) -- never for a missing
        # credential, a malformed ID, an already-existing marker, or any
        # other pre-send failure, and never merely because run_live_smoke()
        # was CALLED (Phase 3F.1 correction 2 requirement 1: "preflight
        # passed" is never conflated with "transmission started"). If
        # writing the marker itself raises, that exception propagates out
        # of AllowlistedHttpClient.get() uncaught -- no socket attempt is
        # made this call. From there it is caught by
        # literature_acquisition_adapter.py's own ``_safe_get`` (which
        # catches ANY exception a transport's ``.get()`` raises, so a raw
        # wire_url embedded in one can never leak outward unsanitized) and
        # surfaces as an ordinary FAILED step in the returned report --
        # this function makes no attempt to catch it a second time or paper
        # over it; either way, zero real communication occurred.
        _write_marker(marker_path, refused=False)

    # No try/except around this call. Only a genuine bug OUTSIDE every
    # adapter-wrapped client.get() call (e.g. in this module's own
    # non-transport code) could still escape run_live_smoke() uncaught;
    # every transport-layer failure -- including one from
    # _mark_first_attempt above -- is already caught and reported via
    # report.errors by the time this returns (Phase 3F.1 correction 2
    # requirement 1 -- this replaces the previous, inaccurate try/except
    # that wrote the marker on ANY exception regardless of whether a real
    # attempt had begun).
    report = run_live_smoke(
        mode=mode, nct_id=normalized_nct_id, pmid=args.pmid, max_articles=args.max_articles,
        max_fulltext_fetches=args.max_fulltext_fetches, out_dir=out_dir,
        on_first_attempt=_mark_first_attempt,
    )
    print()
    if args.json:
        # Defense in depth, matching the text path below: report_to_jsonable
        # is already curated to be metadata-only, but this scrub catches a
        # raw secret VALUE the same way format_report_for_print's _safe()
        # call does (Phase 3F.1 correction requirement 2).
        print(_safe(json.dumps(report_to_jsonable(report), indent=2, default=str), secrets))
    else:
        print(format_report_for_print(report, secrets=secrets))

    if report.refused:
        # run_live_smoke() only ever sets refused_reason BEFORE constructing
        # a client (missing credentials, a malformed ID) -- zero network
        # activity occurred, so there is nothing for the one-time marker to
        # protect against repeating, and _mark_first_attempt above never ran.
        return 1
    return 0 if not report.errors else 1


__all__ = [
    "DEFAULT_MAX_ARTICLES",
    "DEFAULT_MAX_FULLTEXT_FETCHES",
    "DEFAULT_RATE_LIMIT_RPS",
    "DEFAULT_TIMEOUT_SECONDS",
    "LITERATURE_LIVE_SMOKE_USER_AGENT",
    "MAX_REDIRECTS",
    "MAX_RETRIES",
    "NCBI_API_KEY_ENV_VAR",
    "NCBI_EMAIL_ENV_VAR",
    "NCBI_TOOL_ENV_VAR",
    "PMID_RE",
    "TARGETED_MAX_FULLTEXT_FETCHES_CAP",
    "CredentialStatus",
    "LiveSmokePlan",
    "LiveSmokeReport",
    "LiveSmokeStatus",
    "NcbiCredentials",
    "analyze_capture",
    "build_plan",
    "format_plan_for_print",
    "format_report_for_print",
    "main",
    "report_to_jsonable",
    "resolve_ncbi_credentials",
    "run_live_smoke",
]
