"""Phase 3F.1: a manual, human-run Literature (PubMed/Europe PMC) Live Smoke
tool -- OFFLINE this phase.

**No real NCBI/Europe PMC network call is made or intended in this phase,
and nothing here promotes any step to ``ImplementationStatus.LIVE_VERIFIED``.**
This module exists to build out the plan/CLI/credential-gating/Capture-
Manifest/offline-replay surface a FUTURE live-communication phase would
need, and to prove -- against a fake transport and a real loopback server,
never the real internet -- that surface behaves correctly before that real
communication is ever attempted. Every test in
``tests/unit/test_literature_live_smoke_offline.py`` /
``tests/integration/test_literature_live_smoke_transport.py`` exercises
this module's logic against a fake transport or a local loopback test
server, exactly like ``sec_live_smoke.py``/``clinicaltrials_live_smoke.py``.

**NEVER imported by ``cli.py``/``pipeline.py``/any production code path.**

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
whether each is CONFIGURED (a plain boolean) is ever displayed. A refusal
(missing credential, malformed ID) or anything before the first real
request is attempted never writes the one-time LAST_RUN marker.

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

Capture Manifest replay (Phase 3F.1 requirement 7): a live request's wire
URL carries ``email``/``api_key`` (NCBI's own courtesy-identification
convention), so the digest ``AllowlistedHttpClient._save_response`` names
each saved file with (``sha256(wire_url)[:24]``) cannot be recomputed
offline without knowing those secret values -- and ``analyze_capture``
below must work with ZERO credentials configured. It therefore scans every
``*.manifest.json`` actually present in the capture directory and
classifies each by its manifest's own (already-sanitized) ``requested_url``
shape, rather than guessing a filename. Every manifest is resolved through
the SHARED ``capture_manifest.read_manifest`` (never a reimplementation),
so ``VERIFIED``/``MISSING``/``BODY_MISSING``/``MALFORMED``/
``UNSUPPORTED_SCHEMA``/``HASH_MISMATCH``/``CONTENT_LENGTH_MISMATCH``/
``IO_ERROR`` are resolved identically to every other Live Smoke tool, and
every non-``VERIFIED``, non-``MISSING`` status is surfaced explicitly as an
Evidence Integrity failure.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..collectors.http import DEFAULT_MAX_REDIRECTS
from ..collectors.literature import (
    ParsedPubmedArticle,
    check_pubmed_xml_shape,
    parse_europepmc_fulltext_xml,
    parse_europepmc_search_response,
    parse_pubmed_articleset,
)
from ..schemas.enums import ResearchDomain
from .acquisition_executor import AcquisitionExecutor, ExecutionReport
from .acquisition_planning import AcquisitionMethod
from .capture_manifest import ManifestReadStatus, read_manifest
from .checks import SubjectScope
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
)
from .sec_live_smoke import AllowlistedHttpClient, _safe
from .source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    ImplementationStatus,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetKind,
)

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

NCBI_TOOL_ENV_VAR = "IRA_NCBI_TOOL"
NCBI_EMAIL_ENV_VAR = "IRA_NCBI_EMAIL"
NCBI_API_KEY_ENV_VAR = "IRA_NCBI_API_KEY"

#: NCBI PMIDs are plain decimal integers; generous upper bound on digit
#: count, never a guess at a real PMID's current maximum length.
PMID_RE = re.compile(r"^\d{1,9}$")


@dataclass(frozen=True)
class NcbiCredentials:
    """Duck-typed to match ``literature_acquisition_adapter.py``'s own
    ``_eutils_params``/``_secret_values`` expectations (``ncbi_tool``/
    ``ncbi_email``/``ncbi_api_key`` attributes). Never logged, printed, or
    placed in a plan/report/Document/Capture Manifest anywhere in THIS
    module; the adapter it is handed to uses these values for exactly one
    thing -- building a wire URL -- per that module's own wire_url/
    public_url discipline."""

    ncbi_tool: str
    ncbi_email: str
    ncbi_api_key: str


@dataclass(frozen=True)
class CredentialStatus:
    """Configured/not-configured booleans ONLY -- never the values
    themselves (Phase 3F.1 requirement 3)."""

    tool_configured: bool
    email_configured: bool
    #: Optional; never contributes to a refusal on its own.
    api_key_configured: bool


def resolve_ncbi_credentials(env: dict[str, str] | None = None) -> tuple[NcbiCredentials, CredentialStatus, list[str]]:
    """Reads the three NCBI courtesy-identification env vars directly from
    the environment -- deliberately NOT through ``config.Settings``, whose
    ``ncbi_tool`` carries a silent non-empty default this tool must NOT
    inherit (this tool refuses on an unset/blank ``IRA_NCBI_TOOL``, exactly
    like ``sec_live_smoke.py``'s own ``IRA_SEC_USER_AGENT`` refusal, rather
    than quietly using a placeholder). Returns
    ``(credentials, status, missing_required_env_var_names)`` --
    ``IRA_NCBI_API_KEY`` is always optional and never appears in the
    missing list.
    """
    source = env if env is not None else os.environ
    tool = (source.get(NCBI_TOOL_ENV_VAR) or "").strip()
    email = (source.get(NCBI_EMAIL_ENV_VAR) or "").strip()
    api_key = (source.get(NCBI_API_KEY_ENV_VAR) or "").strip()
    missing = [name for name, value in ((NCBI_TOOL_ENV_VAR, tool), (NCBI_EMAIL_ENV_VAR, email)) if not value]
    credentials = NcbiCredentials(ncbi_tool=tool, ncbi_email=email, ncbi_api_key=api_key)
    status = CredentialStatus(
        tool_configured=bool(tool), email_configured=bool(email), api_key_configured=bool(api_key),
    )
    return credentials, status, missing


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


@dataclass
class LiveSmokeReport:
    """Everything to report after the run -- diagnostics, never an
    investment conclusion, and never an abstract/full-text body."""

    plan: LiveSmokePlan
    refused_reason: str | None = None
    requested_urls: list[str] = field(default_factory=list)
    http_statuses: dict[str, int | None] = field(default_factory=dict)
    request_count: int = 0
    cache_hit_count: int = 0
    execution_report: ExecutionReport | None = None
    pmids_located: list[str] = field(default_factory=list)
    #: One curated, JSON-safe dict per article -- metadata/diagnostics
    #: only; abstract/full-text body text is never included here.
    articles: list[dict[str, Any]] = field(default_factory=list)
    coverage_complete: bool = True
    budget_excluded_pmids: list[str] = field(default_factory=list)
    budget_skipped_europepmc_search_pmids: list[str] = field(default_factory=list)
    budget_skipped_fulltext_pmcids: list[str] = field(default_factory=list)
    #: Diagnostic only -- this phase never promotes anything, so this is
    #: always empty in practice; kept for structural parity with
    #: sec/clinicaltrials/form4's own Live Smoke reports.
    live_verified_candidates: list[str] = field(default_factory=list)
    anthropic_api_calls: int = 0
    web_search_calls: int = 0
    external_llm_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.refused_reason is not None


def build_plan(
    *,
    mode: str,
    nct_id: str | None,
    pmid: str | None,
    max_articles: int,
    max_fulltext_fetches: int,
    status: CredentialStatus,
) -> LiveSmokePlan:
    if mode == "discovery":
        _limits, max_requests, formula = _discovery_budget(max_articles, max_fulltext_fetches)
    else:
        _limits, max_requests, formula = _targeted_budget(max_fulltext_fetches)
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
        max_fulltext_fetches=max_fulltext_fetches,
        ncbi_tool_configured=status.tool_configured,
        ncbi_email_configured=status.email_configured,
        ncbi_api_key_configured=status.api_key_configured,
    )


def _build_graph(target_id: str = "target_lit_smoke") -> SourceRoutingGraph:
    l1 = AcquisitionStep(
        step_id="l1", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id=LITERATURE_ADAPTER_ID,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    f = AcquisitionStep(
        step_id="f", target_id=target_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=LITERATURE_ADAPTER_ID,
        depends_on_step_ids=("l1",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    p = AcquisitionStep(
        step_id="p", target_id=target_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=LITERATURE_ADAPTER_ID,
        depends_on_step_ids=("f",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    requirement = EvidenceRequirement(
        requirement_id="req_lit_smoke", serves_legacy_need_ids=("live_smoke_literature",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.SCIENCE_TECHNOLOGY,
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.LITERATURE_ARTICLE,
        required_step_ids=("l1", "f", "p"), serves_requirement_ids=("req_lit_smoke",),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(l1, f, p))


def _collect_transport_diagnostics(client: Any, report: LiveSmokeReport) -> None:
    """Same generic, getattr-based approach as sec_live_smoke.py/
    clinicaltrials_live_smoke.py's own helper -- degrades cleanly for a
    minimal duck-typed fake transport. ``client.requested_urls`` is already
    the sanitized public-URL form (Phase 3F.0.3), so nothing here needs its
    own sanitization pass."""
    seen = list(getattr(client, "requested_urls", report.requested_urls))
    deduped = list(dict.fromkeys(seen))
    report.requested_urls = deduped
    report.request_count = getattr(client, "requests_made", len(deduped))
    report.cache_hit_count = getattr(client, "cache_hits", 0)
    cache = getattr(client, "_cache", None)
    if isinstance(cache, dict):
        for url, result in cache.items():
            report.http_statuses[url] = getattr(result, "status", None)
    for url in report.requested_urls:
        report.http_statuses.setdefault(url, None)


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
) -> LiveSmokeReport:
    """Run one Literature Live Smoke pass. ``http_client``/``env`` are
    injectable for offline testing only -- production callers (``scripts/
    literature_live_smoke.py``) leave both as ``None`` so this resolves the
    real environment and constructs a real, host-allowlisted client.

    Never called from ``Pipeline.run()``; never makes an Anthropic/LLM/Web
    Search call; makes no real network call in THIS phase either way (see
    module docstring), but the code path here is written exactly as a
    future live-communication phase would run it.
    """
    if mode not in ("discovery", "targeted"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'discovery' or 'targeted')")

    credentials, status, missing = resolve_ncbi_credentials(env)
    normalized_nct_id = normalize_nct_id(nct_id) if nct_id else None
    normalized_pmid = (pmid or "").strip()
    effective_max_fulltext_fetches = (
        max_fulltext_fetches if mode == "discovery"
        else min(max_fulltext_fetches, TARGETED_MAX_FULLTEXT_FETCHES_CAP)
    )
    plan = build_plan(
        mode=mode, nct_id=normalized_nct_id, pmid=normalized_pmid or None, max_articles=max_articles,
        max_fulltext_fetches=effective_max_fulltext_fetches, status=status,
    )
    report = LiveSmokeReport(plan=plan)

    # Credential check FIRST -- before validating the ID -- so a missing
    # env var is never masked by an unrelated ID-format refusal (Phase
    # 3F.1 requirement 3). Nothing has been constructed yet that could ever
    # make a request.
    if missing:
        report.refused_reason = (
            f"missing required environment variable(s): {', '.join(missing)} -- refusing to make "
            f"any request ({NCBI_API_KEY_ENV_VAR} remains optional)"
        )
        return report

    if mode == "discovery":
        if not normalized_nct_id or not NCT_ID_RE.match(normalized_nct_id):
            report.refused_reason = f"malformed or missing NCT ID for discovery mode: {nct_id!r}"
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
            report.refused_reason = f"malformed or missing PMID for targeted mode: {pmid!r}"
            return report
        limits, _, _ = _targeted_budget(effective_max_fulltext_fetches)
        reference = LiteratureReference(ct_pmid=normalized_pmid, max_pmids=1)

    client = http_client if http_client is not None else AllowlistedHttpClient(
        user_agent=LITERATURE_LIVE_SMOKE_USER_AGENT, allowed_hosts=ALLOWED_HOSTS,
        timeout=DEFAULT_TIMEOUT_SECONDS, rate_limit_rps=DEFAULT_RATE_LIMIT_RPS, max_retries=MAX_RETRIES,
        max_requests=plan.max_requests, max_redirects=MAX_REDIRECTS, out_dir=out_dir, source="literature",
    )

    graph = _build_graph()
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(
        client, credentials,
        max_search_requests=limits.max_search_requests, max_articles=limits.max_articles,
        max_fulltext_fetches=limits.max_fulltext_fetches, max_total_requests=limits.max_total_requests,
    )
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    execution_report = executor.run(graph, literature_references={"target_lit_smoke": reference})
    report.execution_report = execution_report
    _collect_transport_diagnostics(client, report)

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
                            "europepmc_is_open_access": bool(doc.get("europepmc_is_open_access", False)),
                            "europepmc_in_epmc": bool(doc.get("europepmc_in_epmc", False)),
                            "full_text_acquired": bool(doc.get("full_text_acquired", False)),
                            "document_id": doc_id,
                            "document_authority": stored.document.authority.value if stored else None,
                            "document_content_kind": stored.document.content_kind.value if stored else None,
                        }
                    )
            if step_result.status in (
                StepStatus.FAILED, StepStatus.NOT_FOUND, StepStatus.ZERO_RESULTS, StepStatus.SKIPPED_DUE_TO_BUDGET,
            ):
                report.errors.append(f"{step_result.step_id}: {step_result.status} -- {step_result.failure_reason}")

    for target_report in execution_report.target_reports:
        if target_report.outcome.value == "ACQUIRED":
            report.live_verified_candidates.extend(
                f"{target_report.target_id}:{r.step_id}" for r in target_report.step_results
            )
    return report


def format_plan_for_print(plan: LiveSmokePlan) -> str:
    return "\n".join(
        [
            "=== Literature (PubMed/Europe PMC) Live Smoke plan (nothing sent yet, OFFLINE this phase) ===",
            f"mode: {plan.mode}",
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
    if report.refused:
        lines.append(f"REFUSED: {report.refused_reason}")
        lines.append("requested_urls: []")
        lines.append("http_request_count: 0")
        return "\n".join(_safe(line, secrets) for line in lines)

    lines.append(f"requested_urls ({len(report.requested_urls)}):")
    for url in report.requested_urls:
        lines.append(f"  {url} -> status={report.http_statuses.get(url)}")
    lines.append(f"http_request_count: {report.request_count}")
    lines.append(f"cache_hit_count: {report.cache_hit_count}")
    lines.append(f"pmids_located: {report.pmids_located}")
    lines.append(f"coverage_complete: {report.coverage_complete}")
    lines.append(f"budget_excluded_pmids: {report.budget_excluded_pmids}")
    lines.append(f"budget_skipped_europepmc_search_pmids: {report.budget_skipped_europepmc_search_pmids}")
    lines.append(f"budget_skipped_fulltext_pmcids: {report.budget_skipped_fulltext_pmcids}")
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
    if report.execution_report is not None:
        lines.append(f"diagnostics: {report.execution_report.diagnostics}")
    if report.errors:
        lines.append(f"errors: {report.errors}")
    return "\n".join(_safe(line, secrets) for line in lines)


def report_to_jsonable(report: LiveSmokeReport) -> dict[str, Any]:
    """A curated, JSON-safe dict -- deliberately NOT ``dataclasses.asdict``
    of the whole report, since ``report.execution_report`` embeds step
    payloads that (for the FETCH step) hold real ``Document``/
    ``EuropePmcSearchResult`` objects, and a ``Document`` carries the FULL
    fetched text. Every field below is one already curated by
    ``run_live_smoke`` to be metadata-only (Phase 3F.1 requirement 6)."""
    return {
        "plan": asdict(report.plan),
        "refused_reason": report.refused_reason,
        "requested_urls": report.requested_urls,
        "http_statuses": report.http_statuses,
        "request_count": report.request_count,
        "cache_hit_count": report.cache_hit_count,
        "pmids_located": report.pmids_located,
        "articles": report.articles,
        "coverage_complete": report.coverage_complete,
        "budget_excluded_pmids": report.budget_excluded_pmids,
        "budget_skipped_europepmc_search_pmids": report.budget_skipped_europepmc_search_pmids,
        "budget_skipped_fulltext_pmcids": report.budget_skipped_fulltext_pmcids,
        "live_verified_candidates": report.live_verified_candidates,
        "anthropic_api_calls": report.anthropic_api_calls,
        "web_search_calls": report.web_search_calls,
        "external_llm_tokens": report.external_llm_tokens,
        "diagnostics": (asdict(report.execution_report.diagnostics) if report.execution_report is not None else None),
        "errors": report.errors,
    }


# -- offline Capture Manifest replay (Phase 3F.1 requirement 7) -------------


def _classify_capture(requested_url: str) -> str:
    """Classifies a captured request by its (already-sanitized)
    ``requested_url`` shape -- never by a recomputed filename digest (see
    module docstring: the digest is derived from the WIRE url, which
    carries a secret this function must never need)."""
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


# -- one-time marker (Phase 3F.1 requirement 3: never written on refusal) ---


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
            "Phase 3F.1: Literature (PubMed/Europe PMC) Live Smoke -- OFFLINE this phase. No real "
            "NCBI/Europe PMC network call is made or intended; this validates the plan/CLI/"
            "credential-gating/Capture-Manifest-replay surface ahead of a future live-communication "
            "phase. Exactly one of --nct-id (discovery) or --pmid (targeted) is required, unless "
            "--analyze-capture is given."
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
    report = run_live_smoke(
        mode=mode, nct_id=normalized_nct_id, pmid=args.pmid, max_articles=args.max_articles,
        max_fulltext_fetches=args.max_fulltext_fetches, out_dir=out_dir,
    )
    print()
    if args.json:
        print(json.dumps(report_to_jsonable(report), indent=2, default=str))
    else:
        print(format_report_for_print(report, secrets=secrets))

    if report.refused:
        # run_live_smoke() only ever sets refused_reason BEFORE constructing
        # a client (missing credentials, a malformed ID) -- zero network
        # activity occurred, so there is nothing for the one-time marker to
        # protect against repeating.
        return 1
    _write_marker(marker_path, refused=False)
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
