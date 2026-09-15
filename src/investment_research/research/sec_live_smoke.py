"""Phase 3C: a manual, human-run, ONE-TIME live smoke test against the real
SEC EDGAR public endpoints, using a real, publicly-traded company's filing
(Apple Inc., CIK 320193 -- a real, publicly documented SEC identifier, not
personal or sensitive information) purely as a control filing to check
Phase 3B's real-format assumptions against an actual response.

**NEVER imported by ``cli.py``/``pipeline.py``/any production code path.**
Running this module makes REAL network calls; nothing else in this
repository does that, and this module is never wired into anything that
runs automatically (no CI hook, no pytest collection triggers it). Every
test in ``tests/unit/test_sec_live_smoke_offline.py`` /
``tests/integration/test_sec_live_smoke_transport.py`` exercises this
module's logic against a fake transport or a local loopback test server --
never the real internet.

Invoke via ``scripts/sec_live_smoke.py`` (see that file's own docstring for
the exact command). Requirements enforced here, not just documented:

* ``IRA_SEC_USER_AGENT`` is read from the environment at call time. If it is
  unset, blank, or still the built-in placeholder
  (``config.Settings``'s own default), this module makes ZERO network
  calls and refuses -- the caller decides what to print about that; this
  module itself never logs or prints the user-agent value or any address it
  may contain (Settings.secret_values() now includes it too, so even a
  ``setup_logging``-routed message would be scrubbed as defense in depth).
* Every request host is checked against ``ALLOWED_HOSTS`` -- both the
  initial URL and, for a redirect, the target URL BEFORE it is followed
  (``_AllowlistedRedirectHandler``), and again against the final response
  URL after the fact (defense in depth).
* At most ``MAX_LIVE_GETS`` real HTTP GETs are made in one run -- enforced
  by ``AllowlistedHttpClient.get()`` itself, not merely by planning fewer.
* At most one retry, only for a transient network-level failure -- never
  for a 4xx, a redirect refusal, or a rate-limit response.
* A conservative rate limit, well under SEC's own published fair-use
  guidance.
* No Anthropic API call, no LLM, no Web Search, no Full DD, no
  ``Pipeline.run()`` -- this module imports none of them.
* ``ImplementationStatus`` is never promoted in code here: a successful run
  only REPORTS which steps would be LIVE_VERIFIED candidates, in the
  returned diagnostics; nothing in ``research/source_routing_catalog.py``
  is touched by this module.
* A single-run guard: after any attempted run, a marker file records that
  fact, and a second invocation refuses (``--force-rerun`` overrides it) --
  encoding "one time" into the tool itself rather than relying solely on
  operator discipline.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..collectors.http import FetchResult, RateLimiter
from ..collectors.sec_edgar import FILING_INDEX_URL, SUBMISSIONS_URL
from ..logging_setup import SecretRedactingFilter
from ..schemas.enums import UNKNOWN, FetchOutcome, ResearchDomain
from .acquisition_executor import AcquisitionExecutor, ExecutionReport
from .acquisition_planning import AcquisitionMethod
from .capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    CaptureManifest,
    ManifestReadStatus,
    compute_content_hash,
    manifest_path_for,
    read_manifest,
    utc_now_iso,
    write_manifest,
)
from .checks import SubjectScope
from .document_store import DocumentStore
from .sec_acquisition_adapters import (
    DEFAULT_EXHIBIT_PRIORITY as PRODUCTION_EXHIBIT_PRIORITY,
)
from .sec_acquisition_adapters import (
    SecExhibitAdapter,
    SecExhibitSelectionRequest,
    SecFilingReference,
    SecPrimaryDocumentAdapter,
    _content_is_valid_body,
    _html_to_text,
    _parse_filing_detail_table,
    classify_exhibit_entry,
)
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

#: Read-only public SEC hosts. Never widened at runtime by anything this
#: module does -- a redirect or a final response pointing anywhere else is
#: refused, not "helpfully" followed.
ALLOWED_HOSTS: frozenset[str] = frozenset({"data.sec.gov", "www.sec.gov"})

#: Hard cap on real HTTP GETs in one run -- enforced in code, not aspiration.
MAX_LIVE_GETS = 6

DEFAULT_TIMEOUT_SECONDS = 20.0
#: One request per 2 seconds -- well under SEC's published fair-use guidance
#: (10 req/s), appropriate for a five-request manual smoke test.
DEFAULT_RATE_LIMIT_RPS = 0.5
#: 0 extra retries by default; at most 1, and never for a definitive
#: outcome (2xx/404/blocked/rate-limited) -- only for a transient
#: network-level failure (timeout/connection error).
MAX_RETRIES = 1

#: Apple Inc.'s real, publicly documented SEC CIK -- used ONLY as this
#: script's control-filing default. Never added to research/
#: source_routing_catalog.py's production catalog (Phase 3C requirement 18).
DEFAULT_APPLE_CIK = 320193

SEC_USER_AGENT_ENV_VAR = "IRA_SEC_USER_AGENT"
#: Must equal config.Settings' own default for sec_user_agent -- verified by
#: test_placeholder_matches_config_default in the offline test suite, so
#: this can never silently drift out of sync with config.py.
PLACEHOLDER_SEC_USER_AGENT = "investment-research-agent contact@example.com"

#: Exhibit priority for this control filing: Apple's 10-K realistically
#: carries only SOX certifications (EX-31/EX-32) as narrative-eligible
#: exhibits, not a press release or a material agreement -- widening the
#: priority list to include them exercises the full exhibit FETCH/PARSE/
#: DocumentStore path against a real response, which a press-release-only
#: default would likely find nothing to select.
DEFAULT_EXHIBIT_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "certification",
    "EX-31",
    "EX-32",
    "EX-99",
    "material agreement",
)


class HostAllowlistError(Exception):
    """Raised (and always caught) when a request or redirect would leave
    ``ALLOWED_HOSTS``. Never propagates out of ``AllowlistedHttpClient``."""


class MaxRequestsExceededError(Exception):
    """Raised when a caller (a bug, never a correct one) tries to make more
    than ``max_requests`` real GETs in one run."""


def _valid_user_agent_or_none(value: str | None) -> str | None:
    """The single validation rule for what counts as a real, usable SEC
    User-Agent -- unset/blank/the built-in placeholder all map to ``None``.
    Both ``resolve_user_agent`` (reading the environment) and
    ``run_live_smoke`` (validating an explicitly-passed value, e.g. from a
    test) route through this ONE function, so an empty string or the
    placeholder can never slip past one path while being caught by the
    other."""
    stripped = (value or "").strip()
    if not stripped or stripped == PLACEHOLDER_SEC_USER_AGENT:
        return None
    return stripped


class _Unset:
    """Sentinel type for ``run_live_smoke``'s ``user_agent`` parameter --
    distinguishes "the caller didn't pass anything" (resolve from the real
    environment) from "the caller explicitly passed ``None``/blank/the
    placeholder" (validate that exact value, never silently falling back to
    the ambient environment instead). ``None`` cannot serve as that
    "omitted" marker itself, because ``None`` is also a value a caller can
    legitimately pass on purpose (e.g. a test proving a resolved-to-None
    value is refused)."""

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET = _Unset()


def resolve_user_agent(env: dict[str, str] | None = None) -> str | None:
    """The configured SEC User-Agent, or ``None`` if it is unset, blank, or
    still the built-in placeholder. Never logs or returns anything when the
    answer is "no" beyond this boolean-shaped ``None`` -- the caller decides
    what to print, and never prints the value itself either way."""
    import os

    source = env if env is not None else os.environ
    return _valid_user_agent_or_none(source.get(SEC_USER_AGENT_ENV_VAR))


def _safe(text: str, secrets: list[str]) -> str:
    """Scrub ``text`` of any configured secret before it is ever printed --
    defense in depth on top of never interpolating a secret in the first
    place."""
    return SecretRedactingFilter(secrets).scrub(text)


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuses to follow a redirect whose target host is not in
    ``allowed_hosts`` -- checked BEFORE the redirect is ever taken (Phase 3C
    requirement 11)."""

    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        super().__init__()
        self._allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlparse(newurl).hostname
        if host not in self._allowed_hosts:
            raise HostAllowlistError(f"refused redirect to disallowed host: {host!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass
class AllowlistedHttpClient:
    """A minimal, purpose-built, read-only GET client for this smoke test
    only -- deliberately NOT ``collectors.http.HttpClient``, which does not
    expose or re-validate the final URL after a redirect. Returns the same
    ``collectors.http.FetchResult`` that shape, so it is drop-in compatible
    with ``SecPrimaryDocumentAdapter``/``SecExhibitAdapter``'s duck-typed
    ``http.get(url) -> FetchResult``-like expectation.
    """

    user_agent: str
    allowed_hosts: frozenset[str] = ALLOWED_HOSTS
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    rate_limit_rps: float = DEFAULT_RATE_LIMIT_RPS
    max_retries: int = MAX_RETRIES
    max_requests: int = MAX_LIVE_GETS
    out_dir: Path | None = None
    #: Which live-smoke module this client belongs to ("sec" /
    #: "clinicaltrials") -- recorded in each capture's manifest so a shared
    #: capture directory's manifests stay attributable (Phase 3D.4
    #: requirement 4: one manifest type, one save rule, for every source).
    source: str = "unknown"
    #: Every URL passed to ``.get()``, in order, duplicates included -- the
    #: same public convention ``FakeHttpClient`` (the offline test double)
    #: exposes, so diagnostics code can read this one attribute generically
    #: regardless of which transport it was actually given.
    requested_urls: list[str] = field(default_factory=list, init=False, repr=False)
    _cache_hits: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.rate_limit_rps)
        self._cache: dict[str, FetchResult] = {}
        self._requests_made = 0
        self._opener = urllib.request.build_opener(_AllowlistedRedirectHandler(self.allowed_hosts))
        #: requested URL -> the URL the response actually came from (after
        #: any allowed redirect) -- FetchResult itself carries no field for
        #: this (it is shared with production collectors.http.FetchResult),
        #: so it is tracked here instead, purely for manifest purposes.
        self._final_urls: dict[str, str] = {}
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def requests_made(self) -> int:
        return self._requests_made

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    def get(self, url: str, **_kwargs: Any) -> FetchResult:
        self.requested_urls.append(url)
        if url in self._cache:
            self._cache_hits += 1
            cached = self._cache[url]
            return FetchResult(
                url=cached.url, outcome=cached.outcome, status=cached.status, body=cached.body,
                headers=dict(cached.headers), attempts=cached.attempts, from_cache=True,
                error=cached.error, elapsed_ms=cached.elapsed_ms,
            )

        host = urllib.parse.urlparse(url).hostname
        if host not in self.allowed_hosts:
            result = FetchResult(url=url, outcome=FetchOutcome.BLOCKED, error=f"refused disallowed host: {host!r}")
            self._cache[url] = result
            return result

        if self._requests_made >= self.max_requests:
            raise MaxRequestsExceededError(
                f"live GET cap of {self.max_requests} reached; refusing to request {url!r}"
            )

        result = self._get_with_retry(url)
        self._requests_made += 1
        self._cache[url] = result
        if self.out_dir is not None and result.ok:
            self._save_response(url, result)
        return result

    def _get_with_retry(self, url: str) -> FetchResult:
        attempt = 0
        result: FetchResult | None = None
        while attempt <= self.max_retries:
            attempt += 1
            self._limiter.wait()
            result = self._do_get(url, attempt=attempt)
            # Never retry a definitive outcome -- only a transient one.
            if result.outcome != FetchOutcome.ERROR and result.outcome != FetchOutcome.TIMEOUT:
                return result
            if attempt > self.max_retries:
                return result
        assert result is not None
        return result

    def _do_get(self, url: str, *, attempt: int) -> FetchResult:
        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            "Accept-Encoding": "gzip",
        }
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        started = time.monotonic()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    with contextlib.suppress(OSError):
                        body = gzip.decompress(body)
                final_host = urllib.parse.urlparse(response.geturl()).hostname
                if final_host not in self.allowed_hosts:
                    # Defense in depth: the redirect handler should already
                    # have refused this, but the FINAL url is re-checked too
                    # (Phase 3C requirement 11).
                    return FetchResult(
                        url=url, outcome=FetchOutcome.BLOCKED, attempts=attempt,
                        error=f"final response host {final_host!r} not allowlisted",
                    )
                self._final_urls[url] = response.geturl()
                return FetchResult(
                    url=url, outcome=FetchOutcome.OK, status=int(response.status), body=body,
                    headers=dict(response.headers), attempts=attempt,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            if status == 404:
                return FetchResult(url=url, outcome=FetchOutcome.NOT_FOUND, status=status, attempts=attempt, error=f"HTTP {status}")
            if status == 429:
                return FetchResult(url=url, outcome=FetchOutcome.RATE_LIMITED, status=status, attempts=attempt, error=f"HTTP {status}: {exc.reason}")
            if status in (403, 407):
                return FetchResult(url=url, outcome=FetchOutcome.BLOCKED, status=status, attempts=attempt, error=f"HTTP {status}: {exc.reason}")
            return FetchResult(url=url, outcome=FetchOutcome.ERROR, status=status, attempts=attempt, error=f"HTTP {status}: {exc.reason}")
        except HostAllowlistError as exc:
            return FetchResult(url=url, outcome=FetchOutcome.BLOCKED, attempts=attempt, error=str(exc))
        except TimeoutError:
            return FetchResult(url=url, outcome=FetchOutcome.TIMEOUT, attempts=attempt, error="timed out")
        except (urllib.error.URLError, OSError) as exc:
            return FetchResult(url=url, outcome=FetchOutcome.ERROR, attempts=attempt, error=f"{type(exc).__name__}: {exc}")

    def _save_response(self, url: str, result: FetchResult) -> None:
        """Body bytes, plus a Capture Manifest recording the real UTC
        retrieval time -- never headers (which could echo request metadata)
        and never the User-Agent/API key/email (Phase 3C requirement 29,
        Phase 3D.4 requirement 3)."""
        assert self.out_dir is not None
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        suffix = ".json" if url.endswith(".json") else (".htm" if url.endswith((".htm", ".html")) or "index.htm" in url else ".bin")
        path = self.out_dir / f"{digest}{suffix}"
        with contextlib.suppress(OSError):
            path.write_bytes(result.body)
        with contextlib.suppress(OSError):
            manifest = CaptureManifest(
                schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
                source=self.source,
                requested_url=url,
                final_url=self._final_urls.get(url, url),
                http_status=result.status,
                # Recorded NOW, at the moment of capture -- never from the
                # out_dir's name, a file's mtime, or anything the caller
                # supplies later (Phase 3D.4 requirements 1/7).
                capture_retrieved_at=utc_now_iso(),
                content_hash=compute_content_hash(result.body),
                content_length=len(result.body),
            )
            write_manifest(self.out_dir, digest, manifest)


@dataclass(frozen=True)
class LiveSmokePlan:
    """What this run intends to do -- printed BEFORE any network call
    (Phase 3C requirement 19)."""

    allowed_hosts: tuple[str, ...]
    max_requests: int
    timeout_seconds: float
    max_retries: int
    rate_limit_rps: float
    cik: int
    submissions_url: str


@dataclass(frozen=True)
class RequestBreakdown:
    """Phase 3C.1: physical (transport-layer) request counts, classified by
    SEC URL shape ALONE -- entirely independent of the executor's own
    LOGICAL step-level counters (``logical_direct_api_steps``/
    ``logical_direct_http_steps``, read from ``ExecutionDiagnostics`` after
    it tallies what each ``StepExecutionResult`` itself reported making).

    The two are different measurements of the same run: this one counts
    distinct URLs actually seen on the wire; the other counts what each
    adapter step SAID it did. They happen to agree for this module's own
    graph (every physical GET here is reported by exactly one step), but
    this dataclass never assumes that -- ``categorized_total`` is asserted
    equal to ``physical_http_requests`` by a dedicated invariant test
    instead of being assumed."""

    physical_http_requests: int = 0
    submissions_requests: int = 0
    directory_index_requests: int = 0
    filing_detail_requests: int = 0
    document_body_requests: int = 0
    cache_hits: int = 0
    logical_direct_api_steps: int = 0
    logical_direct_http_steps: int = 0

    @property
    def categorized_total(self) -> int:
        return (
            self.submissions_requests
            + self.directory_index_requests
            + self.filing_detail_requests
            + self.document_body_requests
        )


@dataclass
class LiveSmokeReport:
    """Everything to report after the run -- diagnostics, never an
    investment conclusion (Phase 3C requirement 12/27)."""

    plan: LiveSmokePlan
    refused_reason: str | None = None
    requested_urls: list[str] = field(default_factory=list)
    http_statuses: dict[str, int | None] = field(default_factory=dict)
    request_count: int = 0
    cache_hit_count: int = 0
    request_breakdown: RequestBreakdown | None = None
    execution_report: ExecutionReport | None = None
    primary_document_id: str | None = None
    exhibit_document_id: str | None = None
    fixture_diff: dict[str, Any] = field(default_factory=dict)
    #: Per-artifact MATCH/COMPATIBLE_VARIATION/INCOMPATIBLE/NOT_OBSERVED,
    #: keyed by artifact name (submissions_json/directory_index_json/
    #: filing_detail_html/primary_ixbrl_document/exhibit_document) -- see
    #: ``_run_all_artifact_comparisons``. Distinct from ``fixture_diff``
    #: (submissions-only, kept for backward compatibility).
    fixture_comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Per-stored-document authority/claim-shape diagnostics (Phase 3C.1
    #: section 4) -- see ``_document_diagnostics``.
    document_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    #: Which SEC exhibit was actually selected this run, why, and whether
    #: production's own default priority would have selected it at all --
    #: see ``_audit_exhibit_selection``.
    exhibit_selection_audit: dict[str, Any] = field(default_factory=dict)
    #: Fine-grained capability confirmations (Phase 3C.1 section 5) -- see
    #: ``_capability_confirmations``. Each entry states whether ONE specific
    #: capability was actually exercised and confirmed THIS run, never a
    #: blanket "the target was ACQUIRED" rollup.
    capability_confirmations: list[dict[str, Any]] = field(default_factory=list)
    #: Steps whose target reached ACQUIRED (structurally -- every required
    #: step completed) AND whose acquired content is relevant to a real
    #: investment-research evidentiary need (matches production's own
    #: default exhibit priority, or is the primary document, which is
    #: always relevant).
    live_verified_candidates: list[str] = field(default_factory=list)
    #: Steps whose target reached ACQUIRED structurally, but whose content
    #: is NOT one production's own default priority would have selected
    #: (e.g. a certification exhibit fetched only because THIS script's own
    #: smoke-only priority list ranks it above EX-99/material agreement).
    #: Never conflated with ``live_verified_candidates`` -- structural
    #: completion is not the same claim as requirement satisfaction.
    structurally_completed_not_requirement_relevant: list[str] = field(default_factory=list)
    anthropic_api_calls: int = 0
    web_search_calls: int = 0
    external_llm_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.refused_reason is not None


def build_plan(cik: int) -> LiveSmokePlan:
    return LiveSmokePlan(
        allowed_hosts=tuple(sorted(ALLOWED_HOSTS)),
        max_requests=MAX_LIVE_GETS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        rate_limit_rps=DEFAULT_RATE_LIMIT_RPS,
        cik=cik,
        submissions_url=SUBMISSIONS_URL.format(cik=cik),
    )


def _latest_10k(submissions_payload: dict[str, Any]) -> dict[str, str] | None:
    """The most recent Form 10-K in ``submissions_payload``, by filing date.
    Never guesses a specific accession -- returns ``None`` if none is
    present, rather than falling back to some other form."""
    recent = ((submissions_payload or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    candidates = [i for i, form in enumerate(forms) if form == "10-K"]
    if not candidates:
        return None
    dates = recent.get("filingDate") or []
    best = max(candidates, key=lambda i: dates[i] if i < len(dates) else "")
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    report_dates = recent.get("reportDate") or []
    return {
        "accession": accessions[best] if best < len(accessions) else "",
        "primary_document": documents[best] if best < len(documents) else "",
        "filing_date": dates[best] if best < len(dates) else UNKNOWN,
        "report_date": report_dates[best] if best < len(report_dates) else UNKNOWN,
    }


def _build_graph(target_kind_ids: tuple[str, str]) -> SourceRoutingGraph:
    primary_id, exhibit_id = target_kind_ids
    pl1 = AcquisitionStep(
        step_id="pl1", target_id=primary_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id="sec_primary_document_adapter",
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    pf = AcquisitionStep(
        step_id="pf", target_id=primary_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=("pl1",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    pp = AcquisitionStep(
        step_id="pp", target_id=primary_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_primary_document_adapter",
        depends_on_step_ids=("pf",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    primary_requirement = EvidenceRequirement(
        requirement_id="req_primary", serves_legacy_need_ids=("live_smoke_primary",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.REGULATORY,
    )
    primary_target = AcquisitionTarget(
        target_id=primary_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=("pl1", "pf", "pp"), serves_requirement_ids=("req_primary",),
    )

    el1 = AcquisitionStep(
        step_id="el1", target_id=exhibit_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id="sec_exhibit_enumeration",
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    ef = AcquisitionStep(
        step_id="ef", target_id=exhibit_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_exhibit_enumeration",
        depends_on_step_ids=("el1",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    ep = AcquisitionStep(
        step_id="ep", target_id=exhibit_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="sec_exhibit_enumeration",
        depends_on_step_ids=("ef",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    exhibit_requirement = EvidenceRequirement(
        requirement_id="req_exhibit", serves_legacy_need_ids=("live_smoke_exhibit",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    exhibit_target = AcquisitionTarget(
        target_id=exhibit_id, target_kind=TargetKind.SEC_EXHIBIT,
        required_step_ids=("el1", "ef", "ep"), serves_requirement_ids=("req_exhibit",),
    )

    return SourceRoutingGraph(
        requirements=(primary_requirement, exhibit_requirement),
        targets=(primary_target, exhibit_target),
        steps=(pl1, pf, pp, el1, ef, ep),
    )


def _diff_against_fixtures(submissions_payload: dict[str, Any]) -> dict[str, Any]:
    """A STRUCTURAL comparison (key paths present, never content) against
    Phase 3B's synthetic fixture -- confirms the real submissions payload's
    shape still matches what the adapters assume, without comparing any
    actual filing content (which will of course differ -- Apple's 10-K is
    not the synthetic TESTCO 10-K)."""

    def key_paths(obj: Any, prefix: str = "") -> set[str]:
        paths: set[str] = set()
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                paths.add(path)
                paths |= key_paths(value, path)
        elif isinstance(obj, list) and obj:
            paths |= key_paths(obj[0], f"{prefix}[]")
        return paths

    from . import sec_acquisition_adapters as _adapters_module

    fixture_path = (
        Path(_adapters_module.__file__).resolve().parents[3]
        / "tests" / "fixtures" / "sec_edgar_real_format" / "submissions_testco.json"
    )
    try:
        fixture_payload = json.loads(fixture_path.read_text())
    except OSError as exc:
        return {"error": f"could not read Phase 3B fixture for comparison: {exc}"}

    live_paths = key_paths(submissions_payload)
    fixture_paths = key_paths(fixture_payload)
    return {
        "keys_only_in_live": sorted(live_paths - fixture_paths),
        "keys_only_in_fixture": sorted(fixture_paths - live_paths),
        "keys_in_both": len(live_paths & fixture_paths),
    }


def _classify_sec_url(url: str) -> str:
    """One of "submissions"/"directory_index"/"filing_detail"/
    "document_body" -- by URL SHAPE alone, never by what a step claims to
    have done. Used only for the run-wide physical request BREAKDOWN
    (``RequestBreakdown``); the more specific primary-vs-exhibit split used
    for fixture comparison is done separately, via ``_category_urls``,
    because two different document bodies both fall under
    "document_body" here."""
    if "/submissions/" in url:
        return "submissions"
    if url.endswith("/index.json"):
        return "directory_index"
    if re.search(r"-index\.htm$", url):
        return "filing_detail"
    return "document_body"


def _build_request_breakdown(report: LiveSmokeReport) -> RequestBreakdown:
    counts = {"submissions": 0, "directory_index": 0, "filing_detail": 0, "document_body": 0}
    for url in report.requested_urls:
        counts[_classify_sec_url(url)] += 1
    diagnostics = report.execution_report.diagnostics if report.execution_report is not None else None
    return RequestBreakdown(
        physical_http_requests=report.request_count,
        submissions_requests=counts["submissions"],
        directory_index_requests=counts["directory_index"],
        filing_detail_requests=counts["filing_detail"],
        document_body_requests=counts["document_body"],
        cache_hits=report.cache_hit_count,
        logical_direct_api_steps=diagnostics.direct_api_requests if diagnostics else 0,
        logical_direct_http_steps=diagnostics.direct_http_requests if diagnostics else 0,
    )


def _category_urls(
    cik: int, accession: str, primary_document: str, exhibit_filename: str | None
) -> dict[str, str]:
    """The 5 fully deterministic URLs for one filing's artifacts, given
    identifiers a caller already has in hand -- used identically by a live
    run (which learns these from the submissions response and the
    exhibit's LOCATE payload) and by ``analyze_capture`` (which takes them
    as explicit CLI arguments, since no network call is made to learn them
    offline)."""
    accession_nodash = accession.replace("-", "")
    urls = {
        "submissions": SUBMISSIONS_URL.format(cik=cik),
        "directory_index": FILING_INDEX_URL.format(cik=cik, accession_nodash=accession_nodash, document="index.json"),
        "filing_detail": FILING_INDEX_URL.format(
            cik=cik, accession_nodash=accession_nodash, document=f"{accession}-index.htm"
        ),
    }
    if primary_document:
        urls["primary_document"] = FILING_INDEX_URL.format(
            cik=cik, accession_nodash=accession_nodash, document=primary_document
        )
    if exhibit_filename:
        urls["exhibit_document"] = FILING_INDEX_URL.format(
            cik=cik, accession_nodash=accession_nodash, document=exhibit_filename
        )
    return urls


def _artifact_bodies_from_client(client: Any, urls: list[str]) -> dict[str, str]:
    """Best-effort url->response-text lookup for post-hoc artifact
    comparison, generic across ``AllowlistedHttpClient`` (private
    ``_cache``) and ``FakeHttpClient`` (public ``responses``) -- never
    assumes one specific client type, and never raises when a client
    exposes neither."""
    bodies: dict[str, str] = {}
    cache = getattr(client, "_cache", None)
    if isinstance(cache, dict):
        for url in urls:
            result = cache.get(url)
            if result is not None and getattr(result, "ok", False):
                bodies[url] = getattr(result, "text", "") or ""
    responses = getattr(client, "responses", None)
    if isinstance(responses, dict):
        for url in urls:
            if url not in bodies:
                result = responses.get(url)
                if result is not None and getattr(result, "ok", False):
                    bodies[url] = getattr(result, "text", "") or ""
    return bodies


_MATCH = "MATCH"
_COMPATIBLE_VARIATION = "COMPATIBLE_VARIATION"
_INCOMPATIBLE = "INCOMPATIBLE"
_NOT_OBSERVED = "NOT_OBSERVED"


@dataclass(frozen=True)
class ArtifactComparison:
    artifact: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


def _fixture_file_text(name: str) -> str | None:
    from . import sec_acquisition_adapters as _adapters_module

    fixture_path = (
        Path(_adapters_module.__file__).resolve().parents[3]
        / "tests" / "fixtures" / "sec_edgar_real_format" / name
    )
    try:
        return fixture_path.read_text()
    except OSError:
        return None


def _compare_submissions(live_text: str | None) -> ArtifactComparison:
    if not live_text:
        return ArtifactComparison("submissions_json", _NOT_OBSERVED)
    try:
        payload = json.loads(live_text)
    except json.JSONDecodeError as exc:
        return ArtifactComparison("submissions_json", _INCOMPATIBLE, {"error": f"not valid JSON: {exc}"})
    diff = _diff_against_fixtures(payload)
    if "error" in diff:
        return ArtifactComparison("submissions_json", _NOT_OBSERVED, diff)
    if not diff["keys_only_in_fixture"]:
        status = _MATCH
    elif diff["keys_in_both"] > 0:
        status = _COMPATIBLE_VARIATION
    else:
        status = _INCOMPATIBLE
    return ArtifactComparison("submissions_json", status, diff)


#: Real EDGAR ``index.json`` per-item field names (Phase 3B's own
#: documented shape -- see sec_acquisition_adapters.py's module docstring
#: and tests/fixtures/sec_edgar_real_format/MANIFEST.md).
_EXPECTED_DIRECTORY_ITEM_FIELDS = frozenset({"last-modified", "name", "type", "size"})


def _compare_directory_index(live_text: str | None) -> ArtifactComparison:
    if not live_text:
        return ArtifactComparison("directory_index_json", _NOT_OBSERVED)
    try:
        payload = json.loads(live_text)
    except json.JSONDecodeError as exc:
        return ArtifactComparison("directory_index_json", _INCOMPATIBLE, {"error": f"not valid JSON: {exc}"})
    items = ((payload or {}).get("directory") or {}).get("item") or []
    if not items:
        return ArtifactComparison("directory_index_json", _INCOMPATIBLE, {"error": "no directory.item entries found"})
    live_fields = set(items[0].keys())
    if live_fields == _EXPECTED_DIRECTORY_ITEM_FIELDS:
        status = _MATCH
    elif live_fields & _EXPECTED_DIRECTORY_ITEM_FIELDS:
        status = _COMPATIBLE_VARIATION
    else:
        status = _INCOMPATIBLE
    return ArtifactComparison(
        "directory_index_json", status,
        {"live_fields": sorted(live_fields), "expected_fields": sorted(_EXPECTED_DIRECTORY_ITEM_FIELDS)},
    )


def _compare_filing_detail(live_html: str | None) -> ArtifactComparison:
    if not live_html:
        return ArtifactComparison("filing_detail_html", _NOT_OBSERVED)
    candidates = _parse_filing_detail_table(live_html)
    if not candidates:
        return ArtifactComparison("filing_detail_html", _INCOMPATIBLE, {"error": "no Document Format Files table rows parsed"})
    complete = [c for c in candidates if c.exhibit_type != UNKNOWN and c.description != UNKNOWN]
    status = _MATCH if complete else _COMPATIBLE_VARIATION
    return ArtifactComparison(
        "filing_detail_html", status,
        {"rows_parsed": len(candidates), "rows_with_type_and_description": len(complete)},
    )


def _compare_primary_document(live_html: str | None) -> ArtifactComparison:
    if not live_html:
        return ArtifactComparison("primary_ixbrl_document", _NOT_OBSERVED)
    text = _html_to_text(live_html)
    if not _content_is_valid_body(text):
        return ArtifactComparison(
            "primary_ixbrl_document", _INCOMPATIBLE,
            {"error": "failed minimum content validation (index/rejection page or too short)"},
        )
    has_ix_namespace = bool(re.search(r"<ix:", live_html, re.I))
    has_ix_header = bool(re.search(r"<ix:header", live_html, re.I))
    has_ix_hidden = bool(re.search(r"ix:hidden", live_html, re.I))
    has_visible_facts = bool(re.search(r"<ix:non(fraction|numeric)", live_html, re.I))
    status = _MATCH if (has_ix_namespace and has_visible_facts) else _COMPATIBLE_VARIATION
    return ArtifactComparison(
        "primary_ixbrl_document", status,
        {
            "content_length": len(live_html),
            "extracted_text_length": len(text),
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "has_ix_namespace": has_ix_namespace,
            "has_ix_header": has_ix_header,
            "has_ix_hidden": has_ix_hidden,
            "has_visible_ix_facts": has_visible_facts,
        },
    )


def _compare_exhibit_document(live_html: str | None, category: str | None) -> ArtifactComparison:
    if not live_html:
        return ArtifactComparison("exhibit_document", _NOT_OBSERVED)
    text = _html_to_text(live_html)
    if not _content_is_valid_body(text):
        return ArtifactComparison(
            "exhibit_document", _INCOMPATIBLE, {"error": "failed minimum content validation", "category": category},
        )
    if category is None:
        status = _COMPATIBLE_VARIATION
    elif category in {"PRESS_RELEASE", "MATERIAL_AGREEMENT", "CERTIFICATION", "OTHER_NARRATIVE"}:
        status = _MATCH
    else:
        status = _COMPATIBLE_VARIATION
    return ArtifactComparison(
        "exhibit_document", status,
        {
            "category": category,
            "content_length": len(live_html),
            "extracted_text_length": len(text),
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        },
    )


def _run_all_artifact_comparisons(
    bodies: Mapping[str, str], exhibit_category: str | None = None
) -> list[ArtifactComparison]:
    """All 5 artifacts, compared INDIVIDUALLY (Phase 3C.1 section 3) --
    never collapsed into one submissions-only diff. ``bodies`` maps
    category name ("submissions"/"directory_index"/"filing_detail"/
    "primary_document"/"exhibit_document") to raw response text; a missing
    key reports NOT_OBSERVED for that artifact rather than guessing."""
    return [
        _compare_submissions(bodies.get("submissions")),
        _compare_directory_index(bodies.get("directory_index")),
        _compare_filing_detail(bodies.get("filing_detail")),
        _compare_primary_document(bodies.get("primary_document")),
        _compare_exhibit_document(bodies.get("exhibit_document"), exhibit_category),
    ]


def _exhibit_locate_payload(execution_report: ExecutionReport) -> Mapping[str, Any] | None:
    for target_report in execution_report.target_reports:
        if target_report.target_id != "target_exhibit":
            continue
        for step_result in target_report.step_results:
            if step_result.step_id == "el1":
                return step_result.payload
    return None


def _audit_exhibit_selection(
    execution_report: ExecutionReport | None, priority_keywords: tuple[str, ...]
) -> dict[str, Any]:
    """Why THIS exhibit was selected, and whether production's own default
    priority (``sec_acquisition_adapters.DEFAULT_EXHIBIT_PRIORITY``) would
    have selected it too -- distinguishes "this smoke script's own,
    deliberately broader priority list happened to pick a certification"
    from "production would also acquire this," which are different claims
    (Phase 3C.1 section 1)."""
    if execution_report is None:
        return {"status": _NOT_OBSERVED, "note": "the exhibit LOCATE step was never attempted this run"}
    payload = _exhibit_locate_payload(execution_report)
    if not payload:
        return {"status": _NOT_OBSERVED, "note": "exhibit LOCATE produced no payload (failed, or never reached, this run)"}

    category = payload.get("category", UNKNOWN)
    exhibit_type = str(payload.get("exhibit_type", ""))
    haystack = f"{exhibit_type} {category}".lower()
    matched_smoke_keyword = next((kw for kw in priority_keywords if kw.lower() in haystack), None)
    would_production_select = any(kw.lower() in haystack for kw in PRODUCTION_EXHIBIT_PRIORITY)

    return {
        "selected_filename": payload.get("filename"),
        "selected_exhibit_type": exhibit_type,
        "selected_category": category,
        "selected_via_smoke_priority_keyword": matched_smoke_keyword,
        "smoke_only_priority_keywords_used": list(priority_keywords),
        "production_default_priority_keywords": list(PRODUCTION_EXHIBIT_PRIORITY),
        "would_production_default_select_this_category": would_production_select,
        "not_yet_fetched_other_matches": payload.get("not_yet_fetched", []),
        "detail_html_unavailable": payload.get("detail_html_unavailable", False),
        "note": (
            "The selected exhibit's type/category also matches one of production's own "
            "default priority keywords."
            if would_production_select
            else (
                "This exhibit was selected ONLY because this script's OWN "
                "DEFAULT_EXHIBIT_PRIORITY_KEYWORDS (smoke-only -- exercises the exhibit "
                "FETCH/PARSE/DocumentStore path against a real response) ranks "
                "certification/EX-31/EX-32 ABOVE EX-99/material agreement. Production's "
                "own default priority (sec_acquisition_adapters.DEFAULT_EXHIBIT_PRIORITY) "
                "does not include certification keywords at all, so under production "
                "defaults this run's target_exhibit would most likely have reported "
                "NOT_FOUND instead -- unless 'not_yet_fetched_other_matches' below is "
                "non-empty, in which case a production-relevant candidate DID exist in "
                "this accession but lost only to this script's own (broader, "
                "certification-first) priority ordering, not to absence."
            )
        ),
    }


#: Every distinct capability this run COULD confirm, in the order Phase
#: 3C.1 section 5 lists them. Kept as one flat function (not a lookup
#: table) since each capability's confirmation predicate reads a different
#: combination of step outcomes.
def _capability_confirmations(execution_report: ExecutionReport | None) -> list[dict[str, Any]]:
    outcomes: Mapping[str, StepStatus] = execution_report.outcomes if execution_report is not None else {}
    exhibit_payload: Mapping[str, Any] = (
        (_exhibit_locate_payload(execution_report) or {}) if execution_report is not None else {}
    )
    category = exhibit_payload.get("category")
    detail_unavailable = exhibit_payload.get("detail_html_unavailable", True)

    def status_of(step_id: str) -> StepStatus | None:
        return outcomes.get(step_id)

    results: list[dict[str, Any]] = []

    def add(name: str, confirmed: bool, reason: str) -> None:
        results.append({"capability": name, "confirmed": confirmed, "reason": reason})

    add(
        "SEC submissions transport", status_of("pl1") is StepStatus.URL_RESOLVED,
        "primary LOCATE (pl1) requires a successful submissions fetch before it can resolve a URL"
        if status_of("pl1") is StepStatus.URL_RESOLVED else "pl1 did not reach URL_RESOLVED this run",
    )
    add(
        "primary document resolution", status_of("pl1") is StepStatus.URL_RESOLVED,
        "pl1 reached URL_RESOLVED" if status_of("pl1") is StepStatus.URL_RESOLVED else "pl1 did not resolve a URL this run",
    )
    add(
        "primary body fetch", status_of("pf") is StepStatus.BODY_FETCHED,
        "pf reached BODY_FETCHED" if status_of("pf") is StepStatus.BODY_FETCHED else "pf did not fetch a body this run",
    )
    add(
        "primary body parser", status_of("pp") is StepStatus.PARSED,
        "pp reached PARSED" if status_of("pp") is StepStatus.PARSED else "pp did not parse this run",
    )
    add(
        "directory index transport/parser", status_of("pl1") is StepStatus.URL_RESOLVED,
        "the accession directory index.json is fetched inside pl1's own LOCATE call"
        if status_of("pl1") is StepStatus.URL_RESOLVED else "pl1 (which fetches the directory) did not complete this run",
    )
    filing_detail_ok = status_of("el1") is StepStatus.URL_RESOLVED and not detail_unavailable
    add(
        "filing detail transport/parser", filing_detail_ok,
        "el1 fetched and parsed the filing detail HTML" if filing_detail_ok
        else "filing detail HTML was unavailable, or el1 did not complete, this run",
    )
    add(
        "exhibit enumeration", status_of("el1") is StepStatus.URL_RESOLVED,
        "el1 enumerated and selected a candidate" if status_of("el1") is StepStatus.URL_RESOLVED else "el1 did not complete this run",
    )
    classified = category not in (None, "UNKNOWN")
    add(
        "exhibit classification", classified,
        f"selected candidate classified as {category!r}" if classified else "no exhibit was classified this run",
    )
    add(
        "generic exhibit body fetch", status_of("ef") is StepStatus.BODY_FETCHED,
        "ef reached BODY_FETCHED" if status_of("ef") is StepStatus.BODY_FETCHED else "ef did not fetch a body this run",
    )
    cert_confirmed = status_of("ep") is StepStatus.PARSED and category == "CERTIFICATION"
    add(
        "certification body parser", cert_confirmed,
        "ep parsed a CERTIFICATION-category exhibit" if cert_confirmed
        else (f"not confirmed this run -- selected category was {category!r}, not CERTIFICATION"
              if status_of("ep") is StepStatus.PARSED else "ep did not parse this run"),
    )
    add(
        "EX-99 selection", category == "PRESS_RELEASE",
        "selected category was PRESS_RELEASE" if category == "PRESS_RELEASE"
        else f"not confirmed this run -- selected category was {category!r}",
    )
    add(
        "EX-10 selection", category == "MATERIAL_AGREEMENT",
        "selected category was MATERIAL_AGREEMENT" if category == "MATERIAL_AGREEMENT"
        else f"not confirmed this run -- selected category was {category!r}",
    )
    add(
        "Requirement適合判定 (acquired evidence satisfies a real investment-research EvidenceRequirement)",
        False,
        "never confirmable by this script: its EvidenceRequirement objects are structural "
        "placeholders with no content-relevance criteria, so a step reaching ACQUIRED here is "
        "a mechanical/transport fact only -- never a judgment that the evidence answers any "
        "actual research question",
    )
    return results


def _document_diagnostics(store: DocumentStore, accession: str) -> list[dict[str, Any]]:
    """Authority/claim-shape diagnostics for every document this run
    stored under ``accession`` (Phase 3C.1 section 4). ``independent_
    confirmation``/``company_claim`` are ``schemas.fact.Fact`` fields, not
    ``Document`` fields -- this script only ever constructs ``Document``s
    (never a ``Fact``), so they cannot be read off a stored document; the
    printed report notes this explicitly instead of fabricating either
    field here."""
    return [
        {
            "document_id": stored.document_id,
            "document_role": stored.document_role.value,
            "url": stored.document.url,
            "authority": stored.document.authority.value,
            "is_company_ir": stored.document.is_company_ir,
            "content_kind": stored.document.content_kind.value,
        }
        for stored in store.resolve_by_accession(accession)
    ]


def _collect_transport_diagnostics(client: Any, report: LiveSmokeReport) -> None:
    """Fold whatever request/cache bookkeeping ``client`` exposes into
    ``report`` -- generically, via ``getattr`` with safe fallbacks, rather
    than assuming ``AllowlistedHttpClient``-specific private internals.

    The real ``AllowlistedHttpClient`` exposes a public ``requested_urls``
    list, a ``requests_made``/``cache_hits`` property pair, and (privately)
    a ``_cache`` dict that also carries HTTP status per URL. A minimal
    duck-typed test double such as ``FakeHttpClient`` exposes only a public
    ``requested_urls`` list and nothing else -- this must degrade cleanly
    rather than raising ``AttributeError`` on the parts it lacks.
    """
    seen = list(getattr(client, "requested_urls", report.requested_urls))
    deduped = list(dict.fromkeys(seen))
    report.requested_urls = deduped
    # The fallback must be based on the DEDUPED list, matching what
    # AllowlistedHttpClient.requests_made itself counts (real network
    # completions -- at most one per distinct URL, repeats being cache
    # hits) -- never the raw call count, or report.request_count would
    # silently disagree with len(report.requested_urls) for a transport
    # (like FakeHttpClient) that has no caching concept of its own and so
    # requests the same URL more than once at the call level.
    report.request_count = getattr(client, "requests_made", len(deduped))
    report.cache_hit_count = getattr(client, "cache_hits", 0)

    cache = getattr(client, "_cache", None)
    if isinstance(cache, dict):
        for url, result in cache.items():
            report.http_statuses[url] = getattr(result, "status", None)
    for url in report.requested_urls:
        report.http_statuses.setdefault(url, None)


def run_live_smoke(
    cik: int = DEFAULT_APPLE_CIK,
    *,
    user_agent: str | None | _Unset = _UNSET,
    http_client: Any | None = None,
    out_dir: Path | None = None,
    exhibit_priority_keywords: tuple[str, ...] = DEFAULT_EXHIBIT_PRIORITY_KEYWORDS,
) -> LiveSmokeReport:
    """Run the control-filing smoke test. ``http_client`` is injectable for
    offline testing only -- production callers (``scripts/sec_live_smoke.py``)
    leave it as ``None`` so this constructs a real, host-allowlisted client.

    ``user_agent`` has three distinct states, not two:

    * omitted entirely (the default, ``_UNSET``) -- production behaviour:
      resolve from the real ``IRA_SEC_USER_AGENT`` environment variable.
    * explicitly passed as ``None``, blank, or the built-in placeholder --
      validated as that exact value and refused if invalid, WITHOUT ever
      falling back to the ambient environment. This matters because a
      caller (a test, deliberately) may want to prove that an
      already-resolved-to-invalid value is refused, even when the real
      environment this process happens to be running in has a perfectly
      valid ``IRA_SEC_USER_AGENT`` set (e.g. a human running this on their
      own Mac, where the tests must not depend on that env var being
      unset to pass).
    * explicitly passed as a real value -- used as-is, taking priority
      over the ambient environment.

    Never called from Pipeline.run(); never makes an Anthropic/LLM/Web
    Search call (report.anthropic_api_calls/web_search_calls/
    external_llm_tokens are always 0 -- there is no code path here that
    could increment them).
    """
    plan = build_plan(cik)
    report = LiveSmokeReport(plan=plan)

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
        user_agent=resolved_user_agent, out_dir=out_dir, source="sec",
    )

    submissions_result = client.get(plan.submissions_url)
    report.http_statuses[plan.submissions_url] = getattr(submissions_result, "status", None)
    _collect_transport_diagnostics(client, report)
    report.request_breakdown = _build_request_breakdown(report)
    if not submissions_result.ok:
        report.errors.append(f"submissions fetch failed: {submissions_result.outcome} {submissions_result.error}")
        return report

    submissions_payload = submissions_result.json() or {}
    report.fixture_diff = _diff_against_fixtures(submissions_payload)

    latest = _latest_10k(submissions_payload)
    if latest is None or not latest["accession"] or not latest["primary_document"]:
        report.errors.append(f"no Form 10-K found in CIK {cik}'s submissions listing")
        _collect_transport_diagnostics(client, report)
        report.request_breakdown = _build_request_breakdown(report)
        return report

    graph = _build_graph(("target_primary", "target_exhibit"))
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={
            "sec_primary_document_adapter": SecPrimaryDocumentAdapter(client),
            "sec_exhibit_enumeration": SecExhibitAdapter(client),
        },
        document_store=store,
    )
    filing_ref = SecFilingReference(
        cik=cik, accession=latest["accession"], primary_document=latest["primary_document"],
        form="10-K", filing_date=latest["filing_date"], report_date=latest["report_date"],
    )
    exhibit_selection = SecExhibitSelectionRequest(
        cik=cik, accession=latest["accession"], primary_document=latest["primary_document"],
        priority_keywords=exhibit_priority_keywords,
    )

    execution_report = executor.run(
        graph,
        filing_references={"target_primary": filing_ref},
        exhibit_selectors={"target_exhibit": exhibit_selection},
    )
    report.execution_report = execution_report
    _collect_transport_diagnostics(client, report)
    report.request_breakdown = _build_request_breakdown(report)

    for target_report in execution_report.target_reports:
        for step_result in target_report.step_results:
            if step_result.document_id and target_report.target_id == "target_primary":
                report.primary_document_id = step_result.document_id
            if step_result.document_id and target_report.target_id == "target_exhibit":
                report.exhibit_document_id = step_result.document_id

    report.document_diagnostics = _document_diagnostics(store, latest["accession"])
    report.exhibit_selection_audit = _audit_exhibit_selection(execution_report, exhibit_priority_keywords)
    report.capability_confirmations = _capability_confirmations(execution_report)

    exhibit_payload = _exhibit_locate_payload(execution_report) or {}
    category_urls = _category_urls(
        cik, latest["accession"], latest["primary_document"], exhibit_payload.get("filename"),
    )
    bodies_by_url = _artifact_bodies_from_client(client, list(category_urls.values()))
    bodies_by_category = {name: bodies_by_url[url] for name, url in category_urls.items() if url in bodies_by_url}
    exhibit_category = exhibit_payload.get("category")
    report.fixture_comparisons = {
        comparison.artifact: {"status": comparison.status, "detail": comparison.detail}
        for comparison in _run_all_artifact_comparisons(bodies_by_category, exhibit_category)
    }

    # Structural completion (every required step reached its completion
    # condition) is NOT the same claim as requirement satisfaction -- an
    # exhibit target that only reached ACQUIRED because THIS script's own
    # smoke-only priority list selected a certification is reported
    # separately from genuinely requirement-relevant evidence (Phase 3C.1
    # section 5). The primary document target is always requirement-
    # relevant when acquired; it is never selected by a priority-keyword
    # heuristic in the first place.
    production_relevant_exhibit = report.exhibit_selection_audit.get("would_production_default_select_this_category", False)
    for target_report in execution_report.target_reports:
        if target_report.outcome.value != "ACQUIRED":
            continue
        candidates = [f"{target_report.target_id}:{r.step_id}" for r in target_report.step_results]
        if target_report.target_id == "target_exhibit" and not production_relevant_exhibit:
            report.structurally_completed_not_requirement_relevant.extend(candidates)
        else:
            report.live_verified_candidates.extend(candidates)

    return report


def format_report_for_print(report: LiveSmokeReport, *, secrets: list[str]) -> str:
    """Human-readable diagnostic text -- every line passed through the
    secret scrubber before being returned, as defense in depth (Phase 3C
    requirement 4/20)."""
    lines: list[str] = []
    lines.append("=== SEC Live Smoke (Phase 3C) -- diagnostics, not an investment conclusion ===")
    lines.append(f"allowed_hosts: {report.plan.allowed_hosts}")
    lines.append(f"max_requests: {report.plan.max_requests}")
    lines.append(f"timeout_seconds: {report.plan.timeout_seconds}")
    lines.append(f"max_retries: {report.plan.max_retries}")
    lines.append(f"rate_limit_rps: {report.plan.rate_limit_rps}")
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
    if report.request_breakdown is not None:
        rb = report.request_breakdown
        logical_total = rb.logical_direct_api_steps + rb.logical_direct_http_steps
        lines.append(f"request_breakdown.physical_http_requests: {rb.physical_http_requests}")
        lines.append(
            f"  by category: submissions={rb.submissions_requests} "
            f"directory_index={rb.directory_index_requests} filing_detail={rb.filing_detail_requests} "
            f"document_body={rb.document_body_requests} (categorized_total={rb.categorized_total})"
        )
        lines.append(f"request_breakdown.cache_hits: {rb.cache_hits}")
        lines.append(
            f"request_breakdown.logical: direct_api_steps={rb.logical_direct_api_steps} "
            f"direct_http_steps={rb.logical_direct_http_steps} (logical_total={logical_total})"
        )
    lines.append(f"primary_document_id: {report.primary_document_id}")
    lines.append(f"exhibit_document_id: {report.exhibit_document_id}")
    lines.append(f"fixture_diff (submissions only, legacy): {report.fixture_diff}")
    lines.append("fixture_comparisons (5 artifacts, MATCH/COMPATIBLE_VARIATION/INCOMPATIBLE/NOT_OBSERVED):")
    for artifact, comparison in report.fixture_comparisons.items():
        lines.append(f"  {artifact}: {comparison['status']} {comparison['detail']}")
    lines.append("document_diagnostics (authority/is_company_ir/content_kind per stored document):")
    for doc in report.document_diagnostics:
        lines.append(f"  {doc}")
    lines.append(
        "  note: independent_confirmation/company_claim are schemas.fact.Fact fields, not "
        "Document fields -- this script never constructs a Fact, only Documents, so it cannot "
        "report either here; by production convention a STATUTORY_FILING/is_company_ir=True "
        "document would produce company_claim=True/independent_confirmation=False if a Fact "
        "were later derived from it."
    )
    lines.append(f"exhibit_selection_audit: {report.exhibit_selection_audit}")
    lines.append("capability_confirmations (fine-grained, never a blanket target rollup):")
    for capability in report.capability_confirmations:
        lines.append(f"  [{'x' if capability['confirmed'] else ' '}] {capability['capability']}: {capability['reason']}")
    lines.append(
        f"live_verified_candidates (structural completion AND requirement-relevant, diagnostic "
        f"only, no code was changed): {report.live_verified_candidates}"
    )
    lines.append(
        "structurally_completed_not_requirement_relevant (steps whose target completed but whose "
        f"content is not one production's own default priority would select): "
        f"{report.structurally_completed_not_requirement_relevant}"
    )
    lines.append(f"anthropic_api_calls: {report.anthropic_api_calls}")
    lines.append(f"web_search_calls: {report.web_search_calls}")
    lines.append(f"external_llm_tokens: {report.external_llm_tokens}")
    if report.execution_report is not None:
        lines.append(f"diagnostics: {report.execution_report.diagnostics}")
    if report.errors:
        lines.append(f"errors: {report.errors}")
    return "\n".join(_safe(line, secrets) for line in lines)


def format_plan_for_print(plan: LiveSmokePlan) -> str:
    """Printed BEFORE any network call (Phase 3C requirement 19). The
    accession-dependent URLs are not knowable until the submissions
    response is in hand, so only that first, fully-deterministic URL is
    shown; the rest are described by template."""
    return (
        "=== SEC Live Smoke plan (nothing sent yet) ===\n"
        f"allowed_hosts: {plan.allowed_hosts}\n"
        f"max_requests: {plan.max_requests}\n"
        f"timeout_seconds: {plan.timeout_seconds}\n"
        f"max_retries: {plan.max_retries}\n"
        f"rate_limit_rps: {plan.rate_limit_rps}\n"
        f"cik: {plan.cik}\n"
        f"GET 1 of at most {plan.max_requests} (submissions metadata): {plan.submissions_url}\n"
        f"GET 2..{plan.max_requests} (directory index.json, filing detail HTML, primary document "
        "body, exhibit body) are resolved from the submissions response and are not known until "
        "then; the request count is hard-capped in code regardless."
    )


def _find_captured_file(capture_dir: Path, url: str) -> tuple[str, Path] | None:
    """Recomputes ``AllowlistedHttpClient._save_response``'s exact naming
    convention (``sha256(url)[:24] + suffix``) -- returns the digest (the
    manifest's own key) alongside the body file's path, or ``None`` if
    nothing was ever saved for this URL."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    for suffix in (".json", ".htm", ".bin"):
        candidate = capture_dir / f"{digest}{suffix}"
        if candidate.is_file():
            return digest, candidate
    return None


def _load_captured_bodies_and_manifests(
    capture_dir: Path, category_urls: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Read already-saved response BODIES back off disk, never making a
    network call, and resolve each capture's Capture Manifest status
    alongside it (Phase 3D.4 / Phase 3D.4.1).

    A category with NEITHER a body NOR a manifest is simply omitted from
    ``manifests`` (unchanged pre-3D.4 behavior -- ``categories_missing``
    already reports this). Otherwise every category gets one
    ``capture_manifest_status`` naming exactly which case applies:
    ``VERIFIED`` (manifest present, schema recognized, hash/length agree
    with the body), ``MISSING`` (no manifest -- an unremarkable pre-3D.4
    capture, NOT an Evidence Integrity failure, as long as the body is
    present), or one of the Evidence Integrity failure statuses
    (``MALFORMED``/``UNSUPPORTED_SCHEMA``/``HASH_MISMATCH``/
    ``CONTENT_LENGTH_MISMATCH``/``IO_ERROR``/``BODY_MISSING`` -- a manifest
    exists claiming a capture was made, but the artifact it describes
    cannot be found to re-verify or analyze at all, which IS an Evidence
    Integrity failure, corrected in Phase 3D.4.1.1) -- never conflated
    with ``MISSING`` (Phase 3D.4.1 requirement 6). Only ``VERIFIED`` ever
    populates ``capture_retrieved_at``; every other status reports
    ``UNKNOWN``."""
    bodies: dict[str, str] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for category, url in category_urls.items():
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        found = _find_captured_file(capture_dir, url)
        raw: bytes | None = None
        if found is not None:
            _, path = found
            raw = path.read_bytes()
            bodies[category] = raw.decode("utf-8", errors="replace")
        manifest_exists = manifest_path_for(capture_dir, digest).is_file()
        if raw is None and not manifest_exists:
            continue  # nothing captured for this category at all
        result = read_manifest(capture_dir, digest, body=raw)
        verified_manifest = result.manifest if result.status is ManifestReadStatus.VERIFIED else None
        manifests[category] = {
            "capture_manifest_status": result.status.value,
            "capture_manifest_schema_version": result.manifest.schema_version if result.manifest is not None else None,
            "capture_manifest_error": result.error_reason,
            "capture_retrieved_at": verified_manifest.capture_retrieved_at if verified_manifest is not None else UNKNOWN,
        }
    return bodies, manifests


def analyze_capture(
    capture_dir: Path,
    *,
    cik: int,
    accession: str,
    primary_document: str,
    exhibit_filename: str | None = None,
    exhibit_type: str = "",
) -> dict[str, Any]:
    """Re-analyze ALREADY-SAVED response bodies from a prior live run,
    entirely offline. Makes ZERO network calls, touches NO marker file, and
    is safe to run any number of times -- unlike ``run_live_smoke``, this
    is not a "live" acquisition attempt at all. Never reads or requires
    ``IRA_SEC_USER_AGENT``.

    Requires the caller to already know the accession/primary_document
    (and, to also compare the exhibit body, the exhibit's filename/type)
    -- normally exactly what the ORIGINAL run's own printed diagnostics
    already reported; URL-to-category mapping is still recomputed by this
    caller, never read back from disk.

    Since Phase 3D.4, each saved body also has its own Capture Manifest
    (``<digest>.manifest.json``) recording the real UTC
    ``capture_retrieved_at``, a content hash, and where it came from --
    included here per-category as ``capture_manifests``, each entry naming
    one ``capture_manifest_status`` (Phase 3D.4.1's ``ManifestReadStatus``:
    ``VERIFIED``/``MISSING``/``BODY_MISSING``/``MALFORMED``/
    ``UNSUPPORTED_SCHEMA``/``HASH_MISMATCH``/``CONTENT_LENGTH_MISMATCH``/
    ``IO_ERROR``) rather than a bare ``True``/``False``/``None`` -- a
    missing manifest (an ordinary pre-3D.4 capture) is never conflated with
    a corrupt or disagreeing one (a genuine Evidence Integrity failure).
    Only ``VERIFIED`` ever populates ``capture_retrieved_at``; every other
    status reports ``UNKNOWN``.
    """
    urls = _category_urls(cik, accession, primary_document, exhibit_filename)
    bodies, manifests = _load_captured_bodies_and_manifests(capture_dir, urls)
    exhibit_category = None
    if exhibit_filename:
        exhibit_category = classify_exhibit_entry(
            filename=exhibit_filename, exhibit_type=exhibit_type, description="", primary_document=primary_document,
        ).value
    comparisons = _run_all_artifact_comparisons(bodies, exhibit_category)
    return {
        "capture_dir": str(capture_dir),
        "urls_expected": urls,
        "categories_found": sorted(bodies.keys()),
        "categories_missing": sorted(set(urls) - set(bodies)),
        "exhibit_category_guess": exhibit_category,
        "comparisons": [
            {"artifact": c.artifact, "status": c.status, "detail": c.detail} for c in comparisons
        ],
        "capture_manifests": manifests,
    }


def _marker_state(marker_path: Path) -> dict[str, Any] | None:
    if not marker_path.is_file():
        return None
    try:
        return json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_marker(marker_path: Path, *, refused: bool) -> None:
    """Records that a live run was attempted -- never the user-agent value,
    never any filing content. A subsequent invocation refuses to run again
    unless ``--force-rerun`` is passed (Phase 3C's "one time" as a coded
    guard, not just operator discipline)."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {"attempted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "refused": refused},
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point -- see ``scripts/sec_live_smoke.py`` for the exact
    command a human runs. Never imported by cli.py/pipeline.py."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="sec_live_smoke",
        description=(
            "Phase 3C: one-time SEC EDGAR live smoke test against a real "
            "control filing (default: Apple Inc.'s latest Form 10-K). "
            "Read-only GETs to data.sec.gov/www.sec.gov only. Requires "
            "IRA_SEC_USER_AGENT to already be set in this shell's environment."
        ),
    )
    parser.add_argument(
        "--cik", type=int, default=DEFAULT_APPLE_CIK,
        help="SEC CIK of the control-filing issuer (default: 320193, Apple Inc.)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory to save raw response BODIES (never headers/User-Agent). "
        "Default: data/live_smoke/sec_edgar/<UTC timestamp>/ (gitignored).",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Bypass the one-time marker guard and run again anyway.",
    )
    parser.add_argument(
        "--marker-path", type=Path, default=None,
        help="Override the one-time marker file path (mainly for testing).",
    )
    parser.add_argument(
        "--analyze-capture", type=Path, default=None, metavar="DIR",
        help="Offline-only mode: re-analyze already-saved response bodies from a prior live run "
        "under DIR. Makes NO network call and does not touch the one-time marker -- safe to run "
        "any number of times. Requires --accession and --primary-document (and, to also compare "
        "the exhibit body, --exhibit-filename); ignores --force-rerun/--out-dir.",
    )
    parser.add_argument(
        "--accession", type=str, default=None,
        help="Accession number for --analyze-capture, e.g. 0000320193-25-000079.",
    )
    parser.add_argument(
        "--primary-document", type=str, default=None,
        help="Primary document filename for --analyze-capture, e.g. aapl-20250927.htm.",
    )
    parser.add_argument(
        "--exhibit-filename", type=str, default=None,
        help="Exhibit filename for --analyze-capture, optional -- omit to skip the exhibit comparison.",
    )
    parser.add_argument(
        "--exhibit-type", type=str, default="",
        help="Best-effort exhibit Type (e.g. 'EX-31.1') for --analyze-capture's classification; optional.",
    )
    args = parser.parse_args(argv)

    if args.analyze_capture is not None:
        if not args.accession or not args.primary_document:
            print(
                "REFUSED: --analyze-capture requires --accession and --primary-document "
                "(this mode makes no network access; it only reads already-saved files)."
            )
            return 1
        result = analyze_capture(
            args.analyze_capture, cik=args.cik, accession=args.accession,
            primary_document=args.primary_document, exhibit_filename=args.exhibit_filename,
            exhibit_type=args.exhibit_type,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if not result["categories_missing"] else 1

    repo_root = Path(__file__).resolve().parents[3]
    marker_path = args.marker_path or (repo_root / "data" / "live_smoke" / "sec_edgar" / "LAST_RUN.json")
    out_dir = args.out_dir or (
        repo_root / "data" / "live_smoke" / "sec_edgar" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )

    plan = build_plan(args.cik)
    print(format_plan_for_print(plan))

    previous = _marker_state(marker_path)
    if previous is not None and not args.force_rerun:
        print(
            f"\nREFUSED: a previous live smoke run is already recorded at {marker_path} "
            f"(attempted_at={previous.get('attempted_at')}). Phase 3C is meant to run once; "
            "pass --force-rerun to override deliberately."
        )
        return 2

    user_agent = resolve_user_agent()
    secrets = [user_agent] if user_agent else []
    if user_agent is None:
        print(
            "\nREFUSED: IRA_SEC_USER_AGENT is unset, blank, or still the built-in placeholder. "
            "No SEC request was made. Set it in this shell's environment (the value itself is "
            "never logged or printed by this tool) and re-run."
        )
        # Deliberately NOT writing the one-time marker here: zero network
        # activity occurred, so there is nothing for the "ran once" guard to
        # protect against repeating -- only an ACTUAL attempt (below, where
        # at least the submissions GET is made) counts as the one run.
        return 1

    report = run_live_smoke(args.cik, user_agent=user_agent, out_dir=out_dir)
    print()
    print(format_report_for_print(report, secrets=secrets))
    _write_marker(marker_path, refused=report.refused)
    return 0 if not report.refused and not report.errors else 1
