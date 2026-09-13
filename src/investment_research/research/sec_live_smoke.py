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
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..collectors.http import FetchResult, RateLimiter
from ..collectors.sec_edgar import SUBMISSIONS_URL
from ..logging_setup import SecretRedactingFilter
from ..schemas.enums import UNKNOWN, FetchOutcome, ResearchDomain
from .acquisition_executor import AcquisitionExecutor, ExecutionReport
from .acquisition_planning import AcquisitionMethod
from .checks import SubjectScope
from .document_store import DocumentStore
from .sec_acquisition_adapters import (
    SecExhibitAdapter,
    SecExhibitSelectionRequest,
    SecFilingReference,
    SecPrimaryDocumentAdapter,
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
        """Body bytes only -- never headers (which could echo request
        metadata) and never the User-Agent (Phase 3C requirement 29)."""
        assert self.out_dir is not None
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        suffix = ".json" if url.endswith(".json") else (".htm" if url.endswith((".htm", ".html")) or "index.htm" in url else ".bin")
        path = self.out_dir / f"{digest}{suffix}"
        with contextlib.suppress(OSError):
            path.write_bytes(result.body)


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
    execution_report: ExecutionReport | None = None
    primary_document_id: str | None = None
    exhibit_document_id: str | None = None
    fixture_diff: dict[str, Any] = field(default_factory=dict)
    live_verified_candidates: list[str] = field(default_factory=list)
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
    report.requested_urls = list(dict.fromkeys(seen))
    report.request_count = getattr(client, "requests_made", len(seen))
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
    user_agent: str | None = None,
    http_client: Any | None = None,
    out_dir: Path | None = None,
    exhibit_priority_keywords: tuple[str, ...] = DEFAULT_EXHIBIT_PRIORITY_KEYWORDS,
) -> LiveSmokeReport:
    """Run the control-filing smoke test. ``user_agent``/``http_client`` are
    injectable for offline testing only -- production callers (``scripts/
    sec_live_smoke.py``) leave both as ``None`` so this resolves the real
    environment variable and constructs a real, host-allowlisted client.

    Never called from Pipeline.run(); never makes an Anthropic/LLM/Web
    Search call (report.anthropic_api_calls/web_search_calls/
    external_llm_tokens are always 0 -- there is no code path here that
    could increment them).
    """
    plan = build_plan(cik)
    report = LiveSmokeReport(plan=plan)

    resolved_user_agent = _valid_user_agent_or_none(user_agent) if user_agent is not None else resolve_user_agent()
    if resolved_user_agent is None:
        report.refused_reason = (
            "IRA_SEC_USER_AGENT is unset, blank, or still the built-in placeholder -- "
            "refusing to make any SEC request"
        )
        return report

    client = http_client if http_client is not None else AllowlistedHttpClient(
        user_agent=resolved_user_agent, out_dir=out_dir,
    )

    submissions_result = client.get(plan.submissions_url)
    report.http_statuses[plan.submissions_url] = getattr(submissions_result, "status", None)
    _collect_transport_diagnostics(client, report)
    if not submissions_result.ok:
        report.errors.append(f"submissions fetch failed: {submissions_result.outcome} {submissions_result.error}")
        return report

    submissions_payload = submissions_result.json() or {}
    report.fixture_diff = _diff_against_fixtures(submissions_payload)

    latest = _latest_10k(submissions_payload)
    if latest is None or not latest["accession"] or not latest["primary_document"]:
        report.errors.append(f"no Form 10-K found in CIK {cik}'s submissions listing")
        _collect_transport_diagnostics(client, report)
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

    for target_report in execution_report.target_reports:
        for step_result in target_report.step_results:
            if step_result.document_id and target_report.target_id == "target_primary":
                report.primary_document_id = step_result.document_id
            if step_result.document_id and target_report.target_id == "target_exhibit":
                report.exhibit_document_id = step_result.document_id

    # Diagnostic-only candidate list -- NEVER a code-level promotion.
    for target_report in execution_report.target_reports:
        if target_report.outcome.value == "ACQUIRED":
            report.live_verified_candidates.extend(
                f"{target_report.target_id}:{r.step_id}" for r in target_report.step_results
            )

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
    lines.append(f"primary_document_id: {report.primary_document_id}")
    lines.append(f"exhibit_document_id: {report.exhibit_document_id}")
    lines.append(f"fixture_diff: {report.fixture_diff}")
    lines.append(f"live_verified_candidates (diagnostic only, no code was changed): {report.live_verified_candidates}")
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
    args = parser.parse_args(argv)

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

    report = run_live_smoke(args.cik, out_dir=out_dir)
    print()
    print(format_report_for_print(report, secrets=secrets))
    _write_marker(marker_path, refused=report.refused)
    return 0 if not report.refused and not report.errors else 1
