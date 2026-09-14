"""Phase 3D: a manual, human-run, ONE-TIME live smoke test against the real
ClinicalTrials.gov API v2 single-study endpoint, for ONE explicitly-supplied
NCT ID.

**NEVER imported by ``cli.py``/``pipeline.py``/any production code path.**
Running this module makes a REAL network call; nothing else in this
repository does that outside ``sec_live_smoke.py``, and this module is never
wired into anything that runs automatically. Every test in
``tests/unit/test_clinicaltrials_live_smoke_offline.py`` exercises this
module's logic against a fake transport -- never the real internet.

No NCT ID is ever hard-coded here or in ``clinicaltrials_acquisition_
adapter.py`` (Phase 3D requirement 3/12): the caller supplies it, on the
command line, every time.

Deliberately reuses ``sec_live_smoke.AllowlistedHttpClient`` (a purpose-built,
read-only, host-allowlisted GET client with redirect re-validation, a hard
request cap, retry/rate limiting, and response caching) rather than
duplicating that transport -- only the allowed host and the URL/plan/report
shape differ here.

Unlike SEC EDGAR, ClinicalTrials.gov's API v2 documents no mandatory
User-Agent/contact-email convention, so there is nothing to refuse on if
``IRA_CLINICALTRIALS_USER_AGENT`` is unset -- a generic default is used
instead. If the env var IS set, its value is still never printed or logged
(defense in depth, via the same secret-scrubbing ``sec_live_smoke._safe``
already uses).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..collectors.clinicaltrials import STUDY_DETAIL_URL, parse_study
from ..schemas.enums import UNKNOWN, ResearchDomain
from .acquisition_executor import AcquisitionExecutor, ExecutionReport
from .acquisition_planning import AcquisitionMethod
from .checks import SubjectScope
from .clinicaltrials_acquisition_adapter import (
    CLINICALTRIALS_ADAPTER_ID,
    NCT_ID_RE,
    ClinicalTrialsStudyAdapter,
    ClinicalTrialsStudyReference,
)
from .document_store import DocumentStore
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

ALLOWED_HOSTS: frozenset[str] = frozenset({"clinicaltrials.gov"})
#: Hard cap on real HTTP GETs in one run. One NCT ID, one endpoint -- there
#: is only ever one real request to make; the margin of 1 exists solely to
#: bound an accidental retry-loop bug, never to plan for more real traffic.
MAX_LIVE_GETS = 2
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_RATE_LIMIT_RPS = 0.5
MAX_RETRIES = 1

CT_USER_AGENT_ENV_VAR = "IRA_CLINICALTRIALS_USER_AGENT"
#: ClinicalTrials.gov's API v2 documents no mandatory User-Agent/contact
#: convention (unlike SEC EDGAR's Fair Access policy) -- this default
#: identifies the tool without embedding anyone's contact information, and
#: is used whenever the optional env var above is unset.
DEFAULT_CT_USER_AGENT = "investment-research-agent-clinicaltrials-live-smoke/1.0"


#: Kept equal to config.Settings.stale_after_days's own default by a
#: dedicated test (test_placeholder_matches_config_default's sibling for
#: this constant) -- one staleness threshold for the whole system, never a
#: second independently-chosen number.
REGISTRY_STALE_AFTER_DAYS = 400


def _partial_date_passed(candidate: str, today_date: str) -> bool | None:
    """Whether ``candidate`` (an ISO date, possibly PARTIAL -- ``"2026"``,
    ``"2026-03"``, or ``"2026-03-15"``) is entirely before ``today_date``
    (a full ``YYYY-MM-DD``), compared ONLY at candidate's own precision --
    never padded or completed into a fuller date than CT.gov itself
    asserted. Returns ``None`` (never a guess) if candidate is UNKNOWN/empty.

    Purely a date-ordering fact -- callers must never read a ``True`` here
    as "the trial is late/failed/delayed"; it says nothing beyond "the
    calendar period CT.gov registered has elapsed" (Phase 3D.2 requirement 2).
    """
    if not candidate or candidate == UNKNOWN:
        return None
    length = min(len(candidate), len(today_date))
    return candidate[:length] < today_date[:length]


def _days_since(date_str: str, today_date: str) -> int | None:
    """Whole days between a FULL ``YYYY-MM-DD`` date and ``today_date`` --
    ``None`` for a partial or UNKNOWN date rather than guessing a day count
    from incomplete precision."""
    import re as _re
    from datetime import date as _date

    if not date_str or date_str == UNKNOWN or not _re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return None
    try:
        return (_date.fromisoformat(today_date) - _date.fromisoformat(date_str)).days
    except ValueError:
        return None


@dataclass(frozen=True)
class RegistryFreshnessDiagnostics:
    """Date-only facts about how current this registry record is -- every
    field here is a comparison between two dates CT.gov itself supplied
    (or the retrieval timestamp this script itself recorded), never an
    inference about trial status, completion, delay, or failure (Phase
    3D.2 requirement 2). ``retrieved_at`` is kept structurally separate
    from every study-registered date -- it is never treated as, compared
    to represent, or substituted for a study date.
    """

    retrieved_at: str
    last_update_post: str
    last_update_submit: str
    primary_completion: str
    primary_completion_type: str
    completion_date: str
    completion_date_type: str
    #: None when last_update_post is absent/partial (never guessed).
    days_since_last_update_post: int | None
    stale_after_days_threshold: int
    #: None when days_since_last_update_post is None.
    stale_registry_record: bool | None
    #: None unless primary_completion_type == "ESTIMATED" AND the date is
    #: present -- an ACTUAL-typed date having "passed" is not an
    #: informative fact (it already happened by definition).
    estimated_primary_completion_date_passed: bool | None
    estimated_completion_date_passed: bool | None


def _build_freshness_diagnostics(parsed: dict[str, Any], retrieved_at: str) -> RegistryFreshnessDiagnostics:
    today_date = retrieved_at[:10]
    last_update_post = parsed.get("last_update_post", UNKNOWN)
    days_since = _days_since(last_update_post, today_date)
    primary_completion = parsed.get("primary_completion", UNKNOWN)
    primary_completion_type = parsed.get("primary_completion_type", UNKNOWN)
    completion_date = parsed.get("completion_date", UNKNOWN)
    completion_date_type = parsed.get("completion_date_type", UNKNOWN)
    return RegistryFreshnessDiagnostics(
        retrieved_at=retrieved_at,
        last_update_post=last_update_post,
        last_update_submit=parsed.get("last_update_submit", UNKNOWN),
        primary_completion=primary_completion,
        primary_completion_type=primary_completion_type,
        completion_date=completion_date,
        completion_date_type=completion_date_type,
        days_since_last_update_post=days_since,
        stale_after_days_threshold=REGISTRY_STALE_AFTER_DAYS,
        stale_registry_record=(days_since > REGISTRY_STALE_AFTER_DAYS) if days_since is not None else None,
        estimated_primary_completion_date_passed=(
            _partial_date_passed(primary_completion, today_date)
            if primary_completion_type == "ESTIMATED" else None
        ),
        estimated_completion_date_passed=(
            _partial_date_passed(completion_date, today_date)
            if completion_date_type == "ESTIMATED" else None
        ),
    )


def resolve_user_agent(env: dict[str, str] | None = None) -> str:
    """Never refuses -- ClinicalTrials.gov has no mandatory User-Agent
    convention to enforce. Returns the configured value if
    ``IRA_CLINICALTRIALS_USER_AGENT`` is set, else a generic default. The
    caller never prints this value either way (defense in depth via
    ``sec_live_smoke._safe``)."""
    import os

    source = env if env is not None else os.environ
    value = (source.get(CT_USER_AGENT_ENV_VAR) or "").strip()
    return value or DEFAULT_CT_USER_AGENT


@dataclass(frozen=True)
class LiveSmokePlan:
    """What this run intends to do -- printed BEFORE any network call."""

    allowed_hosts: tuple[str, ...]
    max_requests: int
    timeout_seconds: float
    max_retries: int
    rate_limit_rps: float
    nct_id: str
    study_url: str


@dataclass
class LiveSmokeReport:
    """Everything to report after the run -- diagnostics, never an
    investment conclusion."""

    plan: LiveSmokePlan
    refused_reason: str | None = None
    requested_urls: list[str] = field(default_factory=list)
    http_statuses: dict[str, int | None] = field(default_factory=dict)
    request_count: int = 0
    cache_hit_count: int = 0
    execution_report: ExecutionReport | None = None
    document_id: str | None = None
    parsed_fields: dict[str, Any] = field(default_factory=dict)
    fixture_diff: dict[str, Any] = field(default_factory=dict)
    document_diagnostics: dict[str, Any] = field(default_factory=dict)
    freshness: RegistryFreshnessDiagnostics | None = None
    live_verified_candidates: list[str] = field(default_factory=list)
    anthropic_api_calls: int = 0
    web_search_calls: int = 0
    external_llm_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.refused_reason is not None


def normalize_nct_id(nct_id: str) -> str:
    """The ONLY normalization ever applied to a caller-supplied NCT ID:
    whitespace-trimmed, uppercased. Never partial, never a guess -- every
    caller (run_live_smoke, main(), analyze_capture) must still validate
    the RESULT against NCT_ID_RE afterward; normalizing never widens what
    counts as a valid id, it only tolerates harmless human input variance
    (a trailing newline from copy-paste, lowercase letters)."""
    return (nct_id or "").strip().upper()


def build_plan(nct_id: str) -> LiveSmokePlan:
    return LiveSmokePlan(
        allowed_hosts=tuple(sorted(ALLOWED_HOSTS)),
        max_requests=MAX_LIVE_GETS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        rate_limit_rps=DEFAULT_RATE_LIMIT_RPS,
        nct_id=nct_id,
        study_url=STUDY_DETAIL_URL.format(nct_id=nct_id),
    )


def _build_graph(target_id: str = "target_ct_smoke") -> SourceRoutingGraph:
    l1 = AcquisitionStep(
        step_id="l1", target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id=CLINICALTRIALS_ADAPTER_ID,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    f = AcquisitionStep(
        step_id="f", target_id=target_id, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id=CLINICALTRIALS_ADAPTER_ID,
        depends_on_step_ids=("l1",), completion_condition=StepStatus.STRUCTURED_RECORD_RETRIEVED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    p = AcquisitionStep(
        step_id="p", target_id=target_id, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API, adapter_id=CLINICALTRIALS_ADAPTER_ID,
        depends_on_step_ids=("f",), completion_condition=StepStatus.REQUIRED_FIELDS_PARSED,
        implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    requirement = EvidenceRequirement(
        requirement_id="req_ct_smoke", serves_legacy_need_ids=("live_smoke_clinicaltrials",),
        subject_scope=SubjectScope.COMPANY, domain=ResearchDomain.SCIENCE_TECHNOLOGY,
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.CLINICALTRIALS_RECORD,
        required_step_ids=("l1", "f", "p"), serves_requirement_ids=("req_ct_smoke",),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(l1, f, p))


def _diff_against_fixture(study_payload: dict[str, Any]) -> dict[str, Any]:
    """A STRUCTURAL comparison (key paths present, never content) against
    Phase 3D's synthetic fixture -- confirms the real study payload's shape
    still matches what the adapter/parser assumes."""

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

    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "tests" / "fixtures" / "clinicaltrials_v2_real_format" / "study_recruiting_interventional.json"
    )
    try:
        fixture_payload = json.loads(fixture_path.read_text())
    except OSError as exc:
        return {"error": f"could not read Phase 3D fixture for comparison: {exc}"}

    live_paths = key_paths(study_payload)
    fixture_paths = key_paths(fixture_payload)
    return {
        "keys_only_in_live": sorted(live_paths - fixture_paths),
        "keys_only_in_fixture": sorted(fixture_paths - live_paths),
        "keys_in_both": len(live_paths & fixture_paths),
    }


def _collect_transport_diagnostics(client: Any, report: LiveSmokeReport) -> None:
    """Same generic, getattr-based approach as sec_live_smoke.py's own
    helper -- degrades cleanly for a minimal duck-typed fake transport."""
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
    nct_id: str,
    *,
    user_agent: str | None = None,
    http_client: Any | None = None,
    out_dir: Path | None = None,
) -> LiveSmokeReport:
    """Run the single-study smoke test. ``user_agent``/``http_client`` are
    injectable for offline testing only -- production callers (``scripts/
    clinicaltrials_live_smoke.py``) leave both as ``None`` so this resolves
    the (optional) environment variable and constructs a real,
    host-allowlisted client.

    Never called from Pipeline.run(); never makes an Anthropic/LLM/Web
    Search call.
    """
    normalized_nct_id = normalize_nct_id(nct_id)
    plan = build_plan(normalized_nct_id)
    report = LiveSmokeReport(plan=plan)

    if not NCT_ID_RE.match(normalized_nct_id):
        report.refused_reason = (
            f"malformed NCT ID: {nct_id!r} (normalized: {normalized_nct_id!r}) -- "
            "refusing to make any request"
        )
        return report
    nct_id = normalized_nct_id

    resolved_user_agent = user_agent if user_agent is not None else resolve_user_agent()
    client = http_client if http_client is not None else AllowlistedHttpClient(
        user_agent=resolved_user_agent, allowed_hosts=ALLOWED_HOSTS, timeout=DEFAULT_TIMEOUT_SECONDS,
        rate_limit_rps=DEFAULT_RATE_LIMIT_RPS, max_retries=MAX_RETRIES, max_requests=MAX_LIVE_GETS,
        out_dir=out_dir,
    )

    graph = _build_graph()
    store = DocumentStore()
    executor = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(client)},
        document_store=store,
    )
    execution_report = executor.run(
        graph, study_references={"target_ct_smoke": ClinicalTrialsStudyReference(nct_id=nct_id)},
    )
    report.execution_report = execution_report
    _collect_transport_diagnostics(client, report)

    for target_report in execution_report.target_reports:
        for step_result in target_report.step_results:
            if step_result.document_id:
                report.document_id = step_result.document_id
            if step_result.step_id == "f" and step_result.payload.get("study"):
                report.fixture_diff = _diff_against_fixture(step_result.payload["study"])
            if step_result.step_id == "p" and step_result.payload.get("parsed"):
                report.parsed_fields = step_result.payload["parsed"]
            if step_result.status is StepStatus.FAILED or step_result.status is StepStatus.NOT_FOUND:
                report.errors.append(f"{step_result.step_id}: {step_result.status} -- {step_result.failure_reason}")

    stored = store.resolve_by_accession(nct_id)
    if stored:
        latest = stored[-1]
        report.document_diagnostics = {
            "document_id": latest.document_id,
            "url": latest.document.url,
            "authority": latest.document.authority.value,
            "is_company_ir": latest.document.is_company_ir,
            "content_kind": latest.document.content_kind.value,
            "content_hash": latest.content_hash(),
            "retrieved_at": latest.document.retrieved_at,
        }
        if report.parsed_fields:
            report.freshness = _build_freshness_diagnostics(report.parsed_fields, latest.document.retrieved_at)

    for target_report in execution_report.target_reports:
        if target_report.outcome.value == "ACQUIRED":
            report.live_verified_candidates.extend(
                f"{target_report.target_id}:{r.step_id}" for r in target_report.step_results
            )

    return report


def format_report_for_print(report: LiveSmokeReport, *, secrets: list[str]) -> str:
    lines: list[str] = []
    lines.append("=== ClinicalTrials.gov Live Smoke (Phase 3D) -- diagnostics, not an investment conclusion ===")
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
    lines.append(f"document_id: {report.document_id}")
    lines.append(f"document_diagnostics: {report.document_diagnostics}")
    lines.append(
        "  note: independent_confirmation/company_claim are schemas.fact.Fact fields, not "
        "Document fields -- this script never constructs a Fact, only a Document, so it cannot "
        "report either here."
    )
    lines.append(f"parsed_fields: {report.parsed_fields}")
    lines.append(f"freshness: {report.freshness}")
    lines.append(
        "  note: freshness fields are date comparisons ONLY -- never read stale_registry_record "
        "or estimated_*_date_passed as completion/delay/failure; they say nothing beyond "
        "'this much calendar time has elapsed since CT.gov's own registered dates'."
    )
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
    return (
        "=== ClinicalTrials.gov Live Smoke plan (nothing sent yet) ===\n"
        f"allowed_hosts: {plan.allowed_hosts}\n"
        f"max_requests: {plan.max_requests}\n"
        f"timeout_seconds: {plan.timeout_seconds}\n"
        f"max_retries: {plan.max_retries}\n"
        f"rate_limit_rps: {plan.rate_limit_rps}\n"
        f"nct_id: {plan.nct_id}\n"
        f"GET 1 of at most {plan.max_requests}: {plan.study_url}"
    )


def _load_captured_body(capture_dir: Path, url: str) -> str | None:
    import hashlib

    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    for suffix in (".json", ".htm", ".bin"):
        candidate = capture_dir / f"{digest}{suffix}"
        if candidate.is_file():
            return candidate.read_bytes().decode("utf-8", errors="replace")
    return None


def analyze_capture(capture_dir: Path, *, nct_id: str, as_of: str | None = None) -> dict[str, Any]:
    """Re-analyze an already-saved response body from a prior live run,
    entirely offline. Makes ZERO network calls, touches NO marker file, and
    is safe to run any number of times.

    ``as_of`` (an ISO ``YYYY-MM-DDTHH:MM:SS+00:00`` timestamp, or at least a
    ``YYYY-MM-DD`` date) is the reference point for freshness diagnostics --
    ``_save_response`` never saves a capture timestamp (only the raw body,
    per Phase 3C requirement 29), so there is no reliable "retrieved_at" to
    recover from the file itself; a caller who wants freshness diagnostics
    from a re-analysis must supply one explicitly (e.g. the date the ORIGINAL
    live run actually happened, from its own printed report). Without it,
    freshness is simply omitted -- never guessed from the file's mtime or
    today's real date.
    """
    normalized_nct_id = normalize_nct_id(nct_id)
    if not NCT_ID_RE.match(normalized_nct_id):
        return {
            "capture_dir": str(capture_dir), "found": False,
            "error": f"malformed NCT ID: {nct_id!r} (normalized: {normalized_nct_id!r})",
        }
    url = STUDY_DETAIL_URL.format(nct_id=normalized_nct_id)
    body = _load_captured_body(capture_dir, url)
    if body is None:
        return {"capture_dir": str(capture_dir), "url": url, "found": False}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return {"capture_dir": str(capture_dir), "url": url, "found": True, "error": f"invalid JSON: {exc}"}
    parsed = parse_study(payload) if isinstance(payload, dict) and "protocolSection" in payload else None
    freshness = _build_freshness_diagnostics(parsed, as_of) if parsed is not None and as_of else None
    return {
        "capture_dir": str(capture_dir),
        "url": url,
        "found": True,
        "schema_ok": parsed is not None,
        "parsed_fields": parsed,
        "freshness": freshness,
        "fixture_diff": _diff_against_fixture(payload) if isinstance(payload, dict) else {"error": "not a JSON object"},
    }


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
            {"attempted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "refused": refused},
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point -- see ``scripts/clinicaltrials_live_smoke.py`` for
    the exact command a human runs. Never imported by cli.py/pipeline.py."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="clinicaltrials_live_smoke",
        description=(
            "Phase 3D: one-time ClinicalTrials.gov live smoke test for one explicitly-supplied "
            "NCT ID. Read-only GET to clinicaltrials.gov only."
        ),
    )
    parser.add_argument("--nct-id", type=str, default=None, help="The NCT ID to fetch, e.g. NCT01234567. Required.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to save the raw response BODY.")
    parser.add_argument("--force-rerun", action="store_true", help="Bypass the one-time marker guard.")
    parser.add_argument("--marker-path", type=Path, default=None, help="Override the one-time marker file path.")
    parser.add_argument(
        "--analyze-capture", type=Path, default=None, metavar="DIR",
        help="Offline-only mode: re-analyze an already-saved response body under DIR. Makes NO "
        "network call and does not touch the one-time marker. Requires --nct-id.",
    )
    parser.add_argument(
        "--as-of", type=str, default=None,
        help="ISO date/timestamp to use as the freshness reference point for --analyze-capture "
        "(e.g. the date the ORIGINAL live run happened, from its own printed report). Freshness "
        "diagnostics are omitted, never guessed from the file's mtime or today's date, if this "
        "is not given.",
    )
    args = parser.parse_args(argv)

    if not args.nct_id:
        print("REFUSED: --nct-id is required.")
        return 1
    normalized_nct_id = normalize_nct_id(args.nct_id)

    if args.analyze_capture is not None:
        result = analyze_capture(args.analyze_capture, nct_id=normalized_nct_id, as_of=args.as_of)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("found") else 1

    repo_root = Path(__file__).resolve().parents[3]
    marker_path = args.marker_path or (repo_root / "data" / "live_smoke" / "clinicaltrials" / "LAST_RUN.json")
    out_dir = args.out_dir or (
        repo_root / "data" / "live_smoke" / "clinicaltrials" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )

    # Displayed BEFORE any network call, using the SAME normalized value
    # run_live_smoke() will actually use -- the plan shown must never
    # silently differ from what gets executed.
    plan = build_plan(normalized_nct_id)
    print(format_plan_for_print(plan))

    previous = _marker_state(marker_path)
    if previous is not None and not args.force_rerun:
        print(
            f"\nREFUSED: a previous live smoke run is already recorded at {marker_path} "
            f"(attempted_at={previous.get('attempted_at')}). Pass --force-rerun to override deliberately."
        )
        return 2

    user_agent = resolve_user_agent()
    secrets = [user_agent] if user_agent and user_agent != DEFAULT_CT_USER_AGENT else []
    report = run_live_smoke(normalized_nct_id, out_dir=out_dir)
    print()
    print(format_report_for_print(report, secrets=secrets))
    if report.refused:
        # run_live_smoke() only ever sets refused_reason BEFORE constructing
        # a client (a malformed NCT ID) -- zero network activity occurred,
        # so there is nothing for the "ran once" guard to protect against
        # repeating. Never consume the one-time marker slot for this.
        return 1
    _write_marker(marker_path, refused=False)
    return 0 if not report.errors else 1
