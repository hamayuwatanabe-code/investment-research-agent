"""Command-line interface.

    python main.py DEMOBIO --fixtures
    python main.py CRBP --full-dd
    python main.py CRBP --update
    python main.py DEMOBIO --kill-test --fixtures
    python main.py --compare CRBP CNTB
    python main.py --screen explosive --fixtures

Live network access is OFF by default (``--live`` enables it), because a run
that silently reaches the network behaves differently from one that does not,
and the difference must be a deliberate choice.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from .collectors.base import CollectionResult
from .collectors.clinicaltrials import ClinicalTrialsCollector
from .collectors.extraction import DocumentCollector
from .collectors.fda import FdaCollector
from .collectors.fixtures import FixtureCollector
from .collectors.http import HttpClient
from .collectors.search import build_search_provider
from .collectors.sec_edgar import SecEdgarCollector
from .config import get_settings
from .llm.client import LLMBudget, LLMClient
from .logging_setup import setup_logging
from .orchestrator.pipeline import Pipeline, ResearchResult, new_run_id
from .reporting.report import adaptive_literature_info, render_report
from .research.adversarial import build_plan, run_adversarial_search
from .research.anthropic_web import AnthropicWebResearchProvider
from .research.corpus import CorpusResearchProvider
from .research.literature_acquisition_adapter import ALLOWED_HOSTS as LITERATURE_ALLOWED_HOSTS
from .research.literature_pipeline_integration import (
    DEFAULT_MAX_ARTICLES as LITERATURE_DEFAULT_MAX_ARTICLES,
)
from .research.literature_pipeline_integration import (
    DEFAULT_MAX_FULLTEXT_FETCHES as LITERATURE_DEFAULT_MAX_FULLTEXT_FETCHES,
)
from .research.literature_pipeline_integration import (
    DEFAULT_RATE_LIMIT_RPS as ADAPTIVE_RATE_LIMIT_RPS,
)
from .research.literature_pipeline_integration import (
    DEFAULT_TIMEOUT_SECONDS as ADAPTIVE_TIMEOUT_SECONDS,
)
from .research.literature_pipeline_integration import (
    LITERATURE_PIPELINE_USER_AGENT,
    LiteraturePipelineRequest,
    LiteraturePipelineRequestError,
    bundle_diagnostics,
    run_literature_pipeline_acquisition,
    validate_literature_pipeline_request,
)
from .research.literature_pipeline_integration import (
    MAX_REDIRECTS as ADAPTIVE_MAX_REDIRECTS,
)
from .research.literature_pipeline_integration import (
    MAX_RETRIES as ADAPTIVE_MAX_RETRIES,
)
from .research.ncbi_credentials import (
    NCBI_API_KEY_ENV_VAR,
    NCBI_EMAIL_ENV_VAR,
    NCBI_TOOL_ENV_VAR,
)
from .research.provider import (
    CompositeResearchProvider,
    NullResearchProvider,
    ResearchProvider,
)
from .research.sec_live_smoke import AllowlistedHttpClient
from .schemas.enums import Provenance, RunStatus
from .scoring.adaptive_acquisition_plan import READY_MAX_REQUESTS
from .storage.db import open_db
from .storage.repository import Repository

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="investment-research",
        description=(
            "Falsification-first investment research. The goal is to eliminate wrong "
            "hypotheses early, not to find reasons to buy."
        ),
    )
    parser.add_argument("ticker", nargs="?", help="ticker to research, e.g. CRBP")
    parser.add_argument("--full-dd", action="store_true", help="full due diligence (all agents)")
    parser.add_argument(
        "--update", action="store_true", help="re-run and diff against the last thesis"
    )
    parser.add_argument(
        "--kill-test", action="store_true", help="kill gate only; skip the bull case"
    )
    parser.add_argument("--catalyst", action="store_true", help="catalyst calendar only")
    parser.add_argument(
        "--compare", nargs="+", metavar="TICKER", help="compare two or more tickers"
    )
    parser.add_argument(
        "--screen", metavar="MODE", help="screen fixtures/known tickers, e.g. explosive"
    )
    parser.add_argument(
        "--fixtures", action="store_true", help="use SYNTHETIC fixture data (never real research)"
    )
    parser.add_argument(
        "--corpus",
        action="store_true",
        help=(
            "replay a CAPTURED corpus of real documents from data/corpus/. Real issuer, real "
            "URLs, captured at a stated time -- not a live fetch, and the report says so."
        ),
    )
    parser.add_argument("--live", action="store_true", help="allow outbound network calls")
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "use Claude for the eight interpretive agents. Requires credentials; without them "
            "the deterministic agents run and the report records that."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="claude-sonnet-5",
        help="model id for --llm (default claude-sonnet-5; pass claude-opus-5 for a "
        "red-team/audit run)",
    )
    parser.add_argument(
        "--llm-effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="reasoning effort for the eight INTERPRETIVE agents (Regulatory/Science/"
        "Competitive/Contradiction/Kill/Bear/Bull/Blind Judge) only. Web discovery and "
        "primary-source fetch/extraction never inherit this -- see --research-effort.",
    )
    parser.add_argument(
        "--research-effort",
        default="low",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="reasoning effort for web discovery (adversarial search, follow-up query "
        "generation) and primary-source fetch/extraction (default low: a live smoke test "
        "showed two --llm-effort=high discovery searches alone consuming 76,891 tokens). "
        "Does not change --llm-effort for the interpretive agents; pass a higher value "
        "explicitly only for a deliberately deeper research pass.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=500_000,
        help=(
            "maximum total LLM tokens for the run (default 500000: a materially safer "
            "ceiling for normal live Sonnet research after a live smoke test showed a "
            "single server-side web search consuming 18k+ tokens; pass a higher explicit "
            "value for a deliberately larger run). If this is insufficient, the run reports "
            "RESEARCH STATUS: INCOMPLETE/BLOCKED_PENDING_VERIFICATION with no action label "
            "rather than completing on partial research."
        ),
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="run the separated bear and bull web search passes (requirement P3)",
    )
    parser.add_argument("--resume", metavar="RUN_ID", help="resume an interrupted run")
    parser.add_argument(
        "--price", type=float, help="current share price, if not otherwise available"
    )
    parser.add_argument("--company-name", help="company name (improves registry lookups)")
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument("--report-out", type=Path, help="write the report to a file")
    parser.add_argument(
        "--portfolio",
        type=Path,
        help=(
            "JSON file with your position (shares, cost_basis, portfolio_value, stop_loss, "
            "tax_notes). Read ONLY by the portfolio step, which runs after the blind verdict "
            "is fixed; no research agent ever sees it."
        ),
    )
    parser.add_argument("--db", type=Path, help="override the SQLite path")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--document-first-literature",
        action="store_true",
        help=(
            "Phase 4.2A, default OFF: acquire ONE explicit literature reference "
            "(--literature-pmid XOR --literature-nct-id) via the Document-First Literature path "
            "(AcquisitionExecutor -> PubMedLiteratureAdapter -> DocumentStore -> Evidence/Chunk "
            "Projection) and feed the result into this run alongside the existing structured "
            "collectors, before adversarial discovery. Requires --live and exactly one of "
            "--literature-pmid/--literature-nct-id; never combinable with --fixtures/--corpus; "
            "requires IRA_NCBI_TOOL/IRA_NCBI_EMAIL to be configured in the environment "
            "(IRA_NCBI_API_KEY is optional). An explicit reference is never promoted to 'the "
            "current program' for the ticker -- see research/literature_pipeline_integration.py."
        ),
    )
    parser.add_argument(
        "--literature-pmid", metavar="PMID", help="explicit PMID for --document-first-literature"
    )
    parser.add_argument(
        "--literature-nct-id",
        metavar="NCT_ID",
        help="explicit NCT ID for --document-first-literature",
    )
    parser.add_argument(
        "--literature-max-articles",
        type=int,
        default=LITERATURE_DEFAULT_MAX_ARTICLES,
        help=(
            f"--document-first-literature NCT discovery mode only: max PMIDs to process "
            f"(default {LITERATURE_DEFAULT_MAX_ARTICLES})"
        ),
    )
    parser.add_argument(
        "--literature-max-fulltext-fetches",
        type=int,
        default=LITERATURE_DEFAULT_MAX_FULLTEXT_FETCHES,
        help=(
            "--document-first-literature: max Europe PMC full-text fetches (default "
            f"{LITERATURE_DEFAULT_MAX_FULLTEXT_FETCHES}; a single explicit PMID is additionally "
            "capped at 1 regardless of this value)"
        ),
    )
    parser.add_argument(
        "--adaptive-literature",
        action="store_true",
        help=(
            "Phase 4.3J, default OFF: when combined with --live (and never with --fixtures/"
            "--corpus -- that combination is REJECTED at startup, exit code 2, before any "
            "ticker is researched), inject a real NCBI/Europe PMC transport into Pipeline so "
            "Phase 4.3G's automatically-resolved Adaptive Literature Acquisition plan (if "
            "READY) can execute. Requires IRA_NCBI_TOOL/IRA_NCBI_EMAIL in the environment; "
            "without them, IF a READY plan is actually reached for a given ticker, the "
            "acquisition is REFUSED (zero HTTP for this step only -- not the same as zero "
            "communication for the whole run) and that ticker's run reports RESEARCH STATUS: "
            "BLOCKED_PENDING_VERIFICATION with no action label. This credential check is NOT "
            "performed at startup, since whether any ticker's plan ever reaches READY cannot "
            "be known before research runs. Never shares a transport with "
            "--document-first-literature or across tickers; never promoted over an explicit "
            "PMID/NCT reference used this invocation (SKIPPED_EXPLICIT_OVERRIDE)."
        ),
    )
    return parser


class AdaptiveLiteratureFlagError(ValueError):
    """Phase 4.3J: raised by ``validate_adaptive_literature_flags`` for an
    invalid --adaptive-literature flag combination -- distinguishable
    from a generic ``ValueError`` so ``main()`` can catch it and print/
    exit(2) without a traceback, exactly like
    ``LiteraturePipelineRequestError``. Every raise site fires strictly
    BEFORE ``settings``/the DB/the ``HttpClient``/any ticker is even
    touched -- there is no silent fall-through to the pre-existing
    NOT_ATTEMPTED_NO_RUNNER no-op for an invalid combination; the run
    never starts at all."""


def validate_adaptive_literature_flags(
    *, enabled: bool, live: bool, use_fixtures: bool, use_corpus: bool,
) -> None:
    """Phase 4.3J: validate --adaptive-literature's own flag combination,
    independently of ``validate_literature_pipeline_request`` (the
    explicit PMID/NCT path's own validator) -- the two flags are
    orthogonal and this never duplicates or re-implements that one.

    ``enabled=False`` (the default; every pre-4.3J invocation) is a
    no-op -- returns immediately, so this function changes nothing about
    any existing flag combination that does not pass --adaptive-literature.

    ``enabled=True`` is rejected (``AdaptiveLiteratureFlagError``) when:

    1. ``--fixtures``/``--corpus`` is also set.
       The function that re-verifies a READY plan and runs the bridge
       (``orchestrator.adaptive_literature_step``'s own dependency, in
       ``research/``) always calls ``validate_literature_pipeline_request`` with
       ``use_fixtures=False, use_corpus=False`` HARD-CODED -- it never
       inspects this CLI invocation's own --fixtures/--corpus state.
       Without this check
       here, a real transport would be constructed and handed to Pipeline
       even during a --fixtures run, and the ONLY thing stopping it from
       making a real NCBI/Europe PMC request would be whether a given
       ticker's own Program Identity happens to resolve to READY --
       exactly the kind of silent, data-dependent behavior change this
       phase's own design review was written to prevent (CLAUDE.md:
       fixture runs never make real requests, unconditionally).
    2. ``--live`` is NOT also set. This path can never reach the network
       without it, mirroring the explicit PMID/NCT path's own identical
       rule.

    Deliberately NEVER checks NCBI credential configuration here --
    whether any ticker's plan ever reaches READY depends on that
    ticker's own live research this invocation has not yet done, so a
    credential check at this point would reject flag combinations that
    might never even need a credential. See ``run_one()``'s own runtime
    REFUSED handling for that case instead.
    """
    if not enabled:
        return
    if use_fixtures or use_corpus:
        raise AdaptiveLiteratureFlagError(
            "--adaptive-literature cannot be combined with --fixtures/--corpus -- refusing "
            "before any ticker is researched rather than silently taking the "
            "NOT_ATTEMPTED_NO_RUNNER no-op path"
        )
    if not live:
        raise AdaptiveLiteratureFlagError(
            "--adaptive-literature requires --live -- refusing before any ticker is "
            "researched rather than silently taking the NOT_ATTEMPTED_NO_RUNNER no-op path"
        )


#: Phase 4.3J: the only environment variable NAMES ever read for
#: --adaptive-literature's own transport injection -- never the full
#: process environment. Mirrors research/ncbi_credentials.py's own
#: closed set exactly (that module is the single place these three
#: names are defined; never redefined here).
_ADAPTIVE_ENV_VAR_NAMES = (NCBI_TOOL_ENV_VAR, NCBI_EMAIL_ENV_VAR, NCBI_API_KEY_ENV_VAR)


def build_adaptive_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Phase 4.3J: a minimal dict carrying ONLY the (at most) three NCBI
    courtesy-identification variables, extracted from ``environ`` --
    never the full process environment. This is deliberately narrower
    than simply passing ``os.environ`` through: ``resolve_ncbi_
    credentials`` (research/ncbi_credentials.py) only ever reads these
    three keys regardless, but keeping the injected mapping itself this
    narrow means no OTHER secret (e.g. IRA_ANTHROPIC_API_KEY) is even
    reachable from this code path, now or if it is ever extended. Values
    are never logged, printed, or placed in any diagnostic by this
    function or anything downstream of it (research/
    literature_pipeline_integration.py's own module docstring; also
    independently redacted from any log line by ``config.Settings.
    secret_values()``, which already includes the SAME env vars read
    under ``config.Settings.ncbi_email``/``ncbi_api_key``)."""
    return {name: environ[name] for name in _ADAPTIVE_ENV_VAR_NAMES if name in environ}


def build_adaptive_http_client() -> AllowlistedHttpClient:
    """Phase 4.3J: one fresh, independent ``AllowlistedHttpClient`` for
    the Adaptive Literature Acquisition step -- never shared with the
    explicit PMID/NCT path's own client (that path builds its own,
    internally, inside ``run_literature_pipeline_acquisition`` when it is
    not given one; the two paths are mutually exclusive in effect per
    plan.status's SKIPPED_EXPLICIT_OVERRIDE rule, so sharing would add
    coupling for no benefit) and never shared across tickers (a caller
    must call this ONCE PER TICKER -- see run_one() -- since this
    client's own ``max_requests``/cache/attempt counters are
    instance-scoped; sharing one instance across --compare/--screen
    targets would let an earlier ticker's attempts silently consume a
    later ticker's budget).

    Every setting below is read from research/literature_pipeline_
    integration.py's own constants (never redefined/duplicated here):
    the same user agent, allowlisted hosts (research/
    literature_acquisition_adapter.ALLOWED_HOSTS), timeout, rate limit,
    retry count, and redirect cap the explicit PMID/NCT path's own
    internally-constructed client uses. ``max_requests`` is fixed to
    ``READY_MAX_REQUESTS`` (6) -- the physical-attempt (retry-inclusive)
    cap the bridge that re-verifies a READY plan itself independently
    verifies that same plan's own ``max_requests`` field already equals,
    so a mismatch here would make every READY plan REFUSED rather than
    silently using a different budget. ``out_dir`` is left at its
    default (``None``): no Capture Manifest, no LAST_RUN marker is ever
    written by this call, mirroring research/literature_pipeline_
    integration.py's own stated boundary for the explicit path."""
    return AllowlistedHttpClient(
        user_agent=LITERATURE_PIPELINE_USER_AGENT,
        allowed_hosts=LITERATURE_ALLOWED_HOSTS,
        timeout=ADAPTIVE_TIMEOUT_SECONDS,
        rate_limit_rps=ADAPTIVE_RATE_LIMIT_RPS,
        max_retries=ADAPTIVE_MAX_RETRIES,
        max_requests=READY_MAX_REQUESTS,
        max_redirects=ADAPTIVE_MAX_REDIRECTS,
        source="cli_adaptive_literature",
        additional_secret_query_params=frozenset({"tool"}),
        save_failed_responses=True,
    )


def collect(
    ticker: str,
    company_name: str,
    *,
    settings,
    http: HttpClient,
    use_fixtures: bool,
) -> tuple[list[CollectionResult], dict]:
    """Run the collectors.  Returns results plus any fixture metadata."""
    metadata: dict = {}
    if use_fixtures:
        fixtures = FixtureCollector(settings.fixture_dir)
        metadata = fixtures.metadata(ticker)
        return [fixtures.collect(ticker, company_name)], metadata

    results = [
        SecEdgarCollector(http).collect(ticker, company_name),
        ClinicalTrialsCollector(http).collect(ticker, company_name),
        FdaCollector(http).collect(ticker, company_name),
    ]
    return results, metadata


def build_research_stack(ticker: str, args: argparse.Namespace, settings, llm):
    """Assemble the research providers for this run (ADR 0005).

    Order matters: a captured corpus answers from documents already in hand, and
    live web research is the general channel. Whichever serves a query is
    recorded on the result, so the report can say how each fact was obtained.
    """
    providers: list[ResearchProvider] = []
    if args.corpus:
        providers.append(CorpusResearchProvider(settings.corpus_dir, ticker))
    if args.live and llm is not None:
        providers.append(
            AnthropicWebResearchProvider(llm, research_effort=args.research_effort)
        )
    if not providers:
        return NullResearchProvider()
    return providers[0] if len(providers) == 1 else CompositeResearchProvider(providers)


def run_one(
    ticker: str,
    args: argparse.Namespace,
    settings,
    repo: Repository,
    http: HttpClient,
    *,
    literature_request: LiteraturePipelineRequest | None = None,
) -> ResearchResult:
    fixtures = FixtureCollector(settings.fixture_dir)
    corpus = CorpusResearchProvider(settings.corpus_dir, ticker)

    metadata: dict = {}
    capture_info: dict = {}
    if args.fixtures:
        metadata = fixtures.metadata(ticker)
    elif args.corpus:
        metadata = corpus.metadata(ticker)
        capture_info = corpus.capture_info(ticker)

    company_name = args.company_name or metadata.get("company_name") or ticker.upper()
    price = args.price if args.price is not None else metadata.get("price")

    budget = LLMBudget(max_total_tokens=args.token_budget)
    llm = None
    if args.llm:
        llm = LLMClient(
            api_key=settings.anthropic_api_key,
            model=args.llm_model,
            effort=args.llm_effort,
            budget=budget,
        )
        usable, reason = llm.available()
        if not usable:
            log.warning("--llm requested but unavailable: %s", reason)

    research = build_research_stack(ticker, args, settings, llm)
    chunks: list = []
    adversarial = None

    # Requirement F: collection runs BEFORE adversarial web search (moved
    # ahead of it deliberately) so the expensive web-search pass can see
    # which required domains a structured collector (SEC EDGAR /
    # ClinicalTrials.gov / openFDA) already directly, auditably covers and
    # skip searching them again -- a live measurement showed ~22k actual
    # tokens per web-search call even at low effort, so six independent
    # required-domain searches alone could exhaust a 60k discovery quota.
    if args.corpus:
        documents = corpus.documents(ticker)
        collector = DocumentCollector(
            documents,
            provenance=Provenance.CAPTURED,
            note=(
                f"CAPTURED corpus replay: {capture_info.get('document_count', 0)} document(s) "
                f"captured {capture_info.get('captured_at', 'UNKNOWN')} via "
                f"{capture_info.get('captured_via', 'UNKNOWN')}. Real data about a real issuer, "
                "but NOT fetched live at run time."
            ),
        )
        results = [collector.collect(ticker, company_name)]
        chunks = collector.chunks
    else:
        results, _ = collect(
            ticker,
            company_name,
            settings=settings,
            http=http,
            use_fixtures=args.fixtures,
        )

    # Phase 4.2A: default-OFF Literature Document-First acquisition, run
    # AFTER the existing structured collectors (above) and BEFORE
    # adversarial web search (below) -- new Literature Evidence/Chunk
    # Projection facts must be present before FactCollector/Evidence
    # Integrity/domain agents run inside Pipeline.run(), never added
    # afterward as late evidence. `literature_request` is `None` for every
    # run that did not pass --document-first-literature (validated once in
    # `main()`), so this block never CALLS `run_literature_pipeline_
    # acquisition` and never constructs a DocumentStore/AcquisitionExecutor/
    # HTTP client for every pre-existing invocation of this CLI -- this is
    # NOT the same claim as "zero extra imports": the
    # `literature_pipeline_integration` module (and everything it imports)
    # is loaded unconditionally at CLI import time, by this file's own
    # top-level `from .research.literature_pipeline_integration import
    # ...` -- see that module's own docstring for the exact, narrow ways a
    # flag-OFF run's Python-level footprint differs from before this phase
    # while its observable output (facts/chunks/verdict/report/--json)
    # does not.
    literature_bundle = None
    if literature_request is not None:
        literature_bundle = run_literature_pipeline_acquisition(literature_request, ticker=ticker)
        if literature_bundle.collection_result is not None:
            results = [*results, literature_bundle.collection_result]
        if literature_bundle.chunks:
            chunks = [*chunks, *literature_bundle.chunks]

    if args.adversarial:
        # Requirement M2: search results are discovery evidence, never facts.
        # They are NOT merged into the document set that feeds extraction --
        # adversarial.discovery (SearchQueryRecord/SearchHit) is what the Kill
        # Agent's prompt context and the completeness gate read instead. A
        # search snippet must never become a Fact.
        #
        # `research` is the single provider object for this run -- built once
        # by build_research_stack() above, whether that is a corpus replay, the
        # live Anthropic provider, a composite of both, or NullResearchProvider
        # when neither --corpus nor --live is set.
        #
        # build_plan() decides per-QUERY-INTENT skipping from `results`
        # (requirement E) -- never per-domain: a collector merely touching a
        # domain (e.g. a zero-result Drugs@FDA search) must never silently
        # drop that domain's entire adversarial query set. No unresolved
        # questions exist yet at this point in the run (Stage 3 domain
        # agents, which raise them, have not executed) -- passed as ()
        # deliberately; with today's collectors this is moot regardless,
        # since `_COLLECTOR_REDUNDANT_INTENTS` never marks a real collector's
        # intent as redundant in the first place (requirements B/C).
        adversarial = run_adversarial_search(
            research,
            build_plan(ticker, company_name, collection_results=results),
            llm=llm,
            run_id=ticker, ticker=ticker, research_effort=args.research_effort,
        )

    mode = "standard"
    if args.full_dd:
        mode = "full-dd"
    elif args.kill_test:
        mode = "kill-test"
    elif args.catalyst:
        mode = "catalyst"
    elif args.update:
        mode = "update"

    # Phase 4.3J: --adaptive-literature's own runner injection. Gated on
    # the SAME condition validate_adaptive_literature_flags() already
    # enforced at startup (args.adaptive_literature and args.live and not
    # fixtures/corpus) -- re-checked here, per ticker, rather than trusted
    # from that one startup call, so a future caller of run_one() that
    # skips main()'s own validation still gets the same fixtures/corpus
    # protection. A fresh AllowlistedHttpClient is built PER TICKER (never
    # shared across --compare/--screen targets, never shared with the
    # explicit PMID/NCT path's own client) -- see build_adaptive_http_
    # client()'s own docstring. Constructing it makes no network call by
    # itself; this is safe to do even on --resume, since whether it is
    # ever actually used is judged entirely inside Pipeline.run()/
    # run_adaptive_literature_step() from persisted checkpoints, never
    # re-decided here.
    enable_adaptive = args.adaptive_literature and args.live and not args.fixtures and not args.corpus
    adaptive_http_client = build_adaptive_http_client() if enable_adaptive else None
    adaptive_env = build_adaptive_env(os.environ) if enable_adaptive else None

    pipeline = Pipeline(
        repo,
        build_search_provider(http, settings),
        today=date.today(),
        stale_after_days=settings.stale_after_days,
        research=research,
        llm=llm,
        adversarial=adversarial,
        chunks=chunks,
        capture_info=capture_info,
        direct_acquisition_info=bundle_diagnostics(
            literature_bundle, enabled=literature_request is not None
        ),
        adaptive_http_client=adaptive_http_client,
        adaptive_env=adaptive_env,
    )
    return pipeline.run(
        ticker,
        company_name,
        results,
        mode=mode,
        price=price,
        aliases=metadata.get("aliases", ()),
        # Read only by the portfolio step, which runs after the blind verdict
        # is fixed. Every research agent's isolation policy denies it; see
        # orchestrator/isolation.py.
        user_preferences=load_portfolio(args.portfolio),
        offline=not args.live,
        run_id=args.resume or None,
        resume=bool(args.resume),
    )


def load_portfolio(path: Path | None) -> dict | None:
    """Load the user's position, if one was supplied."""
    if path is None:
        return None
    if not path.is_file():
        log.warning("portfolio file not found: %s", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("malformed portfolio file %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _token_diagnostics(result: ResearchResult) -> dict:
    """Where the tokens went (requirement G): by stage, by agent, and search/
    fetch execution counts, so the next run's spend is visible up front."""
    budget = result.llm_budget
    adversarial = result.adversarial
    escalation = result.escalation
    return {
        "remaining": budget.remaining if budget else None,
        "by_stage": budget.stage_summary() if budget else {},
        "by_agent": budget.by_agent() if budget else {},
        "search": {
            "executed": adversarial.executed if adversarial else 0,
            "unexecuted": len(adversarial.unexecuted) if adversarial else 0,
            "unexecuted_due_to_budget": (
                len(adversarial.unexecuted_due_to_budget) if adversarial else 0
            ),
            "deduplicated": len(adversarial.deduplicated) if adversarial else 0,
            # Requirement F: mandatory-template queries skipped because a
            # structured collector already directly covered that domain --
            # never a budget cut, never UNSEARCHED.
            "skipped_due_to_direct_coverage": (
                len(adversarial.skipped_due_to_direct_coverage) if adversarial else 0
            ),
        },
        # Requirement F: per-batch cost/outcome diagnostics for the batched
        # required-domain research pass -- how many underlying calls the
        # mandatory bear/bull work actually needed, and what each cost.
        "batches": [
            {
                "batch_id": diag.batch_id,
                "intent_ids": diag.intent_ids,
                "domains": diag.domains,
                "server_tool_uses": diag.server_tool_uses,
                "prompt_tokens": diag.prompt_tokens,
                "output_tokens": diag.output_tokens,
                "actual_total_tokens": diag.actual_total_tokens,
                "search_result_count": diag.search_result_count,
                "completed_intent_ids": diag.completed_intent_ids,
                "incomplete_intent_ids": diag.incomplete_intent_ids,
                "ambiguous_results": diag.ambiguous_results,
                "retry_rounds": diag.retry_rounds,
            }
            for diag in (adversarial.batch_diagnostics if adversarial else [])
        ],
        "fetch": {
            "attempted": escalation.fetches_attempted if escalation else 0,
            "failed": escalation.fetches_failed if escalation else 0,
            # Requirement 2: a pre-send budget/availability rejection is
            # never lumped into "the fetch API failed" -- this is the subset
            # of `failed` above that never reached the network.
            "not_sent": escalation.fetches_not_sent if escalation else 0,
            # Requirement E: per-fetch audit trail entry count -- the report's
            # escalation appendix carries the full detail (subject, url,
            # authority, candidate rank, outcome, failure reason, tokens,
            # body obtained/answered, supporting sentence).
            "audit_entries": len(escalation.fetch_log) if escalation else 0,
        },
    }


def result_to_json(result: ResearchResult) -> dict:
    verdict = result.verdict
    card = result.scorecard
    payload = {
        "run_id": result.context.run_id,
        "ticker": result.context.ticker,
        "status": str(result.context.status),
        "used_fixtures": result.context.use_fixtures,
        "evidence_confidence": (
            result.confidence_breakdown.score if result.confidence_breakdown else None
        ),
        "action": (str(verdict.action) if verdict and verdict.action else None),
        "research_status": str(verdict.research_status) if verdict else None,
        "headline": verdict.headline if verdict else None,
        "reasoning": list(verdict.reasoning) if verdict else [],
        "thesis_breakers": list(verdict.thesis_breakers) if verdict else [],
        "critical_red_flags": list(verdict.critical_red_flags) if verdict else [],
        "caveats": list(verdict.caveats) if verdict else [],
        "blocking_verification_required": (
            list(verdict.blocking_verification_required) if verdict else []
        ),
        "blocked": bool(verdict.blocked) if verdict else None,
        "blocked_reason": verdict.blocked_reason if verdict else "",
        "evidence_sufficiency": (
            {
                "sufficient": result.evidence_sufficiency.sufficient,
                "decision_grade_fact_total": result.evidence_sufficiency.decision_grade_fact_total,
                "domains": {
                    str(domain): {
                        "search_status": str(entry.search_status),
                        "evidence_sufficiency_status": str(entry.evidence_sufficiency_status),
                        "decision_grade_fact_count": entry.decision_grade_fact_count,
                        "total_fact_count": entry.total_fact_count,
                    }
                    for domain, entry in result.evidence_sufficiency.domains.items()
                },
            }
            if result.evidence_sufficiency
            else None
        ),
        "research_coverage": (
            {
                domain: entry.status
                for domain, entry in ((str(d), e) for d, e in result.completeness.coverage.items())
            }
            if result.completeness
            else {}
        ),
        "llm_agents_used": result.llm_agents_used,
        "llm_tokens": result.llm_budget.used_total if result.llm_budget else 0,
        "token_diagnostics": _token_diagnostics(result),
        "quarantined_sources": [
            {"source_id": q.source_id, "url": q.url, "field": q.field, "value": q.value}
            for q in result.quarantined_sources
        ],
        "capture_info": result.capture_info,
        "max_kill_level": str(verdict.kill_gate.max_level) if verdict else None,
        "kill_gate": [a.to_row() for a in verdict.kill_gate.assessments] if verdict else [],
        "scores": card.scores if card else {},
        "capped_by_kill_gate": list(card.capped_by_kill_gate) if card else [],
        "fact_count": len(result.bus.facts),
        "contradictions": len(result.bus.contradictions),
        "failures": result.failures,
        "thesis_version": result.thesis_version,
        "portfolio_guidance": result.portfolio_guidance or None,
    }
    # Phase 4.2A correction 1: omit the key entirely when empty (the
    # default-OFF case -- see ResearchResult.direct_acquisition_info's own
    # docstring and bundle_diagnostics(None, enabled=False)) so a run that
    # never used --document-first-literature gets the exact same --json
    # keys this repository produced before Phase 4.2A existed.
    if result.direct_acquisition_info:
        payload["direct_acquisition_info"] = result.direct_acquisition_info
    # Phase 4.3J: same "omit when empty" contract as direct_acquisition_info
    # above -- adaptive_literature_info() returns {} for every outcome that
    # occurs identically whether or not --adaptive-literature was ever
    # passed (see its own docstring in reporting/report.py), so a run that
    # never used the flag, and never resumed a run_id an earlier invocation
    # used it for, gets the exact same --json keys as before this phase.
    info = adaptive_literature_info(result)
    if info:
        payload["adaptive_literature_info"] = info
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Phase 4.2A: validated BEFORE settings/logging/DB/HttpClient are even
    # constructed, so a rejected --document-first-literature invocation
    # makes zero HTTP attempts and touches no other resource either. A
    # flag-off invocation with no literature identifiers (the pre-existing
    # default) returns None here and this block is otherwise inert.
    try:
        literature_request = validate_literature_pipeline_request(
            enabled=args.document_first_literature,
            pmid=args.literature_pmid,
            nct_id=args.literature_nct_id,
            live=args.live,
            use_fixtures=args.fixtures,
            use_corpus=args.corpus,
            max_articles=args.literature_max_articles,
            max_fulltext_fetches=args.literature_max_fulltext_fetches,
        )
    except LiteraturePipelineRequestError as exc:
        print(f"--document-first-literature: {exc}", file=sys.stderr)
        return 2

    # Phase 4.3J: validated BEFORE settings/logging/DB/HttpClient are even
    # constructed, same placement as the explicit PMID/NCT check above --
    # an invalid --adaptive-literature combination never silently falls
    # through to the pre-existing NOT_ATTEMPTED_NO_RUNNER no-op; the run
    # never starts. A flag-off invocation (the default) is a no-op here.
    try:
        validate_adaptive_literature_flags(
            enabled=args.adaptive_literature,
            live=args.live,
            use_fixtures=args.fixtures,
            use_corpus=args.corpus,
        )
    except AdaptiveLiteratureFlagError as exc:
        print(f"--adaptive-literature: {exc}", file=sys.stderr)
        return 2

    settings = get_settings()
    if args.db:
        settings.db_path = args.db
    settings.ensure_dirs()

    run_id = new_run_id()
    setup_logging(
        settings.log_dir,
        run_id,
        secrets=settings.secret_values(),
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    targets: list[str] = []
    if args.compare:
        targets = [t.upper() for t in args.compare]
    elif args.screen:
        fixtures = FixtureCollector(settings.fixture_dir)
        targets = fixtures.available() if args.fixtures else []
        if not targets:
            print(
                "--screen needs a candidate universe. With --fixtures it screens the fixture "
                "set; a live universe source is not configured.",
                file=sys.stderr,
            )
            return 2
    elif args.ticker:
        targets = [args.ticker.upper()]
    else:
        build_parser().print_help()
        return 2

    if literature_request is not None and len(targets) != 1:
        print(
            "--document-first-literature supports exactly one ticker per invocation in this "
            "phase (an explicit PMID/NCT ID reference is not automatically shared across "
            "--compare/--screen targets).",
            file=sys.stderr,
        )
        return 2

    http = HttpClient(
        user_agent=settings.sec_user_agent,
        timeout=settings.http_timeout,
        max_retries=settings.http_max_retries,
        rate_limit_rps=settings.rate_limit_rps,
        cache_dir=settings.cache_dir,
        offline=not args.live,
    )
    conn = open_db(settings.db_path)
    repo = Repository(conn)

    results: list[ResearchResult] = []
    exit_code = 0
    try:
        for ticker in targets:
            result = run_one(
                ticker, args, settings, repo, http, literature_request=literature_request
            )
            results.append(result)
            if result.context.status != RunStatus.COMPLETE:
                exit_code = 1
    finally:
        conn.close()

    if args.as_json:
        print(json.dumps([result_to_json(r) for r in results], indent=2, default=str))
        return exit_code

    for result in results:
        report = render_report(result)
        print(report)
        if args.report_out:
            path = (
                args.report_out
                if len(results) == 1
                else args.report_out.with_name(
                    f"{args.report_out.stem}_{result.context.ticker}{args.report_out.suffix}"
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            print(f"\n[report written to {path}]", file=sys.stderr)

    if len(results) > 1:
        print(_comparison_table(results))

    return exit_code


def _comparison_table(results: list[ResearchResult]) -> str:
    lines = ["", "=" * 78, "COMPARISON", "=" * 78]
    lines.append(
        "Compared on evidence confidence and worst kill level FIRST. There is no combined "
        "score, and a ranking by upside alone is not offered."
    )
    lines.append("")
    lines.append(f"  {'ticker':<10} {'conf':>5} {'kill':>5} {'action':<18} {'status'}")
    for result in results:
        verdict = result.verdict
        confidence = result.confidence_breakdown.score if result.confidence_breakdown else 0.0
        lines.append(
            f"  {result.context.ticker:<10} {confidence:>5.1f} "
            f"{str(verdict.kill_gate.max_level) if verdict else '-':>5} "
            f"{str(verdict.action) if verdict else '-':<18} {result.context.status}"
        )
    return "\n".join(lines)
