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
import sys
from collections.abc import Sequence
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
from .reporting.report import render_report
from .research.adversarial import build_plan, run_adversarial_search
from .research.anthropic_web import AnthropicWebResearchProvider
from .research.corpus import CorpusResearchProvider
from .research.provider import (
    CompositeResearchProvider,
    NullResearchProvider,
    ResearchProvider,
)
from .schemas.enums import Provenance, RunStatus
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
    return parser


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
        "fetch": {
            "attempted": escalation.fetches_attempted if escalation else 0,
            "failed": escalation.fetches_failed if escalation else 0,
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
    return {
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
            result = run_one(ticker, args, settings, repo, http)
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
