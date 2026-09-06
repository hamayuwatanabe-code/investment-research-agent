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
from .collectors.fda import FdaCollector
from .collectors.fixtures import FixtureCollector
from .collectors.http import HttpClient
from .collectors.search import build_search_provider
from .collectors.sec_edgar import SecEdgarCollector
from .config import get_settings
from .logging_setup import setup_logging
from .orchestrator.pipeline import Pipeline, ResearchResult, new_run_id
from .reporting.report import render_report
from .schemas.enums import RunStatus
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
    parser.add_argument("--live", action="store_true", help="allow outbound network calls")
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


def run_one(
    ticker: str,
    args: argparse.Namespace,
    settings,
    repo: Repository,
    http: HttpClient,
) -> ResearchResult:
    fixtures = FixtureCollector(settings.fixture_dir)
    metadata = fixtures.metadata(ticker) if args.fixtures else {}
    company_name = args.company_name or metadata.get("company_name") or ticker.upper()
    price = args.price if args.price is not None else metadata.get("price")

    results, _ = collect(
        ticker,
        company_name,
        settings=settings,
        http=http,
        use_fixtures=args.fixtures,
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
        "action": str(verdict.action) if verdict else None,
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
