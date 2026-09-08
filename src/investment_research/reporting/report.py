"""Final report renderer.

Section order is fixed by requirement 22 and is itself a piece of the design:
the verdict, the evidence confidence, the red flags and *what would kill the
thesis* all come before the bull case.  A reader who stops after two sections
still gets the disconfirming information.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..orchestrator.isolation import Channel
from ..orchestrator.pipeline import ResearchResult
from ..schemas.enums import (
    UNKNOWN,
    KillLevel,
    Materiality,
    Provenance,
    RunStatus,
)
from ..schemas.evaluation import SCORE_DIMENSIONS
from .citations import CitationValidator

#: Sections rendered in each mode. The full report is always the default; the
#: narrow modes exist so a focused question gets a focused answer, and every one
#: of them keeps sections 1-4 (verdict, confidence, red flags, thesis breakers).
MODE_SECTIONS: dict[str, tuple[int, ...]] = {
    "kill-test": (1, 2, 3, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20),
    "catalyst": (1, 2, 3, 4, 11, 14, 15, 16, 17, 19, 20),
}

SECTION_ORDER = (
    "1. Verdict",
    "2. Evidence Confidence",
    "3. Critical Red Flags",
    "4. What Would Kill The Thesis",
    "5. Verified Facts",
    "6. Contradictions",
    "7. Bull Case",
    "8. Bear Case",
    "9. Regulatory",
    "10. Capital Structure / Dilution",
    "11. Science / Technology",
    "12. Competition",
    "13. Valuation",
    "14. Catalysts (JST)",
    "15. Scenarios",
    "16. Scores",
    "17. Immediate Action",
    "18. What Changed Since Previous Run",
    "19. Unresolved Questions",
    "20. Sources",
)


def _money(value: Any) -> str:
    if value is None or value == UNKNOWN:
        return UNKNOWN
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f}{suffix}"
    return f"${value:,.2f}"


def _num(value: Any) -> str:
    if value is None or value == UNKNOWN:
        return UNKNOWN
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


class ReportRenderer:
    def __init__(self, result: ResearchResult) -> None:
        self.result = result
        self.bus = result.bus
        self.validator = CitationValidator(list(self.bus.sources), list(self.bus.facts))
        self.citation_issues: list[str] = []

    # -- helpers -----------------------------------------------------------
    def _channel(self, name: str) -> dict[str, Any]:
        evaluation = self.bus.channels.get(name)
        return evaluation.payload if evaluation else {}

    def _summary(self, name: str) -> str:
        evaluation = self.bus.channels.get(name)
        return evaluation.summary if evaluation else "Not produced (agent did not run or failed)."

    @staticmethod
    def _wrap(text: str, width: int = 76, indent: str = "") -> str:
        import textwrap

        return (
            "\n".join(
                textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)
            )
            or text
        )

    def _safe(self, text: str) -> str:
        clean, report = self.validator.sanitize(text)
        if not report.clean:
            self.citation_issues.append(
                f"unknown urls={report.unknown_urls} fact_ids={report.unknown_fact_ids} "
                f"accessions={report.malformed_accessions} nct={report.malformed_nct}"
            )
        return clean

    # -- rendering ---------------------------------------------------------
    def render(self) -> str:
        ctx = self.result.context
        lines: list[str] = []
        add = lines.append

        add("=" * 78)
        add(f"INVESTMENT RESEARCH REPORT -- {ctx.ticker}")
        add(f"Company: {ctx.company_name}")
        add(f"Run ID: {ctx.run_id}   Mode: {ctx.mode}")
        add(f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        add("=" * 78)

        if ctx.use_fixtures:
            add("")
            add("!" * 78)
            add("SYNTHETIC FIXTURE DATA -- THIS IS NOT REAL RESEARCH.")
            add("Every source below uses the fixture:// scheme and describes no real issuer.")
            add("This run exists to exercise the pipeline, not to inform a trading decision.")
            add("!" * 78)

        if ctx.status != RunStatus.COMPLETE:
            add("")
            add("*" * 78)
            add(f"*** {ctx.status} ***")
            add("One or more agents or collectors did not complete. This analysis is partial")
            add("and must not be treated as a finished assessment.")
            for failure in self.result.failures[:12]:
                add(f"  - {failure}")
            if len(self.result.failures) > 12:
                add(f"  ... and {len(self.result.failures) - 12} more (see logs)")
            add("*" * 78)

        renderers = (
            self._section_1_verdict,
            self._section_2_confidence,
            self._section_3_red_flags,
            self._section_4_thesis_breakers,
            self._section_5_verified_facts,
            self._section_6_contradictions,
            self._section_7_bull,
            self._section_8_bear,
            self._section_9_regulatory,
            self._section_10_capital,
            self._section_11_science,
            self._section_12_competition,
            self._section_13_valuation,
            self._section_14_catalysts,
            self._section_15_scenarios,
            self._section_16_scores,
            self._section_17_action,
            self._section_18_changes,
            self._section_19_unresolved,
            self._section_20_sources,
        )
        wanted = MODE_SECTIONS.get(ctx.mode)
        if wanted is not None:
            add("")
            add(
                f"[{ctx.mode} mode: showing sections "
                f"{', '.join(str(n) for n in wanted)} of 20. "
                "Run without the mode flag for the full report.]"
            )
        for number, renderer in enumerate(renderers, start=1):
            if wanted is None or number in wanted:
                add(renderer())

        # Phase 2 appendices. Defined as module-level functions taking the
        # renderer, so the class stays readable and the sections stay grouped
        # with the concerns they document.
        add(_section_research_provenance(self))
        add(_section_search_coverage(self))
        add(_section_evidence_sufficiency(self))
        add(_section_escalation(self))
        add(_section_traceability(self))
        add(_section_cost(self))

        if self.citation_issues:
            add("")
            add("-" * 78)
            add("CITATION VALIDATION WARNINGS")
            add("-" * 78)
            add(
                "References marked [UNVERIFIED CITATION] could not be matched to a document "
                "retrieved during this run. They are shown rather than removed."
            )
            for issue in self.citation_issues[:10]:
                add(f"  - {issue}")

        return "\n".join(lines)

    # -- sections ----------------------------------------------------------
    def _header(self, title: str) -> str:
        return f"\n{'-' * 78}\n{title}\n{'-' * 78}"

    def _section_1_verdict(self) -> str:
        from ..schemas.enums import ResearchStatus

        verdict = self.result.verdict
        out = [self._header(SECTION_ORDER[0])]
        if verdict is None:
            out.append("RESEARCH_STATUS: BLOCKED_PENDING_VERIFICATION")
            out.append("FINAL_ACTION: NONE")
            out.append("NO VERDICT: the blind judge did not complete. Treat as INCOMPLETE.")
            return "\n".join(out)

        out.append(f"RESEARCH_STATUS: {verdict.research_status}")
        if verdict.research_status is not ResearchStatus.COMPLETE:
            out.append("FINAL_ACTION: NONE")
        else:
            out.append(f"FINAL_ACTION: {verdict.action}")

        # Every K3+ finding, confirmed and provisional alike, named explicitly
        # -- "Potential X: K5 PROVISIONAL" is a legitimate and required output
        # (requirement DG2/DG8), never collapsed into silence or into a
        # confident label it has not earned.
        for assessment in sorted(
            verdict.kill_gate.assessments, key=lambda a: -a.level.level
        ):
            if assessment.level.level < 3:
                continue
            if assessment.confirmation.value == "CONFIRMED":
                out.append(f"{assessment.category}: {assessment.level} CONFIRMED")
            else:
                out.append(f"POTENTIAL_{assessment.category}: {assessment.level} PROVISIONAL")

        if verdict.blocked:
            out.append("")
            out.append("FINAL VERDICT: BLOCKED")
            out.append(self._wrap(verdict.blocked_reason))
            if verdict.blocking_verification_required:
                out.append("")
                out.append("BLOCKING_VERIFICATION_REQUIRED:")
                out.extend(f"- {self._safe(r)}" for r in verdict.blocking_verification_required)
            out.append("")
            out.append(
                "No action label is issued. Not AVOID, not WAIT_FOR_EVENT -- an action label "
                "asserts a judgement, and there is not enough research here to support one."
            )
        out.append(f"Judged blind (identity withheld from the judge): {verdict.judged_blind}")
        out.append(f"Worst kill level: {verdict.kill_gate.max_level}")
        out.append(f"Headline: {self._safe(verdict.headline)}")
        for caveat in verdict.caveats:
            out.append(f"CAVEAT: {caveat}")
        out.append("")
        out.append("Reasoning:")
        out.extend(f"  - {self._safe(r)}" for r in verdict.reasoning)
        return "\n".join(out)

    def _section_2_confidence(self) -> str:
        out = [self._header(SECTION_ORDER[1])]
        breakdown = self.result.confidence_breakdown
        if breakdown is None:
            out.append("Evidence confidence could not be computed.")
            return "\n".join(out)
        out.append(f"EVIDENCE CONFIDENCE: {breakdown.score} / 10")
        out.append("")
        out.append(
            "This is reported before the scores and before the action, and it is a separate "
            "question from whether the upside is large."
        )
        out.append("")
        out.append(f"  facts in evidence set        : {breakdown.fact_count}")
        out.append(f"  Tier 1/2 primary sources     : {breakdown.primary_source_ratio:.0%}")
        out.append(f"  decision-grade evidence      : {breakdown.decision_grade_ratio:.0%}")
        out.append(f"  independently corroborated   : {breakdown.corroboration_ratio:.0%}")
        out.append(f"  company claims (uncorrob.)   : {breakdown.company_claim_ratio:.0%}")
        out.append(f"  stale by event date          : {breakdown.stale_ratio:.0%}")
        out.append(f"  blocking unknowns            : {breakdown.blocking_unknowns}")
        out.append(f"  kill categories not searched : {breakdown.unsearched_categories}")
        if breakdown.penalties:
            out.append("")
            out.append("Penalties applied:")
            out.extend(f"  - {p}" for p in breakdown.penalties)
        return "\n".join(out)

    def _section_3_red_flags(self) -> str:
        out = [self._header(SECTION_ORDER[2])]
        verdict = self.result.verdict
        flags = list(verdict.critical_red_flags) if verdict else []
        if not flags:
            out.append("No K3+ kill assessment was produced.")
        out.extend(f"  [!] {self._safe(f)}" for f in flags)

        critical = [f for f in self.bus.risk_flags if f.severity == Materiality.CRITICAL]
        if critical:
            out.append("")
            out.append("Critical risk flags raised by domain agents:")
            for flag in critical:
                out.append(f"  [!] {flag.title}")
                out.append(f"      {self._safe(flag.detail)}")
                if flag.fact_ids:
                    out.append(f"      facts: {', '.join(f for f in flag.fact_ids if f)}")
        return "\n".join(out)

    def _section_4_thesis_breakers(self) -> str:
        out = [self._header(SECTION_ORDER[3])]
        out.append(
            "Stated before the bull case on purpose: these are the findings that would "
            "invalidate an investment thesis here."
        )
        out.append("")
        verdict = self.result.verdict
        breakers = list(verdict.thesis_breakers) if verdict else []
        if not breakers:
            out.append("  (none identified -- which may mean the evidence set is too thin)")
        out.extend(f"  - {self._safe(b)}" for b in breakers)

        kill = self._channel(Channel.KILL)
        findings = kill.get("findings", [])
        if findings:
            out.append("")
            out.append("Kill Agent findings:")
            for finding in sorted(findings, key=lambda f: -KillLevel(f["level"]).level)[:12]:
                marker = "PRIMARY" if finding.get("primary_source") else "weak sourcing"
                out.append(
                    f"  [{finding['level']}] {finding['category']}: {finding['title']} ({marker})"
                )
                out.append(f"        {self._safe(finding['detail'])[:400]}")
        return "\n".join(out)

    def _section_5_verified_facts(self) -> str:
        out = [self._header(SECTION_ORDER[4])]
        facts = sorted(
            self.bus.facts,
            key=lambda f: (-f.materiality.rank, f.source_tier.rank, f.category.value),
        )
        if not facts:
            out.append("No facts were collected. Nothing below this line is supported.")
            return "\n".join(out)
        out.append(f"{len(facts)} fact(s). Class and tier are shown for every one.")
        out.append("")
        for fact in facts:
            marker = {
                "VERIFIED": "[V]",
                "PARTIALLY_VERIFIED": "[P]",
                "NOT_VERIFIED": "[ ]",
                "CONTRADICTED": "[X]",
                "NOT_FOUND": "[-]",
                "INSUFFICIENT_EVIDENCE": "[?]",
            }.get(str(fact.verified_status), "[ ]")
            out.append(
                f"{marker} ({fact.category}/{fact.evidence_class}/{fact.source_tier}"
                f"/conf {fact.confidence:.2f}{'/STALE' if fact.stale else ''}) {fact.claim}"
            )
            out.append(
                f"      source: {self._safe(fact.source_url)} | event_date={fact.event_date} "
                f"published={fact.publication_date} | id={fact.fact_id}"
            )
            if fact.notes:
                out.append(f"      note: {fact.notes}")
        return "\n".join(out)

    def _section_6_contradictions(self) -> str:
        out = [self._header(SECTION_ORDER[5])]
        if not self.bus.contradictions:
            out.append("No contradictions detected by the mechanical checks.")
            checks = self._channel(Channel.CONTRADICTIONS).get("checks_run", [])
            if checks:
                out.append(f"Checks run: {', '.join(checks)}")
            return "\n".join(out)
        out.append(
            "Conflicts are reported as conflicts. They are not reconciled into a single story."
        )
        for contradiction in sorted(self.bus.contradictions, key=lambda c: -c.severity.rank):
            out.append("")
            out.append(f"  [{contradiction.severity}] {contradiction.kind}")
            out.append(f"    {self._safe(contradiction.description)}")
            out.append(f"    A: {self._safe(contradiction.left_summary)}")
            out.append(f"    B: {self._safe(contradiction.right_summary)}")
        return "\n".join(out)

    def _section_7_bull(self) -> str:
        out = [self._header(SECTION_ORDER[6])]
        payload = self._channel(Channel.BULL)
        out.append(self._safe(self._summary(Channel.BULL)))
        out.append("")
        out.append("(Built without sight of the bear case or the kill findings.)")
        points = payload.get("points", [])
        if not points:
            out.append("")
            out.append("  No evidence-backed undervaluation argument could be constructed.")
        for point in points:
            out.append(f"  + {self._safe(point['claim'])}")
            if point.get("fact_ids"):
                out.append(f"      facts: {', '.join(point['fact_ids'])}")
        dropped = payload.get("dropped_unsupported_points", [])
        if dropped:
            out.append("")
            out.append(f"  {len(dropped)} unsupported point(s) were dropped for lacking a fact.")
        return "\n".join(out)

    def _section_8_bear(self) -> str:
        out = [self._header(SECTION_ORDER[7])]
        payload = self._channel(Channel.BEAR)
        out.append(self._safe(self._summary(Channel.BEAR)))
        out.append("")
        out.append("(Built without sight of the bull case.)")
        for mechanism in payload.get("mechanisms", []):
            out.append(f"  - [{mechanism.get('severity')}] {self._safe(mechanism['mechanism'])}")
            out.append(f"      {self._safe(mechanism['path'])}")
        return "\n".join(out)

    def _section_9_regulatory(self) -> str:
        out = [self._header(SECTION_ORDER[8])]
        payload = self._channel(Channel.REGULATORY)
        if not payload:
            out.append("Regulatory agent produced no output.")
            return "\n".join(out)
        out.append(
            f"REGULATOR POSITION ON PRIMARY ENDPOINT: {payload.get('endpoint_position', UNKNOWN)}"
        )
        out.append("")
        out.append("What the regulator has agreed:")
        agreed = payload.get("agreed", [])
        out.extend(f"  + {self._safe(a)}" for a in agreed) if agreed else out.append(
            "  (nothing found)"
        )
        out.append("")
        out.append("What the regulator has NOT agreed / adverse findings:")
        not_agreed = payload.get("not_agreed", [])
        out.extend(f"  - {self._safe(n)}" for n in not_agreed) if not_agreed else out.append(
            "  (nothing found)"
        )
        out.append("")
        out.append("Unresolved:")
        unresolved = payload.get("unresolved", [])
        out.extend(f"  ? {self._safe(u)}" for u in unresolved) if unresolved else out.append(
            "  (none recorded)"
        )
        framing = payload.get("company_framing", [])
        if framing:
            out.append("")
            out.append("Company characterizations (NOT regulator statements):")
            out.extend(f"  ~ {self._safe(f)}" for f in framing)
        return "\n".join(out)

    def _section_10_capital(self) -> str:
        out = [self._header(SECTION_ORDER[9])]
        payload = self._channel(Channel.CAPITAL_STRUCTURE)
        if not payload:
            out.append("Capital structure agent produced no output.")
            return "\n".join(out)
        out.append(f"As of: {payload.get('as_of', UNKNOWN)}")
        out.append(f"  basic shares            : {_num(payload.get('basic_shares'))}")
        out.append(f"  fully diluted shares    : {_num(payload.get('fully_diluted_shares'))}")
        out.append(f"  dilution overhang       : {payload.get('dilution_overhang_pct', UNKNOWN)}%")
        out.append("  components:")
        for key, value in (payload.get("dilution_components") or {}).items():
            out.append(f"      {key:<22}: {_num(value)}")
        out.append(f"  cash                    : {_money(payload.get('cash'))}")
        out.append(f"  debt                    : {_money(payload.get('debt'))}")
        out.append(f"  implied quarterly burn  : {_money(payload.get('quarterly_burn'))}")
        out.append(f"  runway (months)         : {payload.get('runway_months', UNKNOWN)}")
        out.append(f"  ATM capacity            : {_money(payload.get('atm_capacity'))}")
        out.append(f"  shelf effective         : {payload.get('shelf_effective', UNKNOWN)}")
        out.append(f"  going concern doubt     : {payload.get('going_concern', UNKNOWN)}")
        out.append(
            f"  listing compliance issue: {payload.get('listing_compliance_issue', UNKNOWN)}"
        )
        missing = payload.get("unknown_fields", [])
        if missing:
            out.append("")
            out.append(
                "  INCOMPLETE: the following dilution components were not found, so the fully "
                "diluted figure is a FLOOR: " + ", ".join(missing)
            )
        return "\n".join(out)

    def _section_11_science(self) -> str:
        out = [self._header(SECTION_ORDER[10])]
        payload = self._channel(Channel.SCIENCE)
        if not payload:
            out.append("Science agent produced no output.")
            return "\n".join(out)
        out.append(self._safe(self._summary(Channel.SCIENCE)))
        out.append("")
        for key in (
            "phase",
            "randomized",
            "masked",
            "enrollment",
            "endpoint_type",
            "primary_endpoint",
            "surrogate_validation",
            "design_quality_score",
            "signals_present",
            "signals_absent",
        ):
            if key in payload:
                out.append(f"  {key:<22}: {self._safe(str(payload[key]))}")
        for note in payload.get("design_notes", []):
            out.append(f"  - {note}")
        return "\n".join(out)

    def _section_12_competition(self) -> str:
        out = [self._header(SECTION_ORDER[11])]
        payload = self._channel(Channel.COMPETITIVE)
        if not payload:
            out.append("Competitive agent produced no output.")
            return "\n".join(out)
        out.append(f"Competitors identified: {payload.get('competitor_count', 0)}")
        for competitor in payload.get("competitors", []):
            marker = "AHEAD" if competitor.get("ahead") else "     "
            out.append(f"  [{marker}] {self._safe(competitor['claim'])}")
        out.append("")
        out.append(f"  TAM: {_money(payload.get('tam'))}")
        out.append(f"  SAM: {_num(payload.get('sam'))} (units depend on the source; see notes)")
        out.append(f"  SOM: {payload.get('som') if payload.get('som') is not None else UNKNOWN}")
        for note in payload.get("market_size_notes", []):
            out.append(f"  note: {self._safe(note)}")
        missing = payload.get("dimensions_missing", [])
        if missing:
            out.append(f"  comparison dimensions with no evidence: {', '.join(missing)}")
        return "\n".join(out)

    def _section_13_valuation(self) -> str:
        out = [self._header(SECTION_ORDER[12])]
        payload = self._channel(Channel.VALUATION)
        if not payload:
            out.append("Valuation agent produced no output.")
            return "\n".join(out)
        out.append(f"  price                    : {_money(payload.get('price'))}")
        out.append(f"  basic market cap         : {_money(payload.get('basic_market_cap'))}")
        out.append(
            f"  fully diluted market cap : {_money(payload.get('fully_diluted_market_cap'))}"
        )
        out.append(f"  enterprise value         : {_money(payload.get('enterprise_value'))}")
        out.append(f"  cash-adjusted EV         : {_money(payload.get('cash_adjusted_ev'))}")
        out.append("")
        out.append("What each multiple REQUIRES (not what it would be nice to have):")
        for requirement in payload.get("requirements", []):
            out.append(f"  - {self._safe(requirement)}")
        if payload.get("unreachable_multiples"):
            out.append("")
            out.append(
                "  UNREACHABLE on the stated market size: "
                + ", ".join(payload["unreachable_multiples"])
            )
        for assumption in payload.get("assumptions", []):
            out.append(f"  assumption: {assumption}")
        if payload.get("unknowns"):
            out.append(f"  unknown inputs: {', '.join(payload['unknowns'])}")
        return "\n".join(out)

    def _section_14_catalysts(self) -> str:
        out = [self._header(SECTION_ORDER[13])]
        payload = self._channel(Channel.CATALYSTS)
        events = payload.get("events", [])
        out.append(
            f"Timezone: {payload.get('timezone', 'Asia/Tokyo (JST)')}   Today: {payload.get('today_jst', UNKNOWN)}"
        )
        if not events:
            out.append("No dated forward catalyst found in the evidence.")
            return "\n".join(out)
        for event in events:
            out.append("")
            out.append(
                f"  [{event['horizon']}] {event['date_jst']} "
                f"(date confidence {event['date_confidence']})"
            )
            out.append(f"      event   : {self._safe(event['event'])}")
            out.append(f"      bull    : {event['bull_outcome']}")
            out.append(f"      bear    : {event['bear_outcome']}")
            out.append(f"      priced? : {event['market_pricing']}")
            out.append(f"      source  : {self._safe(event['information_source'])}")
        return "\n".join(out)

    def _section_15_scenarios(self) -> str:
        out = [self._header(SECTION_ORDER[14])]
        if not self.result.scenarios:
            out.append("No scenarios computed.")
            return "\n".join(out)
        for scenario in self.result.scenarios:
            low, high = scenario.probability_range
            out.append("")
            out.append(f"  {scenario.name} ({scenario.confidence})")
            out.append(f"    probability   : {low:.0%} - {high:.0%}")
            if scenario.price_range:
                out.append(
                    f"    price range   : {_money(scenario.price_range[0])} - "
                    f"{_money(scenario.price_range[1])}"
                )
            out.append(f"    market cap    : {_money(scenario.market_cap)}")
            out.append(f"    FD market cap : {_money(scenario.fully_diluted_market_cap)}")
            out.append(f"    horizon       : {scenario.time_horizon}")
            for condition in scenario.required_conditions:
                out.append(f"    requires      : {self._safe(condition)}")
            for condition in scenario.failure_conditions:
                out.append(f"    fails if      : {self._safe(condition)}")
        return "\n".join(out)

    def _section_16_scores(self) -> str:
        out = [self._header(SECTION_ORDER[15])]
        card = self.result.scorecard
        if card is None:
            out.append("No scorecard produced.")
            return "\n".join(out)
        out.append(f"EVIDENCE CONFIDENCE (read this first): {card.evidence_confidence} / 10")
        out.append("")
        out.append("There is deliberately NO overall score. A single number would let a large")
        out.append("upside estimate outvote a disqualifying fact, which is the failure mode")
        out.append("this system exists to prevent.")
        out.append("")
        out.append(f"  {'dimension':<28} {'score':>6}  {'conf':>5}   note")
        for dimension in SCORE_DIMENSIONS:
            value = card.scores.get(dimension)
            confidence = card.per_score_confidence.get(dimension)
            capped = " [CAPPED BY KILL GATE]" if dimension in card.capped_by_kill_gate else ""
            score_text = UNKNOWN if value is None else f"{value:.1f}"
            confidence_text = "-" if confidence is None else f"{confidence:.1f}"
            out.append(
                f"  {dimension:<28} {score_text:>6} {confidence_text:>5}   "
                f"{card.rationale.get(dimension, '')}{capped}"
            )
        return "\n".join(out)

    def _section_17_action(self) -> str:
        out = [self._header(SECTION_ORDER[16])]
        verdict = self.result.verdict
        if verdict is None:
            out.append("No action: the judge did not complete.")
            return "\n".join(out)
        out.append(
            f"EVIDENCE CONFIDENCE: {verdict.evidence_confidence} / 10   (shown before the action)"
        )
        out.append(f"RESEARCH_STATUS: {verdict.research_status}")
        out.append(f"ACTION: {verdict.action if verdict.action is not None else 'NONE'}")
        if self.result.context.status != RunStatus.COMPLETE:
            out.append(
                f"This action is provisional: the run is marked {self.result.context.status}."
            )
        if self.result.context.use_fixtures:
            out.append("This action is derived from SYNTHETIC data and must not be acted on.")

        guidance = self.result.portfolio_guidance
        if guidance:
            out.append("")
            out.append("Position guidance (the only step that saw your holdings, and it ran")
            out.append("after the verdict above was already fixed):")
            if guidance.get("portfolio_concentration_pct") is not None:
                out.append(
                    f"  concentration: {guidance['portfolio_concentration_pct']}% of portfolio"
                )
            if guidance.get("unrealised_pct") is not None:
                out.append(f"  unrealised   : {guidance['unrealised_pct']}%")
            for line in guidance.get("guidance", []):
                out.append(f"  - {self._safe(line)}")
        return "\n".join(out)

    def _section_18_changes(self) -> str:
        out = [self._header(SECTION_ORDER[17])]
        diff = self.result.thesis_diff
        if not diff:
            out.append("No thesis version recorded.")
            return "\n".join(out)
        out.append(f"Thesis version: v{self.result.thesis_version}")
        for key in ("WHAT_CHANGED", "WHY_CHANGED"):
            out.append("")
            out.append(f"{key}:")
            for entry in diff.get(key, []):
                out.append(f"  - {entry}")
        new_facts = diff.get("NEW_FACT", [])
        removed = diff.get("REMOVED_ASSUMPTION", [])
        out.append("")
        out.append(f"NEW_FACT: {len(new_facts)} new fact id(s)")
        out.append(f"REMOVED_ASSUMPTION: {len(removed)} fact id(s) no longer present")
        score_change = diff.get("SCORE_CHANGE", {})
        if score_change:
            out.append("SCORE_CHANGE:")
            for dimension, (old, new) in score_change.items():
                out.append(f"  {dimension}: {old} -> {new}")
        return "\n".join(out)

    def _section_19_unresolved(self) -> str:
        out = [self._header(SECTION_ORDER[18])]
        if not self.bus.unresolved:
            out.append("No unresolved questions recorded (which is itself unusual).")
            return "\n".join(out)
        blocking = [u for u in self.bus.unresolved if u.blocking]
        others = [u for u in self.bus.unresolved if not u.blocking]
        if blocking:
            out.append("BLOCKING -- these prevent a confident conclusion:")
            for question in blocking:
                out.append(f"  [!] {self._safe(question.question)}")
                out.append(
                    f"      why: {question.why_it_matters}  (raised by {question.raised_by})"
                )
        if others:
            out.append("")
            out.append("Other open questions:")
            for question in others:
                out.append(f"  ? {self._safe(question.question)}  (raised by {question.raised_by})")
        return "\n".join(out)

    def _section_20_sources(self) -> str:
        out = [self._header(SECTION_ORDER[19])]
        sources = sorted(self.bus.sources, key=lambda s: (s.tier.rank, s.url))
        if not sources:
            out.append("No sources retrieved.")
            return "\n".join(out)
        for source in sources:
            best_date, kind = source.best_date()
            fixture = " [FIXTURE]" if source.provenance == Provenance.FIXTURE else ""
            out.append(f"  [{source.tier}]{fixture} {source.title}")
            out.append(f"      {self._safe(source.url)}")
            out.append(
                f"      date used: {best_date or UNKNOWN} ({kind or 'none'}) | "
                f"published={source.published_date} event={source.event_date} "
                f"filed={source.filing_date}"
            )
        out.append("")
        out.append(
            "Dates: 'date used' is the event or effective date wherever one exists. "
            "A publication date is never treated as the date the event occurred."
        )
        return "\n".join(out)


def _fmt_status(status: str) -> str:
    return {
        "SEARCHED": "[searched]",
        "PARTIAL": "[partial ]",
        "UNSEARCHED": "[NOT DONE]",
        "FAILED": "[FAILED  ]",
    }.get(status, f"[{status}]")


def _appendix(title: str) -> str:
    return f"\n{'=' * 78}\nAPPENDIX -- {title}\n{'=' * 78}"


def render_report(result: ResearchResult) -> str:
    return ReportRenderer(result).render()


# ---------------------------------------------------------------------------
# Phase 2 appendices, bound onto ReportRenderer.
# ---------------------------------------------------------------------------
def _section_research_provenance(self: ReportRenderer) -> str:
    """How the evidence was obtained (ADR 0005)."""
    out = [_appendix("A. RESEARCH PROVENANCE -- how this evidence was obtained")]
    capture = self.result.capture_info
    if capture:
        out.append("CAPTURED CORPUS REPLAY")
        out.append(
            self._wrap(
                f"This run replayed {capture.get('document_count', 0)} document(s) about a real "
                f"issuer, captured at {capture.get('captured_at', UNKNOWN)} via "
                f"{capture.get('captured_via', UNKNOWN)}. The URLs are real and the documents "
                "are real. They were NOT fetched live at run time, so anything published after "
                "the capture time is absent from this analysis by construction."
            )
        )
        out.append("")

    kinds: dict[str, int] = {}
    paths: dict[str, int] = {}
    for source in self.bus.sources:
        kinds[str(source.provenance)] = kinds.get(str(source.provenance), 0) + 1
    for fact in self.bus.facts:
        paths[str(fact.provenance)] = paths.get(str(fact.provenance), 0) + 1
    out.append("Source provenance:")
    for name, count in sorted(kinds.items()):
        out.append(f"  {name:<12} {count}")

    summary_sources = [s for s in self.bus.sources if not s.content_kind.is_primary_text]
    if summary_sources:
        out.append("")
        out.append(
            self._wrap(
                f"{len(summary_sources)} source(s) are search-engine summaries rather than the "
                "document itself. A summary about a filing is not the filing: quotations drawn "
                "from it have not been checked against the original."
            )
        )
    return "\n".join(out)


def _section_search_coverage(self: ReportRenderer) -> str:
    """Search Completeness Gate (requirement P6)."""
    out = [_appendix("B. SEARCH COMPLETENESS GATE")]
    completeness = self.result.completeness
    if completeness is None:
        out.append("Completeness was not assessed for this run.")
        return "\n".join(out)

    out.append(f"{completeness.searched_count} of 6 required research domains examined.")
    out.append("")
    out.append(f"  {'domain':<22} {'status':<11} {'queries':>7} {'docs':>5}  paths")
    for name, status, queries, documents, paths in completeness.summary_rows():
        out.append(f"  {name:<22} {_fmt_status(status):<11} {queries:>7} {documents:>5}  {paths}")
    if completeness.blocked:
        out.append("")
        out.append("*** VERDICT BLOCKED ***")
        out.append(self._wrap(completeness.reason()))
    else:
        out.append("")
        out.append("All required domains were examined; a verdict is permitted.")
    return "\n".join(out)


def _section_evidence_sufficiency(self: ReportRenderer) -> str:
    """Evidence Sufficiency Matrix (Decision-Grade Evidence Gate, requirement DG5).

    Deliberately separate from the Search Completeness Gate above: that
    section answers "did we look at this domain"; this one answers "did what
    we found actually settle anything". A domain can be SEARCHED there and
    INSUFFICIENT here, and that gap -- not an unsearched domain -- is what
    blocked LGVN/CNTB/CRBP from a confident verdict on the captured corpus.
    """
    out = [_appendix("C. EVIDENCE SUFFICIENCY MATRIX")]
    matrix = self.result.evidence_sufficiency
    if matrix is None:
        out.append("The Evidence Sufficiency Matrix was not assessed for this run.")
        return "\n".join(out)

    out.append(f"Decision-grade facts in this run: {matrix.decision_grade_fact_total}")
    out.append("")
    out.append(f"  {'domain':<22} {'search_status':<13} {'sufficiency':<12} {'grade/total'}")
    for name, search_status, sufficiency_status, grade, total, reason in matrix.summary_rows():
        out.append(
            f"  {name:<22} {search_status:<13} {sufficiency_status:<12} {grade}/{total}"
        )
        out.append(f"{self._wrap(reason, indent='      ')}")
    if matrix.provisional_kill_findings:
        out.append("")
        out.append("PROVISIONAL kill findings still requiring verification:")
        out.extend(f"  - {self._safe(f)}" for f in matrix.provisional_kill_findings)
    if matrix.unresolved_material_claims:
        out.append("")
        out.append("Unresolved MATERIAL/CRITICAL claims:")
        out.extend(f"  - {self._safe(c)}" for c in matrix.unresolved_material_claims)
    out.append("")
    if matrix.sufficient:
        out.append("The Evidence Sufficiency Matrix is satisfied; an Action may be issued.")
    else:
        out.append("*** EVIDENCE SUFFICIENCY MATRIX NOT SATISFIED -- ACTION WITHHELD ***")
        for reason in matrix.blocking_reasons():
            out.append(f"  - {self._wrap(reason, indent='    ')}")
    return "\n".join(out)


def _section_escalation(self: ReportRenderer) -> str:
    """Primary-source escalation (requirement P4)."""
    out = [_appendix("C. PRIMARY-SOURCE ESCALATION")]
    escalation = self.result.escalation
    if escalation is None:
        out.append("No escalation was attempted (no research provider available).")
        return "\n".join(out)
    if not escalation.attempts:
        out.append("No material claim required escalation.")
        return "\n".join(out)

    out.append(
        self._wrap(
            "A material claim carried only by weak sourcing is escalated to a primary source. "
            "A company press release does not count as confirmation of what a regulator said; "
            "only a statutory filing or the regulator's own document does."
        )
    )
    out.append("")
    out.append(f"  attempted : {len(escalation.attempts)}")
    out.append(f"  confirmed : {len(escalation.confirmed)}")
    out.append(f"  UNCONFIRMED: {len(escalation.unconfirmed)}")
    for attempt in escalation.unconfirmed:
        out.append("")
        out.append(f"  [UNVERIFIED_MATERIAL_CLAIM] ({attempt.reason})")
        out.append(f"      {self._safe(attempt.claim)}")
        if attempt.queries:
            out.append(f"      searched: {'; '.join(attempt.queries[:3])}")
    for attempt in escalation.confirmed:
        out.append("")
        out.append(f"  [CONFIRMED] ({attempt.reason}) {self._safe(attempt.claim)[:150]}")
        out.append(f"      confirmed by: {self._safe(attempt.confirming_url)}")
    return "\n".join(out)


def _section_traceability(self: ReportRenderer) -> str:
    """Citation traceability index (requirement P7)."""
    out = [_appendix("D. CITATION TRACEABILITY INDEX")]
    index = self.result.traceability
    if index is None or not index.links:
        out.append("No traceability index was built.")
        return "\n".join(out)
    out.append(
        "Every material claim above resolves through this chain: "
        "claim -> fact_id -> source_id -> URL -> publication date -> event date."
    )
    out.append("")
    for link in index.rows():
        out.append(f"  {link.fact_id}")
        out.append(f"      claim    : {link.claim[:150]}")
        out.append(f"      source   : {link.source_id}")
        out.append(f"      url      : {self._safe(link.url)}")
        out.append(f"      published: {link.publication_date}   event: {link.event_date}")
        out.append(
            f"      tier     : {link.source_tier}   class: {link.evidence_class}   "
            f"status: {link.verified_status}"
        )
    return "\n".join(out)


def _section_cost(self: ReportRenderer) -> str:
    """Model usage and cost accounting (requirement P9)."""
    out = [_appendix("E. ANALYSIS PROVENANCE AND COST")]
    used = self.result.llm_agents_used
    budget = self.result.llm_budget

    if not used:
        out.append(
            self._wrap(
                "No agent was model-backed in this run. Every analysis below came from the "
                "deterministic rule-based agents. This is stated plainly because a "
                "rule-based reading and a model reading are different things, and the reader "
                "is entitled to know which one produced the text."
            )
        )
    else:
        out.append(f"LLM-backed agents: {', '.join(sorted(set(used)))}")
        out.append(
            "All other agents -- share counts, runway, market cap, valuation arithmetic, the "
            "Kill Gate, tiering, routing and isolation -- were deterministic."
        )
    if budget and budget.calls:
        out.append("")
        out.append(f"  tokens used : {budget.used_total:,} of {budget.max_total_tokens:,} budget")
        out.append(f"  remaining   : {budget.remaining:,}")
        out.append(f"  calls       : {len(budget.calls)}")
        out.append("")
        out.append("  tokens by stage (requirement: discovery must not starve later stages):")
        for stage, info in budget.stage_summary().items():
            cap = f"{info['cap']:,}" if info["cap"] is not None else "no quota"
            remaining = f"{info['remaining']:,}" if info["remaining"] is not None else "n/a"
            flag = " [EXHAUSTED]" if info["exhausted"] else ""
            out.append(
                f"      {stage:<14} used {info['used']:>9,} / cap {cap:<10} "
                f"remaining {remaining:<10}{flag}"
            )
        out.append("")
        out.append("  tokens by agent/call:")
        for agent_id, tokens in sorted(budget.by_agent().items()):
            out.append(f"      {agent_id:<22} {tokens:>9,}")

    adversarial = self.result.adversarial
    if adversarial is not None:
        out.append("")
        out.append("  adversarial search (bear/bull discovery):")
        out.append(f"      executed          : {adversarial.executed}")
        out.append(f"      unexecuted        : {len(adversarial.unexecuted)}")
        out.append(
            f"      unexecuted (budget): {len(adversarial.unexecuted_due_to_budget)}"
        )
        out.append(f"      deduplicated      : {len(adversarial.deduplicated)}")

    escalation = self.result.escalation
    if escalation is not None:
        out.append("")
        out.append("  primary-source escalation (fetch):")
        out.append(f"      fetches attempted : {escalation.fetches_attempted}")
        out.append(f"      fetches failed    : {escalation.fetches_failed}")
        out.append(f"      confirmed         : {len(escalation.confirmed)}")
        out.append(f"      unconfirmed       : {len(escalation.unconfirmed)}")

    if self.result.chunks:
        out.append("")
        out.append(
            self._wrap(
                f"Evidence was chunked into {len(self.result.chunks)} chunk(s) and each agent "
                "received only the chunks relevant to it, within a token budget. No agent read "
                "the whole corpus."
            )
        )
    return "\n".join(out)
