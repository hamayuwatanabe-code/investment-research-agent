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

        add(self._section_1_verdict())
        add(self._section_2_confidence())
        add(self._section_3_red_flags())
        add(self._section_4_thesis_breakers())
        add(self._section_5_verified_facts())
        add(self._section_6_contradictions())
        add(self._section_7_bull())
        add(self._section_8_bear())
        add(self._section_9_regulatory())
        add(self._section_10_capital())
        add(self._section_11_science())
        add(self._section_12_competition())
        add(self._section_13_valuation())
        add(self._section_14_catalysts())
        add(self._section_15_scenarios())
        add(self._section_16_scores())
        add(self._section_17_action())
        add(self._section_18_changes())
        add(self._section_19_unresolved())
        add(self._section_20_sources())

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
        verdict = self.result.verdict
        out = [self._header(SECTION_ORDER[0])]
        if verdict is None:
            out.append("NO VERDICT: the blind judge did not complete. Treat as INCOMPLETE.")
            return "\n".join(out)
        out.append(f"Action: {verdict.action}")
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
        out.append(f"ACTION: {verdict.action}")
        if self.result.context.status != RunStatus.COMPLETE:
            out.append(
                f"This action is provisional: the run is marked {self.result.context.status}."
            )
        if self.result.context.use_fixtures:
            out.append("This action is derived from SYNTHETIC data and must not be acted on.")
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


def render_report(result: ResearchResult) -> str:
    return ReportRenderer(result).render()
