"""Evaluation-side structures: kill assessments, scores, scenarios, verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import (
    Action,
    KillCategory,
    KillConfirmation,
    KillLevel,
    ResearchStatus,
    RunStatus,
    ScenarioName,
)


@dataclass
class KillFinding:
    """A single candidate lethal problem found by the Kill Agent."""

    category: KillCategory
    level: KillLevel
    title: str
    detail: str
    fact_ids: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    evidence_is_primary: bool = False
    #: Whether this specific finding is backed by at least one decision-grade
    #: fact. Distinct from ``evidence_is_primary`` (a tier judgement about the
    #: document) and from ``level`` (severity if true). A K5 finding whose only
    #: support is a search snippet pointing at a Tier 1 filing is PROVISIONAL,
    #: not CONFIRMED, until the filing body is actually read.
    confirmation: KillConfirmation = KillConfirmation.PROVISIONAL


@dataclass
class KillAssessment:
    """Per-category K0-K5 outcome (requirement 5)."""

    category: KillCategory
    level: KillLevel
    rationale: str
    findings: tuple[KillFinding, ...] = ()
    evidence_confidence: float = 0.0
    #: CONFIRMED when at least one finding at this assessment's severity is
    #: decision-grade-backed; PROVISIONAL otherwise. Reported as "Potential
    #: <category>: <level> PROVISIONAL" vs "<category>: <level> CONFIRMED" --
    #: the Decision-Grade Evidence Gate's central distinction (requirement DG2).
    confirmation: KillConfirmation = KillConfirmation.PROVISIONAL

    def to_row(self) -> dict[str, Any]:
        return {
            "category": str(self.category),
            "level": str(self.level),
            "rationale": self.rationale,
            "evidence_confidence": self.evidence_confidence,
            "confirmation": str(self.confirmation),
        }


@dataclass
class KillGateResult:
    assessments: tuple[KillAssessment, ...]
    unsearched_categories: tuple[KillCategory, ...] = ()

    @property
    def max_level(self) -> KillLevel:
        """Worst severity found, CONFIRMED or PROVISIONAL alike.

        This answers "how bad, if true" and is used for reporting and for the
        conservative score cap (requirement 5/18) -- an unverified K5 concern
        still means no optimistic score is warranted. It must NOT be used to
        decide whether a final Action such as AVOID may be emitted; that
        decision requires :attr:`max_confirmed_level`.
        """
        if not self.assessments:
            return KillLevel.K0
        return max((a.level for a in self.assessments), key=lambda k: k.level)

    @property
    def max_confirmed_level(self) -> KillLevel:
        """Worst severity among CONFIRMED (decision-grade-backed) assessments.

        This is the level an Action may be based on. A PROVISIONAL K5 -- a
        search snippet pointing at a filing nobody has read yet -- must never
        drive AVOID; it drives a demand for verification instead.
        """
        confirmed = [a for a in self.assessments if a.confirmation is KillConfirmation.CONFIRMED]
        if not confirmed:
            return KillLevel.K0
        return max((a.level for a in confirmed), key=lambda k: k.level)

    @property
    def provisional_major_or_worse(self) -> tuple[KillAssessment, ...]:
        """K3+ assessments that are still PROVISIONAL -- open verification work."""
        return tuple(
            a
            for a in self.assessments
            if a.level.level >= 3 and a.confirmation is KillConfirmation.PROVISIONAL
        )

    @property
    def disqualifying(self) -> tuple[KillAssessment, ...]:
        return tuple(a for a in self.assessments if a.level.level >= 4)

    @property
    def major(self) -> tuple[KillAssessment, ...]:
        return tuple(a for a in self.assessments if a.level.level >= 3)

    def by_category(self, category: KillCategory) -> KillAssessment | None:
        for a in self.assessments:
            if a.category == category:
                return a
        return None

    @classmethod
    def from_channel_payload(cls, payload: dict[str, Any]) -> KillGateResult:
        """Reconstruct a gate from the serialized ``Channel.KILL`` payload.

        The single place that deserializes ``KillAssessment.to_row()`` rows, so
        every consumer (pipeline, deterministic judge, LLM judge) parses
        ``confirmation`` the same way instead of three copies drifting apart.
        """
        assessments = tuple(
            KillAssessment(
                category=KillCategory(row["category"]),
                level=KillLevel(row["level"]),
                rationale=row.get("rationale", ""),
                evidence_confidence=float(row.get("evidence_confidence", 0.0)),
                confirmation=KillConfirmation(
                    row.get("confirmation", KillConfirmation.PROVISIONAL.value)
                ),
            )
            for row in payload.get("kill_gate", [])
        )
        unsearched = tuple(
            KillCategory(name) for name in payload.get("unsearched_categories", [])
        )
        return cls(assessments=assessments, unsearched_categories=unsearched)


#: The independent score dimensions (requirement 7).  There is deliberately NO
#: aggregate/overall score: collapsing these into one number is exactly how a
#: seductive narrative survives contact with a single disqualifying fact.
SCORE_DIMENSIONS: tuple[str, ...] = (
    "evidence_quality",
    "management_quality",
    "science_technology_quality",
    "clinical_quality",
    "regulatory_quality",
    "financial_strength",
    "capital_structure_quality",
    "competitive_moat",
    "tam_quality",
    "sam_som_quality",
    "growth_quality",
    "valuation",
    "catalyst_strength",
    "catalyst_timing",
    "market_pricing",
    "flow_microstructure",
    "immediate_buy",
    "explosive_potential",
    "long_term_multibagger",
    "downside_risk",
    "risk_reward",
    "time_efficiency",
)

#: Dimensions that express "this is a good investment".  The Kill Gate caps
#: every one of them (requirement 5 / regression test 18).
INVESTMENT_QUALITY_DIMENSIONS: tuple[str, ...] = (
    "immediate_buy",
    "risk_reward",
    "long_term_multibagger",
    "regulatory_quality",
    "clinical_quality",
    "financial_strength",
    "capital_structure_quality",
)


@dataclass
class ScoreCard:
    """Independent scores, each 0-10, each with its own evidence confidence.

    ``evidence_confidence`` is reported *separately and first* (requirement 9):
    "Explosive Potential 9.5 / Evidence Confidence 4" is a legitimate and
    important output.
    """

    ticker: str
    run_id: str
    scores: dict[str, float] = field(default_factory=dict)
    per_score_confidence: dict[str, float] = field(default_factory=dict)
    evidence_confidence: float = 0.0
    rationale: dict[str, str] = field(default_factory=dict)
    capped_by_kill_gate: tuple[str, ...] = ()

    def get(self, name: str) -> float | None:
        return self.scores.get(name)

    def missing_dimensions(self) -> tuple[str, ...]:
        return tuple(d for d in SCORE_DIMENSIONS if d not in self.scores)


@dataclass
class Scenario:
    """One of the five mandatory scenarios (requirement 8)."""

    name: ScenarioName
    probability_range: tuple[float, float]
    price_range: tuple[float, float] | None
    market_cap: float | None
    fully_diluted_market_cap: float | None
    time_horizon: str
    required_conditions: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    confidence: str = "LOW_CONFIDENCE"
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "prob_low": self.probability_range[0],
            "prob_high": self.probability_range[1],
            "price_low": None if self.price_range is None else self.price_range[0],
            "price_high": None if self.price_range is None else self.price_range[1],
            "market_cap": self.market_cap,
            "fully_diluted_market_cap": self.fully_diluted_market_cap,
            "time_horizon": self.time_horizon,
            "required_conditions": "\n".join(self.required_conditions),
            "failure_conditions": "\n".join(self.failure_conditions),
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class Verdict:
    """Final output of the Blind Judge, de-anonymized afterwards.

    ``action`` is ``None`` when the Search Completeness Gate blocked the run.
    That is deliberate: an action label asserts a judgement, and there is no
    honest action to assert over research that never examined a required domain.
    """

    action: Action | None
    evidence_confidence: float
    headline: str
    reasoning: tuple[str, ...]
    thesis_breakers: tuple[str, ...]
    critical_red_flags: tuple[str, ...]
    kill_gate: KillGateResult
    run_status: RunStatus = RunStatus.COMPLETE
    judged_blind: bool = True
    anonymized_label: str = "Company X"
    caveats: tuple[str, ...] = ()
    #: Set by the Search Completeness Gate (requirement P6).
    blocked: bool = False
    blocked_reason: str = ""
    #: The Decision-Grade Evidence Gate's verdict on whether there is enough
    #: verified evidence to bear an Action at all. ``action`` may be non-``None``
    #: only when this is COMPLETE -- not merely when every domain was searched,
    #: and not merely when the pipeline ran without failures.
    research_status: ResearchStatus = ResearchStatus.COMPLETE
    #: What must be retrieved and verified before an Action can be issued. Only
    #: populated when ``research_status`` is not COMPLETE.
    blocking_verification_required: tuple[str, ...] = ()


@dataclass
class ValuationMath:
    """Requirement 10: what each multiple actually *requires*."""

    price: float | None
    basic_shares: float | None
    fully_diluted_shares: float | None
    basic_market_cap: float | None
    fully_diluted_market_cap: float | None
    cash: float | None
    debt: float | None
    enterprise_value: float | None
    cash_adjusted_ev: float | None
    multiples: dict[str, dict[str, Any]] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()


@dataclass
class CatalystEvent:
    """Requirement 11.  All dates normalised to JST."""

    event_id: str
    horizon: str
    date_jst: str
    date_confidence: str
    event: str
    expected_outcome: str
    bull_outcome: str
    bear_outcome: str
    market_pricing: str
    information_source: str
    fact_ids: tuple[str, ...] = ()

    def to_row(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["fact_ids"] = ",".join(self.fact_ids)
        return d
