"""Controlled vocabularies.

Every enum here is part of the system's contract.  Free-text substitutes are
rejected by the validators in :mod:`investment_research.schemas.validation`,
because the whole point of this system is that an agent cannot smuggle an
evaluation through a string field.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """`str`-valued enum (3.10-compatible replacement for `enum.StrEnum`)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class EvidenceClass(StrEnum):
    """Fact vs. opinion separation (requirement 1B).

    A claim MUST be classified before it can be stored.  Nothing downstream is
    permitted to treat COMPANY_CLAIM or ANALYST_OPINION as if it were a
    VERIFIED_FACT.
    """

    VERIFIED_FACT = "VERIFIED_FACT"
    COMPANY_CLAIM = "COMPANY_CLAIM"
    INDEPENDENT_EVIDENCE = "INDEPENDENT_EVIDENCE"
    ANALYST_OPINION = "ANALYST_OPINION"
    MARKET_INFERENCE = "MARKET_INFERENCE"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    UNVERIFIED_CLAIM = "UNVERIFIED_CLAIM"


#: Evidence classes that may, on their own, support a material investment
#: conclusion.  See :func:`investment_research.scoring.evidence_confidence`.
DECISION_GRADE_CLASSES = frozenset(
    {EvidenceClass.VERIFIED_FACT, EvidenceClass.INDEPENDENT_EVIDENCE}
)


class SourceTier(StrEnum):
    """Source hierarchy (requirement 2).  Lower tier number == more primary."""

    TIER_1 = "TIER_1"  # regulators, exchanges, statutory filings
    TIER_2 = "TIER_2"  # peer-reviewed, company IR / transcripts, gov research
    TIER_3 = "TIER_3"  # major financial press, trade press, expert interviews
    TIER_4 = "TIER_4"  # sell-side analysts, broker research, price targets
    TIER_5 = "TIER_5"  # social media, retail blogs, message boards
    UNKNOWN = "UNKNOWN"

    @property
    def rank(self) -> int:
        return {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3, "TIER_4": 4, "TIER_5": 5}.get(self.value, 9)

    @property
    def is_primary(self) -> bool:
        return self.rank <= 2


#: Tiers that may NOT, alone, settle a material investment question (req. 2).
NON_DECISIVE_TIERS = frozenset({SourceTier.TIER_4, SourceTier.TIER_5, SourceTier.UNKNOWN})


class VerifiedStatus(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    NOT_FOUND = "NOT_FOUND"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


#: The single sentinel used everywhere a value is unknown.  Requirement 1G:
#: missing data is never imputed.
UNKNOWN = "UNKNOWN"


class Materiality(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

    @property
    def rank(self) -> int:
        return {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "INFORMATIONAL": 1,
        }[self.value]


class FactCategory(StrEnum):
    REGULATORY = "REGULATORY"
    CLINICAL = "CLINICAL"
    SCIENCE = "SCIENCE"
    TECHNOLOGY = "TECHNOLOGY"
    FINANCIAL = "FINANCIAL"
    CAPITAL_STRUCTURE = "CAPITAL_STRUCTURE"
    LIQUIDITY = "LIQUIDITY"
    GOVERNANCE = "GOVERNANCE"
    INSIDER = "INSIDER"
    COMMERCIAL = "COMMERCIAL"
    COMPETITION = "COMPETITION"
    MARKET_SIZE = "MARKET_SIZE"
    CONTRACTS = "CONTRACTS"
    LEGAL = "LEGAL"
    ACCOUNTING = "ACCOUNTING"
    LISTING = "LISTING"
    CATALYST = "CATALYST"
    MICROSTRUCTURE = "MICROSTRUCTURE"
    MANAGEMENT = "MANAGEMENT"
    OTHER = "OTHER"


class KillCategory(StrEnum):
    REGULATORY_KILL = "REGULATORY_KILL"
    CLINICAL_KILL = "CLINICAL_KILL"
    SCIENCE_KILL = "SCIENCE_KILL"
    CAPITAL_KILL = "CAPITAL_KILL"
    COMMERCIAL_KILL = "COMMERCIAL_KILL"
    GOVERNANCE_KILL = "GOVERNANCE_KILL"
    ACCOUNTING_KILL = "ACCOUNTING_KILL"
    LIQUIDITY_KILL = "LIQUIDITY_KILL"


#: Categories that must always be reported, even when the assessment is K0
#: (requirement 5: "最低限 ... を出す").
MANDATORY_KILL_CATEGORIES = (
    KillCategory.REGULATORY_KILL,
    KillCategory.CLINICAL_KILL,
    KillCategory.CAPITAL_KILL,
    KillCategory.COMMERCIAL_KILL,
    KillCategory.GOVERNANCE_KILL,
    KillCategory.LIQUIDITY_KILL,
)


class KillLevel(StrEnum):
    """K0-K5 severity ladder (requirement 5)."""

    K0 = "K0"  # no material concern found
    K1 = "K1"  # minor
    K2 = "K2"  # meaningful but manageable
    K3 = "K3"  # major red flag
    K4 = "K4"  # severe / normally avoid
    K5 = "K5"  # investment disqualifier

    @property
    def level(self) -> int:
        return int(self.value[1])

    @classmethod
    def from_level(cls, n: int) -> KillLevel:
        return cls(f"K{max(0, min(5, int(n)))}")


class Action(StrEnum):
    """The only permitted final action labels (requirement 23)."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    BUY_ON_PULLBACK = "BUY_ON_PULLBACK"
    WAIT_FOR_EVENT = "WAIT_FOR_EVENT"
    HOLD = "HOLD"
    PARTIAL_TAKE_PROFIT = "PARTIAL_TAKE_PROFIT"
    CONSIDER_SELL = "CONSIDER_SELL"
    AVOID = "AVOID"


class Horizon(StrEnum):
    """Catalyst horizons (requirement 11)."""

    T0 = "T0"  # intraday
    T1 = "T1"  # 1-10 trading days
    T2 = "T2"  # 1-3 months
    T3 = "T3"  # 6-18 months
    T4 = "T4"  # 3-10 years

    @property
    def label(self) -> str:
        return {
            "T0": "intraday",
            "T1": "1-10 trading days",
            "T2": "1-3 months",
            "T3": "6-18 months",
            "T4": "3-10 years",
        }[self.value]


class ScenarioName(StrEnum):
    DISASTER = "DISASTER"
    BEAR = "BEAR"
    BASE = "BASE"
    BULL = "BULL"
    EXTREME_BULL = "EXTREME_BULL"


class DateKind(StrEnum):
    """Date-integrity discrimination (requirement 3)."""

    PUBLISHED_DATE = "published_date"
    EVENT_DATE = "event_date"
    EFFECTIVE_DATE = "effective_date"
    FILING_DATE = "filing_date"


class RunStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE_RESEARCH = "INCOMPLETE_RESEARCH"
    FAILED = "FAILED"


class FetchOutcome(StrEnum):
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    BLOCKED = "BLOCKED"  # egress policy / 403 / 407
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    DISABLED = "DISABLED"  # collector switched off (no credentials, offline mode)


class Provenance(StrEnum):
    """Hard separation between production data and mock data (requirement 24)."""

    LIVE = "LIVE"
    CACHE = "CACHE"
    FIXTURE = "FIXTURE"  # mock; may never be presented as real research
