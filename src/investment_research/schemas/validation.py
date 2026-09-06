"""Schema validation and evaluation-leak detection.

Two jobs:

1. Structural validation of :class:`Fact` / :class:`Source` records before they
   are persisted (requirement 14: schema validation).
2. Detecting *evaluative language* where only facts are permitted
   (requirement 1D).  The Fact Collector and the raw-fact channel must stay
   free of "buy", "強気", "9/10", "explosive candidate" and friends.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .enums import (
    UNKNOWN,
    EvidenceClass,
    FactCategory,
    Materiality,
    SourceTier,
    VerifiedStatus,
)
from .fact import Fact, Source, parse_iso_date


class SchemaError(ValueError):
    """Raised when a record violates its contract."""


class EvaluationLeak(ValueError):
    """Raised when an evaluation appears where only facts are permitted."""


# --- evaluative-language detection -----------------------------------------
# Deliberately conservative: these patterns match *recommendations and ratings*,
# not neutral descriptive words.  "revenue declined" is a fact; "avoid this
# stock" is not.
_EVAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rating_scale", re.compile(r"\b(?:10|[0-9](?:\.[0-9])?)\s*/\s*10\b")),
    (
        "action_label",
        re.compile(r"\b(?:strong[_ ]buy|buy the dip|must[- ]buy|table[- ]pounding)\b", re.I),
    ),
    (
        "recommendation",
        re.compile(
            r"\b(?:we|i)\s+(?:recommend|rate|would\s+buy|would\s+sell)\b|"
            r"\b(?:recommend(?:ed|ation)?\s+(?:to\s+)?(?:buy|sell|accumulate|short))\b",
            re.I,
        ),
    ),
    (
        "hype",
        re.compile(
            r"\b(?:explosive\s+candidate|multi-?bagger|moon\s?shot|to\s+the\s+moon|no-?brainer|slam\s+dunk)\b",
            re.I,
        ),
    ),
    (
        "ranking",
        re.compile(
            r"\b(?:top\s+pick|best\s+idea|highest\s+conviction|rank(?:ed)?\s*#\s*\d+)\b", re.I
        ),
    ),
    ("jp_eval", re.compile(r"(?:買い推奨|強気|爆発候補|大化け|絶対に買|買うべき)")),
)


def find_evaluative_language(text: str) -> list[str]:
    """Return the names of evaluative patterns found in ``text``."""
    if not text:
        return []
    return [name for name, pattern in _EVAL_PATTERNS if pattern.search(text)]


def assert_evaluation_free(text: str, *, where: str) -> None:
    """Fail loudly if evaluative language appears in a facts-only channel."""
    hits = find_evaluative_language(text)
    if hits:
        raise EvaluationLeak(
            f"evaluative language {hits} found in facts-only context {where!r}: {text[:200]!r}"
        )


# --- record validation ------------------------------------------------------
_URL_RE = re.compile(r"^(https?://[^\s\"'<>]+|fixture://[^\s]+|local://[^\s]+)$")


def validate_source_url(url: str) -> None:
    """Requirement 14/15: source URLs must be well-formed or absent.

    ``fixture://`` marks mock data and is accepted so that the mock/production
    separation is explicit rather than disguised as a real URL.
    """
    if url in ("", UNKNOWN):
        raise SchemaError("source url is empty; use NOT_FOUND semantics instead of a blank url")
    if not _URL_RE.match(url):
        raise SchemaError(f"malformed source url: {url!r}")


def validate_date_field(value: str, field_name: str) -> None:
    if value == UNKNOWN:
        return
    if parse_iso_date(value) is None:
        raise SchemaError(f"{field_name} is neither UNKNOWN nor a parseable date: {value!r}")


def validate_source(source: Source) -> None:
    validate_source_url(source.url)
    if not isinstance(source.tier, SourceTier):
        raise SchemaError(f"source.tier must be a SourceTier, got {type(source.tier)!r}")
    for name in ("published_date", "event_date", "effective_date", "filing_date"):
        validate_date_field(getattr(source, name), f"source.{name}")


def validate_fact(fact: Fact, *, facts_only: bool = True) -> None:
    """Validate a fact record before persistence.

    ``facts_only`` runs the evaluation-leak check over the claim text.
    """
    if not fact.fact_id.startswith("fact_"):
        raise SchemaError(f"malformed fact_id: {fact.fact_id!r}")
    if not fact.ticker:
        raise SchemaError("fact.ticker is required")
    if not isinstance(fact.category, FactCategory):
        raise SchemaError("fact.category must be a FactCategory")
    if not isinstance(fact.evidence_class, EvidenceClass):
        raise SchemaError("fact.evidence_class must be an EvidenceClass (requirement 1B)")
    if not isinstance(fact.source_tier, SourceTier):
        raise SchemaError("fact.source_tier must be a SourceTier")
    if not isinstance(fact.verified_status, VerifiedStatus):
        raise SchemaError("fact.verified_status must be a VerifiedStatus")
    if not isinstance(fact.materiality, Materiality):
        raise SchemaError("fact.materiality must be a Materiality")
    if not (0.0 <= float(fact.confidence) <= 1.0):
        raise SchemaError(f"fact.confidence out of range: {fact.confidence}")
    if not fact.claim.strip():
        raise SchemaError("fact.claim is empty")
    validate_source_url(fact.source_url)
    for name in ("publication_date", "event_date", "effective_date", "filing_date"):
        validate_date_field(getattr(fact, name), f"fact.{name}")
    if fact.version < 1:
        raise SchemaError("fact.version must be >= 1")
    if facts_only:
        assert_evaluation_free(fact.claim, where=f"fact:{fact.fact_id}")

    # A company's own statement can never be a VERIFIED_FACT on its own.
    if (
        fact.company_claim
        and fact.evidence_class == EvidenceClass.VERIFIED_FACT
        and not fact.independent_confirmation
    ):
        raise SchemaError(
            "a company claim cannot be classified VERIFIED_FACT without independent "
            "confirmation (requirement 1B)"
        )
    # Tier 4/5 alone cannot produce a decision-grade verified fact.
    if (
        fact.evidence_class == EvidenceClass.VERIFIED_FACT
        and fact.source_tier in (SourceTier.TIER_4, SourceTier.TIER_5, SourceTier.UNKNOWN)
        and not fact.independent_confirmation
    ):
        raise SchemaError(
            f"tier {fact.source_tier} cannot alone establish a VERIFIED_FACT (requirement 2)"
        )


def validate_facts(facts: Iterable[Fact], **kwargs: Any) -> None:
    for fact in facts:
        validate_fact(fact, **kwargs)
