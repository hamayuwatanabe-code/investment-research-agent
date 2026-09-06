"""Agent isolation: whitelist projection + leakage scanning.

This is the safety core of the system (requirements 1C, 1D, 4/14, 6, 10, 17).

Principle
---------
An agent is never handed the evidence bus.  It is handed an
:class:`~investment_research.schemas.agent_io.AgentInput` **projected** through
its :class:`IsolationPolicy`.  Anything the policy does not name simply does not
exist for that agent.

Two independent mechanisms, because one is not enough:

1. **Projection** (structural).  Denied channels are never copied into the
   input.  This alone prevents the ordinary case.
2. **Leakage scanning** (behavioural).  Each evaluation carries rare
   ``fingerprint_tokens``.  Before an input is released, the whole payload is
   serialized and scanned for the fingerprints of denied channels and for
   identity strings the agent must not see.  This catches the *indirect* case:
   a Bull narrative laundered into a "fact", or a company name surviving in a
   source URL inside a supposedly blind pack.

A leak is a hard failure (``LeakageError``), never a warning.  A research system
that silently degrades its own independence is worse than one that stops.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schemas.agent_io import AgentInput, Evaluation, RiskFlag
from ..schemas.fact import Contradiction, Fact, Source, UnresolvedQuestion

log = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """Raised when forbidden information reaches an agent's input."""


# --- channel names ----------------------------------------------------------
class Channel:
    RAW_FACTS = "raw_facts"
    VERIFIED_EVIDENCE = "verified_evidence"
    REGULATORY = "regulatory_findings"
    CAPITAL_STRUCTURE = "capital_structure"
    SCIENCE = "science_findings"
    COMPETITIVE = "competitive_findings"
    CATALYSTS = "catalysts"
    MICROSTRUCTURE = "microstructure"
    VALUATION = "valuation_math"
    KILL = "kill_findings"
    CONTRADICTIONS = "contradictions"
    BEAR = "bear_case"
    BULL = "bull_case"
    SCENARIOS = "scenarios"
    VERDICT = "verdict"
    USER_PREFERENCES = "user_preferences"
    PRIOR_EVALUATION = "prior_evaluation"
    IDENTITY = "identity"


@dataclass(frozen=True)
class IsolationPolicy:
    """What one agent may read.

    ``reads`` is a strict whitelist of evaluation channels.  ``sees_identity``
    controls whether the ticker/company name is present at all.  ``sees_facts``
    controls the shared verified-evidence set.
    """

    agent_id: str
    reads: frozenset[str] = frozenset()
    sees_identity: bool = True
    sees_facts: bool = True
    sees_risk_flags: bool = True
    sees_contradictions: bool = True
    sees_user_preferences: bool = False
    anonymize: bool = False
    rationale: str = ""

    def denies(self, channel: str) -> bool:
        return channel not in self.reads


_FACT_CONSUMERS = frozenset({Channel.VERIFIED_EVIDENCE})

#: The isolation table.  Every entry states *why*, because a future edit that
#: quietly widens a policy is the failure mode this system exists to prevent.
POLICIES: dict[str, IsolationPolicy] = {
    "fact_collector": IsolationPolicy(
        agent_id="fact_collector",
        reads=frozenset(),
        sees_facts=False,
        sees_risk_flags=False,
        sees_contradictions=False,
        rationale=(
            "Requirement 1A/4-1: collects without evaluating. Sees no analysis at all so it "
            "cannot search selectively for evidence that supports an existing view."
        ),
    ),
    "evidence_integrity": IsolationPolicy(
        agent_id="evidence_integrity",
        reads=frozenset({Channel.RAW_FACTS}),
        sees_risk_flags=False,
        rationale="Requirement 4-2: verifies raw facts; needs no downstream opinion.",
    ),
    "regulatory": IsolationPolicy(
        agent_id="regulatory",
        reads=frozenset(_FACT_CONSUMERS),
        rationale="Requirement 4-3: reads the shared verified evidence set only.",
    ),
    "capital_structure": IsolationPolicy(
        agent_id="capital_structure",
        reads=frozenset(_FACT_CONSUMERS),
        rationale="Requirement 4-4: arithmetic over filings; no narrative input.",
    ),
    "science": IsolationPolicy(
        agent_id="science",
        reads=frozenset(_FACT_CONSUMERS),
        rationale="Requirement 4-5: trial/technology assessment from evidence only.",
    ),
    "competitive": IsolationPolicy(
        agent_id="competitive",
        reads=frozenset(_FACT_CONSUMERS),
        rationale="Requirement 4-9: peer comparison from evidence only.",
    ),
    "catalyst": IsolationPolicy(
        agent_id="catalyst",
        reads=frozenset(_FACT_CONSUMERS),
        rationale="Requirement 4-11: dated events from evidence only.",
    ),
    "microstructure": IsolationPolicy(
        agent_id="microstructure",
        reads=frozenset(_FACT_CONSUMERS),
        rationale=(
            "Requirement 4-12: positioning data must not be mixed with fundamental "
            "judgement, so it reads no fundamental analysis."
        ),
    ),
    "contradiction": IsolationPolicy(
        agent_id="contradiction",
        reads=frozenset(
            {
                Channel.VERIFIED_EVIDENCE,
                Channel.REGULATORY,
                Channel.CAPITAL_STRUCTURE,
                Channel.SCIENCE,
                Channel.COMPETITIVE,
            }
        ),
        rationale=(
            "Requirement 4-13: mechanical cross-checking needs the domain extractions "
            "(which are facts, not verdicts). It never reads bull/bear."
        ),
    ),
    "kill_agent": IsolationPolicy(
        agent_id="kill_agent",
        reads=frozenset(
            {
                Channel.VERIFIED_EVIDENCE,
                Channel.REGULATORY,
                Channel.CAPITAL_STRUCTURE,
                Channel.SCIENCE,
                Channel.CONTRADICTIONS,
            }
        ),
        rationale=(
            "Requirement 4-6/27: hunts for disqualifying facts. Must never see a bull case "
            "-- there must be nothing for it to defend."
        ),
    ),
    "bear_agent": IsolationPolicy(
        agent_id="bear_agent",
        reads=frozenset(
            {
                Channel.VERIFIED_EVIDENCE,
                Channel.REGULATORY,
                Channel.CAPITAL_STRUCTURE,
                Channel.SCIENCE,
                Channel.COMPETITIVE,
                Channel.CONTRADICTIONS,
            }
        ),
        rationale="Requirement 1C/4-7: BULL channel is denied. Same evidence, no rebuttal loop.",
    ),
    "bull_agent": IsolationPolicy(
        agent_id="bull_agent",
        reads=frozenset(
            {
                Channel.VERIFIED_EVIDENCE,
                Channel.REGULATORY,
                Channel.CAPITAL_STRUCTURE,
                Channel.SCIENCE,
                Channel.COMPETITIVE,
                Channel.CONTRADICTIONS,
            }
        ),
        rationale=(
            "Requirement 1C/4-8: BEAR and KILL channels are denied. The bull case must be "
            "built from evidence, not written as a rebuttal."
        ),
    ),
    "valuation": IsolationPolicy(
        agent_id="valuation",
        reads=frozenset({Channel.VERIFIED_EVIDENCE, Channel.CAPITAL_STRUCTURE}),
        rationale=("Requirement 4-10: sees share counts (facts) but no other agent's evaluation."),
    ),
    "blind_judge": IsolationPolicy(
        agent_id="blind_judge",
        reads=frozenset(
            {
                Channel.VERIFIED_EVIDENCE,
                Channel.KILL,
                Channel.CONTRADICTIONS,
                Channel.BULL,
                Channel.BEAR,
                Channel.VALUATION,
                Channel.CATALYSTS,
                Channel.REGULATORY,
                Channel.CAPITAL_STRUCTURE,
                Channel.SCIENCE,
                Channel.COMPETITIVE,
                Channel.SCENARIOS,
            }
        ),
        sees_identity=False,
        anonymize=True,
        sees_user_preferences=False,
        rationale=(
            "Requirement 6/10/14: judges an anonymized pack. Identity, prior scores, prior "
            "rankings and every user preference are denied so the verdict cannot be anchored."
        ),
    ),
    "portfolio": IsolationPolicy(
        agent_id="portfolio",
        reads=frozenset(
            {
                Channel.VERIFIED_EVIDENCE,
                Channel.KILL,
                Channel.CONTRADICTIONS,
                Channel.BULL,
                Channel.BEAR,
                Channel.VALUATION,
                Channel.CATALYSTS,
                Channel.MICROSTRUCTURE,
                Channel.SCENARIOS,
                Channel.VERDICT,
                Channel.USER_PREFERENCES,
                Channel.REGULATORY,
                Channel.CAPITAL_STRUCTURE,
                Channel.SCIENCE,
                Channel.COMPETITIVE,
            }
        ),
        sees_user_preferences=True,
        rationale=(
            "Requirement 10: the ONLY agent permitted to see holdings, cost basis, tax and "
            "concentration -- and it runs after the blind verdict is fixed."
        ),
    ),
}


def policy_for(agent_id: str) -> IsolationPolicy:
    try:
        return POLICIES[agent_id]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise LeakageError(
            f"agent {agent_id!r} has no isolation policy; refusing to run it. "
            "Add an explicit policy rather than defaulting to permissive."
        ) from exc


# --- the bus ----------------------------------------------------------------
@dataclass
class EvidenceBus:
    """Append-only store of everything produced during a run."""

    facts: list[Fact] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    unresolved: list[UnresolvedQuestion] = field(default_factory=list)
    channels: dict[str, Evaluation] = field(default_factory=dict)

    def publish(self, evaluation: Evaluation) -> None:
        self.channels[evaluation.channel] = evaluation

    def add_facts(self, facts: Iterable[Fact]) -> None:
        known = {f.fact_id for f in self.facts}
        for f in facts:
            if f.fact_id not in known:
                self.facts.append(f)
                known.add(f.fact_id)

    def add_sources(self, sources: Iterable[Source]) -> None:
        known = {s.source_id for s in self.sources}
        for s in sources:
            if s.source_id not in known:
                self.sources.append(s)
                known.add(s.source_id)

    def fact_by_id(self, fact_id: str) -> Fact | None:
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        return None


# --- leakage scanning -------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.%]*")


def _serialize(obj: Any) -> str:
    """Stable text rendering of an arbitrary payload for scanning."""

    def default(o: Any) -> Any:
        if hasattr(o, "__dict__"):
            return {k: v for k, v in vars(o).items() if not k.startswith("_")}
        return str(o)

    try:
        return json.dumps(obj, default=default, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(obj)


def identity_markers(
    ticker: str | None, company_name: str | None, aliases: Sequence[str] = ()
) -> list[str]:
    """Strings whose presence would de-anonymize a blind pack."""
    markers: list[str] = []
    for value in (ticker, company_name, *aliases):
        if not value:
            continue
        value = str(value).strip()
        if len(value) < 2:
            continue
        markers.append(value)
        # company-name stems: "Corbus Pharmaceuticals Holdings" -> "Corbus"
        head = value.split()[0]
        if len(head) >= 4 and head.lower() not in _GENERIC_NAME_WORDS:
            markers.append(head)
    return sorted(set(markers), key=len, reverse=True)


_GENERIC_NAME_WORDS = {
    "the",
    "inc",
    "inc.",
    "corp",
    "corp.",
    "company",
    "holdings",
    "group",
    "pharmaceuticals",
    "pharma",
    "therapeutics",
    "biosciences",
    "technologies",
}


@dataclass
class LeakageScanner:
    """Scans a projected payload for forbidden content."""

    denied_fingerprints: Mapping[str, Sequence[str]] = field(default_factory=dict)
    forbidden_identity: Sequence[str] = ()

    def scan(self, agent_id: str, payload: Any) -> list[str]:
        text = _serialize(payload)
        lowered = text.lower()
        violations: list[str] = []

        for channel, tokens in self.denied_fingerprints.items():
            for token in tokens:
                token = str(token).strip().lower()
                if len(token) < 12 or " " not in token:
                    continue  # only multi-word shingles are diagnostic
                if token in lowered:
                    violations.append(
                        f"denied channel {channel!r} fingerprint {token!r} present in "
                        f"input to {agent_id!r}"
                    )
        for marker in self.forbidden_identity:
            if not marker:
                continue
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])", re.I)
            if pattern.search(text):
                violations.append(
                    f"identity marker {marker!r} present in blind input to {agent_id!r}"
                )
        return violations


class IsolationGuard:
    """Builds each agent's input and refuses to release a leaking one."""

    def __init__(self, bus: EvidenceBus, *, strict: bool = True) -> None:
        self.bus = bus
        self.strict = strict
        self.violations: list[str] = []
        #: agent_id -> SourceRefMap for anonymized packs (never handed to agents)
        self.source_ref_maps: dict[str, Any] = {}

    def project(
        self,
        agent_id: str,
        *,
        ticker: str,
        company_name: str,
        aliases: Sequence[str] = (),
        facts: Sequence[Fact] | None = None,
        params: Mapping[str, Any] | None = None,
        user_preferences: Mapping[str, Any] | None = None,
    ) -> AgentInput:
        policy = policy_for(agent_id)
        selected_facts = tuple(facts if facts is not None else self.bus.facts)

        channels = {name: ev for name, ev in self.bus.channels.items() if name in policy.reads}

        params_out: dict[str, Any] = dict(params or {})
        if policy.sees_user_preferences and user_preferences:
            params_out["user_preferences"] = dict(user_preferences)
        elif user_preferences:
            # Requirement 10: preferences are dropped, not merely ignored.
            params_out.pop("user_preferences", None)

        risk_flags = tuple(self.bus.risk_flags) if policy.sees_risk_flags else ()
        contradictions = tuple(self.bus.contradictions) if policy.sees_contradictions else ()
        unresolved = tuple(self.bus.unresolved)

        if policy.anonymize:
            from .anonymize import (
                anonymize_contradictions,
                anonymize_pack,
                anonymize_risk_flags,
                anonymize_unresolved,
            )

            selected_facts, channels, params_out, ref_map = anonymize_pack(
                selected_facts, channels, params_out, ticker, company_name, aliases
            )
            # Everything else in the pack carries prose too, and prose carries
            # the company name.
            risk_flags = anonymize_risk_flags(risk_flags, ticker, company_name, aliases)
            contradictions = anonymize_contradictions(contradictions, ticker, company_name, aliases)
            unresolved = anonymize_unresolved(unresolved, ticker, company_name, aliases)
            # Kept out of band: the map holds real URLs and is used only to
            # restore citations AFTER the blind verdict is fixed.
            self.source_ref_maps[agent_id] = ref_map

        agent_input = AgentInput(
            agent_id=agent_id,
            run_id=params_out.get("run_id", "UNKNOWN"),
            ticker=None if not policy.sees_identity else ticker,
            company_name=None if not policy.sees_identity else company_name,
            facts=selected_facts if policy.sees_facts else (),
            sources=tuple(self.bus.sources) if policy.sees_facts and not policy.anonymize else (),
            risk_flags=risk_flags,
            contradictions=contradictions,
            unresolved=unresolved,
            channels=channels,
            params=params_out,
        )

        self._assert_clean(policy, agent_input, ticker, company_name, aliases, user_preferences)
        return agent_input

    def _assert_clean(
        self,
        policy: IsolationPolicy,
        agent_input: AgentInput,
        ticker: str,
        company_name: str,
        aliases: Sequence[str],
        user_preferences: Mapping[str, Any] | None,
    ) -> None:
        denied_fingerprints: dict[str, Sequence[str]] = {}
        for name, evaluation in self.bus.channels.items():
            if name in policy.reads:
                continue
            tokens = list(evaluation.fingerprint_tokens)
            denied_fingerprints[name] = tokens

        forbidden_identity: list[str] = []
        if not policy.sees_identity:
            from .anonymize import ANON_LABEL

            # A marker identical to the anonymisation label is already
            # anonymous; flagging it would make the pack impossible to build.
            forbidden_identity = [
                marker
                for marker in identity_markers(ticker, company_name, aliases)
                if marker.strip().lower() != ANON_LABEL.lower()
            ]

        scanner = LeakageScanner(denied_fingerprints, forbidden_identity)
        violations = scanner.scan(policy.agent_id, agent_input)

        if not policy.sees_user_preferences and user_preferences:
            pref_text = _serialize(agent_input).lower()
            for key, value in user_preferences.items():
                needle = str(value).strip().lower()
                if len(needle) >= 5 and needle in pref_text:
                    violations.append(
                        f"user preference {key!r} leaked into input to {policy.agent_id!r}"
                    )
            if "user_preferences" in agent_input.params:
                violations.append(f"user_preferences present in params for {policy.agent_id!r}")

        for channel in self.bus.channels:
            if channel not in policy.reads and channel in agent_input.channels:
                violations.append(
                    f"denied channel {channel!r} present in input to {policy.agent_id!r}"
                )

        if violations:
            self.violations.extend(violations)
            log.error("isolation violation: %s", violations)
            if self.strict:
                raise LeakageError("; ".join(violations))


def _shingles(text: str, n: int = 4) -> set[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def derive_fingerprints(
    texts: Sequence[str], baseline_texts: Sequence[str] = (), *, limit: int = 40, n: int = 4
) -> tuple[str, ...]:
    """Word-shingles that are *novel to this evaluation*.

    Naive single-token fingerprints false-positive constantly: a Bear agent may
    legitimately write "pipeline" because the shared evidence says so.  What is
    diagnostic of a leak is an agent reproducing a phrase that exists **only**
    in a denied evaluation.  So the fingerprint is the set of 4-word shingles
    from the evaluation's own prose minus every shingle already present in the
    shared evidence baseline.
    """
    baseline: set[str] = set()
    for text in baseline_texts:
        baseline |= _shingles(text, n)
    novel: list[str] = []
    for text in texts:
        for shingle in sorted(_shingles(text, n) - baseline):
            if len(shingle) < 12:
                continue
            if shingle not in novel:
                novel.append(shingle)
            if len(novel) >= limit:
                return tuple(novel)
    return tuple(novel)


#: Backwards-compatible alias used by agents that have no baseline to compare
#: against (they pass the shared evidence in explicitly where it matters).
def fingerprint_tokens(*texts: str, limit: int = 40) -> tuple[str, ...]:
    return derive_fingerprints(list(texts), (), limit=limit)
