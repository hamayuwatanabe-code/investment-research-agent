"""Agent 3: Regulatory / Legal.

This agent exists because of one specific failure: a thesis built on
designations, DMC continuations and analyst enthusiasm, while the regulator's
actual position -- that the primary endpoint was not acceptable to establish
effectiveness -- sat unread in a filing.

Its discipline is therefore narrow and strict:

* A company adjective ("constructive", "productive", "aligned with FDA") is
  **never** evidence about the regulator's position.  It is extracted as a
  company characterization and held next to the regulator's own words.
* The output is structured as three lists: what the regulator agreed to, what it
  did **not** agree to, and what is unresolved.  The middle list is the one that
  kills theses, so it is never omitted or summarized away.
* When the regulator's position on endpoint acceptability is simply not in
  evidence, that is reported as an unresolved blocking question -- not as an
  absence of risk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import UNKNOWN, FactCategory, Materiality, SourceTier
from ..schemas.fact import Fact, UnresolvedQuestion
from .base import Agent

log = logging.getLogger(__name__)


# --- regulator-position patterns -------------------------------------------
# Matched against filing text, which is where an adverse regulator position is
# usually disclosed verbatim.
@dataclass(frozen=True)
class RegulatoryPattern:
    key: str
    label: str
    pattern: re.Pattern[str]
    severity: Materiality
    disagreement: bool


ADVERSE_PATTERNS: tuple[RegulatoryPattern, ...] = (
    RegulatoryPattern(
        "endpoint_not_acceptable",
        "FDA does not accept the primary endpoint as adequate to establish effectiveness",
        re.compile(
            r"(?i)(?:does not|did not|would not)\s+(?:consider|agree|view|regard|accept)[^.]{0,120}"
            r"(?:primary\s+)?endpoint[^.]{0,120}(?:appropriate|adequate|acceptable|sufficient|"
            r"establish\s+effectiveness|establish\s+efficacy)"
            r"|endpoint[^.]{0,80}(?:not|isn't)\s+(?:considered\s+)?(?:appropriate|adequate|acceptable)"
            r"[^.]{0,60}(?:establish|support)\s+(?:effectiveness|efficacy)"
        ),
        Materiality.CRITICAL,
        True,
    ),
    RegulatoryPattern(
        "additional_trial_required",
        "Regulator requires an additional adequate and well-controlled trial",
        re.compile(
            r"(?i)(?:additional|another|a\s+second)\s+(?:adequate\s+and\s+well[- ]controlled\s+)?"
            r"(?:trial|study|studies)[^.]{0,100}?\b(?:required|necessary|needed|must\s+be\s+conducted)\b"
        ),
        Materiality.CRITICAL,
        True,
    ),
    RegulatoryPattern(
        "clinical_hold",
        "Clinical hold",
        re.compile(r"(?i)\bclinical\s+hold\b"),
        Materiality.CRITICAL,
        True,
    ),
    RegulatoryPattern(
        "complete_response_letter",
        "Complete Response Letter",
        re.compile(r"(?i)\bcomplete\s+response\s+letter\b|\bCRL\b"),
        Materiality.CRITICAL,
        True,
    ),
    RegulatoryPattern(
        "refuse_to_file",
        "Refuse-to-File",
        re.compile(r"(?i)\brefuse[- ]to[- ]file\b|\brefusal\s+to\s+file\b"),
        Materiality.CRITICAL,
        True,
    ),
    RegulatoryPattern(
        "surrogate_not_accepted",
        "Surrogate endpoint not accepted as reasonably likely to predict benefit",
        re.compile(
            r"(?i)surrogate[^.]{0,100}(?:not\s+(?:reasonably\s+likely|accepted|validated)|"
            r"(?:does not|did not)\s+(?:agree|consider))"
        ),
        Materiality.CRITICAL,
        True,
    ),
    RegulatoryPattern(
        "accelerated_approval_not_available",
        "Accelerated approval pathway not available or not agreed",
        re.compile(
            r"(?i)accelerated\s+approval[^.]{0,120}(?:not\s+(?:available|eligible|appropriate)|"
            r"(?:does not|did not|would not)\s+(?:agree|support))"
        ),
        Materiality.CRITICAL,
        True,
    ),
    RegulatoryPattern(
        "inspection_483_warning",
        "Inspection findings (Form 483 / Warning Letter)",
        re.compile(r"(?i)\bform\s+483\b|\bwarning\s+letter\b|\buntitled\s+letter\b"),
        Materiality.HIGH,
        True,
    ),
    RegulatoryPattern(
        "cmc_deficiency",
        "CMC / manufacturing deficiency",
        re.compile(
            r"(?i)\b(?:CMC|chemistry,?\s+manufacturing)\b[^.]{0,120}"
            r"(?:deficienc|issue|concern|comment|inadequate)"
        ),
        Materiality.HIGH,
        True,
    ),
)

FAVOURABLE_PATTERNS: tuple[RegulatoryPattern, ...] = (
    RegulatoryPattern(
        "spa_agreement",
        "Special Protocol Assessment agreement in place",
        re.compile(r"(?i)special\s+protocol\s+assessment[^.]{0,80}(?:agreement|agreed|reached)"),
        Materiality.HIGH,
        False,
    ),
    RegulatoryPattern(
        "endpoint_agreed",
        "Regulator agreed the primary endpoint",
        re.compile(
            r"(?i)(?:FDA|EMA|PMDA|agency)[^.]{0,80}agreed[^.]{0,80}(?:primary\s+)?endpoint"
        ),
        Materiality.CRITICAL,
        False,
    ),
    RegulatoryPattern(
        "breakthrough",
        "Breakthrough Therapy designation",
        re.compile(r"(?i)\bbreakthrough\s+therapy\b"),
        Materiality.MEDIUM,
        False,
    ),
    RegulatoryPattern(
        "fast_track",
        "Fast Track designation",
        re.compile(r"(?i)\bfast\s+track\b"),
        Materiality.LOW,
        False,
    ),
    RegulatoryPattern(
        "orphan_drug",
        "Orphan Drug designation",
        re.compile(r"(?i)\borphan\s+drug\b"),
        Materiality.LOW,
        False,
    ),
    RegulatoryPattern(
        "rmat",
        "RMAT designation",
        re.compile(r"(?i)\bRMAT\b|regenerative\s+medicine\s+advanced\s+therapy"),
        Materiality.MEDIUM,
        False,
    ),
    RegulatoryPattern(
        "priority_review",
        "Priority Review",
        re.compile(r"(?i)\bpriority\s+review\b"),
        Materiality.MEDIUM,
        False,
    ),
)

#: Adjectives companies use to describe regulator interactions.  Extracted so
#: they can be shown *as company characterizations*, never as regulator facts.
COMPANY_FRAMING_RE = re.compile(
    r"(?i)\b(constructive|productive|positive|encouraging|collaborative|aligned|"
    r"supportive|favorable|favourable|on\s+track)\b"
)

#: Designations that say nothing about efficacy-evidence acceptability.  Naming
#: this explicitly matters: designation count is the classic false comfort.
PROCEDURAL_ONLY_DESIGNATIONS = {"fast_track", "orphan_drug", "priority_review"}

MEETING_RE = re.compile(r"(?i)\btype\s+([abcd])\s+meeting\b|\bpre-?(?:IND|NDA|BLA)\s+meeting\b")


class RegulatoryAgent(Agent):
    agent_id = "regulatory"
    purpose = "Extract what the regulator agreed to, refused, and left unresolved"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        relevant = [
            f
            for f in data.facts
            if f.category
            in (FactCategory.REGULATORY, FactCategory.CLINICAL, FactCategory.LEGAL)
            or _looks_regulatory(f)
        ]

        agreed: list[str] = []
        not_agreed: list[str] = []
        unresolved_items: list[str] = []
        company_framings: list[str] = []
        matched_keys: set[str] = set()
        events: list[dict] = []

        for fact in relevant:
            text = f"{fact.claim} {fact.notes}"
            excerpt = fact.value if isinstance(fact.value, str) else ""
            haystack = f"{text} {excerpt}"

            for pattern in ADVERSE_PATTERNS:
                if pattern.pattern.search(haystack):
                    matched_keys.add(pattern.key)
                    entry = f"{pattern.label} [{fact.fact_id}] (source tier {fact.source_tier})"
                    not_agreed.append(entry)
                    out.risk_flags.append(
                        RiskFlag(
                            flag_id=f"reg_{pattern.key}_{fact.fact_id[-6:]}",
                            category=FactCategory.REGULATORY,
                            title=pattern.label,
                            detail=(
                                f"Regulator position extracted from {fact.source_title!r} "
                                f"({fact.source_tier}): {fact.claim}"
                            ),
                            severity=pattern.severity,
                            fact_ids=(fact.fact_id,),
                            raised_by=self.agent_id,
                        )
                    )
                    events.append(
                        {
                            "event_id": f"reg_{pattern.key}_{fact.fact_id[-8:]}",
                            "regulator": _regulator_of(fact),
                            "event_type": pattern.key,
                            "event_date": fact.event_date,
                            "agreed": UNKNOWN,
                            "not_agreed": pattern.label,
                            "unresolved": UNKNOWN,
                            "company_framing": "",
                            "regulator_statement": fact.claim[:1000],
                            "fact_ids": fact.fact_id,
                        }
                    )

            for pattern in FAVOURABLE_PATTERNS:
                if pattern.pattern.search(haystack):
                    matched_keys.add(pattern.key)
                    note = ""
                    if pattern.key in PROCEDURAL_ONLY_DESIGNATIONS:
                        note = (
                            " -- procedural designation only: says nothing about whether the "
                            "efficacy evidence or endpoint will be accepted"
                        )
                    agreed.append(f"{pattern.label}{note} [{fact.fact_id}]")

            if fact.company_claim and COMPANY_FRAMING_RE.search(fact.claim):
                adjectives = sorted(set(m.lower() for m in COMPANY_FRAMING_RE.findall(fact.claim)))
                company_framings.append(
                    f"Company describes the interaction as {adjectives} [{fact.fact_id}] "
                    f"-- characterization only, not a regulator statement"
                )

            if MEETING_RE.search(haystack):
                matched_keys.add("regulator_meeting")

        # --- the decisive question -----------------------------------------
        endpoint_position = self._endpoint_position(matched_keys)
        if endpoint_position == "UNKNOWN":
            unresolved_items.append(
                "No evidence found stating whether the regulator accepts the primary endpoint "
                "as adequate to establish effectiveness"
            )
            out.unresolved.append(
                UnresolvedQuestion(
                    question=(
                        "Has the regulator agreed that the primary endpoint is adequate to "
                        "establish effectiveness for the intended indication?"
                    ),
                    why_it_matters=(
                        "This single question has more power to invalidate a clinical-stage "
                        "thesis than every designation, DMC continuation and analyst target "
                        "combined. Absence of evidence here is not a favourable answer."
                    ),
                    blocking=True,
                    category=FactCategory.REGULATORY,
                    raised_by=self.agent_id,
                )
            )
            out.risk_flags.append(
                RiskFlag(
                    flag_id="reg_endpoint_position_unknown",
                    category=FactCategory.REGULATORY,
                    title="Regulator position on primary endpoint is unknown",
                    detail=(
                        "No primary-source evidence of the regulator's view on endpoint "
                        "acceptability was obtained. Treated as unexamined, not as clean."
                    ),
                    severity=Materiality.HIGH,
                    raised_by=self.agent_id,
                )
            )

        if company_framings and endpoint_position == "REJECTED":
            unresolved_items.append(
                "Company characterization of the regulator interaction conflicts with the "
                "regulator position found in primary filings"
            )

        for question in _STANDING_QUESTIONS:
            if question["key"] not in matched_keys:
                out.unresolved.append(
                    UnresolvedQuestion(
                        question=question["text"],
                        why_it_matters=question["why"],
                        blocking=bool(question.get("blocking")),
                        category=FactCategory.REGULATORY,
                        raised_by=self.agent_id,
                    )
                )

        payload = {
            "endpoint_position": endpoint_position,
            "agreed": agreed,
            "not_agreed": not_agreed,
            "unresolved": unresolved_items,
            "company_framing": company_framings,
            "matched_signals": sorted(matched_keys),
            "regulatory_events": events,
            "primary_source_backed": [
                f.fact_id
                for f in relevant
                if f.source_tier in (SourceTier.TIER_1, SourceTier.TIER_2)
            ],
        }
        summary = (
            f"Regulator position on primary endpoint: {endpoint_position}. "
            f"{len(agreed)} agreed item(s), {len(not_agreed)} refused/adverse item(s), "
            f"{len(unresolved_items)} unresolved."
        )
        out.evaluation = self.evaluation(
            Channel.REGULATORY,
            summary,
            tuple(not_agreed + unresolved_items),
            payload,
            self.baseline_from(data),
        )
        out.metrics["endpoint_position"] = endpoint_position
        out.metrics["adverse_signals"] = len(not_agreed)
        return out

    @staticmethod
    def _endpoint_position(matched: set[str]) -> str:
        if "endpoint_not_acceptable" in matched or "additional_trial_required" in matched:
            return "REJECTED"
        if "surrogate_not_accepted" in matched:
            return "REJECTED"
        if "endpoint_agreed" in matched or "spa_agreement" in matched:
            return "AGREED"
        return "UNKNOWN"


_STANDING_QUESTIONS = (
    {
        "key": "regulator_meeting",
        "text": "What was discussed and decided at the most recent Type A/B/C meeting?",
        "why": "Meeting minutes state the regulator's position in its own words.",
        "blocking": False,
    },
    {
        "key": "spa_agreement",
        "text": "Is there a Special Protocol Assessment, and does it remain in force?",
        "why": "An SPA is the strongest available evidence that a trial design is acceptable.",
        "blocking": False,
    },
    {
        "key": "complete_response_letter",
        "text": "Has the programme received a Complete Response Letter?",
        "why": "A prior CRL usually names the exact deficiency that must be cured.",
        "blocking": False,
    },
    {
        "key": "cmc_deficiency",
        "text": "Are there open CMC or manufacturing issues?",
        "why": "CMC failures delay approvals independently of clinical results.",
        "blocking": False,
    },
)


def _looks_regulatory(fact: Fact) -> bool:
    text = fact.claim.lower()
    return any(
        token in text
        for token in ("fda", "ema", "pmda", "regulator", "agency", "approval", "endpoint")
    )


def _regulator_of(fact: Fact) -> str:
    text = fact.claim.lower()
    for name in ("fda", "ema", "pmda", "mhra", "health canada", "nmpa"):
        if name in text:
            return name.upper()
    return UNKNOWN
