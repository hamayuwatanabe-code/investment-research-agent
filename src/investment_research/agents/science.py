"""Agent 5: Science / Technology.

Two modes, selected from the evidence rather than from a user hint.

**Biotech**: trial design quality is assessed on things that are checkable --
randomization, masking, comparator, enrollment, endpoint type -- and the single
most consequential distinction is drawn explicitly: is the primary endpoint a
*clinical* outcome or a *biomarker/surrogate*?  A surrogate endpoint that no
regulator has accepted, in an underpowered single-arm study, is a weak design no
matter how strong the mechanism story is.

**Technology**: differentiation is assessed on deployment evidence, customer
validation and replicability rather than on architecture description.

Nothing here scores the *investment*.  It scores the evidence about the asset.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import UNKNOWN, FactCategory, Materiality
from ..schemas.fact import Fact, UnresolvedQuestion
from .base import Agent

log = logging.getLogger(__name__)

_SURROGATE_TERMS = (
    "biomarker",
    "surrogate",
    "change from baseline in",
    "serum level",
    "plasma level",
    "response rate",
    "progression-free",
    "tumor size",
    "score at week",
    "composite score",
)
_CLINICAL_OUTCOME_TERMS = (
    "overall survival",
    "mortality",
    "hospitalization",
    "time to death",
    "symptom",
    "functional",
    "quality of life",
    "event-free survival",
    "clinical outcome",
)
_RANDOMIZED_RE = re.compile(r"(?i)\brandomi[sz]ed\b")
_NONRANDOM_RE = re.compile(r"(?i)\bnon[- ]randomi[sz]ed\b|\bsingle[- ]arm\b|allocation is n/?a")
_MASKED_RE = re.compile(r"(?i)\b(double|triple|quadruple)\b|\bmasking is (double|triple|quadruple)")
_OPENLABEL_RE = re.compile(r"(?i)\bopen[- ]label\b|masking is none")
_ENROLLMENT_RE = re.compile(r"(?i)enrollment is (\d+)")
_PHASE_RE = re.compile(r"(?i)\bphase\s+(?:is\s+)?(?:phase\s*)?([0-4](?:/[0-4])?[ab]?)\b")
_POSTHOC_RE = re.compile(r"(?i)\bpost[- ]hoc\b|\bsubgroup analysis\b|\bexploratory endpoint\b")

_TECH_SIGNALS = (
    ("patents", ("patent", "intellectual property")),
    ("deployment", ("deployed", "in production", "installed base", "shipments")),
    ("customer_validation", ("customer", "design win", "purchase order", "contract award")),
    ("switching_cost", ("switching cost", "integration", "certified", "qualified supplier")),
    ("scalability", ("yield", "capacity", "fab", "production scal", "manufacturing scal")),
    ("trl", ("technology readiness", "trl ")),
)


class ScienceAgent(Agent):
    agent_id = "science"
    purpose = "Assess trial design / technology differentiation from checkable evidence"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        clinical = data.facts_in(FactCategory.CLINICAL, FactCategory.SCIENCE)
        technology = data.facts_in(FactCategory.TECHNOLOGY, FactCategory.COMMERCIAL)
        mode = "biotech" if len(clinical) >= len(technology) and clinical else "technology"

        payload: dict[str, Any] = {"mode": mode}
        if mode == "biotech":
            payload.update(self._assess_trial(clinical, out))
        else:
            payload.update(self._assess_technology(technology, out))

        summary = f"Mode {mode}. " + (
            f"Primary endpoint type: {payload.get('endpoint_type', UNKNOWN)}; "
            f"design: {payload.get('design_summary', UNKNOWN)}; "
            f"design quality {payload.get('design_quality_score', UNKNOWN)}/10."
            if mode == "biotech"
            else f"Differentiation evidence found for: {payload.get('signals_present', [])}."
        )
        out.evaluation = self.evaluation(
            Channel.SCIENCE, summary, (), payload, self.baseline_from(data)
        )
        return out

    # -- biotech -----------------------------------------------------------
    def _assess_trial(self, facts: tuple[Fact, ...], out: AgentOutput) -> dict[str, Any]:
        text = " ".join(f.claim for f in facts).lower()

        endpoint_fact = next(
            (
                f
                for f in facts
                if "primary outcome measure" in f.claim.lower()
                or "primary endpoint" in f.claim.lower()
            ),
            None,
        )
        endpoint_text = endpoint_fact.claim.lower() if endpoint_fact else ""
        if any(term in endpoint_text for term in _CLINICAL_OUTCOME_TERMS):
            endpoint_type = "CLINICAL_OUTCOME"
        elif any(term in endpoint_text for term in _SURROGATE_TERMS):
            endpoint_type = "SURROGATE_OR_BIOMARKER"
        else:
            endpoint_type = UNKNOWN

        randomized = bool(_RANDOMIZED_RE.search(text)) and not bool(_NONRANDOM_RE.search(text))
        masked = bool(_MASKED_RE.search(text)) and not bool(_OPENLABEL_RE.search(text))
        enrollment_match = _ENROLLMENT_RE.search(text)
        enrollment = int(enrollment_match.group(1)) if enrollment_match else None
        phase_match = _PHASE_RE.search(text)
        phase = phase_match.group(1) if phase_match else UNKNOWN
        post_hoc = bool(_POSTHOC_RE.search(text))

        # Independent evidence that the surrogate tracks clinical benefit.
        surrogate_validated = UNKNOWN
        for fact in facts:
            claim = fact.claim.lower()
            if "correlat" in claim or "predict" in claim:
                surrogate_validated = (
                    "NOT_SUPPORTED"
                    if (
                        "did not correlate" in claim
                        or "no correlation" in claim
                        or "not predict" in claim
                    )
                    else "SUPPORTED"
                )
                if surrogate_validated == "NOT_SUPPORTED":
                    out.risk_flags.append(
                        RiskFlag(
                            flag_id="sci_surrogate_unvalidated",
                            category=FactCategory.SCIENCE,
                            title="Independent evidence that the surrogate does not track clinical outcome",
                            detail=fact.claim,
                            severity=Materiality.CRITICAL,
                            fact_ids=(fact.fact_id,),
                            raised_by=self.agent_id,
                        )
                    )
                break

        score = 5.0
        notes: list[str] = []
        if randomized:
            score += 1.5
        else:
            score -= 2.0
            notes.append("not randomized (or randomization not evidenced)")
        if masked:
            score += 1.5
        else:
            score -= 1.0
            notes.append("not double-blinded (or masking not evidenced)")
        if endpoint_type == "CLINICAL_OUTCOME":
            score += 1.5
        elif endpoint_type == "SURROGATE_OR_BIOMARKER":
            score -= 1.5
            notes.append("primary endpoint is a surrogate/biomarker rather than a clinical outcome")
        else:
            score -= 1.0
            notes.append("primary endpoint type could not be determined")
        if enrollment is not None:
            if enrollment < 100:
                score -= 1.0
                notes.append(f"small trial (n={enrollment}); effect estimates will be imprecise")
        else:
            notes.append("enrollment unknown")
        if surrogate_validated == "NOT_SUPPORTED":
            score -= 2.0
            notes.append("surrogate is contradicted by independent literature")
        if post_hoc:
            score -= 0.5
            notes.append("post-hoc or subgroup framing present in the evidence")

        if endpoint_type == "SURROGATE_OR_BIOMARKER":
            out.risk_flags.append(
                RiskFlag(
                    flag_id="sci_surrogate_endpoint",
                    category=FactCategory.CLINICAL,
                    title="Primary endpoint is a surrogate or biomarker measure",
                    detail=(
                        "A surrogate primary endpoint only supports approval if the regulator "
                        "accepts it as reasonably likely to predict clinical benefit. That "
                        "acceptance is a separate, independently verifiable fact."
                    ),
                    severity=Materiality.HIGH,
                    fact_ids=(endpoint_fact.fact_id,) if endpoint_fact else (),
                    raised_by=self.agent_id,
                )
            )
        if enrollment is not None and enrollment < 100:
            out.risk_flags.append(
                RiskFlag(
                    flag_id="sci_small_n",
                    category=FactCategory.CLINICAL,
                    title=f"Small sample size (n={enrollment})",
                    detail="Statistical power for a clinically meaningful effect is likely limited.",
                    severity=Materiality.MEDIUM,
                    raised_by=self.agent_id,
                )
            )
        for question, why in (
            (
                "What is the pre-specified statistical power and assumed effect size?",
                "An unpowered trial can fail on a real effect.",
            ),
            (
                "What is the comparator arm and the current standard of care?",
                "Superiority claims are meaningless without the comparator.",
            ),
            (
                "What competitor clinical data exist in the same indication?",
                "A competitor with a clinical endpoint changes the bar.",
            ),
        ):
            out.unresolved.append(
                UnresolvedQuestion(
                    question=question,
                    why_it_matters=why,
                    category=FactCategory.CLINICAL,
                    raised_by=self.agent_id,
                )
            )

        return {
            "phase": phase,
            "randomized": randomized,
            "masked": masked,
            "enrollment": enrollment,
            "endpoint_type": endpoint_type,
            "primary_endpoint": endpoint_fact.claim if endpoint_fact else UNKNOWN,
            "surrogate_validation": surrogate_validated,
            "post_hoc_signals": post_hoc,
            "design_quality_score": round(max(0.0, min(10.0, score)), 1),
            "design_summary": (
                f"{'randomized' if randomized else 'not randomized'}, "
                f"{'masked' if masked else 'open-label/unknown masking'}, "
                f"n={enrollment if enrollment is not None else UNKNOWN}"
            ),
            "design_notes": notes,
        }

    # -- technology --------------------------------------------------------
    def _assess_technology(self, facts: tuple[Fact, ...], out: AgentOutput) -> dict[str, Any]:
        text = " ".join(f.claim for f in facts).lower()
        present = [name for name, terms in _TECH_SIGNALS if any(t in text for t in terms)]
        absent = [name for name, _ in _TECH_SIGNALS if name not in present]
        for name in absent:
            out.unresolved.append(
                UnresolvedQuestion(
                    question=f"No evidence found for technology signal: {name}",
                    why_it_matters=(
                        "Technical differentiation that never shows up in deployments, "
                        "customers or patents is an assertion, not a moat."
                    ),
                    category=FactCategory.TECHNOLOGY,
                    raised_by=self.agent_id,
                )
            )
        if not present:
            out.risk_flags.append(
                RiskFlag(
                    flag_id="sci_no_tech_evidence",
                    category=FactCategory.TECHNOLOGY,
                    title="No independent evidence of technical differentiation",
                    detail="Differentiation rests on description rather than deployment evidence.",
                    severity=Materiality.HIGH,
                    raised_by=self.agent_id,
                )
            )
        return {
            "signals_present": present,
            "signals_absent": absent,
            "design_quality_score": round(10.0 * len(present) / len(_TECH_SIGNALS), 1),
        }
