"""LLM-backed implementations of the eight interpretive agents (requirement P2).

Which agents are here, and why only these:

    Regulatory, Science, Kill, Bear, Bull, Competitive, Contradiction, Blind Judge

All eight do *document interpretation* -- reading prose and deciding what it
means. That is what a language model is good at and what a regular expression is
bad at: "the FDA no longer refers to the trial as pivotal" carries a
disqualifying meaning that no pattern list anticipates.

What stays deterministic, and is NOT here: share counts, fully diluted maths,
runway, market cap, valuation arithmetic, Kill Gate enforcement, source tiering,
evidence routing and isolation. Those are the parts where a plausible-looking
wrong answer is most damaging, so they never depend on a model.

Every agent below therefore *proposes*; the deterministic Kill Gate still
*disposes*. An LLM cannot lower a kill level or raise a capped score.
"""

from __future__ import annotations

import logging
from typing import Any

from ..collectors.documents import Chunk, build_evidence_pack
from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import UNKNOWN, FactCategory, Materiality
from ..schemas.fact import Contradiction, UnresolvedQuestion, make_contradiction_id
from .llm_base import LLMAgent, PromptBuildResult, cited_only

log = logging.getLogger(__name__)

_SEVERITY = {
    "CRITICAL": Materiality.CRITICAL,
    "HIGH": Materiality.HIGH,
    "MEDIUM": Materiality.MEDIUM,
    "LOW": Materiality.LOW,
    "INFORMATIONAL": Materiality.INFORMATIONAL,
}

_EVIDENCE_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["statement", "evidence_ids"],
    "properties": {
        "statement": {"type": "string", "maxLength": 1200},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "severity": {
            "type": "string",
            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"],
        },
    },
}


def _pack_for(agent_id: str, data: AgentInput, budget: int) -> tuple[Any, str, set[str]]:
    """Build this agent's evidence pack and the id set its citations may use."""
    chunks: list[Chunk] = list(data.params.get("_chunks") or [])
    valid: set[str] = {f.fact_id for f in data.facts}
    if chunks:
        pack = build_evidence_pack(chunks, agent_id, budget_tokens=budget)
        valid |= {c.chunk_id for c in pack.chunks}
        return pack, pack.render(), valid

    lines = [
        f"[{f.fact_id}] ({f.category}/{f.evidence_class}/{f.source_tier}"
        f"/confidence {f.confidence:.2f}) {f.claim}\n"
        f"    source: {f.source_title} | url: {f.source_url} | "
        f"published: {f.publication_date} | event: {f.event_date}"
        for f in data.facts
    ]
    return None, "\n".join(lines) or "(no evidence)", valid


def _flags(items: list[dict[str, Any]], agent_id: str, category: FactCategory) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    for index, item in enumerate(items):
        flags.append(
            RiskFlag(
                flag_id=f"{agent_id}_llm_{index}",
                category=category,
                title=str(item["statement"])[:160],
                detail=str(item["statement"]),
                severity=_SEVERITY.get(str(item.get("severity", "MEDIUM")), Materiality.MEDIUM),
                fact_ids=tuple(item.get("evidence_ids", [])),
                raised_by=agent_id,
            )
        )
    return flags


# --------------------------------------------------------------------------
class LLMRegulatoryAgent(LLMAgent):
    agent_id = "regulatory"
    tool_name = "submit_regulatory_analysis"
    tool_description = "Report what the regulator agreed, refused and left unresolved."
    pack_budget_tokens = 14000
    system_extra = """\
You are the Regulatory agent. Your single most important job is to separate what the REGULATOR
said from what the COMPANY said about what the regulator said.

Answer one question above all others: has the regulator agreed that the trial's primary endpoint
is adequate to establish effectiveness? Read carefully for indirect phrasings, because this is
rarely stated plainly. All of the following mean the endpoint is NOT accepted:
  - "is not sufficient to demonstrate efficacy"
  - "does not consider the endpoint appropriate to establish effectiveness"
  - "no longer refers to the trial as pivotal" (a withdrawal of registrational status)
  - "recommended that only the most objective measures ... are likely to be acceptable"
  - a requirement for an additional adequate and well-controlled trial

A press release headline calling a meeting "constructive" is a company characterisation. Record
it as such, next to what the regulator actually said. If both appear, that is a contradiction
and you must report it.

Designations (Fast Track, Orphan Drug, Rare Pediatric Disease, RMAT, Priority Review) are
procedural. They say NOTHING about whether the efficacy evidence or endpoint will be accepted,
and must be labelled that way wherever you list them."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["endpoint_position", "agreed", "not_agreed", "unresolved"],
        "properties": {
            "endpoint_position": {
                "type": "string",
                "enum": ["AGREED", "REJECTED", "UNKNOWN", "NOT_APPLICABLE"],
            },
            "endpoint_position_reasoning": {"type": "string", "maxLength": 2000},
            "agreed": {"type": "array", "items": _EVIDENCE_ITEM, "maxItems": 15},
            "not_agreed": {"type": "array", "items": _EVIDENCE_ITEM, "maxItems": 15},
            "unresolved": {"type": "array", "items": _EVIDENCE_ITEM, "maxItems": 15},
            "company_framing": {"type": "array", "items": _EVIDENCE_ITEM, "maxItems": 10},
            "procedural_designations": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
            },
            "registrational_status_changed": {"type": "boolean"},
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        pack, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)
        prompt = (
            "Analyse the regulatory position of the company described by this evidence.\n\n"
            "EVIDENCE:\n" + rendered + "\n\n"
            "Report:\n"
            "1. endpoint_position: has the regulator agreed the primary endpoint is adequate to "
            "establish effectiveness? Use NOT_APPLICABLE only if this business has no approval "
            "pathway at all.\n"
            "2. agreed / not_agreed / unresolved: what the regulator agreed, refused, and left "
            "open. Cite evidence ids for each.\n"
            "3. company_framing: any company adjective about a regulator interaction.\n"
            "4. registrational_status_changed: true if a trial previously called pivotal or "
            "registrational is no longer described that way.\n"
        )
        return PromptBuildResult(prompt=prompt, pack=pack, cited_ids=tuple(valid))

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        _, _, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)

        not_agreed, dropped = cited_only(payload.get("not_agreed", []), valid)
        agreed, _ = cited_only(payload.get("agreed", []), valid)
        unresolved, _ = cited_only(payload.get("unresolved", []), valid)
        framing, _ = cited_only(payload.get("company_framing", []), valid)

        position = str(payload.get("endpoint_position", UNKNOWN))
        out.risk_flags.extend(_flags(not_agreed, self.agent_id, FactCategory.REGULATORY))

        if payload.get("registrational_status_changed"):
            out.risk_flags.append(
                RiskFlag(
                    flag_id="reg_registrational_status_withdrawn",
                    category=FactCategory.REGULATORY,
                    title="Trial is no longer described as pivotal or registrational",
                    detail=(
                        "A change in registrational status is a regulator withdrawing the "
                        "premise the valuation rests on, without any new trial result."
                    ),
                    severity=Materiality.CRITICAL,
                    raised_by=self.agent_id,
                )
            )
        if position == "UNKNOWN":
            out.unresolved.append(
                UnresolvedQuestion(
                    question=(
                        "Has the regulator agreed the primary endpoint is adequate to establish "
                        "effectiveness?"
                    ),
                    why_it_matters=(
                        "This one question invalidates more clinical-stage theses than every "
                        "designation and analyst target combined."
                    ),
                    blocking=True,
                    category=FactCategory.REGULATORY,
                    raised_by=self.agent_id,
                )
            )
        for item in unresolved:
            out.unresolved.append(
                UnresolvedQuestion(
                    question=str(item["statement"]),
                    why_it_matters="Raised by regulatory analysis of the evidence.",
                    category=FactCategory.REGULATORY,
                    raised_by=self.agent_id,
                )
            )

        result = {
            "endpoint_position": position,
            "endpoint_position_reasoning": payload.get("endpoint_position_reasoning", ""),
            "agreed": [i["statement"] for i in agreed],
            "not_agreed": [i["statement"] for i in not_agreed],
            "unresolved": [i["statement"] for i in unresolved],
            "company_framing": [i["statement"] for i in framing],
            "procedural_designations": payload.get("procedural_designations", []),
            "registrational_status_changed": bool(
                payload.get("registrational_status_changed", False)
            ),
            "regulatory_events": [],
            "primary_source_backed": sorted(
                {i for item in not_agreed for i in item.get("evidence_ids", [])}
            ),
            "dropped_uncited": len(dropped),
        }
        out.evaluation = self.evaluation(
            Channel.REGULATORY,
            f"Regulator position on primary endpoint: {position}. "
            f"{len(agreed)} agreed, {len(not_agreed)} refused/adverse, {len(unresolved)} open.",
            tuple(i["statement"] for i in not_agreed),
            result,
            self.baseline_from(data),
        )
        out.metrics["endpoint_position"] = position
        out.metrics["dropped_uncited"] = len(dropped)
        return out


# --------------------------------------------------------------------------
class LLMScienceAgent(LLMAgent):
    agent_id = "science"
    tool_name = "submit_science_analysis"
    tool_description = "Assess trial design or technology differentiation from the evidence."
    system_extra = """\
You are the Science/Technology agent. Assess the asset on checkable attributes, not on the
persuasiveness of the mechanism story.

For a clinical asset the single most consequential distinction is whether the primary endpoint
is a CLINICAL OUTCOME (mortality, transplant-free survival, hospitalisation, MACE, symptoms) or
a SURROGATE/BIOMARKER (an imaging measure such as ejection fraction, a lab value, a composite
score). A surrogate only supports approval if a regulator accepts it as reasonably likely to
predict clinical benefit -- a separate fact you must not assume.

For a technology asset, differentiation that never appears in deployments, customers, patents or
independent testing is an assertion, not a moat."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode", "design_quality_score", "findings"],
        "properties": {
            "mode": {"type": "string", "enum": ["biotech", "technology", "unknown"]},
            "endpoint_type": {
                "type": "string",
                "enum": ["CLINICAL_OUTCOME", "SURROGATE_OR_BIOMARKER", "UNKNOWN"],
            },
            "primary_endpoint": {"type": "string", "maxLength": 600},
            "phase": {"type": "string", "maxLength": 40},
            "randomized": {"type": "string", "enum": ["YES", "NO", "UNKNOWN"]},
            "masked": {"type": "string", "enum": ["YES", "NO", "UNKNOWN"]},
            "enrollment": {"type": "integer", "minimum": 0},
            "surrogate_validation": {
                "type": "string",
                "enum": ["SUPPORTED", "NOT_SUPPORTED", "UNKNOWN"],
            },
            "design_quality_score": {"type": "number", "minimum": 0, "maximum": 10},
            "design_notes": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "findings": {"type": "array", "items": _EVIDENCE_ITEM, "maxItems": 12},
            "signals_present": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        pack, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)
        return PromptBuildResult(
            prompt=(
                "Assess the scientific or technical quality of the asset from this evidence.\n\n"
                "EVIDENCE:\n" + rendered + "\n\n"
                "State whether the primary endpoint is a clinical outcome or a surrogate, and "
                "score design quality 0-10 on checkable attributes only. Anything not evidenced "
                "is UNKNOWN."
            ),
            pack=pack,
            cited_ids=tuple(valid),
        )

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        _, _, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)
        findings, dropped = cited_only(payload.get("findings", []), valid)
        out.risk_flags.extend(_flags(findings, self.agent_id, FactCategory.CLINICAL))

        endpoint_type = str(payload.get("endpoint_type", UNKNOWN))
        if endpoint_type == "SURROGATE_OR_BIOMARKER":
            out.risk_flags.append(
                RiskFlag(
                    flag_id="sci_surrogate_endpoint",
                    category=FactCategory.CLINICAL,
                    title="Primary endpoint is a surrogate or biomarker measure",
                    detail=(
                        "A surrogate endpoint supports approval only if the regulator accepts it "
                        "as reasonably likely to predict clinical benefit -- an independent fact."
                    ),
                    severity=Materiality.HIGH,
                    raised_by=self.agent_id,
                )
            )

        result = {
            "mode": payload.get("mode", "unknown"),
            "endpoint_type": endpoint_type,
            "primary_endpoint": payload.get("primary_endpoint", UNKNOWN),
            "phase": payload.get("phase", UNKNOWN),
            "randomized": payload.get("randomized", UNKNOWN) == "YES",
            "masked": payload.get("masked", UNKNOWN) == "YES",
            "enrollment": payload.get("enrollment"),
            "surrogate_validation": payload.get("surrogate_validation", UNKNOWN),
            "design_quality_score": float(payload.get("design_quality_score", 5.0)),
            "design_notes": payload.get("design_notes", []),
            "signals_present": payload.get("signals_present", []),
            "dropped_uncited": len(dropped),
        }
        out.evaluation = self.evaluation(
            Channel.SCIENCE,
            f"Mode {result['mode']}. Endpoint type {endpoint_type}; design quality "
            f"{result['design_quality_score']}/10.",
            tuple(i["statement"] for i in findings),
            result,
            self.baseline_from(data),
        )
        return out


# --------------------------------------------------------------------------
class LLMCompetitiveAgent(LLMAgent):
    agent_id = "competitive"
    tool_name = "submit_competitive_analysis"
    tool_description = "Compare against peers and separate TAM from SAM and SOM."
    system_extra = """\
You are the Competitive Intelligence agent. Separate three different things that are routinely
conflated:
  TAM  everyone with the condition or need
  SAM  those this company could serve given its likely label, geography and channel
  SOM  those it could realistically capture given competitors and distribution

A company slide multiplying prevalence by an assumed price is an assumption, not a market. Say
so. If the evidence supports only one of the three, the other two are UNKNOWN -- do not derive
them."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["competitors", "tam_basis", "som_basis"],
        "properties": {
            "competitors": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "evidence_ids"],
                    "properties": {
                        "name": {"type": "string", "maxLength": 200},
                        "position": {"type": "string", "maxLength": 600},
                        "ahead": {"type": "boolean"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "tam_basis": {"type": "string", "maxLength": 800},
            "sam_basis": {"type": "string", "maxLength": 800},
            "som_basis": {"type": "string", "maxLength": 800},
            "dimensions_covered": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
            "findings": {"type": "array", "items": _EVIDENCE_ITEM, "maxItems": 10},
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        pack, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)
        return PromptBuildResult(
            prompt=(
                "Identify competitors and characterise the addressable market.\n\n"
                "EVIDENCE:\n" + rendered + "\n\n"
                "Name each competitor you can evidence, say whether it is further advanced, and "
                "describe the basis for TAM, SAM and SOM separately. Where a figure rests on a "
                "company assumption, say so."
            ),
            pack=pack,
            cited_ids=tuple(valid),
        )

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        _, _, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)
        competitors, dropped = cited_only(payload.get("competitors", []), valid)
        for competitor in competitors:
            if competitor.get("ahead"):
                out.risk_flags.append(
                    RiskFlag(
                        flag_id=f"comp_ahead_{abs(hash(competitor['name'])) % 100000}",
                        category=FactCategory.COMPETITION,
                        title=f"Competitor further advanced: {competitor['name']}",
                        detail=str(competitor.get("position", "")),
                        severity=Materiality.HIGH,
                        fact_ids=tuple(competitor.get("evidence_ids", [])),
                        raised_by=self.agent_id,
                    )
                )
        if payload.get("som_basis", "").strip().upper() in ("", UNKNOWN):
            out.unresolved.append(
                UnresolvedQuestion(
                    question="What share of the serviceable market is realistically obtainable?",
                    why_it_matters=(
                        "Multiples derived from TAM rather than SOM are the standard way a "
                        "small-cap thesis overstates its own ceiling."
                    ),
                    category=FactCategory.MARKET_SIZE,
                    raised_by=self.agent_id,
                )
            )

        result = {
            "competitors": competitors,
            "competitor_count": len(competitors),
            "tam": None,
            "sam": None,
            "som": None,
            "tam_basis": payload.get("tam_basis", UNKNOWN),
            "sam_basis": payload.get("sam_basis", UNKNOWN),
            "som_basis": payload.get("som_basis", UNKNOWN),
            "market_size_notes": [
                f"TAM basis: {payload.get('tam_basis', UNKNOWN)}",
                f"SAM basis: {payload.get('sam_basis', UNKNOWN)}",
                f"SOM basis: {payload.get('som_basis', UNKNOWN)}",
            ],
            "dimensions_covered": payload.get("dimensions_covered", []),
            "dimensions_missing": [],
            "dropped_uncited": len(dropped),
        }
        if len(competitors) < 3:
            out.degraded = True
            out.errors.append(
                f"only {len(competitors)} competitor(s) evidenced; at least 3 are required"
            )
        out.evaluation = self.evaluation(
            Channel.COMPETITIVE,
            f"{len(competitors)} competitor(s) identified from evidence.",
            (),
            result,
            self.baseline_from(data),
        )
        return out


# --------------------------------------------------------------------------
class LLMContradictionAgent(LLMAgent):
    agent_id = "contradiction"
    tool_name = "submit_contradictions"
    tool_description = "Report semantic contradictions in the evidence without resolving them."
    system_extra = """\
You are the Contradiction agent. Find places where the evidence disagrees with itself, and
report the disagreement. You must NEVER resolve a contradiction toward the more attractive
reading, and you must never resolve one at all.

Look especially for:
  - a company adjective about a regulator interaction versus the regulator's actual position
  - a press release versus the statutory filing describing the same matter
  - an analyst view versus a primary source
  - a status that changed without being announced as a change (for example a trial described as
    pivotal in one document and not in a later one)
  - guidance that moved between two dates
  - a market size inconsistent with the stated patient population"""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["contradictions"],
        "properties": {
            "contradictions": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "description", "left", "right", "severity"],
                    "properties": {
                        "kind": {"type": "string", "maxLength": 80},
                        "description": {"type": "string", "maxLength": 1200},
                        "left": {"type": "string", "maxLength": 600},
                        "right": {"type": "string", "maxLength": 600},
                        "left_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "right_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "severity": {
                            "type": "string",
                            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        },
                    },
                },
            }
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        pack, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)
        return PromptBuildResult(
            prompt=(
                "Find contradictions within this evidence set. Quote both sides.\n\n"
                "EVIDENCE:\n" + rendered + "\n\n"
                "Report each contradiction with its kind, both sides, and a severity. Do not "
                "resolve any of them."
            ),
            pack=pack,
            cited_ids=tuple(valid),
        )

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        _, _, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens)
        ticker = data.ticker or UNKNOWN

        kept = 0
        dropped = 0
        for item in payload.get("contradictions", []):
            left_ids = [i for i in (item.get("left_evidence_ids") or []) if i in valid]
            right_ids = [i for i in (item.get("right_evidence_ids") or []) if i in valid]
            if not left_ids or not right_ids:
                dropped += 1
                continue
            kept += 1
            out.contradictions.append(
                Contradiction(
                    contradiction_id=make_contradiction_id(
                        ticker, str(item["kind"]), left_ids[0], right_ids[0]
                    ),
                    ticker=ticker,
                    kind=str(item["kind"]),
                    description=str(item["description"]),
                    left_fact_id=left_ids[0],
                    right_fact_id=right_ids[0],
                    left_summary=str(item["left"]),
                    right_summary=str(item["right"]),
                    severity=_SEVERITY.get(str(item["severity"]), Materiality.MEDIUM),
                )
            )

        critical = sum(
            1 for c in out.contradictions if c.severity == Materiality.CRITICAL
        )
        out.evaluation = self.evaluation(
            Channel.CONTRADICTIONS,
            f"{kept} contradiction(s) found, {critical} critical. Conflicts are reported as "
            "conflicts, not reconciled.",
            (),
            {
                "contradictions": [c.to_row() for c in out.contradictions],
                "count": kept,
                "critical_count": critical,
                "checks_run": ["llm_semantic_contradiction_scan"],
                "dropped_uncited": dropped,
            },
            self.baseline_from(data),
        )
        out.metrics["contradiction_count"] = kept
        return out
