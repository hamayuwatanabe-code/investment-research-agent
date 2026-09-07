"""LLM-backed Kill, Bear, Bull and Blind Judge agents (requirement P2).

The isolation properties from Phase 1 hold unchanged here, and are now enforced
at two levels rather than one: the agent's ``AgentInput`` is projected through
its policy, and the rendered prompt is scanned before transmission.

Note what the LLM is and is not allowed to decide:

* The Kill agent **proposes** levels; the deterministic Kill Gate still computes
  the binding assessment and the score caps. A model cannot argue a K5 down.
* The Blind Judge **reasons** over an anonymised pack, but the action rules that
  turn a K4/K5 into AVOID, and the "confidence before action" ordering, remain
  deterministic post-conditions applied to whatever it returns.
"""

from __future__ import annotations

import logging
from typing import Any

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput, RiskFlag
from ..schemas.enums import (
    UNKNOWN,
    FactCategory,
    KillCategory,
    KillLevel,
    Materiality,
    RunStatus,
)
from ..schemas.evaluation import Verdict
from ..schemas.fact import UnresolvedQuestion
from .blind_judge import BlindJudgeAgent, _rebuild_gate
from .llm_agents import _EVIDENCE_ITEM, _SEVERITY, _pack_for
from .llm_base import LLMAgent, PromptBuildResult, cited_only

log = logging.getLogger(__name__)

_KILL_CATEGORIES = [c.value for c in KillCategory]
_KILL_LEVELS = [level.value for level in KillLevel]


class LLMKillAgent(LLMAgent):
    agent_id = "kill_agent"
    tool_name = "submit_kill_findings"
    tool_description = "Report disqualifying findings. Do not defend the candidate."
    pack_budget_tokens = 16000
    system_extra = """\
You are the Kill agent. Your only job is to find reasons to DISCARD this candidate. You are not
balancing anything, and you have deliberately not been shown any bull case -- there is nothing
here for you to defend.

Report a finding for every disqualifying or seriously damaging fact you can evidence, across:
REGULATORY, CLINICAL, SCIENCE, CAPITAL, COMMERCIAL, GOVERNANCE, ACCOUNTING, LIQUIDITY.

Assign each a level: K0 none, K1 minor, K2 meaningful but manageable, K3 major red flag,
K4 severe / normally avoid, K5 disqualifier.

The level you assign is a PROPOSAL. A deterministic gate computes the binding assessment from
the same evidence and will not accept a level that the evidence does not support. So do not
inflate, and do not soften: state what the evidence shows.

Weight primary sources heavily. A serious claim carried only by commentary is a flag for
verification, not a disqualification on its own."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["category", "level", "title", "detail", "evidence_ids"],
                    "properties": {
                        "category": {"type": "string", "enum": _KILL_CATEGORIES},
                        "level": {"type": "string", "enum": _KILL_LEVELS},
                        "title": {"type": "string", "maxLength": 200},
                        "detail": {"type": "string", "maxLength": 1500},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "primary_source": {"type": "boolean"},
                    },
                },
            },
            "follow_up_queries": {
                "type": "array",
                "maxItems": 15,
                "items": {"type": "string", "maxLength": 200},
            },
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        pack, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens, self.chunks)
        searches = self.discovery_summary
        return PromptBuildResult(
            prompt=(
                "Find every reason to discard this candidate.\n\n"
                "EVIDENCE:\n"
                + rendered
                + "\n\n"
                + (f"ADVERSARIAL SEARCH RESULTS:\n{searches}\n\n" if searches else "")
                + "Report findings with category, level and evidence ids. Also propose "
                "follow_up_queries: specific web searches that would confirm or refute the "
                "most damaging possibilities implied by this evidence."
            ),
            pack=pack,
            cited_ids=tuple(valid),
        )

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        _, _, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens, self.chunks)
        findings, dropped = cited_only(payload.get("findings", []), valid)

        proposed: list[dict[str, Any]] = []
        for item in findings:
            level = KillLevel(str(item["level"]))
            category = KillCategory(str(item["category"]))
            proposed.append(
                {
                    "category": str(category),
                    "level": str(level),
                    "title": str(item["title"]),
                    "detail": str(item["detail"]),
                    "fact_ids": list(item.get("evidence_ids", [])),
                    "primary_source": bool(item.get("primary_source", False)),
                }
            )
            if level.level >= 3:
                out.risk_flags.append(
                    RiskFlag(
                        flag_id=f"kill_llm_{category.value.lower()}_{len(out.risk_flags)}",
                        category=_kill_to_fact_category(category),
                        title=f"{category} proposed {level}: {item['title']}",
                        detail=str(item["detail"]),
                        severity=(Materiality.CRITICAL if level.level >= 4 else Materiality.HIGH),
                        fact_ids=tuple(item.get("evidence_ids", [])),
                        raised_by=self.agent_id,
                    )
                )

        out.evaluation = self.evaluation(
            Channel.KILL,
            f"{len(proposed)} proposed kill finding(s); the deterministic gate computes the "
            "binding assessment.",
            tuple(f["title"] for f in proposed),
            {
                "llm_proposed_findings": proposed,
                "follow_up_queries": payload.get("follow_up_queries", []),
                "dropped_uncited": dropped and len(payload.get("findings", [])) - len(findings),
            },
            self.baseline_from(data),
        )
        out.metrics["proposed_findings"] = len(proposed)
        out.metrics["follow_up_queries"] = len(payload.get("follow_up_queries", []))
        return out


def _kill_to_fact_category(category: KillCategory) -> FactCategory:
    from .kill_agent import _kill_to_fact_category as mapper

    return mapper(category)


# --------------------------------------------------------------------------
class LLMBearAgent(LLMAgent):
    agent_id = "bear_agent"
    tool_name = "submit_bear_case"
    tool_description = "Construct the most plausible failure scenario from the evidence."
    system_extra = """\
You are the Bear agent. Construct the most plausible scenario in which this investment loses
money, grounded entirely in the evidence you were given.

You have NOT been shown the bull case, and you are not rebutting anything. Build an argument.

Distinguish yourself from the Kill agent: it hunts for facts that disqualify; you describe how a
loss actually occurs even when nothing is formally disqualifying. Dilution before a catalyst,
a readout on an endpoint no regulator accepts, a competitor arriving first -- these are paths,
not verdicts.

If the evidence is too thin to specify a failure path, say so. That is a real finding and more
useful than an invented mechanism."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["mechanisms", "primary_failure_path"],
        "properties": {
            "primary_failure_path": {"type": "string", "maxLength": 600},
            "mechanisms": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["mechanism", "path", "severity", "evidence_ids"],
                    "properties": {
                        "mechanism": {"type": "string", "maxLength": 300},
                        "path": {"type": "string", "maxLength": 1500},
                        "severity": {
                            "type": "string",
                            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        },
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        pack, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens, self.chunks)
        return PromptBuildResult(
            prompt=(
                "Construct the most plausible way this investment loses money.\n\n"
                "EVIDENCE:\n" + rendered + "\n\n"
                "Give each mechanism a concrete path from the current situation to the loss, "
                "with evidence ids."
            ),
            pack=pack,
            cited_ids=tuple(valid),
        )

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        _, _, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens, self.chunks)
        mechanisms, dropped = cited_only(payload.get("mechanisms", []), valid)
        ordered = sorted(
            mechanisms,
            key=lambda m: -_SEVERITY.get(str(m.get("severity", "MEDIUM")), Materiality.MEDIUM).rank,
        )
        summary = str(payload.get("primary_failure_path", "")) or (
            ordered[0]["mechanism"] if ordered else "No specific failure path could be evidenced."
        )
        out.evaluation = self.evaluation(
            Channel.BEAR,
            summary,
            tuple(f"{m['mechanism']}: {m['path']}" for m in ordered[:6]),
            {
                "mechanisms": [
                    {
                        "mechanism": m["mechanism"],
                        "path": m["path"],
                        "severity": m.get("severity", "MEDIUM"),
                        "fact_ids": m.get("evidence_ids", []),
                    }
                    for m in ordered
                ],
                "primary_failure_path": summary,
                "evidence_count": len(valid),
                "dropped_uncited": len(dropped),
            },
            self.baseline_from(data),
        )
        out.metrics["mechanism_count"] = len(ordered)
        return out


# --------------------------------------------------------------------------
class LLMBullAgent(LLMAgent):
    agent_id = "bull_agent"
    tool_name = "submit_bull_case"
    tool_description = "Assess whether the evidence supports undervaluation."
    system_extra = """\
You are the Bull agent. Assess whether the evidence supports the market undervaluing this asset.

You have NOT been shown the bear case or the kill findings. You are building an argument from
evidence, not defending against objections you cannot see.

Two hard constraints:
  - Every point must cite evidence. A point you cannot cite must be dropped, not hedged.
  - A finding that contradicts the thesis is not support merely because it is independent and
    scientific. Read what the evidence actually says before citing it.

If no evidence-backed undervaluation case can be built, say so plainly. That is a legitimate
finding, not a failure of imagination."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["points"],
        "properties": {
            "points": {"type": "array", "items": _EVIDENCE_ITEM, "maxItems": 12},
            "summary": {"type": "string", "maxLength": 800},
            "no_case_reason": {"type": "string", "maxLength": 800},
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        pack, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens, self.chunks)
        return PromptBuildResult(
            prompt=(
                "Assess whether this evidence supports the market undervaluing the asset.\n\n"
                "EVIDENCE:\n" + rendered + "\n\n"
                "Give each point with its evidence ids. If there is no evidence-backed case, "
                "return an empty points list and explain why in no_case_reason."
            ),
            pack=pack,
            cited_ids=tuple(valid),
        )

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        _, _, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens, self.chunks)
        points, dropped = cited_only(payload.get("points", []), valid)

        if not points:
            summary = str(payload.get("no_case_reason", "")) or (
                "No verified evidence supports an undervaluation case."
            )
            out.unresolved.append(
                UnresolvedQuestion(
                    question="What verifiable evidence would support an undervaluation case?",
                    why_it_matters="A bull case that cannot be built from evidence is a narrative.",
                    category=FactCategory.OTHER,
                    raised_by=self.agent_id,
                )
            )
        else:
            summary = str(payload.get("summary", "")) or (
                f"{len(points)} evidence-backed argument(s) for undervaluation."
            )

        out.evaluation = self.evaluation(
            Channel.BULL,
            summary,
            tuple(p["statement"] for p in points),
            {
                "points": [
                    {
                        "claim": p["statement"],
                        "fact_ids": p.get("evidence_ids", []),
                        "category": "llm",
                    }
                    for p in points
                ],
                "dropped_unsupported_points": [p["statement"] for p in dropped],
                "decision_grade_fact_count": len(data.decision_grade_facts()),
                "evidence_used": sorted({i for p in points for i in p.get("evidence_ids", [])}),
            },
            self.baseline_from(data),
        )
        out.metrics["supported_points"] = len(points)
        out.metrics["dropped_points"] = len(dropped)
        return out


# --------------------------------------------------------------------------
class LLMBlindJudgeAgent(LLMAgent):
    agent_id = "blind_judge"
    tool_name = "submit_verdict"
    tool_description = "Return the reasoning and thesis breakers for an anonymised candidate."
    pack_budget_tokens = 20000
    system_extra = """\
You are the Blind Judge. You are evaluating an anonymised candidate referred to only as
Company X. You do not know its name or ticker, you have not been told any previous assessment,
and you do not know whether anyone holds it. Do not speculate about any of that.

Reason from the evidence, the kill findings, the contradictions, and both cases, and produce:
  - the findings that would invalidate the thesis (thesis_breakers)
  - the critical red flags
  - your reasoning

Do NOT choose the final action label. A deterministic rule set applies the action from the kill
gate, the evidence confidence and the research completeness, and it will overrule any preference
you express. Your job is the reasoning that makes that decision legible."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reasoning", "thesis_breakers", "critical_red_flags"],
        "properties": {
            "headline": {"type": "string", "maxLength": 600},
            "reasoning": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "thesis_breakers": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "critical_red_flags": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "decisive_consideration": {"type": "string", "maxLength": 800},
        },
    }

    def build_prompt(self, data: AgentInput) -> PromptBuildResult:
        _, rendered, valid = _pack_for(self.agent_id, data, self.pack_budget_tokens, self.chunks)
        sections = [f"EVIDENCE ABOUT COMPANY X:\n{rendered}"]
        for label, channel in (
            ("KILL FINDINGS", Channel.KILL),
            ("CONTRADICTIONS", Channel.CONTRADICTIONS),
            ("BULL CASE", Channel.BULL),
            ("BEAR CASE", Channel.BEAR),
            ("VALUATION", Channel.VALUATION),
            ("CATALYSTS", Channel.CATALYSTS),
        ):
            evaluation = data.channel(channel)
            if evaluation:
                points = "\n".join(f"- {p}" for p in evaluation.points[:12])
                sections.append(f"{label}:\n{evaluation.summary}\n{points}")
        sections.append(
            f"EVIDENCE CONFIDENCE: {data.params.get('evidence_confidence', 'UNKNOWN')} / 10\n"
            f"RESEARCH STATUS: {data.params.get('run_status', 'UNKNOWN')}"
        )
        return PromptBuildResult(
            prompt="\n\n====\n\n".join(sections)
            + "\n\n====\n\nProduce your reasoning, the thesis breakers and the critical red "
            "flags. Do not choose an action label.",
            cited_ids=tuple(valid),
        )

    def interpret(self, payload: dict[str, Any], data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)
        kill_channel = data.channel(Channel.KILL)
        gate = _rebuild_gate(kill_channel.payload if kill_channel else {})
        confidence = float(data.params.get("evidence_confidence", 0.0))
        run_status = RunStatus(data.params.get("run_status", RunStatus.COMPLETE.value))
        regulatory = data.channel(Channel.REGULATORY)
        endpoint_position = (
            regulatory.payload.get("endpoint_position", UNKNOWN) if regulatory else UNKNOWN
        )

        # The action is decided by the same deterministic rules as the
        # non-LLM judge. The model supplies reasoning, never the label.
        action, action_reason = BlindJudgeAgent._select_action(
            gate, confidence, endpoint_position, run_status
        )

        reasoning = [
            f"Evidence confidence is {confidence}/10, evaluated before any action label.",
            *[str(r) for r in payload.get("reasoning", [])],
            action_reason,
        ]
        headline = str(payload.get("headline", "")) or (
            f"Evidence confidence {confidence}/10; worst kill level {gate.max_level}; "
            f"regulator endpoint position {endpoint_position}; action {action}."
        )
        verdict = Verdict(
            action=action,
            evidence_confidence=confidence,
            headline=headline,
            reasoning=tuple(reasoning),
            thesis_breakers=tuple(str(b) for b in payload.get("thesis_breakers", [])),
            critical_red_flags=tuple(str(f) for f in payload.get("critical_red_flags", [])),
            kill_gate=gate,
            run_status=run_status,
            judged_blind=data.ticker is None,
            anonymized_label=str(data.params.get("anonymized_label", "Company X")),
            caveats=tuple(BlindJudgeAgent._caveats(run_status, confidence, gate)),
        )
        out.evaluation = self.evaluation(
            Channel.VERDICT,
            headline,
            tuple(reasoning),
            {
                "action": str(action),
                "evidence_confidence": confidence,
                "headline": headline,
                "reasoning": reasoning,
                "thesis_breakers": list(verdict.thesis_breakers),
                "critical_red_flags": list(verdict.critical_red_flags),
                "judged_blind": verdict.judged_blind,
                "max_kill_level": str(gate.max_level),
                "decisive_consideration": payload.get("decisive_consideration", ""),
            },
            self.baseline_from(data),
        )
        out.metrics["verdict"] = str(action)
        out.metrics["judged_blind"] = verdict.judged_blind
        out.metrics["_verdict_object"] = verdict
        if not verdict.judged_blind:
            out.degraded = True
            out.errors.append("judge received identity information; verdict is not blind")
        return out
