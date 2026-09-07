"""Agent 14: Blind Neutral Judge.

Receives an anonymized evidence pack: no ticker, no company name, no source
URLs (opaque S-NNN refs instead), no prior scores, no prior ranking, and none of
the user's preferences or holdings.  The isolation guard enforces all of that
and raises if any identity marker survives (requirements 6, 10, 14).

The judgement itself is rule-based, in a fixed order, and every rule is
falsifiable.  The order matters: **evidence confidence is established before the
action label**, per requirement 23.
"""

from __future__ import annotations

import logging

from ..orchestrator.isolation import Channel
from ..schemas.agent_io import AgentInput, AgentOutput
from ..schemas.enums import UNKNOWN, Action, RunStatus
from ..schemas.evaluation import KillGateResult, Verdict
from .base import Agent

log = logging.getLogger(__name__)

#: Below this evidence confidence, no buy-side action is available whatever the
#: rest of the analysis says (requirement 9/23).
MIN_CONFIDENCE_FOR_BUY = 6.0
MIN_CONFIDENCE_FOR_ANY_ACTION = 3.0


class BlindJudgeAgent(Agent):
    agent_id = "blind_judge"
    purpose = "Decide from anonymized evidence, with no knowledge of identity or holdings"

    def run(self, data: AgentInput) -> AgentOutput:
        out = AgentOutput(agent_id=self.agent_id)

        kill_channel = data.channel(Channel.KILL)
        gate = KillGateResult.from_channel_payload(kill_channel.payload if kill_channel else {})
        evidence_confidence = float(data.params.get("evidence_confidence", 0.0))
        run_status = RunStatus(data.params.get("run_status", RunStatus.COMPLETE.value))

        regulatory = data.channel(Channel.REGULATORY)
        endpoint_position = (
            regulatory.payload.get("endpoint_position", UNKNOWN) if regulatory else UNKNOWN
        )
        contradictions = data.channel(Channel.CONTRADICTIONS)
        critical_contradictions = (
            contradictions.payload.get("critical_count", 0) if contradictions else 0
        )
        bear = data.channel(Channel.BEAR)
        bull = data.channel(Channel.BULL)

        reasoning: list[str] = []
        red_flags: list[str] = []
        breakers: list[str] = []

        # 1. Kill gate first (requirement 27: kill test before bull case).
        # PROVISIONAL findings are reported with equal prominence to CONFIRMED
        # ones -- hiding an unverified K5 would be worse than never having
        # looked -- but the label makes clear which is which, because only a
        # CONFIRMED finding may drive the final Action (requirement DG2).
        for assessment in sorted(gate.assessments, key=lambda a: -a.level.level):
            if assessment.level.level >= 3:
                label = (
                    f"{assessment.category} = {assessment.level}"
                    if assessment.confirmation.value == "CONFIRMED"
                    else f"Potential {assessment.category} = {assessment.level} (PROVISIONAL)"
                )
                red_flags.append(f"{label}: {assessment.rationale}")
        if gate.unsearched_categories:
            red_flags.append(
                "Kill categories never searched: "
                + ", ".join(str(c) for c in gate.unsearched_categories)
                + " -- reported as unexamined, not clean"
            )

        # 2. Evidence confidence before any action (requirement 23).
        reasoning.append(
            f"Evidence confidence is {evidence_confidence}/10, evaluated before any action label."
        )

        # 3. Action selection.
        action, action_reason = self._select_action(
            gate, evidence_confidence, endpoint_position, run_status
        )
        reasoning.append(action_reason)

        if endpoint_position == "REJECTED":
            breakers.append(
                "Already broken: the regulator does not accept the primary endpoint as adequate "
                "to establish effectiveness. No trial result fixes this."
            )
        elif endpoint_position == "UNKNOWN":
            breakers.append(
                "Evidence that the regulator does not accept the primary endpoint. This is "
                "currently unverified in either direction and would invalidate the thesis."
            )
        if bear:
            breakers.extend(
                f"Bear mechanism: {m.get('mechanism')}"
                for m in bear.payload.get("mechanisms", [])[:4]
            )
        if critical_contradictions:
            red_flags.append(
                f"{critical_contradictions} critical contradiction(s) in the evidence set"
            )
            reasoning.append(
                "Contradictions are reported unresolved rather than reconciled into one story."
            )

        if bull and not bull.payload.get("points"):
            reasoning.append("No evidence-backed undervaluation argument could be constructed.")

        headline = self._headline(action, gate, endpoint_position, evidence_confidence)

        verdict = Verdict(
            action=action,
            evidence_confidence=evidence_confidence,
            headline=headline,
            reasoning=tuple(reasoning),
            thesis_breakers=tuple(breakers),
            critical_red_flags=tuple(red_flags),
            kill_gate=gate,
            run_status=run_status,
            judged_blind=data.ticker is None,
            anonymized_label=str(data.params.get("anonymized_label", "Company X")),
            caveats=tuple(self._caveats(run_status, evidence_confidence, gate)),
        )

        payload = {
            "action": str(action),
            "evidence_confidence": evidence_confidence,
            "headline": headline,
            "reasoning": list(reasoning),
            "thesis_breakers": list(breakers),
            "critical_red_flags": list(red_flags),
            "judged_blind": verdict.judged_blind,
            "max_kill_level": str(gate.max_level),
        }
        out.evaluation = self.evaluation(
            Channel.VERDICT, headline, tuple(reasoning), payload, self.baseline_from(data)
        )
        out.metrics["verdict"] = str(action)
        out.metrics["judged_blind"] = verdict.judged_blind
        out.metrics["_verdict_object"] = verdict
        if not verdict.judged_blind:
            out.degraded = True
            out.errors.append(
                "judge received identity information; the verdict is not a blind verdict"
            )
        return out

    # -- decision rules ----------------------------------------------------
    @staticmethod
    def _select_action(
        gate: KillGateResult,
        evidence_confidence: float,
        endpoint_position: str,
        run_status: RunStatus,
    ) -> tuple[Action, str]:
        """Choose an Action from CONFIRMED kill severity only.

        Requirement DG2/DG6: a PROVISIONAL finding -- however severe if
        true -- must never by itself select AVOID or any other Action. It is
        surfaced as a red flag and, at the pipeline level, as a reason the
        Evidence Sufficiency Matrix withholds an Action entirely
        (BLOCKED_PENDING_VERIFICATION). This function only ever sees the
        question "given what is actually confirmed, what follows" -- the
        caller is responsible for nulling the result when research is not
        COMPLETE.
        """
        max_confirmed = gate.max_confirmed_level
        if max_confirmed.level >= 5:
            return Action.AVOID, (
                f"A CONFIRMED K5 disqualifier is present ({_worst(gate, confirmed_only=True)}). "
                "No upside estimate overrides a disqualifying fact."
            )
        if max_confirmed.level == 4:
            return Action.AVOID, (
                f"A CONFIRMED K4 severe issue is present ({_worst(gate, confirmed_only=True)}). "
                "The default at K4 is to avoid."
            )
        if run_status == RunStatus.INCOMPLETE_RESEARCH:
            return Action.WAIT_FOR_EVENT, (
                "Research is incomplete: parts of the mandatory evidence set were not obtained, "
                "so no buy-side action is available yet."
            )
        if evidence_confidence < MIN_CONFIDENCE_FOR_ANY_ACTION:
            return Action.AVOID, (
                f"Evidence confidence {evidence_confidence}/10 is too low to support any "
                "position, in either direction."
            )
        if max_confirmed.level == 3:
            return Action.WAIT_FOR_EVENT, (
                f"A CONFIRMED K3 major red flag is present ({_worst(gate, confirmed_only=True)}). "
                "The flag must be resolved before a position is justified."
            )
        if evidence_confidence < MIN_CONFIDENCE_FOR_BUY:
            return Action.WAIT_FOR_EVENT, (
                f"Evidence confidence {evidence_confidence}/10 is below the threshold for a buy "
                "decision. The gap is in the evidence, not in the opportunity."
            )
        if endpoint_position == "UNKNOWN":
            return Action.WAIT_FOR_EVENT, (
                "The regulator's position on the primary endpoint is unverified, which is the "
                "single highest-value unknown in this kind of thesis."
            )
        if max_confirmed.level == 2:
            return Action.BUY_ON_PULLBACK, (
                "Meaningful but manageable confirmed issues; entry price matters."
            )
        if endpoint_position == "AGREED" and evidence_confidence >= 8.0:
            return Action.BUY, (
                "The regulator has agreed the endpoint and the evidence set is strong."
            )
        return Action.HOLD, "No disqualifying issue and no strong evidence-backed entry case."

    @staticmethod
    def _headline(
        action: Action, gate: KillGateResult, endpoint_position: str, confidence: float
    ) -> str:
        return (
            f"Evidence confidence {confidence}/10; worst kill level {gate.max_level}; "
            f"regulator endpoint position {endpoint_position}; action {action}."
        )

    @staticmethod
    def _caveats(run_status: RunStatus, confidence: float, gate: KillGateResult) -> list[str]:
        caveats: list[str] = []
        if run_status == RunStatus.INCOMPLETE_RESEARCH:
            caveats.append(
                "INCOMPLETE RESEARCH: one or more agents or collectors failed. This is not a "
                "completed analysis."
            )
        if confidence < 5.0:
            caveats.append(
                "Evidence confidence is low; the scores describe a possibility, not an "
                "established situation."
            )
        if gate.unsearched_categories:
            caveats.append(
                "Some kill categories were never searched; their K0 means 'not examined'."
            )
        return caveats


def _worst(gate: KillGateResult, *, confirmed_only: bool = False) -> str:
    pool = gate.assessments
    if confirmed_only:
        pool = tuple(a for a in pool if a.confirmation.value == "CONFIRMED")
    worst = max(pool, key=lambda a: a.level.level, default=None)
    return f"{worst.category} {worst.level}" if worst else UNKNOWN
