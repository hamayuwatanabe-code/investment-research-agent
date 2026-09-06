"""Thesis versioning (requirement 12).

A new run never silently replaces the previous conclusion.  It produces a new
thesis version recording WHAT_CHANGED, WHY_CHANGED, NEW_FACT,
REMOVED_ASSUMPTION and SCORE_CHANGE, so that a change of mind is visible as a
change of mind -- including a change in the system's own favour.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..schemas.evaluation import ScoreCard, Verdict


def diff_against_previous(
    previous: sqlite3.Row | None,
    verdict: Verdict,
    card: ScoreCard,
    current_fact_ids: set[str],
    previous_fact_ids: set[str],
) -> dict[str, Any]:
    if previous is None:
        return {
            "WHAT_CHANGED": ["First recorded thesis for this ticker."],
            "WHY_CHANGED": ["No prior run to compare against."],
            "NEW_FACT": sorted(current_fact_ids),
            "REMOVED_ASSUMPTION": [],
            "SCORE_CHANGE": {},
            "is_first": True,
        }

    changed: list[str] = []
    why: list[str] = []

    if str(previous["action"]) != str(verdict.action):
        changed.append(f"Action: {previous['action']} -> {verdict.action}")
        why.append("The action label changed; see the kill gate and evidence confidence below.")
    if abs(float(previous["evidence_confidence"]) - verdict.evidence_confidence) >= 0.5:
        changed.append(
            f"Evidence confidence: {previous['evidence_confidence']} -> {verdict.evidence_confidence}"
        )
        why.append("The quantity or quality of obtained evidence changed.")
    if str(previous["max_kill_level"]) != str(verdict.kill_gate.max_level):
        changed.append(
            f"Worst kill level: {previous['max_kill_level']} -> {verdict.kill_gate.max_level}"
        )
        why.append("A kill-category assessment moved.")

    try:
        snapshot = json.loads(previous["snapshot"] or "{}")
    except (json.JSONDecodeError, TypeError):
        snapshot = {}
    previous_scores: dict[str, float] = snapshot.get("scores", {})
    score_change: dict[str, list[float]] = {}
    for dimension, value in card.scores.items():
        old = previous_scores.get(dimension)
        if old is not None and abs(float(old) - float(value)) >= 0.5:
            score_change[dimension] = [float(old), float(value)]

    new_facts = sorted(current_fact_ids - previous_fact_ids)
    removed = sorted(previous_fact_ids - current_fact_ids)

    if new_facts:
        changed.append(f"{len(new_facts)} new fact(s) entered the evidence set.")
    if removed:
        changed.append(
            f"{len(removed)} previously-held fact(s) are no longer in the evidence set."
        )
        why.append(
            "A fact that disappears is not a fact that was disproven; it may indicate a "
            "collection failure and should be checked."
        )
    if not changed:
        changed.append("No material change from the previous thesis version.")

    return {
        "WHAT_CHANGED": changed,
        "WHY_CHANGED": why or ["No material change."],
        "NEW_FACT": new_facts,
        "REMOVED_ASSUMPTION": removed,
        "SCORE_CHANGE": score_change,
        "is_first": False,
    }


def snapshot_for(card: ScoreCard, verdict: Verdict, fact_ids: set[str]) -> dict[str, Any]:
    return {
        "scores": card.scores,
        "evidence_confidence": card.evidence_confidence,
        "action": str(verdict.action),
        "max_kill_level": str(verdict.kill_gate.max_level),
        "fact_ids": sorted(fact_ids),
    }
