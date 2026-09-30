"""Programme/thesis-relevance resolution (requirement D).

A live run picked an unrelated historical Phase 2 trial for Science's design
assessment, and separately let a terminated/withdrawn trial in a DIFFERENT
programme/indication drive a company-level K4 kill finding -- simply because
it was *a* trial with a matching status keyword, not because it was the trial
the current investment thesis actually rests on.

This module determines, from evidence alone, which clinical programme is the
thesis-relevant *current lead* one: asset/product, trial identifier,
development phase, current status, recency, and any explicit company
description of a programme as lead/current/pivotal/registrational. It never
simply picks the first Phase 2/3 trial encountered, and it never guesses: when
the evidence does not resolve to one programme, ``relevance_unresolved`` is
True and ``trial_id`` stays UNKNOWN -- callers (Science, the Kill Gate) must
treat that as "we could not tell", never as "there is no current programme".

Deliberately deterministic and dependency-free, like the Kill Gate itself: two
components that route which trial's evidence is allowed to drive a company-
level disqualification must not depend on a model's mood.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schemas.enums import UNKNOWN, FactCategory
from ..schemas.fact import Fact

NCT_RE = re.compile(r"\b(NCT[-A-Z0-9]{4,})\b")

#: A fact explicitly describing a programme as the current/lead/pivotal one.
#: Deliberately broad on wording (company disclosures phrase this many ways)
#: but narrow on requiring one of these specific relationship words -- being
#: a Phase 2/3 trial is not, by itself, evidence of being the CURRENT one.
_LEAD_MARKER_RE = re.compile(
    r"(?i)\b(?:lead|current|pivotal|registrational)\b[^.]{0,60}"
    r"\b(?:program|programme|candidate|trial|study|asset|indication)\b"
    r"|\b(?:program|programme|candidate|trial|study)\b[^.]{0,60}"
    r"\b(?:is|as)\b[^.]{0,20}\b(?:the\s+)?(?:lead|current|pivotal|registrational)\b"
)


@dataclass(frozen=True)
class ProgramCandidate:
    """One clinical programme (trial identifier) found in the evidence."""

    trial_id: str
    status: str = UNKNOWN
    phase: str = UNKNOWN
    has_lead_marker: bool = False
    most_recent_date: str = UNKNOWN
    fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramResolution:
    """Which clinical programme is the thesis-relevant current one.

    ``trial_id`` is UNKNOWN whenever this cannot be determined from evidence
    -- never a guess. ``relevance_unresolved`` is what downstream Science and
    Kill Gate scoping must check before treating a trial-specific fact as
    describing the current programme: True means "could not tell", not
    "there is no current programme".
    """

    trial_id: str = UNKNOWN
    status: str = UNKNOWN
    phase: str = UNKNOWN
    relevance_unresolved: bool = True
    rationale: str = ""
    candidates: tuple[ProgramCandidate, ...] = field(default_factory=tuple)


def canonical_program_resolution(resolution: ProgramResolution) -> dict[str, Any]:
    """A complete, JSON-safe, order-stable view of every semantic field on
    this ``ProgramResolution`` and its ``candidates`` (Phase 4.3D correction
    1).

    This is the ONE place that view is built -- both
    :func:`canonical_program_resolution_fingerprint` (used by
    ``agents.base.inputs_hash``) and any future caller needing the full
    semantic content read it from here, never a partial or ad hoc
    reconstruction. It is deliberately broader than ``agents.llm_agents``'s
    own ``_program_resolution_projection`` (the LLMScienceAgent prompt's
    limited, model-facing view of trial_id/relevance_unresolved/rationale/
    candidates{trial_id,status,phase}) -- that one stays narrow on purpose,
    for what a model needs to see; this one must be complete, for what
    "did the resolution actually change" needs to detect.

    Never depends on dataclass ``repr()``, memory identity, ``hash()``, or
    set/dict iteration order: every field is read explicitly by name, dict
    keys are the fixed literals below (a plain Python dict already preserves
    insertion order, and callers that serialize this with ``json.dumps`` may
    also pass ``sort_keys=True`` for a second, independent guarantee), and
    ``fact_ids`` -- the one field whose members carry no order-dependent
    meaning of their own (which facts mention this trial, not a ranking) --
    is explicitly sorted rather than left in whatever incidental order
    ``resolve_current_program`` happened to build it in. ``candidates``
    itself is NOT reordered: its tuple order is ``resolve_current_program``'s
    own deterministic build order, which two calls over the same facts always
    reproduce identically -- reordering it would erase that reproducibility
    guarantee instead of protecting it.
    """
    return {
        "trial_id": resolution.trial_id,
        "status": resolution.status,
        "phase": resolution.phase,
        "relevance_unresolved": resolution.relevance_unresolved,
        "rationale": resolution.rationale,
        "candidates": [
            {
                "trial_id": c.trial_id,
                "status": c.status,
                "phase": c.phase,
                "has_lead_marker": c.has_lead_marker,
                "most_recent_date": c.most_recent_date,
                "fact_ids": sorted(c.fact_ids),
            }
            for c in resolution.candidates
        ],
    }


def canonical_program_resolution_fingerprint(resolution: ProgramResolution | None) -> str:
    """The canonical JSON text of ``resolution``, or ``""`` when there is
    none -- exactly what ``agents.base.inputs_hash`` folds into an agent's
    input fingerprint. ``sort_keys=True`` is a second, independent guarantee
    of stable key order on top of ``canonical_program_resolution``'s own
    (already insertion-ordered) dict, so this string is reproducible
    regardless of how that dict was constructed.
    """
    if resolution is None:
        return ""
    return json.dumps(
        canonical_program_resolution(resolution), sort_keys=True, ensure_ascii=False
    )


def resolve_current_program(facts: Sequence[Fact]) -> ProgramResolution:
    """Resolve the current lead clinical programme from the evidence in hand.

    Priority, each evidence-based rather than assumed:

    1. Only one clinical programme in the evidence -> trivially that one.
    2. Exactly one candidate is explicitly described (by the company or a
       registry entry) as lead/current/pivotal/registrational -> that one,
       regardless of its status (a genuinely terminated CURRENT programme is
       still the current programme).
    3. Otherwise, if exactly one candidate's most recent evidence date is
       unambiguously later than every other candidate's -> that one, as a
       recency tiebreaker.
    4. Otherwise unresolved: multiple candidates, no way to prefer one.
    """
    clinical = [f for f in facts if f.category in (FactCategory.CLINICAL, FactCategory.SCIENCE)]
    by_trial: dict[str, list[Fact]] = {}
    order: list[str] = []
    for fact in clinical:
        match = NCT_RE.search(fact.claim)
        if not match:
            continue
        trial_id = match.group(1)
        if trial_id not in by_trial:
            by_trial[trial_id] = []
            order.append(trial_id)
        by_trial[trial_id].append(fact)

    if not by_trial:
        return ProgramResolution(
            rationale="no trial identifier found anywhere in the clinical evidence"
        )

    candidates: list[ProgramCandidate] = []
    for trial_id in order:
        trial_facts = by_trial[trial_id]
        status = UNKNOWN
        phase = UNKNOWN
        has_marker = False
        most_recent = UNKNOWN
        for fact in trial_facts:
            claim_lower = fact.claim.lower()
            if "overall status is" in claim_lower and isinstance(fact.value, str):
                status = fact.value
            if "phase is" in claim_lower and isinstance(fact.value, str):
                phase = fact.value
            if _LEAD_MARKER_RE.search(fact.claim):
                has_marker = True
            if fact.event_date and fact.event_date != UNKNOWN and (
                most_recent == UNKNOWN or fact.event_date > most_recent
            ):
                most_recent = fact.event_date
        candidates.append(
            ProgramCandidate(
                trial_id=trial_id,
                status=status,
                phase=phase,
                has_lead_marker=has_marker,
                most_recent_date=most_recent,
                fact_ids=tuple(f.fact_id for f in trial_facts),
            )
        )

    if len(candidates) == 1:
        only = candidates[0]
        return ProgramResolution(
            trial_id=only.trial_id,
            status=only.status,
            phase=only.phase,
            relevance_unresolved=False,
            rationale=f"only one clinical programme in evidence: {only.trial_id}",
            candidates=tuple(candidates),
        )

    marked = [c for c in candidates if c.has_lead_marker]
    if len(marked) == 1:
        chosen = marked[0]
        return ProgramResolution(
            trial_id=chosen.trial_id,
            status=chosen.status,
            phase=chosen.phase,
            relevance_unresolved=False,
            rationale=(
                f"{chosen.trial_id} is explicitly described in the evidence as the "
                "lead/current/pivotal/registrational programme"
            ),
            candidates=tuple(candidates),
        )

    if not marked:
        dated = [c for c in candidates if c.most_recent_date != UNKNOWN]
        dated.sort(key=lambda c: c.most_recent_date, reverse=True)
        if len(dated) >= 1 and (len(dated) == 1 or dated[0].most_recent_date > dated[1].most_recent_date):
            newest = dated[0]
            return ProgramResolution(
                trial_id=newest.trial_id,
                status=newest.status,
                phase=newest.phase,
                relevance_unresolved=False,
                rationale=(
                    f"{newest.trial_id} has the most recent evidence dates among "
                    f"{len(candidates)} clinical programmes found, with no explicit lead "
                    "designation for any of them"
                ),
                candidates=tuple(candidates),
            )

    return ProgramResolution(
        relevance_unresolved=True,
        rationale=(
            f"{len(candidates)} distinct clinical programmes in evidence "
            f"({', '.join(c.trial_id for c in candidates)}) with no unambiguous lead "
            "designation or recency signal -- the current thesis-relevant programme could "
            "not be determined from evidence"
        ),
        candidates=tuple(candidates),
    )
