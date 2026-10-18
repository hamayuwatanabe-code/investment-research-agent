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


#: Phase 4.3L: a prose disclosure that can change WHICH programme is current
#: -- a reported primary-endpoint outcome, a discontinuation, a strategic
#: review, or a lost pivotal status. A lead/current/pivotal description
#: dated before (or on the same day as) any such disclosure no longer
#: proves the current programme. Deliberately broad: a false positive only
#: makes the resolution UNRESOLVED (findings stay provisional and block),
#: while a miss would let a stale description scope a real failure on
#: another trial down to a "different programme" K1. Registry status
#: lines ("overall status is TERMINATED") are NOT matched -- a terminated
#: historical trial does not, by itself, say anything about the lead.
_SUPERSEDING_DISCLOSURE_RE = re.compile(
    r"(?i)"
    r"\b(?:did\s+not|didn't|failed\s+to|does\s+not|fails\s+to)\s+(?:meet|achieve|reach|demonstrate)\b"
    r"[^.]{0,80}\bendpoint"
    r"|\bmiss(?:ed|es)\b[^.]{0,60}\bendpoint"
    r"|\b(?:met|achieved)\b[^.]{0,40}\bprimary\s+(?:efficacy\s+)?endpoint"
    r"|\bprimary\s+(?:efficacy\s+)?endpoint\b[^.]{0,60}\b(?:was|were)\s+(?:not\s+)?(?:met|achieved)"
    r"|\bprimary\s+(?:efficacy\s+)?endpoint\b[^.]{0,80}\bnot\s+statistically\s+significant"
    r"|\b(?:discontinu\w*|terminat\w*|halt\w*|suspend\w*|wind(?:ing)?\s+down|deprioriti[sz]\w*)\b"
    r"[^.]{0,60}\b(?:program|programme|trial|study|development)\b"
    r"|\bstrategic\s+(?:review|alternatives|options|re-?prioriti[sz]ation)\b"
    r"|\bexplor\w*\s+(?:all\s+)?(?:strategic\s+)?(?:options|alternatives)\b"
    r"|\bno\s+longer\b[^.]{0,60}\b(?:pivotal|registrational|lead)\b"
    r"|\bregulatory\s+(?:view|status\s+change|consultation\s+outcome)\s+reported\s+by\s+the\s+issuer\b"
)

def is_superseding_disclosure(fact: Fact) -> bool:
    """Whether ``fact`` reports something that can change WHICH programme
    is current (see ``_SUPERSEDING_DISCLOSURE_RE``) -- read from its claim
    and, when a string, its value (an issuer-8-K fact carries its
    classifier's reading there; research/sec_material_8k.py)."""
    value = fact.value if isinstance(fact.value, str) else ""
    return bool(_SUPERSEDING_DISCLOSURE_RE.search(f"{fact.claim} {value}"))


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _disclosure_date(fact: Fact) -> str | None:
    """The latest ISO date known for a fact (event, publication, filing,
    effective) -- when it was said, as best the record shows. ``None``
    when no date is known: such a fact cannot be placed in time."""
    dates = [
        value[:10]
        for value in (fact.event_date, fact.publication_date, fact.filing_date, fact.effective_date)
        if value and value != UNKNOWN and _ISO_DATE_RE.match(value)
    ]
    return max(dates) if dates else None


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
       still the current programme) -- UNLESS a superseding disclosure (a
       reported primary-endpoint outcome, a discontinuation, a strategic
       review, a lost pivotal status; see ``_SUPERSEDING_DISCLOSURE_RE``)
       is dated on or after the latest such description, or cannot be
       dated. Then the description no longer proves what is current.
    3. Otherwise unresolved: multiple candidates, no way to prove one.

    Phase 4.3L removed the former recency tiebreaker ("the candidate with
    the latest evidence date"): registry ``event_date`` is the PLANNED
    primary completion date, so a newly registered trial always won it, and
    a wrong choice scoped a real primary-endpoint failure on another trial
    down to a K1 "different programme" finding that no longer blocked the
    Action. Recency alone never proves which programme is current.
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

    #: Latest date of each candidate's lead/current/pivotal description
    #: (None: at least one such description, none of them datable).
    marker_dates: dict[str, str | None] = {}
    superseding = [f for f in facts if is_superseding_disclosure(f)]

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
                when = _disclosure_date(fact)
                previous = marker_dates.get(trial_id)
                if trial_id not in marker_dates or (
                    when is not None and (previous is None or when > previous)
                ):
                    marker_dates[trial_id] = when
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
        marker_date = marker_dates.get(chosen.trial_id)
        superseded_by = [
            f for f in superseding
            if marker_date is None
            or (when := _disclosure_date(f)) is None
            or when >= marker_date
        ]
        if not superseding or not superseded_by:
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
        return ProgramResolution(
            relevance_unresolved=True,
            rationale=(
                f"{chosen.trial_id}'s lead/current/pivotal description "
                f"(latest dated {marker_date or 'UNKNOWN'}) is superseded by "
                f"{len(superseded_by)} later or undatable disclosure(s) reporting a trial "
                "outcome, discontinuation, strategic review or lost pivotal status "
                f"({', '.join(f.fact_id for f in superseded_by[:5])}) -- the current "
                "thesis-relevant programme cannot be proven from evidence"
            ),
            candidates=tuple(candidates),
        )

    return ProgramResolution(
        relevance_unresolved=True,
        rationale=(
            f"{len(candidates)} distinct clinical programmes in evidence "
            f"({', '.join(c.trial_id for c in candidates)}) with no unambiguous lead "
            "designation -- the current thesis-relevant programme could not be determined "
            "from evidence (recency alone never decides it)"
        ),
        candidates=tuple(candidates),
    )
