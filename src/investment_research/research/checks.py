"""Type shapes for the acquisition/evaluation split proposed in Phase 0B.

Phase 1 scope only: these are pure dataclasses. Nothing here is constructed by
the pipeline, an agent, or a collector this turn -- see ``legacy_catalog.py``
for the one concrete producer of ``LegacyResearchNeed`` rows (a static catalog,
not a running component).

Two families are kept deliberately separate, per the Phase 0B design review:

* ``LegacyResearchNeed`` / ``LegacyResearchNeedResult`` -- the 49 existing
  adversarial-search templates (``BEAR_TEMPLATES``, ``BULL_TEMPLATES``,
  ``KILL_QUERY_TEMPLATES``), reframed as *acquisition* asks: "did we look, and
  what did we get back". A ``LegacyResearchNeed`` carries no investment-
  judgment status -- it is not EVIDENCED, not CONTRADICTED, not confirmed or
  refuted. Whether the evidence it turned up says anything damaging is a
  question for the Kill/Bear/Bull agents that already exist; conflating "we
  found documents" with "we found a problem" was the exact confusion Phase 0B
  was called to resolve.

* ``CheckDefinition`` / ``CheckResult`` -- the shape already implicit in
  ``scoring/kill_gate.py``'s 19 ``KillRule`` entries: a named, evaluable
  proposition over already-collected facts, generalized so a future rule
  engine is not KillRule-specific. ``CheckResult.supporting_fact_ids`` is the
  *only* evidence-linkage field on either side of this split -- there is
  deliberately no ``Fact.supporting_check_ids``, so a fact never carries a
  back-reference to whatever happened to evaluate it; the check owns the
  citation, not the fact (mirrors the existing ``KillFinding.fact_ids`` shape
  in ``schemas/evaluation.py``).

``DomainCompletenessResult`` is a third, smaller shape: a per-
``ResearchDomain`` aggregate distinct from both of the above -- it answers
"was this whole required domain covered", not "did one specific need resolve"
or "did one specific check fire".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..schemas.enums import KillCategory, KillLevel, ResearchDomain, SearchStatus
from .document_store import DocumentRole


class AcquisitionOrigin(str, Enum):
    """Which legacy template family a ``LegacyResearchNeed`` came from."""

    BEAR = "BEAR"
    BULL = "BULL"
    KILL = "KILL"


class AcquisitionStatus(str, Enum):
    """Whether a ``LegacyResearchNeed`` was looked into -- never whether what
    it found was good or bad news. That axis does not exist on this type.
    """

    PENDING = "PENDING"
    ACQUIRED = "ACQUIRED"
    ACQUIRED_ZERO_RESULTS = "ACQUIRED_ZERO_RESULTS"
    SKIPPED_DUE_TO_BUDGET = "SKIPPED_DUE_TO_BUDGET"
    SKIPPED_DUE_TO_DIRECT_COVERAGE = "SKIPPED_DUE_TO_DIRECT_COVERAGE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class LegacyResearchNeed:
    """One of the 49 individually-addressable legacy adversarial templates.

    No status field: acquisition outcome lives on ``LegacyResearchNeedResult``,
    keyed by ``legacy_need_id``, so this type itself can never be asked "did
    this find something" -- an important separation, since these templates are
    adversarial/interpretive questions ("is there dilution", "did the trial
    fail") that acquisition alone can never answer.
    """

    legacy_need_id: str
    origin: AcquisitionOrigin
    #: The raw, unformatted template text, e.g. ``"{c} going concern"``.
    original_template: str
    domain: ResearchDomain
    #: Exact-match dedup key: ``original_template`` with every ``{t}``/``{c}``/
    #: ``{d}`` placeholder replaced by the literal token ``ENTITY``, lowercased
    #: and whitespace-normalized. Two needs merge in Phase 1 if and only if
    #: this key AND ``domain`` both match exactly -- no fuzzy or token-overlap
    #: comparison (see ``legacy_catalog.equivalence_key``).
    equivalence_key: str
    #: The document authority that would actually settle this need, when
    #: known (e.g. REGULATOR for an FDA-concern query). UNKNOWN where the
    #: legacy template is agnostic to who says it.
    required_authority: str = "UNKNOWN"
    expected_document_roles: tuple[DocumentRole, ...] = ()


@dataclass
class LegacyResearchNeedResult:
    """Acquisition outcome for one ``LegacyResearchNeed``.

    Deliberately thin: this says what was retrieved, not what it means.
    """

    legacy_need_id: str
    acquisition_status: AcquisitionStatus
    #: The unified (post-dedup) research need this legacy need resolved into,
    #: when Phase 2 wiring exists. ``None`` in Phase 1, since nothing here is
    #: connected to acquisition yet.
    research_need_id: str | None = None
    document_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class CheckDefinition:
    """A named, evaluable proposition over already-collected facts.

    Generalizes the shape already used by ``scoring.kill_gate.KillRule`` (19
    entries) so a future rule engine need not be Kill-specific. Not wired to
    ``KILL_RULES`` in Phase 1 -- this is the target shape, not a replacement.
    """

    check_id: str
    category: KillCategory
    title: str
    level_primary: KillLevel
    level_secondary: KillLevel
    explanation: str


@dataclass
class CheckResult:
    """Outcome of evaluating one ``CheckDefinition`` against collected facts.

    ``supporting_fact_ids`` is the sole evidence-linkage field in this whole
    type family -- there is no ``Fact.supporting_check_ids`` (see module
    docstring). A fact is never told what evaluated it; a check always says
    what it found.
    """

    check_id: str
    triggered: bool
    level: KillLevel
    supporting_fact_ids: tuple[str, ...] = ()
    detail: str = ""


@dataclass
class DomainCompletenessResult:
    """Per-``ResearchDomain`` aggregate, distinct from both need- and
    check-level results above: whether this whole required domain was
    covered, not whether one specific need resolved or one specific check
    fired.
    """

    domain: ResearchDomain
    status: SearchStatus
    #: legacy_need_ids (post-dedup research_need_ids in Phase 2) whose
    #: acquisition covered this domain.
    covered_by: tuple[str, ...] = field(default_factory=tuple)
    #: Whether a material/critical unresolved question remains open for this
    #: domain regardless of search status (mirrors the existing blocking-
    #: override logic in ``scoring/evidence_sufficiency.py``, not reused here
    #: since this type is not wired to the pipeline in Phase 1).
    has_blocking_unresolved_question: bool = False
