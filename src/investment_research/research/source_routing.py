"""Source Routing Graph v2 (Phase 2.6): EvidenceRequirement / AcquisitionTarget
/ AcquisitionStep, expressed as an explicit AND/OR step DAG.

Phase 2.6 scope only. Pure types and pure functions -- no network call, no
pipeline wiring, no execution.

This corrects Phase 2.5's defect: a "Route" there was a flat priority chain
whose SUCCESS state was the same regardless of whether it had actually
located a URL, fetched a body, or parsed one -- so a NORMAL scenario assuming
"the Direct route succeeds" halted at metadata/URL discovery and never
modeled a body fetch at all (``Direct HTTP requests == 0`` even for SEC
targets). Here, obtaining a real document is always a chain of separately
gated steps:

    LOCATE (find/resolve a URL, or a metadata pointer)
    -> FETCH (retrieve the body)
    -> PARSE (turn it into something Evidence Integrity can read)

and a target is complete only once every REQUIRED step -- not merely the
first one -- has reached its own success state. ``AcquisitionTarget`` states
this as an explicit dependency graph:

* ``required_step_ids`` -- AND: every one of these must reach its own
  ``completion_condition`` for the target to be complete.
* ``alternative_step_groups`` -- OR within a group, AND across groups: at
  least one member of each group must succeed. A later step whose
  ``depends_on_step_ids`` names a member of one of these groups is satisfied
  once ANY member of that group succeeds (see ``resolve_next_steps``).

``resolve_next_steps``/``is_target_complete`` are pure: given step outcomes
(real, from an execution this module never performs, or hypothetical, as
``routing_budget_scenarios``'s NORMAL/DEGRADED/WORST use), they only decide
what a state machine would try next, and whether a target counts as done.
Neither ever claims a claim was confirmed -- see ``checks.py``'s
Acquisition/Evidence/Check boundary, which this module continues to respect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from ..schemas.enums import ContentKind, DocumentAuthority, ResearchDomain
from .acquisition_planning import AcquisitionMethod
from .checks import SubjectScope


class StepKind(str, Enum):
    LOCATE = "LOCATE"
    FETCH = "FETCH"
    PARSE = "PARSE"


class StepStatus(str, Enum):
    """Fine-grained per-step outcome. Deliberately NOT the same vocabulary as
    ``checks.AcquisitionStatus`` (that is the coarser per-LegacyResearchNeed/
    CoverageLedger axis) -- this one exists so a LOCATE step's success is
    structurally incapable of being read as a FETCH or PARSE success.
    """

    PLANNED = "PLANNED"
    LOCATED_METADATA = "LOCATED_METADATA"
    URL_RESOLVED = "URL_RESOLVED"
    BODY_FETCHED = "BODY_FETCHED"
    PARSED = "PARSED"
    #: Structured-API equivalents of BODY_FETCHED/PARSED (requirement 2):
    #: a registry/API record retrieved whole, vs. the specific fields this
    #: requirement needs actually being present in it.
    STRUCTURED_RECORD_RETRIEVED = "STRUCTURED_RECORD_RETRIEVED"
    REQUIRED_FIELDS_PARSED = "REQUIRED_FIELDS_PARSED"
    ZERO_RESULTS = "ZERO_RESULTS"
    NOT_FOUND = "NOT_FOUND"
    NOT_PUBLICLY_AVAILABLE = "NOT_PUBLICLY_AVAILABLE"
    FAILED = "FAILED"
    SKIPPED_DUE_TO_BUDGET = "SKIPPED_DUE_TO_BUDGET"
    MANUAL_VERIFICATION_REQUIRED = "MANUAL_VERIFICATION_REQUIRED"

    @property
    def is_document_content(self) -> bool:
        """Whether this status means "Evidence Integrity has something to
        read" -- never true for a locator-only outcome."""
        return self in (StepStatus.BODY_FETCHED, StepStatus.PARSED, StepStatus.REQUIRED_FIELDS_PARSED)


class FailurePolicy(str, Enum):
    #: This step is not optional on its path: if it does not reach its own
    #: completion_condition, the target cannot complete via this path.
    REQUIRED = "REQUIRED"
    #: This step is one candidate within an alternative_step_group: failing
    #: it makes the group try its next untried member, if any.
    ALTERNATIVE = "ALTERNATIVE"


class ImplementationStatus(str, Enum):
    """Whether the adapter/collector a step names actually exists in this
    repository. Phase 2.7 requirement 3: declaring a route in the catalog is
    not the same claim as it being executable -- a NEW_DIRECT_ADAPTER step
    for a collector nobody has written yet is NOT_IMPLEMENTED, and no
    simulation may treat it as a working path except under the explicitly
    aspirational ASSUME_IMPLEMENTED_ROUTES_SUCCEED assumption.
    """

    IMPLEMENTED = "IMPLEMENTED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    DISABLED = "DISABLED"


class PlanStatus(str, Enum):
    """Whether a plan is structurally achievable given what is ACTUALLY
    implemented today -- never a claim about whether research has been
    executed or completed (see ExecutionStatus/ResearchStatus for that).
    """

    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    #: Feasible overall, but one or more BLOCKING requirements have no
    #: implemented path to content acquisition at all (e.g. a regulator's
    #: own non-public position).
    FEASIBLE_WITH_BLOCKING_GAPS = "FEASIBLE_WITH_BLOCKING_GAPS"


class ExecutionStatus(str, Enum):
    """Whether acquisition actually RAN -- distinct from whether a plan for
    it is feasible (PlanStatus) and from whether the run's evidence clears
    the bar for an Action (ResearchStatus, schemas.enums). This module never
    executes anything, so every simulation result you will find here is
    NOT_RUN; ExecutionStatus exists as a vocabulary for callers that do
    execute (Phase 3+), not as something routing_budget_scenarios.py sets.
    """

    NOT_RUN = "NOT_RUN"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class TargetAcquisitionOutcome(str, Enum):
    """Per-target classification of a (real or simulated) set of step
    outcomes. Distinct from ``checks.AcquisitionStatus`` -- this is what
    ``project_step_results_to_coverage`` (``step_graph_coverage.py``)
    translates INTO that coarser, per-LegacyResearchNeed vocabulary.
    """

    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    IN_PROGRESS = "IN_PROGRESS"
    #: The target's REQUIRED steps and every alternative group each reached
    #: a genuine content-bearing success (never merely LOCATE/metadata, and
    #: never NOT_PUBLICLY_AVAILABLE/MANUAL_VERIFICATION_REQUIRED).
    ACQUIRED = "ACQUIRED"
    #: Acquisition was conclusively determined NOT to be possible through any
    #: public channel. This is a resolution, not a success -- it NEVER
    #: satisfies ``is_target_complete`` and is reported separately.
    ACQUISITION_EXHAUSTED_NOT_PUBLIC = "ACQUISITION_EXHAUSTED_NOT_PUBLIC"
    #: Every remaining path was skipped for budget reasons, not attempted.
    BLOCKED_BY_BUDGET = "BLOCKED_BY_BUDGET"
    #: The only remaining path required a step whose adapter does not exist
    #: in this repository yet.
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    FAILED = "FAILED"


class SimulationAssumption(str, Enum):
    """What a projection is allowed to assume about routes that do not exist
    in code yet. Never mixed silently -- every report states which one
    produced which numbers (Phase 2.7 requirement 4).
    """

    #: Aspirational: every declared route, implemented or not, succeeds when
    #: reached. A statement about the DESIGN, never about what would happen
    #: if run today.
    ASSUME_IMPLEMENTED_ROUTES_SUCCEED = "ASSUME_IMPLEMENTED_ROUTES_SUCCEED"
    #: Grounded: a NOT_IMPLEMENTED step can never succeed; only code that
    #: exists today is assumed capable of running (and, under this
    #: assumption alone, assumed to succeed when it runs).
    CURRENT_IMPLEMENTATION_ONLY = "CURRENT_IMPLEMENTATION_ONLY"
    #: CURRENT_IMPLEMENTATION_ONLY's implementation constraint, plus: every
    #: implemented discovery-risk step returns a soft failure (ZERO_RESULTS)
    #: when reached.
    DIRECT_FAILURES = "DIRECT_FAILURES"
    #: CURRENT_IMPLEMENTATION_ONLY's implementation constraint, plus: every
    #: implemented discovery-risk step returns a hard failure (FAILED) when
    #: reached.
    WORST_CASE = "WORST_CASE"

    @property
    def respects_implementation_status(self) -> bool:
        return self is not SimulationAssumption.ASSUME_IMPLEMENTED_ROUTES_SUCCEED


class TargetKind(str, Enum):
    SEC_FILING_METADATA = "SEC_FILING_METADATA"
    SEC_PRIMARY_DOCUMENT = "SEC_PRIMARY_DOCUMENT"
    SEC_EXHIBIT = "SEC_EXHIBIT"
    FORM4_FILING = "FORM4_FILING"
    CLINICALTRIALS_RECORD = "CLINICALTRIALS_RECORD"
    LITERATURE_ARTICLE = "LITERATURE_ARTICLE"
    FDA_PUBLIC_DATASET_RECORD = "FDA_PUBLIC_DATASET_RECORD"
    FDA_NONPUBLIC_CORRESPONDENCE = "FDA_NONPUBLIC_CORRESPONDENCE"
    ISSUER_IR_MATERIAL = "ISSUER_IR_MATERIAL"
    GENERIC_WEB_DOCUMENT = "GENERIC_WEB_DOCUMENT"


class TokenCostClass(str, Enum):
    ZERO = "ZERO"
    LOW = "LOW"
    HIGH = "HIGH"


class RequestCostClass(str, Enum):
    FREE = "FREE"
    RATE_LIMITED = "RATE_LIMITED"
    PAID_LLM_CALL = "PAID_LLM_CALL"


class RequirementPriorityTier(str, Enum):
    """Requirement 8's fixed priority order for bounded search selection when
    Direct routes cannot cover every requirement. Lower tier number = served
    first when only ``MAX_WEB_SEARCH_USES`` web searches are affordable.
    """

    BLOCKING_REGULATORY = "BLOCKING_REGULATORY"
    CONTRADICTION_FALSIFICATION = "CONTRADICTION_FALSIFICATION"
    CURRENT_PROGRAM_SCIENCE = "CURRENT_PROGRAM_SCIENCE"
    CAPITAL_SURVIVAL = "CAPITAL_SURVIVAL"
    REMAINING_GAPS = "REMAINING_GAPS"

    @property
    def rank(self) -> int:
        return {
            "BLOCKING_REGULATORY": 1,
            "CONTRADICTION_FALSIFICATION": 2,
            "CURRENT_PROGRAM_SCIENCE": 3,
            "CAPITAL_SURVIVAL": 4,
            "REMAINING_GAPS": 5,
        }[self.value]


def priority_tier_for(*, domain: ResearchDomain, blocking_if_unresolved: bool) -> RequirementPriorityTier:
    """Deterministic, code-visible mapping from (domain, blocking) to
    requirement 8's fixed priority order. Never randomized, never tuned per
    run -- the same inputs always rank the same way.
    """
    if blocking_if_unresolved and domain is ResearchDomain.REGULATORY:
        return RequirementPriorityTier.BLOCKING_REGULATORY
    if domain is ResearchDomain.CONTRADICTION:
        return RequirementPriorityTier.CONTRADICTION_FALSIFICATION
    if domain is ResearchDomain.SCIENCE_TECHNOLOGY:
        return RequirementPriorityTier.CURRENT_PROGRAM_SCIENCE
    if domain is ResearchDomain.CAPITAL_STRUCTURE:
        return RequirementPriorityTier.CAPITAL_SURVIVAL
    return RequirementPriorityTier.REMAINING_GAPS


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    serves_legacy_need_ids: tuple[str, ...]
    subject_scope: SubjectScope
    domain: ResearchDomain
    program_scope: str = "UNKNOWN"
    claim_scope: str = ""
    required_authorities: tuple[DocumentAuthority, ...] = ()
    independence_requirement: bool = False
    date_scope: str = "ANY"
    blocking_if_unresolved: bool = False

    @property
    def priority_tier(self) -> RequirementPriorityTier:
        return priority_tier_for(domain=self.domain, blocking_if_unresolved=self.blocking_if_unresolved)


@dataclass(frozen=True)
class AcquisitionStep:
    step_id: str
    target_id: str
    step_kind: StepKind
    acquisition_method: AcquisitionMethod
    depends_on_step_ids: tuple[str, ...] = ()
    failure_policy: FailurePolicy = FailurePolicy.REQUIRED
    authority: DocumentAuthority = DocumentAuthority.UNKNOWN
    expected_content_kind: ContentKind = ContentKind.FULL_DOCUMENT
    #: The StepStatus value that counts as THIS step's own success.
    completion_condition: StepStatus = StepStatus.BODY_FETCHED
    adapter_id: str = "UNKNOWN"
    token_cost_class: TokenCostClass = TokenCostClass.ZERO
    request_cost_class: RequestCostClass = RequestCostClass.FREE
    #: Whether ``adapter_id`` actually exists as working code in this
    #: repository today. Declaring a route is not the same claim as it being
    #: runnable -- see ``ImplementationStatus``.
    implementation_status: ImplementationStatus = ImplementationStatus.IMPLEMENTED

    @property
    def required(self) -> bool:
        return self.failure_policy is FailurePolicy.REQUIRED

    @property
    def sends_to_web_search(self) -> bool:
        return self.acquisition_method.requires_web_search_budget

    @property
    def never_constitutes_content_acquisition(self) -> bool:
        """Whether reaching THIS step's own completion_condition can ever
        mean a document was actually acquired. False for
        NOT_PUBLICLY_AVAILABLE/MANUAL_VERIFICATION_REQUIRED -- their
        "success" is a conclusive non-acquisition resolution, never content
        (Phase 2.7 requirement 5)."""
        return self.acquisition_method in (
            AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
            AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED,
        )


@dataclass(frozen=True)
class AcquisitionTarget:
    target_id: str
    target_kind: TargetKind
    issuer_identifier: str = "UNKNOWN"
    program_identifier: str = "UNKNOWN"
    #: AND set: every one of these step_ids must reach its own
    #: completion_condition for this target to be complete.
    required_step_ids: tuple[str, ...] = ()
    #: AND-of-ORs: each inner tuple is a group where at least one member
    #: must succeed; every group must be satisfied.
    alternative_step_groups: tuple[tuple[str, ...], ...] = ()
    serves_requirement_ids: tuple[str, ...] = ()

    def completion_condition_label(self, steps: Sequence[AcquisitionStep]) -> str:
        """Human-readable description derived from the actual step graph --
        never a separately-maintained enum that could drift out of sync with
        ``required_step_ids``/``alternative_step_groups``."""
        by_id = {s.step_id: s for s in steps if s.target_id == self.target_id}
        parts = [by_id[sid].completion_condition.value for sid in self.required_step_ids if sid in by_id]
        for group in self.alternative_step_groups:
            labels = [by_id[sid].completion_condition.value for sid in group if sid in by_id]
            if labels:
                parts.append("(" + " OR ".join(labels) + ")")
        return " AND ".join(parts) if parts else "NONE"


@dataclass
class SourceRoutingGraph:
    requirements: tuple[EvidenceRequirement, ...] = ()
    targets: tuple[AcquisitionTarget, ...] = ()
    steps: tuple[AcquisitionStep, ...] = ()

    def steps_for_target(self, target_id: str) -> list[AcquisitionStep]:
        return [s for s in self.steps if s.target_id == target_id]

    def targets_for_requirement(self, requirement_id: str) -> list[AcquisitionTarget]:
        return [t for t in self.targets if requirement_id in t.serves_requirement_ids]

    def requirements_for_need(self, legacy_need_id: str) -> list[EvidenceRequirement]:
        return [r for r in self.requirements if legacy_need_id in r.serves_legacy_need_ids]

    def all_served_legacy_need_ids(self) -> set[str]:
        served: set[str] = set()
        for requirement in self.requirements:
            served.update(requirement.serves_legacy_need_ids)
        return served


def _group_containing(target: AcquisitionTarget, step_id: str) -> tuple[str, ...] | None:
    for group in target.alternative_step_groups:
        if step_id in group:
            return group
    return None


def _dependency_satisfied(
    target: AcquisitionTarget,
    step: AcquisitionStep,
    by_id: Mapping[str, AcquisitionStep],
    outcomes: Mapping[str, StepStatus],
) -> bool:
    """Whether every dependency ``step`` cites currently holds.

    Two distinct citation shapes are supported, disambiguated by what the
    step actually lists:

    * Citing a single member of an alternative_step_group ("I depend
      specifically on THIS one") requires exactly that member's own
      completion_condition -- e.g. a structured-record parse step that
      depends only on the Direct fetch step, never on that fetch's web-
      search alternative.
    * Citing an ENTIRE alternative_step_group ("I depend on whichever of
      these succeeded") is satisfied by any one member succeeding -- e.g. a
      body-fetch step that can consume a URL resolved by either the Direct
      locator or its web-search fallback.

    The distinction is structural, not a flag: a group is treated as
    "wholly cited" only when every one of its members appears in
    ``step.depends_on_step_ids``.
    """
    for dep_id in step.depends_on_step_ids:
        dep = by_id.get(dep_id)
        if dep is None:
            return False
        group = _group_containing(target, dep_id)
        if group is not None and set(group) <= set(step.depends_on_step_ids):
            if not any(outcomes.get(sid) == by_id[sid].completion_condition for sid in group if sid in by_id):
                return False
        elif outcomes.get(dep_id) != dep.completion_condition:
            return False
    return True


def resolve_next_steps(
    target: AcquisitionTarget,
    steps: Sequence[AcquisitionStep],
    outcomes: Mapping[str, StepStatus],
) -> list[AcquisitionStep]:
    """Pure DAG step: which of ``target``'s steps are eligible to attempt
    next, given outcomes already recorded for some of them.

    A required step is offered once, when its dependencies are satisfied and
    it has not been attempted. An alternative_step_group offers only its
    next untried, dependency-satisfied member -- and offers nothing once any
    member has already succeeded (its own completion_condition was reached).
    """
    by_id = {s.step_id: s for s in steps if s.target_id == target.target_id}
    next_steps: list[AcquisitionStep] = []

    for step_id in target.required_step_ids:
        step = by_id.get(step_id)
        if step is None or step_id in outcomes:
            continue
        if _dependency_satisfied(target, step, by_id, outcomes):
            next_steps.append(step)

    for group in target.alternative_step_groups:
        group_steps = [by_id[sid] for sid in group if sid in by_id]
        if any(outcomes.get(s.step_id) == s.completion_condition for s in group_steps):
            continue
        for step in group_steps:
            if step.step_id in outcomes:
                continue
            if _dependency_satisfied(target, step, by_id, outcomes):
                next_steps.append(step)
                break

    return next_steps


def is_target_complete(
    target: AcquisitionTarget,
    steps: Sequence[AcquisitionStep],
    outcomes: Mapping[str, StepStatus],
) -> bool:
    """Whether every required step and every alternative group has reached
    its own completion_condition, via a step that can actually constitute
    content acquisition. A LOCATE-only or metadata-only outcome
    (``URL_RESOLVED``/``LOCATED_METADATA``) never satisfies a FETCH/PARSE
    step's own condition, so it can never complete a target on its own.
    A NOT_PUBLICLY_AVAILABLE or MANUAL_VERIFICATION_REQUIRED step reaching
    its own "completion_condition" NEVER counts here either (Phase 2.7
    requirement 5) -- that is a conclusive non-acquisition resolution, not a
    success; see ``target_acquisition_outcome`` for how it IS reported.
    A target with neither required steps nor alternative groups can never be
    complete -- there is nothing that could constitute acquiring it.
    """
    by_id = {s.step_id: s for s in steps if s.target_id == target.target_id}
    if not target.required_step_ids and not target.alternative_step_groups:
        return False
    for step_id in target.required_step_ids:
        step = by_id.get(step_id)
        if step is None or step.never_constitutes_content_acquisition:
            return False
        if outcomes.get(step_id) != step.completion_condition:
            return False
    for group in target.alternative_step_groups:
        group_steps = [by_id[sid] for sid in group if sid in by_id]
        if not any(
            outcomes.get(s.step_id) == s.completion_condition and not s.never_constitutes_content_acquisition
            for s in group_steps
        ):
            return False
    return True


def target_acquisition_outcome(
    target: AcquisitionTarget,
    steps: Sequence[AcquisitionStep],
    outcomes: Mapping[str, StepStatus],
) -> TargetAcquisitionOutcome:
    """Classify a target's (real or simulated) step outcomes.

    Ordering matters: ACQUIRED is checked first (via ``is_target_complete``,
    which already excludes non-content methods), then the terminal negative
    resolutions, then whether the target is simply still reachable
    (IN_PROGRESS) or has never been touched (NOT_ATTEMPTED).
    """
    if is_target_complete(target, steps, outcomes):
        return TargetAcquisitionOutcome.ACQUIRED

    by_id = {s.step_id: s for s in steps if s.target_id == target.target_id}
    recorded = {sid: outcome for sid, outcome in outcomes.items() if sid in by_id}
    if not recorded:
        return TargetAcquisitionOutcome.NOT_ATTEMPTED

    exhausted = resolve_next_steps(target, steps, outcomes) == []

    if any(outcome == StepStatus.NOT_PUBLICLY_AVAILABLE for outcome in recorded.values()):
        return TargetAcquisitionOutcome.ACQUISITION_EXHAUSTED_NOT_PUBLIC
    if exhausted and any(outcome == StepStatus.SKIPPED_DUE_TO_BUDGET for outcome in recorded.values()):
        return TargetAcquisitionOutcome.BLOCKED_BY_BUDGET
    if exhausted and any(
        outcome == StepStatus.FAILED and by_id[sid].implementation_status is ImplementationStatus.NOT_IMPLEMENTED
        for sid, outcome in recorded.items()
    ):
        return TargetAcquisitionOutcome.NOT_IMPLEMENTED
    if exhausted:
        return TargetAcquisitionOutcome.FAILED
    return TargetAcquisitionOutcome.IN_PROGRESS


def target_plan_status(target: AcquisitionTarget, steps: Sequence[AcquisitionStep]) -> PlanStatus:
    """Whether ``target`` has ANY implemented path to genuine content
    acquisition, independent of whether that path would actually succeed on
    a given run. Never mutates anything, never executes anything.

    INFEASIBLE covers two structurally different reasons, both real:
    * every required/alternative step is NOT_IMPLEMENTED, with no viable
      fallback -- an implementation gap; or
    * the target's own required steps are inherently incapable of content
      acquisition (e.g. its only step is NOT_PUBLICLY_AVAILABLE) -- not an
      implementation gap at all, but a target that can never be "acquired".
    Requirement 3: a NEW_DIRECT_ADAPTER route existing in the catalog is
    never, by itself, read as making a target FEASIBLE.
    """
    by_id = {s.step_id: s for s in steps if s.target_id == target.target_id}

    def usable(step: AcquisitionStep) -> bool:
        return step.implementation_status is ImplementationStatus.IMPLEMENTED and not step.never_constitutes_content_acquisition

    if not target.required_step_ids and not target.alternative_step_groups:
        return PlanStatus.INFEASIBLE

    for step_id in target.required_step_ids:
        step = by_id.get(step_id)
        if step is None or not usable(step):
            return PlanStatus.INFEASIBLE
    for group in target.alternative_step_groups:
        group_steps = [by_id[sid] for sid in group if sid in by_id]
        if not any(usable(s) for s in group_steps):
            return PlanStatus.INFEASIBLE
    return PlanStatus.FEASIBLE


def compute_plan_status(graph: SourceRoutingGraph) -> tuple[PlanStatus, int]:
    """Aggregate PlanStatus over the whole graph, plus the count of BLOCKING
    requirements with no implemented path to content acquisition at all
    (``blocking_unresolved_requirements`` -- Phase 2.7 requirement 1/5).
    Read-only; never executes or mutates ``graph``.
    """
    infeasible_target_ids = {
        t.target_id
        for t in graph.targets
        if target_plan_status(t, graph.steps_for_target(t.target_id)) is PlanStatus.INFEASIBLE
    }
    blocking_unresolved = 0
    for requirement in graph.requirements:
        if not requirement.blocking_if_unresolved:
            continue
        targets = graph.targets_for_requirement(requirement.requirement_id)
        if targets and all(t.target_id in infeasible_target_ids for t in targets):
            blocking_unresolved += 1

    if blocking_unresolved > 0:
        return PlanStatus.FEASIBLE_WITH_BLOCKING_GAPS, blocking_unresolved
    if graph.targets and infeasible_target_ids == {t.target_id for t in graph.targets}:
        return PlanStatus.INFEASIBLE, 0
    return PlanStatus.FEASIBLE, 0
