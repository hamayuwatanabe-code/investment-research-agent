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
    """How far a step's adapter/collector has actually progressed toward
    running for real. Phase 2.7's binary IMPLEMENTED/NOT_IMPLEMENTED
    collapsed several genuinely different claims into one -- "a generic
    HttpClient could fetch this" is not the same claim as "a dedicated
    adapter exists", which is not the same claim as "that adapter is
    registered with an executor", which is not the same claim as "the
    production pipeline calls it", which is not the same claim as "this was
    actually verified against mock/fake transport", which is not the same
    claim as "this was verified against the real site/API". Never promote a
    step past the level actually reached this turn (Phase 3A requirement 1).

    The levels are listed in the order Phase 3A's authorization gave them;
    OFFLINE_VERIFIED and LIVE_VERIFIED are verification checkpoints that, in
    this repository, are reached by direct executor tests using injected
    fake transport -- they do not require PIPELINE_WIRED first (pipeline
    connection is explicitly forbidden in Phase 3A), so a step can be
    EXECUTOR_WIRED and OFFLINE_VERIFIED while still not PIPELINE_WIRED.
    """

    #: The route exists only as catalog data -- no code backs it at all.
    DECLARED = "DECLARED"
    #: A generic, source-agnostic primitive (e.g. ``HttpClient``) could serve
    #: this step, but no source-specific adapter exists.
    PRIMITIVE_AVAILABLE = "PRIMITIVE_AVAILABLE"
    #: A dedicated adapter for this exact source exists as code.
    ADAPTER_IMPLEMENTED = "ADAPTER_IMPLEMENTED"
    #: The adapter is registered with an ``AcquisitionExecutor`` and can
    #: actually be invoked to run this step, in isolation.
    EXECUTOR_WIRED = "EXECUTOR_WIRED"
    #: The production ``Pipeline.run()`` calls this step. Never true before
    #: Phase 3A explicitly authorizes pipeline connection.
    PIPELINE_WIRED = "PIPELINE_WIRED"
    #: Verified against injected fake/mock transport in this repository's
    #: test suite -- no real network call was made.
    OFFLINE_VERIFIED = "OFFLINE_VERIFIED"
    #: Verified against the real site/API. Never true in this environment
    #: (no live network, no Live API credential reaches this code).
    LIVE_VERIFIED = "LIVE_VERIFIED"
    #: Deliberately switched off (distinct from never having been built).
    DISABLED = "DISABLED"

    @property
    def rank(self) -> int:
        return {
            "DECLARED": 0,
            "PRIMITIVE_AVAILABLE": 1,
            "ADAPTER_IMPLEMENTED": 2,
            "EXECUTOR_WIRED": 3,
            "PIPELINE_WIRED": 4,
            "OFFLINE_VERIFIED": 5,
            "LIVE_VERIFIED": 6,
            "DISABLED": -1,
        }[self.value]

    @property
    def is_executor_ready(self) -> bool:
        """Whether an ``AcquisitionExecutor`` could actually invoke this step
        today, in isolation (never a claim about pipeline connection)."""
        return self in (
            ImplementationStatus.EXECUTOR_WIRED,
            ImplementationStatus.OFFLINE_VERIFIED,
            ImplementationStatus.LIVE_VERIFIED,
        )

    @property
    def has_runnable_code(self) -> bool:
        """Whether SOME code -- a generic primitive, a dedicated adapter, or
        better -- exists that could attempt this step today, called directly
        rather than through an ``AcquisitionExecutor``. Weaker than
        ``is_executor_ready``: a primitive or adapter can exist without any
        executor wiring at all. False only for DECLARED (nothing behind it)
        and DISABLED (deliberately switched off)."""
        return self.rank >= ImplementationStatus.PRIMITIVE_AVAILABLE.rank


class PlanStatus(str, Enum):
    """Whether a plan is structurally achievable -- and, now, whether that
    achievability has actually been exercised by an executor -- never a
    claim about whether research has been executed or completed for a real
    run (see ExecutionStatus/ResearchStatus for that).

    Phase 3A correction: Phase 2.7's FEASIBLE conflated "an adapter exists
    somewhere in the catalog" with "something could actually run it". These
    five values separate structural feasibility from executor readiness and
    from budget:
    """

    #: No executor-ready path exists for one or more REQUIRED steps, with no
    #: viable alternative -- nothing could run this today even in isolation.
    NOT_EXECUTABLE = "NOT_EXECUTABLE"
    #: An executor-ready path exists for at least one target, but not for
    #: every target -- some remain out of reach given what is genuinely
    #: executor-wired today (an adapter that was never built, or a target
    #: whose steps are all executor-wired but the run-level web-search cap
    #: leaves some of them unserved once budget is considered downstream).
    EXECUTABLE_BOUNDED_INCOMPLETE = "EXECUTABLE_BOUNDED_INCOMPLETE"
    #: Feasible only under the aspirational ASSUME_IMPLEMENTED_ROUTES_SUCCEED
    #: projection -- a statement about a possible future design, not about
    #: what could run today.
    PROJECTED_FEASIBLE_WITH_GAPS = "PROJECTED_FEASIBLE_WITH_GAPS"
    #: Every required adapter is executor-wired, all required targets can
    #: complete within the projected budget, and no unresolved structural
    #: REQUIRED-criticality requirement remains blocking.
    EXECUTABLE_COMPLETE = "EXECUTABLE_COMPLETE"
    #: Executor-ready, but the required search volume itself cannot fit the
    #: discovery budget even before considering the hard cap.
    BUDGET_INFEASIBLE = "BUDGET_INFEASIBLE"


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


class RequirementCriticality(str, Enum):
    """How much an unresolved requirement should matter -- replaces Phase
    2.7's blanket ``blocking_if_unresolved: bool``, which forced every FDA
    regulator-confirmation requirement to be unconditionally blocking merely
    because it happened to be non-public (Phase 3A requirement 3).

    Whether a CONDITIONAL_BLOCKING requirement actually blocks depends on
    facts a static catalog cannot know without a real run -- whether the
    claim concerns the company's CURRENT/lead/registrational program,
    whether it is material to the investment decision, whether issuer
    disclosure alone already resolves the question, and whether an
    alternative primary source exists. Phase 3A does not implement that
    materiality assessment; a CONDITIONAL_BLOCKING requirement is reported
    as ``requires_materiality_assessment`` (PENDING_MATERIALITY_ASSESSMENT)
    rather than resolved either way, never defaulted to blocking or to
    non-blocking.
    """

    #: Always counts as blocking if unresolved -- no runtime condition gates
    #: it (e.g. the issuer's own required statutory disclosure).
    REQUIRED = "REQUIRED"
    #: Blocking ONLY if a materiality assessment (not implemented in Phase
    #: 3A) determines the specific conditions apply. Never silently resolved
    #: to blocking OR to non-blocking by this catalog.
    CONDITIONAL_BLOCKING = "CONDITIONAL_BLOCKING"
    #: Never blocks completion; valuable if obtained, not required.
    BEST_EFFORT = "BEST_EFFORT"


def priority_tier_for(*, domain: ResearchDomain, criticality: RequirementCriticality) -> RequirementPriorityTier:
    """Deterministic, code-visible mapping from (domain, criticality) to
    requirement 8's fixed priority order. Never randomized, never tuned per
    run -- the same inputs always rank the same way. A CONDITIONAL_BLOCKING
    requirement is never treated as REQUIRED-strength for prioritization --
    it competes on domain alone, since whether it actually blocks is still
    pending a materiality assessment this module does not perform.
    """
    if criticality is RequirementCriticality.REQUIRED and domain is ResearchDomain.REGULATORY:
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
    criticality: RequirementCriticality = RequirementCriticality.REQUIRED

    @property
    def priority_tier(self) -> RequirementPriorityTier:
        return priority_tier_for(domain=self.domain, criticality=self.criticality)

    @property
    def requires_materiality_assessment(self) -> bool:
        """PENDING_MATERIALITY_ASSESSMENT: true for a CONDITIONAL_BLOCKING
        requirement, whose actual blocking-ness this catalog does not (and,
        without a real run's facts, cannot) resolve either way."""
        return self.criticality is RequirementCriticality.CONDITIONAL_BLOCKING


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
    #: runnable -- see ``ImplementationStatus``. Defaults to the honest
    #: baseline (nothing stated -> nothing earned): a step that does not
    #: explicitly claim otherwise is DECLARED, never assumed IMPLEMENTED.
    implementation_status: ImplementationStatus = ImplementationStatus.DECLARED

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
        outcome == StepStatus.FAILED and not by_id[sid].implementation_status.is_executor_ready
        for sid, outcome in recorded.items()
    ):
        return TargetAcquisitionOutcome.NOT_IMPLEMENTED
    if exhausted:
        return TargetAcquisitionOutcome.FAILED
    return TargetAcquisitionOutcome.IN_PROGRESS


def target_plan_status(target: AcquisitionTarget, steps: Sequence[AcquisitionStep]) -> PlanStatus:
    """Whether ``target`` has ANY EXECUTOR-READY path to genuine content
    acquisition today, independent of whether that path would actually
    succeed on a given run. Never mutates anything, never executes anything.

    Per-target, this collapses to two values -- NOT_EXECUTABLE or
    EXECUTABLE_COMPLETE -- for two structurally different reasons a target
    can fail to be executor-ready:
    * every required/alternative step falls short of EXECUTOR_WIRED (Phase
      3A requirement 1's ladder), with no viable fallback -- an
      implementation/wiring gap; or
    * the target's own required steps are inherently incapable of content
      acquisition (e.g. its only step is NOT_PUBLICLY_AVAILABLE) -- not a
      wiring gap at all, but a target that can never be "acquired".
    (BUDGET_INFEASIBLE and the two PROJECTED_*/EXECUTABLE_BOUNDED_INCOMPLETE
    graph-level distinctions are computed in ``compute_plan_status`` and in
    ``routing_budget_scenarios.py``, which know about budget and about
    multiple targets; a single target has no "some targets, not others" case
    of its own.) Requirement 1: a step declared but not EXECUTOR_WIRED is
    never, by itself, read as making a target EXECUTABLE_COMPLETE.
    """
    by_id = {s.step_id: s for s in steps if s.target_id == target.target_id}

    def usable(step: AcquisitionStep) -> bool:
        return step.implementation_status.is_executor_ready and not step.never_constitutes_content_acquisition

    if not target.required_step_ids and not target.alternative_step_groups:
        return PlanStatus.NOT_EXECUTABLE

    for step_id in target.required_step_ids:
        step = by_id.get(step_id)
        if step is None or not usable(step):
            return PlanStatus.NOT_EXECUTABLE
    for group in target.alternative_step_groups:
        group_steps = [by_id[sid] for sid in group if sid in by_id]
        if not any(usable(s) for s in group_steps):
            return PlanStatus.NOT_EXECUTABLE
    return PlanStatus.EXECUTABLE_COMPLETE


def compute_plan_status(graph: SourceRoutingGraph) -> tuple[PlanStatus, int, int]:
    """Aggregate PlanStatus over the whole graph from EXECUTOR-READINESS
    alone (never budget, never the aspirational assumption -- those live in
    ``routing_budget_scenarios.py``, which knows about both), plus two counts:

    * ``blocking_unresolved_requirements`` -- REQUIRED-criticality
      requirements with no executor-ready path to content acquisition at all
      (Phase 2.7 requirement 1/5, Phase 3A requirement 3: only REQUIRED
      criticality counts here unconditionally).
    * ``pending_materiality_requirements`` -- CONDITIONAL_BLOCKING
      requirements whose only targets are not executor-ready. Phase 3A
      requirement 3 forbids defaulting these to blocking OR to non-blocking
      without a real run's materiality facts, so they are counted
      separately, never folded into ``blocking_unresolved_requirements``.

    Read-only; never executes or mutates ``graph``.
    """
    executable_target_ids = {
        t.target_id
        for t in graph.targets
        if target_plan_status(t, graph.steps_for_target(t.target_id)) is PlanStatus.EXECUTABLE_COMPLETE
    }
    all_target_ids = {t.target_id for t in graph.targets}
    not_executable_target_ids = all_target_ids - executable_target_ids

    blocking_unresolved = 0
    pending_materiality = 0
    for requirement in graph.requirements:
        targets = graph.targets_for_requirement(requirement.requirement_id)
        if not targets or not all(t.target_id in not_executable_target_ids for t in targets):
            continue
        if requirement.criticality is RequirementCriticality.REQUIRED:
            blocking_unresolved += 1
        elif requirement.criticality is RequirementCriticality.CONDITIONAL_BLOCKING:
            pending_materiality += 1

    if not graph.targets or not_executable_target_ids == all_target_ids:
        return PlanStatus.NOT_EXECUTABLE, blocking_unresolved, pending_materiality
    if not_executable_target_ids:
        return PlanStatus.EXECUTABLE_BOUNDED_INCOMPLETE, blocking_unresolved, pending_materiality
    return PlanStatus.EXECUTABLE_COMPLETE, blocking_unresolved, pending_materiality
