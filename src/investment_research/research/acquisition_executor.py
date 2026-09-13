"""AcquisitionExecutor: executes a Source Routing Graph's steps for real.

Phase 3A scope only. This module is NOT called from the production pipeline
(``Pipeline.run()``) -- see CLAUDE.md's Phase 3A forbidden-changes list.
Every test in this repository drives it with injected fake adapters/HTTP
transport; no real network call is ever made from here.

Adapters are supplied by the caller through dependency injection
(``AdapterProtocol.execute(step, context) -> StepExecutionResult``), keyed by
``AcquisitionStep.adapter_id``. This module knows nothing about SEC EDGAR,
HTTP, or any other concrete source -- see ``sec_acquisition_adapters.py`` for
the two adapters Phase 3A actually implements.

What this module is responsible for (Phase 3A requirement 4):

* Executing only steps whose dependencies are satisfied (``resolve_next_steps``
  from ``source_routing.py`` -- the same pure DAG walk the budget-scenario
  simulation uses, so "what would actually run" and "what the simulation
  assumed" share one definition of eligibility).
* Executing REQUIRED steps in the order the target declares them, and
  selecting one member of an ``alternative_step_group`` per the existing
  ALTERNATIVE-failure-policy semantics.
* Never repeating the same URL/API request within one run (dedup) --
  enforced via ``ExecutionContext.request_cache``, a run-scoped cache keyed
  by a string the *adapter* constructs (this module has no notion of what a
  "same request" means for a given source).
* Recording a ``StepExecutionResult`` for every attempted step, retaining
  failure reasons.
* Never sending a ``NOT_PUBLICLY_AVAILABLE``/``MANUAL_VERIFICATION_REQUIRED``
  step to any adapter or network call -- these resolve to their own
  ``completion_condition`` directly, in this module, with no adapter lookup.
* Never executing a real Web Search (Phase 3A requirement 7): a
  ``sends_to_web_search`` step with no registered adapter is recorded
  NOT_IMPLEMENTED; the run-level ``MAX_WEB_SEARCH_USES`` hard cap is
  enforced even when a test *does* register a fake web-search adapter.
* Never marking a Target complete before PARSE succeeds -- this module does
  not decide completion itself; it defers entirely to
  ``source_routing.target_acquisition_outcome``, which already encodes that
  rule (Phase 2.7 requirement 5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .document_store import DocumentStore
from .source_routing import (
    AcquisitionMethod,
    AcquisitionStep,
    AcquisitionTarget,
    RequirementCriticality,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetAcquisitionOutcome,
    resolve_next_steps,
    target_acquisition_outcome,
)

#: Requirement 8's hard, run-level cap, shared with routing_budget_scenarios.py.
#: Duplicated here (not imported) deliberately: this module must never depend
#: on the simulation module, and the two are asserted equal by a dedicated
#: test (Phase 3A requirement 12 -- invariant preservation).
MAX_WEB_SEARCH_USES = 3

_NEVER_NETWORKED_METHODS = (
    AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
    AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED,
)


@dataclass(frozen=True)
class StepExecutionResult:
    """What actually happened when one step was attempted (or resolved
    without an attempt, for a NOT_PUBLICLY_AVAILABLE/MANUAL_VERIFICATION_
    REQUIRED terminal step). Adapters construct these; the executor never
    invents one on an adapter's behalf beyond the no-adapter-registered and
    terminal-resolution cases, which are its own responsibility.
    """

    step_id: str
    status: StepStatus
    detail: str = ""
    #: The DocumentStore identity a body-fetching/parsing step produced, when
    #: it produced one. ``None`` for a LOCATE-only or non-content step.
    document_id: str | None = None
    #: Whether this result was served from ``ExecutionContext.request_cache``
    #: rather than by making a request -- the dedup/caching requirement
    #: (Phase 3A requirement 10). A cache hit always carries
    #: ``http_requests_made == api_requests_made == 0``.
    cache_hit: bool = False
    http_requests_made: int = 0
    api_requests_made: int = 0
    #: Data this step's dependents may need (e.g. a LOCATE step's resolved
    #: URL/CIK/accession, read by the FETCH step that depends on it via
    #: ``ExecutionContext.payload_for``).
    payload: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: str = ""


@runtime_checkable
class AdapterProtocol(Protocol):
    """What an adapter must implement to be registered with the executor.

    An adapter is looked up by ``AcquisitionStep.adapter_id`` and is handed
    exactly the step it must attempt plus the shared ``ExecutionContext`` --
    never the whole graph, never the DocumentStore's internals beyond what
    ``ExecutionContext.document_store`` exposes.
    """

    def execute(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult: ...


@dataclass
class ExecutionContext:
    """Everything an adapter call for one target can see, plus the run-scoped
    state every adapter call within the SAME run shares.

    Mutated in place by ``AcquisitionExecutor`` as steps complete; adapters
    read from it and write their own cache entries into ``request_cache``,
    but never execute another step themselves.
    """

    target: AcquisitionTarget
    document_store: DocumentStore
    settings: Any = None
    #: step_id -> the StepExecutionResult.payload of every step already
    #: executed for THIS target (reset per target by the executor).
    payloads: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Run-scoped (never reset between targets): an adapter-chosen cache key
    #: (e.g. the literal URL, or ``f"sec_submissions:{cik}"``) to the result
    #: of the first call that used it. Backs "the same primary document
    #: requested by multiple EvidenceRequirements -> exactly 1 HTTP GET"
    #: (Phase 3A requirement 10).
    request_cache: dict[str, StepExecutionResult] = field(default_factory=dict)
    #: Per-target inputs an adapter needs that the abstract step-DAG has no
    #: field for -- Phase 3A does not wire a real pipeline that would resolve
    #: a concrete filing reference automatically (see module docstring: this
    #: executor is not pipeline-connected). Keyed by target_id; a caller
    #: driving a real filing populates these before calling ``run()``.
    filing_references: Mapping[str, Any] = field(default_factory=dict)
    exhibit_selectors: Mapping[str, Any] = field(default_factory=dict)

    def payload_for(self, step_id: str) -> Mapping[str, Any]:
        return self.payloads.get(step_id, {})

    def filing_reference_for(self, target_id: str) -> Any:
        return self.filing_references.get(target_id)

    def exhibit_selector_for(self, target_id: str) -> Any:
        return self.exhibit_selectors.get(target_id)


@dataclass
class ExecutionDiagnostics:
    """Offline diagnostics (Phase 3A requirement 11). Every count here is
    something that actually happened this run -- never an estimate, never a
    projection. ``external_llm_tokens`` is always 0: this module calls no
    model, ever.
    """

    planned_steps: int = 0
    executed_steps: int = 0
    skipped_dependency_steps: int = 0
    cache_hits: int = 0
    direct_api_requests: int = 0
    direct_http_requests: int = 0
    body_fetch_successes: int = 0
    body_fetch_failures: int = 0
    metadata_only_results: int = 0
    parsed_documents: int = 0
    duplicate_acquisition_avoided: int = 0
    incomplete_targets: int = 0
    not_public_requirements: int = 0
    pending_materiality_requirements: int = 0
    web_search_steps_not_executed: int = 0
    external_llm_tokens: int = 0


@dataclass(frozen=True)
class TargetExecutionReport:
    target_id: str
    outcome: TargetAcquisitionOutcome
    step_results: tuple[StepExecutionResult, ...]


@dataclass(frozen=True)
class ExecutionReport:
    outcomes: Mapping[str, StepStatus]
    target_reports: tuple[TargetExecutionReport, ...]
    diagnostics: ExecutionDiagnostics

    def outcome_for(self, target_id: str) -> TargetAcquisitionOutcome | None:
        for report in self.target_reports:
            if report.target_id == target_id:
                return report.outcome
        return None


class AcquisitionExecutor:
    """Executes a ``SourceRoutingGraph`` against injected adapters.

    Never called by ``Pipeline.run()`` in this repository -- see the module
    docstring. Construct one per run; it holds no state that must survive
    across runs.
    """

    def __init__(
        self,
        *,
        adapters: Mapping[str, AdapterProtocol],
        document_store: DocumentStore,
        settings: Any = None,
        max_web_search_uses: int = MAX_WEB_SEARCH_USES,
    ) -> None:
        self._adapters = dict(adapters)
        self._document_store = document_store
        self._settings = settings
        self._max_web_search_uses = max_web_search_uses

    def run(
        self,
        graph: SourceRoutingGraph,
        *,
        filing_references: Mapping[str, Any] | None = None,
        exhibit_selectors: Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        outcomes: dict[str, StepStatus] = {}
        request_cache: dict[str, StepExecutionResult] = {}
        diagnostics = ExecutionDiagnostics(planned_steps=len(graph.steps))
        target_reports: list[TargetExecutionReport] = []
        web_search_uses_made = 0

        for target in graph.targets:
            steps = graph.steps_for_target(target.target_id)
            context = ExecutionContext(
                target=target,
                document_store=self._document_store,
                settings=self._settings,
                request_cache=request_cache,
                filing_references=filing_references or {},
                exhibit_selectors=exhibit_selectors or {},
            )
            step_results: list[StepExecutionResult] = []

            for _ in range(len(steps) + 1):
                next_batch = resolve_next_steps(target, steps, outcomes)
                if not next_batch:
                    break
                progressed = False
                for step in next_batch:
                    if step.step_id in outcomes:
                        continue
                    result, web_search_uses_made = self._execute_one(
                        step, context, diagnostics, web_search_uses_made
                    )
                    outcomes[step.step_id] = result.status
                    context.payloads[step.step_id] = result.payload
                    step_results.append(result)
                    self._tally(step, result, diagnostics)
                    progressed = True
                if not progressed:
                    break

            outcome = target_acquisition_outcome(target, steps, outcomes)
            target_reports.append(
                TargetExecutionReport(
                    target_id=target.target_id, outcome=outcome, step_results=tuple(step_results)
                )
            )
            if outcome is not TargetAcquisitionOutcome.ACQUIRED:
                diagnostics.incomplete_targets += 1
            attempted_ids = {r.step_id for r in step_results}
            diagnostics.skipped_dependency_steps += sum(
                1 for s in steps if s.step_id not in attempted_ids
            )

        self._tally_requirements(graph, target_reports, diagnostics)
        return ExecutionReport(
            outcomes=outcomes, target_reports=tuple(target_reports), diagnostics=diagnostics
        )

    # -- per-step execution --------------------------------------------
    def _execute_one(
        self,
        step: AcquisitionStep,
        context: ExecutionContext,
        diagnostics: ExecutionDiagnostics,
        web_search_uses_made: int,
    ) -> tuple[StepExecutionResult, int]:
        diagnostics.executed_steps += 1

        if step.acquisition_method in _NEVER_NETWORKED_METHODS:
            return (
                StepExecutionResult(
                    step_id=step.step_id,
                    status=step.completion_condition,
                    detail="terminal non-content resolution; never sent to any adapter or network call",
                ),
                web_search_uses_made,
            )

        if step.sends_to_web_search:
            adapter = self._adapters.get(step.adapter_id)
            if adapter is None:
                diagnostics.web_search_steps_not_executed += 1
                return (
                    StepExecutionResult(
                        step_id=step.step_id,
                        status=StepStatus.FAILED,
                        failure_reason=(
                            "web search is not executed in Phase 3A "
                            "(NOT_IMPLEMENTED/UNAVAILABLE, requirement 7)"
                        ),
                    ),
                    web_search_uses_made,
                )
            if web_search_uses_made >= self._max_web_search_uses:
                diagnostics.web_search_steps_not_executed += 1
                return (
                    StepExecutionResult(
                        step_id=step.step_id,
                        status=StepStatus.SKIPPED_DUE_TO_BUDGET,
                        failure_reason=f"the {self._max_web_search_uses}-web-search hard cap was already reached",
                    ),
                    web_search_uses_made,
                )
            result = adapter.execute(step, context)
            return result, web_search_uses_made + 1

        adapter = self._adapters.get(step.adapter_id)
        if adapter is None:
            return (
                StepExecutionResult(
                    step_id=step.step_id,
                    status=StepStatus.FAILED,
                    failure_reason=f"no adapter registered for '{step.adapter_id}' (NOT_IMPLEMENTED)",
                ),
                web_search_uses_made,
            )
        return adapter.execute(step, context), web_search_uses_made

    # -- diagnostics -----------------------------------------------------
    @staticmethod
    def _tally(step: AcquisitionStep, result: StepExecutionResult, diagnostics: ExecutionDiagnostics) -> None:
        if result.cache_hit:
            diagnostics.cache_hits += 1
            diagnostics.duplicate_acquisition_avoided += 1
        if step.acquisition_method is AcquisitionMethod.EXISTING_DIRECT_API:
            diagnostics.direct_api_requests += result.api_requests_made
        else:
            diagnostics.direct_http_requests += result.http_requests_made

        succeeded = result.status == step.completion_condition
        if step.step_kind is StepKind.FETCH:
            if succeeded:
                diagnostics.body_fetch_successes += 1
            elif result.status in (StepStatus.FAILED, StepStatus.NOT_FOUND, StepStatus.ZERO_RESULTS):
                diagnostics.body_fetch_failures += 1
        elif step.step_kind is StepKind.PARSE and succeeded:
            diagnostics.parsed_documents += 1
        elif step.step_kind is StepKind.LOCATE and result.status in (
            StepStatus.URL_RESOLVED,
            StepStatus.LOCATED_METADATA,
        ):
            diagnostics.metadata_only_results += 1

    @staticmethod
    def _tally_requirements(
        graph: SourceRoutingGraph,
        target_reports: Sequence[TargetExecutionReport],
        diagnostics: ExecutionDiagnostics,
    ) -> None:
        outcome_by_target = {r.target_id: r.outcome for r in target_reports}
        for requirement in graph.requirements:
            req_targets = graph.targets_for_requirement(requirement.requirement_id)
            if not req_targets:
                continue
            req_outcomes = [outcome_by_target.get(t.target_id) for t in req_targets]
            if all(o is TargetAcquisitionOutcome.ACQUISITION_EXHAUSTED_NOT_PUBLIC for o in req_outcomes):
                diagnostics.not_public_requirements += 1
            elif requirement.criticality is RequirementCriticality.CONDITIONAL_BLOCKING and all(
                o is not TargetAcquisitionOutcome.ACQUIRED for o in req_outcomes
            ):
                diagnostics.pending_materiality_requirements += 1
