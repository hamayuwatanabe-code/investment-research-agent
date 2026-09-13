"""acquisition_executor.py: AcquisitionExecutor driven entirely by fake
adapters -- Phase 3A requirement 4. No network, no real adapter; this file
only proves the executor's own DAG-driving, dedup, cap-enforcement and
diagnostics logic. See test_sec_acquisition_adapters.py for the real SEC
adapters against a FakeHttpClient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from investment_research.research.acquisition_executor import (
    AcquisitionExecutor,
    ExecutionContext,
    StepExecutionResult,
)
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentStore
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    ImplementationStatus,
    RequirementCriticality,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetAcquisitionOutcome,
    TargetKind,
)
from investment_research.schemas.enums import ResearchDomain


@dataclass
class FakeAdapter:
    """A scripted adapter: returns one canned result per step_id, in order,
    and counts how many times it was actually called (never from cache)."""

    results: dict[str, StepExecutionResult]
    calls: list[str] = field(default_factory=list)

    def execute(self, step: AcquisitionStep, context: ExecutionContext) -> StepExecutionResult:
        self.calls.append(step.step_id)
        return self.results[step.step_id]


def _document_target(tag: str, *, locate_method=AcquisitionMethod.EXISTING_DIRECT_API):
    tid, rid = f"target_{tag}", f"req_{tag}"
    l1 = AcquisitionStep(
        step_id=f"l1_{tag}", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=locate_method, adapter_id="fake_locate",
        completion_condition=StepStatus.URL_RESOLVED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    f = AcquisitionStep(
        step_id=f"f_{tag}", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="fake_fetch",
        depends_on_step_ids=(f"l1_{tag}",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    p = AcquisitionStep(
        step_id=f"p_{tag}", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="fake_parse",
        depends_on_step_ids=(f"f_{tag}",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{tag}",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CAPITAL_STRUCTURE,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=(f"l1_{tag}", f"f_{tag}", f"p_{tag}"), serves_requirement_ids=(rid,),
    )
    return requirement, target, [l1, f, p]


def test_full_chain_executes_in_dependency_order_and_completes_the_target():
    requirement, target, steps = _document_target("a")
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))

    locate = FakeAdapter({"l1_a": StepExecutionResult(step_id="l1_a", status=StepStatus.URL_RESOLVED, payload={"url": "https://example.test/doc"})})
    fetch = FakeAdapter({"f_a": StepExecutionResult(step_id="f_a", status=StepStatus.BODY_FETCHED, document_id="doc_1", http_requests_made=1)})
    parse = FakeAdapter({"p_a": StepExecutionResult(step_id="p_a", status=StepStatus.PARSED, document_id="doc_1")})

    executor = AcquisitionExecutor(
        adapters={"fake_locate": locate, "fake_fetch": fetch, "fake_parse": parse},
        document_store=DocumentStore(),
    )
    report = executor.run(graph)

    assert locate.calls == ["l1_a"]
    assert fetch.calls == ["f_a"]
    assert parse.calls == ["p_a"]
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert report.diagnostics.body_fetch_successes == 1
    assert report.diagnostics.parsed_documents == 1
    assert report.diagnostics.incomplete_targets == 0


def test_metadata_only_never_completes_a_target_via_the_executor():
    requirement, target, steps = _document_target("b")
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))

    locate = FakeAdapter({"l1_b": StepExecutionResult(step_id="l1_b", status=StepStatus.URL_RESOLVED)})
    fetch = FakeAdapter({"f_b": StepExecutionResult(step_id="f_b", status=StepStatus.FAILED, failure_reason="connection reset")})

    executor = AcquisitionExecutor(adapters={"fake_locate": locate, "fake_fetch": fetch}, document_store=DocumentStore())
    report = executor.run(graph)

    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert report.diagnostics.metadata_only_results == 1
    assert report.diagnostics.body_fetch_failures == 1
    assert report.diagnostics.incomplete_targets == 1
    # PARSE was never reached because FETCH failed -- never attempted, so it
    # must not appear as an executed step.
    assert "p_b" not in report.outcomes


def test_not_publicly_available_step_never_reaches_an_adapter_or_network_call():
    step = AcquisitionStep(
        step_id="npa", target_id="t_npa", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE, adapter_id="none",
        completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
    )
    requirement = EvidenceRequirement(
        requirement_id="req_npa", serves_legacy_need_ids=("synthetic_npa",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY, criticality=RequirementCriticality.CONDITIONAL_BLOCKING,
    )
    target = AcquisitionTarget(
        target_id="t_npa", target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE,
        required_step_ids=("npa",), serves_requirement_ids=("req_npa",),
    )
    graph = SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(step,))

    # No adapter registered at all for adapter_id "none" -- if the executor
    # ever tried to call one, this would KeyError/AttributeError immediately.
    executor = AcquisitionExecutor(adapters={}, document_store=DocumentStore())
    report = executor.run(graph)

    assert report.outcomes["npa"] is StepStatus.NOT_PUBLICLY_AVAILABLE
    assert report.outcome_for("t_npa") is TargetAcquisitionOutcome.ACQUISITION_EXHAUSTED_NOT_PUBLIC
    assert report.diagnostics.not_public_requirements == 1
    # pending_materiality_requirements is reserved for a CONDITIONAL_BLOCKING
    # requirement that did NOT resolve to conclusively-not-public -- this one
    # already has a conclusive resolution, so it is never double-counted here.
    assert report.diagnostics.pending_materiality_requirements == 0


def test_web_search_step_with_no_registered_adapter_is_recorded_not_implemented_never_called():
    step = AcquisitionStep(
        step_id="ws", target_id="t_ws", step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY, adapter_id="anthropic_web_search",
        completion_condition=StepStatus.URL_RESOLVED,
    )
    target = AcquisitionTarget(target_id="t_ws", target_kind=TargetKind.GENERIC_WEB_DOCUMENT, required_step_ids=("ws",))
    graph = SourceRoutingGraph(targets=(target,), steps=(step,))

    executor = AcquisitionExecutor(adapters={}, document_store=DocumentStore())
    report = executor.run(graph)

    assert report.outcomes["ws"] is StepStatus.FAILED
    assert report.diagnostics.web_search_steps_not_executed == 1


def test_web_search_hard_cap_is_enforced_even_with_a_registered_fake_adapter():
    """Tests MAY simulate a web-search success (item 7's explicit allowance),
    but the run-level MAX_WEB_SEARCH_USES cap still applies -- a 4th web
    search step in the same run is never executed even if an adapter exists."""
    steps = []
    targets = []
    ws_adapter = FakeAdapter({})
    for i in range(4):
        tid = f"t_ws_{i}"
        step = AcquisitionStep(
            step_id=f"ws_{i}", target_id=tid, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY, adapter_id="fake_web_search",
            completion_condition=StepStatus.URL_RESOLVED,
        )
        ws_adapter.results[f"ws_{i}"] = StepExecutionResult(step_id=f"ws_{i}", status=StepStatus.URL_RESOLVED)
        steps.append(step)
        targets.append(AcquisitionTarget(target_id=tid, target_kind=TargetKind.GENERIC_WEB_DOCUMENT, required_step_ids=(f"ws_{i}",)))

    graph = SourceRoutingGraph(targets=tuple(targets), steps=tuple(steps))
    executor = AcquisitionExecutor(adapters={"fake_web_search": ws_adapter}, document_store=DocumentStore(), max_web_search_uses=3)
    report = executor.run(graph)

    executed = [sid for sid in ("ws_0", "ws_1", "ws_2", "ws_3") if report.outcomes[sid] is StepStatus.URL_RESOLVED]
    capped = [sid for sid in ("ws_0", "ws_1", "ws_2", "ws_3") if report.outcomes[sid] is StepStatus.SKIPPED_DUE_TO_BUDGET]
    assert len(executed) == 3
    assert len(capped) == 1
    assert len(ws_adapter.calls) == 3


def test_diagnostics_never_report_external_llm_tokens():
    report = AcquisitionExecutor(adapters={}, document_store=DocumentStore()).run(SourceRoutingGraph())
    assert report.diagnostics.external_llm_tokens == 0
