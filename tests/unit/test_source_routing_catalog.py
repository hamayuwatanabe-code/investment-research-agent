"""source_routing_catalog.py: the 31-group step-DAG catalog.

Confirms every real group is individually classified, that requirement/
target/step counts are measured separately from 31, and the specific
structural guarantees Phase 2.6 requires: metadata/URL success alone never
completes a target, ClinicalTrials and PubMed are separate targets, issuer
disclosure and regulator confirmation are separate requirements, primary
document and exhibit are separate targets.
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.legacy_catalog import LEGACY_CATALOG, group_legacy_needs
from investment_research.research.source_routing import (
    ImplementationStatus,
    PlanStatus,
    RequirementCriticality,
    StepKind,
    StepStatus,
    TargetKind,
    compute_plan_status,
    is_target_complete,
    resolve_next_steps,
    target_plan_status,
)
from investment_research.research.source_routing_catalog import (
    _ARCHETYPE_BY_KEY,
    build_source_routing_graph,
    routing_coverage_counts,
)


def test_every_one_of_the_31_real_groups_has_an_explicit_archetype():
    groups = group_legacy_needs(LEGACY_CATALOG)
    real_keys = {key for key, _domain, _scope in groups}
    assert real_keys == set(_ARCHETYPE_BY_KEY)
    assert len(_ARCHETYPE_BY_KEY) == 31


def test_counts_are_measured_separately_not_equal_to_31():
    counts = routing_coverage_counts()
    assert counts.legacy_needs == 49
    assert counts.equivalence_groups == 31
    assert counts.evidence_requirements != 31
    assert counts.acquisition_targets != 31
    assert counts.evidence_requirements > counts.equivalence_groups


def test_all_49_legacy_rows_are_connected_to_some_requirement():
    graph = build_source_routing_graph()
    assert graph.all_served_legacy_need_ids() == {n.legacy_need_id for n in LEGACY_CATALOG}


def _walk_full(target, steps):
    outcomes: dict = {}
    for _ in range(len(steps) + 1):
        nxt = resolve_next_steps(target, steps, outcomes)
        if not nxt:
            break
        for s in nxt:
            outcomes[s.step_id] = s.completion_condition
    return outcomes


# --- SEC metadata/URL success alone never completes a target ---------------
def test_sec_locate_success_alone_does_not_complete_the_target():
    graph = build_source_routing_graph()
    req = graph.requirements_for_need("bear_5")[0]  # "company dilution"
    target = graph.targets_for_requirement(req.requirement_id)[0]
    steps = graph.steps_for_target(target.target_id)
    locate_step = next(s for s in steps if s.acquisition_method is AcquisitionMethod.EXISTING_DIRECT_API)
    outcomes = {locate_step.step_id: StepStatus.URL_RESOLVED}
    assert not is_target_complete(target, steps, outcomes)


def test_sec_body_fetch_and_parse_completes_the_target():
    graph = build_source_routing_graph()
    req = graph.requirements_for_need("bear_5")[0]
    target = graph.targets_for_requirement(req.requirement_id)[0]
    steps = graph.steps_for_target(target.target_id)
    outcomes = _walk_full(target, steps)
    assert is_target_complete(target, steps, outcomes)
    assert any(o == StepStatus.BODY_FETCHED for o in outcomes.values())
    assert any(o == StepStatus.PARSED for o in outcomes.values())


# --- primary document vs. exhibit -------------------------------------------
def test_partnership_agreement_primary_and_exhibit_are_separate_targets():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bull_1")
    assert len(reqs) == 2
    kinds = {graph.targets_for_requirement(r.requirement_id)[0].target_kind for r in reqs}
    assert kinds == {TargetKind.SEC_PRIMARY_DOCUMENT, TargetKind.SEC_EXHIBIT}
    target_ids = {graph.targets_for_requirement(r.requirement_id)[0].target_id for r in reqs}
    assert len(target_ids) == 2


def test_exhibit_target_also_requires_fetch_and_parse_not_just_locate():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bull_1")
    exhibit_req = next(r for r in reqs if graph.targets_for_requirement(r.requirement_id)[0].target_kind is TargetKind.SEC_EXHIBIT)
    target = graph.targets_for_requirement(exhibit_req.requirement_id)[0]
    steps = graph.steps_for_target(target.target_id)
    locate_only = {s.step_id: StepStatus.URL_RESOLVED for s in steps if s.acquisition_method is AcquisitionMethod.NEW_DIRECT_ADAPTER}
    assert not is_target_complete(target, steps, locate_only)
    full = _walk_full(target, steps)
    assert is_target_complete(target, steps, full)


# --- issuer disclosure vs. regulator confirmation ---------------------------
def test_issuer_disclosure_and_regulator_confirmation_are_separate_requirements():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_0")  # "company fda concern"
    assert len(reqs) == 2
    disclosure = next(r for r in reqs if not r.independence_requirement)
    regulator = next(r for r in reqs if r.independence_requirement)
    disclosure_target = graph.targets_for_requirement(disclosure.requirement_id)[0]
    regulator_target = graph.targets_for_requirement(regulator.requirement_id)[0]
    assert disclosure_target.target_id != regulator_target.target_id


def test_completing_issuer_disclosure_does_not_complete_regulator_requirement():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_0")
    disclosure = next(r for r in reqs if not r.independence_requirement)
    regulator = next(r for r in reqs if r.independence_requirement)
    disclosure_target = graph.targets_for_requirement(disclosure.requirement_id)[0]
    regulator_target = graph.targets_for_requirement(regulator.requirement_id)[0]

    disclosure_steps = graph.steps_for_target(disclosure_target.target_id)
    disclosure_outcomes = _walk_full(disclosure_target, disclosure_steps)
    assert is_target_complete(disclosure_target, disclosure_steps, disclosure_outcomes)

    regulator_steps = graph.steps_for_target(regulator_target.target_id)
    assert not is_target_complete(regulator_target, regulator_steps, disclosure_outcomes)
    assert not is_target_complete(regulator_target, regulator_steps, {})


def test_regulator_requirement_never_reaches_web_search():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_0")
    regulator = next(r for r in reqs if r.independence_requirement)
    target = graph.targets_for_requirement(regulator.requirement_id)[0]
    steps = graph.steps_for_target(target.target_id)
    assert all(not s.sends_to_web_search for s in steps)
    assert all(s.acquisition_method is AcquisitionMethod.NOT_PUBLICLY_AVAILABLE for s in steps)


# --- ClinicalTrials vs. PubMed/literature ------------------------------------
def test_clinicaltrials_and_literature_are_separate_target_kinds():
    graph = build_source_routing_graph()
    ct_req = graph.requirements_for_need("bear_4")[0]  # "company failed trial"
    lit_req = graph.requirements_for_need("bull_0")[0]  # "company clinical data results"
    ct_target = graph.targets_for_requirement(ct_req.requirement_id)[0]
    lit_target = graph.targets_for_requirement(lit_req.requirement_id)[0]
    assert ct_target.target_kind is TargetKind.CLINICALTRIALS_RECORD
    assert lit_target.target_kind is TargetKind.LITERATURE_ARTICLE
    assert ct_target.target_id != lit_target.target_id


def test_clinicaltrials_success_does_not_complete_literature_requirement():
    graph = build_source_routing_graph()
    ct_req = graph.requirements_for_need("bear_4")[0]
    lit_req = graph.requirements_for_need("bull_0")[0]
    ct_target = graph.targets_for_requirement(ct_req.requirement_id)[0]
    lit_target = graph.targets_for_requirement(lit_req.requirement_id)[0]

    ct_steps = graph.steps_for_target(ct_target.target_id)
    ct_outcomes = _walk_full(ct_target, ct_steps)
    assert is_target_complete(ct_target, ct_steps, ct_outcomes)

    lit_steps = graph.steps_for_target(lit_target.target_id)
    assert not is_target_complete(lit_target, lit_steps, ct_outcomes)
    assert not is_target_complete(lit_target, lit_steps, {})


def test_literature_target_uses_new_direct_adapter_for_pubmed():
    graph = build_source_routing_graph()
    lit_req = graph.requirements_for_need("bull_0")[0]
    target = graph.targets_for_requirement(lit_req.requirement_id)[0]
    steps = graph.steps_for_target(target.target_id)
    pubmed_step = next(s for s in steps if s.adapter_id == "pubmed_europepmc")
    assert pubmed_step.acquisition_method is AcquisitionMethod.NEW_DIRECT_ADAPTER


# --- Web Search snippet is never Evidence -----------------------------------
def test_web_search_locate_success_alone_never_completes_a_web_only_target():
    graph = build_source_routing_graph()
    req = graph.requirements_for_need("bear_13")[0]  # "company criticism"
    target = graph.targets_for_requirement(req.requirement_id)[0]
    steps = graph.steps_for_target(target.target_id)
    locate_step = next(s for s in steps if s.acquisition_method is AcquisitionMethod.WEB_SEARCH_DISCOVERY)
    outcomes = {locate_step.step_id: StepStatus.URL_RESOLVED}
    assert not is_target_complete(target, steps, outcomes)


def test_web_only_target_requires_discovered_url_fetch_and_parse():
    graph = build_source_routing_graph()
    req = graph.requirements_for_need("bear_13")[0]
    target = graph.targets_for_requirement(req.requirement_id)[0]
    steps = graph.steps_for_target(target.target_id)
    assert len(steps) == 3  # LOCATE, FETCH, PARSE all required
    full = _walk_full(target, steps)
    assert is_target_complete(target, steps, full)
    assert any(o == StepStatus.BODY_FETCHED for o in full.values())


# --- no silent drop / no unclassified group ---------------------------------
def test_building_graph_raises_if_a_real_group_is_left_unclassified(monkeypatch):
    import investment_research.research.source_routing_catalog as catalog_module

    trimmed = dict(catalog_module._ARCHETYPE_BY_KEY)
    trimmed.pop("company criticism")
    monkeypatch.setattr(catalog_module, "_ARCHETYPE_BY_KEY", trimmed)
    raised = False
    try:
        catalog_module.build_source_routing_graph()
    except ValueError:
        raised = True
    assert raised


# =========================== Phase 3A: implementation status ===============
def test_pubmed_adapter_is_now_offline_verified():
    """PubMed/Europe PMC was DECLARED through Phase 3E.4 -- untouched by
    Phase 3A/3D/3E/3E.4 (requirement 1: never promote to an unearned
    level). Phase 3F implemented ``PubMedLiteratureAdapter``/
    ``EuropePmcFullTextAdapter`` and proved them against a fake HTTP
    double and the real-format fixtures in
    ``tests/fixtures/literature_real_format/``, earning OFFLINE_VERIFIED
    for its own direct LOCATE/FETCH/PARSE steps -- never LIVE_VERIFIED (no
    real network call was ever made this phase). Mirrors Form 4's own
    DECLARED -> OFFLINE_VERIFIED promotion in Phase 3E; see
    ``test_only_sec_clinicaltrials_and_literature_direct_adapters_are_offline_verified``."""
    graph = build_source_routing_graph()
    declared_adapters = {
        s.adapter_id
        for s in graph.steps
        if s.implementation_status is ImplementationStatus.DECLARED and s.adapter_id not in ("local_parser", "none")
    }
    assert "pubmed_europepmc" not in declared_adapters


def test_only_sec_clinicaltrials_and_literature_direct_adapters_are_offline_verified():
    """Phase 3A requirement 1/5/6, extended by Phase 3D and Phase 3F:
    AcquisitionExecutor plus the SEC primary/exhibit adapters,
    ClinicalTrialsStudyAdapter's structured (known-NCT-ID) path, and (Phase
    3F) PubMedLiteratureAdapter's direct LOCATE/FETCH/PARSE path are built
    and proven against a fake HTTP double, so (and only so) their steps
    earn OFFLINE_VERIFIED. ``"local_parser"`` also appears here now: the
    structured-or-web archetype's PARSE step for ANY direct source (SEC
    uses its own dedicated adapter id for PARSE too, but ClinicalTrials'
    structured chain reuses the generic "local_parser" id for its PARSE
    step) shares that id with every other still-unimplemented structured
    archetype's PARSE step -- this set only ever contains the ids of steps
    actually AT OFFLINE_VERIFIED, never steps at a lower (or, since Phase
    3E.4, higher) rung that happen to share the same generic id.
    ``form4_xml_parser`` is EXCLUDED from this set as of Phase 3E.4 -- its
    3 direct steps are now LIVE_VERIFIED, not OFFLINE_VERIFIED (see
    ``test_form4_direct_adapter_is_live_verified``). Web search stays
    below is_executor_ready -- EXECUTOR_WIRED/PIPELINE_WIRED must never
    appear anywhere in this catalog at all, and LIVE_VERIFIED must appear
    ONLY on Form4's 3 steps (never on literature's, per Phase 3F's explicit
    no-Live-communication scope)."""
    graph = build_source_routing_graph()
    offline_verified_adapters = {
        s.adapter_id for s in graph.steps if s.implementation_status is ImplementationStatus.OFFLINE_VERIFIED
    }
    assert offline_verified_adapters == {
        "sec_primary_document_adapter", "sec_exhibit_enumeration", "clinicaltrials_api", "local_parser",
        "pubmed_europepmc",
    }
    live_verified_steps = [s for s in graph.steps if s.implementation_status is ImplementationStatus.LIVE_VERIFIED]
    assert {s.adapter_id for s in live_verified_steps} == {"form4_xml_parser"}
    assert len(live_verified_steps) == 3
    executor_ready_ids = {s.adapter_id for s in graph.steps if s.implementation_status.is_executor_ready}
    assert executor_ready_ids == offline_verified_adapters | {"form4_xml_parser"}
    assert not any(s.implementation_status is ImplementationStatus.EXECUTOR_WIRED for s in graph.steps)
    assert not any(s.implementation_status is ImplementationStatus.PIPELINE_WIRED for s in graph.steps)


def test_form4_direct_adapter_is_live_verified():
    """Phase 3E.4: every step of the form4 archetype's direct chain
    (LOCATE -> FETCH -> PARSE) -- exactly 3 steps -- is LIVE_VERIFIED,
    never merely the LOCATE step, and never any OTHER route/step in this
    catalog (the DISABLED web-search LOCATE alternative stays DISABLED).
    Earned by a real Mac Live Smoke run against a real issuer; see
    research/form4_live_smoke.py and research/source_routing_catalog.py's
    _form4() for the full provenance."""
    graph = build_source_routing_graph()
    form4_targets = [t for t in graph.targets if t.target_kind is TargetKind.FORM4_FILING]
    assert form4_targets
    for target in form4_targets:
        direct_steps = [
            s for s in graph.steps_for_target(target.target_id)
            if s.adapter_id == "form4_xml_parser"
        ]
        assert {s.step_kind for s in direct_steps} == {StepKind.LOCATE, StepKind.FETCH, StepKind.PARSE}
        assert len(direct_steps) == 3
        assert all(s.implementation_status is ImplementationStatus.LIVE_VERIFIED for s in direct_steps)

        other_steps = [
            s for s in graph.steps_for_target(target.target_id)
            if s.adapter_id != "form4_xml_parser"
        ]
        assert other_steps  # the web-search LOCATE alternative genuinely exists
        assert all(s.implementation_status is not ImplementationStatus.LIVE_VERIFIED for s in other_steps)


def test_real_catalog_plan_status_is_executable_bounded_incomplete():
    """Phase 3A requirement 2: with the SEC primary/exhibit adapters now
    genuinely executor-ready (OFFLINE_VERIFIED) but every other source
    (ClinicalTrials web fallback, Form 4, PubMed, web-only) still below that
    bar, the honest aggregate is a MIX -- EXECUTABLE_BOUNDED_INCOMPLETE, never
    NOT_EXECUTABLE (that would ignore the real SEC promotion) and never
    EXECUTABLE_COMPLETE (that would overclaim the untouched sources). The 6
    fda_dual regulator-confirmation requirements remain pending_materiality,
    never folded into blocking (requirement 3)."""
    graph = build_source_routing_graph()
    status, blocking, pending = compute_plan_status(graph)
    assert status is PlanStatus.EXECUTABLE_BOUNDED_INCOMPLETE
    assert blocking > 0
    assert pending == 6  # exactly the 6 fda_dual regulator-confirmation requirements


def test_sec_primary_and_exhibit_targets_are_now_individually_executable():
    graph = build_source_routing_graph()
    sec_target_kinds = {TargetKind.SEC_PRIMARY_DOCUMENT, TargetKind.SEC_EXHIBIT}
    for target in graph.targets:
        if target.target_kind not in sec_target_kinds:
            continue
        assert target_plan_status(target, graph.steps_for_target(target.target_id)) is PlanStatus.EXECUTABLE_COMPLETE


def test_untouched_targets_remain_not_executable():
    """Web-only targets were never touched by Phase 3A/3D/3E/3F -- they
    must still read NOT_EXECUTABLE, never silently upgraded by association
    with the SEC/ClinicalTrials/Form4/literature promotions. Form4 and
    literature targets are EXCLUDED from this check -- see
    ``test_form4_targets_are_now_individually_executable`` and
    ``test_literature_targets_are_now_individually_executable``."""
    graph = build_source_routing_graph()
    non_promoted_kinds = {TargetKind.GENERIC_WEB_DOCUMENT}
    checked = 0
    for target in graph.targets:
        if target.target_kind not in non_promoted_kinds:
            continue
        checked += 1
        assert target_plan_status(target, graph.steps_for_target(target.target_id)) is PlanStatus.NOT_EXECUTABLE
    assert checked > 0


def test_literature_targets_are_now_individually_executable():
    """Phase 3F: with PubMedLiteratureAdapter's direct path genuinely
    executor-ready (OFFLINE_VERIFIED), every literature target's direct
    alternative group is usable, so the whole target reads
    EXECUTABLE_COMPLETE -- even though its web-search fallback path
    remains untouched (DISABLED), because only ONE member of each
    alternative group needs to be usable (mirrors
    ``test_form4_targets_are_now_individually_executable``)."""
    graph = build_source_routing_graph()
    literature_targets = [t for t in graph.targets if t.target_kind is TargetKind.LITERATURE_ARTICLE]
    assert literature_targets
    for target in literature_targets:
        assert target_plan_status(target, graph.steps_for_target(target.target_id)) is PlanStatus.EXECUTABLE_COMPLETE


def test_form4_targets_are_now_individually_executable():
    """Phase 3E: with Form4Adapter's direct path genuinely executor-ready
    (OFFLINE_VERIFIED), every Form4 target's direct alternative group is
    usable, so the whole target reads EXECUTABLE_COMPLETE -- even though
    its web-search fallback path remains untouched (DISABLED), because
    only ONE member of each alternative group needs to be usable (mirrors
    ``test_clinicaltrials_targets_are_now_individually_executable``)."""
    graph = build_source_routing_graph()
    checked = 0
    for target in graph.targets:
        if target.target_kind is not TargetKind.FORM4_FILING:
            continue
        checked += 1
        assert target_plan_status(target, graph.steps_for_target(target.target_id)) is PlanStatus.EXECUTABLE_COMPLETE
    assert checked > 0


def test_clinicaltrials_targets_are_now_individually_executable():
    """Phase 3D: with ClinicalTrialsStudyAdapter's structured path
    genuinely executor-ready (OFFLINE_VERIFIED), every ClinicalTrials
    target's structured-fetch alternative group is usable, so the whole
    target reads EXECUTABLE_COMPLETE -- even though its web-search
    fallback path remains untouched (DISABLED), because only ONE member of
    each alternative group needs to be usable."""
    graph = build_source_routing_graph()
    checked = 0
    for target in graph.targets:
        if target.target_kind is not TargetKind.CLINICALTRIALS_RECORD:
            continue
        checked += 1
        assert target_plan_status(target, graph.steps_for_target(target.target_id)) is PlanStatus.EXECUTABLE_COMPLETE
    assert checked > 0


def test_fda_regulator_confirmation_is_conditional_blocking_not_required():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_0")  # "company fda concern"
    regulator = next(r for r in reqs if r.independence_requirement)
    disclosure = next(r for r in reqs if not r.independence_requirement)
    assert regulator.criticality is RequirementCriticality.CONDITIONAL_BLOCKING
    assert regulator.requires_materiality_assessment
    assert disclosure.criticality is RequirementCriticality.REQUIRED
    assert not disclosure.requires_materiality_assessment


def test_routing_coverage_counts_reports_implementation_ladder_split():
    counts = routing_coverage_counts()
    assert counts.declared_steps > 0  # PubMed -- untouched this phase
    assert counts.primitive_available_steps > 0  # generic web-fallback fetch/parse
    # Phase 3D promoted ClinicalTrials' structured direct fetch/parse from
    # ADAPTER_IMPLEMENTED to OFFLINE_VERIFIED (a real adapter now exists AND
    # is proven against a fake transport) -- nothing else in this catalog
    # still sits at ADAPTER_IMPLEMENTED (every other archetype's direct
    # path is either OFFLINE_VERIFIED (SEC, ClinicalTrials) or LIVE_VERIFIED
    # (Form 4, Phase 3E.4) or explicitly DECLARED (PubMed/Europe PMC)), so
    # this rung is legitimately empty now.
    assert counts.adapter_implemented_steps == 0
    assert counts.executor_wired_steps == 0  # no step is wired-but-unverified
    assert counts.pipeline_wired_steps == 0  # Pipeline.run() connection forbidden this phase
    assert counts.offline_verified_steps > 0  # the SEC + ClinicalTrials direct chains, proven this phase
    # Phase 3E.4: exactly Form4's own LOCATE/FETCH/PARSE, earned by a real
    # Mac Live Smoke run -- see test_form4_direct_adapter_is_live_verified.
    assert counts.live_verified_steps == 3
    ladder_total = (
        counts.declared_steps + counts.primitive_available_steps + counts.adapter_implemented_steps
        + counts.executor_wired_steps + counts.pipeline_wired_steps + counts.offline_verified_steps
        + counts.live_verified_steps + counts.disabled_steps
    )
    assert ladder_total == counts.locate_steps + counts.fetch_steps + counts.parse_steps


def test_routing_coverage_counts_reports_criticality_split():
    counts = routing_coverage_counts()
    assert counts.conditional_blocking_requirements == 6  # the 6 fda_dual regulator sub-requirements
    assert counts.required_requirements + counts.conditional_blocking_requirements + counts.best_effort_requirements == counts.evidence_requirements
