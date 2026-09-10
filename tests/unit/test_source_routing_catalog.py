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
    StepStatus,
    TargetKind,
    is_target_complete,
    resolve_next_steps,
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
