"""source_routing_catalog.py: the 31-group routing catalog.

Confirms every real group is individually classified (never a silent
WEB_SEARCH_DISCOVERY default), that requirement/target/route counts are
measured separately from the 31 group count, and the specific structural
guarantees Phase 2.5 requires (authority separation, primary-vs-exhibit
targets).
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.legacy_catalog import LEGACY_CATALOG, group_legacy_needs
from investment_research.research.source_routing import DocumentAuthority, TargetKind
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


def test_no_group_defaults_silently_to_web_search():
    """The Phase 2 defect this turn fixes: NOT every archetype is
    web_search-only -- most real groups route Direct-first."""
    graph = build_source_routing_graph()
    counts = routing_coverage_counts(graph)
    assert counts.existing_direct_api_routes > 0
    assert counts.known_url_http_routes > 0
    assert counts.new_direct_adapter_routes > 0
    assert counts.not_publicly_available_routes > 0
    # Some web search remains -- this is not "eliminate all search", it is
    # "never default to search without justification".
    assert counts.web_search_discovery_routes > 0


def test_counts_are_measured_separately_not_equal_to_31():
    counts = routing_coverage_counts()
    assert counts.legacy_needs == 49
    assert counts.equivalence_groups == 31
    # None of these is forced to equal 31 -- they are independently derived.
    assert counts.evidence_requirements != 31
    assert counts.acquisition_targets != 31
    assert counts.evidence_requirements == counts.acquisition_targets  # true of this catalog, not definitionally required
    assert counts.evidence_requirements > counts.equivalence_groups


def test_all_49_legacy_rows_are_connected_to_some_requirement():
    graph = build_source_routing_graph()
    assert graph.all_served_legacy_need_ids() == {n.legacy_need_id for n in LEGACY_CATALOG}


def test_no_requirement_serves_zero_legacy_needs():
    """Silent-drop guard: every EvidenceRequirement must actually trace back
    to at least one real legacy need."""
    graph = build_source_routing_graph()
    assert all(len(r.serves_legacy_need_ids) > 0 for r in graph.requirements)


# --- 1 ResearchNeed -> multiple EvidenceRequirements ------------------------
def test_fda_concern_group_raises_two_evidence_requirements():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_0")  # "company fda concern" group
    assert len(reqs) == 2


def test_issuer_claim_and_regulator_confirmation_are_separate_requirements():
    """Requirement 5's worked example: SEC-filing disclosure by the issuer is
    never the same requirement as the regulator's own confirmation."""
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("kill_0")  # same group as bear_0
    assert len(reqs) == 2
    disclosure = next(r for r in reqs if not r.independence_requirement)
    regulator = next(r for r in reqs if r.independence_requirement)
    assert disclosure.requirement_id != regulator.requirement_id
    assert DocumentAuthority.REGULATOR in regulator.required_authorities
    assert DocumentAuthority.REGULATOR not in disclosure.required_authorities
    assert disclosure.independence_requirement is False
    assert regulator.independence_requirement is True

    # Confirming the disclosure requirement must never be read as having
    # confirmed the regulator requirement -- they are different rows.
    disclosure_targets = graph.targets_for_requirement(disclosure.requirement_id)
    regulator_targets = graph.targets_for_requirement(regulator.requirement_id)
    assert {t.target_id for t in disclosure_targets}.isdisjoint({t.target_id for t in regulator_targets})


def test_regulator_requirement_routes_not_publicly_available_never_to_search():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_0")
    regulator = next(r for r in reqs if r.independence_requirement)
    targets = graph.targets_for_requirement(regulator.requirement_id)
    assert len(targets) == 1
    routes = graph.routes_for_target(targets[0].target_id)
    assert len(routes) == 1
    assert routes[0].acquisition_method is AcquisitionMethod.NOT_PUBLICLY_AVAILABLE
    assert not routes[0].acquisition_method.requires_web_search_budget


# --- multiple ResearchNeeds -> 1 AcquisitionTarget --------------------------
def test_multiple_legacy_needs_share_one_target_via_one_requirement():
    graph = build_source_routing_graph()
    # "company dilution" group: bear_5, kill_4, kill_22
    reqs = graph.requirements_for_need("bear_5")
    assert len(reqs) == 1
    requirement = reqs[0]
    assert set(requirement.serves_legacy_need_ids) == {"bear_5", "kill_4", "kill_22"}
    targets = graph.targets_for_requirement(requirement.requirement_id)
    assert len(targets) == 1


# --- 1 Target -> multiple priority routes -----------------------------------
def test_a_sec_chain_target_has_three_priority_ordered_routes():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_5")
    target = graph.targets_for_requirement(reqs[0].requirement_id)[0]
    routes = graph.routes_for_target(target.target_id)
    assert [r.priority for r in routes] == [1, 2, 3]
    assert [r.acquisition_method for r in routes] == [
        AcquisitionMethod.EXISTING_DIRECT_API,
        AcquisitionMethod.KNOWN_URL_HTTP,
        AcquisitionMethod.WEB_SEARCH_DISCOVERY,
    ]


# --- primary document vs exhibit are separate targets -----------------------
def test_partnership_agreement_has_separate_primary_and_exhibit_targets():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bull_1")  # "company partnership agreement"
    assert len(reqs) == 2
    kinds = set()
    for requirement in reqs:
        targets = graph.targets_for_requirement(requirement.requirement_id)
        assert len(targets) == 1
        kinds.add(targets[0].target_kind)
    assert kinds == {TargetKind.SEC_PRIMARY_DOCUMENT, TargetKind.SEC_EXHIBIT}


def test_exhibit_target_uses_new_direct_adapter_for_enumeration():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bull_1")
    exhibit_req = next(
        r for r in reqs if graph.targets_for_requirement(r.requirement_id)[0].target_kind is TargetKind.SEC_EXHIBIT
    )
    target = graph.targets_for_requirement(exhibit_req.requirement_id)[0]
    routes = graph.routes_for_target(target.target_id)
    assert routes[0].acquisition_method is AcquisitionMethod.NEW_DIRECT_ADAPTER
    assert routes[0].adapter_id == "sec_exhibit_enumeration"


# --- Form 4 / insider selling ------------------------------------------------
def test_insider_selling_routes_through_form4_new_direct_adapter_first():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_12")  # "company insider selling"
    target = graph.targets_for_requirement(reqs[0].requirement_id)[0]
    assert target.target_kind == TargetKind.FORM4_FILING
    routes = graph.routes_for_target(target.target_id)
    assert routes[0].acquisition_method is AcquisitionMethod.NEW_DIRECT_ADAPTER
    assert routes[0].adapter_id == "form4_xml_parser"


# --- web-only groups have no Direct alternative -----------------------------
def test_criticism_has_no_direct_route_at_all():
    graph = build_source_routing_graph()
    reqs = graph.requirements_for_need("bear_13")  # "company criticism"
    target = graph.targets_for_requirement(reqs[0].requirement_id)[0]
    routes = graph.routes_for_target(target.target_id)
    assert len(routes) == 1
    assert routes[0].acquisition_method is AcquisitionMethod.WEB_SEARCH_DISCOVERY


def test_building_graph_raises_if_a_real_group_is_left_unclassified(monkeypatch):
    """Silent-drop guard at the builder level: a group with no archetype
    entry must raise, never fall through to an implicit default."""
    import investment_research.research.source_routing_catalog as catalog_module

    trimmed = dict(catalog_module._ARCHETYPE_BY_KEY)
    trimmed.pop("company criticism")
    monkeypatch.setattr(catalog_module, "_ARCHETYPE_BY_KEY", trimmed)
    try:
        catalog_module.build_source_routing_graph()
        raised = False
    except ValueError:
        raised = True
    assert raised, "an unclassified real group must raise, not default silently"
