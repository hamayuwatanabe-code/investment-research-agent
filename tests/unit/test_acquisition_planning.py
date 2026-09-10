"""AcquisitionTask / AcquisitionPlan: grouping, method classification, and the
NOT_PUBLICLY_AVAILABLE-never-goes-to-search guarantee (Phase 2 requirement 4).
"""

from __future__ import annotations

from investment_research.research.acquisition_planning import (
    DEFAULT_ACQUISITION_METHOD,
    AcquisitionMethod,
    AcquisitionTask,
    build_acquisition_plan,
)
from investment_research.research.budget_feasibility import assess_budget_feasibility
from investment_research.research.checks import AcquisitionStatus, SubjectScope
from investment_research.research.legacy_catalog import LEGACY_CATALOG
from investment_research.schemas.enums import ResearchDomain


def test_plan_has_exactly_31_tasks_serving_all_49_needs():
    plan = build_acquisition_plan()
    assert len(plan.tasks) == 31
    assert plan.all_served_legacy_need_ids() == {n.legacy_need_id for n in LEGACY_CATALOG}


def test_every_task_id_is_unique():
    plan = build_acquisition_plan()
    ids = [t.task_id for t in plan.tasks]
    assert len(ids) == len(set(ids))


def test_task_for_need_finds_the_right_task_for_a_grouped_need():
    plan = build_acquisition_plan()
    task_for_bear0 = plan.task_for_need("bear_0")
    task_for_kill0 = plan.task_for_need("kill_0")
    assert task_for_bear0 is not None
    assert task_for_bear0.task_id == task_for_kill0.task_id
    assert {"bear_0", "kill_0", "kill_18"} <= set(task_for_bear0.serves_legacy_need_ids)


def test_default_method_is_web_search_discovery_for_all_real_groups():
    """Grounded in code (research/adversarial.py's own empty
    _COLLECTOR_REDUNDANT_INTENTS): none of today's 31 groups is settled by an
    existing structured collector, so nothing here is classified
    EXISTING_DIRECT_API without justification this module does not have."""
    plan = build_acquisition_plan()
    assert all(t.acquisition_method is DEFAULT_ACQUISITION_METHOD for t in plan.tasks)
    assert DEFAULT_ACQUISITION_METHOD is AcquisitionMethod.WEB_SEARCH_DISCOVERY


def test_method_override_is_respected_per_equivalence_key():
    plan = build_acquisition_plan(method_for={"company fda concern": AcquisitionMethod.KNOWN_URL_HTTP})
    task = plan.task_for_need("bear_0")
    assert task is not None
    assert task.acquisition_method is AcquisitionMethod.KNOWN_URL_HTTP


# --- requirement 4: NOT_PUBLICLY_AVAILABLE is a real, distinguished method --
def test_not_publicly_available_task_is_never_counted_as_a_search():
    """A synthetic task standing in for the exact example in the spec: the
    text of a non-public regulator meeting minute is NOT_PUBLICLY_AVAILABLE,
    while whether an issuer disclosed the same meeting in an SEC filing is a
    separate, genuinely searchable question -- the two must never collapse
    into one task or one status."""
    minutes_task = AcquisitionTask(
        task_id="task_synthetic_minutes",
        equivalence_key="program type c meeting minutes text",
        subject_scope=SubjectScope.PROGRAM,
        domain=ResearchDomain.REGULATORY,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
        serves_legacy_need_ids=("synthetic_minutes_need",),
    )
    disclosure_task = AcquisitionTask(
        task_id="task_synthetic_disclosure",
        equivalence_key="company type c meeting disclosed in filing",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.REGULATORY,
        acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
        serves_legacy_need_ids=("synthetic_disclosure_need",),
    )
    from investment_research.research.acquisition_planning import AcquisitionPlan

    plan = AcquisitionPlan(tasks=(minutes_task, disclosure_task))
    feasibility = assess_budget_feasibility(plan)

    assert not minutes_task.acquisition_method.requires_web_search_budget
    assert disclosure_task.acquisition_method.requires_web_search_budget
    assert feasibility.expected_server_tool_uses == 1  # only the disclosure task
    assert feasibility.web_search_tasks == 1


def test_manual_verification_required_is_distinguishable_from_unsearched():
    task = AcquisitionTask(
        task_id="task_synthetic_manual",
        equivalence_key="company manual step",
        subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CONTRADICTION,
        acquisition_method=AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED,
        status=AcquisitionStatus.MANUAL_VERIFICATION_REQUIRED,
        serves_legacy_need_ids=("synthetic_manual_need",),
    )
    assert task.status is AcquisitionStatus.MANUAL_VERIFICATION_REQUIRED
    assert task.status is not AcquisitionStatus.UNSEARCHED
    assert not task.acquisition_method.requires_web_search_budget
