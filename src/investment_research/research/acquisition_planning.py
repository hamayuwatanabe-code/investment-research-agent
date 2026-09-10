"""AcquisitionTask / AcquisitionPlan: pure planning over the legacy catalog.

Phase 2 scope only. This module makes no network call, is not wired into the
pipeline, any agent, or any collector, and does not touch ``search_batch()``/
``fetch()``/the Anthropic provider/CLI Live wiring. It answers exactly one
question, offline: given the 49 ``LegacyResearchNeed`` rows, what set of
retrieval units (``AcquisitionTask``) would actually need to run, and by what
method, if acquisition were ever executed. Whether that acquisition succeeds,
and what it means, is out of scope here -- see ``coverage_ledger.py`` for
per-need outcome tracking and ``checks.py``'s module docstring for the
Acquisition/Evidence/Check boundary this whole split exists to preserve.

Responsibilities are kept deliberately separate from ``document_store.py``:
``DocumentStore`` answers "is this the same Document we already have" once
something has actually been fetched; this module never fetches anything and
holds no document content at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..schemas.enums import ResearchDomain
from .checks import AcquisitionStatus, LegacyResearchNeed, SubjectScope
from .document_store import DocumentRole
from .legacy_catalog import LEGACY_CATALOG, group_legacy_needs


class AcquisitionMethod(str, Enum):
    """How a task's document(s) would actually be retrieved, if acquisition
    ran. Deliberately distinguishes several things a single "web search" bucket
    would otherwise hide.

    ``NOT_PUBLICLY_AVAILABLE`` is its own method, never a
    ``WEB_SEARCH_DISCOVERY`` that is merely expected to fail: a document that
    does not exist in any public location must never be sent to web search at
    all. Doing so would burn search budget on a query that can only ever
    return zero results, and risks that zero-result outcome being misread as
    "we searched and confirmed nothing exists" rather than "we always knew
    this specific document isn't public" -- a real regulator meeting minute is
    the canonical example: the minute ITSELF is not publicly available, but an
    issuer's SEC filing or IR page *describing* that meeting may well be
    (``KNOWN_URL_HTTP``/``WEB_SEARCH_DISCOVERY``) -- the two must never be
    conflated.
    """

    #: A structured collector already in this repo (SEC EDGAR, ClinicalTrials.gov,
    #: openFDA) genuinely settles this exact question by having executed.
    #: Per research/adversarial.py's own finding, NONE of today's 31 groups
    #: currently qualify -- every one is an adversarial/interpretive question
    #: a structured payload's mere existence cannot answer.
    EXISTING_DIRECT_API = "EXISTING_DIRECT_API"
    #: A free, public, structured API exists but has no collector yet
    #: (e.g. PubMed/Europe PMC, per the Phase 0B review) -- unimplemented,
    #: not unavailable.
    NEW_DIRECT_ADAPTER = "NEW_DIRECT_ADAPTER"
    #: The exact document URL is already known/derivable (e.g. a specific SEC
    #: filing/exhibit URL) -- a plain HTTP GET, no search needed.
    KNOWN_URL_HTTP = "KNOWN_URL_HTTP"
    #: Needs a general web search to find where the answer might live.
    WEB_SEARCH_DISCOVERY = "WEB_SEARCH_DISCOVERY"
    #: Needs the Anthropic server-side web_fetch tool specifically (e.g. to
    #: retrieve the body behind a URL already found).
    ANTHROPIC_WEB_FETCH = "ANTHROPIC_WEB_FETCH"
    #: The underlying document itself is known not to be publicly retrievable.
    #: Never sent to web search (see class docstring).
    NOT_PUBLICLY_AVAILABLE = "NOT_PUBLICLY_AVAILABLE"
    #: Requires a human step this system does not automate.
    MANUAL_VERIFICATION_REQUIRED = "MANUAL_VERIFICATION_REQUIRED"

    @property
    def requires_web_search_budget(self) -> bool:
        """Whether this method would consume a web-search-shaped server-tool
        use if actually executed. False for every non-search method,
        including NOT_PUBLICLY_AVAILABLE and MANUAL_VERIFICATION_REQUIRED --
        neither is ever sent to search."""
        return self in (
            AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            AcquisitionMethod.ANTHROPIC_WEB_FETCH,
        )


#: Default classification for every group built from today's real 49-row
#: catalog: every one of the 31 groups is, as things stand, a generic
#: adversarial/interpretive search phrase ("{c} going concern", "{t} FDA
#: concern", ...) with no structured collector able to settle it
#: (research/adversarial.py's own `_COLLECTOR_REDUNDANT_INTENTS` is empty for
#: exactly this reason) and no known specific document URL to fetch directly.
#: WEB_SEARCH_DISCOVERY is therefore the honest default -- it is also, not
#: coincidentally, how these exact templates are actually executed today
#: (`research.adversarial.run_adversarial_search`/`collectors.search.kill_queries`).
#: NOT_PUBLICLY_AVAILABLE/MANUAL_VERIFICATION_REQUIRED/EXISTING_DIRECT_API are
#: real, valid values on `AcquisitionTask.acquisition_method` (exercised by
#: the offline acceptance harness with synthetic tasks) but this module does
#: not invent one for any of today's real 31 groups without code-grounded
#: justification -- doing so would be exactly the kind of fabricated
#: classification the project's "never fill a gap" rule forbids.
DEFAULT_ACQUISITION_METHOD = AcquisitionMethod.WEB_SEARCH_DISCOVERY


@dataclass
class AcquisitionTask:
    """One retrieval unit: either a single legacy need or an exact-match
    group of them (``legacy_catalog.group_legacy_needs``), reframed as
    something that would need to be acquired.
    """

    task_id: str
    equivalence_key: str
    subject_scope: SubjectScope
    domain: ResearchDomain
    acquisition_method: AcquisitionMethod
    required_authority: str = "UNKNOWN"
    expected_document_roles: tuple[DocumentRole, ...] = ()
    #: EVERY original ``legacy_need_id`` this task's acquisition would serve,
    #: never truncated to a representative sample -- fan-out safety (Phase 2
    #: requirement D) depends on this list being complete.
    serves_legacy_need_ids: tuple[str, ...] = ()
    status: AcquisitionStatus = AcquisitionStatus.UNSEARCHED


@dataclass
class AcquisitionPlan:
    """The full set of ``AcquisitionTask``s built from a legacy catalog."""

    tasks: tuple[AcquisitionTask, ...] = field(default_factory=tuple)

    def all_served_legacy_need_ids(self) -> set[str]:
        served: set[str] = set()
        for task in self.tasks:
            served.update(task.serves_legacy_need_ids)
        return served

    def task_for_need(self, legacy_need_id: str) -> AcquisitionTask | None:
        for task in self.tasks:
            if legacy_need_id in task.serves_legacy_need_ids:
                return task
        return None


def build_acquisition_plan(
    catalog: tuple[LegacyResearchNeed, ...] = LEGACY_CATALOG,
    *,
    method_for: dict[str, AcquisitionMethod] | None = None,
) -> AcquisitionPlan:
    """Build an ``AcquisitionPlan`` by grouping ``catalog`` via the same
    exact-match ``group_legacy_needs`` Phase 1/2 dedup uses.

    Over ``LEGACY_CATALOG`` this produces exactly 31 tasks, together serving
    all 49 original legacy_need_ids with none dropped (asserted by
    ``tests/unit/test_acquisition_planning.py``).

    ``method_for`` optionally overrides ``DEFAULT_ACQUISITION_METHOD`` per
    equivalence_key, for callers (tests, a future manual classification pass)
    that have grounds to classify a specific group differently. Absent an
    override, every task defaults to WEB_SEARCH_DISCOVERY (see
    ``DEFAULT_ACQUISITION_METHOD``).
    """
    by_id = {need.legacy_need_id: need for need in catalog}
    groups = group_legacy_needs(catalog)
    overrides = method_for or {}

    tasks: list[AcquisitionTask] = []
    for index, ((equivalence_key, domain, subject_scope), need_ids) in enumerate(groups.items()):
        representative = by_id[need_ids[0]]
        tasks.append(
            AcquisitionTask(
                task_id=f"task_{index:03d}",
                equivalence_key=equivalence_key,
                subject_scope=subject_scope,
                domain=domain,
                acquisition_method=overrides.get(equivalence_key, DEFAULT_ACQUISITION_METHOD),
                required_authority=representative.required_authority,
                expected_document_roles=representative.expected_document_roles,
                serves_legacy_need_ids=need_ids,
            )
        )
    return AcquisitionPlan(tasks=tuple(tasks))
