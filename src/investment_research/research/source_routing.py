"""Source Routing Graph: EvidenceRequirement / AcquisitionTarget / AcquisitionRoute.

Phase 2.5 scope only. Pure types and pure functions -- no network call, no
pipeline wiring, no execution. This module replaces the Phase 2
one-need-equals-one-method assumption (``acquisition_planning.AcquisitionTask``,
kept for backward compatibility and its own passing tests, but no longer the
model this project treats as authoritative) with an explicit many-to-many
graph:

    LegacyResearchNeed --(many)--> EvidenceRequirement --(many)--> AcquisitionTarget --(priority-ordered)--> AcquisitionRoute

* One ``LegacyResearchNeed`` (or exact-equivalence group of them) can raise
  MORE THAN ONE ``EvidenceRequirement`` -- most importantly, a claim about
  what a REGULATOR said and a claim about what the ISSUER disclosed a
  regulator said are two different propositions with two different
  authorities, never one requirement (Phase 2.5 requirement 5).
* One ``AcquisitionTarget`` (one real document, e.g. "the primary 10-Q body")
  can satisfy several ``EvidenceRequirement``s.
* One ``AcquisitionTarget`` has one or more priority-ordered
  ``AcquisitionRoute``s -- different ways of obtaining the SAME document,
  cheapest/free-est first (EXISTING_DIRECT_API before KNOWN_URL_HTTP before
  WEB_SEARCH_DISCOVERY before ANTHROPIC_WEB_FETCH before
  NOT_PUBLICLY_AVAILABLE/MANUAL_VERIFICATION_REQUIRED).

31 is the count of exact-equivalence GROUPS (``legacy_catalog.group_legacy_needs``).
It is not, and must never be read as, the count of ``EvidenceRequirement``s,
``AcquisitionTarget``s, or routes -- see ``source_routing_catalog.py`` for the
actual, separately-measured counts.

Acquiring a document through a route, even successfully, never promotes any
``LegacyResearchNeed`` to an evidence-level status -- see ``checks.py``'s
module docstring for the Acquisition/Evidence/Check boundary this module
also respects. ``resolve_active_route`` below decides only which route a
state machine would try next; it never claims a document was fetched or a
claim confirmed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from ..schemas.enums import DocumentAuthority
from .acquisition_planning import AcquisitionMethod
from .checks import AcquisitionStatus, SubjectScope
from .document_store import DocumentRole


class TargetKind(str, Enum):
    """What real-world kind of document/record an ``AcquisitionTarget`` is."""

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
    """Coarse LLM-token cost band for a route, if it were executed."""

    ZERO = "ZERO"  # direct API/HTTP: no LLM call at all
    LOW = "LOW"  # a targeted Anthropic web_fetch of an already-known URL
    HIGH = "HIGH"  # an open-ended web_search discovery call


class RequestCostClass(str, Enum):
    """Coarse network-request cost/constraint band for a route."""

    FREE = "FREE"
    RATE_LIMITED = "RATE_LIMITED"
    PAID_LLM_CALL = "PAID_LLM_CALL"


@dataclass(frozen=True)
class EvidenceRequirement:
    """One distinct proposition that would need to be confirmed.

    Two requirements about superficially "the same" query can differ in
    everything that matters: who would need to be the speaker
    (``required_authorities``), whether independent (non-issuer) confirmation
    is required at all, and what document roles could possibly satisfy them.
    """

    requirement_id: str
    #: EVERY legacy_need_id (across every LegacyResearchNeed that raised this
    #: requirement) -- never a representative sample.
    serves_legacy_need_ids: tuple[str, ...]
    subject_scope: SubjectScope
    #: A specific programme identifier when subject_scope is PROGRAM;
    #: "UNKNOWN" for a company-wide requirement. Never a real drug/programme
    #: name in this catalog -- filled in only when a real run instantiates it.
    program_scope: str = "UNKNOWN"
    #: Free-text description of exactly what is being claimed and by whom,
    #: e.g. "issuer disclosed FDA meeting content" vs. "FDA itself stated
    #: this" -- the axis requirement 5 exists to keep apart.
    claim_scope: str = ""
    required_authorities: tuple[DocumentAuthority, ...] = ()
    #: True only for a requirement that specifically needs a non-issuer
    #: (REGULATOR/INDEPENDENT) speaker -- an issuer's own filing can never
    #: satisfy one of these, however primary the filing's tier.
    independence_requirement: bool = False
    #: Coarse recency/applicability window, e.g. "CURRENT_PROGRAM_ONLY" or
    #: "ANY" -- deliberately a plain string, not a date-arithmetic type, since
    #: Phase 2.5 does no date computation.
    date_scope: str = "ANY"
    acceptable_document_roles: tuple[DocumentRole, ...] = ()
    blocking_if_unresolved: bool = False


@dataclass(frozen=True)
class AcquisitionTarget:
    """One real document (or structured record) that, if obtained, could
    satisfy one or more ``EvidenceRequirement``s."""

    target_id: str
    target_kind: TargetKind
    #: Placeholder-level identifiers only -- "UNKNOWN" in this static catalog;
    #: a real run would fill in the run's own issuer/programme identifiers,
    #: never a value hardcoded here.
    issuer_identifier: str = "UNKNOWN"
    program_identifier: str = "UNKNOWN"
    document_role: DocumentRole = DocumentRole.UNKNOWN
    authority: DocumentAuthority = DocumentAuthority.UNKNOWN
    lookup_parameters: Mapping[str, str] = field(default_factory=dict)
    #: EVERY EvidenceRequirement this target could satisfy.
    serves_requirement_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcquisitionRoute:
    """One way of obtaining one ``AcquisitionTarget``, at one priority."""

    route_id: str
    target_id: str
    #: Lower number = tried first. Unique per target.
    priority: int
    acquisition_method: AcquisitionMethod
    #: Names the specific collector/adapter this route would use, e.g.
    #: "sec_edgar_submissions", "clinicaltrials_api", "pubmed" (unimplemented
    #: -- NEW_DIRECT_ADAPTER), "http_client", "anthropic_web_search". Purely
    #: documentary in Phase 2.5 -- nothing here calls it.
    adapter_id: str = "UNKNOWN"
    expected_authority: DocumentAuthority = DocumentAuthority.UNKNOWN
    token_cost_class: TokenCostClass = TokenCostClass.ZERO
    request_cost_class: RequestCostClass = RequestCostClass.FREE
    #: The set of outcomes on the PRECEDING route (this route's target,
    #: previous priority) that make this route eligible to be attempted.
    #: Meaningless (never consulted) for the lowest-priority route on a
    #: target, which is always eligible first.
    fallback_conditions: tuple[AcquisitionStatus, ...] = ()
    #: The set of outcomes on THIS route, once attempted, after which no
    #: further route on this target is attempted.
    terminal_conditions: tuple[AcquisitionStatus, ...] = (AcquisitionStatus.ACQUIRED,)


@dataclass
class SourceRoutingGraph:
    requirements: tuple[EvidenceRequirement, ...] = ()
    targets: tuple[AcquisitionTarget, ...] = ()
    routes: tuple[AcquisitionRoute, ...] = ()

    def routes_for_target(self, target_id: str) -> list[AcquisitionRoute]:
        return sorted(
            (r for r in self.routes if r.target_id == target_id), key=lambda r: r.priority
        )

    def targets_for_requirement(self, requirement_id: str) -> list[AcquisitionTarget]:
        return [t for t in self.targets if requirement_id in t.serves_requirement_ids]

    def requirements_for_need(self, legacy_need_id: str) -> list[EvidenceRequirement]:
        return [r for r in self.requirements if legacy_need_id in r.serves_legacy_need_ids]

    def all_served_legacy_need_ids(self) -> set[str]:
        served: set[str] = set()
        for requirement in self.requirements:
            served.update(requirement.serves_legacy_need_ids)
        return served


def resolve_active_route(
    routes: Sequence[AcquisitionRoute], outcomes: Mapping[str, AcquisitionStatus]
) -> AcquisitionRoute | None:
    """Pure state-machine step: given one target's routes (any order) and a
    map of route_id -> outcome for routes ALREADY attempted, return the route
    that should be attempted next, or ``None`` if the chain has reached a
    terminal outcome (or is exhausted) and no further route should run.

    This performs no acquisition and calls nothing external. It only answers
    "what would the plan do next", given outcomes the caller supplies (real
    ones, from an execution this module never performs, or hypothetical ones,
    as ``budget_feasibility``'s NORMAL/DEGRADED/WORST scenarios use).
    """
    ordered = sorted(routes, key=lambda r: r.priority)
    previous_outcome: AcquisitionStatus | None = None
    for index, route in enumerate(ordered):
        if index > 0:
            if previous_outcome is None:
                # A later route was asked about before its predecessor was
                # ever attempted -- conservatively, nothing beyond the
                # predecessor is eligible yet.
                return None
            if previous_outcome not in route.fallback_conditions:
                # The predecessor's outcome does not authorize falling
                # through to this route -- the chain stops here.
                return None
        outcome = outcomes.get(route.route_id)
        if outcome is None:
            return route  # not yet attempted -- this is the next step
        if outcome in route.terminal_conditions:
            return None  # this route's own outcome ends the chain
        previous_outcome = outcome
    return None  # every route on this target has been attempted
