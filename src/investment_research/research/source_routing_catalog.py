"""The Phase 2.5 Source Routing Graph, built from the real 49-row legacy
catalog / 31 exact-equivalence groups (``legacy_catalog.group_legacy_needs``).

31 is the group count only. This module deliberately measures several other
counts separately (``routing_coverage_counts()``): the number of
``EvidenceRequirement``s raised (more than 31, because six REGULATORY groups
about FDA correspondence split into two requirements each -- see
``_ARCHETYPE_BY_KEY``'s ``"fda_dual"`` entries and requirement 5's authority-
separation rule), the number of ``AcquisitionTarget``s, and the number of
routes by method.

Every one of the 31 groups is classified by a genuinely code-grounded
archetype -- never "Direct-first" merely by default (Phase 2's mistake this
turn corrects). The grounding for each archetype:

* ``sec_chain`` / ``sec_chain_with_exhibit`` -- the claim is the kind of
  event a US issuer is required to disclose (8-K/10-K/10-Q/424B/DEF 14A):
  dilution, going concern, warrants, a reverse split, delisting, litigation,
  an auditor change, a financing, accounting/restatement/SEC-investigation
  disclosures, an FDA designation announcement, a partnership/collaboration
  agreement. ``SecEdgarCollector`` genuinely fetches filing metadata for
  free (``EXISTING_DIRECT_API``); when the metadata resolves a
  ``primaryDocument``, ``FILING_INDEX_URL`` genuinely resolves to the primary
  document body (a plain HTTP GET, ``KNOWN_URL_HTTP``) -- both zero LLM
  tokens. Neither route, even on success, is treated as confirming the
  underlying adversarial claim (only that a candidate document was
  obtained) -- Web Search remains the fallback when the SEC document, once
  read, does not actually contain the claim.
* ``fda_dual`` -- requirement 5's own worked example, generalized: whether
  the ISSUER disclosed a regulator interaction (``sec_chain``, STATUTORY_
  FILING/COMPANY_IR authority, never independent) is a SEPARATE requirement
  from whether the REGULATOR itself said so. The six groups classified here
  (FDA concern, regulatory risk, an endpoint concern under either template,
  a clinical hold, a Complete Response Letter) name exactly the substance of
  ``collectors/fda.py``'s own documented ``NON_API_REGULATORY_QUESTIONS``
  (Type A/B/C meeting minutes, CRL text, clinical hold notices, SPA/
  surrogate-endpoint agreement) -- no API this repository has ever reaches
  the regulator's own document for these, so the independent-confirmation
  requirement routes NOT_PUBLICLY_AVAILABLE, never to Web Search.
* ``clinicaltrials_web`` -- ``ClinicalTrialsCollector`` genuinely returns a
  structured, free ``status``/``primary_completion`` field directly relevant
  to "failed trial" and catalyst-timing claims (not a full answer, but a
  real, free, structured first attempt) before any paid search.
* ``clinicaltrials_pubmed_web`` -- adds Europe PMC/PubMed, a free public API
  this repository does not yet implement (``NEW_DIRECT_ADAPTER``, per the
  Phase 0B review), genuinely relevant to literature-grounded claims
  (clinical data results, mechanism/efficacy, a safety signal) before Web
  Search.
* ``form4`` -- insider selling is exactly what Form 4 reports; XML parsing
  is unimplemented (``NEW_DIRECT_ADAPTER``, per this repo's own documented
  gap) but the target is real and free once built.
* ``web_only`` -- criticism, a short thesis, and "is a competitor better"
  are third-party sentiment/analysis with no registry, filing, or structured
  API that would ever contain them. No Direct route is invented for these.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.enums import DocumentAuthority, ResearchDomain
from .acquisition_planning import AcquisitionMethod
from .checks import AcquisitionStatus, SubjectScope
from .document_store import DocumentRole
from .legacy_catalog import LEGACY_CATALOG, group_legacy_needs
from .source_routing import (
    AcquisitionRoute,
    AcquisitionTarget,
    EvidenceRequirement,
    RequestCostClass,
    SourceRoutingGraph,
    TargetKind,
    TokenCostClass,
)

#: Every one of the real 31 groups, by its exact equivalence_key, mapped to
#: the archetype that builds its requirements/targets/routes. See the module
#: docstring for the grounding behind each archetype choice.
_ARCHETYPE_BY_KEY: dict[str, str] = {
    # -- fda_dual: issuer disclosure vs. regulator's own position ----------
    "company fda concern": "fda_dual",
    "company regulatory risk": "fda_dual",
    "program endpoint concern": "fda_dual",
    "company endpoint": "fda_dual",
    "company clinical hold": "fda_dual",
    "company complete response letter": "fda_dual",
    # -- clinicaltrials_web: registry status/date is a real free signal -----
    "company delayed catalyst": "clinicaltrials_web",
    "company failed trial": "clinicaltrials_web",
    "company upcoming catalyst readout": "clinicaltrials_web",
    "company failed": "clinicaltrials_web",
    # -- clinicaltrials_pubmed_web: literature-grounded claims --------------
    "program safety concern": "clinicaltrials_pubmed_web",
    "company clinical data results": "clinicaltrials_pubmed_web",
    "program mechanism efficacy": "clinicaltrials_pubmed_web",
    # -- form4: insider selling is literally what Form 4 reports -----------
    "company insider selling": "form4",
    # -- web_only: no Direct source exists for third-party sentiment --------
    "company criticism": "web_only",
    "company short thesis": "web_only",
    "program competitor superiority": "web_only",
    # -- sec_chain_with_exhibit: the agreement text is typically an exhibit -
    "company partnership agreement": "sec_chain_with_exhibit",
    # -- sec_chain: routine 8-K/10-K/10-Q/424B/DEF 14A disclosure -----------
    "company dilution": "sec_chain",
    "company going concern": "sec_chain",
    "company warrant": "sec_chain",
    "company reverse split": "sec_chain",
    "company delisting": "sec_chain",
    "company lawsuit": "sec_chain",
    "company auditor": "sec_chain",
    "company fda designation": "sec_chain",
    "company cash runway financing": "sec_chain",
    "company accounting": "sec_chain",
    "company sec investigation": "sec_chain",
    "company restatement": "sec_chain",
    "company offering priced": "sec_chain",
}


def _sec_document_routes(
    index_tag: str, target_id: str
) -> list[AcquisitionRoute]:
    return [
        AcquisitionRoute(
            route_id=f"route_{index_tag}_1",
            target_id=target_id,
            priority=1,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
            adapter_id="sec_edgar_submissions",
            expected_authority=DocumentAuthority.STATUTORY_FILING,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.RATE_LIMITED,
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{index_tag}_2",
            target_id=target_id,
            priority=2,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
            adapter_id="http_client",
            expected_authority=DocumentAuthority.STATUTORY_FILING,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.FREE,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{index_tag}_3",
            target_id=target_id,
            priority=3,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            adapter_id="anthropic_web_search",
            expected_authority=DocumentAuthority.UNKNOWN,
            token_cost_class=TokenCostClass.HIGH,
            request_cost_class=RequestCostClass.PAID_LLM_CALL,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]


def _sec_chain(index: int, need_ids: tuple[str, ...], domain: ResearchDomain, subject_scope: SubjectScope, key: str) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id = f"req_{tag}"
    target_id = f"target_{tag}"
    requirement = EvidenceRequirement(
        requirement_id=req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"issuer disclosed in a filing: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        independence_requirement=False,
        acceptable_document_roles=(DocumentRole.PRIMARY_DOCUMENT,),
        blocking_if_unresolved=domain in (ResearchDomain.REGULATORY, ResearchDomain.CAPITAL_STRUCTURE),
    )
    target = AcquisitionTarget(
        target_id=target_id,
        target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        document_role=DocumentRole.PRIMARY_DOCUMENT,
        authority=DocumentAuthority.STATUTORY_FILING,
        serves_requirement_ids=(req_id,),
    )
    return SourceRoutingGraph(
        requirements=(requirement,), targets=(target,), routes=tuple(_sec_document_routes(tag, target_id))
    )


def _sec_chain_with_exhibit(index: int, need_ids: tuple[str, ...], domain: ResearchDomain, subject_scope: SubjectScope, key: str) -> SourceRoutingGraph:
    """requirement 4's literal "SEC exhibit enumeration" example, and
    requirement 10's required "primary document and exhibit are separate
    targets" test: an agreement announced in an 8-K's BODY is one target;
    the agreement's own TEXT, when filed as an exhibit to that same 8-K, is
    a second, separate target -- enumerating which exhibit exists at all is
    not something SecEdgarCollector does today (NEW_DIRECT_ADAPTER)."""
    body_tag = f"{index:03d}a"
    body_req_id = f"req_{body_tag}"
    body_target_id = f"target_{body_tag}"
    body_requirement = EvidenceRequirement(
        requirement_id=body_req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"issuer disclosed in a filing (body): {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        acceptable_document_roles=(DocumentRole.PRIMARY_DOCUMENT,),
    )
    body_target = AcquisitionTarget(
        target_id=body_target_id,
        target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        document_role=DocumentRole.PRIMARY_DOCUMENT,
        authority=DocumentAuthority.STATUTORY_FILING,
        serves_requirement_ids=(body_req_id,),
    )
    body_routes = _sec_document_routes(body_tag, body_target_id)

    exhibit_tag = f"{index:03d}b"
    exhibit_req_id = f"req_{exhibit_tag}"
    exhibit_target_id = f"target_{exhibit_tag}"
    exhibit_requirement = EvidenceRequirement(
        requirement_id=exhibit_req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"the underlying agreement text itself (exhibit): {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING,),
        acceptable_document_roles=(DocumentRole.EXHIBIT,),
    )
    exhibit_target = AcquisitionTarget(
        target_id=exhibit_target_id,
        target_kind=TargetKind.SEC_EXHIBIT,
        document_role=DocumentRole.EXHIBIT,
        authority=DocumentAuthority.STATUTORY_FILING,
        serves_requirement_ids=(exhibit_req_id,),
    )
    exhibit_routes = [
        AcquisitionRoute(
            route_id=f"route_{exhibit_tag}_1",
            target_id=exhibit_target_id,
            priority=1,
            acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
            adapter_id="sec_exhibit_enumeration",
            expected_authority=DocumentAuthority.STATUTORY_FILING,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.RATE_LIMITED,
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{exhibit_tag}_2",
            target_id=exhibit_target_id,
            priority=2,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
            adapter_id="http_client",
            expected_authority=DocumentAuthority.STATUTORY_FILING,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.FREE,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{exhibit_tag}_3",
            target_id=exhibit_target_id,
            priority=3,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            adapter_id="anthropic_web_search",
            token_cost_class=TokenCostClass.HIGH,
            request_cost_class=RequestCostClass.PAID_LLM_CALL,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]

    return SourceRoutingGraph(
        requirements=(body_requirement, exhibit_requirement),
        targets=(body_target, exhibit_target),
        routes=tuple(body_routes + exhibit_routes),
    )


def _fda_dual(index: int, need_ids: tuple[str, ...], domain: ResearchDomain, subject_scope: SubjectScope, key: str) -> SourceRoutingGraph:
    """Requirement 5's own worked example: issuer disclosure and regulator
    confirmation are two requirements, never one."""
    disclosure_tag = f"{index:03d}a"
    disclosure_req_id = f"req_{disclosure_tag}"
    disclosure_target_id = f"target_{disclosure_tag}"
    disclosure_requirement = EvidenceRequirement(
        requirement_id=disclosure_req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"issuer disclosed in a filing: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        independence_requirement=False,
        acceptable_document_roles=(DocumentRole.PRIMARY_DOCUMENT,),
        blocking_if_unresolved=True,
    )
    disclosure_target = AcquisitionTarget(
        target_id=disclosure_target_id,
        target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        document_role=DocumentRole.PRIMARY_DOCUMENT,
        authority=DocumentAuthority.STATUTORY_FILING,
        serves_requirement_ids=(disclosure_req_id,),
    )
    disclosure_routes = _sec_document_routes(disclosure_tag, disclosure_target_id)

    regulator_tag = f"{index:03d}b"
    regulator_req_id = f"req_{regulator_tag}"
    regulator_target_id = f"target_{regulator_tag}"
    regulator_requirement = EvidenceRequirement(
        requirement_id=regulator_req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"the regulator itself stated: {key}",
        required_authorities=(DocumentAuthority.REGULATOR,),
        independence_requirement=True,
        acceptable_document_roles=(DocumentRole.PRIMARY_DOCUMENT,),
        blocking_if_unresolved=True,
    )
    regulator_target = AcquisitionTarget(
        target_id=regulator_target_id,
        target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE,
        document_role=DocumentRole.PRIMARY_DOCUMENT,
        authority=DocumentAuthority.REGULATOR,
        serves_requirement_ids=(regulator_req_id,),
    )
    regulator_routes = [
        AcquisitionRoute(
            route_id=f"route_{regulator_tag}_1",
            target_id=regulator_target_id,
            priority=1,
            acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE,
            adapter_id="none",
            expected_authority=DocumentAuthority.REGULATOR,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.FREE,
            terminal_conditions=(AcquisitionStatus.NOT_PUBLICLY_AVAILABLE,),
        ),
    ]

    return SourceRoutingGraph(
        requirements=(disclosure_requirement, regulator_requirement),
        targets=(disclosure_target, regulator_target),
        routes=tuple(disclosure_routes + regulator_routes),
    )


def _clinicaltrials_web(index: int, need_ids: tuple[str, ...], domain: ResearchDomain, subject_scope: SubjectScope, key: str) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id = f"req_{tag}"
    target_id = f"target_{tag}"
    requirement = EvidenceRequirement(
        requirement_id=req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"trial registry / literature signal for: {key}",
        required_authorities=(DocumentAuthority.REGISTRY, DocumentAuthority.INDEPENDENT),
        acceptable_document_roles=(DocumentRole.STRUCTURED_API_RECORD,),
    )
    target = AcquisitionTarget(
        target_id=target_id,
        target_kind=TargetKind.CLINICALTRIALS_RECORD,
        document_role=DocumentRole.STRUCTURED_API_RECORD,
        authority=DocumentAuthority.REGISTRY,
        serves_requirement_ids=(req_id,),
    )
    routes = [
        AcquisitionRoute(
            route_id=f"route_{tag}_1",
            target_id=target_id,
            priority=1,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
            adapter_id="clinicaltrials_api",
            expected_authority=DocumentAuthority.REGISTRY,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.FREE,
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{tag}_2",
            target_id=target_id,
            priority=2,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            adapter_id="anthropic_web_search",
            token_cost_class=TokenCostClass.HIGH,
            request_cost_class=RequestCostClass.PAID_LLM_CALL,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), routes=tuple(routes))


def _clinicaltrials_pubmed_web(index: int, need_ids: tuple[str, ...], domain: ResearchDomain, subject_scope: SubjectScope, key: str) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id = f"req_{tag}"
    target_id = f"target_{tag}"
    requirement = EvidenceRequirement(
        requirement_id=req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"trial registry / literature signal for: {key}",
        required_authorities=(DocumentAuthority.REGISTRY, DocumentAuthority.INDEPENDENT),
        acceptable_document_roles=(DocumentRole.STRUCTURED_API_RECORD, DocumentRole.JOURNAL_ARTICLE),
    )
    #: Simplification, documented here: this single AcquisitionTarget stands
    #: for "the best structured/literature source found", served by three
    #: alternative routes -- not three byte-identical documents. Splitting
    #: ClinicalTrials.gov and Europe PMC into two separate Target rows would
    #: be more literal but adds no information Phase 2.5 uses; the route
    #: chain (Direct registry, then Direct literature, then Web) is what
    #: matters for planning.
    target = AcquisitionTarget(
        target_id=target_id,
        target_kind=TargetKind.CLINICALTRIALS_RECORD,
        document_role=DocumentRole.STRUCTURED_API_RECORD,
        authority=DocumentAuthority.REGISTRY,
        serves_requirement_ids=(req_id,),
    )
    routes = [
        AcquisitionRoute(
            route_id=f"route_{tag}_1",
            target_id=target_id,
            priority=1,
            acquisition_method=AcquisitionMethod.EXISTING_DIRECT_API,
            adapter_id="clinicaltrials_api",
            expected_authority=DocumentAuthority.REGISTRY,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.FREE,
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{tag}_2",
            target_id=target_id,
            priority=2,
            acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
            adapter_id="pubmed_europepmc",
            expected_authority=DocumentAuthority.INDEPENDENT,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.FREE,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{tag}_3",
            target_id=target_id,
            priority=3,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            adapter_id="anthropic_web_search",
            token_cost_class=TokenCostClass.HIGH,
            request_cost_class=RequestCostClass.PAID_LLM_CALL,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), routes=tuple(routes))


def _form4(index: int, need_ids: tuple[str, ...], domain: ResearchDomain, subject_scope: SubjectScope, key: str) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id = f"req_{tag}"
    target_id = f"target_{tag}"
    requirement = EvidenceRequirement(
        requirement_id=req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"insider transaction filing for: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING,),
        acceptable_document_roles=(DocumentRole.STRUCTURED_API_RECORD,),
    )
    target = AcquisitionTarget(
        target_id=target_id,
        target_kind=TargetKind.FORM4_FILING,
        document_role=DocumentRole.STRUCTURED_API_RECORD,
        authority=DocumentAuthority.STATUTORY_FILING,
        serves_requirement_ids=(req_id,),
    )
    routes = [
        AcquisitionRoute(
            route_id=f"route_{tag}_1",
            target_id=target_id,
            priority=1,
            acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
            adapter_id="form4_xml_parser",
            expected_authority=DocumentAuthority.STATUTORY_FILING,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.RATE_LIMITED,
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{tag}_2",
            target_id=target_id,
            priority=2,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP,
            adapter_id="http_client",
            expected_authority=DocumentAuthority.STATUTORY_FILING,
            token_cost_class=TokenCostClass.ZERO,
            request_cost_class=RequestCostClass.FREE,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED,),
        ),
        AcquisitionRoute(
            route_id=f"route_{tag}_3",
            target_id=target_id,
            priority=3,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            adapter_id="anthropic_web_search",
            token_cost_class=TokenCostClass.HIGH,
            request_cost_class=RequestCostClass.PAID_LLM_CALL,
            fallback_conditions=(AcquisitionStatus.ACQUIRED_ZERO_RESULTS, AcquisitionStatus.FAILED),
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), routes=tuple(routes))


def _web_only(index: int, need_ids: tuple[str, ...], domain: ResearchDomain, subject_scope: SubjectScope, key: str) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id = f"req_{tag}"
    target_id = f"target_{tag}"
    requirement = EvidenceRequirement(
        requirement_id=req_id,
        serves_legacy_need_ids=need_ids,
        subject_scope=subject_scope,
        claim_scope=f"third-party sentiment/analysis: {key}",
        required_authorities=(DocumentAuthority.INDEPENDENT,),
        acceptable_document_roles=(DocumentRole.JOURNAL_ARTICLE, DocumentRole.PRESS_RELEASE),
    )
    target = AcquisitionTarget(
        target_id=target_id,
        target_kind=TargetKind.GENERIC_WEB_DOCUMENT,
        authority=DocumentAuthority.INDEPENDENT,
        serves_requirement_ids=(req_id,),
    )
    routes = [
        AcquisitionRoute(
            route_id=f"route_{tag}_1",
            target_id=target_id,
            priority=1,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY,
            adapter_id="anthropic_web_search",
            token_cost_class=TokenCostClass.HIGH,
            request_cost_class=RequestCostClass.PAID_LLM_CALL,
            terminal_conditions=(AcquisitionStatus.ACQUIRED, AcquisitionStatus.ACQUIRED_ZERO_RESULTS),
        ),
    ]
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), routes=tuple(routes))


_BUILDERS = {
    "sec_chain": _sec_chain,
    "sec_chain_with_exhibit": _sec_chain_with_exhibit,
    "fda_dual": _fda_dual,
    "clinicaltrials_web": _clinicaltrials_web,
    "clinicaltrials_pubmed_web": _clinicaltrials_pubmed_web,
    "form4": _form4,
    "web_only": _web_only,
}


def build_source_routing_graph() -> SourceRoutingGraph:
    """Build the full graph over all 31 real groups. Deterministic: iterates
    ``group_legacy_needs(LEGACY_CATALOG)`` in its own (catalog) order."""
    groups = group_legacy_needs(LEGACY_CATALOG)
    requirements: list[EvidenceRequirement] = []
    targets: list[AcquisitionTarget] = []
    routes: list[AcquisitionRoute] = []

    for index, ((equivalence_key, domain, subject_scope), need_ids) in enumerate(groups.items()):
        archetype = _ARCHETYPE_BY_KEY.get(equivalence_key)
        if archetype is None:
            raise ValueError(
                f"group {equivalence_key!r} (domain={domain}) has no routing archetype -- "
                "every one of the 31 real groups must be individually classified, never "
                "defaulted"
            )
        builder = _BUILDERS[archetype]
        sub_graph = builder(index, need_ids, domain, subject_scope, equivalence_key)
        requirements.extend(sub_graph.requirements)
        targets.extend(sub_graph.targets)
        routes.extend(sub_graph.routes)

    return SourceRoutingGraph(
        requirements=tuple(requirements), targets=tuple(targets), routes=tuple(routes)
    )


@dataclass(frozen=True)
class RoutingCoverageCounts:
    legacy_needs: int
    equivalence_groups: int
    evidence_requirements: int
    acquisition_targets: int
    existing_direct_api_routes: int
    new_direct_adapter_routes: int
    known_url_http_routes: int
    web_search_discovery_routes: int
    anthropic_web_fetch_routes: int
    not_publicly_available_routes: int
    manual_verification_required_routes: int


def routing_coverage_counts(graph: SourceRoutingGraph | None = None) -> RoutingCoverageCounts:
    """Every count requirement 2 asks to be measured separately -- 31 is
    never substituted for any of these."""
    graph = graph or build_source_routing_graph()

    def count(method: AcquisitionMethod) -> int:
        return sum(1 for r in graph.routes if r.acquisition_method is method)

    return RoutingCoverageCounts(
        legacy_needs=len(LEGACY_CATALOG),
        equivalence_groups=len(group_legacy_needs(LEGACY_CATALOG)),
        evidence_requirements=len(graph.requirements),
        acquisition_targets=len(graph.targets),
        existing_direct_api_routes=count(AcquisitionMethod.EXISTING_DIRECT_API),
        new_direct_adapter_routes=count(AcquisitionMethod.NEW_DIRECT_ADAPTER),
        known_url_http_routes=count(AcquisitionMethod.KNOWN_URL_HTTP),
        web_search_discovery_routes=count(AcquisitionMethod.WEB_SEARCH_DISCOVERY),
        anthropic_web_fetch_routes=count(AcquisitionMethod.ANTHROPIC_WEB_FETCH),
        not_publicly_available_routes=count(AcquisitionMethod.NOT_PUBLICLY_AVAILABLE),
        manual_verification_required_routes=count(AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED),
    )
