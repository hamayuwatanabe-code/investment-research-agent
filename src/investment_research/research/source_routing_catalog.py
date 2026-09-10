"""The Phase 2.6 Source Routing Graph: every real 31 exact-equivalence group
classified into a code-grounded step-DAG archetype.

31 is the group count only -- see ``routing_coverage_counts()`` for the
separately-measured requirement/target/step counts.

Every archetype expresses acquiring a document as LOCATE -> FETCH -> PARSE
(or, for a structured API, FETCH(structured) -> PARSE(fields)), never as a
single flat "success" state -- see ``source_routing.py``'s module docstring
for why Phase 2.5's flat Route model let a NORMAL scenario stop at metadata/
URL discovery for an SEC target without ever fetching a body.

Archetype grounding (unchanged reasoning from Phase 2.5, restructured into
step chains):

* ``sec_chain`` / ``sec_chain_with_exhibit`` -- routine 8-K/10-K/10-Q/424B/
  DEF 14A disclosure. Direct locate (``SecEdgarCollector``'s free submissions
  fetch, resolving a URL when ``primaryDocument`` is populated) with a
  web-search locate as its only alternative entry point; either way, a
  ``KNOWN_URL_HTTP`` body fetch and a parse step are REQUIRED for the target
  to complete -- reaching only the locate stage, by either path, never
  completes anything.
* ``fda_dual`` -- issuer disclosure (a ``sec_chain``) is one
  ``EvidenceRequirement``; the regulator's own position is a SEPARATE
  requirement whose only target has a single step, terminal at
  NOT_PUBLICLY_AVAILABLE, and NEVER reachable from web search (grounded in
  ``collectors/fda.py``'s own documented ``NON_API_REGULATORY_QUESTIONS``).
* ``clinicaltrials_web`` -- ``ClinicalTrialsCollector``'s free structured
  ``status``/``primary_completion`` fields, with a full web-search-locate ->
  fetch -> parse fallback chain if the structured attempt does not resolve
  the claim. Registry-status claims only (failed trial, catalyst timing).
* ``literature_web`` -- peer-reviewed efficacy/safety/reproducibility claims
  route through Europe PMC/PubMed (``NEW_DIRECT_ADAPTER``, unimplemented but
  free) first, with the same web-search fallback chain. Kept as a
  COMPLETELY SEPARATE requirement/target from ``clinicaltrials_web`` --
  Phase 2.5's single-target simplification (ClinicalTrials/PubMed/Web on one
  target) is retired: a ClinicalTrials success never completes a
  literature-grounded requirement, and vice versa.
* ``form4`` -- insider selling is exactly what Form 4 reports
  (``NEW_DIRECT_ADAPTER``, unimplemented), same fallback shape.
* ``web_only`` -- criticism/short thesis/"is a competitor better" have no
  registry, filing, or structured API that would ever contain them. No
  Direct alternative exists -- but a web-search locate alone STILL never
  completes the target; fetch and parse of the discovered URL remain
  required (requirement 6: a search snippet is never Evidence).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.enums import DocumentAuthority, ResearchDomain
from .acquisition_planning import AcquisitionMethod
from .legacy_catalog import LEGACY_CATALOG, group_legacy_needs
from .source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    RequestCostClass,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetKind,
    TokenCostClass,
)

_ARCHETYPE_BY_KEY: dict[str, str] = {
    "company fda concern": "fda_dual",
    "company regulatory risk": "fda_dual",
    "program endpoint concern": "fda_dual",
    "company endpoint": "fda_dual",
    "company clinical hold": "fda_dual",
    "company complete response letter": "fda_dual",
    "company delayed catalyst": "clinicaltrials_web",
    "company failed trial": "clinicaltrials_web",
    "company upcoming catalyst readout": "clinicaltrials_web",
    "company failed": "clinicaltrials_web",
    "program safety concern": "literature_web",
    "company clinical data results": "literature_web",
    "program mechanism efficacy": "literature_web",
    "company insider selling": "form4",
    "company criticism": "web_only",
    "company short thesis": "web_only",
    "program competitor superiority": "web_only",
    "company partnership agreement": "sec_chain_with_exhibit",
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

_BLOCKING_DOMAINS = (ResearchDomain.REGULATORY, ResearchDomain.CAPITAL_STRUCTURE)


def _document_chain(
    tag: str,
    target_id: str,
    *,
    direct_method: AcquisitionMethod,
    direct_adapter: str,
    direct_authority: DocumentAuthority,
    direct_locate_completion: StepStatus = StepStatus.URL_RESOLVED,
) -> tuple[list[AcquisitionStep], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """LOCATE(direct) altOR LOCATE(web) -> FETCH(http, required) -> PARSE(required).

    Reaching only the locate stage -- by either path -- never completes the
    target: FETCH and PARSE are both in ``required_step_ids``.
    """
    l1, l2, f, p = f"step_{tag}_L1", f"step_{tag}_L2", f"step_{tag}_F", f"step_{tag}_P"
    steps = [
        AcquisitionStep(
            step_id=l1, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=direct_method, adapter_id=direct_adapter,
            authority=direct_authority, completion_condition=direct_locate_completion,
            failure_policy=FailurePolicy.ALTERNATIVE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.RATE_LIMITED,
        ),
        AcquisitionStep(
            step_id=l2, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY, adapter_id="anthropic_web_search",
            completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
            token_cost_class=TokenCostClass.HIGH, request_cost_class=RequestCostClass.PAID_LLM_CALL,
        ),
        AcquisitionStep(
            step_id=f, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="http_client",
            depends_on_step_ids=(l1, l2), completion_condition=StepStatus.BODY_FETCHED,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=p, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="local_parser",
            depends_on_step_ids=(f,), completion_condition=StepStatus.PARSED,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
    ]
    return steps, (f, p), ((l1, l2),)


def _structured_or_web_chain(
    tag: str,
    target_id: str,
    *,
    direct_method: AcquisitionMethod,
    direct_adapter: str,
    direct_authority: DocumentAuthority,
) -> tuple[list[AcquisitionStep], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Direct structured fetch, OR (only if that fails) a full web-search
    locate -> fetch -> parse chain. Two alt-groups: an entry gate (which
    path to attempt) and a terminal gate (which path's own terminal step
    actually completed) -- a mid-chain step (e.g. a resolved URL with no
    fetch yet) satisfies neither.
    """
    s1, sp = f"step_{tag}_S1", f"step_{tag}_SP"
    lw, fw, pw = f"step_{tag}_LW", f"step_{tag}_FW", f"step_{tag}_PW"
    steps = [
        AcquisitionStep(
            step_id=s1, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=direct_method, adapter_id=direct_adapter, authority=direct_authority,
            completion_condition=StepStatus.STRUCTURED_RECORD_RETRIEVED, failure_policy=FailurePolicy.ALTERNATIVE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=sp, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=direct_method, adapter_id="local_parser",
            depends_on_step_ids=(s1,), completion_condition=StepStatus.REQUIRED_FIELDS_PARSED,
            failure_policy=FailurePolicy.ALTERNATIVE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=lw, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY, adapter_id="anthropic_web_search",
            completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
            token_cost_class=TokenCostClass.HIGH, request_cost_class=RequestCostClass.PAID_LLM_CALL,
        ),
        AcquisitionStep(
            step_id=fw, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="http_client",
            depends_on_step_ids=(lw,), completion_condition=StepStatus.BODY_FETCHED,
            failure_policy=FailurePolicy.ALTERNATIVE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=pw, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="local_parser",
            depends_on_step_ids=(fw,), completion_condition=StepStatus.PARSED,
            failure_policy=FailurePolicy.ALTERNATIVE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
    ]
    return steps, (), ((s1, lw), (sp, pw))


def _not_publicly_available_target(tag: str, target_id: str) -> tuple[list[AcquisitionStep], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    step_id = f"step_{tag}_NPA"
    step = AcquisitionStep(
        step_id=step_id, target_id=target_id, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NOT_PUBLICLY_AVAILABLE, adapter_id="none",
        authority=DocumentAuthority.REGULATOR, completion_condition=StepStatus.NOT_PUBLICLY_AVAILABLE,
        token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
    )
    return [step], (step_id,), ()


def _web_only_chain(tag: str, target_id: str) -> tuple[list[AcquisitionStep], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """No Direct alternative exists. Still a REQUIRED three-step chain --
    URL_RESOLVED alone (a search snippet) is never Evidence (requirement 6).
    """
    lw, fw, pw = f"step_{tag}_LW", f"step_{tag}_FW", f"step_{tag}_PW"
    steps = [
        AcquisitionStep(
            step_id=lw, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY, adapter_id="anthropic_web_search",
            completion_condition=StepStatus.URL_RESOLVED,
            token_cost_class=TokenCostClass.HIGH, request_cost_class=RequestCostClass.PAID_LLM_CALL,
        ),
        AcquisitionStep(
            step_id=fw, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="http_client",
            depends_on_step_ids=(lw,), completion_condition=StepStatus.BODY_FETCHED,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=pw, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="local_parser",
            depends_on_step_ids=(fw,), completion_condition=StepStatus.PARSED,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
    ]
    return steps, (lw, fw, pw), ()


def _sec_chain(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id, target_id = f"req_{tag}", f"target_{tag}"
    steps, required, alt_groups = _document_chain(
        tag, target_id, direct_method=AcquisitionMethod.EXISTING_DIRECT_API,
        direct_adapter="sec_edgar_submissions", direct_authority=DocumentAuthority.STATUTORY_FILING,
    )
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"issuer disclosed in a filing: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        blocking_if_unresolved=domain in _BLOCKING_DOMAINS,
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=required, alternative_step_groups=alt_groups, serves_requirement_ids=(req_id,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))


def _sec_chain_with_exhibit(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    body_tag = f"{index:03d}a"
    body_req_id, body_target_id = f"req_{body_tag}", f"target_{body_tag}"
    body_steps, body_required, body_alt = _document_chain(
        body_tag, body_target_id, direct_method=AcquisitionMethod.EXISTING_DIRECT_API,
        direct_adapter="sec_edgar_submissions", direct_authority=DocumentAuthority.STATUTORY_FILING,
    )
    body_requirement = EvidenceRequirement(
        requirement_id=body_req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"issuer disclosed in a filing (body): {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        blocking_if_unresolved=domain in _BLOCKING_DOMAINS,
    )
    body_target = AcquisitionTarget(
        target_id=body_target_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=body_required, alternative_step_groups=body_alt, serves_requirement_ids=(body_req_id,),
    )

    exhibit_tag = f"{index:03d}b"
    exhibit_req_id, exhibit_target_id = f"req_{exhibit_tag}", f"target_{exhibit_tag}"
    exhibit_steps, exhibit_required, exhibit_alt = _document_chain(
        exhibit_tag, exhibit_target_id, direct_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        direct_adapter="sec_exhibit_enumeration", direct_authority=DocumentAuthority.STATUTORY_FILING,
        direct_locate_completion=StepStatus.URL_RESOLVED,
    )
    exhibit_requirement = EvidenceRequirement(
        requirement_id=exhibit_req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"the underlying agreement text itself (exhibit): {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING,),
    )
    exhibit_target = AcquisitionTarget(
        target_id=exhibit_target_id, target_kind=TargetKind.SEC_EXHIBIT,
        required_step_ids=exhibit_required, alternative_step_groups=exhibit_alt,
        serves_requirement_ids=(exhibit_req_id,),
    )

    return SourceRoutingGraph(
        requirements=(body_requirement, exhibit_requirement),
        targets=(body_target, exhibit_target),
        steps=tuple(body_steps + exhibit_steps),
    )


def _fda_dual(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    disclosure_tag = f"{index:03d}a"
    disclosure_req_id, disclosure_target_id = f"req_{disclosure_tag}", f"target_{disclosure_tag}"
    disclosure_steps, disclosure_required, disclosure_alt = _document_chain(
        disclosure_tag, disclosure_target_id, direct_method=AcquisitionMethod.EXISTING_DIRECT_API,
        direct_adapter="sec_edgar_submissions", direct_authority=DocumentAuthority.STATUTORY_FILING,
    )
    disclosure_requirement = EvidenceRequirement(
        requirement_id=disclosure_req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"issuer disclosed in a filing: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        blocking_if_unresolved=True,
    )
    disclosure_target = AcquisitionTarget(
        target_id=disclosure_target_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=disclosure_required, alternative_step_groups=disclosure_alt,
        serves_requirement_ids=(disclosure_req_id,),
    )

    regulator_tag = f"{index:03d}b"
    regulator_req_id, regulator_target_id = f"req_{regulator_tag}", f"target_{regulator_tag}"
    regulator_steps, regulator_required, regulator_alt = _not_publicly_available_target(regulator_tag, regulator_target_id)
    regulator_requirement = EvidenceRequirement(
        requirement_id=regulator_req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"the regulator itself stated: {key}",
        required_authorities=(DocumentAuthority.REGULATOR,), independence_requirement=True,
        blocking_if_unresolved=True,
    )
    regulator_target = AcquisitionTarget(
        target_id=regulator_target_id, target_kind=TargetKind.FDA_NONPUBLIC_CORRESPONDENCE,
        required_step_ids=regulator_required, alternative_step_groups=regulator_alt,
        serves_requirement_ids=(regulator_req_id,),
    )

    return SourceRoutingGraph(
        requirements=(disclosure_requirement, regulator_requirement),
        targets=(disclosure_target, regulator_target),
        steps=tuple(disclosure_steps + regulator_steps),
    )


def _clinicaltrials_web(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id, target_id = f"req_{tag}", f"target_{tag}"
    steps, required, alt_groups = _structured_or_web_chain(
        tag, target_id, direct_method=AcquisitionMethod.EXISTING_DIRECT_API,
        direct_adapter="clinicaltrials_api", direct_authority=DocumentAuthority.REGISTRY,
    )
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"trial registry status/date signal for: {key}",
        required_authorities=(DocumentAuthority.REGISTRY,),
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.CLINICALTRIALS_RECORD,
        required_step_ids=required, alternative_step_groups=alt_groups, serves_requirement_ids=(req_id,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))


def _literature_web(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    """A COMPLETELY SEPARATE requirement/target family from
    ``clinicaltrials_web`` -- never the same target, never satisfied by a
    ClinicalTrials.gov success (Phase 2.6 requirement 4)."""
    tag = f"{index:03d}a"
    req_id, target_id = f"req_{tag}", f"target_{tag}"
    steps, required, alt_groups = _structured_or_web_chain(
        tag, target_id, direct_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        direct_adapter="pubmed_europepmc", direct_authority=DocumentAuthority.INDEPENDENT,
    )
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"peer-reviewed literature signal for: {key}",
        required_authorities=(DocumentAuthority.INDEPENDENT,),
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.LITERATURE_ARTICLE,
        required_step_ids=required, alternative_step_groups=alt_groups, serves_requirement_ids=(req_id,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))


def _form4(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id, target_id = f"req_{tag}", f"target_{tag}"
    steps, required, alt_groups = _document_chain(
        tag, target_id, direct_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        direct_adapter="form4_xml_parser", direct_authority=DocumentAuthority.STATUTORY_FILING,
    )
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"insider transaction filing for: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING,),
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.FORM4_FILING,
        required_step_ids=required, alternative_step_groups=alt_groups, serves_requirement_ids=(req_id,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))


def _web_only(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id, target_id = f"req_{tag}", f"target_{tag}"
    steps, required, alt_groups = _web_only_chain(tag, target_id)
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"third-party sentiment/analysis: {key}",
        required_authorities=(DocumentAuthority.INDEPENDENT,),
    )
    target = AcquisitionTarget(
        target_id=target_id, target_kind=TargetKind.GENERIC_WEB_DOCUMENT,
        required_step_ids=required, alternative_step_groups=alt_groups, serves_requirement_ids=(req_id,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=tuple(steps))


_BUILDERS = {
    "sec_chain": _sec_chain,
    "sec_chain_with_exhibit": _sec_chain_with_exhibit,
    "fda_dual": _fda_dual,
    "clinicaltrials_web": _clinicaltrials_web,
    "literature_web": _literature_web,
    "form4": _form4,
    "web_only": _web_only,
}


def build_source_routing_graph() -> SourceRoutingGraph:
    groups = group_legacy_needs(LEGACY_CATALOG)
    requirements: list[EvidenceRequirement] = []
    targets: list[AcquisitionTarget] = []
    steps: list[AcquisitionStep] = []

    for index, ((equivalence_key, domain, subject_scope), need_ids) in enumerate(groups.items()):
        archetype = _ARCHETYPE_BY_KEY.get(equivalence_key)
        if archetype is None:
            raise ValueError(
                f"group {equivalence_key!r} (domain={domain}) has no routing archetype -- "
                "every one of the 31 real groups must be individually classified, never defaulted"
            )
        sub_graph = _BUILDERS[archetype](index, need_ids, domain, subject_scope, equivalence_key)
        requirements.extend(sub_graph.requirements)
        targets.extend(sub_graph.targets)
        steps.extend(sub_graph.steps)

    return SourceRoutingGraph(requirements=tuple(requirements), targets=tuple(targets), steps=tuple(steps))


@dataclass(frozen=True)
class RoutingCoverageCounts:
    legacy_needs: int
    equivalence_groups: int
    evidence_requirements: int
    acquisition_targets: int
    locate_steps: int
    fetch_steps: int
    parse_steps: int
    existing_direct_api_steps: int
    new_direct_adapter_steps: int
    known_url_http_steps: int
    web_search_discovery_steps: int
    anthropic_web_fetch_steps: int
    not_publicly_available_steps: int
    manual_verification_required_steps: int


def routing_coverage_counts(graph: SourceRoutingGraph | None = None) -> RoutingCoverageCounts:
    graph = graph or build_source_routing_graph()

    def by_method(method: AcquisitionMethod) -> int:
        return sum(1 for s in graph.steps if s.acquisition_method is method)

    def by_kind(kind: StepKind) -> int:
        return sum(1 for s in graph.steps if s.step_kind is kind)

    return RoutingCoverageCounts(
        legacy_needs=len(LEGACY_CATALOG),
        equivalence_groups=len(group_legacy_needs(LEGACY_CATALOG)),
        evidence_requirements=len(graph.requirements),
        acquisition_targets=len(graph.targets),
        locate_steps=by_kind(StepKind.LOCATE),
        fetch_steps=by_kind(StepKind.FETCH),
        parse_steps=by_kind(StepKind.PARSE),
        existing_direct_api_steps=by_method(AcquisitionMethod.EXISTING_DIRECT_API),
        new_direct_adapter_steps=by_method(AcquisitionMethod.NEW_DIRECT_ADAPTER),
        known_url_http_steps=by_method(AcquisitionMethod.KNOWN_URL_HTTP),
        web_search_discovery_steps=by_method(AcquisitionMethod.WEB_SEARCH_DISCOVERY),
        anthropic_web_fetch_steps=by_method(AcquisitionMethod.ANTHROPIC_WEB_FETCH),
        not_publicly_available_steps=by_method(AcquisitionMethod.NOT_PUBLICLY_AVAILABLE),
        manual_verification_required_steps=by_method(AcquisitionMethod.MANUAL_VERIFICATION_REQUIRED),
    )
