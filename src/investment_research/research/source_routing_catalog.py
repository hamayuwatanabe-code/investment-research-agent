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
    ImplementationStatus,
    RequestCostClass,
    RequirementCriticality,
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

#: Phase 3A requirements 4/5/6: ``acquisition_executor.AcquisitionExecutor``
#: plus ``sec_acquisition_adapters.SecPrimaryDocumentAdapter``/
#: ``SecExhibitAdapter`` now exist as code and are proven, in this
#: repository's own test suite, against an injected FakeHttpClient --
#: genuinely OFFLINE_VERIFIED, never LIVE_VERIFIED (no real network call was
#: ever made). These adapter_id strings match exactly what those adapters
#: register under; steps below that use them are promoted from
#: ADAPTER_IMPLEMENTED/DECLARED to OFFLINE_VERIFIED accordingly -- an earned
#: promotion, not a blanket one: every OTHER step in this catalog (web
#: search, ClinicalTrials, PubMed, Form 4) is untouched by Phase 3A and stays
#: at its pre-existing, honest level.
_SEC_PRIMARY_DOCUMENT_ADAPTER_ID = "sec_primary_document_adapter"
_SEC_EXHIBIT_ADAPTER_ID = "sec_exhibit_enumeration"

#: REQUIRED-strength (never conditional) domains for the plain sec_chain/
#: sec_chain_with_exhibit archetypes -- capital-structure/regulatory issuer
#: disclosure is the company's own statutory obligation, not contingent on a
#: later materiality assessment. (fda_dual's REGULATOR-side sub-requirement
#: is handled separately, as CONDITIONAL_BLOCKING -- see ``_fda_dual``.)
_REQUIRED_DOMAINS = (ResearchDomain.REGULATORY, ResearchDomain.CAPITAL_STRUCTURE)


def _document_chain(
    tag: str,
    target_id: str,
    *,
    direct_method: AcquisitionMethod,
    direct_adapter: str,
    direct_authority: DocumentAuthority,
    direct_locate_completion: StepStatus = StepStatus.URL_RESOLVED,
    direct_implementation_status: ImplementationStatus = ImplementationStatus.ADAPTER_IMPLEMENTED,
    fetch_adapter_id: str = "http_client",
    fetch_implementation_status: ImplementationStatus = ImplementationStatus.PRIMITIVE_AVAILABLE,
    parse_adapter_id: str = "local_parser",
    parse_implementation_status: ImplementationStatus = ImplementationStatus.PRIMITIVE_AVAILABLE,
) -> tuple[list[AcquisitionStep], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """LOCATE(direct) altOR LOCATE(web) -> FETCH(http, required) -> PARSE(required).

    Reaching only the locate stage -- by either path -- never completes the
    target: FETCH and PARSE are both in ``required_step_ids``. The Direct
    locate's ``direct_implementation_status`` defaults ADAPTER_IMPLEMENTED
    (the genuinely-existing SEC EDGAR/ClinicalTrials collector code); callers
    whose adapter does not exist yet (SEC exhibit enumeration prior to this
    module wiring it, Form 4 XML, PubMed/Europe PMC) must pass DECLARED
    explicitly -- declaring the route is never itself a claim that it runs
    (Phase 3A requirement 1). The web-search locate is always DISABLED here:
    Phase 3A explicitly does not execute Web Search from this graph (Phase 3A
    requirement 7), regardless of what search code exists elsewhere in the
    application. ``fetch_implementation_status``/``parse_implementation_status``
    default to PRIMITIVE_AVAILABLE (a generic ``HttpClient``/local parser
    exists); callers whose FETCH/PARSE is backed by a genuinely offline-
    verified, executor-wired adapter (the SEC primary-document/exhibit
    adapters) pass the earned, higher status explicitly.
    """
    l1, l2, f, p = f"step_{tag}_L1", f"step_{tag}_L2", f"step_{tag}_F", f"step_{tag}_P"
    steps = [
        AcquisitionStep(
            step_id=l1, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=direct_method, adapter_id=direct_adapter,
            authority=direct_authority, completion_condition=direct_locate_completion,
            failure_policy=FailurePolicy.ALTERNATIVE, implementation_status=direct_implementation_status,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.RATE_LIMITED,
        ),
        AcquisitionStep(
            step_id=l2, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY, adapter_id="anthropic_web_search",
            completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
            implementation_status=ImplementationStatus.DISABLED,
            token_cost_class=TokenCostClass.HIGH, request_cost_class=RequestCostClass.PAID_LLM_CALL,
        ),
        AcquisitionStep(
            step_id=f, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=fetch_adapter_id,
            depends_on_step_ids=(l1, l2), completion_condition=StepStatus.BODY_FETCHED,
            implementation_status=fetch_implementation_status,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=p, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=parse_adapter_id,
            depends_on_step_ids=(f,), completion_condition=StepStatus.PARSED,
            implementation_status=parse_implementation_status,
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
    direct_implementation_status: ImplementationStatus = ImplementationStatus.ADAPTER_IMPLEMENTED,
) -> tuple[list[AcquisitionStep], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Direct structured fetch, OR (only if that fails) a full web-search
    locate -> fetch -> parse chain. Two alt-groups: an entry gate (which
    path to attempt) and a terminal gate (which path's own terminal step
    actually completed) -- a mid-chain step (e.g. a resolved URL with no
    fetch yet) satisfies neither. ``direct_implementation_status`` defaults
    ADAPTER_IMPLEMENTED (ClinicalTrialsCollector); callers using an
    unimplemented adapter (PubMed/Europe PMC) must pass DECLARED explicitly.
    The web-search fallback is always DISABLED for LOCATE (Phase 3A
    requirement 7) and PRIMITIVE_AVAILABLE for its generic FETCH/PARSE (a
    plain HttpClient/local parser exists, but nothing source-specific).
    """
    s1, sp = f"step_{tag}_S1", f"step_{tag}_SP"
    lw, fw, pw = f"step_{tag}_LW", f"step_{tag}_FW", f"step_{tag}_PW"
    steps = [
        AcquisitionStep(
            step_id=s1, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=direct_method, adapter_id=direct_adapter, authority=direct_authority,
            completion_condition=StepStatus.STRUCTURED_RECORD_RETRIEVED, failure_policy=FailurePolicy.ALTERNATIVE,
            implementation_status=direct_implementation_status,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=sp, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=direct_method, adapter_id="local_parser",
            depends_on_step_ids=(s1,), completion_condition=StepStatus.REQUIRED_FIELDS_PARSED,
            failure_policy=FailurePolicy.ALTERNATIVE, implementation_status=direct_implementation_status,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=lw, target_id=target_id, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.WEB_SEARCH_DISCOVERY, adapter_id="anthropic_web_search",
            completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.ALTERNATIVE,
            implementation_status=ImplementationStatus.DISABLED,
            token_cost_class=TokenCostClass.HIGH, request_cost_class=RequestCostClass.PAID_LLM_CALL,
        ),
        AcquisitionStep(
            step_id=fw, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="http_client",
            depends_on_step_ids=(lw,), completion_condition=StepStatus.BODY_FETCHED,
            failure_policy=FailurePolicy.ALTERNATIVE, implementation_status=ImplementationStatus.PRIMITIVE_AVAILABLE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=pw, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="local_parser",
            depends_on_step_ids=(fw,), completion_condition=StepStatus.PARSED,
            failure_policy=FailurePolicy.ALTERNATIVE, implementation_status=ImplementationStatus.PRIMITIVE_AVAILABLE,
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
            implementation_status=ImplementationStatus.DISABLED,
            token_cost_class=TokenCostClass.HIGH, request_cost_class=RequestCostClass.PAID_LLM_CALL,
        ),
        AcquisitionStep(
            step_id=fw, target_id=target_id, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="http_client",
            depends_on_step_ids=(lw,), completion_condition=StepStatus.BODY_FETCHED,
            implementation_status=ImplementationStatus.PRIMITIVE_AVAILABLE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
        AcquisitionStep(
            step_id=pw, target_id=target_id, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id="local_parser",
            depends_on_step_ids=(fw,), completion_condition=StepStatus.PARSED,
            implementation_status=ImplementationStatus.PRIMITIVE_AVAILABLE,
            token_cost_class=TokenCostClass.ZERO, request_cost_class=RequestCostClass.FREE,
        ),
    ]
    return steps, (lw, fw, pw), ()


def _sec_chain(index, need_ids, domain, subject_scope, key) -> SourceRoutingGraph:
    tag = f"{index:03d}a"
    req_id, target_id = f"req_{tag}", f"target_{tag}"
    steps, required, alt_groups = _document_chain(
        tag, target_id, direct_method=AcquisitionMethod.EXISTING_DIRECT_API,
        direct_adapter=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, direct_authority=DocumentAuthority.STATUTORY_FILING,
        direct_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        fetch_adapter_id=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, fetch_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        parse_adapter_id=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, parse_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    requirement = EvidenceRequirement(
        requirement_id=req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"issuer disclosed in a filing: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        criticality=RequirementCriticality.REQUIRED if domain in _REQUIRED_DOMAINS else RequirementCriticality.BEST_EFFORT,
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
        direct_adapter=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, direct_authority=DocumentAuthority.STATUTORY_FILING,
        direct_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        fetch_adapter_id=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, fetch_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        parse_adapter_id=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, parse_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    body_requirement = EvidenceRequirement(
        requirement_id=body_req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"issuer disclosed in a filing (body): {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        criticality=RequirementCriticality.REQUIRED if domain in _REQUIRED_DOMAINS else RequirementCriticality.BEST_EFFORT,
    )
    body_target = AcquisitionTarget(
        target_id=body_target_id, target_kind=TargetKind.SEC_PRIMARY_DOCUMENT,
        required_step_ids=body_required, alternative_step_groups=body_alt, serves_requirement_ids=(body_req_id,),
    )

    exhibit_tag = f"{index:03d}b"
    exhibit_req_id, exhibit_target_id = f"req_{exhibit_tag}", f"target_{exhibit_tag}"
    exhibit_steps, exhibit_required, exhibit_alt = _document_chain(
        exhibit_tag, exhibit_target_id, direct_method=AcquisitionMethod.NEW_DIRECT_ADAPTER,
        direct_adapter=_SEC_EXHIBIT_ADAPTER_ID, direct_authority=DocumentAuthority.STATUTORY_FILING,
        direct_locate_completion=StepStatus.URL_RESOLVED,
        direct_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        fetch_adapter_id=_SEC_EXHIBIT_ADAPTER_ID, fetch_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        parse_adapter_id=_SEC_EXHIBIT_ADAPTER_ID, parse_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
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
        direct_adapter=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, direct_authority=DocumentAuthority.STATUTORY_FILING,
        direct_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        fetch_adapter_id=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, fetch_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
        parse_adapter_id=_SEC_PRIMARY_DOCUMENT_ADAPTER_ID, parse_implementation_status=ImplementationStatus.OFFLINE_VERIFIED,
    )
    disclosure_requirement = EvidenceRequirement(
        requirement_id=disclosure_req_id, serves_legacy_need_ids=need_ids, subject_scope=subject_scope, domain=domain,
        claim_scope=f"issuer disclosed in a filing: {key}",
        required_authorities=(DocumentAuthority.STATUTORY_FILING, DocumentAuthority.COMPANY_IR),
        criticality=RequirementCriticality.REQUIRED,
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
        # Phase 3A requirement 3: the regulator's own non-public correspondence
        # is never uniformly blocking merely because it is non-public. Whether
        # it actually blocks depends on program-currency, materiality, and
        # whether issuer disclosure/an alternative source already resolves it
        # -- facts this static catalog does not have at program_scope=UNKNOWN.
        # Reported as CONDITIONAL_BLOCKING (-> PENDING_MATERIALITY_ASSESSMENT),
        # never defaulted to blocking or to non-blocking.
        criticality=RequirementCriticality.CONDITIONAL_BLOCKING,
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
        direct_implementation_status=ImplementationStatus.DECLARED,
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
        direct_implementation_status=ImplementationStatus.DECLARED,
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
    #: Per-``ImplementationStatus`` ladder rung (Phase 3A requirement 1) --
    #: replaces Phase 2.7's binary implemented_steps/not_implemented_steps,
    #: which could not distinguish "a generic primitive exists" from "an
    #: adapter exists" from "an executor could run it today".
    declared_steps: int
    primitive_available_steps: int
    adapter_implemented_steps: int
    executor_wired_steps: int
    pipeline_wired_steps: int
    offline_verified_steps: int
    live_verified_steps: int
    disabled_steps: int
    #: Requirement-level criticality split (Phase 3A requirement 3).
    required_requirements: int
    conditional_blocking_requirements: int
    best_effort_requirements: int


def routing_coverage_counts(graph: SourceRoutingGraph | None = None) -> RoutingCoverageCounts:
    graph = graph or build_source_routing_graph()

    def by_method(method: AcquisitionMethod) -> int:
        return sum(1 for s in graph.steps if s.acquisition_method is method)

    def by_kind(kind: StepKind) -> int:
        return sum(1 for s in graph.steps if s.step_kind is kind)

    def by_implementation(status: ImplementationStatus) -> int:
        return sum(1 for s in graph.steps if s.implementation_status is status)

    def by_criticality(criticality: RequirementCriticality) -> int:
        return sum(1 for r in graph.requirements if r.criticality is criticality)

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
        declared_steps=by_implementation(ImplementationStatus.DECLARED),
        primitive_available_steps=by_implementation(ImplementationStatus.PRIMITIVE_AVAILABLE),
        adapter_implemented_steps=by_implementation(ImplementationStatus.ADAPTER_IMPLEMENTED),
        executor_wired_steps=by_implementation(ImplementationStatus.EXECUTOR_WIRED),
        pipeline_wired_steps=by_implementation(ImplementationStatus.PIPELINE_WIRED),
        offline_verified_steps=by_implementation(ImplementationStatus.OFFLINE_VERIFIED),
        live_verified_steps=by_implementation(ImplementationStatus.LIVE_VERIFIED),
        disabled_steps=by_implementation(ImplementationStatus.DISABLED),
        required_requirements=by_criticality(RequirementCriticality.REQUIRED),
        conditional_blocking_requirements=by_criticality(RequirementCriticality.CONDITIONAL_BLOCKING),
        best_effort_requirements=by_criticality(RequirementCriticality.BEST_EFFORT),
    )
