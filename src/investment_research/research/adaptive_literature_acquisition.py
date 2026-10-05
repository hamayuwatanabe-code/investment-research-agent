"""Phase 4.3H: an offline bridge from Phase 4.3G's pure, side-effect-free
``AdaptiveAcquisitionPlan`` to the existing Literature Document-First
acquisition path::

    AdaptiveAcquisitionPlan
      -> validate_literature_pipeline_request()
      -> run_literature_pipeline_acquisition()
      -> AcquisitionExecutor -> PubMedLiteratureAdapter -> DocumentStore
      -> Literature Evidence/Chunk Projection
      -> LiteraturePipelineBundle

This module adds no new PubMed/Europe PMC parsing, no new request-budget
math, and no new credential handling -- ``validate_literature_pipeline_
request``/``run_literature_pipeline_acquisition`` (``research/
literature_pipeline_integration.py``) are reused exactly as Phase 4.2A
built them. It decides ONLY whether a Phase 4.3G plan is a genuine,
re-verified ``READY`` worth turning into one acquisition request, and
carries the resulting ``LiteraturePipelineBundle`` back out untouched.

**Not connected to ``orchestrator/pipeline.py``, ``cli.py``, or the Action
Gate.** Phase 4.3H verifies this bridge offline, against an injected Fake
HTTP transport, only -- see ``tests/unit/
test_adaptive_literature_acquisition.py``'s own isolation tests. Wiring
this into ``Pipeline.run()`` (the FactCollector conversion, a second full
Evidence Integrity pass over the new facts, and a resume-snapshot design
for this step) is explicitly future work, not attempted here.

Correction 1's final priority contract: ``plan.status`` is judged FIRST,
unconditionally, before ``http_client``/``env`` are even inspected. Any
of the five non-``READY`` statuses (``NO_ACTION``/``UNRESOLVED``/
``CONFLICTED``/``SKIPPED_EXPLICIT_OVERRIDE``/``REFUSED``) is always
``SKIPPED`` -- regardless of whether ``http_client``/``env`` are
present, absent, or malformed -- with zero calls into
``validate_literature_pipeline_request``/``run_literature_pipeline_
acquisition`` and zero HTTP. Only once a plan has passed that gate (i.e.
only for a CLAIMED ``READY`` plan) does the required-injection guard
run: ``http_client``/``env`` are REQUIRED, explicitly-injected
parameters with no default value -- a caller must always pass both,
even if passing the literal ``None``. Passing ``None`` for either is
never treated as "use the real HTTP client" / "use the real process
environment": it is a ``REFUSED`` outcome for that ``READY`` plan,
checked before ``validate_literature_pipeline_request`` is called and
before any of this module's other ``READY`` re-verification checks.
This is a STRICTER contract than ``run_literature_pipeline_acquisition``'s
own optional ``http_client``/``env`` parameters (which DO fall back to a
real, network-capable ``AllowlistedHttpClient``/the real environment when
omitted) -- this bridge exists specifically so that fallback can never be
reached by accident from Phase 4.3G's automatic planning path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..schemas.enums import StrEnum
from ..scoring.adaptive_acquisition_plan import (
    READY_MAX_REQUESTS,
    REFERENCE_MODE_NCT_ID,
    AcquisitionPlanStatus,
    AdaptiveAcquisitionPlan,
)
from ..scoring.identifier_validation import validate_strict_nct_id
from .literature_pipeline_integration import (
    DEFAULT_MAX_ARTICLES,
    DEFAULT_MAX_FULLTEXT_FETCHES,
    LiteraturePipelineBundle,
    LiteraturePipelineRequestError,
    run_literature_pipeline_acquisition,
    validate_literature_pipeline_request,
)


class LiteratureHttpTransport(Protocol):
    """The minimal shape ``run_literature_pipeline_acquisition`` actually
    calls on its own ``http_client`` argument -- satisfied structurally by
    both ``research.sec_live_smoke.AllowlistedHttpClient`` (the real,
    network-capable transport) and ``tests.unit.
    _literature_fixture_support.FakeHttpClient``/``RaisingHttpClient``
    (this phase's own offline test doubles), without this module naming
    either concrete class or falling back to ``Any``. This module never
    calls ``.get()`` itself -- the injected transport is only ever handed
    through, unopened, to ``run_literature_pipeline_acquisition``; this
    Protocol exists so a wrongly-shaped argument is still a real,
    checked, static-typing error at every call site, not a silently
    accepted ``Any``.
    """

    def get(self, url: str, **kwargs: object) -> object: ...


class AdaptiveExecutionStatus(StrEnum):
    """A closed set of outcomes for one ``execute_adaptive_literature_plan``
    call -- never a bare bool, mirroring ``AcquisitionPlanStatus``'s own
    convention."""

    #: The plan itself was not ``READY`` (``NO_ACTION``/``UNRESOLVED``/
    #: ``CONFLICTED``/``SKIPPED_EXPLICIT_OVERRIDE``/``REFUSED``) -- nothing
    #: to acquire, and this bridge never second-guesses the plan's own
    #: status. Judged FIRST, before ``http_client``/``env`` are even
    #: inspected: a non-``READY`` plan is ``SKIPPED`` regardless of
    #: whether those were injected, missing, or malformed. Zero HTTP
    #: requests.
    SKIPPED = "SKIPPED"
    #: Reached only for a CLAIMED ``READY`` plan. Either the required
    #: ``http_client``/``env`` injection was missing (``None`` -- checked
    #: first, within this ``READY``-only path), OR the plan failed this
    #: module's own independent re-verification (see ``execute_
    #: adaptive_literature_plan``'s docstring), OR ``validate_literature_
    #: pipeline_request`` itself rejected the derived request. Zero HTTP
    #: requests in every case.
    REFUSED = "REFUSED"
    #: The acquisition ran and ``LiteraturePipelineBundle.coverage_complete``
    #: is ``True``.
    COMPLETE = "COMPLETE"
    #: The acquisition ran (not refused) but
    #: ``LiteraturePipelineBundle.coverage_complete`` is ``False`` -- a
    #: provider failure, a partial result, or any other incompleteness
    #: ``run_literature_pipeline_acquisition`` itself already recorded as
    #: a structured field, never a crash.
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class AdaptiveLiteratureExecution:
    """The result of one ``execute_adaptive_literature_plan`` call --
    frozen, mirroring every other Phase 4.3G/4.3H result type. ``bundle``
    is the SAME ``LiteraturePipelineBundle`` ``run_literature_pipeline_
    acquisition`` returned, never rebuilt, re-copied, or stripped by this
    module -- including a ``refused`` bundle, which is kept as-is (never
    discarded) so a caller can still read its own ``refused_reason``/
    ``unresolved_reasons``. ``None`` only for ``SKIPPED``/a pre-acquisition
    ``REFUSED`` (never attempted). ``rationale`` is always short and safe:
    never a raw unvalidated input value, a credential, or a wire URL.
    """

    status: AdaptiveExecutionStatus
    plan: AdaptiveAcquisitionPlan
    bundle: LiteraturePipelineBundle | None
    rationale: str = ""


def _refused(plan: AdaptiveAcquisitionPlan, rationale: str) -> AdaptiveLiteratureExecution:
    return AdaptiveLiteratureExecution(
        status=AdaptiveExecutionStatus.REFUSED, plan=plan, bundle=None, rationale=rationale,
    )


def execute_adaptive_literature_plan(
    plan: AdaptiveAcquisitionPlan,
    *,
    ticker: str,
    http_client: LiteratureHttpTransport | None,
    env: Mapping[str, str] | None,
) -> AdaptiveLiteratureExecution:
    """Re-verify a ``READY`` plan, then (only if every check passes) run
    ONE Literature Document-First acquisition against it.

    Priority order (Correction 1's final contract):

    1. ``plan.status`` is judged FIRST, unconditionally, before
       ``http_client``/``env`` are even inspected. Non-``READY`` plans
       (``NO_ACTION``/``UNRESOLVED``/``CONFLICTED``/
       ``SKIPPED_EXPLICIT_OVERRIDE``/``REFUSED``) are always ``SKIPPED``
       -- this bridge never second-guesses what Phase 4.3G's own
       ``build_adaptive_acquisition_plan`` already decided, and it makes
       no difference here whether ``http_client``/``env`` were injected,
       ``None``, or malformed: a non-``READY`` plan never reaches that
       check at all.
    2. Only for a CLAIMED ``READY`` plan does the required-injection
       guard run: ``http_client``/``env`` being ``None`` is ``REFUSED``
       -- this bridge never falls back to a real, network-capable
       transport or the real process environment.
    3. A ``READY`` plan that passed the injection guard is independently
       re-verified against every one of the following before any
       request is built; failing ANY of them is ``REFUSED``, never a
       guess or a silent correction:

       a. ``plan.status is AcquisitionPlanStatus.READY``.
       b. ``plan.requires_external_communication is True``.
       c. ``plan.reference_mode == REFERENCE_MODE_NCT_ID``.
       d. ``validate_strict_nct_id(plan.nct_id)`` succeeds.
       e. That canonical NCT id equals ``plan.nct_id`` exactly (already
          canonical -- never silently re-canonicalized here).
       f. ``plan.pmid`` is ``""`` or ``"UNKNOWN"`` (never a specific
          pmid -- Phase 4.3G's own plan never populates one for
          ``READY``).
       g. ``plan.max_requests`` is a plain ``int`` (not ``bool``) equal
          to ``READY_MAX_REQUESTS`` (6).

    Only once every check above passes does this call
    ``validate_literature_pipeline_request`` -- always with
    ``enabled=True, pmid=None, nct_id=<the re-verified canonical NCT id>,
    live=True, use_fixtures=False, use_corpus=False,
    max_articles=DEFAULT_MAX_ARTICLES (3),
    max_fulltext_fetches=DEFAULT_MAX_FULLTEXT_FETCHES (1)``, and a COPY of
    the injected ``env`` -- never a ``LiteraturePipelineRequest``
    constructed by hand, which would bypass that validator's own input
    rules. ``live=True`` here satisfies that validator's existing request
    contract; it is not this module granting itself real-network
    permission -- whether any request actually reaches a network depends
    entirely on which transport was injected, and every test for this
    module injects an offline Fake one. A ``LiteraturePipelineRequestError``
    from that call is ``REFUSED``, with zero HTTP requests made (the
    validator itself guarantees that: every rejection fires before any
    network-capable object exists).

    Only once validation succeeds does this call ``run_literature_
    pipeline_acquisition`` -- with the SAME injected ``http_client`` and
    the SAME ``env`` copy, explicitly. The returned bundle (including its
    ``collection_result``/``chunks``/counts/``unresolved_reasons``) is
    never modified, copied piecemeal, or rebuilt -- only inspected, to
    choose this call's own ``AdaptiveExecutionStatus``:

    * ``bundle.refused`` is ``True`` -- ``REFUSED``, the bundle is KEPT
      (never discarded) so its own ``refused_reason`` stays inspectable.
    * ``bundle.coverage_complete`` is ``True`` -- ``COMPLETE``.
    * otherwise -- ``INCOMPLETE``.
    """
    if plan.status is not AcquisitionPlanStatus.READY:
        # Judged FIRST, unconditionally -- before http_client/env are
        # even inspected. A non-READY plan is SKIPPED regardless of
        # whether those were injected, None, or malformed.
        return AdaptiveLiteratureExecution(
            status=AdaptiveExecutionStatus.SKIPPED,
            plan=plan,
            bundle=None,
            rationale=f"plan status is {plan.status}, not READY -- nothing to acquire",
        )

    if http_client is None:
        return _refused(
            plan, "http_client was not injected; this bridge never falls back to a real transport"
        )
    if env is None:
        return _refused(
            plan, "env was not injected; this bridge never falls back to the real process environment"
        )

    if plan.requires_external_communication is not True:
        return _refused(plan, "READY plan does not have requires_external_communication=True")
    if plan.reference_mode != REFERENCE_MODE_NCT_ID:
        return _refused(plan, "READY plan's reference_mode is not NCT_ID")
    canonical_nct = validate_strict_nct_id(plan.nct_id)
    if canonical_nct is None:
        return _refused(plan, "READY plan's nct_id failed independent strict-NCT re-validation")
    if canonical_nct != plan.nct_id:
        return _refused(plan, "READY plan's nct_id is not already in canonical form")
    if plan.pmid not in ("", "UNKNOWN"):
        return _refused(plan, "READY plan unexpectedly carries a non-empty pmid")
    if type(plan.max_requests) is not int:
        return _refused(plan, "READY plan's max_requests is not a plain int")
    if plan.max_requests != READY_MAX_REQUESTS:
        return _refused(
            plan, "READY plan's max_requests does not equal the fixed request budget"
        )

    env_copy = dict(env)
    try:
        request = validate_literature_pipeline_request(
            enabled=True,
            pmid=None,
            nct_id=canonical_nct,
            live=True,
            use_fixtures=False,
            use_corpus=False,
            max_articles=DEFAULT_MAX_ARTICLES,
            max_fulltext_fetches=DEFAULT_MAX_FULLTEXT_FETCHES,
            env=env_copy,
        )
    except LiteraturePipelineRequestError as exc:
        return _refused(plan, f"request validation refused this plan: {exc}")

    if request is None:
        # Unreachable given enabled=True always either returns a request
        # or raises -- but this module never trusts that invariant
        # silently (same precedent as the strict-NCT re-check above).
        return _refused(plan, "request validation returned no request unexpectedly")

    bundle = run_literature_pipeline_acquisition(
        request, ticker=ticker, http_client=http_client, env=env_copy,
    )

    if bundle.refused:
        return AdaptiveLiteratureExecution(
            status=AdaptiveExecutionStatus.REFUSED,
            plan=plan,
            bundle=bundle,
            rationale="the acquisition itself was refused before any request (see bundle.refused_reason)",
        )
    if bundle.coverage_complete:
        return AdaptiveLiteratureExecution(
            status=AdaptiveExecutionStatus.COMPLETE,
            plan=plan,
            bundle=bundle,
            rationale=f"acquisition completed with full coverage for {canonical_nct}",
        )
    return AdaptiveLiteratureExecution(
        status=AdaptiveExecutionStatus.INCOMPLETE,
        plan=plan,
        bundle=bundle,
        rationale=f"acquisition ran but coverage is incomplete for {canonical_nct}",
    )


__all__ = [
    "AdaptiveExecutionStatus",
    "AdaptiveLiteratureExecution",
    "LiteratureHttpTransport",
    "execute_adaptive_literature_plan",
]
