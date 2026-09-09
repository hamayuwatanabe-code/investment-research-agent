"""Cost-safe batched required-domain research (the v4 token-starvation fix).

The problem, restated from a live run: an Anthropic server-side web-search
call costs ~22k actual tokens even at ``research_effort=low``. Six
independent required-domain queries -- one per REGULATORY / CAPITAL_STRUCTURE
/ SCIENCE_TECHNOLOGY / COMPETITION / CATALYST / CONTRADICTION gap -- cannot
fit a 60k discovery quota: a live run spent 66,731 tokens on three searches,
skipped twenty more to budget, and left required domains unsearched.

Raising the global token budget does not fix this; it only delays the same
failure. The fix is architectural: **separate the research INTENT (what
question, for what domain, is being asked) from the API CALL that answers
it.** One Anthropic message, with the web_search tool available for several
uses, can carry evidence for several clearly-labelled intents at once --
while every intent still gets its own, fully independent audit record and
completion status, exactly as if it had been issued as a solitary call.

Nothing here changes what counts as evidence: a search result is still a
pointer (``ContentKind.METADATA_ONLY``), never a fact, and per-intent
candidate identification is handed off to the existing question-aware
escalation pass (``research/escalation.py``) for the actual fetch-and-verify
work. Batching only ever reduces how many expensive discovery *calls* are
needed to raise every candidate; it never promotes a candidate to
decision-grade evidence itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..collectors.documents import Document
from ..schemas.enums import (
    UNKNOWN,
    FetchOutcome,
    IntentStatus,
    QueryPurpose,
    ResearchDomain,
    ResearchPath,
)
from ..schemas.fact import UnresolvedQuestion
from .discovery import DiscoveryLog, SearchQueryRecord, hits_from_documents, make_query_id
from .provider import ResearchProvider, ResearchQuery, ResearchResult

log = logging.getLogger(__name__)

#: A batch this large would itself risk the same per-call cost blowup
#: batching exists to avoid -- kept well below what a single server-tool
#: call's max_uses ceiling should ever be asked to carry (requirement F).
MAX_INTENTS_PER_BATCH = 6

#: Statuses that count as this intent having genuinely been examined --
#: either with or without evidence -- as opposed to never having been
#: reached at all this run.
_EXECUTED_STATUSES = (IntentStatus.EXECUTED_WITH_EVIDENCE, IntentStatus.EXECUTED_ZERO_RESULTS)

#: Domains treated as falsification-adjacent for batch ordering (requirement
#: D): REGULATORY carries the "is there a regulator concern" adversarial
#: questions closest to a kill finding, and CONTRADICTION is disconfirming
#: evidence by construction. Both are scheduled in the FIRST batch, ahead of
#: COMPETITION/CATALYST/SCIENCE_TECHNOLOGY/CAPITAL_STRUCTURE gaps, regardless
#: of template order.
_FALSIFICATION_DOMAINS = frozenset({ResearchDomain.REGULATORY, ResearchDomain.CONTRADICTION})


@dataclass
class ResearchIntent:
    """One research question a domain gap or an unresolved question raises.

    This is the unit priority, source-routing and per-intent completeness
    all operate on (requirement A) -- deliberately NOT the same unit as an
    underlying API call, which may serve several intents at once.
    """

    intent_id: str
    domain: ResearchDomain
    question: str
    #: Allowed source domains (e.g. ``("sec.gov",)``), empty meaning
    #: unrestricted. Requirement B/9: two intents with different, non-empty
    #: restrictions must never be silently merged into one shared filter.
    source_restrictions: tuple[str, ...] = ()
    #: Lower runs first within its wave (see ``build_research_batches``).
    priority: int = 50
    rationale: str = ""
    #: A material/critical (blocking) unresolved question -- always
    #: scheduled in the first wave, ahead of generic domain templates
    #: (requirement D).
    critical: bool = False
    origin_fact_id: str | None = None
    status: IntentStatus = IntentStatus.PENDING
    documents: list[Document] = field(default_factory=list)
    detail: str = ""
    path: ResearchPath = ResearchPath.NONE

    @property
    def executed(self) -> bool:
        return self.status in _EXECUTED_STATUSES

    def to_research_result(self) -> ResearchResult:
        """A ``ResearchResult``-shaped view of this intent's outcome, for
        callers (the Search Completeness Gate, adversarial's existing
        result lists) that read a flat list of ``ResearchResult`` objects.
        Never mutates the intent; a pure projection.
        """
        query = ResearchQuery(
            query=self.question,
            domain=self.domain,
            stance="bear",
            allowed_domains=self.source_restrictions,
            rationale=self.rationale,
        )
        outcome = {
            IntentStatus.EXECUTED_WITH_EVIDENCE: FetchOutcome.OK,
            IntentStatus.EXECUTED_ZERO_RESULTS: FetchOutcome.NOT_FOUND,
            IntentStatus.SKIPPED_DUE_TO_BUDGET: FetchOutcome.DISABLED,
            IntentStatus.INCOMPLETE_RESPONSE: FetchOutcome.DISABLED,
            IntentStatus.ERROR: FetchOutcome.ERROR,
            IntentStatus.PENDING: FetchOutcome.DISABLED,
        }[self.status]
        return ResearchResult(
            query=query,
            documents=list(self.documents),
            outcome=outcome,
            path=self.path,
            executed=self.executed,
            error="" if self.executed else self.detail,
        )


@dataclass
class ResearchBatch:
    """A small group of intents scheduled to share as few underlying API
    calls as possible (requirement B)."""

    batch_id: str
    intents: list[ResearchIntent] = field(default_factory=list)

    @property
    def intent_ids(self) -> list[str]:
        return [intent.intent_id for intent in self.intents]

    @property
    def domains(self) -> list[ResearchDomain]:
        seen: list[ResearchDomain] = []
        for intent in self.intents:
            if intent.domain not in seen:
                seen.append(intent.domain)
        return seen


@dataclass
class BatchDiagnostics:
    """Per-batch cost/outcome diagnostics (requirement F)."""

    batch_id: str
    intent_ids: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    server_tool_uses: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    actual_total_tokens: int = 0
    search_result_count: int = 0
    completed_intent_ids: list[str] = field(default_factory=list)
    incomplete_intent_ids: list[str] = field(default_factory=list)


def intents_from_unresolved_questions(
    questions: Sequence[UnresolvedQuestion],
    *,
    max_questions: int = 6,
) -> list[ResearchIntent]:
    """Material/critical unresolved questions, as intents (requirement D:
    these are scheduled ahead of every generic domain template)."""
    from ..scoring.completeness import research_domain_for_category  # deferred: avoid import cycle
    from .escalation import domains_for_category

    intents: list[ResearchIntent] = []
    for index, question in enumerate(q for q in questions if q.blocking):
        if index >= max_questions:
            break
        domain = research_domain_for_category(question.category) or ResearchDomain.CONTRADICTION
        intents.append(
            ResearchIntent(
                intent_id=f"uq_{index}_{question.category.value.lower()}",
                domain=domain,
                question=question.question,
                source_restrictions=domains_for_category(question.category),
                priority=0,
                rationale=f"material unresolved question: {question.question[:120]}",
                critical=True,
            )
        )
    return intents


def _wave_rank(intent: ResearchIntent) -> int:
    """Sort key for priority WITHIN a wave (requirement D): critical
    (blocking unresolved question) intents rank first, then REGULATORY/
    CONTRADICTION (falsification-adjacent) domain gaps, then everything
    else. Purely a tiebreak among intents already in the same wave -- it
    never moves an intent between waves (that would risk starving a
    domain of its own guaranteed first-batch slot, see
    ``build_research_batches``).
    """
    if intent.critical:
        return 0
    if intent.domain in _FALSIFICATION_DOMAINS:
        return 1
    return 2


def build_research_batches(
    intents: Sequence[ResearchIntent],
    *,
    max_intents_per_batch: int = MAX_INTENTS_PER_BATCH,
) -> list[ResearchBatch]:
    """Group intents into a small number of priority-ordered batches.

    Two guarantees, both required, that would otherwise trade off against
    each other under budget pressure:

    1. **Every required domain gets covered even if only ONE batch ever
       runs.** The first wave contains every critical (blocking unresolved
       question) intent PLUS exactly one intent per distinct domain --
       enough, on its own, to let the Search Completeness Gate see all six
       domains examined even if a later batch is never reached.
    2. **Critical work is never starved by generic templates** (requirement
       D): within each wave, critical intents rank first, then REGULATORY/
       CONTRADICTION (falsification-adjacent), then the rest.

    Everything not needed for that first per-domain sweep -- a domain's
    SECOND, THIRD, ... template query -- goes in the following wave(s). An
    overflow beyond ``max_intents_per_batch`` spills into an additional
    batch of the SAME wave rather than silently dropping or merging
    intents; nothing here ever changes an intent's own domain or source
    restrictions.
    """
    ordered = sorted(intents, key=lambda intent: (intent.priority, intent.intent_id))

    seen_domains: set[ResearchDomain] = set()
    first_wave: list[ResearchIntent] = []
    rest: list[ResearchIntent] = []
    for intent in ordered:
        if intent.critical:
            first_wave.append(intent)
        elif intent.domain not in seen_domains:
            first_wave.append(intent)
            seen_domains.add(intent.domain)
        else:
            rest.append(intent)

    first_wave.sort(key=lambda intent: (_wave_rank(intent), intent.priority, intent.intent_id))
    rest.sort(key=lambda intent: (_wave_rank(intent), intent.priority, intent.intent_id))

    batches: list[ResearchBatch] = []
    batch_index = 0
    for wave in (first_wave, rest):
        for start in range(0, len(wave), max_intents_per_batch):
            chunk = wave[start : start + max_intents_per_batch]
            if not chunk:
                continue
            batch_index += 1
            batches.append(ResearchBatch(batch_id=f"batch_{batch_index}", intents=chunk))
    return batches


def _apply_result_to_intent(intent: ResearchIntent, result: ResearchResult | None) -> None:
    """Set ``intent.status``/``documents``/``detail`` from its own result.

    ``result is None`` means the batch response never addressed this
    intent at all (requirement C) -- never inferred as zero results, and
    never silently borrowed from a neighboring intent's evidence.
    """
    if result is None:
        intent.status = IntentStatus.INCOMPLETE_RESPONSE
        intent.detail = "omitted from the batch response"
        return
    intent.path = result.path
    if not result.executed:
        error = str(result.error)
        intent.status = (
            IntentStatus.SKIPPED_DUE_TO_BUDGET if error.startswith("BudgetExceeded") else IntentStatus.ERROR
        )
        intent.detail = error
        return
    intent.documents = list(result.documents)
    if result.error:
        intent.status = IntentStatus.ERROR
        intent.detail = result.error
    elif result.documents:
        intent.status = IntentStatus.EXECUTED_WITH_EVIDENCE
    else:
        intent.status = IntentStatus.EXECUTED_ZERO_RESULTS


def execute_research_batch(
    batch: ResearchBatch,
    provider: ResearchProvider,
    *,
    agent_id: str,
    discovery: DiscoveryLog,
    run_id: str,
    ticker: str,
    query_purpose: QueryPurpose = QueryPurpose.BEAR,
    llm_agent_id: str | None = None,
) -> BatchDiagnostics:
    """Execute one batch, updating every intent's status/documents in place.

    Requirement A/C: if ``provider`` exposes a ``search_batch`` method
    (duck-typed -- only ``AnthropicWebResearchProvider`` implements one
    today), it is used to serve the whole batch in as few underlying calls
    as it can manage while still returning a per-intent result mapping. A
    provider that cannot batch (corpus replay, no research configured, a
    test double) is served with one ``.search()`` call per intent instead
    -- correctness is identical either way; only the call count differs.
    Every intent is recorded into ``discovery`` exactly as it would be for
    a solitary query, so per-intent audit provenance never depends on
    which path served it (requirement A).

    ``agent_id`` is the orchestrating caller's own identity (e.g.
    "adversarial_search") and is what every recorded ``SearchQueryRecord``
    carries. ``llm_agent_id`` (defaulting to ``agent_id`` when omitted) is
    what is passed to the PROVIDER call instead -- a finer-grained tag (e.g.
    "adversarial_bear") used for per-stance LLM budget/token attribution,
    exactly mirroring how a solitary, unbatched query already separates the
    two (see ``run_adversarial_search``'s own ``_execute``).
    """
    diagnostics = BatchDiagnostics(
        batch_id=batch.batch_id,
        intent_ids=batch.intent_ids,
        domains=[str(domain) for domain in batch.domains],
    )
    provider_agent_id = llm_agent_id or agent_id
    search_batch = getattr(provider, "search_batch", None)
    if callable(search_batch):
        results_by_intent, meta = search_batch(batch.intents, agent_id=provider_agent_id)
        diagnostics.server_tool_uses = int(meta.get("server_tool_uses", 0))
        diagnostics.prompt_tokens = int(meta.get("prompt_tokens", 0))
        diagnostics.output_tokens = int(meta.get("output_tokens", 0))
        diagnostics.actual_total_tokens = int(meta.get("actual_total_tokens", 0))
    else:
        results_by_intent = {}
        for intent in batch.intents:
            query = ResearchQuery(
                query=intent.question,
                domain=intent.domain,
                stance="bear",
                allowed_domains=intent.source_restrictions,
                rationale=intent.rationale,
            )
            results_by_intent[intent.intent_id] = provider.search(query, agent_id=provider_agent_id)

    for intent in batch.intents:
        _apply_result_to_intent(intent, results_by_intent.get(intent.intent_id))
        record = SearchQueryRecord(
            query_id=make_query_id(agent_id, query_purpose, intent.question, run_id),
            run_id=run_id,
            ticker=ticker,
            agent_id=agent_id,
            query_purpose=query_purpose,
            query_text=intent.question,
            origin_fact_id=intent.origin_fact_id,
            results_count=len(intent.documents),
            executed=intent.executed,
            provider=getattr(provider, "name", UNKNOWN),
            rationale=intent.rationale,
            outcome=str(intent.status),
        )
        discovery.record_query(record)
        discovery.record_hits(
            hits_from_documents(
                intent.documents, query=record, provider=record.provider, path=intent.path
            )
        )
        diagnostics.search_result_count += len(intent.documents)
        if intent.executed:
            diagnostics.completed_intent_ids.append(intent.intent_id)
        else:
            diagnostics.incomplete_intent_ids.append(intent.intent_id)
    return diagnostics


def run_research_batches(
    batches: Sequence[ResearchBatch],
    provider: ResearchProvider,
    *,
    agent_id: str,
    discovery: DiscoveryLog,
    run_id: str,
    ticker: str,
    query_purpose: QueryPurpose = QueryPurpose.BEAR,
    llm_agent_id: str | None = None,
) -> list[BatchDiagnostics]:
    """Execute batches in priority order, stopping cleanly once the
    discovery budget can no longer fund another (requirement F).

    A batch already executed keeps its completed intents' statuses exactly
    as recorded (requirement H8): a batch that comes back entirely
    SKIPPED_DUE_TO_BUDGET (the underlying provider refused every intent in
    it) stops the run from attempting any LATER batch -- their intents are
    marked SKIPPED_DUE_TO_BUDGET explicitly rather than left dangling as
    PENDING -- but a batch that only PARTIALLY succeeded (some intents
    completed, one omitted or budget-cut) never triggers this: the next
    batch still gets its turn. No exception ever propagates out of this
    function, so a batch failure can never poison a later pipeline stage's
    own, separately-scoped budget (requirement F/7).
    """
    all_diagnostics: list[BatchDiagnostics] = []
    exhausted = False
    for batch in batches:
        if exhausted:
            for intent in batch.intents:
                intent.status = IntentStatus.SKIPPED_DUE_TO_BUDGET
                intent.detail = "discovery budget already exhausted by an earlier batch"
            all_diagnostics.append(
                BatchDiagnostics(
                    batch_id=batch.batch_id,
                    intent_ids=batch.intent_ids,
                    domains=[str(domain) for domain in batch.domains],
                    incomplete_intent_ids=list(batch.intent_ids),
                )
            )
            continue
        try:
            diagnostics = execute_research_batch(
                batch,
                provider,
                agent_id=agent_id,
                discovery=discovery,
                run_id=run_id,
                ticker=ticker,
                query_purpose=query_purpose,
                llm_agent_id=llm_agent_id,
            )
        except Exception as exc:  # noqa: BLE001 - a batch failure never poisons the run
            log.warning("research batch %s failed: %s", batch.batch_id, exc)
            for intent in batch.intents:
                intent.status = IntentStatus.ERROR
                intent.detail = f"{type(exc).__name__}: {exc}"
            diagnostics = BatchDiagnostics(
                batch_id=batch.batch_id,
                intent_ids=batch.intent_ids,
                domains=[str(domain) for domain in batch.domains],
                incomplete_intent_ids=list(batch.intent_ids),
            )
        all_diagnostics.append(diagnostics)
        if batch.intents and all(
            intent.status is IntentStatus.SKIPPED_DUE_TO_BUDGET for intent in batch.intents
        ):
            exhausted = True
    return all_diagnostics
