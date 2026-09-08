"""Adversarial search: bear and bull queries, executed separately (req. P3).

Two properties matter more than the query list itself.

**Separation.** Bear and bull searches run as separate passes with separate
budgets, and their results are attributed to their stance. Pooling them before
the isolation boundary would let the bull agent see what the bear search turned
up, which is the rebuttal loop the whole system is built to avoid.

**Follow-ups.** A fixed query list only finds the problems someone anticipated.
So the collected evidence is fed back to the model, which proposes further
queries -- "search for the 2024 meeting that first called this trial pivotal" is
the kind of query no static list contains, and is exactly how the LGVN-type
finding gets confirmed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schemas.enums import QueryPurpose, ResearchDomain
from .discovery import DiscoveryLog, SearchQueryRecord, hits_from_documents, make_query_id
from .provider import ResearchProvider, ResearchQuery, ResearchResult

log = logging.getLogger(__name__)

#: Prefix on ResearchResult.error set by AnthropicWebResearchProvider when a
#: BudgetExceeded abort is caught (see research/anthropic_web.py). Used here
#: to distinguish "we did not look because the discovery budget ran out"
#: from every other reason a query went unexecuted.
_BUDGET_ERROR_PREFIX = "BudgetExceeded"

#: The mandatory bear-side set (requirement P3). ``{t}`` is the ticker,
#: ``{c}`` the company name, ``{d}`` a drug or programme name where known.
BEAR_TEMPLATES: tuple[tuple[str, ResearchDomain], ...] = (
    ("{t} FDA concern", ResearchDomain.REGULATORY),
    ("{c} regulatory risk", ResearchDomain.REGULATORY),
    ("{d} endpoint concern", ResearchDomain.REGULATORY),
    ("{c} failed trial", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("{c} dilution", ResearchDomain.CAPITAL_STRUCTURE),
    ("{c} going concern", ResearchDomain.CAPITAL_STRUCTURE),
    ("{c} warrant", ResearchDomain.CAPITAL_STRUCTURE),
    ("{c} reverse split", ResearchDomain.CAPITAL_STRUCTURE),
    ("{c} delisting", ResearchDomain.CAPITAL_STRUCTURE),
    ("{c} lawsuit", ResearchDomain.CONTRADICTION),
    ("{c} auditor", ResearchDomain.CONTRADICTION),
    ("{c} insider selling", ResearchDomain.CONTRADICTION),
    ("{c} criticism", ResearchDomain.CONTRADICTION),
    ("{c} short thesis", ResearchDomain.COMPETITION),
    ("{d} competitor superiority", ResearchDomain.COMPETITION),
    ("{d} safety concern", ResearchDomain.SCIENCE_TECHNOLOGY),
)

#: The bull-side set. Requirement 13 forbids a positive-only search; it equally
#: forbids a negative-only one, so both are declared and both are executed.
BULL_TEMPLATES: tuple[tuple[str, ResearchDomain], ...] = (
    ("{c} clinical data results", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("{c} partnership agreement", ResearchDomain.COMPETITION),
    ("{c} FDA designation", ResearchDomain.REGULATORY),
    ("{c} cash runway financing", ResearchDomain.CAPITAL_STRUCTURE),
    ("{d} mechanism efficacy", ResearchDomain.SCIENCE_TECHNOLOGY),
    ("{c} upcoming catalyst readout", ResearchDomain.CATALYST),
)

FOLLOW_UP_SYSTEM = """\
You generate additional web search queries for an investment research system whose purpose is to
eliminate wrong investment hypotheses using primary sources.

Given evidence already collected, propose searches that would CONFIRM OR REFUTE the most
consequential possibilities it implies. Good follow-up queries are specific: they name the
document, the meeting, the endpoint, the trial or the counterparty. "company risk" is useless;
"ELPIS II 2024 Type C meeting pivotal designation" is useful.

Do not propose queries designed to find supporting material for a bull case unless explicitly
asked for bull-stance queries. Return queries only, no commentary."""

FOLLOW_UP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["queries"],
    "properties": {
        "queries": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query", "domain", "rationale"],
                "properties": {
                    "query": {"type": "string", "maxLength": 200},
                    "domain": {
                        "type": "string",
                        "enum": [d.value for d in ResearchDomain],
                    },
                    "rationale": {"type": "string", "maxLength": 400},
                },
            },
        }
    },
}


@dataclass
class SearchPlan:
    bear: list[ResearchQuery] = field(default_factory=list)
    bull: list[ResearchQuery] = field(default_factory=list)

    def all(self) -> list[ResearchQuery]:
        return [*self.bear, *self.bull]


@dataclass
class AdversarialOutcome:
    """Results of the two passes, kept apart.

    ``documents`` on each ``ResearchResult`` remain here for callers that need
    the raw provider response (e.g. escalation, which fetches bodies). They are
    NOT facts and must never be extracted into facts directly (requirement M2)
    -- ``discovery`` is the audited, purpose-tagged record of what was searched
    and found, and it is what agents and the completeness gate should read.
    """

    bear_results: list[ResearchResult] = field(default_factory=list)
    bull_results: list[ResearchResult] = field(default_factory=list)
    follow_up_queries: list[ResearchQuery] = field(default_factory=list)
    executed: int = 0
    unexecuted: list[str] = field(default_factory=list)
    #: The subset of `unexecuted` specifically caused by the discovery stage
    #: budget quota being exhausted -- "we did not look because we ran out of
    #: budget", never conflated with "we looked and found nothing" or any
    #: other reason a query did not run. The Search Completeness Gate reads
    #: `executed` on each ResearchResult either way, so a required domain
    #: that lost its only query to a budget cutoff still correctly reads
    #: UNSEARCHED/FAILED, not silently SEARCHED.
    unexecuted_due_to_budget: list[str] = field(default_factory=list)
    #: Queries dropped as near-duplicates before ever being issued (requirement
    #: F: deduplicate semantically overlapping queries). Distinct from
    #: `unexecuted`: these were never even attempted, by design, not cut off.
    deduplicated: list[str] = field(default_factory=list)
    #: Auditable, purpose-separated query and hit log (requirement M3).
    discovery: DiscoveryLog = field(default_factory=DiscoveryLog)

    def bear_documents(self) -> list:
        return [d for r in self.bear_results for d in r.documents]

    def bull_documents(self) -> list:
        return [d for r in self.bull_results for d in r.documents]

    def all_documents(self) -> list:
        """All discovered documents, deduplicated.

        Kept for callers that need to attempt body retrieval (escalation), NOT
        for fact extraction -- see the class docstring.
        """
        seen: set[str] = set()
        out = []
        for document in self.bear_documents() + self.bull_documents():
            if document.doc_id in seen:
                continue
            seen.add(document.doc_id)
            out.append(document)
        return out


def build_plan(ticker: str, company: str, programmes: Sequence[str] = ()) -> SearchPlan:
    """Expand the mandatory templates for this candidate."""
    subject = company or ticker
    drug = programmes[0] if programmes else subject

    def expand(
        templates: tuple[tuple[str, ResearchDomain], ...], stance: str
    ) -> list[ResearchQuery]:
        seen: set[str] = set()
        queries: list[ResearchQuery] = []
        for template, domain in templates:
            text = template.format(t=ticker, c=subject, d=drug).strip()
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            queries.append(
                ResearchQuery(
                    query=text,
                    domain=domain,
                    stance=stance,
                    rationale=f"mandatory {stance} query",
                )
            )
        return queries

    return SearchPlan(bear=expand(BEAR_TEMPLATES, "bear"), bull=expand(BULL_TEMPLATES, "bull"))


def _prioritize_domain_coverage(queries: Sequence[ResearchQuery]) -> list[ResearchQuery]:
    """Reorder so every domain's first query runs before any domain's second.

    Guarantees required-domain coverage survives a budget cutoff (requirement
    F): if a stance's queries stop partway through, every domain that had any
    query in this stance already got at least one attempt, rather than one
    domain's extra queries exhausting the quota before a domain later in the
    original template list ever got a turn. A stable sort within each group
    keeps the original relative order otherwise.
    """
    seen_domains: set[ResearchDomain] = set()
    first_pass: list[ResearchQuery] = []
    rest: list[ResearchQuery] = []
    for query in queries:
        if query.domain not in seen_domains:
            first_pass.append(query)
            seen_domains.add(query.domain)
        else:
            rest.append(query)
    return [*first_pass, *rest]


def _query_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedupe_semantically(
    queries: Sequence[ResearchQuery],
    *,
    threshold: float = 0.85,
    seed_tokens: Sequence[set[str]] = (),
) -> tuple[list[ResearchQuery], list[str]]:
    """Drop a query whose normalized keyword set nearly duplicates an earlier one.

    A cheap token-overlap heuristic, not a semantic model -- good enough to
    catch "the same question asked two different ways" (most likely among
    LLM-generated follow-ups) without spending another LLM call on it. Returns
    the deduplicated list plus the text of every query dropped, so the caller
    can record what was skipped and why (requirement F). ``seed_tokens`` lets
    a caller dedupe a new batch (e.g. follow-ups) against queries already
    issued in an earlier batch (e.g. the mandatory bear pass) without
    re-executing or re-listing them.
    """
    kept: list[ResearchQuery] = []
    kept_tokens: list[set[str]] = list(seed_tokens)
    dropped: list[str] = []
    for query in queries:
        tokens = _query_tokens(query.query)
        if any(_jaccard(tokens, other) >= threshold for other in kept_tokens):
            dropped.append(query.query)
            continue
        kept.append(query)
        kept_tokens.append(tokens)
    return kept, dropped


def run_adversarial_search(
    provider: ResearchProvider,
    plan: SearchPlan,
    *,
    llm: Any = None,
    max_follow_ups: int = 8,
    run_id: str = "",
    ticker: str = "",
    agent_id: str = "adversarial_search",
) -> AdversarialOutcome:
    """Execute the bear pass, then the bull pass, then LLM follow-ups.

    Order is deliberate: the bear pass runs first and its results seed the
    follow-up generation, so the follow-ups hunt for disconfirmation rather than
    elaborating a story the bull pass has already started telling.

    Every query and every hit is recorded on ``outcome.discovery`` with its
    purpose (requirement M3), so a Bear-purpose hit can never be handed to the
    Bull agent's context (or vice versa) -- the isolation is enforced by which
    purposes an agent is allowed to read (``purposes_for_agent``), not by hoping
    nothing gets mixed up downstream.

    Adaptive, not 30 blind calls (requirement F): within each stance, every
    required domain's first query runs before any domain's second
    (``_prioritize_domain_coverage``), near-duplicate queries are dropped
    before ever being issued (``_dedupe_semantically``), follow-ups are
    generated only from the mandatory bear pass's own results, and every
    query the discovery budget quota could not afford is recorded as such
    (``unexecuted_due_to_budget``) rather than looking searched or clean. If
    ``llm`` is given, this sets its budget's current stage to ``"discovery"``
    for the duration of this call (see ``LLMBudget.set_stage``).
    """
    outcome = AdversarialOutcome()
    run_id = run_id or ticker or "unknown-run"
    if llm is not None:
        llm.budget.set_stage("discovery")

    def _execute(
        query: ResearchQuery,
        purpose: QueryPurpose,
        *,
        llm_agent_id: str,
        origin_fact_id: str | None = None,
    ) -> ResearchResult:
        result = provider.search(query, agent_id=llm_agent_id)
        record = SearchQueryRecord(
            query_id=make_query_id(agent_id, purpose, query.query, run_id),
            run_id=run_id,
            ticker=ticker,
            agent_id=agent_id,
            query_purpose=purpose,
            query_text=query.query,
            origin_fact_id=origin_fact_id,
            results_count=len(result.documents),
            executed=result.executed,
            provider=getattr(provider, "name", "unknown"),
            rationale=query.rationale,
            outcome=str(result.outcome),
        )
        outcome.discovery.record_query(record)
        outcome.discovery.record_hits(
            hits_from_documents(
                result.documents, query=record, provider=record.provider, path=result.path
            )
        )
        return result

    def _record(result: ResearchResult, query: ResearchQuery) -> None:
        if result.executed:
            outcome.executed += 1
            return
        outcome.unexecuted.append(query.query)
        if str(result.error).startswith(_BUDGET_ERROR_PREFIX):
            outcome.unexecuted_due_to_budget.append(query.query)

    # --- bear: required-domain coverage first, then near-duplicates dropped
    bear_queries = _prioritize_domain_coverage(plan.bear)
    bear_queries, bear_dropped = _dedupe_semantically(bear_queries)
    outcome.deduplicated.extend(bear_dropped)
    bear_query_tokens = [_query_tokens(q.query) for q in bear_queries]

    for query in bear_queries:
        result = _execute(query, QueryPurpose.BEAR, llm_agent_id="adversarial_bear")
        outcome.bear_results.append(result)
        _record(result, query)

    # Follow-ups are optional, discovery-budget-scoped work: skip generating
    # them at all once the discovery stage quota is already spent, rather
    # than paying for a follow-up proposal call that could never be executed.
    discovery_exhausted = llm is not None and "discovery" in llm.budget.stage_exhausted
    if llm is not None and not discovery_exhausted:
        follow_ups = _generate_follow_ups(llm, outcome, max_follow_ups)
        # Never propose searching for something the mandatory bear pass
        # already asked.
        follow_ups, follow_up_dropped = _dedupe_semantically(
            follow_ups, seed_tokens=bear_query_tokens
        )
        outcome.deduplicated.extend(follow_up_dropped)
        outcome.follow_up_queries = follow_ups
        for query in follow_ups:
            result = _execute(query, QueryPurpose.BEAR, llm_agent_id="adversarial_followup")
            outcome.bear_results.append(result)
            _record(result, query)

    # --- bull: separate pass, same treatment -----------------------------
    bull_queries = _prioritize_domain_coverage(plan.bull)
    bull_queries, bull_dropped = _dedupe_semantically(bull_queries)
    outcome.deduplicated.extend(bull_dropped)

    for query in bull_queries:
        result = _execute(query, QueryPurpose.BULL, llm_agent_id="adversarial_bull")
        outcome.bull_results.append(result)
        _record(result, query)

    return outcome


def _generate_follow_ups(llm: Any, outcome: AdversarialOutcome, limit: int) -> list[ResearchQuery]:
    usable, _ = llm.available()
    if not usable:
        return []
    evidence = "\n".join(
        f"- [{document.doc_id}] {document.title}: {document.text[:300]}"
        for document in outcome.bear_documents()[:40]
    )
    if not evidence:
        return []
    try:
        payload = llm.structured(
            agent_id="adversarial_followup",
            system=FOLLOW_UP_SYSTEM,
            prompt=(
                "Evidence collected so far by the disconfirming search pass:\n\n"
                f"{evidence}\n\n"
                "Propose additional searches that would confirm or refute the most consequential "
                "possibilities this evidence implies."
            ),
            tool_name="propose_queries",
            tool_description="Propose additional disconfirming search queries.",
            schema=FOLLOW_UP_SCHEMA,
            max_tokens=4000,
        )
    except Exception as exc:  # noqa: BLE001 - a failed follow-up is not fatal
        log.warning("follow-up query generation failed: %s", exc)
        return []

    queries: list[ResearchQuery] = []
    for item in payload.get("queries", [])[:limit]:
        queries.append(
            ResearchQuery(
                query=str(item["query"]),
                domain=ResearchDomain(str(item["domain"])),
                stance="bear",
                rationale=str(item.get("rationale", "")),
            )
        )
    return queries
