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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schemas.enums import ResearchDomain
from .provider import ResearchProvider, ResearchQuery, ResearchResult

log = logging.getLogger(__name__)

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
    """Results of the two passes, kept apart."""

    bear_results: list[ResearchResult] = field(default_factory=list)
    bull_results: list[ResearchResult] = field(default_factory=list)
    follow_up_queries: list[ResearchQuery] = field(default_factory=list)
    executed: int = 0
    unexecuted: list[str] = field(default_factory=list)

    def bear_documents(self) -> list:
        return [d for r in self.bear_results for d in r.documents]

    def bull_documents(self) -> list:
        return [d for r in self.bull_results for d in r.documents]

    def all_documents(self) -> list:
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


def run_adversarial_search(
    provider: ResearchProvider,
    plan: SearchPlan,
    *,
    llm: Any = None,
    max_follow_ups: int = 8,
) -> AdversarialOutcome:
    """Execute the bear pass, then the bull pass, then LLM follow-ups.

    Order is deliberate: the bear pass runs first and its results seed the
    follow-up generation, so the follow-ups hunt for disconfirmation rather than
    elaborating a story the bull pass has already started telling.
    """
    outcome = AdversarialOutcome()

    for query in plan.bear:
        result = provider.search(query)
        outcome.bear_results.append(result)
        if result.executed:
            outcome.executed += 1
        else:
            outcome.unexecuted.append(query.query)

    if llm is not None:
        follow_ups = _generate_follow_ups(llm, outcome, max_follow_ups)
        outcome.follow_up_queries = follow_ups
        for query in follow_ups:
            result = provider.search(query)
            outcome.bear_results.append(result)
            if result.executed:
                outcome.executed += 1
            else:
                outcome.unexecuted.append(query.query)

    for query in plan.bull:
        result = provider.search(query)
        outcome.bull_results.append(result)
        if result.executed:
            outcome.executed += 1
        else:
            outcome.unexecuted.append(query.query)

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
            agent_id="adversarial_search",
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
