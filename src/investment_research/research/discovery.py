"""Search discovery records, kept separate from evidence (requirement M2).

A search result is a **pointer**, not a finding. Its title, snippet and engine
summary tell you where to look; they are not the document, and nobody has read
the document yet. Storing them as facts means a paraphrase written by a search
engine ends up quoted as though it were the filing -- which is the same
substitution, one layer earlier, that this whole system exists to prevent.

So discovery lives in its own store:

    SearchQueryRecord   what was asked, by whom, for what purpose, when
    SearchHit           what came back: title, url, snippet, provider, timestamp

Facts are extracted from **document bodies**. When a body cannot be retrieved,
the claim is recorded with ``SEARCH_EVIDENCE`` or
``PRIMARY_SOURCE_IDENTIFIED_BUT_NOT_FETCHED`` status -- never as verified
evidence, and never silently mixed in with facts read from a real document.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..schemas.enums import UNKNOWN, QueryPurpose, ResearchPath, SourceTier

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_query_id(agent_id: str, purpose: QueryPurpose, query_text: str, run_id: str) -> str:
    payload = f"{run_id}|{agent_id}|{purpose}|{query_text.strip().lower()}"
    return "q_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class SearchQueryRecord:
    """One issued search, auditable after the fact (requirement M3).

    Every field the requirement names is stored, including ``origin_fact_id``:
    a follow-up query generated from a fact must be traceable back to the fact
    that provoked it, or "the model decided to search for this" is unauditable.
    """

    query_id: str
    run_id: str
    ticker: str
    agent_id: str
    query_purpose: QueryPurpose
    query_text: str
    origin_fact_id: str | None = None
    created_at: str = field(default_factory=_now)
    results_count: int = 0
    executed: bool = True
    provider: str = UNKNOWN
    rationale: str = ""
    outcome: str = UNKNOWN

    def to_row(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "run_id": self.run_id,
            "ticker": self.ticker.upper(),
            "agent_id": self.agent_id,
            "query_purpose": str(self.query_purpose),
            "query_text": self.query_text,
            "origin_fact_id": self.origin_fact_id,
            "created_at": self.created_at,
            "results_count": self.results_count,
            "executed": int(self.executed),
            "provider": self.provider,
            "rationale": self.rationale[:500],
            "outcome": self.outcome,
        }


@dataclass
class SearchHit:
    """One search result. Discovery evidence, never a fact.

    ``body_retrieved`` is the field that decides everything downstream: until it
    is true, nothing here may become verified evidence.
    """

    hit_id: str
    query_id: str
    run_id: str
    ticker: str
    title: str
    url: str
    snippet: str = ""
    provider: str = UNKNOWN
    research_path: ResearchPath = ResearchPath.NONE
    query_purpose: QueryPurpose = QueryPurpose.NEUTRAL
    captured_at: str = field(default_factory=_now)
    published_date: str = UNKNOWN
    tier: SourceTier = SourceTier.UNKNOWN
    rank: int = 0
    #: True only once the document body behind this URL has actually been
    #: fetched and stored. A search snippet is not a body.
    body_retrieved: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "hit_id": self.hit_id,
            "query_id": self.query_id,
            "run_id": self.run_id,
            "ticker": self.ticker.upper(),
            "title": self.title[:500],
            "url": self.url,
            "snippet": self.snippet[:2000],
            "provider": self.provider,
            "research_path": str(self.research_path),
            "query_purpose": str(self.query_purpose),
            "captured_at": self.captured_at,
            "published_date": self.published_date,
            "tier": str(self.tier),
            "rank": self.rank,
            "body_retrieved": int(self.body_retrieved),
        }


def make_hit_id(query_id: str, url: str) -> str:
    return "hit_" + hashlib.sha256(f"{query_id}|{url}".encode()).hexdigest()[:16]


@dataclass
class DiscoveryLog:
    """Everything the search layer found, per run."""

    queries: list[SearchQueryRecord] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)

    def record_query(self, record: SearchQueryRecord) -> None:
        self.queries.append(record)

    def record_hits(self, hits: Iterable[SearchHit]) -> None:
        known = {hit.hit_id for hit in self.hits}
        for hit in hits:
            if hit.hit_id not in known:
                self.hits.append(hit)
                known.add(hit.hit_id)

    # -- purpose-scoped views (requirement M3) -----------------------------
    def queries_for(self, purposes: Sequence[QueryPurpose]) -> list[SearchQueryRecord]:
        wanted = set(purposes)
        return [q for q in self.queries if q.query_purpose in wanted]

    def hits_for(self, purposes: Sequence[QueryPurpose]) -> list[SearchHit]:
        wanted = set(purposes)
        return [h for h in self.hits if h.query_purpose in wanted]

    def hits_for_agent(self, agent_id: str) -> list[SearchHit]:
        from ..schemas.enums import purposes_for_agent

        return self.hits_for(tuple(purposes_for_agent(agent_id)))

    def urls_for_agent(self, agent_id: str) -> set[str]:
        return {hit.url for hit in self.hits_for_agent(agent_id)}

    def unfetched_primary_urls(self) -> list[SearchHit]:
        """Hits pointing at a primary source whose body was never retrieved."""
        return [h for h in self.hits if h.tier.is_primary and not h.body_retrieved]

    def summary(self) -> dict[str, Any]:
        by_purpose: dict[str, int] = {}
        for query in self.queries:
            key = str(query.query_purpose)
            by_purpose[key] = by_purpose.get(key, 0) + 1
        return {
            "queries": len(self.queries),
            "executed": sum(1 for q in self.queries if q.executed),
            "hits": len(self.hits),
            "bodies_retrieved": sum(1 for h in self.hits if h.body_retrieved),
            "by_purpose": by_purpose,
        }


def hits_from_documents(
    documents: Sequence[Any],
    *,
    query: SearchQueryRecord,
    provider: str,
    path: ResearchPath,
) -> list[SearchHit]:
    """Turn a provider's returned documents into discovery hits.

    ``body_retrieved`` follows the document's own ``content_kind``: only a
    document whose actual text was retrieved counts as a fetched body.
    """
    hits: list[SearchHit] = []
    for rank, document in enumerate(documents):
        hits.append(
            SearchHit(
                hit_id=make_hit_id(query.query_id, document.url),
                query_id=query.query_id,
                run_id=query.run_id,
                ticker=query.ticker,
                title=document.title,
                url=document.url,
                snippet=(document.text or "")[:2000],
                provider=provider,
                research_path=path,
                query_purpose=query.query_purpose,
                published_date=document.published_date,
                tier=document.tier,
                rank=rank,
                body_retrieved=document.content_kind.is_primary_text,
            )
        )
    return hits
