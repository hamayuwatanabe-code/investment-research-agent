"""Research provider abstraction (ADR 0005).

One interface, several channels, and every fact records which channel served it.

    ANTHROPIC_WEB   Anthropic server-side web_search / web_fetch. The primary
                    path: search and fetch execute on Anthropic's infrastructure,
                    so a restricted client environment does not have to be
                    circumvented -- it simply is not involved.
    DIRECT_API      SEC EDGAR, ClinicalTrials.gov v2, openFDA. Structured and
                    free; preferred wherever the environment can reach them,
                    because parsing beats searching for structured data.
    SEARCH_API      Tavily / Brave / an MCP search server, via the existing
                    SearchProvider interface. Configuration, not architecture.
    CORPUS          Replay of documents captured earlier, with the capture time
                    recorded and surfaced in the report.

A provider never silently substitutes one path for another: the path is part of
the result, and the report states which paths served the run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..collectors.documents import Document
from ..schemas.enums import FetchOutcome, ResearchDomain, ResearchPath, SearchStatus

log = logging.getLogger(__name__)


@dataclass
class ResearchQuery:
    """One question put to the outside world."""

    query: str
    domain: ResearchDomain
    #: "bear" queries hunt for disconfirming evidence; "bull" for supporting.
    #: They are executed separately and their results are never pooled before
    #: the isolation boundary (requirement P3).
    stance: str = "neutral"
    max_results: int = 6
    allowed_domains: tuple[str, ...] = ()
    rationale: str = ""

    def key(self) -> str:
        return f"{self.stance}:{self.domain}:{self.query}"


@dataclass
class ResearchResult:
    """What one query produced."""

    query: ResearchQuery
    documents: list[Document] = field(default_factory=list)
    outcome: FetchOutcome = FetchOutcome.OK
    path: ResearchPath = ResearchPath.NONE
    executed: bool = True
    error: str = ""
    tokens_used: int = 0

    @property
    def ok(self) -> bool:
        return self.executed and self.outcome == FetchOutcome.OK


@dataclass
class DomainCoverage:
    """Whether a required research domain was actually searched (P6)."""

    domain: ResearchDomain
    status: SearchStatus = SearchStatus.UNSEARCHED
    queries_executed: int = 0
    queries_attempted: int = 0
    documents_found: int = 0
    paths: tuple[ResearchPath, ...] = ()
    detail: str = ""

    @property
    def searched(self) -> bool:
        return self.status in (SearchStatus.SEARCHED, SearchStatus.PARTIAL)


class ResearchProvider(Protocol):
    """The contract every research channel implements.

    ``agent_id`` is passed through to an LLM-backed provider so its calls are
    tagged for stage-aware budget quotas and per-stage diagnostics (e.g.
    "adversarial_bear" vs "escalation"); a provider with no LLM behind it
    (corpus, null, or any future non-LLM channel) simply ignores it.
    """

    name: str
    path: ResearchPath

    def available(self) -> tuple[bool, str]:
        """``(usable, reason)``. A provider says why it cannot run."""
        ...

    def search(self, query: ResearchQuery, *, agent_id: str = "research") -> ResearchResult: ...

    def fetch(
        self, url: str, *, reason: str = "", agent_id: str = "research"
    ) -> Document | None: ...


class NullResearchProvider:
    """No research channel configured.

    Returns ``executed=False`` so callers can distinguish "searched and found
    nothing" from "never searched" -- the distinction the Search Completeness
    Gate is built on.
    """

    name = "none"
    path = ResearchPath.NONE

    def available(self) -> tuple[bool, str]:
        return False, "no research provider configured"

    def search(self, query: ResearchQuery, *, agent_id: str = "research") -> ResearchResult:
        return ResearchResult(
            query=query,
            outcome=FetchOutcome.DISABLED,
            path=ResearchPath.NONE,
            executed=False,
            error="no research provider configured",
        )

    def fetch(self, url: str, *, reason: str = "", agent_id: str = "research") -> Document | None:
        return None


class CompositeResearchProvider:
    """Tries providers in order and records which one served each query.

    Order is deliberate: a structured API answers a structured question better
    than a search engine, so DIRECT_API providers are tried first where they
    apply, and web search is the general fallback.
    """

    name = "composite"
    path = ResearchPath.NONE

    def __init__(self, providers: Sequence[ResearchProvider]) -> None:
        self.providers = list(providers)

    def available(self) -> tuple[bool, str]:
        reasons: list[str] = []
        for provider in self.providers:
            usable, reason = provider.available()
            if usable:
                return True, f"{provider.name} available"
            reasons.append(f"{provider.name}: {reason}")
        return False, "; ".join(reasons) or "no providers"

    def search(self, query: ResearchQuery, *, agent_id: str = "research") -> ResearchResult:
        last: ResearchResult | None = None
        for provider in self.providers:
            usable, reason = provider.available()
            if not usable:
                last = ResearchResult(
                    query=query,
                    outcome=FetchOutcome.DISABLED,
                    path=provider.path,
                    executed=False,
                    error=reason,
                )
                continue
            result = provider.search(query, agent_id=agent_id)
            if result.ok and result.documents:
                return result
            last = result
        return last or NullResearchProvider().search(query, agent_id=agent_id)

    def fetch(self, url: str, *, reason: str = "", agent_id: str = "research") -> Document | None:
        for provider in self.providers:
            usable, _ = provider.available()
            if not usable:
                continue
            document = provider.fetch(url, reason=reason, agent_id=agent_id)
            if document is not None:
                return document
        return None
