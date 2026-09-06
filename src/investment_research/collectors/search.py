"""Pluggable web-search provider (requirement 13: bull AND bear searches).

Search is optional.  With no provider configured the system does not silently
skip the bear-side search: it records every query it *would* have run as an
unexecuted query, and the report says which kill-searches were never performed.
A kill category that was never searched is reported as UNSEARCHED, not K0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from ..schemas.enums import FetchOutcome
from .http import HttpClient

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    published_date: str = "UNKNOWN"


@dataclass
class SearchResponse:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    outcome: FetchOutcome = FetchOutcome.OK
    error: str = ""
    executed: bool = True


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:  # pragma: no cover
        ...


class NullSearchProvider:
    """No search configured.

    Returns ``executed=False`` so callers can distinguish "searched and found
    nothing" from "never searched" -- a distinction this system treats as
    load-bearing.
    """

    name = "none"

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:
        return SearchResponse(
            query=query,
            outcome=FetchOutcome.DISABLED,
            error="no search provider configured (IRA_SEARCH_PROVIDER=none)",
            executed=False,
        )


class TavilySearchProvider:
    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, http: HttpClient, api_key: str) -> None:
        self.http = http
        self.api_key = api_key

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:
        if not self.api_key:
            return SearchResponse(
                query=query,
                outcome=FetchOutcome.DISABLED,
                error="TAVILY_API_KEY not set",
                executed=False,
            )
        result = self.http.get(
            self.endpoint,
            params={"api_key": self.api_key, "query": query, "max_results": limit},
        )
        if not result.ok:
            return SearchResponse(query=query, outcome=result.outcome, error=result.error)
        payload = result.json() or {}
        hits = [
            SearchHit(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", ""))[:1000],
                published_date=str(item.get("published_date", "UNKNOWN")),
            )
            for item in payload.get("results", [])
            if item.get("url")
        ]
        return SearchResponse(query=query, hits=hits)


class BraveSearchProvider:
    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, http: HttpClient, api_key: str) -> None:
        self.http = http
        self.api_key = api_key

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:
        if not self.api_key:
            return SearchResponse(
                query=query,
                outcome=FetchOutcome.DISABLED,
                error="BRAVE_API_KEY not set",
                executed=False,
            )
        result = self.http.get(
            self.endpoint,
            params={"q": query, "count": limit},
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
        )
        if not result.ok:
            return SearchResponse(query=query, outcome=result.outcome, error=result.error)
        payload = result.json() or {}
        hits = [
            SearchHit(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("description", ""))[:1000],
                published_date=str(item.get("age", "UNKNOWN")),
            )
            for item in (payload.get("web", {}) or {}).get("results", [])
            if item.get("url")
        ]
        return SearchResponse(query=query, hits=hits)


def build_search_provider(http: HttpClient, settings) -> SearchProvider:
    provider = (settings.search_provider or "none").lower()
    if provider == "tavily" and settings.tavily_api_key:
        return TavilySearchProvider(http, settings.tavily_api_key)
    if provider == "brave" and settings.brave_api_key:
        return BraveSearchProvider(http, settings.brave_api_key)
    if provider not in ("none", ""):
        log.warning("search provider %r requested but no API key present", provider)
    return NullSearchProvider()


# --- query construction -----------------------------------------------------
#: Requirement 4-6: the Kill Agent's mandatory search set.
KILL_QUERY_TEMPLATES: tuple[str, ...] = (
    "{t} FDA concern",
    "{t} regulatory risk",
    "{t} failed",
    "{t} endpoint",
    "{t} dilution",
    "{t} going concern",
    "{t} reverse split",
    "{t} delisting",
    "{t} lawsuit",
    "{t} accounting",
    "{t} auditor",
    "{t} clinical hold",
    "{t} insider selling",
    "{t} short thesis",
    "{t} Complete Response Letter",
    "{t} SEC investigation",
    "{t} restatement",
    "{t} offering priced",
)

#: Requirement 13: a positive-only search is forbidden, so the bull-side set is
#: declared alongside the bear-side set and both are executed.
BULL_QUERY_TEMPLATES: tuple[str, ...] = (
    "{t} clinical data results",
    "{t} partnership agreement",
    "{t} contract award",
    "{t} FDA designation",
    "{t} revenue guidance",
    "{t} pipeline update",
)


def kill_queries(ticker: str, company_name: str = "") -> list[str]:
    subject = company_name or ticker
    queries = [tpl.format(t=ticker) for tpl in KILL_QUERY_TEMPLATES]
    if company_name:
        queries += [tpl.format(t=subject) for tpl in KILL_QUERY_TEMPLATES[:8]]
    return queries


def bull_queries(ticker: str, company_name: str = "") -> list[str]:
    subject = company_name or ticker
    return [tpl.format(t=subject) for tpl in BULL_QUERY_TEMPLATES]
