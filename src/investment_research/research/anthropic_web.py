"""Anthropic server-side web research (ADR 0005, the primary path).

Uses the Messages API server tools ``web_search_20260318`` and
``web_fetch_20260318``. Both execute on Anthropic's infrastructure, which is why
this is the right answer for a client environment with restricted egress: the
program is not reaching those sites at all, so there is nothing to circumvent.

Two API facts shape the code:

* ``web_fetch`` only fetches URLs already present in the conversation. So a
  fetch is issued as a follow-up turn carrying the URL, never as a cold call.
* Server-tool failures return HTTP 200 with an error object inside the result
  block rather than raising. A ``web_search_tool_result`` whose ``content`` is a
  dict is an error; a list is success. Getting that branch wrong silently turns
  a failed search into "no results found", which this system must never do.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..collectors.documents import Document
from ..collectors.tiering import classify_authority, classify_tier
from ..llm.client import BudgetExceeded
from ..schemas.enums import (
    UNKNOWN,
    ContentKind,
    DocumentAuthority,
    FetchOutcome,
    Provenance,
    ResearchPath,
)
from ..schemas.fact import Source
from .provider import ResearchQuery, ResearchResult

#: doc_type label to attach for each DocumentAuthority, matching the
#: vocabulary escalation.py's admissibility check already reads
#: (_PRIMARY_DOC_TYPES = {"filing", "registry", "regulator", "docket"}).
_DOC_TYPE_FOR_AUTHORITY: dict[DocumentAuthority, str] = {
    DocumentAuthority.REGULATOR: "regulator",
    DocumentAuthority.STATUTORY_FILING: "filing",
    DocumentAuthority.REGISTRY: "registry",
    DocumentAuthority.COMPANY_IR: "press_release",
    DocumentAuthority.INDEPENDENT: "independent",
    DocumentAuthority.UNKNOWN: "unknown",
}

log = logging.getLogger(__name__)

#: Server tool type strings. The dated variants with dynamic filtering require
#: Opus 4.6+/Sonnet 4.6+; the basic variants are the fallback for older models.
#: 20260318 is the current variant, verified live against claude-sonnet-5 and
#: claude-opus-5; it supersedes the older 20260209 dated tools. A compatibility
#: fallback to the basic (non-dated-dynamic-filtering) variants is kept for
#: models that don't support server-side dynamic filtering at all.
WEB_SEARCH_TOOL = "web_search_20260318"
WEB_FETCH_TOOL = "web_fetch_20260318"
WEB_SEARCH_TOOL_BASIC = "web_search_20250305"
WEB_FETCH_TOOL_BASIC = "web_fetch_20250910"

#: Callers permitted to invoke the server-side web tools directly (as opposed
#: to via server-side code execution / programmatic tool calling). This
#: provider only ever issues them directly from the top-level agent turn.
_DIRECT_CALLER = ["direct"]

#: Search is discovery-only: the useful payload comes back as
#: ``web_search_tool_result`` content, not model prose, so a search call never
#: needs a long completion. Kept far below the interpretive agents' output
#: ceilings on purpose (cost control, requirement P9).
SEARCH_MAX_TOKENS = 1024

#: Models that support the dynamic-filtering variants.
_DYNAMIC_FILTER_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
    "claude-fable-5-1",
)

SEARCH_SYSTEM = (
    "You are a research retrieval assistant for an investment research system whose purpose "
    "is to eliminate wrong investment hypotheses using primary sources.\n"
    "Use the web_search tool to answer the query. Prefer regulator, exchange and statutory "
    "filing sources over commentary. Do not summarise, interpret, rank or evaluate the "
    "results, and do not offer any view about the security. Return only what you found.\n"
    "Never invent a URL, a date, or a quotation. If you found nothing, say NOTHING FOUND."
)


def tool_types_for(model: str) -> tuple[str, str]:
    """Pick the server-tool variants this model supports."""
    if any(model.startswith(prefix) for prefix in _DYNAMIC_FILTER_MODELS):
        return WEB_SEARCH_TOOL, WEB_FETCH_TOOL
    return WEB_SEARCH_TOOL_BASIC, WEB_FETCH_TOOL_BASIC


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AnthropicWebResearchProvider:
    """Web search and fetch through Anthropic's server-side tools."""

    name = "anthropic_web"
    path = ResearchPath.ANTHROPIC_WEB

    def __init__(
        self,
        llm,  # investment_research.llm.client.LLMClient
        *,
        max_uses_per_query: int = 5,
        max_content_tokens: int = 20000,
        research_effort: str = "low",
    ) -> None:
        self.llm = llm
        self.max_uses_per_query = max_uses_per_query
        self.max_content_tokens = max_content_tokens
        # Research (discovery/fetch) runs at its own, independently-controlled
        # effort -- see --research-effort. It must never inherit --llm-effort,
        # which governs the eight INTERPRETIVE agents only: a single high-effort
        # discovery pass burned 76,891 tokens on two searches in a live run.
        self.research_effort = research_effort

    def available(self) -> tuple[bool, str]:
        return self.llm.available()

    # -- search ------------------------------------------------------------
    def search(self, query: ResearchQuery, *, agent_id: str = "anthropic_web_search") -> ResearchResult:
        usable, reason = self.available()
        if not usable:
            return ResearchResult(
                query=query,
                outcome=FetchOutcome.DISABLED,
                path=self.path,
                executed=False,
                error=reason,
            )

        search_tool, _ = tool_types_for(self.llm.model)
        tool: dict[str, Any] = {
            "type": search_tool,
            "name": "web_search",
            "max_uses": self.max_uses_per_query,
        }
        if search_tool == WEB_SEARCH_TOOL:
            tool["allowed_callers"] = _DIRECT_CALLER
        if query.allowed_domains:
            tool["allowed_domains"] = list(query.allowed_domains)

        try:
            response = self.llm.raw_message(
                system=SEARCH_SYSTEM,
                messages=[{"role": "user", "content": query.query}],
                tools=[tool],
                max_tokens=SEARCH_MAX_TOKENS,
                agent_id=agent_id,
                effort=self.research_effort,
            )
        except BudgetExceeded as exc:
            # A budget cutoff is "we did not look", not "we looked and found
            # nothing" -- executed=False so the Search Completeness Gate marks
            # this domain FAILED/UNSEARCHED rather than silently SEARCHED with
            # zero documents (requirement P6 / CLAUDE.md rule 8).
            log.warning("web search skipped for %r: %s", query.query, exc)
            return ResearchResult(
                query=query,
                outcome=FetchOutcome.DISABLED,
                path=self.path,
                executed=False,
                error=f"BudgetExceeded: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            log.warning("web search failed for %r: %s", query.query, exc)
            return ResearchResult(
                query=query,
                outcome=FetchOutcome.ERROR,
                path=self.path,
                error=f"{type(exc).__name__}: {exc}",
            )

        documents, error = self._documents_from_response(response, query)
        # Evidence integrity is untouched by this cap: a truncated result is
        # still METADATA_ONLY, still a pointer, never promoted to a fact.
        documents = documents[: query.max_results]
        outcome = FetchOutcome.OK if not error else FetchOutcome.ERROR
        if not documents and not error:
            outcome = FetchOutcome.NOT_FOUND
        return ResearchResult(
            query=query,
            documents=documents,
            outcome=outcome,
            path=self.path,
            error=error,
            tokens_used=self.llm.last_usage_tokens,
        )

    def _documents_from_response(
        self, response: Any, query: ResearchQuery
    ) -> tuple[list[Document], str]:
        return parse_search_response(response, path=self.path)

    # -- fetch -------------------------------------------------------------
    def fetch(
        self,
        url: str,
        *,
        reason: str = "",
        agent_id: str = "anthropic_web_fetch",
        known: Source | Document | None = None,
    ) -> Document | None:
        """Fetch one URL's full text.

        ``web_fetch`` only fetches URLs already in the conversation, so the URL
        is placed in the user turn and the model is instructed to fetch it.

        ``known`` is the already-collected ``Source`` or ``Document`` (a
        SearchHit-shaped pointer) that led here, when there is one. Its real
        metadata -- title, publisher, published/event/effective/filing date,
        accession, tier -- is carried forward onto the fetched Document
        wherever it is actually known (requirement B2); nothing is invented
        for a field ``known`` does not have. Retrieval time
        (``retrieved_at``, stamped below) is NEVER used as a substitute for
        any of those dates (requirement B1) -- a historical filing fetched
        today stays dated when it was filed, not when it was fetched.
        """
        usable, _ = self.available()
        if not usable:
            return None

        _, fetch_tool = tool_types_for(self.llm.model)
        tool: dict[str, Any] = {
            "type": fetch_tool,
            "name": "web_fetch",
            "max_uses": 2,
            "max_content_tokens": self.max_content_tokens,
            "citations": {"enabled": True},
        }
        if fetch_tool == WEB_FETCH_TOOL:
            tool["allowed_callers"] = _DIRECT_CALLER
        prompt = (
            f"Fetch this URL and return its substantive text verbatim, without summarising, "
            f"interpreting or evaluating it: {url}\n"
            f"Reason this document is needed: {reason or 'primary source verification'}\n"
            "If the fetch fails, reply exactly FETCH FAILED."
        )
        try:
            response = self.llm.raw_message(
                system=SEARCH_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                max_tokens=self.max_content_tokens,
                agent_id=agent_id,
                effort=self.research_effort,
            )
        except BudgetExceeded as exc:
            log.warning("web fetch skipped for %s: %s", url, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("web fetch failed for %s: %s", url, exc)
            return None

        # `retrieved` here is when Anthropic's server retrieved the page --
        # retrieval time, same axis as our own retrieved_at below. It is
        # deliberately NEVER used as published_date/event_date/etc.
        text, fetched_title, retrieved = _extract_fetch_result(response)
        if not text:
            return None

        seed = _seed_from(known)
        is_company_ir = bool(getattr(known, "is_company_ir", False))
        authority = classify_authority(url, is_company_ir=is_company_ir)
        return Document(
            doc_id=_doc_id(url),
            url=url,
            title=seed.get("title") or fetched_title or url,
            publisher=seed.get("publisher") or _publisher(url),
            published_date=seed.get("published_date", UNKNOWN),
            event_date=seed.get("event_date", UNKNOWN),
            effective_date=seed.get("effective_date", UNKNOWN),
            filing_date=seed.get("filing_date", UNKNOWN),
            accession=seed.get("accession", UNKNOWN),
            doc_type=seed.get("doc_type") or _DOC_TYPE_FOR_AUTHORITY[authority],
            is_company_ir=authority is DocumentAuthority.COMPANY_IR,
            text=text,
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            research_path=self.path,
            # Retrieval time only -- see the docstring above.
            retrieved_at=_now(),
            tier=seed.get("tier") or classify_tier(url, is_company_ir=is_company_ir),
            authority=authority,
        )


#: Fields carried forward from an already-known Source/Document seed, when
#: (and only when) the seed actually has them -- never invented.
_SEED_DATE_FIELDS = ("published_date", "event_date", "effective_date", "filing_date")


def _seed_from(known: Source | Document | None) -> dict[str, Any]:
    """Extract known, non-UNKNOWN metadata from an already-collected
    Source or SearchHit-shaped Document, for carrying forward onto a
    fetched Document (requirement B2). Both dataclasses share these field
    names; a field either object does not have, or holds UNKNOWN in, is
    simply absent from the result -- callers fall back to UNKNOWN, never to
    retrieval time.
    """
    if known is None:
        return {}
    seed: dict[str, Any] = {}
    for field_name in ("title", "publisher", *_SEED_DATE_FIELDS, "accession"):
        value = getattr(known, field_name, None)
        if value and value != UNKNOWN:
            seed[field_name] = value
    tier = getattr(known, "tier", None)
    if tier is not None and str(tier) != "UNKNOWN":
        seed["tier"] = tier
    doc_type = getattr(known, "doc_type", None)
    if doc_type and doc_type not in ("unknown", "web"):
        seed["doc_type"] = doc_type
    return seed


def _extract_fetch_result(response: Any) -> tuple[str, str, str]:
    """Pull document text out of a ``web_fetch_tool_result`` block."""
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "web_fetch_tool_result":
            continue
        content = getattr(block, "content", None)
        if content is None and isinstance(block, dict):
            content = block.get("content")
        if content is None:
            continue
        if _get(content, "error_code"):
            log.warning("web fetch error: %s", _get(content, "error_code"))
            return "", "", ""
        document = _get(content, "content") or {}
        source = _get(document, "source") or {}
        text = _get(source, "data") or ""
        title = _get(document, "title") or ""
        retrieved = _get(content, "retrieved_at") or ""
        return str(text), str(title), str(retrieved)
    return "", "", ""


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _doc_id(url: str) -> str:
    import hashlib

    return "web_" + hashlib.sha256(url.encode()).hexdigest()[:16]


def _publisher(url: str) -> str:
    from urllib.parse import urlparse

    try:
        host = urlparse(url).hostname or UNKNOWN
    except ValueError:
        return UNKNOWN
    return host[4:] if host.startswith("www.") else host


def parse_search_response(
    response: Any, *, path: ResearchPath = ResearchPath.ANTHROPIC_WEB
) -> tuple[list[Document], str]:
    """Turn a Messages API response into documents.

    Module-level so it can be exercised directly against recorded API payloads.
    The error branch is the point: a ``web_search_tool_result`` whose ``content``
    is a mapping is a *failure*, and must not be read as an empty result set.
    """
    documents: list[Document] = []
    error = ""
    for block in getattr(response, "content", None) or (
        response.get("content", []) if isinstance(response, dict) else []
    ):
        block_type = _get(block, "type")
        if block_type != "web_search_tool_result":
            continue
        content = _get(block, "content")
        if content is None:
            continue
        if isinstance(content, dict) or not isinstance(content, (list, tuple)):
            code = _get(content, "error_code") or "unknown_error"
            error = f"web_search error: {code}"
            log.warning("web search returned an error block: %s", code)
            continue
        for item in content:
            url = _get(item, "url")
            if not url:
                continue
            authority = classify_authority(url)
            documents.append(
                Document(
                    doc_id=_doc_id(url),
                    url=url,
                    title=_get(item, "title") or url,
                    publisher=_publisher(url),
                    # Only what the search engine actually reports about the
                    # page; never invented, and never retrieval time (see
                    # retrieved_at below, which IS retrieval time).
                    published_date=_get(item, "page_age") or UNKNOWN,
                    doc_type=_DOC_TYPE_FOR_AUTHORITY[authority],
                    is_company_ir=authority is DocumentAuthority.COMPANY_IR,
                    text=_get(item, "text") or "",
                    # A search result is a pointer, not the document. Fetch is
                    # what upgrades it to FULL_DOCUMENT.
                    content_kind=ContentKind.METADATA_ONLY,
                    provenance=Provenance.LIVE,
                    research_path=path,
                    retrieved_at=_now(),
                    tier=classify_tier(url, is_company_ir=authority is DocumentAuthority.COMPANY_IR),
                    authority=authority,
                )
            )
    return documents, error
