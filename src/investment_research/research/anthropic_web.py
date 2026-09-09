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
import re
from collections.abc import Sequence
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
from .batching import ResearchIntent
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

#: The label convention a batched call relies on to attribute each search to
#: the intent it served (requirement A/B). Kept deliberately mechanical --
#: an exact marker line, not prose the parser would have to interpret --
#: because attribution here is a correctness boundary, not a formatting
#: nicety: an unattributed result must never be guessed into an intent's
#: evidence.
BATCH_INTENT_MARKER_RE = re.compile(r"--\s*INTENT\s+(\S+?)\s*--")

#: A batched call still runs at cost-controlled effort (requirement F); it
#: answers several intents, so it is allowed a larger completion ceiling
#: than a single-intent search, but still far below an interpretive agent's.
BATCH_SEARCH_MAX_TOKENS = 4096

BATCH_SEARCH_SYSTEM = (
    "You are a research retrieval assistant for an investment research system whose purpose "
    "is to eliminate wrong investment hypotheses using primary sources.\n"
    "You will be given several separately labelled research INTENTS. For EACH intent, in the "
    "order given:\n"
    "1. Emit a line containing exactly `-- INTENT <intent_id> --` (the exact id given, nothing "
    "else on that line).\n"
    "2. Immediately call the web_search tool to research that intent's question, honoring any "
    "source restriction stated for it.\n"
    "3. Do this even if you already have relevant knowledge -- only tool results count as "
    "evidence here.\n"
    "Do not combine, skip, reorder or merge intents; do not search for anything not asked. "
    "Prefer regulator, exchange and statutory filing sources over commentary. Do not "
    "summarise, interpret, rank or evaluate what you find, and do not offer any view about "
    "the security. Never invent a URL, a date, or a quotation. If an intent's search finds "
    "nothing, still emit its marker and still call the tool, then move on."
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
        max_uses_per_batch: int = 8,
    ) -> None:
        self.llm = llm
        self.max_uses_per_query = max_uses_per_query
        self.max_content_tokens = max_content_tokens
        # Research (discovery/fetch) runs at its own, independently-controlled
        # effort -- see --research-effort. It must never inherit --llm-effort,
        # which governs the eight INTERPRETIVE agents only: a single high-effort
        # discovery pass burned 76,891 tokens on two searches in a live run.
        self.research_effort = research_effort
        # Requirement F: a conservative, explicit ceiling on how many
        # server-tool uses ONE batched call may spend, independent of how
        # many intents are in the batch -- a batch is never allowed to
        # consume the whole discovery quota by itself.
        self.max_uses_per_batch = max_uses_per_batch

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

    # -- batched search ------------------------------------------------------
    def search_batch(
        self, intents: Sequence[ResearchIntent], *, agent_id: str = "anthropic_web_batch"
    ) -> tuple[dict[str, ResearchResult], dict[str, int]]:
        """Serve several ``ResearchIntent``s from as few underlying calls as
        possible (requirement A/B).

        One message is sent with the web_search tool available for several
        uses. The model is instructed (``BATCH_SEARCH_SYSTEM``) to emit a
        ``-- INTENT <id> --`` marker immediately before searching for each
        intent, in order; the response is then parsed by walking its content
        blocks and attributing every ``web_search_tool_result`` to whichever
        marker most recently preceded it (``_parse_batch_response``).

        Returns ``(results_by_intent_id, diagnostics)``. An intent whose id
        never appears as a key in the returned mapping was never addressed
        by the model at all -- the caller (``research.batching``) reads that
        as ``IntentStatus.INCOMPLETE_RESPONSE``, never as zero results.
        Requirement B: when every intent in the batch shares the exact same
        (non-empty) source restriction, it is applied at the tool level; a
        batch mixing different restrictions leaves the tool unrestricted and
        each intent's own restriction is enforced afterward by filtering its
        attributed documents to matching hosts -- never blended across
        intents with incompatible requirements.
        """
        empty_meta = {"server_tool_uses": 0, "prompt_tokens": 0, "output_tokens": 0, "actual_total_tokens": 0}
        usable, reason = self.available()
        if not intents:
            return {}, empty_meta
        if not usable:
            return (
                {
                    intent.intent_id: ResearchResult(
                        query=_query_for_intent(intent),
                        outcome=FetchOutcome.DISABLED,
                        path=self.path,
                        executed=False,
                        error=reason,
                    )
                    for intent in intents
                },
                empty_meta,
            )

        search_tool, _ = tool_types_for(self.llm.model)
        max_uses = min(len(intents) + 1, self.max_uses_per_batch)
        tool: dict[str, Any] = {"type": search_tool, "name": "web_search", "max_uses": max_uses}
        if search_tool == WEB_SEARCH_TOOL:
            tool["allowed_callers"] = _DIRECT_CALLER
        shared_restrictions = _shared_source_restrictions(intents)
        if shared_restrictions:
            tool["allowed_domains"] = list(shared_restrictions)

        prompt = _build_batch_prompt(intents)
        try:
            response = self.llm.raw_message(
                system=BATCH_SEARCH_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                max_tokens=BATCH_SEARCH_MAX_TOKENS,
                agent_id=agent_id,
                effort=self.research_effort,
            )
        except BudgetExceeded as exc:
            log.warning("batched web search skipped: %s", exc)
            error = f"BudgetExceeded: {exc}"
            return (
                {
                    intent.intent_id: ResearchResult(
                        query=_query_for_intent(intent),
                        outcome=FetchOutcome.DISABLED,
                        path=self.path,
                        executed=False,
                        error=error,
                    )
                    for intent in intents
                },
                empty_meta,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            log.warning("batched web search failed: %s", exc)
            error = f"{type(exc).__name__}: {exc}"
            return (
                {
                    intent.intent_id: ResearchResult(
                        query=_query_for_intent(intent), outcome=FetchOutcome.ERROR,
                        path=self.path, error=error,
                    )
                    for intent in intents
                },
                empty_meta,
            )

        documents_by_intent, server_tool_uses = _parse_batch_response(response, intents, path=self.path)
        results: dict[str, ResearchResult] = {}
        for intent in intents:
            if intent.intent_id not in documents_by_intent:
                continue  # never attributed -- see docstring; not guessed at
            documents = documents_by_intent[intent.intent_id]
            if intent.source_restrictions:
                documents = [d for d in documents if _host_matches(d.url, intent.source_restrictions)]
            results[intent.intent_id] = ResearchResult(
                query=_query_for_intent(intent),
                documents=documents[:6],
                path=self.path,
                tokens_used=self.llm.last_usage_tokens,
            )

        last_call = self.llm.budget.calls[-1] if self.llm.budget.calls else None
        meta = {
            "server_tool_uses": server_tool_uses,
            "prompt_tokens": last_call.input_tokens if last_call else 0,
            "output_tokens": last_call.output_tokens if last_call else 0,
            "actual_total_tokens": last_call.total_tokens if last_call else 0,
        }
        return results, meta

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


def _iter_content_blocks(response: Any) -> Sequence[Any]:
    return getattr(response, "content", None) or (
        response.get("content", []) if isinstance(response, dict) else []
    )


def _documents_from_result_content(
    content: Any, *, path: ResearchPath
) -> tuple[list[Document], str]:
    """Turn ONE ``web_search_tool_result`` block's ``content`` into documents.

    Shared by ``parse_search_response`` (one query, one call) and the
    batched-call parser (``_parse_batch_response``) so both apply the exact
    same error/tiering/authority rules to a result block, wherever in a
    response it appears.
    """
    if content is None:
        return [], ""
    if isinstance(content, dict) or not isinstance(content, (list, tuple)):
        code = _get(content, "error_code") or "unknown_error"
        log.warning("web search returned an error block: %s", code)
        return [], f"web_search error: {code}"
    documents: list[Document] = []
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
    return documents, ""


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
    for block in _iter_content_blocks(response):
        if _get(block, "type") != "web_search_tool_result":
            continue
        block_documents, block_error = _documents_from_result_content(_get(block, "content"), path=path)
        documents.extend(block_documents)
        if block_error:
            error = block_error
    return documents, error


def _query_for_intent(intent: ResearchIntent) -> ResearchQuery:
    return ResearchQuery(
        query=intent.question,
        domain=intent.domain,
        stance="bear",
        allowed_domains=intent.source_restrictions,
        rationale=intent.rationale,
    )


def _shared_source_restrictions(intents: Sequence[ResearchIntent]) -> tuple[str, ...]:
    """The single source restriction to apply at the TOOL level for this
    batch, or ``()`` when intents disagree (requirement B/9): incompatible
    restrictions are never blended into one shared filter -- each intent's
    own restriction is instead enforced afterward, per-document, by
    ``_host_matches``.
    """
    restrictions = {intent.source_restrictions for intent in intents if intent.source_restrictions}
    if len(restrictions) == 1:
        return next(iter(restrictions))
    return ()


def _host_matches(url: str, allowed_domains: Sequence[str]) -> bool:
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == domain.lower() or host.endswith("." + domain.lower()) for domain in allowed_domains)


def _build_batch_prompt(intents: Sequence[ResearchIntent]) -> str:
    """One user turn listing every intent in this batch, labelled so the
    response can be parsed back into per-intent evidence (requirement A)."""
    sections = []
    for intent in intents:
        restriction = (
            f"Restrict this search to: {', '.join(intent.source_restrictions)}."
            if intent.source_restrictions
            else "No source restriction for this intent."
        )
        sections.append(
            f"INTENT {intent.intent_id} (domain: {intent.domain}):\n"
            f"{intent.question}\n"
            f"{restriction}"
        )
    return (
        "Research the following intents, one at a time, in the exact order given. For each, "
        "first emit its `-- INTENT <id> --` marker line, then call web_search for it.\n\n"
        + "\n\n".join(sections)
    )


def _parse_batch_response(
    response: Any, intents: Sequence[ResearchIntent], *, path: ResearchPath
) -> tuple[dict[str, list[Document]], int]:
    """Attribute every search result in a batched response to the intent
    whose marker most recently preceded it (requirement A/C).

    Returns ``(documents_by_intent_id, server_tool_uses)``. An intent_id is
    a key in the returned mapping if and only if the model emitted its
    marker at least once -- even with an empty document list, that still
    means "addressed, zero results" (``IntentStatus.EXECUTED_ZERO_RESULTS``),
    distinct from an intent whose id never appears at all (``INCOMPLETE_
    RESPONSE`` -- the caller in ``research.batching`` reads that from a
    plain absent key). A ``web_search_tool_result`` seen before any marker
    is line is never attributed anywhere -- guessing an owner for it would
    defeat the entire point of per-intent auditability.
    """
    known_ids = {intent.intent_id for intent in intents}
    documents_by_intent: dict[str, list[Document]] = {}
    current_id: str | None = None
    server_tool_uses = 0
    for block in _iter_content_blocks(response):
        block_type = _get(block, "type")
        if block_type == "text":
            text = str(_get(block, "text") or "")
            for match in BATCH_INTENT_MARKER_RE.finditer(text):
                marker_id = match.group(1)
                if marker_id in known_ids:
                    current_id = marker_id
                    documents_by_intent.setdefault(current_id, [])
        elif block_type == "server_tool_use" and _get(block, "name") == "web_search":
            server_tool_uses += 1
        elif block_type == "web_search_tool_result":
            if current_id is None:
                log.warning("batched web search result seen before any INTENT marker; dropped")
                continue
            block_documents, _error = _documents_from_result_content(_get(block, "content"), path=path)
            documents_by_intent[current_id].extend(block_documents)
    return documents_by_intent, server_tool_uses
