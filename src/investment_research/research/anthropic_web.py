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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..collectors.documents import Document
from ..collectors.tiering import classify_authority, classify_tier
from ..llm.client import (
    SERVER_TOOL_TOKEN_RESERVE,
    SERVER_TOOL_TOKEN_RESERVE_LOW_EFFORT,
    BudgetExceeded,
    LLMBudget,
)
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
        # Set on every `fetch()` call, right before it returns None -- the
        # only way a caller (research/escalation.py's fetch audit) can tell
        # a PRE-SEND rejection (provider unavailable, preflight
        # BudgetExceeded: `last_fetch_sent=False`, never reached the network)
        # apart from a genuine SEND-then-fail (an actual API/SDK exception,
        # or a sent request whose response carried no usable text/an error
        # block: `last_fetch_sent=True`). A successful fetch resets both to
        # their "nothing to report" defaults. Mirrors the existing
        # `last_usage_tokens` convention on `LLMClient`.
        self.last_fetch_error: str = ""
        self.last_fetch_sent: bool = True

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

        Returns ``(results_by_intent_id, diagnostics)``. Two DISTINCT reasons
        an intent can end up not-executed, and this method never conflates
        them: (1) it was excluded from this call entirely by the
        split/allocate budget decision (``_affordable_prefix``) -- a fact
        this method knows for CERTAIN, since it never sent it -- and comes
        back as an explicit, ``executed=False`` result whose error starts
        ``"BudgetExceeded: excluded ... by split/allocate"``, which the
        caller (``research.batching``) reads as
        ``IntentStatus.SKIPPED_DUE_TO_BUDGET``; (2) it WAS included in the
        request but the model's response never actually addressed it (no
        marker, no search, or the response was truncated) -- this is the
        only case whose id is absent from the returned mapping entirely,
        which the caller reads as ``IntentStatus.INCOMPLETE_RESPONSE``. Only
        as many of ``intents`` as a single call can honestly afford, given
        what actually remains of the global AND current-stage budget, are
        ever included in the request -- never all of them on the hope that
        the real cost will happen to fit. Requirement B: when every intent
        in the batch shares the exact same (non-empty) source restriction,
        it is applied at the tool level; a batch mixing different
        restrictions leaves the tool unrestricted and each intent's own
        restriction is enforced afterward by filtering its attributed
        documents to matching hosts -- never blended across intents with
        incompatible requirements.
        """
        empty_meta = {
            "server_tool_uses": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "actual_total_tokens": 0,
            "ambiguous_results": 0,
        }
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

        # Split/allocate (never a flat guess): only as many of `intents` as a
        # single call can honestly afford, given what actually remains of the
        # global AND the current stage's budget -- see _affordable_prefix.
        # Intents beyond the returned prefix are NEVER sent this call. That
        # exclusion is a structural fact this method already knows for
        # certain -- it must come back as an explicit, per-intent
        # SKIPPED_DUE_TO_BUDGET-shaped result, never silently absent from
        # `results` (absence reads as INCOMPLETE_RESPONSE downstream, which
        # is reserved for an intent that WAS sent but whose response never
        # arrived -- a genuinely different, less certain situation).
        included = _affordable_prefix(
            intents, self.llm.budget, effort=self.research_effort, max_uses_per_batch=self.max_uses_per_batch
        )
        included_ids = {intent.intent_id for intent in included}
        deferred = [intent for intent in intents if intent.intent_id not in included_ids]

        results: dict[str, ResearchResult] = {}
        if deferred:
            deferred_error = (
                f"BudgetExceeded: excluded from this call by split/allocate -- did not fit the "
                f"remaining budget at dispatch time ({self.llm.budget.remaining} global token(s) "
                f"/ {self.llm.budget.stage_remaining(self.llm.budget.current_stage)} stage "
                "token(s) left). Never sent -- distinct from a sent intent whose response never "
                "arrived. The caller may re-evaluate it against whatever budget remains once "
                "this call's actual usage is recorded."
            )
            log.info(
                "batched web search: %d of %d intent(s) excluded by split/allocate: %s",
                len(deferred), len(intents), [intent.intent_id for intent in deferred],
            )
            for intent in deferred:
                results[intent.intent_id] = ResearchResult(
                    query=_query_for_intent(intent),
                    outcome=FetchOutcome.DISABLED,
                    path=self.path,
                    executed=False,
                    error=deferred_error,
                )

        if not included:
            log.warning("batched web search skipped: no intent fits the remaining budget")
            return results, empty_meta

        search_tool, _ = tool_types_for(self.llm.model)
        max_uses = min(len(included) + 1, self.max_uses_per_batch)
        tool: dict[str, Any] = {"type": search_tool, "name": "web_search", "max_uses": max_uses}
        if search_tool == WEB_SEARCH_TOOL:
            tool["allowed_callers"] = _DIRECT_CALLER
        shared_restrictions = _shared_source_restrictions(included)
        if shared_restrictions:
            tool["allowed_domains"] = list(shared_restrictions)

        prompt = _build_batch_prompt(included)
        reserved_tokens = _batch_reservation(included, max_uses=max_uses, effort=self.research_effort)
        try:
            response = self.llm.raw_message(
                system=BATCH_SEARCH_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                max_tokens=BATCH_SEARCH_MAX_TOKENS,
                agent_id=agent_id,
                effort=self.research_effort,
                reserved_tokens=reserved_tokens,
            )
        except BudgetExceeded as exc:
            log.warning("batched web search skipped: %s", exc)
            error = f"BudgetExceeded: {exc}"
            for intent in included:
                results[intent.intent_id] = ResearchResult(
                    query=_query_for_intent(intent),
                    outcome=FetchOutcome.DISABLED,
                    path=self.path,
                    executed=False,
                    error=error,
                )
            return results, empty_meta
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            log.warning("batched web search failed: %s", exc)
            error = f"{type(exc).__name__}: {exc}"
            for intent in included:
                results[intent.intent_id] = ResearchResult(
                    query=_query_for_intent(intent), outcome=FetchOutcome.ERROR,
                    path=self.path, error=error,
                )
            return results, empty_meta

        parsed = _parse_batch_response(response, included, path=self.path)
        for intent in included:
            issued = parsed.issued_tool_use_ids.get(intent.intent_id, set())
            unresolved = issued - parsed.resolved_tool_use_ids
            if intent.intent_id not in parsed.search_completed and not unresolved:
                # No REAL web_search_tool_result block was ever attributed
                # to this intent, and there is no outstanding issued call
                # either -- the model never actually addressed it at all
                # (requirement 1).
                continue
            documents = parsed.documents_by_intent.get(intent.intent_id, [])
            if intent.source_restrictions:
                documents = [d for d in documents if _host_matches(d.url, intent.source_restrictions)]
            error = parsed.error_by_intent.get(intent.intent_id, "")
            if unresolved:
                # At least one web_search call ISSUED for this intent never
                # got a result block back before the response ended -- even
                # though a different search for the SAME intent may have
                # resolved and produced the documents below. Never read as
                # complete success; any evidence already obtained survives.
                incomplete_note = f"IncompleteIntent: {len(unresolved)} unresolved web_search call(s)"
                error = f"{incomplete_note}; {error}" if error else incomplete_note
                results[intent.intent_id] = ResearchResult(
                    query=_query_for_intent(intent),
                    documents=documents[:6],
                    outcome=FetchOutcome.ERROR,
                    path=self.path,
                    executed=False,
                    error=error,
                    tokens_used=self.llm.last_usage_tokens,
                )
                continue
            results[intent.intent_id] = ResearchResult(
                query=_query_for_intent(intent),
                documents=documents[:6],
                outcome=FetchOutcome.ERROR if error else FetchOutcome.OK,
                path=self.path,
                error=error,
                tokens_used=self.llm.last_usage_tokens,
            )
        server_tool_uses = parsed.server_tool_uses

        last_call = self.llm.budget.calls[-1] if self.llm.budget.calls else None
        meta = {
            "server_tool_uses": server_tool_uses,
            "prompt_tokens": last_call.input_tokens if last_call else 0,
            "output_tokens": last_call.output_tokens if last_call else 0,
            "actual_total_tokens": last_call.total_tokens if last_call else 0,
            "ambiguous_results": parsed.ambiguous_results,
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
        usable, reason = self.available()
        if not usable:
            self.last_fetch_sent = False
            self.last_fetch_error = f"provider unavailable: {reason}"
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
            self.last_fetch_sent = False
            self.last_fetch_error = f"BudgetExceeded: {exc}"
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("web fetch failed for %s: %s", url, exc)
            self.last_fetch_sent = True
            self.last_fetch_error = f"{type(exc).__name__}: {exc}"
            return None

        # `retrieved` here is when Anthropic's server retrieved the page --
        # retrieval time, same axis as our own retrieved_at below. It is
        # deliberately NEVER used as published_date/event_date/etc.
        text, fetched_title, retrieved = _extract_fetch_result(response)
        if not text:
            # The request WAS sent and answered -- an empty/error result is
            # a genuine send-then-fail, never a pre-send rejection.
            self.last_fetch_sent = True
            self.last_fetch_error = "web_fetch returned no usable text (tool error or empty result)"
            return None

        self.last_fetch_sent = True
        self.last_fetch_error = ""
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

    Missing/``None`` content and a malformed (non-list, non-error-dict)
    shape are never read as a genuine zero-result search -- only an actual
    ``content == []`` (a well-formed, empty list) is. Conflating the two
    would let a broken or truncated response silently pass as "searched,
    nothing found".
    """
    if content is None:
        log.warning("web search result block has no content (missing/None)")
        return [], "web_search error: missing result content"
    if isinstance(content, dict) or not isinstance(content, (list, tuple)):
        # Either a plain error dict, or the real SDK's typed error object
        # (e.g. ``WebSearchToolResultError``) -- both expose ``error_code``.
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
    batch, or ``()`` when intents disagree (requirement 3/9).

    A tool-level restriction applies to EVERY search in the call, including
    one issued for an intent that itself has no restriction at all -- so it
    may only be set when EVERY intent in the batch shares the exact same
    NON-EMPTY restriction. An unrestricted intent mixed in with restricted
    ones (``[("sec.gov",), ()]``) must never have its own search narrowed
    just because a sibling intent happens to want ``sec.gov`` -- that
    silently over-restricts the unrestricted intent's search. Whenever
    intents disagree (any two distinct values, empty included), this
    returns ``()`` and each intent's own restriction is instead enforced
    afterward, per-document, by ``_host_matches``.
    """
    distinct = {intent.source_restrictions for intent in intents}
    if len(distinct) != 1:
        return ()
    (only,) = distinct
    return only


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


#: Same conservative chars/token heuristic ``LLMClient._default_reservation``
#: uses, kept local so this module's own reservation math doesn't reach into
#: a client-private constant.
_CHARS_PER_TOKEN = 4


def _batch_reservation(intents: Sequence[ResearchIntent], *, max_uses: int, effort: str) -> int:
    """Conservative preflight token estimate for a batched call over exactly
    ``intents``, honest about how many searches ``max_uses`` actually allows.

    A live production run showed a single 6-intent batched call (``max_uses``
    up to 7) spend ~102,000 actual tokens against a 60,000-token discovery
    stage quota -- because the preflight reservation this module used to pass
    to ``raw_message`` was a FLAT one-search reserve
    (``SERVER_TOOL_TOKEN_RESERVE[_LOW_EFFORT]``), regardless of how many
    searches the request's own ``max_uses`` ceiling permitted. That flat
    reserve was calibrated (see its own docstring) against a single search,
    before batching existed; batching several searches into one call to cut
    the CALL count does not cut the TOKEN count, since real cost tracks the
    number of searches actually issued, not the number of API calls that
    contain them. Scaling the reserve by ``max_uses`` here makes the
    preflight check (``LLMBudget.check``) honestly reject a batch whose
    worst-case real cost would not fit, BEFORE anything is sent, instead of
    discovering the overshoot only after the fact via ``LLMBudget.record``'s
    post-hoc stage-exhaustion latch (by which point the tokens are already
    spent and every other required domain this stage still owes a search to
    is starved).
    """
    per_search = SERVER_TOOL_TOKEN_RESERVE_LOW_EFFORT if effort == "low" else SERVER_TOOL_TOKEN_RESERVE
    prompt = _build_batch_prompt(intents)
    prompt_chars = len(BATCH_SEARCH_SYSTEM) + len(prompt)
    return prompt_chars // _CHARS_PER_TOKEN + BATCH_SEARCH_MAX_TOKENS + max_uses * per_search


def _affordable_prefix(
    intents: Sequence[ResearchIntent],
    budget: LLMBudget,
    *,
    effort: str,
    max_uses_per_batch: int,
) -> list[ResearchIntent]:
    """The longest PREFIX of ``intents`` (priority order preserved) whose
    scaled reservation (see ``_batch_reservation``) fits what actually
    remains of both the global budget and the current stage's own quota.

    This is the split/allocate half of the same fix: rather than gambling
    the whole batch on one oversized call (silently starving every OTHER
    required domain's search once that one call blows the stage's real
    quota), a batch that does not fit is served PARTIALLY -- as many of its
    intents as a single call can honestly afford -- so the stage's budget
    gets spent across several smaller, successful calls instead of one call
    that either wildly overshoots or (once ``_batch_reservation`` is honest
    about its cost) gets rejected outright. Intents beyond the returned
    prefix are simply never included in this call; the caller
    (``research.batching``) reads their absence from the response as
    ``IntentStatus.INCOMPLETE_RESPONSE``, exactly as it already does for an
    intent the model omitted from its reply -- never inferred as a
    completed, zero-result search. An empty list means not even a
    single-intent call fits right now.

    Deliberately NOT a claim that this eliminates overshoot risk entirely --
    a server-side tool's real cost is still not knowable before the response
    comes back (see ``BudgetExceeded``'s own docstring); this only makes the
    preflight estimate track batch size instead of ignoring it.
    """
    stage_remaining = budget.stage_remaining(budget.current_stage)
    limit = budget.remaining if stage_remaining is None else min(budget.remaining, stage_remaining)
    for count in range(len(intents), 0, -1):
        prefix = list(intents[:count])
        max_uses = min(count + 1, max_uses_per_batch)
        if _batch_reservation(prefix, max_uses=max_uses, effort=effort) <= limit:
            return prefix
    return []


@dataclass
class _BatchParseResult:
    #: Documents actually attributed to each intent (may be empty for an
    #: intent whose search genuinely completed with zero results).
    documents_by_intent: dict[str, list[Document]] = field(default_factory=dict)
    #: An intent_id is a member of this set if and only if at least one
    #: REAL ``web_search_tool_result`` block was attributed to it -- a
    #: marker, or a ``server_tool_use`` request with no matching result
    #: block, is never enough on its own (requirement 1). The caller reads
    #: absence from this set as ``IntentStatus.INCOMPLETE_RESPONSE``,
    #: regardless of whether ``documents_by_intent`` happens to have a key
    #: for it. Membership here does NOT by itself mean every search issued
    #: for the intent was resolved -- see ``issued_tool_use_ids`` /
    #: ``resolved_tool_use_ids`` for that.
    search_completed: set[str] = field(default_factory=set)
    #: Tool-error text attributed to each intent (requirement 2) -- kept
    #: alongside any documents that same intent DID get from a different,
    #: successful search, never silently discarded and never allowed to
    #: read as complete, uncomplicated success.
    error_by_intent: dict[str, str] = field(default_factory=dict)
    server_tool_uses: int = 0
    #: Every ``server_tool_use`` (web_search call) id issued while a KNOWN
    #: intent was the active marker, keyed by that intent. Used together
    #: with ``resolved_tool_use_ids`` to detect a search that was ISSUED for
    #: an intent but never got a result block back before the response
    #: ended -- an unresolved call that must never be silently read as that
    #: intent having completed successfully, even when a DIFFERENT search
    #: for the same intent did resolve.
    issued_tool_use_ids: dict[str, set[str]] = field(default_factory=dict)
    #: Every ``tool_use_id`` that a ``web_search_tool_result`` block was
    #: actually seen and successfully attributed for.
    resolved_tool_use_ids: set[str] = field(default_factory=set)
    #: Result blocks that could not be attributed to any intent at all --
    #: no resolvable ``tool_use_id``, or a ``tool_use_id`` that does not
    #: resolve to a known intent. Never guessed at via marker order; only
    #: counted, for diagnostics (requirement 1).
    ambiguous_results: int = 0


def _parse_batch_response(
    response: Any, intents: Sequence[ResearchIntent], *, path: ResearchPath
) -> _BatchParseResult:
    """Attribute every search result in a batched response to the intent
    that actually requested it (requirements 1/2/4).

    Attribution is ID-based ONLY: a ``server_tool_use`` block carries its own
    ``id``; the ``web_search_tool_result`` block that answers it carries a
    matching ``tool_use_id``. Recording ``tool_use_id -> intent`` as each
    search is issued (only while a KNOWN intent marker is active) and
    looking a result's ``tool_use_id`` up in that map is correct even when
    result blocks arrive in a different order than their searches were
    issued. A result block with no ``tool_use_id`` at all, or whose
    ``tool_use_id`` fails to resolve to a known intent, is NEVER guessed at
    by falling back to marker order -- genuine ambiguity (e.g. "search A ->
    search B -> an id-less result") is recorded as such (``ambiguous_results``)
    and the result is dropped, not assigned to whichever intent happened to
    be current.

    An UNKNOWN marker (an id the batch never asked for) invalidates the
    current attribution context entirely: a ``server_tool_use`` seen after
    it is never recorded as issued for whichever KNOWN intent came before
    the unknown marker (requirement 4).

    Every issued ``server_tool_use`` id is tracked per intent
    (``issued_tool_use_ids``) alongside every id a result block actually
    resolved (``resolved_tool_use_ids``), so the caller can tell a search
    that never got a result back -- even when a DIFFERENT search for the
    SAME intent did succeed -- from a genuinely fully-resolved intent.
    """
    known_ids = {intent.intent_id for intent in intents}
    out = _BatchParseResult()
    current_id: str | None = None
    tool_use_id_to_intent: dict[str, str] = {}

    for block in _iter_content_blocks(response):
        block_type = _get(block, "type")
        if block_type == "text":
            text = str(_get(block, "text") or "")
            for match in BATCH_INTENT_MARKER_RE.finditer(text):
                marker_id = match.group(1)
                if marker_id in known_ids:
                    current_id = marker_id
                else:
                    log.warning(
                        "batched web search: unknown INTENT marker %r; results are not "
                        "attributed to the prior intent until the next known marker",
                        marker_id,
                    )
                    current_id = None
        elif block_type == "server_tool_use" and _get(block, "name") == "web_search":
            out.server_tool_uses += 1
            tool_use_id = _get(block, "id")
            if tool_use_id and current_id is not None:
                tool_use_id_to_intent[str(tool_use_id)] = current_id
                out.issued_tool_use_ids.setdefault(current_id, set()).add(str(tool_use_id))
        elif block_type == "web_search_tool_result":
            result_tool_use_id = _get(block, "tool_use_id")
            if not result_tool_use_id:
                # No id on the result at all -- attribution genuinely cannot
                # be determined (e.g. search A -> search B -> an id-less
                # result). Never guessed via marker order; recorded as
                # ambiguous and dropped.
                log.warning(
                    "batched web search result has no tool_use_id; attribution is ambiguous, "
                    "dropped rather than guessed"
                )
                out.ambiguous_results += 1
                continue
            target_id = tool_use_id_to_intent.get(str(result_tool_use_id))
            if target_id is None:
                log.warning(
                    "batched web search result tool_use_id %r does not resolve to a known "
                    "intent; dropped rather than guessed",
                    result_tool_use_id,
                )
                out.ambiguous_results += 1
                continue

            out.resolved_tool_use_ids.add(str(result_tool_use_id))
            out.search_completed.add(target_id)
            block_documents, block_error = _documents_from_result_content(_get(block, "content"), path=path)
            out.documents_by_intent.setdefault(target_id, [])
            out.documents_by_intent[target_id].extend(block_documents)
            if block_error:
                existing = out.error_by_intent.get(target_id, "")
                out.error_by_intent[target_id] = f"{existing}; {block_error}" if existing else block_error
    return out
