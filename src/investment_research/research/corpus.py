"""Captured-corpus research provider.

Replays real documents about a real issuer that were captured at a stated time
through a stated channel. This is neither live research nor synthetic fixture
data, and it is labelled as its own thing so it can never be mistaken for
either: ``Provenance.CAPTURED``, with the capture timestamp carried into the
report.

Its purpose is to make the extraction and analysis path testable on genuine
prose. A pattern that finds "does not consider the primary endpoint appropriate"
in a synthetic fixture proves very little; finding "the FDA no longer refers to
the trial as pivotal" in real wording proves the thing that matters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..collectors.documents import Document
from ..collectors.tiering import classify_tier
from ..schemas.enums import (
    UNKNOWN,
    ContentKind,
    FetchOutcome,
    Provenance,
    ResearchPath,
)
from .provider import ResearchQuery, ResearchResult

log = logging.getLogger(__name__)


class CorpusResearchProvider:
    """Serves documents from ``data/corpus/<TICKER>.json``."""

    name = "corpus"
    path = ResearchPath.CORPUS

    def __init__(self, corpus_dir: str | Path, ticker: str = "") -> None:
        self.corpus_dir = Path(corpus_dir)
        self.ticker = ticker.upper()
        self._payload: dict[str, Any] | None = None

    # -- loading -----------------------------------------------------------
    def available_tickers(self) -> list[str]:
        return sorted(p.stem.upper() for p in self.corpus_dir.glob("*.json"))

    def load(self, ticker: str | None = None) -> dict[str, Any] | None:
        ticker = (ticker or self.ticker).upper()
        path = self.corpus_dir / f"{ticker}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.error("malformed corpus %s: %s", path, exc)
            return None
        self._payload = payload
        return payload

    def metadata(self, ticker: str | None = None) -> dict[str, Any]:
        payload = self._payload or self.load(ticker)
        return (payload or {}).get("company", {})

    def capture_info(self, ticker: str | None = None) -> dict[str, Any]:
        payload = self._payload or self.load(ticker)
        if not payload:
            return {}
        return {
            "captured_at": payload.get("captured_at", UNKNOWN),
            "captured_via": payload.get("captured_via", UNKNOWN),
            "document_count": len(payload.get("documents", [])),
        }

    def documents(self, ticker: str | None = None) -> list[Document]:
        payload = self._payload or self.load(ticker)
        if not payload:
            return []
        captured_at = payload.get("captured_at", UNKNOWN)
        out: list[Document] = []
        for entry in payload.get("documents", []):
            url = entry.get("url", "")
            if not url.startswith("http"):
                log.warning("corpus document without a real URL skipped: %r", url)
                continue
            out.append(
                Document(
                    doc_id=entry.get("doc_id") or url,
                    url=url,
                    title=entry.get("title", UNKNOWN),
                    publisher=entry.get("publisher", UNKNOWN),
                    published_date=entry.get("published_date", UNKNOWN),
                    event_date=entry.get("event_date", UNKNOWN),
                    filing_date=entry.get("filing_date", UNKNOWN),
                    doc_type=entry.get("doc_type", "unknown"),
                    is_company_ir=bool(entry.get("is_company_ir", False)),
                    text=entry.get("text", ""),
                    content_kind=ContentKind(entry.get("content_kind", "SEARCH_SUMMARY")),
                    provenance=Provenance.CAPTURED,
                    research_path=ResearchPath.CORPUS,
                    retrieved_at=captured_at,
                    tier=classify_tier(url, is_company_ir=bool(entry.get("is_company_ir"))),
                )
            )
        return out

    # -- provider interface ------------------------------------------------
    def available(self) -> tuple[bool, str]:
        if not self.corpus_dir.is_dir():
            return False, f"no corpus directory at {self.corpus_dir}"
        if self.ticker and not (self.corpus_dir / f"{self.ticker}.json").is_file():
            return False, f"no captured corpus for {self.ticker}"
        return True, "corpus available"

    def search(self, query: ResearchQuery, *, agent_id: str = "research") -> ResearchResult:
        """Keyword-match the query against the captured documents.

        A corpus search is genuinely a search -- it can return nothing -- but it
        cannot discover anything that was not captured. That limit is reported
        rather than hidden: the domain coverage records CORPUS as the path, and
        the report states the capture time.
        """
        documents = self.documents()
        if not documents:
            return ResearchResult(
                query=query,
                outcome=FetchOutcome.NOT_FOUND,
                path=self.path,
                error=f"no captured corpus for {self.ticker}",
            )
        terms = [t for t in _terms(query.query) if len(t) > 3]
        scored: list[tuple[int, Document]] = []
        for document in documents:
            haystack = f"{document.title} {document.text}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, document))
        scored.sort(key=lambda pair: (-pair[0], pair[1].doc_id))
        matched = [document for _, document in scored[: query.max_results]]
        return ResearchResult(
            query=query,
            documents=matched,
            outcome=FetchOutcome.OK if matched else FetchOutcome.NOT_FOUND,
            path=self.path,
        )

    def fetch(
        self, url: str, *, reason: str = "", agent_id: str = "research", known: Any = None
    ) -> Document | None:
        # Captured documents already carry their real, captured metadata --
        # `known` (an already-collected Source/Document pointer) has nothing
        # to add here that the corpus entry does not already have.
        for document in self.documents():
            if document.url == url:
                return document
        return None


def _terms(query: str) -> list[str]:
    import re

    return [t for t in re.split(r"[^A-Za-z0-9]+", query.lower()) if t]
