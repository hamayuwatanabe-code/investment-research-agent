"""SEC EDGAR collector.

Endpoints used (all free, no key; SEC requires a descriptive User-Agent):

* ``https://www.sec.gov/files/company_tickers.json``      ticker -> CIK
* ``https://data.sec.gov/submissions/CIK##########.json`` filing history
* ``https://data.sec.gov/api/xbrl/companyconcept/...``    XBRL facts
* ``https://efts.sec.gov/LATEST/search-index?q=...``      full-text search

Everything produced here is Tier 1 by source, but the *evidence class* still
depends on content: a company's own 8-K narrative is a COMPANY_CLAIM published
in a Tier-1 venue, and the Evidence Integrity agent treats it accordingly.
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas.enums import UNKNOWN, FactCategory, FetchOutcome, Provenance, SourceTier
from ..schemas.fact import RawFact, Source, make_source_id
from .base import CollectionResult
from .http import HttpClient

log = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

#: Filing forms this system cares about, mapped to the fact category they feed.
FORM_CATEGORIES: dict[str, FactCategory] = {
    "10-K": FactCategory.FINANCIAL,
    "10-Q": FactCategory.FINANCIAL,
    "8-K": FactCategory.OTHER,
    "S-1": FactCategory.CAPITAL_STRUCTURE,
    "S-3": FactCategory.CAPITAL_STRUCTURE,
    "S-3ASR": FactCategory.CAPITAL_STRUCTURE,
    "424B3": FactCategory.CAPITAL_STRUCTURE,
    "424B4": FactCategory.CAPITAL_STRUCTURE,
    "424B5": FactCategory.CAPITAL_STRUCTURE,
    "DEF 14A": FactCategory.GOVERNANCE,
    "SC 13D": FactCategory.GOVERNANCE,
    "SC 13G": FactCategory.GOVERNANCE,
    "4": FactCategory.INSIDER,
    "3": FactCategory.INSIDER,
    "5": FactCategory.INSIDER,
    "NT 10-K": FactCategory.ACCOUNTING,
    "NT 10-Q": FactCategory.ACCOUNTING,
    "25-NSE": FactCategory.LISTING,
}

TRACKED_FORMS = tuple(FORM_CATEGORIES)


class SecEdgarCollector:
    name = "sec_edgar"

    def __init__(self, http: HttpClient, *, max_filings: int = 60) -> None:
        self.http = http
        self.max_filings = max_filings

    # -- ticker -> CIK -----------------------------------------------------
    def resolve_cik(self, ticker: str) -> tuple[int | None, str, FetchOutcome]:
        result = self.http.get(TICKER_MAP_URL)
        if not result.ok:
            return None, UNKNOWN, result.outcome
        payload = result.json()
        if not isinstance(payload, dict):
            return None, UNKNOWN, FetchOutcome.ERROR
        wanted = ticker.upper()
        for entry in payload.values():
            if isinstance(entry, dict) and str(entry.get("ticker", "")).upper() == wanted:
                return int(entry["cik_str"]), str(entry.get("title", UNKNOWN)), FetchOutcome.OK
        return None, UNKNOWN, FetchOutcome.NOT_FOUND

    # -- collection --------------------------------------------------------
    def collect(self, ticker: str, company_name: str = UNKNOWN) -> CollectionResult:
        out = CollectionResult(collector=self.name)
        cik, resolved_name, outcome = self.resolve_cik(ticker)
        out.attempted_urls.append(TICKER_MAP_URL)
        if cik is None:
            out.outcome = outcome
            out.errors.append(f"could not resolve CIK for {ticker}: {outcome}")
            out.notes.append(
                "SEC identity not established; every SEC-derived fact is NOT_FOUND for this run"
            )
            return out
        if resolved_name != UNKNOWN:
            out.notes.append(f"resolved_company_name={resolved_name}")

        url = SUBMISSIONS_URL.format(cik=cik)
        out.attempted_urls.append(url)
        submissions = self.http.get(url)
        if not submissions.ok:
            out.outcome = submissions.outcome
            out.errors.append(f"submissions fetch failed: {submissions.outcome} {submissions.error}")
            return out
        payload = submissions.json()
        if not isinstance(payload, dict):
            out.outcome = FetchOutcome.ERROR
            out.errors.append("submissions payload was not a JSON object")
            return out

        out.raw_facts.extend(self._filings_to_facts(ticker, cik, payload, out))
        return out

    def _filings_to_facts(
        self, ticker: str, cik: int, payload: dict[str, Any], out: CollectionResult
    ) -> list[RawFact]:
        recent = (payload.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        report_dates = recent.get("reportDate") or []
        accessions = recent.get("accessionNumber") or []
        documents = recent.get("primaryDocument") or []
        descriptions = recent.get("primaryDocDescription") or []
        items = recent.get("items") or []

        facts: list[RawFact] = []
        for index, form in enumerate(forms[: self.max_filings * 3]):
            if form not in FORM_CATEGORIES:
                continue
            if len(facts) >= self.max_filings:
                break
            accession = accessions[index] if index < len(accessions) else UNKNOWN
            document = documents[index] if index < len(documents) else ""
            filing_date = dates[index] if index < len(dates) else UNKNOWN
            report_date = report_dates[index] if index < len(report_dates) else UNKNOWN
            description = descriptions[index] if index < len(descriptions) else form
            item_codes = items[index] if index < len(items) else ""

            url = FILING_INDEX_URL.format(
                cik=cik,
                accession_nodash=str(accession).replace("-", ""),
                document=document or "",
            )
            source = Source(
                source_id=make_source_id(url, description),
                url=url,
                title=f"{form} {description}".strip(),
                tier=SourceTier.TIER_1,
                publisher="SEC EDGAR",
                filing_date=filing_date or UNKNOWN,
                event_date=report_date or filing_date or UNKNOWN,
                published_date=filing_date or UNKNOWN,
                accession=str(accession),
                provenance=Provenance.LIVE,
            )
            out.sources.append(source)
            claim = f"{form} filed {filing_date}" + (
                f" (items {item_codes})" if item_codes else ""
            ) + (f"; period {report_date}" if report_date else "")
            facts.append(
                RawFact(
                    ticker=ticker.upper(),
                    category=FORM_CATEGORIES[form],
                    claim=claim,
                    source=source,
                    value=form,
                    unit="form_type",
                    company_claim=True,  # the filer authored it
                    collector=self.name,
                )
            )
        return facts

    def company_facts(self, cik: int) -> tuple[dict[str, Any] | None, FetchOutcome]:
        """XBRL company facts (shares outstanding, cash, etc.)."""
        result = self.http.get(COMPANY_FACTS_URL.format(cik=cik))
        return (result.json() if result.ok else None), result.outcome


def extract_xbrl_metric(
    company_facts: dict[str, Any], tag: str, unit: str = "USD"
) -> tuple[float | None, str]:
    """Pull the most recent value of one XBRL tag.

    Returns ``(value, as_of_date)``; ``(None, UNKNOWN)`` when the tag is absent.
    Never interpolates a missing figure.
    """
    facts = ((company_facts or {}).get("facts") or {}).get("us-gaap") or {}
    entry = facts.get(tag)
    if not entry:
        return None, UNKNOWN
    units: dict[str, list[dict[str, Any]]] = entry.get("units") or {}
    series: list[dict[str, Any]] = units.get(unit) or next(iter(units.values()), [])
    dated = [item for item in series if item.get("end")]
    if not dated:
        return None, UNKNOWN
    latest = max(dated, key=lambda item: str(item.get("end")))
    try:
        return float(latest["val"]), str(latest.get("end", UNKNOWN))
    except (KeyError, TypeError, ValueError):
        return None, UNKNOWN
