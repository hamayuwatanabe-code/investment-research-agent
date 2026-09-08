"""FDA collectors (openFDA + Drugs@FDA style documents).

Scope note, stated plainly because it matters for how the output is read:
openFDA exposes labels, approvals, recalls and adverse events.  It does **not**
expose meeting minutes, Type A/B/C correspondence, SPAs or Complete Response
Letters.  Those are the documents that decide whether an endpoint is acceptable
-- exactly the information whose absence caused the failure this system exists
to prevent.

So this collector reports what it can obtain and, for everything it cannot,
emits an explicit unresolved question rather than an optimistic silence.
"""

from __future__ import annotations

import logging

from ..schemas.enums import UNKNOWN, FactCategory, FetchOutcome, Provenance, SourceTier
from ..schemas.fact import RawFact, Source, make_source_id
from .base import CollectionResult
from .http import HttpClient

log = logging.getLogger(__name__)

DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
LABEL_URL = "https://api.fda.gov/drug/label.json"
ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"
DEVICE_PMA_URL = "https://api.fda.gov/device/pma.json"

#: Regulatory questions openFDA cannot answer.  Surfaced as unresolved questions
#: so the report shows the hole instead of hiding it.
NON_API_REGULATORY_QUESTIONS = (
    "Has FDA agreed that the primary endpoint is adequate to establish effectiveness?",
    "What did FDA agree to, and explicitly not agree to, at the most recent Type A/B/C meeting?",
    "Is there a Special Protocol Assessment, and is it still in force?",
    "Has the programme ever received a Complete Response Letter or a clinical hold?",
    "What CMC or manufacturing issues remain open?",
    "Does FDA consider the proposed surrogate endpoint reasonably likely to predict benefit?",
)


class FdaCollector:
    name = "fda"

    def __init__(self, http: HttpClient, *, limit: int = 20) -> None:
        self.http = http
        self.limit = limit

    def collect(self, ticker: str, company_name: str = UNKNOWN) -> CollectionResult:
        out = CollectionResult(collector=self.name)
        if not company_name or company_name == UNKNOWN:
            out.outcome = FetchOutcome.NOT_FOUND
            out.errors.append("no company name available; FDA sponsor search skipped")
            return out

        search = f'sponsor_name:"{company_name}"'
        result = self.http.get(DRUGSFDA_URL, params={"search": search, "limit": self.limit})
        out.attempted_urls.append(result.url)
        if result.outcome == FetchOutcome.NOT_FOUND:
            # Drugs@FDA's own normal response for "no matching application"
            # is an HTTP 404 -- that is this API's documented zero-result
            # shape, not a network or server failure. This translation is
            # made explicitly, only here, at this collector's own semantic
            # boundary: HttpClient's NOT_FOUND stays a generic, collector-
            # agnostic outcome, and CollectionResult.degraded is never
            # globally taught that NOT_FOUND means success. A 404 from any
            # other collector is unaffected and still degrades as before.
            out.outcome = FetchOutcome.OK
            out.zero_results = True
            out.notes.append(
                f"ZERO_RESULTS: no Drugs@FDA applications listed for {company_name!r}. "
                "This means only that Drugs@FDA has no matching application on file for "
                "this sponsor -- it does NOT resolve whether FDA has had any other "
                "interaction with this programme. Type A/B/C meetings, endpoint "
                "acceptability, a Special Protocol Assessment, CMC issues, and a prior "
                "clinical hold all remain unresolved unless separately researched."
            )
            return out
        if not result.ok:
            out.outcome = result.outcome
            out.errors.append(f"drugsfda fetch failed: {result.outcome} {result.error}")
            return out

        payload = result.json() or {}
        for record in payload.get("results", []):
            application = record.get("application_number", UNKNOWN)
            url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={application}"
            source = Source(
                source_id=make_source_id(url, application),
                url=url,
                title=f"Drugs@FDA {application}",
                tier=SourceTier.TIER_1,
                publisher="FDA",
                provenance=Provenance.LIVE,
            )
            out.sources.append(source)
            for product in record.get("products", []) or [{}]:
                out.raw_facts.append(
                    RawFact(
                        ticker=ticker.upper(),
                        category=FactCategory.REGULATORY,
                        claim=(
                            f"{application}: {product.get('brand_name', UNKNOWN)} "
                            f"marketing status {product.get('marketing_status', UNKNOWN)}"
                        ),
                        source=source,
                        value=product.get("marketing_status", UNKNOWN),
                        unit="marketing_status",
                        company_claim=False,
                        collector=self.name,
                    )
                )
        return out
