"""ClinicalTrials.gov (API v2) collector.

Trial design detail is where a biotech thesis usually lives or dies, so this
collector extracts *design*, not just existence: randomization, masking,
comparator, enrollment, and the primary outcome measure verbatim.  The verbatim
primary endpoint matters because the failure this system is built to prevent was
a mismatch between the company's endpoint and the regulator's view of it.
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas.enums import UNKNOWN, FactCategory, FetchOutcome, Provenance, SourceTier
from ..schemas.fact import RawFact, Source, make_source_id
from .base import CollectionResult
from .http import HttpClient

log = logging.getLogger(__name__)

STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"


class ClinicalTrialsCollector:
    name = "clinicaltrials"

    def __init__(self, http: HttpClient, *, page_size: int = 25) -> None:
        self.http = http
        self.page_size = page_size

    def collect(self, ticker: str, company_name: str = UNKNOWN) -> CollectionResult:
        out = CollectionResult(collector=self.name)
        if not company_name or company_name == UNKNOWN:
            out.outcome = FetchOutcome.NOT_FOUND
            out.errors.append(
                "no company name available; sponsor search skipped rather than guessed"
            )
            return out

        params = {
            "query.spons": company_name,
            "pageSize": self.page_size,
            "format": "json",
        }
        result = self.http.get(STUDIES_URL, params=params)
        out.attempted_urls.append(result.url)
        if not result.ok:
            out.outcome = result.outcome
            out.errors.append(f"studies fetch failed: {result.outcome} {result.error}")
            return out
        payload = result.json()
        if not isinstance(payload, dict):
            out.outcome = FetchOutcome.ERROR
            out.errors.append("studies payload was not a JSON object")
            return out

        studies = payload.get("studies") or []
        if not studies:
            out.outcome = FetchOutcome.NOT_FOUND
            out.notes.append(f"no registered studies found for sponsor {company_name!r}")
            return out

        for study in studies:
            parsed = parse_study(study)
            if not parsed.get("nct_id"):
                continue
            url = STUDY_URL.format(nct_id=parsed["nct_id"])
            source = Source(
                source_id=make_source_id(url, parsed.get("title", "")),
                url=url,
                title=parsed.get("title", "") or parsed["nct_id"],
                tier=SourceTier.TIER_1,
                publisher="ClinicalTrials.gov",
                published_date=parsed.get("last_update", UNKNOWN),
                event_date=parsed.get("primary_completion", UNKNOWN),
                provenance=Provenance.LIVE,
            )
            out.sources.append(source)
            for claim, value, unit in _study_claims(parsed):
                out.raw_facts.append(
                    RawFact(
                        ticker=ticker.upper(),
                        category=FactCategory.CLINICAL,
                        claim=claim,
                        source=source,
                        value=value,
                        unit=unit,
                        company_claim=True,  # registry entries are sponsor-authored
                        collector=self.name,
                    )
                )
        return out


def parse_study(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten one API v2 study record.  Absent fields stay ``UNKNOWN``."""
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    design = protocol.get("designModule") or {}
    outcomes = protocol.get("outcomesModule") or {}
    sponsor = protocol.get("sponsorCollaboratorsModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}

    design_info = design.get("designInfo") or {}
    masking = (design_info.get("maskingInfo") or {}).get("masking", UNKNOWN)
    enrollment_info = design.get("enrollmentInfo") or {}
    primary_outcomes = outcomes.get("primaryOutcomes") or []
    secondary_outcomes = outcomes.get("secondaryOutcomes") or []
    phases = design.get("phases") or []

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", UNKNOWN),
        "sponsor": ((sponsor.get("leadSponsor") or {}).get("name", UNKNOWN)),
        "status": status.get("overallStatus", UNKNOWN),
        "phase": ",".join(phases) if phases else UNKNOWN,
        "enrollment": enrollment_info.get("count"),
        "enrollment_type": enrollment_info.get("type", UNKNOWN),
        "allocation": design_info.get("allocation", UNKNOWN),
        "masking": masking or UNKNOWN,
        "primary_purpose": design_info.get("primaryPurpose", UNKNOWN),
        "intervention_model": design_info.get("interventionModel", UNKNOWN),
        "primary_endpoint": (
            primary_outcomes[0].get("measure", UNKNOWN) if primary_outcomes else UNKNOWN
        ),
        "primary_endpoint_timeframe": (
            primary_outcomes[0].get("timeFrame", UNKNOWN) if primary_outcomes else UNKNOWN
        ),
        "secondary_endpoints": [o.get("measure", "") for o in secondary_outcomes],
        "arms": [a.get("label", "") for a in (arms.get("armGroups") or [])],
        "primary_completion": (status.get("primaryCompletionDateStruct") or {}).get(
            "date", UNKNOWN
        ),
        "primary_completion_type": (status.get("primaryCompletionDateStruct") or {}).get(
            "type", UNKNOWN
        ),
        "last_update": (status.get("lastUpdatePostDateStruct") or {}).get("date", UNKNOWN),
        "start_date": (status.get("startDateStruct") or {}).get("date", UNKNOWN),
    }


def _study_claims(parsed: dict[str, Any]) -> list[tuple[str, Any, str]]:
    nct = parsed["nct_id"]
    claims: list[tuple[str, Any, str]] = [
        (f"{nct} phase is {parsed['phase']}", parsed["phase"], "phase"),
        (f"{nct} overall status is {parsed['status']}", parsed["status"], "status"),
        (
            f"{nct} primary outcome measure is: {parsed['primary_endpoint']}",
            parsed["primary_endpoint"],
            "endpoint",
        ),
        (
            f"{nct} allocation is {parsed['allocation']}, masking is {parsed['masking']}",
            f"{parsed['allocation']}/{parsed['masking']}",
            "design",
        ),
    ]
    if parsed.get("enrollment") is not None:
        claims.append(
            (
                f"{nct} enrollment is {parsed['enrollment']} ({parsed['enrollment_type']})",
                parsed["enrollment"],
                "participants",
            )
        )
    if parsed.get("primary_completion") != UNKNOWN:
        claims.append(
            (
                f"{nct} primary completion date is {parsed['primary_completion']} "
                f"({parsed['primary_completion_type']})",
                parsed["primary_completion"],
                "date",
            )
        )
    return claims
