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
#: The API v2 SINGLE-study endpoint -- returns one study record's JSON
#: directly (the same per-study shape ``parse_study`` already expects for
#: an item of ``STUDIES_URL``'s ``studies`` array), not wrapped in a
#: ``{"studies": [...]}`` envelope. Used when the NCT ID is already known,
#: so no sponsor-name search (and no Web Search) is needed at all -- see
#: ``research/clinicaltrials_acquisition_adapter.py``.
STUDY_DETAIL_URL = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
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
            out.raw_facts.extend(raw_facts_from_study(ticker, parsed, source, collector=self.name))
        return out


def _date_struct(module: dict[str, Any], key: str) -> tuple[str, str]:
    """A CT.gov ``XDateStruct`` (``{"date": ..., "type": "ACTUAL"|"ESTIMATED"}``)
    -- returns ``(date, type)``, both ``UNKNOWN`` if the struct or the date
    itself is absent. The date string is passed through verbatim, including
    a partial (year-only or year-month) value -- never padded, guessed, or
    completed into a full date that CT.gov itself did not assert."""
    struct = module.get(key) or {}
    return struct.get("date", UNKNOWN), struct.get("type", UNKNOWN)


def parse_study(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten one API v2 study record.  Absent fields stay ``UNKNOWN`` (or,
    for a genuinely list-shaped field, an empty list) -- never guessed, and
    never crashes on a field CT.gov omitted or a value of the wrong shape.

    Every date returned here is the SPECIFIC registered date it names
    (start/primary-completion/completion/last-update-submit/last-update-
    post) -- never conflated with each other and never with retrieval time,
    which this function has no opinion about at all (callers set
    ``retrieved_at``/``Document.retrieved_at`` separately, never from one of
    these).
    """
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    design = protocol.get("designModule") or {}
    outcomes = protocol.get("outcomesModule") or {}
    sponsor = protocol.get("sponsorCollaboratorsModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    eligibility = protocol.get("eligibilityModule") or {}
    contacts_locations = protocol.get("contactsLocationsModule") or {}
    conditions_module = protocol.get("conditionsModule") or {}
    references_module = protocol.get("referencesModule") or {}

    design_info = design.get("designInfo") or {}
    masking_info = design_info.get("maskingInfo") or {}
    masking = masking_info.get("masking", UNKNOWN)
    enrollment_info = design.get("enrollmentInfo") or {}
    primary_outcomes = outcomes.get("primaryOutcomes") or []
    secondary_outcomes = outcomes.get("secondaryOutcomes") or []
    phases = design.get("phases") or []

    start_date, start_date_type = _date_struct(status, "startDateStruct")
    primary_completion, primary_completion_type = _date_struct(status, "primaryCompletionDateStruct")
    completion_date, completion_date_type = _date_struct(status, "completionDateStruct")
    last_update_post, last_update_post_type = _date_struct(status, "lastUpdatePostDateStruct")

    responsible_party = sponsor.get("responsibleParty") or {}
    collaborators = sponsor.get("collaborators") or []

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", UNKNOWN),
        "official_title": ident.get("officialTitle", UNKNOWN),
        "organization": ((ident.get("organization") or {}).get("fullName", UNKNOWN)),
        "sponsor": ((sponsor.get("leadSponsor") or {}).get("name", UNKNOWN)),
        "collaborators": [c.get("name", "") for c in collaborators],
        "responsible_party_type": responsible_party.get("type", UNKNOWN),
        "status": status.get("overallStatus", UNKNOWN),
        "why_stopped": status.get("whyStopped", UNKNOWN),
        "study_type": design.get("studyType", UNKNOWN),
        "phases": list(phases),
        "phase": ",".join(phases) if phases else UNKNOWN,
        "enrollment": enrollment_info.get("count"),
        "enrollment_type": enrollment_info.get("type", UNKNOWN),
        "allocation": design_info.get("allocation", UNKNOWN),
        "masking": masking or UNKNOWN,
        "masking_who": list(masking_info.get("whoMasked") or []),
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
        "interventions": [i.get("name", "") for i in (arms.get("interventions") or [])],
        "conditions": list(conditions_module.get("conditions") or []),
        "eligibility_criteria": eligibility.get("eligibilityCriteria", UNKNOWN),
        "eligibility_sex": eligibility.get("sex", UNKNOWN),
        "eligibility_minimum_age": eligibility.get("minimumAge", UNKNOWN),
        "eligibility_maximum_age": eligibility.get("maximumAge", UNKNOWN),
        "locations": [
            {
                "facility": loc.get("facility", UNKNOWN),
                "city": loc.get("city", UNKNOWN),
                "country": loc.get("country", UNKNOWN),
                "status": loc.get("status", UNKNOWN),
            }
            for loc in (contacts_locations.get("locations") or [])
        ],
        "references": [
            {"pmid": r.get("pmid", UNKNOWN), "citation": r.get("citation", UNKNOWN)}
            for r in (references_module.get("references") or [])
        ],
        # Presence of results, not their content or interpretation -- see
        # ``research/clinicaltrials_acquisition_adapter.py``'s module
        # docstring for why "resultsSection absent" must never be read as
        # "the trial failed".
        "has_results": bool(study.get("hasResults", False)),
        "start_date": start_date,
        "start_date_type": start_date_type,
        "primary_completion": primary_completion,
        "primary_completion_type": primary_completion_type,
        "completion_date": completion_date,
        "completion_date_type": completion_date_type,
        "last_update_post": last_update_post,
        "last_update_post_type": last_update_post_type,
        "last_update_submit": status.get("lastUpdateSubmitDate", UNKNOWN),
        "study_first_submit_date": status.get("studyFirstSubmitDate", UNKNOWN),
        # Backward-compatible alias -- pre-existing callers read "last_update".
        "last_update": last_update_post,
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
    if parsed.get("completion_date") != UNKNOWN:
        claims.append(
            (
                f"{nct} completion date is {parsed['completion_date']} "
                f"({parsed['completion_date_type']})",
                parsed["completion_date"],
                "date",
            )
        )
    if parsed.get("study_type") != UNKNOWN:
        claims.append((f"{nct} study type is {parsed['study_type']}", parsed["study_type"], "study_type"))
    # Presence of a results submission, NEVER its content or interpretation
    # -- "no resultsSection" and "the trial failed" are different statements
    # (see clinicaltrials_acquisition_adapter.py).
    claims.append(
        (f"{nct} has a results section on ClinicalTrials.gov: {parsed.get('has_results', False)}",
         bool(parsed.get("has_results", False)), "has_results")
    )
    return claims


def raw_facts_from_study(
    ticker: str,
    parsed: dict[str, Any],
    source: Source,
    *,
    collector: str = "clinicaltrials",
    document_id: str | None = None,
) -> list[RawFact]:
    """Build the registered-fact RawFacts for one already-parsed study.

    Shared by ``ClinicalTrialsCollector.collect()`` (sponsor-name search,
    never has a Document) and ``clinicaltrials_acquisition_adapter.py``
    (known-NCT-ID direct fetch, which DOES store a Document and so can set
    ``document_id`` -- ``RawFact.document_id`` stays ``None`` only for a
    path that genuinely never had one, never as a default guess).
    """
    return [
        RawFact(
            ticker=ticker.upper(),
            category=FactCategory.CLINICAL,
            claim=claim,
            source=source,
            value=value,
            unit=unit,
            company_claim=True,  # registry entries are sponsor-authored
            collector=collector,
            document_id=document_id,
        )
        for claim, value, unit in _study_claims(parsed)
    ]
