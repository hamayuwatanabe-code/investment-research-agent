"""Search Completeness Gate (requirement P6).

Phase 1 reported unsearched kill categories. This is stronger: six research
domains must be SEARCHED before any final verdict is issued at all. If one is
not, the run produces

    FINAL VERDICT: BLOCKED
    RESEARCH STATUS: INCOMPLETE

and **no action label is emitted**. Not AVOID, not WAIT_FOR_EVENT -- no label.
An action implies a judgement, and a judgement over an unexamined domain is
exactly the thing this system exists to prevent. "We have not looked" is not a
position on the security.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..research.provider import DomainCoverage
from ..schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    FactCategory,
    ResearchDomain,
    ResearchPath,
    SearchStatus,
)

log = logging.getLogger(__name__)

#: FactCategory -> the ResearchDomain it counts toward, for matching an
#: UnresolvedQuestion's category against a query intent's domain (used by
#: the intent-level skip check in research/adversarial.py). Kept here,
#: alongside the completeness vocabulary it is part of.
_CATEGORY_TO_DOMAIN: dict[FactCategory, ResearchDomain] = {
    FactCategory.REGULATORY: ResearchDomain.REGULATORY,
    FactCategory.CLINICAL: ResearchDomain.SCIENCE_TECHNOLOGY,
    FactCategory.SCIENCE: ResearchDomain.SCIENCE_TECHNOLOGY,
    FactCategory.TECHNOLOGY: ResearchDomain.SCIENCE_TECHNOLOGY,
    FactCategory.CAPITAL_STRUCTURE: ResearchDomain.CAPITAL_STRUCTURE,
    FactCategory.FINANCIAL: ResearchDomain.CAPITAL_STRUCTURE,
    FactCategory.LIQUIDITY: ResearchDomain.CAPITAL_STRUCTURE,
    FactCategory.COMPETITION: ResearchDomain.COMPETITION,
    FactCategory.MARKET_SIZE: ResearchDomain.COMPETITION,
    FactCategory.CATALYST: ResearchDomain.CATALYST,
}


def research_domain_for_category(category: FactCategory) -> ResearchDomain | None:
    """The ``ResearchDomain`` a fact/question ``category`` counts toward, or
    ``None`` when it maps to no required domain (e.g. GOVERNANCE, LEGAL)."""
    return _CATEGORY_TO_DOMAIN.get(category)


#: Which structured collector's SUCCESSFUL execution merely TOUCHES which
#: required research domain -- auditable, but never by itself a claim that
#: the domain is complete (this is the exact regression this replaces: a
#: zero-result Drugs@FDA lookup or a ClinicalTrials.gov sponsor search used
#: to be read as "REGULATORY"/"SCIENCE_TECHNOLOGY" is DONE). A collector
#: touching a domain contributes only PARTIAL coverage; see
#: ``_SUFFICIENCY_CHECKS`` for what, if anything, can promote a domain past
#: that to genuinely complete.
_COLLECTOR_TOUCHES_DOMAIN: dict[str, frozenset[ResearchDomain]] = {
    "sec_edgar": frozenset({ResearchDomain.CAPITAL_STRUCTURE, ResearchDomain.REGULATORY}),
    "clinicaltrials": frozenset({ResearchDomain.SCIENCE_TECHNOLOGY}),
    "fda": frozenset({ResearchDomain.REGULATORY}),
}

#: RawFact.unit tags that, TOGETHER, must all have been actually extracted
#: (from filing bodies / XBRL -- never inferred from a filing merely
#: existing) before CAPITAL_STRUCTURE may be marked genuinely complete.
#: SecEdgarCollector.collect() today only ever records filing METADATA
#: (unit="form_type") -- so this checklist is never satisfied by that path
#: alone, by design (requirement D).
CAPITAL_STRUCTURE_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "basic_shares_outstanding",
        "fully_diluted_shares",
        "warrants_outstanding",
        "cash_and_equivalents",
        "total_debt",
        "atm_shelf_remaining",
    }
)


def _capital_structure_sufficient(collection_results: Sequence[Any]) -> bool:
    """Whether the CAPITAL_STRUCTURE required-field checklist is satisfied.

    Requirement D: a successful submissions-history fetch (filing metadata
    only) is never enough. Only actual extraction of the required capital
    fields themselves -- recorded as ``RawFact.unit`` tags -- promotes this
    domain past PARTIAL.
    """
    present: set[str] = set()
    for result in collection_results:
        if not getattr(result, "ok", False):
            continue
        for raw_fact in getattr(result, "raw_facts", ()) or ():
            unit = getattr(raw_fact, "unit", None)
            if unit in CAPITAL_STRUCTURE_REQUIRED_FIELDS:
                present.add(unit)
    return present >= CAPITAL_STRUCTURE_REQUIRED_FIELDS


def _regulatory_sufficient(collection_results: Sequence[Any]) -> bool:
    """openFDA/Drugs@FDA can never, by itself, complete REGULATORY.

    Requirement B: meeting outcome, endpoint acceptability, an SPA, CMC
    resolution and CRL/correspondence history are all outside what openFDA's
    API exposes (see ``collectors.fda.NON_API_REGULATORY_QUESTIONS``) --
    true regardless of how many applications a Drugs@FDA search returns, and
    regardless of a zero-result search too. There is currently no collector
    in this system that can establish REGULATORY sufficiency on its own.
    """
    return False


def _science_sufficient(collection_results: Sequence[Any]) -> bool:
    """ClinicalTrials.gov establishes registry facts only.

    Requirement C: phase, status, enrollment, design and endpoint WORDING
    are directly established. Peer-reviewed efficacy, biological
    plausibility/mechanism, endpoint validation, competitor evidence and
    reproducibility are not -- and remain open regardless of how successful
    the registry search was. There is currently no collector in this system
    that can establish SCIENCE_TECHNOLOGY sufficiency on its own.
    """
    return False


#: Per-domain sufficiency check (requirement A): the ONLY way a domain may
#: be promoted to ``SearchStatus.DIRECTLY_RESEARCHED``. A domain absent from
#: this mapping can never be marked sufficient by a collector at all.
_SUFFICIENCY_CHECKS: dict[ResearchDomain, Callable[[Sequence[Any]], bool]] = {
    ResearchDomain.CAPITAL_STRUCTURE: _capital_structure_sufficient,
    ResearchDomain.REGULATORY: _regulatory_sufficient,
    ResearchDomain.SCIENCE_TECHNOLOGY: _science_sufficient,
}


def direct_collector_coverage(
    collection_results: Sequence[Any],
) -> tuple[frozenset[ResearchDomain], frozenset[ResearchDomain]]:
    """``(sufficient, touched)`` domains a structured collector covered.

    Requirement A: "a successful collector execution is auditable coverage,
    but does not automatically mean the entire ResearchDomain is complete."
    ``touched`` is every domain a collector that actually ran successfully
    (``result.ok``) is mapped to -- real, auditable, but PARTIAL at most.
    ``sufficient`` is the (today, always small-or-empty) subset that ALSO
    passed that domain's own sufficiency checklist -- the only domains that
    may read ``SearchStatus.DIRECTLY_RESEARCHED``. Neither is ever inferred
    from facts existing afterward.
    """
    touched: set[ResearchDomain] = set()
    for result in collection_results:
        if not getattr(result, "ok", False):
            continue
        collector = getattr(result, "collector", "")
        touched |= _COLLECTOR_TOUCHES_DOMAIN.get(collector, frozenset())

    sufficient = {
        domain
        for domain in touched
        if _SUFFICIENCY_CHECKS.get(domain, lambda _r: False)(collection_results)
    }
    return frozenset(sufficient), frozenset(touched)


@dataclass
class CompletenessResult:
    coverage: dict[ResearchDomain, DomainCoverage] = field(default_factory=dict)

    @property
    def missing(self) -> list[ResearchDomain]:
        return [
            domain
            for domain in REQUIRED_RESEARCH_DOMAINS
            if not self.coverage.get(domain, DomainCoverage(domain)).searched
        ]

    @property
    def blocked(self) -> bool:
        return bool(self.missing)

    @property
    def searched_count(self) -> int:
        return len(REQUIRED_RESEARCH_DOMAINS) - len(self.missing)

    def reason(self) -> str:
        if not self.blocked:
            return ""
        names = ", ".join(str(domain) for domain in self.missing)
        return (
            f"{len(self.missing)} of {len(REQUIRED_RESEARCH_DOMAINS)} required research domains "
            f"were not searched: {names}. A verdict over an unexamined domain would assert more "
            f"than the research supports."
        )

    def summary_rows(self) -> list[tuple[str, str, int, int, str]]:
        rows: list[tuple[str, str, int, int, str]] = []
        for domain in REQUIRED_RESEARCH_DOMAINS:
            entry = self.coverage.get(domain, DomainCoverage(domain))
            rows.append(
                (
                    str(domain),
                    str(entry.status),
                    entry.queries_executed,
                    entry.documents_found,
                    ", ".join(str(p) for p in entry.paths) or "none",
                )
            )
        return rows


def assess_completeness(
    *,
    search_results: list,
    facts_by_domain: dict[ResearchDomain, int] | None = None,
    agents_run: set[str] | None = None,
    collection_results: Sequence[Any] = (),
) -> CompletenessResult:
    """Decide, per domain, whether research actually happened.

    A domain counts as SEARCHED when a web/discovery query for it executed;
    DIRECTLY_RESEARCHED only when a structured collector's execution record
    ALSO passes that domain's own sufficiency checklist (requirement A/D --
    e.g. today, nothing does for REGULATORY or SCIENCE_TECHNOLOGY, by
    design: see ``_regulatory_sufficient``/``_science_sufficient``); PARTIAL
    when a collector merely touched the domain (a genuine execution record,
    but not sufficiency) or the agent responsible for it produced evidence
    from the collected corpus. None of these is ever inferred from facts
    existing alone -- there is always an actual query or collector
    execution record behind it. PARTIAL is real, auditable coverage, but it
    does NOT satisfy the final required-domain gate (see
    ``DomainCoverage.searched``) -- only SEARCHED or DIRECTLY_RESEARCHED do.
    """
    result = CompletenessResult()
    facts_by_domain = facts_by_domain or {}
    agents_run = agents_run or set()
    sufficient_direct, touched_direct = direct_collector_coverage(collection_results)

    #: Which agent, having run and produced evidence, satisfies which domain.
    agent_for_domain = {
        ResearchDomain.REGULATORY: "regulatory",
        ResearchDomain.CAPITAL_STRUCTURE: "capital_structure",
        ResearchDomain.SCIENCE_TECHNOLOGY: "science",
        ResearchDomain.COMPETITION: "competitive",
        ResearchDomain.CATALYST: "catalyst",
        ResearchDomain.CONTRADICTION: "contradiction",
    }

    for domain in REQUIRED_RESEARCH_DOMAINS:
        entry = DomainCoverage(domain=domain)
        paths: list[ResearchPath] = []
        for search in search_results:
            if search.query.domain is not domain:
                continue
            entry.queries_attempted += 1
            if search.executed:
                entry.queries_executed += 1
                entry.documents_found += len(search.documents)
                if search.path not in paths:
                    paths.append(search.path)
        entry.paths = tuple(paths)

        agent_evidence = facts_by_domain.get(domain, 0)
        agent_ran = agent_for_domain.get(domain) in agents_run

        if entry.queries_executed and entry.documents_found:
            entry.status = SearchStatus.SEARCHED
            entry.detail = (
                f"{entry.queries_executed} query/queries executed, "
                f"{entry.documents_found} document(s)"
            )
        elif entry.queries_executed:
            entry.status = SearchStatus.SEARCHED
            entry.detail = f"{entry.queries_executed} query/queries executed, no documents returned"
        elif domain in sufficient_direct:
            entry.status = SearchStatus.DIRECTLY_RESEARCHED
            if ResearchPath.DIRECT_API not in paths:
                paths.append(ResearchPath.DIRECT_API)
                entry.paths = tuple(paths)
            entry.detail = (
                "a structured collector satisfied this domain's own sufficiency checklist "
                "(no web search needed)"
            )
        elif domain in touched_direct or (agent_ran and agent_evidence):
            # Requirement A/B/C: a collector genuinely executed for this
            # domain (even a clean zero-result search, e.g. Drugs@FDA
            # finding no applications) -- real, auditable coverage, but
            # never enough by itself to call the domain complete. Never
            # described as "searched": PARTIAL is a distinct status that
            # does NOT satisfy the final gate (DomainCoverage.searched).
            entry.status = SearchStatus.PARTIAL
            entry.detail = (
                "a structured collector touched this domain (not sufficient on its own)"
                if domain in touched_direct and not agent_evidence
                else f"no dedicated query executed, but {agent_evidence} fact(s) were analysed "
                "from the collected corpus"
            )
        elif entry.queries_attempted:
            entry.status = SearchStatus.FAILED
            entry.detail = f"{entry.queries_attempted} query/queries attempted, none executed"
        else:
            entry.status = SearchStatus.UNSEARCHED
            entry.detail = "no query attempted and no evidence analysed"

        result.coverage[domain] = entry

    return result
