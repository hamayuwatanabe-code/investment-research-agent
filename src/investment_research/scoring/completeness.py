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
from dataclasses import dataclass, field

from ..research.provider import DomainCoverage
from ..schemas.enums import (
    REQUIRED_RESEARCH_DOMAINS,
    ResearchDomain,
    ResearchPath,
    SearchStatus,
)

log = logging.getLogger(__name__)


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
) -> CompletenessResult:
    """Decide, per domain, whether research actually happened.

    A domain counts as SEARCHED when a query for it executed **or** when the
    agent responsible for it produced evidence from the collected corpus. Both
    are real examination; neither is inferred from the other.
    """
    result = CompletenessResult()
    facts_by_domain = facts_by_domain or {}
    agents_run = agents_run or set()

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
            entry.detail = (
                f"{entry.queries_executed} query/queries executed, no documents returned"
            )
        elif agent_ran and agent_evidence:
            entry.status = SearchStatus.PARTIAL
            entry.detail = (
                f"no dedicated query executed, but {agent_evidence} fact(s) were analysed "
                "from the collected corpus"
            )
        elif entry.queries_attempted:
            entry.status = SearchStatus.FAILED
            entry.detail = (
                f"{entry.queries_attempted} query/queries attempted, none executed"
            )
        else:
            entry.status = SearchStatus.UNSEARCHED
            entry.detail = "no query attempted and no evidence analysed"

        result.coverage[domain] = entry

    return result
