# Agent 1 -- Raw Fact Collector

**Agent id**: `fact_collector`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["fact_collector"]`
**Implementation**: `src/investment_research/agents/fact_collector.py`

## Purpose

Collect observable facts about the company and evaluate none of them.

## MUST see

- Nothing but the ticker and the collector results. No facts, no risk flags, no analysis of any kind.

## MUST NOT see

- Every evaluation channel. It must not know what anyone thinks, so it cannot search selectively in support of a view it already holds.

## Output contract

A list of `RawFact` records. Each carries a claim, a source with its tier and dates, and a `company_claim` boolean. There is deliberately **no** field in which to record a view.

Required coverage: 10-K, 10-Q, 8-K, S-1/S-3, 424B, DEF 14A, 13D/13G, Form 4, FDA documents, ClinicalTrials.gov, trial protocol, government contracts, peer-reviewed papers, company IR, earnings transcripts, financing, warrants, convertibles, ATM, debt, cash, share count, fully diluted securities, management, competition, market size, customer concentration, backlog, pipeline, recent events.

## Hard rules

1. Never write an opinion, rating, recommendation or forecast. Claims are checked against an evaluative-language detector and a violation is dropped with an error, not stored.
2. Never infer a number that is not stated. If a filing does not give the diluted count, the diluted count is UNKNOWN.
3. Every required coverage target that produced nothing becomes an explicit gap in the report.
