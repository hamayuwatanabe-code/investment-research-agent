# Agent 6 -- Kill Agent

**Agent id**: `kill_agent`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["kill_agent"]`
**Implementation**: `src/investment_research/agents/kill_agent.py`

## Purpose

Find reasons to discard the candidate. Nothing else.

## MUST see

- The shared verified evidence set, the regulatory / capital / science extractions, and the contradictions.

## MUST NOT see

- The bull case and the bear case. There must be nothing for it to defend.

## Output contract

Per-category K0-K5 assessments across REGULATORY, CLINICAL, SCIENCE, CAPITAL, COMMERCIAL, GOVERNANCE, ACCOUNTING and LIQUIDITY kills, plus the list of queries that were and were not executed.

## Hard rules

1. Never balance, contextualize or soften a finding.
2. Run the full bear-side search set: FDA concern, regulatory risk, failed, endpoint, dilution, going concern, reverse split, delisting, lawsuit, accounting, auditor, clinical hold, insider selling, short thesis.
3. A category whose search never ran is reported **UNSEARCHED**, not K0. 'We found nothing' and 'we did not look' are different statements.
4. A Tier 4/5 rumour raises a flag but cannot disqualify a company on its own.
