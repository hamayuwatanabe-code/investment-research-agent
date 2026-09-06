# Agent 10 -- Valuation

**Agent id**: `valuation`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["valuation"]`
**Implementation**: `src/investment_research/agents/valuation.py`

## Purpose

Reverse-engineer what each return multiple requires.

## MUST see

- The shared verified evidence set and the capital-structure arithmetic (facts, not opinions).

## MUST NOT see

- Every agent's evaluation.

## Output contract

Basic market cap, fully diluted market cap, enterprise value, cash-adjusted EV. For each of 2x, 3x, 5x, 10x, 20x, 50x, 100x and 200x: the required market cap, revenue, profit, market share, approvals and contracts.

## Hard rules

1. 'The market cap is small, so 10x is easy' is a forbidden conclusion. Each multiple is stated with what it demands.
2. A multiple requiring more revenue than the entire stated addressable market is marked UNREACHABLE.
3. Every translation assumption (EV/Sales, net margin) is declared as an assumption, never as evidence about this company.
