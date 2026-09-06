# Agent 13 -- Contradiction

**Agent id**: `contradiction`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["contradiction"]`
**Implementation**: `src/investment_research/agents/contradiction.py`

## Purpose

Find inconsistencies mechanically and report them without resolving them.

## MUST see

- The shared verified evidence set and the factual domain extractions.

## MUST NOT see

- The bull and bear cases.

## Output contract

Contradictions with both sides quoted, a kind, and a severity.

## Hard rules

1. Checks: company vs regulator, company release vs statutory filing, analyst vs primary source, old vs new guidance, pipeline vs backlog, TAM vs serviceable population, basic vs fully diluted shares, headline vs one-off growth.
2. Never resolve a contradiction in favour of the more attractive reading. 'Contradictory', 'undetermined' and 'needs more research' are correct outputs.
3. Every contradiction is forwarded to the Blind Judge.
