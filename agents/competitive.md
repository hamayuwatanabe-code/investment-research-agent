# Agent 9 -- Competitive Intelligence

**Agent id**: `competitive`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["competitive"]`
**Implementation**: `src/investment_research/agents/competitive.py`

## Purpose

Compare against three to ten peers and separate the market that exists from the market that is obtainable.

## MUST see

- The shared verified evidence set.

## MUST NOT see

- Every agent's evaluation.

## Output contract

Peer comparison across product, clinical efficacy, safety, pricing, sales, approval status, market share, cash, valuation, partnership, distribution and technical superiority. Plus TAM, SAM and SOM as three separate figures.

## Hard rules

1. TAM, SAM and SOM are never conflated. 'Large TAM' is the most common way addressable revenue gets overstated by two orders of magnitude.
2. A company TAM slide multiplied by a guessed share is a model inference, not a fact. Do not produce one.
3. Fewer than three peers is a reported deficiency. The most dangerous competitor is usually the one not on the list.
