# Agent 11 -- Catalyst

**Agent id**: `catalyst`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["catalyst"]`
**Implementation**: `src/investment_research/agents/catalyst.py`

## Purpose

Enumerate dated forward events in JST.

## MUST see

- The shared verified evidence set.

## MUST NOT see

- Every agent's evaluation.

## Output contract

Per event: date (JST), date_confidence, event, expected_outcome, bull_outcome, bear_outcome, market_pricing, information_source, horizon T0-T4.

## Hard rules

1. All dates in JST. T0 intraday, T1 1-10 trading days, T2 1-3 months, T3 6-18 months, T4 3-10 years.
2. Guided timing is widened to the end of its window and labelled LOW confidence. 'Q4 2026' is never sharpened into a date.
3. `market_pricing` is UNKNOWN unless there is evidence. A catalyst the market has already discounted is a risk, not a catalyst.
