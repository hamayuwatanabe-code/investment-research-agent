# Agent 14 -- Blind Neutral Judge

**Agent id**: `blind_judge`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["blind_judge"]`
**Implementation**: `src/investment_research/agents/blind_judge.py`

## Purpose

Reach a verdict from the evidence alone.

## MUST see

- An anonymized evidence pack: facts, kill findings, contradictions, bull case, bear case, valuation math, catalysts and scenarios, with the company referred to only as 'Company X' and sources as opaque S-NNN references.

## MUST NOT see

- The ticker, the company name, source URLs, prior scores, prior rankings, prior verdicts, the user's holdings, cost basis, or opinions. All denied by policy and verified by a leakage scanner that fails the run on breach.

## Output contract

An action label, the evidence confidence, the headline, the reasoning, the thesis breakers, the critical red flags and the caveats.

## Hard rules

1. Evidence confidence is established **before** the action label.
2. Kill gate first: K5 or K4 means AVOID regardless of upside. K3 means the flag must be resolved before a position is justified.
3. Incomplete research means no buy-side action is available.
4. Permitted actions only: STRONG_BUY, BUY, BUY_ON_PULLBACK, WAIT_FOR_EVENT, HOLD, PARTIAL_TAKE_PROFIT, CONSIDER_SELL, AVOID.
5. Identity is restored only after the verdict is fixed.
