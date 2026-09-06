# Agent 4 -- Capital Structure

**Agent id**: `capital_structure`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["capital_structure"]`
**Implementation**: `src/investment_research/agents/capital_structure.py`

## Purpose

Reconstruct the full claim on the equity and the financing that is already authorized.

## MUST see

- The shared verified evidence set.

## MUST NOT see

- Every agent's evaluation.

## Output contract

basic shares, fully diluted shares, options, RSUs, public/private warrants, pre-funded warrants, preferred, convertible debt, ATM capacity, shelf status, PIPE, earnout, reverse split history, listing compliance, cash, debt, burn, runway, going concern, and an explicit `unknown_fields` list.

## Hard rules

1. A headline market cap is never a valuation input. Every figure is computed on the fully diluted count.
2. Pre-funded warrants are shares in all but name.
3. A diluted count assembled from partial data is reported as a FLOOR with its missing components named. It is never presented as complete.
4. One disclosure line naming two instrument types is one number. Attributing it to both silently inflates the count.
