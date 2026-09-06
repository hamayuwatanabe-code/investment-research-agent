# Agent 12 -- Market Microstructure

**Agent id**: `microstructure`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["microstructure"]`
**Implementation**: `src/investment_research/agents/microstructure.py`

## Purpose

Report positioning and flow data.

## MUST see

- The shared verified evidence set.

## MUST NOT see

- **Every fundamental analysis.** Denied by policy so a squeeze setup cannot be blended into a business assessment.

## Output contract

short interest, short float, days to cover, borrow availability and fee, options open interest, gamma, volume, relative volume, VWAP, ATR, RSI, support/resistance, institutional ownership, insider ownership, ETF/index exposure.

## Hard rules

1. Never mix these with fundamental judgement. They are reported in their own section.
2. An unknown short interest is not a short interest of zero. Unknown fields are listed.
