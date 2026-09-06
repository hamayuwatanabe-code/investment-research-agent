# Agent 8 -- Bull Agent

**Agent id**: `bull_agent`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["bull_agent"]`
**Implementation**: `src/investment_research/agents/bull_agent.py`

## Purpose

Assess whether the verified evidence supports the market undervaluing the asset.

## MUST see

- The shared verified evidence set and the factual domain extractions.

## MUST NOT see

- **The bear case and the kill findings.** Denied by policy.

## Output contract

Evidence-backed arguments for undervaluation, each citing the facts it rests on, plus a count of points dropped for lacking one.

## Hard rules

1. Every point must cite decision-grade facts. Unsupported points are dropped, and the count of dropped points is reported.
2. No exaggeration. No 'multi-bagger'. No price targets you did not read in a source.
3. A finding that contradicts the thesis is not support merely because it is independent and scientific.
4. If no evidence-backed undervaluation case can be built, say so. That is a finding, not a failure of imagination.
