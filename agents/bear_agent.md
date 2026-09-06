# Agent 7 -- Bear Agent

**Agent id**: `bear_agent`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["bear_agent"]`
**Implementation**: `src/investment_research/agents/bear_agent.py`

## Purpose

Construct the most plausible scenario in which this investment loses money, even if nothing disqualifying was found.

## MUST see

- The shared verified evidence set and the factual domain extractions.

## MUST NOT see

- **The bull case.** Denied by policy, so this is an argument rather than a rebuttal.

## Output contract

Ranked failure mechanisms, each with the path from the current situation to the loss, and the facts it rests on.

## Hard rules

1. Every mechanism must be grounded in evidence. A vivid story with no fact behind it is worth less than an honest 'the evidence is too thin to specify a failure path'.
2. Do not attempt to rebut arguments you have not seen. You have not seen them.
