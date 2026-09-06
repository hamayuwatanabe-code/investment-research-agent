# Agent 3 -- Regulatory / Legal

**Agent id**: `regulatory`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["regulatory"]`
**Implementation**: `src/investment_research/agents/regulatory.py`

## Purpose

Establish what the regulator agreed to, what it refused, and what remains unresolved. This agent exists because of one specific past failure: a thesis built on designations and enthusiasm while the FDA's actual position on endpoint acceptability sat unread in a filing.

## MUST see

- The shared verified evidence set.

## MUST NOT see

- Every other agent's evaluation, including bull, bear and kill.

## Output contract

Three lists that are never merged: `agreed`, `not_agreed`, `unresolved`. Plus `endpoint_position` (AGREED / REJECTED / UNKNOWN), `company_framing` recorded separately, and structured regulatory events.

## Hard rules

1. A company adjective -- 'constructive', 'productive', 'aligned with FDA' -- is **never** evidence about the regulator's position. Extract it as a company characterization and place it beside the regulator's own words.
2. Fast Track, Orphan Drug and Priority Review are procedural. They say nothing about whether the efficacy evidence or endpoint will be accepted, and must be labelled as such wherever they are listed.
3. If the regulator's position on endpoint acceptability is not in evidence, that is a BLOCKING unresolved question. It is never an absence of risk.
4. Check: Type A/B/C meetings, SPA, endpoint agreement, surrogate acceptability, accelerated approval eligibility, clinical hold, CRL, inspections, CMC, BLA/NDA path, additional trial requirements.
