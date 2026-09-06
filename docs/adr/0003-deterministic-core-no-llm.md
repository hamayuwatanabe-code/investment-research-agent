# ADR 0003: The decisive logic is rule-based, not model-based

## Status
Accepted.

## Context

This is an LLM-adjacent system, so the default assumption is that an LLM does
the analysis. But the components that decide whether a thesis dies — the kill
gate, capital-structure arithmetic, valuation math, contradiction detection,
date integrity — are exactly the components where a hallucination or a
sycophantic reading causes the failure this system exists to prevent.

## Decision

The deterministic core has **zero runtime dependencies** and calls no model. Kill
rules are regular expressions with stated severities and explanations. Runway is
arithmetic. Contradiction checks are named, enumerable comparisons. The share
count is a sum whose missing terms are listed.

An LLM provider is an *optional* component, usable for narrative synthesis of
already-verified facts, and its output passes through the citation validator
before reaching the reader.

## Consequences

- The same evidence produces the same verdict on every run, and every step is
  auditable line by line.
- The system runs with no API key, offline, anywhere Python 3.10+ runs.
- Rules are less flexible than a model. A novel phrasing of an adverse
  regulatory finding can be missed, which is why the Regulatory agent also emits
  a BLOCKING unresolved question whenever the endpoint position is not
  positively established. Absence of a rule match is never read as good news.
