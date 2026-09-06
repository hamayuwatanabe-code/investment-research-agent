# ADR 0001: Agent isolation is structural, not prompted

## Status
Accepted.

## Context

Requirement 1C says the Bull and Bear agents must not read each other's output,
and requirement 10 says the user's holdings must never reach the Fact Collector,
Kill Agent, Bear Agent or Blind Judge.

The obvious implementation is to say so in each agent's prompt. That fails in
three ways: it is unverifiable, it degrades silently, and it does nothing about
*indirect* leakage — a bull argument copied into something labelled a "fact"
still reaches the bear agent, and a `sec.gov/corbus-pharmaceuticals-8k.htm`
citation de-anonymizes a blind pack as surely as the company name does.

## Decision

Isolation is enforced by the orchestrator, in two independent layers.

**Projection.** Each agent has an explicit `IsolationPolicy` naming the channels
it may read. The orchestrator builds an `AgentInput` containing only those.
There is no permissive default: an agent without a policy raises rather than
running.

**Leakage scanning.** Each evaluation carries `fingerprint_tokens` — four-word
shingles from its own prose, minus every shingle already present in the shared
evidence baseline. Before an input is released, the serialized payload is
scanned for the fingerprints of denied channels and, for blind agents, for
identity markers. A hit raises `LeakageError` and fails the run.

The baseline subtraction matters. Naive single-token fingerprints false-positive
constantly, because a bear agent may legitimately write "pipeline". What is
diagnostic is an agent reproducing a phrase that exists *only* in a denied
evaluation.

## Consequences

- Isolation is testable, and is tested, in both the direct and indirect cases.
- A leak is a hard failure. A research system that silently degrades its own
  independence is worse than one that stops.
- During the first end-to-end run the scanner caught three real leaks in code
  written moments earlier: the source-reference map (holding real URLs) being
  passed to the Blind Judge inside `params`; risk flags, contradictions and
  unresolved questions reaching the judge unredacted; and an agent's own
  risk-flag prose being misread as a leak because the fingerprint baseline
  covered only facts. None would have been found by reading the code.
- Widening a policy to silence a leakage error is almost always wrong. The
  error is usually correct.
