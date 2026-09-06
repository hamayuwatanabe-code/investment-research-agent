# ADR 0002: There is no overall score

## Status
Accepted.

## Context

Requirement 7 lists twenty-two independent score dimensions and forbids a
combined score. It is worth recording *why*, because the pressure to add one
will recur — a single number is easier to rank, sort and screen on.

## Decision

Twenty-two independent dimensions, each with its own confidence. No aggregate,
no weighted composite, no "conviction score".

Additionally, the Kill Gate caps every investment-quality dimension by the worst
kill level found: K3 caps at 4.5, K4 at 2.5, K5 at 1.0.

`explosive_potential` is deliberately **not** capped. A disqualified company can
still be capable of a violent move, and suppressing that would be its own
distortion. What is capped is the claim that it is a good investment.

## Consequences

- "Explosive Potential 9.5 / Evidence Confidence 4.0 / Immediate Buy 1.0" is a
  legitimate and useful output. It says: this can move, and we have not
  established that it should.
- A single disqualifying fact cannot be outvoted by an arbitrary number of
  attractive ones. Averaging is exactly the arithmetic that let a K5 regulatory
  problem lose to a good story.
- Ranking across candidates is done on evidence confidence and worst kill level
  first — never on upside alone. `--compare` enforces this in its output.
