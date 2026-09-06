# ADR 0004: Fixture data is unmistakable

## Status
Accepted.

## Context

Requirement 24 requires mock and production data to be clearly separated, and
requirement 15 forbids fabricating sources. A fixture that looks like real
research is a fabrication with extra steps.

The constraint that forced the issue: the environment this system was built in
has outbound access to `data.sec.gov`, `clinicaltrials.gov` and `api.fda.gov`
blocked by egress policy. The end-to-end demonstration therefore had to run on
synthetic data.

## Decision

Four independent markers, any one of which is sufficient to identify a fixture
run:

1. Fixture source URLs use the `fixture://` scheme. They cannot be mistaken for,
   or clicked as, real citations, and the fixture collector rejects any entry
   that does not use it.
2. Every fixture source carries `Provenance.FIXTURE`.
3. The report prints a banner: *SYNTHETIC FIXTURE DATA — THIS IS NOT REAL
   RESEARCH.*
4. Evidence confidence is capped at 3.0 and the run is never marked COMPLETE.

Fixture companies are synthetic. `DEMOBIO` is not a real issuer and its facts
describe no real company; it reproduces the *shape* of the target failure.

## Consequences

- The pipeline, the regression test and the example output are all real and
  reproducible without network access.
- A live run needs only `--live`; the collectors are written against the real
  API contracts and are tested against a local server serving API-shaped
  payloads.
- Blocked hosts are reported as `BLOCKED`, never retried, and never worked
  around. The resulting run is honest: zero facts, zero confidence, INCOMPLETE
  RESEARCH, and no conclusions.
