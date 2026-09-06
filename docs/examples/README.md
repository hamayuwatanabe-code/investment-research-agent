# Example outputs

Two real runs of this system, committed verbatim.

## `DEMOBIO-fixture-run.txt`

`python main.py DEMOBIO --fixtures`

A complete run over the synthetic `DEMOBIO` fixture, which reproduces the shape
of the failure this system was built after. The interesting part is the
combination it produces:

```
explosive_potential          9.5
immediate_buy                1.0   [CAPPED BY KILL GATE]
risk_reward                  1.0   [CAPPED BY KILL GATE]
EVIDENCE CONFIDENCE          1.0 / 10
REGULATORY_KILL              K5
ACTION                       AVOID
```

The company can move violently, and there is no case for owning it. The system
says both, separately, without averaging them into one misleading number.

Note also:

- the decisive fact was found in the Risk Factors of a 10-Q, not in the press
  release that called the same meeting "constructive";
- both statements are reported side by side as a CRITICAL contradiction;
- Fast Track and Orphan Drug are listed under "what the regulator has agreed"
  but labelled *procedural designation only — says nothing about whether the
  efficacy evidence or endpoint will be accepted*;
- 200x is marked UNREACHABLE because it would require 127% of the company's own
  stated addressable market.

This is synthetic data and is labelled as such throughout.

## `CRBP-live-run-blocked-network.txt`

`python main.py CRBP --live --company-name "Corbus Pharmaceuticals Holdings, Inc." --price 12.50`

A live run in an environment whose egress policy blocks `data.sec.gov`,
`clinicaltrials.gov` and `api.fda.gov`. It is included because the honest
failure mode is worth showing:

```
*** INCOMPLETE_RESEARCH ***
  - fact_collector: DEGRADED :: sec_edgar: BLOCKED ... clinicaltrials: BLOCKED ... fda: BLOCKED
EVIDENCE CONFIDENCE: 0.0 / 10
  facts in evidence set : 0
ACTION: WAIT_FOR_EVENT
```

Zero facts, zero confidence, no verdict about the company, and the blocked hosts
named. Nothing was inferred, no cached impression of the company was used, and
the run is not reported as a success. Running this with unrestricted network
access requires no code change.
