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

## `DEMOTECH-fixture-run.txt`

`python main.py DEMOTECH --fixtures`

The control case, and the reason it is committed: **a falsification-first system
that always says AVOID would be useless.** `DEMOTECH` is a synthetic company with
no disqualifying facts -- funded for years, no debt, independently corroborated
deployments, three named competitors, an independent market study rather than a
company TAM slide.

```
REGULATORY_KILL              K0
worst kill level             K0
ACTION                       WAIT_FOR_EVENT
```

Not AVOID, and not BUY either: evidence confidence is still low because this is
fixture data and no search provider was configured, so several kill categories
were never searched. The system distinguishes "nothing disqualifying was found"
from "we looked and it is clean", and reports the second only when it is true.

This run also fixed a real false signal: the Regulatory agent used to ask whether
the regulator accepts the primary endpoint even for a technology company that has
no approval pathway, and reported the unanswerable question as a regulatory risk.
It now asks only where an approval actually gates the business -- while still
asking when the evidence set is empty, since an empty set must never be read as
"not regulated".

## `compare-output.txt`

`python main.py --compare DEMOBIO DEMOTECH --fixtures`

Comparison is ordered by evidence confidence and worst kill level, never by
upside. There is no combined score to sort on.

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
