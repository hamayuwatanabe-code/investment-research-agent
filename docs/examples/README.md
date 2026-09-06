# Example outputs

Runs of this system, committed verbatim.

## `LGVN-captured-corpus-run.txt` — the real 2026 case

`python main.py LGVN --corpus --adversarial`

Longeveron, on real documents captured 2026-09-06. This is the live instance of
the failure the whole system was built after, and the system catches it:

```
REGULATOR POSITION ON PRIMARY ENDPOINT: REJECTED
[K5] REGULATORY_KILL: Regulator states the primary endpoint is not sufficient
     to demonstrate efficacy
[K5] REGULATORY_KILL: Regulator no longer treats the trial as pivotal
[K4] CAPITAL_KILL   : Going concern doubt
ACTION: AVOID       EVIDENCE CONFIDENCE: 0.0 / 10
```

The company's own press release is headlined **"Constructive Type C Meeting"**.
The report records that adjective under *Company characterizations (NOT
regulator statements)* and puts the regulator's actual position — that RVEF
"is not sufficient to demonstrate efficacy", and that the FDA "no longer refers
to the ELPIS II trial as pivotal" — in the refused list.

Five FDA designations (Orphan Drug, Fast Track, Rare Pediatric Disease, RMAT,
Priority Review) appear under "what the regulator has agreed", every one
labelled *procedural designation only — says nothing about whether the efficacy
evidence or endpoint will be accepted*. That is the exact conflation that
produced the original loss.

Note what the report does **not** claim: evidence confidence is 0.0/10, seven
material claims are marked `UNVERIFIED_MATERIAL_CLAIM` because the captured
documents are search summaries that could not be escalated to a filing, and the
run is `INCOMPLETE_RESEARCH`. The verdict is AVOID and the system is explicit
that it reached it on thin, unconfirmed sourcing.

## `CNTB-captured-corpus-run.txt` and `CRBP-captured-corpus-run.txt`

The same pipeline on two other real issuers, both returning
`WAIT_FOR_EVENT` at K2 rather than AVOID — the discrimination that shows the
LGVN result comes from LGVN's evidence and not from blanket pessimism. CRBP in
particular reports share count and cash as `UNKNOWN` rather than guessing,
because sec.gov was unreachable at capture time.

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
