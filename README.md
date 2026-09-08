# investment-research-agent

An institutional-grade, **falsification-first** investment research system for
US and Japanese equities, with an emphasis on small caps and asymmetric-return
candidates.

Its primary purpose is not to find good stocks.

> **The goal is to kill wrong investment hypotheses as early as possible — using
> objective facts, primary sources and disconfirming evidence — and to evaluate
> only what survives.**

## Why it is built this way

It was built after a specific failure:

> A small-cap clinical-stage company presented an unusually attractive surface:
> tiny market capitalisation, a near-term clinical event, Fast Track and Orphan
> Drug designations, a Data Monitoring Committee recommending trial
> continuation, NIH support, a very large stated TAM, and bullish sell-side
> coverage. A coherent and exciting narrative was assembled from those signals.
>
> One fact, available in a filing, invalidated all of it: **the FDA did not
> consider the primary endpoint appropriate to establish efficacy.**

Every signal in that list was true. The narrative was still wrong. So the
system is designed around a single question — *what would make this wrong, and
is it already the case?* — asked before anything else.

`tests/regression/test_lgvn_type_failure.py` reconstructs exactly that case
against an anonymous company and asserts the system now catches it.

## Phase 2: live research and real LLM agents

Eight agents run on Claude when credentials are present, over live web research
through Anthropic's server-side `web_search` / `web_fetch` tools
([ADR 0005](docs/adr/0005-live-research-provider.md)):

```bash
export IRA_ANTHROPIC_API_KEY=sk-ant-...
python3 main.py LGVN --live --llm --adversarial
```

The interpretive work — reading a filing, spotting a semantic contradiction,
judging trial design — goes to the model. The arithmetic and the gates do not:
share counts, runway, market cap, valuation, the Kill Gate, source tiering and
isolation stay deterministic, because a plausible-looking wrong answer there is
what this system exists to prevent. **The model proposes; the rule-based core
disposes.** An LLM cannot lower a kill level, raise a capped score, or choose the
final action label.

Without credentials the deterministic agents run and the report says so — it
never presents a rule-based reading as a model analysis.

### The real case: LGVN

[`docs/examples/LGVN-captured-corpus-run.txt`](docs/examples/LGVN-captured-corpus-run.txt)
is a run over real Longeveron documents captured 2026-09-06:

```
REGULATOR POSITION ON PRIMARY ENDPOINT: REJECTED
[K5] Regulator states the primary endpoint is not sufficient to demonstrate efficacy
[K5] Regulator no longer treats the trial as pivotal
[K4] Going concern doubt
ACTION: AVOID      EVIDENCE CONFIDENCE: 0.0 / 10
```

The company's press release is headlined *"Constructive Type C Meeting"*. The
report files that adjective under **Company characterizations (NOT regulator
statements)**, and puts the FDA's actual position in the refused list. All five
FDA designations are listed as *procedural designation only*.

## Quick start

No dependencies to install for the core — Python 3.10+ and the standard library.

```bash
git clone <this repo> && cd investment-research-agent
python3 -m pip install pytest ruff mypy        # dev tooling only
cp .env.example .env                           # optional; everything works empty

python3 main.py DEMOBIO --fixtures             # synthetic end-to-end run
python3 -m pytest -q                           # 233 tests
```

Live research (network off by default):

```bash
python3 main.py CRBP --live --company-name "Corbus Pharmaceuticals Holdings, Inc." --price 12.50
```

### Modes

| Command | Meaning |
|---|---|
| `main.py TICKER` | standard run |
| `main.py TICKER --full-dd` | full due diligence |
| `main.py TICKER --update` | re-run and diff against the stored thesis |
| `main.py TICKER --kill-test` | kill gate emphasis |
| `main.py TICKER --catalyst` | catalyst calendar emphasis |
| `main.py --compare CRBP CNTB` | compare, ranked by confidence and kill level first |
| `main.py --screen explosive --fixtures` | screen the fixture universe |
| `--portfolio FILE` | your position, read **only** after the blind verdict is fixed |
| `--corpus` | replay a CAPTURED corpus of real documents from `data/corpus/` |
| `--live` | permit outbound network calls |
| `--llm` | use Claude for the eight interpretive agents |
| `--adversarial` | run the separated bear and bull search passes |
| `--resume RUN_ID` | restart an interrupted run from its last good stage |
| `--token-budget N` | cap total LLM tokens for the run |
| `--fixtures` | use synthetic data (always labelled) |
| `--json` / `--report-out FILE` | machine-readable / file output |

Exit code is `1` when the run is `INCOMPLETE_RESEARCH`, so it fails loudly in a
pipeline.

## What it actually produces

From the committed example run ([full output](docs/examples/DEMOBIO-fixture-run.txt)):

```
explosive_potential          9.5
immediate_buy                1.0   [CAPPED BY KILL GATE]
risk_reward                  1.0   [CAPPED BY KILL GATE]
EVIDENCE CONFIDENCE          1.0 / 10
REGULATORY_KILL              K5
ACTION                       AVOID
```

That combination is the point. The company can move violently *and* there is no
case for owning it, and the system states both without averaging them into one
misleading number.

The control case matters just as much — a system that always says AVOID would be
useless. [`DEMOTECH`](docs/examples/DEMOTECH-fixture-run.txt), a synthetic
company with no disqualifying facts, comes back `K0` and `WAIT_FOR_EVENT`: not
avoided, and not bought either, because several kill categories were never
searched and the system will not report an unsearched category as clean.

## The ten rules

1. **No conclusion-first research.** The Fact Collector has no field for a view
   and sees no analysis, so it cannot search selectively for support.
2. **Fact and opinion separated by type.** Every claim carries an evidence
   class: `VERIFIED_FACT`, `COMPANY_CLAIM`, `INDEPENDENT_EVIDENCE`,
   `ANALYST_OPINION`, `MARKET_INFERENCE`, `MODEL_INFERENCE`, `UNVERIFIED_CLAIM`.
   A company statement stays a company claim however primary the venue.
3. **Bull and Bear never read each other.** Same evidence, no rebuttal loop.
4. **Evaluations do not flow downstream.** Facts, sources, tiers, confidences,
   contradictions and risk flags may pass. Ratings, rankings, verdicts and
   anything about your position may not.
5. **Falsification first.** The Kill Agent runs before the Bull Agent.
6. **Contradictions beat coherence.** Conflicting facts are reported as a
   conflict, never reconciled toward the more attractive reading.
7. **Missing data is never filled in.** `UNKNOWN` / `NOT FOUND` /
   `INSUFFICIENT EVIDENCE`.
8. **Unsearched is not clean.** A kill category whose search never ran is
   reported `UNSEARCHED`, not K0.
9. **No overall score.** 22 independent dimensions; see
   [ADR 0002](docs/adr/0002-no-overall-score.md).
10. **No fabricated citations.** URLs, SEC accessions, NCT ids and analyst
    targets are validated against documents actually retrieved.

## Architecture

```
main.py
  └── cli.py
        └── orchestrator/pipeline.py
              │
              ├── EvidenceBus              append-only, channel-addressed
              ├── IsolationGuard           whitelist projection + leakage scan
              ├── Collectors               SEC EDGAR · ClinicalTrials.gov · FDA · search · fixtures
              ├── 14 agents                (below)
              ├── KillGate                 K0-K5, deterministic
              ├── ScoreCard                22 dimensions, capped by the kill gate
              ├── Repository               SQLite, append-only versioned facts
              └── ReportRenderer           20 sections, citation-validated
```

Stage order is fixed and load-bearing:

```
COLLECT → VERIFY → DOMAIN → CONTRADICT → KILL → BEAR ∥ BULL → VALUE
        → SCENARIO → BLIND JUDGE → DE-ANONYMISE → REPORT
```

### The fourteen agents

| # | Agent | Role | Cannot see |
|---|---|---|---|
| 1 | Fact Collector | collect, never evaluate | everything |
| 2 | Evidence Integrity | classify, date, corroborate | downstream analysis |
| 3 | Regulatory / Legal | what the regulator agreed, refused, left open | all evaluations |
| 4 | Capital Structure | full diluted count, runway, financing capacity | all evaluations |
| 5 | Science / Technology | trial design or deployment evidence | all evaluations |
| 6 | Kill Agent | disqualifying facts only | bull and bear |
| 7 | Bear Agent | most plausible failure path | **bull** |
| 8 | Bull Agent | evidence for undervaluation | **bear, kill** |
| 9 | Competitive Intelligence | peers; TAM vs SAM vs SOM | all evaluations |
| 10 | Valuation | what each 2x–200x multiple requires | all but share counts |
| 11 | Catalyst | dated events in JST, T0–T4 | all evaluations |
| 12 | Microstructure | positioning and flow | **all fundamentals** |
| 13 | Contradiction | mechanical cross-checks | bull and bear |
| 14 | Blind Judge | the verdict | **identity, prior scores, your holdings** |
| — | Portfolio | applies the fixed verdict to your position | nothing — but runs **last** |

Prompt contracts are in [`agents/`](agents/).

### How isolation is enforced

Not by prompt. By the orchestrator, in two layers:

- **Projection** — each agent gets an `AgentInput` built from a whitelist. There
  is no permissive default; an agent without a policy refuses to run.
- **Leakage scanning** — each evaluation carries four-word shingles unique to its
  own prose (shingles already in the shared evidence are subtracted, so shared
  vocabulary does not false-positive). Any denied channel's fingerprint, or any
  identity marker in a blind pack, raises `LeakageError` and fails the run.

This caught three real leaks during the first end-to-end run, including the
source-reference map — which holds the real `sec.gov` URLs — being handed to the
Blind Judge inside its parameters. See
[ADR 0001](docs/adr/0001-isolation-is-structural-not-prompted.md).

## Gates added in Phase 2

**Search Completeness Gate.** Six domains — Regulatory, Capital Structure,
Science/Technology, Competition, Catalyst, Contradiction — must be searched.
If one is not:

```
FINAL VERDICT: BLOCKED
RESEARCH STATUS: INCOMPLETE
```

and **no action label is emitted at all**. Not AVOID, not WAIT_FOR_EVENT. An
action asserts a judgement, and a judgement over an unexamined domain asserts
more than the research supports.

**Primary-source escalation.** A material claim carried only by Tier 3–5
reporting is escalated to a filing or a regulator document. When it cannot be
confirmed it becomes `UNVERIFIED_MATERIAL_CLAIM` — distinct from `NOT_VERIFIED`,
because "we tried specifically and failed" is a stronger warning than "we did
not confirm". A company press release does **not** count as confirmation of what
a regulator said.

**Citation traceability.** `claim → fact_id → source_id → URL → publication date
→ event date`, printed as an appendix. An attributed sentence — "the FDA said",
"analysts expect", "the study showed" — with no resolvable citation is a report
violation, because that phrasing is exactly how an unsourced assertion acquires
the authority of a source.

**Three provenance states, not two.** `LIVE`, `CAPTURED` (real documents about a
real issuer, captured at a stated time and replayed) and `FIXTURE` (synthetic).
Alongside it `ContentKind` records how much of a document is in hand: a
`SEARCH_SUMMARY` is a search engine's summary *about* a filing, not the filing.
Confidence is capped accordingly and material claims are escalated.

**Cost control.** Documents are chunked and each agent receives only the chunks
scored relevant to it, within a token budget. No agent reads a whole 10-K, and
per-agent token spend is reported.

## Source hierarchy

| Tier | Sources | Can it settle a material question alone? |
|---|---|---|
| 1 | SEC EDGAR, FDA, ClinicalTrials.gov, NIH, NASA, DoD, DHS, procurement, regulators, exchanges, statutory filings | yes |
| 2 | peer-reviewed research, PubMed, conference primary work, transcripts, company IR, government research | yes |
| 3 | Reuters, Bloomberg, WSJ, FT, Nikkei, trade press, expert interviews | supporting |
| 4 | sell-side, broker research, price targets | **no** |
| 5 | Seeking Alpha, X, Reddit, Stocktwits, blogs, boards | **no** |

Host matching is longest-suffix, so `pubmed.ncbi.nlm.nih.gov` is Tier 2
literature rather than Tier 1 via `nih.gov`. Reprints are detected by shingle
overlap: a wire story carried by five outlets counts once.

Dates distinguish `published_date`, `event_date`, `effective_date` and
`filing_date`, and recency is judged on when the event happened.

## Kill gate

Eight categories scored K0–K5, with `K4`/`K5` capping every investment-quality
score:

| | |
|---|---|
| K0 | no material concern found |
| K1 | minor |
| K2 | meaningful but manageable |
| K3 | major red flag — caps quality at 4.5 |
| K4 | severe, normally avoid — caps at 2.5 |
| K5 | investment disqualifier — caps at 1.0 |

`explosive_potential` is deliberately **not** capped: a disqualified company can
still move violently, and hiding that would be its own distortion.

## Storage

SQLite (`data/research.db`). Facts are append-only and versioned — a revised
fact gets `version + 1` and the predecessor is marked `superseded_by`, enforced
by a database trigger. Tables: `companies`, `facts`, `sources`,
`contradictions`, `regulatory_events`, `clinical_trials`, `capital_structure`,
`insider_trades`, `catalysts`, `scores`, `kill_assessments`, `scenarios`,
`thesis_versions`, `agent_runs`, `runs`, `fetch_log`.

Each new run writes a thesis version recording `WHAT_CHANGED`, `WHY_CHANGED`,
`NEW_FACT`, `REMOVED_ASSUMPTION` and `SCORE_CHANGE`, so a change of mind is
visible as one.

## Report

Twenty sections in a fixed order. Sections 3 and 4 — the red flags and *what
would kill the thesis* — come before the bull case, so a reader who stops early
still gets the disconfirming information.

```
1 Verdict · 2 Evidence Confidence · 3 Critical Red Flags · 4 What Would Kill The Thesis
5 Verified Facts · 6 Contradictions · 7 Bull Case · 8 Bear Case · 9 Regulatory
10 Capital Structure / Dilution · 11 Science / Technology · 12 Competition
13 Valuation · 14 Catalysts (JST) · 15 Scenarios · 16 Scores · 17 Immediate Action
18 What Changed Since Previous Run · 19 Unresolved Questions · 20 Sources
```

Action labels: `STRONG_BUY`, `BUY`, `BUY_ON_PULLBACK`, `WAIT_FOR_EVENT`, `HOLD`,
`PARTIAL_TAKE_PROFIT`, `CONSIDER_SELL`, `AVOID`. Evidence confidence is always
printed before the action.

## Testing

```bash
python3 -m pytest -q                    # 233 tests
python3 -m pytest tests/regression -q   # the LGVN-type guard
```

| Suite | Covers |
|---|---|
| unit | schemas, tiers, dates, duplicates, isolation, kill gate, evidence classification, valuation, capital structure, citations |
| integration | HTTP retry/backoff/timeout/cache over real sockets; SEC and CT.gov parsing; full pipeline with SQLite; zero-evidence path |
| regression | the LGVN-type failure, with a control run proving the kill comes from the facts and not from blanket pessimism |

A post-edit hook (`.claude/hooks/post_edit_checks.sh`) runs format, lint, type
check and all three suites, and exits non-zero on failure.

## Configuration and security

Everything is optional; the system runs with an empty `.env` and reports the
resulting gaps rather than filling them.

| Variable | Purpose |
|---|---|
| `IRA_SEC_USER_AGENT` | SEC requires a descriptive UA with a contact address |
| `IRA_ANTHROPIC_API_KEY` | optional narrative synthesis of verified facts |
| `IRA_SEARCH_PROVIDER` + key | `none` / `tavily` / `brave` — enables the kill-search set |
| `IRA_MARKET_PROVIDER` + key | optional microstructure data |

Keys are read from the environment only, `.env` is gitignored, and the log
formatter redacts known secret values and secret-shaped tokens. External web
content is treated as data, never instruction. A 403/407 from a proxy is an
egress-policy denial: it is reported, never retried or routed around.

## Assumptions recorded during the build

Decisions taken without asking, per the brief:

1. **Deterministic core, optional LLM.** The kill gate, capital arithmetic,
   valuation and contradiction detection are rule-based, because that is exactly
   where a hallucination causes the target failure
   ([ADR 0003](docs/adr/0003-deterministic-core-no-llm.md)).
2. **Zero runtime dependencies.** Standard library only.
3. **Network off by default.** A run that silently reaches the network behaves
   differently from one that does not; that must be a deliberate choice.
4. **Fixture companies are synthetic.** Attributing invented facts to a real
   issuer would itself be fabrication
   ([ADR 0004](docs/adr/0004-mock-and-production-data-separation.md)).
5. **Holdings reach exactly one component, and it runs last.** `--portfolio
   FILE` is read only by the Portfolio step, after the blind verdict is fixed,
   so knowing that a position is held — and at what price — cannot influence
   what the evidence was judged to mean. It never revises the verdict; it
   answers a different question: given this verdict, what should happen to the
   position? A test asserts the verdict and every score are byte-identical with
   and without a position supplied.
6. **`explosive_potential` is not capped by the kill gate**; investment-quality
   dimensions are. See ADR 0002.
7. **EV/Sales 5x and a 20% net margin** are the default translations from
   required market cap to required revenue and profit. Both are printed as
   assumptions on every run.

## Known limitations

Stated plainly rather than left to be discovered:

- **Regulatory meeting minutes, Type A/B/C correspondence, SPAs and CRLs are not
  available from any public API.** The system extracts them when a company
  discloses them in a filing, and otherwise raises a blocking unresolved
  question. It cannot obtain what is not published.
- **Kill searches need a search provider.** Without one they are enumerated and
  reported as *not executed*, and the affected categories come back `UNSEARCHED`.
- **Japanese equities**: TDnet and EDINET are tier-classified but do not yet have
  dedicated collectors. JP coverage currently depends on the search provider.
- **Microstructure** needs a market-data subscription; without one the fields are
  `UNKNOWN`, which is reported rather than defaulted.
- **Kill rules are pattern-based**, so a novel phrasing of an adverse finding can
  be missed. This is why an unestablished regulator position is always a blocking
  question — absence of a rule match is never read as good news.
- **Scenario probabilities are coarse and labelled low-confidence.** They are
  ranges on purpose; narrower ones would imply precision the evidence lacks.
- `--screen` requires a candidate universe and currently screens only the fixture
  set; no live universe source is configured.
- **Resume exists but re-collects on stale evidence.** `--resume RUN_ID` restarts
  from the last successful stage; if any restored fact is past the freshness
  threshold for its category, it restarts from collection instead. That is
  deliberate — resuming onto a stale regulatory fact is worse than starting
  over, because it would look current.
- **Insider detail is shallow.** Without Form 4 XML parsing, the individual, the
  role and the price are usually not in the filing text; they are stored as
  `UNKNOWN` rather than inferred.
- **No live LLM or live web run has ever been executed from this repository's
  environment.** Both code paths are implemented and tested against a mock
  Anthropic API speaking the real wire protocol, but the container has no
  credential available to the program and its egress policy blocks
  `sec.gov`, `clinicaltrials.gov`, `api.fda.gov` and the news domains. The real
  ticker runs therefore use `CAPTURED` corpora, and every report says so.
- **The captured corpora are search summaries, not source documents.** Direct
  fetch of the underlying filings and releases was blocked, so what is stored is
  a search engine's summary *about* each document. The system marks these
  `SEARCH_SUMMARY`, caps their confidence, and escalates their material claims —
  which is why the LGVN run reports evidence confidence 0.0/10 despite reaching
  a correct and well-sourced AVOID.
- **Corpus capture is manual.** Refreshing `data/corpus/` means re-running the
  searches by hand; there is no scheduled capture job.
- **Anthropic web search is US-only**, which constrains the Japanese-equity
  ambition until a JP-capable provider is configured.

## Licence

Private, for the author's own research use. Nothing here is investment advice.
