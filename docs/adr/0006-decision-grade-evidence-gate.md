# ADR 0006: The Decision-Grade Evidence Gate

## Status
Accepted.

## Context

Requirement M1 (see the MUST-1/2/3 work) made sure a search snippet can never
be *stored* as a `VERIFIED_FACT`. Running LGVN, CNTB and CRBP against their
captured corpora with that gate in place produced zero `VERIFIED` facts in
every one of the three runs — exactly as intended, since none of their
document bodies had actually been fetched. But the Blind Judge still emitted
`AVOID` for LGVN and `WAIT_FOR_EVENT` for CNTB/CRBP, computed from Kill Gate
*severity* and an evidence-confidence *score*, neither of which asked whether
any fact behind that severity was decision-grade. A K5 built entirely from an
unread search result produced exactly the same confident label a K5 built from
a read filing would have. That is the FDA-endpoint failure this project was
built after, reproduced from the other direction: a plausible, structurally
correct-looking pipeline emitting a confident answer that the underlying
evidence does not support.

Fixing the storage layer (MUST-1) was necessary but not sufficient. The gap was
architectural: nothing between "the facts are correctly classified" and "an
Action is emitted" ever asked "is *any* of this actually verified".

## Decision

**Investment Action requires COMPLETE research status, and research status is
computed from an evidence-sufficiency test distinct from severity.**

1. `ResearchStatus` (`COMPLETE` / `INCOMPLETE` / `BLOCKED_PENDING_VERIFICATION`)
   is tracked on `Verdict` alongside — not instead of — the existing
   `RunStatus` (pipeline health) and the Search Completeness Gate (was every
   domain searched). `Verdict.action` may be non-`None` only when
   `research_status is COMPLETE`. `WAIT_FOR_EVENT` is an Action like any
   other; insufficient evidence gets no label, not that one.

2. Every `KillFinding`/`KillAssessment` carries a `KillConfirmation`
   (`PROVISIONAL` / `CONFIRMED`) computed from whether the fact(s) behind it
   are decision-grade (`Fact.is_decision_grade`), *separate from* `KillLevel`.
   Severity says how bad, if true; confirmation says whether "if true" has
   been settled. `KillGateResult.max_level` (used for score caps and
   reporting) and `KillGateResult.max_confirmed_level` (used to select an
   Action) are deliberately two different properties, so a provisional K5
   from a search snippet cannot silently do the work of a confirmed one.

3. The `EvidenceSufficiencyMatrix` (`scoring/evidence_sufficiency.py`) is the
   second half of a distinction the Search Completeness Gate only started:
   `search_status` ("did we look at this domain") and
   `evidence_sufficiency_status` ("did what we found settle anything") are
   tracked as two separate fields per domain. A domain can be fully SEARCHED
   and still INSUFFICIENT. The matrix blocks the run when: zero decision-grade
   facts exist anywhere; any K3+ kill finding is still PROVISIONAL; an
   unresolved MATERIAL/CRITICAL claim remains; or a domain carrying a material
   (MEDIUM+) claim has no decision-grade backing and no CONFIRMED rule-based
   disqualifier standing in for it.

4. The rule-based/arithmetic exception. Cash-runway and similar structured
   arithmetic disqualifiers (CLAUDE.md: "what must stay rule-based, always")
   are allowed to register as CONFIRMED even when their input figures are
   `COMPANY_CLAIM` — the arithmetic is auditable and the input is structured,
   not a free-text assertion. This is narrow and one-directional: it lets a
   confirmed *kill* finding stand in for a domain's decision-grade requirement
   (a going-concern K4 substitutes for a missing decision-grade capital-
   structure fact), never the reverse.

5. Symmetry is structural, not policy. `Fact.is_decision_grade` and the
   sufficiency matrix have no branch on sentiment; the same test that keeps an
   unread regulatory rejection from producing AVOID keeps an unread "FDA
   agreed the endpoint" from producing BUY. Regression coverage
   (`tests/regression/test_decision_grade_gate.py`) exercises both directions
   deliberately, including a case where `_select_action` alone *would* return
   BUY, to show the pipeline-level gate — not that function — is what carries
   the guarantee.

6. The Blind Judge's LLM path never held this gate anyway (the action label
   was always computed by the same deterministic `_select_action`, never by
   the model), but the prompt now separates decision-grade evidence from a
   `NON_DECISION_GRADE / DO NOT USE FOR FINAL VERDICT` section explicitly, and
   a regression test feeds the model a payload arguing for the opposite
   conclusion to prove the returned action does not move.

## Consequences

- Re-running LGVN/CNTB/CRBP against the same captured corpora now reports
  `RESEARCH_STATUS: BLOCKED_PENDING_VERIFICATION` / `FINAL_ACTION: NONE` for
  all three, with `POTENTIAL_<CATEGORY>_KILL: <level> PROVISIONAL` lines
  naming exactly what would need a fetched document body to become CONFIRMED.
  This is a **less** confident output than before on the same evidence, which
  is the point: the evidence did not change, only the honesty about it did.
- Two existing Phase 1 fixture regressions (`test_pipeline.py`,
  `test_lgvn_type_failure.py`) changed from asserting a bare `AVOID` to
  asserting `BLOCKED_PENDING_VERIFICATION` plus a `CONFIRMED` regulatory
  finding, because their fixtures also carry a second, genuinely
  single-sourced (`COMPANY_CLAIM`) finding that the stricter gate correctly
  leaves PROVISIONAL. The regulatory disqualifier itself did not get weaker;
  a second, previously-unexamined gap got surfaced.
- `kill_assessments` gained a `confirmation` column and `thesis_versions`
  gained `research_status`/`blocking_verification_required`; a new
  `evidence_sufficiency` table was added. `storage/db.py` now applies
  additive `ALTER TABLE` migrations for tables that already exist under an
  older shape, rather than relying on `CREATE TABLE IF NOT EXISTS` alone,
  which does nothing to an existing table's columns.
