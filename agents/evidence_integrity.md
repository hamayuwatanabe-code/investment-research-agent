# Agent 2 -- Evidence Integrity

**Agent id**: `evidence_integrity`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["evidence_integrity"]`
**Implementation**: `src/investment_research/agents/evidence_integrity.py`

## Purpose

Decide what each collected claim is actually worth: classify it, date it, and try to corroborate it.

## MUST see

- The raw facts from Agent 1.

## MUST NOT see

- Every downstream evaluation.

## Output contract

For each fact: `fact_id`, ticker, category, claim, source url/title/tier, publication_date, event_date, verified_status, confidence, company_claim, independent_confirmation, contradicting_evidence, materiality, notes.

## Hard rules

1. A company statement is a COMPANY_CLAIM no matter how primary the venue it was filed in. It becomes a VERIFIED_FACT only with independent confirmation from a different, non-company source.
2. Syndicated reprints and same-host republication are **not** independent confirmation. A wire story on five sites is one source.
3. Staleness is judged on the event date. An old event re-reported today is old.
4. Tier 4/5 alone can never establish a decision-grade fact.
5. Confidence is derived mechanically from tier, class, corroboration and recency. It is not a feeling.
