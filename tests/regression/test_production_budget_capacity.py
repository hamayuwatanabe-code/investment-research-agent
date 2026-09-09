"""Offline, no-API-call capacity plan for the production discovery budget.

This does NOT call the real Anthropic API and does NOT run Full DD. It
computes, from the SAME functions the production CLI actually uses to build
its research plan (``build_plan``, ``kill_queries``, ``build_research_batches``,
``_batch_reservation``), exactly how many mandatory queries a real run issues,
how they get batched, and what each batch's/query's own preflight
RESERVATION would be under production defaults (--token-budget 200000,
discovery stage = 30% = 60000) -- entirely deterministic, code-based
arithmetic, never a live measurement.

Two SEPARATE numbers are computed, and they must never be confused:

1. The RESERVATION total (CONFIRMED, deterministic): the exact sum
   ``_batch_reservation``/``LLMClient._default_reservation`` would compute
   for every batch/query production actually builds. This is the
   conservative UPPER BOUND the preflight check enforces -- it does not
   depend on what the live API would actually charge.
2. An ESTIMATED actual-cost RANGE, built only from the two empirical data
   points already on record in this project's own chat history (a 2-question
   smoke test: 31492 tokens / 2 searches; one production discovery batch:
   102192 tokens / 6 intents). This range is NEVER treated as a fixed
   per-search unit price -- batching amortizes shared prompt/output overhead
   differently at different batch sizes, and effort/thinking budgets vary --
   it is reported only as a plausible band, clearly labeled as an estimate.

The conclusion this file locks in as a regression (so it breaks loudly if
the underlying arithmetic ever silently changes): under CURRENT production
defaults, neither number fits the 60,000-token discovery stage quota, by a
wide margin, and splitting batches into smaller pieces does not change that
-- total required RESERVATION is a property of how many searches must be
issued, not how many calls carry them.
"""

from __future__ import annotations

import pytest

from investment_research.collectors.search import kill_queries
from investment_research.llm.client import LLMClient
from investment_research.research.adversarial import (
    _dedupe_semantically,
    _prioritize_domain_coverage,
    build_plan,
)
from investment_research.research.anthropic_web import (
    SEARCH_MAX_TOKENS,
    SEARCH_SYSTEM,
    _batch_reservation,
)
from investment_research.research.batching import ResearchIntent, build_research_batches

pytestmark = pytest.mark.regression

# Production defaults (cli.py's own --token-budget/DEFAULT_STAGE_QUOTAS).
PRODUCTION_TOTAL_BUDGET = 200_000
DISCOVERY_STAGE_SHARE = 0.30
DISCOVERY_STAGE_CAP = int(PRODUCTION_TOTAL_BUDGET * DISCOVERY_STAGE_SHARE)  # 60,000

# The two empirical data points already reported in this project's own chat
# history for this session -- NOT re-derived from a live call here, and
# NEVER read as a fixed per-search unit price (see module docstring).
_SMOKE_TEST_TOKENS, _SMOKE_TEST_SEARCHES = 31_492, 2
_PRODUCTION_BATCH_TOKENS, _PRODUCTION_BATCH_SEARCHES = 102_192, 6
_EMPIRICAL_PER_SEARCH_LOW = min(_SMOKE_TEST_TOKENS / _SMOKE_TEST_SEARCHES, _PRODUCTION_BATCH_TOKENS / _PRODUCTION_BATCH_SEARCHES)
_EMPIRICAL_PER_SEARCH_HIGH = max(_SMOKE_TEST_TOKENS / _SMOKE_TEST_SEARCHES, _PRODUCTION_BATCH_TOKENS / _PRODUCTION_BATCH_SEARCHES)

TICKER, COMPANY = "TESTCO", "Generic Biotech Holdings"


def _bear_bull_batches() -> tuple[list, list]:
    """Exactly what research/adversarial.py's _run_stance_batched() builds
    for a real run: mandatory templates -> domain-prioritized -> semantically
    deduped -> ResearchIntents -> build_research_batches()."""
    plan = build_plan(TICKER, COMPANY)
    bear_q = _prioritize_domain_coverage(plan.bear)
    bear_q, _ = _dedupe_semantically(bear_q)
    bull_q, _ = _dedupe_semantically(plan.bull)

    bear_intents = [
        ResearchIntent(intent_id=f"bear_{i}", domain=q.domain, question=q.query, priority=i)
        for i, q in enumerate(bear_q)
    ]
    bull_intents = [
        ResearchIntent(intent_id=f"bull_{i}", domain=q.domain, question=q.query, priority=i)
        for i, q in enumerate(bull_q)
    ]
    return build_research_batches(bear_intents), build_research_batches(bull_intents)


def _reservation_total(batches: list, *, effort: str = "low") -> int:
    total = 0
    for batch in batches:
        max_uses = min(len(batch.intents) + 1, 8)  # AnthropicWebResearchProvider default
        total += _batch_reservation(batch.intents, max_uses=max_uses, effort=effort)
    return total


def _kill_reservation_total(queries: list[str], *, effort: str = "low") -> int:
    """Kill's mandatory queries are NOT batched in production
    (agents/kill_agent.py's _search_via_research calls
    ResearchProvider.search() once per query) -- this mirrors that exactly,
    using the real LLMClient._default_reservation() formula so the number
    is never a hand-duplicated approximation of production's own math."""
    llm = LLMClient()  # no credentials needed for this pure arithmetic
    total = 0
    for query in queries:
        total += llm._default_reservation(
            system=SEARCH_SYSTEM,
            messages=[{"role": "user", "content": query}],
            tools=[{"type": "web_search_tool"}],  # any web_search-prefixed type triggers the reserve
            max_tokens=SEARCH_MAX_TOKENS,
            effort=effort,
        )
    return total


def test_production_capacity_plan_reservation_total_vastly_exceeds_the_discovery_stage_cap():
    """CONFIRMED (deterministic, code-based): the exact preflight
    RESERVATION production's own batching/kill code would compute for every
    mandatory query, summed, against the 60,000-token discovery stage cap.
    This is the conservative upper bound the system itself enforces before
    any real API cost is even considered."""
    bear_batches, bull_batches = _bear_bull_batches()
    bear_reserve = _reservation_total(bear_batches)
    bull_reserve = _reservation_total(bull_batches)

    kq = kill_queries(TICKER, COMPANY)
    kill_reserve_full = _kill_reservation_total(kq)

    # Kill/Bear overlap: kill_agent.py's _covered_by() already skips a kill
    # query whose distinctive tail matches an ALREADY-EXECUTED bear/bull
    # query text. Best case (every overlapping bear/bull query actually
    # executed) removes those from Kill's own reservation need; this is
    # fragile in practice (it only helps for whichever bear/bull queries
    # happen to survive the SAME budget crunch), so both numbers are
    # reported, never only the optimistic one.
    plan = build_plan(TICKER, COMPANY)
    bear_bull_texts = {q.query.lower() for q in (*plan.bear, *plan.bull)}
    kq_overlapping = [q for q in kq if q.lower() in bear_bull_texts]
    kq_best_case = [q for q in kq if q.lower() not in bear_bull_texts]
    kill_reserve_best_case = _kill_reservation_total(kq_best_case)

    total_worst_case = bear_reserve + bull_reserve + kill_reserve_full
    total_best_case = bear_reserve + bull_reserve + kill_reserve_best_case

    report_lines = [
        "",
        "=" * 78,
        "PRODUCTION DISCOVERY-BUDGET CAPACITY PLAN (offline, code-based, no API call)",
        "=" * 78,
        f"bear mandatory queries (post-dedup): {sum(len(b.intents) for b in bear_batches)} "
        f"in {len(bear_batches)} batch(es), reservation = {bear_reserve:,}",
        f"bull mandatory queries (post-dedup): {sum(len(b.intents) for b in bull_batches)} "
        f"in {len(bull_batches)} batch(es), reservation = {bull_reserve:,}",
        f"kill mandatory queries: {len(kq)} (UNBATCHED -- one search() call each), "
        f"reservation = {kill_reserve_full:,}",
        f"  of which {len(kq_overlapping)} exact-text-match a bear/bull query "
        "(kill_agent.py's own dedup skips these IF that bear/bull query already executed)",
        f"  best-case kill reservation (assuming all {len(kq_overlapping)} overlaps were skipped) "
        f"= {kill_reserve_best_case:,}",
        "-" * 78,
        f"TOTAL reservation, worst case:  {total_worst_case:,}",
        f"TOTAL reservation, best case:   {total_best_case:,}",
        f"discovery stage cap (30% of {PRODUCTION_TOTAL_BUDGET:,}): {DISCOVERY_STAGE_CAP:,}",
        f"SHORTFALL (worst case):  {total_worst_case - DISCOVERY_STAGE_CAP:,} tokens "
        f"({total_worst_case / DISCOVERY_STAGE_CAP:.1f}x the stage cap)",
        f"SHORTFALL (best case):   {total_best_case - DISCOVERY_STAGE_CAP:,} tokens "
        f"({total_best_case / DISCOVERY_STAGE_CAP:.1f}x the stage cap)",
        "-" * 78,
        f"ESTIMATED range (NOT a fixed unit price -- see module docstring): "
        f"{_EMPIRICAL_PER_SEARCH_LOW:,.0f}-{_EMPIRICAL_PER_SEARCH_HIGH:,.0f} tokens/search, "
        f"anchored to a {_SMOKE_TEST_SEARCHES}-search smoke test ({_SMOKE_TEST_TOKENS:,} tokens) "
        f"and a {_PRODUCTION_BATCH_SEARCHES}-intent production batch ({_PRODUCTION_BATCH_TOKENS:,} tokens).",
        "=" * 78,
    ]
    print("\n".join(report_lines))  # captured by `pytest -s` for reporting

    # CONFIRMED, deterministic finding: even the RESERVATION scheme alone
    # (a conservative upper bound, not a live measurement) cannot schedule
    # every mandatory query -- worst case AND best case -- within the
    # discovery stage's 60,000-token cap. Splitting batches into smaller
    # pieces changes how many CALLS carry the searches, never how many
    # tokens of reservation the searches themselves require.
    assert total_worst_case > DISCOVERY_STAGE_CAP * 3, (
        "this locks in the current shortfall as a regression: if a future change to templates, "
        "kill_queries(), the reservation formula, or the stage quota shrinks this gap, this "
        "assertion should be revisited deliberately, not silently"
    )
    assert total_best_case > DISCOVERY_STAGE_CAP * 3

    # Kill's own mandatory set, taken ENTIRELY ALONE (Bear/Bull consuming
    # nothing), still cannot fit -- the shortfall is not solely a Bear/Bull
    # problem; Kill sharing the same "discovery" stage budget is itself
    # already over capacity.
    assert kill_reserve_full > DISCOVERY_STAGE_CAP


def test_batching_kills_queries_does_not_reduce_the_reservation_total():
    """本番規模の確認: batching does not, by itself, reduce required
    consumption -- it only reduces how many CALLS carry the same searches.
    Hypothetically batching Kill's mandatory queries the same way Bear/Bull
    already are must not come back materially cheaper in RESERVATION terms
    (the scaled-by-max_uses reservation this system now uses on purpose
    means a batch's total reserve tracks its search count, not its call
    count) -- this is exactly why simply "splitting into smaller batches"
    was never going to close the gap on its own."""
    kq = kill_queries(TICKER, COMPANY)
    unbatched_reserve = _kill_reservation_total(kq)

    from investment_research.schemas.enums import ResearchDomain

    kill_intents = [
        ResearchIntent(intent_id=f"kill_{i}", domain=ResearchDomain.REGULATORY, question=q, priority=i)
        for i, q in enumerate(kq)
    ]
    batches = build_research_batches(kill_intents)
    batched_reserve = _reservation_total(batches)

    # Batching is not a free lunch: it does not come in materially UNDER
    # the unbatched total (it may even exceed it, since this system's
    # reservation now correctly scales with `max_uses`, i.e. with how many
    # searches a batch's own call may issue).
    assert batched_reserve >= unbatched_reserve * 0.8, (
        f"batched={batched_reserve:,} vs unbatched={unbatched_reserve:,} -- batching should never "
        "look like a large, free reduction in required reservation"
    )
