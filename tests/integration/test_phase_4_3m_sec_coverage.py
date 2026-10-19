"""Phase 4.3M: SEC primary-source coverage and the Action Gate.

What is asserted is never a real filing's wording -- every issuer, accession,
date and sentence is SYNTHETIC (see ``_sec_issuer_fixtures``). The point is the
mechanism, across issuers with different item mixes:

* every 8-K in the window is accounted for PER ACCESSION (read / beyond the cap /
  out of budget / failed / not read by policy / not interpretable), never only in a note;
* the item policy decides what is read, what is merely reported as NOT EXAMINED,
  and what blocks because no extractor exists for it;
* the 8-K and the 10-K / 10-Q shares of the run's GET budget are separate and fail closed;
* an unread / unreadable filing withholds the Action, and a resume refuses a record
  written under another scope and judges the RESTORED ledger, not this invocation's;
* the documents this system does NOT read are shown to let an Action through, so the
  residual risk is a tested fact, not a footnote.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import investment_research.orchestrator.pipeline as pipeline_module
from investment_research.collectors.base import CollectionResult
from investment_research.orchestrator import resume as resume_module
from investment_research.reporting.report import render_report
from investment_research.research import sec_material_8k as m8k
from investment_research.research import sec_periodic_reports as periodic
from investment_research.research.sec_coverage import (
    FINANCING_RELEASE_CONDITIONS,
    CoverageStatus,
    PolicyClass,
    classify_items,
    ledger_consistency_reasons,
)
from investment_research.research.sec_live_smoke import (
    BODY_TOO_LARGE_ERROR,
    AllowlistedHttpClient,
    MaxRequestsExceededError,
)
from investment_research.schemas.enums import (
    EvidenceClass,
    FactCategory,
    FetchOutcome,
    KillCategory,
    KillLevel,
    ResearchStatus,
    RunStatus,
)
from investment_research.scoring.kill_gate import evaluate_kill_gate
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository
from tests.conftest import make_fact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sec_issuer_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.integration

TODAY = fx.TODAY
TICKER = "DEMOBIO"
UNVERIFIED = m8k.UNVERIFIED_PREFIX
UNVERIFIED_PERIODIC = periodic.UNVERIFIED_PERIODIC_PREFIX
FINANCING_PREFIX = "FINANCING CONDITION REQUIRES VERIFICATION:"

GOING_CONCERN = (
    "Our independent registered public accounting firm has included an explanatory paragraph "
    "expressing substantial doubt about our ability to continue as a going concern."
)
TENK = fx.Filing("10-K", "2026-03-10", "", 900, primary_name="tenk.htm",
                 primary_text=fx.periodic_body("Our operations continued as planned. "))
TENQ = fx.Filing("10-Q", "2026-08-10", "", 901, primary_name="tenq.htm",
                 primary_text=fx.periodic_body("Liquidity is described in Note 1. "))


def clean_issuer(extra: list[fx.Filing] | None = None, **kw) -> fx.Issuer:
    """A filer whose latest 10-K and 10-Q exist and say nothing notable."""
    filings = [fx.Filing(**{**TENK.__dict__}), fx.Filing(**{**TENQ.__dict__}), *(extra or [])]
    filings.sort(key=lambda f: (f.filed, f.n), reverse=True)
    return fx.Issuer(1700001, filings, **kw)


def collect_both(issuer, *, http="default", older=0, today=TODAY, **kwargs):
    fake = fx.FakeSec(issuer.routes()) if http == "default" else http
    common = {"cik": issuer.cik, "recent": issuer.recent(), "today": today,
              "older_listing_pages": older, **kwargs}
    material = m8k.collect_material_8k_disclosures(TICKER, http=fake, **common)
    periodic_result = periodic.collect_periodic_reports(TICKER, http=fake, **common)
    return material, periodic_result, fake


def rows_by_accession(*results: CollectionResult) -> dict[str, dict]:
    return {row["accession"]: row for r in results for row in r.coverage}


# =============================================================================
# 1. The item policy
# =============================================================================
@pytest.mark.parametrize(
    ("items", "policy"),
    [
        ((), PolicyClass.MATERIAL),  # nothing stated: cannot be judged, so read
        (("8.01",), PolicyClass.MATERIAL),
        (("1.01", "3.02", "9.01"), PolicyClass.MATERIAL),
        (("3.01",), PolicyClass.MATERIAL),
        (("4.02",), PolicyClass.MATERIAL),
        (("2.02", "9.01"), PolicyClass.PERIODIC_RESULTS),
        (("5.02",), PolicyClass.ADMINISTRATIVE),
        (("5.02", "5.07", "9.01"), PolicyClass.ADMINISTRATIVE),
        (("5.02", "8.01"), PolicyClass.MATERIAL),  # the strictest item decides
        (("2.02", "8.01"), PolicyClass.MATERIAL),
        (("1.05",), PolicyClass.UNSUPPORTED),  # no extractor -> blocks
        (("5.02", "1.03"), PolicyClass.UNSUPPORTED),
        (("8.01", "3.03"), PolicyClass.UNSUPPORTED),
        (("9.99",), PolicyClass.UNSUPPORTED),  # an unknown code is never assumed harmless
    ],
)
def test_item_policy_classes(items, policy):
    assert classify_items(items) is policy


def test_policy_sets_are_disjoint_and_the_read_set_is_what_the_scope_says():
    from investment_research.research import sec_coverage as c

    assert not (c.MATERIAL_ITEMS & c.ADMINISTRATIVE_ITEMS)
    assert not (c.PERIODIC_RESULTS_ITEMS & c.ADMINISTRATIVE_ITEMS)
    assert m8k.MATERIAL_8K_ITEMS == c.MATERIAL_ITEMS | c.PERIODIC_RESULTS_ITEMS
    assert {"1.01", "2.03", "3.02", "3.01"} <= m8k.MATERIAL_8K_ITEMS


# =============================================================================
# 2. Per-accession tracking across issuers with different item layouts
# =============================================================================
def test_alpha_biotech_every_accession_is_accounted_for_and_nothing_is_unverified():
    issuer = fx.alpha_biotech()
    material, _, fake = collect_both(issuer)
    assert not material.errors, material.errors
    rows = {r["accession"]: r for r in material.coverage}
    assert len(rows) == len(issuer.filings) == 18  # one row per 8-K, none missing

    read = [r for r in rows.values() if r["status"] == "READ"]
    not_read = [r for r in rows.values() if r["status"] == "NOT_READ_POLICY"]
    assert len(read) == 7 and len(not_read) == 11  # 5 material + newest 2 earnings / 10 officers + 1 older earnings
    assert all(r["blocking"] is False for r in rows.values())
    assert sum(r["policy"] == "ADMINISTRATIVE" for r in not_read) == 10
    older_earnings = [r for r in not_read if r["policy"] == "PERIODIC_RESULTS"]
    assert [r["filing_date"] for r in older_earnings] == ["2026-03-02"]  # the 3rd-newest 2.02
    # all reads fit the 8-K share of the GET budget, ADMINISTRATIVE ones cost nothing
    assert len(fake.requested) <= m8k.MAX_OUTBOUND_GETS
    assert not any("0000010000" in u for u in fake.requested)


def test_alpha_listing_notices_and_the_trial_result_become_company_claim_facts():
    material, _, _ = collect_both(fx.alpha_biotech())
    by_value = {}
    for fact in material.raw_facts:
        by_value.setdefault(fact.value, []).append(fact)
    assert len(by_value["exchange listing notice reported by the issuer (issuer SEC disclosure)"]) == 2
    assert len(by_value["primary endpoint was not met (issuer 8-K disclosure)"]) == 1
    assert all(f.company_claim for f in material.raw_facts)


def test_alpha_financing_exhibit_is_listed_as_not_read_and_does_not_block():
    material, _, _ = collect_both(fx.alpha_biotech())
    march = next(r for r in material.coverage if r["filing_date"] == "2026-03-12")
    assert march["status"] == "READ" and march["blocking"] is False
    assert any("EX-10.1" in d and "not supported" in d for d in march["documents_not_read"])


def test_beta_device_financing_only_issuer_reads_its_financing_items():
    material, _, fake = collect_both(fx.beta_device())
    rows = {r["filing_date"]: r for r in material.coverage}
    assert rows["2026-09-02"]["status"] == "READ" and rows["2026-06-20"]["status"] == "READ"
    assert rows["2026-04-04"]["status"] == "NOT_READ_POLICY"  # 5.07 only
    assert any("ex10-1.htm" in d for d in rows["2026-09-02"]["documents_not_read"])
    assert not material.errors


def test_gamma_heavy_filer_budget_exhaustion_is_per_accession_and_fails_closed():
    issuer = fx.gamma_heavy(20)
    fake = fx.FakeSec(issuer.routes())
    material = m8k.collect_material_8k_disclosures(
        TICKER, cik=issuer.cik, recent=issuer.recent(), http=m8k.SubBudgetHttp(fake, m8k.MAX_OUTBOUND_GETS),
        today=TODAY,
    )
    rows = material.coverage
    assert len(rows) == 22  # 20 material + 2 unsupported, each accounted for
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    assert by_status["READ"] == m8k.MAX_FILINGS == 12
    assert by_status["UNREAD_BEYOND_CAP"] == 8
    assert by_status["NOT_READ_UNSUPPORTED"] == 2
    # every blocking row has an UNVERIFIED reason naming its accession
    for r in rows:
        if r["blocking"]:
            assert any(r["accession"] in e and e.startswith(UNVERIFIED) for e in material.errors), r
    assert len([e for e in material.errors if e.startswith(UNVERIFIED)]) == 10
    assert len(fake.requested) <= m8k.MAX_OUTBOUND_GETS  # 3 per filing here: 12 filings = 36
    # the OLDEST material filings are the ones left beyond the cap
    beyond = sorted(r["filing_date"] for r in rows if r["status"] == "UNREAD_BEYOND_CAP")
    read = sorted(r["filing_date"] for r in rows if r["status"] == "READ")
    assert max(beyond) <= min(read)


def test_ledger_and_errors_agree_for_every_issuer():
    for issuer in (fx.alpha_biotech(), fx.beta_device(), fx.gamma_heavy(15), clean_issuer()):
        material, periodic_result, _ = collect_both(issuer)
        for result, prefix in ((material, UNVERIFIED), (periodic_result, UNVERIFIED_PERIODIC)):
            summary = {"errors": result.errors, "coverage": result.coverage}
            assert ledger_consistency_reasons(summary, prefix=prefix) == []


def test_priority_material_filings_are_read_before_earnings_releases():
    filings = [fx.Filing("8-K", f"2026-09-{d:02d}", "8.01", 400 + d, [fx.ex99("x.")]) for d in range(1, 13)]
    filings += [
        fx.Filing("8-K", "2026-09-28", "2.02,9.01", 500, [fx.ex99("Results.")]),
        fx.Filing("8-K", "2026-09-27", "2.02,9.01", 501, [fx.ex99("Results.")]),
    ]
    issuer = fx.Issuer(1900007, filings)
    material, _, _ = collect_both(issuer)
    rows = {r["accession"][-3:]: r for r in material.coverage}
    assert rows["500"]["status"] == "UNREAD_BEYOND_CAP" and rows["501"]["status"] == "UNREAD_BEYOND_CAP"
    assert rows["500"]["blocking"] and rows["501"]["blocking"]  # a blocked earnings release is named
    assert sum(r["status"] == "READ" for r in rows.values()) == 12


def test_offline_every_candidate_is_unread_and_blocking_and_unsupported_items_still_block():
    issuer = fx.gamma_heavy(3)
    material = m8k.collect_material_8k_disclosures(
        TICKER, cik=issuer.cik, recent=issuer.recent(), http=None, today=TODAY,
        unavailable_reason="offline run",
    )
    statuses = {r["status"] for r in material.coverage}
    assert statuses == {"UNREAD_NO_CLIENT", "NOT_READ_UNSUPPORTED"}
    assert all(r["blocking"] for r in material.coverage)


def test_unsupported_item_blocks_without_being_read():
    issuer = fx.gamma_heavy(0)
    fake = fx.FakeSec(issuer.routes())
    material = m8k.collect_material_8k_disclosures(
        TICKER, cik=issuer.cik, recent=issuer.recent(), http=fake, today=TODAY,
    )
    assert fake.requested == []  # nothing is fetched for a filing we cannot interpret
    assert len([e for e in material.errors if "no extractor" in e]) == 2
    assert {r["policy"] for r in material.coverage} == {"UNSUPPORTED"}


def test_listing_truncation_is_detected_not_assumed_complete():
    issuer = fx.Issuer(1700002, [fx.Filing("8-K", "2026-08-01", "8.01", 1, [fx.ex99("x.")])], older_pages=2)
    material, periodic_result, _ = collect_both(issuer, older=2)
    assert any("reaches back only to 2026-08-01" in e for e in material.errors)
    assert any("reaches back only to" in e for e in periodic_result.errors)
    # a listing that DOES reach back over the window is not flagged
    full = fx.Issuer(1700003, [fx.Filing("8-K", "2025-10-01", "8.01", 1, [fx.ex99("x.")])], older_pages=2)
    material2, _, _ = collect_both(full, older=2)
    assert not any("reaches back only" in e for e in material2.errors)


# =============================================================================
# 3. The listed gate-visible events (and their negatives)
# =============================================================================
@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("On July 1, 2026, the Company received a letter from Nasdaq notifying it that the bid "
         "price had closed below the minimum bid price requirement for continued listing.",
         "LISTING_DEFICIENCY_REPORTED"),
        ("On Feb 9, 2026 the Company received a notice from Nasdaq that it was not in compliance "
         "with the minimum stockholders' equity requirement for continued listing.",
         "LISTING_DEFICIENCY_REPORTED"),
        (GOING_CONCERN, "GOING_CONCERN_DOUBT_REPORTED"),
        ("The Audit Committee dismissed Alpha LLP as the Company's independent registered public "
         "accounting firm.", "AUDITOR_CHANGE_REPORTED"),
        ("The Company concluded that its previously issued financial statements should no longer "
         "be relied upon and will be restated.", "ACCOUNTING_RELIABILITY_REPORTED"),
        ("Management identified a material weakness in internal control over financial reporting.",
         "ACCOUNTING_RELIABILITY_REPORTED"),
    ],
)
def test_reported_events_become_facts_the_kill_gate_can_see(sentence, kind):
    assert [s.kind for s in m8k.extract_statements(sentence)] == [kind]
    gate = evaluate_kill_gate(
        [make_fact(sentence, category=FactCategory.OTHER, value=m8k._FACT_VALUE[kind],
                   evidence_class=EvidenceClass.COMPANY_CLAIM, company_claim=True)],
        [],
    )
    assert any(a.level >= KillLevel.K3 and a.findings for a in gate.assessments), kind


@pytest.mark.parametrize(
    "sentence",
    [
        "We could be subject to delisting if our bid price stays below the minimum bid price.",
        "If we are unable to raise capital, there may be substantial doubt about our ability to "
        "continue as a going concern.",
        "Management concluded that substantial doubt about our ability to continue as a going "
        "concern was alleviated.",
        "We did not identify any material weakness in our internal control over financial reporting.",
        "The material weakness identified in 2024 has been remediated.",
        "The Company has not changed its independent registered public accounting firm.",
        "Our common stock is listed on the Nasdaq Capital Market.",
    ],
)
def test_hypothetical_resolved_or_ordinary_wording_is_not_an_event_fact(sentence):
    assert [s.kind for s in m8k.extract_statements(sentence)] == []


# =============================================================================
# 4. Periodic reports: scope, budget, reading, fail closed
# =============================================================================
def test_periodic_selection_takes_the_newest_10k_and_the_10q_after_it():
    issuer = fx.Issuer(1700010, [
        fx.Filing("10-K", "2025-03-01", "", 1), fx.Filing("10-K", "2026-03-10", "", 2),
        fx.Filing("10-Q", "2025-11-05", "", 3),  # older than the newest 10-K: not selected
        fx.Filing("10-Q", "2026-05-08", "", 4), fx.Filing("10-Q", "2026-08-10", "", 5),
        fx.Filing("10-K/A", "2026-04-20", "", 6),
    ])
    selection = periodic.select_periodic_reports(issuer.recent(), today=TODAY)
    assert [c.filing_date for c in selection.selected] == ["2026-03-10", "2026-08-10"]
    amend = [e for e in selection.entries if e.form == "10-K/A"]
    assert len(amend) == 1 and amend[0].status is CoverageStatus.NOT_READ_POLICY
    assert not amend[0].blocking


def test_a_required_report_missing_from_the_listing_is_not_found_and_blocks():
    only_8k = fx.Issuer(1700011, [fx.Filing("8-K", "2026-08-01", "8.01", 1)])
    sel = periodic.select_periodic_reports(only_8k.recent(), today=TODAY)
    assert sel.selected == [] and [e.status for e in sel.entries] == [CoverageStatus.NOT_FOUND]
    result = periodic.collect_periodic_reports(
        TICKER, cik=1700011, recent=only_8k.recent(), http=fx.FakeSec({}), today=TODAY,
    )
    assert result.errors and result.errors[0].startswith(UNVERIFIED_PERIODIC)
    assert "no 10-K filed within" in result.errors[0]


def test_a_10q_is_required_once_it_is_due_but_not_right_after_the_10k():
    tenk_only = fx.Issuer(1700012, [fx.Filing("10-K", "2026-03-10", "", 1)])
    due = periodic.select_periodic_reports(tenk_only.recent(), today=TODAY)  # 220 days later
    assert [e.form for e in due.entries if e.status is CoverageStatus.NOT_FOUND] == ["10-Q"]
    fresh = periodic.select_periodic_reports(tenk_only.recent(), today=date(2026, 4, 1))
    assert fresh.entries == []  # not due yet


def test_clean_periodic_reports_are_read_and_do_not_block():
    issuer = clean_issuer()
    _, result, fake = collect_both(issuer)
    assert not result.errors
    assert {r["status"] for r in result.coverage} == {"READ"}
    assert len(fake.requested) == 4  # index + body, twice
    assert len(fake.requested) <= periodic.MAX_OUTBOUND_GETS == 8


def test_going_concern_in_a_10k_is_a_company_claim_fact_the_gate_acts_on():
    tenk = fx.Filing(**{**TENK.__dict__, "primary_text": fx.periodic_body(GOING_CONCERN + " ")})
    issuer = fx.Issuer(1700013, [tenk, fx.Filing(**{**TENQ.__dict__})])
    _, result, _ = collect_both(issuer)
    (fact,) = list(result.raw_facts)
    assert fact.claim == GOING_CONCERN and fact.company_claim is True
    assert fact.category is FactCategory.LIQUIDITY
    assert fact.source.title.startswith("10-K ") and fact.source.accession.endswith("-26-000900")
    assert not result.errors  # read and classified; the FINDING (not a read failure) will block


@pytest.mark.parametrize(
    "outcome",
    [FetchOutcome.TIMEOUT, FetchOutcome.NOT_FOUND],
    ids=["timeout", "not-found"],
)
def test_an_unread_periodic_report_is_unverified_and_blocking(outcome):
    issuer = clean_issuer()
    routes = issuer.routes()
    tenq = next(f for f in issuer.filings if f.form == "10-Q")
    routes[f"{issuer.base(tenq)}/{tenq.primary_name}"] = outcome
    _, result, _ = collect_both(issuer, http=fx.FakeSec(routes))
    unread = [r for r in result.coverage if r["form"] == "10-Q"]
    assert unread[0]["status"] == "UNREAD_FETCH_FAILED" and unread[0]["blocking"]
    assert any(e.startswith(UNVERIFIED_PERIODIC) and unread[0]["accession"] in e for e in result.errors)


def test_a_too_short_periodic_body_is_invalid_not_clean():
    issuer = clean_issuer()
    routes = issuer.routes()
    tenk = next(f for f in issuer.filings if f.form == "10-K")
    routes[f"{issuer.base(tenk)}/{tenk.primary_name}"] = "<p>too short</p>"
    _, result, _ = collect_both(issuer, http=fx.FakeSec(routes))
    row = next(r for r in result.coverage if r["form"] == "10-K")
    assert row["status"] == "UNREAD_INVALID" and row["blocking"]


def test_offline_periodic_reports_are_unread_and_blocking():
    issuer = clean_issuer()
    result = periodic.collect_periodic_reports(
        TICKER, cik=issuer.cik, recent=issuer.recent(), http=None, today=TODAY,
        unavailable_reason="offline run",
    )
    assert {r["status"] for r in result.coverage} == {"UNREAD_NO_CLIENT"}
    assert len([e for e in result.errors if e.startswith(UNVERIFIED_PERIODIC)]) == 2


def test_unclassified_periodic_sentences_are_counted_not_blocking_and_not_hidden():
    ambiguous = "The primary endpoint did not show a statistically significant improvement. "
    tenk = fx.Filing(**{**TENK.__dict__, "primary_text": fx.periodic_body(ambiguous)})
    issuer = fx.Issuer(1700014, [tenk, fx.Filing(**{**TENQ.__dict__})])
    _, result, _ = collect_both(issuer)
    row = next(r for r in result.coverage if r["form"] == "10-K")
    assert row["status"] == "READ" and row["blocking"] is False
    assert row["unclassified_statements"] == 1  # visible in the ledger
    assert not result.errors


# =============================================================================
# 5. The GET budget: separate shares, one hard total, a body-size limit
# =============================================================================
def test_sub_budget_view_stops_new_requests_and_leaves_the_other_share():
    fake = fx.FakeSec({"https://www.sec.gov/a": "x" * 10})
    view = m8k.SubBudgetHttp(fake, 2)
    view.get("https://www.sec.gov/a")
    view.get("https://www.sec.gov/a")
    with pytest.raises(MaxRequestsExceededError):
        view.get("https://www.sec.gov/a")
    other = m8k.SubBudgetHttp(fake, 2)  # a second view starts from the client's current count
    other.get("https://www.sec.gov/a")
    assert fake.attempts_made == 3


def test_the_run_total_is_the_sum_of_the_two_shares():
    from investment_research import cli

    assert cli.SEC_DOCUMENT_MAX_GETS == m8k.MAX_OUTBOUND_GETS + periodic.MAX_OUTBOUND_GETS == 56
    assert m8k.MAX_OUTBOUND_GETS == m8k.MAX_FILINGS * m8k.GETS_PER_FILING
    assert periodic.MAX_OUTBOUND_GETS == periodic.MAX_REPORTS * periodic.GETS_PER_REPORT


def test_an_exhausted_share_leaves_periodic_reports_unread_and_blocking_not_clean():
    issuer = clean_issuer()
    fake = fx.FakeSec(issuer.routes())
    result = periodic.collect_periodic_reports(
        TICKER, cik=issuer.cik, recent=issuer.recent(), http=m8k.SubBudgetHttp(fake, 1),
        today=TODAY,
    )
    statuses = {r["form"]: r["status"] for r in result.coverage}
    assert statuses["10-K"] == "UNREAD_BUDGET" and statuses["10-Q"] == "UNREAD_BUDGET"
    assert len(result.errors) == 2 and all(e.startswith(UNVERIFIED_PERIODIC) for e in result.errors)


class _BigBodyHandler(BaseHTTPRequestHandler):
    hits = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).hits += 1
        body = b"A" * 5000
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_an_oversized_body_is_refused_not_truncated_and_not_retried():
    _BigBodyHandler.hits = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _BigBodyHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = AllowlistedHttpClient(
            user_agent="phase-4-3m-test/1.0", allowed_hosts=frozenset({"127.0.0.1"}), timeout=2.0,
            rate_limit_rps=1000.0, max_retries=2, max_requests=6, max_body_bytes=1000, source="t",
        )
        result = client.get(f"http://127.0.0.1:{httpd.server_port}/big")
    finally:
        httpd.shutdown()
    assert result.outcome is FetchOutcome.ERROR and result.error == BODY_TOO_LARGE_ERROR
    assert result.body == b"" and not result.ok
    assert _BigBodyHandler.hits == 1 and client.attempts_made == 1  # a retry would be the same body


# =============================================================================
# 6. The Action Gate (noise-free pipeline: the baseline genuinely reaches an Action)
# =============================================================================
def run_pipeline(monkeypatch, material, periodic_result, *, repo=None, run_id="m-run", resume=False,
                 skip_harness=False):
    if not skip_harness:
        fx.install_noise_free_pipeline_harness(monkeypatch)
    monkeypatch.setattr(resume_module, "is_stale", lambda _f, _t: False)
    repo = repo or Repository(open_db(":memory:"))
    company, price, aliases, base = fx.demobio_base()
    result = fx.pipeline_for(repo).run(
        TICKER, company, [base, material, periodic_result], price=price, aliases=aliases,
        run_id=run_id, resume=resume,
    )
    return result, repo


def blocking(result, prefix):
    return [r for r in result.verdict.blocking_verification_required if r.startswith(prefix)]


def test_baseline_clean_filer_reaches_an_action(monkeypatch):
    material, periodic_result, _ = collect_both(clean_issuer())
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    assert result.context.status is RunStatus.COMPLETE
    assert result.verdict.action is not None
    assert len(result.sec_coverage) == 2  # the ledger travels with the result


def test_heavy_filer_beyond_the_cap_and_unsupported_items_withhold_the_action(monkeypatch):
    issuer = fx.gamma_heavy(15)
    issuer.filings += [fx.Filing(**{**TENK.__dict__}), fx.Filing(**{**TENQ.__dict__})]
    fake = fx.FakeSec(issuer.routes())
    kw = {"cik": issuer.cik, "recent": issuer.recent(), "today": TODAY}
    material = m8k.collect_material_8k_disclosures(
        TICKER, http=m8k.SubBudgetHttp(fake, m8k.MAX_OUTBOUND_GETS), **kw,
    )
    periodic_result = periodic.collect_periodic_reports(
        TICKER, http=m8k.SubBudgetHttp(fake, periodic.MAX_OUTBOUND_GETS), **kw,
    )
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    reasons = blocking(result, UNVERIFIED)
    assert len(reasons) == 3 + 2  # 15-12 beyond the cap + 2 uninterpretable item codes
    assert any("no extractor" in r for r in reasons)
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION
    assert not blocking(result, UNVERIFIED_PERIODIC)  # the 10-K / 10-Q share was not starved


def test_an_unread_10q_withholds_the_action_by_itself(monkeypatch):
    issuer = clean_issuer()
    routes = issuer.routes()
    tenq = next(f for f in issuer.filings if f.form == "10-Q")
    routes[f"{issuer.base(tenq)}/{tenq.primary_name}"] = FetchOutcome.TIMEOUT
    material, periodic_result, _ = collect_both(issuer, http=fx.FakeSec(routes))
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    assert blocking(result, UNVERIFIED_PERIODIC) and not blocking(result, UNVERIFIED)
    assert result.verdict.action is None


def _gate_over(repo, result, needle: str):
    """The Kill Gate over ONLY the stored fact(s) that quote ``needle`` -- the
    DEMOBIO fixture carries facts of its own (a decision-grade going-concern
    one among them), which must not be mistaken for what these filings add."""
    facts = [f for f in repo.facts_for_resume(result.context.run_id) if needle in f.claim]
    assert facts, needle
    assert all(f.evidence_class is EvidenceClass.COMPANY_CLAIM and not f.is_decision_grade for f in facts)
    return evaluate_kill_gate(facts, [])


def test_going_concern_in_the_annual_report_reaches_the_kill_gate_as_a_provisional_k4(monkeypatch):
    tenk = fx.Filing(**{**TENK.__dict__, "primary_text": fx.periodic_body(GOING_CONCERN + " ")})
    issuer = fx.Issuer(1700013, [tenk, fx.Filing(**{**TENQ.__dict__})])
    material, periodic_result, _ = collect_both(issuer)
    result, repo = run_pipeline(monkeypatch, material, periodic_result)
    gate = _gate_over(repo, result, "substantial doubt")
    capital = next(a for a in gate.assessments if a.category == KillCategory.CAPITAL_KILL)
    assert capital.level is KillLevel.K4 and capital.confirmation.value == "PROVISIONAL"
    assert capital in gate.provisional_major_or_worse  # what withholds the Action


def test_a_listing_notice_in_an_8k_reaches_the_kill_gate_as_a_provisional_k3(monkeypatch):
    issuer = clean_issuer([fx.Filing("8-K", "2026-07-02", "3.01", 3, primary_text=fx.primary_8k(
        "On July 1, 2026, the Company received a letter from the Nasdaq Listing Qualifications "
        "Department notifying it that the bid price had closed below the minimum bid price "
        "requirement for continued listing. "
    ))])
    material, periodic_result, _ = collect_both(issuer)
    assert not material.errors
    result, repo = run_pipeline(monkeypatch, material, periodic_result)
    gate = _gate_over(repo, result, "minimum bid price")
    listing = next(a for a in gate.assessments if a.category == KillCategory.LIQUIDITY_KILL)
    assert listing.level is KillLevel.K3 and listing.confirmation.value == "PROVISIONAL"
    assert listing in gate.provisional_major_or_worse


# =============================================================================
# 7. Financing condition: explicit blocker, explicit release conditions, no auto-release
# =============================================================================
FIN_SENTENCE = (
    "The Second Closing Trigger Date shall occur upon satisfaction or waiver of the Milestone set "
    "forth in the Purchase Agreement, being the Company's receipt of written confirmation from the "
    "FDA that a statistically significant improvement in the primary endpoint of the trial would "
    "be sufficient to support a marketing application, and the Investors shall purchase the "
    "second tranche of Shares at the Second Closing."
)
CLOSED_SENTENCE = (
    "On October 1, 2026, the Company completed the Second Closing and received gross proceeds "
    "of $5 million."
)


def financing_issuer(extra: list[fx.Filing] | None = None) -> fx.Issuer:
    return clean_issuer([fx.Filing("8-K", "2026-09-02", "1.01,3.02", 5, [fx.ex99(FIN_SENTENCE)]), *(extra or [])])


def test_financing_condition_reason_states_what_clears_it_and_that_nothing_automatic_does(monkeypatch):
    material, periodic_result, _ = collect_both(financing_issuer())
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    (reason,) = blocking(result, FINANCING_PREFIX)
    assert FINANCING_RELEASE_CONDITIONS in reason
    for required in ("executed financing agreement", "later issuer SEC filing", "FDA's own communication",
                     "COMPANY_CLAIM", "No later issuer statement"):
        assert required in reason
    assert result.verdict.action is None
    assert not blocking(result, UNVERIFIED)  # the blocker is the financing condition alone


def test_a_later_issuer_statement_that_the_closing_happened_does_not_release_it(monkeypatch):
    later = fx.Filing("8-K", "2026-10-02", "3.02", 6, [fx.ex99(CLOSED_SENTENCE)])
    material, periodic_result, _ = collect_both(financing_issuer([later]))
    assert [f.claim for f in material.raw_facts if "Second Closing Trigger" in f.claim]
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    assert blocking(result, FINANCING_PREFIX)  # still blocked: the closing is another COMPANY_CLAIM
    assert result.verdict.action is None


def test_financing_blocker_survives_resume_and_is_judged_from_restored_evidence(monkeypatch):
    material, periodic_result, _ = collect_both(financing_issuer())
    first, repo = run_pipeline(monkeypatch, material, periodic_result, run_id="fin-resume")
    assert blocking(first, FINANCING_PREFIX)
    clean_m, clean_p, _ = collect_both(clean_issuer())  # this invocation's fresh read is clean
    second, _ = run_pipeline(monkeypatch, clean_m, clean_p, repo=repo, run_id="fin-resume", resume=True)
    assert blocking(second, FINANCING_PREFIX)
    assert second.verdict.action is None


# =============================================================================
# 8. Resume: both scopes, both paths, the restored ledger decides
# =============================================================================
PATHS = pytest.mark.parametrize("crash_before_verify", [True, False], ids=["collect-path", "verify-path"])


def first_run(monkeypatch, material, periodic_result, *, crash_before_verify):
    fx.install_noise_free_pipeline_harness(monkeypatch)
    monkeypatch.setattr(resume_module, "is_stale", lambda _f, _t: False)
    repo = Repository(open_db(":memory:"))
    company, price, aliases, base = fx.demobio_base()
    kwargs = {"price": price, "aliases": aliases, "run_id": "m-scope"}
    pipe = fx.pipeline_for(repo)
    if crash_before_verify:
        def boom(*_a, **_k):
            raise RuntimeError("simulated crash")

        with monkeypatch.context() as m:
            m.setattr(pipeline_module, "run_full_evidence_integrity_pass", boom)
            with pytest.raises(RuntimeError, match="simulated crash"):
                pipe.run(TICKER, company, [base, material, periodic_result], **kwargs)
    else:
        pipe.run(TICKER, company, [base, material, periodic_result], **kwargs)
    return repo


def edit_snapshot(repo, collector, edit):
    row = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = 'm-scope' AND stage = '_collect_snapshot'"
    ).fetchone()
    payload = json.loads(row["payload"])
    summary = next(c for c in payload["collectors"] if c["collector"] == collector)
    edit(summary)
    repo.conn.execute(
        "UPDATE run_checkpoints SET payload = ? WHERE run_id = 'm-scope' AND stage = '_collect_snapshot'",
        (json.dumps(payload),),
    )
    repo.conn.commit()


def resume(monkeypatch, repo):
    material, periodic_result, fake = collect_both(clean_issuer())  # a clean, CURRENT fresh read
    company, price, aliases, base = fx.demobio_base()
    result = fx.pipeline_for(repo).run(
        TICKER, company, [base, material, periodic_result], price=price, aliases=aliases,
        run_id="m-scope", resume=True,
    )
    return result


@PATHS
def test_both_scopes_are_saved_with_ids_in_the_collect_snapshot(monkeypatch, crash_before_verify):
    material, periodic_result, _ = collect_both(clean_issuer())
    repo = first_run(monkeypatch, material, periodic_result, crash_before_verify=crash_before_verify)
    row = repo.conn.execute(
        "SELECT payload FROM run_checkpoints WHERE run_id = 'm-scope' AND stage = '_collect_snapshot'"
    ).fetchone()
    summaries = {c["collector"]: c for c in json.loads(row["payload"])["collectors"]}
    assert summaries[m8k.MATERIAL_8K_COLLECTOR]["scope"]["id"] == m8k.MATERIAL_8K_SCOPE_ID
    assert summaries[periodic.PERIODIC_COLLECTOR]["scope"]["id"] == periodic.PERIODIC_SCOPE_ID
    assert summaries[periodic.PERIODIC_COLLECTOR]["scope"]["max_outbound_gets"] == 8
    assert summaries[periodic.PERIODIC_COLLECTOR]["coverage"]  # the ledger is saved too


@PATHS
def test_resume_with_current_scopes_adds_no_scope_reason_and_reports_the_restored_ledger(
    monkeypatch, crash_before_verify,
):
    material, periodic_result, _ = collect_both(clean_issuer())
    repo = first_run(monkeypatch, material, periodic_result, crash_before_verify=crash_before_verify)
    result = resume(monkeypatch, repo)
    assert not blocking(result, UNVERIFIED) and not blocking(result, UNVERIFIED_PERIODIC)
    assert {r["collector"] for r in result.sec_coverage} == {"sec_periodic_reports"}
    assert len(result.sec_coverage) == 2


@PATHS
@pytest.mark.parametrize("collector, prefix", [
    (m8k.MATERIAL_8K_COLLECTOR, UNVERIFIED),
    (periodic.PERIODIC_COLLECTOR, UNVERIFIED_PERIODIC),
], ids=["8-K-scope", "periodic-scope"])
@pytest.mark.parametrize("edit", [
    lambda s: s.pop("scope", None),
    lambda s: s.__setitem__("scope", {"id": "an-older-scope/v1"}),
    lambda s: s.__setitem__("scope", "garbage"),
], ids=["missing", "older", "malformed"])
def test_resume_refuses_a_record_from_another_scope_without_any_request(
    monkeypatch, crash_before_verify, collector, prefix, edit,
):
    material, periodic_result, _ = collect_both(clean_issuer())
    repo = first_run(monkeypatch, material, periodic_result, crash_before_verify=crash_before_verify)
    edit_snapshot(repo, collector, edit)
    result = resume(monkeypatch, repo)
    reasons = [r for r in blocking(result, prefix) if "selection scope" in r]
    assert len(reasons) == 1
    assert result.verdict.action is None
    assert result.verdict.research_status is ResearchStatus.BLOCKED_PENDING_VERIFICATION


@PATHS
def test_a_snapshot_that_predates_periodic_coverage_is_refused(monkeypatch, crash_before_verify):
    material, periodic_result, _ = collect_both(clean_issuer())
    sec_edgar = CollectionResult(collector="sec_edgar")
    fx.install_noise_free_pipeline_harness(monkeypatch)
    monkeypatch.setattr(resume_module, "is_stale", lambda _f, _t: False)
    repo = Repository(open_db(":memory:"))
    company, price, aliases, base = fx.demobio_base()
    # a collection with the SEC list and an 8-K record but NO periodic collector (pre-4.3M)
    fx.pipeline_for(repo).run(TICKER, company, [base, sec_edgar, material], price=price,
                              aliases=aliases, run_id="m-scope")
    second = fx.pipeline_for(repo).run(TICKER, company, [base, sec_edgar, material], price=price,
                                       aliases=aliases, run_id="m-scope", resume=True)
    assert any("predates periodic-report coverage" in r for r in blocking(second, UNVERIFIED_PERIODIC))
    assert second.verdict.action is None


@PATHS
def test_an_edited_ledger_cannot_hide_a_blocking_row(monkeypatch, crash_before_verify):
    issuer = clean_issuer()
    routes = issuer.routes()
    tenq = next(f for f in issuer.filings if f.form == "10-Q")
    routes[f"{issuer.base(tenq)}/{tenq.primary_name}"] = FetchOutcome.TIMEOUT
    material, periodic_result, _ = collect_both(issuer, http=fx.FakeSec(routes))
    repo = first_run(monkeypatch, material, periodic_result, crash_before_verify=crash_before_verify)

    def drop_errors(summary):  # the error list is emptied; the ledger still says UNREAD
        summary["errors"] = []

    edit_snapshot(repo, periodic.PERIODIC_COLLECTOR, drop_errors)
    result = resume(monkeypatch, repo)
    derived = [r for r in blocking(result, UNVERIFIED_PERIODIC) if "no matching unverified reason" in r]
    assert len(derived) == 1
    assert result.verdict.action is None


@PATHS
def test_a_malformed_ledger_row_blocks(monkeypatch, crash_before_verify):
    material, periodic_result, _ = collect_both(clean_issuer())
    repo = first_run(monkeypatch, material, periodic_result, crash_before_verify=crash_before_verify)
    edit_snapshot(repo, periodic.PERIODIC_COLLECTOR, lambda s: s["coverage"].append({"nonsense": 1}))
    result = resume(monkeypatch, repo)
    assert any("malformed" in r for r in blocking(result, UNVERIFIED_PERIODIC))


# =============================================================================
# 9. Report and JSON
# =============================================================================
def test_report_lists_every_accession_with_blocking_rows_marked(monkeypatch):
    issuer = fx.gamma_heavy(14)
    issuer.filings += [fx.Filing(**{**TENK.__dict__}), fx.Filing(**{**TENQ.__dict__})]
    fake = fx.FakeSec(issuer.routes())
    kw = {"cik": issuer.cik, "recent": issuer.recent(), "today": TODAY}
    material = m8k.collect_material_8k_disclosures(TICKER, http=m8k.SubBudgetHttp(fake, 48), **kw)
    periodic_result = periodic.collect_periodic_reports(TICKER, http=m8k.SubBudgetHttp(fake, 8), **kw)
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    report = render_report(result)
    assert "SEC FILING COVERAGE (per accession)" in report
    for row in result.sec_coverage:
        assert row["accession"] in report
    assert "UNREAD_BEYOND_CAP" in report and "NOT_READ_UNSUPPORTED" in report
    assert " *" in report and "NOT EXAMINED" in report


def test_alpha_report_states_what_was_not_examined(monkeypatch):
    material, periodic_result, _ = collect_both(fx.alpha_biotech())
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    report = render_report(result)
    assert "NOT_READ_POLICY" in report and "ADMINISTRATIVE" in report
    assert "EX-10.1" in report  # the unread agreement exhibit is visible


# =============================================================================
# 10. Document types this system does NOT read: an Action still reaches the user.
#     These tests exist so the residual risk is a demonstrated fact.
# =============================================================================
def test_UNSUPPORTED_agreement_exhibits_are_not_read_and_an_action_can_still_be_issued(monkeypatch):
    """A financing 8-K whose terms live only in an EX-10 agreement: the body
    is read and says nothing, the agreement is never read, the Action is issued."""
    issuer = clean_issuer([fx.Filing(
        "8-K", "2026-09-02", "1.01", 5, other_exhibits=[("ex10-1.htm", "EX-10.1")],
        primary_text=fx.primary_8k("The Company entered into the agreement described in Exhibit 10.1. "),
    )])
    material, periodic_result, fake = collect_both(issuer)
    assert not any("ex10-1" in u for u in fake.requested)
    row = next(r for r in material.coverage if r["filing_date"] == "2026-09-02")
    assert row["status"] == "READ" and row["documents_not_read"]
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    assert result.verdict.action is not None  # RESIDUAL RISK: terms in the exhibit are unread


def test_UNSUPPORTED_prospectuses_and_foreign_forms_are_not_tracked_and_an_action_can_be_issued(monkeypatch):
    """424B5 / S-3 / SC 13D / Form 4 in the listing are never read (only the
    metadata fact exists): they do not block."""
    issuer = clean_issuer([
        fx.Filing("424B5", "2026-09-03", "", 7), fx.Filing("S-3", "2026-08-01", "", 8),
        fx.Filing("SC 13D", "2026-07-01", "", 9), fx.Filing("4", "2026-06-01", "", 10),
    ])
    material, periodic_result, _ = collect_both(issuer)
    assert all(r["form"] in {"8-K", "8-K/A", "10-K", "10-Q"} for r in rows_by_accession(
        material, periodic_result).values())
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    assert result.verdict.action is not None  # RESIDUAL RISK: these filings' bodies are unread


def test_UNSUPPORTED_foreign_private_issuer_has_no_10k_and_blocks_instead_of_passing(monkeypatch):
    issuer = fx.Issuer(1700020, [fx.Filing("20-F", "2026-04-01", "", 1), fx.Filing("6-K", "2026-08-01", "", 2)])
    material, periodic_result, _ = collect_both(issuer)
    result, _ = run_pipeline(monkeypatch, material, periodic_result)
    assert any("no 10-K filed within" in r for r in blocking(result, UNVERIFIED_PERIODIC))
    assert result.verdict.action is None


# =============================================================================
# 11. Kill-rule vocabulary added for gate visibility (pos/neg)
# =============================================================================
@pytest.mark.parametrize("claim", [
    "The Audit Committee dismissed Alpha LLP as the independent registered public accounting firm.",
    "Beta LLP resigned as the Company's certifying accountant.",
])
def test_auditor_change_rule_reads_the_item_4_01_wording(claim):
    gate = evaluate_kill_gate([make_fact(claim, category=FactCategory.ACCOUNTING)], [])
    accounting = next(a for a in gate.assessments if a.category == KillCategory.ACCOUNTING_KILL)
    assert accounting.level >= KillLevel.K2


@pytest.mark.parametrize("claim", [
    "The Audit Committee approved the engagement of an independent registered public accounting firm.",
    "The Company's common stock is listed on the Nasdaq Capital Market.",
])
def test_unrelated_accounting_or_listing_wording_does_not_fire(claim):
    gate = evaluate_kill_gate([make_fact(claim, category=FactCategory.ACCOUNTING)], [])
    assert all(a.level < KillLevel.K2 for a in gate.assessments if a.category in (
        KillCategory.ACCOUNTING_KILL, KillCategory.LIQUIDITY_KILL))


# =============================================================================
# 12. Resume across issuers: the RESTORED ledger decides, whatever the layout
# =============================================================================
def _issuers_with_periodic():
    out = []
    for build in (fx.alpha_biotech, fx.beta_device, lambda: fx.gamma_heavy(15)):
        issuer = build()
        issuer.filings += [fx.Filing(**{**TENK.__dict__}), fx.Filing(**{**TENQ.__dict__})]
        out.append(issuer)
    return out


@PATHS
@pytest.mark.parametrize("issuer", _issuers_with_periodic(), ids=["alpha", "beta", "gamma"])
def test_resume_restores_the_first_invocations_ledger_not_the_fresh_clean_read(
    monkeypatch, crash_before_verify, issuer,
):
    fake = fx.FakeSec(issuer.routes())
    kw = {"cik": issuer.cik, "recent": issuer.recent(), "today": TODAY}
    material = m8k.collect_material_8k_disclosures(
        TICKER, http=m8k.SubBudgetHttp(fake, m8k.MAX_OUTBOUND_GETS), **kw,
    )
    periodic_result = periodic.collect_periodic_reports(
        TICKER, http=m8k.SubBudgetHttp(fake, periodic.MAX_OUTBOUND_GETS), **kw,
    )
    repo = first_run(monkeypatch, material, periodic_result, crash_before_verify=crash_before_verify)
    result = resume(monkeypatch, repo)  # this invocation's own read is the clean filer's

    restored = {(r["collector"], r["accession"], r["status"]) for r in result.sec_coverage}
    expected = {("sec_material_8k", r["accession"], r["status"]) for r in material.coverage}
    expected |= {("sec_periodic_reports", r["accession"], r["status"]) for r in periodic_result.coverage}
    assert restored == expected
    # the restored record's blocking rows still block (nothing is re-read or forgotten)
    expected_blocking = sum(1 for r in [*material.coverage, *periodic_result.coverage] if r["blocking"])
    reasons = blocking(result, UNVERIFIED) + blocking(result, UNVERIFIED_PERIODIC)
    assert len([r for r in reasons if "selection scope" not in r]) == expected_blocking
    if expected_blocking:
        assert result.verdict.action is None


# =============================================================================
# 13. A long report is read in bounded time
# =============================================================================
def test_a_multi_megabyte_periodic_report_is_extracted_in_reasonable_time():
    import time

    filler = (
        "We have incurred significant losses since inception and expect to continue to incur "
        "losses. Our product candidates may not receive regulatory approval. "
    ) * 20000
    assert len(filler) > 2_500_000
    started = time.monotonic()
    statements = m8k.extract_statements(filler + GOING_CONCERN)
    elapsed = time.monotonic() - started
    assert [s.kind for s in statements] == ["GOING_CONCERN_DOUBT_REPORTED"]
    assert elapsed < 60, elapsed
