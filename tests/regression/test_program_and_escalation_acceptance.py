"""Live-validation acceptance test (requirement H): C + D + G together.

A fully generic, synthetic end-to-end scenario combining three defects fixed
together in this task:

* **C** (unresolved-question-driven escalation): the company has an
  approval-gated business with no evidence either way on regulator endpoint
  acceptability, which raises a MATERIAL, blocking unresolved question. The
  collector already found a Tier 1 SEC filing pointer (a bare "Form 8-K was
  filed" fact -- metadata, not the filing's substantive body). A search
  snippet alone could never confirm the material question; only fetching the
  filing's actual body can. Escalation must fetch that body, find the
  adverse regulator language, and turn it into a decision-grade fact.
* **D** (programme/thesis-relevance resolution): the company has an old,
  terminated, unrelated trial in a DIFFERENT indication, and a current,
  active lead trial (explicitly described as such) with a surrogate/
  biomarker endpoint in a different indication. The old trial must not
  auto-generate a company-level kill, and Science must evaluate the current
  trial, not the old one.
* **G** (decision-gate output consistency): with several required research
  domains never searched at all (no adversarial pass configured), the
  completeness gate must withhold the Action entirely -- FINAL_ACTION stays
  NONE -- and no rendered surface may leak an actionable label.

No hard-coded real-world ticker, company, trial name, meeting type or
conclusion appears anywhere in this file.
"""

from __future__ import annotations

from datetime import date

import pytest

from investment_research.collectors.base import CollectionResult
from investment_research.collectors.documents import Document
from investment_research.collectors.search import NullSearchProvider
from investment_research.orchestrator.isolation import Channel
from investment_research.orchestrator.pipeline import Pipeline
from investment_research.reporting.report import render_report
from investment_research.schemas.enums import (
    ContentKind,
    FactCategory,
    KillCategory,
    KillConfirmation,
    KillLevel,
    Provenance,
    ResearchPath,
    ResearchStatus,
    SourceTier,
)
from investment_research.schemas.fact import RawFact, Source, make_source_id
from investment_research.scoring.decision_gate_consistency import find_action_labels

pytestmark = pytest.mark.regression

TODAY = date(2026, 9, 6)
TICKER = "GENBIO"
NAME = "Generic Biotech Holdings"

OLD_TRIAL = "NCT10000001"  # historical, different indication, terminated
CURRENT_TRIAL = "NCT20000002"  # current lead programme, different indication
FILING_URL = "fixture://sec/GENBIO/8-K-endpoint-correspondence"

_ADVERSE_BODY = (
    "In written responses following its most recent correspondence with the sponsor, the "
    "regulator advised the company that the agency has not agreed the primary endpoint is "
    "adequate, and that the endpoint is not sufficient to establish effectiveness for the "
    "intended indication; an additional adequate and well-controlled trial would be required "
    "to support a marketing application."
)


class _FilingBodyProvider:
    """A ResearchProvider double that answers a fetch() for one known filing
    URL with its adverse body, and refuses any search (so the test proves
    escalation used the already-collected source rather than searching)."""

    name = "fake_research"
    path = ResearchPath.ANTHROPIC_WEB

    def __init__(self, bodies: dict[str, str]) -> None:
        self.bodies = bodies
        self.search_calls = 0
        self.fetch_calls: list[str] = []

    def available(self) -> tuple[bool, str]:
        return True, "ready"

    def search(self, query, *, agent_id: str = "research"):
        self.search_calls += 1
        from investment_research.research.provider import ResearchResult
        from investment_research.schemas.enums import FetchOutcome

        return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=self.path)

    def fetch(self, url: str, *, reason: str = "", agent_id: str = "research"):
        self.fetch_calls.append(url)
        text = self.bodies.get(url)
        if text is None:
            return None
        return Document(
            doc_id="filing_body",
            url=url,
            title="Form 8-K",
            publisher="SEC",
            published_date="2026-08-01",
            doc_type="filing",
            is_company_ir=False,
            text=text,
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.FIXTURE,
            research_path=self.path,
            tier=SourceTier.TIER_1,
        )


def _source(url: str, title: str, tier: SourceTier, event: str) -> Source:
    return Source(
        source_id=make_source_id(url, title),
        url=url,
        title=title,
        tier=tier,
        event_date=event,
        published_date=event,
        filing_date=event,
        provenance=Provenance.FIXTURE,
    )


FILING_SOURCE = _source(FILING_URL, "Form 8-K", SourceTier.TIER_1, "2026-08-01")
OLD_TRIAL_SOURCE = _source(f"fixture://ctgov/{OLD_TRIAL}", "Registry record", SourceTier.TIER_1, "2019-06-01")
CURRENT_TRIAL_SOURCE = _source(
    f"fixture://ctgov/{CURRENT_TRIAL}", "Registry record", SourceTier.TIER_1, "2026-06-01"
)
COMPANY_SOURCE = _source("fixture://ir/GENBIO/pr", "Company statement", SourceTier.TIER_2, "2026-06-01")


def _raw(claim: str, source: Source, category: FactCategory, *, company_claim: bool = False) -> RawFact:
    return RawFact(
        ticker=TICKER,
        category=category,
        claim=claim,
        source=source,
        company_claim=company_claim,
        collector="regression_fixture",
    )


FACTS = [
    # -- old, unrelated, terminated trial (indication A) --------------------
    _raw(f"{OLD_TRIAL} phase is Phase 2", OLD_TRIAL_SOURCE, FactCategory.CLINICAL),
    _raw(f"{OLD_TRIAL} overall status is TERMINATED", OLD_TRIAL_SOURCE, FactCategory.CLINICAL),
    # -- current lead trial (indication B), surrogate/biomarker endpoint ----
    _raw(f"{CURRENT_TRIAL} phase is Phase 3", CURRENT_TRIAL_SOURCE, FactCategory.CLINICAL),
    _raw(f"{CURRENT_TRIAL} overall status is RECRUITING", CURRENT_TRIAL_SOURCE, FactCategory.CLINICAL),
    _raw(
        f"{CURRENT_TRIAL} allocation is Randomized, masking is DOUBLE",
        CURRENT_TRIAL_SOURCE,
        FactCategory.CLINICAL,
    ),
    _raw(f"{CURRENT_TRIAL} enrollment is 300 (ESTIMATED)", CURRENT_TRIAL_SOURCE, FactCategory.CLINICAL),
    _raw(
        f"{CURRENT_TRIAL} primary outcome measure is: change from baseline in biomarker X",
        CURRENT_TRIAL_SOURCE,
        FactCategory.CLINICAL,
    ),
    _raw(
        f"Generic Biotech Holdings describes {CURRENT_TRIAL} as its current lead registrational "
        "programme",
        COMPANY_SOURCE,
        FactCategory.CLINICAL,
        company_claim=True,
    ),
    # -- the filing POINTER only: metadata, not the filing's substantive body
    _raw(
        "Form 8-K filed disclosing correspondence with the regulator regarding the primary "
        "endpoint for the current programme",
        FILING_SOURCE,
        FactCategory.REGULATORY,
        company_claim=True,
    ),
]


def build_collection() -> CollectionResult:
    sources = {f.source.source_id: f.source for f in FACTS}
    return CollectionResult(
        collector="regression_fixture",
        raw_facts=list(FACTS),
        sources=list(sources.values()),
        provenance=Provenance.FIXTURE,
    )


@pytest.fixture
def provider():
    return _FilingBodyProvider({FILING_URL: _ADVERSE_BODY})


@pytest.fixture
def result(repo, provider):
    pipeline = Pipeline(repo, NullSearchProvider(), today=TODAY, research=provider)
    return pipeline.run(TICKER, NAME, [build_collection()], price=2.10)


# --- C: escalation resolves the material unresolved question ---------------
def test_regulatory_agent_raised_a_material_unresolved_question(result):
    questions = [q.question for q in result.bus.unresolved if q.blocking]
    assert any("endpoint" in q.lower() and "adequate" in q.lower() for q in questions)


def test_escalation_used_the_already_collected_filing_not_a_new_search(provider, result):
    assert FILING_URL in provider.fetch_calls
    # The Kill Agent's OWN mandatory queries also legitimately use this same
    # live provider (requirement E), so the provider-wide search_calls count
    # is not zero -- what matters here is that THIS question's own escalation
    # attempt never fell through to search at all, because the already-
    # collected filing answered it directly.
    regulatory_attempts = [
        a for a in result.escalation.attempts if "endpoint" in a.claim.lower()
    ]
    assert regulatory_attempts
    assert regulatory_attempts[0].confirmed
    assert regulatory_attempts[0].searched is False, (
        "a plausible primary source was already collected; escalation must not force a "
        "redundant search for it"
    )


def test_the_escalated_regulatory_fact_is_decision_grade_and_reaches_kill(result):
    escalated = [
        f for f in result.bus.facts if "not sufficient to establish effectiveness" in f.claim
    ]
    assert escalated, "the fetched filing body must have produced a new fact"
    fact = escalated[0]
    assert fact.is_decision_grade

    regulatory_kill = result.verdict.kill_gate.by_category(KillCategory.REGULATORY_KILL)
    assert regulatory_kill.level is KillLevel.K5
    assert regulatory_kill.confirmation is KillConfirmation.CONFIRMED


# --- D: programme resolution -------------------------------------------------
def test_science_evaluated_the_current_programme_not_the_old_one(result):
    science_payload = result.bus.channels[Channel.SCIENCE].payload
    assert science_payload["program_resolved"] == CURRENT_TRIAL
    assert science_payload["endpoint_type"] == "SURROGATE_OR_BIOMARKER"


def test_the_old_terminated_trial_did_not_become_a_company_level_kill(result):
    # The Verdict's own reconstructed kill_gate (from_channel_payload) never
    # carries per-finding detail (KillAssessment.to_row() does not serialize
    # it) -- the flat, per-finding record lives on the KILL channel payload.
    findings = result.bus.channels[Channel.KILL].payload["findings"]
    old_trial_findings = [f for f in findings if OLD_TRIAL in f["detail"]]
    assert old_trial_findings, "the old trial's status must still be recorded as a finding"
    assert all(KillLevel(f["level"]).level <= KillLevel.K1.level for f in old_trial_findings)
    assert all("DIFFERENT PROGRAMME" in f["detail"] for f in old_trial_findings)


# --- G: decision-gate output consistency ------------------------------------
def test_final_action_stays_none_when_evidence_completeness_is_insufficient(result):
    assert result.verdict.research_status is not ResearchStatus.COMPLETE
    assert result.verdict.action is None


def test_no_actionable_label_leaks_anywhere_in_the_rendered_report(result):
    report = render_report(result)
    not_issued_sentence = (
        "No action label is issued. Not AVOID, not WAIT_FOR_EVENT -- an action label "
        "asserts a judgement, and there is not enough research here to support one."
    )
    sanitized = report.replace(not_issued_sentence, "")
    leaked = find_action_labels(sanitized)
    assert leaked == [], f"actionable label(s) leaked into the rendered report: {leaked}"
    assert "FINAL_ACTION: NONE" in report
