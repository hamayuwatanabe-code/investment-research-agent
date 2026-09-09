"""Structured-metadata-first candidate ranking, capped initial fetches, exhibit
following, and per-fetch audit diagnostics (requirements D and E).

The defect: with dozens of collected SEC filings, the old ranking used only
lexical title overlap ("8-K CURRENT REPORT" tells nothing apart), and the
per-question fetch cap (``max_candidates_per_question=3``) meant three
fetches were spent whenever three Tier-1 sources happened to exist, blind to
which one was actually likely to answer the question. The report also only
ever said "fetches attempted: N, fetches failed: M" with no way to tell
which URL, for which question, with what outcome.
"""

from __future__ import annotations

from investment_research.collectors.documents import Document
from investment_research.research.escalation import (
    EscalationReport,
    FetchAttempt,
    _attempt_fetch_for_question,
    _rank_candidates_for_question,
    _score_candidates,
    escalate_unresolved_questions,
)
from investment_research.research.provider import (
    ResearchResult,
)
from investment_research.schemas.enums import (
    ContentKind,
    DocumentAuthority,
    FactCategory,
    FetchOutcome,
    Provenance,
    ResearchPath,
    SourceTier,
)
from investment_research.schemas.fact import Source, UnresolvedQuestion, make_source_id
from tests.conftest import make_fact


def _source(
    url: str,
    title: str,
    *,
    accession: str = "UNKNOWN",
    filing_date: str = "UNKNOWN",
    tier: SourceTier = SourceTier.TIER_1,
) -> Source:
    return Source(
        source_id=make_source_id(url, title),
        url=url,
        title=title,
        tier=tier,
        accession=accession,
        filing_date=filing_date,
    )


_REGULATORY_QUESTION = UnresolvedQuestion(
    question="Does the regulator consider the primary endpoint appropriate to establish "
    "effectiveness for the intended indication?",
    why_it_matters="A rejected endpoint invalidates the registrational path.",
    blocking=True,
    category=FactCategory.REGULATORY,
)


# --- D: structured-metadata-first ranking ------------------------------------
def test_structured_metadata_ranks_category_linked_and_dated_source_first():
    strong = _source(
        "https://www.sec.gov/Archives/x/8-K-regulatory.htm",
        "Form 8-K",
        accession="0001234567-26-000123",
        filing_date="2026-05-08",
    )
    weak = _source("https://www.sec.gov/Archives/x/10-K-annual.htm", "Form 10-K (Annual Report)")
    linked_fact = make_fact(
        "The agency raised concerns about the endpoint",
        category=FactCategory.REGULATORY,
        url=strong.url,
    )

    ranked = _rank_candidates_for_question(
        _REGULATORY_QUESTION, [weak, strong], [linked_fact]
    )
    assert ranked[0].url == strong.url, (
        "a source backing a same-category fact, with a known accession and filing date, must "
        "outrank an unlinked, undated source even without stronger title overlap"
    )


def test_capital_structure_source_hosted_off_routed_domain_scores_lower():
    on_domain = _source("https://www.sec.gov/Archives/x/8-K.htm", "Form 8-K")
    off_domain = _source("https://www.globenewswire.com/news/x", "Press Release")
    scored = {
        s.url: score
        for s, score in _score_candidates(
            UnresolvedQuestion(
                question="What is the fully diluted share count?",
                why_it_matters="Dilution.",
                blocking=True,
                category=FactCategory.CAPITAL_STRUCTURE,
            ),
            [on_domain, off_domain],
            [],
        )
    }
    assert scored[on_domain.url] > scored[off_domain.url]


# --- D: ambiguous ranking -> ONE tie-break search, capped initial fetches ---
class _RecordingProvider:
    """Records every search/fetch call. Fetches for every URL EXCEPT
    ``answering_url`` return a document that does not answer the question, so
    the "the answer is at the second position tried" path is exercised
    without accidentally short-circuiting on the very first fetch."""

    name = "fake"
    path = ResearchPath.ANTHROPIC_WEB

    def __init__(self, answering_url: str | None = None, answer_text: str = "") -> None:
        self.answering_url = answering_url
        self.answer_text = answer_text or (
            "The agency stated it does not consider the primary endpoint appropriate to "
            "establish effectiveness for the intended indication."
        )
        self.search_calls: list[str] = []
        self.fetch_calls: list[str] = []

    def available(self):
        return True, "ready"

    def search(self, query, *, agent_id: str = "research"):
        self.search_calls.append(query.query)
        return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=self.path)

    def fetch(self, url, *, reason="", agent_id: str = "research", known=None):
        self.fetch_calls.append(url)
        text = (
            self.answer_text
            if url == self.answering_url
            else "Unrelated boilerplate text about the company's operations."
        )
        return Document(
            doc_id=url,
            url=url,
            title="Form 8-K",
            publisher="sec.gov",
            published_date="2026-08-01",
            doc_type="filing",
            text=text,
            content_kind=ContentKind.FULL_DOCUMENT,
            provenance=Provenance.LIVE,
            tier=SourceTier.TIER_1,
            authority=DocumentAuthority.STATUTORY_FILING,
        )


def test_ambiguous_candidate_set_issues_one_tie_break_search_and_caps_initial_fetches():
    # Many equally-unstructured Tier-1 sources: no accession, no filing date,
    # no linked fact -- genuinely ambiguous, exactly the case a live run
    # spent three blind fetches on. The second one happens to answer, so the
    # run must stop there rather than continuing to expand or search further.
    sources = [
        _source(f"https://www.sec.gov/Archives/x/filing-{i}.htm", "8-K CURRENT REPORT")
        for i in range(5)
    ]
    answering_url = sources[1].url
    provider = _RecordingProvider(answering_url=answering_url)
    facts, _ = escalate_unresolved_questions(
        [_REGULATORY_QUESTION],
        sources,
        provider,
        company="Generic Biotech Holdings",
        ticker="TESTCO",
        run_id="r1",
    )
    assert facts, "the answering candidate must have produced a fact"
    assert len(provider.search_calls) == 1, (
        f"expected exactly ONE tie-break discovery search for an ambiguous candidate set, got "
        f"{provider.search_calls}"
    )
    assert len(provider.fetch_calls) <= 2, (
        f"must not spend more than 2 initial fetches simply because more Tier-1 sources exist, "
        f"got {provider.fetch_calls}"
    )


# --- D: exhibit following within the same attempt ---------------------------
class _ExhibitProvider:
    name = "fake"
    path = ResearchPath.ANTHROPIC_WEB

    def __init__(self, filing_url: str, exhibit_url: str, exhibit_answer: str) -> None:
        self.filing_url = filing_url
        self.exhibit_url = exhibit_url
        self.exhibit_answer = exhibit_answer
        self.fetch_calls: list[str] = []

    def available(self):
        return True, "ready"

    def search(self, query, *, agent_id: str = "research"):
        return ResearchResult(query=query, outcome=FetchOutcome.NOT_FOUND, path=self.path)

    def fetch(self, url, *, reason="", agent_id: str = "research", known=None):
        self.fetch_calls.append(url)
        if url == self.filing_url:
            return Document(
                doc_id="filing",
                url=url,
                title="Form 8-K",
                publisher="sec.gov",
                doc_type="filing",
                accession="0001234567-26-000999",
                text="The company filed this report. See Exhibit 99.1 for further detail.",
                content_kind=ContentKind.FULL_DOCUMENT,
                provenance=Provenance.LIVE,
                tier=SourceTier.TIER_1,
                authority=DocumentAuthority.STATUTORY_FILING,
            )
        if url == self.exhibit_url:
            return Document(
                doc_id="exhibit",
                url=url,
                title="Exhibit 99.1",
                publisher="sec.gov",
                doc_type="filing",
                text=self.exhibit_answer,
                content_kind=ContentKind.FULL_DOCUMENT,
                provenance=Provenance.LIVE,
                tier=SourceTier.TIER_1,
                authority=DocumentAuthority.STATUTORY_FILING,
            )
        return None


def test_exhibit_referenced_by_a_filing_is_followed_in_the_same_attempt():
    filing_url = "https://www.sec.gov/Archives/x/8-K.htm"
    exhibit_url = "https://www.sec.gov/Archives/x/ex99-1.htm"
    filing = _source(filing_url, "Form 8-K", accession="0001234567-26-000999")
    exhibit = _source(exhibit_url, "Exhibit 99.1", accession="0001234567-26-000999")
    linked_fact = make_fact(
        "regulatory update", category=FactCategory.REGULATORY, url=filing_url
    )

    provider = _ExhibitProvider(
        filing_url,
        exhibit_url,
        "In the attached exhibit, the agency stated it does not consider the primary endpoint "
        "appropriate to establish effectiveness for the intended indication.",
    )
    facts, report = escalate_unresolved_questions(
        [_REGULATORY_QUESTION],
        [filing, exhibit],
        provider,
        company="Generic Biotech Holdings",
        ticker="TESTCO",
        run_id="r1",
        facts=[linked_fact],
    )
    assert facts, "the exhibit's answer must have produced a fact"
    assert facts[0].source_url == exhibit_url
    assert exhibit_url in provider.fetch_calls
    exhibit_records = [entry for entry in report.fetch_log if entry.url == exhibit_url]
    assert exhibit_records
    assert exhibit_records[0].candidate_rank == -2
    assert exhibit_records[0].body_answered is True


# --- E: per-fetch audit diagnostics ------------------------------------------
def test_fetch_log_records_full_audit_fields_for_every_attempt():
    provider = _RecordingProvider()
    sources = [_source("https://www.sec.gov/Archives/x/only-candidate.htm", "Form 8-K")]
    _, report = escalate_unresolved_questions(
        [_REGULATORY_QUESTION],
        sources,
        provider,
        company="Generic Biotech Holdings",
        ticker="TESTCO",
        run_id="r1",
    )
    assert report.fetch_log, "every fetch attempt must be recorded"
    entry = report.fetch_log[0]
    assert isinstance(entry, FetchAttempt)
    assert entry.url
    assert entry.authority
    assert entry.outcome in ("answered", "no_answer", "fetch_failed", "confirmed")
    assert entry.body_obtained is True
    assert entry.body_answered is False
    assert entry.subject_id


# --- pre-send budget rejection vs. a genuinely sent-then-failed fetch -------
class _NotSentFetchProvider:
    """A fetch that never reached the network -- a preflight budget/
    availability rejection, exactly what AnthropicWebResearchProvider.fetch()
    now signals via its own last_fetch_sent/last_fetch_error attributes."""

    name = "fake"
    path = ResearchPath.ANTHROPIC_WEB
    last_fetch_sent = False
    last_fetch_error = "BudgetExceeded: stage 'escalation' budget exhausted"

    def available(self):
        return True, "ready"

    def fetch(self, url, *, reason="", agent_id="research", known=None):
        return None


class _SentThenFailedFetchProvider:
    """A fetch that WAS sent and then genuinely failed (an actual API/SDK
    exception, or a sent request whose response carried no usable text)."""

    name = "fake"
    path = ResearchPath.ANTHROPIC_WEB
    last_fetch_sent = True
    last_fetch_error = "APIStatusError: 500 internal server error"

    def available(self):
        return True, "ready"

    def fetch(self, url, *, reason="", agent_id="research", known=None):
        return None


class _LegacyFetchProvider:
    """A provider that exposes no last_fetch_sent/last_fetch_error signal at
    all (e.g. a corpus/null provider) -- must read exactly as before this
    distinction existed: sent=True, outcome="fetch_failed", never silently
    misclassified as a budget rejection just because the signal is absent."""

    name = "fake"
    path = ResearchPath.ANTHROPIC_WEB

    def available(self):
        return True, "ready"

    def fetch(self, url, *, reason="", agent_id="research", known=None):
        return None


def test_fetch_audit_distinguishes_pre_send_budget_rejection_from_a_sent_then_failed_fetch():
    """送信前の予算拒否を「取得APIが失敗」と一括表示しない: a fetch the
    provider never even attempted to send must be reported distinctly
    (outcome="not_sent", sent=False, fetches_not_sent incremented) from a
    fetch that WAS sent and then genuinely failed (outcome="fetch_failed",
    sent=True) -- both currently land in fetches_failed for backward
    compatibility, but only the genuinely-sent case is "the fetch API
    failed"."""
    candidate = _source("https://www.sec.gov/Archives/x/8-K.htm", "Form 8-K")

    not_sent_report = EscalationReport()
    _attempt_fetch_for_question(
        _NotSentFetchProvider(), candidate, 1, _REGULATORY_QUESTION, "q1", not_sent_report
    )
    assert not_sent_report.fetches_attempted == 1
    assert not_sent_report.fetches_failed == 1
    assert not_sent_report.fetches_not_sent == 1
    entry = not_sent_report.fetch_log[0]
    assert entry.outcome == "not_sent"
    assert entry.sent is False
    assert "BudgetExceeded" in entry.failure_reason

    sent_report = EscalationReport()
    _attempt_fetch_for_question(
        _SentThenFailedFetchProvider(), candidate, 1, _REGULATORY_QUESTION, "q1", sent_report
    )
    assert sent_report.fetches_attempted == 1
    assert sent_report.fetches_failed == 1
    assert sent_report.fetches_not_sent == 0, "a genuinely sent-then-failed fetch must never count as not-sent"
    entry2 = sent_report.fetch_log[0]
    assert entry2.outcome == "fetch_failed"
    assert entry2.sent is True
    assert "APIStatusError" in entry2.failure_reason


def test_fetch_audit_defaults_to_sent_true_when_the_provider_exposes_no_signal():
    candidate = _source("https://www.sec.gov/Archives/x/8-K.htm", "Form 8-K")
    report = EscalationReport()
    _attempt_fetch_for_question(_LegacyFetchProvider(), candidate, 1, _REGULATORY_QUESTION, "q1", report)
    entry = report.fetch_log[0]
    assert entry.sent is True
    assert entry.outcome == "fetch_failed"
    assert report.fetches_not_sent == 0
