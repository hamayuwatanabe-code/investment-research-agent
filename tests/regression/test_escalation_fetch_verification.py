"""Regression suite: primary-source escalation must fetch before confirming.

DEFECT 2: ``AnthropicWebResearchProvider.search()`` intentionally returns
METADATA_ONLY documents (a search hit is a pointer, not a finding), but
``escalation.escalate()`` was checking ``_confirms()`` directly against those
search results. ``_confirms()`` correctly refuses any non-primary-text
document, so a live search hit could never become confirming evidence through
this path -- escalation was structurally incapable of ever confirming
anything live.

The fix threads the full chain through ``escalate()``:

    ResearchQuery -> provider.search() -> SearchHit-shaped candidate URL
        -> provider.fetch(candidate_url) -> FULL_DOCUMENT
        -> _confirms(fetched_body, fact) -> only then promote the Fact

These tests use a wholly synthetic, generic company/claim/URL set -- no real
ticker, company name, drug name, or specific regulatory history is
hard-coded anywhere in this file or in the production code it exercises. The
pattern being modeled (a search hit points at a plausible primary source, the
fetched body either does or does not actually state the claim) is general to
any material regulatory claim, not specific to any one issuer.
"""

from __future__ import annotations

from investment_research.collectors.documents import Document
from investment_research.llm.client import BudgetExceeded
from investment_research.research.escalation import (
    _confirms,
    _plausible_primary_candidate,
    escalate,
)
from investment_research.research.provider import ResearchQuery, ResearchResult
from investment_research.schemas.enums import (
    ContentKind,
    EvidenceClass,
    FetchOutcome,
    ResearchPath,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Fact
from tests.conftest import make_fact

#: A generic, non-issuer-specific material regulatory claim. Deliberately
#: shaped to match escalation.py's own "regulator_position" pattern
#: (regulator/agency ... advised/not sufficient/...) without naming any real
#: company, drug, trial or regulator interaction.
GENERIC_REGULATOR_CONCERN = (
    "The regulator advised that the primary endpoint is not sufficient to support approval"
)

COMPANY = "Generic Biotech Holdings"


class _FakeEscalationProvider:
    """A minimal ResearchProvider double giving full control over search vs. fetch.

    ``documents_by_domain`` simulates a real provider honoring
    ``ResearchQuery.allowed_domains``: a query restricted to one domain only
    surfaces hits registered under that domain, so the per-domain fetch
    gating in ``escalate()`` is exercised the same way it would be live.
    """

    name = "fake_escalation_provider"

    def __init__(
        self,
        *,
        documents_by_domain: dict[str, list[Document]] | None = None,
        fetch_map: dict[str, Document] | None = None,
        fetch_raises_for: dict[str, Exception] | None = None,
        search_disabled: bool = False,
    ):
        self._documents_by_domain = documents_by_domain or {}
        self._fetch_map = fetch_map or {}
        self._fetch_raises_for = fetch_raises_for or {}
        self._search_disabled = search_disabled
        self.search_calls: list[ResearchQuery] = []
        self.fetch_calls: list[str] = []

    def available(self) -> tuple[bool, str]:
        return True, "ready"

    def search(self, query: ResearchQuery) -> ResearchResult:
        self.search_calls.append(query)
        if self._search_disabled:
            return ResearchResult(
                query=query,
                outcome=FetchOutcome.DISABLED,
                path=ResearchPath.ANTHROPIC_WEB,
                executed=False,
                error="BudgetExceeded: LLM token budget already exhausted",
            )
        domain = query.allowed_domains[0] if query.allowed_domains else None
        documents = list(self._documents_by_domain.get(domain, []))
        outcome = FetchOutcome.OK if documents else FetchOutcome.NOT_FOUND
        return ResearchResult(
            query=query, documents=documents, outcome=outcome, path=ResearchPath.ANTHROPIC_WEB
        )

    def fetch(self, url: str, *, reason: str = "") -> Document | None:
        self.fetch_calls.append(url)
        if url in self._fetch_raises_for:
            raise self._fetch_raises_for[url]
        return self._fetch_map.get(url)


def _weak_fact(**overrides) -> Fact:
    defaults = {
        "tier": SourceTier.TIER_3,
        "url": "https://www.generic-news-wire.example/article",
        "content_kind": ContentKind.SEARCH_SUMMARY,
        "verified": VerifiedStatus.SEARCH_EVIDENCE,
    }
    defaults.update(overrides)
    return make_fact(GENERIC_REGULATOR_CONCERN, **defaults)


def _search_hit(url: str, *, tier: SourceTier = SourceTier.TIER_1) -> Document:
    """A METADATA_ONLY search result -- a pointer, never confirming evidence."""
    return Document(
        doc_id=f"hit::{url}",
        url=url,
        title="Regulatory correspondence summary",
        text="",  # a search snippet's text is irrelevant -- METADATA_ONLY bars it regardless
        content_kind=ContentKind.METADATA_ONLY,
        tier=tier,
        doc_type="web",
    )


def _fetched_body(url: str, text: str, *, tier: SourceTier = SourceTier.TIER_1) -> Document:
    """A FULL_DOCUMENT -- what a real fetch of the candidate URL returns."""
    return Document(
        doc_id=f"doc::{url}",
        url=url,
        title="Regulatory correspondence",
        text=text,
        content_kind=ContentKind.FULL_DOCUMENT,
        tier=tier,
        doc_type="regulator",
    )


# --- _confirms() never accepts a search hit, however well it matches --------
def test_confirms_refuses_a_metadata_only_hit_even_with_matching_text():
    """NEVER promote a search snippet or METADATA_ONLY document."""
    fact = _weak_fact()
    hit = _search_hit("https://www.fda.gov/x")
    hit.text = GENERIC_REGULATOR_CONCERN  # perfect textual match, still METADATA_ONLY
    assert _confirms(hit, fact) is False


# --- _plausible_primary_candidate(): tier/domain pre-filter -----------------
def test_plausible_primary_candidate_requires_matching_host_and_primary_tier():
    primary_and_matching = Document(
        doc_id="d1", url="https://www.fda.gov/x", title="t", tier=SourceTier.TIER_1
    )
    assert _plausible_primary_candidate(primary_and_matching, "fda.gov") is True

    wrong_host = Document(
        doc_id="d2", url="https://www.unrelated.example/x", title="t", tier=SourceTier.TIER_1
    )
    assert _plausible_primary_candidate(wrong_host, "fda.gov") is False

    weak_tier = Document(
        doc_id="d3", url="https://www.fda.gov/x", title="t", tier=SourceTier.TIER_5
    )
    assert _plausible_primary_candidate(weak_tier, "fda.gov") is False


# --- Regression 1: fetched body confirms -> only then promoted -------------
def test_escalation_confirms_material_claim_only_via_fetched_primary_body():
    """Generic pattern: search returns a primary-source pointer containing a
    material regulator concern; the fetched body contains the concern; only
    the fetched body permits VERIFIED/CONFIRMED status.
    """
    url = "https://www.fda.gov/regulatory-update"
    hit = _search_hit(url)
    fetched = _fetched_body(url, GENERIC_REGULATOR_CONCERN + " for this application.")
    provider = _FakeEscalationProvider(
        documents_by_domain={"sec.gov": [], "fda.gov": [hit]},
        fetch_map={url: fetched},
    )

    fact = _weak_fact()
    updated, report = escalate([fact], provider, company=COMPANY)
    result = updated[0]

    assert provider.fetch_calls == [url], "must fetch the candidate before it can confirm anything"
    assert result.verified_status is VerifiedStatus.VERIFIED
    assert result.evidence_class is EvidenceClass.INDEPENDENT_EVIDENCE
    assert result.content_kind is ContentKind.FULL_DOCUMENT
    assert result.is_decision_grade is True
    assert report.attempts[0].confirmed is True
    # Recorded from the fetched document, never from the search hit.
    assert report.attempts[0].confirming_url == fetched.url


def test_escalation_never_fetches_an_implausible_candidate():
    """Do not blindly fetch every hit: a Tier 5 blog hit is never fetched."""
    good_url = "https://www.fda.gov/regulatory-update"
    bad_hit = _search_hit("https://blog.example.com/opinion", tier=SourceTier.TIER_5)
    good_hit = _search_hit(good_url)
    fetched = _fetched_body(good_url, GENERIC_REGULATOR_CONCERN + " for this application.")
    provider = _FakeEscalationProvider(
        documents_by_domain={"sec.gov": [], "fda.gov": [bad_hit, good_hit]},
        fetch_map={good_url: fetched},
    )

    updated, _ = escalate([_weak_fact()], provider, company=COMPANY)

    assert provider.fetch_calls == [good_url]
    assert updated[0].verified_status is VerifiedStatus.VERIFIED


# --- Regression 2: fetch fails or the body does not confirm -> unverified --
def test_escalation_leaves_claim_unverified_when_fetch_fails():
    """Same search hit; the fetch fails (returns None) -> stays unverified."""
    url = "https://www.fda.gov/regulatory-update"
    hit = _search_hit(url)
    provider = _FakeEscalationProvider(
        documents_by_domain={"sec.gov": [], "fda.gov": [hit]},
        fetch_map={},  # fetch() returns None for every URL
    )

    updated, report = escalate([_weak_fact()], provider, company=COMPANY)
    result = updated[0]

    assert provider.fetch_calls == [url]
    assert result.verified_status is VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM
    assert result.is_decision_grade is False
    assert not report.attempts[0].confirmed


def test_escalation_leaves_claim_unverified_when_fetched_body_does_not_confirm():
    """Same search hit; the fetched body exists but does not state the claim."""
    url = "https://www.fda.gov/regulatory-update"
    hit = _search_hit(url)
    unrelated = _fetched_body(
        url, "This document discusses routine administrative filing deadlines."
    )
    provider = _FakeEscalationProvider(
        documents_by_domain={"sec.gov": [], "fda.gov": [hit]},
        fetch_map={url: unrelated},
    )

    updated, report = escalate([_weak_fact()], provider, company=COMPANY)
    result = updated[0]

    assert result.verified_status is VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM
    assert result.evidence_class != EvidenceClass.INDEPENDENT_EVIDENCE
    assert result.is_decision_grade is False
    assert not report.attempts[0].confirmed


def test_escalation_fetch_raising_never_confirms_and_is_not_swallowed_as_success():
    """A fetch that raises (e.g. a transport error) must never be read as confirmation."""
    url = "https://www.fda.gov/regulatory-update"
    hit = _search_hit(url)
    provider = _FakeEscalationProvider(
        documents_by_domain={"sec.gov": [], "fda.gov": [hit]},
        fetch_raises_for={url: RuntimeError("connection reset")},
    )

    updated, _ = escalate([_weak_fact()], provider, company=COMPANY)
    assert updated[0].verified_status is VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM
    assert updated[0].is_decision_grade is False


# --- BudgetExceeded: incomplete/unverified, never a false negative ---------
def test_escalation_budget_exhaustion_during_search_is_unverified_not_a_false_negative():
    provider = _FakeEscalationProvider(search_disabled=True)

    updated, report = escalate([_weak_fact()], provider, company=COMPANY)
    result = updated[0]

    assert provider.fetch_calls == [], "a search that never executed must never be fetched from"
    assert result.verified_status is VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM
    assert result.is_decision_grade is False
    assert "budget" in result.notes.lower()


def test_escalation_budget_exhaustion_during_fetch_is_unverified_not_a_false_negative():
    url = "https://www.fda.gov/regulatory-update"
    hit = _search_hit(url)
    provider = _FakeEscalationProvider(
        documents_by_domain={"sec.gov": [], "fda.gov": [hit]},
        fetch_raises_for={url: BudgetExceeded("LLM token budget already exhausted")},
    )

    updated, _ = escalate([_weak_fact()], provider, company=COMPANY)
    result = updated[0]

    assert result.verified_status is VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM
    assert result.is_decision_grade is False
