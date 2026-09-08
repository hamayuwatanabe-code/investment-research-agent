"""Primary-source escalation (requirement P4).

The rule: a **material** claim carried only by Tier 3-5 reporting must not reach
the Final Judge as though it were established. The system attempts to confirm it
in a primary source, and when it cannot, marks it
``UNVERIFIED_MATERIAL_CLAIM`` -- a status barred from decision-grade use.

Why a distinct status rather than NOT_VERIFIED: the two mean different things.
NOT_VERIFIED is "we did not confirm this". UNVERIFIED_MATERIAL_CLAIM is "this
matters, we tried specifically to confirm it in a filing or a regulator
document, and we could not". The second is a much stronger warning, and it is
also a to-do list for a human.

Escalation runs news -> filing -> regulator, in that order, because the cheapest
confirmation is usually a company filing that repeats the claim under liability.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..collectors.documents import Document, split_sentences
from ..schemas.enums import (
    UNKNOWN,
    ContentKind,
    DocumentAuthority,
    EvidenceClass,
    FactCategory,
    FetchOutcome,
    Materiality,
    ResearchDomain,
    SourceTier,
    VerifiedStatus,
)
from ..schemas.fact import Fact, Source, UnresolvedQuestion, make_fact_id, make_source_id
from .provider import ResearchProvider, ResearchQuery

#: What evidence_class/company_claim/independent_confirmation a fetched
#: document's DocumentAuthority may support (requirement B3). The central
#: distinction: a statutory issuer filing is authoritative evidence that the
#: issuer made the filed disclosure -- decision-grade for THAT -- but it is
#: NOT independent confirmation of an underlying regulator communication a
#: genuine regulator-issued document would be. A company press release stays
#: COMPANY_CLAIM and never becomes decision-grade merely because its body
#: was fetched.
_EVIDENCE_FOR_AUTHORITY: dict[DocumentAuthority, tuple[EvidenceClass, bool, bool]] = {
    DocumentAuthority.REGULATOR: (EvidenceClass.INDEPENDENT_EVIDENCE, False, True),
    DocumentAuthority.STATUTORY_FILING: (EvidenceClass.VERIFIED_FACT, True, False),
    DocumentAuthority.REGISTRY: (EvidenceClass.VERIFIED_FACT, True, False),
    DocumentAuthority.COMPANY_IR: (EvidenceClass.COMPANY_CLAIM, True, False),
    DocumentAuthority.INDEPENDENT: (EvidenceClass.INDEPENDENT_EVIDENCE, False, True),
    DocumentAuthority.UNKNOWN: (EvidenceClass.UNVERIFIED_CLAIM, False, False),
}


def _evidence_for_authority(authority: DocumentAuthority) -> tuple[EvidenceClass, bool, bool]:
    """``(evidence_class, company_claim, independent_confirmation)`` for a
    fetched document of this authority. See ``_EVIDENCE_FOR_AUTHORITY``."""
    return _EVIDENCE_FOR_AUTHORITY[authority]

log = logging.getLogger(__name__)

#: The full generic domain set -- used only as the fallback for a category
#: this system has no more specific routing for (requirement C).
PRIMARY_DOMAINS: tuple[str, ...] = (
    "sec.gov",
    "fda.gov",
    "clinicaltrials.gov",
    "ema.europa.eu",
    "nih.gov",
)

#: Question/claim category -> the primary-source domains actually relevant to
#: it, in preference order (requirement C). Escalating every question against
#: every domain regardless of topic is expensive AND wrong: a live run
#: escalated a fully-diluted-share-count question to fda.gov, which cannot
#: possibly answer it and only spends budget that a regulatory or scientific
#: question needed. A category absent from this table falls back to
#: ``PRIMARY_DOMAINS`` (the full generic set) rather than searching nothing.
_DOMAIN_ROUTING: dict[FactCategory, tuple[str, ...]] = {
    # REGULATORY / endpoint / FDA interaction.
    FactCategory.REGULATORY: ("sec.gov", "fda.gov", "clinicaltrials.gov"),
    FactCategory.CLINICAL: ("clinicaltrials.gov", "fda.gov", "nih.gov"),
    # CAPITAL_STRUCTURE / dilution / warrants / shares / financing -- sec.gov
    # only. fda.gov and clinicaltrials.gov can never answer a share-count or
    # financing question, and must never be queried for one.
    FactCategory.CAPITAL_STRUCTURE: ("sec.gov",),
    FactCategory.LIQUIDITY: ("sec.gov",),
    FactCategory.FINANCIAL: ("sec.gov",),
    # SCIENCE / trial design -- registries and government scientific sources.
    FactCategory.SCIENCE: ("clinicaltrials.gov", "pubmed.ncbi.nlm.nih.gov", "nih.gov"),
    FactCategory.TECHNOLOGY: ("clinicaltrials.gov", "pubmed.ncbi.nlm.nih.gov", "nih.gov"),
    # COMPETITION -- regulator labels, trial registries, statutory filings,
    # peer-reviewed sources.
    FactCategory.COMPETITION: (
        "clinicaltrials.gov",
        "fda.gov",
        "sec.gov",
        "pubmed.ncbi.nlm.nih.gov",
    ),
    FactCategory.COMMERCIAL: ("sec.gov", "fda.gov", "clinicaltrials.gov"),
    # CATALYST -- statutory filings, company IR, regulator calendars/registries.
    FactCategory.CATALYST: ("sec.gov", "fda.gov", "clinicaltrials.gov"),
    # GOVERNANCE / accounting -- sec.gov, exchange/regulator sources.
    FactCategory.GOVERNANCE: ("sec.gov",),
    FactCategory.ACCOUNTING: ("sec.gov",),
    FactCategory.INSIDER: ("sec.gov",),
    FactCategory.LISTING: ("sec.gov",),
    FactCategory.CONTRACTS: ("sec.gov",),
    FactCategory.LEGAL: ("sec.gov",),
    FactCategory.MARKET_SIZE: ("sec.gov",),
    FactCategory.MICROSTRUCTURE: ("sec.gov",),
    FactCategory.MANAGEMENT: ("sec.gov",),
}


def domains_for_category(category: FactCategory) -> tuple[str, ...]:
    """The primary-source domains relevant to ``category`` (requirement C).

    Never searches an obviously irrelevant domain for a category with a more
    specific routing (e.g. fda.gov for a capital-structure question); a
    category with no specific routing falls back to the full generic set
    rather than being left unsearched.
    """
    return _DOMAIN_ROUTING.get(category, PRIMARY_DOMAINS)

#: A claim is material enough to require escalation if it says one of these
#: things. Deliberately narrow: escalating everything would be noise.
_MATERIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "regulator_position",
        re.compile(
            r"(?i)\b(?:fda|ema|pmda|regulator|agency)\b.{0,120}"
            r"(?:not sufficient|does not|no longer|advised|required|rejected)"
        ),
    ),
    ("endpoint", re.compile(r"(?i)\bendpoint\b.{0,80}(?:not|insufficient|inadequate|cannot)")),
    ("pivotal_status", re.compile(r"(?i)\bno longer\b.{0,40}\b(?:pivotal|registrational)\b")),
    ("going_concern", re.compile(r"(?i)\b(?:going concern|substantial doubt)\b")),
    ("clinical_hold", re.compile(r"(?i)\bclinical hold\b")),
    ("trial_failure", re.compile(r"(?i)\b(?:failed|did not meet|missed)\b.{0,40}\bendpoint\b")),
    (
        "investigation",
        re.compile(r"(?i)\b(?:sec investigation|subpoena|wells notice|restatement)\b"),
    ),
    ("delisting", re.compile(r"(?i)\b(?:delisting|minimum bid price|non-?compliance)\b")),
)


@dataclass
class EscalationAttempt:
    fact_id: str
    claim: str
    reason: str
    queries: list[str] = field(default_factory=list)
    confirmed: bool = False
    confirming_url: str = ""
    confirming_tier: str = UNKNOWN
    searched: bool = False
    note: str = ""


@dataclass
class FetchAttempt:
    """One fetch, fully auditable (requirement E).

    ``fetches_attempted: 12, fetches_failed: 9`` on its own does not say
    WHICH twelve URLs, for WHICH question, or WHY nine of them failed. One of
    these is recorded for every fetch this module issues, whether it came
    from an already-collected source, a tie-breaking discovery search, or an
    exhibit followed from a filing already fetched in the same attempt.
    """

    #: The fact_id being escalated, or the unresolved question's own text
    #: (truncated) when there is no fact_id -- this module escalates both.
    subject_id: str
    url: str
    #: DocumentAuthority of the candidate BEFORE fetching (from the known
    #: Source, when known) -- "UNKNOWN" when nothing was known about it yet.
    authority: str
    #: 1-based position of this URL among the ranked candidates it was drawn
    #: from; -1 when it came from a tie-breaking discovery search instead of
    #: the already-ranked candidate list; -2 when it was an exhibit followed
    #: from another fetch in the same attempt (filing -> exhibit chain).
    candidate_rank: int
    outcome: str  # "answered" | "confirmed" | "no_answer" | "fetch_failed"
    failure_reason: str = ""
    tokens_used: int = 0
    body_obtained: bool = False
    body_answered: bool = False
    supporting_sentence: str = ""


@dataclass
class EscalationReport:
    attempts: list[EscalationAttempt] = field(default_factory=list)
    #: Diagnostics (requirement G): how many candidate URLs were actually
    #: fetched, and how many of those fetches came back empty (failed,
    #: raised, or budget-cut) versus produced a body.
    fetches_attempted: int = 0
    fetches_failed: int = 0
    #: Per-fetch audit trail (requirement E) -- the appendix-ready detail
    #: behind the two counters above.
    fetch_log: list[FetchAttempt] = field(default_factory=list)

    @property
    def confirmed(self) -> list[EscalationAttempt]:
        return [a for a in self.attempts if a.confirmed]

    @property
    def unconfirmed(self) -> list[EscalationAttempt]:
        return [a for a in self.attempts if a.searched and not a.confirmed]

    @property
    def not_attempted(self) -> list[EscalationAttempt]:
        return [a for a in self.attempts if not a.searched]


def needs_escalation(fact: Fact) -> tuple[bool, str]:
    """Whether this fact is a material claim resting on weak sourcing."""
    if fact.source_tier in (SourceTier.TIER_1, SourceTier.TIER_2) and fact.independent_confirmation:
        return False, ""
    if fact.source_tier.rank <= 2 and not fact.company_claim:
        # A regulator's or exchange's own document already is the primary source.
        return False, ""
    for name, pattern in _MATERIAL_PATTERNS:
        if pattern.search(fact.claim):
            return True, name
    if fact.materiality == Materiality.CRITICAL:
        return True, "critical_materiality"
    return False, ""


#: Document types that carry statutory liability or come from the regulator
#: itself. A press release does not qualify, however primary the wire that
#: distributed it.
_PRIMARY_DOC_TYPES = frozenset({"filing", "registry", "regulator", "docket"})


def _distinctive_terms(text: str) -> set[str]:
    return {term for term in re.split(r"[^a-z0-9]+", text.lower()) if len(term) > 5}


def _is_admissible_primary_document(document: Document) -> bool:
    """Whether a fetched document is even eligible to settle a claim/question.

    A company press release is not a primary source for what a regulator
    said -- it is the company's account of it, which is the exact conflation
    this system exists to stop. Company-controlled material qualifies only
    when it is a statutory filing (or an exhibit/attachment to one), which
    carries liability that a release does not. And only a FETCHED body
    counts: a search engine's summary about a filing is not the filing.
    """
    if not document.tier.is_primary:
        return False
    if document.is_company_ir and document.doc_type not in _PRIMARY_DOC_TYPES:
        return False
    return document.content_kind.is_primary_text


def _confirms(document: Document, fact: Fact) -> bool:
    """Whether a candidate primary document actually supports the claim.

    Two independent bars, both required:

    1. **The document must be able to confirm this** -- see
       ``_is_admissible_primary_document``.
    2. **It must actually say the same thing.** Overlapping distinctive terms,
       not merely being about the same company: a 10-K that mentions the FDA
       does not confirm a specific statement about what the FDA said.
    """
    if not _is_admissible_primary_document(document):
        return False
    haystack = f"{document.title} {document.text}".lower()
    if not haystack.strip():
        return False
    terms = _distinctive_terms(fact.claim)
    if not terms:
        return False
    overlap = sum(1 for term in terms if term in haystack)
    return overlap >= max(3, len(terms) // 4)


def _extract_answering_sentence(document: Document, question: UnresolvedQuestion) -> str | None:
    """The sentence in a fetched body that actually answers this question.

    Positive, negative and contradictory answers are treated symmetrically:
    this only checks whether the body actually *addresses* the question
    (shares enough of its distinctive vocabulary), never which direction the
    answer points. A body that merely mentions the general topic without
    engaging the specific question does not count -- and only verbatim text
    from the fetched body is ever returned, never a paraphrase.
    """
    if not _is_admissible_primary_document(document):
        return None
    terms = _distinctive_terms(f"{question.question} {question.why_it_matters}")
    if not terms:
        return None
    threshold = max(3, len(terms) // 3)
    best_sentence: str | None = None
    best_overlap = 0
    for sentence in split_sentences(document.text):
        overlap = sum(1 for term in terms if term in sentence.lower())
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    if best_sentence is not None and best_overlap >= threshold:
        return best_sentence
    return None


#: How much stronger a top score must be than the runner-up to count as
#: unambiguous (requirement D). Below this margin, a tie-breaking discovery
#: search is issued before any fetch is spent guessing among near-equal
#: candidates.
_UNAMBIGUOUS_SCORE_MARGIN = 2.0

#: Recognized statutory filing types, checked against a source's own title --
#: structured metadata the collector already recorded, never a guess.
_FILING_TYPE_HINTS: tuple[str, ...] = ("8-K", "10-K", "10-Q", "S-1", "S-3", "424B", "DEF 14A", "6-K")

_EXHIBIT_REFERENCE_RE = re.compile(r"(?i)\bexhibit\s+\d+\.\d+\b")


def _recency_sort_key(source: Source) -> str:
    date = source.filing_date if source.filing_date != UNKNOWN else source.event_date
    return date if date != UNKNOWN else "0000-00-00"


def _score_candidates(
    question: UnresolvedQuestion, sources: Sequence[Source], facts: Sequence[Fact]
) -> list[tuple[Source, float]]:
    """Score already-collected primary sources by relevance to ``question``,
    using STRUCTURED metadata first (requirement D) -- never lexical title
    overlap alone. A live run had dozens of collected SEC filings for one
    material question; "8-K CURRENT REPORT" title overlap could not tell them
    apart, and three fetches were spent essentially at random.

    Signals, strongest first, all drawn from fields the collector already
    populated (never inferred or guessed):

    1. The source backs at least one collected fact in the SAME category as
       the question -- the collector already filed this source under the
       same topic.
    2. The source's host is one of the domains this question's category is
       actually routed to (requirement C) -- a source outside the routed
       domain set is far less likely to be the right one.
    3. A known accession number -- marks a genuinely identified statutory
       filing, not a placeholder pointer.
    4. A known filing/event date at all.
    5. Lexical title overlap with the question -- the weakest signal, used
       only to break remaining ties, never as the primary ranking basis.

    Ties within a score band keep their original relative order (a stable
    sort), then break by recency (a more recent filing is more likely to
    speak to a currently-unresolved question than an older one).
    """
    same_category_urls = {
        f.source_url for f in facts if f.category == question.category and f.source_url
    }
    question_terms = _distinctive_terms(f"{question.question} {question.why_it_matters}")
    routed_domains = set(domains_for_category(question.category))

    def _score(source: Source) -> float:
        score = 0.0
        if source.url in same_category_urls:
            score += 4.0
        try:
            host = (urlparse(source.url).hostname or "").lower()
        except ValueError:
            host = ""
        if any(host == domain or host.endswith("." + domain) for domain in routed_domains):
            score += 2.0
        if source.accession and source.accession != UNKNOWN:
            score += 1.0
        if source.filing_date != UNKNOWN or source.event_date != UNKNOWN:
            score += 0.5
        if question_terms and _distinctive_terms(source.title) & question_terms:
            score += 0.25
        return score

    scored = [(source, _score(source)) for source in sources]
    scored.sort(key=lambda pair: (pair[1], _recency_sort_key(pair[0])), reverse=True)
    return scored


def _rank_candidates_for_question(
    question: UnresolvedQuestion, sources: Sequence[Source], facts: Sequence[Fact]
) -> list[Source]:
    """Order already-collected primary sources by relevance to ``question``
    (requirement D). See ``_score_candidates`` for the ranking signals."""
    return [source for source, _score in _score_candidates(question, sources, facts)]


def _ranking_is_ambiguous(scored: Sequence[tuple[Source, float]]) -> bool:
    """Whether the top-ranked candidate(s) do not clearly stand out.

    Requirement D: "if the candidate set is still broad/ambiguous, issue ONE
    narrowly-scoped discovery search to identify the most likely primary
    source before spending multiple fetches." Ambiguous means: there is no
    candidate at all (nothing to rank), the best score is itself
    uninformative (0 -- no structural signal matched anything), or the top
    two candidates are within ``_UNAMBIGUOUS_SCORE_MARGIN`` of each other so
    picking one over the other would be a guess rather than a ranking.
    """
    if not scored:
        return True
    top_score = scored[0][1]
    if top_score <= 0:
        return True
    if len(scored) > 1:
        runner_up = scored[1][1]
        if (top_score - runner_up) < _UNAMBIGUOUS_SCORE_MARGIN:
            return True
    return False


def _same_filing_exhibits(document: Document, candidates: Sequence[Source]) -> list[Source]:
    """Already-collected sources that are exhibits/attachments of the SAME
    statutory filing as ``document`` (requirement D: filing -> exhibit
    traceability), matched by accession number -- structured metadata, never
    a URL guess.
    """
    if document.accession == UNKNOWN or not document.accession:
        return []
    return [
        source
        for source in candidates
        if source.accession == document.accession and source.url != document.url
    ]


def _plausible_primary_candidate(document: Document, domain: str) -> bool:
    """Whether a search hit is worth spending a fetch on.

    A fetch is a real cost (LLM/tool budget), so this is a cheap pre-filter on
    the SearchHit-shaped candidate alone -- it never confirms anything by
    itself. Only ``_confirms()``, run against the FETCHED body, does that. Two
    checks, both required: the hit's own tier classification (derived from its
    URL, same as any other search result) must already look primary, and its
    host must actually be the domain this query was restricted to -- a search
    tool's domain filter is a request, not a guarantee.
    """
    if not document.tier.is_primary:
        return False
    try:
        host = urlparse(document.url).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    domain = domain.lower()
    return host == domain or host.endswith("." + domain)


def escalate(
    facts: Sequence[Fact],
    provider: ResearchProvider,
    *,
    company: str,
    max_facts: int = 12,
    max_queries_per_fact: int = 2,
) -> tuple[list[Fact], EscalationReport]:
    """Try to confirm weakly-sourced material claims in primary sources.

    Returns the facts with statuses updated, plus a report of what was tried.
    """
    from dataclasses import replace

    report = EscalationReport()
    usable, reason = provider.available()
    updated: list[Fact] = []
    escalated = 0

    for fact in facts:
        required, why = needs_escalation(fact)
        if not required:
            updated.append(fact)
            continue

        attempt = EscalationAttempt(fact_id=fact.fact_id, claim=fact.claim[:300], reason=why)
        if not usable or escalated >= max_facts:
            attempt.note = (
                reason if not usable else f"escalation budget of {max_facts} facts exhausted"
            )
            report.attempts.append(attempt)
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM,
                    notes=(fact.notes + "; " if fact.notes else "")
                    + f"material claim not escalated to a primary source ({attempt.note})",
                )
            )
            continue

        escalated += 1
        attempt.searched = True
        confirming: Document | None = None
        budget_cut_short = False
        # Requirement C: routed by this fact's own category, not a fixed
        # "search everything" domain list.
        for domain in domains_for_category(fact.category)[:max_queries_per_fact]:
            query = ResearchQuery(
                query=f"{company} {_key_terms(fact.claim)}",
                domain=ResearchDomain.REGULATORY,
                stance="verify",
                allowed_domains=(domain,),
                rationale=f"primary-source confirmation of {why}",
            )
            attempt.queries.append(f"{query.query} site:{domain}")
            result = provider.search(query, agent_id="escalation")
            if not result.executed:
                # "We did not look" (budget cutoff, provider unavailable, ...),
                # never conflated with "we looked and it isn't there".
                if result.outcome is FetchOutcome.DISABLED:
                    budget_cut_short = True
                continue

            # The search hit is a pointer (METADATA_ONLY): it can tell us a
            # candidate URL is worth fetching, but it can never itself confirm
            # anything -- _confirms() below only runs against a fetched body.
            # Kept as full SearchHit-shaped Documents (not just URLs) so the
            # fetch below can carry forward whatever real metadata the hit
            # already had (requirement B2).
            candidates = [
                document for document in result.documents
                if _plausible_primary_candidate(document, domain)
            ]
            for candidate in candidates[:max_queries_per_fact]:
                url = candidate.url
                report.fetches_attempted += 1
                try:
                    fetched = provider.fetch(
                        url, reason=f"confirm: {why}", agent_id="escalation", known=candidate
                    )
                except Exception as exc:  # noqa: BLE001 - a fetch failure never confirms
                    log.warning("escalation fetch failed for %s: %s", url, exc)
                    fetched = None
                if fetched is None:
                    report.fetches_failed += 1
                    # Fetch failure (including a BudgetExceeded abort caught by
                    # the provider) leaves the claim unverified, never silently
                    # confirmed and never treated as a contradiction.
                    report.fetch_log.append(
                        FetchAttempt(
                            subject_id=fact.fact_id,
                            url=url,
                            authority=str(candidate.tier),
                            candidate_rank=-1,
                            outcome="fetch_failed",
                            failure_reason="fetch returned no document (network/parse/budget)",
                        )
                    )
                    continue
                confirms_it = _confirms(fetched, fact)
                report.fetch_log.append(
                    FetchAttempt(
                        subject_id=fact.fact_id,
                        url=url,
                        authority=str(fetched.authority),
                        candidate_rank=-1,
                        outcome="confirmed" if confirms_it else "no_answer",
                        tokens_used=0,
                        body_obtained=True,
                        body_answered=confirms_it,
                    )
                )
                if confirms_it:
                    confirming = fetched
                    break
            if confirming:
                break

        if confirming is not None:
            attempt.confirmed = True
            # Recorded from the FETCHED document only -- never from the search
            # hit that merely pointed at it.
            attempt.confirming_url = confirming.url
            attempt.confirming_tier = str(confirming.tier)
            # Requirement B3: WHO authored the confirming document decides
            # what this confirmation actually establishes. A statutory
            # issuer filing confirms "the issuer disclosed this" -- not
            # independent confirmation of an underlying regulator statement,
            # which only a genuine regulator-issued document supports.
            evidence_class, _, independent_confirmation = _evidence_for_authority(
                confirming.authority
            )
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.VERIFIED,
                    evidence_class=evidence_class,
                    # The claim was read from an actual document body just now
                    # (the confirming fetch) -- that is what content_kind
                    # records, independent of how the original claim was
                    # sourced (requirement M1).
                    content_kind=ContentKind.FULL_DOCUMENT,
                    independent_confirmation=independent_confirmation,
                    corroborating_source_ids=(*fact.corroborating_source_ids, confirming.doc_id),
                    confidence=min(1.0, fact.confidence + 0.25),
                    notes=(fact.notes + "; " if fact.notes else "")
                    + f"escalated and confirmed in a primary source body "
                    f"({confirming.authority}): {confirming.url}",
                )
            )
        else:
            note = "material claim; primary-source confirmation attempted and not found"
            if budget_cut_short:
                note = (
                    "material claim; primary-source confirmation incomplete -- the LLM token "
                    "budget was exhausted before every candidate domain could be searched or "
                    "fetched (not the same as searching and finding nothing)"
                )
            updated.append(
                replace(
                    fact,
                    verified_status=VerifiedStatus.UNVERIFIED_MATERIAL_CLAIM,
                    confidence=min(fact.confidence, 0.3),
                    notes=(fact.notes + "; " if fact.notes else "") + note,
                )
            )
        report.attempts.append(attempt)

    return updated, report


def escalate_unresolved_questions(
    questions: Sequence[UnresolvedQuestion],
    sources: Sequence[Source],
    provider: ResearchProvider,
    *,
    company: str,
    ticker: str = UNKNOWN,
    run_id: str = UNKNOWN,
    facts: Sequence[Fact] = (),
    max_questions: int = 6,
    max_candidates_per_question: int = 3,
) -> tuple[list[Fact], EscalationReport]:
    """Attempt primary-source confirmation for MATERIAL/CRITICAL unresolved
    questions (requirement C) -- escalation is not limited to weak facts.

    A live run had a material unresolved question (regulator endpoint
    acceptability) and reported "no material claim required escalation",
    because the existing :func:`escalate` only ever looks at ``Fact`` objects.
    A ``blocking`` :class:`UnresolvedQuestion` is this system's existing
    "material/critical" marker (the Decision-Grade Evidence Gate already reads
    it that way), so that is what drives this pass.

    Already-collected Tier 1/2 sources are tried BEFORE issuing any new
    search -- a filing the SEC/ClinicalTrials/FDA collectors already found is
    fetched directly rather than forcing a redundant web search for something
    already in hand. A source's own body is fetched (not merely its listing),
    so a filing's exhibits/attachments are reachable the same way a filing
    itself is: whatever URL is in ``sources``.

    Only a FETCHED body can create decision-grade evidence: a body that does
    not actually address the question (see ``_extract_answering_sentence``)
    leaves the question unresolved, and a search hit alone is never promoted
    to a fact.
    """
    report = EscalationReport()
    new_facts: list[Fact] = []
    usable, reason = provider.available()
    material = [q for q in questions if q.blocking][:max_questions]
    primary_sources = [s for s in sources if s.tier.is_primary]
    #: Requirement D: fetch only the top 1-2 plausible bodies initially,
    #: never three simply because three Tier-1 sources exist. Expansion (up
    #: to ``max_candidates_per_question``) only happens if those fail to
    #: answer the question and budget remains.
    initial_fetch_cap = min(2, max_candidates_per_question)

    for question in material:
        attempt = EscalationAttempt(
            fact_id="",
            claim=question.question[:300],
            reason="unresolved_material_question",
        )
        if not usable:
            attempt.note = reason
            report.attempts.append(attempt)
            continue

        subject_id = question.question[:120]
        already_tried_urls: set[str] = set()

        def _try_candidates(
            candidates: list[tuple[Source | Document, int]],
            *,
            question: UnresolvedQuestion = question,
            subject_id: str = subject_id,
            already_tried_urls: set[str] = already_tried_urls,
            provider: ResearchProvider = provider,
            primary_sources: list[Source] = primary_sources,
            report: EscalationReport = report,
        ) -> tuple[str | None, Document | None]:
            for candidate, rank in candidates:
                if candidate.url in already_tried_urls:
                    continue
                already_tried_urls.add(candidate.url)
                found_answer, fetched = _attempt_fetch_for_question(
                    provider, candidate, rank, question, subject_id, report
                )
                if found_answer is not None:
                    return found_answer, fetched
                # Requirement D: statutory-filing exhibit following, within
                # THIS SAME attempt -- a filing that references an exhibit
                # relevant to the question is followed before spending a
                # fresh search, preserving filing -> exhibit traceability.
                if fetched is not None and _EXHIBIT_REFERENCE_RE.search(fetched.text or ""):
                    for exhibit_source in _same_filing_exhibits(fetched, primary_sources):
                        if exhibit_source.url in already_tried_urls:
                            continue
                        already_tried_urls.add(exhibit_source.url)
                        exhibit_answer, exhibit_fetched = _attempt_fetch_for_question(
                            provider, exhibit_source, -2, question, subject_id, report
                        )
                        if exhibit_answer is not None:
                            return exhibit_answer, exhibit_fetched
            return None, None

        # 1. Rank already-collected primary sources by STRUCTURED metadata
        #    (requirement D): category linkage, domain routing, accession,
        #    date -- never lexical title overlap alone.
        scored = _score_candidates(question, primary_sources, facts)
        ranked_candidates: list[Source] = [source for source, _ in scored]

        # 1b. If the ranking is still ambiguous, issue ONE narrowly-scoped
        #     discovery search (this question's single best-routed domain)
        #     to identify the most likely primary source BEFORE spending
        #     multiple fetches guessing among near-equal candidates.
        discovered_first: list[Document] = []
        routed = domains_for_category(question.category)
        if _ranking_is_ambiguous(scored) and routed:
            domain = routed[0]
            query = ResearchQuery(
                query=f"{company} {_key_terms(question.question)}",
                domain=ResearchDomain.REGULATORY,
                stance="verify",
                allowed_domains=(domain,),
                rationale=f"identify the most likely primary source for: {subject_id}",
            )
            attempt.queries.append(f"{query.query} site:{domain} [tie-break]")
            result = provider.search(query, agent_id="escalation_question")
            if result.executed:
                discovered_first = [
                    document
                    for document in result.documents
                    if _plausible_primary_candidate(document, domain)
                ][:initial_fetch_cap]

        # 2. Fetch only the top plausible bodies initially: the tie-breaking
        #    discovery result (if any) first, then the structurally ranked
        #    already-collected candidates.
        fetch_plan: list[tuple[Source | Document, int]] = [
            (document, -1) for document in discovered_first
        ] + [(source, rank) for rank, source in enumerate(ranked_candidates, start=1)]
        initial_batch = fetch_plan[:initial_fetch_cap]
        expansion_batch = fetch_plan[initial_fetch_cap:max_candidates_per_question]

        answer, answering_document = _try_candidates(initial_batch)

        # 3. Expand only if the initial batch failed to answer it and budget
        #    (more ranked candidates) remains.
        if answer is None and expansion_batch:
            answer, answering_document = _try_candidates(expansion_batch)

        # 4. Only issue a broader search when nothing already collected or
        #    discovered answered it -- routed by the QUESTION's own category
        #    (requirement C), never a fixed "search everything" domain list.
        if answer is None:
            attempt.searched = True
            for domain in domains_for_category(question.category)[:2]:
                query = ResearchQuery(
                    query=f"{company} {_key_terms(question.question)}",
                    domain=ResearchDomain.REGULATORY,
                    stance="verify",
                    allowed_domains=(domain,),
                    rationale=f"resolve unresolved question: {subject_id}",
                )
                attempt.queries.append(f"{query.query} site:{domain}")
                result = provider.search(query, agent_id="escalation_question")
                if not result.executed:
                    continue
                candidates = [
                    document
                    for document in result.documents
                    if _plausible_primary_candidate(document, domain)
                ]
                answer, answering_document = _try_candidates(
                    [(c, -1) for c in candidates[:max_candidates_per_question]]
                )
                if answer is not None:
                    break

        if answer is not None and answering_document is not None:
            attempt.confirmed = True
            attempt.confirming_url = answering_document.url
            attempt.confirming_tier = str(answering_document.tier)
            new_facts.append(
                _fact_from_answered_question(
                    answer, answering_document, question, ticker=ticker, run_id=run_id
                )
            )
        else:
            attempt.note = (
                "unresolved material question; primary-source confirmation attempted and the "
                "fetched body(ies) did not address it"
            )
        report.attempts.append(attempt)

    return new_facts, report


def _attempt_fetch_for_question(
    provider: ResearchProvider,
    candidate: Source | Document,
    rank: int,
    question: UnresolvedQuestion,
    subject_id: str,
    report: EscalationReport,
) -> tuple[str | None, Document | None]:
    """Fetch one candidate for an unresolved question.

    Always records a :class:`FetchAttempt` (requirement E), whatever the
    outcome. Returns ``(answer, document)`` when the fetched body answers
    the question; ``(None, document)`` when it fetched cleanly but did not
    answer it (the caller uses ``document`` to check for an exhibit
    reference); ``(None, None)`` when the fetch itself failed.
    """
    report.fetches_attempted += 1
    try:
        fetched = provider.fetch(
            candidate.url,
            reason=f"resolve unresolved question: {subject_id}",
            known=candidate,
        )
    except Exception as exc:  # noqa: BLE001 - a fetch failure never resolves
        log.warning("escalation (question) fetch failed for %s: %s", candidate.url, exc)
        fetched = None
    if fetched is None:
        report.fetches_failed += 1
        report.fetch_log.append(
            FetchAttempt(
                subject_id=subject_id,
                url=candidate.url,
                authority=str(DocumentAuthority.UNKNOWN),
                candidate_rank=rank,
                outcome="fetch_failed",
                failure_reason="fetch returned no document (network/parse/budget)",
            )
        )
        return None, None

    sentence = _extract_answering_sentence(fetched, question)
    answered = sentence is not None
    report.fetch_log.append(
        FetchAttempt(
            subject_id=subject_id,
            url=candidate.url,
            authority=str(fetched.authority),
            candidate_rank=rank,
            outcome="answered" if answered else "no_answer",
            body_obtained=True,
            body_answered=answered,
            supporting_sentence=sentence or "",
        )
    )
    if answered:
        return sentence, fetched
    return None, fetched


def _fact_from_answered_question(
    answer: str,
    document: Document,
    question: UnresolvedQuestion,
    *,
    ticker: str,
    run_id: str,
) -> Fact:
    """Build a Fact from a fetched body that answered a material unresolved
    question. Verbatim text only -- never a paraphrase.

    Requirement B5: dates are the document's OWN known dates, never
    retrieval time (fetch() itself already never substitutes one for the
    other, but this function must not reintroduce that substitution either).
    evidence_class/company_claim/independent_confirmation are derived from
    the fetched document's real DocumentAuthority -- never unconditionally
    "independent" just because a body was successfully fetched.
    """
    evidence_class, company_claim, independent_confirmation = _evidence_for_authority(
        document.authority
    )
    # Prefer the document's own event_date; fall back to published_date only
    # when no event_date is known. Neither ever falls back to retrieved_at.
    event_date = document.event_date if document.event_date != UNKNOWN else document.published_date
    verified_status = (
        VerifiedStatus.VERIFIED
        if document.authority is not DocumentAuthority.UNKNOWN
        else VerifiedStatus.NOT_VERIFIED
    )
    return Fact(
        fact_id=make_fact_id(ticker, question.category, answer, document.url, event_date),
        ticker=ticker,
        category=question.category,
        claim=answer,
        evidence_class=evidence_class,
        source_id=make_source_id(document.url, document.title),
        source_url=document.url,
        source_title=document.title,
        source_tier=document.tier,
        publication_date=document.published_date,
        event_date=event_date,
        effective_date=document.effective_date,
        filing_date=document.filing_date,
        verified_status=verified_status,
        confidence=0.75 if independent_confirmation or evidence_class == EvidenceClass.VERIFIED_FACT else 0.3,
        company_claim=company_claim,
        materiality=Materiality.CRITICAL,
        provenance=document.provenance,
        run_id=run_id,
        notes=(
            f"escalated from a material unresolved question via a fetched primary-source body "
            f"({document.authority}): {question.question[:200]}"
        ),
        content_kind=ContentKind.FULL_DOCUMENT,
        independent_confirmation=independent_confirmation,
    )


def _key_terms(claim: str, limit: int = 10) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9-]+", claim) if len(w) > 4]
    return " ".join(words[:limit])
