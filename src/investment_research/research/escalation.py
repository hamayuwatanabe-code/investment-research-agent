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

#: Domains searched when escalating, in preference order.
PRIMARY_DOMAINS: tuple[str, ...] = (
    "sec.gov",
    "fda.gov",
    "clinicaltrials.gov",
    "ema.europa.eu",
    "nih.gov",
)

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
class EscalationReport:
    attempts: list[EscalationAttempt] = field(default_factory=list)
    #: Diagnostics (requirement G): how many candidate URLs were actually
    #: fetched, and how many of those fetches came back empty (failed,
    #: raised, or budget-cut) versus produced a body.
    fetches_attempted: int = 0
    fetches_failed: int = 0

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


def _rank_candidates_for_question(
    question: UnresolvedQuestion, sources: Sequence[Source], facts: Sequence[Fact]
) -> list[Source]:
    """Order already-collected primary sources by relevance to ``question``.

    Two signals, either sufficient to rank a source ahead of an unranked one:
    1. The source backs at least one collected fact in the SAME category as
       the question (e.g. a REGULATORY question and a REGULATORY fact whose
       ``source_url`` matches) -- the strongest available signal, since it
       means the collector filed this source under the same topic.
    2. The source's own title shares distinctive vocabulary with the
       question -- catches a source not yet backing any fact (or backing one
       in a different category) whose listing itself names the topic (e.g.
       "Form 8-K, Regulatory Update").

    Ties keep their original relative order (a stable sort), so this never
    reorders otherwise-equal candidates arbitrarily.
    """
    same_category_urls = {
        f.source_url for f in facts if f.category == question.category and f.source_url
    }
    question_terms = _distinctive_terms(f"{question.question} {question.why_it_matters}")

    def _score(source: Source) -> int:
        score = 0
        if source.url in same_category_urls:
            score += 2
        if question_terms and _distinctive_terms(source.title) & question_terms:
            score += 1
        return score

    return sorted(sources, key=_score, reverse=True)


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
        for domain in PRIMARY_DOMAINS[:max_queries_per_fact]:
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
                    continue
                if _confirms(fetched, fact):
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

        answer: str | None = None
        answering_document: Document | None = None

        # 1. Already-collected Tier 1/2 sources first (requirement C: don't
        #    force a redundant search when a plausible primary source, e.g. a
        #    filing the SEC collector already found, is already in hand).
        #    Ranked by topical relevance to THIS question -- a source behind a
        #    fact in the same category, or whose own title overlaps the
        #    question's vocabulary, is tried before an unrelated one, so the
        #    per-question fetch cap is spent on the most plausible candidate
        #    first rather than whatever happened to be collected first.
        ranked_candidates = _rank_candidates_for_question(question, primary_sources, facts)
        for source in ranked_candidates[:max_candidates_per_question]:
            report.fetches_attempted += 1
            try:
                fetched = provider.fetch(
                    source.url,
                    reason=f"resolve unresolved question: {question.question[:120]}",
                    known=source,
                )
            except Exception as exc:  # noqa: BLE001 - a fetch failure never resolves
                log.warning("escalation (question) fetch failed for %s: %s", source.url, exc)
                fetched = None
            if fetched is None:
                report.fetches_failed += 1
                continue
            sentence = _extract_answering_sentence(fetched, question)
            if sentence is not None:
                answer, answering_document = sentence, fetched
                break

        # 2. Only search when nothing already collected answered it.
        if answer is None:
            attempt.searched = True
            for domain in PRIMARY_DOMAINS[:2]:
                query = ResearchQuery(
                    query=f"{company} {_key_terms(question.question)}",
                    domain=ResearchDomain.REGULATORY,
                    stance="verify",
                    allowed_domains=(domain,),
                    rationale=f"resolve unresolved question: {question.question[:120]}",
                )
                attempt.queries.append(f"{query.query} site:{domain}")
                result = provider.search(query, agent_id="escalation_question")
                if not result.executed:
                    continue
                candidates = [
                    document for document in result.documents
                    if _plausible_primary_candidate(document, domain)
                ]
                for candidate in candidates[:max_candidates_per_question]:
                    url = candidate.url
                    report.fetches_attempted += 1
                    try:
                        fetched = provider.fetch(
                            url,
                            reason=f"resolve unresolved question: {question.question[:120]}",
                            known=candidate,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("escalation (question) fetch failed for %s: %s", url, exc)
                        fetched = None
                    if fetched is None:
                        report.fetches_failed += 1
                        continue
                    sentence = _extract_answering_sentence(fetched, question)
                    if sentence is not None:
                        answer, answering_document = sentence, fetched
                        break
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
