"""PubMed (NCBI E-utilities) and Europe PMC XML/JSON parsing -- pure
functions only, mirroring ``collectors/form4.py``'s "pure functions only"
scope exactly. Nothing here makes a network call; see
``research/literature_acquisition_adapter.py`` for the LOCATE/FETCH/PARSE
adapters that acquire the bodies this module parses.

Official specs modeled: NCBI E-utilities
(https://www.ncbi.nlm.nih.gov/books/NBK25497/) and the Europe PMC RESTful
Web Service (https://europepmc.org/RestfulWebService).

Real PubMed XML structure modeled (element names, unabbreviated)::

    PubmedArticleSet
        PubmedArticle+
            MedlineCitation
                PMID
                Article
                    Journal (Title, ISOAbbreviation, JournalIssue/PubDate)
                    ArticleTitle
                    Abstract / AbstractText* (Label attr for structured abstracts)
                    AuthorList / Author* (LastName/ForeName or CollectiveName,
                        AffiliationInfo/Affiliation)
                    Language
                    PublicationTypeList / PublicationType*
                    ArticleDate (DateType="Electronic")
                    GrantList / Grant* (GrantID, Agency, Country)
                    DataBankList / DataBank
                        (DataBankName == "ClinicalTrials.gov") ->
                        AccessionNumberList / AccessionNumber* (NCT ids)
                CoiStatement
                MeshHeadingList / MeshHeading / DescriptorName*
                KeywordList / Keyword*
            PubmedData
                History
                PublicationStatus
                ArticleIdList / ArticleId* (IdType: pubmed/pmc/doi)
                CommentsCorrectionsList / CommentsCorrections*
                    (RefType: RetractionIn/ErratumIn/CorrectionIn/...; PMID; RefSource)

Evidence boundary (Phase 3F requirement 6, corrected in Phase 3F.0.1,
CRITICAL -- mirrors ``collectors/form4.py``'s own precedent for a
different evidence class): a document being retrievable through PubMed or
Europe PMC, and even a specific sentence it contains, is NEVER itself
"peer-reviewed", "efficacy confirmed", "safety confirmed", "endpoint
accepted by the FDA", independent confirmation, or a ``VERIFIED_FACT``.
PubMed indexes far more than MEDLINE-reviewed journal articles -- online
books, editorials, letters, and (via NIH's Preprint Pilot) preprints all
carry a PMID -- and Europe PMC's own search explicitly covers preprint
servers. This module therefore NEVER infers peer review from indexing
alone or from a "Journal Article" publication type alone: every parsed
article/search result carries its own ``PublicationStage`` (what KIND of
document this is, from the source's own structured metadata) and
``PeerReviewStatus`` (``UNKNOWN`` unless an explicit negative signal --
e.g. a "Preprint" publication type or a Europe PMC preprint-server source
code -- says otherwise; NEVER set to ``CONFIRMED`` by this module).
``ContentKind.FULL_DOCUMENT`` on an Europe PMC open-access fetch means
only "the full text was retrieved" -- it carries no implication about peer
review, quality, or independent confirmation.

Every RawFact this module builds carries ``FactCategory.SCIENCE``,
``RawFact.source_authority=DocumentAuthority.BIOMEDICAL_LITERATURE`` (the
PRIMARY signal ``agents/evidence_integrity.py`` uses to route these facts
to ``EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION``, which is excluded
from ``DECISION_GRADE_CLASSES`` -- the ``unit`` prefix below is retained
only as auxiliary, human-readable grouping, never as the classification
key itself), and ``company_claim=False`` always -- never inferred from
author affiliation alone, however many authors are company-employed or
company-funded. A "reported result" RawFact's claim is always the
DOCUMENT'S OWN wording (a sentence drawn from its abstract/full text),
never converted into a confirmed-efficacy statement at this stage -- that
conversion never happens anywhere in this system (CLAUDE.md rule 1/7).

Retracted articles: ``retracted=True`` is kept structurally, but this
module does NOT delete, hide, or renumber anything about the article --
a retracted paper's facts are still generated (never decision-grade
regardless -- see above), with the retraction itself surfaced as its own
RawFact. Corrections/errata preserve their relation to the original
article (``related_pmid``) and never overwrite it.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..schemas.enums import (
    UNKNOWN,
    ContentKind,
    DocumentAuthority,
    FactCategory,
    PeerReviewStatus,
    PublicationStage,
)
from ..schemas.fact import RawFact, Source

#: NCBI's own DataBankName for a ClinicalTrials.gov cross-reference --
#: exact match only, never a fuzzy "contains clinicaltrials".
_CLINICALTRIALS_DATABANK_NAME = "ClinicalTrials.gov"

_NCT_RE = re.compile(r"\bNCT\d{8}\b")


class XmlShapeError(str, Enum):
    """Mirrors ``collectors/form4.py``'s ``XmlShapeError`` precedent: WHY a
    response body is not usable, never one opaque generic failure."""

    MALFORMED = "MALFORMED"
    NOT_PUBMED_ARTICLE_SET = "NOT_PUBMED_ARTICLE_SET"
    EMPTY = "EMPTY"
    HTML_ERROR_PAGE = "HTML_ERROR_PAGE"


#: A cheap, conservative signal that a body is an HTML error page served
#: instead of XML (a proxy/WAF block page, a 429/5xx rendered as HTML) --
#: never used to reject genuine XML that happens to contain these words in
#: prose, since this only fires when the DOCUMENT ROOT itself is HTML.
_HTML_ROOT_TAGS = frozenset({"html", "HTML"})


def check_pubmed_xml_shape(xml_text: str) -> tuple[XmlShapeError | None, str]:
    """Is this body well-formed XML, rooted at ``<PubmedArticleSet>``?

    Returns ``(None, "")`` on success, or ``(XmlShapeError, reason)`` on any
    failure -- never raises (Phase 3F requirement: malformed XML / HTML
    error page / empty body must each be their own diagnosable outcome).
    """
    if not xml_text or not xml_text.strip():
        return XmlShapeError.EMPTY, "response body was empty"
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return XmlShapeError.MALFORMED, f"malformed XML: {exc}"
    if root.tag in _HTML_ROOT_TAGS:
        return XmlShapeError.HTML_ERROR_PAGE, "response body is an HTML page, not PubMed XML"
    if root.tag != "PubmedArticleSet":
        return (
            XmlShapeError.NOT_PUBMED_ARTICLE_SET,
            f"root element is <{root.tag}>, not <PubmedArticleSet>",
        )
    return None, ""


def _text(el: ET.Element | None, path: str, default: str = UNKNOWN) -> str:
    if el is None:
        return default
    found = el.find(path)
    if found is None or found.text is None:
        return default
    stripped = found.text.strip()
    return stripped or default


def _pub_date(date_el: ET.Element | None) -> str:
    """A ``PubDate``/``ArticleDate``-shaped element's Year/Month/Day (or
    ``MedlineDate`` free-text fallback, NLM's own convention for an
    imprecise date) -- never a guessed day when only year/month is known."""
    if date_el is None:
        return UNKNOWN
    medline_date = _text(date_el, "MedlineDate")
    if medline_date != UNKNOWN:
        return medline_date
    year = _text(date_el, "Year")
    if year == UNKNOWN:
        return UNKNOWN
    month = _text(date_el, "Month")
    day = _text(date_el, "Day")
    parts = [year]
    if month != UNKNOWN:
        parts.append(month)
        if day != UNKNOWN:
            parts.append(day)
    return "-".join(parts)


@dataclass(frozen=True)
class AbstractSection:
    label: str  # e.g. "BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS", or UNKNOWN for unstructured
    text: str


@dataclass(frozen=True)
class LiteratureAuthor:
    last_name: str
    fore_name: str
    collective_name: str
    affiliations: tuple[str, ...]


@dataclass(frozen=True)
class LiteratureGrant:
    grant_id: str
    agency: str
    country: str


@dataclass(frozen=True)
class CommentCorrection:
    """One ``CommentsCorrections`` entry -- retraction/correction/erratum/
    related-article relation, kept SEPARATE from the article it points at,
    never used to overwrite the original (Phase 3F requirement 6)."""

    ref_type: str  # e.g. "RetractionIn", "ErratumIn", "CorrectionIn", "RetractionOf", "UpdateIn"
    ref_source: str
    related_pmid: str


@dataclass(frozen=True)
class ParsedPubmedArticle:
    pmid: str
    pmcid: str
    doi: str
    article_title: str
    abstract_sections: tuple[AbstractSection, ...]
    journal_title: str
    journal_iso_abbreviation: str
    publication_date: str
    electronic_publication_date: str
    publication_types: tuple[str, ...]
    authors: tuple[LiteratureAuthor, ...]
    language: str
    keywords: tuple[str, ...]
    mesh_terms: tuple[str, ...]
    grants: tuple[LiteratureGrant, ...]
    coi_statement: str
    nct_ids: tuple[str, ...]
    comments_corrections: tuple[CommentCorrection, ...]
    article_status: str  # PublicationStatus, e.g. "ppublish", "epublish", "aheadofprint"
    #: Phase 3F.0.1: what KIND of document this is, from PubMed's own
    #: PublicationTypeList -- never inferred from title/abstract wording.
    #: See ``_publication_stage_from_types`` for the exact mapping.
    publication_stage: PublicationStage = PublicationStage.UNKNOWN
    #: Phase 3F.0.1: whether peer review is actually confirmed. This module
    #: NEVER sets this to CONFIRMED (indexing, and even "Journal Article",
    #: are not evidence of peer review) -- only ever UNKNOWN or, for an
    #: explicit negative signal (a "Preprint" publication type),
    #: NOT_PEER_REVIEWED. See ``_peer_review_status_from_stage``.
    peer_review_status: PeerReviewStatus = PeerReviewStatus.UNKNOWN

    @property
    def has_abstract(self) -> bool:
        return any(section.text for section in self.abstract_sections)

    @property
    def is_retracted(self) -> bool:
        return any(cc.ref_type == "RetractionIn" for cc in self.comments_corrections)

    @property
    def is_correction_or_erratum(self) -> bool:
        return any(cc.ref_type in ("ErratumIn", "CorrectionIn") for cc in self.comments_corrections)


def _parse_authors(article_el: ET.Element) -> tuple[LiteratureAuthor, ...]:
    authors: list[LiteratureAuthor] = []
    for author_el in article_el.findall("AuthorList/Author"):
        collective = _text(author_el, "CollectiveName", "")
        affiliations = tuple(
            (aff.text or "").strip()
            for aff in author_el.findall("AffiliationInfo/Affiliation")
            if aff.text and aff.text.strip()
        )
        authors.append(
            LiteratureAuthor(
                last_name=_text(author_el, "LastName", ""),
                fore_name=_text(author_el, "ForeName", ""),
                collective_name=collective,
                affiliations=affiliations,
            )
        )
    return tuple(authors)


def _parse_grants(article_el: ET.Element) -> tuple[LiteratureGrant, ...]:
    return tuple(
        LiteratureGrant(
            grant_id=_text(grant_el, "GrantID"),
            agency=_text(grant_el, "Agency"),
            country=_text(grant_el, "Country"),
        )
        for grant_el in article_el.findall("GrantList/Grant")
    )


def _parse_nct_ids(article_el: ET.Element) -> tuple[str, ...]:
    """NCT ids ONLY from a ``DataBank`` whose ``DataBankName`` is literally
    ``ClinicalTrials.gov`` -- never a regex sweep of the whole document
    (which would also match an NCT id merely mentioned in prose, a
    materially weaker signal this module keeps separate by not extracting
    it here at all)."""
    nct_ids: list[str] = []
    for databank_el in article_el.findall("DataBankList/DataBank"):
        name = _text(databank_el, "DataBankName")
        if name != _CLINICALTRIALS_DATABANK_NAME:
            continue
        for acc_el in databank_el.findall("AccessionNumberList/AccessionNumber"):
            value = (acc_el.text or "").strip()
            if _NCT_RE.fullmatch(value):
                nct_ids.append(value)
    return tuple(dict.fromkeys(nct_ids))  # de-duplicated, order preserved


def _parse_comments_corrections(pubmed_data_el: ET.Element | None) -> tuple[CommentCorrection, ...]:
    if pubmed_data_el is None:
        return ()
    return tuple(
        CommentCorrection(
            ref_type=cc_el.get("RefType", UNKNOWN),
            ref_source=_text(cc_el, "RefSource"),
            related_pmid=_text(cc_el, "PMID"),
        )
        for cc_el in pubmed_data_el.findall("CommentsCorrectionsList/CommentsCorrections")
    )


def _parse_article_ids(pubmed_data_el: ET.Element | None) -> tuple[str, str]:
    """``(pmcid, doi)`` from ``PubmedData/ArticleIdList`` -- the PMID
    itself is read separately from ``MedlineCitation/PMID`` (the
    authoritative element), never inferred solely from this list."""
    pmcid, doi = UNKNOWN, UNKNOWN
    if pubmed_data_el is None:
        return pmcid, doi
    for id_el in pubmed_data_el.findall("ArticleIdList/ArticleId"):
        id_type = id_el.get("IdType", "")
        value = (id_el.text or "").strip()
        if not value:
            continue
        if id_type == "pmc":
            pmcid = value
        elif id_type == "doi":
            doi = value
    return pmcid, doi


#: Real NLM PublicationType values mapped to PublicationStage, checked in
#: priority order (most specific first). Structured-metadata-only -- never
#: inferred from title/abstract text (Phase 3F.0.1 requirement 2). These
#: are NLM's own controlled-vocabulary strings, matched exactly (case-
#: sensitive); an unrecognized or absent type never defaults to
#: JOURNAL_ARTICLE (see ``_publication_stage_from_types``).
_PUBLICATION_TYPE_STAGE_PRIORITY: tuple[tuple[str, PublicationStage], ...] = (
    ("Preprint", PublicationStage.PREPRINT),
    ("Retraction of Publication", PublicationStage.RETRACTION),
    ("Published Erratum", PublicationStage.ERRATUM),
    ("Corrected and Republished Article", PublicationStage.CORRECTION),
    ("Editorial", PublicationStage.EDITORIAL),
    ("Letter", PublicationStage.LETTER),
    ("Books and Documents", PublicationStage.BOOK_OR_CHAPTER),
    ("Book Chapter", PublicationStage.BOOK_OR_CHAPTER),
    ("Online Book", PublicationStage.BOOK_OR_CHAPTER),
    ("Journal Article", PublicationStage.JOURNAL_ARTICLE),
)


def _publication_stage_from_types(publication_types: tuple[str, ...]) -> PublicationStage:
    """The FIRST matching entry in ``_PUBLICATION_TYPE_STAGE_PRIORITY``
    (most specific type first) wins -- e.g. an erratum notice that is ALSO
    tagged "Journal Article" is still classified ERRATUM, never
    JOURNAL_ARTICLE. ``OTHER`` when ``publication_types`` is non-empty but
    nothing recognized matches (e.g. "Review", "Comment"); ``UNKNOWN`` when
    the source supplied no publication type at all -- never guessed."""
    for candidate, stage in _PUBLICATION_TYPE_STAGE_PRIORITY:
        if candidate in publication_types:
            return stage
    return PublicationStage.OTHER if publication_types else PublicationStage.UNKNOWN


#: PublicationStage values that are themselves an explicit "not peer
#: reviewed" signal (Phase 3F.0.1 requirement 2). Deliberately narrow: this
#: module never infers NOT_PEER_REVIEWED from any other stage, and never
#: infers CONFIRMED from ANY stage, including JOURNAL_ARTICLE -- PubMed
#: indexing a document as a "Journal Article" is not evidence that peer
#: review happened, only that the piece is journal-shaped.
_NOT_PEER_REVIEWED_STAGES = frozenset({PublicationStage.PREPRINT})


def _peer_review_status_from_stage(stage: PublicationStage) -> PeerReviewStatus:
    if stage in _NOT_PEER_REVIEWED_STAGES:
        return PeerReviewStatus.NOT_PEER_REVIEWED
    return PeerReviewStatus.UNKNOWN


def _parse_pubmed_article(article_el: ET.Element) -> ParsedPubmedArticle:
    citation = article_el.find("MedlineCitation")
    article = citation.find("Article") if citation is not None else None
    pubmed_data = article_el.find("PubmedData")
    journal = article.find("Journal") if article is not None else None

    abstract_sections: list[AbstractSection] = []
    if article is not None:
        for text_el in article.findall("Abstract/AbstractText"):
            label = text_el.get("Label", UNKNOWN)
            abstract_sections.append(AbstractSection(label=label, text=(text_el.text or "").strip()))

    publication_types = tuple(
        (pt.text or "").strip()
        for pt in (article.findall("PublicationTypeList/PublicationType") if article is not None else [])
        if pt.text and pt.text.strip()
    )
    keywords = tuple(
        (kw.text or "").strip()
        for kw in (citation.findall("KeywordList/Keyword") if citation is not None else [])
        if kw.text and kw.text.strip()
    )
    mesh_terms = tuple(
        (dn.text or "").strip()
        for dn in (citation.findall("MeshHeadingList/MeshHeading/DescriptorName") if citation is not None else [])
        if dn.text and dn.text.strip()
    )
    pmcid, doi = _parse_article_ids(pubmed_data)
    publication_stage = _publication_stage_from_types(publication_types)
    peer_review_status = _peer_review_status_from_stage(publication_stage)

    return ParsedPubmedArticle(
        pmid=_text(citation, "PMID"),
        pmcid=pmcid,
        doi=doi,
        article_title=_text(article, "ArticleTitle"),
        abstract_sections=tuple(abstract_sections),
        journal_title=_text(journal, "Title"),
        journal_iso_abbreviation=_text(journal, "ISOAbbreviation"),
        publication_date=_pub_date(journal.find("JournalIssue/PubDate") if journal is not None else None),
        electronic_publication_date=_pub_date(
            article.find("ArticleDate") if article is not None else None
        ),
        publication_types=publication_types,
        authors=_parse_authors(article) if article is not None else (),
        language=_text(article, "Language"),
        keywords=keywords,
        mesh_terms=mesh_terms,
        grants=_parse_grants(article) if article is not None else (),
        coi_statement=_text(citation, "CoiStatement", ""),
        nct_ids=_parse_nct_ids(article) if article is not None else (),
        comments_corrections=_parse_comments_corrections(pubmed_data),
        article_status=_text(pubmed_data, "PublicationStatus"),
        publication_stage=publication_stage,
        peer_review_status=peer_review_status,
    )


def parse_pubmed_articleset(xml_text: str) -> list[ParsedPubmedArticle] | None:
    """Parses a real ``<PubmedArticleSet>`` EFetch response, one or more
    ``PubmedArticle`` children (batch fetch -- Phase 3F requirement: never
    fetch a large PMID set one at a time). Returns ``None`` for anything
    that fails ``check_pubmed_xml_shape``; a well-formed set with zero
    ``PubmedArticle`` children returns an empty list (a real, if unusual,
    EFetch response shape -- distinct from ``None``, which means the body
    itself was unusable)."""
    shape_error, _reason = check_pubmed_xml_shape(xml_text)
    if shape_error is not None:
        return None
    root = ET.fromstring(xml_text)
    return [_parse_pubmed_article(article_el) for article_el in root.findall("PubmedArticle")]


# -- Europe PMC ---------------------------------------------------------


#: Europe PMC's own ``source`` code for its aggregated preprint servers
#: (bioRxiv, medRxiv, Research Square, ...) -- the one explicit, structured
#: "not peer reviewed" signal this module reads from Europe PMC (Phase
#: 3F.0.1 requirement 2's "Europe PMC preprintはNOT_PEER_REVIEWED").
_EUROPEPMC_PREPRINT_SOURCE_CODES = frozenset({"PPR"})


@dataclass(frozen=True)
class EuropePmcSearchResult:
    """One ``resultList.result[]`` entry from Europe PMC's ``/search``
    endpoint -- metadata ONLY (Phase 3F requirement 3: open-access status
    is a claim about availability, never itself the full text)."""

    pmid: str
    pmcid: str
    doi: str
    title: str
    is_open_access: bool
    in_epmc: bool
    license: str
    journal_title: str
    pub_year: str
    #: Europe PMC's own result-source code (e.g. "MED" = MEDLINE, "PPR" =
    #: preprint server, "PMC" = PMC-only, "AGR"/"CBA" = other aggregated
    #: sources). ``UNKNOWN`` when the response omitted it.
    source: str = UNKNOWN

    @property
    def publication_stage(self) -> PublicationStage:
        if self.source in _EUROPEPMC_PREPRINT_SOURCE_CODES:
            return PublicationStage.PREPRINT
        return PublicationStage.UNKNOWN

    @property
    def peer_review_status(self) -> PeerReviewStatus:
        """Phase 3F.0.1 requirement 2: NOT_PEER_REVIEWED only for a
        confirmed preprint-server source; otherwise UNKNOWN -- Europe PMC
        listing a result under "MED" (MEDLINE) is still not, by itself,
        proof that peer review happened (it may be an editorial, a letter,
        or a MEDLINE-indexed preprint)."""
        if self.source in _EUROPEPMC_PREPRINT_SOURCE_CODES:
            return PeerReviewStatus.NOT_PEER_REVIEWED
        return PeerReviewStatus.UNKNOWN


def parse_europepmc_search_response(payload_text: str) -> list[EuropePmcSearchResult] | None:
    """Parses Europe PMC's real ``format=json`` search response shape:
    ``{"resultList": {"result": [...]}}``. Returns ``None`` for malformed
    JSON or a response missing the expected top-level shape -- never a
    best-effort partial guess."""
    try:
        payload = json.loads(payload_text) if payload_text else None
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    result_list = payload.get("resultList")
    if not isinstance(result_list, dict) or "result" not in result_list:
        return None
    results: list[EuropePmcSearchResult] = []
    for entry in result_list.get("result") or []:
        if not isinstance(entry, dict):
            continue
        results.append(
            EuropePmcSearchResult(
                pmid=str(entry.get("pmid", UNKNOWN)),
                pmcid=str(entry.get("pmcid", UNKNOWN)),
                doi=str(entry.get("doi", UNKNOWN)),
                title=str(entry.get("title", UNKNOWN)),
                is_open_access=str(entry.get("isOpenAccess", "N")).upper() == "Y",
                in_epmc=str(entry.get("inEPMC", "N")).upper() == "Y",
                license=str(entry.get("license", "")),
                journal_title=str(entry.get("journalTitle", UNKNOWN)),
                pub_year=str(entry.get("pubYear", UNKNOWN)),
                source=str(entry.get("source", UNKNOWN)),
            )
        )
    return results


@dataclass(frozen=True)
class FullTextSection:
    title: str
    text: str


@dataclass(frozen=True)
class ParsedEuropePmcFullText:
    """A genuinely obtained open-access full text (JATS-like XML). Kept
    STRUCTURALLY separate from ``ParsedPubmedArticle``'s abstract: this
    dataclass is never returned for an abstract-only or non-OA article
    (Phase 3F requirement 2/6)."""

    sections: tuple[FullTextSection, ...]

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"{s.title}\n{s.text}" if s.title else s.text for s in self.sections)


class FullTextXmlShapeError(str, Enum):
    MALFORMED = "MALFORMED"
    NOT_ARTICLE = "NOT_ARTICLE"
    HTML_ERROR_PAGE = "HTML_ERROR_PAGE"
    EMPTY = "EMPTY"


def check_europepmc_fulltext_xml_shape(xml_text: str) -> tuple[FullTextXmlShapeError | None, str]:
    if not xml_text or not xml_text.strip():
        return FullTextXmlShapeError.EMPTY, "response body was empty"
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return FullTextXmlShapeError.MALFORMED, f"malformed XML: {exc}"
    if root.tag in _HTML_ROOT_TAGS:
        return FullTextXmlShapeError.HTML_ERROR_PAGE, "response body is an HTML page, not full-text XML"
    if root.tag != "article":
        return FullTextXmlShapeError.NOT_ARTICLE, f"root element is <{root.tag}>, not <article>"
    return None, ""


def parse_europepmc_fulltext_xml(xml_text: str) -> ParsedEuropePmcFullText | None:
    """Parses a real Europe PMC ``fullTextXML`` (JATS-like) response:
    ``<article><body><sec><title/><p/>...</sec>...</body></article>``.
    Returns ``None`` on any shape failure. Only ``<body>`` sections are
    read -- ``<front>``/``<back>`` (metadata, references) are deliberately
    not treated as body text."""
    shape_error, _reason = check_europepmc_fulltext_xml_shape(xml_text)
    if shape_error is not None:
        return None
    root = ET.fromstring(xml_text)
    body = root.find("body")
    if body is None:
        return ParsedEuropePmcFullText(sections=())
    sections: list[FullTextSection] = []
    for sec_el in body.findall("sec"):
        title = _text(sec_el, "title", "")
        paragraphs = [
            (p.text or "").strip() for p in sec_el.findall("p") if p.text and p.text.strip()
        ]
        sections.append(FullTextSection(title=title, text=" ".join(paragraphs)))
    return ParsedEuropePmcFullText(sections=tuple(sections))


# -- RawFact generation ---------------------------------------------------

#: Every literature RawFact's ``unit`` is prefixed with this -- auxiliary,
#: human-readable grouping ONLY (Phase 3F.0.1). The actual classification
#: signal ``agents/evidence_integrity.py::_classify`` uses is
#: ``RawFact.source_authority=DocumentAuthority.BIOMEDICAL_LITERATURE``,
#: set unconditionally by ``_make_raw_fact`` below -- renaming or dropping
#: this prefix would change no evidence-class outcome.
LITERATURE_UNIT_PREFIX = "literature_"

_ENDPOINT_SENTENCE_RE = re.compile(r"[^.]*\bendpoint\b[^.]*\.", re.IGNORECASE)
_ADVERSE_EVENT_SENTENCE_RE = re.compile(
    r"[^.]*\b(?:adverse event|adverse reaction|serious adverse|toxicit(?:y|ies))\b[^.]*\.",
    re.IGNORECASE,
)
_RESULT_LABELS = frozenset({"RESULTS", "CONCLUSION", "CONCLUSIONS", "FINDINGS"})
_DESIGN_LABELS = frozenset({"METHODS", "DESIGN", "OBJECTIVE", "OBJECTIVES", "BACKGROUND"})


def _make_raw_fact(
    ticker: str,
    claim: str,
    source: Source,
    *,
    unit: str,
    value: Any = UNKNOWN,
    collector: str,
    document_id: str | None,
    content_kind: ContentKind,
) -> RawFact:
    return RawFact(
        ticker=ticker.upper(),
        category=FactCategory.SCIENCE,
        claim=claim,
        source=source,
        value=value,
        unit=f"{LITERATURE_UNIT_PREFIX}{unit}",
        # Never inferred from author affiliation alone (Phase 3F
        # requirement 6) -- a publication's authors are not the issuer
        # speaking, whoever they are employed by.
        company_claim=False,
        collector=collector,
        document_id=document_id,
        content_kind=content_kind,
        # Phase 3F.0.1: the PRIMARY signal agents/evidence_integrity.py
        # uses to classify this fact -- set unconditionally on every
        # literature RawFact, regardless of unit/content_kind.
        source_authority=DocumentAuthority.BIOMEDICAL_LITERATURE,
    )


def raw_facts_from_pubmed_article(
    ticker: str,
    parsed: ParsedPubmedArticle,
    source: Source,
    *,
    collector: str = "literature_pubmed",
    document_id: str | None = None,
) -> list[RawFact]:
    """One RawFact per distinct fact TYPE (Phase 3F requirement 7) -- never
    merged into one undifferentiated "paper says X" blob. A "reported
    result" RawFact's claim is always the paper's OWN wording; it is never
    converted to "efficacy proven" here or anywhere downstream of this
    function (that conversion never happens in this system at all)."""
    facts: list[RawFact] = []
    has_abstract = parsed.has_abstract
    content_kind = ContentKind.EXCERPT if has_abstract else ContentKind.METADATA_ONLY

    facts.append(
        _make_raw_fact(
            ticker,
            f"A PubMed-indexed publication exists: PMID {parsed.pmid}"
            + (f", PMCID {parsed.pmcid}" if parsed.pmcid != UNKNOWN else "")
            + (f", DOI {parsed.doi}" if parsed.doi != UNKNOWN else ""),
            source, unit="publication_exists", value=parsed.pmid,
            collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
        )
    )
    if parsed.article_title != UNKNOWN:
        facts.append(
            _make_raw_fact(
                ticker, f"Article title (PMID {parsed.pmid}): {parsed.article_title}",
                source, unit="article_title", value=parsed.article_title,
                collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
            )
        )
    if parsed.publication_date != UNKNOWN:
        facts.append(
            _make_raw_fact(
                ticker,
                f"Publication date (PMID {parsed.pmid}): {parsed.publication_date}"
                + (
                    f"; electronic publication date: {parsed.electronic_publication_date}"
                    if parsed.electronic_publication_date != UNKNOWN
                    else ""
                ),
                source, unit="publication_date", value=parsed.publication_date,
                collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
            )
        )
    for pt in parsed.publication_types:
        facts.append(
            _make_raw_fact(
                ticker, f"Publication type (PMID {parsed.pmid}): {pt}",
                source, unit="publication_type", value=pt,
                collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
            )
        )
    # Phase 3F.0.1 requirement 2: publication_stage/peer_review_status are
    # ALWAYS their own explicit facts, kept separate from "publication
    # type" above -- a downstream reader must never have to re-derive
    # "was this peer reviewed" from a raw NLM type string itself.
    facts.append(
        _make_raw_fact(
            ticker,
            f"Publication stage (PMID {parsed.pmid}): {parsed.publication_stage.value}",
            source, unit="publication_stage", value=parsed.publication_stage.value,
            collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
        )
    )
    facts.append(
        _make_raw_fact(
            ticker,
            f"Peer review status (PMID {parsed.pmid}): {parsed.peer_review_status.value} "
            "(PubMed/Europe PMC indexing alone is never treated as confirmation of peer review)",
            source, unit="peer_review_status", value=parsed.peer_review_status.value,
            collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
        )
    )
    for nct_id in parsed.nct_ids:
        facts.append(
            _make_raw_fact(
                ticker,
                f"PMID {parsed.pmid} registers a trial registration identifier: {nct_id}",
                source, unit="trial_registration_id", value=nct_id,
                collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
            )
        )
    for section in parsed.abstract_sections:
        if not section.text:
            continue
        label = section.label.upper() if section.label != UNKNOWN else UNKNOWN
        if label in _DESIGN_LABELS:
            facts.append(
                _make_raw_fact(
                    ticker, f"Study design wording (PMID {parsed.pmid}, {section.label}): {section.text}",
                    source, unit="study_design_wording", value=section.label,
                    collector=collector, document_id=document_id, content_kind=content_kind,
                )
            )
        elif label in _RESULT_LABELS:
            facts.append(
                _make_raw_fact(
                    ticker, f"Reported result wording (PMID {parsed.pmid}, {section.label}): {section.text}",
                    source, unit="reported_result_wording", value=section.label,
                    collector=collector, document_id=document_id, content_kind=content_kind,
                )
            )
        for match in _ENDPOINT_SENTENCE_RE.finditer(section.text):
            facts.append(
                _make_raw_fact(
                    ticker, f"Endpoint wording (PMID {parsed.pmid}): {match.group(0).strip()}",
                    source, unit="endpoint_wording",
                    collector=collector, document_id=document_id, content_kind=content_kind,
                )
            )
        for match in _ADVERSE_EVENT_SENTENCE_RE.finditer(section.text):
            facts.append(
                _make_raw_fact(
                    ticker, f"Adverse-event wording (PMID {parsed.pmid}): {match.group(0).strip()}",
                    source, unit="adverse_event_wording",
                    collector=collector, document_id=document_id, content_kind=content_kind,
                )
            )
    if parsed.grants or parsed.coi_statement:
        grant_text = "; ".join(
            f"{g.grant_id} ({g.agency}, {g.country})" for g in parsed.grants if g.grant_id != UNKNOWN
        )
        claim = f"Funding/conflict-of-interest disclosure (PMID {parsed.pmid})"
        if grant_text:
            claim += f": grants={grant_text}"
        if parsed.coi_statement:
            claim += f"; COI statement: {parsed.coi_statement}"
        facts.append(
            _make_raw_fact(
                ticker, claim, source, unit="funding_coi",
                collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
            )
        )
    if parsed.comments_corrections:
        for cc in parsed.comments_corrections:
            facts.append(
                _make_raw_fact(
                    ticker,
                    f"Retraction/correction status (PMID {parsed.pmid}): {cc.ref_type} "
                    f"(related PMID: {cc.related_pmid}, source: {cc.ref_source})",
                    source, unit="retraction_correction_status", value=cc.ref_type,
                    collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
                )
            )
    return facts


def raw_facts_from_europepmc_fulltext(
    ticker: str,
    pmid: str,
    search_result: EuropePmcSearchResult,
    parsed_fulltext: ParsedEuropePmcFullText | None,
    source: Source,
    *,
    collector: str = "literature_europepmc",
    document_id: str | None = None,
) -> list[RawFact]:
    """Open-access status and full-text availability are always their own,
    SEPARATE facts (Phase 3F requirement 3/8's "must clearly separate
    metadata-only, abstract-only, and open-access-full-text" and
    requirement 6's non-OA-never-full-text-acquired boundary).

    Phase 3F.0.1: ``ContentKind.FULL_DOCUMENT`` on the full-text-acquired
    fact below means only "the full text was retrieved" -- it carries no
    implication about peer review, and ``search_result.peer_review_status``
    is surfaced as its own, separate fact so nothing downstream can read
    "full text acquired" as "peer-reviewed and confirmed".
    """
    facts = [
        _make_raw_fact(
            ticker,
            f"Europe PMC open-access status (PMID {pmid}): isOpenAccess="
            f"{search_result.is_open_access}, license={search_result.license or UNKNOWN}",
            source, unit="open_access_status", value=search_result.is_open_access,
            collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
        ),
        _make_raw_fact(
            ticker,
            f"Peer review status (PMID {pmid}, Europe PMC source={search_result.source}): "
            f"{search_result.peer_review_status.value} (full-text acquisition, if any, is a "
            "separate fact and never itself confirms peer review)",
            source, unit="peer_review_status", value=search_result.peer_review_status.value,
            collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
        ),
    ]
    if parsed_fulltext is not None and parsed_fulltext.sections:
        facts.append(
            _make_raw_fact(
                ticker,
                f"Europe PMC open-access full text acquired (PMID {pmid}, "
                f"{len(parsed_fulltext.sections)} section(s)); this means only that the full "
                "text was retrieved, not that it is peer-reviewed, high-quality, or "
                "independently confirmed",
                source, unit="full_text_availability", value=True,
                collector=collector, document_id=document_id, content_kind=ContentKind.FULL_DOCUMENT,
            )
        )
    else:
        facts.append(
            _make_raw_fact(
                ticker, f"Europe PMC full text unavailable (PMID {pmid}): abstract only",
                source, unit="full_text_availability", value=False,
                collector=collector, document_id=document_id, content_kind=ContentKind.METADATA_ONLY,
            )
        )
    return facts


__all__ = [
    "AbstractSection",
    "CommentCorrection",
    "EuropePmcSearchResult",
    "FullTextSection",
    "FullTextXmlShapeError",
    "LiteratureAuthor",
    "LiteratureGrant",
    "LITERATURE_UNIT_PREFIX",
    "ParsedEuropePmcFullText",
    "ParsedPubmedArticle",
    "XmlShapeError",
    "check_europepmc_fulltext_xml_shape",
    "check_pubmed_xml_shape",
    "parse_europepmc_fulltext_xml",
    "parse_europepmc_search_response",
    "parse_pubmed_articleset",
    "raw_facts_from_europepmc_fulltext",
    "raw_facts_from_pubmed_article",
]
