"""Phase 3F: pure-function tests for ``collectors/literature.py`` -- PubMed
XML / Europe PMC XML/JSON parsing and RawFact generation, against the
real-format fixtures in ``tests/fixtures/literature_real_format/`` (see
that directory's MANIFEST.md).
"""

from __future__ import annotations

from investment_research.collectors.literature import (
    LITERATURE_UNIT_PREFIX,
    AbstractSection,
    EuropePmcSearchResult,
    FullTextXmlShapeError,
    ParsedPubmedArticle,
    XmlShapeError,
    check_europepmc_fulltext_xml_shape,
    check_pubmed_xml_shape,
    parse_europepmc_fulltext_xml,
    parse_europepmc_search_response,
    parse_pubmed_articleset,
    raw_facts_from_europepmc_fulltext,
    raw_facts_from_pubmed_article,
)
from investment_research.schemas.enums import (
    UNKNOWN,
    ContentKind,
    DocumentAuthority,
    FactCategory,
    PeerReviewStatus,
    PublicationStage,
    SourceTier,
)
from investment_research.schemas.fact import Source

from . import _literature_fixture_support as fx


def _parse_one(name: str) -> ParsedPubmedArticle:
    articles = parse_pubmed_articleset(fx.fixture_text(name))
    assert articles is not None and len(articles) == 1
    return articles[0]


def _source(url: str = "https://eutils.ncbi.nlm.nih.gov/x") -> Source:
    return Source(source_id="src_test", url=url, title="t", tier=SourceTier.TIER_2)


# --- PubMed XML shape gate ----------------------------------------------------
def test_check_pubmed_xml_shape_accepts_normal_article():
    error, _reason = check_pubmed_xml_shape(fx.fixture_text("normal_abstract.xml"))
    assert error is None


def test_check_pubmed_xml_shape_rejects_malformed():
    error, _reason = check_pubmed_xml_shape(fx.fixture_text("malformed.xml"))
    assert error is XmlShapeError.MALFORMED


def test_check_pubmed_xml_shape_rejects_html_error_page():
    error, _reason = check_pubmed_xml_shape(fx.fixture_text("html_error_page.htm"))
    assert error is XmlShapeError.HTML_ERROR_PAGE


def test_check_pubmed_xml_shape_rejects_empty_body():
    error, _reason = check_pubmed_xml_shape("")
    assert error is XmlShapeError.EMPTY


def test_parse_pubmed_articleset_returns_none_for_unusable_body():
    assert parse_pubmed_articleset(fx.fixture_text("malformed.xml")) is None
    assert parse_pubmed_articleset(fx.fixture_text("html_error_page.htm")) is None
    assert parse_pubmed_articleset("") is None


# --- field extraction -----------------------------------------------------
def test_normal_abstract_fields():
    article = _parse_one("normal_abstract.xml")
    assert article.pmid == "90000001"
    assert "Sample Compound" in article.article_title
    assert article.journal_title == "Journal of Fictional Oncology"
    assert article.publication_date == "2025-Jun"
    assert article.electronic_publication_date == "2025-05-20"
    assert article.language == "eng"
    assert "Journal Article" in article.publication_types
    assert article.has_abstract is True
    assert article.is_retracted is False
    assert article.is_correction_or_erratum is False


def test_structured_abstract_sections_and_endpoint_extraction():
    article = _parse_one("structured_abstract.xml")
    labels = [s.label for s in article.abstract_sections]
    assert labels == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    units = {f.unit for f in facts}
    assert f"{LITERATURE_UNIT_PREFIX}study_design_wording" in units
    assert f"{LITERATURE_UNIT_PREFIX}reported_result_wording" in units
    assert f"{LITERATURE_UNIT_PREFIX}endpoint_wording" in units


def test_no_abstract_article_has_no_abstract_and_is_metadata_only():
    article = _parse_one("no_abstract.xml")
    assert article.has_abstract is False
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    assert all(f.content_kind is not ContentKind.FULL_DOCUMENT for f in facts)
    assert any(f.content_kind is ContentKind.METADATA_ONLY for f in facts)


def test_multiple_authors_and_collective_name():
    article = _parse_one("multiple_authors.xml")
    assert len(article.authors) == 3
    named = [a for a in article.authors if a.last_name]
    assert any(a.last_name == "Beta" and len(a.affiliations) == 2 for a in named)
    collective = [a for a in article.authors if a.collective_name]
    assert collective and collective[0].collective_name == "Fictional Study Group"


def test_sponsor_funded_grants_extracted_but_company_claim_stays_false():
    article = _parse_one("sponsor_funded.xml")
    assert article.grants
    assert article.grants[0].agency == "Sample Biotech Holdings, Inc."
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    assert facts  # non-empty
    # Requirement 6: author/funding affiliation NEVER infers company_claim.
    assert all(f.company_claim is False for f in facts)
    funding_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}funding_coi"]
    assert funding_facts and "SAMPLE-GRANT-0001" in funding_facts[0].claim


def test_conflict_of_interest_statement_extracted():
    article = _parse_one("conflict_of_interest.xml")
    assert "paid consultant" in article.coi_statement
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    funding_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}funding_coi"]
    assert funding_facts and "paid consultant" in funding_facts[0].claim


def test_nct_id_extracted_only_from_clinicaltrials_databank():
    article = _parse_one("nct_id_present.xml")
    assert article.nct_ids == ("NCT09990001",)
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    trial_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}trial_registration_id"]
    assert len(trial_facts) == 1
    assert trial_facts[0].value == "NCT09990001"


def test_ids_complete_extracts_pmid_pmcid_doi():
    article = _parse_one("ids_complete.xml")
    assert article.pmid == "90000008"
    assert article.pmcid == "PMC9990008"
    assert article.doi == "10.9999/fict.2025.00008"


def test_correction_erratum_retraction_relations():
    correction = _parse_one("correction.xml")
    assert correction.is_correction_or_erratum is True
    assert correction.is_retracted is False
    assert correction.comments_corrections[0].ref_type == "CorrectionIn"
    assert correction.comments_corrections[0].related_pmid == "90000019"

    erratum = _parse_one("erratum.xml")
    assert erratum.is_correction_or_erratum is True
    assert erratum.comments_corrections[0].ref_type == "ErratumIn"

    retracted = _parse_one("retracted.xml")
    assert retracted.is_retracted is True
    assert retracted.is_correction_or_erratum is False
    facts = raw_facts_from_pubmed_article("TEST", retracted, _source())
    # A retracted article's facts are still generated -- retraction never
    # silently deletes or hides anything (Phase 3F requirement 6).
    assert facts
    status_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}retraction_correction_status"]
    assert status_facts and status_facts[0].value == "RetractionIn"


def test_batch_articleset_parses_both_articles():
    articles = parse_pubmed_articleset(fx.fixture_text("batch_articleset.xml"))
    assert articles is not None
    assert {a.pmid for a in articles} == {"90000015", "90000016"}


# --- RawFact generation: evidence-boundary invariants ------------------------
def test_every_raw_fact_is_science_category_with_literature_unit_prefix():
    article = _parse_one("structured_abstract.xml")
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    assert facts
    for f in facts:
        assert f.category is FactCategory.SCIENCE
        assert f.unit.startswith(LITERATURE_UNIT_PREFIX)
        assert f.company_claim is False


def test_publication_exists_title_date_type_facts_present():
    article = _parse_one("normal_abstract.xml")
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    units = {f.unit for f in facts}
    for suffix in ("publication_exists", "article_title", "publication_date", "publication_type"):
        assert f"{LITERATURE_UNIT_PREFIX}{suffix}" in units


def test_adverse_event_wording_extracted_without_efficacy_conflation():
    article = ParsedPubmedArticle(
        pmid="90000050", pmcid=UNKNOWN, doi=UNKNOWN, article_title="Fictional AE study",
        abstract_sections=(
            AbstractSection(
                label="RESULTS",
                text="Three patients experienced a serious adverse event during the fictional study.",
            ),
        ),
        journal_title=UNKNOWN, journal_iso_abbreviation=UNKNOWN, publication_date=UNKNOWN,
        electronic_publication_date=UNKNOWN, publication_types=(), authors=(), language=UNKNOWN,
        keywords=(), mesh_terms=(), grants=(), coi_statement="", nct_ids=(),
        comments_corrections=(), article_status=UNKNOWN,
    )
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    ae_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}adverse_event_wording"]
    assert ae_facts
    assert "serious adverse event" in ae_facts[0].claim
    # The claim is kept as wording only -- never converted into an efficacy
    # or safety conclusion (no fact this function ever produces asserts
    # "confirmed"/"proven"/"established").
    assert "confirmed" not in ae_facts[0].claim.lower()
    assert "proven" not in ae_facts[0].claim.lower()


def test_reported_result_wording_never_says_efficacy_confirmed():
    """The paper's OWN reported-result wording is carried verbatim -- this
    function itself never appends any efficacy/safety confirmation
    language (Phase 3F requirement 6/7)."""
    article = _parse_one("structured_abstract.xml")
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    result_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}reported_result_wording"]
    assert result_facts
    for f in result_facts:
        assert "efficacy confirmed" not in f.claim.lower()
        assert "safety confirmed" not in f.claim.lower()


# --- Europe PMC ----------------------------------------------------------------
def test_parse_europepmc_search_response_open_access():
    results = parse_europepmc_search_response(fx.fixture_text("europepmc_search_oa.json"))
    assert results is not None and len(results) == 1
    r = results[0]
    assert r.pmid == "90000008"
    assert r.pmcid == "PMC9990008"
    assert r.is_open_access is True
    assert r.in_epmc is True
    assert r.license == "cc by"


def test_parse_europepmc_search_response_non_open_access():
    results = parse_europepmc_search_response(fx.fixture_text("europepmc_search_non_oa.json"))
    assert results is not None and len(results) == 1
    assert results[0].is_open_access is False
    assert results[0].in_epmc is False


def test_parse_europepmc_search_response_malformed_returns_none():
    assert parse_europepmc_search_response("not json") is None
    assert parse_europepmc_search_response('{"unexpected": true}') is None


def test_europepmc_fulltext_sections_distinct_from_abstract():
    parsed = parse_europepmc_fulltext_xml(fx.fixture_text("europepmc_fulltext_oa.xml"))
    assert parsed is not None
    titles = [s.title for s in parsed.sections]
    assert titles == ["Introduction", "Methods", "Results"]
    assert "fictional full-text introduction content" in parsed.full_text.lower()


def test_check_europepmc_fulltext_xml_shape_rejects_wrong_root():
    error, _reason = check_europepmc_fulltext_xml_shape(fx.fixture_text("structured_abstract.xml"))
    assert error is FullTextXmlShapeError.NOT_ARTICLE


def test_check_europepmc_fulltext_xml_shape_rejects_html_error_page():
    error, _reason = check_europepmc_fulltext_xml_shape(fx.fixture_text("html_error_page.htm"))
    assert error is FullTextXmlShapeError.HTML_ERROR_PAGE


def test_raw_facts_from_europepmc_fulltext_open_access_vs_not():
    search_oa = EuropePmcSearchResult(
        pmid="90000008", pmcid="PMC9990008", doi="10.9999/fict.2025.00008",
        title="t", is_open_access=True, in_epmc=True, license="cc by",
        journal_title="j", pub_year="2025",
    )
    parsed_fulltext = parse_europepmc_fulltext_xml(fx.fixture_text("europepmc_fulltext_oa.xml"))
    facts_oa = raw_facts_from_europepmc_fulltext("TEST", "90000008", search_oa, parsed_fulltext, _source())
    availability_oa = [f for f in facts_oa if f.unit == f"{LITERATURE_UNIT_PREFIX}full_text_availability"]
    assert availability_oa and availability_oa[0].value is True
    assert availability_oa[0].content_kind is ContentKind.FULL_DOCUMENT

    search_non_oa = EuropePmcSearchResult(
        pmid="90000001", pmcid=UNKNOWN, doi=UNKNOWN, title="t", is_open_access=False,
        in_epmc=False, license="", journal_title="j", pub_year="2025",
    )
    facts_non_oa = raw_facts_from_europepmc_fulltext("TEST", "90000001", search_non_oa, None, _source())
    availability_non_oa = [f for f in facts_non_oa if f.unit == f"{LITERATURE_UNIT_PREFIX}full_text_availability"]
    assert availability_non_oa and availability_non_oa[0].value is False
    assert availability_non_oa[0].content_kind is not ContentKind.FULL_DOCUMENT
    assert all(f.company_claim is False for f in facts_non_oa)


# --- Phase 3F.0.1: publication_stage / peer_review_status --------------------
def test_normal_journal_article_peer_review_status_is_unknown_never_confirmed():
    """Requirement 7: a PubMed 'Journal Article' publication type alone
    must never set peer_review_status=CONFIRMED."""
    article = _parse_one("normal_abstract.xml")
    assert article.publication_stage is PublicationStage.JOURNAL_ARTICLE
    assert article.peer_review_status is PeerReviewStatus.UNKNOWN
    assert article.peer_review_status is not PeerReviewStatus.CONFIRMED


def test_pubmed_preprint_is_not_peer_reviewed():
    article = _parse_one("preprint.xml")
    assert article.publication_stage is PublicationStage.PREPRINT
    assert article.peer_review_status is PeerReviewStatus.NOT_PEER_REVIEWED
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    status_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}peer_review_status"]
    assert status_facts and status_facts[0].value == "NOT_PEER_REVIEWED"


def test_editorial_and_letter_are_not_journal_article_stage():
    """Requirement 7: an editorial/letter must never be treated as clinical
    efficacy evidence -- its stage is distinct from JOURNAL_ARTICLE, and
    its peer_review_status is never CONFIRMED."""
    editorial = _parse_one("editorial.xml")
    assert editorial.publication_stage is PublicationStage.EDITORIAL
    assert editorial.publication_stage is not PublicationStage.JOURNAL_ARTICLE
    assert editorial.peer_review_status is not PeerReviewStatus.CONFIRMED

    letter = _parse_one("letter.xml")
    assert letter.publication_stage is PublicationStage.LETTER
    assert letter.publication_stage is not PublicationStage.JOURNAL_ARTICLE
    assert letter.peer_review_status is not PeerReviewStatus.CONFIRMED


def test_online_book_chapter_is_never_treated_as_peer_reviewed():
    book = _parse_one("online_book_chapter.xml")
    assert book.publication_stage is PublicationStage.BOOK_OR_CHAPTER
    assert book.publication_stage is not PublicationStage.JOURNAL_ARTICLE
    assert book.peer_review_status is not PeerReviewStatus.CONFIRMED


def test_publication_stage_never_confirmed_by_this_module_for_any_fixture():
    """No fixture in this real-format set, and no code path in this module,
    ever produces PeerReviewStatus.CONFIRMED (requirement 2's central
    correction)."""
    for name in (
        "normal_abstract.xml", "structured_abstract.xml", "no_abstract.xml",
        "multiple_authors.xml", "sponsor_funded.xml", "conflict_of_interest.xml",
        "nct_id_present.xml", "ids_complete.xml", "correction.xml", "erratum.xml",
        "retracted.xml", "preprint.xml", "editorial.xml", "letter.xml",
        "online_book_chapter.xml",
    ):
        article = _parse_one(name)
        assert article.peer_review_status is not PeerReviewStatus.CONFIRMED


def test_europepmc_preprint_search_result_is_not_peer_reviewed():
    results = parse_europepmc_search_response(fx.fixture_text("europepmc_search_preprint.json"))
    assert results is not None and len(results) == 1
    result = results[0]
    assert result.source == "PPR"
    assert result.publication_stage is PublicationStage.PREPRINT
    assert result.peer_review_status is PeerReviewStatus.NOT_PEER_REVIEWED
    facts = raw_facts_from_europepmc_fulltext("TEST", "90000022", result, None, _source())
    status_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}peer_review_status"]
    assert status_facts and status_facts[0].value == "NOT_PEER_REVIEWED"


def test_europepmc_medline_source_peer_review_status_stays_unknown():
    results = parse_europepmc_search_response(fx.fixture_text("europepmc_search_oa.json"))
    assert results is not None
    # MEDLINE-sourced ("MED") is not a preprint-server code, but that is
    # never read as CONFIRMED either -- MEDLINE indexes editorials, letters
    # and preprints too, so UNKNOWN is the only honest default.
    assert results[0].source == "MED"
    assert results[0].peer_review_status is PeerReviewStatus.UNKNOWN
    assert results[0].peer_review_status is not PeerReviewStatus.CONFIRMED


def test_open_access_full_text_acquisition_never_sets_independent_confirmation():
    """Requirement 7: OA full-text acquisition alone never implies
    independent_confirmation -- that is enforced downstream by
    evidence_integrity.py/escalation.py (see
    test_literature_evidence_boundary.py), but the RawFact itself must
    also never claim it in its own wording."""
    search_oa = EuropePmcSearchResult(
        pmid="90000008", pmcid="PMC9990008", doi="10.9999/fict.2025.00008",
        title="t", is_open_access=True, in_epmc=True, license="cc by",
        journal_title="j", pub_year="2025",
    )
    parsed_fulltext = parse_europepmc_fulltext_xml(fx.fixture_text("europepmc_fulltext_oa.xml"))
    facts = raw_facts_from_europepmc_fulltext("TEST", "90000008", search_oa, parsed_fulltext, _source())
    fulltext_facts = [f for f in facts if f.unit == f"{LITERATURE_UNIT_PREFIX}full_text_availability"]
    assert fulltext_facts
    claim_lower = fulltext_facts[0].claim.lower()
    assert "peer-reviewed" not in claim_lower or "not" in claim_lower or "never" in claim_lower
    assert "independently confirmed" not in claim_lower or "not" in claim_lower or "never" in claim_lower


# --- Phase 3F.0.1: source_authority is set on every literature RawFact -----
def test_every_raw_fact_carries_biomedical_literature_authority():
    article = _parse_one("structured_abstract.xml")
    facts = raw_facts_from_pubmed_article("TEST", article, _source())
    assert facts
    for f in facts:
        assert f.source_authority is DocumentAuthority.BIOMEDICAL_LITERATURE

    search_oa = EuropePmcSearchResult(
        pmid="90000008", pmcid="PMC9990008", doi="10.9999/fict.2025.00008",
        title="t", is_open_access=True, in_epmc=True, license="cc by",
        journal_title="j", pub_year="2025",
    )
    epmc_facts = raw_facts_from_europepmc_fulltext("TEST", "90000008", search_oa, None, _source())
    for f in epmc_facts:
        assert f.source_authority is DocumentAuthority.BIOMEDICAL_LITERATURE
