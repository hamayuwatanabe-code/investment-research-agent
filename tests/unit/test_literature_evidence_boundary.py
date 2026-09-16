"""Phase 3F: the BIOMEDICAL_PUBLICATION_ASSERTION EvidenceClass /
BIOMEDICAL_LITERATURE DocumentAuthority boundary -- audited across every
consumer (evidence_integrity.py, escalation.py, fact_collector.py,
validation.py, storage round-trip), proving no path promotes a literature
fact to VERIFIED_FACT or independent_confirmation=True, and that ordinary
(non-literature) SCIENCE-category evidence is completely unaffected.
Mirrors ``tests/unit/test_form4_evidence_boundary.py``'s own structure.
"""

from __future__ import annotations

import pytest

from investment_research.agents.evidence_integrity import EvidenceIntegrityAgent
from investment_research.agents.fact_collector import FactCollectorAgent
from investment_research.collectors.literature import (
    LITERATURE_UNIT_PREFIX,
    raw_facts_from_pubmed_article,
)
from investment_research.orchestrator.anonymize import anonymize_facts
from investment_research.research.escalation import _EVIDENCE_FOR_AUTHORITY, _evidence_for_authority
from investment_research.schemas.agent_io import AgentInput
from investment_research.schemas.enums import (
    DECISION_GRADE_CLASSES,
    ContentKind,
    DocumentAuthority,
    EvidenceClass,
    FactCategory,
    SourceTier,
    VerifiedStatus,
)
from investment_research.schemas.fact import Fact, Source, make_source_id
from investment_research.schemas.validation import SchemaError, validate_fact
from investment_research.storage.db import open_db
from investment_research.storage.repository import Repository
from tests.conftest import TODAY, make_fact

from . import _literature_fixture_support as fx


def _pubmed_article(name: str = "normal_abstract.xml"):
    from investment_research.collectors.literature import parse_pubmed_articleset

    return parse_pubmed_articleset(fx.fixture_text(name))[0]


# --- schemas/enums.py: the new values exist and are correctly excluded -----
def test_peer_reviewed_publication_assertion_excluded_from_decision_grade():
    assert EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION not in DECISION_GRADE_CLASSES
    assert EvidenceClass.VERIFIED_FACT in DECISION_GRADE_CLASSES
    assert EvidenceClass.INDEPENDENT_EVIDENCE in DECISION_GRADE_CLASSES


def test_peer_reviewed_literature_is_distinct_from_independent_and_registry():
    assert DocumentAuthority.BIOMEDICAL_LITERATURE != DocumentAuthority.INDEPENDENT
    assert DocumentAuthority.BIOMEDICAL_LITERATURE != DocumentAuthority.REGISTRY
    assert DocumentAuthority.BIOMEDICAL_LITERATURE != DocumentAuthority.COMPANY_IR


# --- research/escalation.py: complete mapping, correct triple --------------
def test_escalation_evidence_for_authority_covers_every_documentauthority_value():
    for authority in DocumentAuthority:
        assert authority in _EVIDENCE_FOR_AUTHORITY, f"{authority} has no _EVIDENCE_FOR_AUTHORITY entry"
        _evidence_for_authority(authority)  # must not raise


def test_escalation_peer_reviewed_literature_maps_to_the_dedicated_class():
    evidence_class, company_claim, independent_confirmation = _evidence_for_authority(
        DocumentAuthority.BIOMEDICAL_LITERATURE
    )
    assert evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
    assert company_claim is False
    assert independent_confirmation is False


def test_escalation_independent_mapping_unchanged():
    """Requirement: DocumentAuthority.INDEPENDENT's own (pre-existing)
    mapping is completely unaffected by the new literature authority."""
    assert _evidence_for_authority(DocumentAuthority.INDEPENDENT) == (
        EvidenceClass.INDEPENDENT_EVIDENCE, False, True,
    )


# --- agents/fact_collector.py: safe initial placeholder --------------------
def test_fact_collector_initial_classification_of_a_literature_fact_is_never_verified():
    article = _pubmed_article()
    source = Source(
        source_id=make_source_id("https://eutils.ncbi.nlm.nih.gov/x", "t"),
        url="https://eutils.ncbi.nlm.nih.gov/x", title="t", tier=SourceTier.TIER_2,
        content_kind=ContentKind.EXCERPT,
    )
    raw_facts = raw_facts_from_pubmed_article("SAMPB", article, source, document_id="doc_lit_test")
    assert raw_facts
    for raw in raw_facts:
        fact = FactCollectorAgent._to_fact(raw, run_id="test_run", provenance=raw.source.provenance)
        assert fact.evidence_class is not EvidenceClass.VERIFIED_FACT
        assert fact.evidence_class is not EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
        assert fact.company_claim is False


# --- agents/evidence_integrity.py: classification + forced non-confirmation
def _literature_fact(
    claim: str, *, unit: str, source_url: str, event_date: str = "2025-06-01",
    source_authority: DocumentAuthority = DocumentAuthority.BIOMEDICAL_LITERATURE,
) -> Fact:
    return Fact(
        fact_id=f"fact_{abs(hash((claim, source_url, unit)))}",
        ticker="SAMPB",
        category=FactCategory.SCIENCE,
        claim=claim,
        evidence_class=EvidenceClass.UNVERIFIED_CLAIM,
        source_id=make_source_id(source_url, "PubMed"),
        source_url=source_url,
        source_title="PubMed",
        source_tier=SourceTier.TIER_2,
        event_date=event_date,
        company_claim=False,
        unit=f"{LITERATURE_UNIT_PREFIX}{unit}",
        content_kind=ContentKind.EXCERPT,
        source_authority=source_authority,
    )


def test_literature_fact_classified_as_peer_reviewed_publication_assertion():
    fact = _literature_fact(
        "Reported result wording (PMID 90000002): positive fictional result",
        unit="reported_result_wording", source_url="https://eutils.ncbi.nlm.nih.gov/a",
    )
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=(fact,),
    )
    assessed = list(agent.run(agent_input).facts)
    assert len(assessed) == 1
    assert assessed[0].evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
    assert assessed[0].independent_confirmation is False
    assert assessed[0].verified_status is VerifiedStatus.NOT_VERIFIED
    assert not assessed[0].is_decision_grade


def test_two_literature_facts_with_identical_claim_text_never_confirm_each_other():
    """Mirrors Form4's own precedent (Phase 3E.4 requirement 10): peer
    review alone must never set independent_confirmation=True, including
    two papers' coincidentally similar wording 'confirming' each other."""
    claim = "Reported result wording (PMID X): identical fictional wording"
    fact_a = _literature_fact(claim, unit="reported_result_wording", source_url="https://eutils.ncbi.nlm.nih.gov/a")
    fact_b = _literature_fact(claim, unit="reported_result_wording", source_url="https://eutils.ncbi.nlm.nih.gov/b")
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=(fact_a, fact_b),
    )
    assessed = list(agent.run(agent_input).facts)
    assert len(assessed) == 2
    for fact in assessed:
        assert fact.independent_confirmation is False
        assert fact.corroborating_source_ids == ()
        assert fact.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION


def test_non_literature_science_fact_classification_is_unaffected():
    """The ``unit`` discriminator scopes the override to literature facts
    only -- an ordinary SCIENCE-category fact from a DIFFERENT producer
    (e.g. ClinicalTrials.gov, unit not prefixed 'literature_') keeps its
    pre-existing classification exactly."""
    fact = Fact(
        fact_id="fact_ct_science", ticker="SAMPB", category=FactCategory.SCIENCE,
        claim="Trial status: recruiting", evidence_class=EvidenceClass.UNVERIFIED_CLAIM,
        source_id=make_source_id("https://clinicaltrials.gov/x", "CT"),
        source_url="https://clinicaltrials.gov/x", source_title="CT", source_tier=SourceTier.TIER_2,
        event_date="2025-06-01", company_claim=False, unit="trial_status",
    )
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=(fact,),
    )
    assessed = list(agent.run(agent_input).facts)[0]
    assert assessed.evidence_class is not EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
    assert assessed.evidence_class is EvidenceClass.INDEPENDENT_EVIDENCE


def test_classification_keyed_on_authority_survives_a_renamed_unit():
    """Phase 3F.0.1 requirement 3: unit naming is auxiliary only -- a
    literature fact classifies correctly even under a unit name the
    collector has never used, as long as source_authority is set."""
    fact = _literature_fact(
        "Reported result wording: some finding", unit="some_brand_new_unit_name",
        source_url="https://eutils.ncbi.nlm.nih.gov/renamed",
    )
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=(fact,),
    )
    assessed = list(agent.run(agent_input).facts)[0]
    assert assessed.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION


def test_literature_shaped_unit_name_from_a_different_authority_never_auto_classifies():
    """Phase 3F.0.1 requirement 3: a fact carrying a 'literature_'-prefixed
    unit but a DIFFERENT (or UNKNOWN) source_authority must never be
    classified as BIOMEDICAL_PUBLICATION_ASSERTION -- unit name alone
    proves nothing."""
    fact = _literature_fact(
        "Reported result wording: some finding", unit="reported_result_wording",
        source_url="https://example.test/not-actually-literature",
        source_authority=DocumentAuthority.UNKNOWN,
    )
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=(fact,),
    )
    assessed = list(agent.run(agent_input).facts)[0]
    assert assessed.evidence_class is not EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION


def test_open_access_full_text_derived_fact_never_gets_independent_confirmation():
    """Phase 3F.0.1 requirement 7/10: even a fact drawn from an actually-
    retrieved Europe PMC OPEN-ACCESS full text (ContentKind.FULL_DOCUMENT,
    so it clears the ``is_search_derived`` gate and reaches the authority
    check) still ends up independent_confirmation=False -- full-text
    acquisition is never independent confirmation."""
    fact = _literature_fact(
        "Europe PMC open-access full text acquired (PMID 90000008, 3 section(s)); this means only "
        "that the full text was retrieved, not that it is peer-reviewed, high-quality, or "
        "independently confirmed",
        unit="full_text_availability", source_url="https://www.ebi.ac.uk/europepmc/x",
    )
    fact = Fact(**{**fact.__dict__, "content_kind": ContentKind.FULL_DOCUMENT})
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=(fact,),
    )
    assessed = list(agent.run(agent_input).facts)[0]
    assert assessed.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION
    assert assessed.independent_confirmation is False
    assert not assessed.is_decision_grade


def test_sponsor_funding_alone_never_infers_company_claim_through_evidence_integrity():
    from investment_research.collectors.literature import parse_pubmed_articleset

    article = parse_pubmed_articleset(fx.fixture_text("sponsor_funded.xml"))[0]
    source = Source(
        source_id=make_source_id("https://eutils.ncbi.nlm.nih.gov/x", "t"),
        url="https://eutils.ncbi.nlm.nih.gov/x", title="t", tier=SourceTier.TIER_2,
        content_kind=ContentKind.EXCERPT,
    )
    raw_facts = raw_facts_from_pubmed_article("SAMPB", article, source, document_id="doc_sponsor")
    facts = [FactCollectorAgent._to_fact(r, run_id="r", provenance=r.source.provenance) for r in raw_facts]
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=tuple(facts),
    )
    assessed = list(agent.run(agent_input).facts)
    assert assessed
    for fact in assessed:
        assert fact.company_claim is False
        assert fact.independent_confirmation is False


def test_non_science_company_claim_confirmation_path_is_unaffected():
    """Requirement: an ordinary company claim, confirmed by an independent
    Tier-1 source, still upgrades to VERIFIED_FACT exactly as before -- the
    Phase 3F override is scoped to literature-prefixed SCIENCE facts only."""
    claim = "The company completed a $50 million private placement"
    company = make_fact(claim, category=FactCategory.CAPITAL_STRUCTURE, company_claim=True, tier=SourceTier.TIER_2)
    confirming = make_fact(
        claim, category=FactCategory.CAPITAL_STRUCTURE, company_claim=False, tier=SourceTier.TIER_1,
        url="https://www.sec.gov/other-filing",
    )
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="TEST", company_name="Test Co",
        facts=(company, confirming),
    )
    assessed = {f.source_url: f for f in agent.run(agent_input).facts}
    assert assessed[company.source_url].independent_confirmation is True
    assert assessed[company.source_url].evidence_class is EvidenceClass.VERIFIED_FACT


# --- schemas/validation.py: defense-in-depth schema guard -------------------
def test_validate_fact_rejects_peer_reviewed_assertion_with_independent_confirmation_true():
    fact = make_fact(
        "Reported result wording", category=FactCategory.SCIENCE,
        evidence_class=EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION,
        company_claim=False, independent_confirmation=True, tier=SourceTier.TIER_2,
    )
    with pytest.raises(SchemaError, match="independent_confirmation=True"):
        validate_fact(fact)


def test_validate_fact_accepts_peer_reviewed_assertion_with_independent_confirmation_false():
    fact = make_fact(
        "Reported result wording", category=FactCategory.SCIENCE,
        evidence_class=EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION,
        company_claim=False, independent_confirmation=False, tier=SourceTier.TIER_2,
        verified=VerifiedStatus.NOT_VERIFIED,
    )
    validate_fact(fact)  # must not raise


# --- storage round-trip: the new enum value persists and reads back --------
def test_biomedical_publication_assertion_round_trips_through_sqlite():
    conn = open_db(":memory:")
    repo = Repository(conn)
    repo.upsert_company("SAMPB", "Sample Biotech Holdings, Inc.")
    fact = make_fact(
        "Reported result wording", ticker="SAMPB", category=FactCategory.SCIENCE,
        evidence_class=EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION,
        company_claim=False, independent_confirmation=False, tier=SourceTier.TIER_2,
        verified=VerifiedStatus.NOT_VERIFIED,
    )
    repo.save_source(
        Source(source_id=fact.source_id, url=fact.source_url, title=fact.source_title, tier=fact.source_tier)
    )
    repo.save_fact(fact)

    row = repo.latest_fact_row(fact.fact_id)
    assert row is not None
    assert row["evidence_class"] == "BIOMEDICAL_PUBLICATION_ASSERTION"
    assert EvidenceClass(row["evidence_class"]) is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION


def test_source_authority_round_trips_through_sqlite():
    """Phase 3F.0.2 requirement 3: a literature fact's source_authority
    persists and reads back exactly; other fact producers (which never set
    it) keep reading back the pre-existing UNKNOWN default unaffected --
    see tests/unit/test_db_migration.py for the additive-column migration
    (NULL old row -> 'UNKNOWN') itself."""
    conn = open_db(":memory:")
    repo = Repository(conn)
    repo.upsert_company("SAMPB", "Sample Biotech Holdings, Inc.")

    literature_fact = _literature_fact(
        "Reported result wording", unit="reported_result_wording",
        source_url="https://eutils.ncbi.nlm.nih.gov/x",
    )
    repo.save_source(
        Source(
            source_id=literature_fact.source_id, url=literature_fact.source_url,
            title=literature_fact.source_title, tier=literature_fact.source_tier,
        )
    )
    repo.save_fact(literature_fact)
    row = repo.latest_fact_row(literature_fact.fact_id)
    assert row is not None
    assert row["source_authority"] == "BIOMEDICAL_LITERATURE"
    assert DocumentAuthority(row["source_authority"]) is DocumentAuthority.BIOMEDICAL_LITERATURE

    other_fact = make_fact("An ordinary SEC-sourced claim", ticker="SAMPB", category=FactCategory.CAPITAL_STRUCTURE)
    assert other_fact.source_authority is DocumentAuthority.UNKNOWN
    repo.save_source(
        Source(source_id=other_fact.source_id, url=other_fact.source_url, title=other_fact.source_title, tier=other_fact.source_tier)
    )
    repo.save_fact(other_fact)
    other_row = repo.latest_fact_row(other_fact.fact_id)
    assert other_row is not None
    assert other_row["source_authority"] == "UNKNOWN"


# --- retraction: never decision-grade regardless of confirmation status ----
def test_retracted_article_facts_never_decision_grade():
    from investment_research.collectors.literature import parse_pubmed_articleset

    article = parse_pubmed_articleset(fx.fixture_text("retracted.xml"))[0]
    source = Source(
        source_id=make_source_id("https://eutils.ncbi.nlm.nih.gov/x", "t"),
        url="https://eutils.ncbi.nlm.nih.gov/x", title="t", tier=SourceTier.TIER_2,
        content_kind=ContentKind.EXCERPT,
    )
    raw_facts = raw_facts_from_pubmed_article("SAMPB", article, source, document_id="doc_retracted")
    facts = [FactCollectorAgent._to_fact(r, run_id="r", provenance=r.source.provenance) for r in raw_facts]
    agent = EvidenceIntegrityAgent(today=TODAY)
    agent_input = AgentInput(
        agent_id="evidence_integrity", run_id="r", ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
        facts=tuple(facts),
    )
    assessed = list(agent.run(agent_input).facts)
    assert assessed
    for fact in assessed:
        assert not fact.is_decision_grade


# --- Phase 3F.0.2 requirement 3: Blind Judge never sees literature provenance
def test_blind_judge_input_never_reveals_literature_source_authority_or_identity():
    """A literature fact converted to a Blind Judge input (anonymize_facts)
    must never carry source_authority, document_id, a real source URL, or
    collector/adapter identity -- scanning the FULL serialized payload for
    every marker string a leak could take."""
    article = _pubmed_article("structured_abstract.xml")
    source = Source(
        source_id=make_source_id("https://eutils.ncbi.nlm.nih.gov/x", "t"),
        url="https://eutils.ncbi.nlm.nih.gov/x", title="t", tier=SourceTier.TIER_2,
        content_kind=ContentKind.EXCERPT,
    )
    raw_facts = raw_facts_from_pubmed_article("SAMPB", article, source, document_id="doc_lit_blind_test")
    facts = [FactCollectorAgent._to_fact(r, run_id="r", provenance=r.source.provenance) for r in raw_facts]
    agent = EvidenceIntegrityAgent(today=TODAY)
    assessed = list(
        agent.run(
            AgentInput(
                agent_id="evidence_integrity", run_id="r", ticker="SAMPB",
                company_name="Sample Biotech Holdings, Inc.", facts=tuple(facts),
            )
        ).facts
    )
    assert assessed
    assert any(f.evidence_class is EvidenceClass.BIOMEDICAL_PUBLICATION_ASSERTION for f in assessed)
    assert any(f.source_authority is DocumentAuthority.BIOMEDICAL_LITERATURE for f in assessed)

    blind_facts, _ref_map = anonymize_facts(
        assessed, ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
    )
    assert blind_facts

    # Field NAMES (source_authority=, document_id=) always appear in a
    # dataclass repr regardless of value -- these markers check the VALUES
    # that would actually identify the producer/document if leaked.
    forbidden_markers = (
        "BIOMEDICAL_LITERATURE",
        "doc_lit_blind_test",
        "eutils.ncbi.nlm.nih.gov",
        "literature_pubmed",
        "collected_by=literature",
    )
    for fact in blind_facts:
        assert fact.source_authority is DocumentAuthority.UNKNOWN
        assert fact.document_id is None
        # The full dataclass repr is the strongest available proxy for "any
        # serialized payload a Blind Judge prompt-builder might construct
        # from this object" -- every field, not just the ones this test
        # happens to name.
        serialized = repr(fact)
        for marker in forbidden_markers:
            assert marker not in serialized, f"{marker!r} leaked into blind fact: {serialized[:400]}"


def test_blind_judge_notes_strip_collector_identity_but_keep_other_notes_content():
    """The collected_by=<collector> fragment is redacted, but the rest of
    notes (when present) is otherwise preserved/redacted only for identity
    markers, not wholesale deleted."""
    fact = _literature_fact(
        "Reported result wording", unit="reported_result_wording",
        source_url="https://eutils.ncbi.nlm.nih.gov/y",
    )
    fact = Fact(**{**fact.__dict__, "notes": "collected_by=literature_pubmed; some other diagnostic note"})
    blind_facts, _ref_map = anonymize_facts(
        [fact], ticker="SAMPB", company_name="Sample Biotech Holdings, Inc.",
    )
    assert len(blind_facts) == 1
    assert "literature_pubmed" not in blind_facts[0].notes
    assert "collected_by=" not in blind_facts[0].notes or "[REDACTED]" in blind_facts[0].notes
    assert "some other diagnostic note" in blind_facts[0].notes


def test_non_literature_fact_anonymization_is_unaffected_by_source_authority_field():
    """A pre-existing (non-literature) fact producer, whose source_authority
    was already UNKNOWN, sees no behavior change from this fix -- the field
    is simply confirmed UNKNOWN before and after anonymization."""
    fact = make_fact("The company completed a private placement", category=FactCategory.CAPITAL_STRUCTURE)
    assert fact.source_authority is DocumentAuthority.UNKNOWN
    blind_facts, _ref_map = anonymize_facts(
        [fact], ticker="TEST", company_name="Test Co",
    )
    assert blind_facts[0].source_authority is DocumentAuthority.UNKNOWN
