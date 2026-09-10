"""Phase 1: DocumentStore identity, versioning and duplicate-content tracking.

No network, no pipeline wiring -- these exercise the store directly per the
"pure types/catalog/DocumentStore/lineage only" Phase 1 scope. All test data
uses the synthetic "TESTCO" placeholder, never a real issuer.
"""

from __future__ import annotations

from investment_research.collectors.documents import Document
from investment_research.research.document_store import DocumentRole, DocumentStore
from investment_research.schemas.enums import DocumentAuthority


def _doc(doc_id: str, url: str, text: str, authority: DocumentAuthority) -> Document:
    return Document(doc_id=doc_id, url=url, title=doc_id, text=text, authority=authority)


def test_resolve_by_accession_returns_a_list_never_a_single_document():
    store = DocumentStore()
    store.put(
        _doc("d1", "https://sec.gov/x/0001/index.htm", "index page", DocumentAuthority.STATUTORY_FILING),
        document_id="d1",
        accession="0001-26-000123",
        filename="index.htm",
        document_role=DocumentRole.FILING_INDEX,
    )
    store.put(
        _doc("d2", "https://sec.gov/x/0001/ex10.htm", "exhibit body", DocumentAuthority.STATUTORY_FILING),
        document_id="d2",
        accession="0001-26-000123",
        filename="ex10.htm",
        document_role=DocumentRole.EXHIBIT,
    )

    result = store.resolve_by_accession("0001-26-000123")

    assert isinstance(result, list)
    assert {d.document_id for d in result} == {"d1", "d2"}


def test_same_accession_different_filename_stays_separate():
    store = DocumentStore()
    a = store.put(
        _doc("d1", "https://sec.gov/x/0001/primary.htm", "primary text", DocumentAuthority.STATUTORY_FILING),
        document_id="d1",
        accession="0001-26-000123",
        filename="primary.htm",
        document_role=DocumentRole.PRIMARY_DOCUMENT,
    )
    b = store.put(
        _doc("d2", "https://sec.gov/x/0001/ex99.htm", "exhibit text", DocumentAuthority.STATUTORY_FILING),
        document_id="d2",
        accession="0001-26-000123",
        filename="ex99.htm",
        document_role=DocumentRole.EXHIBIT,
    )

    assert a.document_id != b.document_id
    assert store.resolve_by_accession_and_file("0001-26-000123", "primary.htm").document_id == "d1"
    assert store.resolve_by_accession_and_file("0001-26-000123", "ex99.htm").document_id == "d2"


def test_same_accession_index_vs_primary_document_differ_in_role_and_id():
    store = DocumentStore()
    index = store.put(
        _doc("d1", "https://sec.gov/x/0001/", "directory listing", DocumentAuthority.STATUTORY_FILING),
        document_id="d1",
        accession="0001-26-000123",
        filename="",
        document_role=DocumentRole.FILING_INDEX,
    )
    primary = store.put(
        _doc("d2", "https://sec.gov/x/0001/primary.htm", "10-K text", DocumentAuthority.STATUTORY_FILING),
        document_id="d2",
        accession="0001-26-000123",
        filename="primary.htm",
        document_role=DocumentRole.PRIMARY_DOCUMENT,
    )

    assert index.document_id != primary.document_id
    assert index.filename != primary.filename
    assert index.document_role is DocumentRole.FILING_INDEX
    assert primary.document_role is DocumentRole.PRIMARY_DOCUMENT


def test_content_hash_match_with_different_authority_never_auto_merges():
    store = DocumentStore()
    same_text = "The company announced a new financing arrangement."
    store.put(
        _doc("issuer1", "https://ir.testco.example/press/1", same_text, DocumentAuthority.COMPANY_IR),
        document_id="issuer1",
        canonical_url="https://ir.testco.example/press/1",
    )
    store.put(
        _doc("wire1", "https://wire.example/press/1", same_text, DocumentAuthority.INDEPENDENT),
        document_id="wire1",
        canonical_url="https://wire.example/press/1",
    )

    # Two distinct Document identities remain -- no merge happened.
    assert store.get("issuer1") is not None
    assert store.get("wire1") is not None
    assert store.get("issuer1").document_id != store.get("wire1").document_id

    relations = store.duplicate_content_relations()
    assert len(relations) == 1
    relation = relations[0]
    assert set(relation.document_ids) == {"issuer1", "wire1"}
    assert relation.same_authority is False


def test_content_hash_match_with_same_authority_is_flagged_same_authority_true():
    store = DocumentStore()
    same_text = "Identical regulator correspondence text."
    store.put(
        _doc("r1", "https://fda.gov/letters/a", same_text, DocumentAuthority.REGULATOR),
        document_id="r1",
        canonical_url="https://fda.gov/letters/a",
    )
    store.put(
        _doc("r2", "https://fda.gov/letters/a-mirror", same_text, DocumentAuthority.REGULATOR),
        document_id="r2",
        canonical_url="https://fda.gov/letters/a-mirror",
    )

    relations = store.duplicate_content_relations()
    assert len(relations) == 1
    assert relations[0].same_authority is True


def test_same_url_new_content_creates_a_new_version_and_old_stays_retrievable():
    store = DocumentStore()
    url = "https://ir.testco.example/press/latest"
    v1 = store.put(
        _doc("p1", url, "Q1 update: cash position stable.", DocumentAuthority.COMPANY_IR),
        document_id="p1",
        canonical_url=url,
    )
    v2 = store.put(
        _doc("p2", url, "Q1 update, corrected: cash position revised downward.", DocumentAuthority.COMPANY_IR),
        document_id="p2",
        canonical_url=url,
    )

    assert v1.document_id != v2.document_id
    assert v2.version == v1.version + 1
    assert v2.previous_version_id == v1.document_id

    # The old version is still fully retrievable under its own id.
    old = store.get(v1.document_id)
    assert old is not None
    assert old.document.text == "Q1 update: cash position stable."

    history = store.version_history(v2.document_id)
    assert [d.document_id for d in history] == [v1.document_id, v2.document_id]


def test_identical_content_reregistered_at_same_url_is_idempotent():
    store = DocumentStore()
    url = "https://ir.testco.example/press/stable"
    text = "No change in guidance this quarter."
    first = store.put(
        _doc("s1", url, text, DocumentAuthority.COMPANY_IR), document_id="s1", canonical_url=url
    )
    second = store.put(
        _doc("s1-retry", url, text, DocumentAuthority.COMPANY_IR),
        document_id="s1-retry",
        canonical_url=url,
    )

    assert second.document_id == first.document_id
    assert second.version == 1
    assert store.get("s1-retry") is None
