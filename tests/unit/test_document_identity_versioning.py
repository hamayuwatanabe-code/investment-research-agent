"""Phase 3D.1: Document identity/versioning invariants, shared across every
adapter that stores a Document (SEC primary, SEC exhibit, ClinicalTrials) --
never adapter-specific, never re-implemented per source.

No real network call anywhere in this file.
"""

from __future__ import annotations

import pytest

from investment_research.collectors.documents import Document
from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.clinicaltrials_acquisition_adapter import (
    CLINICALTRIALS_ADAPTER_ID,
    ClinicalTrialsStudyAdapter,
    ClinicalTrialsStudyReference,
)
from investment_research.research.document_store import (
    CorruptVersionChainError,
    DocumentIdentityError,
    DocumentRole,
    DocumentStore,
    StoredDocument,
    derive_document_id,
)
from investment_research.research.sec_acquisition_adapters import (
    SecExhibitAdapter,
    SecPrimaryDocumentAdapter,
)
from investment_research.schemas.enums import DocumentAuthority

from . import _clinicaltrials_fixture_support as ctfx
from . import _sec_fixture_support as secfx
from ._clinicaltrials_fixture_support import FakeHttpClient as CtFakeHttpClient
from ._sec_fixture_support import ACCESSION, PRIMARY_DOCUMENT
from ._sec_fixture_support import FakeHttpClient as SecFakeHttpClient
from .test_sec_acquisition_adapters import (
    _exhibit_graph,
    _primary_document_graph,
    _ref,
    _selection,
)


def _doc(url: str, text: str, *, authority: DocumentAuthority = DocumentAuthority.REGISTRY) -> Document:
    return Document(doc_id="placeholder", url=url, title="t", text=text, authority=authority)


# --- derive_document_id: content-dependent, single source of truth ---------
def test_derive_document_id_is_content_dependent():
    same_a = derive_document_id("x", "https://example.test/a", "content-1")
    same_b = derive_document_id("x", "https://example.test/a", "content-1")
    different = derive_document_id("x", "https://example.test/a", "content-2")
    assert same_a == same_b
    assert same_a != different


def test_derive_document_id_is_url_dependent_too():
    a = derive_document_id("x", "https://example.test/a", "same-content")
    b = derive_document_id("x", "https://example.test/b", "same-content")
    assert a != b


# --- generic DocumentStore invariants (source-agnostic) ---------------------
def test_same_url_same_content_dedups_no_new_version():
    store = DocumentStore()
    url = "https://example.test/doc"
    doc_id = derive_document_id("t", url, "hello")
    first = store.put(_doc(url, "hello"), document_id=doc_id, accession="ACC", filename="f")
    second = store.put(_doc(url, "hello"), document_id=doc_id, accession="ACC", filename="f")
    assert first is second
    assert second.version == 1
    assert len(store.resolve_by_canonical_url(url)) == 1


def test_same_url_different_content_creates_v2_with_correct_previous():
    store = DocumentStore()
    url = "https://example.test/doc"
    v1_id = derive_document_id("t", url, "version one")
    v2_id = derive_document_id("t", url, "version two")
    v1 = store.put(_doc(url, "version one"), document_id=v1_id, accession="ACC", filename="f")
    v2 = store.put(_doc(url, "version two"), document_id=v2_id, accession="ACC", filename="f")
    assert v2.version == 2
    assert v2.previous_version_id == v1.document_id
    assert v1.document_id != v2.document_id


def test_three_updates_produce_v1_v2_v3_chain():
    store = DocumentStore()
    url = "https://example.test/doc"
    ids = [derive_document_id("t", url, f"content {i}") for i in range(3)]
    stored = [
        store.put(_doc(url, f"content {i}"), document_id=ids[i], accession="ACC", filename="f")
        for i in range(3)
    ]
    assert [s.version for s in stored] == [1, 2, 3]
    assert stored[1].previous_version_id == stored[0].document_id
    assert stored[2].previous_version_id == stored[1].document_id
    history = store.version_history(stored[2].document_id)
    assert [h.document_id for h in history] == ids


def test_old_version_still_retrievable_by_its_own_id():
    store = DocumentStore()
    url = "https://example.test/doc"
    v1_id, v2_id = derive_document_id("t", url, "one"), derive_document_id("t", url, "two")
    store.put(_doc(url, "one"), document_id=v1_id, accession="ACC", filename="f")
    store.put(_doc(url, "two"), document_id=v2_id, accession="ACC", filename="f")
    fetched = store.get(v1_id)
    assert fetched is not None
    assert fetched.document.text == "one"
    assert fetched.version == 1


def test_canonical_url_index_last_entry_is_the_latest_version():
    store = DocumentStore()
    url = "https://example.test/doc"
    for i in range(3):
        store.put(
            _doc(url, f"content {i}"), document_id=derive_document_id("t", url, f"content {i}"),
            accession="ACC", filename="f",
        )
    versions = store.resolve_by_canonical_url(url)
    assert len(versions) == 3
    assert versions[-1].version == 3
    assert versions[0].version == 1


# --- self-reference / cycle prevention --------------------------------------
def test_put_rejects_a_self_referencing_document_id():
    store = DocumentStore()
    url = "https://example.test/doc"
    v1_id = derive_document_id("t", url, "one")
    store.put(_doc(url, "one"), document_id=v1_id, accession="ACC", filename="f")
    with pytest.raises(DocumentIdentityError):
        # Deliberately reusing v1's OWN id for different content -- exactly
        # the bug that caused an infinite loop before this fix.
        store.put(_doc(url, "different content"), document_id=v1_id, accession="ACC", filename="f")


def test_put_rejects_a_document_id_reused_from_an_earlier_ancestor():
    store = DocumentStore()
    url = "https://example.test/doc"
    v1_id = derive_document_id("t", url, "one")
    v2_id = derive_document_id("t", url, "two")
    store.put(_doc(url, "one"), document_id=v1_id, accession="ACC", filename="f")
    store.put(_doc(url, "two"), document_id=v2_id, accession="ACC", filename="f")
    with pytest.raises(DocumentIdentityError):
        # v1_id is not the IMMEDIATE predecessor of a third put (v2 is), but
        # it IS an ancestor -- still refused, a non-adjacent cycle is a
        # cycle too.
        store.put(_doc(url, "three"), document_id=v1_id, accession="ACC", filename="f")


def test_version_history_raises_on_a_corrupt_cycle_even_if_constructed_directly():
    """If a version chain were ever corrupted by some path OTHER than
    put() (e.g. a bug in a future migration), version_history() must
    report it as broken -- never silently truncate or treat it as a valid,
    complete history."""
    store = DocumentStore()
    doc_a = StoredDocument(
        document_id="doc_a", document=_doc("https://example.test/x", "a"),
        accession="ACC", filename="f", document_role=DocumentRole.UNKNOWN,
        version=2, previous_version_id="doc_b",
    )
    doc_b = StoredDocument(
        document_id="doc_b", document=_doc("https://example.test/x", "b"),
        accession="ACC", filename="f", document_role=DocumentRole.UNKNOWN,
        version=1, previous_version_id="doc_a",  # cycle: a -> b -> a
    )
    store._by_id["doc_a"] = doc_a  # noqa: SLF001 -- deliberately bypassing put() to simulate corruption
    store._by_id["doc_b"] = doc_b  # noqa: SLF001
    with pytest.raises(CorruptVersionChainError):
        store.version_history("doc_a")


# --- different URLs, same content: duplicate tracking, never a merge -------
def test_different_urls_same_content_are_tracked_as_duplicate_never_merged():
    store = DocumentStore()
    id_a = derive_document_id("t", "https://a.test/doc", "identical body")
    id_b = derive_document_id("t", "https://b.test/doc", "identical body")
    a = store.put(_doc("https://a.test/doc", "identical body"), document_id=id_a, accession="ACC_A", filename="f")
    b = store.put(_doc("https://b.test/doc", "identical body"), document_id=id_b, accession="ACC_B", filename="f")
    assert a.document_id != b.document_id  # never merged into one Document
    relations = store.duplicate_content_relations()
    assert any(set(r.document_ids) == {a.document_id, b.document_id} for r in relations)


def test_same_content_different_authority_is_never_merged_and_authority_is_preserved():
    store = DocumentStore()
    id_a = derive_document_id("t", "https://company.test/pr", "identical body")
    id_b = derive_document_id("t", "https://wire.test/pr", "identical body")
    a = store.put(
        _doc("https://company.test/pr", "identical body", authority=DocumentAuthority.COMPANY_IR),
        document_id=id_a, accession="ACC_A", filename="f",
    )
    b = store.put(
        _doc("https://wire.test/pr", "identical body", authority=DocumentAuthority.INDEPENDENT),
        document_id=id_b, accession="ACC_B", filename="f",
    )
    assert a.authority is DocumentAuthority.COMPANY_IR
    assert b.authority is DocumentAuthority.INDEPENDENT  # never overwritten to match a's
    relations = store.duplicate_content_relations()
    relation = next(r for r in relations if set(r.document_ids) == {a.document_id, b.document_id})
    assert relation.same_authority is False


# --- adapter integration: SEC primary/exhibit + ClinicalTrials, real chains -
def test_sec_primary_document_same_url_different_content_creates_v2():
    store = DocumentStore()
    v1_body = secfx.fixture_text("primary_10k_ixbrl.htm")
    v2_body = v1_body.replace("31.5", "45.2")  # a plausible amended cash figure
    assert v1_body != v2_body

    responses_v1 = secfx.default_responses()
    responses_v1[secfx.document_url(PRIMARY_DOCUMENT)] = secfx.ok(v1_body)
    http_v1 = SecFakeHttpClient(responses=responses_v1)
    executor_v1 = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http_v1)}, document_store=store,
    )
    graph = _primary_document_graph("v1")
    executor_v1.run(graph, filing_references={"target_v1": _ref()})
    v1_docs = store.resolve_by_accession(ACCESSION)
    assert len(v1_docs) == 1

    responses_v2 = secfx.default_responses()
    responses_v2[secfx.document_url(PRIMARY_DOCUMENT)] = secfx.ok(v2_body)
    http_v2 = SecFakeHttpClient(responses=responses_v2)
    executor_v2 = AcquisitionExecutor(
        adapters={"sec_primary_document_adapter": SecPrimaryDocumentAdapter(http_v2)}, document_store=store,
    )
    graph2 = _primary_document_graph("v2")
    executor_v2.run(graph2, filing_references={"target_v2": _ref()})

    all_docs = store.resolve_by_accession(ACCESSION)
    assert len(all_docs) == 2
    latest = next(d for d in all_docs if d.version == 2)
    assert latest.previous_version_id == v1_docs[0].document_id
    history = store.version_history(latest.document_id)
    assert len(history) == 2


def test_sec_exhibit_same_url_different_content_creates_v2():
    store = DocumentStore()
    exhibit_filename = "testco-20251231_ex99-1.htm"
    v1_body = secfx.fixture_text("exhibit_99_1_press_release.htm")
    v2_body = v1_body.replace("GBH-101", "GBH-101B")
    assert v1_body != v2_body

    responses_v1 = secfx.default_responses()
    responses_v1[secfx.document_url(exhibit_filename)] = secfx.ok(v1_body)
    http_v1 = SecFakeHttpClient(responses=responses_v1)
    executor_v1 = AcquisitionExecutor(
        adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http_v1)}, document_store=store,
    )
    graph = _exhibit_graph("v1")
    executor_v1.run(graph, exhibit_selectors={"target_v1": _selection()})
    v1_docs = [d for d in store.resolve_by_accession(ACCESSION) if d.document_role is DocumentRole.EXHIBIT]
    assert len(v1_docs) == 1

    responses_v2 = secfx.default_responses()
    responses_v2[secfx.document_url(exhibit_filename)] = secfx.ok(v2_body)
    http_v2 = SecFakeHttpClient(responses=responses_v2)
    executor_v2 = AcquisitionExecutor(
        adapters={"sec_exhibit_enumeration": SecExhibitAdapter(http_v2)}, document_store=store,
    )
    graph2 = _exhibit_graph("v2")
    executor_v2.run(graph2, exhibit_selectors={"target_v2": _selection()})

    all_exhibit_docs = [d for d in store.resolve_by_accession(ACCESSION) if d.document_role is DocumentRole.EXHIBIT]
    assert len(all_exhibit_docs) == 2
    latest = next(d for d in all_exhibit_docs if d.version == 2)
    assert latest.previous_version_id == v1_docs[0].document_id


def test_clinicaltrials_same_url_same_content_dedups():
    store = DocumentStore()
    http = CtFakeHttpClient(responses=ctfx.default_responses())
    executor = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(http)}, document_store=store,
    )
    from .test_clinicaltrials_acquisition_adapter import _one_target_graph

    graph, _ = _one_target_graph("target_dedup_a")
    executor.run(graph, study_references={"target_dedup_a": ClinicalTrialsStudyReference(nct_id=ctfx.NCT_ID)})
    graph2, _ = _one_target_graph("target_dedup_b")
    executor.run(graph2, study_references={"target_dedup_b": ClinicalTrialsStudyReference(nct_id=ctfx.NCT_ID)})

    docs = store.resolve_by_accession(ctfx.NCT_ID)
    assert len(docs) == 1  # identical content -- no new version


def test_clinicaltrials_same_url_different_content_creates_v2():
    store = DocumentStore()
    from .test_clinicaltrials_acquisition_adapter import _one_target_graph

    http_v1 = CtFakeHttpClient(responses={ctfx.study_url(ctfx.NCT_ID): ctfx.ok(ctfx.fixture_text("study_recruiting_interventional.json"))})
    executor_v1 = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(http_v1)}, document_store=store,
    )
    graph1, _ = _one_target_graph("target_v1")
    executor_v1.run(graph1, study_references={"target_v1": ClinicalTrialsStudyReference(nct_id=ctfx.NCT_ID)})
    v1_docs = store.resolve_by_accession(ctfx.NCT_ID)
    assert len(v1_docs) == 1

    http_v2 = CtFakeHttpClient(responses={ctfx.study_url(ctfx.NCT_ID): ctfx.ok(ctfx.fixture_text("study_recruiting_interventional_v2.json"))})
    executor_v2 = AcquisitionExecutor(
        adapters={CLINICALTRIALS_ADAPTER_ID: ClinicalTrialsStudyAdapter(http_v2)}, document_store=store,
    )
    graph2, _ = _one_target_graph("target_v2")
    executor_v2.run(graph2, study_references={"target_v2": ClinicalTrialsStudyReference(nct_id=ctfx.NCT_ID)})

    all_docs = store.resolve_by_accession(ctfx.NCT_ID)
    assert len(all_docs) == 2
    latest = next(d for d in all_docs if d.version == 2)
    assert latest.previous_version_id == v1_docs[0].document_id
