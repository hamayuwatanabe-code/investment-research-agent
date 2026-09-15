"""Phase 3E.2.2: ownership-filing ``primaryDocument`` XSL-display-path
compatibility.

A real SEC Live Smoke run against issuer CIK 1721484 (NOT reproduced
here -- no real CIK, person, or accession is hard-coded anywhere in this
file) found that SEC's own submissions metadata can report a Form 4/4-A's
``primaryDocument`` as an XSLT display path
(``xslF345X06/marketforms-73885.xml``), which the general
``sec_acquisition_adapters._validate_filename`` correctly refuses (it
contains a ``/``) -- causing a real, valid Form 4 candidate to fail with
``LOCATE_FAILED`` even though the underlying raw XML is genuinely
reachable at the accession root.

This file covers, entirely offline:

1. ``collectors.form4.normalize_ownership_primary_document()`` in
   isolation -- every required acceptance/rejection case, pure, no HTTP.
2. ``Form4Adapter``'s FETCH stage wired to a locally-built
   ``FakeHttpClient`` (independent of ``_form4_fixture_support.py``'s
   shared ``_SCENARIOS``/``default_responses()``, so nothing here can
   perturb any existing shared-fixture test) proving: an XSL-wrapped
   candidate is fetched via its RAW XML URL (never the XSL path), a
   basename absent from the directory index is refused rather than
   guessed, and a body that is not real ``ownershipDocument`` XML (e.g.
   the XSL stylesheet's own HTML rendering) is never accepted as
   ACQUIRED.
"""

from __future__ import annotations

import json

import pytest

from investment_research.collectors.form4 import (
    NormalizedOwnershipPrimaryDocument,
    OwnershipPrimaryDocumentRejection,
    normalize_ownership_primary_document,
)
from investment_research.collectors.sec_edgar import (
    FILING_INDEX_URL,
    cik_for_archives,
)
from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentStore
from investment_research.research.form4_acquisition_adapter import (
    FORM4_ADAPTER_ID,
    Form4Adapter,
    Form4IssuerReference,
)
from investment_research.research.source_routing import (
    AcquisitionStep,
    AcquisitionTarget,
    EvidenceRequirement,
    FailurePolicy,
    ImplementationStatus,
    SourceRoutingGraph,
    StepKind,
    StepStatus,
    TargetAcquisitionOutcome,
    TargetKind,
)
from investment_research.schemas.enums import UNKNOWN, ResearchDomain

from . import _form4_fixture_support as fx

# ---------------------------------------------------------------------------
# 1. normalize_ownership_primary_document() -- pure, no HTTP
# ---------------------------------------------------------------------------


def test_bare_basename_normalizes_with_unknown_wrapper():
    normalized, rejection = normalize_ownership_primary_document("primary_doc.xml")
    assert rejection is None
    assert normalized == NormalizedOwnershipPrimaryDocument(
        original_primary_document="primary_doc.xml",
        xsl_wrapper_path=UNKNOWN,
        normalized_xml_filename="primary_doc.xml",
    )


@pytest.mark.parametrize("wrapper", ["xslF345X06", "xslF345X03", "xslF345X01", "xslF345X99"])
def test_valid_xsl_wrapper_versions_normalize(wrapper):
    primary_document = f"{wrapper}/sample-99001.xml"
    normalized, rejection = normalize_ownership_primary_document(primary_document)
    assert rejection is None
    assert normalized.original_primary_document == primary_document
    assert normalized.xsl_wrapper_path == wrapper
    assert normalized.normalized_xml_filename == "sample-99001.xml"


@pytest.mark.parametrize(
    "primary_document,expected",
    [
        ("", OwnershipPrimaryDocumentRejection.EMPTY),
        ("/xslF345X06/safe.xml", OwnershipPrimaryDocumentRejection.ABSOLUTE_PATH),
        ("../safe.xml", OwnershipPrimaryDocumentRejection.PATH_TRAVERSAL),
        ("xslF345X06/../safe.xml", OwnershipPrimaryDocumentRejection.PATH_TRAVERSAL),
        ("xslF345X06/./safe.xml", OwnershipPrimaryDocumentRejection.CURRENT_DIR_SEGMENT),
        ("xslF345X06\\safe.xml", OwnershipPrimaryDocumentRejection.BACKSLASH),
        ("xslF345X06/safe.xml?download=1", OwnershipPrimaryDocumentRejection.QUERY_OR_FRAGMENT),
        ("xslF345X06/safe.xml#frag", OwnershipPrimaryDocumentRejection.QUERY_OR_FRAGMENT),
        ("http://evil.example.com/safe.xml", OwnershipPrimaryDocumentRejection.SCHEME_OR_NETLOC),
        ("https://evil.example.com/safe.xml", OwnershipPrimaryDocumentRejection.SCHEME_OR_NETLOC),
        ("a/b/safe.xml", OwnershipPrimaryDocumentRejection.TOO_MANY_PATH_SEGMENTS),
        ("a/b/c/safe.xml", OwnershipPrimaryDocumentRejection.TOO_MANY_PATH_SEGMENTS),
        ("xslF999X06/safe.xml", OwnershipPrimaryDocumentRejection.UNRECOGNIZED_XSL_WRAPPER),
        ("notxsl/safe.xml", OwnershipPrimaryDocumentRejection.UNRECOGNIZED_XSL_WRAPPER),
        ("xslF345X06/safe.htm", OwnershipPrimaryDocumentRejection.NOT_XML),
        ("xslF345X06/safe", OwnershipPrimaryDocumentRejection.NOT_XML),
        ("xslF345X06/", OwnershipPrimaryDocumentRejection.EMPTY_BASENAME),
        ("xslF345X06/safe%2e%2e.xml", OwnershipPrimaryDocumentRejection.PERCENT_ENCODED),
        ("xslF345X06/safe%2fxml", OwnershipPrimaryDocumentRejection.PERCENT_ENCODED),
        ("xslF345X06/safe%5cxml", OwnershipPrimaryDocumentRejection.PERCENT_ENCODED),
        ("xslF345X06/safe\x00.xml", OwnershipPrimaryDocumentRejection.CONTROL_CHARACTERS),
        ("xslF345X06/safe\t.xml", OwnershipPrimaryDocumentRejection.CONTROL_CHARACTERS),
    ],
)
def test_rejections(primary_document, expected):
    normalized, rejection = normalize_ownership_primary_document(primary_document)
    assert normalized is None
    assert rejection is expected


def test_exactly_one_of_normalized_or_rejection_is_returned():
    for value in ("primary_doc.xml", "xslF345X06/safe.xml", "", "../evil.xml"):
        normalized, rejection = normalize_ownership_primary_document(value)
        assert (normalized is None) != (rejection is None)


# ---------------------------------------------------------------------------
# 2. Form4Adapter FETCH stage -- locally-built FakeHttpClient, independent
#    of _form4_fixture_support.py's shared _SCENARIOS/default_responses().
# ---------------------------------------------------------------------------

_ACCESSION_SUFFIX = "000090"
_ACCESSION = f"{fx.ISSUER_CIK}-26-{_ACCESSION_SUFFIX}"
_ACCESSION_NODASH = _ACCESSION.replace("-", "")

_OWNERSHIP_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <issuer>
        <issuerCik>0001112223</issuerCik>
        <issuerName>Sample Biotech Holdings, Inc.</issuerName>
        <issuerTradingSymbol>SAMPB</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0005556667</rptOwnerCik>
            <rptOwnerName>Sample Reporting Person</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector><isOfficer>0</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-02-27</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>12.34</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts><sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
            <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <remarks>None</remarks>
    <ownerSignature><signatureName>/s/ Sample Reporting Person</signatureName><signatureDate>2026-03-01</signatureDate></ownerSignature>
</ownershipDocument>
"""

_XSL_HTML_RENDERING = "<html><head><title>Form 4</title></head><body>rendered via XSL, not raw XML</body></html>"


def _submissions_payload(primary_document: str, *, form: str = "4") -> dict:
    return {
        "cik": fx.ISSUER_CIK_INT, "name": fx.ISSUER_NAME, "tickers": [fx.ISSUER_TICKER],
        "filings": {
            "recent": {
                "accessionNumber": [_ACCESSION], "form": [form], "primaryDocument": [primary_document],
                "filingDate": ["2026-03-01"], "reportDate": ["2026-03-01"],
            },
            "files": [],
        },
    }


def _directory_url() -> str:
    return FILING_INDEX_URL.format(cik=cik_for_archives(fx.ISSUER_CIK_INT), accession_nodash=_ACCESSION_NODASH, document="index.json")


def _raw_xml_url(basename: str) -> str:
    return FILING_INDEX_URL.format(cik=cik_for_archives(fx.ISSUER_CIK_INT), accession_nodash=_ACCESSION_NODASH, document=basename)


def _xsl_display_url(wrapper: str, basename: str) -> str:
    """The XSL-styled display URL -- built ONLY so a test can assert it is
    NEVER requested (the fix's whole point: fetch the raw XML at the
    accession root, never this path)."""
    return FILING_INDEX_URL.format(
        cik=cik_for_archives(fx.ISSUER_CIK_INT), accession_nodash=_ACCESSION_NODASH, document=f"{wrapper}/{basename}",
    )


def _directory_payload(*names: str) -> dict:
    return {"directory": {"item": [{"name": name, "type": "4"} for name in names]}}


def _graph() -> SourceRoutingGraph:
    tid, rid = "target_a", "req_a"
    l1 = AcquisitionStep(
        step_id="l1", target_id=tid, step_kind=StepKind.LOCATE,
        acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id=FORM4_ADAPTER_ID,
        completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    f = AcquisitionStep(
        step_id="f", target_id=tid, step_kind=StepKind.FETCH,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=FORM4_ADAPTER_ID,
        depends_on_step_ids=("l1",), completion_condition=StepStatus.BODY_FETCHED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    p = AcquisitionStep(
        step_id="p", target_id=tid, step_kind=StepKind.PARSE,
        acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=FORM4_ADAPTER_ID,
        depends_on_step_ids=("f",), completion_condition=StepStatus.PARSED,
        implementation_status=ImplementationStatus.EXECUTOR_WIRED,
    )
    requirement = EvidenceRequirement(
        requirement_id=rid, serves_legacy_need_ids=("synthetic_a",), subject_scope=SubjectScope.COMPANY,
        domain=ResearchDomain.CONTRADICTION,
    )
    target = AcquisitionTarget(
        target_id=tid, target_kind=TargetKind.FORM4_FILING,
        required_step_ids=("l1", "f", "p"), serves_requirement_ids=(rid,),
    )
    return SourceRoutingGraph(requirements=(requirement,), targets=(target,), steps=(l1, f, p))


def _run(http):
    graph = _graph()
    target = graph.targets[0]
    store = DocumentStore()
    executor = AcquisitionExecutor(adapters={FORM4_ADAPTER_ID: Form4Adapter(http)}, document_store=store)
    ref = Form4IssuerReference(issuer_cik=fx.ISSUER_CIK_INT)
    report = executor.run(graph, form4_references={target.target_id: ref})
    return report, store, target


def _by_step_prefix(report, prefix):
    return next(r for r in report.target_reports[0].step_results if r.step_id.startswith(prefix))


# --- the real-world fix: XSL-wrapped primaryDocument fetches the RAW XML ---
def test_xsl_wrapped_primary_document_fetches_raw_xml_not_the_xsl_path():
    basename = "sample-99001.xml"
    wrapper = "xslF345X06"
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(_submissions_payload(f"{wrapper}/{basename}"))),
        _directory_url(): fx.ok(json.dumps(_directory_payload(basename))),
        _raw_xml_url(basename): fx.ok(_OWNERSHIP_XML),
    })
    report, store, target = _run(http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED

    fetch_result = _by_step_prefix(report, "f")
    fetched_entry = fetch_result.payload["fetched"][0]
    assert fetched_entry["original_primary_document"] == f"{wrapper}/{basename}"
    assert fetched_entry["normalized_xml_filename"] == basename
    assert fetched_entry["xsl_wrapper_path"] == wrapper
    assert fetched_entry["directory_index_verified"] is True
    assert fetched_entry["resolved_raw_xml_url"] == _raw_xml_url(basename)
    assert fetched_entry["ownership_document_verified"] is True
    assert fetched_entry["issuer_cik_verified"] is True

    # The XSL display path itself is NEVER requested.
    assert _xsl_display_url(wrapper, basename) not in http.requested_urls
    assert _raw_xml_url(basename) in http.requested_urls

    # Phase 3E.2.2 requirement 5: exactly 1 submissions + 2 per-candidate
    # (directory index + raw XML) physical requests -- the request-cap
    # formula and cache/physical-GET separation are unaffected by this fix.
    assert http.requested_urls == [fx.submissions_url(), _directory_url(), _raw_xml_url(basename)]


@pytest.mark.parametrize("wrapper", ["xslF345X06", "xslF345X03", "xslF345X01"])
def test_xsl_wrapper_version_variants_all_succeed(wrapper):
    basename = "sample-99001.xml"
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(_submissions_payload(f"{wrapper}/{basename}"))),
        _directory_url(): fx.ok(json.dumps(_directory_payload(basename))),
        _raw_xml_url(basename): fx.ok(_OWNERSHIP_XML),
    })
    report, store, target = _run(http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED


def test_simple_basename_still_succeeds_unaffected():
    basename = "sample-99001.xml"
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(_submissions_payload(basename))),
        _directory_url(): fx.ok(json.dumps(_directory_payload(basename))),
        _raw_xml_url(basename): fx.ok(_OWNERSHIP_XML),
    })
    report, store, target = _run(http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    fetched_entry = _by_step_prefix(report, "f").payload["fetched"][0]
    assert fetched_entry["xsl_wrapper_path"] == UNKNOWN


# --- directory index is the source of truth, never the XSL path itself ---
def test_basename_absent_from_directory_index_is_refused_never_guessed():
    basename = "sample-99001.xml"
    wrapper = "xslF345X06"
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(_submissions_payload(f"{wrapper}/{basename}"))),
        _directory_url(): fx.ok(json.dumps(_directory_payload("some-other-file.xml"))),
        # deliberately NOT registering _raw_xml_url(basename) -- if the
        # adapter ever guessed instead of verifying, this would 404
        # rather than the assertion below failing loudly.
    })
    report, store, target = _run(http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch_result = _by_step_prefix(report, "f")
    failure = fetch_result.payload["fetch_failures"][0]
    assert failure["directory_index_verified"] is False
    assert failure["normalized_xml_filename"] == basename
    assert _raw_xml_url(basename) not in http.requested_urls


# --- an XSL-rendered HTML body is never accepted as ACQUIRED --------------
def test_html_rendering_at_resolved_url_is_refused_never_acquired():
    """If the resolved raw-XML URL ever served the XSL stylesheet's own
    HTML rendering instead of the raw XML (e.g. a future SEC change), the
    shape check must refuse it -- never silently promoted to ACQUIRED."""
    basename = "sample-99001.xml"
    wrapper = "xslF345X06"
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(_submissions_payload(f"{wrapper}/{basename}"))),
        _directory_url(): fx.ok(json.dumps(_directory_payload(basename))),
        _raw_xml_url(basename): fx.ok(_XSL_HTML_RENDERING),
    })
    report, store, target = _run(http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    failure = _by_step_prefix(report, "f").payload["fetch_failures"][0]
    assert failure["ownership_document_verified"] is False
    assert "MALFORMED" in failure["reason"] or "MISSING_OWNERSHIP_DOCUMENT" in failure["reason"]


# --- malformed/unsafe primaryDocument shapes refused before any directory GET
@pytest.mark.parametrize(
    "primary_document",
    [
        "../evil.xml",
        "xslF345X06/../../etc/passwd.xml",
        "a/b/c.xml",
        "xslF999X06/safe.xml",
        "safe.htm",
    ],
)
def test_unsafe_primary_document_refused_before_any_directory_request(primary_document):
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(_submissions_payload(primary_document))),
    })
    report, store, target = _run(http)
    assert report.outcome_for(target.target_id) is not TargetAcquisitionOutcome.ACQUIRED
    failure = _by_step_prefix(report, "f").payload["fetch_failures"][0]
    assert "original_primary_document" in failure
    assert failure["original_primary_document"] == primary_document
    # No directory index request was made for this candidate -- refused at
    # the normalize step, before any HTTP call.
    assert _directory_url() not in http.requested_urls


# --- Form 3/5 exclusion / CIK mismatch / Form 4/A are unaffected ----------
def test_form4a_candidate_with_xsl_wrapped_primary_document_still_succeeds():
    basename = "sample-99002.xml"
    wrapper = "xslF345X06"
    xml_4a = _OWNERSHIP_XML.replace("<documentType>4</documentType>", "<documentType>4/A</documentType>")
    http = fx.FakeHttpClient(responses={
        fx.submissions_url(): fx.ok(json.dumps(_submissions_payload(f"{wrapper}/{basename}", form="4/A"))),
        _directory_url(): fx.ok(json.dumps(_directory_payload(basename))),
        _raw_xml_url(basename): fx.ok(xml_4a),
    })
    report, store, target = _run(http)
    assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    fetched_entry = _by_step_prefix(report, "f").payload["fetched"][0]
    assert fetched_entry["xsl_wrapper_path"] == wrapper
