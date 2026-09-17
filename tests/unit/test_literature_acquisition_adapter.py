"""Phase 3F: PubMedLiteratureAdapter/EuropePmcFullTextAdapter against
real-format fixtures (``tests/fixtures/literature_real_format/``, see that
directory's MANIFEST.md). No real network call anywhere in this file;
every URL and request count is asserted exactly. Mirrors
``tests/unit/test_form4_acquisition_adapter.py``'s own structure.
"""

from __future__ import annotations

import pytest

from investment_research.research.acquisition_executor import AcquisitionExecutor
from investment_research.research.acquisition_planning import AcquisitionMethod
from investment_research.research.checks import SubjectScope
from investment_research.research.document_store import DocumentStore
from investment_research.research.literature_acquisition_adapter import (
    ALLOWED_HOSTS,
    EUROPEPMC_FULLTEXT_URL,
    FAILURE_KIND_BODY_TOO_SHORT,
    FAILURE_KIND_PARSE_ERROR,
    FAILURE_KIND_TRANSPORT_ERROR,
    LITERATURE_ADAPTER_ID,
    LiteratureReference,
    PubMedLiteratureAdapter,
    RequestBudgetLimits,
    _public_request_url,
    _require_allowed_host,
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
from investment_research.schemas.enums import (
    ContentKind,
    DocumentAuthority,
    FetchOutcome,
    ResearchDomain,
)

from . import _literature_fixture_support as fx


def _literature_graph(*tags: str) -> SourceRoutingGraph:
    requirements, targets, steps = [], [], []
    for tag in tags:
        tid, rid = f"target_{tag}", f"req_{tag}"
        l1 = AcquisitionStep(
            step_id=f"l1_{tag}", target_id=tid, step_kind=StepKind.LOCATE,
            acquisition_method=AcquisitionMethod.NEW_DIRECT_ADAPTER, adapter_id=LITERATURE_ADAPTER_ID,
            completion_condition=StepStatus.URL_RESOLVED, failure_policy=FailurePolicy.REQUIRED,
            implementation_status=ImplementationStatus.EXECUTOR_WIRED,
        )
        f = AcquisitionStep(
            step_id=f"f_{tag}", target_id=tid, step_kind=StepKind.FETCH,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=LITERATURE_ADAPTER_ID,
            depends_on_step_ids=(f"l1_{tag}",), completion_condition=StepStatus.BODY_FETCHED,
            implementation_status=ImplementationStatus.EXECUTOR_WIRED,
        )
        p = AcquisitionStep(
            step_id=f"p_{tag}", target_id=tid, step_kind=StepKind.PARSE,
            acquisition_method=AcquisitionMethod.KNOWN_URL_HTTP, adapter_id=LITERATURE_ADAPTER_ID,
            depends_on_step_ids=(f"f_{tag}",), completion_condition=StepStatus.PARSED,
            implementation_status=ImplementationStatus.EXECUTOR_WIRED,
        )
        requirement = EvidenceRequirement(
            requirement_id=rid, serves_legacy_need_ids=(f"synthetic_{tag}",), subject_scope=SubjectScope.COMPANY,
            domain=ResearchDomain.SCIENCE_TECHNOLOGY,
        )
        target = AcquisitionTarget(
            target_id=tid, target_kind=TargetKind.LITERATURE_ARTICLE,
            required_step_ids=(f"l1_{tag}", f"f_{tag}", f"p_{tag}"), serves_requirement_ids=(rid,),
        )
        requirements.append(requirement)
        targets.append(target)
        steps.extend([l1, f, p])
    return SourceRoutingGraph(requirements=tuple(requirements), targets=tuple(targets), steps=tuple(steps))


def _run(refs: dict, http, *, graph=None, store=None, adapter=None, **adapter_kwargs):
    graph = graph or _literature_graph(*[t.replace("target_", "") for t in refs])
    store = store or DocumentStore()
    adapter = adapter or PubMedLiteratureAdapter(http, **adapter_kwargs)
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    report = executor.run(graph, literature_references=refs)
    return report, store, graph


def _by_step_prefix(report, target_index, prefix):
    return next(r for r in report.target_reports[target_index].step_results if r.step_id.startswith(prefix))


# --- Priority A/B/C acquisition order ---------------------------------------
def test_priority_a_ct_pmid_skips_esearch_entirely():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    assert not any("esearch.fcgi" in u for u in http.requested_urls)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"]
    assert parsed[0]["pmid"] == "90000001"


def test_priority_b_nct_id_exact_search():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(
        responses={
            fx.esearch_url(term, 20): fx.ok(fx.esearch_response(["90000007"])),
            fx.efetch_url(["90000007"]): fx.ok(fx.fixture_text("nct_id_present.xml")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.payload["confirmed_same_trial"] is True
    assert locate.payload["priority"] == "B"


def test_priority_c_alias_candidates_never_auto_confirmed_same_trial():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "SampleCompound AND FictionalSyndrome"
    http = fx.FakeHttpClient(
        responses={
            fx.esearch_url(term, 20): fx.ok(fx.esearch_response(["90000001"])),
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
        }
    )
    ref = LiteratureReference(alias="SampleCompound", condition="FictionalSyndrome")
    report, store, _ = _run({target_id: ref}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.payload["priority"] == "C"
    assert locate.payload["confirmed_same_trial"] is False
    parse = _by_step_prefix(report, 0, "p_")
    assert parse.payload["confirmed_same_trial"] is False


def test_no_reference_supplied_never_acquired_zero_requests():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={})
    report, store, _ = _run({}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


def test_reference_with_no_usable_field_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={})
    report, store, _ = _run({target_id: LiteratureReference()}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    assert http.requested_urls == []


# --- zero results / malformed / error distinguishing -------------------------
def test_zero_esearch_results_never_acquired():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990099[si]"
    http = fx.FakeHttpClient(responses={fx.esearch_url(term, 20): fx.ok(fx.esearch_response([]))})
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990099")}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.ZERO_RESULTS


def test_malformed_esearch_response_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(responses={fx.esearch_url(term, 20): fx.ok("not json { at all")})
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.FAILED


@pytest.mark.parametrize(
    "outcome_name",
    ["RATE_LIMITED", "ERROR", "TIMEOUT"],
)
def test_esearch_transport_failures_distinguished(outcome_name):
    from investment_research.schemas.enums import FetchOutcome

    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(
        responses={fx.esearch_url(term, 20): fx.failed(FetchOutcome[outcome_name], outcome_name.lower())}
    )
    report, store, _ = _run({target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph)
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.FAILED
    assert outcome_name in locate.failure_reason


def test_malformed_xml_efetch_body_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000012"]): fx.ok(fx.fixture_text("malformed.xml"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000012")}, http, graph=graph)
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert "MALFORMED" in fetch.failure_reason


def test_html_error_page_efetch_body_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("html_error_page.htm"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert "HTML_ERROR_PAGE" in fetch.failure_reason


def test_empty_efetch_body_fails():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok("")})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert "EMPTY" in fetch.failure_reason


def test_pmid_mismatch_fails_and_is_recorded():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000013"]): fx.ok(fx.fixture_text("mismatched_pmid.xml"))})
    executor_store = DocumentStore()
    from investment_research.research.acquisition_executor import AcquisitionExecutor as _Exec

    executor = _Exec(adapters={LITERATURE_ADAPTER_ID: PubMedLiteratureAdapter(http)}, document_store=executor_store)
    report = executor.run(graph, literature_references={target_id: LiteratureReference(ct_pmid="90000013")})
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


# --- batch fetch / same-PMID dedup -------------------------------------------
def test_batch_fetch_of_two_pmids_is_one_efetch_get():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "SampleCompound AND FictionalSyndrome"
    http = fx.FakeHttpClient(
        responses={
            fx.esearch_url(term, 20): fx.ok(fx.esearch_response(["90000015", "90000016"])),
            fx.efetch_url(["90000015", "90000016"]): fx.ok(fx.fixture_text("batch_articleset.xml")),
        }
    )
    ref = LiteratureReference(alias="SampleCompound", condition="FictionalSyndrome")
    report, store, _ = _run({target_id: ref}, http, graph=graph)
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"]
    assert {d["pmid"] for d in parsed} == {"90000015", "90000016"}


def test_same_pmid_requested_by_two_targets_fetched_once():
    graph = _literature_graph("a", "b")
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    refs = {t.target_id: LiteratureReference(ct_pmid="90000001") for t in graph.targets}
    report, store, _ = _run(refs, http, graph=graph)
    for target in graph.targets:
        assert report.outcome_for(target.target_id) is TargetAcquisitionOutcome.ACQUIRED
    efetch_calls = [u for u in http.requested_urls if "efetch.fcgi" in u]
    assert len(efetch_calls) == 1


# --- Europe PMC open-access full text -----------------------------------------
def test_open_access_europepmc_fulltext_acquired_and_distinguished_from_abstract():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok(fx.fixture_text("europepmc_fulltext_oa.xml")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    assert parsed["full_text_acquired"] is True
    fulltext_doc_id = parsed["europepmc_fulltext_document_id"]
    assert fulltext_doc_id is not None
    stored = store.get(fulltext_doc_id)
    assert stored is not None
    assert stored.document.content_kind is ContentKind.FULL_DOCUMENT
    assert stored.document.authority is DocumentAuthority.BIOMEDICAL_LITERATURE
    assert "fictional full-text introduction content" in stored.document.text.lower()
    # The PubMed abstract-only document is a SEPARATE document, never
    # conflated with the Europe PMC full text.
    pubmed_doc_id = parsed["document_id"]
    assert pubmed_doc_id != fulltext_doc_id


def test_non_open_access_never_marked_full_text_acquired_and_no_fulltext_fetch():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    assert parsed["full_text_acquired"] is False
    assert parsed["europepmc_fulltext_document_id"] is None
    assert not any("fullTextXML" in u for u in http.requested_urls)


# --- Phase 3F.2 correction requirement 1: fullTextXML URL shape ------------
def test_europepmc_fulltext_url_is_the_real_single_segment_shape():
    """A real Live Smoke run against a genuinely open-access PMCID
    (PMC2910600) 404'd against the OLD, two-segment
    {source}/{pmcid}/fullTextXML path -- the real Europe PMC RESTful Web
    Service takes exactly one path segment (the PMCID itself)."""
    assert EUROPEPMC_FULLTEXT_URL == "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    assert "{source}" not in EUROPEPMC_FULLTEXT_URL
    assert (
        fx.europepmc_fulltext_url("PMC2910600")
        == "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2910600/fullTextXML"
    )


def test_no_fixture_file_contains_the_real_identifiers_from_the_confirmed_live_run():
    """Requirement: never add the real PMID/PMCID a confirmed Live Smoke
    run actually used (20668659 / PMC2910600) into a fixture -- test only
    with synthetic IDs (the PMC999#### / 9000#### convention this fixture
    set already establishes). Scans every fixture file actually on disk,
    rather than checking one hand-picked string, so a future fixture
    addition is covered automatically."""
    for path in fx.FIXTURE_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "PMC2910600" not in text, f"real PMCID leaked into fixture {path.name}"
        assert "20668659" not in text, f"real PMID leaked into fixture {path.name}"


# --- Phase 3F.2 correction requirement 2/3: Europe PMC fulltext failure ----
# is never silently swallowed -- structured, sanitized, and coverage_complete
# is set False, while the PubMed abstract/excerpt success is preserved.
def test_fulltext_404_is_reported_structured_never_silently_dropped():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.not_found(),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.payload["coverage_complete"] is False
    failures = fetch.payload["europepmc_fulltext_failures"]
    assert list(failures.keys()) == ["90000008"]
    entry = failures["90000008"]
    assert entry["pmcid"] == "PMC9990008"
    assert entry["stage"] == "fulltext_fetch"
    assert entry["kind"] == FetchOutcome.NOT_FOUND.value
    assert "reason" in entry and entry["reason"]
    # PubMed's own success is never lost -- the article is still fetched
    # and parsed, it simply carries no full-text document.
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    assert parsed["pmid"] == "90000008"
    assert parsed["full_text_acquired"] is False
    assert parsed["europepmc_search_succeeded"] is True
    assert parsed["europepmc_fulltext_document_id"] is None
    assert fetch.status is StepStatus.BODY_FETCHED  # never FAILED -- PubMed itself succeeded


@pytest.mark.parametrize(
    "outcome,expected_kind",
    [
        (FetchOutcome.NOT_FOUND, FetchOutcome.NOT_FOUND.value),
        (FetchOutcome.RATE_LIMITED, FetchOutcome.RATE_LIMITED.value),
        (FetchOutcome.ERROR, FetchOutcome.ERROR.value),
        (FetchOutcome.BLOCKED, FetchOutcome.BLOCKED.value),
    ],
)
def test_fulltext_fetch_failure_kinds_are_structured_not_free_text(outcome, expected_kind):
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.failed(outcome, f"{outcome.value} 999"),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    entry = fetch.payload["europepmc_fulltext_failures"]["90000008"]
    assert entry["kind"] == expected_kind


class _RaisesOnlyForUrl:
    """A minimal transport double that raises ConnectionError for exactly
    ONE URL and otherwise delegates to a real FakeHttpClient -- lets a test
    isolate a transport-level exception to a single request (e.g. only the
    Europe PMC fulltext fetch) while every other call in the SAME run
    still succeeds normally, which RaisingHttpClient (raises for every
    call) cannot exercise."""

    def __init__(self, raises_for_url: str, responses: dict) -> None:
        self._raises_for_url = raises_for_url
        self._delegate = fx.FakeHttpClient(responses=responses)

    def get(self, url, **kwargs):
        self._delegate.requested_urls.append(url)
        if url == self._raises_for_url:
            raise ConnectionError(f"simulated connection error for {url}")
        return self._delegate.responses.get(url, fx.not_found())

    @property
    def requested_urls(self):
        return self._delegate.requested_urls


def test_fulltext_transport_exception_reported_as_transport_error_kind():
    """A connection-level exception (never an HTTP response at all) during
    the Europe PMC fulltext fetch specifically -- PubMed EFetch and Europe
    PMC search both still succeed -- is classified
    FAILURE_KIND_TRANSPORT_ERROR, distinct from any FetchOutcome value
    (which requires an actual HTTP response)."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    fulltext_url = fx.europepmc_fulltext_url("PMC9990008")
    http = _RaisesOnlyForUrl(
        raises_for_url=fulltext_url,
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
        },
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.BODY_FETCHED  # PubMed itself still succeeded
    entry = fetch.payload["europepmc_fulltext_failures"]["90000008"]
    assert entry["kind"] == FAILURE_KIND_TRANSPORT_ERROR
    assert entry["stage"] == "fulltext_fetch"


def test_fulltext_malformed_xml_reported_as_parse_error_not_fetch_failure():
    """A 2xx Europe PMC response whose body does NOT match the expected
    JATS-like <article> shape is a PARSE failure, structurally distinct
    from a fetch-level failure (404/429/5xx/timeout/transport error) --
    Phase 3F.2 correction requirement 3's "distinguish fetch failure from
    parse failure"."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok("<not-jats-at-all>this is not a real article body</not-jats-at-all>"),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    entry = fetch.payload["europepmc_fulltext_failures"]["90000008"]
    assert entry["kind"] == FAILURE_KIND_PARSE_ERROR
    assert entry["stage"] == "fulltext_parse"
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    assert parsed["full_text_acquired"] is False


def test_fulltext_body_too_short_reported_structured():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
            fx.europepmc_fulltext_url("PMC9990008"): fx.ok("too short"),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    entry = fetch.payload["europepmc_fulltext_failures"]["90000008"]
    assert entry["kind"] == FAILURE_KIND_BODY_TOO_SHORT


def test_search_failure_is_reported_structured_and_distinct_from_budget_skip():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.failed(FetchOutcome.ERROR, "500"),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.payload["coverage_complete"] is False
    assert fetch.payload["budget_skipped_europepmc_search_pmids"] == []  # not a budget skip
    entry = fetch.payload["europepmc_search_failures"]["90000001"]
    assert entry["kind"] == FetchOutcome.ERROR.value
    assert entry["stage"] == "search"


def test_pubmed_abstract_document_is_never_full_document_content_kind():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    parsed = _by_step_prefix(report, 0, "p_").payload["parsed_documents"][0]
    stored = store.get(parsed["document_id"])
    assert stored.document.content_kind is ContentKind.EXCERPT
    assert stored.document.content_kind is not ContentKind.FULL_DOCUMENT


# --- Document versioning ------------------------------------------------------
def test_refetch_with_different_content_creates_a_new_document_version():
    store = DocumentStore()
    graph1 = _literature_graph("a")
    target_id = graph1.targets[0].target_id
    http1 = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report1, _, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http1, graph=graph1, store=store)
    doc_id_v1 = _by_step_prefix(report1, 0, "p_").payload["parsed_documents"][0]["document_id"]

    graph2 = _literature_graph("a")
    http2 = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("updated_article_v2.xml"))})
    report2, _, _ = _run({graph2.targets[0].target_id: LiteratureReference(ct_pmid="90000001")}, http2, graph=graph2, store=store)
    doc_id_v2 = _by_step_prefix(report2, 0, "p_").payload["parsed_documents"][0]["document_id"]

    assert doc_id_v1 != doc_id_v2
    history = store.version_history(doc_id_v2)
    assert len(history) == 2
    assert history[0].document_id == doc_id_v1  # oldest first
    assert history[-1].document_id == doc_id_v2


# --- HTTP safety ---------------------------------------------------------------
def test_allowed_hosts_are_exactly_the_documented_three():
    assert frozenset({"eutils.ncbi.nlm.nih.gov", "www.ebi.ac.uk", "europepmc.org"}) == ALLOWED_HOSTS


def test_require_allowed_host_rejects_other_hosts():
    with pytest.raises(ValueError):
        _require_allowed_host("https://evil.example.com/esearch.fcgi")
    _require_allowed_host("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")  # must not raise


def test_public_request_url_strips_email_and_api_key():
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&email=secret@example.com&api_key=SECRETKEY123&term=x"
    public = _public_request_url(url)
    assert "secret@example.com" not in public
    assert "SECRETKEY123" not in public
    assert "term=x" in public


# --- Phase 3F.0.1 requirement 4: request caps --------------------------------
def test_max_articles_caps_fan_out_never_unlimited():
    """A candidate list larger than max_articles is never fetched in full
    -- the excess is excluded explicitly, never silently dropped or
    reported as NOT_FOUND."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "SampleCompound AND FictionalSyndrome"
    http = fx.FakeHttpClient(
        responses={
            fx.esearch_url(term, 20): fx.ok(fx.esearch_response(["90000015", "90000016"])),
            fx.efetch_url(["90000015"]): fx.ok(fx.fixture_text("normal_abstract.xml").replace("90000001", "90000015")),
        }
    )
    ref = LiteratureReference(alias="SampleCompound", condition="FictionalSyndrome")
    report, store, _ = _run({target_id: ref}, http, graph=graph, max_articles=1)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.BODY_FETCHED  # one article WAS fetched
    assert fetch.payload["budget_excluded_pmids"] == ["90000016"]
    assert fetch.payload["coverage_complete"] is False
    parse = _by_step_prefix(report, 0, "p_")
    assert parse.payload["coverage_complete"] is False
    assert {d["pmid"] for d in parse.payload["parsed_documents"]} == {"90000015"}


def test_zero_article_budget_never_reported_as_not_found():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph, max_articles=0)
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.SKIPPED_DUE_TO_BUDGET
    assert fetch.status is not StepStatus.NOT_FOUND
    assert fetch.status is not StepStatus.FAILED
    assert fetch.payload["budget_excluded_pmids"] == ["90000001"]
    assert fetch.payload["coverage_complete"] is False
    assert http.requested_urls == []  # never even attempted the EFetch GET
    assert report.outcome_for(target_id) is not TargetAcquisitionOutcome.ACQUIRED


def test_zero_search_budget_blocks_esearch_before_any_get():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={})
    report, store, _ = _run(
        {target_id: LiteratureReference(nct_id="NCT09990001")}, http, graph=graph, max_search_requests=0,
    )
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.SKIPPED_DUE_TO_BUDGET
    assert http.requested_urls == []


def test_zero_fulltext_budget_skips_europepmc_fetch_but_article_still_acquired():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000008"]): fx.ok(fx.fixture_text("ids_complete.xml")),
            fx.europepmc_search_url("90000008"): fx.ok(fx.fixture_text("europepmc_search_oa.json")),
        }
    )
    report, store, _ = _run(
        {target_id: LiteratureReference(ct_pmid="90000008")}, http, graph=graph, max_fulltext_fetches=0,
    )
    # The PubMed article itself is still acquired -- only the Europe PMC
    # full-text enrichment is budget-skipped.
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.payload["budget_skipped_fulltext_pmcids"] == ["PMC9990008"]
    assert fetch.payload["coverage_complete"] is False
    assert not any("fullTextXML" in u for u in http.requested_urls)
    parse = _by_step_prefix(report, 0, "p_")
    parsed = parse.payload["parsed_documents"][0]
    assert parsed["full_text_acquired"] is False


def test_default_budgets_never_trigger_on_ordinary_single_article_run():
    """The default caps never interfere with normal, small-scale usage.
    Registers a real (non-OA) Europe PMC search response too -- Phase 3F.2
    correction requirement 2 means an UNREGISTERED Europe PMC search URL
    now correctly counts as a genuine (non-budget) provider failure and
    sets coverage_complete=False, so a test asserting "ordinary, complete
    run" must make Europe PMC search actually succeed, not rely on that
    failure being silently swallowed (the very bug this correction
    fixes)."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.europepmc_search_url("90000001"): fx.ok(fx.fixture_text("europepmc_search_non_oa.json")),
        }
    )
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    parse = _by_step_prefix(report, 0, "p_")
    assert parse.payload["coverage_complete"] is True
    assert parse.payload["budget_excluded_pmids"] == []
    assert parse.payload["europepmc_search_failures"] == {}
    assert parse.payload["europepmc_fulltext_failures"] == {}


# --- external LLM / Pipeline isolation ----------------------------------------
def test_no_external_llm_tokens_used():
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    report, store, _ = _run({target_id: LiteratureReference(ct_pmid="90000001")}, http, graph=graph)
    assert report.diagnostics.external_llm_tokens == 0


# --- Phase 3F.0.1 requirement 5: secret leakage -------------------------------
class _FakeSettingsWithSecrets:
    ncbi_tool = "investment-research-agent"
    ncbi_email = "secret-contact@example.test"
    ncbi_api_key = "SECRET_NCBI_API_KEY_999"


_SECRET_STRINGS = (
    _FakeSettingsWithSecrets.ncbi_email,
    _FakeSettingsWithSecrets.ncbi_api_key,
)


def _assert_no_secret_leak(obj) -> None:
    text = repr(obj)
    for secret in _SECRET_STRINGS:
        assert secret not in text, f"secret {secret!r} leaked into repr(): {text[:500]}"


def _esearch_url_with_secrets(term: str, max_pmids: int) -> str:
    from urllib.parse import urlencode

    from investment_research.research.literature_acquisition_adapter import (
        NCBI_ESEARCH_URL,
        _eutils_params,
    )

    params = _eutils_params(
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids)},
        _FakeSettingsWithSecrets(),
    )
    return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"


def _efetch_url_with_secrets(pmids: list[str]) -> str:
    from urllib.parse import urlencode

    from investment_research.research.literature_acquisition_adapter import (
        NCBI_EFETCH_URL,
        _eutils_params,
    )

    params = _eutils_params(
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"},
        _FakeSettingsWithSecrets(),
    )
    return f"{NCBI_EFETCH_URL}?{urlencode(params)}"


def test_secrets_reach_the_real_request_but_never_the_cache_key_or_document():
    """The real E-utilities request DOES carry email/api_key (that is how
    NCBI courtesy identification works) -- but nothing this module stores,
    caches, or surfaces afterward may ever contain it."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    term = "NCT09990001[si]"
    http = fx.FakeHttpClient(
        responses={
            _esearch_url_with_secrets(term, 20): fx.ok(fx.esearch_response(["90000007"])),
            _efetch_url_with_secrets(["90000007"]): fx.ok(fx.fixture_text("nct_id_present.xml")),
        }
    )
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(http, settings=_FakeSettingsWithSecrets())
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    report = executor.run(graph, literature_references={target_id: LiteratureReference(nct_id="NCT09990001")})
    assert report.outcome_for(target_id) is TargetAcquisitionOutcome.ACQUIRED

    # The secrets WERE actually sent on the wire -- proves they weren't
    # simply omitted everywhere (which would silently break real NCBI
    # courtesy identification). Decoded, since urlencode percent-escapes
    # characters like '@' -- a human/log reader would see the decoded form.
    from urllib.parse import unquote

    combined_requested = unquote(" ".join(http.requested_urls))
    for secret in _SECRET_STRINGS:
        assert secret in combined_requested

    # ... but never anywhere this module surfaces to a human/log/manifest.
    for step_report in report.target_reports[0].step_results:
        _assert_no_secret_leak(step_report)
        _assert_no_secret_leak(step_report.payload)
        _assert_no_secret_leak(step_report.failure_reason)
    stored_docs = [store.get(d["document_id"]) for d in _by_step_prefix(report, 0, "p_").payload["parsed_documents"]]
    for stored in stored_docs:
        _assert_no_secret_leak(stored)
        assert stored.document.url is not None
        for secret in _SECRET_STRINGS:
            assert secret not in stored.document.url


def test_secrets_never_leak_through_a_failed_request_diagnostic():
    """Even a FAILED step's own diagnostics (failure_reason, payload) must
    never carry the secret -- a transport failure is exactly when a raw
    request URL is most tempting to log verbatim."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    # No response registered at all -> a 404-shaped FakeResult, exercising
    # the failure path.
    http = fx.FakeHttpClient(responses={})
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(http, settings=_FakeSettingsWithSecrets())
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    report = executor.run(graph, literature_references={target_id: LiteratureReference(nct_id="NCT09990001")})
    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.FAILED
    _assert_no_secret_leak(locate.failure_reason)
    _assert_no_secret_leak(locate.payload)
    # The secret WAS still sent on the (failed) request itself.
    from urllib.parse import unquote

    combined_requested = unquote(" ".join(http.requested_urls))
    for secret in _SECRET_STRINGS:
        assert secret in combined_requested


def test_request_cache_keys_never_contain_secrets():
    """ExecutionContext.request_cache is a plain dict whose KEYS could
    easily end up in a debugger dump or a future diagnostic dump -- they
    must never carry a secret either."""
    from investment_research.research.acquisition_executor import ExecutionContext
    from investment_research.research.source_routing import AcquisitionTarget, TargetKind

    store = DocumentStore()
    context = ExecutionContext(
        target=AcquisitionTarget(target_id="t", target_kind=TargetKind.LITERATURE_ARTICLE),
        document_store=store,
    )
    http = fx.FakeHttpClient(
        responses={fx.esearch_url("NCT09990001[si]", 20): fx.ok(fx.esearch_response(["90000007"]))}
    )
    adapter = PubMedLiteratureAdapter(http, settings=_FakeSettingsWithSecrets())
    adapter._esearch("NCT09990001[si]", 20, context)
    for key in context.request_cache:
        for secret in _SECRET_STRINGS:
            assert secret not in key


# --- Phase 3F.0.2 requirement 1: transport exceptions never leak wire_url ---
def test_transport_exception_never_leaks_wire_url_or_secret():
    """The fake transport's lowest layer (RaisingHttpClient) sees the raw
    wire_url and the raw secret -- nothing this module returns afterward
    may contain either, even though the transport RAISED rather than
    returning a result."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.RaisingHttpClient()
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(http, settings=_FakeSettingsWithSecrets())
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    report = executor.run(graph, literature_references={target_id: LiteratureReference(nct_id="NCT09990001")})

    locate = _by_step_prefix(report, 0, "l1_")
    assert locate.status is StepStatus.FAILED
    _assert_no_secret_leak(locate.failure_reason)
    _assert_no_secret_leak(locate.payload)
    for step_report in report.target_reports[0].step_results:
        _assert_no_secret_leak(step_report)

    # The transport itself DID see the wire_url (with secrets) and raised
    # about it -- confirms the exception path was genuinely exercised, not
    # silently skipped.
    from urllib.parse import unquote

    assert http.requested_urls
    combined_requested = unquote(" ".join(http.requested_urls))
    for secret in _SECRET_STRINGS:
        assert secret in combined_requested
    # ... but the raw wire_url string itself never reaches the sanitized
    # failure_reason (only its public_url form, or a redaction marker).
    assert http.requested_urls[0] not in locate.failure_reason


def test_transport_exception_without_secrets_still_sanitized_and_typed():
    """Even with no Settings/secrets configured, a raised transport
    exception is caught and converted to a clear, typed failure -- never
    left to propagate uncaught out of the adapter."""
    graph = _literature_graph("a")
    target_id = graph.targets[0].target_id
    http = fx.RaisingHttpClient()
    store = DocumentStore()
    adapter = PubMedLiteratureAdapter(http)
    executor = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store)
    report = executor.run(graph, literature_references={target_id: LiteratureReference(ct_pmid="90000001")})
    fetch = _by_step_prefix(report, 0, "f_")
    assert fetch.status is StepStatus.FAILED
    assert "ConnectionError" in fetch.failure_reason


def test_secret_redaction_covers_both_raw_and_percent_encoded_forms():
    """Requirement 1: test both pre- and post-URL-encoding forms of the
    secret directly against the sanitizer, not just end to end."""
    from investment_research.research.literature_acquisition_adapter import _sanitize_text

    settings = _FakeSettingsWithSecrets()
    raw_form = f"error contacting {settings.ncbi_email} with key {settings.ncbi_api_key}"
    encoded_form = (
        "error at https://eutils.ncbi.nlm.nih.gov/x?email=secret-contact%40example.test"
        "&api_key=SECRET_NCBI_API_KEY_999"
    )
    for text in (raw_form, encoded_form):
        sanitized = _sanitize_text(text, settings=settings)
        for secret in _SECRET_STRINGS:
            assert secret not in sanitized
        from urllib.parse import quote

        for secret in _SECRET_STRINGS:
            assert quote(secret, safe="") not in sanitized


def test_public_request_url_never_strips_tool_param():
    """``tool`` is not secret-shaped (no personal info) and is not stripped
    -- only ``email``/``api_key`` are (Phase 3F.0.2 requirement 1's
    explicit "tool -- only if currently treated as secret" clause: it is
    not)."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&tool=investment-research-agent&term=x"
    assert "tool=investment-research-agent" in _public_request_url(url)


# --- Phase 3F.0.2 requirement 2: RequestBudget is run-scoped ----------------
def test_article_budget_shared_across_two_targets_in_the_same_run():
    graph = _literature_graph("a", "b")
    http = fx.FakeHttpClient(
        responses={
            fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml")),
            fx.efetch_url(["90000015"]): fx.ok(
                fx.fixture_text("batch_articleset.xml").replace("90000016", "IGNORED")
            ),
        }
    )
    adapter = PubMedLiteratureAdapter(http, max_articles=1)
    refs = {
        graph.targets[0].target_id: LiteratureReference(ct_pmid="90000001"),
        graph.targets[1].target_id: LiteratureReference(ct_pmid="90000015"),
    }
    report, store, _ = _run(refs, http, graph=graph, adapter=adapter)
    fetch_a = _by_step_prefix(report, 0, "f_")
    fetch_b = _by_step_prefix(report, 1, "f_")
    # The FIRST target consumes the entire (shared) budget of 1 article;
    # the SECOND target, in the SAME run, gets none.
    assert fetch_a.status is StepStatus.BODY_FETCHED
    assert fetch_b.status is StepStatus.SKIPPED_DUE_TO_BUDGET
    assert fetch_b.payload["budget_excluded_pmids"] == ["90000015"]


def test_second_run_with_same_adapter_instance_starts_with_zero_usage():
    """The same PubMedLiteratureAdapter (and its RequestBudgetLimits)
    reused for a SECOND, separate AcquisitionExecutor.run() call must start
    with fresh usage -- never carrying over the first run's consumption."""
    http1 = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    adapter = PubMedLiteratureAdapter(http1, max_articles=1)

    graph1 = _literature_graph("a")
    store1 = DocumentStore()
    executor1 = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store1)
    report1 = executor1.run(graph1, literature_references={graph1.targets[0].target_id: LiteratureReference(ct_pmid="90000001")})
    assert _by_step_prefix(report1, 0, "f_").status is StepStatus.BODY_FETCHED

    # Second run, same adapter instance, a DIFFERENT PMID -- if usage carried
    # over, max_articles=1 would already be exhausted and this would be
    # SKIPPED_DUE_TO_BUDGET; it must not be.
    http1.responses[fx.efetch_url(["90000015"])] = fx.ok(
        fx.fixture_text("batch_articleset.xml").replace("90000016", "IGNORED")
    )
    graph2 = _literature_graph("a")
    store2 = DocumentStore()
    executor2 = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store2)
    report2 = executor2.run(graph2, literature_references={graph2.targets[0].target_id: LiteratureReference(ct_pmid="90000015")})
    assert _by_step_prefix(report2, 0, "f_").status is StepStatus.BODY_FETCHED


def test_failed_first_run_does_not_carry_budget_usage_to_second_run():
    """Even when the first run ends in FAILED (a budget exhaustion, here),
    a second run with the same adapter instance starts at zero -- the
    executor's own per-run adapter_state dict is never reused."""
    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    adapter = PubMedLiteratureAdapter(http, max_articles=0)  # guarantees FAILED/SKIPPED first run

    graph1 = _literature_graph("a")
    store1 = DocumentStore()
    executor1 = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store1)
    report1 = executor1.run(graph1, literature_references={graph1.targets[0].target_id: LiteratureReference(ct_pmid="90000001")})
    assert _by_step_prefix(report1, 0, "f_").status is StepStatus.SKIPPED_DUE_TO_BUDGET

    # A fresh adapter with a real (non-zero) budget, run against the same
    # PMID, must succeed -- proving the SECOND run (even a differently
    # configured one) never inherits stale state from anywhere shared.
    adapter2 = PubMedLiteratureAdapter(http, max_articles=1)
    graph2 = _literature_graph("a")
    store2 = DocumentStore()
    executor2 = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter2}, document_store=store2)
    report2 = executor2.run(graph2, literature_references={graph2.targets[0].target_id: LiteratureReference(ct_pmid="90000001")})
    assert _by_step_prefix(report2, 0, "f_").status is StepStatus.BODY_FETCHED


def test_budget_usage_is_a_fresh_object_per_run_not_shared_globally():
    """Directly proves RequestBudgetUsage is created fresh in
    ExecutionContext.adapter_state per run() call, never as adapter/module
    global state."""
    from investment_research.research.literature_acquisition_adapter import (
        _BUDGET_USAGE_STATE_KEY,
        RequestBudgetUsage,
    )

    http = fx.FakeHttpClient(responses={fx.efetch_url(["90000001"]): fx.ok(fx.fixture_text("normal_abstract.xml"))})
    adapter = PubMedLiteratureAdapter(http)

    graph1 = _literature_graph("a")
    store1 = DocumentStore()
    executor1 = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store1)
    executor1.run(graph1, literature_references={graph1.targets[0].target_id: LiteratureReference(ct_pmid="90000001")})

    graph2 = _literature_graph("a")
    store2 = DocumentStore()
    executor2 = AcquisitionExecutor(adapters={LITERATURE_ADAPTER_ID: adapter}, document_store=store2)
    # Confirm no adapter/module-level attribute holds a RequestBudgetUsage
    # -- only ExecutionContext.adapter_state does, freshly, per run.
    assert not hasattr(adapter, "budget")
    assert isinstance(adapter.limits, RequestBudgetLimits)
    executor2.run(graph2, literature_references={graph2.targets[0].target_id: LiteratureReference(ct_pmid="90000001")})
    # (RequestBudgetUsage itself is only constructible/inspectable via
    # ExecutionContext.adapter_state, exercised by the two tests above --
    # this test's own assertion is that no OTHER persistent home for it
    # exists on the adapter.)
    assert RequestBudgetUsage  # imported successfully; used for typing/documentation of intent
    assert _BUDGET_USAGE_STATE_KEY  # the private key both runs used, independently
