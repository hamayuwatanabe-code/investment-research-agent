"""Shared support for tests driving ``PubMedLiteratureAdapter``/
``EuropePmcFullTextAdapter`` against the real-format fixtures in
``tests/fixtures/literature_real_format/`` (see that directory's
MANIFEST.md for provenance). Not a test module itself -- no ``test_*``
functions live here, so pytest never collects it directly. Mirrors
``tests/unit/_form4_fixture_support.py``'s own ``FakeHttpClient``/
``FakeResult`` pattern exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

from investment_research.research.literature_acquisition_adapter import (
    EUROPEPMC_FULLTEXT_URL,
    EUROPEPMC_SEARCH_URL,
    NCBI_EFETCH_URL,
    NCBI_ESEARCH_URL,
)
from investment_research.schemas.enums import FetchOutcome

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "literature_real_format"


def fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def esearch_url(term: str, max_pmids: int) -> str:
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(max_pmids)}
    return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"


def efetch_url(pmids: list[str]) -> str:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    return f"{NCBI_EFETCH_URL}?{urlencode(params)}"


def europepmc_search_url(pmid: str) -> str:
    params = {"query": f"ext_id:{pmid} AND src:med", "format": "json"}
    return f"{EUROPEPMC_SEARCH_URL}?{urlencode(params)}"


def europepmc_fulltext_url(pmcid: str) -> str:
    return EUROPEPMC_FULLTEXT_URL.format(pmcid=pmcid)


def esearch_response(pmids: list[str]) -> str:
    """The real NCBI ESearch ``retmode=json`` response shape."""
    import json

    return json.dumps(
        {
            "header": {"type": "esearch", "version": "0.3"},
            "esearchresult": {
                "count": str(len(pmids)), "retmax": str(len(pmids)), "retstart": "0",
                "idlist": pmids,
            },
        }
    )


@dataclass
class FakeResult:
    ok: bool
    outcome: FetchOutcome
    _body: str
    error: str = ""

    @property
    def text(self) -> str:
        return self._body

    def json(self) -> object:
        """Mirrors ``collectors/http.py``'s real ``HttpResponse.json()``:
        malformed JSON returns ``None``, never raises."""
        import json

        if not self._body:
            return None
        try:
            return json.loads(self._body)
        except json.JSONDecodeError:
            return None


def ok(body: str) -> FakeResult:
    return FakeResult(ok=True, outcome=FetchOutcome.OK, _body=body)


def not_found(error: str = "404") -> FakeResult:
    return FakeResult(ok=False, outcome=FetchOutcome.NOT_FOUND, _body="", error=error)


def failed(outcome: FetchOutcome, error: str) -> FakeResult:
    return FakeResult(ok=False, outcome=outcome, _body="", error=error)


@dataclass
class FakeHttpClient:
    """Records every URL requested, in order, and returns a scripted
    response per URL -- so a test can assert exact URLs and exact request
    counts (never inferred). A URL with no registered response is treated
    as a 404, never silently synthesized."""

    responses: dict[str, FakeResult]
    requested_urls: list[str] = field(default_factory=list)
    #: Populated as requests are made, keyed by the SAME ``url`` argument
    #: passed to ``.get()`` -- mirrors ``AllowlistedHttpClient``'s private
    #: ``_cache`` well enough for shared, transport-agnostic diagnostics
    #: code (``_collect_transport_diagnostics``) to read structured
    #: per-URL outcomes generically, without a transport-specific code path
    #: for the real client vs. this fake one.
    _cache: dict[str, FakeResult] = field(default_factory=dict, init=False, repr=False)
    #: Invoked exactly once, immediately before the FIRST ``.get()`` call
    #: this instance ever serves -- mirrors ``AllowlistedHttpClient.
    #: on_first_attempt`` so offline tests can exercise marker-timing
    #: contracts without a real socket. ``None`` (the default) is a no-op,
    #: matching every existing caller's behaviour exactly.
    on_first_attempt: Callable[[], None] | None = None
    _first_attempt_signaled: bool = field(default=False, init=False, repr=False)

    def get(self, url: str, **kwargs: object) -> FakeResult:
        if self.on_first_attempt is not None and not self._first_attempt_signaled:
            self.on_first_attempt()
            self._first_attempt_signaled = True
        self.requested_urls.append(url)
        result = self.responses.get(url, not_found("no fake response registered for this URL"))
        self._cache[url] = result
        return result


@dataclass
class RaisingHttpClient:
    """A fake transport whose ``.get()`` RAISES instead of returning a
    result -- mirrors what a real ``urllib``-backed transport can do (a
    connection error, a malformed-response exception whose message embeds
    the full request URL). This is deliberately "the fake transport's
    lowest layer" that DOES see the raw wire URL (via ``requested_urls``,
    recorded before raising) and the exception message it raises -- exactly
    the layer Phase 3F.0.2's leakage tests are allowed to inspect for the
    secret; every object/string ABOVE this layer must not carry it."""

    requested_urls: list[str] = field(default_factory=list)
    #: When set, the exception message embeds this exact URL (as a raw
    #: ``urllib``-style exception would) -- lets a test assert the raw
    #: wire_url never survives past ``_safe_get``.
    exception_message_template: str = "urlopen error for {url}: [Errno -2] Name or service not known"
    #: Same contract as ``FakeHttpClient.on_first_attempt`` -- invoked once,
    #: immediately before the first ``.get()`` call, BEFORE the raise below.
    #: Lets a test prove that a marker-writing callback fires even when the
    #: transport itself then fails (an "attempt started, then failed" case,
    #: as opposed to a pre-send failure that never reaches this transport
    #: at all).
    on_first_attempt: Callable[[], None] | None = None
    _first_attempt_signaled: bool = field(default=False, init=False, repr=False)
    #: Mirrors ``FakeHttpClient``'s ``_cache`` -- always empty here, since
    #: this transport never returns a result to cache, but present so
    #: generic diagnostics code that does ``getattr(client, "_cache", None)``
    #: sees a well-formed (if empty) dict rather than nothing.
    _cache: dict[str, FakeResult] = field(default_factory=dict, init=False, repr=False)

    def get(self, url: str, **kwargs: object) -> FakeResult:
        if self.on_first_attempt is not None and not self._first_attempt_signaled:
            self.on_first_attempt()
            self._first_attempt_signaled = True
        self.requested_urls.append(url)
        raise ConnectionError(self.exception_message_template.format(url=url))
