"""Shared support for tests driving ``PubMedLiteratureAdapter``/
``EuropePmcFullTextAdapter`` against the real-format fixtures in
``tests/fixtures/literature_real_format/`` (see that directory's
MANIFEST.md for provenance). Not a test module itself -- no ``test_*``
functions live here, so pytest never collects it directly. Mirrors
``tests/unit/_form4_fixture_support.py``'s own ``FakeHttpClient``/
``FakeResult`` pattern exactly.
"""

from __future__ import annotations

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
    return EUROPEPMC_FULLTEXT_URL.format(source="PMC", pmcid=pmcid)


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

    def get(self, url: str, **kwargs: object) -> FakeResult:
        self.requested_urls.append(url)
        return self.responses.get(url, not_found("no fake response registered for this URL"))


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

    def get(self, url: str, **kwargs: object) -> FakeResult:
        self.requested_urls.append(url)
        raise ConnectionError(self.exception_message_template.format(url=url))
