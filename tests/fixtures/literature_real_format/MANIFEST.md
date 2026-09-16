# PubMed / Europe PMC real-format fixtures (Phase 3F)

**Provenance**: every file in this directory is **synthetic** XML/JSON
shaped to match NCBI E-utilities' publicly documented PubMed XML response
schema (https://www.ncbi.nlm.nih.gov/books/NBK25497/) and the Europe PMC
RESTful Web Service's documented response shapes
(https://europepmc.org/RestfulWebService). None of it was captured from a
live request -- this session has no live network access, and Phase 3F
forbids one. `Document.provenance` for anything built from these fixtures
is `LIVE` only in the sense that
`research/literature_acquisition_adapter.py`'s own code sets it
unconditionally on a successful fetch (mirroring the SEC/ClinicalTrials/
Form4 adapters exactly) -- every test in this repository drives that
adapter against an injected fake HTTP double, never the real internet, so
no *actual* live network call is ever made from a test.

Every PMID (`900000xx`), NCT ID (`NCT0999xxxx`), PMCID (`PMC999xxxx`), DOI
(`10.9999/fict.*`), author name, journal title, and institution name in
this directory is wholly fictional. No real company, drug, ticker, trial,
or person is named anywhere. No secret or credential-shaped value appears
in any file.

## Files

| File | Scenario | Notes |
|---|---|---|
| `normal_abstract.xml` | Normal PubMed article with an unstructured abstract | PMID `90000001` |
| `structured_abstract.xml` | Structured abstract (`BACKGROUND`/`METHODS`/`RESULTS`/`CONCLUSIONS` labels) | PMID `90000002`; exercises study-design/reported-result/endpoint wording separation |
| `no_abstract.xml` | No `<Abstract>` element at all | PMID `90000003`; must resolve to `ContentKind.METADATA_ONLY`, never a guessed abstract |
| `multiple_authors.xml` | Multiple authors, multiple affiliations per author, and a `CollectiveName` group author | PMID `90000004` |
| `sponsor_funded.xml` | `GrantList` naming a sponsor as the funding agency | PMID `90000005`; `company_claim` must still be `False` |
| `conflict_of_interest.xml` | A genuine `CoiStatement` | PMID `90000006` |
| `nct_id_present.xml` | `DataBankList`/`DataBank[DataBankName="ClinicalTrials.gov"]`/`AccessionNumber` carrying `NCT09990001` | PMID `90000007` |
| `ids_complete.xml` | PMID, PMCID, and DOI all populated in `ArticleIdList` | PMID `90000008`; paired with `europepmc_search_oa.json`/`europepmc_fulltext_oa.xml` for the open-access full-text scenario |
| `correction.xml` | `CommentsCorrections RefType="CorrectionIn"` | PMID `90000009`; related PMID `90000019` (no fixture body needed -- the relation itself is what's tested) |
| `erratum.xml` | `CommentsCorrections RefType="ErratumIn"` | PMID `90000010`; related PMID `90000020` |
| `retracted.xml` | `CommentsCorrections RefType="RetractionIn"` | PMID `90000011`; related PMID `90000021`; facts are still generated, never decision-grade |
| `malformed.xml` | Not well-formed XML (an unclosed tag) | PMID `90000012` never actually parses |
| `html_error_page.htm` | An HTML page (e.g. a 503) served instead of XML | Must never be accepted as PubMed/Europe-PMC-fulltext XML |
| `empty_body.xml` | Zero-length response body | Must never be accepted as a body |
| `mismatched_pmid.xml` | The XML's own `<PMID>` (`90000099`) deliberately differs from the PMID a test requests (`90000013`) | Exercises PMID-mismatch -> `FAILED` handling |
| `updated_article_v2.xml` | Same PMID as `normal_abstract.xml` (`90000001`), genuinely different abstract text | A re-fetch with different content -- must form a new `DocumentStore` version, never overwrite the original |
| `batch_articleset.xml` | A single `PubmedArticleSet` response containing TWO `PubmedArticle` entries (PMIDs `90000015`/`90000016`) | Proves batch EFetch (never one PMID at a time) |
| `europepmc_search_oa.json` | Europe PMC `/search` response: `isOpenAccess=Y`, `inEPMC=Y`, PMCID `PMC9990008` | Paired with `ids_complete.xml` (same PMID `90000008`) |
| `europepmc_search_non_oa.json` | Europe PMC `/search` response: `isOpenAccess=N`, `inEPMC=N` | Paired with `normal_abstract.xml` (PMID `90000001`) -- full text must never be marked acquired |
| `europepmc_fulltext_oa.xml` | A JATS-like `<article><body><sec>...</sec></body></article>` open-access full text | Genuinely distinct sections from any PubMed abstract -- never conflated |

A "duplicate article" scenario (the same PMID requested twice, e.g. via two
different `EvidenceRequirement`s) deliberately reuses `normal_abstract.xml`
rather than adding a byte-identical second file -- the interesting behavior
under test is the adapter's own dedup (`ExecutionContext.request_cache`
under the `literature_pmid:<pmid>` key), not the fixture content.
