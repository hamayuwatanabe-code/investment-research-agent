# SEC Form 4 (Section 16 ownership) real-format fixtures (Phase 3E, extended
# in Phase 3E.1)

**Provenance**: every file in this directory is **synthetic** XML/HTML
shaped to match SEC EDGAR's publicly documented `ownershipDocument` XML
schema (the same schema underlies Form 3, Form 4, and Form 5, distinguished
by `documentType`). None of it was captured from a live request — this
session has no live network access, and Phase 3E/3E.1 forbid one.
`Document.provenance` for anything built from these fixtures is `LIVE`
only in the sense that `research/form4_acquisition_adapter.py`'s own code
sets it unconditionally on a successful fetch (mirroring the SEC/
ClinicalTrials adapters exactly) — every test in this repository drives
that adapter against an injected `FakeHttpClient`, never the real
internet, so no *actual* live network call is ever made from a test.

The issuer is a wholly fictional "Sample Biotech Holdings, Inc." (ticker
`SAMPB`, CIK `0001112223`); the reporting owner is a wholly fictional
"Sample Reporting Person" (CIK `0005556667`). Since Phase 3E.1, every
accession lives under the ISSUER's own CIK (`0001112223`) -- issuer-driven
discovery queries the issuer's own submissions.json, never a separately
supplied "filer" CIK (see `research/form4_acquisition_adapter.py`'s module
docstring for why "filer_cik" was retired in favor of "issuer_cik"). No
real company, ticker, CIK, insider name, or accession number is used
anywhere in this directory or in the tests that consume it. No secret or
credential-shaped value appears in any file.

`submissions.json`/`index.json`/`company_tickers.json` response bodies are
built programmatically by `tests/unit/_form4_fixture_support.py` (the
exact real parallel-array `filings.recent` shape
`tests/fixtures/sec_edgar_real_format` already models as static files)
rather than duplicated here as dozens of near-identical static JSON files
— only the ownership XML/HTML bodies below are static.

## Files

| File | Scenario | Notes |
|---|---|---|
| `normal_market_purchase.xml` | Ordinary open-market purchase | `transactionCode=P`, `acquired_or_disposed=A` |
| `normal_market_sale.xml` | Ordinary open-market sale | `transactionCode=S`, `acquired_or_disposed=D` |
| `option_exercise.xml` | Exercise of a derivative security | Derivative table, `transactionCode=M` |
| `tax_withholding.xml` | Payment/withholding of tax via securities | Non-derivative, `transactionCode=F` |
| `grant_award.xml` | Grant or award | Non-derivative, `transactionCode=A` |
| `gift.xml` | Bona fide gift | Non-derivative, `transactionCode=G` |
| `derivative_transaction.xml` | Purchase of a derivative security | `transactionCode=P` on the derivative table, full `exercise_price`/`exercise_date`/`expiration_date`/`underlying_security_*` |
| `indirect_ownership.xml` | Indirect beneficial ownership | `directOrIndirectOwnership=I`, `natureOfOwnership="By Family Trust"` |
| `multiple_transactions.xml` | Two transactions in one filing | Proves transactions are never merged/summed |
| `footnote_10b5_1.xml` | A transaction explicitly tied to a Rule 10b5-1 plan | The transaction's `footnoteId` references a footnote whose text says so explicitly |
| `form4a_amendment.xml` | A Form 4/A amendment | `documentType=4/A`, filed under its own accession -- never treated as an overwrite of an original Form 4 |
| `missing_fields.xml` | Optional fields genuinely absent | No `issuerTradingSymbol`, no `officerTitle` -- both must resolve to `UNKNOWN`, never guessed |
| `missing_required_fields.xml` | REQUIRED field absent | No `issuerCik` at all -- PARSE must fail distinctly from FETCH's shape gate |
| `body_says_form3.xml` | Submissions metadata says Form 4, XML body says Form 3 | Caught at FETCH (defense in depth beyond LOCATE's own metadata cross-check) |
| `missing_ownership_document.xml` | Wrong XML root entirely | Root is `<edgarSubmission>`, not `<ownershipDocument>` |
| `malformed_xml.xml` | Not well-formed XML | An unclosed tag |
| `html_error_page.htm` | SEC's automated-traffic block page, served instead of XML | Must never be accepted as an ownership document body |
| `empty_body.xml` | Zero-length response body | Must never be accepted as a body |
| `cik_mismatch.xml` | `issuerCik` disagrees with the requested issuer | `0009998887` instead of `0001112223` -- excluded at FETCH, never stored as ACQUIRED evidence (Phase 3E.1) |
| `reconciled_amendment.xml` | A Form 4/A whose remarks explicitly reference `normal_market_purchase`'s own accession | `documentType=4/A`; reconciliation resolves to RECONCILED (Phase 3E.1) |
| `older_filing_on_continuation_page.xml` | A filing reachable ONLY via a `filings.files` continuation page, never `filings.recent` | Proves pagination is followed, not just the first ~1000 filings (Phase 3E.1) |
| `checkbox_and_date_present.xml` | A document-level Rule 10b5-1 checkbox (`aff10b5One=1`) AND a `dateOfOriginalSubmission` element, both genuinely present | Used ONLY by `tests/unit/test_form4_live_smoke_offline.py`'s raw-XML diagnostic scanner tests (Phase 3E.2) -- deliberately NOT added to `_form4_fixture_support.py`'s shared `_SCENARIOS`/`CLEANLY_ACQUIRABLE_SCENARIOS`, since several Phase 3E.1 tests assert that no shared fixture has this checkbox present |
| `checkbox_false_no_date.xml` | A document-level Rule 10b5-1 checkbox with raw value `0` (no `dateOfOriginalSubmission`) | Mirrors the real Phase 3E.3 Mac Live Smoke finding: a normal Form 4 with `aff10b5One=0` parses to `False`, never `None`; also NOT added to the shared `_SCENARIOS` for the same reason as above |
| `amendment_with_date_of_original_submission.xml` | A Form 4/A with a genuinely present `dateOfOriginalSubmission`, remarks referencing no other accession | `documentType=4/A`; reconciliation resolves to UNRESOLVED; used by Phase 3E.3's targeted-mode / Form 4/A diagnostic tests, not added to the shared `_SCENARIOS` |

(The `metadata_says_form3` scenario -- SEC's own submissions metadata
records the accession as Form 3, not Form 4/4-A -- and the
`unknown_accession` scenario need no XML body at all: both are refused by
LOCATE before any body would ever be fetched. `form4a_amendment.xml`
-- unchanged from Phase 3E -- deliberately references NO accession in its
remarks, so its own reconciliation resolves to UNRESOLVED, never guessed
related to any other filing by period/owner coincidence.)
