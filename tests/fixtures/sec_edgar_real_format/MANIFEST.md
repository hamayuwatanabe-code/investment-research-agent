# SEC EDGAR real-format fixtures (Phase 3B)

**Provenance**: every file in this directory is **synthetic** content shaped
to match SEC EDGAR's publicly documented response formats (the submissions
JSON schema, the `index.json` directory listing schema, and the
`{accession}-index.htm` "Document Format Files" table layout). None of it was
captured from a live request — this session has no live network access, and
Phase 3B forbids one. `Document.provenance` for anything built from these
fixtures is therefore `Provenance.FIXTURE`, never `Provenance.CAPTURED`
(`CAPTURED` means a real document, captured at a stated time, replayed —
that claim would be false here).

The issuer is a wholly fictional "Generic Biotech Holdings, Inc." (ticker
`TESTCO`, CIK `9999999`, matching the placeholder already used by
`tests/unit/test_sec_acquisition_adapters.py`). No real company, ticker, CIK,
or accession number is used anywhere in this directory. No secret or
credential-shaped value appears in any file.

Generated: this Phase 3B session (see the owning commit for the exact date).

## Files

| File | Mirrors | Notes |
|---|---|---|
| `submissions_testco.json` | `GET https://data.sec.gov/submissions/CIK{cik:010d}.json` | `filings.recent` parallel arrays, including `isInlineXBRL` |
| `directory_index_10k.json` | `GET .../Archives/edgar/data/{cik}/{accession-nodash}/index.json` | Real limitation modeled: `type` is populated for exhibits/primary doc but **not** for XBRL viewer report files (`R1.htm`...), and there is **no** Sequence/Description field at all |
| `filing_detail_10k.htm` | `GET .../Archives/edgar/data/{cik}/{accession-nodash}/{accession}-index.htm` | The "Document Format Files" `<table>` -- the only reliable source of Sequence + Description together |
| `primary_10k_ixbrl.htm` | The 10-K primary document itself | Inline XBRL: visible `<ix:nonFraction>`/`<ix:nonNumeric>` tags plus a non-visible `<ix:header>` block that must never appear in extracted text |
| `exhibit_99_1_press_release.htm` | EX-99.1 | Narrative, company-authored |
| `exhibit_10_1_material_agreement.htm` | EX-10.1 | Narrative, company-authored |
| `exhibit_31_1_certification.htm` | EX-31.1 | Boilerplate SOX certification -- fetchable but never prioritized |
| `error_page_undeclared_automated_tool.htm` | SEC's automated-traffic block page | Must never be accepted as a body |
| `error_page_rate_limited.htm` | A generic rate-limit/503-style HTML error page | Must never be accepted as a body |
| `index_page_directory_listing.htm` | The bare Apache-style "Index of /Archives/..." page (not the filing detail page) | Must never be accepted as `FULL_DOCUMENT` |
| `empty_document.htm` | A `<html><body></body></html>` stub | Below the minimum body length |
| `too_short_body.htm` | A few words only | Below the minimum body length |

## Known limitation (documented, not silently worked around)

SEC's real `index.json` does not carry Sequence or Description for any item,
and its `type` field is unreliable for XBRL viewer artifacts. The adapter
therefore treats `index.json` as authoritative only for
filename/size/last-modified, and fetches the filing detail HTML
(`{accession}-index.htm`) to obtain Sequence + Description + a trustworthy
Type. If the detail HTML fetch fails, the adapter falls back to
`index.json`-only classification (Type only, Sequence=0, Description=UNKNOWN)
and records this degradation explicitly in the LOCATE result's payload
(`"detail_html_unavailable": true`) rather than silently proceeding as if
nothing were missing.
