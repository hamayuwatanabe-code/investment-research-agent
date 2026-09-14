# ClinicalTrials.gov API v2 fixtures (Phase 3D)

All studies here are **synthetic** -- no real sponsor, drug, condition, NCT
ID, or trial. None of these NCT IDs correspond to a real registered study;
they were chosen from an unused numeric range purely to look shaped like a
real one (`NCT#########`). Never present these as research, and never treat
their content as describing anything real.

These model the real ClinicalTrials.gov API v2 **single-study** endpoint
(`GET https://clinicaltrials.gov/api/v2/studies/{nctId}`), whose response is
one study record's JSON directly -- `{"protocolSection": {...}, "hasResults":
bool, ...}` -- NOT wrapped in the list endpoint's `{"studies": [...],
"nextPageToken": ...}` envelope. `collectors/clinicaltrials.py`'s existing
`parse_study()` already models this same per-study shape for an item of the
list endpoint's `studies` array; the single-study endpoint returns that
identical shape directly.

| File | Models |
|---|---|
| `study_recruiting_interventional.json` | Normal case: INTERVENTIONAL, Phase 2, RECRUITING, no results yet, full design/eligibility/locations/references modules populated. |
| `study_recruiting_interventional_v2.json` | The SAME NCT ID as the file above, with `overallStatus` changed to `ACTIVE_NOT_RECRUITING`, a higher enrollment count, and a later `lastUpdatePostDateStruct` -- an updated version of the same study, for `DocumentStore` versioning. |
| `study_completed_with_results.json` | COMPLETED, `hasResults: true`, with a minimal `resultsSection` (presence only -- this repository never interprets result content). |
| `study_terminated.json` | TERMINATED, with `whyStopped` populated. |
| `study_withdrawn.json` | WITHDRAWN, zero enrollment, no start date (the study never started). |
| `study_missing_status_and_dates.json` | `overallStatus` and every date struct absent entirely -- proves the parser never guesses a status or a date it was not given. |
| `study_partial_dates.json` | Every date struct present but partial (`"2026"`, `"2026-03"`) -- proves partial CT.gov dates are passed through verbatim, never padded/completed. |
| `study_malformed.txt` | Deliberately invalid JSON syntax (truncated) -- a schema/parse error, distinct from a well-formed-but-wrong-shape payload (constructed inline in tests) and from an HTTP-level failure (404/429/5xx/timeout, exercised via the fake transport's own outcomes, not a fixture file). |

None of these fixtures is ever fetched over a real network connection by
this repository's tests -- see `tests/unit/_clinicaltrials_fixture_support.py`
and `tests/unit/test_clinicaltrials_acquisition_adapter.py`.
