#!/usr/bin/env python3
"""Phase 3E.2/3E.3: Form 4 structure-verification live smoke test entry
point. Run this on a machine where ``IRA_SEC_USER_AGENT`` is already set
in the shell environment (this tool never creates or requires a ``.env``
file, and never prints the value):

    python3 scripts/form4_live_smoke.py --issuer-cik 320193
    python3 scripts/form4_live_smoke.py --issuer-cik 320193 --lookback-days 365 --max-filings 5
    python3 scripts/form4_live_smoke.py --issuer-cik 320193 --json
    python3 scripts/form4_live_smoke.py --issuer-cik 320193 --force-rerun

Targeted mode (Phase 3E.3) -- verify exactly ONE known accession (e.g. a
real Form 4/A) in at most 3 real GETs, never a broader sweep:

    python3 scripts/form4_live_smoke.py --issuer-cik 320193 --accession 0000320193-26-000001

To re-analyze an already-saved capture directory OFFLINE (no network, no
marker, safe to re-run any number of times):

    python3 scripts/form4_live_smoke.py --analyze-capture data/live_smoke/form4/<TIMESTAMP>

See ``python3 scripts/form4_live_smoke.py --help`` for every option, and
``src/investment_research/research/form4_live_smoke.py``'s module
docstring for exactly what this does and does not do:

* Only ``--issuer-cik`` is required for discovery mode -- discovery is
  entirely issuer-driven, exactly like ``Form4Adapter`` itself. No
  primary-document filename or reporting-owner CIK is ever accepted as
  input, in either mode. Targeted mode additionally takes ``--accession``
  -- primaryDocument is still always resolved from SEC's own submissions
  metadata, never from a CLI argument.
* Read-only GET requests to data.sec.gov / www.sec.gov ONLY. Discovery
  mode's cap is a formula derived from ``--max-filings``; targeted mode's
  cap is a fixed 3 (submissions, directory index, raw XML body) with no
  ``filings.files`` continuation-page search. Both are printed in the
  plan before any request is sent; redirect targets are re-validated
  against the same host allowlist in both modes.
* Zero Anthropic API calls, zero LLM calls, zero Web Search calls, no
  Full DD, and this NEVER connects to Pipeline.run() -- it is a
  standalone diagnostic script, never imported by production code.
* Never promotes anything in ``research/source_routing_catalog.py`` to
  ``LIVE_VERIFIED`` -- that module is never imported here.
* Refuses outright, before any network call, if IRA_SEC_USER_AGENT is
  unset, blank, or still the repository's built-in placeholder value.
* Discovery mode: if the issuer's own submissions produce no Form 4/4-A
  filings at all, reports ``DISCOVERY_NOT_SUPPORTED`` and stops -- never
  guesses a different CIK. Targeted mode: an accession not present in
  ``filings.recent`` reports ``ACCESSION_NOT_FOUND``, never a
  continuation-page search.
* ``--json`` prints the FULL report (every discovery candidate included,
  never truncated by the human-readable summary) as JSON.
* Meant to run ONCE -- a marker file records the attempt (only once real
  communication began) and a second invocation refuses unless
  ``--force-rerun`` is passed deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from investment_research.research.form4_live_smoke import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
