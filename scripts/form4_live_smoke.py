#!/usr/bin/env python3
"""Phase 3E.2: Form 4 structure-verification live smoke test entry point.
Run this on a machine where ``IRA_SEC_USER_AGENT`` is already set in the
shell environment (this tool never creates or requires a ``.env`` file,
and never prints the value):

    python3 scripts/form4_live_smoke.py --issuer-cik 320193
    python3 scripts/form4_live_smoke.py --issuer-cik 320193 --lookback-days 365 --max-filings 5
    python3 scripts/form4_live_smoke.py --issuer-cik 320193 --force-rerun

To re-analyze an already-saved capture directory OFFLINE (no network, no
marker, safe to re-run any number of times):

    python3 scripts/form4_live_smoke.py --analyze-capture data/live_smoke/form4/<TIMESTAMP>

See ``python3 scripts/form4_live_smoke.py --help`` for every option, and
``src/investment_research/research/form4_live_smoke.py``'s module
docstring for exactly what this does and does not do:

* Only ``--issuer-cik`` is required -- discovery is entirely issuer-
  driven, exactly like ``Form4Adapter`` itself. No accession, primary-
  document filename, or reporting-owner CIK is ever accepted as input.
* Read-only GET requests to data.sec.gov / www.sec.gov ONLY, hard-capped
  by a formula derived from ``--max-filings`` (printed in the plan before
  any request is sent), redirect targets re-validated against the same
  host allowlist.
* Zero Anthropic API calls, zero LLM calls, zero Web Search calls, no
  Full DD, and this NEVER connects to Pipeline.run() -- it is a
  standalone diagnostic script, never imported by production code.
* Never promotes anything in ``research/source_routing_catalog.py`` to
  ``LIVE_VERIFIED`` -- that module is never imported here.
* Refuses outright, before any network call, if IRA_SEC_USER_AGENT is
  unset, blank, or still the repository's built-in placeholder value.
* If the issuer's own submissions produce no Form 4/4-A filings at all,
  reports ``DISCOVERY_NOT_SUPPORTED`` and stops -- never guesses a
  different CIK.
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
