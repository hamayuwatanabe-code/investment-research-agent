#!/usr/bin/env python3
"""Phase 3D: ClinicalTrials.gov-only live smoke test entry point. Run this
on a machine with real internet access to fetch ONE explicitly-supplied NCT
ID's registry record:

    python3 scripts/clinicaltrials_live_smoke.py --nct-id NCT01234567
    python3 scripts/clinicaltrials_live_smoke.py --nct-id NCT01234567 --out-dir /tmp/ct_smoke
    python3 scripts/clinicaltrials_live_smoke.py --nct-id NCT01234567 --force-rerun

To re-analyze an already-saved capture OFFLINE (no network, no marker,
safe to re-run any number of times):

    python3 scripts/clinicaltrials_live_smoke.py --nct-id NCT01234567 \
        --analyze-capture data/live_smoke/clinicaltrials/<TIMESTAMP>

See ``python3 scripts/clinicaltrials_live_smoke.py --help`` for every
option, and ``src/investment_research/research/clinicaltrials_live_smoke.py``'s
module docstring for exactly what this does and does not do:

* A single read-only GET to clinicaltrials.gov ONLY, hard-capped at 2 real
  requests per run.
* Zero Anthropic API calls, zero LLM calls, zero Web Search calls, no Full
  DD, and this NEVER connects to Pipeline.run() -- it is a standalone
  diagnostic script, never imported by production code.
* No NCT ID is ever hard-coded here or in production code -- you must pass
  one explicitly with --nct-id.
* Meant to run ONCE per NCT ID -- a marker file records the attempt and a
  second invocation refuses unless ``--force-rerun`` is passed deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from investment_research.research.clinicaltrials_live_smoke import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
