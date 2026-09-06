#!/usr/bin/env python3
"""Entry point.

    python main.py DEMOBIO --fixtures
    python main.py CRBP --full-dd

See ``python main.py --help`` for the full set of modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from investment_research.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
