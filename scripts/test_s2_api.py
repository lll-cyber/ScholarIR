#!/usr/bin/env python3
"""Quick S2 API smoke (loads ScholarIR/.env).

  PYTHONPATH=src python3 scripts/test_s2_api.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "tests" / "test_api" / "test_s2_api.py"), run_name="__main__")
