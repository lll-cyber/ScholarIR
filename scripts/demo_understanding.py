#!/usr/bin/env python3
"""Demo Query Understanding on samples from PaSa Auto/RealScholarQuery test sets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.config import PASA_REAL_JSONL, PASA_TEST_JSONL
from scholar_ir.query_understanding import understand
from scholar_ir.query_understanding.flow_log import format_understanding_flow

# How many lines to take from the head of each file
N_AUTO = 3
N_REAL = 3


def load_samples(path: Path, n: int, source: str) -> list[dict]:
    if not path.exists():
        print(f"[warn] missing {path}", file=sys.stderr)
        return []
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            rows.append(
                {
                    "source": source,
                    "qid": row.get("qid", f"{source}_{i}"),
                    "question": row["question"],
                }
            )
    return rows


def main():
    use_llm = "--llm" in sys.argv or "--local" in sys.argv or "--deepseek" in sys.argv
    use_heuristic = "--heuristic" in sys.argv
    if use_heuristic:
        use_llm = False
    if "--deepseek" in sys.argv:
        os.environ.setdefault("SCHOLAR_IR_LLM_BACKEND", "deepseek")
    samples = load_samples(PASA_REAL_JSONL, N_REAL, "RealScholarQuery") + load_samples(
        PASA_TEST_JSONL, N_AUTO, "AutoScholarQuery"
    )
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        sys.exit(1)

    for item in samples:
        q = item["question"]
        r = understand(q, {"use_llm": use_llm, "max_subqueries": 5, "verbose": True})
        print("=" * 60)
        print(f"[{item['source']}] {item['qid']}")
        print(format_understanding_flow(r))
        print(
            "criteria:",
            [
                f"{c.get('type') or c.get('name')}={c.get('text') or c.get('description')}"
                for c in r.relevance_criteria
                if c.get("weight", 0) > 0 or c.get("required")
            ],
        )


if __name__ == "__main__":
    main()
