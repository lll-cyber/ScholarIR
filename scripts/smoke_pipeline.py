"""Smoke: full pipeline on one PaSa query.

Usage:
  PYTHONPATH=src python3 scripts/smoke_pipeline.py
  PYTHONPATH=src python3 scripts/smoke_pipeline.py --deepseek
  PYTHONPATH=src python3 scripts/smoke_pipeline.py --qid AutoScholarQuery_test_3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir import run
from scholar_ir.config import DEFAULT_PER_QUERY_TOPK, PASA_TEST_JSONL
from scholar_ir.eval import arxiv_only, score_sample
from scholar_ir.llm.deepseek_client import deepseek_configured


def _load_row(path: Path, qid: str | None):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if qid is None:
                return row
            if row.get("qid") == qid:
                return row
    raise SystemExit(f"qid not found: {qid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None, help="PaSa AutoScholar qid")
    ap.add_argument("--topk", type=int, default=DEFAULT_PER_QUERY_TOPK)
    ap.add_argument("--max-subqueries", type=int, default=3)
    ap.add_argument("--deepseek", action="store_true", help="Understanding via DeepSeek API")
    ap.add_argument("--heuristic", action="store_true", help="rule-based Understanding only")
    ap.add_argument(
        "--sources",
        default="arxiv,openalex",
        help="Comma-separated retrieval sources (default: arxiv,openalex)",
    )
    ap.add_argument(
        "--expand",
        action="store_true",
        help="Enable citation expansion in filter",
    )
    args = ap.parse_args()

    use_llm = not args.heuristic
    if args.deepseek:
        use_llm = True
        os.environ.setdefault("SCHOLAR_IR_LLM_BACKEND", "deepseek")
    if use_llm and not deepseek_configured() and os.getenv("SCHOLAR_IR_LLM_BACKEND", "auto") == "deepseek":
        print("[warn] DEEPSEEK_API_KEY missing; may fall back to heuristic", file=sys.stderr)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    question = "papers about retrieval-augmented generation for question answering"
    gold_ids: list = []
    qid = "manual"
    if PASA_TEST_JSONL.exists():
        row = _load_row(PASA_TEST_JSONL, args.qid)
        question = row["question"]
        gold_ids = row.get("answer_arxiv_id") or []
        qid = row.get("qid", qid)

    print("understanding:", "deepseek" if use_llm else "heuristic")
    print("sources:", sources)
    print("expand:", args.expand)
    result = run(
        question,
        {
            "understanding": {"use_llm": use_llm, "max_subqueries": args.max_subqueries},
            "retrieval": {
                "sources": sources,
                "per_query_topk": args.topk,
            },
            "judge": {
                "max_return": 20,
                "expand_citations": args.expand,
                "seed_top_k": 5,
                "ref_limit": 10,
                "cit_limit": 10,
                "expand_max_total": 50,
                "use_llm_for_expanded": True,
                "llm_top_k_expanded": 15,
            },
        },
    )

    sample = score_sample(
        result.paper_ids,
        gold_ids,
        qid=qid,
        pred_arxiv_only=True,
        latency_ms=int(result.metrics_local.get("latency_ms") or 0),
    )

    print("qid:", qid)
    print("question:", question[:160])
    print("intent:", result.understanding.intent)
    print("slots:", result.understanding.slots)
    print("sub_queries:", [sq.text for sq in result.understanding.sub_queries])
    print("n_candidates:", len(result.retrieval.candidates))
    print("paper_ids (all):", result.paper_ids)
    print("paper_ids (arxiv):", arxiv_only(result.paper_ids))
    print("gold_ids:", gold_ids)
    print(
        f"P/R/F1 (arxiv-only pred): "
        f"{sample.precision:.4f} {sample.recall:.4f} {sample.f1:.4f}"
    )
    print("metrics:", result.metrics_local)
    print("--- titles ---")
    for p in result.retrieval.candidates[:8]:
        print(f"  [{p.paper_id}] {p.title[:90]}")


if __name__ == "__main__":
    main()
