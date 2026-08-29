#!/usr/bin/env python3
"""Evaluate SPAR baseline on PaSa Auto/Real ScholarQuery (macro set F1).

Uses baselines/SPAR full pipeline with DeepSeek LLM (from ScholarIR/.env).

Usage:
  PYTHONPATH=src python3 scripts/eval_spar.py --split auto --limit 5
  PYTHONPATH=src python3 scripts/eval_spar.py --split real --limit 5 --out outputs/eval_spar_real5_deepseek.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SPAR_ROOT = ROOT.parent / "baselines" / "SPAR"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SPAR_ROOT))

from scholar_ir.config import OUTPUT_ROOT, PASA_REAL_JSONL, PASA_TEST_JSONL, _load_env_files
from scholar_ir.eval import arxiv_only, load_pasa_jsonl, macro_average, score_sample
from scholar_ir.llm.deepseek_client import deepseek_configured

_load_env_files()


def _setup_spar_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("SCHOLAR_IR_LLM_BACKEND", "deepseek")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    os.environ.setdefault("LLM_MODEL_NAME", model)

    import global_config as gc

    gc.LLM_MODEL_NAME = model
    gc.RERANK_MODEL = model
    gc.SEARCH_ROUTES = [s.strip() for s in args.sources.split(",") if s.strip()]
    gc.SEARCH_ROUTE = gc.SEARCH_ROUTES
    gc.QUERY_NUM_PRUNED = args.query_num_pruned
    gc.DO_REFERENCE_SEARCH = False
    gc.ENABLE_RERANK = False


def _import_spar_tree():
    from pipeline_spar import AcademicSearchTree

    return AcademicSearchTree


def _pred_ids_from_spar(docs: Dict[str, Any], max_return: int) -> List[str]:
    ranked = sorted(
        docs.values(),
        key=lambda d: float(d.get("sim_score", 0) or 0),
        reverse=True,
    )
    out: List[str] = []
    seen: set[str] = set()
    for doc in ranked:
        aid = str(doc.get("arxivId") or doc.get("paper_id") or "").split("v")[0].strip()
        if not aid or aid in seen:
            continue
        if arxiv_only([aid]):
            out.append(aid)
            seen.add(aid)
        if len(out) >= max_return:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SPAR baseline eval on PaSa jsonl")
    ap.add_argument("--split", choices=["auto", "real"], default="auto")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-return", type=int, default=20, help="max pred arxiv ids")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-docs", type=int, default=10, help="SPAR relevance_doc_num")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--query-num-pruned", type=int, default=2)
    ap.add_argument(
        "--sources",
        default="arxiv",
        help="SPAR SEARCH_ROUTES (comma-separated), default arxiv only for fair compare",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    if not deepseek_configured():
        raise SystemExit("DEEPSEEK_API_KEY not set (load ScholarIR/.env)")

    _setup_spar_env(args)
    AcademicSearchTree = _import_spar_tree()

    path = PASA_TEST_JSONL if args.split == "auto" else PASA_REAL_JSONL
    if not path.exists():
        raise SystemExit(f"missing dataset: {path}")

    rows = load_pasa_jsonl(path, limit=args.limit, offset=args.offset)
    samples = []
    llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    print(
        f"SPAR eval split={args.split} n={len(rows)} sources={args.sources} "
        f"llm={llm_model} max_depth={args.max_depth} path={path}"
    )

    for i, row in enumerate(rows):
        t0 = time.time()
        err = None
        pred: List[str] = []
        n_cand = 0
        trace: Dict[str, Any] = {}

        try:
            agent = AcademicSearchTree(
                max_depth=args.max_depth,
                max_docs=args.max_docs,
                similarity_threshold=args.score_thresh,
            )
            docs = agent.search(row.question, end_date="")
            n_cand = len(docs)
            pred = _pred_ids_from_spar(docs, args.max_return)
            trace = {
                "n_searched_docs": len(agent.root.searched_docs),
                "expanded_queries_info": agent.root.extra.get("expanded_queries_info"),
                "n_returned": len(docs),
            }
        except Exception as e:
            err = str(e)
            traceback.print_exc()

        latency = int((time.time() - t0) * 1000)
        sample = score_sample(
            pred,
            row.gold_ids,
            qid=row.qid,
            pred_arxiv_only=True,
            latency_ms=latency,
            extra={
                "n_candidates": n_cand,
                "error": err,
                "spar_trace": trace,
            },
        )
        samples.append(sample)
        print(
            f"[{i+1}/{len(rows)}] {row.qid}  "
            f"F1={sample.f1:.3f} P={sample.precision:.3f} R={sample.recall:.3f}  "
            f"pred={sample.n_pred_arxiv}/{sample.n_pred} gold={sample.n_gold}  "
            f"cand={n_cand} {latency}ms"
            + (f" ERR={err}" if err else "")
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = macro_average(samples)
    print("=" * 60)
    print(
        f"MACRO n={summary.n}  "
        f"P={summary.macro_p:.4f}  R={summary.macro_r:.4f}  F1={summary.macro_f1:.4f}"
    )
    print(
        f"mean latency_ms={sum(s.latency_ms for s in samples) / max(len(samples), 1):.0f}"
    )

    payload = {
        "baseline": "SPAR",
        "split": args.split,
        "path": str(path),
        "sources": [s.strip() for s in args.sources.split(",") if s.strip()],
        "limit": args.limit,
        "offset": args.offset,
        "llm": llm_model,
        "spar_config": {
            "max_depth": args.max_depth,
            "max_docs": args.max_docs,
            "score_thresh": args.score_thresh,
            "query_num_pruned": args.query_num_pruned,
        },
        "pred_arxiv_only": True,
        "macro": {
            "n": summary.n,
            "precision": summary.macro_p,
            "recall": summary.macro_r,
            "f1": summary.macro_f1,
        },
        "samples": [
            {
                "qid": s.qid,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
                "n_pred": s.n_pred,
                "n_pred_arxiv": s.n_pred_arxiv,
                "n_gold": s.n_gold,
                "paper_ids": s.paper_ids,
                "latency_ms": s.latency_ms,
                "extra": s.extra,
            }
            for s in samples
        ],
    }

    out = args.out
    if out is None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_ROOT / f"eval_spar_{args.split}_n{summary.n}_deepseek.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
