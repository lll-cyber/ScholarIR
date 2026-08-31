#!/usr/bin/env python3
"""Evaluate ScholarIR on PaSa Auto/Real ScholarQuery (macro set F1).

Writes:
  - detailed .log (per-sample understanding flow + retrieval + F1)
  - .json (macro + per-sample preds / understanding / retrieval_trace)

Usage:
  PYTHONPATH=src python3 scripts/eval_pasa.py --split auto --limit 5 --deepseek
  PYTHONPATH=src python3 scripts/eval_pasa.py --split real --limit 10 --deepseek
  PYTHONPATH=src python3 scripts/eval_pasa.py --split auto --limit 20 --out outputs/eval_auto20.json

  # nohup (unbuffered recommended):
  nohup env PYTHONPATH=src python3 -u scripts/eval_pasa.py --split auto --limit 5 --deepseek \
    > logs/eval_pasa_nohup.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir import run
from scholar_ir.config import LOG_ROOT, OUTPUT_ROOT, PASA_REAL_JSONL, PASA_TEST_JSONL
from scholar_ir.eval import load_pasa_jsonl, macro_average, score_sample
from scholar_ir.llm.deepseek_client import deepseek_configured
from scholar_ir.query_understanding.flow_log import (
    format_understanding_flow,
    understanding_to_debug_dict,
)


def _setup_logging(log_path: Path) -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_pasa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    u_log = logging.getLogger("scholar_ir.query_understanding")
    u_log.setLevel(logging.INFO)
    u_log.handlers.clear()
    u_log.addHandler(fh)
    u_log.propagate = False
    return logger


def _run_one(
    question: str,
    *,
    use_llm: bool,
    max_subqueries: int,
    sources: List[str],
    topk: int,
    max_return: int,
    arxiv_only: bool = True,
) -> Any:
    u_opts: Dict[str, Any] = {
        "use_llm": use_llm,
        "max_subqueries": max_subqueries,
        "max_lexical_swaps": 3,
        "enable_semantic": True,
        "verbose": True,
    }
    f_opts: Dict[str, Any] = {
        "max_return": max_return,
        "arxiv_only": arxiv_only,
        # 引用扩展：召回覆盖是当前瓶颈，从种子论文的 reference/citation 补池
        "expand_citations": True,
    }
    return run(
        question,
        {
            "understanding": u_opts,
            "query_understanding": u_opts,
            "retrieval": {"sources": sources, "per_query_topk": topk},
            "search": {"sources": sources, "per_query_topk": topk},
            "judge": f_opts,
            "filter": f_opts,
            "ranking": {"max_return": max_return, "arxiv_only": arxiv_only},
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="PaSa set-F1 eval")
    ap.add_argument("--split", choices=["auto", "real"], default="auto")
    ap.add_argument("--limit", type=int, default=5, help="max samples (default 5)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--max-subqueries", type=int, default=5)
    ap.add_argument("--max-return", type=int, default=20)
    ap.add_argument("--deepseek", action="store_true", help="Understanding via DeepSeek API")
    ap.add_argument("--heuristic", action="store_true", help="rule-based Understanding only")
    ap.add_argument(
        "--sources",
        default="arxiv,openalex",
        help="comma-separated: arxiv,openalex,semantic (default: arxiv,openalex)",
    )
    ap.add_argument(
        "--pred-all-ids",
        action="store_true",
        help="score with all pred ids (default: arxiv-shaped only)",
    )
    ap.add_argument(
        "--keep-non-arxiv",
        action="store_true",
        help="allow W…/S2-hash ids to occupy max_return (default: drop them)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sleep", type=float, default=0.2, help="pause between queries")
    ap.add_argument("--tag", default="", help="optional filename tag for log/json")
    args = ap.parse_args()

    use_llm = not args.heuristic
    if args.deepseek:
        use_llm = True
        os.environ.setdefault("SCHOLAR_IR_LLM_BACKEND", "deepseek")
    llm_mode = "heuristic"
    if use_llm:
        llm_mode = "deepseek" if deepseek_configured() else "llm(auto)"

    path = PASA_TEST_JSONL if args.split == "auto" else PASA_REAL_JSONL
    if not path.exists():
        raise SystemExit(f"missing dataset: {path}")

    rows = load_pasa_jsonl(path, limit=args.limit, offset=args.offset)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""

    log_path = LOG_ROOT / f"eval_pasa_{args.split}_n{args.limit}{tag}_{stamp}.log"
    logger = _setup_logging(log_path)
    logger.info("log file: %s", log_path)

    samples = []

    logger.info("=" * 72)
    logger.info(
        "split=%s n=%s sources=%s understanding=%s topk=%s path=%s",
        args.split,
        len(rows),
        sources,
        llm_mode,
        args.topk,
        path,
    )

    for i, row in enumerate(rows):
        logger.info("-" * 72)
        logger.info("[%s/%s] %s", i + 1, len(rows), row.qid)
        logger.info("Q: %s", row.question)

        t0 = time.time()
        retrieval_trace: list | None = None
        understanding_dbg: dict | None = None
        err: Optional[str] = None
        try:
            result = _run_one(
                row.question,
                use_llm=use_llm,
                max_subqueries=args.max_subqueries,
                sources=sources,
                topk=args.topk,
                max_return=args.max_return,
                arxiv_only=not args.keep_non_arxiv,
            )
            pred = result.paper_ids
            latency = int(
                result.metrics_local.get("latency_ms") or (time.time() - t0) * 1000
            )
            n_cand = len(result.retrieval.candidates)
            retrieval_trace = result.retrieval.trace
            understanding_dbg = understanding_to_debug_dict(result.understanding)
            flow = format_understanding_flow(
                result.understanding,
                title="[scholar] understanding flow",
            )
            logger.info("\n%s", flow)
            logger.info(
                "retrieval stats: %s",
                json.dumps(result.retrieval.stats, ensure_ascii=False),
            )
            for t in retrieval_trace or []:
                slim = {k: v for k, v in t.items() if k not in ("params", "_papers")}
                logger.info("  retrieval: %s", json.dumps(slim, ensure_ascii=False))
        except Exception as e:
            pred = []
            latency = int((time.time() - t0) * 1000)
            n_cand = 0
            err = str(e)
            logger.exception("FAILED: %s", e)

        trace_errors = []
        if retrieval_trace:
            for t in retrieval_trace:
                if t.get("error"):
                    trace_errors.append(
                        {"qid": t.get("qid"), "source": t.get("source"), "error": t["error"]}
                    )

        sample = score_sample(
            pred,
            row.gold_ids,
            qid=row.qid,
            pred_arxiv_only=not args.pred_all_ids,
            latency_ms=latency,
            extra={
                "n_candidates": n_cand,
                "error": err,
                "retrieval_errors": trace_errors or None,
                "understanding": understanding_dbg,
                "retrieval_trace": retrieval_trace,
            },
        )
        samples.append(sample)
        logger.info(
            "F1=%.3f P=%.3f R=%.3f pred=%s/%s gold=%s %sms%s",
            sample.f1,
            sample.precision,
            sample.recall,
            sample.n_pred_arxiv,
            sample.n_pred,
            sample.n_gold,
            latency,
            f" ERR={err}" if err else "",
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = macro_average(samples)
    logger.info("=" * 72)
    logger.info(
        "MACRO n=%s P=%.4f R=%.4f F1=%.4f mean_lat=%.0fms",
        summary.n,
        summary.macro_p,
        summary.macro_r,
        summary.macro_f1,
        sum(s.latency_ms for s in samples) / max(len(samples), 1),
    )

    out = args.out
    if out is None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_ROOT / f"eval_{args.split}_n{summary.n}{tag}_{stamp}.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stamp": stamp,
        "log": str(log_path),
        "split": args.split,
        "path": str(path),
        "sources": sources,
        "limit": args.limit,
        "offset": args.offset,
        "topk": args.topk,
        "max_subqueries": args.max_subqueries,
        "understanding": llm_mode,
        "use_llm": use_llm,
        "pred_arxiv_only": not args.pred_all_ids,
        "arxiv_only_truncate": not args.keep_non_arxiv,
        "macro": {
            "n": summary.n,
            "precision": summary.macro_p,
            "recall": summary.macro_r,
            "f1": summary.macro_f1,
            "mean_latency_ms": sum(s.latency_ms for s in samples) / max(len(samples), 1),
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

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)
    logger.info("DONE. log=%s", log_path)


if __name__ == "__main__":
    main()
