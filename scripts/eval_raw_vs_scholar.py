#!/usr/bin/env python3
"""End-to-end PaSa eval: raw query baseline vs ScholarIR understanding.

Writes:
  - detailed .log (per-sample understanding flow + F1)
  - .json (macro + per-sample preds / understanding / retrieval_trace)

Usage:
  PYTHONPATH=src python3 scripts/eval_raw_vs_scholar.py --split auto --limit 5 --deepseek
  PYTHONPATH=src python3 scripts/eval_raw_vs_scholar.py --split real --limit 5 --deepseek
  PYTHONPATH=src python3 scripts/eval_raw_vs_scholar.py --split both --limit 5 --deepseek
  # single source: --sources arxiv
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
    logger = logging.getLogger("eval_raw_vs_scholar")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    # Also pipe understanding verbose logs into the same file
    u_log = logging.getLogger("scholar_ir.query_understanding")
    u_log.setLevel(logging.INFO)
    u_log.handlers.clear()
    u_log.addHandler(fh)
    u_log.propagate = False
    return logger


def _run_one(
    question: str,
    *,
    mode: str,
    use_llm: bool,
    max_subqueries: int,
    sources: List[str],
    topk: int,
    max_return: int,
) -> Any:
    u_opts: Dict[str, Any] = {
        "use_llm": use_llm and mode == "scholar",
        "max_subqueries": max_subqueries,
        "max_lexical_swaps": 3,
        "enable_semantic": True,
        "verbose": True,
        "raw_only": mode == "raw",
    }
    return run(
        question,
        {
            "understanding": u_opts,
            "query_understanding": u_opts,
            "retrieval": {"sources": sources, "per_query_topk": topk},
            "search": {"sources": sources, "per_query_topk": topk},
            "judge": {"max_return": max_return},
            "filter": {"max_return": max_return},
        },
    )


def eval_split(
    *,
    split: str,
    limit: int,
    offset: int,
    modes: List[str],
    use_llm: bool,
    sources: List[str],
    topk: int,
    max_subqueries: int,
    max_return: int,
    sleep: float,
    logger: logging.Logger,
    stamp: str,
) -> Dict[str, Any]:
    path = PASA_TEST_JSONL if split == "auto" else PASA_REAL_JSONL
    if not path.exists():
        raise SystemExit(f"missing dataset: {path}")
    rows = load_pasa_jsonl(path, limit=limit, offset=offset)

    logger.info("=" * 72)
    logger.info(
        "split=%s n=%s sources=%s modes=%s understanding=%s path=%s",
        split,
        len(rows),
        sources,
        modes,
        "deepseek" if use_llm and deepseek_configured() else ("llm" if use_llm else "off"),
        path,
    )

    by_mode: Dict[str, List] = {m: [] for m in modes}
    detailed_samples: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        logger.info("-" * 72)
        logger.info("[%s/%s] %s", i + 1, len(rows), row.qid)
        logger.info("Q: %s", row.question)
        sample_blob: Dict[str, Any] = {
            "qid": row.qid,
            "question": row.question,
            "n_gold": len(row.gold_ids),
            "modes": {},
        }

        for mode in modes:
            t0 = time.time()
            err: Optional[str] = None
            retrieval_trace = None
            understanding_dbg = None
            try:
                result = _run_one(
                    row.question,
                    mode=mode,
                    use_llm=use_llm,
                    max_subqueries=max_subqueries,
                    sources=sources,
                    topk=topk,
                    max_return=max_return,
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
                    title=f"[mode={mode}] understanding flow",
                )
                logger.info("\n%s", flow)
            except Exception as e:
                pred = []
                latency = int((time.time() - t0) * 1000)
                n_cand = 0
                err = str(e)
                logger.exception("[%s] FAILED: %s", mode, e)

            scored = score_sample(
                pred,
                row.gold_ids,
                qid=row.qid,
                pred_arxiv_only=True,
                latency_ms=latency,
                extra={
                    "mode": mode,
                    "n_candidates": n_cand,
                    "error": err,
                    "understanding": understanding_dbg,
                    "retrieval_trace": retrieval_trace,
                },
            )
            by_mode[mode].append(scored)
            logger.info(
                "[%s] F1=%.3f P=%.3f R=%.3f  pred=%s/%s gold=%s  %sms%s",
                mode,
                scored.f1,
                scored.precision,
                scored.recall,
                scored.n_pred_arxiv,
                scored.n_pred,
                scored.n_gold,
                latency,
                f" ERR={err}" if err else "",
            )
            sample_blob["modes"][mode] = {
                "precision": scored.precision,
                "recall": scored.recall,
                "f1": scored.f1,
                "n_pred": scored.n_pred,
                "n_pred_arxiv": scored.n_pred_arxiv,
                "n_gold": scored.n_gold,
                "paper_ids": scored.paper_ids,
                "latency_ms": scored.latency_ms,
                "n_candidates": n_cand,
                "error": err,
                "understanding": understanding_dbg,
                "retrieval_trace": retrieval_trace,
            }
            if sleep > 0:
                time.sleep(sleep)

        detailed_samples.append(sample_blob)

    macros = {}
    for mode, samples in by_mode.items():
        summary = macro_average(samples)
        macros[mode] = {
            "n": summary.n,
            "precision": summary.macro_p,
            "recall": summary.macro_r,
            "f1": summary.macro_f1,
            "mean_latency_ms": sum(s.latency_ms for s in samples) / max(len(samples), 1),
        }
        logger.info(
            "MACRO[%s] n=%s P=%.4f R=%.4f F1=%.4f mean_lat=%.0fms",
            mode,
            summary.n,
            summary.macro_p,
            summary.macro_r,
            summary.macro_f1,
            macros[mode]["mean_latency_ms"],
        )

    payload = {
        "stamp": stamp,
        "split": split,
        "path": str(path),
        "sources": sources,
        "limit": limit,
        "offset": offset,
        "topk": topk,
        "max_subqueries": max_subqueries,
        "modes": modes,
        "use_llm": use_llm,
        "macro": macros,
        "samples": detailed_samples,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw vs ScholarIR PaSa eval")
    ap.add_argument("--split", choices=["auto", "real", "both"], default="both")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-subqueries", type=int, default=5)
    ap.add_argument("--max-return", type=int, default=20)
    ap.add_argument("--deepseek", action="store_true")
    ap.add_argument("--heuristic", action="store_true")
    ap.add_argument(
        "--sources",
        default="arxiv,openalex,semantic",
        help="comma-separated retrieval sources (default: all three)",
    )
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument(
        "--modes",
        default="raw,scholar",
        help="comma-separated: raw,scholar",
    )
    ap.add_argument("--tag", default="", help="optional filename tag")
    args = ap.parse_args()

    use_llm = not args.heuristic
    if args.deepseek:
        use_llm = True
        os.environ.setdefault("SCHOLAR_IR_LLM_BACKEND", "deepseek")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"eval_raw_vs_scholar{tag}_{stamp}.log"
    logger = _setup_logging(log_path)
    logger.info("log file: %s", log_path)

    splits = ["auto", "real"] if args.split == "both" else [args.split]
    all_payloads = []
    for split in splits:
        payload = eval_split(
            split=split,
            limit=args.limit,
            offset=args.offset,
            modes=modes,
            use_llm=use_llm,
            sources=sources,
            topk=args.topk,
            max_subqueries=args.max_subqueries,
            max_return=args.max_return,
            sleep=args.sleep,
            logger=logger,
            stamp=stamp,
        )
        out_json = OUTPUT_ROOT / f"eval_raw_vs_scholar_{split}_n{args.limit}{tag}_{stamp}.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", out_json)
        all_payloads.append({"split": split, "json": str(out_json), "macro": payload["macro"]})

    summary_path = OUTPUT_ROOT / f"eval_raw_vs_scholar_summary{tag}_{stamp}.json"
    summary_path.write_text(
        json.dumps(
            {"stamp": stamp, "log": str(log_path), "runs": all_payloads},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote summary %s", summary_path)
    logger.info("DONE. log=%s", log_path)


if __name__ == "__main__":
    main()
