"""Smoke: full pipeline on one PaSa query.

Usage:
  PYTHONPATH=src python3 scripts/smoke_pipeline.py --deepseek
  PYTHONPATH=src python3 scripts/smoke_pipeline.py --deepseek --sources arxiv,openalex,semantic
  PYTHONPATH=src python3 scripts/smoke_pipeline.py --deepseek --out-dir outputs/smoke_run
  PYTHONPATH=src python3 scripts/smoke_pipeline.py --qid AutoScholarQuery_test_3 --dump-json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _paper_brief(p) -> Dict[str, Any]:
    return {
        "paper_id": p.paper_id,
        "title": p.title,
        "year": p.year,
        "source": p.source,
    }


def _scored_brief(s) -> Dict[str, Any]:
    return {
        "paper_id": s.paper.paper_id,
        "title": s.paper.title,
        "year": s.paper.year,
        "score": s.score,
        "reason": s.reason,
        "features": dict(s.features or {}),
    }


def _dump_run_artifacts(
    out_dir: Path,
    *,
    qid: str,
    question: str,
    result,
    sample,
    options: Dict[str, Any],
) -> None:
    """Persist intermediate stages + final organize (list + graph)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    u = result.understanding
    r = result.retrieval
    j = result.judge
    org = result.organized

    _write_json(
        out_dir / "00_meta.json",
        {
            "qid": qid,
            "question": question,
            "options": options,
            "metrics": result.metrics_local,
            "eval": {
                "precision": sample.precision,
                "recall": sample.recall,
                "f1": sample.f1,
                "gold_ids": list(getattr(sample, "gold_ids", []) or []),
            },
            "paper_ids": result.paper_ids,
            "paper_ids_arxiv": arxiv_only(result.paper_ids),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    _write_json(
        out_dir / "01_understanding.json",
        {
            "raw_question": u.raw_question,
            "intent": u.intent,
            "slots": u.slots,
            "relevance_criteria": u.relevance_criteria,
            "sub_queries": [
                {
                    "qid": sq.qid,
                    "text": sq.text,
                    "channel": sq.channel,
                    "angle": sq.angle,
                    "mode": sq.mode,
                    "modifiers": sq.modifiers,
                }
                for sq in u.sub_queries
            ],
            "trace": u.trace,
        },
    )
    _write_json(
        out_dir / "02_retrieval.json",
        {
            "stats": r.stats,
            "trace": r.trace,
            "n_candidates": len(r.candidates),
            "candidates": [_paper_brief(p) for p in r.candidates],
        },
    )
    _write_json(
        out_dir / "03_filter.json",
        {
            "n_scored": len(j.scored),
            "n_selected": len(j.selected),
            "paper_ids": j.paper_ids,
            "scored": [_scored_brief(s) for s in j.scored],
            "selected": [_paper_brief(p) for p in j.selected],
        },
    )
    if org is not None:
        payload = org.to_dict()
        _write_json(out_dir / "04_organize.json", payload)
        # Convenience splits for frontend / debugging
        _write_json(
            out_dir / "04a_list.json",
            {
                "view": "list",
                "summary": org.summary,
                "groups": org.groups,
                "items": org.items,
                "funnel": org.funnel,
                "query_view": org.query_view,
            },
        )
        if org.graph:
            _write_json(out_dir / "04b_graph.json", org.graph)

    print(f"saved run artifacts -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None, help="PaSa AutoScholar qid")
    ap.add_argument("--topk", type=int, default=DEFAULT_PER_QUERY_TOPK)
    ap.add_argument("--max-subqueries", type=int, default=3)
    ap.add_argument("--deepseek", action="store_true", help="Understanding via DeepSeek API")
    ap.add_argument("--heuristic", action="store_true", help="rule-based Understanding only")
    ap.add_argument(
        "--sources",
        default="arxiv,openalex,semantic",
        help="Comma-separated retrieval sources (default: arxiv,openalex,semantic)",
    )
    ap.add_argument(
        "--expand",
        action="store_true",
        help="Enable citation expansion in filter",
    )
    ap.add_argument(
        "--dump-json",
        default=None,
        help="Write organize payload (list+graph) to this JSON path",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Save full intermediate + final artifacts under this directory",
    )
    ap.add_argument(
        "--view",
        default="auto",
        choices=["auto", "list", "graph"],
        help="Organize view mode (default auto: dense cites → graph)",
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip citation-graph fetch (faster; list only)",
    )
    ap.add_argument(
        "--no-arxiv-only",
        action="store_true",
        help="Keep non-arXiv ids in filter/ranking (default: drop them for PaSa-friendly ids)",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help="Also write Python logging (understanding flow etc.) to this file",
    )
    args = ap.parse_args()

    use_llm = not args.heuristic
    if args.deepseek:
        use_llm = True
        os.environ.setdefault("SCHOLAR_IR_LLM_BACKEND", "deepseek")
    if use_llm and not deepseek_configured() and os.getenv("SCHOLAR_IR_LLM_BACKEND", "auto") == "deepseek":
        print("[warn] DEEPSEEK_API_KEY missing; may fall back to heuristic", file=sys.stderr)

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path, encoding="utf-8"),
            ],
            force=True,
        )
        print("logging ->", log_path)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    question = "papers about retrieval-augmented generation for question answering"
    gold_ids: list = []
    qid = "manual"
    if PASA_TEST_JSONL.exists():
        row = _load_row(PASA_TEST_JSONL, args.qid)
        question = row["question"]
        gold_ids = row.get("answer_arxiv_id") or []
        qid = row.get("qid", qid)

    run_options: Dict[str, Any] = {
        "understanding": {"use_llm": use_llm, "max_subqueries": args.max_subqueries},
        "retrieval": {
            "sources": sources,
            "per_query_topk": args.topk,
        },
        "judge": {
            "max_return": 20,
            "arxiv_only": (not args.no_arxiv_only),
            "expand_citations": args.expand,
            "seed_top_k": 5,
            "ref_limit": 10,
            "cit_limit": 10,
            "expand_max_total": 50,
            "use_llm_for_expanded": True,
            "llm_top_k_expanded": 15,
        },
        "ranking": {
            "threshold": 0.5,
            "max_return": 20,
            "arxiv_only": (not args.no_arxiv_only),
        },
        "organize": {
            "view": args.view,
            "build_graph": (not args.no_graph),
        },
    }

    print("understanding:", "deepseek" if use_llm else "heuristic")
    print("sources:", sources)
    print("expand:", args.expand)
    print("arxiv_only:", not args.no_arxiv_only)
    print("organize:", run_options["organize"])
    result = run(question, run_options)

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

    org = result.organized
    if org is not None:
        print("--- organize (前端展示视图) ---")
        print("summary:", org.summary)
        print("view:", org.view)
        qv = org.query_view
        print("query_view.intent:", qv.get("intent"), "| topic:", qv.get("topic"))
        print("query_view.keywords:", qv.get("keywords"))
        print("query_view.constraints:", qv.get("constraints"))
        funnel = " -> ".join(
            f"{s['label']}({s['count']})" for s in org.funnel.get("stages", [])
        )
        print("funnel:", funnel)
        print(
            "groups:",
            ", ".join(f"{g['label']}={g['count']}" for g in org.groups),
            f"| high_threshold={org.funnel.get('high_threshold')}",
        )
        for it in org.items:
            tag = "HIGH" if it["tier"] == "highly_relevant" else "part"
            print(f"  #{it['rank']:>2} [{tag}] {it['score']:.4f} {it['title'][:70]}")
            print(f"        {it['year']} | cites={it['citation_count']} | {it['url']}")
            print(f"        入选: {it.get('selection_reason') or ''}")
            evidence = (it.get("match_reasons") or [])[1:]
            if evidence:
                print(f"        证据: {' / '.join(evidence)}")

        if org.graph:
            g = org.graph
            stats = g.get("stats") or {}
            print(
                f"--- graph (view={org.view}) nodes={stats.get('n_nodes')} "
                f"edges={stats.get('n_edges')} reason={stats.get('reason')!r} ---"
            )
            for e in (g.get("edges") or [])[:12]:
                print(f"  {e.get('source')} -cites-> {e.get('target')}")
        else:
            print("--- graph: (not built) ---")

        if args.dump_json:
            out = Path(args.dump_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(org.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("dumped organize payload ->", out)

    if args.out_dir:
        _dump_run_artifacts(
            Path(args.out_dir),
            qid=qid,
            question=question,
            result=result,
            sample=sample,
            options=run_options,
        )


if __name__ == "__main__":
    main()
