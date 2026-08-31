"""Orchestrator — four stages (赛题流水线):

  (1) query_understanding  查询理解与分解
  (2) search + filter      自主搜索（含 broaden/narrow 迭代）+ 过滤不相干/低质量
  (3) ranking              论文综合排序（多特征 + embedding 融合）
  (4) organize             搜索结果归纳整理（骨架）
"""

from __future__ import annotations

import time
from typing import Any, Dict

from scholar_ir.filter import filter_papers
from scholar_ir.organize import organize
from scholar_ir.query_understanding import understand
from scholar_ir.ranking import rank
from scholar_ir.search import retrieve_iterative
from scholar_ir.search.s2_client import reset_rate_limiter
from scholar_ir.types import PipelineResult


def run(question: str, options: Dict[str, Any] | None = None) -> PipelineResult:
    options = options or {}
    # Make sure any updated S2_RATE_LIMIT_RPS from env takes effect
    reset_rate_limiter()
    t0 = time.time()

    # (1) 查询理解与分解
    u_opts = options.get("query_understanding") or options.get("understanding")
    understanding = understand(question, u_opts)

    # (2) 自主搜索（候选不足自动 broaden / 过多自动 narrow）+ 候选过滤
    s_opts = options.get("search") or options.get("retrieval")
    retrieval = retrieve_iterative(understanding, s_opts)

    f_opts = options.get("filter") or options.get("judge")
    filter_result = filter_papers(
        understanding,
        retrieval.candidates,
        f_opts,
    )

    # (3) 论文综合排序（骨架：透传 filter 结果）
    ranking_result = rank(
        understanding,
        filter_result,
        options.get("ranking"),
    )

    # (4) 搜索结果归纳整理（骨架：透传排序列表）
    organized = organize(
        understanding,
        ranking_result,
        options.get("organize"),
    )

    selected = ranking_result.selected
    paper_ids = ranking_result.paper_ids or [p.paper_id for p in selected if p.paper_id]

    latency_ms = int((time.time() - t0) * 1000)
    return PipelineResult(
        understanding=understanding,
        retrieval=retrieval,
        judge=filter_result,  # type field kept for eval compat; stage-2 filter
        paper_ids=paper_ids,
        metrics_local={
            "latency_ms": latency_ms,
            "n_sub_queries": len(understanding.sub_queries),
            "n_candidates": len(retrieval.candidates),
            "n_selected": len(paper_ids),
            "retrieval_stats": retrieval.stats,
            "stages": {
                "query_understanding": True,
                "search_filter": True,
                "ranking": True,
                "organize": True,
            },
            "organize_preview": getattr(organized, "summary", None),
        },
    )
