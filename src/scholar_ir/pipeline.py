"""Orchestrator — four stages (赛题流水线):

  (1) query_understanding  查询理解与分解
  (2) search + filter      自主搜索（broaden/narrow）+ 过滤/LLM/impact/归一化
  (3) ranking              按 filter 分数排序 + threshold / arxiv_only / max_return
  (4) organize             分档列表 + 引用关系图 + 自然语言入选理由

打分与 impact 融合在 filter；ranking 不再做 embedding / 权重重算。

Logging:
  - 默认打 stage 摘要到 logger ``scholar_ir.pipeline``（需调用方或 options 配 handler）
  - ``options["log_file"]`` 或环境变量 ``SCHOLAR_IR_LOG_FILE``：追加写文件
  - ``options["verbose"]=False`` 可关掉 stage 日志
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from scholar_ir.filter import filter_papers
from scholar_ir.organize import organize
from scholar_ir.query_understanding import understand
from scholar_ir.ranking import rank
from scholar_ir.search import retrieve_iterative
from scholar_ir.search.s2_client import reset_rate_limiter
from scholar_ir.types import PipelineResult

logger = logging.getLogger("scholar_ir.pipeline")
_FILE_HANDLER_PATHS: set[str] = set()


def _configure_pipeline_logging(options: Dict[str, Any]) -> None:
    """Attach an optional FileHandler once per path; never hijack root if already set."""
    log_file = options.get("log_file") or os.getenv("SCHOLAR_IR_LOG_FILE") or ""
    log_file = str(log_file).strip()
    if not log_file:
        return
    path = str(Path(log_file).expanduser().resolve())
    if path in _FILE_HANDLER_PATHS:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    handler.setLevel(logging.INFO)
    # Attach to scholar_ir package logger so understanding/filter/s2 also land here
    pkg = logging.getLogger("scholar_ir")
    pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)
    # Avoid duplicate console spam if root already has handlers
    pkg.propagate = True
    _FILE_HANDLER_PATHS.add(path)
    logger.info("pipeline file logging -> %s", path)


def run(question: str, options: Dict[str, Any] | None = None) -> PipelineResult:
    options = options or {}
    verbose = bool(options.get("verbose", True))
    _configure_pipeline_logging(options)

    # Make sure any updated S2_RATE_LIMIT_RPS from env takes effect
    reset_rate_limiter()
    t0 = time.time()
    if verbose:
        logger.info("pipeline start q=%r", (question or "")[:160])

    # (1) 查询理解与分解
    u_opts = options.get("query_understanding") or options.get("understanding")
    understanding = understand(question, u_opts)
    if verbose:
        logger.info(
            "stage1 understanding intent=%s n_sub_queries=%d",
            understanding.intent,
            len(understanding.sub_queries),
        )

    # (2) 自主搜索（候选不足自动 broaden / 过多自动 narrow）+ 候选过滤
    s_opts = options.get("search") or options.get("retrieval")
    retrieval = retrieve_iterative(understanding, s_opts)
    if verbose:
        logger.info(
            "stage2a search n_candidates=%d stats=%s",
            len(retrieval.candidates),
            retrieval.stats,
        )

    f_opts = options.get("filter") or options.get("judge")
    filter_result = filter_papers(
        understanding,
        retrieval.candidates,
        f_opts,
    )
    if verbose:
        logger.info(
            "stage2b filter n_scored=%d n_selected=%d",
            len(filter_result.scored),
            len(filter_result.selected),
        )

    # (3) 排序截断（分数以 filter 为准；本阶段不做 embedding）
    ranking_result = rank(
        understanding,
        filter_result,
        options.get("ranking"),
    )
    if verbose:
        logger.info(
            "stage3 ranking n_selected=%d",
            len(ranking_result.selected),
        )

    # (4) 搜索结果归纳整理（分档 + 漏斗 + 图 + 入选理由）
    organized = organize(
        understanding,
        ranking_result,
        options.get("organize"),
        retrieval=retrieval,
        filter_result=filter_result,
    )
    if verbose:
        gstats = (organized.graph or {}).get("stats") or {}
        logger.info(
            "stage4 organize view=%s items=%d graph_edges=%s summary=%s",
            organized.view,
            len(organized.items),
            gstats.get("n_edges"),
            (organized.summary or "")[:120],
        )

    selected = ranking_result.selected
    paper_ids = ranking_result.paper_ids or [p.paper_id for p in selected if p.paper_id]

    latency_ms = int((time.time() - t0) * 1000)
    if verbose:
        logger.info(
            "pipeline done latency_ms=%d n_selected=%d",
            latency_ms,
            len(paper_ids),
        )

    return PipelineResult(
        understanding=understanding,
        retrieval=retrieval,
        judge=filter_result,  # type field kept for eval compat; stage-2 filter
        paper_ids=paper_ids,
        organized=organized,
        metrics_local={
            "latency_ms": latency_ms,
            "n_sub_queries": len(understanding.sub_queries),
            "n_candidates": len(retrieval.candidates),
            "n_selected": len(paper_ids),
            "n_highly_relevant": organized.funnel.get("n_highly_relevant"),
            "n_partially_relevant": organized.funnel.get("n_partially_relevant"),
            "retrieval_stats": retrieval.stats,
            "stages": {
                "query_understanding": True,
                "search_filter": True,
                "ranking": True,
                "organize": True,
            },
            "organize_preview": organized.summary,
        },
    )
