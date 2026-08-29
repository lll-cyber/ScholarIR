"""Search (stage-2): Understanding → candidates via per-source adapt.

与 filter 一起构成「自主搜索策略」阶段（迭代式检索后续补齐）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from scholar_ir.config import DEFAULT_PER_QUERY_TOPK, DEFAULT_SOURCES
from scholar_ir.search.adapt.arxiv_api import adapt_arxiv, search_arxiv_detail
from scholar_ir.search.adapt.openalex import (
    adapt_openalex,
    search_openalex_detail,
)
from scholar_ir.search.adapt.semantic import (
    adapt_semantic,
    search_semantic_detail,
)
from scholar_ir.types import PaperRef, RetrievalResult, SubQuery, UnderstandingResult


def paper_dict_to_ref(paper: Dict[str, Any], source: str = "") -> PaperRef:
    """Normalize common SPAR/api_web paper dicts into PaperRef."""
    paper_id = (
        paper.get("paper_id")
        or paper.get("arxivId")
        or paper.get("arxiv_id")
        or paper.get("id")
        or paper.get("paperId")
        or ""
    )
    year = paper.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    return PaperRef(
        paper_id=str(paper_id),
        title=paper.get("title", "") or "",
        abstract=paper.get("abstract", "") or paper.get("summary", "") or "",
        year=year if isinstance(year, int) else None,
        source=source or paper.get("source", ""),
        raw=paper,
    )


def _run_source(
    source: str,
    sq: SubQuery,
    slots: Dict[str, Any],
    topk: int,
    dry_run: bool,
    *,
    s2_rate_limit: bool = True,
) -> Dict[str, Any]:
    """Return trace entry + papers list under key '_papers'."""
    if source == "semantic":
        req = adapt_semantic(sq, slots, limit=topk)
        entry: Dict[str, Any] = {
            "qid": sq.qid,
            "source": source,
            "text_used": req.text_used,
            "params": dict(req.params),
            "has_api_key": bool(req.headers.get("x-api-key")),
        }
        if dry_run:
            entry["status"] = "dry_run"
            entry["hits"] = []
            entry["_papers"] = []
            return entry
        detail = search_semantic_detail(
            sq, slots, limit=topk, rate_limit=s2_rate_limit
        )
        entry["http_status"] = detail.status_code
        if detail.error:
            entry["error"] = detail.error
        if detail.waited_s:
            entry["s2_waited_s"] = round(detail.waited_s, 3)
        if detail.retries:
            entry["s2_retries"] = detail.retries
        papers = detail.papers
    elif source == "openalex":
        req = adapt_openalex(sq, slots, limit=topk)
        entry = {
            "qid": sq.qid,
            "source": source,
            "text_used": req.text_used,
            "params": dict(req.params),
            "filter_parts": list(req.filter_parts),
        }
        if dry_run:
            entry["status"] = "dry_run"
            entry["hits"] = []
            entry["_papers"] = []
            return entry
        detail = search_openalex_detail(sq, slots, limit=topk)
        entry["http_status"] = detail.status_code
        if detail.error:
            entry["error"] = detail.error
        papers = detail.papers
    elif source == "arxiv":
        req = adapt_arxiv(sq, slots, limit=topk)
        entry = {
            "qid": sq.qid,
            "source": source,
            "text_used": req.text_used,
            "params": req.to_dict(),
        }
        if dry_run:
            entry["status"] = "dry_run"
            entry["hits"] = []
            entry["_papers"] = []
            return entry
        detail = search_arxiv_detail(sq, slots, limit=topk)
        entry["http_status"] = detail.status_code
        if detail.error:
            entry["error"] = detail.error
        papers = detail.papers
    else:
        return {
            "qid": sq.qid,
            "source": source,
            "status": "skipped",
            "note": "unknown source (implemented: arxiv, openalex, semantic)",
            "hits": [],
            "_papers": [],
        }

    entry["status"] = "ok" if papers else "empty_or_error"
    entry["hits"] = [p.paper_id for p in papers]
    entry["n_hits"] = len(papers)
    entry["_papers"] = papers
    return entry


def retrieve(
    understanding: UnderstandingResult,
    options: Dict[str, Any] | None = None,
) -> RetrievalResult:
    """Run adapted searches. Default source: arxiv (native ids, no API key)."""
    options = options or {}
    sources = options.get("sources") or list(DEFAULT_SOURCES)
    topk = int(options.get("per_query_topk", DEFAULT_PER_QUERY_TOPK))
    dry_run = bool(options.get("dry_run", False))
    s2_rate_limit = not bool(options.get("s2_no_rate_limit", False))
    semantic_max_queries = int(options.get("semantic_max_queries", 2))

    slots = understanding.slots or {}
    sub_queries: List[SubQuery] = understanding.sub_queries or []

    candidates: List[PaperRef] = []
    seen = set()
    trace: List[Dict[str, Any]] = []
    n_api_calls = 0

    for i, sq in enumerate(sub_queries):
        for source in sources:
            # Limit S2 calls to the top-K most important sub-queries
            if source == "semantic" and i >= max(0, semantic_max_queries):
                trace.append({
                    "qid": sq.qid,
                    "source": source,
                    "status": "skipped",
                    "note": f"semantic budget exhausted (semantic_max_queries={semantic_max_queries})",
                    "hits": [],
                })
                continue
            entry = _run_source(
                source, sq, slots, topk, dry_run, s2_rate_limit=s2_rate_limit
            )
            papers: List[PaperRef] = entry.pop("_papers", [])
            if not dry_run and entry.get("status") not in ("skipped", "dry_run"):
                n_api_calls += 1
            for p in papers:
                key = p.paper_id or p.title
                if key and key not in seen:
                    seen.add(key)
                    candidates.append(p)
            trace.append(entry)

    return RetrievalResult(
        candidates=candidates,
        trace=trace,
        stats={
            "n_api_calls": n_api_calls,
            "n_candidates": len(candidates),
            "sources": sources,
            "per_query_topk": topk,
            "dry_run": dry_run,
            "s2_rate_limit": s2_rate_limit and "semantic" in sources,
        },
    )
