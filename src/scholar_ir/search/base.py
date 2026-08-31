"""Search (stage-2): Understanding → candidates via per-source adapt.

与 filter 一起构成「自主搜索」阶段；broaden/narrow 迭代见 ``iterate.py``。
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
from scholar_ir.search.dedup import canonical_paper_id
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


# ---------- Semantic-source budget allocation ----------

# Higher = more important.  Sub_queries are sorted by this before the budget
# cap is applied, so semantic (S2) always sees the highest-value queries.
_PRIORITY_BY_ANGLE: Dict[str, int] = {
    "core": 100,
    "synonym": 80,
    "abbrev": 70,
    "entity": 60,
    "conceptual": 50,
    "metadata": 30,
    "raw": 10,
}
_PRIORITY_BY_MODE = {"semantic": 1000, "decomposition": 50}


def _subquery_priority(sq: SubQuery) -> int:
    """Composite priority score for the sub-query."""
    base = _PRIORITY_BY_ANGLE.get(getattr(sq, "angle", ""), 40)
    mode_bonus = _PRIORITY_BY_MODE.get(getattr(sq, "mode", ""), 0)
    channel_bonus = 0
    if getattr(sq, "channel", "") == "semantic":
        channel_bonus = 1000
    return base + mode_bonus + channel_bonus


def _select_semantic_budget(
    sub_queries: List[SubQuery],
    semantic_max_queries: int,
) -> set:
    """Pick the set of sub_query qids eligible for semantic (S2) source.

    Allocation is by priority, not by position.  semantic-channel subqueries
    are always eligible; the remaining budget is filled by the top-scored
    sub_queries (angle core > synonym > abbrev > ... > raw).

    Returns:
        set of qid strings eligible for S2 calls.
    """
    if semantic_max_queries <= 0 or not sub_queries:
        return set()

    # Always-eligible: anything explicitly semantic-channeled or semantic-mode.
    always: List[SubQuery] = [
        sq for sq in sub_queries
        if getattr(sq, "channel", "") == "semantic"
        or getattr(sq, "mode", "") == "semantic"
    ]

    # Remaining, sorted by priority descending, stable.
    remaining: List[SubQuery] = [sq for sq in sub_queries if sq not in always]
    remaining.sort(key=_subquery_priority, reverse=True)

    eligible: List[SubQuery] = list(always)
    for sq in remaining:
        if len(eligible) >= semantic_max_queries:
            break
        eligible.append(sq)

    return {
        getattr(sq, "qid", "") or f"i{i}"
        for i, sq in enumerate(eligible)
    }


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

    # Pre-compute semantic budget allocation by priority (not position).
    # Mode/semantic-channel sub_queries always pass; remaining are picked
    # in priority order (core > synonym > abbrev > entity > ...).
    semantic_eligible = _select_semantic_budget(sub_queries, semantic_max_queries)

    candidates: List[PaperRef] = []
    # `seen` uses canonical keys so the same paper from arxiv/openalex/s2
    # only enters the candidate list once.
    seen = set()
    trace: List[Dict[str, Any]] = []
    n_api_calls = 0

    for i, sq in enumerate(sub_queries):
        sqid = getattr(sq, "qid", None) or f"i{i}"
        sq_semantic_eligible = sqid in semantic_eligible
        for source in sources:
            # Limit S2 calls to the priority-selected sub-queries
            if source == "semantic" and not sq_semantic_eligible:
                trace.append({
                    "qid": sqid,
                    "source": source,
                    "status": "skipped",
                    "note": (
                        f"semantic budget exhausted (semantic_max_queries="
                        f"{semantic_max_queries}); priority rank outside budget"
                    ),
                    "priority": _subquery_priority(sq),
                    "angle": getattr(sq, "angle", ""),
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
                ckey = canonical_paper_id(p, index=len(candidates))
                if not ckey or ckey in seen:
                    continue
                seen.add(ckey)
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
