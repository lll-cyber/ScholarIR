"""Stage (3) 论文综合排序 — 基于 filter 结果做多特征融合重排。

输入：
  - understanding: 查询意图/槽位
  - filter_result: 含 scored / selected / paper_ids

输出：
  - 重新排序后的 JudgeResult（scored/selected/paper_ids 已更新）

融合特征：
  - filter_score: filter 阶段的相关度分数（主信号）
  - citation_score: 引用数权威性（对数归一化）
  - recency_score: 年份新度
  - venue_score: 顶会/顶刊加分
  - title_density_score: 标题中查询词命中密度
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from scholar_ir.types import JudgeResult, PaperRef, ScoredPaper, UnderstandingResult

# 轻量停用词表（与 filter/base.py 保持一致即可）
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "about", "regarding", "by", "as", "at", "from", "since", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "please", "could", "you",
    "find", "papers", "paper", "provide", "list", "show", "search", "get",
    "what", "how", "any", "some", "me", "tell", "i", "we",
    "based", "using", "via", "techniques", "methods", "approaches",
}

# 常见顶会/顶刊子串（小写）
_TOP_VENUES = {
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
    "acl", "emnlp", "naacl", "coling", "tacl",
    "aaai", "ijcai", "kdd", "www", "sigir", "icde", "mlsys",
    "tpami", "ijcv", "jmlr", "tkde", "tods", "tois",
    "nature", "science", "cell",
}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _extract_tokens(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()):
        tok = raw.strip("-")
        if len(tok) <= 1 or tok in _STOPWORDS:
            continue
        tokens.add(tok)
    return tokens


def _collect_query_tokens(understanding: UnderstandingResult) -> Set[str]:
    """收集查询侧关键词（同义词/变体/实例）。"""
    slots = understanding.slots or {}
    parts: List[str] = [understanding.raw_question]

    for key in ("topic", "method", "dataset", "domain"):
        val = slots.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    for term in slots.get("terms") or []:
        if not isinstance(term, dict):
            continue
        for k in ("text", "abbrev"):
            v = term.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        for lst in (term.get("synonyms") or [], term.get("instances") or []):
            for v in lst:
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

    skeleton = slots.get("query_skeleton")
    if isinstance(skeleton, dict):
        core = skeleton.get("core_text")
        if isinstance(core, str) and core.strip():
            parts.append(core.strip())
        for part in skeleton.get("parts") or []:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt.strip())
            for v in part.get("variants") or []:
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

    for crit in understanding.relevance_criteria:
        if isinstance(crit, dict):
            desc = crit.get("description")
            if isinstance(desc, str) and desc.strip():
                parts.append(desc.strip())

    tokens: Set[str] = set()
    for p in parts:
        tokens.update(_extract_tokens(p))
    return tokens


def _citation_count(paper: PaperRef) -> int:
    """从不同 source 的 raw 里尽量读出引用数。"""
    raw = paper.raw or {}
    for key in ("citationCount", "citation_count", "cited_by_count", "num_cited_by"):
        val = raw.get(key)
        if isinstance(val, int):
            return max(0, val)
        if isinstance(val, str) and val.isdigit():
            return max(0, int(val))
    return 0


def _venue_str(paper: PaperRef) -> str:
    raw = paper.raw or {}
    for key in ("venue", "journal", "publicationVenue"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val.lower()
    return ""


def _compute_stats(
    papers: List[PaperRef],
    query_tokens: Set[str],
) -> Dict[str, Any]:
    """计算排序所需的统计量。"""
    citations = [_citation_count(p) for p in papers]
    max_citations = max(citations) if citations else 0

    years = [p.year for p in papers if p.year is not None]
    min_year = min(years) if years else 2000
    max_year = max(years) if years else 2026

    return {
        "query_tokens": query_tokens,
        "max_citations": max_citations,
        "min_year": min_year,
        "max_year": max_year,
    }


def _citation_score(paper: PaperRef, max_citations: int) -> float:
    if max_citations <= 0:
        return 0.0
    cc = _citation_count(paper)
    return math.log1p(cc) / math.log1p(max_citations)


def _recency_score(paper: PaperRef, min_year: int, max_year: int) -> float:
    if paper.year is None or max_year <= min_year:
        return 0.5
    return (paper.year - min_year) / (max_year - min_year)


def _venue_score(paper: PaperRef) -> float:
    venue = _venue_str(paper)
    if not venue:
        return 0.0
    for top in _TOP_VENUES:
        if top in venue:
            return 1.0
    return 0.0


def _title_density_score(paper: PaperRef, query_tokens: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    title_tokens = _extract_tokens(paper.title or "")
    if not title_tokens:
        return 0.0
    matched = query_tokens & title_tokens
    return len(matched) / len(query_tokens)


def _rank_score(
    scored: ScoredPaper,
    understanding: UnderstandingResult,
    stats: Dict[str, Any],
    weights: Dict[str, float],
) -> Tuple[float, str]:
    """计算单篇论文的最终综合分与理由。"""
    paper = scored.paper

    filter_score = max(0.0, min(1.0, scored.score))
    citation = _citation_score(paper, stats["max_citations"])
    recency = _recency_score(paper, stats["min_year"], stats["max_year"])
    venue = _venue_score(paper)
    title_density = _title_density_score(paper, stats["query_tokens"])

    final = (
        weights.get("filter", 0.50) * filter_score +
        weights.get("citation", 0.20) * citation +
        weights.get("recency", 0.15) * recency +
        weights.get("venue", 0.10) * venue +
        weights.get("title", 0.05) * title_density
    )
    final = round(max(0.0, min(1.0, final)), 4)

    reason = (
        f"rank={final}; "
        f"filter={filter_score}, citation={round(citation,2)}, "
        f"recency={round(recency,2)}, venue={round(venue,2)}, "
        f"title={round(title_density,2)}"
    )
    return final, reason


def rank(
    understanding: UnderstandingResult,
    filter_result: JudgeResult,
    options: Dict[str, Any] | None = None,
) -> JudgeResult:
    """对 filter 结果做多特征融合重排。

    Args:
        understanding: 查询理解结果。
        filter_result: filter 阶段输出（含 scored/selected）。
        options: 配置项
            - max_return: 最终返回数量（默认 20）
            - weights: 各特征权重，默认
                {filter: 0.5, citation: 0.2, recency: 0.15, venue: 0.1, title: 0.05}
            - threshold: 最低综合分（默认 0.0，不过滤）

    Returns:
        重新排序后的 JudgeResult。
    """
    options = options or {}
    max_return = int(options.get("max_return", 20))
    threshold = float(options.get("threshold", 0.0))
    weights = dict(options.get("weights") or {})
    arxiv_only = bool(options.get("arxiv_only", True))

    # 只重排 filter 已经选中的论文，不复活被丢弃的候选
    selected_ids = {p.paper_id for p in filter_result.selected if p.paper_id}
    pool = [s for s in filter_result.scored if s.paper.paper_id in selected_ids]

    if not pool:
        return filter_result

    query_tokens = _collect_query_tokens(understanding)
    papers = [s.paper for s in pool]
    stats = _compute_stats(papers, query_tokens)

    rescored: List[ScoredPaper] = []
    for s in pool:
        score, reason = _rank_score(s, understanding, stats, weights)
        rescored.append(ScoredPaper(paper=s.paper, score=score, reason=reason))

    rescored.sort(key=lambda s: s.score, reverse=True)
    survivors = [s.paper for s in rescored if s.score >= threshold]
    if arxiv_only:
        from scholar_ir.eval import is_arxiv_id

        survivors = [p for p in survivors if p.paper_id and is_arxiv_id(p.paper_id)]
    selected = survivors[:max_return]

    return JudgeResult(
        scored=rescored,
        selected=selected,
        paper_ids=[p.paper_id for p in selected if p.paper_id],
    )
