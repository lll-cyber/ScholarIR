"""Stage (4) 搜索结果归纳整理 — 面向前端展示的结构化输出。

产出内容，前端可直接渲染：
  - query_view: 系统如何理解这个问题（意图、槽位、实际打出的检索式）
  - funnel:     检索漏斗（候选 → 过滤 → 排序），让用户看到收敛过程
  - groups:     相关性分档（高度相关 / 部分相关），对应赛题的两档要求
  - items:      每篇论文的结构化字段（含 selection_reason 自然语言入选说明 + 特征证据）
  - graph:      入选论文的引用关系图（nodes + cites edges；多源回退）
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scholar_ir.organize.graph import (
    build_citation_graph,
    resolve_view,
    should_build_graph,
)
from scholar_ir.types import (
    JudgeResult,
    OrganizeResult,
    RetrievalResult,
    ScoredPaper,
    UnderstandingResult,
)

# 分档阈值：按 intent 区分，宽口径意图放低门槛
_TIER_THRESHOLDS: Dict[str, float] = {
    "survey": 0.45,
    "broad": 0.45,
    "related": 0.50,
    "method": 0.55,
    "dataset": 0.55,
    "specific": 0.60,
}
_DEFAULT_HIGH_THRESHOLD = 0.55
# 相关性判定分低于此值时，无论综合分多高都不进"高度相关"档，
# 避免引用数/时效性等工程特征把实际不相关的论文顶上来
_MIN_FILTER_FOR_HIGH = 0.5

# 特征名 → 前端展示用中文标签
_FEATURE_LABELS: Dict[str, str] = {
    "relevance": "相关性判定",
    "filter": "相关性判定",  # 兼容旧字段名
    "keyword_coverage": "关键词覆盖",
    "llm": "LLM 判定",
    "blended": "综合得分",
    "normalized": "跨意图归一",
    "embedding": "语义相似度",
    "citation": "引用影响力",
    "recency": "时效性",
    "venue": "发表场所",
    "title": "标题命中",
}

# 证据条里跳过的派生/重复特征（主文案已覆盖相关性）
_REASON_SKIP_FEATURES = frozenset({
    "normalized",
    "blended",
    "impact_skipped",
})

_IMPACT_TAIL_RE = re.compile(r";?\s*impact\[[^\]]*\]\s*$", re.I)
_LLM_REASON_RE = re.compile(
    r"^(?:expanded_)?llm\s*:\s*[0-9.]+\s*;\s*(.+)$",
    re.I | re.DOTALL,
)
_KW_ONLY_RE = re.compile(r"^keyword_coverage\s*:\s*[0-9.]+\s*$", re.I)


def _high_threshold(intent: str, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    return _TIER_THRESHOLDS.get((intent or "").strip().lower(), _DEFAULT_HIGH_THRESHOLD)


def _arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def _paper_url(paper_id: str, raw: Dict[str, Any]) -> str:
    """尽量给出一个可点击链接，供前端跳转。"""
    for key in ("url", "openAccessPdf", "landing_page_url", "doi_url"):
        val = raw.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
        if isinstance(val, dict):
            inner = val.get("url")
            if isinstance(inner, str) and inner.startswith("http"):
                return inner

    doi = raw.get("doi")
    if isinstance(doi, str) and doi.strip():
        doi = doi.strip()
        return doi if doi.startswith("http") else f"https://doi.org/{doi}"

    from scholar_ir.eval import is_arxiv_id

    if paper_id and is_arxiv_id(paper_id):
        return _arxiv_url(paper_id)
    return ""


def _citation_count(raw: Dict[str, Any]) -> Optional[int]:
    for key in ("citationCount", "citation_count", "cited_by_count", "num_cited_by"):
        val = raw.get(key)
        if isinstance(val, int):
            return max(0, val)
        if isinstance(val, str) and val.isdigit():
            return int(val)
    return None


def _venue(raw: Dict[str, Any]) -> str:
    for key in ("venue", "journal", "publicationVenue"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            name = val.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return ""


def _top_reasons(features: Dict[str, float], n: int = 3) -> List[str]:
    """挑出得分最高的几个特征，转成百分比证据条（辅助，非主文案）。"""
    if not features:
        return []
    ranked = sorted(
        (
            (name, val)
            for name, val in features.items()
            if name not in _REASON_SKIP_FEATURES
            and not str(name).endswith("_skipped")
        ),
        key=lambda kv: float(kv[1]) if isinstance(kv[1], (int, float)) else -1.0,
        reverse=True,
    )
    out: List[str] = []
    for name, val in ranked:
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num <= 0.0:
            continue
        # bool True from impact_skipped already skipped; skip non-score flags
        if isinstance(val, bool):
            continue
        label = _FEATURE_LABELS.get(name, name)
        out.append(f"{label} {round(num * 100)}%")
        if len(out) >= n:
            break
    return out


def extract_llm_reason_text(reason: str) -> str:
    """从 ScoredPaper.reason 中抽出 LLM 的自然语言判定句。

    常见格式::
      llm:0.9;Proposes a deep unfolding...;impact[cit=1.0,...]
      expanded_llm:0.85;Cites seed X on method Y;impact[...]
    """
    text = (reason or "").strip()
    if not text:
        return ""
    text = _IMPACT_TAIL_RE.sub("", text).strip().rstrip(";").strip()
    if not text or _KW_ONLY_RE.match(text):
        return ""
    m = _LLM_REASON_RE.match(text)
    if m:
        body = m.group(1).strip().rstrip(";").strip()
        # 偶发二次嵌套 impact
        body = _IMPACT_TAIL_RE.sub("", body).strip().rstrip(";").strip()
        return body
    # 无 llm: 前缀但也不像纯 keyword 标记时，不当作 NL（避免泄露 debug 串）
    return ""


def _tier_lead(tier: str) -> str:
    if tier == "highly_relevant":
        return "入选「高度相关」"
    if tier == "partially_relevant":
        return "入选「部分相关」"
    return "入选结果列表"


def _heuristic_selection_reason(
    scored: ScoredPaper,
    understanding: UnderstandingResult,
    tier: str,
) -> str:
    """无 LLM 短句时，用槽位 + 特征拼一条可读入选说明。"""
    slots = understanding.slots or {}
    topic = slots.get("topic") or ""
    method = slots.get("method") or ""
    feats = scored.features or {}
    bits: List[str] = []

    if topic:
        bits.append(f"与主题「{topic}」相关")
    if method:
        bits.append(f"方法线索「{method}」")

    title_hit = float(feats.get("title") or 0.0)
    if title_hit >= 0.5:
        bits.append("标题关键词命中较高")

    rel = feats.get("relevance", feats.get("llm", feats.get("keyword_coverage")))
    try:
        if rel is not None and float(rel) > 0:
            bits.append(f"相关性约 {round(float(rel) * 100)}%")
    except (TypeError, ValueError):
        pass

    cit = float(feats.get("citation") or 0.0)
    if cit >= 0.7:
        bits.append("引用影响力突出")

    if not bits:
        intent = understanding.intent or "broad"
        bits.append(f"综合排序通过（intent={intent}）")

    return f"{_tier_lead(tier)}：" + "；".join(bits) + "。"


def build_selection_reason(
    scored: ScoredPaper,
    understanding: UnderstandingResult,
    tier: str,
) -> str:
    """面向用户的「为什么这篇论文入选」自然语言说明。"""
    llm_text = extract_llm_reason_text(scored.reason or "")
    lead = _tier_lead(tier)
    if llm_text:
        # LLM 句多为英文 brief；保留原文，前面加中文档位引导
        return f"{lead}：{llm_text}"
    return _heuristic_selection_reason(scored, understanding, tier)


def build_match_reasons(
    scored: ScoredPaper,
    understanding: UnderstandingResult,
    tier: str,
    *,
    n_features: int = 3,
) -> List[str]:
    """match_reasons[0] = 自然语言入选说明；其后为特征证据条。"""
    selection = build_selection_reason(scored, understanding, tier)
    evidence = _top_reasons(scored.features or {}, n=n_features)
    # 避免把整段 selection 再复制进证据；证据保持短标签
    out = [selection] if selection else []
    out.extend(evidence)
    return out


def _abstract_snippet(abstract: str, max_chars: int) -> str:
    """截断摘要：优先在句子边界断开，其次词边界，避免半截词/半截句。"""
    text = " ".join((abstract or "").split())
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    window = text[:max_chars]
    min_keep = max(1, max_chars // 2)

    # 优先：窗口内最后一个完整句子（最长且仍 ≤ max_chars 的句末前缀）
    sentence_end = -1
    for i, ch in enumerate(window):
        if ch in ".!?。；;":
            if ch in "。；;" or i + 1 >= len(window) or window[i + 1].isspace():
                sentence_end = i + 1
    if sentence_end > 0:
        return window[:sentence_end].rstrip() + "…"

    # 其次：最后一个空白（要求至少保留半窗，避免过短）
    space = window.rfind(" ")
    if space >= min_keep:
        return window[:space].rstrip() + "…"

    return window.rstrip() + "…"


def _relevance_score(features: Dict[str, Any]) -> float:
    """读取纯相关性分（不含 citation/recency 等 impact），供高度相关档守卫。"""
    for key in ("relevance", "llm", "keyword_coverage", "filter"):
        if key in features:
            try:
                return float(features[key])
            except (TypeError, ValueError):
                continue
    return 1.0


def _build_item(
    scored: ScoredPaper,
    rank_index: int,
    tier: str,
    snippet_chars: int,
    understanding: UnderstandingResult,
) -> Dict[str, Any]:
    paper = scored.paper
    raw = paper.raw or {}
    selection_reason = build_selection_reason(scored, understanding, tier)
    return {
        "rank": rank_index,
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "venue": _venue(raw),
        "citation_count": _citation_count(raw),
        "source": paper.source,
        "url": _paper_url(paper.paper_id, raw),
        "score": round(scored.score, 4),
        "tier": tier,
        "features": dict(scored.features or {}),
        # 主文案：为什么入选；match_reasons[0] 与此一致，后接特征证据
        "selection_reason": selection_reason,
        "match_reasons": build_match_reasons(scored, understanding, tier),
        "abstract_snippet": _abstract_snippet(paper.abstract, snippet_chars),
        "debug_reason": scored.reason,
    }


def _build_query_view(understanding: UnderstandingResult) -> Dict[str, Any]:
    """把查询理解结果整理成前端可展示的"系统怎么理解你的问题"。"""
    slots = understanding.slots or {}

    keywords: List[str] = []
    for term in slots.get("terms") or []:
        if isinstance(term, dict):
            text = term.get("text")
            if isinstance(text, str) and text.strip():
                keywords.append(text.strip())

    constraints: Dict[str, Any] = {}
    for key in ("year_from", "year_to"):
        val = slots.get(key)
        if val:
            constraints[key] = val
    negation = slots.get("negation")
    if negation:
        constraints["exclude"] = negation

    return {
        "raw_question": understanding.raw_question,
        "intent": understanding.intent,
        "topic": slots.get("topic") or "",
        "keywords": keywords,
        "constraints": constraints,
        "sub_queries": [
            {"qid": sq.qid, "text": sq.text, "angle": sq.angle}
            for sq in understanding.sub_queries
        ],
    }


def _build_funnel(
    retrieval: Optional[RetrievalResult],
    filter_result: Optional[JudgeResult],
    n_ranked: int,
    n_high: int,
    n_partial: int,
) -> Dict[str, Any]:
    """检索漏斗：让用户看到每一步筛掉了多少。"""
    stages: List[Dict[str, Any]] = []

    if retrieval is not None:
        stats = retrieval.stats or {}
        stages.append({
            "stage": "retrieved",
            "label": "多源召回",
            "count": len(retrieval.candidates),
            "detail": {
                "api_calls": stats.get("n_api_calls"),
                "iterate_rounds": stats.get("iterate_rounds"),
            },
        })

    if filter_result is not None:
        stages.append({
            "stage": "filtered",
            "label": "过滤后",
            "count": len(filter_result.selected),
        })

    stages.append({
        "stage": "ranked",
        "label": "综合排序",
        "count": n_ranked,
    })

    return {
        "stages": stages,
        "n_highly_relevant": n_high,
        "n_partially_relevant": n_partial,
    }


def _build_summary(
    understanding: UnderstandingResult,
    funnel: Dict[str, Any],
    n_high: int,
    n_partial: int,
    top_title: str,
) -> str:
    intent = understanding.intent or "broad"
    n_retrieved = 0
    for stage in funnel.get("stages") or []:
        if stage.get("stage") == "retrieved":
            n_retrieved = stage.get("count") or 0
            break

    parts = [f"意图识别为 {intent}"]
    if n_retrieved:
        parts.append(f"多源召回 {n_retrieved} 篇候选")
    parts.append(f"经过滤与排序后给出 {n_high + n_partial} 篇")
    if n_high:
        parts.append(f"其中高度相关 {n_high} 篇")
    if n_partial:
        parts.append(f"部分相关 {n_partial} 篇")
    summary = "，".join(parts) + "。"
    if top_title:
        summary += f" 最相关：{top_title}"
    return summary


def _enrich_summary_with_graph(summary: str, graph: Dict[str, Any]) -> str:
    stats = graph.get("stats") or {}
    n_edges = int(stats.get("n_edges") or 0)
    if n_edges > 0:
        return summary + f" 引用关系图含 {n_edges} 条边。"
    return summary


def organize(
    understanding: UnderstandingResult,
    ranking_result: JudgeResult,
    options: Dict[str, Any] | None = None,
    retrieval: Optional[RetrievalResult] = None,
    filter_result: Optional[JudgeResult] = None,
) -> OrganizeResult:
    """把排序结果整理成前端可直接渲染的结构。

    Args:
        understanding: 查询理解结果。
        ranking_result: 排序阶段输出。
        options: 配置项
            - view: ``list`` | ``graph`` | ``auto``（默认 auto：边够密才切到 graph）
            - build_graph: 强制开/关拉边（与 intent 无关；默认 ≥2 篇就拉）
            - graph_min_edges / graph_min_density: auto 视图的稠密阈值
            - high_threshold: 高度相关档阈值（默认按 intent 自适应）
            - min_filter_for_high: 进入高度相关档所需的最低纯相关性分（默认 0.5；
              读 features.relevance / llm / keyword_coverage，不含 impact）
            - snippet_chars: 摘要截断长度（默认 240；按句子边界截断）
            - graph_seed_k / graph_ref_limit / graph_cit_limit: 图构建预算
        retrieval: 检索结果，用于构造漏斗（可选）。
        filter_result: 过滤结果，用于构造漏斗（可选）。

    Returns:
        OrganizeResult，调用 to_dict() 即得前端契约 JSON。
    """
    options = options or {}
    snippet_chars = int(options.get("snippet_chars", 240))
    high_threshold = _high_threshold(
        understanding.intent, options.get("high_threshold")
    )
    min_filter_for_high = float(
        options.get("min_filter_for_high", _MIN_FILTER_FOR_HIGH)
    )

    # 只展示最终入选的论文，按排序结果的顺序
    selected_ids = [p.paper_id for p in ranking_result.selected if p.paper_id]
    scored_map = {
        s.paper.paper_id: s for s in ranking_result.scored if s.paper.paper_id
    }
    ordered = [scored_map[pid] for pid in selected_ids if pid in scored_map]

    items: List[Dict[str, Any]] = []
    high_items: List[Dict[str, Any]] = []
    partial_items: List[Dict[str, Any]] = []

    for i, scored in enumerate(ordered, 1):
        filter_score = _relevance_score(scored.features or {})
        is_high = (
            scored.score >= high_threshold
            and filter_score >= min_filter_for_high
        )
        tier = "highly_relevant" if is_high else "partially_relevant"
        item = _build_item(scored, i, tier, snippet_chars, understanding)
        items.append(item)
        if tier == "highly_relevant":
            high_items.append(item)
        else:
            partial_items.append(item)

    groups = [
        {
            "key": "highly_relevant",
            "label": "高度相关",
            "count": len(high_items),
            "paper_ids": [it["paper_id"] for it in high_items],
        },
        {
            "key": "partially_relevant",
            "label": "部分相关",
            "count": len(partial_items),
            "paper_ids": [it["paper_id"] for it in partial_items],
        },
    ]

    funnel = _build_funnel(
        retrieval, filter_result, len(items), len(high_items), len(partial_items)
    )
    funnel["high_threshold"] = round(high_threshold, 4)
    funnel["min_filter_for_high"] = round(min_filter_for_high, 4)

    summary = _build_summary(
        understanding,
        funnel,
        len(high_items),
        len(partial_items),
        items[0]["title"] if items else "",
    )

    # 建图：按数据/预算，不按 intent；展示视图再按边密度决定
    graph: Dict[str, Any] = {}
    if should_build_graph(options, n_papers=len(ordered)) and ordered:
        selected_papers = [s.paper for s in ordered]
        graph = build_citation_graph(selected_papers, items, options)
        summary = _enrich_summary_with_graph(summary, graph)

    view = resolve_view(options, graph)

    return OrganizeResult(
        items=items,
        groups=groups,
        summary=summary,
        funnel=funnel,
        query_view=_build_query_view(understanding),
        view=view,
        graph=graph,
    )
