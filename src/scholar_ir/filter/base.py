"""Filter (stage-2): coarse rules + lightweight relevance scoring + optional citation expansion.

Pipeline:
  1) Deduplicate candidates by paper_id / normalized title.
  2) Hard rules: year range, explicit negation.
  3) Lightweight keyword-coverage scoring on survivors (with synonyms/variants).
  4) LLM rescore on the Top-K survivors (optional).
  5) Citation expansion: fetch references/citations of highly-relevant seeds (optional).
  6) Score expanded papers and merge them back into the pool.
  7) Intent-aware threshold + max_return truncation.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

from scholar_ir.llm import deepseek_chat, deepseek_configured
from scholar_ir.eval import is_arxiv_id
from scholar_ir.search.s2_client import (
    get_paper_citations,
    get_paper_references,
    s2_configured,
)
from scholar_ir.types import JudgeResult, PaperRef, ScoredPaper, UnderstandingResult

# 轻量停用词表，仅用于关键词覆盖度计算
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "about", "regarding", "by", "as", "at", "from", "since", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "please", "could", "you",
    "find", "papers", "paper", "provide", "list", "show", "search", "get",
    "what", "how", "any", "some", "me", "tell", "i", "we",
    # 泛化词汇：在学术查询中通常不具区分性
    "based", "using", "via", "techniques", "methods", "approaches",
}


def _normalize_text(text: str) -> str:
    """统一转小写并压缩空白。"""
    return " ".join((text or "").lower().split())


def _extract_tokens(text: str) -> Set[str]:
    """抽取有意义的英文/数字 token，去掉停用词和过短词。"""
    tokens: Set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()):
        tok = raw.strip("-")
        if len(tok) <= 1 or tok in _STOPWORDS:
            continue
        tokens.add(tok)
    return tokens


def _paper_text(paper: PaperRef) -> str:
    """合并论文标题和摘要作为判定文本。"""
    return _normalize_text(f"{paper.title} {paper.abstract}")


def _paper_key(paper: PaperRef) -> str:
    """去重 key：优先 paper_id，否则规范化标题。"""
    key = (paper.paper_id or "").strip()
    if key:
        return key
    return re.sub(r"[^a-z0-9]+", "", (paper.title or "").lower())


def _deduplicate(candidates: List[PaperRef]) -> List[PaperRef]:
    """按 paper_id 去重；无 id 时按规范化标题去重。"""
    seen: Set[str] = set()
    out: List[PaperRef] = []
    for p in candidates:
        key = _paper_key(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _year_ok(paper: PaperRef, slots: Dict[str, Any]) -> bool:
    """年份硬约束。论文无年份时不过滤。"""
    year = paper.year
    if year is None:
        return True
    yf = slots.get("year_from")
    yt = slots.get("year_to")
    if yf is not None and year < int(yf):
        return False
    if yt is not None and year > int(yt):
        return False
    return True


def _negation_hit(paper: PaperRef, negations: List[str]) -> bool:
    """否定词过滤：否定短语完整出现在标题/摘要中。"""
    if not negations:
        return False
    text = _paper_text(paper)
    for neg in negations:
        neg_norm = _normalize_text(neg)
        if not neg_norm:
            continue
        # 优先整短语匹配；避免单个词被切分后误伤
        if neg_norm in text:
            return True
    return False


def _collect_query_tokens(understanding: UnderstandingResult) -> Set[str]:
    """收集查询侧所有关键词，包括同义词、变体、实例。

    来源：raw_question、slots、terms、query_skeleton、relevance_criteria。
    """
    slots = understanding.slots or {}
    parts: List[str] = [understanding.raw_question]

    # 顶层槽位
    for key in ("topic", "method", "dataset", "domain"):
        val = slots.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # terms 层：text / abbrev / synonyms / instances
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

    # query_skeleton 层：core_text / parts / variants
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

    # relevance criteria 描述
    for crit in understanding.relevance_criteria:
        if isinstance(crit, dict):
            desc = crit.get("description")
            if isinstance(desc, str) and desc.strip():
                parts.append(desc.strip())

    tokens: Set[str] = set()
    for p in parts:
        tokens.update(_extract_tokens(p))
    return tokens


def _keyword_coverage_score(
    paper: PaperRef,
    understanding: UnderstandingResult,
) -> float:
    """基于查询侧完整关键词集合与论文标题/摘要的覆盖度打分。

    查询侧已包含同义词和变体，因此对 broad/survey 查询更宽容。
    """
    query_tokens = _collect_query_tokens(understanding)
    if not query_tokens:
        return 0.5

    paper_tokens = _extract_tokens(_paper_text(paper))
    if not paper_tokens:
        return 0.0

    matched = query_tokens & paper_tokens
    coverage = len(matched) / len(query_tokens)

    # 标题命中的权重更高
    title_tokens = _extract_tokens(paper.title or "")
    title_matched = query_tokens & title_tokens
    title_bonus = 0.15 * (len(title_matched) / max(len(query_tokens), 1))

    score = min(1.0, coverage + title_bonus)
    return round(score, 4)


def _intent_threshold(intent: str, override: Optional[float]) -> float:
    """根据 intent 选择默认 threshold；用户可显式覆盖。"""
    if override is not None:
        return float(override)
    intent_norm = (intent or "").lower()
    if intent_norm in {"survey", "broad"}:
        return 0.05
    if intent_norm == "method":
        return 0.15
    return 0.25


def _score_paper(
    paper: PaperRef,
    understanding: UnderstandingResult,
    options: Dict[str, Any],
) -> Tuple[float, str]:
    """对单篇论文打分并给出理由（仅规则 + 关键词，不含 LLM）。"""
    slots = understanding.slots or {}

    # 1. 年份硬约束
    if options.get("rule_year", True) and not _year_ok(paper, slots):
        return 0.0, "year_mismatch"

    # 2. 否定词过滤
    negations = slots.get("negation") or []
    if options.get("rule_negation", True) and _negation_hit(paper, negations):
        return 0.0, "negation_hit"

    # 3. 轻量关键词覆盖度打分（已包含 topic/method 及同义词）
    score = _keyword_coverage_score(paper, understanding)
    reason = f"keyword_coverage:{score}"
    return score, reason


def _s2_dict_to_ref(raw: Dict[str, Any]) -> Optional[PaperRef]:
    """把 S2 paper dict 转成 PaperRef；优先用 arXiv id 作为 paper_id。"""
    ext = raw.get("externalIds") or {}
    arxiv_id = ""
    if isinstance(ext, dict) and ext.get("ArXiv"):
        arxiv_id = str(ext["ArXiv"]).split("/")[-1].split("v")[0]

    paper_id = arxiv_id or raw.get("paperId") or ""
    title = raw.get("title") or ""
    if not paper_id and not title:
        return None

    year = raw.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)

    return PaperRef(
        paper_id=str(paper_id),
        title=title,
        abstract=raw.get("abstract") or "",
        year=year if isinstance(year, int) else None,
        source="semantic",
        raw=raw,
    )


def _build_llm_prompt(
    understanding: UnderstandingResult,
    papers: List[PaperRef],
    seed_notes: Optional[Dict[str, str]] = None,
) -> str:
    """构造 LLM 判定 prompt。可选 seed_notes 说明论文与种子的引用关系。"""
    criteria_lines: List[str] = []
    for crit in understanding.relevance_criteria:
        if isinstance(crit, dict):
            desc = crit.get("description") or ""
            must = crit.get("must_have") or crit.get("must", "")
            if desc:
                criteria_lines.append(f"- {desc}")
            if must:
                criteria_lines.append(f"  必须包含: {must}")

    criteria_text = "\n".join(criteria_lines) if criteria_lines else "无额外约束。"

    paper_blocks: List[str] = []
    for idx, p in enumerate(papers, 1):
        title = (p.title or "").strip()
        abstract = (p.abstract or "").strip()
        if len(abstract) > 900:
            abstract = abstract[:900].rsplit(" ", 1)[0] + " ..."
        note = ""
        if seed_notes and p.paper_id and p.paper_id in seed_notes:
            note = f"\n    relation: {seed_notes[p.paper_id]}"
        block = textwrap.dedent(f"""\
            [{idx}] id: {p.paper_id or 'N/A'}
            title: {title}{note}
            abstract: {abstract}
        """)
        paper_blocks.append(block)

    return textwrap.dedent(f"""\
        You are an expert academic-paper relevance judge.

        User query: {understanding.raw_question}
        Intent: {understanding.intent}

        Relevance criteria:
        {criteria_text}

        Score each candidate paper from 0.0 (irrelevant) to 1.0 (highly relevant).
        If a candidate is related to a seed paper via citation, use that as weak evidence of relevance.
        Output a single JSON array with exactly one object per candidate, in the same order:

        [
          {{"paper_id": "<id>", "score": 0.85, "reason": "brief reason"}},
          ...
        ]

        Candidate papers:
        {''.join(chr(10) + b for b in paper_blocks)}
    """)


def _parse_llm_json(
    text: str,
    expected_ids: List[str],
) -> Dict[str, Tuple[float, str]]:
    """解析 LLM 返回的 JSON array，返回 {paper_id: (score, reason)}。"""
    text = (text or "").strip()
    if not text:
        return {}

    # 有时候模型会包裹在 markdown code block 里
    if "```json" in text:
        text = text.split("```json", 1)[-1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[-1]
        text = text.split("```", 1)[0]

    text = text.strip()
    try:
        data = json.loads(text)
    except Exception:
        # 尝试截取第一个 [ ... ]
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except Exception:
            return {}

    if not isinstance(data, list):
        return {}

    result: Dict[str, Tuple[float, str]] = {}
    expected_set = set(expected_ids)
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = item.get("paper_id")
        if pid not in expected_set:
            continue
        score = item.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        score = max(0.0, min(1.0, score))
        reason = str(item.get("reason") or "").strip()
        result[pid] = (score, reason)
    return result


def _llm_rescore_topk(
    understanding: UnderstandingResult,
    papers: List[PaperRef],
    options: Dict[str, Any],
    seed_notes: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, str]]:
    """用 LLM 对 Top-K 候选重新打分。失败返回空 dict。"""
    if not options.get("use_llm", True):
        return {}
    if not papers:
        return {}
    if not deepseek_configured():
        return {}

    prompt = _build_llm_prompt(understanding, papers, seed_notes=seed_notes)
    response = deepseek_chat(
        [
            {"role": "system", "content": "You are a precise academic paper relevance judge. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=float(options.get("llm_temperature", 0.2)),
        max_tokens=int(options.get("llm_max_tokens", 1024)),
    )
    if not response:
        return {}

    return _parse_llm_json(response, [p.paper_id for p in papers if p.paper_id])


def _select_seeds(
    scored_sorted: List[ScoredPaper],
    options: Dict[str, Any],
) -> List[PaperRef]:
    """从第一轮筛选结果里选种子。"""
    seed_top_k = int(options.get("seed_top_k", 3))
    seed_min_score = float(options.get("seed_min_score", 0.3))
    if seed_top_k <= 0:
        return []

    seeds: List[PaperRef] = []
    for s in scored_sorted:
        if s.score < seed_min_score:
            continue
        seeds.append(s.paper)
        if len(seeds) >= seed_top_k:
            break
    return seeds


def _expand_citations(
    seeds: List[PaperRef],
    seen_keys: Set[str],
    options: Dict[str, Any],
) -> Tuple[List[PaperRef], Dict[str, str]]:
    """对种子做引用扩展，返回 (扩展论文列表, seed_notes)。

    seed_notes: {paper_id -> "referenced by / cites seed_title"}
    """
    if not seeds:
        return [], {}
    if not s2_configured():
        return [], {}

    ref_limit = int(options.get("ref_limit", 5))
    cit_limit = int(options.get("cit_limit", 5))
    expand_max_total = int(options.get("expand_max_total", 30))

    expanded: List[PaperRef] = []
    seed_notes: Dict[str, str] = {}

    for seed in seeds:
        seed_title = (seed.title or seed.paper_id or "seed").strip()
        seed_key = _paper_key(seed)

        # 向后扩展：references
        if ref_limit > 0 and len(expanded) < expand_max_total:
            _, refs = get_paper_references(
                seed.paper_id,
                limit=min(ref_limit, expand_max_total - len(expanded)),
            )
            for raw in refs:
                paper = _s2_dict_to_ref(raw)
                if paper is None:
                    continue
                key = _paper_key(paper)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                expanded.append(paper)
                note = f"cited by seed '{seed_title}'"
                if paper.paper_id:
                    seed_notes[paper.paper_id] = note

        # 向前扩展：citations
        if cit_limit > 0 and len(expanded) < expand_max_total:
            _, cits = get_paper_citations(
                seed.paper_id,
                limit=min(cit_limit, expand_max_total - len(expanded)),
            )
            for raw in cits:
                paper = _s2_dict_to_ref(raw)
                if paper is None:
                    continue
                key = _paper_key(paper)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                expanded.append(paper)
                note = f"cites seed '{seed_title}'"
                if paper.paper_id:
                    seed_notes[paper.paper_id] = note

        if len(expanded) >= expand_max_total:
            break

    return expanded, seed_notes


def _score_expanded_papers(
    expanded: List[PaperRef],
    understanding: UnderstandingResult,
    options: Dict[str, Any],
    seed_notes: Dict[str, str],
) -> List[ScoredPaper]:
    """对扩展论文打分（规则 + 关键词 + 可选 LLM）。"""
    # 规则 + 关键词初分
    scored: List[ScoredPaper] = []
    for paper in expanded:
        score, reason = _score_paper(paper, understanding, options)
        scored.append(ScoredPaper(paper=paper, score=score, reason=reason))

    # 可选 LLM 精筛（默认只让 top 扩展论文进 LLM）
    if options.get("use_llm_for_expanded", True):
        llm_top_k_expanded = int(options.get("llm_top_k_expanded", 10))
        survivors = [s for s in scored if s.score > 0.0][:llm_top_k_expanded]
        if survivors:
            llm_map = _llm_rescore_topk(
                understanding,
                [s.paper for s in survivors],
                options,
                seed_notes=seed_notes,
            )
            if llm_map:
                for s in survivors:
                    pid = s.paper.paper_id
                    if pid and pid in llm_map:
                        llm_score, llm_reason = llm_map[pid]
                        s.score = round(llm_score, 4)
                        s.reason = f"expanded_llm:{llm_score};{llm_reason}"

    return scored


def filter_papers(
    understanding: UnderstandingResult,
    candidates: List[PaperRef],
    options: Dict[str, Any] | None = None,
) -> JudgeResult:
    """Coarse filter + lightweight scoring + optional LLM rescore + citation expansion.

    Args:
        understanding: Query understanding result with intent/slots/criteria.
        candidates: Papers returned from search stage.
        options: Controls for threshold, max_return, and rule switches.
            - threshold: minimum score to keep (auto by intent if omitted)
            - max_return: maximum papers to return (default 20)
            - arxiv_only: if True (default), drop non-arXiv ids before truncation
              so W… / S2 hash do not occupy max_return slots (PaSa-friendly)
            - rule_year: enable year filtering (default True)
            - rule_negation: enable negation filtering (default True)
            - use_llm: enable LLM rescore on Top-K survivors (default True)
            - llm_top_k: number of survivors to send to LLM (default 15)
            - llm_temperature: LLM temperature (default 0.2)
            - llm_max_tokens: LLM max tokens (default 1024)
            - expand_citations: enable citation expansion (default False)
            - seed_top_k: number of seed papers (default 3)
            - seed_min_score: minimum score to be a seed (default 0.3)
            - ref_limit: references per seed (default 5)
            - cit_limit: citations per seed (default 5)
            - expand_max_total: max expanded papers (default 30)
            - use_llm_for_expanded: LLM judge for expanded papers (default True)
            - llm_top_k_expanded: LLM top-k for expanded papers (default 10)

    Returns:
        JudgeResult with scored/selected/paper_ids.
    """
    options = options or {}
    max_return = int(options.get("max_return", 20))
    arxiv_only = bool(options.get("arxiv_only", True))
    threshold = _intent_threshold(
        understanding.intent,
        options.get("threshold"),
    )
    llm_top_k = int(options.get("llm_top_k", 15))

    # 去重
    deduped = _deduplicate(candidates)

    scored: List[ScoredPaper] = []
    for paper in deduped:
        score, reason = _score_paper(paper, understanding, options)
        scored.append(ScoredPaper(paper=paper, score=score, reason=reason))

    # 按关键词分数初步排序
    scored_sorted = sorted(scored, key=lambda s: s.score, reverse=True)

    # 选出 Top-K 幸存者交给 LLM 精判（排除硬规则丢弃的 score=0）
    if llm_top_k > 0:
        survivors_for_llm = [
            s for s in scored_sorted if s.score > 0.0
        ][:llm_top_k]
        llm_map = _llm_rescore_topk(
            understanding,
            [s.paper for s in survivors_for_llm],
            options,
        )
        if llm_map:
            for s in survivors_for_llm:
                pid = s.paper.paper_id
                if pid and pid in llm_map:
                    llm_score, llm_reason = llm_map[pid]
                    s.score = round(llm_score, 4)
                    s.reason = f"llm:{llm_score};{llm_reason}"

    # 用 LLM 更新后的分数重新排序
    scored_sorted = sorted(scored, key=lambda s: s.score, reverse=True)

    # 引用扩展（可选）
    if options.get("expand_citations", False):
        seeds = _select_seeds(scored_sorted, options)
        if seeds:
            seen_keys = {_paper_key(s.paper) for s in scored_sorted}
            expanded, seed_notes = _expand_citations(seeds, seen_keys, options)
            if expanded:
                expanded_scored = _score_expanded_papers(
                    expanded, understanding, options, seed_notes
                )
                scored_sorted.extend(expanded_scored)
                scored_sorted = sorted(scored_sorted, key=lambda s: s.score, reverse=True)

    survivors = [s.paper for s in scored_sorted if s.score >= threshold]
    if arxiv_only:
        # Non-arXiv ids never match PaSa gold; drop before max_return so they
        # do not crowd out arXiv papers.
        survivors = [
            p for p in survivors if p.paper_id and is_arxiv_id(p.paper_id)
        ]
    selected = survivors[:max_return]

    return JudgeResult(
        scored=scored_sorted,
        selected=selected,
        paper_ids=[p.paper_id for p in selected if p.paper_id],
    )


# Backward-compatible alias
judge = filter_papers
