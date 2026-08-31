"""Filter (stage-2): coarse rules + lightweight relevance scoring + optional citation expansion.

Channel contract with query_understanding.slot_usage:
  - api_filter slots (year_from/year_to/venue/authors): enforced at search stage;
    year_from/year_to re-checked here as a defense-in-depth hard rule.
  - query_material slots (topic, method, dataset, domain, terms, query_skeleton):
    consumed via _collect_query_tokens (keyword coverage) and surfaced to the
    LLM judge through relevance_criteria.
  - judge_only slot (negation): enforced here as _negation_hit hard rule.

Pipeline:
  1) Deduplicate candidates by paper_id / normalized title.
  2) Hard rules: year range, explicit negation.
  3) Lightweight keyword-coverage scoring on survivors (with synonyms/variants).
  4) LLM rescore on the Top-K survivors (optional; batched + concurrency degradation).
  5) Citation expansion: fetch references/citations of highly-relevant seeds (optional).
  6) Intent-aware impact blending (citation / recency / venue / title density).
  7) Cross-intent score normalization (threshold → 0.5 on public ``.score``).
  8) Pass-bar (normalized ≥ 0.5) + max_return truncation.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from scholar_ir.embeddings import (
    cosine_similarity,
    embedding_configured,
    encode,
)
from scholar_ir.llm import deepseek_chat, deepseek_configured
from scholar_ir.eval import is_arxiv_id
from scholar_ir.search.dedup import canonical_paper_id
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

# 顶会/顶刊子串（小写），用于 venue 影响力打分
_TOP_VENUES = {
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
    "acl", "emnlp", "naacl", "coling", "tacl",
    "aaai", "ijcai", "kdd", "www", "sigir", "icde", "mlsys",
    "tpami", "ijcv", "jmlr", "tkde", "tods", "tois",
    "nature", "science", "cell",
}

# Impact 子信号按 intent 分配的权重（关键词+LLM+语义相似度+引用+年份+venue+title 密度）。
# 各 intent 的总权重=1。intent 未命中时退回 default。
# embedding 权重从 rel 中划出：语义相似度与关键词相关性度量的是同一件事
# （查询-论文匹配程度），不应挤压 citation/recency 等独立工程特征的份额。
# embedding 服务不可用时，_blend_score 会把该权重回补给 rel。
_IMPACT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "method":   {"rel": 0.68, "embedding": 0.12, "citation": 0.05, "recency": 0.05, "venue": 0.04, "title": 0.06},
    "survey":   {"rel": 0.45, "embedding": 0.10, "citation": 0.18, "recency": 0.10, "venue": 0.10, "title": 0.07},
    "specific": {"rel": 0.66, "embedding": 0.12, "citation": 0.04, "recency": 0.04, "venue": 0.04, "title": 0.10},
    "broad":    {"rel": 0.66, "embedding": 0.12, "citation": 0.06, "recency": 0.06, "venue": 0.05, "title": 0.05},
    "default":  {"rel": 0.66, "embedding": 0.12, "citation": 0.06, "recency": 0.06, "venue": 0.05, "title": 0.05},
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


def _paper_key(paper: PaperRef, *, index: Optional[int] = None) -> str:
    """Cross-source stable key: arxiv > DOI > title+pid > index sentinel.

    `index` disambiguates papers with no identifier at all (last-resort).
    """
    return canonical_paper_id(paper, index=index)


def _deduplicate(candidates: List[PaperRef]) -> List[PaperRef]:
    """Deduplicate candidates across sources by canonical id."""
    seen: Set[str] = set()
    out: List[PaperRef] = []
    for i, p in enumerate(candidates):
        key = _paper_key(p, index=i)
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


# ---------- Impact signals (merged from ranking) ----------


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


def _citation_score(paper: PaperRef, max_citations: int) -> float:
    if max_citations <= 0:
        return 0.0
    return math.log1p(_citation_count(paper)) / math.log1p(max_citations)


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


def _impact_weights(intent: str, override: Optional[Dict[str, float]]) -> Dict[str, float]:
    """选择 intent 对应的权重表；用户可显式覆盖。"""
    if override:
        return dict(override)
    intent_norm = (intent or "").lower()
    return dict(_IMPACT_WEIGHTS.get(intent_norm) or _IMPACT_WEIGHTS["default"])


def _compute_impact_stats(
    papers: List[PaperRef],
) -> Tuple[int, int, int]:
    """统计引用数与年份的最大最小值，用于 citation / recency 归一化。"""
    citations = [_citation_count(p) for p in papers]
    years = [p.year for p in papers if p.year is not None]
    max_citations = max(citations) if citations else 0
    min_year = min(years) if years else 2000
    max_year = max(years) if years else 2026
    return max_citations, min_year, max_year


def _impact_features(
    paper: PaperRef,
    query_tokens: Set[str],
    max_citations: int,
    min_year: int,
    max_year: int,
) -> Dict[str, float]:
    """单篇论文的 4 个外部影响子信号。"""
    return {
        "citation": round(_citation_score(paper, max_citations), 4),
        "recency": round(_recency_score(paper, min_year, max_year), 4),
        "venue": round(_venue_score(paper), 4),
        "title": round(_title_density_score(paper, query_tokens), 4),
    }


def _embed_query_text(understanding: UnderstandingResult) -> str:
    """query 侧编码文本：优先 query_skeleton.core_text。

    core_text 已由 understanding 剥离「Which papers ...」这类疑问外壳，
    比原始问句更接近论文摘要的陈述式表述；缺失时回退到原始问题。
    """
    slots = understanding.slots or {}
    skeleton = slots.get("query_skeleton")
    if isinstance(skeleton, dict):
        core = (skeleton.get("core_text") or "").strip()
        if core:
            return core
    return (understanding.raw_question or "").strip()


def _embed_doc_text(paper: PaperRef) -> str:
    """doc 侧编码文本：title + abstract。

    不含 venue / year —— 它们已在 _venue_score / _recency_score 里作为
    独立特征参与融合，重复放进向量文本会重复计权。
    """
    parts = [paper.title or ""]
    abstract = getattr(paper, "abstract", "") or ""
    if abstract:
        parts.append(abstract)
    return " ".join(p for p in parts if p).strip()


def _embedding_scores(
    understanding: UnderstandingResult,
    papers: List[PaperRef],
) -> List[float]:
    """批量计算 query 与各论文的语义相似度，返回与 ``papers`` 等长的分数列表。

    服务未配置或任一环节失败时返回空列表，由调用方降级（不阻断主流程）。
    无摘要且无标题的论文得 0 分。
    """
    if not papers or not embedding_configured():
        return []

    query_text = _embed_query_text(understanding)
    if not query_text:
        return []

    doc_texts = [_embed_doc_text(p) for p in papers]
    # 只对非空文本请求编码，空文本直接给 0
    fillable = [i for i, t in enumerate(doc_texts) if t]
    if not fillable:
        return []

    vectors = encode([query_text] + [doc_texts[i] for i in fillable])
    if not vectors or len(vectors) != len(fillable) + 1:
        return []

    query_vector = vectors[0]
    scores = [0.0] * len(papers)
    for pos, vec in zip(fillable, vectors[1:]):
        scores[pos] = round(cosine_similarity(query_vector, vec), 4)
    return scores


def _blend_score(
    relevance: float,
    impact: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """rel (keyword+llm) 与 impact 子信号按 weights 融合，归一到 [0, 1]。

    ``impact`` 缺少 embedding 键时（服务不可用），其权重回补给 rel，
    保证总权重恒为 1，分数尺度与阈值判定不因降级而漂移。
    """
    rel_w = float(weights.get("rel", 0.0))
    emb_w = float(weights.get("embedding", 0.0))
    if "embedding" in impact:
        emb_term = emb_w * float(impact["embedding"])
    else:
        rel_w += emb_w
        emb_term = 0.0
    rel = max(0.0, min(1.0, relevance))
    raw = (
        rel_w * rel
        + emb_term
        + float(weights.get("citation", 0.0)) * float(impact["citation"])
        + float(weights.get("recency", 0.0)) * float(impact["recency"])
        + float(weights.get("venue", 0.0)) * float(impact["venue"])
        + float(weights.get("title", 0.0)) * float(impact["title"])
    )
    return round(max(0.0, min(1.0, raw)), 4)


def _normalize_cross_intent(score: float, threshold: float) -> float:
    """把 raw score 映射到跨 intent 可比的尺度：threshold → 0.5。

    不同 intent 的 `_intent_threshold` / impact 权重不同，同一 raw 分不可直接比。
    以该 intent 的通过门槛为锚点做分段线性映射：
      - score <= 0           → 0（硬规则丢弃）
      - [0, threshold]       → [0, 0.5]
      - [threshold, 1]       → [0.5, 1]
    这样「刚好过线」在所有 intent 上都是 ~0.5，下游 organize / ranking
    可用统一阈值解读。
    """
    s = max(0.0, min(1.0, float(score)))
    if s <= 0.0:
        return 0.0
    t = max(0.0, min(1.0, float(threshold)))
    if t <= 0.0:
        # 无通过门槛：保持原分
        return round(s, 4)
    if s >= t:
        span = max(1e-9, 1.0 - t)
        return round(0.5 + 0.5 * (s - t) / span, 4)
    return round(0.5 * s / t, 4)


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


def _concurrency_schedule(configured: int) -> List[int]:
    """并发降级阶梯，例如 8 -> [8, 4, 1]。"""
    schedule: List[int] = []
    for tier in (32, 16, 8, 4, 1):
        if tier <= configured and tier not in schedule:
            schedule.append(tier)
    if not schedule:
        schedule.append(1)
    elif schedule[-1] != 1:
        schedule.append(1)
    return schedule


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate_limit" in text or "too many requests" in text


def _score_one_batch(
    understanding: UnderstandingResult,
    papers: List[PaperRef],
    options: Dict[str, Any],
    seed_notes: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, str]]:
    """对一个 batch 调用一次 LLM。失败抛异常，由调用方决定是否重试。"""
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
        raise RuntimeError("empty LLM response")

    parsed = _parse_llm_json(response, [p.paper_id for p in papers if p.paper_id])
    if not parsed:
        raise RuntimeError("unparseable LLM response")
    return parsed


def _llm_rescore_topk(
    understanding: UnderstandingResult,
    papers: List[PaperRef],
    options: Dict[str, Any],
    seed_notes: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[float, str]]:
    """用 LLM 对 Top-K 候选分 batch 并发打分。

    - 候选按 llm_batch_size 切分，避免单次 prompt 过长导致截断。
    - 并发按 llm_concurrency 起步，失败的 batch 在更低并发档位重试（如 8 -> 4 -> 1）。
    - 触发限流时立即整体降档，避免继续打满。
    - 全部失败返回空 dict，调用方沿用关键词分数。
    """
    if not options.get("use_llm", True):
        return {}
    if not papers:
        return {}
    if not deepseek_configured():
        return {}

    batch_size = max(1, int(options.get("llm_batch_size", 8)))
    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]

    result: Dict[str, Tuple[float, str]] = {}
    if len(batches) == 1:
        try:
            return _score_one_batch(understanding, batches[0], options, seed_notes)
        except Exception:
            return {}

    configured = min(32, max(1, int(options.get("llm_concurrency", 8))))
    remaining = list(enumerate(batches, 1))
    for tier in _concurrency_schedule(configured):
        if not remaining:
            break
        failed: List[Tuple[int, List[PaperRef]]] = []
        worklist = list(remaining)
        backoff = False
        for start in range(0, len(worklist), tier):
            chunk = worklist[start : start + tier]
            with ThreadPoolExecutor(max_workers=tier) as executor:
                futures = {
                    executor.submit(
                        _score_one_batch, understanding, batch, options, seed_notes
                    ): (idx, batch)
                    for idx, batch in chunk
                }
                for future in as_completed(futures):
                    idx, batch = futures[future]
                    try:
                        result.update(future.result())
                    except Exception as exc:
                        failed.append((idx, batch))
                        if _is_rate_limit_error(exc):
                            backoff = True
            if backoff:
                failed.extend(worklist[start + len(chunk) :])
                break
        remaining = failed

    return result


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
    """Coarse filter + lightweight scoring + optional LLM rescore + citation expansion
    + impact-aware blending.

    Pipeline:
      1) Deduplicate candidates by paper_id / normalized title.
      2) Hard rules: year range, explicit negation.
      3) Lightweight keyword-coverage scoring on survivors (with synonyms/variants).
      4) LLM rescore on the Top-K survivors (optional; batched + concurrency degradation).
      5) Citation expansion: fetch references/citations of highly-relevant seeds (optional).
      6) Intent-aware impact blending: blend relevance score with citation/recency/venue/
         title_density using intent-specific weights, then re-sort.
      7) Cross-intent score normalization (threshold → 0.5) + max_return truncation.

    Args:
        understanding: Query understanding result with intent/slots/criteria.
        candidates: Papers returned from search stage.
        options: Controls for threshold, max_return, and rule switches.
            - threshold: raw-score pass bar before normalization (auto by intent if omitted)
            - max_return: maximum papers to return (default 20)
            - arxiv_only: if True, drop non-arXiv ids before truncation so W… /
              S2 hash do not occupy max_return slots (PaSa-friendly).
              Default False; eval/smoke should pass True when scoring arXiv gold.
            - rule_year: enable year filtering (default True)
            - rule_negation: enable negation filtering (default True)
            - use_llm: enable LLM rescore on Top-K survivors (default True)
            - llm_top_k: number of survivors to send to LLM (default 15)
            - llm_batch_size: candidates per LLM call (default 8)
            - llm_concurrency: initial parallel LLM calls, degrades on failure (default 8)
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
            - apply_impact: enable impact blending (default True). Set False for legacy
              behavior where the pre-normalize score == relevance only.
            - use_embedding: enable semantic similarity signal (default True). Requires
              EMBEDDING_API_* config; silently degrades when unavailable.
            - embedding_top_k: max papers sent to the embedding service (default 100).
            - impact_weights: override intent-aware weight table (Dict[str, float]).

    Note:
        ``understanding.relevance_criteria`` (including required method/topic) are
        soft signals for the LLM judge prompt. Keyword path does **not** hard-zero
        papers that omit a required method string (avoids brittle recall loss).

    Returns:
        JudgeResult with scored/selected/paper_ids. Each ScoredPaper.features contains
        sub-signals: keyword_coverage, llm (optional), relevance, citation, recency,
        venue, title, blended (if impact on), normalized (final ``.score``).
        ``.score`` is always the cross-intent normalized value (pass bar ≈ 0.5).
    """
    options = options or {}
    max_return = int(options.get("max_return", 20))
    arxiv_only = bool(options.get("arxiv_only", False))
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
        # 在 features 里记录 keyword 子信号
        kw_score = float(score) if "keyword_coverage" in reason else 0.0
        scored.append(
            ScoredPaper(
                paper=paper,
                score=score,
                reason=reason,
                features={"keyword_coverage": round(kw_score, 4)},
            )
        )

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
                    s.features["llm"] = round(llm_score, 4)

    # 用 LLM 更新后的分数重新排序
    scored_sorted = sorted(scored_sorted, key=lambda s: s.score, reverse=True)

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

    # Intent-aware impact blending (embedding / citation / recency / venue / title density)
    if options.get("apply_impact", True):
        weights = _impact_weights(understanding.intent, options.get("impact_weights"))
        # 排除硬规则丢弃的 score=0：它们没机会跑 impact（除非显式要求）
        all_papers = [s.paper for s in scored_sorted]
        max_citations, min_year, max_year = _compute_impact_stats(all_papers)
        query_tokens = _collect_query_tokens(understanding)

        # 语义相似度：只对通过硬规则的论文编码，省调用；失败返回空列表后降级
        emb_by_id: Dict[int, float] = {}
        if options.get("use_embedding", True):
            emb_candidates = [s for s in scored_sorted if s.score > 0.0]
            emb_top_k = int(options.get("embedding_top_k", 100))
            if emb_top_k > 0:
                emb_candidates = emb_candidates[:emb_top_k]
            if emb_candidates:
                emb_scores = _embedding_scores(
                    understanding, [s.paper for s in emb_candidates]
                )
                if emb_scores:
                    emb_by_id = {
                        id(s): sc for s, sc in zip(emb_candidates, emb_scores)
                    }

        for s in scored_sorted:
            if s.score <= 0.0:
                # 硬规则丢弃的论文：保持 score=0 + 标记原因，不参与 impact
                s.features["impact_skipped"] = True
                continue
            impact = _impact_features(
                s.paper, query_tokens, max_citations, min_year, max_year
            )
            if id(s) in emb_by_id:
                impact["embedding"] = emb_by_id[id(s)]
            s.features.update(impact)
            rel_for_blend = float(
                s.features.get("llm", s.features.get("keyword_coverage", 0.0))
            )
            s.features["relevance"] = round(rel_for_blend, 4)
            new_score = _blend_score(rel_for_blend, impact, weights)
            s.features["blended"] = new_score
            s.score = new_score
            # reason 加入 impact 概览
            emb_part = (
                f"emb={impact['embedding']}," if "embedding" in impact else ""
            )
            impact_summary = (
                f"impact[{emb_part}"
                f"cit={impact['citation']},"
                f"rec={impact['recency']},"
                f"ven={impact['venue']},"
                f"ttl={impact['title']}]"
            )
            s.reason = f"{s.reason};{impact_summary}"
        scored_sorted = sorted(scored_sorted, key=lambda s: s.score, reverse=True)

    # Cross-intent normalization: threshold → 0.5 on the public `.score`
    for s in scored_sorted:
        if s.score <= 0.0:
            s.features.setdefault("relevance", 0.0)
            s.features["normalized"] = 0.0
            s.score = 0.0
            continue
        if "relevance" not in s.features:
            s.features["relevance"] = round(
                float(s.features.get("llm", s.features.get("keyword_coverage", 0.0))),
                4,
            )
        raw = float(s.score)
        norm = _normalize_cross_intent(raw, threshold)
        s.features["normalized"] = norm
        s.score = norm
    scored_sorted = sorted(scored_sorted, key=lambda s: s.score, reverse=True)

    # After normalization, pass bar is 0.5 (threshold→0.5); threshold<=0 keeps all.
    pass_bar = 0.5 if threshold > 0.0 else 0.0
    survivors = [s.paper for s in scored_sorted if s.score >= pass_bar]
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
