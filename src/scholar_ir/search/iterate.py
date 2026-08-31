"""Search iteration controller — broaden / narrow calibration.

参考 paperseek 的迭代校准思路：一次检索后根据候选数量决定
  - 候选过少 → LLM 放宽查询（去掉过窄限定词、加同义词）
  - 候选过多 → LLM 收窄查询（补充核心限定词）
再补充检索一轮并与首轮结果合并去重。

任何 LLM 失败都直接返回首轮结果，不阻断主流程。
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional

from scholar_ir.llm import deepseek_chat, deepseek_configured
from scholar_ir.search.base import retrieve
from scholar_ir.types import PaperRef, RetrievalResult, SubQuery, UnderstandingResult


def _paper_key(paper: PaperRef) -> str:
    return (paper.paper_id or paper.title or "").strip().lower()


def _sample_titles(candidates: List[PaperRef], n: int = 5) -> List[str]:
    return [(p.title or "").strip() for p in candidates[:n] if p.title]


def _build_prompt(
    understanding: UnderstandingResult,
    queries: List[str],
    candidates: List[PaperRef],
    direction: str,
    max_n: int,
) -> str:
    """构造 broaden / narrow prompt。"""
    titles = _sample_titles(candidates)
    titles_text = (
        "\n".join(f"- {t}" for t in titles) if titles else "(no results returned)"
    )
    queries_text = "\n".join(f"- {q}" for q in queries)

    if direction == "broaden":
        goal = textwrap.dedent("""\
            The previous queries returned too few results. Produce BROADER queries:
            - drop overly narrow qualifiers and rare modifiers
            - keep the core research concept intact
            - prefer widely used synonyms or the umbrella term""")
    else:
        goal = textwrap.dedent("""\
            The previous queries returned too many loosely related results. Produce NARROWER queries:
            - add the most discriminative qualifier from the user's intent
            - keep the core research concept intact
            - avoid generic words that match unrelated papers""")

    return textwrap.dedent(f"""\
        You are an expert academic search-query engineer.

        User question: {understanding.raw_question}
        Intent: {understanding.intent}

        Previous queries:
        {queries_text}

        Sample titles returned:
        {titles_text}

        {goal}

        Output ONLY a JSON array of at most {max_n} plain keyword query strings,
        no field labels, no boolean operators, no API parameters:

        ["query one", "query two"]
    """)


def _parse_queries(text: str, max_n: int) -> List[str]:
    """解析 LLM 返回的查询字符串数组。"""
    text = (text or "").strip()
    if not text:
        return []

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    out: List[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            out.append(" ".join(item.split()))
        if len(out) >= max_n:
            break
    return out


def _refine_queries(
    understanding: UnderstandingResult,
    candidates: List[PaperRef],
    direction: str,
    options: Dict[str, Any],
) -> List[SubQuery]:
    """用 LLM 生成 broaden / narrow 后的 sub_queries。"""
    if not deepseek_configured():
        return []

    prev_queries = [sq.text for sq in understanding.sub_queries if sq.text]
    if not prev_queries:
        return []

    max_n = int(options.get("refine_max_queries", 3))
    prompt = _build_prompt(
        understanding, prev_queries, candidates, direction, max_n
    )
    response = deepseek_chat(
        [
            {
                "role": "system",
                "content": "You are a precise academic search-query engineer. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=float(options.get("refine_temperature", 0.3)),
        max_tokens=int(options.get("refine_max_tokens", 512)),
    )
    texts = _parse_queries(response or "", max_n)
    if not texts:
        return []

    # 复用首轮 filters，保持年份等硬约束不丢
    base_filters: Dict[str, Any] = {}
    if understanding.sub_queries:
        base_filters = dict(understanding.sub_queries[0].filters or {})

    seen = {q.lower() for q in prev_queries}
    out: List[SubQuery] = []
    for i, text in enumerate(texts):
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(
            SubQuery(
                qid=f"r{i}",
                text=text,
                channel="keyword",
                filters=dict(base_filters),
                angle=direction,
                mode="lexical",
                angle_source=f"iterate:{direction}",
            )
        )
    return out


def retrieve_iterative(
    understanding: UnderstandingResult,
    options: Dict[str, Any] | None = None,
) -> RetrievalResult:
    """带 broaden / narrow 校准的检索。

    Args:
        understanding: 查询理解结果。
        options: 除 retrieve 的全部选项外，额外支持
            - iterate: 是否启用迭代校准（默认 True）
            - target_min: 候选下限，低于则 broaden（默认 10）
            - target_max: 候选上限，高于则 narrow（默认 200）
            - max_rounds: 额外补充检索轮数（默认 1）
            - refine_max_queries: 每轮生成的新查询数（默认 3）

    Returns:
        合并后的 RetrievalResult，trace 中含 iterate 事件。
    """
    options = options or {}
    result = retrieve(understanding, options)

    if not options.get("iterate", True):
        return result
    if options.get("dry_run", False):
        return result

    target_min = int(options.get("target_min", 10))
    target_max = int(options.get("target_max", 200))
    max_rounds = int(options.get("max_rounds", 1))

    candidates = list(result.candidates)
    seen = {_paper_key(p) for p in candidates}
    trace = list(result.trace)
    n_api_calls = int(result.stats.get("n_api_calls", 0))
    rounds_run = 0

    for round_idx in range(max(0, max_rounds)):
        n = len(candidates)
        if n < target_min:
            direction = "broaden"
        elif n > target_max:
            direction = "narrow"
        else:
            break

        new_queries = _refine_queries(
            understanding, candidates, direction, options
        )
        if not new_queries:
            trace.append({
                "stage": "iterate",
                "round": round_idx + 1,
                "direction": direction,
                "status": "no_new_queries",
                "n_candidates_before": n,
            })
            break

        # 用新 sub_queries 复用同一套检索逻辑
        refined_understanding = UnderstandingResult(
            raw_question=understanding.raw_question,
            intent=understanding.intent,
            slots=understanding.slots,
            relevance_criteria=understanding.relevance_criteria,
            sub_queries=new_queries,
        )
        extra = retrieve(refined_understanding, options)

        added = 0
        for p in extra.candidates:
            key = _paper_key(p)
            if key and key not in seen:
                seen.add(key)
                candidates.append(p)
                added += 1

        trace.extend(extra.trace)
        n_api_calls += int(extra.stats.get("n_api_calls", 0))
        rounds_run += 1
        trace.append({
            "stage": "iterate",
            "round": round_idx + 1,
            "direction": direction,
            "status": "ok",
            "queries": [q.text for q in new_queries],
            "n_candidates_before": n,
            "n_added": added,
            "n_candidates_after": len(candidates),
        })

        if added == 0:
            break

    stats = dict(result.stats)
    stats.update({
        "n_api_calls": n_api_calls,
        "n_candidates": len(candidates),
        "iterate_rounds": rounds_run,
    })

    return RetrievalResult(candidates=candidates, trace=trace, stats=stats)
