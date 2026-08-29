"""Stage (4) 搜索结果归纳整理 — stub.

按用户意图整理搜索结果，返回结构化展示（列表、关系图等）。
当前仅包装 paper_ids / titles。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from scholar_ir.types import JudgeResult, UnderstandingResult


@dataclass
class OrganizeResult:
    """Structured presentation of search results (stub)."""

    items: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    view: str = "list"  # list | graph | ...


def organize(
    understanding: UnderstandingResult,
    ranking_result: JudgeResult,
    options: Dict[str, Any] | None = None,
) -> OrganizeResult:
    """v0 stub: flat list of {paper_id, title, score}."""
    options = options or {}
    view = str(options.get("view", "list"))
    items: List[Dict[str, Any]] = []
    score_map = {s.paper.paper_id: s.score for s in ranking_result.scored}
    for p in ranking_result.selected:
        items.append(
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "score": score_map.get(p.paper_id),
            }
        )
    intent = understanding.intent or "broad"
    summary = f"intent={intent}; n={len(items)} (organize stub)"
    return OrganizeResult(items=items, summary=summary, view=view)
