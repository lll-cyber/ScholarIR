"""Assemble SubQuery list from ExpansionTask[] + Slot Usage plan."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from scholar_ir.types import SubQuery
from scholar_ir.query_understanding.rules import rewrite_by_intent
from scholar_ir.query_understanding.slot_usage import RetrievalPlan


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip(" ,")


def assemble_from_tasks(
    question: str,
    intent: str,
    slots: Dict[str, Any],
    plan: RetrievalPlan,
    *,
    max_n: int = 5,
    add_survey_modifier: bool = False,
) -> List[SubQuery]:
    """Primary path: expand ExpansionTask → SubQuery (unified for all intents)."""
    filters = dict(plan.api_filters)
    tasks = list(plan.tasks or [])[:max_n]

    if not tasks:
        return _fallback_template(
            question, intent, slots, plan, max_n=max_n,
            add_survey_modifier=add_survey_modifier,
        )

    out: List[SubQuery] = []
    seen: set[str] = set()
    for i, task in enumerate(tasks):
        text = _clean(task.text_seed)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        channel = "metadata" if task.transform == "metadata" else "keyword"
        out.append(
            SubQuery(
                qid=f"q{i}",
                text=text,
                channel=channel,
                filters=dict(filters),
                angle=task.transform,
                mode=task.mode,
                modifiers=list(task.modifiers or []),
                angle_source=task.source,
            )
        )
        if len(out) >= max_n:
            break
    return out or _fallback_template(
        question, intent, slots, plan, max_n=max_n,
        add_survey_modifier=add_survey_modifier,
    )


def _fallback_template(
    question: str,
    intent: str,
    slots: Dict[str, Any],
    plan: RetrievalPlan,
    *,
    max_n: int,
    add_survey_modifier: bool = False,
) -> List[SubQuery]:
    topic = (slots.get("topic") or question).strip()
    texts = rewrite_by_intent(
        intent, topic, question, max_n=max_n,
        add_survey_modifier=add_survey_modifier,
    )
    filters = dict(plan.api_filters)
    return [
        SubQuery(
            qid=f"q{i}",
            text=t,
            channel="keyword",
            filters=dict(filters),
            angle="core",
            mode="lexical",
            angle_source="template",
            modifiers=["survey"] if add_survey_modifier and intent == "survey" and i == 0 else [],
        )
        for i, t in enumerate(texts)
    ]


def assemble_sub_queries(
    question: str,
    intent: str,
    slots: Dict[str, Any],
    plan: RetrievalPlan,
    expanded: List | None = None,
    recall_hints: Dict[str, Any] | None = None,
    *,
    max_n: int = 5,
) -> List[SubQuery]:
    """Public entry. Ignores legacy expanded/recall_hints when plan.tasks present."""
    _ = expanded, recall_hints
    return assemble_from_tasks(question, intent, slots, plan, max_n=max_n)
