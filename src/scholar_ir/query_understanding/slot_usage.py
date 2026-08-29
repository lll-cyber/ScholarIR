"""Slot Usage Decision + ExpansionTask builder (code policy, not LLM).

Lexical: high-confidence synonym/abbrev swaps on core_text only.
Do not pad to max_n — quality over quota.
Semantic reformulation is appended later when coverage_gap_likely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from scholar_ir.query_understanding.slots import (
    ensure_query_skeleton,
    skeleton_to_text,
)
from scholar_ir.query_understanding.variant_quality import filter_retrieval_variants


@dataclass
class SlotUsage:
    slot: str
    channel: str  # query_material | api_filter | post_filter | judge_only
    fallback: str = ""


@dataclass
class ExpansionTask:
    mode: str  # lexical | decomposition | semantic
    transform: str  # core | synonym | abbrev | entity | metadata | raw | conceptual
    source: str
    text_seed: str  # fully rendered string
    term_index: Optional[int] = None
    modifiers: List[str] = field(default_factory=list)
    swapped_part: str = ""  # skeleton part id replaced vs core


@dataclass
class RetrievalPlan:
    api_filters: Dict[str, Any] = field(default_factory=dict)
    slot_usages: List[SlotUsage] = field(default_factory=list)
    recall_mode: bool = True
    tasks: List[ExpansionTask] = field(default_factory=list)
    use_llm_expand: bool = False
    include_method_in_text: bool = False


def _nonempty(val: Any) -> bool:
    return val is not None and val != "" and val != []


def apply_slot_usage(
    intent: str,
    slots: Dict[str, Any],
    *,
    recall_mode: bool = True,
    max_n: int = 5,
    max_lexical_swaps: int = 3,
    enable_decomposition: bool = False,
    question: str = "",
    coverage_gap_likely: bool = False,
    semantic_reserve: int = 0,
    add_survey_modifier: bool = False,
) -> RetrievalPlan:
    """Project slots → api_filters + ExpansionTask list (budgeted, not padded)."""
    slots = ensure_query_skeleton(slots)
    usages: List[SlotUsage] = []
    api_filters: Dict[str, Any] = {}

    for key in ("year_from", "year_to", "venue", "authors"):
        if _nonempty(slots.get(key)):
            api_filters[key] = slots[key]
            usages.append(SlotUsage(slot=key, channel="api_filter"))

    if slots.get("negation"):
        usages.append(SlotUsage(slot="negation", channel="judge_only"))

    method = slots.get("method")
    if _nonempty(method):
        if recall_mode:
            usages.append(SlotUsage(slot="method", channel="judge_only"))
        else:
            api_filters["method"] = method
            usages.append(SlotUsage(slot="method", channel="api_filter"))

    if slots.get("topic") or slots.get("query_skeleton"):
        usages.append(SlotUsage(slot="query_skeleton", channel="query_material"))

    if intent == "specific":
        enable_decomposition = False

    lexical_budget = max(1, max_n - max(0, semantic_reserve))
    tasks = _build_tasks(
        intent,
        slots,
        max_n=lexical_budget,
        max_lexical_swaps=max_lexical_swaps,
        enable_decomposition=enable_decomposition,
        question=question,
        coverage_gap_likely=coverage_gap_likely,
        add_survey_modifier=add_survey_modifier,
    )

    return RetrievalPlan(
        api_filters=api_filters,
        slot_usages=usages,
        recall_mode=recall_mode,
        tasks=tasks,
        use_llm_expand=False,
        include_method_in_text=bool(method) and intent == "method",
    )


def _variant_transform(canon: str, variant: str) -> str:
    c, v = canon.lower(), variant.lower()
    if v == c:
        return "core"
    if len(variant) <= 6 and variant.isupper():
        return "abbrev"
    if len(canon) <= 6 and canon.isupper() and len(variant) > len(canon):
        return "abbrev"
    return "synonym"


def _looks_like_paper_title(text: str) -> bool:
    t = (text or "").strip()
    if not t or "?" in t:
        return False
    low = t.lower()
    if low.startswith(
        ("list ", "find ", "show ", "papers ", "search ", "get ", "what ", "how ")
    ):
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 3:
        return False
    capped = sum(1 for w in words if w[:1].isupper() and w[:1].isalpha())
    return capped >= max(3, (len(words) + 1) // 2)


def _specific_nav_tasks(
    slots: Dict[str, Any],
    question: str,
    *,
    coverage_gap_likely: bool = False,
) -> List[ExpansionTask]:
    tasks: List[ExpansionTask] = []
    authors = slots.get("authors")
    if authors:
        if isinstance(authors, list):
            author_text = " ".join(str(a) for a in authors if a).strip()
        else:
            author_text = str(authors).strip()
        if author_text:
            tasks.append(
                ExpansionTask(
                    mode="lexical",
                    transform="metadata",
                    source="slots.authors",
                    text_seed=author_text,
                )
            )

    topic = (slots.get("topic") or "").strip()
    if topic and _looks_like_paper_title(topic):
        tasks.append(
            ExpansionTask(
                mode="lexical",
                transform="metadata",
                source="slots.topic",
                text_seed=topic,
            )
        )

    q_clean = re.sub(r"[?？]+$", "", (question or "").strip())
    if q_clean and not coverage_gap_likely:
        low_q = q_clean.lower()
        if low_q != topic.lower() and not _looks_like_paper_title(q_clean):
            tasks.append(
                ExpansionTask(
                    mode="lexical",
                    transform="raw",
                    source="raw_question",
                    text_seed=q_clean,
                )
            )
    return tasks


def _build_lexical_from_skeleton(
    intent: str,
    slots: Dict[str, Any],
    *,
    max_lexical_swaps: int,
    add_survey_modifier: bool = False,
) -> List[ExpansionTask]:
    skeleton = slots.get("query_skeleton")
    if not isinstance(skeleton, dict):
        skeleton = {}
    has_material = bool(skeleton.get("core_text")) or bool(skeleton.get("parts"))

    use_survey = add_survey_modifier and intent == "survey"

    if not has_material:
        topic = (slots.get("topic") or "").strip()
        if not topic:
            return []
        seed = topic
        mods: List[str] = []
        if use_survey:
            mods = ["survey"]
            if "survey" not in seed.lower() and "review" not in seed.lower():
                seed = f"{seed} survey"
        return [
            ExpansionTask(
                mode="lexical",
                transform="core",
                source="slots.topic",
                text_seed=seed,
                modifiers=mods,
            )
        ]

    parts: List[Dict[str, Any]] = list(skeleton.get("parts") or [])
    modifier = "survey" if use_survey else None
    mods = ["survey"] if use_survey else []

    core_text = skeleton_to_text(skeleton, modifier=modifier)
    tasks: List[ExpansionTask] = [
        ExpansionTask(
            mode="lexical",
            transform="core",
            source="slots.query_skeleton",
            text_seed=core_text,
            modifiers=list(mods),
        )
    ]

    # High-confidence alts only; round-robin; stop early (no pad)
    queues: List[tuple[Dict[str, Any], List[str]]] = []
    for p in parts:
        if not p.get("replaceable"):
            continue
        canon = str(p.get("text") or "").strip()
        alts = filter_retrieval_variants(
            canon,
            list(p.get("variants") or []),
            max_alts=2,
        )
        if alts:
            queues.append((p, alts))

    swaps = 0
    while swaps < max_lexical_swaps:
        progressed = False
        for p, alt in queues:
            if swaps >= max_lexical_swaps:
                break
            if not alt:
                continue
            variant = alt.pop(0)
            pid = str(p.get("id") or "")
            canon = str(p.get("text") or "").strip()
            text = skeleton_to_text(
                skeleton,
                overrides={pid: variant},
                modifier=modifier,
            )
            if text.lower() == core_text.lower():
                continue
            tasks.append(
                ExpansionTask(
                    mode="lexical",
                    transform=_variant_transform(canon, variant),
                    source=f"slots.query_skeleton.parts[{pid}]",
                    text_seed=text,
                    modifiers=list(mods),
                    swapped_part=pid,
                )
            )
            swaps += 1
            progressed = True
        if not progressed:
            break
    return tasks


def _build_decomposition_tasks(
    slots: Dict[str, Any],
    mods: List[str],
    modifier: Optional[str],
) -> List[ExpansionTask]:
    skeleton = slots.get("query_skeleton")
    if not isinstance(skeleton, dict):
        return []
    parts: List[Dict[str, Any]] = list(skeleton.get("parts") or [])
    tasks: List[ExpansionTask] = []
    terms = list(slots.get("terms") or [])
    for t in terms:
        if not t.get("coverage_gap_likely"):
            continue
        instances = [str(x).strip() for x in (t.get("instances") or []) if str(x).strip()]
        if not instances:
            continue
        ttext = str(t.get("text") or "").lower()
        tabbrev = str(t.get("abbrev") or "").lower()
        target_pid = None
        for p in parts:
            pt = str(p.get("text") or "").lower()
            if pt == ttext or (tabbrev and pt == tabbrev):
                target_pid = str(p.get("id") or "")
                break
        if not target_pid:
            for p in parts:
                if p.get("replaceable"):
                    target_pid = str(p.get("id") or "")
                    break
        if not target_pid and parts:
            target_pid = str(parts[0].get("id") or "t0")
        for inst in instances:
            text = skeleton_to_text(
                skeleton,
                overrides={target_pid: inst} if target_pid else None,
                modifier=modifier,
            )
            tasks.append(
                ExpansionTask(
                    mode="decomposition",
                    transform="entity",
                    source=f"slots.terms.instances:{inst}",
                    text_seed=text,
                    modifiers=list(mods),
                    swapped_part=target_pid or "",
                )
            )
    return tasks


def finalize_task_budget(tasks: List[ExpansionTask], max_n: int) -> List[ExpansionTask]:
    """Cap tasks without padding. Conceptual beats raw when budget is tight."""
    cores = [t for t in tasks if t.transform == "core"]
    conceptual = [t for t in tasks if t.transform == "conceptual"]
    metas = [t for t in tasks if t.transform == "metadata"]
    raws = [t for t in tasks if t.transform == "raw"]
    others = [
        t
        for t in tasks
        if t.transform not in ("core", "conceptual", "metadata", "raw")
    ]

    out: List[ExpansionTask] = []
    seen: set[str] = set()

    def _add(t: ExpansionTask) -> bool:
        key = " ".join(t.text_seed.lower().split())
        if not key or key in seen:
            return False
        seen.add(key)
        out.append(t)
        return len(out) >= max_n

    for group in (cores, others, conceptual, metas, raws):
        for t in group:
            if _add(t):
                return out
    return out


def _build_tasks(
    intent: str,
    slots: Dict[str, Any],
    *,
    max_n: int,
    max_lexical_swaps: int,
    enable_decomposition: bool,
    question: str = "",
    coverage_gap_likely: bool = False,
    add_survey_modifier: bool = False,
) -> List[ExpansionTask]:
    tasks: List[ExpansionTask] = []

    if intent == "specific":
        tasks.extend(
            _specific_nav_tasks(slots, question, coverage_gap_likely=coverage_gap_likely)
        )

    tasks.extend(
        _build_lexical_from_skeleton(
            intent,
            slots,
            max_lexical_swaps=max_lexical_swaps,
            add_survey_modifier=add_survey_modifier,
        )
    )

    if enable_decomposition and intent != "specific":
        use_survey = add_survey_modifier and intent == "survey"
        modifier = "survey" if use_survey else None
        mods = ["survey"] if use_survey else []
        tasks.extend(_build_decomposition_tasks(slots, mods, modifier))

    return finalize_task_budget(tasks, max_n)


def apply_usage_decision(
    intent: str,
    slots: Dict[str, Any],
    *,
    recall_mode: bool = True,
    use_llm: bool = True,
    max_n: int = 5,
    max_lexical_swaps: int = 3,
    enable_decomposition: bool = False,
    question: str = "",
    add_survey_modifier: bool = False,
) -> RetrievalPlan:
    _ = use_llm
    return apply_slot_usage(
        intent,
        slots,
        recall_mode=recall_mode,
        max_n=max_n,
        max_lexical_swaps=max_lexical_swaps,
        enable_decomposition=enable_decomposition,
        question=question,
        add_survey_modifier=add_survey_modifier,
    )
