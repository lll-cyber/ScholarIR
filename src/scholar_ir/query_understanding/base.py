"""Stage (1) QueryUnderstanding: NL → UnderstandingResult.

Pipeline:
  1) Round 1: intent + slots (terms[] + query_skeleton + coverage_gap_likely)
  2) Slot Usage → api_filters + high-conf lexical ExpansionTask[] (no pad-to-N)
  3) If coverage_gap_likely: optional semantic reformulation (conceptual)
  4) Assemble SubQuery
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from scholar_ir.types import SubQuery, UnderstandingResult
from scholar_ir.query_understanding.assemble import assemble_from_tasks
from scholar_ir.query_understanding.flow_log import format_understanding_flow
from scholar_ir.query_understanding.llm_extract import try_llm_extract
from scholar_ir.query_understanding.rules import extract_slots_heuristic
from scholar_ir.query_understanding.slot_usage import ExpansionTask, apply_slot_usage, finalize_task_budget
from scholar_ir.query_understanding.slots import empty_slots, ensure_topic_term, slots_to_criteria
from scholar_ir.query_understanding.variant_quality import resolve_coverage_gap_likely

logger = logging.getLogger("scholar_ir.query_understanding")


def _task_snap(t: ExpansionTask) -> Dict[str, Any]:
    return {
        "transform": t.transform,
        "mode": t.mode,
        "text_seed": t.text_seed,
        "source": t.source,
        "swapped_part": t.swapped_part or None,
        "modifiers": list(t.modifiers or []),
    }


def understand(question: str, options: Dict[str, Any] | None = None) -> UnderstandingResult:
    options = options or {}
    max_n = int(options.get("max_subqueries", 5))
    max_lexical_swaps = int(options.get("max_lexical_swaps", 3))
    use_llm = bool(options.get("use_llm", True))
    recall_mode = bool(options.get("recall_mode", True))
    enable_decomposition = bool(options.get("enable_decomposition", False))
    enable_semantic = bool(options.get("enable_semantic", True))
    max_semantic = int(options.get("max_semantic", 2))
    use_llm_expand = bool(options.get("use_llm_expand", False))
    verbose = bool(options.get("verbose", False))
    raw_only = bool(options.get("raw_only", False))
    add_survey_modifier = bool(options.get("add_survey_modifier", False))

    trace: List[Dict[str, Any]] = []

    # --- Baseline: single raw question, no LLM understanding ---
    if raw_only:
        q_clean = (question or "").strip()
        result = UnderstandingResult(
            raw_question=question,
            intent="broad",
            slots=empty_slots(),
            relevance_criteria=[
                {
                    "name": "topic",
                    "type": "topic",
                    "text": q_clean,
                    "required": True,
                    "description": f"Paper is about: {q_clean}",
                    "weight": 1.0,
                }
            ],
            sub_queries=[
                SubQuery(
                    qid="q0",
                    text=q_clean,
                    channel="keyword",
                    angle="raw",
                    mode="lexical",
                    angle_source="raw_only",
                )
            ],
            trace=[
                {
                    "step": "1_raw_baseline",
                    "detail": "raw_only=True → skip Round1/SlotUsage/semantic; one SubQuery = NL",
                }
            ],
        )
        if verbose:
            logger.info("\n%s", format_understanding_flow(result, title="[understanding:raw]"))
        return result

    intent = "broad"
    slots: Dict[str, Any] = empty_slots()
    source = "heuristic"

    # --- Round 1 extract ---
    if use_llm:
        llm_out = try_llm_extract(question)
        if llm_out is not None:
            intent, slots = llm_out
            source = "llm"

    if source == "heuristic":
        intent, slots = extract_slots_heuristic(question)

    slots = ensure_topic_term(slots)
    sk = slots.get("query_skeleton") if isinstance(slots.get("query_skeleton"), dict) else None
    # Single source of truth: resolve once, write back, all logs read slots[...]
    gap = resolve_coverage_gap_likely(slots)
    semantic_reserve = 0
    if use_llm and enable_semantic and not use_llm_expand and gap:
        semantic_reserve = min(max_semantic, max(1, max_n // 2))

    # --- Slot usage / lexical tasks ---
    plan = apply_slot_usage(
        intent,
        slots,
        recall_mode=recall_mode,
        max_n=max_n,
        max_lexical_swaps=max_lexical_swaps,
        enable_decomposition=enable_decomposition,
        question=question,
        coverage_gap_likely=gap,
        semantic_reserve=semantic_reserve,
        add_survey_modifier=add_survey_modifier,
    )
    trace.append(
        {
            "step": "1_round1_extract",
            "detail": f"intent={intent}; extract_source={source}",
            "source": source,
            "coverage_gap_likely": gap,
            "claim": slots.get("claim"),
            "skeleton": {
                "core_text": (sk or {}).get("core_text"),
                "parts": (sk or {}).get("parts"),
            }
            if sk
            else None,
            "n_terms": len(slots.get("terms") or []),
            "n_term_groups": len(slots.get("term_groups") or []),
        }
    )
    sk2 = slots.get("query_skeleton") if isinstance(slots.get("query_skeleton"), dict) else sk
    trace.append(
        {
            "step": "2_slot_usage_lexical",
            "detail": (
                f"build ExpansionTask from core_text + high-conf span swaps "
                f"(max_lexical_swaps={max_lexical_swaps}, no pad-to-{max_n})"
            ),
            "api_filters": dict(plan.api_filters),
            "n_tasks": len(plan.tasks),
            "tasks": [_task_snap(t) for t in plan.tasks],
            "skeleton": {
                "core_text": (sk2 or {}).get("core_text"),
                "parts": (sk2 or {}).get("parts"),
            }
            if sk2
            else None,
        }
    )

    # --- Semantic reformulation ---
    n_sem_added = 0
    if use_llm and enable_semantic and not use_llm_expand and gap:
        remaining = max_n - len(plan.tasks)
        n_sem = min(max_semantic, max(0, remaining))
        detail = (
            f"coverage_gap_likely=True → request up to {n_sem} conceptual reformulations"
            + (f" (intent={intent})" if intent == "specific" else "")
        )
        reforms: List[str] = []
        if n_sem > 0:
            try:
                from scholar_ir.query_understanding.llm_semantic import (
                    try_llm_semantic_reformulate,
                )

                existing = [t.text_seed for t in plan.tasks]
                reforms = try_llm_semantic_reformulate(
                    question,
                    intent,
                    slots,
                    n=n_sem,
                    existing_texts=existing,
                )
                for text in reforms:
                    plan.tasks.append(
                        ExpansionTask(
                            mode="semantic",
                            transform="conceptual",
                            source="llm_semantic_reformulate",
                            text_seed=text,
                        )
                    )
                    n_sem_added += 1
            except Exception as e:
                detail += f"; semantic failed: {e}"
        trace.append(
            {
                "step": "3_semantic_reformulation",
                "detail": detail,
                "coverage_gap_likely": gap,
                "n_semantic": n_sem_added,
                "tasks": [_task_snap(t) for t in plan.tasks if t.transform == "conceptual"],
            }
        )
    else:
        reason = []
        if not gap:
            reason.append("coverage_gap_likely=False")
        if not enable_semantic:
            reason.append("enable_semantic=False")
        if not use_llm:
            reason.append("use_llm=False")
        if use_llm_expand:
            reason.append("use_llm_expand=True")
        trace.append(
            {
                "step": "3_semantic_reformulation",
                "detail": "skipped (" + ", ".join(reason or ["n/a"]) + ")",
                "coverage_gap_likely": gap,
                "n_semantic": 0,
            }
        )

    if use_llm_expand and use_llm:
        try:
            from scholar_ir.query_understanding.llm_expand import try_llm_expand

            expanded = try_llm_expand(question, intent, slots, {}, n=max_n)
            if expanded:
                plan.tasks = [
                    ExpansionTask(
                        mode="lexical",
                        transform="core" if i == 0 else "synonym",
                        source="llm_expand_legacy",
                        text_seed=text,
                    )
                    for i, (text, _angle) in enumerate(expanded[:max_n])
                ]
                trace.append(
                    {
                        "step": "3b_legacy_llm_expand",
                        "detail": "use_llm_expand=True replaced tasks with free LLM expand",
                        "n_tasks": len(plan.tasks),
                    }
                )
        except Exception:
            pass

    plan.tasks = finalize_task_budget(plan.tasks, max_n)

    sub_queries = assemble_from_tasks(
        question,
        intent,
        slots,
        plan,
        max_n=max_n,
        add_survey_modifier=add_survey_modifier,
    )
    trace.append(
        {
            "step": "4_assemble",
            "detail": f"ExpansionTask → SubQuery (n={len(sub_queries)}, max_n={max_n})",
            "n_sub_queries": len(sub_queries),
        }
    )

    result = UnderstandingResult(
        raw_question=question,
        intent=intent,
        slots=slots,
        relevance_criteria=slots_to_criteria(intent, slots, question),
        sub_queries=sub_queries,
        trace=trace,
    )
    if verbose:
        logger.info("\n%s", format_understanding_flow(result, title="[understanding:scholar]"))
    return result
