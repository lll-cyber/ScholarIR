"""Pretty-print / serialize Understanding stage-1 flow for logs."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TextIO

from scholar_ir.types import UnderstandingResult

logger = logging.getLogger("scholar_ir.query_understanding")


def format_understanding_flow(
    result: UnderstandingResult,
    *,
    title: str = "",
) -> str:
    """Human-readable multi-line description of the understanding pipeline."""
    lines: List[str] = []
    if title:
        lines.append(title)
    lines.append(f"Q: {result.raw_question}")
    lines.append(f"intent: {result.intent}")

    for step in result.trace or []:
        name = step.get("step") or step.get("name") or "?"
        detail = step.get("detail") or ""
        lines.append(f"  [{name}] {detail}".rstrip())
        for k in (
            "source",
            "coverage_gap_likely",
            "claim",
            "n_terms",
            "n_term_groups",
            "n_tasks",
            "n_semantic",
            "n_sub_queries",
        ):
            if k in step and step[k] is not None:
                lines.append(f"       {k}={step[k]}")
        if step.get("api_filters"):
            lines.append(f"       api_filters={json.dumps(step['api_filters'], ensure_ascii=False)}")
        if step.get("tasks"):
            for t in step["tasks"]:
                lines.append(
                    f"       task: [{t.get('transform')}|{t.get('mode')}] "
                    f"{t.get('text_seed')}"
                    + (f"  swap={t.get('swapped_part')}" if t.get("swapped_part") else "")
                )
        if step.get("skeleton"):
            sk = step["skeleton"]
            if isinstance(sk, dict):
                lines.append(f"       core_text={sk.get('core_text')!r}")
                for p in sk.get("parts") or []:
                    lines.append(
                        f"       part[{p.get('id')}] {p.get('text')!r} "
                        f"repl={p.get('replaceable')} vars={p.get('variants')}"
                    )

    lines.append("sub_queries:")
    for sq in result.sub_queries:
        bits = [f"angle={sq.angle}", f"mode={sq.mode}"]
        if sq.modifiers:
            bits.append(f"mod={sq.modifiers}")
        if sq.filters:
            bits.append(f"filters={sq.filters}")
        if sq.angle_source:
            bits.append(f"src={sq.angle_source}")
        lines.append(f"  - [{sq.qid}] {sq.text}  ({', '.join(bits)})")

    gap = (result.slots or {}).get("coverage_gap_likely")
    claim = (result.slots or {}).get("claim")
    lines.append(f"coverage_gap_likely={gap}")
    if claim:
        lines.append(f"claim={claim}")
    return "\n".join(lines)


def log_understanding_flow(
    result: UnderstandingResult,
    *,
    title: str = "",
    level: int = logging.INFO,
    file: Optional[TextIO] = None,
) -> str:
    text = format_understanding_flow(result, title=title)
    logger.log(level, text)
    if file is not None:
        file.write(text + "\n")
        file.flush()
    return text


def understanding_to_debug_dict(result: UnderstandingResult) -> Dict[str, Any]:
    return {
        "raw_question": result.raw_question,
        "intent": result.intent,
        "slots": result.slots,
        "coverage_gap_likely": (result.slots or {}).get("coverage_gap_likely"),
        "trace": result.trace,
        "sub_queries": [
            {
                "qid": sq.qid,
                "text": sq.text,
                "angle": sq.angle,
                "mode": getattr(sq, "mode", "lexical"),
                "modifiers": list(getattr(sq, "modifiers", []) or []),
                "filters": sq.filters,
                "channel": sq.channel,
                "angle_source": getattr(sq, "angle_source", ""),
            }
            for sq in result.sub_queries
        ],
        "relevance_criteria": result.relevance_criteria,
    }
