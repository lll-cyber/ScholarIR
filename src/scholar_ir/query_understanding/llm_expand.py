"""Round 2 LLM: mutually exclusive recall sub-queries after Usage Decision."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from scholar_ir.query_understanding.llm_common import call_llm
from scholar_ir.query_understanding.slots import INTENTS
from scholar_ir.vendor_spar.utils import fetch_string

SYSTEM_PROMPT = (
    "You are an academic search query expander for paper retrieval. "
    "Output only valid JSON. Use English keyword-style strings."
)

EXPAND_PROMPT = """Generate {n} mutually exclusive SHORT keyword search strings for recall.

Context (already extracted — do NOT re-extract slots):
- intent: {intent}
- topic: {topic}
- method (text only, not a hard filter): {method}
- recall_hints: {recall_hints}

Rules:
- Each sub-query is 3-12 words, keyword-style (not full sentences)
- Angles must be distinct (e.g. survey, method, application, evaluation, subtopic)
- For intent=survey: at least one query should target survey/review literature
- For intent=method: cover methods, applications, and evaluation/benchmark angles
- Do NOT encode negation or exclusions in queries
- Year/time filters are applied separately — omit years unless essential to meaning
- Prefer recall breadth over precision

Return ONLY valid JSON:
{{
  "sub_queries": [
    {{"text": "<search string>", "angle": "<short angle label>"}},
    ...
  ]
}}

User question:
{question}
"""


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = fetch_string(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _normalize_expanded(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw = data.get("sub_queries") or []
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            text, angle = item.strip(), "core"
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            angle = str(item.get("angle") or "core").strip() or "core"
        else:
            continue
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((text, angle))
    return out


def try_llm_expand(
    question: str,
    intent: str,
    slots: Dict[str, Any],
    recall_hints: Dict[str, Any],
    *,
    n: int = 5,
) -> Optional[List[Tuple[str, str]]]:
    """Return list of (text, angle) or None if LLM unavailable / parse failed."""
    if intent not in INTENTS:
        intent = "broad"
    topic = (slots.get("topic") or question).strip()
    method = slots.get("method") or "null"
    hints_json = json.dumps(recall_hints or {}, ensure_ascii=False)
    prompt = EXPAND_PROMPT.format(
        n=n,
        intent=intent,
        topic=topic,
        method=method,
        recall_hints=hints_json,
        question=question.strip(),
    )
    text = call_llm(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=int(os.getenv("SCHOLAR_IR_EXPAND_MAX_TOKENS", "1024")),
    )
    if not text:
        return None
    data = _parse_json(text)
    if not data:
        return None
    expanded = _normalize_expanded(data)
    return expanded or None
