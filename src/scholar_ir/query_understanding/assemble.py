"""Assemble SubQuery list from ExpansionTask[] + Slot Usage plan."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scholar_ir.types import SubQuery
from scholar_ir.query_understanding.rules import rewrite_by_intent
from scholar_ir.query_understanding.slot_usage import RetrievalPlan


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip(" ,")


# Question-prefix patterns stripped before submitting to search APIs.
# Applied sequentially (not OR-ed) to avoid ^-anchor problems with re.sub.
_PREFIX_PATTERNS: List[str] = [
    # Polite / imperative openers
    r"^(?:could you|can you|please|may i (?:have|ask)|would you)\s+",
    # "I want / I need / I am looking for"
    r"^(?:i (?:want|need|am looking for|i'm looking for|i'd like))\s+"
    r"(?:to\s+)?(?:find|search\s+for|see|know\s+about|have|papers about|papers on|papers regarding)\s+",
    # "Looking for / searching for"
    r"^(?:looking for|searching for)\s+",
    # "What/which/who question openers
    r"^(?:what are|what is|which (?:is|are)|who is|who are)\s+(?:the\s+)?",
    # "Give me / find me / show me / list me" (with optional "papers about X")
    r"^(?:give me|find me|show me|list(?: me)?|provide me with|provide)\s+"
    r"(?:me\s*)?(?:some\s+)?(?:papers? about |papers? on |papers? regarding |research on |studies on )?",
    # "find / get / provide papers on/about X" (without "me")
    r"^(?:find|get|provide|search for)\s+papers?\s+(?:about|on|regarding) ",
    # "list papers about X" (without "me")
    r"^(?:list)\s+(?:papers? about |papers? on |papers? regarding )",
    # "Are there / is there / do you know / do you have"
    r"^(?:are there|is there|do you know|do you have)\s+",
    # "Papers / articles / studies about X" — strip whole prefix
    r"^(?:papers?|articles?|studies?|publications?|works?)\s+"
    r"(?:about|on|regarding|related to|using|with)\s+",
]

_TRAILING_Q = re.compile(r"[?？!！.]+$")
_LEADING_ARTICLE = re.compile(
    r"^(?:papers?|articles?|studies?|publications?|works?)\s+(?:about|on|regarding|related to)\s+",
    re.I,
)
_DEFAULT_MAX_TOKENS = 24


def rewrite_query(
    text: str,
    *,
    intent: str = "",
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    min_tokens: int = 2,
) -> Optional[str]:
    """Standalone query rewrite — strip question form, normalize for search.

    Pipeline:
      1. Collapse whitespace.
      2. Apply each prefix pattern sequentially (strip "please list papers about X").
      3. Strip "papers about X" leading prefix.
      4. Strip trailing punctuation.
      5. Truncate to max_tokens (keep head; search-relevant terms are at the front).
      6. Drop too-short results.

    Returns cleaned text, or None if the rewrite is unusable.
    """
    if not text:
        return None

    t = " ".join(text.split()).strip()
    if not t:
        return None

    # Apply each prefix pattern sequentially to avoid ^-anchor problem.
    for pat in _PREFIX_PATTERNS:
        t = re.sub(pat, "", t, flags=re.I).strip()
    t = _LEADING_ARTICLE.sub("", t).strip()
    t = _TRAILING_Q.sub("", t).strip()
    t = " ".join(t.split())

    # Truncate by tokens (keep head; search-relevant terms are at the front).
    toks = t.split()
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
        t = " ".join(toks)
        # After head-truncation, drop leading prepositions / articles.
        t = re.sub(
            r"^(?:with|for|of|in|on|at|by|from|to|via|using|a|an|the)\s+",
            "",
            t,
            flags=re.I,
        ).strip()

    if len(t.split()) < min_tokens:
        return None

    # Lowercase uniformly; search APIs treat casing uniformly.
    t = t.lower()
    return t or None


def _rewrite_with_intent_fallback(
    text: str,
    intent: str,
    topic: str,
    question: str,
    add_survey_modifier: bool,
) -> str:
    """If standalone rewrite yields a too-broad term, fall back to intent templates."""
    cleaned = rewrite_query(text, intent=intent) or ""
    # If too short (1-2 words) and intent has a template, use it.
    if cleaned and len(cleaned.split()) <= 2 and topic and intent:
        templates = rewrite_by_intent(
            intent, topic, question, max_n=1,
            add_survey_modifier=add_survey_modifier,
        )
        if templates:
            return templates[0]
    return cleaned or text


def assemble_from_tasks(
    question: str,
    intent: str,
    slots: Dict[str, Any],
    plan: RetrievalPlan,
    *,
    max_n: int = 5,
    add_survey_modifier: bool = False,
    rewrite_subqueries: bool = True,
) -> List[SubQuery]:
    """Primary path: expand ExpansionTask → SubQuery (unified for all intents).

    When `rewrite_subqueries=True` (default), each rendered text is passed
    through `rewrite_query` to strip question form and normalize for search APIs.
    """
    filters = dict(plan.api_filters)
    topic = (slots.get("topic") or "").strip()
    tasks = list(plan.tasks or [])[:max_n]

    if not tasks:
        return _fallback_template(
            question, intent, slots, plan, max_n=max_n,
            add_survey_modifier=add_survey_modifier,
            rewrite_subqueries=rewrite_subqueries,
        )

    out: List[SubQuery] = []
    seen: set[str] = set()
    for i, task in enumerate(tasks):
        text = _clean(task.text_seed)
        if not text:
            continue
        if rewrite_subqueries:
            text = _rewrite_with_intent_fallback(
                text, intent, topic, question, add_survey_modifier,
            )
        text = _clean(text)
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
        rewrite_subqueries=rewrite_subqueries,
    )


def _fallback_template(
    question: str,
    intent: str,
    slots: Dict[str, Any],
    plan: RetrievalPlan,
    *,
    max_n: int,
    add_survey_modifier: bool = False,
    rewrite_subqueries: bool = True,
) -> List[SubQuery]:
    topic = (slots.get("topic") or question).strip()
    texts = rewrite_by_intent(
        intent, topic, question, max_n=max_n,
        add_survey_modifier=add_survey_modifier,
    )
    if rewrite_subqueries:
        texts = [
            _rewrite_with_intent_fallback(
                t, intent, topic, question, add_survey_modifier,
            )
            for t in texts
        ]
    filters = dict(plan.api_filters)
    out: List[SubQuery] = []
    seen: set[str] = set()
    for i, t in enumerate(texts):
        cleaned = _clean(t)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            SubQuery(
                qid=f"q{i}",
                text=cleaned,
                channel="keyword",
                filters=dict(filters),
                angle="core",
                mode="lexical",
                angle_source="template",
                modifiers=["survey"] if add_survey_modifier and intent == "survey" and i == 0 else [],
            )
        )
        if len(out) >= max_n:
            break
    return out


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
