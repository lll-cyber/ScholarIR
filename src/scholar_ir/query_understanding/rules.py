"""Rule / heuristic extractors — work without LLM.

intent → rewrite templates (recall)
slots  → year / negation / rough topic (filter)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from scholar_ir.query_understanding.slots import empty_slots, empty_term, ensure_topic_term

_YEAR = re.compile(
    r"(?:since|after|from|post|starting\s+(?:in|from)?|beginning\s+(?:in|from)?)\s*"
    r"(20\d{2}|19\d{2})"
    r"|"
    r"(?:before|until|upto|up\s+to|prior\s+to)\s*(20\d{2}|19\d{2})"
    r"|"
    r"(?:between|from)\s*(20\d{2}|19\d{2})\s*(?:and|to|-|–)\s*(20\d{2}|19\d{2})"
    r"|"
    r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b"
    r"|"
    r"(?:in|year)\s+(20\d{2})\b"
    r"|"
    r"(?:last|past)\s+(\d+)\s+years?",
    re.I,
)

_NEG = re.compile(
    r"(?:not\s+(?:about|including|using)|excluding|except(?:\s+for)?|without|"
    r"不要|不含|排除|而非)\s+([^,.;?!]+)",
    re.I,
)

_SURVEY = re.compile(r"\b(surveys?|reviews?|overview|taxonomy|综述|回顾)\b", re.I)
_METHOD = re.compile(
    r"\b(methods?|approaches?|techniques?|algorithms?|frameworks?|architectures?|"
    r"方法|算法|框架|模型)\b",
    re.I,
)
_DATASET = re.compile(r"\b(datasets?|benchmarks?|corpus|数据[集集]|基准)\b", re.I)
_SPECIFIC = re.compile(
    r"\b(the\s+paper|the\s+study|this\s+work|bib\.|arxiv\.org|"
    r"titled|named)\b",
    re.I,
)
_RELATED = re.compile(
    r"\b(related\s+work|prior\s+work|previous\s+work|相关工作|related\s+to)\b",
    re.I,
)

_STOP_PREFIX = re.compile(
    r"^(?:could you|can you|please|what are|which|give me|list|find|tell me|"
    r"i need|looking for|are there any|provide(?: me)?)\s+",
    re.I,
)
_STOP_FILLER = re.compile(
    r"\b(?:some|any|the|papers?|studies|works?|research|about|on|regarding|"
    r"related to|that|which|me|with|for)\b",
    re.I,
)


def infer_intent(question: str) -> str:
    q = question.strip()
    if _SPECIFIC.search(q):
        return "specific"
    if _SURVEY.search(q):
        return "survey"
    if _RELATED.search(q):
        return "related"
    if _DATASET.search(q) and not _METHOD.search(q):
        return "dataset"
    if _METHOD.search(q):
        return "method"
    if len(q.split()) <= 8:
        return "broad"
    return "broad"


def extract_years(question: str, current_year: int = 2026) -> Tuple[Optional[int], Optional[int]]:
    q = question
    # between A and B
    m = re.search(
        r"(?:between|from)\s*(20\d{2}|19\d{2})\s*(?:and|to|-|–)\s*(20\d{2}|19\d{2})",
        q,
        re.I,
    )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)
    m = re.search(r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)

    year_from = year_to = None
    for m in re.finditer(
        r"(?:since|after|from|post)\s*(20\d{2}|19\d{2})", q, re.I
    ):
        year_from = int(m.group(1))
    for m in re.finditer(
        r"(?:before|until|prior\s+to)\s*(20\d{2}|19\d{2})", q, re.I
    ):
        year_to = int(m.group(1)) - 1
    m = re.search(r"(?:last|past)\s+(\d+)\s+years?", q, re.I)
    if m:
        n = int(m.group(1))
        year_from = current_year - n
    m = re.search(r"(?:in|year)\s+(20\d{2})\b", q, re.I)
    if m and year_from is None and year_to is None:
        y = int(m.group(1))
        return y, y
    return year_from, year_to


def extract_negation(question: str) -> Optional[List[str]]:
    found = []
    for m in _NEG.finditer(question):
        phrase = m.group(1).strip(" .;:")
        if phrase and len(phrase) < 80:
            found.append(phrase)
    return found or None


def rough_topic(question: str) -> str:
    """Strip question wrappers → rough topic phrase for rewrite seed."""
    q = question.strip()
    q = _STOP_PREFIX.sub("", q)
    q = re.sub(r"[?？]+$", "", q).strip()
    # drop year clauses lightly
    q = re.sub(
        r"(?:since|after|before|from|between|last|past)\s+[^,.]*",
        " ",
        q,
        flags=re.I,
    )
    # keep content words: remove only leading fillers repeatedly
    tokens = [t for t in re.split(r"\s+", q) if t]
    # if still long, drop pure filler tokens but keep technical terms
    kept = []
    for t in tokens:
        if _STOP_FILLER.fullmatch(t) and len(kept) == 0:
            continue
        if _STOP_FILLER.fullmatch(t) and len(kept) > 0 and len(kept) < 2:
            continue
        kept.append(t)
    topic = " ".join(kept).strip(" ,.-")
    return topic or question.strip()


def extract_slots_heuristic(question: str) -> Tuple[str, Dict[str, Any]]:
    intent = infer_intent(question)
    slots = empty_slots()
    slots["topic"] = rough_topic(question)
    yf, yt = extract_years(question)
    slots["year_from"] = yf
    slots["year_to"] = yt
    slots["negation"] = extract_negation(question)
    # light method hint if intent is method and phrase exists
    m = re.search(
        r"(?:using|via|based on|with)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})",
        question,
        re.I,
    )
    if m:
        slots["method"] = m.group(1).strip()
    topic = (slots.get("topic") or "").strip()
    terms = []
    if topic:
        terms.append(empty_term(topic, "topic"))
    if slots.get("method"):
        method = str(slots["method"]).strip()
        if method.lower() not in topic.lower():
            terms.append(empty_term(method, "method"))
    slots["terms"] = terms or None
    return intent, ensure_topic_term(slots)


def rewrite_by_intent(
    intent: str,
    topic: str,
    question: str,
    max_n: int = 3,
    add_survey_modifier: bool = False,
) -> List[str]:
    """Intent-conditioned query rewrites for recall. Dedup + cap."""
    topic = (topic or "").strip() or question.strip()
    q_clean = re.sub(r"[?？]+$", "", question.strip())

    variants: List[str] = []

    def add(s: str) -> None:
        s = re.sub(r"\s+", " ", s).strip(" ,")
        if s and s.lower() not in {v.lower() for v in variants}:
            variants.append(s)

    add(topic)
    if intent == "survey":
        if add_survey_modifier:
            add(f"{topic} survey")
            add(f"{topic} literature review")
        else:
            add(topic)
    elif intent == "method":
        add(f"{topic} method")
        add(f"{topic} approach")
    elif intent == "dataset":
        add(f"{topic} dataset")
        add(f"{topic} benchmark")
    elif intent == "specific":
        add(q_clean)
        add(topic)
    elif intent == "related":
        add(f"{topic} related work")
        add(f"{topic}")
    else:  # broad
        add(f"{topic}")
        if add_survey_modifier:
            add(f"{topic} survey")

    # always keep a close-to-original as last resort diversity
    add(q_clean)
    return variants[:max_n]
