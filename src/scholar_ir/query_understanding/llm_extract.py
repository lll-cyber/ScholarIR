"""Round 1 LLM extraction: tightened schema → internal slots.

First-class fields: intent, filters, term_groups, claim, coverage_gap_likely,
query_skeleton. role is optional. Normalized into legacy terms[] for Slot Usage.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from scholar_ir.query_understanding.llm_common import call_llm
from scholar_ir.query_understanding.slots import (
    INTENTS,
    empty_slots,
    ensure_topic_term,
    migrate_legacy_hints_to_terms,
    normalize_query_skeleton,
    normalize_round1_output,
    normalize_terms,
    _apply_true_filters,
)
from scholar_ir.vendor_spar.utils import fetch_string

SYSTEM_PROMPT = (
    "You are an academic search query analyzer for a paper retrieval system. "
    "Output only valid JSON. Prefer compact English retrieval phrases; "
    "preserve comparisons/causality/conditions in core_text and claim. "
    "Variants/abbrevs exist ONLY to improve search recall — never to rewrite for style "
    "or to fill a quota. Do NOT put claim paraphrases into term variants."
)

PROMPT = """Extract a LIGHT Round-1 schema for academic paper search. Rules:

## First-class fields (use these — do NOT invent API topic/method parameters)
- intent: ONE primary routing label only
- filters: ONLY true API/Judge constraints (year/venue/authors/negation). null if unknown
- term_groups: retrieval material. Within a group, variants ≈ soft OR; across groups,
  required=true ≈ AND for Filter. Prefer ZERO weak variants over padding.
- claim: fill when the query asserts a comparison/causality/proposition; NOT a term.
  Leave null for pure terminology queries. Never dump claim paraphrases into variants.
- coverage_gap_likely: true when literal/core phrasing is UNLIKELY in papers and a
  different CONCEPTUAL wording is needed for recall
  (e.g. "smaller dataset better than larger" → papers say "data quality vs quantity").
  false when standard terms already retrieve well.
- query_skeleton: core_text preserves relations; parts MUST be non-overlapping exact
  substrings of core_text (longest span wins if nested).
- dataset / domain: application context — data resources, benchmarks, corpora,
  or disciplinary area. Fill when the question implies WHERE or IN WHICH CONTEXT
  a topic/method is discussed (e.g. "in medical imaging", "on ImageNet",
  "for NLP"). Leave null when not specified. Both are optional hints for
  keyword coverage scoring — not API filter constraints.
- topic / method: the query's primary subject and approach (mirrors term_groups
  role tags for redundancy). topic = the main concept/area; method = the
  algorithm / technique / framework named. Leave null when not specified.
  These are optional hints for keyword coverage scoring.

## term_groups
- canonical: preferred indexing phrase
- variants: ≤1–2 high-confidence synonyms / alternate forms (may omit canonical)
- abbrev: acronym if papers use it, else null
- required / replaceable: Filter hard-constraint / lexical span-swap eligibility
- role: OPTIONAL hint only (topic|method|entity|null). APIs do not consume role.

## variants quality (HIGH CONFIDENCE ONLY)
ALLOWED:
  ✅ video generation ↔ video synthesis
  ✅ LLM ↔ large language model
  ✅ pre-training ↔ pretraining
FORBIDDEN:
  ❌ X → X task / X model / X method / X approach
  ❌ morphology/POS breaks (calibration → calibrating)
  ❌ claim-level paraphrases (put those in claim / leave for conceptual stage)

## Do NOT output final sub_queries

Allowed intent (pick one): {intents}

Return ONLY valid JSON:
{{
  "intent": "<one of allowed>",
  "filters": {{
    "year_from": <int or null>,
    "year_to": <int or null>,
    "venue": "<venue or null>",
    "authors": ["..."] or null,
    "negation": ["..."] or null
  }},
  "coverage_gap_likely": true/false,
  "claim": "<proposition text or null>",
  "topic": "<primary subject or null>",
  "method": "<primary approach/framework or null>",
  "dataset": "<data resource / benchmark / corpus name or null>",
  "domain": "<disciplinary or application area or null>",
  "term_groups": [
    {{
      "canonical": "<concept>",
      "variants": ["<≤2 high-conf forms>"],
      "abbrev": "<acronym or null>",
      "required": true,
      "replaceable": true,
      "role": "topic|method|entity|null"
    }}
  ],
  "query_skeleton": {{
    "core_text": "<relation-preserving retrieval string>",
    "parts": [
      {{
        "id": "p0",
        "text": "<exact substring of core_text>",
        "required": true,
        "replaceable": true,
        "variants": ["<span>", "<≤1–2 high-conf synonym/abbrev>"]
      }}
    ]
  }}
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


def _as_bool(val: Any) -> Optional[bool]:
    if val is None or val == "" or val == "null":
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return bool(val)


def _normalize_slots_legacy(raw_slots: Dict[str, Any]) -> Dict[str, Any]:
    """Compat: old {slots: {topic, method, terms, ...}} LLM responses."""
    slots = empty_slots()
    slots = _apply_true_filters(slots, raw_slots)
    for k in ("topic", "method"):
        if k in raw_slots:
            v = raw_slots[k]
            slots[k] = None if v == "" or v == "null" else v
    slots["coverage_gap_likely"] = _as_bool(raw_slots.get("coverage_gap_likely"))
    claim = raw_slots.get("claim")
    if claim is not None and claim != "" and claim != "null":
        slots["claim"] = str(claim).strip() or None
    slots["terms"] = normalize_terms(raw_slots.get("terms"))
    slots["query_skeleton"] = normalize_query_skeleton(raw_slots.get("query_skeleton"))
    return ensure_topic_term(slots)


def _looks_like_new_schema(data: Dict[str, Any]) -> bool:
    if isinstance(data.get("term_groups"), list):
        return True
    if isinstance(data.get("filters"), dict):
        return True
    if "claim" in data and "slots" not in data:
        return True
    return False


def _normalize(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    intent = str(data.get("intent") or "broad").lower().strip()
    if intent not in INTENTS:
        intent = "broad"

    if _looks_like_new_schema(data):
        slots = normalize_round1_output(data)
    else:
        raw_slots = data.get("slots") or {}
        if not isinstance(raw_slots, dict):
            raw_slots = {}
        slots = _normalize_slots_legacy(raw_slots)
        # Top-level overrides if model mixed schemas
        if data.get("claim") and not slots.get("claim"):
            c = data.get("claim")
            if c not in ("", "null", None):
                slots["claim"] = str(c).strip() or None
        if data.get("coverage_gap_likely") is not None and slots.get("coverage_gap_likely") is None:
            slots["coverage_gap_likely"] = _as_bool(data.get("coverage_gap_likely"))

    if data.get("recall_hints") or data.get("coverage"):
        slots = migrate_legacy_hints_to_terms(slots, data.get("recall_hints"))
        cov = data.get("coverage") or {}
        if isinstance(cov, dict):
            gap = cov.get("coverage_gap_likely")
            if gap is not None and slots.get("coverage_gap_likely") is None:
                slots["coverage_gap_likely"] = bool(gap)
            if slots.get("terms"):
                instances = cov.get("instances")
                terms = list(slots["terms"])
                for i, t in enumerate(terms):
                    if t.get("role") == "topic":
                        t = dict(t)
                        if instances and not t.get("instances"):
                            if isinstance(instances, list):
                                t["instances"] = [
                                    str(x).strip() for x in instances if str(x).strip()
                                ]
                        if gap is not None and t.get("coverage_gap_likely") is None:
                            t["coverage_gap_likely"] = bool(gap)
                        terms[i] = t
                        break
                slots["terms"] = terms

    return intent, ensure_topic_term(slots)


def try_llm_extract(question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (intent, slots) or None. slots includes terms[] (+ term_groups/claim)."""
    prompt = PROMPT.format(intents=", ".join(INTENTS), question=question.strip())
    text = call_llm(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    if not text:
        return None
    data = _parse_json(text)
    if data:
        return _normalize(data)
    return None
