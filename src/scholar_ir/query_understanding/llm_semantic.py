"""Optional semantic reformulation when lexical coverage is likely insufficient.

Only runs when coverage_gap_likely — not to pad to N queries.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from scholar_ir.query_understanding.llm_common import call_llm
from scholar_ir.query_understanding.slots import skeleton_to_text
from scholar_ir.vendor_spar.utils import fetch_string

SYSTEM_PROMPT = (
    "You are an expert academic search strategist. "
    "Output only valid JSON. Your goal: translate a user's natural-language "
    "information need into search concepts that actually appear in paper titles "
    "and abstracts, not stylistic paraphrases."
)

PROMPT = """The user's question may be phrased abstractly, but papers are indexed
by their concrete methods, tasks, models, and applications.

User question:
{question}

Current core retrieval string (preserve the same information need):
{core_text}

Claim / proposition (if any — reformulate this conceptually, do NOT synonym-swap):
{claim}

Intent: {intent}

Generate up to {n} ALTERNATIVE search queries that express the SAME underlying
information need using CONCRETE academic concepts likely to appear in paper
metadata.

Guidelines for each alternative query:
1. Map abstract phrases to concrete methods/models.
   - "hybrid architectures" → "autoencoder GAN", "VAE CNN", "graph attention network",
     "deep unfolding network", "model-based deep learning"
   - "reconstruction-based techniques" → "anomaly detection", "inverse problem",
     "compressive sensing reconstruction", "image reconstruction"
2. Keep the same task, method, or comparison intent; do NOT drift to unrelated topics.
3. Prefer established keyword-style academic English (not prose).
4. Do NOT merely swap synonyms (e.g. "methods" vs "approaches") — that is handled elsewhere.
5. Do NOT add "survey" or "review" unless the user explicitly asks for surveys.
6. If you cannot produce a genuinely different conceptual phrasing, return fewer
   queries or an empty list — NEVER pad.

Return ONLY JSON:
{{
  "queries": [
    {{"text": "<search string>", "reason": "<why this helps retrieval>"}}
  ]
}}
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


def try_llm_semantic_reformulate(
    question: str,
    intent: str,
    slots: Dict[str, Any],
    *,
    n: int = 2,
    existing_texts: Optional[List[str]] = None,
) -> List[str]:
    """Return 0..n conceptual search strings, or [] if LLM unavailable."""
    if n <= 0:
        return []
    sk = slots.get("query_skeleton")
    core = ""
    if isinstance(sk, dict):
        core = skeleton_to_text(sk)
    core = core or (slots.get("topic") or question or "").strip()
    claim = str(slots.get("claim") or "").strip() or "(none)"
    prompt = PROMPT.format(
        question=question.strip(),
        core_text=core,
        claim=claim,
        intent=intent,
        n=n,
    )
    text = call_llm(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    if not text:
        return []
    data = _parse_json(text)
    if not data:
        return []

    existing = {_norm(t) for t in (existing_texts or []) if t}
    existing.add(_norm(core))
    out: List[str] = []
    raw_list = data.get("queries") or data.get("sub_queries") or []
    if not isinstance(raw_list, list):
        return []
    for item in raw_list:
        if isinstance(item, str):
            q = item.strip()
        elif isinstance(item, dict):
            q = str(item.get("text") or "").strip()
        else:
            continue
        q = re.sub(r"\s+", " ", q).strip(" ,")
        if not q or _norm(q) in existing:
            continue
        # Reject near-copies of core (token Jaccard high)
        if _too_similar(core, q):
            continue
        existing.add(_norm(q))
        out.append(q)
        if len(out) >= n:
            break
    return out


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _too_similar(a: str, b: str, thresh: float = 0.75) -> bool:
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / max(len(ta), len(tb)) >= thresh
