"""Lightweight slot schema for Query Understanding.

Round1 (tightened): filters + term_groups + claim + coverage_gap + query_skeleton.
Internal slots keep terms[] (mapped from term_groups) for Slot Usage / criteria.
query_skeleton = core_text (relation-preserving) + replaceable parts (spans).
Lexical expand = render core_text, swap at most ONE replaceable span.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

INTENTS = (
    "survey",
    "method",
    "dataset",
    "specific",
    "broad",
    "related",
)

TERM_ROLES = ("topic", "method", "entity", "other")

LIGHT_SLOT_KEYS = (
    "topic",
    "method",
    "year_from",
    "year_to",
    "venue",
    "authors",
    "negation",
    "terms",
    "term_groups",
    "query_skeleton",
    "coverage_gap_likely",
    "claim",
)

# True API / Judge filters (not topic/method semantic roles)
TRUE_FILTER_KEYS = ("year_from", "year_to", "venue", "authors", "negation")

# raw = original NL; metadata = title/author; conceptual = semantic reformulation
ANGLE_SET = ("core", "synonym", "abbrev", "entity", "metadata", "raw", "conceptual")
MODE_SET = ("lexical", "decomposition", "semantic")


def _normalize_role(role: Any) -> Optional[str]:
    """Optional role hint; None when absent. Never invent."""
    if role is None or role == "" or role == "null":
        return None
    r = str(role).lower().strip()
    if r in TERM_ROLES:
        return r
    return "other"


def empty_term(
    text: str = "",
    role: Optional[str] = "topic",
    *,
    abbrev: Optional[str] = None,
    synonyms: Optional[List[str]] = None,
    instances: Optional[List[str]] = None,
    coverage_gap_likely: Optional[bool] = None,
    required: bool = True,
    replaceable: Optional[bool] = None,
) -> Dict[str, Any]:
    has_variants = bool(abbrev) or bool(synonyms) or bool(instances)
    if replaceable is None:
        replaceable = has_variants
    norm_role = _normalize_role(role) if role is not None else None
    # Compat: callers that pass default "topic" keep topic; explicit None stays None
    if role is None:
        stored_role: Optional[str] = None
    else:
        stored_role = norm_role if norm_role is not None else "other"
    return {
        "text": text,
        "role": stored_role,
        "abbrev": abbrev,
        "synonyms": synonyms,
        "instances": instances,
        "coverage_gap_likely": coverage_gap_likely,
        "required": bool(required),
        "replaceable": bool(replaceable),
    }


def empty_skeleton_part(
    part_id: str,
    text: str,
    *,
    required: bool = True,
    replaceable: bool = False,
    variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """One replaceable span inside core_text (legacy name: part)."""
    return {
        "id": part_id,
        "text": text,
        "required": bool(required),
        "replaceable": bool(replaceable),
        "variants": variants,
    }


def empty_slots() -> Dict[str, Any]:
    return {k: None for k in LIGHT_SLOT_KEYS}


def _as_str_list(val: Any) -> Optional[List[str]]:
    if val is None or val == "" or val == "null":
        return None
    if isinstance(val, str):
        val = [val]
    if not isinstance(val, list):
        return None
    out = [str(x).strip() for x in val if str(x).strip()]
    return out or None


def _dedupe_strs(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def normalize_term(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    # term_groups use canonical; legacy terms use text
    text = str(raw.get("text") or raw.get("canonical") or "").strip()
    if not text:
        return None
    if "role" not in raw or raw.get("role") is None or raw.get("role") == "" or raw.get("role") == "null":
        role: Optional[str] = None
    else:
        role = _normalize_role(raw.get("role"))
    abbrev = raw.get("abbrev")
    if abbrev is None or abbrev == "" or abbrev == "null":
        abbrev = None
    else:
        abbrev = str(abbrev).strip() or None
    gap = raw.get("coverage_gap_likely")
    if gap is None or gap == "" or gap == "null":
        gap_bool: Optional[bool] = None
    else:
        gap_bool = bool(gap)
    required = raw.get("required")
    if required is None or required == "" or required == "null":
        required = True
    replaceable = raw.get("replaceable")
    if replaceable is None or replaceable == "" or replaceable == "null":
        replaceable = None
    else:
        replaceable = bool(replaceable)
    # variants (new) fold into synonyms (legacy), drop self-duplicates of canonical
    synonyms = _as_str_list(raw.get("synonyms")) or []
    variants = _as_str_list(raw.get("variants")) or []
    for v in variants:
        if v.lower() == text.lower():
            continue
        if v.lower() not in {s.lower() for s in synonyms}:
            synonyms.append(v)
    return empty_term(
        text,
        role,
        abbrev=abbrev,
        synonyms=synonyms or None,
        instances=_as_str_list(raw.get("instances")),
        coverage_gap_likely=gap_bool,
        required=bool(required),
        replaceable=replaceable,
    )


def normalize_terms(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if raw is None or raw == "" or raw == "null":
        return None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        t = normalize_term(item)
        if not t:
            continue
        key = t["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out or None


def normalize_term_groups(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize Round1 term_groups[]; preserve structure for trace / future boolean pack."""
    if raw is None or raw == "" or raw == "null":
        return None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or item.get("text") or "").strip()
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        variants = _as_str_list(item.get("variants")) or []
        variants = _dedupe_strs(
            [v for v in variants if v.lower() != key]
        )
        abbrev = item.get("abbrev")
        if abbrev is None or abbrev == "" or abbrev == "null":
            abbrev_s: Optional[str] = None
        else:
            abbrev_s = str(abbrev).strip() or None
        required = item.get("required")
        if required is None or required == "" or required == "null":
            required = True
        replaceable = item.get("replaceable")
        if replaceable is None or replaceable == "" or replaceable == "null":
            replaceable = bool(variants) or bool(abbrev_s)
        else:
            replaceable = bool(replaceable)
        out.append(
            {
                "canonical": canonical,
                "variants": variants or None,
                "abbrev": abbrev_s,
                "required": bool(required),
                "replaceable": bool(replaceable),
                "role": _normalize_role(item.get("role")),
            }
        )
    return out or None


def term_groups_to_terms(term_groups: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Map term_groups → legacy terms[] (text/synonyms) for Slot Usage / criteria."""
    if not term_groups:
        return None
    terms: List[Dict[str, Any]] = []
    for g in term_groups:
        if not isinstance(g, dict):
            continue
        t = normalize_term(
            {
                "canonical": g.get("canonical"),
                "variants": g.get("variants"),
                "abbrev": g.get("abbrev"),
                "required": g.get("required", True),
                "replaceable": g.get("replaceable"),
                "role": g.get("role"),
            }
        )
        if t:
            terms.append(t)
    return terms or None


def _apply_true_filters(slots: Dict[str, Any], filters: Any) -> Dict[str, Any]:
    """Copy year/venue/authors/negation from filters dict into slots."""
    slots = dict(slots)
    if not isinstance(filters, dict):
        return slots
    for k in ("venue", "authors", "negation"):
        if k in filters:
            v = filters[k]
            slots[k] = None if v == "" or v == "null" else v
    for yk in ("year_from", "year_to"):
        if yk not in filters:
            continue
        v = filters.get(yk)
        if v is None or v == "" or v == "null":
            slots[yk] = None
        else:
            try:
                slots[yk] = int(v)
            except (TypeError, ValueError):
                slots[yk] = None
    if slots.get("authors") == "null":
        slots["authors"] = None
    if slots.get("negation") == "null":
        slots["negation"] = None
    return slots


def _sync_compat_topic_method(slots: Dict[str, Any]) -> Dict[str, Any]:
    """Derive legacy topic/method from term_groups/terms for downstream compat."""
    slots = dict(slots)
    terms = list(slots.get("terms") or [])
    if not (slots.get("method") or "").strip():
        for t in terms:
            if t.get("role") == "method" and (t.get("text") or "").strip():
                slots["method"] = str(t["text"]).strip()
                break
    if not (slots.get("topic") or "").strip():
        for t in terms:
            if t.get("role") == "topic" and (t.get("text") or "").strip():
                slots["topic"] = str(t["text"]).strip()
                break
        if not (slots.get("topic") or "").strip():
            for t in terms:
                if t.get("required", True) and (t.get("text") or "").strip():
                    slots["topic"] = str(t["text"]).strip()
                    break
    return slots


def normalize_round1_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tightened Round1 JSON → internal slots.

    First-class: filters, term_groups, claim, coverage_gap_likely, query_skeleton.
    role is optional hint. Maps term_groups → terms for existing pipeline.
    """
    slots = empty_slots()
    filters = data.get("filters")
    if not isinstance(filters, dict):
        # Flat true-filter fields at top level (tolerant)
        filters = {k: data.get(k) for k in TRUE_FILTER_KEYS if k in data}
    slots = _apply_true_filters(slots, filters)

    gap = data.get("coverage_gap_likely")
    if gap is None or gap == "" or gap == "null":
        slots["coverage_gap_likely"] = None
    else:
        slots["coverage_gap_likely"] = bool(gap)

    claim = data.get("claim")
    if claim is None or claim == "" or claim == "null":
        slots["claim"] = None
    else:
        slots["claim"] = str(claim).strip() or None

    term_groups = normalize_term_groups(data.get("term_groups"))
    slots["term_groups"] = term_groups
    terms = term_groups_to_terms(term_groups)
    if terms is None and data.get("terms") is not None:
        terms = normalize_terms(data.get("terms"))
    slots["terms"] = terms
    slots["query_skeleton"] = normalize_query_skeleton(data.get("query_skeleton"))
    slots = _sync_compat_topic_method(slots)
    return ensure_topic_term(slots)


def normalize_skeleton_part(raw: Any, idx: int = 0) -> Optional[Dict[str, Any]]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return empty_skeleton_part(f"t{idx}", text)
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    part_id = str(raw.get("id") or f"t{idx}").strip() or f"t{idx}"
    required = raw.get("required")
    if required is None or required == "" or required == "null":
        required = True
    replaceable = raw.get("replaceable")
    if replaceable is None or replaceable == "" or replaceable == "null":
        replaceable = False
    variants = _as_str_list(raw.get("variants"))
    return empty_skeleton_part(
        part_id,
        text,
        required=bool(required),
        replaceable=bool(replaceable),
        variants=variants,
    )


def _join_parts(parts: List[Dict[str, Any]]) -> str:
    return " ".join(
        str(p.get("text") or "").strip() for p in parts if str(p.get("text") or "").strip()
    )


def normalize_query_skeleton(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize to {core_text, parts}.

    - core_text: relation-preserving retrieval string
    - parts: replaceable spans that appear inside core_text
    Legacy parts-only → core_text = space-join(parts).
    Also accepts replace_spans as alias of parts.
    """
    if raw is None or raw == "" or raw == "null":
        return None
    if isinstance(raw, list):
        raw = {"parts": raw}
    if not isinstance(raw, dict):
        return None

    core_text = str(raw.get("core_text") or raw.get("core_claim") or "").strip() or None

    parts_raw = raw.get("parts")
    if parts_raw is None:
        parts_raw = raw.get("replace_spans")
    parts: List[Dict[str, Any]] = []
    if isinstance(parts_raw, list):
        for i, item in enumerate(parts_raw):
            p = normalize_skeleton_part(item, i)
            if p:
                parts.append(p)

    if not core_text and not parts:
        return None
    if not core_text and parts:
        core_text = _join_parts(parts)
    parts = dedupe_skeleton_parts(core_text or "", parts)
    if not core_text and not parts:
        return None
    if not core_text and parts:
        core_text = _join_parts(parts)
    return {"core_text": core_text, "parts": parts}


def _term_variants(term: Dict[str, Any]) -> List[str]:
    variants: List[str] = []
    text = str(term.get("text") or "").strip()
    if text:
        variants.append(text)
    abbrev = term.get("abbrev")
    if abbrev:
        variants.append(str(abbrev).strip())
    for s in term.get("synonyms") or []:
        if s:
            variants.append(str(s).strip())
    return _dedupe_strs([v for v in variants if v])


def _match_term_for_part(part: Dict[str, Any], terms: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ptext = str(part.get("text") or "").lower()
    for t in terms:
        ttext = str(t.get("text") or "").lower()
        abbrev = str(t.get("abbrev") or "").lower()
        if ptext == ttext or (abbrev and ptext == abbrev):
            return t
        for syn in t.get("synonyms") or []:
            if ptext == str(syn).lower():
                return t
    return None


def _span_first_range(core_text: str, span: str) -> Optional[tuple[int, int]]:
    """Return [start, end) of first case-insensitive occurrence of span in core_text."""
    if not core_text or not span:
        return None
    m = re.search(re.escape(span), core_text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.start(), m.end()


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def dedupe_skeleton_parts(
    core_text: str,
    parts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep parts that appear in core_text; drop duplicates / overlapping spans.

    Prefer longest span when ranges overlap. Exact duplicate text → keep first.
    """
    core_text = (core_text or "").strip()
    if not parts:
        return []

    candidates: List[tuple[int, Dict[str, Any], tuple[int, int]]] = []
    seen_text: set[str] = set()
    for idx, p in enumerate(parts):
        p = dict(p)
        text = str(p.get("text") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        if core_text:
            rng = _span_first_range(core_text, text)
            if rng is None:
                continue  # not a real substring of core_text
        else:
            rng = (0, 0)
        candidates.append((idx, p, rng))

    # Longest span first; stable by original order
    candidates.sort(key=lambda x: (-(x[2][1] - x[2][0]), x[0]))
    kept: List[tuple[int, Dict[str, Any], tuple[int, int]]] = []
    for item in candidates:
        _, _p, rng = item
        if core_text and any(_ranges_overlap(rng, k[2]) for k in kept):
            continue
        kept.append(item)

    # Restore original relative order
    kept.sort(key=lambda x: x[0])
    return [p for _, p, _ in kept]


def merge_terms_into_skeleton(
    skeleton: Dict[str, Any],
    terms: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Optionally enrich *existing* parts with matching term variants.

    terms ≠ parts: do NOT invent new auto* spans from terms.
    Overlapping / duplicate parts are dropped (longest wins).
    """
    terms = terms or []
    core_text = str(skeleton.get("core_text") or "").strip()
    parts: List[Dict[str, Any]] = []
    for p in skeleton.get("parts") or []:
        p = dict(p)
        term = _match_term_for_part(p, terms)
        variants = list(p.get("variants") or [])
        if term:
            # Soft enrich only when span already registered — never add new parts
            variants.extend(_term_variants(term))
            forms = _term_variants(term)
            if len(forms) > 1 or term.get("replaceable"):
                p["replaceable"] = True
            if term.get("required") is False:
                p["required"] = False
        variants = _dedupe_strs([v for v in variants if v])
        canon = str(p.get("text") or "").strip()
        if canon:
            from scholar_ir.query_understanding.variant_quality import (
                filter_retrieval_variants,
            )

            alts = filter_retrieval_variants(canon, variants, max_alts=2)
            variants = _dedupe_strs([canon] + alts)
        p["variants"] = variants or None
        if variants and len(variants) > 1:
            p["replaceable"] = True
        parts.append(p)

    parts = dedupe_skeleton_parts(core_text, parts)
    if not core_text and parts:
        core_text = _join_parts(parts)
    return {"core_text": core_text, "parts": parts}


def build_skeleton_from_terms(terms: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Fallback conjunctive skeleton when LLM omitted core_text."""
    if not terms:
        return None
    parts: List[Dict[str, Any]] = []
    for i, t in enumerate(terms):
        text = str(t.get("text") or "").strip()
        if not text:
            continue
        variants = _term_variants(t)
        replaceable = bool(t.get("replaceable")) or len(variants) > 1
        parts.append(
            empty_skeleton_part(
                f"t{i}",
                text,
                required=bool(t.get("required", True)),
                replaceable=replaceable,
                variants=variants if len(variants) > 1 else None,
            )
        )
    if not parts:
        return None
    return {"core_text": _join_parts(parts), "parts": parts}


def ensure_query_skeleton(slots: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure slots.query_skeleton exists; merge term variants; sync topic."""
    slots = dict(slots)
    terms = list(slots.get("terms") or [])
    skeleton = normalize_query_skeleton(slots.get("query_skeleton"))
    if skeleton is None:
        skeleton = build_skeleton_from_terms(terms)
    if skeleton is None:
        topic = (slots.get("topic") or "").strip()
        if topic:
            skeleton = {
                "core_text": topic,
                "parts": [empty_skeleton_part("t0", topic, required=True, replaceable=False)],
            }
    if skeleton is not None:
        skeleton = merge_terms_into_skeleton(skeleton, terms)
        slots["query_skeleton"] = skeleton
        joined = skeleton_to_text(skeleton)
        if joined and not (slots.get("topic") or "").strip():
            slots["topic"] = joined
    return slots


def _replace_once(text: str, old: str, new: str) -> str:
    """Replace first occurrence of old in text (exact, then case-insensitive)."""
    if not old or not text:
        return text
    idx = text.find(old)
    if idx >= 0:
        return text[:idx] + new + text[idx + len(old) :]
    m = re.search(re.escape(old), text, flags=re.IGNORECASE)
    if m:
        return text[: m.start()] + new + text[m.end() :]
    return text


def skeleton_to_text(
    skeleton: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, str]] = None,
    modifier: Optional[str] = None,
) -> str:
    """Render skeleton: prefer core_text + span swap; else join parts."""
    overrides = overrides or {}
    parts: List[Dict[str, Any]] = list(skeleton.get("parts") or [])
    core = str(skeleton.get("core_text") or "").strip()

    if core:
        text = core
        id_to_part = {str(p.get("id") or ""): p for p in parts}
        for pid, variant in overrides.items():
            part = id_to_part.get(pid)
            if not part:
                continue
            canon = str(part.get("text") or "").strip()
            if not canon or not variant:
                continue
            text = _replace_once(text, canon, variant)
    else:
        toks: List[str] = []
        for p in parts:
            pid = str(p.get("id") or "")
            if pid in overrides:
                tok = overrides[pid]
            else:
                tok = str(p.get("text") or "").strip()
            if tok:
                toks.append(tok)
        text = " ".join(toks)

    if modifier:
        m = modifier.strip()
        if m and m.lower() not in text.lower():
            text = f"{text} {m}".strip()
    return " ".join(text.split())


def ensure_topic_term(slots: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure terms contains a topic term; then ensure query_skeleton."""
    slots = dict(slots)
    terms = list(slots.get("terms") or [])
    topic = (slots.get("topic") or "").strip()
    has_topic_role = any(t.get("role") == "topic" for t in terms)
    if topic and not has_topic_role:
        if not terms:
            terms.insert(0, empty_term(topic, "topic"))
            slots["terms"] = terms
    elif not topic and has_topic_role:
        for t in terms:
            if t.get("role") == "topic":
                slots["topic"] = t.get("text")
                break
    if terms and slots.get("terms") is None:
        slots["terms"] = terms
    return ensure_query_skeleton(slots)


def migrate_legacy_hints_to_terms(
    slots: Dict[str, Any],
    recall_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fold old recall_hints / coverage into slots.terms (compat)."""
    slots = dict(slots)
    terms: List[Dict[str, Any]] = list(slots.get("terms") or [])
    hints = recall_hints or {}

    topic_idx = next((i for i, t in enumerate(terms) if t.get("role") == "topic"), None)
    if topic_idx is None and slots.get("topic"):
        terms.append(empty_term(str(slots["topic"]), "topic"))
        topic_idx = len(terms) - 1

    if topic_idx is not None:
        t = dict(terms[topic_idx])
        if hints.get("abbrev") and not t.get("abbrev"):
            t["abbrev"] = str(hints["abbrev"]).strip() or None
        syn = _as_str_list(hints.get("synonyms"))
        if syn:
            existing = list(t.get("synonyms") or [])
            for s in syn:
                if s.lower() not in {x.lower() for x in existing}:
                    existing.append(s)
            t["synonyms"] = existing or None
        terms[topic_idx] = empty_term(
            t["text"],
            t.get("role") or "topic",
            abbrev=t.get("abbrev"),
            synonyms=t.get("synonyms"),
            instances=t.get("instances"),
            coverage_gap_likely=t.get("coverage_gap_likely"),
        )

    for ent in _as_str_list(hints.get("entities")) or []:
        if ent.lower() not in {x.get("text", "").lower() for x in terms}:
            terms.append(empty_term(ent, "entity"))

    slots["terms"] = terms or None
    return ensure_topic_term(slots)


def slots_to_filters(slots: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if slots.get("year_from") is not None:
        filters["year_from"] = slots["year_from"]
    if slots.get("year_to") is not None:
        filters["year_to"] = slots["year_to"]
    if slots.get("venue"):
        filters["venue"] = slots["venue"]
    if slots.get("authors"):
        filters["authors"] = slots["authors"]
    if slots.get("method"):
        filters["method"] = slots["method"]
    return filters


def _criterion_description(c_type: str, text: str) -> str:
    templates = {
        "topic": "Paper is about: {text}",
        "method": "Involves method/technique: {text}",
        "entity": "Involves entity/concept: {text}",
        "claim": "Addresses claim/comparison: {text}",
        "negation": "Must NOT be primarily about: {text}",
    }
    return templates.get(c_type, "Relevant to: {text}").format(text=text)


def _default_criterion_weight(c_type: str) -> float:
    return {
        "topic": 0.4,
        "method": 0.35,
        "entity": 0.25,
        "claim": 0.2,
        "negation": 0.25,
    }.get(c_type, 0.2)


def _looks_like_claim(text: str) -> bool:
    low = f" {(text or '').lower()} "
    markers = (
        " better ",
        " than ",
        " versus ",
        " vs ",
        " outperform",
        " outperforms ",
        " less ",
        " more ",
        " can produce ",
    )
    return any(m in low for m in markers)


def _criterion_key(c_type: str, text: str) -> tuple[str, str]:
    return c_type, " ".join(text.lower().split())


def slots_to_criteria(intent: str, slots: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    """Derive relevance criteria from required semantic units (terms/claim), not intent."""
    criteria: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(
        c_type: str,
        text: str,
        *,
        required: bool = True,
        weight: Optional[float] = None,
    ) -> None:
        t = (text or "").strip()
        if not t:
            return
        key = _criterion_key(c_type, t)
        if key in seen:
            return
        seen.add(key)
        w = _default_criterion_weight(c_type) if weight is None else weight
        criteria.append(
            {
                "name": c_type,
                "type": c_type,
                "text": t,
                "required": bool(required),
                "description": _criterion_description(c_type, t),
                "weight": w,
            }
        )

    for raw in slots.get("terms") or []:
        if not isinstance(raw, dict):
            continue
        if not raw.get("required", True):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        role = raw.get("role")
        if role:
            role = str(role).lower().strip()
        # role is optional hint; untyped required terms still constrain Filter
        if role in ("topic", "method", "entity"):
            c_type = role
        else:
            c_type = "topic"
        _add(c_type, text, required=True)

    method = slots.get("method")
    if method and str(method).strip():
        _add("method", str(method).strip(), required=True)

    has_topic = any(c.get("type") == "topic" for c in criteria)
    sk = slots.get("query_skeleton")
    skeleton_text = ""
    if isinstance(sk, dict):
        skeleton_text = skeleton_to_text(sk) or str(sk.get("core_text") or "").strip()

    if not has_topic:
        topic = skeleton_text or (slots.get("topic") or question.strip())
        _add("topic", str(topic).strip(), required=True)

    claim = slots.get("claim")
    if claim and str(claim).strip():
        _add("claim", str(claim).strip(), required=True)
    elif skeleton_text and _looks_like_claim(skeleton_text):
        sk_norm = " ".join(skeleton_text.lower().split())
        existing = {
            " ".join(str(c.get("text") or "").lower().split()) for c in criteria
        }
        if sk_norm not in existing:
            _add("claim", skeleton_text, required=True)

    negations = slots.get("negation")
    if negations:
        neg_list = negations if isinstance(negations, list) else [str(negations)]
        for neg in neg_list:
            neg_text = str(neg).strip()
            if neg_text:
                _add("negation", neg_text, required=True, weight=0.0)

    scorable = [
        c
        for c in criteria
        if c.get("type") != "negation" and float(c.get("weight") or 0) > 0
    ]
    total = sum(float(c["weight"]) for c in scorable) or 1.0
    for c in scorable:
        c["weight"] = round(float(c["weight"]) / total, 4)

    criteria.append(
        {
            "name": "intent",
            "type": "intent",
            "text": intent,
            "required": False,
            "description": f"User intent={intent}",
            "weight": 0.0,
        }
    )
    return criteria
