"""High-confidence retrieval variants + coverage-gap helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Weak padding often invented to fill variant lists — not true synonyms
_WEAK_SUFFIXES = (
    "task",
    "tasks",
    "model",
    "models",
    "method",
    "methods",
    "approach",
    "approaches",
    "system",
    "systems",
    "framework",
    "frameworks",
    "technique",
    "techniques",
    "architecture",
    "architectures",
    "network",
    "networks",
    "problem",
    "problems",
)

# Core-text gap heuristic: umbrella heads that rarely appear as-is in titles.
# NOTE: do NOT include bare "models" — "language models" is concrete indexing.
_GAP_INDICATORS = (
    "techniques",
    "approaches",
    "methods",
    "architectures",
    "frameworks",
    "strategies",
    "mechanisms",
)


def _norm_ws(s: str) -> str:
    return " ".join((s or "").lower().split())


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _norm_ws(s))


def _singularize_token(tok: str) -> str:
    t = tok.lower()
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    for suf in ("ches", "shes", "sses", "xes", "zes"):
        if t.endswith(suf) and len(t) > len(suf) + 1:
            return t[: -len(suf)] + suf[0]  # e.g. boxes→box? rough
    if t.endswith("ses") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _is_plural_only_variant(canon: str, variant: str) -> bool:
    """True when difference is only singular↔plural (transformer↔transformers)."""
    ct, vt = _tokens(canon), _tokens(variant)
    if not ct or not vt or len(ct) != len(vt):
        return False
    changed = 0
    for a, b in zip(ct, vt):
        if a == b:
            continue
        if _singularize_token(a) == _singularize_token(b):
            changed += 1
            continue
        return False
    return changed >= 1


def _strip_weak_head(s: str) -> Optional[str]:
    """Strip trailing weak suffix word; return stem or None if no strip."""
    toks = _tokens(s)
    if len(toks) < 2:
        return None
    if toks[-1] in _WEAK_SUFFIXES:
        return " ".join(toks[:-1])
    return None


def _is_weak_suffix_swap(canon: str, variant: str) -> bool:
    """Reject techniques↔methods / X techniques↔X approaches (same stem)."""
    sc, sv = _strip_weak_head(canon), _strip_weak_head(variant)
    if sc is not None and sv is not None and sc == sv:
        return True
    # canon has no weak head but variant only adds one
    cn, vn = _norm_ws(canon), _norm_ws(variant)
    for suf in _WEAK_SUFFIXES:
        if vn == f"{cn} {suf}":
            return True
        if cn == f"{vn} {suf}":
            return True
    return False


def _light_stem(tok: str) -> str:
    t = _singularize_token(tok)
    for suf, cut in (
        ("ating", 5),  # calibrating
        ("ation", 5),  # calibration
        ("tting", 4),  # detecting? detecting→detect via ing
        ("ting", 4),
        ("sion", 4),
        ("tion", 4),
        ("ing", 3),
        ("ied", 3),
        ("ed", 2),
        ("ly", 2),
    ):
        if t.endswith(suf) and len(t) > cut + 2:
            return t[:-cut]
    return t


def _is_morphology_only_variant(canon: str, variant: str) -> bool:
    """Reject POS/morphology swaps: calibration↔calibrating, detection↔detecting bias."""
    ct, vt = _tokens(canon), _tokens(variant)
    if not ct or not vt:
        return False
    # Same multiset of light stems, different surface → morphology / reorder
    sc = sorted(_light_stem(t) for t in ct)
    sv = sorted(_light_stem(t) for t in vt)
    if sc == sv and _norm_ws(canon) != _norm_ws(variant):
        # Allow pure hyphen/space / casing already excluded by caller
        c_compact = re.sub(r"[^a-z0-9]", "", _norm_ws(canon))
        v_compact = re.sub(r"[^a-z0-9]", "", _norm_ws(variant))
        if c_compact == v_compact:
            return False  # pre-training ↔ pretraining: keep
        return True
    # Classic tion↔ting on single-token pairs
    if len(ct) == 1 and len(vt) == 1:
        c, v = ct[0], vt[0]
        if c.endswith("tion") and v.endswith("ting"):
            if _light_stem(c) == _light_stem(v) or c[: -len("tion")] == v[: -len("ting")]:
                return True
        if c.endswith("sion") and v.endswith("sing"):
            return True
    return False


def is_high_confidence_variant(canon: str, variant: str) -> bool:
    """True iff variant is a retrieval-oriented drop-in for canon (not padding)."""
    c = (canon or "").strip()
    v = (variant or "").strip()
    if not c or not v:
        return False
    if _norm_ws(c) == _norm_ws(v):
        return False

    cn, vn = _norm_ws(c), _norm_ws(v)

    # Reject "X" → "X task/model/method/..."
    for suf in _WEAK_SUFFIXES:
        if vn == f"{cn} {suf}" or vn.startswith(f"{cn} {suf} "):
            return False
        if cn == f"{vn} {suf}" or cn.startswith(f"{vn} {suf} "):
            return False

    if _is_plural_only_variant(c, v):
        return False
    if _is_weak_suffix_swap(c, v):
        return False
    if _is_morphology_only_variant(c, v):
        return False

    # Hyphen / space normalization (pre-training ↔ pretraining)
    c_compact = re.sub(r"[^a-z0-9]", "", cn)
    v_compact = re.sub(r"[^a-z0-9]", "", vn)
    if c_compact == v_compact:
        return True

    # Acronym ↔ expansion
    short, long_ = (c, v) if len(c) <= len(v) else (v, c)
    acronymish = (len(short.split()) <= 2 and len(short) <= 8 and short.isupper()) or (
        len(short.split()) == 1 and len(short) <= 6 and short.isalpha()
    )
    if acronymish and len(short) <= 8:
        initials = "".join(w[0] for w in long_.split() if w)
        if short.upper() == initials.upper() or short.upper() == long_.upper():
            return True
        # LLM ↔ large language model (partial)
        if short.isupper() and short.lower() in _norm_ws(long_).replace(" ", ""):
            return True

    c_toks = set(_tokens(c))
    v_toks = set(_tokens(v))
    if c_toks and v_toks:
        overlap = len(c_toks & v_toks) / max(1, min(len(c_toks), len(v_toks)))
        if overlap < 0.34 and not acronymish:
            if c_compact != v_compact and not (
                c_compact in v_compact or v_compact in c_compact
            ):
                # Allow short cross-synonym NPs (less data ↔ smaller dataset)
                if not (len(c.split()) <= 3 and len(v.split()) <= 3):
                    return False

        if len(v.split()) > len(c.split()) + 2 and overlap < 0.6:
            return False

        if len(c.split()) <= 4 and len(v.split()) <= 4:
            if abs(len(c.split()) - len(v.split())) <= 2:
                if overlap >= 0.34 or c_compact in v_compact or v_compact in c_compact:
                    return True
                if len(c.split()) <= 3 and len(v.split()) <= 3:
                    return True
                return False

    return True


def filter_retrieval_variants(
    canon: str,
    variants: Optional[List[str]],
    *,
    max_alts: int = 2,
) -> List[str]:
    """Keep up to max_alts high-confidence alts (excludes canon)."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in variants or []:
        v = str(raw or "").strip()
        if not v:
            continue
        key = _norm_ws(v)
        if key in seen or key == _norm_ws(canon):
            continue
        if not is_high_confidence_variant(canon, v):
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= max_alts:
            break
    return out


def _core_text_gap_likely(slots: Dict[str, Any]) -> bool:
    sk = slots.get("query_skeleton")
    core = ""
    if isinstance(sk, dict):
        core = sk.get("core_text") or ""
    core = core or slots.get("topic") or ""
    if not core:
        return False
    low = " ".join(core.lower().split())
    return any(f" {w}" in f" {low}" or low.endswith(f" {w}") or low == w for w in _GAP_INDICATORS)


def slots_coverage_gap_likely(slots: Dict[str, Any]) -> bool:
    """Aggregate coverage gap signal from slots / terms / claim / core heuristic."""
    if slots.get("coverage_gap_likely") is True:
        return True
    claim = slots.get("claim")
    if isinstance(claim, str) and claim.strip():
        return True
    for t in slots.get("terms") or []:
        if isinstance(t, dict) and t.get("coverage_gap_likely") is True:
            return True
    sk = slots.get("query_skeleton")
    if isinstance(sk, dict) and sk.get("coverage_gap_likely") is True:
        return True
    return _core_text_gap_likely(slots)


def resolve_coverage_gap_likely(slots: Dict[str, Any]) -> bool:
    """Compute gap once, write bool back to slots (single source of truth)."""
    gap = bool(slots_coverage_gap_likely(slots))
    slots["coverage_gap_likely"] = gap
    return gap
