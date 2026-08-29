"""Set-level Precision / Recall / F1 for paper id collections."""

from __future__ import annotations

from typing import Iterable, Set, Tuple

from scholar_ir.vendor_spar.utils import keep_letters


def normalize_id(x: str) -> str:
    x = (x or "").strip().lower()
    # strip arxiv version suffix: 2009.02040v2 -> 2009.02040
    if "v" in x and x.split("v")[0].replace(".", "").isdigit():
        return x.split("v")[0]
    return x


def normalize_title(title: str) -> str:
    return keep_letters(title or "")


def set_prf(
    pred: Iterable[str],
    gold: Iterable[str],
    *,
    as_title: bool = False,
) -> Tuple[float, float, float]:
    """Return precision, recall, f1 over sets."""
    norm = normalize_title if as_title else normalize_id
    p_set: Set[str] = {norm(x) for x in pred if x}
    g_set: Set[str] = {norm(x) for x in gold if x}
    p_set.discard("")
    g_set.discard("")
    if not g_set:
        return 0.0, 0.0, 0.0
    if not p_set:
        return 0.0, 0.0, 0.0
    tp = len(p_set & g_set)
    precision = tp / len(p_set)
    recall = tp / len(g_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
