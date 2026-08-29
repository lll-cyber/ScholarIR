"""Eval helpers: load PaSa jsonl, macro-average set F1."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scholar_ir.eval.metrics import normalize_id, set_prf

_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(v\d+)?$", re.I)


def is_arxiv_id(x: str) -> bool:
    x = normalize_id(x)
    return bool(_ARXIV_ID_RE.match(x))


def arxiv_only(ids: Iterable[str]) -> List[str]:
    return [normalize_id(x) for x in ids if is_arxiv_id(x)]


@dataclass
class EvalRow:
    qid: str
    question: str
    gold_ids: List[str]
    gold_titles: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def load_pasa_jsonl(
    path: Path,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                EvalRow(
                    qid=str(obj.get("qid") or f"row_{i}"),
                    question=obj["question"],
                    gold_ids=list(obj.get("answer_arxiv_id") or []),
                    gold_titles=list(obj.get("answer") or []),
                    meta=obj.get("source_meta") or {},
                )
            )
    return rows


@dataclass
class SampleScore:
    qid: str
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    n_pred_arxiv: int
    paper_ids: List[str]
    latency_ms: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    n: int
    macro_p: float
    macro_r: float
    macro_f1: float
    samples: List[SampleScore]


def score_sample(
    pred_ids: Iterable[str],
    gold_ids: Iterable[str],
    *,
    qid: str = "",
    pred_arxiv_only: bool = True,
    latency_ms: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> SampleScore:
    pred_list = [normalize_id(x) for x in pred_ids if x]
    gold_list = [normalize_id(x) for x in gold_ids if x]
    scored_pred = arxiv_only(pred_list) if pred_arxiv_only else pred_list
    p, r, f1 = set_prf(scored_pred, gold_list)
    return SampleScore(
        qid=qid,
        precision=p,
        recall=r,
        f1=f1,
        n_pred=len(pred_list),
        n_gold=len(gold_list),
        n_pred_arxiv=len(arxiv_only(pred_list)),
        paper_ids=pred_list,
        latency_ms=latency_ms,
        extra=extra or {},
    )


def macro_average(samples: List[SampleScore]) -> EvalSummary:
    n = len(samples)
    if n == 0:
        return EvalSummary(n=0, macro_p=0.0, macro_r=0.0, macro_f1=0.0, samples=[])
    return EvalSummary(
        n=n,
        macro_p=sum(s.precision for s in samples) / n,
        macro_r=sum(s.recall for s in samples) / n,
        macro_f1=sum(s.f1 for s in samples) / n,
        samples=samples,
    )
