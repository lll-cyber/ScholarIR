"""arXiv adapt: method only from SubQuery.filters (Usage Decision)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.search.adapt.arxiv_api import adapt_arxiv
from scholar_ir.types import SubQuery


def test_method_not_from_slots_when_absent_in_filters() -> None:
    sq = SubQuery(qid="q0", text="multimodal audio visual pretraining", filters={})
    slots = {"method": "pre-trained on large-scale datasets"}
    req = adapt_arxiv(sq, slots)
    assert "pre-trained" not in req.query
    assert "large-scale" not in req.query


def test_method_in_filters_is_appended() -> None:
    sq = SubQuery(
        qid="q0",
        text="parameter efficient fine-tuning",
        filters={"method": "LoRA"},
    )
    req = adapt_arxiv(sq, {})
    assert "LoRA" in req.text_used or "lora" in req.text_used.lower()


if __name__ == "__main__":
    test_method_not_from_slots_when_absent_in_filters()
    test_method_in_filters_is_appended()
    print("arxiv adapt tests passed")
