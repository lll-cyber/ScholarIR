"""Unit tests for deterministic relevance criteria + coverage-gap policy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.filter.base import _score_paper
from scholar_ir.query_understanding.slot_usage import (
    apply_slot_usage,
    finalize_task_budget,
    ExpansionTask,
)
from scholar_ir.query_understanding.slots import empty_skeleton_part, empty_term, slots_to_criteria
from scholar_ir.types import PaperRef, UnderstandingResult


def test_criteria_from_required_terms_not_intent() -> None:
    slots = {
        "topic": "video generation",
        "method": None,
        "terms": [
            empty_term("autoregressive transformer", "method", required=True),
            empty_term("video generation", "topic", required=True),
        ],
        "query_skeleton": {
            "core_text": "autoregressive transformer video generation",
            "parts": [
                empty_skeleton_part("p0", "autoregressive transformer", replaceable=True),
                empty_skeleton_part("p1", "video generation", replaceable=True),
            ],
        },
    }
    criteria = slots_to_criteria("survey", slots, "List papers using autoregressive transformer for video generation")
    by_type = {c["type"]: c for c in criteria if c.get("weight", 0) > 0}
    assert "method" in by_type
    assert "topic" in by_type
    assert by_type["method"]["text"] == "autoregressive transformer"
    assert by_type["method"]["required"] is True
    assert by_type["topic"]["text"] == "video generation"


def test_criteria_includes_both_method_terms() -> None:
    slots = {
        "terms": [
            empty_term("target networks", "method", required=True),
            empty_term("Deep Q-learning", "method", required=True),
        ],
        "query_skeleton": {
            "core_text": "target networks Deep Q-learning",
            "parts": [],
        },
    }
    criteria = slots_to_criteria("specific", slots, "target networks for Deep Q-learning")
    methods = [c for c in criteria if c.get("type") == "method" and c.get("required")]
    assert len(methods) == 2
    texts = {c["text"] for c in methods}
    assert texts == {"target networks", "Deep Q-learning"}


def test_filter_required_method_is_soft_not_hard_zero() -> None:
    """required method criteria feed the LLM judge; keyword path does not hard-zero.

    Design choice: brittle token matching on method phrases kills recall
    (synonyms / paraphrases). Hard year/negation rules stay; method/topic
    requirements are soft via relevance_criteria → LLM prompt.
    """
    criteria = slots_to_criteria(
        "survey",
        {
            "terms": [
                empty_term("autoregressive transformer", "method"),
                empty_term("video generation", "topic"),
            ]
        },
        "autoregressive transformer video generation",
    )
    assert any(c.get("type") == "method" and c.get("required") for c in criteria)

    understanding = UnderstandingResult(
        raw_question="autoregressive transformer video generation",
        intent="survey",
        slots={"topic": "video generation"},
        relevance_criteria=criteria,
    )
    diffusion = PaperRef(
        paper_id="p1",
        title="Diffusion models for video generation",
        abstract="We generate videos with diffusion.",
    )
    ar = PaperRef(
        paper_id="p2",
        title="Autoregressive transformer for video generation",
        abstract="We use autoregressive transformers to generate videos.",
    )
    score_bad, reason_bad = _score_paper(diffusion, understanding, {})
    score_good, reason_good = _score_paper(ar, understanding, {})
    # No method_missing hard kill
    assert reason_bad != "method_missing"
    assert score_bad > 0.0
    # Matching paper still ranks higher on keyword coverage
    assert score_good > score_bad
    assert "keyword_coverage" in reason_good


def test_specific_gap_skips_raw() -> None:
    slots = {
        "coverage_gap_likely": True,
        "terms": [empty_term("smaller dataset improves LLM pretraining", "topic")],
        "query_skeleton": {
            "core_text": "smaller dataset improves LLM pretraining",
            "parts": [
                empty_skeleton_part(
                    "p0",
                    "smaller dataset",
                    replaceable=True,
                    variants=["smaller dataset", "less data"],
                ),
            ],
        },
    }
    plan = apply_slot_usage(
        "specific",
        slots,
        max_n=5,
        question="Can smaller datasets produce better LLM models?",
        coverage_gap_likely=True,
        semantic_reserve=2,
    )
    assert not any(t.transform == "raw" for t in plan.tasks)
    assert len(plan.tasks) <= 3


def test_finalize_prefers_conceptual_over_raw() -> None:
    tasks = [
        ExpansionTask("lexical", "core", "core", "core query"),
        ExpansionTask("lexical", "synonym", "swap", "synonym query"),
        ExpansionTask("semantic", "conceptual", "sem", "conceptual hypothesis"),
        ExpansionTask("lexical", "raw", "raw", "original long natural language question"),
    ]
    out = finalize_task_budget(tasks, max_n=3)
    transforms = [t.transform for t in out]
    assert transforms == ["core", "synonym", "conceptual"]


if __name__ == "__main__":
    test_criteria_from_required_terms_not_intent()
    test_criteria_includes_both_method_terms()
    test_filter_required_method_is_soft_not_hard_zero()
    test_specific_gap_skips_raw()
    test_finalize_prefers_conceptual_over_raw()
    print("criteria tests passed")
