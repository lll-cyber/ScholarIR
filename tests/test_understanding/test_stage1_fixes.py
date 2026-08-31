"""Tests for Stage-1 fixes: auto-decompose, heuristic coverage_gap, rewrite_query."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.query_understanding.base import understand
from scholar_ir.query_understanding.slot_usage import (
    ExpansionTask,
    RetrievalPlan,
    should_auto_decompose,
)
from scholar_ir.query_understanding.assemble import (
    rewrite_query,
    assemble_from_tasks,
)
from scholar_ir.query_understanding.rules import (
    _detect_heuristic_claim,
    extract_slots_heuristic,
)


# ---- rewrite_query tests ----

def test_rewrite_strips_question_prefix() -> None:
    cases = [
        ("list papers about deep learning", "deep learning"),
        ("find papers on transformer architecture", "transformer architecture"),
        ("what are the best methods for LLM alignment", "best methods for llm alignment"),
        ("please show me papers about diffusion models", "diffusion models"),
        ("can you find papers using RLHF", "find papers using rlhf"),
        ("i need papers about retrieval augmented generation", "retrieval augmented generation"),
    ]
    for inp, expected in cases:
        got = rewrite_query(inp)
        assert got == expected, f"input: {inp!r} → got {got!r}, expected {expected!r}"


def test_rewrite_keeps_proper_nouns() -> None:
    # "show me" prefix stripped, search-relevant tail kept
    got = rewrite_query("show me research on LoRA fine-tuning")
    # "show me research on" is stripped as prefix; "LoRA fine-tuning" is search-relevant
    assert got == "lora fine-tuning", f"got {got!r}"


def test_rewrite_strips_trailing_punctuation() -> None:
    got = rewrite_query("papers about attention mechanisms???")
    assert got == "attention mechanisms"


def test_rewrite_lowercases() -> None:
    got = rewrite_query("Papers About TRANSFORMER MODELS")
    assert got == "transformer models"


def test_rewrite_truncates_long() -> None:
    long_text = " ".join(["transformer attention mechanism self-supervised learning " for _ in range(10)])
    got = rewrite_query(long_text, max_tokens=6)
    assert len(got.split()) <= 6
    # head is kept (search-relevant terms are at the front)
    assert got.startswith("transformer"), f"expected head kept, got: {got}"


def test_rewrite_drops_too_short() -> None:
    got = rewrite_query("Papers")
    assert got is None


# ---- _detect_heuristic_claim tests ----

def test_claim_better_than() -> None:
    cases = [
        "Is diffusion better than autoregressive for video generation",
        "Which method is better, BERT or GPT for QA",
        "Does Transformer outperform LSTM in translation",
    ]
    for q in cases:
        assert _detect_heuristic_claim(q) is not None, f"missed claim in: {q}"


def test_claim_causality() -> None:
    cases = [
        "How does dropout affect model generalization",
        "What causes catastrophic forgetting in neural networks",
        "Why does scaling improve LLM performance",
    ]
    for q in cases:
        assert _detect_heuristic_claim(q) is not None, f"missed claim in: {q}"


def test_claim_prove_show() -> None:
    cases = [
        "Prove that transformers are better than RNNs",
        "Show that retrieval helps language models",
        "Demonstrate that data efficiency matters for pretraining",
    ]
    for q in cases:
        assert _detect_heuristic_claim(q) is not None, f"missed claim in: {q}"


def test_claim_none_for_boring() -> None:
    cases = [
        "What is attention mechanism",
        "How does BERT work",
        "Papers about neural networks",
    ]
    for q in cases:
        assert _detect_heuristic_claim(q) is None, f"false positive claim in: {q}"


# ---- extract_slots_heuristic coverage_gap tests ----

def test_heuristic_short_topic_sets_gap() -> None:
    for q in ["transformers", "attention", "deep learning"]:
        _, slots = extract_slots_heuristic(q)
        assert slots.get("coverage_gap_likely") is True, f"missed gap for: {q}"


def test_heuristic_claim_sets_gap() -> None:
    _, slots = extract_slots_heuristic(
        "Is diffusion better than GANs for image generation"
    )
    assert slots.get("coverage_gap_likely") is True
    assert slots.get("claim") is not None


def test_heuristic_long_topic_no_false_gap() -> None:
    _, slots = extract_slots_heuristic(
        "attention mechanism in transformer architecture for NLP"
    )
    # long topic, no claim → gap not forced
    assert slots.get("coverage_gap_likely") is not True


# ---- should_auto_decompose tests ----

def test_decompose_true_for_term_with_gap_and_instances() -> None:
    slots = {
        "terms": [
            {"text": "video generation", "role": "topic", "coverage_gap_likely": True,
             "instances": ["text-to-video", "T2V", "video synthesis"]},
        ],
    }
    assert should_auto_decompose(slots) is True


def test_decompose_true_for_multi_aspect_skeleton() -> None:
    slots = {
        "query_skeleton": {
            "core_text": "transformer using preprocessing on dataset",
            "parts": [
                {"id": "p0", "text": "transformer", "required": True, "replaceable": True},
                {"id": "p1", "text": "preprocessing", "required": True, "replaceable": True},
                {"id": "p2", "text": "dataset", "required": False, "replaceable": True},
            ],
        },
    }
    assert should_auto_decompose(slots) is True


def test_decompose_false_for_specific_intent() -> None:
    assert should_auto_decompose({}, intent="specific") is False
    slots = {
        "terms": [
            {"text": "DONE", "role": "topic", "coverage_gap_likely": True,
             "instances": ["instance1"]},
        ],
    }
    assert should_auto_decompose(slots, intent="specific") is False


def test_decompose_false_for_vague_slots() -> None:
    assert should_auto_decompose({"terms": []}) is False
    assert should_auto_decompose({}) is False


def test_decompose_false_for_term_gap_no_instances() -> None:
    slots = {
        "terms": [
            {"text": "deep learning", "role": "topic", "coverage_gap_likely": True,
             "instances": []},
        ],
    }
    assert should_auto_decompose(slots) is False


# ---- Integration: understand() with new fixes ----

def test_understand_auto_decompose_for_multi_aspect() -> None:
    q = "transformer using preprocessing on ImageNet dataset"
    result = understand(q, {"use_llm": False})
    # verify sub_queries exist
    assert len(result.sub_queries) >= 1
    # check decomposition was enabled (auto)
    decomp_trace = [t for t in result.trace if "slot_usage_lexical" in t.get("step", "")]
    if decomp_trace:
        step = decomp_trace[0]
        assert step.get("enable_decomposition") is True or step.get("n_tasks", 0) >= 1


def test_understand_heuristic_claim_sets_gap() -> None:
    q = "Is diffusion better than GANs for image generation"
    result = understand(q, {"use_llm": False})
    # heuristic should set coverage_gap even without LLM
    gap = result.slots.get("coverage_gap_likely")
    assert gap is True, f"coverage_gap_likely should be True, got {gap}"


def test_understand_short_topic_auto_gap() -> None:
    q = "transformers"
    result = understand(q, {"use_llm": False})
    assert result.slots.get("coverage_gap_likely") is True


def test_understand_subqueries_rewritten() -> None:
    q = "list papers about transformer architecture"
    result = understand(q, {"use_llm": False})
    texts = [sq.text for sq in result.sub_queries]
    # should not contain question prefixes
    for t in texts:
        assert not t.startswith("list papers about"), f"unwritten: {t}"
    # should be lowercase
    for t in texts:
        assert t == t.lower(), f"not lowercased: {t}"


if __name__ == "__main__":
    import traceback

    tests = [
        # rewrite
        test_rewrite_strips_question_prefix,
        test_rewrite_keeps_proper_nouns,
        test_rewrite_strips_trailing_punctuation,
        test_rewrite_lowercases,
        test_rewrite_truncates_long,
        test_rewrite_drops_too_short,
        # claim detection
        test_claim_better_than,
        test_claim_causality,
        test_claim_prove_show,
        test_claim_none_for_boring,
        # heuristic gap
        test_heuristic_short_topic_sets_gap,
        test_heuristic_claim_sets_gap,
        test_heuristic_long_topic_no_false_gap,
        # auto decompose
        test_decompose_true_for_term_with_gap_and_instances,
        test_decompose_true_for_multi_aspect_skeleton,
        test_decompose_false_for_specific_intent,
        test_decompose_false_for_vague_slots,
        test_decompose_false_for_term_gap_no_instances,
        # integration
        test_understand_auto_decompose_for_multi_aspect,
        test_understand_heuristic_claim_sets_gap,
        test_understand_short_topic_auto_gap,
        test_understand_subqueries_rewritten,
    ]

    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"  FAILED: {fn.__name__}")
            failed += 1
            continue
        print(f"  ok: {fn.__name__}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
