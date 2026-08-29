"""Unit tests for Slot Usage / ExpansionTask (no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.query_understanding.slot_usage import apply_slot_usage, apply_usage_decision
from scholar_ir.query_understanding.slots import empty_skeleton_part, empty_term, ensure_topic_term


def test_recall_mode_drops_method_filter() -> None:
    slots = {
        "topic": "LoRA fine-tuning",
        "method": "LoRA",
        "year_from": 2020,
        "year_to": None,
        "venue": None,
        "authors": None,
        "negation": ["GPT-4"],
        "terms": [empty_term("LoRA fine-tuning", "topic"), empty_term("LoRA", "method")],
    }
    slots = ensure_topic_term(slots)
    plan = apply_slot_usage("method", slots, recall_mode=True, max_n=5)
    assert plan.api_filters.get("year_from") == 2020
    assert "method" not in plan.api_filters
    assert any(t.transform == "core" for t in plan.tasks)


def test_specific_keeps_lexical_and_true_metadata() -> None:
    slots = ensure_topic_term(
        {
            "topic": "Attention Is All You Need",
            "authors": ["Vaswani"],
            "method": None,
            "terms": [empty_term("Attention Is All You Need", "topic")],
        }
    )
    plan = apply_slot_usage(
        "specific",
        slots,
        max_n=5,
        question="Find the paper Attention Is All You Need by Vaswani",
    )
    assert any(t.transform == "metadata" for t in plan.tasks)
    # author is metadata; title also metadata; should not mislabel NL as metadata only
    assert any("Vaswani" in t.text_seed for t in plan.tasks if t.transform == "metadata")


def test_specific_allows_lexical_variants() -> None:
    slots = ensure_topic_term(
        {
            "topic": "target network Deep Q-learning",
            "terms": [
                empty_term(
                    "target network",
                    "entity",
                    synonyms=["target Q-network"],
                    replaceable=True,
                ),
                empty_term(
                    "Deep Q-learning",
                    "method",
                    abbrev="DQN",
                    replaceable=True,
                ),
            ],
            "query_skeleton": {
                "core_text": "target network Deep Q-learning",
                "parts": [
                    empty_skeleton_part(
                        "t0",
                        "target network",
                        replaceable=True,
                        variants=["target network", "target Q-network"],
                    ),
                    empty_skeleton_part(
                        "t1",
                        "Deep Q-learning",
                        replaceable=True,
                        variants=["Deep Q-learning", "DQN"],
                    ),
                ],
            },
        }
    )
    plan = apply_slot_usage(
        "specific",
        slots,
        max_n=5,
        question="List all papers that use target network in Deep Q-learning",
        enable_decomposition=True,  # must still be ignored for specific
    )
    assert not any(t.mode == "decomposition" for t in plan.tasks)
    texts = [t.text_seed.lower() for t in plan.tasks]
    assert any("dqn" in t for t in texts)
    assert any("target q-network" in t for t in texts)
    # raw NL is angle=raw, not metadata
    raws = [t for t in plan.tasks if t.transform == "raw"]
    assert raws
    assert all(t.transform != "metadata" or "list all" not in t.text_seed.lower() for t in plan.tasks)


def test_skeleton_swap_keeps_all_required_parts() -> None:
    """ICL + LLM + pre-training: synonym must not drop other constraints."""
    slots = {
        "topic": "in-context learning LLM pre-training",
        "terms": [
            empty_term("in-context learning", "topic", required=True, replaceable=False),
            empty_term(
                "LLM",
                "entity",
                abbrev="LLM",
                synonyms=["large language model", "large language models"],
                required=True,
                replaceable=True,
            ),
            empty_term("pre-training", "other", required=True, replaceable=False),
        ],
        "query_skeleton": {
            "core_text": "in-context learning LLM pre-training",
            "parts": [
                empty_skeleton_part("t0", "in-context learning", required=True, replaceable=False),
                empty_skeleton_part(
                    "t1",
                    "LLM",
                    required=True,
                    replaceable=True,
                    variants=["LLM", "large language model", "large language models"],
                ),
                empty_skeleton_part("t2", "pre-training", required=True, replaceable=False),
            ],
        },
    }
    slots = ensure_topic_term(slots)
    plan = apply_slot_usage("broad", slots, max_n=5)
    texts = [t.text_seed.lower() for t in plan.tasks]
    assert any("in-context learning" in t and "pre-training" in t for t in texts)
    for t in texts:
        assert "in-context learning" in t
        assert "pre-training" in t
    assert any("large language model" in t for t in texts)


def test_comparative_core_text_preserves_relation() -> None:
    """Claim/comparison must not be flattened into bag-of-phrases."""
    slots = ensure_topic_term(
        {
            "terms": [
                empty_term("LLM", "entity", synonyms=["large language model"], replaceable=True),
                empty_term("smaller dataset", "other", synonyms=["less data"], replaceable=True),
                empty_term("larger dataset", "other", replaceable=False),
            ],
            "query_skeleton": {
                "core_text": (
                    "smaller dataset can produce better models than larger dataset "
                    "in LLM pre-training"
                ),
                "parts": [
                    empty_skeleton_part(
                        "t0",
                        "LLM",
                        replaceable=True,
                        variants=["LLM", "large language model"],
                    ),
                    empty_skeleton_part(
                        "t1",
                        "smaller dataset",
                        replaceable=True,
                        variants=["smaller dataset", "less data"],
                    ),
                    empty_skeleton_part("t2", "larger dataset", replaceable=False),
                ],
            },
        }
    )
    plan = apply_slot_usage("broad", slots, max_n=5)
    texts = [t.text_seed.lower() for t in plan.tasks]
    for t in texts:
        assert "better" in t and "than" in t
        assert "larger dataset" in t
        # must not be bag-of-phrases style
        assert not t.startswith("large language model pre-training dataset")
    assert any("less data" in t and "better" in t and "than" in t for t in texts)
    assert any("large language model" in t and "better" in t for t in texts)


def test_decomposition_opt_in_keeps_other_parts() -> None:
    slots = {
        "topic": "parameter-efficient fine-tuning NLP",
        "terms": [
            empty_term(
                "parameter-efficient fine-tuning",
                "topic",
                abbrev="PEFT",
                instances=["LoRA", "adapter tuning"],
                coverage_gap_likely=True,
                replaceable=True,
            ),
            empty_term("NLP", "other", required=True, replaceable=False),
        ],
        "query_skeleton": {
            "core_text": "parameter-efficient fine-tuning NLP",
            "parts": [
                empty_skeleton_part(
                    "t0",
                    "parameter-efficient fine-tuning",
                    required=True,
                    replaceable=True,
                    variants=["parameter-efficient fine-tuning", "PEFT"],
                ),
                empty_skeleton_part("t1", "NLP", required=True, replaceable=False),
            ],
        },
    }
    slots = ensure_topic_term(slots)
    plan0 = apply_slot_usage("broad", slots, enable_decomposition=False, max_n=5)
    assert not any(t.mode == "decomposition" for t in plan0.tasks)
    plan1 = apply_slot_usage("broad", slots, enable_decomposition=True, max_n=5)
    decomp = [t for t in plan1.tasks if t.mode == "decomposition"]
    assert decomp
    assert all("nlp" in t.text_seed.lower() for t in decomp)


def test_legacy_alias() -> None:
    plan = apply_usage_decision("survey", {"topic": "RAG", "terms": []}, use_llm=False)
    assert isinstance(plan.api_filters, dict)


def test_weak_variants_filtered() -> None:
    from scholar_ir.query_understanding.variant_quality import (
        filter_retrieval_variants,
        is_high_confidence_variant,
    )

    assert not is_high_confidence_variant("video generation", "video generation task")
    assert not is_high_confidence_variant("video generation", "video generation model")
    assert not is_high_confidence_variant("calibration", "calibrating")
    assert not is_high_confidence_variant("autoregressive transformer", "autoregressive transformers")
    assert not is_high_confidence_variant(
        "reconstruction-based techniques", "reconstruction-based methods"
    )
    assert not is_high_confidence_variant("hybrid architectures", "hybrid models")
    assert not is_high_confidence_variant("bias detection", "detecting bias")
    assert not is_high_confidence_variant("peer review", "peer reviews")
    assert is_high_confidence_variant("video generation", "video synthesis")
    assert is_high_confidence_variant("LLM", "large language model")
    assert is_high_confidence_variant("pre-training", "pretraining")
    assert is_high_confidence_variant("smaller dataset", "less data")
    assert is_high_confidence_variant("Deep Q-learning", "DQN")

    slots = ensure_topic_term(
        {
            "query_skeleton": {
                "core_text": "autoregressive transformer video generation",
                "parts": [
                    empty_skeleton_part(
                        "p0",
                        "autoregressive transformer",
                        replaceable=True,
                        variants=[
                            "autoregressive transformer",
                            "autoregressive model",
                            "autoregressive transformer model",
                            "autoregressive transformers",
                        ],
                    ),
                    empty_skeleton_part(
                        "p1",
                        "video generation",
                        replaceable=True,
                        variants=[
                            "video generation",
                            "video synthesis",
                            "video generation task",
                            "video generation model",
                        ],
                    ),
                ],
            }
        }
    )
    plan = apply_slot_usage("broad", slots, max_n=5, max_lexical_swaps=3)
    texts = [t.text_seed.lower() for t in plan.tasks]
    assert any("video synthesis" in t for t in texts)
    assert any("autoregressive model" in t for t in texts)
    assert not any("video generation task" in t for t in texts)
    assert not any("video generation model" in t for t in texts)
    assert not any("transformer model video" in t for t in texts)
    # Does not pad: core + ≤2 high-conf swaps typically
    assert len(plan.tasks) <= 4
    assert len(plan.tasks) >= 2


def test_no_auto_spans_and_overlap_longest() -> None:
    from scholar_ir.query_understanding.slots import (
        dedupe_skeleton_parts,
        merge_terms_into_skeleton,
    )

    sk = merge_terms_into_skeleton(
        {
            "core_text": "in-context learning LLM pre-training",
            "parts": [empty_skeleton_part("p0", "LLM", replaceable=True, variants=["LLM"])],
        },
        [
            empty_term("LLM", "entity", synonyms=["large language model"]),
            empty_term("in-context learning", "topic", synonyms=["ICL"], replaceable=True),
        ],
    )
    assert not any(str(p["id"]).startswith("auto") for p in sk["parts"])

    parts = dedupe_skeleton_parts(
        "models gain in-context learning capability during pre-training",
        [
            empty_skeleton_part("t0", "in-context learning", replaceable=True),
            empty_skeleton_part("t1", "in-context learning capability", replaceable=True),
        ],
    )
    assert [p["text"] for p in parts] == ["in-context learning capability"]


def test_round_robin_under_lexical_cap() -> None:
    slots = ensure_topic_term(
        {
            "query_skeleton": {
                "core_text": "smaller dataset better than larger dataset in LLM pre-training",
                "parts": [
                    empty_skeleton_part(
                        "p0",
                        "smaller dataset",
                        replaceable=True,
                        variants=["smaller dataset", "less data", "smaller training data"],
                    ),
                    empty_skeleton_part(
                        "p1",
                        "larger dataset",
                        replaceable=True,
                        variants=["larger dataset", "more data"],
                    ),
                    empty_skeleton_part(
                        "p2",
                        "LLM",
                        replaceable=True,
                        variants=["LLM", "large language model"],
                    ),
                ],
            }
        }
    )
    plan = apply_slot_usage("broad", slots, max_n=5, max_lexical_swaps=3)
    swapped = [t.swapped_part for t in plan.tasks if t.swapped_part]
    assert set(swapped) == {"p0", "p1", "p2"}


def test_coverage_gap_helper() -> None:
    from scholar_ir.query_understanding.variant_quality import (
        resolve_coverage_gap_likely,
        slots_coverage_gap_likely,
    )

    assert not slots_coverage_gap_likely({"terms": []})
    assert slots_coverage_gap_likely({"coverage_gap_likely": True})
    assert slots_coverage_gap_likely(
        {"terms": [{"text": "x", "coverage_gap_likely": True}]}
    )
    # claim implies conceptual gap
    assert slots_coverage_gap_likely({"claim": "smaller data can outperform larger data"})
    # "language models" must NOT trip umbrella heuristic (no bare "models")
    assert not slots_coverage_gap_likely(
        {
            "coverage_gap_likely": False,
            "query_skeleton": {
                "core_text": "in-context learning in large language models during pre-training"
            },
        }
    )
    # abstract umbrella heads still trip heuristic
    assert slots_coverage_gap_likely(
        {
            "coverage_gap_likely": False,
            "query_skeleton": {
                "core_text": "hybrid architectures reconstruction-based techniques"
            },
        }
    )
    slots = {
        "coverage_gap_likely": False,
        "query_skeleton": {
            "core_text": "hybrid architectures reconstruction-based techniques"
        },
    }
    assert resolve_coverage_gap_likely(slots) is True
    assert slots["coverage_gap_likely"] is True


if __name__ == "__main__":
    test_recall_mode_drops_method_filter()
    test_specific_keeps_lexical_and_true_metadata()
    test_specific_allows_lexical_variants()
    test_skeleton_swap_keeps_all_required_parts()
    test_comparative_core_text_preserves_relation()
    test_decomposition_opt_in_keeps_other_parts()
    test_legacy_alias()
    test_weak_variants_filtered()
    test_no_auto_spans_and_overlap_longest()
    test_round_robin_under_lexical_cap()
    test_coverage_gap_helper()
    print("slot_usage tests passed")
