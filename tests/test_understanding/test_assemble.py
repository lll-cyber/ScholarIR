"""Unit tests for semantic-skeleton assemble (no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.query_understanding.assemble import assemble_from_tasks
from scholar_ir.query_understanding.slot_usage import apply_slot_usage
from scholar_ir.query_understanding.slots import empty_skeleton_part, empty_term, ensure_topic_term


def test_assemble_lexical_from_skeleton() -> None:
    slots = ensure_topic_term(
        {
            "topic": "retrieval-augmented generation",
            "method": None,
            "year_from": 2020,
            "terms": [
                empty_term(
                    "retrieval-augmented generation",
                    "topic",
                    abbrev="RAG",
                    synonyms=["retrieval augmented generation"],
                    replaceable=True,
                )
            ],
            "query_skeleton": {
                "core_text": "retrieval-augmented generation",
                "parts": [
                    empty_skeleton_part(
                        "t0",
                        "retrieval-augmented generation",
                        required=True,
                        replaceable=True,
                        variants=["retrieval-augmented generation", "RAG"],
                    )
                ],
            },
        }
    )
    plan = apply_slot_usage("survey", slots, max_n=5)
    subs = assemble_from_tasks(
        "survey on RAG since 2020",
        "survey",
        slots,
        plan,
        max_n=5,
    )
    assert len(subs) >= 1
    assert subs[0].filters.get("year_from") == 2020
    assert all(sq.mode == "lexical" for sq in subs)
    core = next(sq for sq in subs if sq.angle == "core")
    assert "survey" in (core.modifiers or []) or "survey" in core.text.lower()


def test_no_query_drift_multi_part() -> None:
    slots = ensure_topic_term(
        {
            "terms": [
                empty_term("in-context learning", "topic", replaceable=False),
                empty_term(
                    "LLM",
                    "entity",
                    synonyms=["large language models"],
                    replaceable=True,
                ),
                empty_term("pre-training", "other", replaceable=False),
            ],
            "query_skeleton": {
                "core_text": "in-context learning LLM pre-training",
                "parts": [
                    empty_skeleton_part("t0", "in-context learning"),
                    empty_skeleton_part(
                        "t1",
                        "LLM",
                        replaceable=True,
                        variants=["LLM", "large language models"],
                    ),
                    empty_skeleton_part("t2", "pre-training"),
                ],
            },
        }
    )
    plan = apply_slot_usage("related", slots, max_n=5)
    subs = assemble_from_tasks("q", "related", slots, plan, max_n=5)
    assert len(subs) >= 2
    for sq in subs:
        low = sq.text.lower()
        assert "in-context learning" in low
        assert "pre-training" in low
        assert len(sq.text.split()) >= 3


def test_specific_metadata_not_raw_nl() -> None:
    slots = ensure_topic_term(
        {
            "topic": "Attention Is All You Need",
            "authors": ["Vaswani"],
            "terms": [empty_term("Attention Is All You Need", "topic")],
        }
    )
    q = "Find the paper Attention Is All You Need"
    plan = apply_slot_usage("specific", slots, max_n=5, question=q)
    subs = assemble_from_tasks(q, "specific", slots, plan, max_n=5)
    meta = [sq for sq in subs if sq.angle == "metadata"]
    assert meta
    assert all(sq.channel == "metadata" for sq in meta)
    assert all("find the paper" not in sq.text.lower() for sq in meta)


def test_specific_lexical_variants_assembled() -> None:
    slots = ensure_topic_term(
        {
            "query_skeleton": {
                "core_text": "autoregressive transformer video generation",
                "parts": [
                    empty_skeleton_part(
                        "t0",
                        "autoregressive transformer",
                        replaceable=True,
                        variants=["autoregressive transformer", "autoregressive model"],
                    ),
                    empty_skeleton_part(
                        "t1",
                        "video generation",
                        replaceable=True,
                        variants=["video generation", "video synthesis"],
                    ),
                ],
            }
        }
    )
    q = "List all papers that use autoregressive transformer to generate videos."
    plan = apply_slot_usage("specific", slots, max_n=5, question=q)
    subs = assemble_from_tasks(q, "specific", slots, plan, max_n=5)
    angles = {sq.angle for sq in subs}
    assert "core" in angles
    assert "synonym" in angles or "abbrev" in angles
    # original NL must not be labeled metadata
    for sq in subs:
        if "list all papers" in sq.text.lower():
            assert sq.angle == "raw"
            assert sq.channel == "keyword"


if __name__ == "__main__":
    test_assemble_lexical_from_skeleton()
    test_no_query_drift_multi_part()
    test_specific_metadata_not_raw_nl()
    test_specific_lexical_variants_assembled()
    print("assemble tests passed")
