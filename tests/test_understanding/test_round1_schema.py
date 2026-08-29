"""Unit tests for tightened Round1 schema normalize (filters/term_groups/claim)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.query_understanding.llm_extract import _normalize
from scholar_ir.query_understanding.slots import normalize_round1_output, slots_to_criteria


CLAIM_EXAMPLE = {
    "intent": "related",
    "filters": {
        "year_from": None,
        "year_to": None,
        "venue": None,
        "authors": None,
        "negation": None,
    },
    "coverage_gap_likely": True,
    "claim": (
        "smaller or curated training data can outperform larger datasets "
        "in language model pretraining"
    ),
    "term_groups": [
        {
            "canonical": "language model pretraining",
            "variants": ["LLM pretraining", "large language model pre-training"],
            "abbrev": "LLM",
            "required": True,
            "replaceable": True,
            "role": "topic",
        },
        {
            "canonical": "dataset size",
            "variants": ["training data scale", "corpus size"],
            "abbrev": None,
            "required": False,
            "replaceable": True,
            "role": None,
        },
    ],
    "query_skeleton": {
        "core_text": "smaller dataset leads to better language model pretraining than larger dataset",
        "parts": [
            {
                "id": "p0",
                "text": "smaller dataset",
                "required": True,
                "replaceable": True,
                "variants": ["smaller dataset", "less data"],
            },
            {
                "id": "p1",
                "text": "language model pretraining",
                "required": True,
                "replaceable": True,
                "variants": ["language model pretraining", "LLM pretraining"],
            },
            {
                "id": "p2",
                "text": "larger dataset",
                "required": True,
                "replaceable": False,
                "variants": ["larger dataset"],
            },
        ],
    },
}

METHOD_EXAMPLE = {
    "intent": "method",
    "filters": {
        "year_from": None,
        "year_to": None,
        "venue": None,
        "authors": None,
        "negation": None,
    },
    "coverage_gap_likely": False,
    "claim": None,
    "term_groups": [
        {
            "canonical": "autoregressive transformer",
            "variants": ["autoregressive model"],
            "abbrev": None,
            "required": True,
            "replaceable": True,
            "role": "method",
        },
        {
            "canonical": "video generation",
            "variants": ["video synthesis"],
            "abbrev": None,
            "required": True,
            "replaceable": True,
            "role": "topic",
        },
    ],
    "query_skeleton": {
        "core_text": "autoregressive transformer video generation",
        "parts": [
            {
                "id": "p0",
                "text": "autoregressive transformer",
                "required": True,
                "replaceable": True,
                "variants": ["autoregressive transformer", "autoregressive model"],
            },
            {
                "id": "p1",
                "text": "video generation",
                "required": True,
                "replaceable": True,
                "variants": ["video generation", "video synthesis"],
            },
        ],
    },
}


def test_normalize_claim_example() -> None:
    slots = normalize_round1_output(CLAIM_EXAMPLE)
    assert slots["coverage_gap_likely"] is True
    assert "outperform" in (slots.get("claim") or "")
    assert slots.get("term_groups") and len(slots["term_groups"]) == 2
    terms = slots.get("terms") or []
    assert len(terms) == 2
    by_text = {t["text"]: t for t in terms}
    lm = by_text["language model pretraining"]
    assert lm["required"] is True
    assert lm["role"] == "topic"
    assert lm["abbrev"] == "LLM"
    assert "LLM pretraining" in (lm.get("synonyms") or [])
    ds = by_text["dataset size"]
    assert ds["required"] is False
    assert ds["role"] is None
    sk = slots["query_skeleton"]
    assert "smaller dataset" in sk["core_text"]
    assert len(sk["parts"]) == 3


def test_normalize_method_example_criteria() -> None:
    intent, slots = _normalize(METHOD_EXAMPLE)
    assert intent == "method"
    assert slots.get("claim") is None
    assert slots.get("method") == "autoregressive transformer"
    assert slots.get("topic") == "video generation"
    criteria = slots_to_criteria(intent, slots, "autoregressive transformer video")
    types = {c["type"]: c for c in criteria if c.get("weight", 0) > 0}
    assert "method" in types
    assert "topic" in types
    assert types["method"]["text"] == "autoregressive transformer"
    assert not any(c.get("type") == "claim" and c.get("weight", 0) > 0 for c in criteria)


def test_claim_enters_criteria() -> None:
    intent, slots = _normalize(CLAIM_EXAMPLE)
    criteria = slots_to_criteria(intent, slots, "smaller dataset question")
    claims = [c for c in criteria if c.get("type") == "claim" and c.get("required")]
    assert len(claims) == 1
    assert "outperform" in claims[0]["text"]


def test_legacy_slots_schema_still_works() -> None:
    intent, slots = _normalize(
        {
            "intent": "survey",
            "slots": {
                "topic": "video generation",
                "method": "LoRA",
                "year_from": 2020,
                "coverage_gap_likely": False,
                "terms": [
                    {"text": "LoRA", "role": "method", "required": True},
                    {"text": "video generation", "role": "topic", "required": True},
                ],
                "query_skeleton": {
                    "core_text": "LoRA video generation",
                    "parts": [
                        {"id": "p0", "text": "LoRA", "required": True, "replaceable": False},
                        {
                            "id": "p1",
                            "text": "video generation",
                            "required": True,
                            "replaceable": True,
                            "variants": ["video generation", "video synthesis"],
                        },
                    ],
                },
            },
        }
    )
    assert intent == "survey"
    assert slots["year_from"] == 2020
    assert slots["method"] == "LoRA"
    texts = {t["text"] for t in slots["terms"]}
    assert "LoRA" in texts and "video generation" in texts


def test_untyped_required_term_still_in_criteria() -> None:
    slots = normalize_round1_output(
        {
            "intent": "broad",
            "filters": {},
            "coverage_gap_likely": False,
            "claim": None,
            "term_groups": [
                {
                    "canonical": "contrastive learning",
                    "variants": [],
                    "abbrev": None,
                    "required": True,
                    "replaceable": False,
                    "role": None,
                }
            ],
            "query_skeleton": {
                "core_text": "contrastive learning",
                "parts": [
                    {
                        "id": "p0",
                        "text": "contrastive learning",
                        "required": True,
                        "replaceable": False,
                    }
                ],
            },
        }
    )
    criteria = slots_to_criteria("broad", slots, "contrastive learning")
    texts = {c["text"] for c in criteria if c.get("required") and c.get("weight", 0) > 0}
    assert "contrastive learning" in texts


if __name__ == "__main__":
    test_normalize_claim_example()
    test_normalize_method_example_criteria()
    test_claim_enters_criteria()
    test_legacy_slots_schema_still_works()
    test_untyped_required_term_still_in_criteria()
    print("round1 schema tests passed")
