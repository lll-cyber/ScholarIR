# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# Slimmed from SPAR utils.py — keep helpers needed by Retrieval/Eval.
# Plotting / Excel helpers were dropped to avoid hard deps.
# ==================================================================

from typing import List
import hashlib
import re
import statistics


def fetch_string(raw_str):
    raw_str = raw_str.strip()
    if "```" in raw_str:
        pattern = r"(?s)(?:```json|```)\n([\s\S]*?)\n```"
        match = re.search(pattern, raw_str)
        if match:
            extracted_json = match.group(1)
        else:
            extracted_json = raw_str.replace("```json", "").replace("```", "")
    else:
        extracted_json = raw_str
    return extracted_json


def calculate_statistics(numbers: List[float]) -> dict:
    if not numbers:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "stdev": 0,
            "min": 0,
            "max": 0,
        }
    return {
        "count": len(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
        "min": min(numbers),
        "max": max(numbers),
    }


def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    return "".join(letters).lower()


def get_md5(string):
    return hashlib.md5(string.encode("utf-8")).hexdigest()
