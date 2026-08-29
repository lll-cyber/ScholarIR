"""Backward-compatible alias → filter.filter_papers."""

from scholar_ir.filter import filter_papers, judge

__all__ = ["filter_papers", "judge"]
