"""Backward-compatible alias → query_understanding."""

from scholar_ir.query_understanding import *  # noqa: F401,F403
from scholar_ir.query_understanding import understand

__all__ = ["understand"]
