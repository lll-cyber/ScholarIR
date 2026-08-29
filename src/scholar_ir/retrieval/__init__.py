"""Backward-compatible alias → search."""

from scholar_ir.search import *  # noqa: F401,F403
from scholar_ir.search import retrieve

__all__ = ["retrieve"]
