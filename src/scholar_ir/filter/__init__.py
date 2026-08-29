"""Stage 2 (part): filter irrelevant / low-quality candidates after search."""

from .base import filter_papers

# Backward-compatible alias
judge = filter_papers

__all__ = ["filter_papers", "judge"]
