"""Backward-compatible re-export → slot_usage.apply_usage_decision."""

from scholar_ir.query_understanding.slot_usage import (  # noqa: F401
    ExpansionTask,
    RetrievalPlan,
    SlotUsage,
    apply_slot_usage,
    apply_usage_decision,
)

__all__ = [
    "ExpansionTask",
    "RetrievalPlan",
    "SlotUsage",
    "apply_slot_usage",
    "apply_usage_decision",
]
