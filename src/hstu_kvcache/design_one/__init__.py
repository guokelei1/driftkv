"""Compact source summaries and personalized candidate-read correction."""

from .adapter import CompatibilityView, SharedReadAdapter, SummaryProjection
from .summary import ProducerSummary

__all__ = [
    "CompatibilityView", "ProducerSummary", "SharedReadAdapter",
    "SummaryProjection",
]
