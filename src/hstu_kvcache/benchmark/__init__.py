"""Release-cost measurement primitives.

These utilities are deliberately independent from training and quality
evaluation.  They construct deterministic random-weight HSTU instances solely
for measuring the release-time cost of persistent K/V state.
"""

from .release_cost import (
    RELEASE_COST_CONFIGURATIONS,
    ReleaseCostConfiguration,
    ReleaseCostEstimate,
    estimate_release_card_hours,
    make_random_hstu,
)

__all__ = [
    "RELEASE_COST_CONFIGURATIONS",
    "ReleaseCostConfiguration",
    "ReleaseCostEstimate",
    "estimate_release_card_hours",
    "make_random_hstu",
]
