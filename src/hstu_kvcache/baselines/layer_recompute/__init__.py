"""DroidSpeak-inspired interval replay with real cached boundary inputs."""

from .core import (
    LayerRecomputeState,
    append,
    append_band,
    capture_state,
    enumerate_intervals,
    profile_intervals,
    recompute_interval,
    retain_latest,
)

__all__ = [
    "LayerRecomputeState",
    "append",
    "append_band",
    "capture_state",
    "enumerate_intervals",
    "profile_intervals",
    "recompute_interval",
    "retain_latest",
]
