from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from hstu_kvcache.models import HSTU, HSTUConfig


@dataclass(frozen=True)
class ReleaseCostConfiguration:
    """Architecture point used by the paper's release-cost table."""

    name: str
    num_layers: int
    context_length: int
    hidden_size: int
    num_heads: int

    def model_config(self, *, num_items: int, num_behaviors: int) -> HSTUConfig:
        return HSTUConfig(
            num_items=num_items,
            num_behaviors=num_behaviors,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_seq_len=self.context_length,
            input_dropout=0.0,
            attn_dropout=0.0,
        )


# Head width stays at 64 for the H256/H512 configurations.  Batch size is a
# runtime choice because the largest context point has quadratic attention
# activation memory; the benchmark defaults to the conservative A40 setting 8.
RELEASE_COST_CONFIGURATIONS: tuple[ReleaseCostConfiguration, ...] = (
    ReleaseCostConfiguration("4L_context512_H128_heads4", 4, 512, 128, 4),
    ReleaseCostConfiguration("6L_context1K_H256", 6, 1024, 256, 4),
    ReleaseCostConfiguration("8L_context2K_H512", 8, 2048, 512, 8),
    ReleaseCostConfiguration("16L_context4K_H512", 16, 4096, 512, 8),
    ReleaseCostConfiguration("24L_context8K_H512", 24, 8192, 512, 8),
)


@dataclass(frozen=True)
class ReleaseCostEstimate:
    sampled_users: int
    target_users: int
    elapsed_seconds: float
    card_hours: float

    @property
    def seconds_per_user(self) -> float:
        return self.elapsed_seconds / self.sampled_users

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self) | {"seconds_per_user": self.seconds_per_user}


def estimate_release_card_hours(
    *, elapsed_seconds: float, sampled_users: int, target_users: int = 10_000_000
) -> ReleaseCostEstimate:
    """Linearly extrapolate a measured one-card wall time to a release batch."""
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    if sampled_users < 1:
        raise ValueError("sampled_users must be positive")
    if target_users < 1:
        raise ValueError("target_users must be positive")
    return ReleaseCostEstimate(
        sampled_users=sampled_users,
        target_users=target_users,
        elapsed_seconds=elapsed_seconds,
        card_hours=elapsed_seconds * target_users / sampled_users / 3600.0,
    )


def make_random_hstu(
    configuration: ReleaseCostConfiguration,
    *,
    seed: int,
    num_items: int = 50_000,
    num_behaviors: int = 8,
) -> HSTU:
    """Create a deterministic untrained model for runtime-only measurement."""
    if num_items < 2:
        raise ValueError("num_items must reserve at least PAD and one item")
    if num_behaviors < 2:
        raise ValueError("num_behaviors must reserve at least PAD and one behavior")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HSTU(configuration.model_config(
            num_items=num_items, num_behaviors=num_behaviors
        ))
    return model.eval()
