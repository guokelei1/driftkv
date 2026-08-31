from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from broadcast_residual import generate_av_broadcast_residual
from hstu_kvcache.models import HSTU, HSTUConfig
from one_release_refinement import (
    build_broadcast_probe_source_cache,
    parameter_cast_maps,
)
from reader_compatibility_correction import intervene_reader_correction


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=64,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=32,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()


def test_compact_probe_source_and_av_sidecar_are_replayable() -> None:
    parent, current = _model(29), _model(31)
    items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
    deltas = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]).float()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    source, layout = build_broadcast_probe_source_cache(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        cast_maps=parameter_cast_maps(parent, current),
        repair_width=8,
    )
    assert layout.carriers == 2
    assert layout.padding_positions == 6
    assert source.seq_len == parent_cache.seq_len == 8
    assert torch.count_nonzero(source.k[:, :, :6]) == 0
    assert torch.count_nonzero(source.v[:, :, :6]) == 0

    residual = generate_av_broadcast_residual(
        current, source, parent_cache, items[:, -1]
    )
    assert residual.replay_max_abs_error < 1e-6
    assert len(residual.corrections) == 2
    assert residual.corrections[0].shape == (1, 2, 8)
    scores, _ = intervene_reader_correction(
        current,
        parent_cache,
        torch.tensor([[9, 10, 11]]),
        torch.tensor([10.0]),
        stage="av_aggregation",
        corrections=residual.corrections,
    )
    assert scores.shape == (1, 3)
