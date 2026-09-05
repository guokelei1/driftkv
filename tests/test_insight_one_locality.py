from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache  # noqa: E402
from insight_one_locality.common import (  # noqa: E402
    HISTORY,
    LOCALITY_CONFIGS,
    PATH_IDS,
    _prefix_step,
    config_mask,
    hybrid_cache,
    stable_topk_mask,
    token_importance_scores,
    token_masks,
)


def test_frozen_locality_matrix_has_expected_counts_and_costs() -> None:
    assert len(LOCALITY_CONFIGS) == 34
    assert len(PATH_IDS) == 36
    assert Counter(config.family for config in LOCALITY_CONFIGS) == {
        "layer": 14,
        "window": 8,
        "token": 12,
    }
    assert sorted({config.cost for config in LOCALITY_CONFIGS if config.family == "layer"}) == [
        1 / 6,
        2 / 6,
        3 / 6,
        4 / 6,
    ]
    assert sorted({config.cost for config in LOCALITY_CONFIGS if config.family == "window"}) == [
        128 / 1024,
        256 / 1024,
        512 / 1024,
        768 / 1024,
    ]


def test_stable_topk_prefers_recent_positions_on_ties() -> None:
    scores = torch.tensor([[1.0, 2.0, 2.0, 2.0, 0.0]])
    mask = stable_topk_mask(scores, 2)
    assert torch.equal(mask, torch.tensor([[False, False, True, True, False]]))


def test_every_config_mask_splices_both_tensors_exactly() -> None:
    generator = torch.Generator().manual_seed(17)
    parent = HSTUKVCache(
        k=torch.randn(6, 2, HISTORY, 4, generator=generator),
        v=torch.randn(6, 2, HISTORY, 4, generator=generator),
        seq_len=HISTORY,
    )
    current = HSTUKVCache(
        k=torch.randn(6, 2, HISTORY, 4, generator=generator),
        v=torch.randn(6, 2, HISTORY, 4, generator=generator),
        seq_len=HISTORY,
    )
    selector_scores = {
        name: torch.randn(2, HISTORY, generator=generator)
        for name in ("ATTN_MASS", "READ_NORM", "PERSISTENCE", "KV_DRIFT", "READ_DELTA")
    }
    selected_tokens = token_masks(selector_scores)
    for config in LOCALITY_CONFIGS:
        mask = config_mask(config, 2, torch.device("cpu"), selected_tokens)
        hybrid = hybrid_cache(parent, current, mask)
        assert torch.equal(hybrid.k[mask], current.k[mask])
        assert torch.equal(hybrid.v[mask], current.v[mask])
        assert torch.equal(hybrid.k[~mask], parent.k[~mask])
        assert torch.equal(hybrid.v[~mask], parent.v[~mask])


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=128,
            num_behaviors=4,
            hidden_size=12,
            num_layers=6,
            num_heads=3,
            head_dim=4,
            max_seq_len=HISTORY,
            temporal_num_freqs=2,
            input_dropout=0.0,
            attn_dropout=0.0,
        )
    ).eval()


def test_selector_reader_trace_matches_native_reader_and_masks_are_finite() -> None:
    parent, current = _model(3), _model(5)
    items = (torch.arange(HISTORY)[None, :] % 100) + 1
    behaviors = (torch.arange(HISTORY)[None, :] % 4) + 1
    deltas = torch.ones(1, HISTORY)
    deltas[:, 0] = 0.0
    candidates = torch.tensor([[101, 102, 103, 104]])
    query_deltas = torch.tensor([7.0])
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    current_cache = current.compute_kv(items, behaviors, deltas)

    x = current.embed_query_tokens(candidates, query_deltas).reshape(
        -1, 1, current.cfg.hidden_size
    )
    for layer, block in enumerate(current.blocks):
        x, _, _ = _prefix_step(current, block, x, parent_cache, layer, candidates.shape[1])
    traced = current.cc_score_head(current.final_norm(x)).reshape(1, -1)
    native = current.score_cc_reuse(parent_cache, candidates, query_deltas)
    # The trace contracts the same tensors with einsum to avoid repeating the
    # 1024-position cache across candidates; accumulation order may differ by
    # a few fp32 ulps from the native matmul path.
    assert torch.allclose(traced, native, atol=3e-6, rtol=1e-6)

    importance = token_importance_scores(
        current,
        parent_cache,
        current_cache,
        candidates,
        query_deltas,
        candidate_chunk=2,
    )
    assert set(importance) == {
        "ATTN_MASS",
        "READ_NORM",
        "PERSISTENCE",
        "KV_DRIFT",
        "READ_DELTA",
    }
    assert all(value.shape == (1, HISTORY) for value in importance.values())
    assert all(bool(torch.isfinite(value).all()) for value in importance.values())
    masks = token_masks(importance)
    assert len(masks) == 12
    assert all(mask.shape == (1, HISTORY) for mask in masks.values())
