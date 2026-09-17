"""The shared competitor path also handles a ten-layer native HSTU model."""

import numpy as np
import torch
from design.competitor_probe import evaluate_batch, path_records, summarize_scores

from hstu_kvcache.baselines.kv_translate import fit
from hstu_kvcache.models import HSTU, HSTUConfig


def _model(seed):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(
        num_items=64, num_behaviors=3, hidden_size=20, num_layers=10,
        num_heads=2, max_seq_len=32, block_variant="hstu_reference",
        activation="silu", input_dropout=0, attn_dropout=0,
    )).eval()


def _events(seed, batch, length):
    rng = torch.Generator().manual_seed(seed)
    return (
        torch.randint(1, 65, (batch, length), generator=rng),
        torch.randint(1, 4, (batch, length), generator=rng),
        torch.randint(1, 30, (batch, length), generator=rng).float(),
    )


@torch.no_grad()
def test_ten_layer_native_competitors_keep_full_and_reuse_endpoints():
    parent, current = _model(31), _model(47)
    # Independent synthetic histories exercise fitting, source-layer selection,
    # and scoring without loading any real model or evaluation panel.
    calibration = _events(101, batch=2, length=12)
    selection = _events(103, batch=1, length=11)
    translator = fit(
        parent.compute_kv(*calibration), current.compute_kv(*calibration),
        num_heads=2, k=2, ridge=0.01,
        selection_source=parent.compute_kv(*selection),
        selection_target=current.compute_kv(*selection),
    )
    assert translator.source_layers.shape == (current.cfg.num_layers, 2)
    assert translator.selection_evaluation == "provided_validation"

    raw = _events(107, batch=2, length=6)
    candidates = torch.arange(20, 32).repeat(2, 1)
    query_deltas = torch.tensor([3., 7.])
    last_layer = len(current.blocks) - 1
    intervals = [(last_layer - 1, last_layer), (0, last_layer)]
    tails = [0, 2, raw[0].shape[1]]
    translators = {"kt_k2": translator}
    scores = evaluate_batch(
        parent, current, *raw, candidates, query_deltas,
        layer_intervals=intervals, tail_lengths=tails,
        translators=translators, candidate_chunk=5,
    )
    for values in scores.values():
        assert values.shape == candidates.shape
        assert values.device.type == "cpu"
        assert torch.isfinite(values).all()
    torch.testing.assert_close(scores[f"lr_0_{last_layer}"], scores["current_exact"])
    torch.testing.assert_close(scores[f"tr_{raw[0].shape[1]}"], scores["current_exact"])
    torch.testing.assert_close(scores["tr_0"], scores["reuse"], rtol=0, atol=0)

    records = path_records(
        intervals, tails, translators, history_length=raw[0].shape[1],
        num_layers=len(current.blocks),
    )
    rows = summarize_scores(
        {name: values.numpy() for name, values in scores.items()},
        "synthetic_ten_layer_edge", records,
    )
    assert {row["config_id"] for row in rows} == set(scores)
    for row in rows:
        assert np.isfinite(row["mean_abs_probability_gap"])
        assert np.isfinite(row["mean_Bernoulli_JS"])
