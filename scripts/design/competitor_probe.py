"""Independent executable competitors under Insight 1's functional metrics.

This adapter consumes explicit history batches and already-fitted translators.
It neither fits on evaluation users nor reads or changes sealed Insight 1 runs.
Single-edge outputs do not establish continuous cache-lifetime behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from insight_one_locality.adjudicate import (
    bernoulli_js,
    rank_correlation,
    sigmoid,
    top10_overlap,
)
from insight_one_locality.common import score_cache_chunked

from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.kv_translate import KVTranslator
from hstu_kvcache.baselines.tail_recompute import recompute_tail


@torch.no_grad()
def evaluate_batch(
    parent,
    current,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    candidates: torch.Tensor,
    query_deltas: torch.Tensor,
    *,
    layer_intervals: list[tuple[int, int]],
    tail_lengths: list[int],
    translators: dict[str, KVTranslator],
    candidate_chunk: int = 8,
) -> dict[str, torch.Tensor]:
    """Score independent migrations from one shared parent snapshot.

    Histories are unpadded, equal-length and causally aligned. Current Exact is
    an evaluation anchor only; its cache is never passed to a migration method.
    Every translator must already be fitted outside this evaluation batch.
    """
    if candidate_chunk < 1:
        raise ValueError("candidate_chunk must be positive")
    modes = parent.training, current.training
    parent.eval()
    current.eval()
    try:
        state = lr.capture_state(parent, item_ids, behaviors, time_deltas)

        def score(cache):
            return score_cache_chunked(current, cache, candidates, query_deltas, candidate_chunk)

        scores = {
            "reuse": score(state.cache),
            "current_exact": score(current.compute_kv(item_ids, behaviors, time_deltas)),
        }
        for start, end in layer_intervals:
            migrated = lr.recompute_interval(
                current, state, item_ids, behaviors, time_deltas, (start, end)
            )
            scores[f"lr_{start}_{end}"] = score(migrated.cache)
        for n in tail_lengths:
            migrated = recompute_tail(
                current, state.cache, item_ids, behaviors, time_deltas, n
            )
            scores[f"tr_{n}"] = score(migrated)
        for config_id, translator in translators.items():
            if config_id != f"kt_k{translator.source_layers.shape[1]}":
                raise ValueError("translator IDs must be kt_k{k} for their source-layer count")
            scores[config_id] = score(translator.apply(state.cache))
        return scores
    finally:
        parent.train(modes[0])
        current.train(modes[1])


def path_records(
    layer_intervals: list[tuple[int, int]],
    tail_lengths: list[int],
    translators: dict[str, KVTranslator],
    history_length: int,
    num_layers: int,
) -> list[dict[str, Any]]:
    """Describe paths; updated K/V fraction is coverage, never compute cost."""
    if history_length < 1 or num_layers < 1:
        raise ValueError("history_length and num_layers must be positive")
    records = [
        {"config_id": "reuse", "family": "anchors", "parameters": {}, "kv_updated_fraction": 0.0},
        {"config_id": "current_exact", "family": "anchors", "parameters": {}, "kv_updated_fraction": 1.0},
    ]
    for start, end in layer_intervals:
        if not 0 <= start <= end < num_layers:
            raise ValueError("layer intervals must be inclusive and within the model")
        records.append({
            "config_id": f"lr_{start}_{end}",
            "family": "layer_recompute",
            "parameters": {"interval": [start, end]},
            "kv_updated_fraction": (end - start + 1) / num_layers,
        })
    for n in tail_lengths:
        if n < 0:
            raise ValueError("tail lengths must be nonnegative")
        records.append({
            "config_id": f"tr_{n}",
            "family": "tail_recompute",
            "parameters": {"n": n, "effective_n": min(n, history_length)},
            "kv_updated_fraction": min(n, history_length) / history_length,
        })
    for config_id, translator in translators.items():
        k = translator.source_layers.shape[1]
        if config_id != f"kt_k{k}":
            raise ValueError("translator IDs must be kt_k{k} for their source-layer count")
        records.append({
            "config_id": config_id,
            "family": "kv_translate",
            "parameters": {
                "k": k,
                "source_layers": translator.source_layers.detach().cpu().tolist(),
                "selection_evaluation": translator.selection_evaluation,
            },
            "kv_updated_fraction": 1.0,
        })
    if len({record["config_id"] for record in records}) != len(records):
        raise ValueError("duplicate configurations would overwrite score paths")
    return records


def summarize_scores(
    scores: dict[str, np.ndarray], edge: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reuse Insight 1 mathematics without its sealed path/cost assumptions.

    Recovery is 1 - mean(abs(p_method-p_exact))/mean(abs(p_reuse-p_exact)),
    over all users and candidates. It is not a mean of per-user recoveries.
    Near-zero Reuse gaps retain every row and return an undefined recovery.
    """
    exact = np.asarray(scores["current_exact"], dtype=np.float64)
    reuse = np.asarray(scores["reuse"], dtype=np.float64)
    if exact.ndim != 2 or not exact.shape[0] or exact.shape[1] < 10:
        raise ValueError("scores must contain users with at least 10 candidates each")
    record_ids = [record["config_id"] for record in records]
    if len(set(record_ids)) != len(record_ids) or set(record_ids) != set(scores):
        raise ValueError("records must describe every score path exactly once")
    if any(np.asarray(values).shape != exact.shape for values in scores.values()):
        raise ValueError("all score paths must have the same [users,candidates] shape")
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("nonfinite logits cannot be silently excluded from metrics")

    exact_probability = sigmoid(exact)
    denominator = float(np.abs(sigmoid(reuse) - exact_probability).mean())
    near_zero = denominator <= 1e-12
    exact_top1 = exact.argmax(axis=1)
    rows = []
    for record in records:
        config_id = record["config_id"]
        observed = np.asarray(scores[config_id], dtype=np.float64)
        probability = sigmoid(observed)
        gap = float(np.abs(probability - exact_probability).mean())
        rows.append({
            "edge": edge,
            "config_id": config_id,
            "family": record["family"],
            "parameters": record["parameters"],
            "kv_updated_fraction": record["kv_updated_fraction"],
            "users": len(exact),
            "candidates_per_user": exact.shape[1],
            "mean_abs_probability_gap": gap,
            "reuse_mean_abs_probability_gap": denominator,
            "probability_gap_recovery": None if near_zero else float(1.0 - gap / denominator),
            "probability_gap_recovery_status": "near_zero_reuse_gap" if near_zero else "defined",
            "mean_abs_logit_gap": float(np.abs(observed - exact).mean()),
            "mean_Bernoulli_JS": float(bernoulli_js(probability, exact_probability).mean()),
            "top1_agreement": float((observed.argmax(axis=1) == exact_top1).mean()),
            "top10_overlap": float(top10_overlap(exact, observed).mean()),
            "rank_correlation": float(rank_correlation(exact, observed).mean()),
        })
    return rows
