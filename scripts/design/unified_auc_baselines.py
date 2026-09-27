"""Calibration-only wrappers for the executable release-snapshot baselines.

The caller owns UID selection, unpadded history grouping and teacher generation.
Layer profiling uses teacher logits, never feedback labels; translation shares
one source-layer ranking across its declared k budgets.
"""

from __future__ import annotations

import torch

from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.kv_translate import (
    KVTranslator,
    fit_affine_ridge,
    select_source_layers,
)
from hstu_kvcache.baselines.kv_translate.core import _features
from insight_one_locality.common import score_cache_chunked


@torch.no_grad()
def calibrate_layer_intervals(
    current, parent_state, events, candidates, query_deltas,
    max_profile_users=32, *, exact_cache=None, exact_logits=None,
    candidate_chunk=8,
):
    """Choose one real contiguous replay per layer count on calibration users.

    Supply exactly one calibration teacher: its Current cache or its logits on
    these same candidates. ``events`` is the aligned (items, behaviors, deltas)
    tuple. All 21 intervals of a six-layer model start independently from the
    supplied Parent state, including its actual boundary inputs. The returned
    tuple is (intervals_by_count, all_records, statistics). Ties use (start,end).
    """
    users = parent_state.cache.k.shape[1]
    if not 0 < users <= max_profile_users:
        raise ValueError("caller must pass only the selected profiling users")
    if (exact_cache is None) == (exact_logits is None):
        raise ValueError("provide exactly one calibration teacher cache or logits")
    if candidates.shape[0] != users or candidate_chunk < 1:
        raise ValueError("candidates must match profiling users and chunk be positive")
    was_training = current.training
    current.eval()
    try:
        if exact_logits is None:
            exact_logits = score_cache_chunked(
                current, exact_cache, candidates, query_deltas, candidate_chunk,
            )
        if exact_logits.shape != candidates.shape:
            raise ValueError("teacher logits must match the calibration candidates")
        teacher = exact_logits.to(device=candidates.device, dtype=torch.float64)
        records = []
        for interval in lr.enumerate_intervals(len(current.blocks), include_reuse=False):
            migrated = lr.recompute_interval(current, parent_state, *events, interval)
            logits = score_cache_chunked(
                current, migrated.cache, candidates, query_deltas, candidate_chunk,
            )
            error = float((logits.double() - teacher).square().mean())
            if not torch.isfinite(torch.tensor(error)):
                raise ValueError("nonfinite profiling error cannot select an interval")
            records.append(dict(
                interval=list(interval), recomputed_layers=interval[1]-interval[0]+1,
                mean_squared_logit_error=error,
            ))
            del migrated, logits
        choices = {}
        for count in range(1, len(current.blocks)+1):
            chosen = min(
                (r for r in records if r["recomputed_layers"] == count),
                key=lambda r: (r["mean_squared_logit_error"], *r["interval"]),
            )
            choices[count] = tuple(chosen["interval"])
        for record in records:
            record["selected"] = tuple(record["interval"]) == choices[record["recomputed_layers"]]
        return choices, records, dict(
            profiling_users=users, candidates_per_user=candidates.shape[1],
            evaluated_intervals=len(records), objective="mean_squared_logit_error",
            labels_used=False, teacher_scope="caller_supplied_calibration_only",
            interval_entry="actual_parent_boundary_input_or_current_embedding",
            tie_break="lexicographic_start_end",
        )
    finally:
        current.train(was_training)


@torch.no_grad()
def fit_translators(source, target, num_heads, ks=(1, 2, 3, 4), ridge=.01):
    """Reuse the existing in-sample layer selection and centered affine ridge.

    K and V have separate shared maps. Each map reads every head from each
    selected source layer, exactly as ``kv_translate.fit``. All rows must be
    valid, aligned calibration tokens; no evaluation cache enters this helper.
    Device and dtype follow the supplied caches and the existing fitter.
    """
    ks = tuple(ks)
    if not ks or len(set(ks)) != len(ks) or min(ks) < 1:
        raise ValueError("ks must contain distinct positive source-layer counts")
    ranking, scores, evaluation = select_source_layers(
        source, target, num_heads, max(ks),
    )
    fitted = {}
    for k in ks:
        selected = ranking[:, :k].clone()
        parameters = []
        for source_values, target_values in ((source.k, target.k), (source.v, target.v)):
            weights, biases = [], []
            for layer, source_layers in enumerate(selected):
                x = _features(source_values, source_layers).flatten(0, 1)
                y = target_values[layer].flatten(0, 1)
                weight, bias = fit_affine_ridge(x, y, ridge)
                weights.append(weight)
                biases.append(bias)
            parameters.extend((torch.stack(weights), torch.stack(biases)))
        fitted[k] = KVTranslator(selected, *parameters, scores, evaluation)
    return fitted
