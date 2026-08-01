from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

from ..models import HSTUKVCache
from ..streaming.sharded_edge import ExternalEmbeddingHSTU
from ..streaming.trainer import build_next_item_targets
from ..streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
)
from .stage45_oldkv import DirectOldKVProgram
from .xp_exact_baseline import XPBaselineRecord, canonical_sha256

PROTOCOL = "evokv_xp_d1_quality_development_v1"
SUFFIX_DIAGNOSTIC_PROTOCOL = (
    "evokv_xp_reuse_exact_suffix_diagnostic_development_v0"
)
ACTION_PLAN_PROTOCOL = "evokv_xp_d1_action_plan_v2_development_v0"
METHODS = (
    "all_reuse",
    "compiled_direct_oldkv",
    "mixed_fixed20",
    "all_exact",
)
REUSE_EXACT_METHODS = ("all_reuse", "all_exact")

T = TypeVar("T")


@dataclass(frozen=True)
class TimedValue:
    value: Any
    median_milliseconds: float
    samples_milliseconds: tuple[float, ...]


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode()
    )
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def direct_program_sha256(program: DirectOldKVProgram) -> str:
    return canonical_sha256(
        {
            "source_version": program.source_version,
            "target_version": program.target_version,
            "weights_sha256": tensor_sha256(program.weights),
            "biases_sha256": tensor_sha256(program.biases),
            "weights_shape": list(program.weights.shape),
            "biases_shape": list(program.biases.shape),
            "dtype": str(program.weights.dtype),
        }
    )


def timed_call(
    function: Callable[[], T],
    device: torch.device,
    repeats: int,
) -> TimedValue:
    if repeats < 1:
        raise ValueError("timing repeats must be positive")
    value = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            value = function()
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)))
        else:
            started = time.perf_counter()
            value = function()
            samples.append((time.perf_counter() - started) * 1000.0)
    return TimedValue(
        value=value,
        median_milliseconds=float(statistics.median(samples)),
        samples_milliseconds=tuple(samples),
    )


@torch.no_grad()
def apply_direct_oldkv(
    program: DirectOldKVProgram,
    source: HSTUKVCache,
) -> HSTUKVCache:
    if (
        source.k.shape != source.v.shape
        or source.k.ndim != 4
        or source.k.shape[0] != program.num_layers
        or source.k.shape[-1] != program.kv_width
        or source.k.device != program.device
    ):
        raise ValueError("XP direct old-K/V cache signature differs")
    joined = torch.cat(
        (
            source.k.to(program.weights.dtype),
            source.v.to(program.weights.dtype),
        ),
        dim=-1,
    )
    shape = joined.shape
    flattened = joined.reshape(shape[0], -1, shape[-1])
    projected = torch.baddbmm(
        program.biases[:, None, :].expand(
            shape[0], flattened.shape[1], shape[-1]
        ),
        flattened,
        program.weights,
    ).reshape(shape)
    return HSTUKVCache(
        k=projected[..., : program.kv_width].contiguous(),
        v=projected[..., program.kv_width :].contiguous(),
        seq_len=source.seq_len,
    )


def cache_relative_error(
    cache: HSTUKVCache,
    exact: HSTUKVCache,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if (
        cache.k.shape != exact.k.shape
        or cache.v.shape != exact.v.shape
        or cache.k.ndim != 4
        or lengths.shape != (cache.k.shape[1],)
    ):
        raise ValueError("XP cache fidelity shapes differ")
    valid = (
        torch.arange(cache.k.shape[2], device=cache.k.device)[None, :]
        < lengths.to(cache.k.device)[:, None]
    )
    mask = valid[None, :, :, None]
    delta = (
        ((cache.k.float() - exact.k.float()) * mask).square().sum((0, 2, 3))
        + ((cache.v.float() - exact.v.float()) * mask).square().sum((0, 2, 3))
    )
    scale = (
        (exact.k.float() * mask).square().sum((0, 2, 3))
        + (exact.v.float() * mask).square().sum((0, 2, 3))
    )
    return delta.sqrt() / scale.sqrt().clamp_min(1e-12)


def _topk_overlap(
    scores: torch.Tensor,
    exact_scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    k = min(topk, scores.shape[1])
    selected = torch.topk(scores, k, dim=1).indices
    exact_selected = torch.topk(exact_scores, k, dim=1).indices
    return (
        (selected[:, :, None] == exact_selected[:, None, :])
        .any(dim=2)
        .float()
        .mean(dim=1)
    )


def recommendation_sums(
    scores: torch.Tensor,
    exact_scores: torch.Tensor,
) -> dict[str, float | int]:
    if (
        scores.ndim != 2
        or scores.shape != exact_scores.shape
        or scores.shape[1] < 2
    ):
        raise ValueError("XP recommendation score shapes differ")
    if scores.shape[0] == 0:
        return {
            "positive_targets": 0,
            "sampled_cross_entropy_sum": 0.0,
            "hit_at_10_sum": 0.0,
            "ndcg_at_10_sum": 0.0,
            "reciprocal_rank_sum": 0.0,
            "score_cosine_sum": 0.0,
            "top10_overlap_sum": 0.0,
            "top1_agreement_sum": 0.0,
        }
    positive = scores[:, :1]
    ranks = 1 + (scores[:, 1:] >= positive).sum(dim=1)
    hit = ranks <= 10
    ndcg = torch.where(
        hit,
        torch.reciprocal(torch.log2(ranks.double() + 1.0)),
        torch.zeros_like(ranks, dtype=torch.float64),
    )
    loss = torch.nn.functional.cross_entropy(
        scores,
        torch.zeros(scores.shape[0], dtype=torch.int64, device=scores.device),
        reduction="sum",
    )
    score_cosine = torch.nn.functional.cosine_similarity(
        scores.float(), exact_scores.float(), dim=1
    )
    return {
        "positive_targets": scores.shape[0],
        "sampled_cross_entropy_sum": float(loss.double().item()),
        "hit_at_10_sum": float(hit.double().sum().item()),
        "ndcg_at_10_sum": float(ndcg.sum().item()),
        "reciprocal_rank_sum": float(
            torch.reciprocal(ranks.double()).sum().item()
        ),
        "score_cosine_sum": float(score_cosine.double().sum().item()),
        "top10_overlap_sum": float(
            _topk_overlap(scores, exact_scores, 10).double().sum().item()
        ),
        "top1_agreement_sum": float(
            (scores.argmax(1) == exact_scores.argmax(1)).double().sum().item()
        ),
    }


def summarize_recommendation_sums(
    values: Mapping[str, float | int],
) -> dict[str, float | int]:
    count = int(values["positive_targets"])
    if count < 1:
        raise ValueError("XP recommendation summary has no targets")
    return {
        "positive_targets": count,
        "sampled_cross_entropy": float(
            values["sampled_cross_entropy_sum"]
        )
        / count,
        "hit_rate_at_10": float(values["hit_at_10_sum"]) / count,
        "ndcg_at_10": float(values["ndcg_at_10_sum"]) / count,
        "mean_reciprocal_rank": float(values["reciprocal_rank_sum"]) / count,
        "score_cosine_to_exact": float(values["score_cosine_sum"]) / count,
        "top10_overlap_with_exact": float(values["top10_overlap_sum"]) / count,
        "top1_agreement_with_exact": float(values["top1_agreement_sum"]) / count,
    }


def summarize_optional_recommendation_sums(
    values: Mapping[str, float | int],
) -> dict[str, float | int | None]:
    if int(values["positive_targets"]) > 0:
        return summarize_recommendation_sums(values)
    return {
        "positive_targets": 0,
        "sampled_cross_entropy": None,
        "hit_rate_at_10": None,
        "ndcg_at_10": None,
        "mean_reciprocal_rank": None,
        "score_cosine_to_exact": None,
        "top10_overlap_with_exact": None,
        "top1_agreement_with_exact": None,
    }


def _float_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.float(),
        v=cache.v.float(),
        seq_len=cache.seq_len,
    )


def cache_storage_roundtrip(
    cache: HSTUKVCache,
    dtype: torch.dtype,
) -> HSTUKVCache:
    if dtype != torch.float16:
        raise ValueError("XP diagnostic cache storage dtype differs")
    return HSTUKVCache(
        k=cache.k.to(dtype=dtype),
        v=cache.v.to(dtype=dtype),
        seq_len=cache.seq_len,
    )


@torch.no_grad()
def evaluate_quality_batch(
    dense: ExternalEmbeddingHSTU,
    embedding: TrainableProjectedModuloEmbedding,
    batch: Mapping[str, torch.Tensor],
    candidates: torch.Tensor,
    old_cache: HSTUKVCache,
    program: DirectOldKVProgram | None,
    history_end: int,
    device: torch.device,
    timing_repeats: int,
    mixed_exact_record_ids: set[int] | frozenset[int],
    methods: Sequence[str] = METHODS,
    suffix_offset_breakdown: bool = False,
    common_cache_storage_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    selected_methods = tuple(methods)
    if (
        not selected_methods
        or len(selected_methods) != len(set(selected_methods))
        or any(name not in METHODS for name in selected_methods)
        or "all_exact" not in selected_methods
        or (
            suffix_offset_breakdown
            and not set(REUSE_EXACT_METHODS).issubset(selected_methods)
        )
        or (
            program is None
            and any(
                name in {"compiled_direct_oldkv", "mixed_fixed20"}
                for name in selected_methods
            )
        )
        or common_cache_storage_dtype not in {None, torch.float16}
    ):
        raise ValueError("XP quality method selection differs")
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)
    labels = batch["labels"].to(device)
    train_mask = batch["train_mask"].to(device)
    lengths = batch["lengths"].to(device)
    records = batch["record_indices"].to(device)
    if not 2 <= history_end < item_ids.shape[1]:
        raise ValueError("XP quality history boundary differs")
    prefix_width = history_end - 1
    real = records >= 0
    prefix_lengths = torch.where(
        real,
        torch.full_like(lengths, prefix_width),
        torch.zeros_like(lengths),
    )
    prefix_ids = item_ids[:, :prefix_width]
    prefix_behaviors = behaviors[:, :prefix_width]
    prefix_deltas = time_deltas[:, :prefix_width]

    def exact_cache_call() -> HSTUKVCache:
        vectors = embedding(prefix_ids, prefix_lengths)
        return dense.core.compute_kv_from_item_embeddings(
            vectors,
            prefix_behaviors,
            prefix_deltas,
            prefix_lengths,
        )

    exact_timed = timed_call(exact_cache_call, device, timing_repeats)
    exact_cache = (
        exact_timed.value
        if common_cache_storage_dtype is None
        else cache_storage_roundtrip(
            exact_timed.value,
            common_cache_storage_dtype,
        )
    )
    old_device = old_cache.to(device)
    if common_cache_storage_dtype is not None and (
        old_device.k.dtype != common_cache_storage_dtype
        or old_device.v.dtype != common_cache_storage_dtype
    ):
        raise ValueError("XP reuse cache storage dtype differs")
    compiled_timed = None
    compiled_cache = None
    if any(
        name in {"compiled_direct_oldkv", "mixed_fixed20"}
        for name in selected_methods
    ):
        compiled_timed = timed_call(
            lambda: apply_direct_oldkv(program, old_device),
            device,
            timing_repeats,
        )
        compiled_cache = compiled_timed.value
    mixed_exact = torch.tensor(
        [
            int(value) in mixed_exact_record_ids
            for value in records.detach().cpu().tolist()
        ],
        dtype=torch.bool,
        device=device,
    )
    mixed_cache = None
    if "mixed_fixed20" in selected_methods:
        if (
            exact_cache.k.dtype != compiled_cache.k.dtype
            or exact_cache.v.dtype != compiled_cache.v.dtype
        ):
            raise ValueError("XP mixed cache storage dtype differs")
        mixed_mask = mixed_exact[None, :, None, None]
        mixed_cache = HSTUKVCache(
            k=torch.where(
                mixed_mask,
                exact_cache.k,
                compiled_cache.k,
            ).contiguous(),
            v=torch.where(
                mixed_mask,
                exact_cache.v,
                compiled_cache.v,
            ).contiguous(),
            seq_len=exact_cache.seq_len,
        )
    available_caches = {
        "all_reuse": old_device,
        "compiled_direct_oldkv": compiled_cache,
        "mixed_fixed20": mixed_cache,
        "all_exact": exact_cache,
    }
    caches = {name: available_caches[name] for name in selected_methods}
    suffix_ids = item_ids[:, prefix_width:-1]
    suffix_behaviors = behaviors[:, prefix_width:-1]
    suffix_deltas = time_deltas[:, prefix_width:-1]
    suffix_lengths = torch.where(
        real,
        torch.full_like(lengths, suffix_ids.shape[1]),
        torch.zeros_like(lengths),
    )
    suffix_vectors = embedding(suffix_ids, suffix_lengths)
    targets, valid = build_next_item_targets(
        item_ids,
        lengths,
        labels,
        train_mask,
    )
    suffix_valid = valid[:, prefix_width:]
    suffix_offsets = (
        torch.arange(
            1,
            suffix_valid.shape[1] + 1,
            dtype=torch.int64,
            device=device,
        )[None, :]
        .expand_as(suffix_valid)[suffix_valid]
    )
    target_record_ids = records[:, None].expand_as(suffix_valid)[suffix_valid]
    positive_count = int(suffix_valid.sum().item())
    candidates = candidates.to(device)
    all_positive_count = int(valid.sum().item())
    if candidates.shape[0] == all_positive_count:
        suffix_positions = (
            torch.arange(valid.shape[1], device=device)[None, :]
            >= prefix_width
        ).expand_as(valid)
        candidates = candidates[suffix_positions[valid]]
    elif candidates.shape[0] != positive_count:
        raise ValueError("XP fixed candidates differ from quality targets")
    if candidates.shape[0] != positive_count:
        raise ValueError("XP fixed candidate suffix selection differs")
    candidate_lengths = torch.full(
        (positive_count,),
        candidates.shape[1],
        dtype=torch.int64,
        device=device,
    )
    candidate_vectors = embedding(candidates, candidate_lengths)

    def scores_for(cache: HSTUKVCache) -> torch.Tensor:
        hidden, _ = dense.core.forward_with_cache_from_item_embeddings(
            _float_cache(cache),
            suffix_vectors,
            suffix_behaviors,
            suffix_deltas,
        )
        return torch.einsum(
            "nh,nch->nc",
            hidden[suffix_valid],
            candidate_vectors,
        )

    score_timings = {
        name: timed_call(
            lambda cache=cache: scores_for(cache),
            device,
            timing_repeats,
        )
        for name, cache in caches.items()
    }
    exact_scores = score_timings["all_exact"].value
    cache_errors = {
        name: cache_relative_error(cache, exact_cache, prefix_lengths)
        .detach()
        .cpu()[real.cpu()]
        .tolist()
        for name, cache in caches.items()
    }
    maintenance_milliseconds = {
        "all_reuse": 0.0,
        "all_exact": exact_timed.median_milliseconds,
    }
    if compiled_timed is not None:
        maintenance_milliseconds["compiled_direct_oldkv"] = (
            compiled_timed.median_milliseconds
        )
    if mixed_cache is not None:
        real_mixed_exact = mixed_exact & real
        if bool(real_mixed_exact.any()) and bool((real & ~mixed_exact).any()):
            mixed_maintenance = (
                exact_timed.median_milliseconds
                + compiled_timed.median_milliseconds
            )
        elif bool(real_mixed_exact.any()):
            mixed_maintenance = exact_timed.median_milliseconds
        else:
            mixed_maintenance = compiled_timed.median_milliseconds
        maintenance_milliseconds["mixed_fixed20"] = mixed_maintenance
    method_reports = {}
    for name, timing in score_timings.items():
        method_report = {
            "recommendation_sums": recommendation_sums(
                timing.value,
                exact_scores,
            ),
            "cache_error_rel": cache_errors[name],
            "maintenance_milliseconds": maintenance_milliseconds[name],
            "maintenance_cost_kind": (
                "zero_reuse_component"
                if name == "all_reuse"
                else "measured_full_batch_component"
                if name != "mixed_fixed20"
                else (
                    "component_bound_selected_route_batch_medians_"
                    "not_end_to_end"
                )
            ),
            "online_suffix_and_score_milliseconds": (
                timing.median_milliseconds
            ),
        }
        if suffix_offset_breakdown:
            method_report["recommendation_sums_by_suffix_offset"] = {
                str(offset): recommendation_sums(
                    timing.value[suffix_offsets == offset],
                    exact_scores[suffix_offsets == offset],
                )
                for offset in range(1, suffix_valid.shape[1] + 1)
            }
        method_reports[name] = method_report
    report = {
        "record_ids": records[real].detach().cpu().tolist(),
        "prefix_tokens_per_real_record": prefix_width,
        "positive_targets": positive_count,
        "methods": method_reports,
        "timing_boundary": {
            "maintenance": (
                "prefix cache reuse, direct-old-K/V affine, or exact current "
                "prefix recomputation"
            ),
            "online": (
                "common current-model suffix consumption and frozen-candidate "
                "dot-product scoring"
            ),
            "warmup_calls_per_measured_batch": 1,
            "measured_repetitions": timing_repeats,
            "clock": "cuda_event" if device.type == "cuda" else "perf_counter",
            "mixed_cost": (
                "component-bound selected exact/compiled batch medians; not an "
                "end-to-end mixed-route runtime"
            ),
        },
    }
    if suffix_offset_breakdown:
        paired_methods = {}
        for name in REUSE_EXACT_METHODS:
            scores = score_timings[name].value
            positive = scores[:, :1]
            ranks = 1 + (scores[:, 1:] >= positive).sum(dim=1)
            cross_entropy = torch.nn.functional.cross_entropy(
                scores,
                torch.zeros(
                    scores.shape[0],
                    dtype=torch.int64,
                    device=scores.device,
                ),
                reduction="none",
            )
            paired_methods[name] = {
                "ranks": ranks.detach().cpu().tolist(),
                "sampled_cross_entropy": (
                    cross_entropy.detach().cpu().double().tolist()
                ),
            }
        report["paired_target_contributions"] = {
            "record_ids": target_record_ids.detach().cpu().tolist(),
            "suffix_offsets": suffix_offsets.detach().cpu().tolist(),
            **paired_methods,
        }
    if common_cache_storage_dtype is not None:
        report["common_cache_storage"] = {
            "storage_dtype": str(common_cache_storage_dtype),
            "consumption_dtype": str(torch.float32),
            "methods": list(REUSE_EXACT_METHODS),
        }
    return report


def merge_batch_reports(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not reports:
        raise ValueError("XP quality reports are empty")
    method_names = tuple(reports[0]["methods"])
    if not method_names or any(
        tuple(report["methods"]) != method_names for report in reports
    ):
        raise ValueError("XP quality report methods differ")
    offset_names = {
        name: tuple(
            reports[0]["methods"][name]
            .get("recommendation_sums_by_suffix_offset", {})
        )
        for name in method_names
    }
    merged: dict[str, dict[str, Any]] = {
        name: {
            "recommendation_sums": defaultdict(float),
            "recommendation_sums_by_suffix_offset": {
                offset: defaultdict(float)
                for offset in offset_names[name]
            },
            "cache_error_rel": [],
            "maintenance_milliseconds": 0.0,
            "online_suffix_and_score_milliseconds": 0.0,
            "maintenance_cost_kinds": set(),
        }
        for name in method_names
    }
    records = []
    paired_rows = []
    for report in reports:
        records.extend(int(value) for value in report["record_ids"])
        methods = report["methods"]
        for name in method_names:
            current = methods[name]
            for key, value in current["recommendation_sums"].items():
                merged[name]["recommendation_sums"][key] += value
            current_offsets = current.get(
                "recommendation_sums_by_suffix_offset",
                {},
            )
            if tuple(current_offsets) != offset_names[name]:
                raise ValueError("XP quality suffix offsets differ")
            for offset, sums in current_offsets.items():
                for key, value in sums.items():
                    merged[name]["recommendation_sums_by_suffix_offset"][
                        offset
                    ][key] += value
            merged[name]["cache_error_rel"].extend(
                current["cache_error_rel"]
            )
            merged[name]["maintenance_milliseconds"] += float(
                current["maintenance_milliseconds"]
            )
            merged[name]["online_suffix_and_score_milliseconds"] += float(
                current["online_suffix_and_score_milliseconds"]
            )
            merged[name]["maintenance_cost_kinds"].add(
                str(current["maintenance_cost_kind"])
            )
        paired = report.get("paired_target_contributions")
        if paired is not None:
            lengths = {
                len(paired["record_ids"]),
                len(paired["suffix_offsets"]),
                *(
                    len(paired[name][field])
                    for name in REUSE_EXACT_METHODS
                    for field in ("ranks", "sampled_cross_entropy")
                ),
            }
            if len(lengths) != 1:
                raise ValueError("XP paired target contribution lengths differ")
            paired_rows.extend(
                (
                    int(paired["record_ids"][index]),
                    int(paired["suffix_offsets"][index]),
                    int(paired["all_reuse"]["ranks"][index]),
                    float(
                        paired["all_reuse"]["sampled_cross_entropy"][index]
                    ),
                    int(paired["all_exact"]["ranks"][index]),
                    float(
                        paired["all_exact"]["sampled_cross_entropy"][index]
                    ),
                )
                for index in range(len(paired["record_ids"]))
            )
    if len(records) != len(set(records)):
        raise ValueError("XP quality record coverage overlaps")
    output = {}
    for name, value in merged.items():
        errors = value["cache_error_rel"]
        output[name] = {
            "recommendation": summarize_recommendation_sums(
                value["recommendation_sums"]
            ),
            "cache_fidelity": {
                "records": len(errors),
                "relative_error_mean": float(statistics.fmean(errors)),
                "relative_error_max": float(max(errors)),
            },
            "gpu_cost": {
                "maintenance_milliseconds_sum_of_batch_medians": value[
                    "maintenance_milliseconds"
                ],
                "online_suffix_and_score_milliseconds_sum_of_batch_medians": value[
                    "online_suffix_and_score_milliseconds"
                ],
                "maintenance_cost_kind": sorted(
                    value["maintenance_cost_kinds"]
                ),
            },
        }
        if offset_names[name]:
            output[name]["recommendation_by_suffix_offset"] = {
                offset: summarize_optional_recommendation_sums(sums)
                for offset, sums in value[
                    "recommendation_sums_by_suffix_offset"
                ].items()
            }
    result = {
        "record_ids": sorted(records),
        "record_ids_sha256": canonical_sha256(
            {"record_ids": sorted(records)}
        ),
        "records": len(records),
        "methods": output,
    }
    if paired_rows:
        paired_rows.sort(key=lambda value: (value[0], value[1]))
        pair_keys = [
            {"record_id": value[0], "suffix_offset": value[1]}
            for value in paired_rows
        ]
        if len(pair_keys) != len(
            {(value["record_id"], value["suffix_offset"]) for value in pair_keys}
        ):
            raise ValueError("XP paired target contribution keys overlap")
        result["paired_target_contributions"] = {
            "targets": len(paired_rows),
            "pair_key_sha256": canonical_sha256({"pairs": pair_keys}),
            "record_ids": [value[0] for value in paired_rows],
            "suffix_offsets": [value[1] for value in paired_rows],
            "all_reuse": {
                "ranks": [value[2] for value in paired_rows],
                "sampled_cross_entropy": [value[3] for value in paired_rows],
            },
            "all_exact": {
                "ranks": [value[4] for value in paired_rows],
                "sampled_cross_entropy": [value[5] for value in paired_rows],
            },
        }
    return result


def _extent(record: XPBaselineRecord) -> dict[str, int]:
    old_end = record.old_start + record.old_length
    target_end = record.target_start + record.target_length
    retained_start = max(record.old_start, record.target_start)
    retained_end = min(old_end, target_end)
    retained = max(0, retained_end - retained_start)
    return {
        "old_start": record.old_start,
        "old_tokens": record.old_length,
        "target_start": record.target_start,
        "target_tokens": record.target_length,
        "retained_tokens": retained,
        "evicted_tokens": record.old_length - retained,
        "append_tokens": record.target_length - retained,
        "retained_offset_in_old": retained_start - record.old_start,
        "retained_offset_in_target": retained_start - record.target_start,
    }


def _stratum(record: XPBaselineRecord) -> str:
    extent = _extent(record)
    target_bin = min((record.target_length - 1) // 128, 3)
    retained_bin = min(
        3,
        4 * extent["retained_tokens"] // max(record.target_length, 1),
    )
    return f"owner{record.owner_rank}:target{target_bin}:retained{retained_bin}"


def select_token_budget(
    records: Sequence[tuple[int, int, str]],
    *,
    selection_salt: str,
    numerator: int = 1,
    denominator: int = 5,
) -> tuple[set[int], dict[str, object]]:
    if (
        not records
        or not selection_salt
        or numerator < 0
        or denominator < 1
        or numerator > denominator
    ):
        raise ValueError("XP token-budget selection differs")
    ids = [int(record_id) for record_id, _, _ in records]
    if (
        len(ids) != len(set(ids))
        or any(
            int(tokens) < 1 or not str(stratum)
            for _, tokens, stratum in records
        )
    ):
        raise ValueError("XP token-budget records differ")
    strata: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record_id, tokens, stratum in records:
        strata[str(stratum)].append((int(record_id), int(tokens)))
    exact_ids: set[int] = set()
    stratum_ledger = []
    for name in sorted(strata):
        ordered = sorted(
            strata[name],
            key=lambda value: (
                hashlib.sha256(
                    f"{selection_salt}:{name}:{value[0]}".encode()
                ).digest(),
                value[0],
            ),
        )
        cumulative = [0]
        for _, tokens in ordered:
            cumulative.append(cumulative[-1] + tokens)
        total = cumulative[-1]
        selected_count = min(
            range(len(cumulative)),
            key=lambda index: (
                abs(cumulative[index] * denominator - total * numerator),
                cumulative[index] * denominator > total * numerator,
                index,
            ),
        )
        selected = ordered[:selected_count]
        selected_tokens = cumulative[selected_count]
        exact_ids.update(record_id for record_id, _ in selected)
        stratum_ledger.append(
            {
                "stratum": name,
                "records": len(ordered),
                "retained_tokens": total,
                "target_exact_retained_tokens": (
                    total * numerator / denominator
                ),
                "exact_records": len(selected),
                "exact_retained_tokens": selected_tokens,
                "actual_record_fraction": len(selected) / len(ordered),
                "actual_retained_token_fraction": selected_tokens / total,
            }
        )
    total_tokens = sum(int(tokens) for _, tokens, _ in records)
    exact_tokens = sum(
        int(tokens)
        for record_id, tokens, _ in records
        if int(record_id) in exact_ids
    )
    ledger: dict[str, object] = {
        "policy": "label_free_extent_strata_then_stable_hash_token_prefix",
        "quality_labels_read": False,
        "per_record_exact_error_read": False,
        "budget_basis": "retained_tokens",
        "target_fraction": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "total_records": len(records),
        "exact_records": len(exact_ids),
        "compiled_records": len(records) - len(exact_ids),
        "total_retained_tokens": total_tokens,
        "target_exact_retained_tokens": (
            total_tokens * numerator / denominator
        ),
        "actual_exact_retained_tokens": exact_tokens,
        "actual_record_fraction": len(exact_ids) / len(records),
        "actual_retained_token_fraction": exact_tokens / total_tokens,
        "selection_salt": selection_salt,
        "strata": stratum_ledger,
    }
    return exact_ids, ledger


def build_action_plan_v2(
    records: Sequence[XPBaselineRecord],
    *,
    benchmark_id: str,
    source_version: int,
    target_version: int,
    program_sha256: str,
    source_checkpoint_sha256: str,
    target_checkpoint_sha256: str,
    workload_sha256: str,
    split_sha256: str,
    selection_salt: str,
) -> dict[str, object]:
    if (
        not records
        or target_version != source_version + 1
        or any(_extent(record)["append_tokens"] != 32 for record in records)
    ):
        raise ValueError("XP ActionPlan v2 requires an adjacent edge and append=32")
    exact_ids, selection = select_token_budget(
        [
            (
                record.record_id,
                _extent(record)["retained_tokens"],
                _stratum(record),
            )
            for record in records
        ],
        selection_salt=(
            f"{selection_salt}:{workload_sha256}:theta{source_version}:"
            f"theta{target_version}"
        ),
    )
    plan_records = []
    for record in sorted(records, key=lambda value: value.record_id):
        plan_records.append(
            {
                "record_id": record.record_id,
                "user_id": record.user_id,
                "owner_rank": record.owner_rank,
                "action": (
                    "exact" if record.record_id in exact_ids else "compiled"
                ),
                **_extent(record),
            }
        )
    plan_hash = canonical_sha256({"records": plan_records})
    return {
        "protocol": ACTION_PLAN_PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "benchmark_id": benchmark_id,
        "source_version": source_version,
        "target_version": target_version,
        "actions": ["compiled", "exact"],
        "selection": selection,
        "bindings": {
            "program_sha256": program_sha256,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "target_checkpoint_sha256": target_checkpoint_sha256,
            "workload_sha256": workload_sha256,
            "split_sha256": split_sha256,
        },
        "extent_contract": {
            "layout": "natural_variable_length_het",
            "append_tokens": 32,
            "capacity_bytes": "valid_kv_bytes",
        },
        "records": plan_records,
        "record_count": len(plan_records),
        "records_sha256": plan_hash,
    }


def split_batches(
    batches: Sequence[Mapping[str, torch.Tensor]],
    fit_batches: int,
    probe_batches: int,
) -> tuple[
    tuple[Mapping[str, torch.Tensor], ...],
    tuple[Mapping[str, torch.Tensor], ...],
]:
    if fit_batches < 1 or probe_batches < 1 or fit_batches + probe_batches > len(batches):
        raise ValueError("XP D1 fit/probe batch split differs")
    fit = tuple(batches[:fit_batches])
    probe = tuple(batches[fit_batches : fit_batches + probe_batches])
    fit_ids = {
        int(value)
        for batch in fit
        for value in batch["record_indices"].tolist()
        if int(value) >= 0
    }
    probe_ids = {
        int(value)
        for batch in probe
        for value in batch["record_indices"].tolist()
        if int(value) >= 0
    }
    if not fit_ids or not probe_ids or fit_ids & probe_ids:
        raise ValueError("XP D1 fit/probe records overlap")
    return fit, probe


def split_identity(
    fit: Sequence[Mapping[str, torch.Tensor]],
    probe: Sequence[Mapping[str, torch.Tensor]],
    qualification: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, object]:
    def ids(values: Sequence[Mapping[str, torch.Tensor]]) -> list[int]:
        return sorted(
            int(record)
            for batch in values
            for record in batch["record_indices"].tolist()
            if int(record) >= 0
        )

    roles = {
        "fit": ids(fit),
        "probe": ids(probe),
        "qualification_test": ids(qualification),
    }
    sets = [set(values) for values in roles.values()]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("XP D1 evaluation roles overlap")
    payload = {
        name: {
            "records": len(values),
            "record_ids_sha256": canonical_sha256({"record_ids": values}),
        }
        for name, values in roles.items()
    }
    return {"roles": payload, "sha256": canonical_sha256(payload)}
