from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ..models import HSTU, HSTUKVCache
from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_next_item_chain import _effective_document
from .kuairand_root_cause import (
    PROTOCOL,
    _atomic_json,
    _evaluation_sequence,
    _load_checkpoint,
    _selected_users,
    _stored_cache,
    file_sha256,
    load_plan,
    make_model,
)
from .qk_protocol_sweep_runner import (
    METRICS,
    nested_popular_candidate_ids,
    nested_uniform_candidate_ids,
)
from .qk_stream_version import cache_relative_error

ROLLOUT_PROTOCOL = "evokv_kuairand_next_item_recursive_rollout_v0"
METHODS = (
    "lag1_monolithic",
    "lag5_monolithic",
    "recursive_rollout",
    "fresh_recompute",
)
HORIZONS = (1, 4, 16, None)


def _horizon_name(value: int | None) -> str:
    return "all" if value is None else f"first_{value}"


def load_next_item_rollout_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    bindings = document.get("checkpoint_bindings", [])
    quality = document.get("quality", {})
    current_version = int(document.get("current_version", -1))
    update_date_indices = list(range(16, current_version + 14))
    binding_versions = list(range(2, current_version + 1))
    if (
        document.get("protocol") != ROLLOUT_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("source_version") != 2
        or current_version not in (7, 8)
        or document.get("total_num_days") != current_version + 15
        or document.get("update_date_indices") != update_date_indices
        or document.get("evaluation_date_index") != current_version + 14
        or document.get("evaluation_history_window_days") != 7
        or document.get("methods") != list(METHODS)
        or quality.get("negative_counts") != [49, 99, 199]
        or quality.get("preferred_negative_count") != 99
        or quality.get("uniform_candidate_seeds") != [61031, 14929, 29063]
        or quality.get("horizons") != [1, 4, 16, "all"]
        or int(quality.get("record_limit_per_rank", 0)) < 1
        or quality.get("cap_user_limit_to_eligible") is not True
        or int(quality.get("bootstrap_samples", 0)) < 1000
        or int(quality.get("bootstrap_seed", 0)) < 1
        or not isinstance(bindings, list)
        or len(bindings) != len(binding_versions)
        or [value.get("version") for value in bindings] != binding_versions
        or file_sha256(document.get("source_config", {}).get("path", ""))
        != document.get("source_config", {}).get("sha256")
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand next-item recursive rollout config differs")
    return document


def _quantize_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        cache.k.to(torch.float16).to(torch.float32),
        cache.v.to(torch.float16).to(torch.float32),
        cache.seq_len,
    )


def _append_cache(
    model: HSTU,
    cache: HSTUKVCache,
    sequence: dict[str, np.ndarray],
    device: torch.device,
) -> HSTUKVCache:
    items = torch.from_numpy(sequence["item_ids"]).long().unsqueeze(0).to(device)
    behaviors = torch.from_numpy(sequence["behaviors"]).long().unsqueeze(0).to(device)
    deltas = torch.from_numpy(sequence["time_deltas"]).float().unsqueeze(0).to(device)
    _, updated = model.forward_with_cache(cache, items, behaviors, deltas)
    return _quantize_cache(updated)


def _slice(sequence: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {
        name: sequence[name][start:stop]
        for name in ("item_ids", "behaviors", "time_deltas")
    }


def _recursive_cache(
    models: dict[int, HSTU],
    prefix: dict[str, np.ndarray],
    timestamps: np.ndarray,
    boundaries: list[tuple[int, int]],
    device: torch.device,
) -> tuple[HSTUKVCache, dict[int, int]]:
    versions = np.full(len(timestamps), 2, dtype=np.int64)
    for timestamp, version in boundaries:
        versions[timestamps >= timestamp] = version
    changes = np.flatnonzero(versions[1:] != versions[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), changes))
    stops = np.concatenate((changes, np.asarray([len(versions)])))
    first_version = int(versions[0])
    cache = _stored_cache(
        models[first_version],
        _slice(prefix, int(starts[0]), int(stops[0])),
        device,
    )
    for start, stop in zip(starts[1:], stops[1:], strict=True):
        version = int(versions[int(start)])
        cache = _append_cache(
            models[version],
            cache,
            _slice(prefix, int(start), int(stop)),
            device,
        )
    counts = {
        int(version): int(np.count_nonzero(versions == version))
        for version in np.unique(versions)
    }
    return cache, counts


def _run_positive_hidden(
    model: HSTU,
    cache: HSTUKVCache,
    suffix: dict[str, np.ndarray],
    labels: np.ndarray,
    chunk: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    current = cache
    for start in range(0, len(suffix["item_ids"]), chunk):
        stop = min(start + chunk, len(suffix["item_ids"]))
        items = torch.from_numpy(suffix["item_ids"][start:stop]).long().unsqueeze(0).to(device)
        behaviors = torch.from_numpy(suffix["behaviors"][start:stop]).long().unsqueeze(0).to(device)
        deltas = torch.from_numpy(suffix["time_deltas"][start:stop]).float().unsqueeze(0).to(device)
        hidden, current = model.forward_with_cache(current, items, behaviors, deltas)
        mask = torch.from_numpy(labels[start:stop]).to(device)
        if bool(torch.any(mask)):
            values.append(hidden[0][mask].detach().cpu())
    return torch.cat(values)


def _candidate_metric_values(scores: torch.Tensor) -> torch.Tensor:
    values = scores.double()
    ranks = 1 + (values[:, :, 1:] >= values[:, :, :1]).sum(dim=-1)
    ranks_float = ranks.double()
    output = torch.zeros(
        (*ranks.shape, len(METRICS)),
        dtype=torch.float64,
        device=scores.device,
    )
    output[:, :, 0] = torch.logsumexp(values, dim=-1) - values[:, :, 0]
    output[:, :, 1] = torch.where(
        ranks <= 5,
        torch.reciprocal(torch.log2(ranks_float + 1.0)),
        torch.zeros_like(ranks_float),
    )
    output[:, :, 2] = torch.where(
        ranks <= 10,
        torch.reciprocal(torch.log2(ranks_float + 1.0)),
        torch.zeros_like(ranks_float),
    )
    output[:, :, 3] = torch.reciprocal(ranks_float)
    output[:, :, 4] = (ranks <= 1).double()
    output[:, :, 5] = (ranks <= 5).double()
    output[:, :, 6] = (ranks <= 10).double()
    return output.cpu()


def nested_logged_unengaged_candidate_ids(
    positive_ids: torch.Tensor,
    day_item_ids: np.ndarray,
    engaged_labels: np.ndarray,
    fallback_items: torch.Tensor,
    *,
    num_prediction_items: int,
    maximum_negative_count: int,
    seed: int | None,
) -> torch.Tensor:
    positives = positive_ids.detach().cpu().long()
    items = np.asarray(day_item_ids, dtype=np.int64)
    labels = np.asarray(engaged_labels, dtype=np.bool_)
    fallback = fallback_items.detach().cpu().long().numpy()
    if (
        positives.ndim != 1
        or items.ndim != 1
        or labels.ndim != 1
        or len(items) != len(labels)
        or maximum_negative_count < 1
        or fallback.ndim != 1
        or len(np.unique(fallback)) != len(fallback)
        or np.any(fallback < 1)
        or np.any(fallback > num_prediction_items)
    ):
        raise ValueError("logged unengaged candidate request differs")
    positive_positions = np.flatnonzero(labels)
    if (
        len(positive_positions) != len(positives)
        or not np.array_equal(items[positive_positions], positives.numpy())
    ):
        raise ValueError("logged positive alignment differs")
    valid_negative_positions = np.flatnonzero(
        (~labels) & (items >= 1) & (items <= num_prediction_items)
    )
    blocked = set(int(value) for value in positives.tolist())
    rows = []
    for ordinal, (positive, position) in enumerate(
        zip(positives.tolist(), positive_positions.tolist(), strict=True)
    ):
        if seed is None:
            ordered_positions = valid_negative_positions[
                np.lexsort(
                    (
                        valid_negative_positions,
                        np.abs(valid_negative_positions - position),
                    )
                )
            ]
        else:
            generator = np.random.default_rng(
                int(seed) + ordinal * 1_000_003 + int(positive) * 97
            )
            ordered_positions = generator.permutation(valid_negative_positions)
        selected = []
        seen = {int(positive)}
        for raw_position in ordered_positions:
            item = int(items[int(raw_position)])
            if item not in blocked and item not in seen:
                selected.append(item)
                seen.add(item)
                if len(selected) == maximum_negative_count:
                    break
        if len(selected) < maximum_negative_count:
            for raw_item in fallback:
                item = int(raw_item)
                if item not in blocked and item not in seen:
                    selected.append(item)
                    seen.add(item)
                    if len(selected) == maximum_negative_count:
                        break
        if len(selected) != maximum_negative_count:
            raise ValueError("logged unengaged candidate coverage differs")
        rows.append(torch.tensor([positive, *selected], dtype=torch.int64))
    if not rows:
        return torch.empty((0, maximum_negative_count + 1), dtype=torch.int64)
    return torch.stack(rows)


@torch.no_grad()
def _candidate_sums(
    current: HSTU,
    hidden: torch.Tensor,
    positives: torch.Tensor,
    popular: dict[str, torch.Tensor],
    day_item_ids: np.ndarray,
    engaged_labels: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    digest: hashlib._Hash,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    quality = config["quality"]
    counts = [int(value) for value in quality["negative_counts"]]
    maximum = max(counts)
    seeds = [int(value) for value in quality["uniform_candidate_seeds"]]
    variant_names = [f"uniform_seed_{value}" for value in seeds]
    variant_names.extend(popular)
    variant_names.append("logged_unengaged_nearest")
    variant_names.extend(f"logged_unengaged_seed_{value}" for value in seeds)
    candidate_sets = [
        nested_uniform_candidate_ids(
            positives,
            num_prediction_items=current.cfg.num_prediction_items,
            maximum_negative_count=maximum,
            seed=seed,
        )
        for seed in seeds
    ]
    candidate_sets.extend(
        nested_popular_candidate_ids(
            positives,
            popular_items,
            maximum_negative_count=maximum,
        )
        for popular_items in popular.values()
    )
    candidate_sets.append(
        nested_logged_unengaged_candidate_ids(
            positives,
            day_item_ids,
            engaged_labels,
            popular["base_period_exposure_popular"],
            num_prediction_items=current.cfg.num_prediction_items,
            maximum_negative_count=maximum,
            seed=None,
        )
    )
    candidate_sets.extend(
        nested_logged_unengaged_candidate_ids(
            positives,
            day_item_ids,
            engaged_labels,
            popular["base_period_exposure_popular"],
            num_prediction_items=current.cfg.num_prediction_items,
            maximum_negative_count=maximum,
            seed=seed,
        )
        for seed in seeds
    )
    target_counts = np.asarray(
        [min(len(positives), value) if value is not None else len(positives) for value in HORIZONS],
        dtype=np.int64,
    )
    sums = np.zeros(
        (
            len(candidate_sets),
            len(HORIZONS),
            len(counts),
            len(METHODS),
            len(METRICS),
        ),
        dtype=np.float64,
    )
    for variant, candidates in enumerate(candidate_sets):
        digest.update(candidates.numpy().astype("<i8", copy=False).tobytes())
        candidate_vectors = current.item_emb.weight[candidates.to(device)]
        scores = torch.einsum("mth,tch->mtc", hidden.to(device), candidate_vectors)
        for count_index, count in enumerate(counts):
            metric_values = _candidate_metric_values(scores[:, :, : count + 1])
            for horizon_index, target_count in enumerate(target_counts):
                sums[variant, horizon_index, count_index] = (
                    metric_values[:, :target_count].sum(dim=1).numpy()
                )
        del candidate_vectors, scores
    return sums, target_counts, variant_names


def _bootstrap_weights(records: int, samples: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.multinomial(
        records,
        np.full(records, 1.0 / records, dtype=np.float64),
        size=samples,
    ).astype(np.float64)


def _summarize(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    variant_names: list[str],
) -> dict[str, Any]:
    quality = config["quality"]
    counts = [int(value) for value in quality["negative_counts"]]
    sums = np.stack([value["sums"] for value in records])
    targets = np.stack([value["targets_by_horizon"] for value in records]).astype(np.float64)
    weights = _bootstrap_weights(
        len(records),
        int(quality["bootstrap_samples"]),
        int(quality["bootstrap_seed"]),
    )
    fresh_index = METHODS.index("fresh_recompute")
    result: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS[:-1]):
        by_variant: dict[str, Any] = {}
        for variant_index, variant in enumerate(variant_names):
            by_horizon: dict[str, Any] = {}
            for horizon_index, horizon in enumerate(HORIZONS):
                denominator = float(targets[:, horizon_index].sum())
                bootstrap_denominator = weights @ targets[:, horizon_index]
                by_count: dict[str, Any] = {}
                for count_index, count in enumerate(counts):
                    metrics: dict[str, Any] = {}
                    for metric_index, metric in enumerate(METRICS):
                        reuse_values = sums[
                            :, variant_index, horizon_index, count_index, method_index, metric_index
                        ]
                        fresh_values = sums[
                            :, variant_index, horizon_index, count_index, fresh_index, metric_index
                        ]
                        reuse = float(reuse_values.sum() / denominator)
                        fresh = float(fresh_values.sum() / denominator)
                        gaps = reuse_values - fresh_values if metric == "cross_entropy" else fresh_values - reuse_values
                        gap = float(gaps.sum() / denominator)
                        bootstrap = (weights @ gaps) / bootstrap_denominator
                        interval = np.quantile(bootstrap, [0.025, 0.975])
                        metrics[metric] = {
                            "reuse": reuse,
                            "recompute": fresh,
                            "recompute_advantage_absolute": gap,
                            "relative_to_reuse_percent": 100.0 * gap / abs(reuse) if reuse else None,
                            "relative_to_recompute_percent": 100.0 * gap / abs(fresh) if fresh else None,
                            "user_cluster_95_interval": [float(interval[0]), float(interval[1])],
                            "positive_with_ci": bool(interval[0] > 0.0),
                        }
                    by_count[str(count)] = {
                        "candidate_count": count + 1,
                        "metrics": metrics,
                    }
                by_horizon[_horizon_name(horizon)] = {
                    "positive_targets": int(denominator),
                    "negative_counts": by_count,
                }
            by_variant[variant] = {"horizons": by_horizon}
        result[method] = {"candidate_variants": by_variant}
    return result


def _variant_pass(metrics: dict[str, Any]) -> bool:
    ranking = ("ndcg_at_5", "ndcg_at_10", "mrr", "hit_rate_at_5", "hit_rate_at_10")
    positive = [
        metric
        for metric in ranking
        if metrics[metric]["positive_with_ci"]
        and metrics[metric]["relative_to_reuse_percent"] is not None
        and metrics[metric]["relative_to_reuse_percent"] >= 5.0
    ]
    return bool(metrics["cross_entropy"]["positive_with_ci"] and len(positive) >= 2)


def _decision(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    quality = config["quality"]
    preferred = str(quality["preferred_negative_count"])
    uniform = [f"uniform_seed_{value}" for value in quality["uniform_candidate_seeds"]]
    opportunities = []
    diagnostics = []
    for method in METHODS[:-1]:
        for horizon in (_horizon_name(value) for value in HORIZONS):
            variants = summary[method]["candidate_variants"]
            passes = {
                variant: _variant_pass(
                    variants[variant]["horizons"][horizon]["negative_counts"][preferred]["metrics"]
                )
                for variant in variants
            }
            row = {"method": method, "horizon": horizon, "variant_passes": passes}
            diagnostics.append(row)
            if all(passes[name] for name in uniform):
                opportunities.append(row)
    return {
        "preferred_negative_count": int(preferred),
        "uniform_seed_robust_opportunities": opportunities,
        "diagnostics": diagnostics,
    }


def run_next_item_rollout_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_next_item_rollout_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand recursive rollout evaluation requires two ranks")
    output = Path(config["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = _effective_document(
            {
                **config,
                "source": {"config": config["source_config"]},
                "evaluation_methods": ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"],
                "record_limit_per_rank": config["quality"]["record_limit_per_rank"],
                "cap_user_limit_to_eligible": True,
            }
        )
        document["data"]["history_window_days"] = config["evaluation_history_window_days"]
        torch.set_float32_matmul_precision("high")
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in [14, 15, *config["update_date_indices"]]:
            plan.ingest_day(dates[int(date_index)])
        models = {}
        binding_by_version = {
            int(value["version"]): value for value in config["checkpoint_bindings"]
        }
        for version in range(2, int(config["current_version"]) + 1):
            model = make_model(document, plan, runtime.device)
            checkpoint = Path(binding_by_version[version]["path"])
            _load_checkpoint(model, checkpoint.parents[1], version)
            model.eval()
            models[version] = model
        current_version = int(config["current_version"])
        current = models[current_version]
        update_date = dates[int(config["update_date_indices"][-1])]
        eval_date = dates[int(config["evaluation_date_index"])]
        selected, eligible = _selected_users(
            plan,
            update_date,
            eval_date,
            int(config["quality"]["record_limit_per_rank"]) * runtime.world_size,
            int(document["quality"]["sampling_seed"]) + current_version * 1009,
            True,
        )
        local_users = selected[runtime.rank :: runtime.world_size]
        boundaries = [
            (
                int(plan.daily_segments[dates[int(date_index)]]["time_ms"].min()),
                version,
            )
            for version, date_index in zip(
                range(3, current_version),
                config["update_date_indices"][1:],
                strict=True,
            )
        ]
        exposure_counts = np.zeros(plan.num_prediction_items + 1, dtype=np.int64)
        engaged_counts = np.zeros(plan.num_prediction_items + 1, dtype=np.int64)
        for date in plan.base_dates:
            frame = plan.daily_segments[date]
            exposed = frame["item_idx"].to_numpy(dtype=np.int64)
            exposed = exposed[(exposed >= 1) & (exposed <= plan.num_prediction_items)]
            engaged = frame.loc[frame["label"] > 0, "item_idx"].to_numpy(dtype=np.int64)
            engaged = engaged[(engaged >= 1) & (engaged <= plan.num_prediction_items)]
            np.add.at(exposure_counts, exposed, 1)
            np.add.at(engaged_counts, engaged, 1)
        ids = np.arange(1, plan.num_prediction_items + 1, dtype=np.int64)
        popular = {
            "base_period_exposure_popular": torch.from_numpy(
                ids[np.lexsort((ids, -exposure_counts[1:]))].copy()
            ),
            "base_period_engaged_popular": torch.from_numpy(
                ids[np.lexsort((ids, -engaged_counts[1:]))].copy()
            ),
        }
        digest = hashlib.sha256()
        records = []
        maximum_cache_duplicate = 0.0
        maximum_hidden_duplicate = 0.0
        maximum_recursive_length_error = 0
        started = time.perf_counter()
        variant_names = []
        for ordinal, user in enumerate(local_users):
            sequence = _evaluation_sequence(plan, user, eval_date)
            user_day = plan.daily_segments[eval_date]
            user_day = user_day[user_day["user_idx"] == user].sort_values("time_ms")
            history = plan._build_seq(user, as_of_timestamp=int(user_day["time_ms"].min()))
            timestamps = history["timestamps"][:-1].copy()
            prefix = sequence["prefix"]
            if len(timestamps) != len(prefix["item_ids"]):
                raise RuntimeError("KuaiRand recursive prefix timestamp alignment differs")
            fresh_a = _stored_cache(current, prefix, runtime.device)
            fresh_b = _stored_cache(current, prefix, runtime.device)
            lag1 = _stored_cache(models[current_version - 1], prefix, runtime.device)
            lag5 = _stored_cache(models[current_version - 5], prefix, runtime.device)
            recursive, lineage = _recursive_cache(
                models,
                prefix,
                timestamps,
                boundaries,
                runtime.device,
            )
            hidden_by_method = {
                "lag1_monolithic": _run_positive_hidden(
                    current,
                    lag1,
                    sequence["suffix"],
                    sequence["labels"],
                    int(document["quality"]["suffix_chunk"]),
                    runtime.device,
                ),
                "lag5_monolithic": _run_positive_hidden(
                    current,
                    lag5,
                    sequence["suffix"],
                    sequence["labels"],
                    int(document["quality"]["suffix_chunk"]),
                    runtime.device,
                ),
                "recursive_rollout": _run_positive_hidden(
                    current,
                    recursive,
                    sequence["suffix"],
                    sequence["labels"],
                    int(document["quality"]["suffix_chunk"]),
                    runtime.device,
                ),
                "fresh_recompute": _run_positive_hidden(
                    current,
                    fresh_a,
                    sequence["suffix"],
                    sequence["labels"],
                    int(document["quality"]["suffix_chunk"]),
                    runtime.device,
                ),
            }
            fresh_duplicate_hidden = _run_positive_hidden(
                current,
                fresh_b,
                sequence["suffix"],
                sequence["labels"],
                int(document["quality"]["suffix_chunk"]),
                runtime.device,
            )
            hidden = torch.stack([hidden_by_method[name] for name in METHODS])
            positives = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
            if hidden.shape[1] != len(positives):
                raise RuntimeError("KuaiRand recursive positive alignment differs")
            sums, targets_by_horizon, variant_names = _candidate_sums(
                current,
                hidden,
                positives,
                popular,
                sequence["targets"],
                sequence["labels"],
                config,
                runtime.device,
                digest,
            )
            maximum_cache_duplicate = max(
                maximum_cache_duplicate,
                float((fresh_a.k - fresh_b.k).abs().max().item()),
                float((fresh_a.v - fresh_b.v).abs().max().item()),
            )
            maximum_hidden_duplicate = max(
                maximum_hidden_duplicate,
                float((hidden_by_method["fresh_recompute"] - fresh_duplicate_hidden).abs().max().item()),
            )
            maximum_recursive_length_error = max(
                maximum_recursive_length_error,
                abs(recursive.seq_len - len(prefix["item_ids"])),
            )
            records.append(
                {
                    "user_id": user,
                    "targets_by_horizon": targets_by_horizon,
                    "sums": sums,
                    "prefix_length": len(prefix["item_ids"]),
                    "lineage": lineage,
                    "cache_relative_error": {
                        "lag1_monolithic": cache_relative_error(lag1, fresh_a),
                        "lag5_monolithic": cache_relative_error(lag5, fresh_a),
                        "recursive_rollout": cache_relative_error(recursive, fresh_a),
                    },
                    "hidden_relative_error": {
                        method: float(
                            torch.linalg.vector_norm(
                                (hidden_by_method[method] - hidden_by_method["fresh_recompute"]).double()
                            )
                            / torch.linalg.vector_norm(
                                hidden_by_method["fresh_recompute"].double()
                            ).clamp_min(1e-12)
                        )
                        for method in METHODS[:-1]
                    },
                }
            )
            if (ordinal + 1) % 32 == 0 or ordinal + 1 == len(local_users):
                print(
                    f"phase=kuairand_recursive_rollout rank={runtime.rank} "
                    f"users={ordinal + 1}/{len(local_users)}",
                    flush=True,
                )
        gathered: list[Any] | None = [None] * runtime.world_size if runtime.is_primary else None
        dist.gather_object(
            {
                "records": records,
                "candidate_sha256": digest.hexdigest(),
                "maximum_cache_duplicate": maximum_cache_duplicate,
                "maximum_hidden_duplicate": maximum_hidden_duplicate,
                "maximum_recursive_length_error": maximum_recursive_length_error,
            },
            gathered,
            dst=0,
        )
        if not runtime.is_primary:
            dist.barrier()
            return None
        combined = sorted(
            [record for shard in gathered for record in shard["records"]],
            key=lambda value: int(value["user_id"]),
        )
        summary = _summarize(combined, config, variant_names)
        feature_summary = {}
        for method in METHODS[:-1]:
            feature_summary[method] = {
                "cache_relative_error_mean": float(
                    np.mean([value["cache_relative_error"][method] for value in combined])
                ),
                "hidden_relative_error_mean": float(
                    np.mean([value["hidden_relative_error"][method] for value in combined])
                ),
            }
        result = {
            "protocol": ROLLOUT_PROTOCOL,
            "source_protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data": metadata,
            "evaluation": {
                "update_date": update_date,
                "evaluation_date": eval_date,
                "eligible_users": eligible,
                "selected_users": len(combined),
                "selected_user_ids_sha256": hashlib.sha256(
                    np.asarray([value["user_id"] for value in combined], dtype="<i8").tobytes()
                ).hexdigest(),
                "candidate_sha256_by_rank": [value["candidate_sha256"] for value in gathered],
                "variants": variant_names,
                "horizons": [_horizon_name(value) for value in HORIZONS],
                "negative_counts": config["quality"]["negative_counts"],
            },
            "mechanism": feature_summary,
            "candidate_quality": summary,
            "decision": _decision(summary, config),
            "sanity": {
                "maximum_fresh_duplicate_cache_absolute_error": max(
                    value["maximum_cache_duplicate"] for value in gathered
                ),
                "maximum_fresh_duplicate_hidden_absolute_error": max(
                    value["maximum_hidden_duplicate"] for value in gathered
                ),
                "maximum_recursive_length_error": max(
                    value["maximum_recursive_length_error"] for value in gathered
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)


def validate_next_item_rollout_result(result: dict[str, Any]) -> None:
    sanity = result.get("sanity", {})
    evaluation = result.get("evaluation", {})
    if (
        result.get("protocol") != ROLLOUT_PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or int(evaluation.get("eligible_users", 0)) < 700
        or int(evaluation.get("selected_users", 0)) > int(evaluation.get("eligible_users", 0))
        or evaluation.get("horizons") != ["first_1", "first_4", "first_16", "all"]
        or evaluation.get("negative_counts") != [49, 99, 199]
        or sanity.get("maximum_fresh_duplicate_cache_absolute_error") != 0.0
        or sanity.get("maximum_fresh_duplicate_hidden_absolute_error") != 0.0
        or sanity.get("maximum_recursive_length_error") != 0
    ):
        raise ValueError("KuaiRand recursive rollout result differs")
