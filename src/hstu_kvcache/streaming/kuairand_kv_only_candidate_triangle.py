from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_kv_only_chain import (
    _effective_document,
    load_kv_only_chain_config,
)
from .kuairand_root_cause import (
    _atomic_json,
    _evaluation_sequence,
    _parameter_distances,
    _run_suffix,
    _selected_users,
    _stored_cache,
    file_sha256,
    load_plan,
    make_model,
)
from .qk_protocol_sweep_runner import nested_uniform_candidate_ids

PROTOCOL = "evokv_kuairand_kv_only_candidate_triangle_v0"
METRICS = (
    "candidate_cross_entropy",
    "ndcg_at_5",
    "ndcg_at_10",
    "mrr",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "hit_rate_at_10",
    "pairwise_win_rate",
)
LOWER_IS_BETTER = {"candidate_cross_entropy"}
TABLE_METRICS = (
    "candidate_cross_entropy",
    "ndcg_at_10",
    "mrr",
    "pairwise_win_rate",
)


def load_candidate_triangle_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    quality = document.get("quality", {})
    execution = document.get("execution", {})
    outputs = document.get("outputs", {})
    bindings = document.get("checkpoint_bindings", [])
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("target_versions") != [2, 3, 4, 5, 6, 7, 8]
        or quality.get("negative_counts") != [99, 499, 999]
        or quality.get("uniform_candidate_seeds") != [61031, 14929, 29063]
        or quality.get("metrics") != list(METRICS)
        or quality.get("primary_diagnostic")
        != {"metric": "pairwise_win_rate", "negative_count": 999}
        or int(quality.get("record_limit_per_rank", 0)) != 512
        or quality.get("cap_user_limit_to_eligible") is not True
        or int(quality.get("target_chunk", 0)) != 64
        or int(quality.get("suffix_chunk", 0)) != 64
        or int(quality.get("bootstrap_samples", 0)) != 2000
        or int(quality.get("bootstrap_seed", 0)) < 1
        or int(quality.get("sampling_seed", 0)) < 1
        or execution.get("cuda_visible_devices") != "0,1"
        or execution.get("world_size") != 2
        or not isinstance(document.get("chain_config", {}).get("path"), str)
        or not isinstance(document.get("chain_config", {}).get("sha256"), str)
        or not isinstance(bindings, list)
        or [value.get("version") for value in bindings] != list(range(1, 9))
        or not all(
            isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str)
            for value in bindings
        )
        or not all(
            isinstance(outputs.get(name), str)
            for name in ("result", "table", "log")
        )
    ):
        raise ValueError("KuaiRand KV-only candidate triangle config differs")
    return document


def _verify_bindings(config: dict[str, Any]) -> None:
    chain = config["chain_config"]
    if file_sha256(chain["path"]) != chain["sha256"]:
        raise ValueError("KuaiRand KV-only chain config hash differs")
    load_kv_only_chain_config(chain["path"])
    for binding in config["checkpoint_bindings"]:
        if file_sha256(binding["path"]) != binding["sha256"]:
            raise ValueError(
                f"KuaiRand KV-only theta{binding['version']} checkpoint hash differs"
            )


def _checkpoint(config: dict[str, Any], version: int) -> Path:
    for binding in config["checkpoint_bindings"]:
        if int(binding["version"]) == version:
            return Path(binding["path"])
    raise ValueError("KuaiRand KV-only candidate checkpoint version differs")


def _hash_ints(values: list[int] | np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _candidate_sets(
    positives: torch.Tensor,
    num_prediction_items: int,
    maximum_negative_count: int,
    seeds: list[int],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        nested_uniform_candidate_ids(
            positives,
            num_prediction_items=num_prediction_items,
            maximum_negative_count=maximum_negative_count,
            seed=seed,
        ).to(torch.int32)
        for seed in seeds
    )


def _metric_sums_from_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("KuaiRand candidate score shape differs")
    values = scores.double()
    positive = values[:, :1]
    negatives = values[:, 1:]
    ranks = 1 + (negatives >= positive).sum(dim=1)
    ranks_float = ranks.double()
    output = torch.zeros(len(values), len(METRICS), dtype=torch.float64, device=values.device)
    output[:, 0] = torch.logsumexp(values, dim=1) - values[:, 0]
    output[:, 1] = torch.where(
        ranks <= 5,
        torch.reciprocal(torch.log2(ranks_float + 1.0)),
        torch.zeros_like(ranks_float),
    )
    output[:, 2] = torch.where(
        ranks <= 10,
        torch.reciprocal(torch.log2(ranks_float + 1.0)),
        torch.zeros_like(ranks_float),
    )
    output[:, 3] = torch.reciprocal(ranks_float)
    output[:, 4] = (ranks <= 1).double()
    output[:, 5] = (ranks <= 5).double()
    output[:, 6] = (ranks <= 10).double()
    output[:, 7] = (
        (positive > negatives).double() + 0.5 * (positive == negatives).double()
    ).mean(dim=1)
    return output.sum(dim=0)


@torch.no_grad()
def _candidate_metric_sums(
    model,
    hidden: torch.Tensor,
    candidates: tuple[torch.Tensor, ...],
    negative_counts: list[int],
    target_chunk: int,
    device: torch.device,
) -> np.ndarray:
    if any(len(value) != len(hidden) for value in candidates):
        raise ValueError("KuaiRand candidate target alignment differs")
    output = torch.zeros(
        len(candidates), len(negative_counts), len(METRICS), dtype=torch.float64
    )
    for variant, candidate_ids in enumerate(candidates):
        for start in range(0, len(hidden), target_chunk):
            stop = min(start + target_chunk, len(hidden))
            ids = candidate_ids[start:stop].long().to(device)
            vectors = model.prediction_item_weight[ids]
            scores = torch.einsum("th,tch->tc", hidden[start:stop].to(device), vectors)
            for count_index, negative_count in enumerate(negative_counts):
                output[variant, count_index] += _metric_sums_from_scores(
                    scores[:, : negative_count + 1]
                ).cpu()
            del ids, vectors, scores
    return output.numpy()


def _bootstrap_interval(
    targets: np.ndarray,
    oriented_sums: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    probabilities = np.full(len(targets), 1.0 / len(targets), dtype=np.float64)
    weights = generator.multinomial(len(targets), probabilities, size=samples)
    values = (weights @ oriented_sums) / (weights @ targets)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def _aggregate_cell(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    target: int,
    source: int,
) -> dict[str, Any]:
    quality = config["quality"]
    seeds = [int(value) for value in quality["uniform_candidate_seeds"]]
    counts = [int(value) for value in quality["negative_counts"]]
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    fresh = np.stack([np.asarray(value["fresh_sums"]) for value in records])
    reuse = np.stack([np.asarray(value["reuse_sums"]) for value in records])
    variants: dict[str, Any] = {}
    for variant_index, candidate_seed in enumerate(seeds):
        by_count: dict[str, Any] = {}
        for count_index, negative_count in enumerate(counts):
            metrics: dict[str, Any] = {}
            for metric_index, metric in enumerate(METRICS):
                fresh_values = fresh[:, variant_index, count_index, metric_index]
                reuse_values = reuse[:, variant_index, count_index, metric_index]
                oriented = (
                    reuse_values - fresh_values
                    if metric in LOWER_IS_BETTER
                    else fresh_values - reuse_values
                )
                fresh_mean = float(fresh_values.sum() / denominator)
                reuse_mean = float(reuse_values.sum() / denominator)
                absolute = float(oriented.sum() / denominator)
                interval = _bootstrap_interval(
                    targets,
                    oriented,
                    int(quality["bootstrap_samples"]),
                    int(quality["bootstrap_seed"])
                    + target * 1_000_003
                    + source * 10_007
                    + variant_index * 101
                    + count_index * 17
                    + metric_index,
                )
                metrics[metric] = {
                    "recompute": fresh_mean,
                    "reuse": reuse_mean,
                    "recompute_advantage_absolute": absolute,
                    "recompute_advantage_relative_percent": (
                        100.0 * absolute / abs(reuse_mean) if reuse_mean else None
                    ),
                    "user_cluster_95_interval": interval,
                    "positive_with_ci": bool(interval[0] > 0.0),
                    "negative_with_ci": bool(interval[1] < 0.0),
                }
            by_count[str(negative_count)] = {
                "candidate_count": negative_count + 1,
                "metrics": metrics,
            }
        variants[f"uniform_seed_{candidate_seed}"] = {
            "negative_counts": by_count
        }
    records = sorted(records, key=lambda value: int(value["user_id"]))
    digest = hashlib.sha256()
    for value in records:
        digest.update(bytes.fromhex(value["candidate_sha256"]))
    return {
        "target_version": target,
        "source_version": source,
        "cache_age": target - source,
        "users": len(records),
        "positive_targets": int(denominator),
        "selected_user_ids_sha256": _hash_ints([value["user_id"] for value in records]),
        "candidate_sets_sha256": digest.hexdigest(),
        "candidate_variants": variants,
    }


def _seed_mean_matrices(
    cells: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    variants = [
        f"uniform_seed_{value}" for value in config["quality"]["uniform_candidate_seeds"]
    ]
    for negative_count in config["quality"]["negative_counts"]:
        by_metric: dict[str, Any] = {}
        for metric in METRICS:
            rows: dict[str, Any] = {}
            for target in config["target_versions"]:
                row: dict[str, float] = {}
                for source in range(1, int(target)):
                    cell = next(
                        value
                        for value in cells
                        if value["target_version"] == target
                        and value["source_version"] == source
                    )
                    values = [
                        cell["candidate_variants"][variant]["negative_counts"][
                            str(negative_count)
                        ]["metrics"][metric]["recompute_advantage_relative_percent"]
                        for variant in variants
                    ]
                    row[str(source)] = float(np.mean(values))
                rows[str(target)] = row
            by_metric[metric] = rows
        output[str(negative_count)] = by_metric
    return output


def _age_summary(cells: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    variants = [
        f"uniform_seed_{value}" for value in config["quality"]["uniform_candidate_seeds"]
    ]
    output: dict[str, Any] = {}
    for negative_count in config["quality"]["negative_counts"]:
        by_metric: dict[str, Any] = {}
        for metric in METRICS:
            by_age: dict[str, Any] = {}
            for age in range(1, 8):
                values = [
                    cell["candidate_variants"][variant]["negative_counts"][
                        str(negative_count)
                    ]["metrics"][metric]["recompute_advantage_relative_percent"]
                    for cell in cells
                    if cell["cache_age"] == age
                    for variant in variants
                ]
                by_age[str(age)] = {
                    "cell_seed_pairs": len(values),
                    "mean_relative_percent": float(np.mean(values)),
                    "minimum_relative_percent": float(np.min(values)),
                    "maximum_relative_percent": float(np.max(values)),
                    "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
                }
            by_metric[metric] = by_age
        output[str(negative_count)] = by_metric
    return output


def _decision(cells: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    variants = [
        f"uniform_seed_{value}" for value in config["quality"]["uniform_candidate_seeds"]
    ]
    output: dict[str, Any] = {}
    for negative_count in config["quality"]["negative_counts"]:
        by_metric: dict[str, Any] = {}
        for metric in METRICS:
            values = [
                cell["candidate_variants"][variant]["negative_counts"][
                    str(negative_count)
                ]["metrics"][metric]
                for cell in cells
                for variant in variants
            ]
            cell_signs = []
            for cell in cells:
                seed_values = [
                    cell["candidate_variants"][variant]["negative_counts"][
                        str(negative_count)
                    ]["metrics"][metric]["recompute_advantage_relative_percent"]
                    for variant in variants
                ]
                cell_signs.append(seed_values)
            by_metric[metric] = {
                "cell_seed_pairs": len(values),
                "positive_cell_seed_pairs": sum(
                    value["recompute_advantage_relative_percent"] > 0.0
                    for value in values
                ),
                "positive_with_ci_cell_seed_pairs": sum(
                    value["positive_with_ci"] for value in values
                ),
                "negative_with_ci_cell_seed_pairs": sum(
                    value["negative_with_ci"] for value in values
                ),
                "cells_all_three_seeds_positive": sum(
                    all(value > 0.0 for value in seed_values)
                    for seed_values in cell_signs
                ),
                "cells_majority_seeds_positive": sum(
                    sum(value > 0.0 for value in seed_values) >= 2
                    for seed_values in cell_signs
                ),
                "total_cells": len(cells),
            }
        output[str(negative_count)] = by_metric
    return output


def _table(matrices: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [
        "# KuaiRand KV-only candidate Recompute-over-Reuse triangle",
        "",
        "Each cell is the mean relative Recompute advantage over Reuse across three frozen uniform-negative seeds. All seed-specific endpoints and confidence intervals are retained in result.json.",
    ]
    for negative_count in config["quality"]["negative_counts"]:
        for metric in TABLE_METRICS:
            lines.extend(
                [
                    "",
                    f"## {metric}, {negative_count} negatives",
                    "",
                    "| target\\source | θ1 | θ2 | θ3 | θ4 | θ5 | θ6 | θ7 | θ8 |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            values = matrices[str(negative_count)][metric]
            for target in range(1, 9):
                row = [f"θ{target}"]
                for source in range(1, 9):
                    if source > target or target == 1:
                        row.append("—")
                    elif source == target:
                        row.append("0.000%")
                    else:
                        row.append(f"{values[str(target)][str(source)]:+.3f}%")
                lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _load_model(config: dict[str, Any], version: int, document, plan, device):
    model = make_model(document, plan, device)
    state = torch.load(_checkpoint(config, version), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model.eval()
    return model


def run_candidate_triangle(config_path: str | Path) -> dict[str, Any] | None:
    config = load_candidate_triangle_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    try:
        if runtime.world_size != int(config["execution"]["world_size"]):
            raise ValueError("KuaiRand KV-only candidate triangle world size differs")
        validation: list[str | None] = [None]
        if runtime.is_primary:
            try:
                _verify_bindings(config)
            except Exception as error:
                validation[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(validation, src=0)
        if validation[0] is not None:
            raise ValueError(validation[0])
        output = Path(config["outputs"]["result"])
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        chain_config = load_kv_only_chain_config(config["chain_config"]["path"])
        document = _effective_document(chain_config)
        document["quality"]["suffix_chunk"] = int(config["quality"]["suffix_chunk"])
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in (14, 15):
            plan.ingest_day(dates[date_index])
        torch.set_float32_matmul_precision("high")
        started = time.perf_counter()
        cells = []
        selected_hashes: dict[str, set[str]] = {}
        quality = config["quality"]
        counts = [int(value) for value in quality["negative_counts"]]
        seeds = [int(value) for value in quality["uniform_candidate_seeds"]]
        for target in config["target_versions"]:
            target = int(target)
            if target > 2:
                plan.ingest_day(dates[13 + target])
            update_date = dates[13 + target]
            evaluation_date = dates[14 + target]
            selected, eligible = _selected_users(
                plan,
                update_date,
                evaluation_date,
                int(quality["record_limit_per_rank"]) * runtime.world_size,
                int(quality["sampling_seed"]) + target * 1009,
                True,
            )
            local_users = selected[runtime.rank :: runtime.world_size]
            current = _load_model(config, target, document, plan, runtime.device)
            workloads = []
            for ordinal, user in enumerate(local_users):
                sequence = _evaluation_sequence(plan, user, evaluation_date)
                positives = torch.from_numpy(
                    sequence["targets"][sequence["labels"]]
                ).long()
                candidates = _candidate_sets(
                    positives,
                    current.cfg.num_prediction_items,
                    max(counts),
                    seeds,
                )
                fresh_cache = _stored_cache(current, sequence["prefix"], runtime.device)
                fresh_hidden = _run_suffix(
                    current,
                    fresh_cache,
                    sequence["suffix"],
                    sequence["labels"],
                    int(quality["suffix_chunk"]),
                    runtime.device,
                )
                if len(fresh_hidden) != len(positives):
                    raise RuntimeError("KuaiRand candidate fresh target alignment differs")
                fresh_sums = _candidate_metric_sums(
                    current,
                    fresh_hidden,
                    candidates,
                    counts,
                    int(quality["target_chunk"]),
                    runtime.device,
                )
                digest = hashlib.sha256()
                for value in candidates:
                    digest.update(value.numpy().astype("<i4", copy=False).tobytes())
                workloads.append(
                    {
                        "user_id": int(user),
                        "sequence": sequence,
                        "targets": len(positives),
                        "candidates": candidates,
                        "fresh_sums": fresh_sums,
                        "candidate_sha256": digest.hexdigest(),
                    }
                )
                if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(local_users):
                    print(
                        f"phase=kuairand_candidate_fresh target={target} rank={runtime.rank} "
                        f"users={ordinal + 1}/{len(local_users)}",
                        flush=True,
                    )
            target_hashes = set()
            for source in range(1, target):
                source_model = _load_model(config, source, document, plan, runtime.device)
                distances = _parameter_distances(source_model, current)
                if any(
                    distances[name]["relative_l2_update"] != 0.0
                    for name in ("item_embedding", "input_encoders", "other_core")
                ):
                    raise RuntimeError("KuaiRand KV-only parameter isolation differs")
                local_records = []
                for ordinal, workload in enumerate(workloads):
                    sequence = workload["sequence"]
                    stale_cache = _stored_cache(
                        source_model, sequence["prefix"], runtime.device
                    )
                    stale_hidden = _run_suffix(
                        current,
                        stale_cache,
                        sequence["suffix"],
                        sequence["labels"],
                        int(quality["suffix_chunk"]),
                        runtime.device,
                    )
                    reuse_sums = _candidate_metric_sums(
                        current,
                        stale_hidden,
                        workload["candidates"],
                        counts,
                        int(quality["target_chunk"]),
                        runtime.device,
                    )
                    local_records.append(
                        {
                            "user_id": workload["user_id"],
                            "targets": workload["targets"],
                            "fresh_sums": workload["fresh_sums"].tolist(),
                            "reuse_sums": reuse_sums.tolist(),
                            "candidate_sha256": workload["candidate_sha256"],
                        }
                    )
                    if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(workloads):
                        print(
                            f"phase=kuairand_candidate_stale target={target} source={source} "
                            f"rank={runtime.rank} users={ordinal + 1}/{len(workloads)}",
                            flush=True,
                        )
                gathered: list[Any] | None = (
                    [None] * runtime.world_size if runtime.is_primary else None
                )
                dist.gather_object(local_records, gathered, dst=0)
                if runtime.is_primary:
                    combined = [value for shard in gathered for value in shard]
                    cell = _aggregate_cell(combined, config, target, source)
                    cell.update(
                        {
                            "update_date": update_date,
                            "evaluation_date": evaluation_date,
                            "eligible_same_user_population": eligible,
                            "parameter_group_distances": distances,
                        }
                    )
                    target_hashes.add(cell["selected_user_ids_sha256"])
                    cells.append(cell)
                del source_model, local_records
                gc.collect()
                torch.cuda.empty_cache()
            if runtime.is_primary:
                selected_hashes[str(target)] = target_hashes
            del workloads, current
            gc.collect()
            torch.cuda.empty_cache()
        if not runtime.is_primary:
            dist.barrier()
            return None
        if any(len(value) != 1 for value in selected_hashes.values()):
            raise RuntimeError("KuaiRand candidate target user sets differ by source")
        matrices = _seed_mean_matrices(cells, config)
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "chain_config": config["chain_config"],
            "data": metadata,
            "serving_semantics": "old-version prefix K/V plus current-model suffix, predicting each engaged next item from a real preceding item",
            "candidate_protocol": {
                "construction": "nested deterministic uniform negatives, identical across Recompute and every Reuse source",
                "negative_counts": counts,
                "uniform_candidate_seeds": seeds,
                "metrics": list(METRICS),
                "cluster_unit": "user",
            },
            "cells": cells,
            "seed_mean_matrices": matrices,
            "age_summary": _age_summary(cells, config),
            "decision": _decision(cells, config),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        table = Path(config["outputs"]["table"])
        table.parent.mkdir(parents=True, exist_ok=True)
        temporary = table.with_suffix(table.suffix + ".tmp")
        temporary.write_text(_table(matrices, config))
        temporary.replace(table)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)
