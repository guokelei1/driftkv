from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_kv_only_candidate_triangle import (
    LOWER_IS_BETTER,
    METRICS,
    _candidate_metric_sums,
    _candidate_sets,
)
from .kuairand_root_cause import (
    _atomic_json,
    _atomic_torch,
    _evaluation_sequence,
    _run_suffix,
    _selected_users,
    _stored_cache,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)
from .kuairand_untied_cache_path_screen import (
    _load_checkpoint_model,
    _seed,
)

PROTOCOL = "evokv_kuairand_kv_strength_screen_v0"
EXPECTED_CANDIDATES = [
    {"id": "kv_lr100_e2", "lr": 0.0001, "epochs": 2},
    {"id": "kv_lr200_e2", "lr": 0.0002, "epochs": 2},
    {"id": "kv_lr500_e2", "lr": 0.0005, "epochs": 2},
    {"id": "kv_lr1000_e2", "lr": 0.001, "epochs": 2},
    {"id": "kv_lr500_e4", "lr": 0.0005, "epochs": 4},
    {"id": "kv_lr1000_e4", "lr": 0.001, "epochs": 4},
]


def load_kv_strength_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    training = document.get("training", {})
    evaluation = document.get("evaluation", {})
    analysis = document.get("analysis", {})
    execution = document.get("execution", {})
    outputs = document.get("outputs", {})
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("source_version") != 1
        or document.get("target_version") != 2
        or document.get("update_date_index") != 15
        or document.get("evaluation_date_index") != 16
        or document.get("candidates") != EXPECTED_CANDIDATES
        or training.get("parameter_group") != "kv_projections_only"
        or training.get("negative_count") != 32
        or training.get("batch_size") != 8
        or float(training.get("weight_decay", -1.0)) != 0.0001
        or evaluation.get("negative_counts") != [99, 999]
        or evaluation.get("uniform_candidate_seeds") != [61031, 14929, 29063]
        or evaluation.get("metrics") != list(METRICS)
        or float(evaluation.get("tuning_fraction", 0.0)) != 0.25
        or int(evaluation.get("record_limit_per_rank", 0)) != 512
        or evaluation.get("cap_user_limit_to_eligible") is not True
        or int(evaluation.get("bootstrap_samples", 0)) != 2000
        or analysis.get("target_scopes")
        != {"all": None, "first_1": 1, "first_4": 4}
        or analysis.get("prefix_cohorts")
        != {"all": 0, "prefix_ge_256": 256, "prefix_at_cap": 511}
        or analysis.get("primary")
        != {"target_scope": "first_4", "prefix_cohort": "prefix_ge_256"}
        or execution.get("training_cuda_visible_devices") != "0"
        or execution.get("evaluation_cuda_visible_devices") != "0,1"
        or execution.get("evaluation_world_size") != 2
        or file_sha256(source.get("base_config", {}).get("path", ""))
        != source.get("base_config", {}).get("sha256")
        or file_sha256(source.get("theta1", {}).get("path", ""))
        != source.get("theta1", {}).get("sha256")
        or not all(
            isinstance(outputs.get(name), str)
            for name in ("checkpoint_root", "training_result", "evaluation_result", "table")
        )
    ):
        raise ValueError("KuaiRand K/V strength screen config differs")
    return document


def _effective_document(config: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(config["source"]["base_config"]["path"]).read_text())
    document["model"]["tie_item_embeddings"] = True
    document["data"]["total_num_days"] = 17
    return document


def _checkpoint(config: dict[str, Any], candidate: str) -> Path:
    return (
        Path(config["outputs"]["checkpoint_root"])
        / candidate
        / "theta_2"
        / "model.pt"
    )


def _kv_parameters(model) -> tuple[list[torch.nn.Parameter], list[str]]:
    parameters = []
    names = []
    for name, parameter in model.named_parameters():
        active = ".attn.k_proj." in name or ".attn.v_proj." in name
        parameter.requires_grad_(active)
        if active:
            parameters.append(parameter)
            names.append(name)
    if not parameters or len(names) != model.cfg.num_layers * 2:
        raise RuntimeError("KuaiRand K/V projection parameter selection differs")
    return parameters, names


@torch.no_grad()
def _kv_distance(previous, current) -> dict[str, dict[str, float]]:
    previous_parameters = dict(previous.named_parameters())
    numerator = 0.0
    denominator = 0.0
    parameters = 0
    for name, value in current.named_parameters():
        if ".attn.k_proj." not in name and ".attn.v_proj." not in name:
            continue
        old = previous_parameters[name]
        numerator += float((value - old).float().square().sum().item())
        denominator += float(old.float().square().sum().item())
        parameters += value.numel()
    return {
        "kv_projections": {
            "parameters": parameters,
            "relative_l2_update": math.sqrt(numerator)
            / max(math.sqrt(denominator), 1e-12),
        },
        "non_kv_parameters": {
            "parameters": sum(
                value.numel()
                for name, value in current.named_parameters()
                if ".attn.k_proj." not in name and ".attn.v_proj." not in name
            ),
            "relative_l2_update": 0.0,
            "validation": "frozen_by_requires_grad_contract_and_checkpoint_manifest",
        },
    }


def _valid_candidate(
    config: dict[str, Any], candidate: dict[str, Any], model_path: Path
) -> dict[str, Any] | None:
    manifest_path = model_path.with_name("manifest.json")
    if not model_path.is_file() or not manifest_path.is_file():
        if model_path.exists() or manifest_path.exists():
            raise FileExistsError(f"KuaiRand K/V candidate is partial: {candidate['id']}")
        return None
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("status") != "complete_development_checkpoint"
        or manifest.get("candidate") != candidate
        or manifest.get("source_model_sha256") != config["source"]["theta1"]["sha256"]
        or manifest.get("model_sha256") != file_sha256(model_path)
    ):
        raise ValueError(f"KuaiRand K/V candidate resume differs: {candidate['id']}")
    return manifest


def run_kv_strength_training(config_path: str | Path) -> dict[str, Any]:
    config = load_kv_strength_config(config_path)
    output = Path(config["outputs"]["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand K/V strength training requires one rank")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    document = _effective_document(config)
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    plan.ingest_day(dates[14])
    source_histories = copy.deepcopy(plan.user_histories)
    source_state = torch.load(
        config["source"]["theta1"]["path"], map_location="cpu", weights_only=True
    )
    records = []
    started = time.perf_counter()
    training = config["training"]
    for candidate in config["candidates"]:
        model_path = _checkpoint(config, str(candidate["id"]))
        manifest = _valid_candidate(config, candidate, model_path)
        if manifest is not None:
            records.append(manifest)
            continue
        plan.user_histories = copy.deepcopy(source_histories)
        update_date = dates[int(config["update_date_index"])]
        plan.ingest_day(update_date)
        model = make_model(document, plan, device)
        model.load_state_dict(source_state)
        parameters, parameter_names = _kv_parameters(model)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(candidate["lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        epochs = []
        for epoch in range(int(candidate["epochs"])):
            _seed(int(training["seed"]) + epoch)
            batches = plan.iter_train_batches(
                update_date,
                int(training["batch_size"]),
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
            epochs.append(
                _train_epoch(
                    model,
                    optimizer,
                    batches,
                    device,
                    int(training["negative_count"]),
                    None,
                    f"kuairand_kv_strength_{candidate['id']}_epoch{epoch + 1}",
                )
            )
        _atomic_torch(model_path, model.state_dict())
        manifest = {
            "protocol": PROTOCOL,
            "status": "complete_development_checkpoint",
            "scientific_result": False,
            "formal_result": False,
            "candidate": candidate,
            "source_model_sha256": config["source"]["theta1"]["sha256"],
            "model_sha256": file_sha256(model_path),
            "model": asdict(model.cfg),
            "parameter_group": training["parameter_group"],
            "trainable_parameter_names": parameter_names,
            "trainable_parameters": sum(value.numel() for value in parameters),
            "training": epochs,
        }
        _atomic_json(model_path.with_name("manifest.json"), manifest)
        records.append(manifest)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "protocol": PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "source": config["source"]["theta1"],
        "candidates": records,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def _metric_family(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for comparison in ("recompute_over_reuse", "current_over_previous"):
        metrics = {}
        for metric in METRICS:
            values = []
            intervals = []
            for seed in config["evaluation"]["uniform_candidate_seeds"]:
                value = summary["candidate_variants"][f"uniform_seed_{seed}"][
                    "negative_counts"
                ]["999"]["comparisons"][comparison]["metrics"][metric]
                values.append(value["advantage_relative_percent"])
                intervals.append(value["user_cluster_95_interval"])
            numeric = [float(value) for value in values if value is not None]
            metrics[metric] = {
                "relative_percent_by_seed": values,
                "minimum_relative_percent": min(numeric) if numeric else None,
                "mean_relative_percent": float(np.mean(numeric)) if numeric else None,
                "positive_all_seeds": len(numeric) == len(values)
                and all(value > 0.0 for value in numeric),
                "positive_with_ci_all_seeds": all(value[0] > 0.0 for value in intervals),
            }
        output[comparison] = metrics
    return output


def _summarize_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    seed_offset: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    seeds = [int(value) for value in evaluation["uniform_candidate_seeds"]]
    counts = [int(value) for value in evaluation["negative_counts"]]
    values = np.stack([np.asarray(value["metric_sums"]) for value in records])
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    generator = np.random.default_rng(
        int(evaluation["bootstrap_seed"]) + seed_offset
    )
    weights = generator.multinomial(
        len(targets),
        np.full(len(targets), 1.0 / len(targets), dtype=np.float64),
        size=int(evaluation["bootstrap_samples"]),
    )
    bootstrap_denominators = weights @ targets
    output: dict[str, Any] = {}
    columns = []
    entries = []
    comparisons = {
        "recompute_over_reuse": (0, 1),
        "current_over_previous": (0, 2),
    }
    for variant_index, candidate_seed in enumerate(seeds):
        by_count = {}
        for count_index, negative_count in enumerate(counts):
            by_comparison = {}
            for comparison, (current_index, baseline_index) in comparisons.items():
                metrics = {}
                for metric_index, metric in enumerate(METRICS):
                    current = values[
                        :, current_index, variant_index, count_index, metric_index
                    ]
                    baseline = values[
                        :, baseline_index, variant_index, count_index, metric_index
                    ]
                    oriented = (
                        baseline - current
                        if metric in LOWER_IS_BETTER
                        else current - baseline
                    )
                    current_mean = float(current.sum() / denominator)
                    baseline_mean = float(baseline.sum() / denominator)
                    advantage = float(oriented.sum() / denominator)
                    metrics[metric] = {
                        "current": current_mean,
                        "baseline": baseline_mean,
                        "advantage_absolute": advantage,
                        "advantage_relative_percent": (
                            100.0 * advantage / abs(baseline_mean)
                            if baseline_mean
                            else None
                        ),
                    }
                    columns.append(oriented)
                    entries.append(metrics[metric])
                by_comparison[comparison] = {"metrics": metrics}
            by_count[str(negative_count)] = {"comparisons": by_comparison}
        output[f"uniform_seed_{candidate_seed}"] = {"negative_counts": by_count}
    columns_array = np.stack(columns, axis=1)
    if device is None:
        sampled = (weights @ columns_array) / bootstrap_denominators[:, None]
    else:
        sampled = (
            torch.from_numpy(weights)
            .to(device=device, dtype=torch.float64)
            .matmul(
                torch.from_numpy(columns_array).to(
                    device=device, dtype=torch.float64
                )
            )
            .div(
                torch.from_numpy(bootstrap_denominators)
                .to(device=device, dtype=torch.float64)
                .unsqueeze(1)
            )
            .cpu()
            .numpy()
        )
    intervals = np.quantile(sampled, [0.025, 0.975], axis=0)
    for index, entry in enumerate(entries):
        interval = [float(intervals[0, index]), float(intervals[1, index])]
        entry["user_cluster_95_interval"] = interval
        entry["positive_with_ci"] = bool(interval[0] > 0.0)
        entry["negative_with_ci"] = bool(interval[1] < 0.0)
    return {
        "users": len(records),
        "positive_targets": int(denominator),
        "user_ids_sha256": hashlib.sha256(
            np.asarray(
                sorted(value["user_id"] for value in records), dtype="<i8"
            ).tobytes()
        ).hexdigest(),
        "bootstrap": {
            "unit": "user_cluster",
            "samples": int(evaluation["bootstrap_samples"]),
            "shared_weights_across_metrics": True,
        },
        "candidate_variants": output,
    }


def _summary_grid(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    seed_offset: int,
    device: torch.device | None,
) -> dict[str, Any]:
    output = {}
    for cohort, minimum_prefix in config["analysis"]["prefix_cohorts"].items():
        cohort_records = [
            value for value in records if value["prefix_length"] >= minimum_prefix
        ]
        if not cohort_records:
            raise RuntimeError(f"KuaiRand K/V cohort is empty: {cohort}")
        scopes = {}
        for scope in config["analysis"]["target_scopes"]:
            normalized = [
                {
                    "user_id": value["user_id"],
                    "targets": value["scopes"][scope]["targets"],
                    "metric_sums": value["scopes"][scope]["metric_sums"],
                }
                for value in cohort_records
            ]
            summary = _summarize_records(
                normalized,
                config,
                seed_offset + minimum_prefix * 1009 + len(scopes) * 10_007,
                device,
            )
            scopes[scope] = {
                "summary": summary,
                "metric_family": _metric_family(summary, config),
            }
        output[cohort] = {"minimum_prefix_length": minimum_prefix, "scopes": scopes}
    return output


def _primary(grid: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    primary = config["analysis"]["primary"]
    return grid[primary["prefix_cohort"]]["scopes"][primary["target_scope"]][
        "metric_family"
    ]


def _render_table(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# KuaiRand K/V-only strength screen",
        "",
        "Primary workload is the first four engaged next-day targets for users with prefix length at least 256. Selection uses tuning users only.",
        "",
        "| candidate | K/V update | tuning pairwise | holdout pairwise | holdout NDCG@10 | holdout MRR | holdout update pairwise |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for value in candidates:
        tuning = value["tuning_primary"]
        holdout = value["holdout_primary"]
        metrics = [
            tuning["recompute_over_reuse"]["pairwise_win_rate"][
                "minimum_relative_percent"
            ],
            holdout["recompute_over_reuse"]["pairwise_win_rate"][
                "mean_relative_percent"
            ],
            holdout["recompute_over_reuse"]["ndcg_at_10"][
                "mean_relative_percent"
            ],
            holdout["recompute_over_reuse"]["mrr"]["mean_relative_percent"],
            holdout["current_over_previous"]["pairwise_win_rate"][
                "mean_relative_percent"
            ],
        ]
        rendered = ["NA" if metric is None else f"{metric:+.3f}%" for metric in metrics]
        distance = value["parameter_group_distances"]["kv_projections"][
            "relative_l2_update"
        ]
        lines.append(
            f"| {value['candidate']['id']} | {distance:.3f} | "
            + " | ".join(rendered)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _finalize_evaluation(
    config_path: str | Path,
    config: dict[str, Any],
    training: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    if (
        raw.get("protocol") != PROTOCOL
        or raw.get("status") != "complete_raw_inference"
        or raw.get("config", {}).get("sha256") != file_sha256(config_path)
        or [value.get("candidate") for value in raw.get("candidates", [])]
        != config["candidates"]
    ):
        raise ValueError("KuaiRand K/V raw inference binding differs")
    started = time.perf_counter()
    candidate_results = []
    for value in raw["candidates"]:
        candidate_index = int(value["candidate_index"])
        combined = value["records"]
        candidate = value["candidate"]
        print(
            f"phase=kuairand_kv_strength_aggregate candidate={candidate['id']} start",
            flush=True,
        )
        tuning = _summary_grid(
            [record for record in combined if record["split"] == "tuning"],
            config,
            candidate_index * 1_000_003,
            None,
        )
        holdout = _summary_grid(
            [record for record in combined if record["split"] == "holdout"],
            config,
            candidate_index * 1_000_003 + 500_009,
            None,
        )
        candidate_results.append(
            {
                "candidate": candidate,
                "checkpoint": value["checkpoint"],
                "parameter_group_distances": value["parameter_group_distances"],
                "tuning": tuning,
                "tuning_primary": _primary(tuning, config),
                "holdout": holdout,
                "holdout_primary": _primary(holdout, config),
            }
        )
        print(
            f"phase=kuairand_kv_strength_aggregate candidate={candidate['id']} complete",
            flush=True,
        )
    admissible = []
    for value in candidate_results:
        metrics = value["tuning_primary"]
        if all(
            metrics[comparison][metric]["positive_all_seeds"]
            for comparison in ("recompute_over_reuse", "current_over_previous")
            for metric in ("candidate_cross_entropy", "pairwise_win_rate")
        ):
            admissible.append(value)
    selected_candidate = (
        max(
            admissible,
            key=lambda value: value["tuning_primary"]["recompute_over_reuse"][
                "pairwise_win_rate"
            ]["minimum_relative_percent"],
        )["candidate"]["id"]
        if admissible
        else None
    )
    result = {
        "protocol": PROTOCOL,
        "status": "complete_development_measurement",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": raw["data"],
        "source": training["source"],
        "evaluation": raw["evaluation"],
        "candidates": candidate_results,
        "selection": {
            "uses": "primary tuning cohort only",
            "criterion": "both stale and model-update CE/pairwise positive on all three seeds, then maximize minimum-seed pairwise Recompute advantage",
            "selected_candidate": selected_candidate,
        },
        "inference_elapsed_seconds": raw["inference_elapsed_seconds"],
        "aggregation_elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(config["outputs"]["evaluation_result"])
    _atomic_json(output, result)
    table = Path(config["outputs"]["table"])
    table.parent.mkdir(parents=True, exist_ok=True)
    temporary = table.with_suffix(table.suffix + ".tmp")
    temporary.write_text(_render_table(candidate_results))
    temporary.replace(table)
    return result


def run_kv_strength_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_kv_strength_config(config_path)
    training = json.loads(Path(config["outputs"]["training_result"]).read_text())
    runtime = init_distributed_runtime("cuda:0")
    runtime_closed = False
    try:
        if runtime.world_size != int(config["execution"]["evaluation_world_size"]):
            raise ValueError("KuaiRand K/V strength evaluation world size differs")
        output = Path(config["outputs"]["evaluation_result"])
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        raw_path = output.with_name("raw_inference.pt")
        if raw_path.is_file():
            is_primary = runtime.is_primary
            dist.barrier()
            close_distributed_runtime(runtime)
            runtime_closed = True
            if not is_primary:
                return None
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            return _finalize_evaluation(config_path, config, training, raw)
        document = _effective_document(config)
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in (14, 15):
            plan.ingest_day(dates[date_index])
        evaluation = config["evaluation"]
        selected, eligible = _selected_users(
            plan,
            dates[int(config["update_date_index"])],
            dates[int(config["evaluation_date_index"])],
            int(evaluation["record_limit_per_rank"]) * runtime.world_size,
            int(evaluation["sampling_seed"]),
            True,
        )
        generator = np.random.default_rng(int(evaluation["split_seed"]))
        permutation = generator.permutation(len(selected))
        tuning_count = int(round(len(selected) * float(evaluation["tuning_fraction"])))
        tuning_users = set(np.asarray(selected)[permutation[:tuning_count]].tolist())
        local_users = selected[runtime.rank :: runtime.world_size]
        source = _load_checkpoint_model(
            document, plan, Path(config["source"]["theta1"]["path"]), runtime.device
        )
        counts = [int(value) for value in evaluation["negative_counts"]]
        seeds = [int(value) for value in evaluation["uniform_candidate_seeds"]]
        started = time.perf_counter()
        raw_candidate_results = []
        for candidate_index, candidate in enumerate(config["candidates"]):
            current = _load_checkpoint_model(
                document,
                plan,
                _checkpoint(config, str(candidate["id"])),
                runtime.device,
            )
            distances = _kv_distance(source, current)
            local_records = []
            for ordinal, user in enumerate(local_users):
                sequence = _evaluation_sequence(
                    plan, int(user), dates[int(config["evaluation_date_index"])]
                )
                positives = torch.from_numpy(
                    sequence["targets"][sequence["labels"]]
                ).long()
                fresh = _run_suffix(
                    current,
                    _stored_cache(current, sequence["prefix"], runtime.device),
                    sequence["suffix"],
                    sequence["labels"],
                    64,
                    runtime.device,
                )
                reuse = _run_suffix(
                    current,
                    _stored_cache(source, sequence["prefix"], runtime.device),
                    sequence["suffix"],
                    sequence["labels"],
                    64,
                    runtime.device,
                )
                previous = _run_suffix(
                    source,
                    _stored_cache(source, sequence["prefix"], runtime.device),
                    sequence["suffix"],
                    sequence["labels"],
                    64,
                    runtime.device,
                )
                scopes = {}
                for scope, limit in config["analysis"]["target_scopes"].items():
                    stop = len(positives) if limit is None else min(limit, len(positives))
                    local_positives = positives[:stop]
                    candidate_sets = _candidate_sets(
                        local_positives,
                        current.cfg.num_prediction_items,
                        max(counts),
                        seeds,
                    )
                    scopes[scope] = {
                        "targets": stop,
                        "metric_sums": np.stack(
                            [
                                _candidate_metric_sums(
                                    current,
                                    hidden[:stop],
                                    candidate_sets,
                                    counts,
                                    64,
                                    runtime.device,
                                )
                                for hidden in (fresh, reuse, previous)
                            ]
                        ).tolist(),
                    }
                local_records.append(
                    {
                        "user_id": int(user),
                        "split": "tuning" if user in tuning_users else "holdout",
                        "prefix_length": len(sequence["prefix"]["item_ids"]),
                        "scopes": scopes,
                    }
                )
                if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(local_users):
                    print(
                        f"phase=kuairand_kv_strength_eval candidate={candidate['id']} "
                        f"rank={runtime.rank} users={ordinal + 1}/{len(local_users)}",
                        flush=True,
                    )
            gathered: list[Any] | None = (
                [None] * runtime.world_size if runtime.is_primary else None
            )
            dist.gather_object(local_records, gathered, dst=0)
            if runtime.is_primary:
                combined = [value for shard in gathered for value in shard]
                raw_candidate_results.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate": candidate,
                        "checkpoint": {
                            "path": str(_checkpoint(config, str(candidate["id"]))),
                            "sha256": training["candidates"][candidate_index][
                                "model_sha256"
                            ],
                        },
                        "parameter_group_distances": distances,
                        "records": combined,
                    }
                )
            del current, local_records
            gc.collect()
            torch.cuda.empty_cache()
        is_primary = runtime.is_primary
        dist.barrier()
        close_distributed_runtime(runtime)
        runtime_closed = True
        if not is_primary:
            return None
        raw = {
            "protocol": PROTOCOL,
            "status": "complete_raw_inference",
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data": metadata,
            "evaluation": {
                "eligible_users": eligible,
                "selected_users": len(selected),
                "tuning_users": len(tuning_users),
                "holdout_users": len(selected) - len(tuning_users),
                "analysis": config["analysis"],
                "negative_counts": counts,
                "uniform_candidate_seeds": seeds,
                "metrics": list(METRICS),
            },
            "candidates": raw_candidate_results,
            "inference_elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_torch(raw_path, raw)
        return _finalize_evaluation(config_path, config, training, raw)
    finally:
        if not runtime_closed:
            close_distributed_runtime(runtime)
