from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
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
    _parameter_distances,
    _run_suffix,
    _selected_users,
    _stored_cache,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)

PROTOCOL = "evokv_kuairand_untied_cache_path_screen_v0"
EXPECTED_CANDIDATES = [
    {"id": "cp_lr100_e2", "lr": 0.0001, "epochs": 2},
    {"id": "cp_lr200_e2", "lr": 0.0002, "epochs": 2},
    {"id": "cp_lr500_e2", "lr": 0.0005, "epochs": 2},
    {"id": "cp_lr1000_e2", "lr": 0.001, "epochs": 2},
    {"id": "cp_lr500_e4", "lr": 0.0005, "epochs": 4},
]
METHODS = ("recompute", "reuse", "previous_fresh")


def load_untied_screen_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    training = document.get("training", {})
    evaluation = document.get("evaluation", {})
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
        or training.get("parameter_group") != "cache_producer"
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
        or execution.get("training_cuda_visible_devices") != "0"
        or execution.get("evaluation_cuda_visible_devices") != "0,1"
        or execution.get("evaluation_world_size") != 2
        or file_sha256(source.get("base_config", {}).get("path", ""))
        != source.get("base_config", {}).get("sha256")
        or file_sha256(source.get("tied_theta1", {}).get("path", ""))
        != source.get("tied_theta1", {}).get("sha256")
        or not all(
            isinstance(outputs.get(name), str)
            for name in ("checkpoint_root", "training_result", "evaluation_result", "table")
        )
    ):
        raise ValueError("KuaiRand untied cache-path screen config differs")
    return document


def _effective_document(config: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(config["source"]["base_config"]["path"]).read_text())
    document["model"]["tie_item_embeddings"] = False
    document["data"]["total_num_days"] = 17
    return document


def _checkpoint(config: dict[str, Any], candidate: str | None, version: int) -> Path:
    root = Path(config["outputs"]["checkpoint_root"])
    if version == 1:
        return root / "theta_1" / "model.pt"
    if version == 2 and candidate is not None:
        return root / candidate / "theta_2" / "model.pt"
    raise ValueError("KuaiRand untied checkpoint binding differs")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _untied_source_state(config: dict[str, Any], model) -> dict[str, torch.Tensor]:
    tied = torch.load(
        config["source"]["tied_theta1"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    state = model.state_dict()
    for name, value in tied.items():
        state[name] = value
    state["output_emb.weight"] = tied["item_emb.weight"][
        : model.cfg.num_prediction_items + 1
    ].clone()
    return state


def _cache_producer_parameters(model) -> tuple[list[torch.nn.Parameter], list[str]]:
    last = model.cfg.num_layers - 1
    parameters = []
    names = []
    for name, parameter in model.named_parameters():
        excluded = name == "output_emb.weight" or name == "final_norm.weight"
        excluded = excluded or name.startswith(
            (
                f"blocks.{last}.attn.q_proj.",
                f"blocks.{last}.attn.out_proj.",
                f"blocks.{last}.gate_proj.",
            )
        )
        parameter.requires_grad_(not excluded)
        if not excluded:
            parameters.append(parameter)
            names.append(name)
    if not parameters or "item_emb.weight" not in names or "output_emb.weight" in names:
        raise RuntimeError("KuaiRand cache-producer parameter selection differs")
    return parameters, names


def run_untied_screen_training(config_path: str | Path) -> dict[str, Any]:
    config = load_untied_screen_config(config_path)
    output = Path(config["outputs"]["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand untied cache-path training requires one rank")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    document = _effective_document(config)
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    plan.ingest_day(dates[14])
    source_histories = copy.deepcopy(plan.user_histories)
    source_model = make_model(document, plan, device)
    source_state = _untied_source_state(config, source_model)
    source_model.load_state_dict(source_state)
    source_path = _checkpoint(config, None, 1)
    source_manifest_path = source_path.with_name("manifest.json")
    source_manifest = (
        json.loads(source_manifest_path.read_text())
        if source_manifest_path.is_file()
        else None
    )
    if source_path.is_file() and source_manifest is not None:
        if (
            source_manifest.get("protocol") != PROTOCOL
            or source_manifest.get("status") != "complete_derived_bootstrap"
            or source_manifest.get("version") != 1
            or source_manifest.get("source_tied_sha256")
            != config["source"]["tied_theta1"]["sha256"]
            or source_manifest.get("model_sha256") != file_sha256(source_path)
            or source_manifest.get("model") != asdict(source_model.cfg)
        ):
            raise ValueError("KuaiRand untied theta1 resume binding differs")
    elif source_path.exists() or source_manifest_path.exists():
        raise FileExistsError("KuaiRand untied theta1 checkpoint is partial")
    else:
        _atomic_torch(source_path, source_model.state_dict())
        _atomic_json(
            source_manifest_path,
            {
            "protocol": PROTOCOL,
            "status": "complete_derived_bootstrap",
            "scientific_result": False,
            "formal_result": False,
            "version": 1,
            "source_tied_sha256": config["source"]["tied_theta1"]["sha256"],
            "model_sha256": file_sha256(source_path),
            "model": asdict(source_model.cfg),
            "construction": "input embedding copied from tied theta1; frozen output embedding initialized from the same prediction rows",
            },
        )
    del source_model
    started = time.perf_counter()
    records = []
    training = config["training"]
    for candidate_index, candidate in enumerate(config["candidates"]):
        plan.user_histories = copy.deepcopy(source_histories)
        update_date = dates[int(config["update_date_index"])]
        plan.ingest_day(update_date)
        model = make_model(document, plan, device)
        model.load_state_dict(source_state)
        parameters, parameter_names = _cache_producer_parameters(model)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(candidate["lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        epochs = []
        for epoch in range(int(candidate["epochs"])):
            _seed(int(training["seed"]) + candidate_index * 1009 + epoch)
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
                    f"kuairand_untied_{candidate['id']}_epoch{epoch + 1}",
                )
            )
        model_path = _checkpoint(config, str(candidate["id"]), 2)
        if model_path.exists():
            raise FileExistsError(f"KuaiRand untied candidate exists: {model_path}")
        _atomic_torch(model_path, model.state_dict())
        record = {
            "protocol": PROTOCOL,
            "status": "complete_development_checkpoint",
            "scientific_result": False,
            "formal_result": False,
            "candidate": candidate,
            "candidate_index": candidate_index,
            "version": 2,
            "date": update_date,
            "model_sha256": file_sha256(model_path),
            "source_model_sha256": file_sha256(source_path),
            "model": asdict(model.cfg),
            "parameter_group": training["parameter_group"],
            "trainable_parameter_names": parameter_names,
            "trainable_parameters": sum(value.numel() for value in parameters),
            "training": epochs,
        }
        _atomic_json(model_path.with_name("manifest.json"), record)
        records.append(record)
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
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "candidates": records,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def _load_checkpoint_model(document, plan, path: Path, device):
    model = make_model(document, plan, device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _bootstrap(
    targets: np.ndarray,
    oriented_sums: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    weights = generator.multinomial(
        len(targets),
        np.full(len(targets), 1.0 / len(targets), dtype=np.float64),
        size=samples,
    )
    values = (weights @ oriented_sums) / (weights @ targets)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def _summarize_records(
    records: list[dict[str, Any]], config: dict[str, Any], seed_offset: int
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    seeds = [int(value) for value in evaluation["uniform_candidate_seeds"]]
    counts = [int(value) for value in evaluation["negative_counts"]]
    values = np.stack([np.asarray(value["metric_sums"]) for value in records])
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    output: dict[str, Any] = {}
    comparisons = {
        "recompute_over_reuse": (0, 1),
        "current_over_previous": (0, 2),
    }
    for variant_index, candidate_seed in enumerate(seeds):
        by_count = {}
        for count_index, negative_count in enumerate(counts):
            by_comparison = {}
            for comparison_index, (comparison, (current_index, baseline_index)) in enumerate(
                comparisons.items()
            ):
                metrics = {}
                for metric_index, metric in enumerate(METRICS):
                    current = values[:, current_index, variant_index, count_index, metric_index]
                    baseline = values[:, baseline_index, variant_index, count_index, metric_index]
                    oriented = (
                        baseline - current
                        if metric in LOWER_IS_BETTER
                        else current - baseline
                    )
                    current_mean = float(current.sum() / denominator)
                    baseline_mean = float(baseline.sum() / denominator)
                    advantage = float(oriented.sum() / denominator)
                    interval = _bootstrap(
                        targets,
                        oriented,
                        int(evaluation["bootstrap_samples"]),
                        int(evaluation["bootstrap_seed"])
                        + seed_offset
                        + variant_index * 101
                        + count_index * 17
                        + comparison_index * 1009
                        + metric_index,
                    )
                    metrics[metric] = {
                        "current": current_mean,
                        "baseline": baseline_mean,
                        "advantage_absolute": advantage,
                        "advantage_relative_percent": (
                            100.0 * advantage / abs(baseline_mean)
                            if baseline_mean
                            else None
                        ),
                        "user_cluster_95_interval": interval,
                        "positive_with_ci": bool(interval[0] > 0.0),
                        "negative_with_ci": bool(interval[1] < 0.0),
                    }
                by_comparison[comparison] = {"metrics": metrics}
            by_count[str(negative_count)] = {"comparisons": by_comparison}
        output[f"uniform_seed_{candidate_seed}"] = {"negative_counts": by_count}
    return {
        "users": len(records),
        "positive_targets": int(denominator),
        "user_ids_sha256": hashlib.sha256(
            np.asarray(sorted(value["user_id"] for value in records), dtype="<i8").tobytes()
        ).hexdigest(),
        "candidate_variants": output,
    }


def _primary_score(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    values = []
    update_ce = []
    stale_ce = []
    for seed in config["evaluation"]["uniform_candidate_seeds"]:
        row = summary["candidate_variants"][f"uniform_seed_{seed}"]["negative_counts"][
            "999"
        ]["comparisons"]
        values.append(
            row["recompute_over_reuse"]["metrics"]["pairwise_win_rate"][
                "advantage_relative_percent"
            ]
        )
        stale_ce.append(
            row["recompute_over_reuse"]["metrics"]["candidate_cross_entropy"][
                "advantage_absolute"
            ]
        )
        update_ce.append(
            row["current_over_previous"]["metrics"]["candidate_cross_entropy"][
                "advantage_absolute"
            ]
        )
    return {
        "pairwise_relative_percent_by_seed": values,
        "minimum_pairwise_relative_percent": min(values),
        "mean_pairwise_relative_percent": float(np.mean(values)),
        "stale_ce_all_positive": all(value > 0.0 for value in stale_ce),
        "current_over_previous_ce_all_positive": all(value > 0.0 for value in update_ce),
    }


def _render_table(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# KuaiRand untied cache-producer screen",
        "",
        "Primary diagnostic is 999-negative pairwise Recompute-over-Reuse. Candidate selection uses tuning users only; holdout values are reported after selection.",
        "",
        "| candidate | LR | epochs | tuning min/mean | tuning update CE+ | holdout min/mean | holdout update CE+ |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for value in candidates:
        candidate = value["candidate"]
        tuning = value["tuning_primary"]
        holdout = value["holdout_primary"]
        lines.append(
            "| {name} | {lr:.4g} | {epochs} | {tmin:+.3f}%/{tmean:+.3f}% | {tupdate} | {hmin:+.3f}%/{hmean:+.3f}% | {hupdate} |".format(
                name=candidate["id"],
                lr=candidate["lr"],
                epochs=candidate["epochs"],
                tmin=tuning["minimum_pairwise_relative_percent"],
                tmean=tuning["mean_pairwise_relative_percent"],
                tupdate=tuning["current_over_previous_ce_all_positive"],
                hmin=holdout["minimum_pairwise_relative_percent"],
                hmean=holdout["mean_pairwise_relative_percent"],
                hupdate=holdout["current_over_previous_ce_all_positive"],
            )
        )
    return "\n".join(lines) + "\n"


def run_untied_screen_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_untied_screen_config(config_path)
    training = json.loads(Path(config["outputs"]["training_result"]).read_text())
    runtime = init_distributed_runtime("cuda:0")
    try:
        if runtime.world_size != int(config["execution"]["evaluation_world_size"]):
            raise ValueError("KuaiRand untied evaluation world size differs")
        output = Path(config["outputs"]["evaluation_result"])
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
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
        split_generator = np.random.default_rng(int(evaluation["split_seed"]))
        permutation = split_generator.permutation(len(selected))
        tuning_count = int(round(len(selected) * float(evaluation["tuning_fraction"])))
        tuning_users = set(np.asarray(selected)[permutation[:tuning_count]].tolist())
        local_users = selected[runtime.rank :: runtime.world_size]
        source = _load_checkpoint_model(
            document, plan, _checkpoint(config, None, 1), runtime.device
        )
        counts = [int(value) for value in evaluation["negative_counts"]]
        seeds = [int(value) for value in evaluation["uniform_candidate_seeds"]]
        started = time.perf_counter()
        candidate_results = []
        for candidate_index, candidate in enumerate(config["candidates"]):
            current = _load_checkpoint_model(
                document,
                plan,
                _checkpoint(config, str(candidate["id"]), 2),
                runtime.device,
            )
            distances = _parameter_distances(source, current)
            if distances["output_item_embedding"]["relative_l2_update"] != 0.0:
                raise RuntimeError("KuaiRand untied output embedding changed")
            local_records = []
            for ordinal, user in enumerate(local_users):
                sequence = _evaluation_sequence(
                    plan, int(user), dates[int(config["evaluation_date_index"])]
                )
                positives = torch.from_numpy(
                    sequence["targets"][sequence["labels"]]
                ).long()
                candidates = _candidate_sets(
                    positives,
                    current.cfg.num_prediction_items,
                    max(counts),
                    seeds,
                )
                current_fresh = _run_suffix(
                    current,
                    _stored_cache(current, sequence["prefix"], runtime.device),
                    sequence["suffix"],
                    sequence["labels"],
                    64,
                    runtime.device,
                )
                current_reuse = _run_suffix(
                    current,
                    _stored_cache(source, sequence["prefix"], runtime.device),
                    sequence["suffix"],
                    sequence["labels"],
                    64,
                    runtime.device,
                )
                previous_fresh = _run_suffix(
                    source,
                    _stored_cache(source, sequence["prefix"], runtime.device),
                    sequence["suffix"],
                    sequence["labels"],
                    64,
                    runtime.device,
                )
                metric_sums = np.stack(
                    [
                        _candidate_metric_sums(
                            current,
                            hidden,
                            candidates,
                            counts,
                            64,
                            runtime.device,
                        )
                        for hidden in (current_fresh, current_reuse, previous_fresh)
                    ]
                )
                local_records.append(
                    {
                        "user_id": int(user),
                        "split": "tuning" if user in tuning_users else "holdout",
                        "targets": len(positives),
                        "metric_sums": metric_sums.tolist(),
                    }
                )
                if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(local_users):
                    print(
                        f"phase=kuairand_untied_eval candidate={candidate['id']} "
                        f"rank={runtime.rank} users={ordinal + 1}/{len(local_users)}",
                        flush=True,
                    )
            gathered: list[Any] | None = (
                [None] * runtime.world_size if runtime.is_primary else None
            )
            dist.gather_object(local_records, gathered, dst=0)
            if runtime.is_primary:
                combined = [value for shard in gathered for value in shard]
                tuning = _summarize_records(
                    [value for value in combined if value["split"] == "tuning"],
                    config,
                    candidate_index * 100_003,
                )
                holdout = _summarize_records(
                    [value for value in combined if value["split"] == "holdout"],
                    config,
                    candidate_index * 100_003 + 50_021,
                )
                candidate_results.append(
                    {
                        "candidate": candidate,
                        "checkpoint": {
                            "path": str(_checkpoint(config, str(candidate["id"]), 2)),
                            "sha256": training["candidates"][candidate_index]["model_sha256"],
                        },
                        "parameter_group_distances": distances,
                        "tuning": tuning,
                        "tuning_primary": _primary_score(tuning, config),
                        "holdout": holdout,
                        "holdout_primary": _primary_score(holdout, config),
                    }
                )
            del current, local_records
            gc.collect()
            torch.cuda.empty_cache()
        if not runtime.is_primary:
            dist.barrier()
            return None
        admissible = [
            value
            for value in candidate_results
            if value["tuning_primary"]["stale_ce_all_positive"]
            and value["tuning_primary"]["current_over_previous_ce_all_positive"]
        ]
        selected_candidate = (
            max(
                admissible,
                key=lambda value: value["tuning_primary"][
                    "minimum_pairwise_relative_percent"
                ],
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
            "data": metadata,
            "source": training["source"],
            "evaluation": {
                "eligible_users": eligible,
                "selected_users": len(selected),
                "tuning_users": len(tuning_users),
                "holdout_users": len(selected) - len(tuning_users),
                "split_seed": evaluation["split_seed"],
                "negative_counts": counts,
                "uniform_candidate_seeds": seeds,
                "metrics": list(METRICS),
            },
            "candidates": candidate_results,
            "selection": {
                "uses": "tuning users only",
                "criterion": "stale CE and current-over-previous CE positive on all three 999-negative seeds, then maximize minimum seed pairwise Recompute advantage",
                "selected_candidate": selected_candidate,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        table = Path(config["outputs"]["table"])
        table.parent.mkdir(parents=True, exist_ok=True)
        temporary = table.with_suffix(table.suffix + ".tmp")
        temporary.write_text(_render_table(candidate_results))
        temporary.replace(table)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)
