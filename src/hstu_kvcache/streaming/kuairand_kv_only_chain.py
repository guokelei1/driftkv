from __future__ import annotations

import gc
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
from .kuairand_history_path_screen import _select_parameters
from .kuairand_next_item_triangle import METRICS, _comparison, _summaries, _table
from .kuairand_root_cause import (
    _atomic_json,
    _atomic_torch,
    _evaluate_edge,
    _popularity_ranks,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)

PROTOCOL = "evokv_kuairand_kv_only_chain_v0"


def load_kv_only_chain_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    training = document.get("training", {})
    evaluation = document.get("evaluation", {})
    execution = document.get("execution", {})
    outputs = document.get("outputs", {})
    bindings = [
        source.get("base_config", {}),
        source.get("theta1", {}),
        source.get("theta2", {}),
    ]
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("total_num_days") != 23
        or training.get("source_version") != 2
        or training.get("target_versions") != [3, 4, 5, 6, 7, 8]
        or training.get("update_date_indices") != [16, 17, 18, 19, 20, 21]
        or training.get("parameter_group") != "kv"
        or int(training.get("epochs_per_update", 0)) != 2
        or float(training.get("lr", 0.0)) != 0.0001
        or float(training.get("weight_decay", -1.0)) != 0.0001
        or training.get("optimizer_lifecycle") != "fresh_adamw_per_update"
        or evaluation.get("target_versions") != [2, 3, 4, 5, 6, 7, 8]
        or evaluation.get("methods")
        != ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"]
        or int(evaluation.get("record_limit_per_rank", 0)) < 1
        or execution.get("training_cuda_visible_devices") != "0"
        or execution.get("evaluation_cuda_visible_devices") != "0,1"
        or execution.get("evaluation_world_size") != 2
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
        or not all(
            isinstance(outputs.get(name), str)
            for name in ("checkpoint_root", "training_result", "evaluation_result", "table")
        )
    ):
        raise ValueError("KuaiRand KV-only chain config differs")
    return document


def _effective_document(config: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(config["source"]["base_config"]["path"]).read_text())
    document["data"]["total_num_days"] = int(config["total_num_days"])
    document["interventions"]["methods"] = list(config["evaluation"]["methods"])
    document["quality"]["record_limit_per_rank"] = int(
        config["evaluation"]["record_limit_per_rank"]
    )
    document["quality"]["cap_user_limit_to_eligible"] = False
    document["quality"]["bootstrap_samples"] = int(
        config["evaluation"]["bootstrap_samples"]
    )
    document["quality"]["bootstrap_seed"] = int(
        config["evaluation"]["bootstrap_seed"]
    )
    return document


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _chain_checkpoint(config: dict[str, Any], version: int) -> Path:
    return Path(config["outputs"]["checkpoint_root"]) / f"theta_{version}" / "model.pt"


def _checkpoint(config: dict[str, Any], version: int) -> Path:
    if version == 1:
        return Path(config["source"]["theta1"]["path"])
    if version == 2:
        return Path(config["source"]["theta2"]["path"])
    if 3 <= version <= 8:
        return _chain_checkpoint(config, version)
    raise ValueError("KuaiRand KV-only checkpoint version differs")


def run_kv_only_chain_training(config_path: str | Path) -> dict[str, Any]:
    config = load_kv_only_chain_config(config_path)
    output = Path(config["outputs"]["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand KV-only chain training requires one rank")
    document = _effective_document(config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for date_index in (14, 15):
        plan.ingest_day(dates[date_index])
    model = make_model(document, plan, device)
    model.load_state_dict(
        torch.load(config["source"]["theta2"]["path"], map_location="cpu", weights_only=True)
    )
    parameters, parameter_names = _select_parameters(model, "kv")
    started = time.perf_counter()
    records = []
    training = config["training"]
    for version, date_index in zip(
        training["target_versions"], training["update_date_indices"], strict=True
    ):
        date = dates[int(date_index)]
        plan.ingest_day(date)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        epochs = []
        for epoch in range(int(training["epochs_per_update"])):
            _seed(int(training["seed"]) + int(version) * 1009 + epoch)
            batches = plan.iter_train_batches(
                date,
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
                    f"kuairand_kv_only_theta{version}_epoch{epoch + 1}",
                )
            )
        model_path = _chain_checkpoint(config, int(version))
        if model_path.exists():
            raise FileExistsError(f"KuaiRand KV-only checkpoint exists: {model_path}")
        _atomic_torch(model_path, model.state_dict())
        record = {
            "protocol": PROTOCOL,
            "status": "complete_development_checkpoint",
            "scientific_result": False,
            "formal_result": False,
            "version": int(version),
            "date": date,
            "config_sha256": file_sha256(config_path),
            "parent_model_sha256": file_sha256(_checkpoint(config, int(version) - 1)),
            "model_sha256": file_sha256(model_path),
            "model": asdict(model.cfg),
            "trainable_parameter_names": parameter_names,
            "trainable_parameters": sum(value.numel() for value in parameters),
            "training": epochs,
        }
        _atomic_json(model_path.with_name("manifest.json"), record)
        records.append(record)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "protocol": PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "model": asdict(model.cfg),
        "parameter_group": "kv",
        "optimizer_lifecycle": training["optimizer_lifecycle"],
        "versions": records,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def _triangle_decision(cells: list[dict[str, Any]], age_summary: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for metric in ("cross_entropy", "mrr", "ndcg_at_10"):
        values = [value["metrics"][metric] for value in cells]
        output[metric] = {
            "positive_cells": sum(value["relative_percent"] > 0.0 for value in values),
            "positive_cells_with_ci": sum(value["positive_with_ci"] for value in values),
            "negative_cells_with_ci": sum(
                value["relative_percent"] < 0.0
                and value["user_cluster_95_interval"][1] < 0.0
                for value in values
            ),
            "age_mean_relative_percent": {
                age: row["mean_relative_percent"]
                for age, row in age_summary[metric].items()
            },
        }
    return output


def run_kv_only_chain_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_kv_only_chain_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand KV-only chain evaluation requires two ranks")
    output = Path(config["outputs"]["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = _effective_document(config)
        torch.set_float32_matmul_precision("high")
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in (14, 15):
            plan.ingest_day(dates[date_index])
        popularity = _popularity_ranks(plan)
        started = time.perf_counter()
        cells = []
        user_hashes: dict[str, set[str]] = {}
        for target in config["evaluation"]["target_versions"]:
            if int(target) > 2:
                plan.ingest_day(dates[13 + int(target)])
            current = make_model(document, plan, runtime.device)
            current.load_state_dict(
                torch.load(_checkpoint(config, int(target)), map_location="cpu", weights_only=True)
            )
            current.eval()
            target_hashes = set()
            for source in range(1, int(target)):
                previous = make_model(document, plan, runtime.device)
                previous.load_state_dict(
                    torch.load(_checkpoint(config, source), map_location="cpu", weights_only=True)
                )
                previous.eval()
                edge = _evaluate_edge(
                    document,
                    plan,
                    previous,
                    current,
                    int(target),
                    dates[13 + int(target)],
                    dates[14 + int(target)],
                    popularity,
                    runtime,
                )
                if runtime.is_primary:
                    assert edge is not None
                    target_hashes.add(edge["selected_user_ids_sha256"])
                    cells.append(
                        {
                            "target_version": int(target),
                            "source_version": source,
                            "cache_age": int(target) - source,
                            "update_date": dates[13 + int(target)],
                            "evaluation_date": dates[14 + int(target)],
                            "metrics": {
                                metric: _comparison(edge, metric) for metric in METRICS
                            },
                            "edge": edge,
                        }
                    )
                del previous
                gc.collect()
                torch.cuda.empty_cache()
            if runtime.is_primary:
                user_hashes[str(target)] = target_hashes
            del current
            gc.collect()
            torch.cuda.empty_cache()
        if not runtime.is_primary:
            dist.barrier()
            return None
        if any(len(values) != 1 for values in user_hashes.values()):
            raise RuntimeError("KuaiRand KV-only target user sets differ")
        if not all(value["edge"]["sanity"]["implementation_passed"] for value in cells):
            raise RuntimeError("KuaiRand KV-only triangle sanity failed")
        matrices, age_summary = _summaries(cells)
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data": metadata,
            "cells": cells,
            "matrices": matrices,
            "age_summary": age_summary,
            "decision": _triangle_decision(cells, age_summary),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        table = Path(config["outputs"]["table"])
        table.parent.mkdir(parents=True, exist_ok=True)
        temporary = table.with_suffix(table.suffix + ".tmp")
        temporary.write_text(_table(matrices))
        temporary.replace(table)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)
