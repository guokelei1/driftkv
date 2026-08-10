from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_next_item_chain import NEXT_ITEM_CHAIN_PROTOCOL, _move_optimizer
from .kuairand_root_cause import (
    PROTOCOL,
    _atomic_json,
    _atomic_torch,
    _evaluate_edge,
    _load_checkpoint,
    _popularity_ranks,
    _save_checkpoint,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)

TRIANGLE_PROTOCOL = "evokv_kuairand_causal_next_item_triangle_v0"
METRICS = (
    "cross_entropy",
    "mrr",
    "ndcg_at_10",
    "hit_rate_at_50",
    "hit_rate_at_200",
)


def load_triangle_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    bindings = [
        source.get("base_config", {}),
        source.get("chain_config", {}),
        source.get("chain_resume", {}),
        source.get("theta7", {}),
    ]
    training = document.get("training", {})
    evaluation = document.get("evaluation", {})
    outputs = document.get("outputs", {})
    if (
        document.get("protocol") != TRIANGLE_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or int(document.get("total_num_days", 0)) != 23
        or training.get("source_version") != 7
        or training.get("target_version") != 8
        or training.get("update_date_index") != 21
        or int(training.get("epochs", 0)) < 1
        or evaluation.get("target_versions") != [2, 3, 4, 5, 6, 7, 8]
        or evaluation.get("methods")
        != ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"]
        or int(evaluation.get("record_limit_per_rank", 0)) < 1
        or not isinstance(outputs.get("checkpoint_root"), str)
        or not isinstance(outputs.get("training_result"), str)
        or not isinstance(outputs.get("evaluation_result"), str)
        or not isinstance(outputs.get("table"), str)
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand causal next-item triangle config differs")
    return document


def _effective_document(config: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(config["source"]["base_config"]["path"]).read_text())
    document["data"]["total_num_days"] = int(config["total_num_days"])
    document["interventions"]["methods"] = list(config["evaluation"]["methods"])
    document["quality"]["record_limit_per_rank"] = int(
        config["evaluation"]["record_limit_per_rank"]
    )
    document["quality"]["cap_user_limit_to_eligible"] = bool(
        config["evaluation"].get("cap_user_limit_to_eligible", False)
    )
    return document


def _checkpoint_root(config: dict[str, Any], version: int) -> Path:
    if version <= 2:
        return Path(config["source"]["base_checkpoint_root"])
    if version <= 7:
        return Path(config["source"]["chain_checkpoint_root"])
    if version == 8:
        return Path(config["outputs"]["checkpoint_root"])
    raise ValueError("KuaiRand causal next-item checkpoint version differs")


def run_triangle_training(config_path: str | Path) -> dict[str, Any]:
    config = load_triangle_config(config_path)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand causal next-item theta8 training is single-rank")
    output = Path(config["outputs"]["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    document = _effective_document(config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for date_index in range(14, int(config["training"]["update_date_index"])):
        plan.ingest_day(dates[date_index])
    model = make_model(document, plan, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(document["training"]["base_lr"]),
        weight_decay=float(document["training"]["weight_decay"]),
    )
    resume = torch.load(
        config["source"]["chain_resume"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    if (
        resume.get("protocol") != NEXT_ITEM_CHAIN_PROTOCOL
        or int(resume.get("version", -1)) != int(config["training"]["source_version"])
        or resume.get("config_sha256") != config["source"]["chain_config"]["sha256"]
    ):
        raise ValueError("KuaiRand causal next-item theta7 resume binding differs")
    model.load_state_dict(resume["model_state"])
    optimizer.load_state_dict(resume["optimizer_state"])
    del resume
    _move_optimizer(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = float(document["training"]["stream_lr"])
    date_index = int(config["training"]["update_date_index"])
    date = dates[date_index]
    plan.ingest_day(date)
    started = time.perf_counter()
    epochs = []
    version = int(config["training"]["target_version"])
    for epoch in range(int(config["training"]["epochs"])):
        np.random.seed(int(document["training"]["seed"]) + version * 1009 + epoch)
        batches = plan.iter_train_batches(
            date,
            int(document["training"]["batch_size"]),
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
                int(document["training"]["negative_count"]),
                document["training"]["maximum_update_batches_per_epoch"],
                f"kuairand_causal_next_item_theta{version}_epoch{epoch + 1}",
            )
        )
    record = {
        "version": version,
        "role": "fixed_schedule_stream_update",
        "date": date,
        "epochs": epochs,
    }
    root = Path(config["outputs"]["checkpoint_root"])
    manifest = _save_checkpoint(
        model,
        root,
        version,
        Path(config_path),
        metadata,
        record,
    )
    _atomic_torch(
        root / "resume.pt",
        {
            "protocol": TRIANGLE_PROTOCOL,
            "config_sha256": file_sha256(config_path),
            "version": version,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
    )
    result = {
        "protocol": TRIANGLE_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "version": version,
        "training": record,
        "manifest": manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def _comparison(edge: dict[str, Any], metric: str) -> dict[str, Any]:
    value = edge["fresh_current_comparisons"]["stale_previous"][metric]
    return {
        "recompute": value["fresh_current"],
        "reuse": value["stale_previous"],
        "absolute": value["fresh_current_advantage_absolute"],
        "relative_percent": value["fresh_current_advantage_relative_percent"],
        "user_cluster_95_interval": value["user_cluster_95_interval"],
        "positive_with_ci": value["fresh_current_advantage_positive_with_ci"],
    }


def _summaries(cells: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    matrices = {}
    age_summary = {}
    for metric in METRICS:
        matrices[metric] = {
            str(target): {
                str(source): next(
                    cell["metrics"][metric]["relative_percent"]
                    for cell in cells
                    if cell["target_version"] == target
                    and cell["source_version"] == source
                )
                for source in range(1, target)
            }
            for target in range(2, 9)
        }
        by_age: dict[int, list[float]] = {}
        for cell in cells:
            by_age.setdefault(cell["cache_age"], []).append(
                cell["metrics"][metric]["relative_percent"]
            )
        age_summary[metric] = {
            str(age): {
                "cells": len(values),
                "mean_relative_percent": float(np.mean(values)),
                "minimum_relative_percent": float(np.min(values)),
                "maximum_relative_percent": float(np.max(values)),
                "positive_fraction": float(np.mean(np.asarray(values) > 0)),
            }
            for age, values in sorted(by_age.items())
        }
    return matrices, age_summary


def _table(matrices: dict[str, Any]) -> str:
    lines = [
        "# KuaiRand causal next-item Recompute-over-Reuse triangle",
        "",
        "Each cell is relative Recompute advantage over direct Reuse. Rows are current model versions and columns are cache source versions.",
    ]
    for metric in METRICS:
        lines.extend(["", f"## {metric}", ""])
        headers = ["target\\source", *[f"θ{value}" for value in range(1, 9)]]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for target in range(1, 9):
            values = [f"θ{target}"]
            for source in range(1, 9):
                if source > target:
                    values.append("—")
                elif source == target:
                    values.append("0.000%")
                elif target == 1:
                    values.append("—")
                else:
                    value = matrices[metric][str(target)][str(source)]
                    values.append(f"{value:+.3f}%")
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def run_triangle_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_triangle_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand causal next-item triangle evaluation requires two ranks")
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
        selected_hashes: dict[str, set[str]] = {}
        for target in config["evaluation"]["target_versions"]:
            if int(target) > 2:
                plan.ingest_day(dates[13 + int(target)])
            current = make_model(document, plan, runtime.device)
            _load_checkpoint(current, _checkpoint_root(config, int(target)), int(target))
            current.eval()
            target_hashes = set()
            for source in range(1, int(target)):
                previous = make_model(document, plan, runtime.device)
                _load_checkpoint(previous, _checkpoint_root(config, source), source)
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
                selected_hashes[str(target)] = target_hashes
            del current
            gc.collect()
            torch.cuda.empty_cache()
        if not runtime.is_primary:
            dist.barrier()
            return None
        if any(len(values) != 1 for values in selected_hashes.values()):
            raise RuntimeError("KuaiRand causal next-item target user sets differ")
        if not all(cell["edge"]["sanity"]["implementation_passed"] for cell in cells):
            raise RuntimeError("KuaiRand causal next-item triangle sanity failed")
        matrices, age_summary = _summaries(cells)
        result = {
            "protocol": TRIANGLE_PROTOCOL,
            "source_protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data": metadata,
            "serving_semantics": "old-version prefix K/V plus current-model suffix, predicting the next item from each real last-item hidden state",
            "metrics": list(METRICS),
            "cells": cells,
            "matrices": matrices,
            "age_summary": age_summary,
            "decision": {
                metric: {
                    "positive_cells": sum(
                        cell["metrics"][metric]["relative_percent"] > 0 for cell in cells
                    ),
                    "positive_with_ci_cells": sum(
                        cell["metrics"][metric]["positive_with_ci"] for cell in cells
                    ),
                    "total_cells": len(cells),
                }
                for metric in METRICS
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        table = Path(config["outputs"]["table"])
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text(_table(matrices))
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)
