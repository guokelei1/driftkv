from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
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

NEXT_ITEM_CHAIN_PROTOCOL = "evokv_kuairand_next_item_chain_extension_v0"


def load_next_item_chain_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    bindings = [
        source.get("config", {}),
        source.get("resume", {}),
        source.get("theta2", {}),
    ]
    if (
        document.get("protocol") != NEXT_ITEM_CHAIN_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("source_version") != 2
        or document.get("current_version") != 7
        or document.get("total_num_days") != 22
        or document.get("update_date_indices") != [16, 17, 18, 19, 20]
        or document.get("evaluation_date_index") != 21
        or document.get("source_versions") != [6, 2]
        or document.get("evaluation_methods")
        != ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"]
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand next-item chain config differs")
    return document


def _effective_document(config: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(config["source"]["config"]["path"]).read_text())
    document["data"]["total_num_days"] = config["total_num_days"]
    document["interventions"]["methods"] = config["evaluation_methods"]
    document["quality"]["record_limit_per_rank"] = config["record_limit_per_rank"]
    document["quality"]["cap_user_limit_to_eligible"] = bool(
        config.get("cap_user_limit_to_eligible", False)
    )
    return document


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def run_next_item_chain_training(config_path: str | Path) -> dict[str, Any]:
    config = load_next_item_chain_config(config_path)
    output = Path(config["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand next-item chain training is single-rank")
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(document["training"]["base_lr"]),
        weight_decay=float(document["training"]["weight_decay"]),
    )
    root = Path(config["checkpoint_root"])
    resume_path = root / "resume.pt"
    config_hash = file_sha256(config_path)
    completed_version = int(config["source_version"])
    records = []
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume.get("protocol") != NEXT_ITEM_CHAIN_PROTOCOL
            or resume.get("config_sha256") != config_hash
            or int(resume.get("version", -1)) < int(config["source_version"])
        ):
            raise ValueError("KuaiRand next-item chain resume differs")
        completed_version = int(resume["version"])
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        for version in range(3, completed_version + 1):
            manifest = json.loads((root / f"theta_{version}" / "manifest.json").read_text())
            records.append(manifest["training"])
    else:
        source_resume = torch.load(
            config["source"]["resume"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        if (
            source_resume.get("protocol") != PROTOCOL
            or source_resume.get("version") != config["source_version"]
            or source_resume.get("config_sha256") != config["source"]["config"]["sha256"]
        ):
            raise ValueError("KuaiRand source optimizer binding differs")
        model.load_state_dict(source_resume["model_state"])
        optimizer.load_state_dict(source_resume["optimizer_state"])
    _move_optimizer(optimizer, device)
    for date_index in config["update_date_indices"][: max(0, completed_version - 2)]:
        plan.ingest_day(dates[int(date_index)])
    for group in optimizer.param_groups:
        group["lr"] = float(document["training"]["stream_lr"])
    started = time.perf_counter()
    for version, date_index in enumerate(config["update_date_indices"], start=3):
        if version <= completed_version:
            continue
        date = dates[int(date_index)]
        plan.ingest_day(date)
        epochs = []
        for epoch in range(int(document["training"]["update_epochs"])):
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
                    f"kuairand_next_item_theta{version}_epoch{epoch + 1}",
                )
            )
        record = {
            "version": version,
            "role": "stream_update_extension",
            "date": date,
            "epochs": epochs,
        }
        _save_checkpoint(model, root, version, Path(config_path), metadata, record)
        _atomic_torch(
            resume_path,
            {
                "protocol": NEXT_ITEM_CHAIN_PROTOCOL,
                "config_sha256": config_hash,
                "version": version,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
        )
        records.append(record)
        completed_version = version
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "protocol": NEXT_ITEM_CHAIN_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": config_hash},
        "data": metadata,
        "model": asdict(model.cfg),
        "versions": records,
        "completed_version": completed_version,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def _source_root(config: dict[str, Any], version: int) -> Path:
    if version <= int(config["source_version"]):
        source_document = json.loads(Path(config["source"]["config"]["path"]).read_text())
        return Path(source_document["outputs"]["checkpoint_root"])
    return Path(config["checkpoint_root"])


def _lag_pass(edge: dict[str, Any]) -> bool:
    stale = edge["fresh_current_comparisons"]["stale_previous"]
    ranking = ["hit_rate_at_50", "hit_rate_at_200", "mrr", "ndcg_at_10"]
    positive_rankings = [
        metric
        for metric in ranking
        if stale[metric]["fresh_current_advantage_positive_with_ci"]
        and stale[metric]["fresh_current_advantage_relative_percent"] >= 5.0
    ]
    return bool(
        stale["cross_entropy"]["fresh_current_advantage_positive_with_ci"]
        and len(positive_rankings) >= 2
    )


def run_next_item_chain_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_next_item_chain_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand next-item chain evaluation requires two ranks")
    output = Path(config["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = _effective_document(config)
        torch.set_float32_matmul_precision("high")
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in [14, 15, *config["update_date_indices"]]:
            plan.ingest_day(dates[int(date_index)])
        current = make_model(document, plan, runtime.device)
        _load_checkpoint(current, Path(config["checkpoint_root"]), config["current_version"])
        current.eval()
        popularity = _popularity_ranks(plan)
        started = time.perf_counter()
        lags = []
        for source_version in config["source_versions"]:
            source = make_model(document, plan, runtime.device)
            _load_checkpoint(source, _source_root(config, int(source_version)), int(source_version))
            source.eval()
            edge = _evaluate_edge(
                document,
                plan,
                source,
                current,
                int(config["current_version"]),
                dates[int(config["update_date_indices"][-1])],
                dates[int(config["evaluation_date_index"])],
                popularity,
                runtime,
            )
            if runtime.is_primary:
                lags.append(
                    {
                        "source_version": source_version,
                        "lag": int(config["current_version"]) - int(source_version),
                        "edge": edge,
                        "passed": _lag_pass(edge),
                    }
                )
            del source
            torch.cuda.empty_cache()
        if not runtime.is_primary:
            dist.barrier()
            return None
        lag1 = next(value for value in lags if value["lag"] == 1)
        lag5 = next(value for value in lags if value["lag"] == 5)
        result = {
            "protocol": NEXT_ITEM_CHAIN_PROTOCOL,
            "source_protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data": metadata,
            "lags": lags,
            "decision": {
                "lag5_passed": lag5["passed"],
                "lag1_passed": lag1["passed"],
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)


def validate_next_item_chain_result(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != NEXT_ITEM_CHAIN_PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or [value.get("lag") for value in result.get("lags", [])] != [1, 5]
        or not all(value["edge"]["sanity"]["implementation_passed"] for value in result["lags"])
    ):
        raise ValueError("KuaiRand next-item chain result differs")
