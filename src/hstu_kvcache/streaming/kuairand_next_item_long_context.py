from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kuairand_next_item_chain import _effective_document, _move_optimizer
from .kuairand_next_item_strength import _grouped_optimizer
from .kuairand_root_cause import (
    PROTOCOL,
    _atomic_json,
    _atomic_torch,
    _save_checkpoint,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)

LONG_CONTEXT_PROTOCOL = "evokv_kuairand_next_item_long_context_chain_v0"


def _expected_candidates() -> list[dict[str, Any]]:
    high = {
        "item_embedding": 0.0001,
        "input_encoders": 0.0002,
        "kv_projections": 0.001,
        "other_core": 0.0002,
    }
    medium = {**high, "kv_projections": 0.0005}
    return [
        {
            "candidate_id": "long7_high_e3",
            "epochs": 3,
            "learning_rates": high,
        },
        {
            "candidate_id": "long7_mid_e3",
            "epochs": 3,
            "learning_rates": medium,
        },
    ]


def load_next_item_long_context_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    bindings = [source.get("config", {}), source.get("resume", {}), source.get("theta2", {})]
    if (
        document.get("protocol") != LONG_CONTEXT_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("source_version") != 2
        or document.get("target_version") != 8
        or document.get("total_num_days") != 23
        or document.get("history_window_days") != 7
        or document.get("update_date_indices") != list(range(16, 22))
        or document.get("candidates") != _expected_candidates()
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand long-context chain config differs")
    return document


def _train_candidate(
    config: dict[str, Any],
    config_path: Path,
    candidate: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    root = Path(config["checkpoint_parent"]) / candidate["candidate_id"]
    output = Path(config["result_parent"]) / candidate["candidate_id"] / "training.json"
    if output.is_file():
        return json.loads(output.read_text())
    document = _effective_document(
        {
            **config,
            "source": config["source"],
            "evaluation_methods": ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"],
            "record_limit_per_rank": 1,
        }
    )
    document["data"]["history_window_days"] = int(config["history_window_days"])
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for date_index in (14, 15):
        plan.ingest_day(dates[date_index])
    model = make_model(document, plan, device)
    source_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(document["training"]["base_lr"]),
        weight_decay=float(document["training"]["weight_decay"]),
    )
    source_resume = torch.load(
        config["source"]["resume"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    if (
        source_resume.get("protocol") != PROTOCOL
        or source_resume.get("version") != 2
        or source_resume.get("config_sha256") != config["source"]["config"]["sha256"]
    ):
        raise ValueError("KuaiRand long-context source optimizer differs")
    model.load_state_dict(source_resume["model_state"])
    source_optimizer.load_state_dict(source_resume["optimizer_state"])
    optimizer = _grouped_optimizer(
        model,
        source_optimizer,
        candidate["learning_rates"],
        float(document["training"]["weight_decay"]),
    )
    del source_optimizer, source_resume
    _move_optimizer(optimizer, device)
    resume_path = root / "resume.pt"
    completed_version = 2
    records = []
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume.get("protocol") != LONG_CONTEXT_PROTOCOL
            or resume.get("candidate_id") != candidate["candidate_id"]
            or resume.get("config_sha256") != file_sha256(config_path)
        ):
            raise ValueError("KuaiRand long-context resume differs")
        completed_version = int(resume["version"])
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        _move_optimizer(optimizer, device)
        for version in range(3, completed_version + 1):
            manifest = json.loads((root / f"theta_{version}" / "manifest.json").read_text())
            records.append(manifest["training"])
    for date_index in config["update_date_indices"][: max(0, completed_version - 2)]:
        plan.ingest_day(dates[int(date_index)])
    started = time.perf_counter()
    for version, date_index in enumerate(config["update_date_indices"], start=3):
        if version <= completed_version:
            continue
        date = dates[int(date_index)]
        plan.ingest_day(date)
        epochs = []
        for epoch in range(int(candidate["epochs"])):
            seed = int(document["training"]["seed"]) + version * 1009 + epoch
            np.random.seed(seed)
            torch.manual_seed(seed)
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
                    f"kuairand_{candidate['candidate_id']}_theta{version}_epoch{epoch + 1}",
                )
            )
        record = {
            "version": version,
            "role": "stream_update_long_context",
            "date": date,
            "history_window_days": config["history_window_days"],
            "candidate": candidate,
            "epochs": epochs,
        }
        _save_checkpoint(model, root, version, config_path, metadata, record)
        _atomic_torch(
            resume_path,
            {
                "protocol": LONG_CONTEXT_PROTOCOL,
                "candidate_id": candidate["candidate_id"],
                "config_sha256": file_sha256(config_path),
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
        "protocol": LONG_CONTEXT_PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "candidate": candidate,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "history_window_days": config["history_window_days"],
        "versions": records,
        "completed_version": completed_version,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    del optimizer, model, plan
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_next_item_long_context_training(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_next_item_long_context_config(config_path)
    output = Path(config["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand long-context training is single-rank")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    candidates = [
        _train_candidate(config, config_path, candidate, device)
        for candidate in config["candidates"]
    ]
    result = {
        "protocol": LONG_CONTEXT_PROTOCOL,
        "status": "complete_development_training_sweep",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "history_window_days": config["history_window_days"],
        "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output, result)
    return result


def validate_next_item_long_context_result(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != LONG_CONTEXT_PROTOCOL
        or result.get("status") != "complete_development_training_sweep"
        or result.get("scientific_result") is not False
        or result.get("history_window_days") != 7
        or len(result.get("candidates", [])) != 2
        or any(value.get("completed_version") != 8 for value in result.get("candidates", []))
        or any(len(value.get("versions", [])) != 6 for value in result.get("candidates", []))
    ):
        raise ValueError("KuaiRand long-context training result differs")
