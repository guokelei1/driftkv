from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kuairand_next_item_chain import _effective_document, _move_optimizer
from .kuairand_next_item_strength import GROUPS, STRENGTH_PROTOCOL, _group_name
from .kuairand_root_cause import (
    _atomic_json,
    _atomic_torch,
    _save_checkpoint,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)

HOLDOUT_PROTOCOL = "evokv_kuairand_next_item_temporal_holdout_v0"


def load_next_item_holdout_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    learning_rates = document.get("learning_rates", {})
    if (
        document.get("protocol") != HOLDOUT_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("source_version") != 7
        or document.get("target_version") != 8
        or document.get("total_num_days") != 23
        or document.get("ingested_date_indices") != [14, 15, 16, 17, 18, 19, 20]
        or document.get("update_date_index") != 21
        or document.get("epochs") != 3
        or learning_rates
        != {
            "item_embedding": 0.0001,
            "input_encoders": 0.0002,
            "kv_projections": 0.001,
            "other_core": 0.0002,
        }
        or file_sha256(source.get("config", {}).get("path", ""))
        != source.get("config", {}).get("sha256")
        or file_sha256(source.get("resume", {}).get("path", ""))
        != source.get("resume", {}).get("sha256")
        or file_sha256(source.get("theta7", {}).get("path", ""))
        != source.get("theta7", {}).get("sha256")
    ):
        raise ValueError("KuaiRand next-item temporal holdout config differs")
    return document


def _optimizer(model, learning_rates: dict[str, float], weight_decay: float):
    grouped: dict[str, list[torch.nn.Parameter]] = {name: [] for name in GROUPS}
    for name, parameter in model.named_parameters():
        grouped[_group_name(name)].append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": grouped[name],
                "lr": float(learning_rates[name]),
                "group_name": name,
            }
            for name in GROUPS
        ],
        weight_decay=weight_decay,
    )


def run_next_item_holdout_training(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_next_item_holdout_config(config_path)
    output = Path(config["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand temporal holdout training is single-rank")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    document = _effective_document(
        {
            **config,
            "source": {"config": config["source"]["config"]},
            "evaluation_methods": ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"],
            "record_limit_per_rank": 1,
        }
    )
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for date_index in config["ingested_date_indices"]:
        plan.ingest_day(dates[int(date_index)])
    model = make_model(document, plan, device)
    optimizer = _optimizer(
        model,
        config["learning_rates"],
        float(document["training"]["weight_decay"]),
    )
    resume = torch.load(
        config["source"]["resume"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    if (
        resume.get("protocol") != STRENGTH_PROTOCOL
        or resume.get("candidate_id") != "kv_focused_high_e3"
        or resume.get("version") != config["source_version"]
    ):
        raise ValueError("KuaiRand temporal holdout source resume differs")
    model.load_state_dict(resume["model_state"])
    optimizer.load_state_dict(resume["optimizer_state"])
    _move_optimizer(optimizer, device)
    date = dates[int(config["update_date_index"])]
    plan.ingest_day(date)
    started = time.perf_counter()
    epochs = []
    version = int(config["target_version"])
    for epoch in range(int(config["epochs"])):
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
                f"kuairand_temporal_holdout_theta{version}_epoch{epoch + 1}",
            )
        )
    record = {
        "version": version,
        "role": "locked_temporal_holdout_update",
        "date": date,
        "candidate_id": "kv_focused_high_e3",
        "learning_rates": config["learning_rates"],
        "epochs": epochs,
    }
    root = Path(config["checkpoint_root"])
    _save_checkpoint(model, root, version, config_path, metadata, record)
    _atomic_torch(
        Path(config["resume_output"]),
        {
            "protocol": HOLDOUT_PROTOCOL,
            "config_sha256": file_sha256(config_path),
            "version": version,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
    )
    result = {
        "protocol": HOLDOUT_PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "model": asdict(model.cfg),
        "source_version": config["source_version"],
        "completed_version": version,
        "training": record,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def validate_next_item_holdout_result(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != HOLDOUT_PROTOCOL
        or result.get("status") != "complete_development_training"
        or result.get("scientific_result") is not False
        or result.get("source_version") != 7
        or result.get("completed_version") != 8
        or len(result.get("training", {}).get("epochs", [])) != 3
    ):
        raise ValueError("KuaiRand temporal holdout training result differs")
