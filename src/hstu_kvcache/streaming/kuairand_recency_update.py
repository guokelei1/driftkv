from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hstu_kvcache.data.kuairand import collate_batch

from .kuairand_engagement import (
    PROTOCOL,
    _atomic_json,
    _evaluate_edge,
    _load_checkpoint,
    _save_checkpoint,
    _train_epoch,
    file_sha256,
    load_engagement_config,
    load_plan,
    make_model,
)

RECENCY_PROTOCOL = "evokv_kuairand_recency_aligned_update_v0"


def load_recency_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    base = document.get("base_config", {})
    training = document.get("training", {})
    if (
        document.get("protocol") != RECENCY_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("tail_fraction") != 0.25
        or document.get("horizons") != [4, 8, 16, 32, 64]
        or training.get("update_epochs") != 2
        or training.get("update_lr") != 0.0002
        or training.get("batch_size") != 16
        or file_sha256(base.get("path", "")) != base.get("sha256")
    ):
        raise ValueError("KuaiRand recency-update config differs")
    return document


def _tail_batches(plan, date, batch_size, tail_fraction, seed):
    day = plan.daily_segments[date]
    ordered = day.sort_values("time_ms")
    first = int(np.floor((1.0 - tail_fraction) * len(ordered)))
    cutoff = int(ordered.iloc[first]["time_ms"])
    sequences = []
    for user in day["user_idx"].unique():
        user = int(user)
        history = plan.user_histories[user]
        sequence = plan._build_seq(user, truncate=len(history["item_ids"]))
        if sequence is None:
            continue
        sequence["train_mask"] = sequence["timestamps"] >= cutoff
        sequences.extend(
            chunk
            for chunk in plan._chunk_sequence(sequence)
            if np.any(chunk["train_mask"])
        )
    generator = np.random.default_rng(seed)
    generator.shuffle(sequences)
    sequences.sort(key=lambda sequence: len(sequence["item_ids"]))
    groups = [sequences[index : index + batch_size] for index in range(0, len(sequences), batch_size)]
    generator.shuffle(groups)
    for group in groups:
        yield collate_batch(group, max_seq_len=plan.max_seq_len, pad_to=None)


def run_recency_update(config_path: str | Path):
    config = load_recency_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    base = load_engagement_config(config["base_config"]["path"])
    effective = json.loads(json.dumps(base))
    effective["training"].update(config["training"])
    plan, metadata = load_plan(base)
    plan.init_base()
    device = torch.device("cuda:0")
    current = make_model(base, plan, device)
    base_root = Path(base["outputs"]["checkpoint_root"])
    root = Path(config["checkpoint_root"])
    _load_checkpoint(current, base_root, 0)
    optimizer = torch.optim.AdamW(
        current.parameters(),
        lr=float(config["training"]["update_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    dates = plan.base_dates + plan.stream_dates
    edges = []
    checkpoints = []
    started = time.monotonic()
    for edge, (update_index, eval_index) in enumerate(
        zip(
            base["evaluation"]["update_date_indices"],
            base["evaluation"]["evaluation_date_indices"],
            strict=True,
        ),
        start=1,
    ):
        update_date = dates[int(update_index)]
        eval_date = dates[int(eval_index)]
        previous = make_model(base, plan, device)
        previous_root = base_root if edge == 1 else root
        _load_checkpoint(previous, previous_root, edge - 1)
        plan.ingest_day(update_date)
        epochs = []
        for epoch in range(int(config["training"]["update_epochs"])):
            batches = _tail_batches(
                plan,
                update_date,
                int(config["training"]["batch_size"]),
                float(config["tail_fraction"]),
                int(config["training"]["seed"]) + edge * 1009 + epoch,
            )
            epochs.append(
                _train_epoch(
                    current,
                    optimizer,
                    batches,
                    effective,
                    device,
                    f"recency_theta{edge}_e{epoch + 1}",
                )
            )
        checkpoints.append(
            _save_checkpoint(current, root, edge, config_path, metadata, epochs)
        )
        previous.eval()
        current.eval()
        values = []
        for horizon in config["horizons"]:
            print(f"phase=recency_eval edge={edge} horizon={horizon}", flush=True)
            values.append(
                _evaluate_edge(
                    base,
                    plan,
                    previous,
                    current,
                    update_date,
                    eval_date,
                    edge,
                    device,
                    max_exposures=int(horizon),
                )
            )
        edges.append({"edge": edge, "values": values, "training": epochs})
    common_positive_horizons = [
        horizon
        for horizon in config["horizons"]
        if all(
            next(
                value
                for value in edge["values"]
                if value["max_exposures"] == horizon
            )["gate"]["passed"]
            for edge in edges
        )
    ]
    result = {
        "protocol": RECENCY_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "base_config": config["base_config"],
        "data": metadata,
        "tail_fraction": config["tail_fraction"],
        "checkpoints": checkpoints,
        "edges": edges,
        "decision": {"common_positive_horizons": common_positive_horizons},
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_recency_update(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != RECENCY_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or len(result.get("edges", [])) != 2
        or not all(len(edge.get("values", [])) == 5 for edge in result["edges"])
        or not all(
            value.get("same_model_sanity_passed")
            for edge in result["edges"]
            for value in edge["values"]
        )
    ):
        raise ValueError("KuaiRand recency-update result differs")
