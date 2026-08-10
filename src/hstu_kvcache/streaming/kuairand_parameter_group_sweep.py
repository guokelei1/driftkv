from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from .kuairand_intraday_update import _split_day

SWEEP_PROTOCOL = "evokv_kuairand_parameter_group_sweep_v0"


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    base = document.get("base_config", {})
    names = [candidate.get("name") for candidate in document.get("candidates", [])]
    if (
        document.get("protocol") != SWEEP_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("horizons") != [4, 8, 16]
        or names != ["head_only", "kv_only", "qkv_only", "backbone_only"]
        or file_sha256(base.get("path", "")) != base.get("sha256")
    ):
        raise ValueError("KuaiRand parameter-group sweep config differs")
    return document


def _reset_histories(plan):
    for history in plan.user_histories.values():
        history["item_ids"] = np.array([], dtype=np.int64)
        history["behaviors"] = np.array([], dtype=np.int64)
        history["time_deltas"] = np.array([], dtype=np.float32)
        history["labels"] = np.array([], dtype=np.int64)
        history["timestamps"] = np.array([], dtype=np.int64)
    plan.init_base()


def _select_parameters(model, group):
    selected = []
    selected_names = []
    for name, parameter in model.named_parameters():
        if group == "head_only":
            active = name.startswith("engagement_head.")
        elif group == "kv_only":
            active = ".attn.k_proj." in name or ".attn.v_proj." in name
        elif group == "qkv_only":
            active = any(f".attn.{projection}." in name for projection in ("q_proj", "k_proj", "v_proj"))
        elif group == "backbone_only":
            active = name.startswith("backbone.")
        else:
            raise ValueError("KuaiRand parameter group differs")
        parameter.requires_grad_(active)
        if active:
            selected.append(parameter)
            selected_names.append(name)
    if not selected:
        raise RuntimeError("KuaiRand parameter group is empty")
    return selected, selected_names


def _candidate_admitted(edges):
    for edge in edges:
        primary = next(value for value in edge["values"] if value["max_exposures"] == 4)
        stale = primary["comparisons"]["recompute_over_reuse"]
        if not (
            stale["average_precision"]["positive_direction_with_ci"]
            and stale["ndcg_at_10"]["positive_direction_with_ci"]
            and max(
                stale["average_precision"]["relative_percent"],
                stale["ndcg_at_10"]["relative_percent"],
            )
            >= 5.0
        ):
            return False
    return True


def run_parameter_group_sweep(config_path: str | Path):
    config = load_sweep_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    base = load_engagement_config(config["base_config"]["path"])
    plan, metadata = load_plan(base)
    device = torch.device("cuda:0")
    base_root = Path(base["outputs"]["checkpoint_root"])
    dates = plan.base_dates + plan.stream_dates
    candidates = []
    started = time.monotonic()
    for candidate in config["candidates"]:
        _reset_histories(plan)
        current = make_model(base, plan, device)
        _load_checkpoint(current, base_root, 0)
        parameters, parameter_names = _select_parameters(current, candidate["name"])
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(candidate["update_lr"]),
            weight_decay=float(candidate["weight_decay"]),
        )
        torch.manual_seed(int(candidate["seed"]))
        torch.cuda.manual_seed_all(int(candidate["seed"]))
        effective = json.loads(json.dumps(base))
        effective["training"].update(candidate)
        candidate_root = Path(config["checkpoint_root"]) / candidate["name"]
        edges = []
        checkpoints = []
        for edge, date_index in enumerate(config["source_date_indices"], start=1):
            source_date = dates[int(date_index)]
            boundary, update_frame, evaluation_frame = _split_day(
                plan.daily_segments[source_date],
                float(config["split_fraction"]),
            )
            update_key = f"{candidate['name']}_{edge}_update"
            evaluation_key = f"{candidate['name']}_{edge}_evaluation"
            plan.daily_segments[update_key] = update_frame
            plan.daily_segments[evaluation_key] = evaluation_frame
            previous = make_model(base, plan, device)
            previous_root = base_root if edge == 1 else candidate_root
            _load_checkpoint(previous, previous_root, edge - 1)
            plan.ingest_day(update_key)
            epochs = []
            for epoch in range(int(candidate["update_epochs"])):
                np.random.seed(int(candidate["seed"]) + edge * 1009 + epoch)
                batches = plan.iter_train_batches(
                    update_key,
                    int(candidate["batch_size"]),
                    all_chunks=True,
                    bucket_by_length=True,
                    pad_to_max_seq_len=False,
                )
                epochs.append(
                    _train_epoch(
                        current,
                        optimizer,
                        batches,
                        effective,
                        device,
                        f"{candidate['name']}_theta{edge}_e{epoch + 1}",
                    )
                )
            checkpoints.append(
                _save_checkpoint(
                    current,
                    candidate_root,
                    edge,
                    config_path,
                    metadata,
                    epochs,
                )
            )
            previous.eval()
            current.eval()
            values = []
            for horizon in config["horizons"]:
                print(
                    f"phase=parameter_group candidate={candidate['name']} "
                    f"edge={edge} horizon={horizon}",
                    flush=True,
                )
                values.append(
                    _evaluate_edge(
                        base,
                        plan,
                        previous,
                        current,
                        update_key,
                        evaluation_key,
                        edge,
                        device,
                        max_exposures=int(horizon),
                    )
                )
            plan.ingest_day(evaluation_key)
            edges.append(
                {
                    "edge": edge,
                    "source_date": source_date,
                    "boundary_time_ms": boundary,
                    "training": epochs,
                    "values": values,
                }
            )
        candidates.append(
            {
                "name": candidate["name"],
                "parameter_names": parameter_names,
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "checkpoints": checkpoints,
                "edges": edges,
                "admitted": _candidate_admitted(edges),
            }
        )
    admitted = [candidate["name"] for candidate in candidates if candidate["admitted"]]
    result = {
        "protocol": SWEEP_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "base_config": config["base_config"],
        "data": metadata,
        "candidates": candidates,
        "decision": {"admitted_candidates": admitted},
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_parameter_group_sweep(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != SWEEP_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or len(result.get("candidates", [])) != 4
        or not all(len(candidate.get("edges", [])) == 2 for candidate in result["candidates"])
        or not all(
            value.get("same_model_sanity_passed")
            for candidate in result["candidates"]
            for edge in candidate["edges"]
            for value in edge["values"]
        )
    ):
        raise ValueError("KuaiRand parameter-group sweep result differs")
