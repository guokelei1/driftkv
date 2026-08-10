from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kuairand_next_item_chain import _effective_document, _move_optimizer
from .kuairand_next_item_holdout import _optimizer
from .kuairand_next_item_strength import STRENGTH_PROTOCOL
from .kuairand_root_cause import (
    _atomic_json,
    _save_checkpoint,
    build_next_item_targets,
    file_sha256,
    load_plan,
    make_model,
)

HARD_UPDATE_PROTOCOL = "evokv_kuairand_next_item_hard_negative_update_v0"
LOGGED_UPDATE_PROTOCOL = "evokv_kuairand_next_item_logged_negative_update_v1"


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
            "candidate_id": "mixed_exposure_high_e3",
            "epochs": 3,
            "learning_rates": high,
            "negative_sampler": {
                "uniform_count": 16,
                "popular_count": 16,
                "popularity_source": "base_period_exposure",
            },
        },
        {
            "candidate_id": "mixed_engaged_high_e3",
            "epochs": 3,
            "learning_rates": high,
            "negative_sampler": {
                "uniform_count": 16,
                "popular_count": 16,
                "popularity_source": "base_period_engaged",
            },
        },
        {
            "candidate_id": "mixed_exposure_mid_e3",
            "epochs": 3,
            "learning_rates": medium,
            "negative_sampler": {
                "uniform_count": 16,
                "popular_count": 16,
                "popularity_source": "base_period_exposure",
            },
        },
        {
            "candidate_id": "hard_exposure_high_e3",
            "epochs": 3,
            "learning_rates": high,
            "negative_sampler": {
                "uniform_count": 0,
                "popular_count": 32,
                "popularity_source": "base_period_exposure",
            },
        },
    ]


def _expected_logged_candidates() -> list[dict[str, Any]]:
    high = {
        "item_embedding": 0.0001,
        "input_encoders": 0.0002,
        "kv_projections": 0.001,
        "other_core": 0.0002,
    }
    medium = {**high, "kv_projections": 0.0005}
    return [
        {
            "candidate_id": "mixed_logged_high_e3",
            "epochs": 3,
            "learning_rates": high,
            "negative_sampler": {
                "uniform_count": 16,
                "popular_count": 0,
                "logged_count": 16,
                "popularity_source": "base_period_exposure",
            },
        },
        {
            "candidate_id": "logged_heavy_high_e3",
            "epochs": 3,
            "learning_rates": high,
            "negative_sampler": {
                "uniform_count": 8,
                "popular_count": 0,
                "logged_count": 24,
                "popularity_source": "base_period_exposure",
            },
        },
        {
            "candidate_id": "mixed_logged_mid_e3",
            "epochs": 3,
            "learning_rates": medium,
            "negative_sampler": {
                "uniform_count": 16,
                "popular_count": 0,
                "logged_count": 16,
                "popularity_source": "base_period_exposure",
            },
        },
        {
            "candidate_id": "logged_heavy_mid_e3",
            "epochs": 3,
            "learning_rates": medium,
            "negative_sampler": {
                "uniform_count": 8,
                "popular_count": 0,
                "logged_count": 24,
                "popularity_source": "base_period_exposure",
            },
        },
    ]


def load_next_item_hard_update_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    bindings = [source.get("config", {}), source.get("resume", {}), source.get("theta7", {})]
    if (
        document.get("protocol") not in (HARD_UPDATE_PROTOCOL, LOGGED_UPDATE_PROTOCOL)
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("source_version") != 7
        or document.get("target_version") != 8
        or document.get("total_num_days") != 23
        or document.get("ingested_date_indices") != list(range(14, 21))
        or document.get("update_date_index") != 21
        or (
            document.get("protocol") == HARD_UPDATE_PROTOCOL
            and document.get("candidates") != _expected_candidates()
        )
        or (
            document.get("protocol") == LOGGED_UPDATE_PROTOCOL
            and document.get("candidates") != _expected_logged_candidates()
        )
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand hard-negative update config differs")
    return document


def _popularity_rankings(plan) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    counts = {
        "base_period_exposure": np.zeros(plan.num_prediction_items + 1, dtype=np.int64),
        "base_period_engaged": np.zeros(plan.num_prediction_items + 1, dtype=np.int64),
    }
    for date in plan.base_dates:
        frame = plan.daily_segments[date]
        exposed = frame["item_idx"].to_numpy(dtype=np.int64)
        exposed = exposed[(exposed >= 1) & (exposed <= plan.num_prediction_items)]
        engaged = frame.loc[frame["label"] > 0, "item_idx"].to_numpy(dtype=np.int64)
        engaged = engaged[(engaged >= 1) & (engaged <= plan.num_prediction_items)]
        np.add.at(counts["base_period_exposure"], exposed, 1)
        np.add.at(counts["base_period_engaged"], engaged, 1)
    ids = np.arange(1, plan.num_prediction_items + 1, dtype=np.int64)
    rankings = {}
    metadata = {}
    for name, values in counts.items():
        ranking = ids[np.lexsort((ids, -values[1:]))].copy()
        rankings[name] = torch.from_numpy(ranking)
        metadata[name] = {
            "events": int(values.sum()),
            "nonzero_items": int(np.count_nonzero(values[1:])),
            "sha256": hashlib.sha256(ranking.astype("<i8", copy=False).tobytes()).hexdigest(),
            "top_items": ranking[:20].tolist(),
        }
    return rankings, metadata


def _popular_negatives(
    positives: torch.Tensor,
    ranking: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if count == 0:
        return torch.empty((*positives.shape, 0), dtype=torch.int64, device=positives.device)
    head = ranking[: count + 1].to(positives.device)
    negatives = head[:count].view(*([1] * positives.ndim), count).expand(
        *positives.shape, count
    )
    replacement = head[count]
    return torch.where(negatives == positives.unsqueeze(-1), replacement, negatives)


def _logged_negatives(
    item_ids: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    count: int,
    num_prediction_items: int,
) -> torch.Tensor:
    if count == 0:
        return torch.empty((*targets.shape, 0), dtype=torch.int64, device=targets.device)
    output = torch.empty((*targets.shape, count), dtype=torch.int64, device=targets.device)
    positions = torch.arange(item_ids.shape[1], device=item_ids.device)
    for batch_index in range(item_ids.shape[0]):
        valid = positions < lengths[batch_index]
        engaged = item_ids[batch_index][valid & labels[batch_index]]
        pool = item_ids[batch_index][
            valid
            & (~labels[batch_index])
            & (item_ids[batch_index] >= 1)
            & (item_ids[batch_index] <= num_prediction_items)
        ]
        if len(engaged):
            pool = pool[~torch.isin(pool, torch.unique(engaged))]
        pool = torch.unique(pool)
        if len(pool):
            indices = torch.randint(
                0,
                len(pool),
                (*targets[batch_index].shape, count),
                device=targets.device,
            )
            output[batch_index] = pool[indices]
        else:
            output[batch_index] = torch.randint(
                1,
                num_prediction_items + 1,
                (*targets[batch_index].shape, count),
                device=targets.device,
            )
    return output


def _training_step(
    model,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sampler: dict[str, Any],
    ranking: torch.Tensor,
) -> tuple[float, int]:
    model.train()
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch["labels"].to(device)
    train_mask = batch["train_mask"].to(device)
    optimizer.zero_grad(set_to_none=True)
    hidden, _ = model(
        item_ids,
        behaviors,
        time_deltas,
        return_kv=False,
        lengths=lengths,
    )
    targets, valid = build_next_item_targets(item_ids, lengths, labels, train_mask)
    target_count = int(valid.sum().item())
    if target_count < 1:
        return 0.0, 0
    positives = targets.unsqueeze(-1)
    uniform_count = int(sampler["uniform_count"])
    if uniform_count:
        uniform = torch.randint(
            1,
            model.cfg.num_prediction_items + 1,
            (*targets.shape, uniform_count),
            device=device,
        )
        uniform = torch.where(
            uniform == positives,
            uniform.remainder(model.cfg.num_prediction_items) + 1,
            uniform,
        )
    else:
        uniform = torch.empty((*targets.shape, 0), dtype=torch.int64, device=device)
    popular = _popular_negatives(targets, ranking, int(sampler["popular_count"]))
    logged = _logged_negatives(
        item_ids,
        labels,
        lengths,
        targets,
        int(sampler.get("logged_count", 0)),
        model.cfg.num_prediction_items,
    )
    candidates = torch.cat((positives, uniform, popular, logged), dim=-1)
    logits = model.item_emb.score(hidden[:, :-1], candidates)
    labels_zero = torch.zeros_like(targets)
    per_target = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        labels_zero.flatten(),
        reduction="none",
    ).view_as(targets)
    loss = (per_target * valid).sum() / valid.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), target_count


def _train_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    batches,
    device: torch.device,
    sampler: dict[str, Any],
    ranking: torch.Tensor,
    phase: str,
) -> dict[str, Any]:
    loss_sum = 0.0
    targets = 0
    sequences = 0
    tokens = 0
    completed_batches = 0
    for batch in batches:
        loss, count = _training_step(model, batch, optimizer, device, sampler, ranking)
        if count:
            loss_sum += loss * count
            targets += count
        sequences += len(batch["lengths"])
        tokens += int(batch["lengths"].sum().item())
        completed_batches += 1
        if completed_batches % 100 == 0:
            print(
                f"phase={phase} batches={completed_batches} targets={targets} "
                f"loss={loss_sum / max(targets, 1):.6f}",
                flush=True,
            )
    if targets < 1:
        raise RuntimeError("KuaiRand hard-negative update produced no targets")
    return {
        "batches": completed_batches,
        "sequences": sequences,
        "tokens": tokens,
        "eligible_targets": targets,
        "target_weighted_sampled_cross_entropy": loss_sum / targets,
    }


def run_next_item_hard_update_training(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_next_item_hard_update_config(config_path)
    output = Path(config["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand hard-negative update training is single-rank")
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
    rankings, ranking_metadata = _popularity_rankings(plan)
    date = dates[int(config["update_date_index"])]
    plan.ingest_day(date)
    source_resume = torch.load(
        config["source"]["resume"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    if (
        source_resume.get("protocol") != STRENGTH_PROTOCOL
        or source_resume.get("candidate_id") != "kv_focused_high_e3"
        or source_resume.get("version") != 7
    ):
        raise ValueError("KuaiRand hard-negative source resume differs")
    started = time.perf_counter()
    candidate_results = []
    for candidate in config["candidates"]:
        candidate_output = Path(config["result_parent"]) / candidate["candidate_id"] / "training.json"
        if candidate_output.is_file():
            candidate_results.append(json.loads(candidate_output.read_text()))
            continue
        model = make_model(document, plan, device)
        optimizer = _optimizer(
            model,
            candidate["learning_rates"],
            float(document["training"]["weight_decay"]),
        )
        model.load_state_dict(source_resume["model_state"])
        optimizer.load_state_dict(source_resume["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = float(candidate["learning_rates"][group["group_name"]])
        _move_optimizer(optimizer, device)
        epochs = []
        ranking = rankings[candidate["negative_sampler"]["popularity_source"]]
        for epoch in range(int(candidate["epochs"])):
            seed = int(document["training"]["seed"]) + 8 * 1009 + epoch
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
                    candidate["negative_sampler"],
                    ranking,
                    f"kuairand_{candidate['candidate_id']}_theta8_epoch{epoch + 1}",
                )
            )
        record = {
            "version": 8,
            "role": "hard_negative_temporal_holdout_update",
            "date": date,
            "candidate": candidate,
            "epochs": epochs,
        }
        root = Path(config["checkpoint_parent"]) / candidate["candidate_id"]
        manifest = _save_checkpoint(model, root, 8, config_path, metadata, record)
        candidate_result = {
            "candidate": candidate,
            "checkpoint": manifest,
            "training": record,
        }
        _atomic_json(candidate_output, candidate_result)
        candidate_results.append(candidate_result)
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    del source_resume
    result = {
        "protocol": config["protocol"],
        "status": "complete_development_training_sweep",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "popularity": ranking_metadata,
        "source_version": 7,
        "completed_version": 8,
        "candidates": candidate_results,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def validate_next_item_hard_update_result(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") not in (HARD_UPDATE_PROTOCOL, LOGGED_UPDATE_PROTOCOL)
        or result.get("status") != "complete_development_training_sweep"
        or result.get("scientific_result") is not False
        or result.get("source_version") != 7
        or result.get("completed_version") != 8
        or len(result.get("candidates", [])) != 4
        or any(
            len(value.get("training", {}).get("epochs", [])) != 3
            for value in result.get("candidates", [])
        )
    ):
        raise ValueError("KuaiRand hard-negative update result differs")
