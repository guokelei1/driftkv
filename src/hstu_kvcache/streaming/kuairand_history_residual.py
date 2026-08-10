from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from hstu_kvcache.data.kuairand import stable_context_hash
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache

from .kuairand_engagement import (
    METHODS,
    METRICS,
    _bootstrap,
    _metric_values,
    file_sha256,
    interleave_batch,
    interleave_sequence,
    load_plan,
)

PROTOCOL = "evokv_kuairand_history_residual_engagement_v0"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data", {})
    training = document.get("training", {})
    evaluation = document.get("evaluation", {})
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("scoring_mode") != "history_residual_only"
        or data.get("base_num_days") != 14
        or data.get("total_num_days") != 17
        or data.get("max_users") != 256
        or data.get("max_original_seq_len") != 64
        or data.get("history_window_days") != 1
        or data.get("preserve_new_item_engagement_labels") is not True
        or training.get("base_epochs") != 1
        or training.get("update_epochs") != 2
        or evaluation.get("update_date_indices") != [14, 15]
        or evaluation.get("evaluation_date_indices") != [15, 16]
        or evaluation.get("selected_users") != 128
        or evaluation.get("max_exposures_per_user") != 64
    ):
        raise ValueError("KuaiRand history-residual config differs")
    for source in data.get("standard_logs", []):
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("KuaiRand history-residual source binding differs")
    semantic_token = data.get("semantic_token", "video_id")
    if semantic_token not in ("video_id", "author_hash"):
        raise ValueError("KuaiRand history-residual semantic token differs")
    if semantic_token == "author_hash":
        feature_source = data.get("feature_source", {})
        if (
            data.get("semantic_hash_buckets") != 65536
            or file_sha256(feature_source.get("path", "")) != feature_source.get("sha256")
        ):
            raise ValueError("KuaiRand history-residual feature binding differs")
    return document


def load_experiment_plan(document: dict[str, Any]):
    plan, metadata = load_plan(document)
    semantic_token = document["data"].get("semantic_token", "video_id")
    if semantic_token == "video_id":
        metadata["semantic_token"] = semantic_token
        return plan, metadata
    source = document["data"]["feature_source"]["path"]
    features = pd.read_csv(source, usecols=["video_id", "author_id"])
    author_by_video = pd.Series(
        features["author_id"].to_numpy(dtype=np.int64),
        index=features["video_id"].to_numpy(dtype=np.int64),
    )
    frame = plan.trace.interactions
    authors = frame["video_id"].map(author_by_video).fillna(-1).to_numpy(dtype=np.int64)
    buckets = int(document["data"]["semantic_hash_buckets"])
    tokens = np.ones(len(frame), dtype=np.int64)
    present = authors >= 0
    tokens[present] = 2 + (
        stable_context_hash(authors[present]) % np.uint64(buckets - 1)
    ).astype(np.int64)
    frame["item_idx"] = tokens
    plan.trace.num_items = buckets
    plan.trace.num_prediction_items = buckets
    plan.trace.context_hash_buckets = 0
    plan.trace.item_map = {}
    plan.daily_segments = {
        date: group.sort_values("time_ms").reset_index(drop=True)
        for date, group in frame.assign(date=frame["date"].astype(str)).groupby("date")
    }
    metadata.update(
        {
            "semantic_token": semantic_token,
            "semantic_hash_buckets": buckets,
            "feature_source": {
                "path": source,
                "sha256": file_sha256(source),
            },
            "num_items": plan.num_items,
            "num_prediction_items": plan.num_prediction_items,
            "missing_feature_rows": int((~present).sum()),
        }
    )
    return plan, metadata


class HistoryResidualEngagementModel(nn.Module):
    def __init__(self, cfg: HSTUConfig) -> None:
        super().__init__()
        self.backbone = HSTU(cfg)
        self.engagement_head = nn.Linear(cfg.hidden_size, 1)
        nn.init.zeros_(self.engagement_head.bias)

    @property
    def cfg(self) -> HSTUConfig:
        return self.backbone.cfg

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def compute_kv(self, *args, **kwargs):
        return self.backbone.compute_kv(*args, **kwargs)

    def forward_with_cache(self, *args, **kwargs):
        return self.backbone.forward_with_cache(*args, **kwargs)

    def isolated_query_hidden(
        self,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        shape = item_ids.shape
        flat_items = item_ids.reshape(-1, 1)
        flat_deltas = time_deltas.reshape(-1, 1)
        flat_behaviors = torch.full_like(flat_items, self.cfg.num_behaviors)
        hidden, _ = self.backbone(flat_items, flat_behaviors, flat_deltas)
        return hidden[:, 0].reshape(*shape, self.cfg.hidden_size)

    def residual_logits(
        self,
        query_hidden: torch.Tensor,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        isolated = self.isolated_query_hidden(item_ids, time_deltas)
        return self.engagement_head(query_hidden - isolated).squeeze(-1)


def make_model(document: dict[str, Any], plan, device: torch.device):
    model = document["model"]
    cfg = HSTUConfig(
        num_items=plan.num_items,
        num_prediction_items=plan.num_prediction_items,
        num_behaviors=plan.num_behaviors,
        hidden_size=int(model["hidden_size"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        head_dim=int(model["head_dim"]),
        max_seq_len=2 * int(document["data"]["max_original_seq_len"]),
        temporal_num_freqs=int(model["temporal_num_freqs"]),
        activation=str(model["activation"]),
        input_dropout=float(model["input_dropout"]),
    )
    return HistoryResidualEngagementModel(cfg).to(device)


def _train_epoch(model, optimizer, batches, document, device, phase: str):
    model.train()
    loss_sum = 0.0
    targets = 0
    started = time.monotonic()
    for batch_index, batch in enumerate(batches):
        value = interleave_batch(batch, model.cfg.num_behaviors, device)
        optimizer.zero_grad(set_to_none=True)
        hidden, _ = model(
            value["item_ids"],
            value["behaviors"],
            value["time_deltas"],
            lengths=value["lengths"],
        )
        original_items = batch["item_ids"].to(device)
        original_deltas = batch["time_deltas"].to(device)
        logits = model.residual_logits(hidden[:, 0::2], original_items, original_deltas)
        selected = value["target_mask"]
        loss = F.binary_cross_entropy_with_logits(logits[selected], value["labels"][selected])
        if not torch.isfinite(loss):
            raise RuntimeError("KuaiRand history-residual training produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(document["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
        count = int(selected.sum().item())
        loss_sum += float(loss.detach().item()) * count
        targets += count
        if (batch_index + 1) % 200 == 0:
            print(
                f"phase={phase} batches={batch_index + 1} targets={targets} "
                f"loss={loss_sum / targets:.6f}",
                flush=True,
            )
    return {
        "phase": phase,
        "targets": targets,
        "mean_binary_cross_entropy": loss_sum / targets,
        "elapsed_seconds": time.monotonic() - started,
    }


def _checkpoint_paths(root: Path, version: int):
    directory = root / f"theta_{version}"
    return directory / "model.pt", directory / "manifest.json"


def _save_checkpoint(model, root, version, config_path, metadata, training):
    model_path, manifest_path = _checkpoint_paths(root, version)
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("KuaiRand history-residual checkpoint already exists")
    _atomic_torch(
        model_path,
        {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}},
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_development_checkpoint",
        "scientific_result": False,
        "version": version,
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(model_path),
        "model": asdict(model.cfg),
        "data": metadata,
        "training": training,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_checkpoint(model, root, version):
    model_path, manifest_path = _checkpoint_paths(root, version)
    manifest = json.loads(manifest_path.read_text())
    if manifest["model_sha256"] != file_sha256(model_path):
        raise ValueError("KuaiRand history-residual checkpoint binding differs")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()


def run_training(config_path: str | Path):
    document = load_config(config_path)
    output = Path(document["outputs"]["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    device = torch.device("cuda:0")
    _seed_everything(int(document["training"]["seed"]))
    plan, metadata = load_experiment_plan(document)
    plan.init_base()
    model = make_model(document, plan, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(document["training"]["base_lr"]),
        weight_decay=float(document["training"]["weight_decay"]),
    )
    root = Path(document["outputs"]["checkpoint_root"])
    versions = []
    started = time.monotonic()
    for epoch in range(int(document["training"]["base_epochs"])):
        np.random.seed(int(document["training"]["seed"]) + epoch)
        batches = plan.iter_base_train_batches(
            int(document["training"]["batch_size"]),
            all_chunks=True,
            bucket_by_length=True,
            pad_to_max_seq_len=False,
        )
        training = _train_epoch(
            model,
            optimizer,
            batches,
            document,
            device,
            f"history_residual_theta0_e{epoch + 1}",
        )
    versions.append(_save_checkpoint(model, root, 0, config_path, metadata, training))
    for group in optimizer.param_groups:
        group["lr"] = float(document["training"]["update_lr"])
    dates = plan.base_dates + plan.stream_dates
    for version, date_index in enumerate(document["evaluation"]["update_date_indices"], start=1):
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
                    document,
                    device,
                    f"history_residual_theta{version}_e{epoch + 1}",
                )
            )
        versions.append(_save_checkpoint(model, root, version, config_path, metadata, epochs))
    result = {
        "protocol": PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "versions": versions,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def _tensor_sequence(sequence, device):
    return (
        torch.as_tensor(sequence["item_ids"], dtype=torch.long, device=device).unsqueeze(0),
        torch.as_tensor(sequence["behaviors"], dtype=torch.long, device=device).unsqueeze(0),
        torch.as_tensor(sequence["time_deltas"], dtype=torch.float32, device=device).unsqueeze(0),
    )


def _empty_cache(reference: HSTUKVCache):
    return HSTUKVCache(reference.k[:, :, :0], reference.v[:, :, :0], 0)


def _summarize(records, document):
    endpoints = {
        method: {
            metric: float(np.mean([value["metrics"][method][metric] for value in records]))
            for metric in METRICS
        }
        for method in METHODS
    }
    pairs = {
        "fresh_update_value": ("recompute", "previous_fresh"),
        "recompute_over_reuse": ("recompute", "reuse"),
        "history_value": ("recompute", "no_prefix"),
    }
    comparisons = {}
    for pair_index, (name, (better, worse)) in enumerate(pairs.items()):
        comparisons[name] = {}
        for metric_index, metric in enumerate(METRICS):
            better_values = np.asarray([value["metrics"][better][metric] for value in records])
            worse_values = np.asarray([value["metrics"][worse][metric] for value in records])
            advantage = (
                worse_values - better_values
                if metric in ("log_loss", "brier")
                else better_values - worse_values
            )
            interval = _bootstrap(
                advantage,
                int(document["evaluation"]["bootstrap_samples"]),
                int(document["evaluation"]["bootstrap_seed"])
                + pair_index * 10000019
                + metric_index * 1000003,
            )
            comparisons[name][metric] = {
                "absolute": float(advantage.mean()),
                "relative_percent": 100.0 * float(advantage.mean()) / float(worse_values.mean()),
                "user_bootstrap_95": interval,
                "positive_direction_with_ci": interval[0] > 0,
            }
    stale = comparisons["recompute_over_reuse"]
    gate = {
        "fresh_update_average_precision": comparisons["fresh_update_value"][
            "average_precision"
        ]["positive_direction_with_ci"],
        "history_average_precision": comparisons["history_value"]["average_precision"][
            "positive_direction_with_ci"
        ],
        "stale_log_loss": stale["log_loss"]["positive_direction_with_ci"],
        "stale_average_precision": stale["average_precision"]["positive_direction_with_ci"],
        "stale_ndcg_at_50": stale["ndcg_at_50"]["positive_direction_with_ci"],
        "stale_rank_magnitude_at_least_5_percent": max(
            stale["average_precision"]["relative_percent"],
            stale["ndcg_at_50"]["relative_percent"],
        )
        >= 5.0,
    }
    gate["passed"] = all(gate.values())
    return {"users": len(records), "endpoints": endpoints, "comparisons": comparisons, "gate": gate}


@torch.no_grad()
def _evaluate_edge(document, plan, previous, current, update_date, eval_date, edge, device):
    update_users = set(plan.daily_segments[update_date]["user_idx"].astype(int))
    evaluation = plan.daily_segments[eval_date]
    maximum = int(document["evaluation"]["max_exposures_per_user"])
    eligible = []
    for user, original in evaluation.groupby("user_idx"):
        frame = original.sort_values("time_ms").iloc[:maximum]
        labels = frame["label"].to_numpy(dtype=np.int64)
        user = int(user)
        if user in update_users and labels.min() == 0 and labels.max() == 1:
            history = plan._build_seq(
                user,
                truncate=int(document["data"]["max_original_seq_len"]),
                as_of_timestamp=int(frame["time_ms"].min()),
            )
            if history is not None and len(history["item_ids"]) >= 2:
                eligible.append(user)
    generator = np.random.default_rng(int(document["evaluation"]["sampling_seed"]) + edge)
    selected = sorted(
        np.asarray(eligible)[
            generator.permutation(len(eligible))[: int(document["evaluation"]["selected_users"])]
        ].tolist()
    )
    if len(selected) != int(document["evaluation"]["selected_users"]):
        raise RuntimeError("KuaiRand history-residual evaluation coverage differs")
    records = []
    sanity = 0.0
    for index, user in enumerate(selected):
        frame = (
            evaluation[evaluation["user_idx"] == user]
            .sort_values("time_ms")
            .iloc[:maximum]
        )
        history = plan._build_seq(
            user,
            truncate=int(document["data"]["max_original_seq_len"]),
            as_of_timestamp=int(frame["time_ms"].min()),
        )
        prefix = interleave_sequence(history, current.cfg.num_behaviors)
        day_sequence = {
            "item_ids": frame["item_idx"].to_numpy(dtype=np.int64),
            "behaviors": frame["behavior"].to_numpy(dtype=np.int64),
            "time_deltas": frame["time_delta"].to_numpy(dtype=np.float32),
        }
        suffix = interleave_sequence(day_sequence, current.cfg.num_behaviors)
        prefix_tensors = _tensor_sequence(prefix, device)
        suffix_tensors = _tensor_sequence(suffix, device)
        previous_cache = previous.compute_kv(*prefix_tensors)
        current_cache = current.compute_kv(*prefix_tensors)
        previous_hidden, _ = previous.forward_with_cache(previous_cache, *suffix_tensors)
        recompute_hidden, _ = current.forward_with_cache(current_cache, *suffix_tensors)
        reuse_hidden, _ = current.forward_with_cache(previous_cache, *suffix_tensors)
        no_prefix_hidden, _ = current.forward_with_cache(_empty_cache(current_cache), *suffix_tensors)
        if index == 0:
            full_hidden, _ = current(
                torch.cat((prefix_tensors[0], suffix_tensors[0]), dim=1),
                torch.cat((prefix_tensors[1], suffix_tensors[1]), dim=1),
                torch.cat((prefix_tensors[2], suffix_tensors[2]), dim=1),
            )
            sanity = float(
                (full_hidden[:, len(prefix["item_ids"]) :] - recompute_hidden).abs().max().item()
            )
        query_items = torch.as_tensor(
            day_sequence["item_ids"],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        query_deltas = torch.as_tensor(
            day_sequence["time_deltas"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        labels = torch.from_numpy(frame["label"].to_numpy(copy=True)).float()
        metrics = {
            "previous_fresh": _metric_values(
                previous.residual_logits(previous_hidden[:, 0::2], query_items, query_deltas)[0].cpu(),
                labels,
            ),
            "recompute": _metric_values(
                current.residual_logits(recompute_hidden[:, 0::2], query_items, query_deltas)[0].cpu(),
                labels,
            ),
            "reuse": _metric_values(
                current.residual_logits(reuse_hidden[:, 0::2], query_items, query_deltas)[0].cpu(),
                labels,
            ),
            "no_prefix": _metric_values(
                current.residual_logits(no_prefix_hidden[:, 0::2], query_items, query_deltas)[0].cpu(),
                labels,
            ),
        }
        records.append(
            {
                "user_id": user,
                "exposures": len(frame),
                "positives": int(labels.sum().item()),
                "prefix_events": len(history["item_ids"]),
                "metrics": metrics,
            }
        )
        if (index + 1) % 32 == 0:
            print(
                f"phase=history_residual_eval edge={edge} users={index + 1}/{len(selected)}",
                flush=True,
            )
    summary = _summarize(records, document)
    summary.update(
        {
            "edge": edge,
            "update_date": update_date,
            "evaluation_date": eval_date,
            "eligible_users": len(eligible),
            "selected_user_ids_sha256": hashlib.sha256(
                np.asarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
            "same_model_incremental_maximum_absolute_error": sanity,
            "same_model_sanity_passed": sanity <= 1e-4,
            "records": records,
        }
    )
    return summary


def run_evaluation(config_path: str | Path):
    document = load_config(config_path)
    output = Path(document["outputs"]["evaluation_result"])
    if output.is_file():
        return json.loads(output.read_text())
    device = torch.device("cuda:0")
    plan, metadata = load_experiment_plan(document)
    plan.init_base()
    previous = make_model(document, plan, device)
    current = make_model(document, plan, device)
    root = Path(document["outputs"]["checkpoint_root"])
    dates = plan.base_dates + plan.stream_dates
    edges = []
    started = time.monotonic()
    for edge, (update_index, eval_index) in enumerate(
        zip(
            document["evaluation"]["update_date_indices"],
            document["evaluation"]["evaluation_date_indices"],
            strict=True,
        ),
        start=1,
    ):
        update_date = dates[int(update_index)]
        eval_date = dates[int(eval_index)]
        plan.ingest_day(update_date)
        _load_checkpoint(previous, root, edge - 1)
        _load_checkpoint(current, root, edge)
        edges.append(
            _evaluate_edge(
                document,
                plan,
                previous,
                current,
                update_date,
                eval_date,
                edge,
                device,
            )
        )
    result = {
        "protocol": PROTOCOL,
        "status": "complete_development_evaluation",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "edges": edges,
        "decision": {
            "positive_edges": [edge["edge"] for edge in edges if edge["gate"]["passed"]],
            "stable_two_edge_positive": all(edge["gate"]["passed"] for edge in edges),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_result(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != PROTOCOL
        or result.get("status") != "complete_development_evaluation"
        or result.get("scientific_result") is not False
        or len(result.get("edges", [])) != 2
        or not all(edge.get("same_model_sanity_passed") for edge in result["edges"])
    ):
        raise ValueError("KuaiRand history-residual result differs")
