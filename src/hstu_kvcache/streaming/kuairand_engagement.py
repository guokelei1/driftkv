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
import torch
import torch.nn as nn
import torch.nn.functional as F

from hstu_kvcache.data import StreamingDataPlan
from hstu_kvcache.models import (
    HSTU,
    DenseHSTUV2,
    DenseHSTUV2Config,
    HSTUConfig,
    HSTUKVCache,
)

PROTOCOL = "evokv_kuairand_candidate_aware_engagement_v0"
METRICS = (
    "log_loss",
    "brier",
    "roc_auc",
    "average_precision",
    "ndcg_at_10",
    "ndcg_at_50",
)
METHODS = ("previous_fresh", "recompute", "reuse", "no_prefix")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


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


def load_engagement_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data")
    model = document.get("model")
    training = document.get("training")
    evaluation = document.get("evaluation")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or not all(isinstance(value, dict) for value in (data, model, training, evaluation))
        or data.get("base_num_days") != 14
        or data.get("total_num_days") not in (17, 21)
        or data.get("max_users") not in (64, 256)
        or data.get("history_window_days") != 1
        or data.get("preserve_new_item_engagement_labels") is not True
        or training.get("base_epochs") != 1
        or training.get("update_epochs") != 2
        or evaluation.get("update_date_indices") != [14, 15]
        or evaluation.get("evaluation_date_indices") != [15, 16]
        or evaluation.get("selected_users") not in (32, 128)
        or model.get("architecture", "legacy_hstu")
        not in ("legacy_hstu", "dense_hstu_v2")
    ):
        raise ValueError("KuaiRand engagement config differs")
    for source in data.get("standard_logs", []):
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("KuaiRand engagement source binding differs")
    return document


def load_plan(document: dict[str, Any]):
    data = document["data"]
    plan = StreamingDataPlan.from_csvs(
        [value["path"] for value in data["standard_logs"]],
        base_num_days=int(data["base_num_days"]),
        total_num_days=int(data["total_num_days"]),
        max_seq_len=int(data["max_original_seq_len"]),
        max_items=data["max_prediction_items"],
        max_users=int(data["max_users"]),
        min_interactions_per_user=int(data["min_interactions_per_user"]),
        fit_vocabulary_on_base=True,
        context_hash_buckets=int(data["context_hash_buckets"]),
        prediction_items_from_engaged_only=True,
        history_window_days=int(data["history_window_days"]),
    )
    frame = plan.trace.interactions
    frame["label"] = (
        frame["is_click"].astype(bool)
        | frame["is_like"].astype(bool)
        | frame["is_follow"].astype(bool)
        | frame["is_comment"].astype(bool)
        | frame["is_forward"].astype(bool)
        | frame["long_view"].astype(bool)
    ).astype(np.int64)
    plan.daily_segments = {
        date: group.sort_values("time_ms").reset_index(drop=True)
        for date, group in frame.assign(date=frame["date"].astype(str)).groupby("date")
    }
    dates = plan.base_dates + plan.stream_dates
    rows = frame.groupby(frame["date"].astype(str)).size()
    positives = frame.groupby(frame["date"].astype(str))["label"].sum()
    metadata = {
        "base_dates": plan.base_dates,
        "stream_dates": plan.stream_dates,
        "num_users": plan.num_users,
        "num_items": plan.num_items,
        "num_prediction_items": plan.num_prediction_items,
        "num_behaviors": plan.num_behaviors,
        "rows_per_date": {date: int(rows.loc[date]) for date in dates},
        "positives_per_date": {date: int(positives.loc[date]) for date in dates},
        "labels_preserved_for_context_hash_items": True,
    }
    return plan, metadata


class CandidateAwareEngagementModel(nn.Module):
    def __init__(
        self,
        cfg: HSTUConfig | DenseHSTUV2Config,
        architecture: str = "legacy_hstu",
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = DenseHSTUV2(cfg) if architecture == "dense_hstu_v2" else HSTU(cfg)
        self.engagement_head = nn.Linear(cfg.hidden_size, 1)
        nn.init.zeros_(self.engagement_head.bias)

    @property
    def cfg(self) -> HSTUConfig | DenseHSTUV2Config:
        return self.backbone.cfg

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def compute_kv(self, *args, **kwargs):
        return self.backbone.compute_kv(*args, **kwargs)

    def forward_with_cache(self, *args, **kwargs):
        return self.backbone.forward_with_cache(*args, **kwargs)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.engagement_head(hidden).squeeze(-1)


def make_model(document: dict[str, Any], plan: StreamingDataPlan, device: torch.device):
    model = document["model"]
    architecture = str(model.get("architecture", "legacy_hstu"))
    common = {
        "num_items": plan.num_items,
        "num_prediction_items": plan.num_prediction_items,
        "num_behaviors": plan.num_behaviors,
        "hidden_size": int(model["hidden_size"]),
        "num_layers": int(model["num_layers"]),
        "num_heads": int(model["num_heads"]),
        "head_dim": int(model["head_dim"]),
        "max_seq_len": 2 * int(document["data"]["max_original_seq_len"]),
        "temporal_num_freqs": int(model["temporal_num_freqs"]),
        "input_dropout": float(model["input_dropout"]),
    }
    if architecture == "dense_hstu_v2":
        common["max_seq_len"] = 4 * int(document["data"]["max_original_seq_len"])
        cfg = DenseHSTUV2Config(**common)
    else:
        cfg = HSTUConfig(**common, activation=str(model["activation"]))
    return CandidateAwareEngagementModel(cfg, architecture).to(device)


def interleave_batch(batch: dict[str, torch.Tensor], query_behavior: int, device: torch.device):
    items = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    deltas = batch["time_deltas"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch["labels"].to(device).float()
    train_mask = batch["train_mask"].to(device)
    batch_size, width = items.shape
    interleaved_items = torch.zeros(batch_size, 2 * width, dtype=torch.long, device=device)
    interleaved_behaviors = torch.zeros_like(interleaved_items)
    interleaved_deltas = torch.zeros(batch_size, 2 * width, dtype=torch.float32, device=device)
    interleaved_items[:, 0::2] = items
    interleaved_items[:, 1::2] = items
    interleaved_behaviors[:, 0::2] = query_behavior
    interleaved_behaviors[:, 1::2] = behaviors
    interleaved_deltas[:, 0::2] = deltas
    valid = torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    interleaved_items[:, 0::2] *= valid
    interleaved_items[:, 1::2] *= valid
    interleaved_behaviors[:, 0::2] *= valid
    interleaved_behaviors[:, 1::2] *= valid
    return {
        "item_ids": interleaved_items,
        "behaviors": interleaved_behaviors,
        "time_deltas": interleaved_deltas,
        "lengths": 2 * lengths,
        "labels": labels,
        "target_mask": train_mask & valid,
    }


def _train_epoch(model, optimizer, batches, document, device, phase: str):
    model.train()
    loss_sum = 0.0
    targets = 0
    started = time.monotonic()
    query_behavior = model.cfg.num_behaviors
    for batch_index, batch in enumerate(batches):
        value = interleave_batch(batch, query_behavior, device)
        optimizer.zero_grad(set_to_none=True)
        hidden, _ = model(
            value["item_ids"],
            value["behaviors"],
            value["time_deltas"],
            lengths=value["lengths"],
        )
        logits = model.logits(hidden[:, 0::2])
        selected = value["target_mask"]
        loss = F.binary_cross_entropy_with_logits(
            logits[selected],
            value["labels"][selected],
        )
        if not torch.isfinite(loss):
            raise RuntimeError("KuaiRand engagement training produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(document["training"]["gradient_clip_norm"]))
        optimizer.step()
        count = int(selected.sum().item())
        loss_sum += float(loss.detach().item()) * count
        targets += count
        if (batch_index + 1) % 100 == 0:
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


def _save_checkpoint(model, root: Path, version: int, config_path: str | Path, metadata, training):
    model_path, manifest_path = _checkpoint_paths(root, version)
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("KuaiRand engagement checkpoint already exists")
    _atomic_torch(
        model_path,
        {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}},
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_development_checkpoint",
        "scientific_result": False,
        "version": version,
        "architecture": model.architecture,
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(model_path),
        "model": asdict(model.cfg),
        "data": metadata,
        "training": training,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_checkpoint(model, root: Path, version: int):
    model_path, manifest_path = _checkpoint_paths(root, version)
    manifest = json.loads(manifest_path.read_text())
    if manifest["model_sha256"] != file_sha256(model_path):
        raise ValueError("KuaiRand engagement checkpoint binding differs")
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()


def run_training(config_path: str | Path):
    document = load_engagement_config(config_path)
    output = Path(document["outputs"]["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    device = torch.device("cuda:0")
    _seed_everything(int(document["training"]["seed"]))
    plan, metadata = load_plan(document)
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
        training = _train_epoch(model, optimizer, batches, document, device, f"engagement_theta0_e{epoch + 1}")
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
                    f"engagement_theta{version}_e{epoch + 1}",
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


def validate_training_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    versions = result.get("versions", [])
    if (
        result.get("protocol") != PROTOCOL
        or result.get("status") != "complete_development_training"
        or result.get("scientific_result") is not False
        or result.get("config", {}).get("sha256")
        != file_sha256(result.get("config", {}).get("path", ""))
        or [value.get("version") for value in versions] != [0, 1, 2]
        or any(
            value.get("architecture")
            != document["model"].get("architecture", "legacy_hstu")
            for value in versions
        )
    ):
        raise ValueError("KuaiRand engagement training result differs")


def interleave_sequence(sequence: dict[str, np.ndarray], query_behavior: int):
    length = len(sequence["item_ids"])
    items = np.empty(2 * length, dtype=np.int64)
    behaviors = np.empty(2 * length, dtype=np.int64)
    deltas = np.zeros(2 * length, dtype=np.float32)
    items[0::2] = sequence["item_ids"]
    items[1::2] = sequence["item_ids"]
    behaviors[0::2] = query_behavior
    behaviors[1::2] = sequence["behaviors"]
    deltas[0::2] = sequence["time_deltas"]
    return {"item_ids": items, "behaviors": behaviors, "time_deltas": deltas}


def _tensor_sequence(sequence, device):
    return (
        torch.as_tensor(sequence["item_ids"], dtype=torch.long, device=device).unsqueeze(0),
        torch.as_tensor(sequence["behaviors"], dtype=torch.long, device=device).unsqueeze(0),
        torch.as_tensor(sequence["time_deltas"], dtype=torch.float32, device=device).unsqueeze(0),
    )


def _empty_cache(reference: HSTUKVCache):
    return HSTUKVCache(reference.k[:, :, :0], reference.v[:, :, :0], 0)


def _metric_values(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.double()
    labels = labels.double()
    probabilities = torch.sigmoid(logits)
    log_loss = F.binary_cross_entropy_with_logits(logits, labels).item()
    brier = ((probabilities - labels) ** 2).mean().item()
    order = torch.argsort(logits, descending=True, stable=True)
    sorted_labels = labels[order]
    positives = int(labels.sum().item())
    negatives = len(labels) - positives
    positive_ranks = torch.nonzero(sorted_labels > 0, as_tuple=False).flatten().double() + 1.0
    average_precision = float((torch.arange(1, positives + 1, dtype=torch.double) / positive_ranks).mean().item())
    rank_sum = positive_ranks.sum().item()
    roc_auc = (positives * (positives + negatives + 1) - rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    output = {
        "log_loss": float(log_loss),
        "brier": float(brier),
        "roc_auc": float(roc_auc),
        "average_precision": average_precision,
    }
    for cutoff in (10, 50):
        gains = sorted_labels[:cutoff]
        discounts = torch.log2(torch.arange(2, len(gains) + 2, dtype=torch.double)).reciprocal()
        dcg = float((gains * discounts).sum().item())
        ideal_count = min(positives, cutoff)
        ideal = float(discounts[:ideal_count].sum().item())
        output[f"ndcg_at_{cutoff}"] = dcg / ideal if ideal else 0.0
    return output


def _bootstrap(values: np.ndarray, samples: int, seed: int):
    generator = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        selected = generator.integers(0, len(values), size=(count, len(values)))
        output[start : start + count] = values[selected].mean(axis=1)
    return [float(value) for value in np.quantile(output, [0.025, 0.975])]


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
        "fresh_update_ap": comparisons["fresh_update_value"]["average_precision"][
            "positive_direction_with_ci"
        ],
        "history_ap": comparisons["history_value"]["average_precision"]["positive_direction_with_ci"],
        "stale_log_loss": stale["log_loss"]["positive_direction_with_ci"],
        "stale_average_precision": stale["average_precision"]["positive_direction_with_ci"],
        "stale_ndcg_at_50": stale["ndcg_at_50"]["positive_direction_with_ci"],
    }
    gate["passed"] = all(gate.values())
    return {"users": len(records), "endpoints": endpoints, "comparisons": comparisons, "gate": gate}


@torch.no_grad()
def _evaluate_edge(
    document,
    plan,
    previous,
    current,
    update_date,
    eval_date,
    edge,
    device,
    prefix_cap: int | None = None,
    max_exposures: int | None = None,
):
    update_users = set(plan.daily_segments[update_date]["user_idx"].astype(int))
    evaluation = plan.daily_segments[eval_date]
    eligible = []
    for user, frame in evaluation.groupby("user_idx"):
        frame = frame.sort_values("time_ms")
        if max_exposures is not None:
            frame = frame.iloc[:max_exposures]
        user = int(user)
        labels = frame["label"].to_numpy(dtype=np.int64)
        if user in update_users and labels.min() == 0 and labels.max() == 1:
            history = plan._build_seq(
                user,
                truncate=prefix_cap,
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
        raise RuntimeError("KuaiRand engagement evaluation coverage differs")
    query_behavior = current.cfg.num_behaviors
    records = []
    sanity = 0.0
    for index, user in enumerate(selected):
        frame = evaluation[evaluation["user_idx"] == user].sort_values("time_ms")
        if max_exposures is not None:
            frame = frame.iloc[:max_exposures]
        history = plan._build_seq(
            user,
            truncate=prefix_cap,
            as_of_timestamp=int(frame["time_ms"].min()),
        )
        prefix = interleave_sequence(history, query_behavior)
        day_sequence = {
            "item_ids": frame["item_idx"].to_numpy(dtype=np.int64),
            "behaviors": frame["behavior"].to_numpy(dtype=np.int64),
            "time_deltas": frame["time_delta"].to_numpy(dtype=np.float32),
        }
        suffix = interleave_sequence(day_sequence, query_behavior)
        prefix_tensors = _tensor_sequence(prefix, device)
        suffix_tensors = _tensor_sequence(suffix, device)
        previous_cache = previous.compute_kv(*prefix_tensors)
        current_cache = current.compute_kv(*prefix_tensors)
        previous_hidden, _ = previous.forward_with_cache(previous_cache, *suffix_tensors)
        recompute_hidden, _ = current.forward_with_cache(current_cache, *suffix_tensors)
        reuse_hidden, _ = current.forward_with_cache(previous_cache, *suffix_tensors)
        no_prefix_hidden, _ = current.forward_with_cache(_empty_cache(current_cache), *suffix_tensors)
        if index == 0:
            full_items = torch.cat((prefix_tensors[0], suffix_tensors[0]), dim=1)
            full_behaviors = torch.cat((prefix_tensors[1], suffix_tensors[1]), dim=1)
            full_deltas = torch.cat((prefix_tensors[2], suffix_tensors[2]), dim=1)
            full_hidden, _ = current(full_items, full_behaviors, full_deltas)
            sanity = float(
                (full_hidden[:, len(prefix["item_ids"]) :] - recompute_hidden).abs().max().item()
            )
        labels = torch.from_numpy(frame["label"].to_numpy(copy=True)).float()
        query_hidden = {
            "previous_fresh": previous_hidden[0, 0::2],
            "recompute": recompute_hidden[0, 0::2],
            "reuse": reuse_hidden[0, 0::2],
            "no_prefix": no_prefix_hidden[0, 0::2],
        }
        metrics = {
            method: _metric_values(
                (previous if method == "previous_fresh" else current).logits(hidden).cpu(),
                labels,
            )
            for method, hidden in query_hidden.items()
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
        if (index + 1) % 16 == 0:
            print(f"phase=engagement_eval edge={edge} users={index + 1}/{len(selected)}", flush=True)
    summary = _summarize(records, document)
    summary.update(
        {
            "edge": edge,
            "prefix_cap": prefix_cap or int(document["data"]["max_original_seq_len"]),
            "max_exposures": max_exposures,
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
    document = load_engagement_config(config_path)
    output = Path(document["outputs"]["evaluation_result"])
    if output.is_file():
        return json.loads(output.read_text())
    device = torch.device("cuda:0")
    plan, metadata = load_plan(document)
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


def validate_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    if (
        result.get("protocol") != PROTOCOL
        or result.get("status") != "complete_development_evaluation"
        or result.get("scientific_result") is not False
        or len(result.get("edges", [])) != 2
        or not all(edge.get("same_model_sanity_passed") for edge in result["edges"])
    ):
        raise ValueError("KuaiRand engagement result differs")
