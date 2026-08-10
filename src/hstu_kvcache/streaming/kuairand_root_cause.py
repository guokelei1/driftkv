from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data import StreamingDataPlan
from ..models import HSTU, HSTUConfig, HSTUKVCache
from .distributed import close_distributed_runtime, init_distributed_runtime
from .qk_root_cause_sanity import METRICS, _bootstrap_interval, _metric_sums
from .qk_stream_version import cache_relative_error, file_sha256
from .trainer import build_next_item_targets

PROTOCOL = "evokv_root_cause_kuairand_natural_day_v0"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _hash_int_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_document(document: dict[str, object]) -> None:
    campaign = document.get("campaign")
    data = document.get("data")
    schedule = document.get("schedule")
    model = document.get("model")
    training = document.get("training")
    interventions = document.get("interventions")
    quality = document.get("quality")
    execution = document.get("execution")
    outputs = document.get("outputs")
    logs = () if not isinstance(data, dict) else data.get("standard_logs")
    methods = () if not isinstance(interventions, dict) else interventions.get("methods")
    expected_methods = [
        "fresh_full_a",
        "fresh_full_b",
        "stale_previous",
        "zero_prefix",
        "no_prefix",
        "wrong_user_fresh",
        "shuffled_prefix",
        "recent_4",
        "recent_16",
        "recent_64",
    ]
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scope") not in ("implementation_canary", "development_opportunity")
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (
                data,
                schedule,
                model,
                training,
                interventions,
                quality,
                execution,
                outputs,
                campaign,
            )
        )
        or not isinstance(logs, list)
        or len(logs) != 2
        or any(not isinstance(value, dict) for value in logs)
        or data.get("random_log_excluded_from_training") is not True
        or data.get("fit_vocabulary_on_base") is not True
        or data.get("prediction_items_from_engaged_only") is not True
        or int(data.get("base_num_days", 0)) != 14
        or int(data.get("total_num_days", 0)) != 17
        or int(data.get("history_window_days", 0)) != 1
        or int(data.get("context_hash_buckets", 0)) < 1
        or schedule.get("update_date_indices") != [14, 15]
        or schedule.get("evaluation_date_indices") != [15, 16]
        or int(model.get("hidden_size", 0)) < 1
        or int(model.get("num_layers", 0)) < 1
        or int(model.get("num_heads", 0)) < 1
        or int(model.get("head_dim", 0)) < 1
        or int(training.get("negative_count", 0)) < 1
        or int(training.get("base_epochs", 0)) < 1
        or int(training.get("update_epochs", 0)) < 1
        or training.get("training_sequences") != "all_chunks"
        or methods != expected_methods
        or interventions.get("recent_lengths") != [4, 16, 64]
        or int(quality.get("record_limit_per_rank", 0)) < 1
        or int(quality.get("target_chunk", 0)) < 1
        or int(quality.get("full_catalog_item_chunk", 0)) < 1
        or int(quality.get("suffix_chunk", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or execution.get("training_cuda_visible_devices") != "0"
        or execution.get("evaluation_cuda_visible_devices") != "0,1"
        or int(execution.get("evaluation_world_size", 0)) != 2
        or not isinstance(campaign.get("path"), str)
        or not isinstance(campaign.get("sha256"), str)
    ):
        raise ValueError("KuaiRand root-cause config differs")


def load_plan(document: dict[str, object]) -> tuple[StreamingDataPlan, dict[str, object]]:
    campaign = document["campaign"]
    if file_sha256(Path(campaign["path"])) != campaign["sha256"]:
        raise ValueError("KuaiRand root-cause campaign hash differs")
    data = document["data"]
    for source in data["standard_logs"]:
        path = Path(source["path"])
        if file_sha256(path) != source["sha256"]:
            raise ValueError(f"KuaiRand source hash differs: {path}")
    plan = StreamingDataPlan.from_csvs(
        [source["path"] for source in data["standard_logs"]],
        base_num_days=int(data["base_num_days"]),
        total_num_days=int(data["total_num_days"]),
        max_seq_len=int(data["max_seq_len"]),
        max_items=data["max_prediction_items"],
        max_users=data["max_users"],
        min_interactions_per_user=int(data["min_interactions_per_user"]),
        fit_vocabulary_on_base=True,
        context_hash_buckets=int(data["context_hash_buckets"]),
        prediction_items_from_engaged_only=True,
        history_window_days=int(data["history_window_days"]),
    )
    ordered_users = np.asarray(
        [value for value, _ in sorted(plan.trace.user_map.items(), key=lambda item: item[1])],
        dtype=np.int64,
    )
    ordered_items = np.asarray(
        [value for value, _ in sorted(plan.trace.item_map.items(), key=lambda item: item[1])],
        dtype=np.int64,
    )
    frame = plan.trace.interactions
    date_rows = frame.groupby(frame["date"].astype(str)).size()
    date_targets = frame.groupby(frame["date"].astype(str))["label"].sum()
    dates = plan.base_dates + plan.stream_dates
    metadata = {
        "base_dates": plan.base_dates,
        "stream_dates": plan.stream_dates,
        "num_users": plan.num_users,
        "num_context_items": plan.num_items,
        "num_prediction_items": plan.num_prediction_items,
        "num_behaviors": plan.num_behaviors,
        "selected_rows": len(frame),
        "eligible_engaged_targets": int(frame["label"].sum()),
        "rows_per_date": {date: int(date_rows.loc[date]) for date in dates},
        "eligible_targets_per_date": {
            date: int(date_targets.loc[date]) for date in dates
        },
        "raw_user_ids_sha256": _hash_int_array(ordered_users),
        "prediction_video_ids_sha256": _hash_int_array(ordered_items),
        "prediction_catalog_rule": "all engaged item ids observed in the first 14 dates",
        "context_rule": "all exposures retained; non-catalog ids use stable context-only buckets",
    }
    return plan, metadata


def make_model(document: dict[str, object], plan: StreamingDataPlan, device: torch.device) -> HSTU:
    model = document["model"]
    cfg = HSTUConfig(
        num_items=plan.num_items,
        num_prediction_items=plan.num_prediction_items,
        num_behaviors=plan.num_behaviors,
        hidden_size=int(model["hidden_size"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        head_dim=int(model["head_dim"]),
        max_seq_len=int(document["data"]["max_seq_len"]),
        temporal_num_freqs=int(model["temporal_num_freqs"]),
        activation=str(model["activation"]),
        input_dropout=float(model["input_dropout"]),
        tie_item_embeddings=bool(model.get("tie_item_embeddings", True)),
    )
    return HSTU(cfg).to(device)


def _checkpoint_paths(root: Path, version: int) -> tuple[Path, Path]:
    directory = root / f"theta_{version}"
    return directory / "model.pt", directory / "manifest.json"


def _valid_checkpoint(
    root: Path,
    version: int,
    config_sha256: str,
    data_metadata: dict[str, object],
) -> bool:
    model_path, manifest_path = _checkpoint_paths(root, version)
    if not model_path.exists() or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return bool(
        manifest.get("protocol") == PROTOCOL
        and manifest.get("version") == version
        and manifest.get("config_sha256") == config_sha256
        and manifest.get("prediction_video_ids_sha256")
        == data_metadata["prediction_video_ids_sha256"]
        and manifest.get("model_sha256") == file_sha256(model_path)
    )


def _save_checkpoint(
    model: HSTU,
    root: Path,
    version: int,
    config_path: Path,
    data_metadata: dict[str, object],
    training_record: dict[str, object],
) -> dict[str, object]:
    model_path, manifest_path = _checkpoint_paths(root, version)
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError(f"partial or existing theta{version} checkpoint")
    _atomic_torch(model_path, model.state_dict())
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_development_checkpoint",
        "scientific_result": False,
        "formal_result": False,
        "version": version,
        "config_sha256": file_sha256(config_path),
        "program_sha256": file_sha256(Path(__file__)),
        "model_sha256": file_sha256(model_path),
        "prediction_video_ids_sha256": data_metadata["prediction_video_ids_sha256"],
        "model": asdict(model.cfg),
        "training": training_record,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_checkpoint(model: HSTU, root: Path, version: int) -> None:
    model_path, _ = _checkpoint_paths(root, version)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))


def _training_step(
    model: HSTU,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    negative_count: int,
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
    if bool(torch.any(targets[valid] > model.cfg.num_prediction_items)):
        raise ValueError("KuaiRand valid target lies outside prediction catalog")
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    positives = safe_targets.unsqueeze(-1)
    negatives = torch.randint(
        1,
        model.cfg.num_prediction_items + 1,
        (*targets.shape, negative_count),
        device=device,
    )
    negatives = torch.where(
        negatives == positives,
        negatives.remainder(model.cfg.num_prediction_items) + 1,
        negatives,
    )
    candidates = torch.cat((positives, negatives), dim=-1)
    logits = model.score_hidden(hidden[:, :-1], candidates)
    labels_zero = torch.zeros_like(safe_targets)
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
    model: HSTU,
    optimizer: torch.optim.Optimizer,
    batches,
    device: torch.device,
    negative_count: int,
    maximum_batches: int | None,
    phase: str,
) -> dict[str, object]:
    loss_sum = 0.0
    targets = 0
    sequences = 0
    tokens = 0
    completed_batches = 0
    for batch_index, batch in enumerate(batches):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        loss, count = _training_step(model, batch, optimizer, device, negative_count)
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
        raise RuntimeError(f"{phase} has no eligible training targets")
    return {
        "batches": completed_batches,
        "sequences": sequences,
        "tokens": tokens,
        "eligible_targets": targets,
        "target_weighted_sampled_cross_entropy": loss_sum / targets,
        "truncated_by_canary_batch_limit": maximum_batches is not None,
    }


def run_training(config_path: Path) -> dict[str, object]:
    document = json.loads(config_path.read_text())
    validate_document(document)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand root-cause training is single-rank")
    output = Path(document["outputs"]["training_result"])
    if output.exists():
        result = json.loads(output.read_text())
        if result.get("status") != "complete_development_training":
            raise FileExistsError("KuaiRand training result exists but is not complete")
        return result
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(document["training"]["seed"]))
    started = time.perf_counter()
    plan, data_metadata = load_plan(document)
    plan.init_base()
    model = make_model(document, plan, device)
    training = document["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["base_lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    root = Path(document["outputs"]["checkpoint_root"])
    config_sha256 = file_sha256(config_path)
    program_sha256 = file_sha256(Path(__file__))
    resume_path = root / "resume.pt"
    completed_version = -1
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume.get("protocol") != PROTOCOL
            or resume.get("config_sha256") != config_sha256
            or resume.get("program_sha256", program_sha256) != program_sha256
            or not _valid_checkpoint(
                root,
                int(resume.get("version", -1)),
                config_sha256,
                data_metadata,
            )
        ):
            raise ValueError("KuaiRand training resume state differs")
        completed_version = int(resume["version"])
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
    records: list[dict[str, object]] = []
    for version in range(completed_version + 1):
        _, manifest_path = _checkpoint_paths(root, version)
        records.append(json.loads(manifest_path.read_text())["training"])
    if completed_version < 0:
        epochs = []
        for epoch in range(int(training["base_epochs"])):
            np.random.seed(int(training["seed"]) + epoch)
            batches = plan.iter_base_train_batches(
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
                    training["maximum_base_batches_per_epoch"],
                    f"kuairand_theta0_epoch{epoch + 1}",
                )
            )
        record = {
            "version": 0,
            "role": "base",
            "dates": plan.base_dates,
            "epochs": epochs,
        }
        _save_checkpoint(model, root, 0, config_path, data_metadata, record)
        _atomic_torch(
            resume_path,
            {
                "protocol": PROTOCOL,
                "config_sha256": config_sha256,
                "program_sha256": program_sha256,
                "version": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
        )
        records.append(record)
        completed_version = 0
    else:
        for date in plan.stream_dates[:completed_version]:
            plan.ingest_day(date)
    for group in optimizer.param_groups:
        group["lr"] = float(training["stream_lr"])
    update_indices = [int(value) for value in document["schedule"]["update_date_indices"]]
    for version, date_index in enumerate(update_indices, start=1):
        date = (plan.base_dates + plan.stream_dates)[date_index]
        if version <= completed_version:
            continue
        plan.ingest_day(date)
        epochs = []
        for epoch in range(int(training["update_epochs"])):
            np.random.seed(int(training["seed"]) + version * 1009 + epoch)
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
                    training["maximum_update_batches_per_epoch"],
                    f"kuairand_theta{version}_epoch{epoch + 1}",
                )
            )
        record = {
            "version": version,
            "role": "stream_update",
            "date": date,
            "epochs": epochs,
        }
        _save_checkpoint(model, root, version, config_path, data_metadata, record)
        _atomic_torch(
            resume_path,
            {
                "protocol": PROTOCOL,
                "config_sha256": config_sha256,
                "program_sha256": program_sha256,
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
        "protocol": PROTOCOL,
        "status": "complete_development_training",
        "scope": document["scope"],
        "scientific_result": False,
        "formal_result": False,
        "round_id": document["round_id"],
        "config": {"path": str(config_path), "sha256": config_sha256},
        "programs": {
            "runner": {
                "path": "src/hstu_kvcache/streaming/kuairand_root_cause.py",
                "sha256": file_sha256(Path(__file__)),
            }
        },
        "data": data_metadata,
        "model": asdict(model.cfg),
        "training": training,
        "versions": records,
        "execution": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "world_size": 1,
            "runtime_seconds": time.perf_counter() - started,
            "qualification_consumed": False,
            "final_consumed": False,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _atomic_json(output, result)
    return result


def _stored_cache(model: HSTU, sequence: dict[str, np.ndarray], device: torch.device) -> HSTUKVCache:
    items = torch.from_numpy(sequence["item_ids"]).long().unsqueeze(0).to(device)
    behaviors = torch.from_numpy(sequence["behaviors"]).long().unsqueeze(0).to(device)
    deltas = torch.from_numpy(sequence["time_deltas"]).float().unsqueeze(0).to(device)
    lengths = torch.tensor([items.shape[1]], dtype=torch.int64, device=device)
    cache = model.compute_kv(items, behaviors, deltas, lengths)
    return HSTUKVCache(
        k=cache.k.to(torch.float16).to(torch.float32),
        v=cache.v.to(torch.float16).to(torch.float32),
        seq_len=cache.seq_len,
    )


def _empty_cache(reference: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(reference.k[:, :, :0].clone(), reference.v[:, :, :0].clone(), 0)


def _zero_cache(reference: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(torch.zeros_like(reference.k), torch.zeros_like(reference.v), reference.seq_len)


def _slice_sequence(sequence: dict[str, np.ndarray], start: int, stop: int | None = None) -> dict[str, np.ndarray]:
    return {
        name: value[start:stop]
        for name, value in sequence.items()
        if name in ("item_ids", "behaviors", "time_deltas")
    }


def _length_matched_donor(sequence: dict[str, np.ndarray], length: int) -> dict[str, np.ndarray]:
    if len(sequence["item_ids"]) < 1:
        raise ValueError("wrong-user donor sequence is empty")
    output = {}
    for name in ("item_ids", "behaviors", "time_deltas"):
        values = sequence[name]
        if len(values) >= length:
            output[name] = values[-length:].copy()
        else:
            output[name] = np.resize(values, length)
    return output


def _evaluation_sequence(plan: StreamingDataPlan, user: int, date: str) -> dict[str, object]:
    day = plan.daily_segments[date]
    user_day = day[day["user_idx"] == user].sort_values("time_ms")
    history = plan._build_seq(user, as_of_timestamp=int(user_day["time_ms"].min()))
    if history is None or len(history["item_ids"]) < 2:
        raise ValueError("evaluation history is too short")
    prefix = {
        "item_ids": history["item_ids"][:-1].copy(),
        "behaviors": history["behaviors"][:-1].copy(),
        "time_deltas": history["time_deltas"][:-1].copy(),
    }
    suffix_items = np.concatenate(
        (history["item_ids"][-1:], user_day["item_idx"].to_numpy(dtype=np.int64)[:-1])
    )
    suffix_behaviors = np.concatenate(
        (history["behaviors"][-1:], user_day["behavior"].to_numpy(dtype=np.int64)[:-1])
    )
    suffix_deltas = np.concatenate(
        (history["time_deltas"][-1:], user_day["time_delta"].to_numpy(dtype=np.float32)[:-1])
    )
    targets = user_day["item_idx"].to_numpy(dtype=np.int64)
    labels = user_day["label"].to_numpy(dtype=np.bool_)
    return {
        "prefix": prefix,
        "suffix": {
            "item_ids": suffix_items,
            "behaviors": suffix_behaviors,
            "time_deltas": suffix_deltas,
        },
        "targets": targets,
        "labels": labels,
        "last_context_items": suffix_items,
        "available_prefix_length": int(history["available_length_before_token_cap"]) - 1,
    }


def _eligible_users(plan: StreamingDataPlan, update_date: str, eval_date: str) -> list[int]:
    update = plan.daily_segments[update_date]
    evaluation = plan.daily_segments[eval_date]
    update_users = set(update.loc[update["label"] > 0, "user_idx"].astype(int))
    eval_users = set(evaluation.loc[evaluation["label"] > 0, "user_idx"].astype(int))
    first_timestamps = evaluation.groupby("user_idx")["time_ms"].min()
    output = []
    for user in sorted(update_users & eval_users):
        history = plan._build_seq(user, as_of_timestamp=int(first_timestamps.loc[user]))
        if history is not None and len(history["item_ids"]) >= 2:
            output.append(user)
    return output


def _selected_users(
    plan: StreamingDataPlan,
    update_date: str,
    eval_date: str,
    total: int,
    seed: int,
    cap_to_eligible: bool = False,
) -> tuple[list[int], int]:
    eligible = _eligible_users(plan, update_date, eval_date)
    if len(eligible) < total:
        if not cap_to_eligible:
            raise RuntimeError("KuaiRand root-cause evaluation user coverage differs")
        total = len(eligible)
    generator = np.random.default_rng(seed)
    selected = sorted(np.asarray(eligible)[generator.permutation(len(eligible))[:total]].tolist())
    return selected, len(eligible)


@torch.no_grad()
def _run_suffix(
    model: HSTU,
    cache: HSTUKVCache,
    suffix: dict[str, np.ndarray],
    labels: np.ndarray,
    chunk: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    current = cache
    for start in range(0, len(suffix["item_ids"]), chunk):
        stop = min(start + chunk, len(suffix["item_ids"]))
        items = torch.from_numpy(suffix["item_ids"][start:stop]).long().unsqueeze(0).to(device)
        behaviors = torch.from_numpy(suffix["behaviors"][start:stop]).long().unsqueeze(0).to(device)
        deltas = torch.from_numpy(suffix["time_deltas"][start:stop]).float().unsqueeze(0).to(device)
        hidden, current = model.forward_with_cache(current, items, behaviors, deltas)
        mask = torch.from_numpy(labels[start:stop]).to(device)
        if bool(torch.any(mask)):
            values.append(hidden[0][mask].detach().cpu())
    if not values:
        return torch.empty((0, model.cfg.hidden_size), dtype=torch.float32)
    return torch.cat(values)


@torch.no_grad()
def _incremental_parity(
    model: HSTU,
    sequence: dict[str, object],
    device: torch.device,
) -> float:
    prefix = sequence["prefix"]
    suffix = _slice_sequence(sequence["suffix"], 0, min(16, len(sequence["suffix"]["item_ids"])))
    prefix = _slice_sequence(prefix, max(0, len(prefix["item_ids"]) - 128))
    items = np.concatenate((prefix["item_ids"], suffix["item_ids"]))
    behaviors = np.concatenate((prefix["behaviors"], suffix["behaviors"]))
    deltas = np.concatenate((prefix["time_deltas"], suffix["time_deltas"]))
    tensors = (
        torch.from_numpy(items).long().unsqueeze(0).to(device),
        torch.from_numpy(behaviors).long().unsqueeze(0).to(device),
        torch.from_numpy(deltas).float().unsqueeze(0).to(device),
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        full, _ = model(*tensors, return_kv=False)
        prefix_tensors = tuple(value[:, : len(prefix["item_ids"])] for value in tensors)
        suffix_tensors = tuple(value[:, len(prefix["item_ids"]):] for value in tensors)
        cache = model.compute_kv(*prefix_tensors)
        incremental, _ = model.forward_with_cache(cache, *suffix_tensors)
        return float(
            (incremental - full[:, len(prefix["item_ids"]):]).abs().max().item()
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


@torch.no_grad()
def _full_catalog_pair(
    model: HSTU,
    hidden: torch.Tensor,
    positives: torch.Tensor,
    target_chunk: int,
    item_chunk: int,
    device: torch.device,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or hidden.shape[0] != 2 or hidden.shape[1] != len(positives):
        raise ValueError("KuaiRand full-catalog pair differs")
    nll_parts = []
    rank_parts = []
    weights = model.prediction_item_weight
    for start in range(0, len(positives), target_chunk):
        stop = min(start + target_chunk, len(positives))
        local_hidden = hidden[:, start:stop].to(device)
        local_positive = positives[start:stop].to(device)
        positive_vectors = weights.index_select(0, local_positive)
        positive_scores = torch.einsum("mth,th->mt", local_hidden, positive_vectors)
        lse = torch.full_like(positive_scores, -torch.inf)
        ranks = torch.zeros_like(positive_scores, dtype=torch.int64)
        for item_start in range(1, model.cfg.num_prediction_items + 1, item_chunk):
            item_stop = min(item_start + item_chunk, model.cfg.num_prediction_items + 1)
            scores = torch.matmul(local_hidden, weights[item_start:item_stop].t())
            lse = torch.logaddexp(lse, torch.logsumexp(scores.float(), dim=-1))
            ids = torch.arange(item_start, item_stop, dtype=torch.int64, device=device)
            positive_mask = local_positive[:, None] == ids[None, :]
            ranks += (
                (scores >= positive_scores.unsqueeze(-1)) & ~positive_mask.unsqueeze(0)
            ).sum(dim=-1)
        nll_parts.append((lse - positive_scores).detach().cpu())
        rank_parts.append((ranks + 1).detach().cpu())
        print(f"phase={phase} targets={stop}/{len(positives)}", flush=True)
    return torch.cat(nll_parts, dim=1), torch.cat(rank_parts, dim=1)


def _parameter_distances(previous: HSTU, current: HSTU) -> dict[str, dict[str, float]]:
    groups = {
        "item_embedding": lambda name: name in ("item_emb.weight", "output_emb.weight"),
        "input_item_embedding": lambda name: name == "item_emb.weight",
        "output_item_embedding": lambda name: name == "output_emb.weight",
        "input_encoders": lambda name: name.startswith(("behavior_emb.", "temporal_enc.", "in_proj.")),
        "kv_projections": lambda name: ".attn.k_proj." in name or ".attn.v_proj." in name,
        "other_core": lambda name: name not in ("item_emb.weight", "output_emb.weight") and not name.startswith(("behavior_emb.", "temporal_enc.", "in_proj.")) and ".attn.k_proj." not in name and ".attn.v_proj." not in name,
    }
    previous_state = previous.state_dict()
    current_state = current.state_dict()
    output = {}
    for group, predicate in groups.items():
        numerator = 0.0
        denominator = 0.0
        parameters = 0
        for name, value in previous_state.items():
            if predicate(name):
                numerator += float((current_state[name] - value).double().square().sum().item())
                denominator += float(value.double().square().sum().item())
                parameters += value.numel()
        output[group] = {
            "parameters": parameters,
            "relative_l2_update": math.sqrt(numerator) / max(math.sqrt(denominator), 1e-12),
        }
    return output


def _popularity_ranks(plan: StreamingDataPlan) -> np.ndarray:
    base = torch.zeros(plan.num_prediction_items + 1, dtype=torch.int64)
    for date in plan.base_dates:
        frame = plan.daily_segments[date]
        values = frame.loc[frame["label"] > 0, "item_idx"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
        base += torch.bincount(torch.from_numpy(values), minlength=len(base))
    ids = np.arange(1, plan.num_prediction_items + 1, dtype=np.int64)
    counts = base[1:].numpy()
    order = np.lexsort((ids, -counts))
    ranks = np.zeros(plan.num_prediction_items + 1, dtype=np.int64)
    ranks[ids[order]] = np.arange(1, len(ids) + 1, dtype=np.int64)
    return ranks


def _ranking_metric_sums(ranks: torch.Tensor) -> dict[str, float]:
    values = _metric_sums(torch.zeros(len(ranks)), ranks)
    values.pop("cross_entropy")
    return values


def _metric_comparison(
    records: list[dict[str, object]],
    left_name: str,
    right_name: str,
    left_getter,
    right_getter,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    output = {}
    for metric_index, metric in enumerate(METRICS):
        left = np.asarray([left_getter(value)[metric] for value in records], dtype=np.float64)
        right = np.asarray([right_getter(value)[metric] for value in records], dtype=np.float64)
        oriented = left - right if metric == "cross_entropy" else right - left
        absolute = float(oriented.sum() / denominator)
        right_mean = float(right.sum() / denominator)
        interval = _bootstrap_interval(
            targets,
            oriented,
            bootstrap_samples,
            bootstrap_seed + metric_index,
        )
        output[metric] = {
            left_name: float(left.sum() / denominator),
            right_name: right_mean,
            f"{right_name}_advantage_absolute": absolute,
            f"{right_name}_advantage_relative_percent": 100.0 * absolute / abs(right_mean) if right_mean else None,
            "user_cluster_95_interval": interval,
            f"{right_name}_advantage_positive_with_ci": interval[0] > 0,
        }
    return output


def _aggregate_edge(
    records: list[dict[str, object]],
    methods: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    records = sorted(records, key=lambda value: int(value["user_id"]))
    targets = np.asarray([value["targets"] for value in records], dtype=np.float64)
    denominator = float(targets.sum())
    endpoints = {
        method: {
            metric: float(sum(value["metric_sums"][method][metric] for value in records) / denominator)
            for metric in METRICS
        }
        for method in [*methods, "previous_fresh"]
    }
    fresh_comparisons = {
        method: _metric_comparison(
            records,
            method,
            "fresh_current",
            lambda value, selected=method: value["metric_sums"][selected],
            lambda value: value["metric_sums"]["fresh_full_a"],
            bootstrap_samples,
            bootstrap_seed + index * 101,
        )
        for index, method in enumerate(methods)
    }
    update_value = _metric_comparison(
        records,
        "previous_fresh",
        "current_fresh",
        lambda value: value["metric_sums"]["previous_fresh"],
        lambda value: value["metric_sums"]["fresh_full_a"],
        bootstrap_samples,
        bootstrap_seed + 10001,
    )
    ranking_baselines = {}
    for baseline in ("base_popularity", "repeat_last_item"):
        ranking_baselines[baseline] = {
            metric: float(
                sum(value["ranking_baselines"][baseline][metric] for value in records)
                / denominator
            )
            for metric in METRICS
            if metric != "cross_entropy"
        }
    sanity = {
        "fresh_duplicate_cache_maximum_absolute_error": max(
            value["sanity"]["fresh_duplicate_cache_maximum_absolute_error"] for value in records
        ),
        "fresh_duplicate_hidden_maximum_absolute_error": max(
            value["sanity"]["fresh_duplicate_hidden_maximum_absolute_error"] for value in records
        ),
        "fresh_duplicate_nll_maximum_absolute_error": max(
            value["sanity"]["fresh_duplicate_nll_maximum_absolute_error"] for value in records
        ),
        "fresh_duplicate_ranks_equal": all(
            value["sanity"]["fresh_duplicate_ranks_equal"] for value in records
        ),
        "incremental_full_forward_maximum_absolute_error": max(
            value["sanity"]["incremental_full_forward_maximum_absolute_error"] for value in records
        ),
    }
    sanity["implementation_passed"] = bool(
        sanity["fresh_duplicate_cache_maximum_absolute_error"] == 0.0
        and sanity["fresh_duplicate_hidden_maximum_absolute_error"] == 0.0
        and sanity["fresh_duplicate_nll_maximum_absolute_error"] <= 1e-7
        and sanity["fresh_duplicate_ranks_equal"]
        and sanity["incremental_full_forward_maximum_absolute_error"] <= 1e-4
    )
    cache_errors = {}
    hidden_errors = {}
    for method in methods:
        cache_values = [value["cache_relative_error"].get(method) for value in records]
        cache_values = [value for value in cache_values if value is not None]
        cache_errors[method] = {
            "records": len(cache_values),
            "mean": float(np.mean(cache_values)) if cache_values else None,
            "median": float(np.median(cache_values)) if cache_values else None,
            "maximum": float(np.max(cache_values)) if cache_values else None,
        }
        values = [value["hidden_relative_error"][method] for value in records]
        hidden_errors[method] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "users": len(records),
        "positive_targets": int(denominator),
        "endpoints": endpoints,
        "fresh_current_comparisons": fresh_comparisons,
        "previous_to_current_fresh_update_value": update_value,
        "ranking_baselines": ranking_baselines,
        "cache_relative_error_from_fresh": cache_errors,
        "hidden_relative_error_from_fresh": hidden_errors,
        "sanity": sanity,
        "history_lengths": {
            "minimum": min(value["prefix_length"] for value in records),
            "median": float(np.median([value["prefix_length"] for value in records])),
            "p95": float(np.percentile([value["prefix_length"] for value in records], 95)),
            "maximum": max(value["prefix_length"] for value in records),
            "distinct": len(set(value["prefix_length"] for value in records)),
            "at_model_cap_fraction": float(
                np.mean([value["prefix_token_truncated"] for value in records])
            ),
        },
        "records_detail": records,
    }


@torch.no_grad()
def _evaluate_edge(
    document: dict[str, object],
    plan: StreamingDataPlan,
    previous: HSTU,
    current: HSTU,
    version: int,
    update_date: str,
    eval_date: str,
    popularity_ranks: np.ndarray,
    runtime,
) -> dict[str, object] | None:
    quality = document["quality"]
    methods = list(document["interventions"]["methods"])
    selected, eligible = _selected_users(
        plan,
        update_date,
        eval_date,
        int(quality["record_limit_per_rank"]) * runtime.world_size,
        int(quality["sampling_seed"]) + version * 1009,
        bool(quality.get("cap_user_limit_to_eligible", False)),
    )
    local_users = selected[runtime.rank::runtime.world_size]
    donors = {user: selected[(selected.index(user) + 1) % len(selected)] for user in local_users}
    buffers = []
    suffix_chunk = int(quality["suffix_chunk"])
    for ordinal, user in enumerate(local_users):
        sequence = _evaluation_sequence(plan, user, eval_date)
        donor_sequence = _evaluation_sequence(plan, donors[user], eval_date)
        prefix = sequence["prefix"]
        fresh_a = _stored_cache(current, prefix, runtime.device)
        fresh_b = _stored_cache(current, prefix, runtime.device)
        stale = _stored_cache(previous, prefix, runtime.device)
        generator = np.random.default_rng(
            int(document["interventions"]["shuffle_seed"]) + version * 1000003 + user
        )
        permutation = generator.permutation(len(prefix["item_ids"]))
        shuffled = {
            "item_ids": prefix["item_ids"][permutation],
            "behaviors": prefix["behaviors"][permutation],
            "time_deltas": prefix["time_deltas"],
        }
        donor_prefix = _length_matched_donor(donor_sequence["prefix"], len(prefix["item_ids"]))
        caches = {
            "fresh_full_a": fresh_a,
            "fresh_full_b": fresh_b,
            "stale_previous": stale,
            "zero_prefix": _zero_cache(fresh_a),
            "no_prefix": _empty_cache(fresh_a),
            "wrong_user_fresh": _stored_cache(current, donor_prefix, runtime.device),
            "shuffled_prefix": _stored_cache(current, shuffled, runtime.device),
        }
        for recent in document["interventions"]["recent_lengths"]:
            caches[f"recent_{recent}"] = _stored_cache(
                current,
                _slice_sequence(prefix, max(0, len(prefix["item_ids"]) - int(recent))),
                runtime.device,
            )
        hidden = {
            method: _run_suffix(
                current,
                caches[method],
                sequence["suffix"],
                sequence["labels"],
                suffix_chunk,
                runtime.device,
            )
            for method in methods
        }
        previous_hidden = _run_suffix(
            previous,
            _stored_cache(previous, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        positive_targets = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
        if len(positive_targets) != len(previous_hidden) or any(
            len(value) != len(positive_targets) for value in hidden.values()
        ):
            raise RuntimeError("KuaiRand evaluation target alignment differs")
        cache_errors = {
            method: (
                cache_relative_error(caches[method], fresh_a)
                if caches[method].k.shape == fresh_a.k.shape
                else None
            )
            for method in methods
        }
        hidden_errors = {
            method: float(
                torch.linalg.vector_norm((hidden[method] - hidden["fresh_full_a"]).double())
                / torch.linalg.vector_norm(hidden["fresh_full_a"].double()).clamp_min(1e-12)
            )
            for method in methods
        }
        parity = _incremental_parity(current, sequence, runtime.device)
        cache_duplicate = float(
            max(
                (fresh_a.k - fresh_b.k).abs().max().item(),
                (fresh_a.v - fresh_b.v).abs().max().item(),
            )
        )
        buffers.append(
            {
                "user_id": user,
                "donor_user_id": donors[user],
                "targets": len(positive_targets),
                "positive_targets": positive_targets,
                "hidden": hidden,
                "previous_hidden": previous_hidden,
                "cache_relative_error": cache_errors,
                "hidden_relative_error": hidden_errors,
                "prefix_length": len(prefix["item_ids"]),
                "available_prefix_length": sequence["available_prefix_length"],
                "prefix_token_truncated": sequence["available_prefix_length"] > len(prefix["item_ids"]),
                "last_context_items": torch.from_numpy(
                    sequence["last_context_items"][sequence["labels"]]
                ).long(),
                "sanity": {
                    "fresh_duplicate_cache_maximum_absolute_error": cache_duplicate,
                    "fresh_duplicate_hidden_maximum_absolute_error": float(
                        (hidden["fresh_full_a"] - hidden["fresh_full_b"]).abs().max().item()
                    ),
                    "incremental_full_forward_maximum_absolute_error": parity,
                },
            }
        )
        del caches, hidden, previous_hidden, fresh_a, fresh_b, stale
        print(
            f"phase=kuairand_edge{version}_hidden rank={runtime.rank} "
            f"user={ordinal + 1}/{len(local_users)} targets={len(positive_targets)}",
            flush=True,
        )
    positives = torch.cat([value["positive_targets"] for value in buffers])
    hidden_by_method = {
        method: torch.cat([value["hidden"][method] for value in buffers])
        for method in methods
    }
    pairs = [methods[index:index + 2] for index in range(0, len(methods), 2)]
    method_scores = {}
    for pair in pairs:
        nll, ranks = _full_catalog_pair(
            current,
            torch.stack([hidden_by_method[name] for name in pair]),
            positives,
            int(quality["target_chunk"]),
            int(quality["full_catalog_item_chunk"]),
            runtime.device,
            f"kuairand_edge{version}_{pair[0]}_{pair[1]}_rank{runtime.rank}",
        )
        for row, name in enumerate(pair):
            method_scores[name] = (nll[row], ranks[row])
    previous_values = torch.cat([value["previous_hidden"] for value in buffers])
    previous_nll, previous_ranks = _full_catalog_pair(
        previous,
        torch.stack((previous_values, previous_values.clone())),
        positives,
        int(quality["target_chunk"]),
        int(quality["full_catalog_item_chunk"]),
        runtime.device,
        f"kuairand_edge{version}_previous_fresh_rank{runtime.rank}",
    )
    start = 0
    records = []
    for value in buffers:
        stop = start + value["targets"]
        metric_sums = {
            method: _metric_sums(
                method_scores[method][0][start:stop],
                method_scores[method][1][start:stop],
            )
            for method in methods
        }
        metric_sums["previous_fresh"] = _metric_sums(
            previous_nll[0, start:stop], previous_ranks[0, start:stop]
        )
        targets = value["positive_targets"].numpy()
        popularity = torch.from_numpy(popularity_ranks[targets]).long()
        repeat_last = torch.where(
            value["last_context_items"] == value["positive_targets"],
            torch.ones(value["targets"], dtype=torch.int64),
            torch.full(
                (value["targets"],),
                current.cfg.num_prediction_items,
                dtype=torch.int64,
            ),
        )
        record = {
            key: selected_value
            for key, selected_value in value.items()
            if key not in (
                "positive_targets",
                "hidden",
                "previous_hidden",
                "last_context_items",
            )
        }
        record["metric_sums"] = metric_sums
        record["ranking_baselines"] = {
            "base_popularity": _ranking_metric_sums(popularity),
            "repeat_last_item": _ranking_metric_sums(repeat_last),
        }
        record["sanity"]["fresh_duplicate_nll_maximum_absolute_error"] = float(
            (method_scores["fresh_full_a"][0][start:stop] - method_scores["fresh_full_b"][0][start:stop]).abs().max().item()
        )
        record["sanity"]["fresh_duplicate_ranks_equal"] = bool(
            torch.equal(
                method_scores["fresh_full_a"][1][start:stop],
                method_scores["fresh_full_b"][1][start:stop],
            )
        )
        records.append(record)
        start = stop
    gathered: list[object] | None = [None] * runtime.world_size if runtime.is_primary else None
    dist.gather_object(records, gathered, dst=0)
    if not runtime.is_primary:
        return None
    combined = [record for shard in gathered for record in shard]
    result = _aggregate_edge(
        combined,
        methods,
        int(quality["bootstrap_samples"]),
        int(quality["bootstrap_seed"]) + version * 1000003,
    )
    result.update(
        {
            "edge": version,
            "update_date": update_date,
            "evaluation_date": eval_date,
            "eligible_same_user_population": eligible,
            "selected_user_ids_sha256": _hash_int_array(np.asarray(selected)),
            "parameter_group_distances": _parameter_distances(previous, current),
        }
    )
    return result


def run_evaluation(config_path: Path) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    validate_document(document)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != int(document["execution"]["evaluation_world_size"]):
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand evaluation world size differs")
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["evaluation_result"])
        if output.exists():
            result = json.loads(output.read_text())
            if result.get("status") != "complete_development_measurement":
                raise FileExistsError("KuaiRand evaluation result exists but is not complete")
            return result if runtime.is_primary else None
        torch.set_float32_matmul_precision("high")
        _seed_everything(int(document["training"]["seed"]) + runtime.rank)
        plan, data_metadata = load_plan(document)
        root = Path(document["outputs"]["checkpoint_root"])
        config_sha256 = file_sha256(config_path)
        for version in range(3):
            if not _valid_checkpoint(root, version, config_sha256, data_metadata):
                raise ValueError(f"KuaiRand theta{version} checkpoint differs")
        models = []
        for version in range(3):
            model = make_model(document, plan, runtime.device)
            _load_checkpoint(model, root, version)
            model.eval()
            models.append(model)
        popularity_ranks = _popularity_ranks(plan)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        edges = []
        update_indices = document["schedule"]["update_date_indices"]
        eval_indices = document["schedule"]["evaluation_date_indices"]
        for version, (update_index, eval_index) in enumerate(
            zip(update_indices, eval_indices, strict=True), start=1
        ):
            update_date = dates[int(update_index)]
            eval_date = dates[int(eval_index)]
            plan.ingest_day(update_date)
            edge = _evaluate_edge(
                document,
                plan,
                models[version - 1],
                models[version],
                version,
                update_date,
                eval_date,
                popularity_ranks,
                runtime,
            )
            if runtime.is_primary:
                edges.append(edge)
        if not runtime.is_primary:
            dist.barrier()
            return None
        implementation = all(edge["sanity"]["implementation_passed"] for edge in edges)
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scope": document["scope"],
            "scientific_result": False,
            "formal_result": False,
            "round_id": document["round_id"],
            "dataset": "KuaiRand-1K-standard",
            "config": {"path": str(config_path), "sha256": config_sha256},
            "programs": {
                "runner": {
                    "path": "src/hstu_kvcache/streaming/kuairand_root_cause.py",
                    "sha256": file_sha256(Path(__file__)),
                }
            },
            "data": data_metadata,
            "schedule": document["schedule"],
            "interventions": document["interventions"],
            "quality": document["quality"],
            "edges": edges,
            "sanity": {"implementation_passed": implementation},
            "execution": {
                "world_size": runtime.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "runtime_seconds": time.perf_counter() - started,
                "qualification_consumed": False,
                "final_consumed": False,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
            },
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)
