from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hstu_kvcache.data.kuairand import collate_batch
from hstu_kvcache.models import HSTU, DenseHSTUV2, DenseHSTUV2Config, HSTUConfig

from .kuairand_next_item_chain import _effective_document
from .kuairand_root_cause import (
    _atomic_json,
    _atomic_torch,
    file_sha256,
    load_plan,
)
from .trainer import build_next_item_targets

ROLLING_CHAIN_PROTOCOL = "evokv_kuairand_next_item_rolling_context_chain_v0"
EXPOSURE_CHAIN_PROTOCOL = "evokv_kuairand_next_item_exposure_objective_chain_v0"


def load_rolling_chain_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    model = document.get("model", {})
    training = document.get("training", {})
    candidates = document.get("candidates", [])
    rolling_expected = [
        ("legacy_raw", "legacy_hstu", False, 1.0),
        ("legacy_norm_t005", "legacy_hstu", True, 0.05),
        ("dense_raw", "dense_hstu_v2", False, 1.0),
        ("dense_norm_t005", "dense_hstu_v2", True, 0.05),
    ]
    exposure_expected = [
        ("legacy_exposure_t010", "legacy_hstu", True, 0.1),
        ("dense_exposure_t010", "dense_hstu_v2", True, 0.1),
    ]
    observed = [
        (
            value.get("candidate_id"),
            value.get("architecture"),
            value.get("normalize_scores"),
            value.get("temperature"),
        )
        for value in candidates
    ]
    protocol = document.get("protocol")
    expected = rolling_expected if protocol == ROLLING_CHAIN_PROTOCOL else exposure_expected
    if (
        protocol not in (ROLLING_CHAIN_PROTOCOL, EXPOSURE_CHAIN_PROTOCOL)
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("total_num_days") != 23
        or document.get("history_window_days") != 7
        or document.get("max_users") != 64
        or document.get("update_date_indices") != list(range(14, 22))
        or model.get("hidden_size") != 64
        or model.get("num_layers") != 2
        or model.get("num_heads") != 4
        or model.get("head_dim") != 16
        or model.get("max_seq_len") != 256
        or training.get("context_tokens") != 192
        or training.get("target_span") != 64
        or training.get("negative_count") != 32
        or training.get("base_epochs") != 1
        or training.get("update_epochs") != 3
        or observed != expected
        or (
            protocol == EXPOSURE_CHAIN_PROTOCOL
            and (
                any(value.get("objective") != "exposure_bce" for value in candidates)
                or any(value.get("positive_weight") != 4.0 for value in candidates)
                or any(value.get("in_catalog_targets_only") is not True for value in candidates)
            )
        )
        or file_sha256(document.get("source_config", {}).get("path", ""))
        != document.get("source_config", {}).get("sha256")
    ):
        raise ValueError("KuaiRand rolling-context chain config differs")
    return document


def effective_document(config: dict[str, Any]) -> dict[str, Any]:
    document = _effective_document(
        {
            **config,
            "source": {"config": config["source_config"]},
            "evaluation_methods": [
                "fresh_full_a",
                "fresh_full_b",
                "stale_previous",
                "no_prefix",
            ],
            "record_limit_per_rank": 1,
            "cap_user_limit_to_eligible": True,
        }
    )
    document["data"]["history_window_days"] = int(config["history_window_days"])
    document["data"]["max_users"] = int(config["max_users"])
    document["data"]["max_seq_len"] = int(config["model"]["max_seq_len"])
    return document


def make_rolling_model(
    config: dict[str, Any],
    candidate: dict[str, Any],
    plan,
    device: torch.device,
) -> HSTU | DenseHSTUV2:
    model = config["model"]
    common = {
        "num_items": plan.num_items,
        "num_prediction_items": plan.num_prediction_items,
        "num_behaviors": plan.num_behaviors,
        "hidden_size": int(model["hidden_size"]),
        "num_layers": int(model["num_layers"]),
        "num_heads": int(model["num_heads"]),
        "head_dim": int(model["head_dim"]),
        "max_seq_len": int(model["max_seq_len"]),
        "temporal_num_freqs": int(model["temporal_num_freqs"]),
        "input_dropout": float(model["input_dropout"]),
    }
    if candidate["architecture"] == "dense_hstu_v2":
        return DenseHSTUV2(DenseHSTUV2Config(**common)).to(device)
    return HSTU(HSTUConfig(**common, activation=str(model["legacy_activation"]))).to(device)


def rolling_sequences(
    histories: list[dict[str, np.ndarray]],
    context_tokens: int,
    target_span: int,
    target_start_by_history: list[int],
) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for history, first_target in zip(
        histories,
        target_start_by_history,
        strict=True,
    ):
        length = len(history["item_ids"])
        if first_target < 1 or first_target >= length:
            continue
        for target_start in range(first_target, length, target_span):
            target_stop = min(length, target_start + target_span)
            sequence_start = max(0, target_start - context_tokens)
            train_mask = np.zeros(target_stop - sequence_start, dtype=np.bool_)
            train_mask[target_start - sequence_start :] = True
            sequence = {
                name: values[sequence_start:target_stop]
                for name, values in history.items()
                if isinstance(values, np.ndarray)
            }
            sequence["train_mask"] = train_mask
            sequences.append(sequence)
    return sequences


def base_rolling_sequences(plan, context_tokens: int, target_span: int) -> list[dict[str, Any]]:
    frames = [plan.daily_segments[date] for date in plan.base_dates]
    frame = __import__("pandas").concat(frames, ignore_index=True)
    histories = [
        plan._frame_sequence(int(user), user_frame)
        for user, user_frame in frame.groupby("user_idx", sort=False)
    ]
    return rolling_sequences(
        histories,
        context_tokens,
        target_span,
        [1] * len(histories),
    )


def update_rolling_sequences(
    plan,
    date: str,
    context_tokens: int,
    target_span: int,
) -> list[dict[str, Any]]:
    day = plan.daily_segments[date]
    histories = []
    starts = []
    for user, user_day in day.groupby("user_idx", sort=False):
        history = plan.user_histories.get(int(user))
        if history is None:
            continue
        timestamps = history["timestamps"]
        first = int(np.searchsorted(timestamps, int(user_day["time_ms"].min())))
        histories.append(history)
        starts.append(first)
    return rolling_sequences(histories, context_tokens, target_span, starts)


def iter_rolling_batches(
    sequences: list[dict[str, Any]],
    batch_size: int,
    max_seq_len: int,
    seed: int,
):
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(sequences))
    ordered = [sequences[int(index)] for index in order]
    ordered.sort(key=lambda value: len(value["item_ids"]))
    groups = [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]
    group_order = generator.permutation(len(groups))
    for index in group_order:
        yield collate_batch(
            groups[int(index)],
            max_seq_len=max_seq_len,
            pad_to=None,
        )


def _score(
    model: HSTU | DenseHSTUV2,
    hidden: torch.Tensor,
    candidates: torch.Tensor,
    normalize: bool,
    temperature: float,
) -> torch.Tensor:
    vectors = model.item_emb.weight[candidates]
    if normalize:
        hidden = F.normalize(hidden, dim=-1)
        vectors = F.normalize(vectors, dim=-1)
    return torch.einsum("blh,blch->blc", hidden, vectors) / temperature


def train_rolling_epoch(
    model: HSTU | DenseHSTUV2,
    optimizer: torch.optim.Optimizer,
    batches,
    candidate: dict[str, Any],
    negative_count: int,
    device: torch.device,
    phase: str,
) -> dict[str, Any]:
    model.train()
    loss_sum = 0.0
    targets_total = 0
    sequence_total = 0
    token_total = 0
    batches_total = 0
    started = time.perf_counter()
    for batch in batches:
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
            lengths=lengths,
        )
        if candidate.get("objective") == "exposure_bce":
            targets = item_ids[:, 1:]
            positions = torch.arange(targets.shape[1], device=device)
            valid = positions.unsqueeze(0) < (lengths - 1).clamp_min(0).unsqueeze(1)
            valid = valid & (item_ids[:, :-1] > 0) & (targets > 0) & train_mask[:, 1:].bool()
        else:
            targets, valid = build_next_item_targets(item_ids, lengths, labels, train_mask)
        if candidate.get("in_catalog_targets_only"):
            valid = valid & (targets <= int(model.cfg.num_prediction_items))
        count = int(valid.sum().item())
        if count < 1:
            continue
        if candidate.get("objective") == "exposure_bce":
            logits = _score(
                model,
                hidden[:, :-1],
                targets.unsqueeze(-1),
                bool(candidate["normalize_scores"]),
                float(candidate["temperature"]),
            ).squeeze(-1)
            losses = F.binary_cross_entropy_with_logits(
                logits,
                labels[:, 1:].to(logits.dtype),
                pos_weight=torch.tensor(
                    float(candidate["positive_weight"]),
                    dtype=logits.dtype,
                    device=device,
                ),
                reduction="none",
            )
        else:
            positives = targets.unsqueeze(-1)
            negatives = torch.randint(
                1,
                int(model.cfg.num_prediction_items) + 1,
                (*targets.shape, negative_count),
                device=device,
            )
            negatives = torch.where(
                negatives == positives,
                negatives.remainder(int(model.cfg.num_prediction_items)) + 1,
                negatives,
            )
            candidates = torch.cat((positives, negatives), dim=-1)
            logits = _score(
                model,
                hidden[:, :-1],
                candidates,
                bool(candidate["normalize_scores"]),
                float(candidate["temperature"]),
            )
            losses = F.cross_entropy(
                logits.flatten(0, 1),
                torch.zeros_like(targets).flatten(),
                reduction="none",
            ).view_as(targets)
        loss = (losses * valid).sum() / valid.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{phase} produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.detach().item()) * count
        targets_total += count
        sequence_total += len(lengths)
        token_total += int(lengths.sum().item())
        batches_total += 1
        if batches_total % 100 == 0:
            print(
                f"phase={phase} batches={batches_total} targets={targets_total} "
                f"loss={loss_sum / targets_total:.6f}",
                flush=True,
            )
    if targets_total < 1:
        raise RuntimeError(f"{phase} has no training targets")
    return {
        "phase": phase,
        "batches": batches_total,
        "sequences": sequence_total,
        "tokens": token_total,
        "eligible_targets": targets_total,
        "target_weighted_sampled_cross_entropy": loss_sum / targets_total,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _checkpoint_paths(root: Path, version: int) -> tuple[Path, Path]:
    directory = root / f"theta_{version}"
    return directory / "model.pt", directory / "manifest.json"


def _save(
    model: HSTU | DenseHSTUV2,
    root: Path,
    version: int,
    config_path: Path,
    candidate: dict[str, Any],
    metadata: dict[str, Any],
    training: dict[str, Any],
    protocol: str,
) -> dict[str, Any]:
    model_path, manifest_path = _checkpoint_paths(root, version)
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError(f"rolling theta{version} already exists")
    _atomic_torch(model_path, model.state_dict())
    manifest = {
        "protocol": protocol,
        "status": "complete_development_checkpoint",
        "scientific_result": False,
        "formal_result": False,
        "version": version,
        "candidate": candidate,
        "config_sha256": file_sha256(config_path),
        "program_sha256": file_sha256(Path(__file__)),
        "model_sha256": file_sha256(model_path),
        "model": asdict(model.cfg),
        "data": {
            "num_users": metadata["num_users"],
            "num_context_items": metadata["num_context_items"],
            "num_prediction_items": metadata["num_prediction_items"],
            "prediction_video_ids_sha256": metadata["prediction_video_ids_sha256"],
        },
        "training": training,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


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
    document = effective_document(config)
    plan, metadata = load_plan(document)
    plan.init_base()
    model = make_rolling_model(config, candidate, plan, device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["base_lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    resume_path = root / "resume.pt"
    completed = -1
    records = []
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume.get("protocol") != config["protocol"]
            or resume.get("candidate_id") != candidate["candidate_id"]
            or resume.get("config_sha256") != file_sha256(config_path)
        ):
            raise ValueError("KuaiRand rolling-context resume differs")
        completed = int(resume["version"])
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
        for version in range(completed + 1):
            records.append(json.loads(_checkpoint_paths(root, version)[1].read_text())["training"])
    started = time.perf_counter()
    if completed < 0:
        sequences = base_rolling_sequences(
            plan,
            int(training["context_tokens"]),
            int(training["target_span"]),
        )
        epochs = []
        for epoch in range(int(training["base_epochs"])):
            epoch_seed = int(training["seed"]) + epoch
            _seed(epoch_seed)
            epochs.append(
                train_rolling_epoch(
                    model,
                    optimizer,
                    iter_rolling_batches(
                        sequences,
                        int(training["batch_size"]),
                        int(config["model"]["max_seq_len"]),
                        epoch_seed,
                    ),
                    candidate,
                    int(training["negative_count"]),
                    device,
                    f"rolling_{candidate['candidate_id']}_theta0_e{epoch + 1}",
                )
            )
        record = {
            "version": 0,
            "role": "base_rolling_context",
            "dates": plan.base_dates,
            "window_count": len(sequences),
            "epochs": epochs,
        }
        _save(model, root, 0, config_path, candidate, metadata, record, config["protocol"])
        _atomic_torch(
            resume_path,
            {
                "protocol": config["protocol"],
                "candidate_id": candidate["candidate_id"],
                "config_sha256": file_sha256(config_path),
                "version": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
        )
        records.append(record)
        completed = 0
        del sequences
    dates = plan.base_dates + plan.stream_dates
    for date_index in config["update_date_indices"][:completed]:
        plan.ingest_day(dates[int(date_index)])
    for group in optimizer.param_groups:
        group["lr"] = float(candidate["stream_lr"])
    for version, date_index in enumerate(config["update_date_indices"], start=1):
        if version <= completed:
            continue
        date = dates[int(date_index)]
        plan.ingest_day(date)
        sequences = update_rolling_sequences(
            plan,
            date,
            int(training["context_tokens"]),
            int(training["target_span"]),
        )
        epochs = []
        for epoch in range(int(training["update_epochs"])):
            epoch_seed = int(training["seed"]) + version * 1009 + epoch
            _seed(epoch_seed)
            epochs.append(
                train_rolling_epoch(
                    model,
                    optimizer,
                    iter_rolling_batches(
                        sequences,
                        int(training["batch_size"]),
                        int(config["model"]["max_seq_len"]),
                        epoch_seed,
                    ),
                    candidate,
                    int(training["negative_count"]),
                    device,
                    f"rolling_{candidate['candidate_id']}_theta{version}_e{epoch + 1}",
                )
            )
        record = {
            "version": version,
            "role": "stream_update_rolling_context",
            "date": date,
            "window_count": len(sequences),
            "epochs": epochs,
        }
        _save(
            model,
            root,
            version,
            config_path,
            candidate,
            metadata,
            record,
            config["protocol"],
        )
        _atomic_torch(
            resume_path,
            {
                "protocol": config["protocol"],
                "candidate_id": candidate["candidate_id"],
                "config_sha256": file_sha256(config_path),
                "version": version,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
        )
        records.append(record)
        completed = version
        del sequences
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "protocol": config["protocol"],
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "candidate": candidate,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "model": asdict(model.cfg),
        "training": training,
        "versions": records,
        "completed_version": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    del optimizer, model, plan
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_rolling_chain(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_rolling_chain_config(config_path)
    output = Path(config["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand rolling-context training is single-rank")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    results = [
        _train_candidate(config, config_path, candidate, device)
        for candidate in config["candidates"]
    ]
    result = {
        "protocol": config["protocol"],
        "status": "complete_development_training_sweep",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "candidates": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output, result)
    return result


def validate_rolling_chain_result(result: dict[str, Any]) -> None:
    candidates = result.get("candidates", [])
    if (
        result.get("protocol") != ROLLING_CHAIN_PROTOCOL
        or result.get("status") != "complete_development_training_sweep"
        or result.get("scientific_result") is not False
        or len(candidates) != 4
        or any(value.get("completed_version") != 8 for value in candidates)
        or any(len(value.get("versions", [])) != 9 for value in candidates)
    ):
        raise ValueError("KuaiRand rolling-context training result differs")


def validate_exposure_chain_result(result: dict[str, Any]) -> None:
    candidates = result.get("candidates", [])
    if (
        result.get("protocol") != EXPOSURE_CHAIN_PROTOCOL
        or result.get("status") != "complete_development_training_sweep"
        or result.get("scientific_result") is not False
        or len(candidates) != 2
        or any(value.get("completed_version") != 8 for value in candidates)
        or any(len(value.get("versions", [])) != 9 for value in candidates)
    ):
        raise ValueError("KuaiRand exposure-objective training result differs")
