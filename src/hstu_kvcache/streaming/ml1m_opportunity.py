from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hstu_kvcache.data.movielens import load_movielens_hard
from hstu_kvcache.models import HSTU, DenseHSTUV2, DenseHSTUV2Config, HSTUConfig
from hstu_kvcache.models.kv_cache import HSTUKVCache

PROTOCOL = "evokv_ml1m_opportunity_factor_screen_v0"
METRICS = (
    "candidate_cross_entropy",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "hit_rate_at_10",
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


def validate_config(document: dict[str, Any]) -> None:
    if document.get("protocol") != PROTOCOL:
        raise ValueError("ML1m opportunity protocol differs")
    if document.get("status") != "ready_for_autonomous_execution":
        raise ValueError("ML1m opportunity status differs")
    if document.get("scientific_result") is not False:
        raise ValueError("ML1m opportunity result scope differs")
    data = document.get("data")
    model = document.get("model")
    training = document.get("training")
    evaluation = document.get("evaluation")
    execution = document.get("execution")
    variants = document.get("variants")
    if not all(isinstance(value, dict) for value in (data, model, training, evaluation, execution)):
        raise ValueError("ML1m opportunity sections differ")
    if not isinstance(variants, list) or [value.get("id") for value in variants] != [
        "a0_legacy_raw_full",
        "a1_legacy_normalized_full",
        "a2_dense_normalized_full",
        "a3_dense_normalized_blocks",
    ]:
        raise ValueError("ML1m opportunity variants differ")
    if int(data.get("user_limit", 0)) < 16 or int(data.get("selection_seed", -1)) < 0:
        raise ValueError("ML1m opportunity user selection differs")
    if not Path(data.get("path", "")).is_dir():
        raise ValueError("ML1m opportunity data path is unavailable")
    if int(model.get("max_seq_len", 0)) < 8:
        raise ValueError("ML1m opportunity max sequence differs")
    if int(training.get("negative_samples", 0)) < 1:
        raise ValueError("ML1m opportunity negatives differ")
    if int(training.get("base_epochs", 0)) < 1 or int(training.get("update_epochs", 0)) < 1:
        raise ValueError("ML1m opportunity epochs differ")
    if int(evaluation.get("bootstrap_samples", 0)) < 100:
        raise ValueError("ML1m opportunity bootstrap differs")
    if execution.get("cuda_visible_devices") != "0":
        raise ValueError("ML1m opportunity GPU binding differs")


def load_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    validate_config(document)
    return document


def _user_order(user_ids: list[str], seed: int) -> list[str]:
    return sorted(
        user_ids,
        key=lambda user_id: hashlib.sha256(f"{seed}:{user_id}".encode()).digest(),
    )


def load_causal_records(document: dict[str, Any]) -> dict[str, Any]:
    data = document["data"]
    split_records = {
        split: {record["user_id"]: record for record in load_movielens_hard(data["path"], split)}
        for split in ("train", "dev", "test")
    }
    common = set(split_records["train"])
    if common != set(split_records["dev"]) or common != set(split_records["test"]):
        raise ValueError("ML1m split user sets differ")
    chronology_failures = []
    candidate_failures = []
    for user_id in common:
        train = split_records["train"][user_id]
        dev = split_records["dev"][user_id]
        test = split_records["test"][user_id]
        expected_dev = np.concatenate((train["history"], train["positive_items"]))
        expected_test = np.concatenate((dev["history"], dev["positive_items"]))
        if not np.array_equal(dev["history"], expected_dev) or not np.array_equal(
            test["history"], expected_test
        ):
            chronology_failures.append(user_id)
        for record in (train, dev, test):
            if len(record["positive_items"]) != 1 or int(record["labels"].sum()) != 1:
                candidate_failures.append(user_id)
    if chronology_failures or candidate_failures:
        raise ValueError("ML1m causal chronology or candidate labels differ")
    selected = _user_order(sorted(common), int(data["selection_seed"]))[: int(data["user_limit"])]
    selected_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    return {
        "splits": split_records,
        "selected_users": selected,
        "selected_users_sha256": selected_hash,
        "available_users": len(common),
    }


def _model_config(document: dict[str, Any], architecture: str):
    cfg = document["model"]
    common = {
        "num_items": int(cfg["num_items"]),
        "num_prediction_items": int(cfg["num_items"]),
        "num_behaviors": 1,
        "hidden_size": int(cfg["hidden_size"]),
        "num_layers": int(cfg["num_layers"]),
        "num_heads": int(cfg["num_heads"]),
        "head_dim": int(cfg["hidden_size"]) // int(cfg["num_heads"]),
        "max_seq_len": int(cfg["max_seq_len"]),
        "input_dropout": float(cfg["input_dropout"]),
    }
    if architecture == "dense_hstu_v2":
        return DenseHSTUV2Config(
            **common,
            output_dropout=float(cfg["output_dropout"]),
        )
    if architecture != "legacy":
        raise ValueError("unknown architecture")
    return HSTUConfig(
        **common,
        gating=str(cfg["legacy_gating"]),
        qk_scale=float(cfg["legacy_qk_scale"]),
        activation=str(cfg["legacy_activation"]),
        attn_dropout=float(cfg["output_dropout"]),
    )


def make_model(document: dict[str, Any], architecture: str, device: torch.device):
    cfg = _model_config(document, architecture)
    model = DenseHSTUV2(cfg) if architecture == "dense_hstu_v2" else HSTU(cfg)
    return model.to(device)


def _base_sequences(records: dict[str, Any], selected: list[str], max_seq_len: int):
    sequences = []
    for user_id in selected:
        record = records[user_id]
        full = np.concatenate((record["history"], record["positive_items"]))
        full = full[-(max_seq_len + 1) :]
        sequences.append((user_id, full[:-1], full[1:]))
    return sequences


def _update_sequences(records: dict[str, Any], selected: list[str], max_seq_len: int):
    return [
        (
            user_id,
            records[user_id]["history"][-max_seq_len:],
            int(records[user_id]["positive_items"][0]),
        )
        for user_id in selected
    ]


def _collate_base(batch, device: torch.device):
    lengths = torch.tensor([len(value[1]) for value in batch], dtype=torch.long, device=device)
    width = int(lengths.max().item())
    items = torch.zeros(len(batch), width, dtype=torch.long, device=device)
    targets = torch.zeros_like(items)
    for row, (_, sequence, sequence_targets) in enumerate(batch):
        length = len(sequence)
        items[row, :length] = torch.as_tensor(sequence, dtype=torch.long, device=device)
        targets[row, :length] = torch.as_tensor(
            sequence_targets,
            dtype=torch.long,
            device=device,
        )
    behaviors = (items != 0).long()
    deltas = torch.zeros_like(items, dtype=torch.float32)
    return items, behaviors, deltas, targets, lengths


def _collate_update(batch, device: torch.device):
    lengths = torch.tensor([len(value[1]) for value in batch], dtype=torch.long, device=device)
    width = int(lengths.max().item())
    items = torch.zeros(len(batch), width, dtype=torch.long, device=device)
    targets = torch.tensor([value[2] for value in batch], dtype=torch.long, device=device)
    for row, (_, sequence, _) in enumerate(batch):
        items[row, : len(sequence)] = torch.as_tensor(sequence, dtype=torch.long, device=device)
    behaviors = (items != 0).long()
    deltas = torch.zeros_like(items, dtype=torch.float32)
    return items, behaviors, deltas, targets, lengths


def _sample_candidates(
    targets: torch.Tensor,
    num_items: int,
    negative_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    negatives = torch.randint(
        1,
        num_items,
        (targets.numel(), negative_samples),
        device=targets.device,
        generator=generator,
    )
    negatives = negatives + (negatives >= targets.unsqueeze(1))
    return torch.cat((targets.unsqueeze(1), negatives), dim=1)


def _score_vectors(
    model,
    hidden: torch.Tensor,
    candidates: torch.Tensor,
    normalized: bool,
    temperature: float,
) -> torch.Tensor:
    candidate_vectors = model.item_emb.weight[candidates]
    if normalized:
        hidden = F.normalize(hidden, dim=-1)
        candidate_vectors = F.normalize(candidate_vectors, dim=-1)
    return torch.einsum("nh,nch->nc", hidden, candidate_vectors) / temperature


def _train(
    model,
    examples,
    document: dict[str, Any],
    normalized: bool,
    phase: str,
    update_scope: str,
    seed: int,
) -> dict[str, Any]:
    training = document["training"]
    if update_scope == "blocks":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.blocks.parameters():
            parameter.requires_grad_(True)
    elif update_scope != "full":
        raise ValueError("unknown update scope")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    epochs = int(training["base_epochs"] if phase == "base" else training["update_epochs"])
    learning_rate = float(training["base_lr"] if phase == "base" else training["update_lr"])
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(seed + 9173)
    model.train()
    epoch_results = []
    started = time.monotonic()
    for epoch in range(epochs):
        order = rng.permutation(len(examples))
        loss_sum = 0.0
        target_count = 0
        for start in range(0, len(order), batch_size):
            batch = [examples[int(index)] for index in order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            if phase == "base":
                items, behaviors, deltas, targets, lengths = _collate_base(batch, device)
                hidden, _ = model(items, behaviors, deltas, lengths=lengths)
                valid = torch.arange(items.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
                selected_hidden = hidden[valid]
                selected_targets = targets[valid]
            else:
                items, behaviors, deltas, selected_targets, lengths = _collate_update(batch, device)
                hidden, _ = model(items, behaviors, deltas, lengths=lengths)
                selected_hidden = model.last_hidden(hidden, lengths)
            candidates = _sample_candidates(
                selected_targets,
                int(document["model"]["num_items"]),
                int(training["negative_samples"]),
                torch_generator,
            )
            scores = _score_vectors(
                model,
                selected_hidden,
                candidates,
                normalized,
                float(training["temperature"] if normalized else 1.0),
            )
            loss = F.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise RuntimeError("ML1m training produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip_norm"]))
            optimizer.step()
            count = scores.shape[0]
            loss_sum += float(loss.detach().item()) * count
            target_count += count
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "mean_sampled_cross_entropy": loss_sum / target_count,
                "targets": target_count,
            }
        )
        print(
            f"phase=ml1m_{phase} scope={update_scope} epoch={epoch + 1}/{epochs} "
            f"loss={loss_sum / target_count:.6f} targets={target_count}",
            flush=True,
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {
        "phase": phase,
        "scope": update_scope,
        "epochs": epoch_results,
        "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }


def _state_delta(previous: dict[str, torch.Tensor], current: dict[str, torch.Tensor]):
    groups: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for name, old in previous.items():
        new = current[name]
        if name.startswith("item_emb"):
            group = "item_embedding"
        elif name.startswith("blocks"):
            group = "blocks"
        elif name.startswith("final_norm"):
            group = "final_norm"
        else:
            group = "input_path"
        groups[group][0] += float((new.double() - old.double()).pow(2).sum().item())
        groups[group][1] += float(old.double().pow(2).sum().item())
    return {
        group: {
            "absolute_l2": math.sqrt(values[0]),
            "relative_l2": math.sqrt(values[0]) / max(math.sqrt(values[1]), 1e-12),
        }
        for group, values in sorted(groups.items())
    }


def _empty_cache(reference: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=reference.k[:, :, :0],
        v=reference.v[:, :, :0],
        seq_len=0,
    )


def _candidate_metrics(scores: torch.Tensor, target_index: torch.Tensor):
    ce = F.cross_entropy(scores, target_index, reduction="none")
    rows = torch.arange(scores.shape[0], device=scores.device)
    positive = scores[rows, target_index]
    ranks = 1 + (scores > positive.unsqueeze(1)).sum(dim=1)
    ranks_float = ranks.float()
    return {
        "candidate_cross_entropy": ce,
        "mrr": ranks_float.reciprocal(),
        "ndcg_at_5": torch.where(
            ranks <= 5,
            torch.log2(ranks_float + 1).reciprocal(),
            torch.zeros_like(ranks_float),
        ),
        "ndcg_at_10": torch.where(
            ranks <= 10,
            torch.log2(ranks_float + 1).reciprocal(),
            torch.zeros_like(ranks_float),
        ),
        "hit_rate_at_1": (ranks <= 1).float(),
        "hit_rate_at_5": (ranks <= 5).float(),
        "hit_rate_at_10": (ranks <= 10).float(),
    }


def _candidate_ids(
    record: dict[str, Any],
    user_id: str,
    num_items: int,
    candidate_count: int | None,
    seed: int,
    filter_seen: bool = False,
) -> np.ndarray:
    positive = int(record["positive_items"][0])
    if candidate_count == num_items:
        return np.arange(1, num_items + 1, dtype=np.int64)
    count = len(record["candidates"]) if candidate_count is None else candidate_count
    if count < 2 or count > num_items:
        raise ValueError("candidate count differs")
    candidates = [positive]
    candidates.extend(int(value) for value in record["candidates"] if int(value) != positive)
    if len(candidates) < count:
        digest = hashlib.sha256(f"{seed}:{user_id}:{count}".encode()).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        excluded = set(candidates)
        if filter_seen:
            excluded.update(int(value) for value in record["history"])
        available = np.asarray(
            [item_id for item_id in range(1, num_items + 1) if item_id not in excluded],
            dtype=np.int64,
        )
        candidates.extend(int(value) for value in generator.choice(
            available,
            size=count - len(candidates),
            replace=False,
        ))
    output = np.asarray(candidates[:count], dtype=np.int64)
    if len(np.unique(output)) != count or int(output[0]) != positive:
        raise ValueError("candidate pool construction differs")
    return output


def _relative_rows(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((value - reference).double(), dim=tuple(range(2, value.ndim)))
    denominator = torch.linalg.vector_norm(reference.double(), dim=tuple(range(2, value.ndim)))
    return numerator / denominator.clamp_min(1e-12)


@torch.no_grad()
def _evaluate(
    previous,
    current,
    records: dict[str, Any],
    selected: list[str],
    document: dict[str, Any],
    normalized: bool,
    candidate_count: int | None = None,
    candidate_seed: int = 0,
    filter_seen: bool = False,
    candidate_map: dict[str, np.ndarray] | None = None,
    candidate_protocol_override: str | None = None,
    prediction_query: bool = False,
) -> dict[str, Any]:
    max_seq_len = int(document["model"]["max_seq_len"])
    groups: dict[int, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for user_id in selected:
        record = records[user_id]
        history = record["history"][-(max_seq_len - 1) :] if prediction_query else record[
            "history"
        ][-max_seq_len:]
        prefix = history if prediction_query else history[:-1]
        suffix = np.asarray([0], dtype=np.int64) if prediction_query else history[-1:]
        groups[len(history)].append(
            (
                user_id,
                prefix,
                suffix,
                (
                    candidate_map[user_id]
                    if candidate_map is not None
                    else _candidate_ids(
                        record,
                        user_id,
                        int(document["model"]["num_items"]),
                        candidate_count,
                        candidate_seed,
                        filter_seen,
                    )
                ),
            )
        )
    if candidate_map is not None:
        candidate_widths = {len(value) for value in candidate_map.values()}
        if len(candidate_widths) != 1 or set(candidate_map) != set(selected):
            raise ValueError("candidate map coverage differs")
        candidate_count = candidate_widths.pop()
    device = next(current.parameters()).device
    batch_size = int(document["evaluation"]["batch_size"])
    temperature = float(document["training"]["temperature"] if normalized else 1.0)
    details = []
    maximum_fresh_hidden_error = 0.0
    maximum_fresh_score_error = 0.0
    started = time.monotonic()
    previous.eval()
    current.eval()
    for history_length, values in sorted(groups.items()):
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            prefix = torch.as_tensor(
                np.stack([value[1] for value in batch]),
                dtype=torch.long,
                device=device,
            )
            suffix = torch.as_tensor(
                np.stack([value[2] for value in batch]),
                dtype=torch.long,
                device=device,
            )
            candidates = torch.as_tensor(
                np.stack([value[3] for value in batch]),
                dtype=torch.long,
                device=device,
            )
            prefix_behaviors = torch.ones_like(prefix)
            suffix_behaviors = torch.ones_like(suffix)
            prefix_deltas = torch.zeros_like(prefix, dtype=torch.float32)
            suffix_deltas = torch.zeros_like(suffix, dtype=torch.float32)
            previous_cache = previous.compute_kv(prefix, prefix_behaviors, prefix_deltas)
            current_cache = current.compute_kv(prefix, prefix_behaviors, prefix_deltas)
            reuse_hidden, _ = current.forward_with_cache(
                previous_cache,
                suffix,
                suffix_behaviors,
                suffix_deltas,
            )
            fresh_incremental, _ = current.forward_with_cache(
                current_cache,
                suffix,
                suffix_behaviors,
                suffix_deltas,
            )
            previous_incremental, _ = previous.forward_with_cache(
                previous_cache,
                suffix,
                suffix_behaviors,
                suffix_deltas,
            )
            no_prefix_hidden, _ = current.forward_with_cache(
                _empty_cache(current_cache),
                suffix,
                suffix_behaviors,
                suffix_deltas,
            )
            full_items = torch.cat((prefix, suffix), dim=1)
            full_behaviors = torch.ones_like(full_items)
            full_deltas = torch.zeros_like(full_items, dtype=torch.float32)
            fresh_full, _ = current(full_items, full_behaviors, full_deltas)
            previous_full, _ = previous(full_items, full_behaviors, full_deltas)
            recompute_vector = fresh_full[:, -1]
            previous_vector = previous_full[:, -1]
            method_vectors = {
                "previous_fresh": previous_vector,
                "recompute": recompute_vector,
                "reuse": reuse_hidden[:, -1],
                "no_prefix": no_prefix_hidden[:, -1],
            }
            method_scores = {
                method: _score_vectors(
                    current if method != "previous_fresh" else previous,
                    vector,
                    candidates,
                    normalized,
                    temperature,
                )
                for method, vector in method_vectors.items()
            }
            duplicate_score = _score_vectors(
                current,
                fresh_incremental[:, -1],
                candidates,
                normalized,
                temperature,
            )
            maximum_fresh_score_error = max(
                maximum_fresh_score_error,
                float((duplicate_score - method_scores["recompute"]).abs().max().item()),
            )
            full_catalog_mode = (
                candidate_map is None
                and candidate_count == int(document["model"]["num_items"])
            )
            if filter_seen and full_catalog_mode:
                seen = torch.zeros_like(candidates, dtype=torch.bool)
                for row, value in enumerate(batch):
                    history = records[value[0]]["history"]
                    seen[row, torch.as_tensor(history, device=device) - 1] = True
                    positive = int(records[value[0]]["positive_items"][0])
                    seen[row, positive - 1] = False
                method_scores = {
                    method: scores.masked_fill(seen, -torch.inf)
                    for method, scores in method_scores.items()
                }
            maximum_fresh_hidden_error = max(
                maximum_fresh_hidden_error,
                float((fresh_incremental[:, -1] - recompute_vector).abs().max().item()),
                float((previous_incremental[:, -1] - previous_vector).abs().max().item()),
            )
            if full_catalog_mode:
                target_index = torch.tensor(
                    [int(records[value[0]]["positive_items"][0]) - 1 for value in batch],
                    dtype=torch.long,
                    device=device,
                )
            else:
                target_index = torch.zeros(len(batch), dtype=torch.long, device=device)
            metric_values = {
                method: _candidate_metrics(scores, target_index)
                for method, scores in method_scores.items()
            }
            cache_k_error = _relative_rows(previous_cache.k, current_cache.k).mean(dim=0)
            cache_v_error = _relative_rows(previous_cache.v, current_cache.v).mean(dim=0)
            hidden_error = torch.linalg.vector_norm(
                (reuse_hidden[:, -1] - recompute_vector).double(),
                dim=1,
            ) / torch.linalg.vector_norm(recompute_vector.double(), dim=1).clamp_min(1e-12)
            for row, value in enumerate(batch):
                details.append(
                    {
                        "user_id": value[0],
                        "history_length": history_length + int(prediction_query),
                        "prefix_length": history_length if prediction_query else history_length - 1,
                        "metrics": {
                            method: {
                                metric: float(metric_values[method][metric][row].item())
                                for metric in METRICS
                            }
                            for method in METHODS
                        },
                        "cache_k_relative_error": float(cache_k_error[row].item()),
                        "cache_v_relative_error": float(cache_v_error[row].item()),
                        "hidden_relative_error": float(hidden_error[row].item()),
                    }
                )
    if candidate_protocol_override is not None:
        candidate_protocol = candidate_protocol_override
    elif candidate_count is None:
        candidate_protocol = "frozen_pilot20"
    elif candidate_count == int(document["model"]["num_items"]):
        candidate_protocol = "full_catalog_seen_filtered" if filter_seen else "full_catalog"
    elif filter_seen:
        candidate_protocol = "pilot20_hard_plus_deterministic_unseen_base_catalog"
    else:
        candidate_protocol = "pilot20_hard_plus_deterministic_base_catalog"
    return {
        "records": details,
        "elapsed_seconds": time.monotonic() - started,
        "candidate_count": (
            len(records[selected[0]]["candidates"])
            if candidate_count is None
            else candidate_count
        ),
        "candidate_protocol": candidate_protocol,
        "prediction_protocol": "learned_query_after_history" if prediction_query else "last_item_hidden",
        "sanity": {
            "maximum_same_model_incremental_hidden_absolute_error": maximum_fresh_hidden_error,
            "maximum_same_model_incremental_score_absolute_error": maximum_fresh_score_error,
            "passed": maximum_fresh_hidden_error <= 1e-4 and maximum_fresh_score_error <= 1e-4,
        },
    }


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        selected = generator.integers(0, len(values), size=(count, len(values)))
        output[start : start + count] = values[selected].mean(axis=1)
    return [float(value) for value in np.quantile(output, [0.025, 0.975])]


def _comparison(
    records: list[dict[str, Any]],
    better: str,
    worse: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for index, metric in enumerate(METRICS):
        better_values = np.asarray(
            [value["metrics"][better][metric] for value in records],
            dtype=np.float64,
        )
        worse_values = np.asarray(
            [value["metrics"][worse][metric] for value in records],
            dtype=np.float64,
        )
        advantage = worse_values - better_values if metric == "candidate_cross_entropy" else better_values - worse_values
        interval = _bootstrap(advantage, samples, seed + index * 1000003)
        baseline = float(worse_values.mean())
        result[metric] = {
            "direction": f"{better}_advantage_over_{worse}",
            "absolute": float(advantage.mean()),
            "relative_percent": 100.0 * float(advantage.mean()) / baseline if baseline else None,
            "user_bootstrap_95": {
                "lower": interval[0],
                "upper": interval[1],
                "samples": samples,
                "seed": seed + index * 1000003,
            },
            "positive_direction_with_ci": bool(interval[0] > 0),
            "positive_user_fraction": float((advantage > 0).mean()),
        }
    return result


def summarize_records(records: list[dict[str, Any]], document: dict[str, Any]) -> dict[str, Any]:
    if not records:
        raise ValueError("ML1m opportunity summary has no records")
    endpoints = {
        method: {
            metric: float(np.mean([value["metrics"][method][metric] for value in records]))
            for metric in METRICS
        }
        for method in METHODS
    }
    samples = int(document["evaluation"]["bootstrap_samples"])
    seed = int(document["evaluation"]["bootstrap_seed"])
    comparisons = {
        "fresh_update_value": _comparison(records, "recompute", "previous_fresh", samples, seed),
        "recompute_over_reuse": _comparison(records, "recompute", "reuse", samples, seed + 101),
        "history_value": _comparison(records, "recompute", "no_prefix", samples, seed + 202),
    }
    stale = comparisons["recompute_over_reuse"]
    ranking_metrics = ("mrr", "ndcg_at_5", "ndcg_at_10", "hit_rate_at_5", "hit_rate_at_10")
    gate = {
        "fresh_update_ce_positive": comparisons["fresh_update_value"]["candidate_cross_entropy"][
            "absolute"
        ]
        > 0,
        "history_ce_positive": comparisons["history_value"]["candidate_cross_entropy"]["absolute"]
        > 0,
        "stale_ce_positive_ci": stale["candidate_cross_entropy"]["positive_direction_with_ci"],
        "stale_ranking_positive_ci": any(stale[metric]["positive_direction_with_ci"] for metric in ranking_metrics),
    }
    return {
        "users": len(records),
        "endpoints": endpoints,
        "comparisons": comparisons,
        "representation_drift": {
            field: {
                "mean": float(np.mean([value[field] for value in records])),
                "median": float(np.median([value[field] for value in records])),
                "p95": float(np.quantile([value[field] for value in records], 0.95)),
            }
            for field in (
                "cache_k_relative_error",
                "cache_v_relative_error",
                "hidden_relative_error",
            )
        },
        "gate": gate,
    }


def summarize_evaluation(evaluation: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    result = summarize_records(evaluation["records"], document)
    result["gate"]["same_model_sanity"] = evaluation["sanity"]["passed"]
    result["gate"]["passed_screen"] = all(result["gate"].values())
    result.update(
        {
        "sanity": evaluation["sanity"],
            "candidate_count": evaluation["candidate_count"],
            "candidate_protocol": evaluation["candidate_protocol"],
            "prediction_protocol": evaluation["prediction_protocol"],
        }
    )
    return result


def _checkpoint_payload(model, architecture: str, metadata: dict[str, Any]):
    return {
        "architecture": architecture,
        "config": vars(model.cfg),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "metadata": metadata,
    }


def _save_checkpoint(path: Path, model, architecture: str, metadata: dict[str, Any]):
    _atomic_torch(path, _checkpoint_payload(model, architecture, metadata))
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def run_factor_screen(config_path: str | Path) -> dict[str, Any]:
    document = load_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        validate_summary(existing, document)
        return existing
    output_root.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(config_path)
    _atomic_json(
        output_root / "resolved_config.json",
        {"source_path": str(config_path), "source_sha256": source_hash, "config": document},
    )
    causal = load_causal_records(document)
    _atomic_json(
        output_root / "selected_users.json",
        {
            "available_users": causal["available_users"],
            "selected_users": causal["selected_users"],
            "selected_users_sha256": causal["selected_users_sha256"],
        },
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if document["execution"]["require_cuda"] and device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m opportunity screen")
    selected = causal["selected_users"]
    max_seq_len = int(document["model"]["max_seq_len"])
    base_examples = _base_sequences(causal["splits"]["train"], selected, max_seq_len)
    update_examples = _update_sequences(causal["splits"]["dev"], selected, max_seq_len)
    base_states: dict[str, dict[str, Any]] = {}
    variant_results = []
    started = time.monotonic()
    for variant_index, variant in enumerate(document["variants"]):
        variant_id = variant["id"]
        architecture = variant["architecture"]
        normalized = bool(variant["normalized_scoring"])
        base_key = f"{architecture}:{normalized}"
        variant_seed = int(document["training"]["seed"])
        print(
            f"phase=ml1m_variant_start variant={variant_id} architecture={architecture} "
            f"normalized={normalized}",
            flush=True,
        )
        if base_key not in base_states:
            _seed_everything(variant_seed)
            base_model = make_model(document, architecture, device)
            base_training = _train(
                base_model,
                base_examples,
                document,
                normalized,
                "base",
                "full",
                variant_seed + 1009,
            )
            base_state = {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}
            base_checkpoint = _save_checkpoint(
                output_root / "checkpoints" / base_key.replace(":", "_") / "theta0.pt",
                base_model,
                architecture,
                {"training": base_training, "selected_users_sha256": causal["selected_users_sha256"]},
            )
            base_states[base_key] = {
                "state_dict": base_state,
                "training": base_training,
                "checkpoint": base_checkpoint,
            }
            del base_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        previous = make_model(document, architecture, device)
        previous.load_state_dict(base_states[base_key]["state_dict"])
        current = deepcopy(previous)
        update_training = _train(
            current,
            update_examples,
            document,
            normalized,
            "update",
            variant["update_scope"],
            variant_seed + 2003 + variant_index * 10007,
        )
        current_state = {name: value.detach().cpu() for name, value in current.state_dict().items()}
        parameter_delta = _state_delta(base_states[base_key]["state_dict"], current_state)
        checkpoint = _save_checkpoint(
            output_root / "checkpoints" / variant_id / "theta1.pt",
            current,
            architecture,
            {
                "training": update_training,
                "selected_users_sha256": causal["selected_users_sha256"],
                "parameter_delta": parameter_delta,
            },
        )
        evaluation = _evaluate(
            previous,
            current,
            causal["splits"]["test"],
            selected,
            document,
            normalized,
        )
        summary = summarize_evaluation(evaluation, document)
        result = {
            "variant": variant,
            "base_training": base_states[base_key]["training"],
            "update_training": update_training,
            "parameter_delta": parameter_delta,
            "checkpoints": {
                "theta0": base_states[base_key]["checkpoint"],
                "theta1": checkpoint,
            },
            "evaluation": summary,
            "records": evaluation["records"],
        }
        variant_path = output_root / "variants" / f"{variant_id}.json"
        _atomic_json(variant_path, result)
        variant_results.append(
            {
                "id": variant_id,
                "result_path": str(variant_path),
                "result_sha256": file_sha256(variant_path),
                "evaluation": summary,
                "parameter_delta": parameter_delta,
            }
        )
        stale = summary["comparisons"]["recompute_over_reuse"]
        print(
            f"phase=ml1m_variant_complete variant={variant_id} "
            f"stale_ce={stale['candidate_cross_entropy']['absolute']:.6f} "
            f"stale_mrr={stale['mrr']['absolute']:.6f} "
            f"passed={summary['gate']['passed_screen']}",
            flush=True,
        )
        del previous, current
        if device.type == "cuda":
            torch.cuda.empty_cache()
    passed = [value["id"] for value in variant_results if value["evaluation"]["gate"]["passed_screen"]]
    summary = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": source_hash},
        "data": {
            "available_users": causal["available_users"],
            "selected_users": len(selected),
            "selected_users_sha256": causal["selected_users_sha256"],
            "target_leakage": False,
            "update_target": "dev positive",
            "held_out_target": "test positive",
            "candidate_pool": "frozen pilot20 test candidates",
        },
        "variants": variant_results,
        "decision": {
            "passed_variants": passed,
            "positive_candidate_found": bool(passed),
            "next": "medium_replication" if passed else "bounded_factor_followup",
        },
        "elapsed_seconds": time.monotonic() - started,
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
    }
    validate_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    variants = summary.get("variants")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete"
        or summary.get("scientific_result") is not False
        or summary.get("formal_result") is not False
        or not isinstance(variants, list)
        or [value.get("id") for value in variants] != [value["id"] for value in document["variants"]]
    ):
        raise ValueError("ML1m opportunity summary differs")
    for variant in variants:
        result_path = Path(variant["result_path"])
        if not result_path.is_file() or file_sha256(result_path) != variant["result_sha256"]:
            raise ValueError("ML1m opportunity variant binding differs")
        evaluation = variant.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("users") != int(
            document["data"]["user_limit"]
        ):
            raise ValueError("ML1m opportunity evaluation coverage differs")
        if not evaluation.get("sanity", {}).get("passed"):
            raise ValueError("ML1m opportunity same-model sanity failed")
