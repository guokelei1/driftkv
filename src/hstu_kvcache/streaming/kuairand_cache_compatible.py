from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from ..models import HSTU
from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_root_cause import (
    METRICS,
    _atomic_json,
    _atomic_torch,
    _evaluation_sequence,
    _hash_int_array,
    _load_checkpoint,
    _metric_comparison,
    _metric_sums,
    _run_suffix,
    _seed_everything,
    _selected_users,
    _stored_cache,
    _valid_checkpoint,
    load_plan,
    make_model,
    validate_document,
)
from .qk_stream_version import file_sha256
from .trainer import build_next_item_targets

PROTOCOL = "evokv_root_cause_kuairand_cache_compatible_v0"


class OutputHead(nn.Module):
    def __init__(self, initial_weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(initial_weight.detach().clone())

    def score(self, hidden: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.weight[candidate_ids]
        return torch.einsum("...h,...ch->...c", hidden, vectors)


def validate_compatible_document(document: dict[str, object]) -> dict[str, object]:
    campaign = document.get("campaign")
    parent = document.get("parent")
    attribution = document.get("attribution")
    training = document.get("training")
    quality = document.get("quality")
    execution = document.get("execution")
    outputs = document.get("outputs")
    decision = document.get("decision_rule")
    revision = document.get("model_revision")
    methods = document.get("methods")
    checkpoints = document.get("checkpoints")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scope") != "development_model_revision"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (
                campaign,
                parent,
                attribution,
                training,
                quality,
                execution,
                outputs,
                decision,
                revision,
            )
        )
        or not isinstance(checkpoints, list)
        or [value.get("version") for value in checkpoints] != [0, 1, 2]
        or methods
        != [
            "base_anchor",
            "compatible_previous",
            "compatible_current_reuse",
            "compatible_current_exact",
            "full_previous",
            "full_current",
        ]
        or int(training.get("batch_size", 0)) != 8
        or int(training.get("negative_count", 0)) != 32
        or int(training.get("update_epochs", 0)) != 2
        or float(training.get("stream_lr", 0.0)) != 0.0001
        or float(training.get("weight_decay", -1.0)) != 0.0001
        or training.get("training_sequences") != "all_chunks"
        or int(quality.get("record_limit_per_rank", 0)) != 128
        or int(quality.get("target_chunk", 0)) < 1
        or int(quality.get("full_catalog_item_chunk", 0)) < 1
        or int(quality.get("suffix_chunk", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or execution.get("training_cuda_visible_devices") != "0"
        or execution.get("evaluation_cuda_visible_devices") != "0,1"
        or int(execution.get("evaluation_world_size", 0)) != 2
        or float(decision.get("primary_pooled_ce_retention_percent", -1.0)) != 50.0
        or float(decision.get("hybrid_pooled_ce_retention_percent", -1.0)) != 25.0
        or decision.get("require_positive_ce_ci_on_each_edge") is not True
        or decision.get("require_positive_ranking_ci_on_each_edge_for_primary") is not True
        or revision.get("input_item_embedding") != "frozen_theta0"
        or revision.get("cache_producing_backbone") != "frozen_theta0"
        or revision.get("candidate_output_head") != "untied_sequential_scorer"
        or revision.get("output_head_rows") != "all_context_items"
        or revision.get("ranking_catalog") != "base_prediction_items_only"
    ):
        raise ValueError("KuaiRand cache-compatible config differs")
    bound_paths = (
        (campaign, "path", "sha256"),
        (parent, "config", "config_sha256"),
        (parent, "training_result", "training_sha256"),
        (parent, "evaluation_result", "evaluation_sha256"),
        (parent, "runner", "runner_sha256"),
        (attribution, "config", "config_sha256"),
        (attribution, "result", "result_sha256"),
        (attribution, "runner", "runner_sha256"),
    )
    for binding, path_name, hash_name in bound_paths:
        if file_sha256(Path(binding[path_name])) != binding[hash_name]:
            raise ValueError(f"KuaiRand cache-compatible binding differs: {path_name}")
    parent_document = json.loads(Path(parent["config"]).read_text())
    validate_document(parent_document)
    for checkpoint in checkpoints:
        manifest = (
            Path(parent["checkpoint_root"])
            / f"theta_{checkpoint['version']}"
            / "manifest.json"
        )
        if file_sha256(manifest) != checkpoint["manifest_sha256"]:
            raise ValueError("KuaiRand cache-compatible parent checkpoint differs")
    return parent_document


def _head_paths(root: Path, version: int) -> tuple[Path, Path]:
    directory = root / f"scorer_{version}"
    return directory / "output_head.pt", directory / "manifest.json"


def _valid_head(
    root: Path,
    version: int,
    config_sha256: str,
    parent_theta0_manifest_sha256: str,
) -> bool:
    head_path, manifest_path = _head_paths(root, version)
    if not head_path.exists() or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return bool(
        manifest.get("protocol") == PROTOCOL
        and manifest.get("version") == version
        and manifest.get("config_sha256") == config_sha256
        and manifest.get("parent_theta0_manifest_sha256")
        == parent_theta0_manifest_sha256
        and manifest.get("output_head_sha256") == file_sha256(head_path)
    )


def _save_head(
    head: OutputHead,
    root: Path,
    version: int,
    config_path: Path,
    parent_theta0_manifest_sha256: str,
    training_record: dict[str, object],
) -> dict[str, object]:
    head_path, manifest_path = _head_paths(root, version)
    if head_path.exists() or manifest_path.exists():
        raise FileExistsError(f"partial or existing compatible scorer{version}")
    _atomic_torch(head_path, {"weight": head.weight.detach().cpu()})
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_development_checkpoint",
        "scientific_result": False,
        "formal_result": False,
        "version": version,
        "config_sha256": file_sha256(config_path),
        "program_sha256": file_sha256(Path(__file__)),
        "parent_theta0_manifest_sha256": parent_theta0_manifest_sha256,
        "output_head_sha256": file_sha256(head_path),
        "shape": list(head.weight.shape),
        "dtype": str(head.weight.dtype),
        "training": training_record,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_head(head: OutputHead, root: Path, version: int) -> None:
    head_path, _ = _head_paths(root, version)
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    head.load_state_dict(state)


def _output_only_training_step(
    backbone: HSTU,
    head: OutputHead,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    negative_count: int,
) -> tuple[float, int]:
    backbone.train()
    head.train()
    item_ids = batch["item_ids"].to(device)
    behaviors = batch["behaviors"].to(device)
    time_deltas = batch["time_deltas"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch["labels"].to(device)
    train_mask = batch["train_mask"].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        hidden, _ = backbone(
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
    negatives = torch.randint(
        1,
        backbone.cfg.num_prediction_items + 1,
        (*targets.shape, negative_count),
        device=device,
    )
    negatives = torch.where(
        negatives == positives,
        negatives.remainder(backbone.cfg.num_prediction_items) + 1,
        negatives,
    )
    candidates = torch.cat((positives, negatives), dim=-1)
    logits = head.score(hidden[:, :-1], candidates)
    labels_zero = torch.zeros_like(targets)
    per_target = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        labels_zero.flatten(),
        reduction="none",
    ).view_as(targets)
    loss = (per_target * valid).sum() / valid.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), target_count


def _train_epoch(
    backbone: HSTU,
    head: OutputHead,
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
        loss, count = _output_only_training_step(
            backbone,
            head,
            batch,
            optimizer,
            device,
            negative_count,
        )
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


def run_compatible_training(config_path: Path) -> dict[str, object]:
    document = json.loads(config_path.read_text())
    parent_document = validate_compatible_document(document)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand cache-compatible training is single-rank")
    output = Path(document["outputs"]["training_result"])
    config_sha256 = file_sha256(config_path)
    if output.exists():
        result = json.loads(output.read_text())
        if (
            result.get("status") != "complete_development_training"
            or result.get("config", {}).get("sha256") != config_sha256
        ):
            raise FileExistsError("KuaiRand cache-compatible training result differs")
        return result
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(document["training"]["seed"]))
    started = time.perf_counter()
    plan, data_metadata = load_plan(parent_document)
    parent = document["parent"]
    parent_root = Path(parent["checkpoint_root"])
    if not _valid_checkpoint(
        parent_root,
        0,
        parent["config_sha256"],
        data_metadata,
    ):
        raise ValueError("KuaiRand cache-compatible theta0 differs")
    backbone = make_model(parent_document, plan, device)
    _load_checkpoint(backbone, parent_root, 0)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    head = OutputHead(backbone.item_emb.weight).to(device)
    training = document["training"]
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(training["stream_lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    root = Path(document["outputs"]["checkpoint_root"])
    resume_path = root / "resume.pt"
    parent_theta0_hash = document["checkpoints"][0]["manifest_sha256"]
    program_sha256 = file_sha256(Path(__file__))
    completed_version = 0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        candidate_version = int(resume.get("version", -1))
        if (
            resume.get("protocol") != PROTOCOL
            or resume.get("config_sha256") != config_sha256
            or resume.get("program_sha256") != program_sha256
            or candidate_version not in (1, 2)
            or not _valid_head(
                root,
                candidate_version,
                config_sha256,
                parent_theta0_hash,
            )
        ):
            raise ValueError("KuaiRand cache-compatible resume differs")
        completed_version = candidate_version
        head.load_state_dict(resume["head_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
    records = []
    for version in range(1, completed_version + 1):
        _, manifest_path = _head_paths(root, version)
        records.append(json.loads(manifest_path.read_text())["training"])
    plan.init_base()
    for date in plan.stream_dates[:completed_version]:
        plan.ingest_day(date)
    dates = plan.base_dates + plan.stream_dates
    for version, date_index in enumerate(
        parent_document["schedule"]["update_date_indices"], start=1
    ):
        if version <= completed_version:
            continue
        date = dates[int(date_index)]
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
                    backbone,
                    head,
                    optimizer,
                    batches,
                    device,
                    int(training["negative_count"]),
                    training["maximum_update_batches_per_epoch"],
                    f"kuairand_compatible_scorer{version}_epoch{epoch + 1}",
                )
            )
        record = {
            "version": version,
            "role": "output_scorer_only_stream_update",
            "date": date,
            "epochs": epochs,
        }
        _save_head(
            head,
            root,
            version,
            config_path,
            parent_theta0_hash,
            record,
        )
        _atomic_torch(
            resume_path,
            {
                "protocol": PROTOCOL,
                "config_sha256": config_sha256,
                "program_sha256": program_sha256,
                "version": version,
                "head_state": head.state_dict(),
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
                "path": "src/hstu_kvcache/streaming/kuairand_cache_compatible.py",
                "sha256": program_sha256,
            }
        },
        "parent": document["parent"],
        "data": data_metadata,
        "model_revision": {
            "input_item_embedding": "frozen_theta0",
            "cache_producing_backbone": "frozen_theta0",
            "candidate_output_head": "untied_sequential_scorer",
            "output_head_shape": list(head.weight.shape),
            "output_head_dtype": str(head.weight.dtype),
        },
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
            "resume_removed_after_complete": True,
        },
    }
    _atomic_json(output, result)
    if resume_path.exists():
        resume_path.unlink()
    return result


@torch.no_grad()
def _full_catalog_multi(
    weights: list[torch.Tensor],
    hidden: torch.Tensor,
    positives: torch.Tensor,
    prediction_items: int,
    target_chunk: int,
    item_chunk: int,
    device: torch.device,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or hidden.shape[0] != len(weights) or hidden.shape[1] != len(positives):
        raise ValueError("KuaiRand cache-compatible full-catalog inputs differ")
    if (
        prediction_items < 1
        or prediction_items >= weights[0].shape[0]
        or any(value.shape != weights[0].shape for value in weights)
    ):
        raise ValueError("KuaiRand cache-compatible scorer geometry differs")
    nll_parts = []
    rank_parts = []
    for start in range(0, len(positives), target_chunk):
        stop = min(start + target_chunk, len(positives))
        local_hidden = hidden[:, start:stop].to(device)
        local_positive = positives[start:stop].to(device)
        positive_scores = torch.stack(
            [
                (local_hidden[index] * weight.index_select(0, local_positive)).sum(dim=-1)
                for index, weight in enumerate(weights)
            ]
        )
        lse = torch.full_like(positive_scores, -torch.inf)
        ranks = torch.zeros_like(positive_scores, dtype=torch.int64)
        for item_start in range(1, prediction_items + 1, item_chunk):
            item_stop = min(item_start + item_chunk, prediction_items + 1)
            scores = torch.stack(
                [
                    torch.matmul(
                        local_hidden[index],
                        weight[item_start:item_stop].t(),
                    )
                    for index, weight in enumerate(weights)
                ]
            )
            lse = torch.logaddexp(lse, torch.logsumexp(scores.float(), dim=-1))
            ids = torch.arange(item_start, item_stop, dtype=torch.int64, device=device)
            positive_mask = local_positive[:, None] == ids[None, :]
            ranks += (
                (scores >= positive_scores.unsqueeze(-1))
                & ~positive_mask.unsqueeze(0)
            ).sum(dim=-1)
        nll_parts.append((lse - positive_scores).detach().cpu())
        rank_parts.append((ranks + 1).detach().cpu())
        print(f"phase={phase} targets={stop}/{len(positives)}", flush=True)
    return torch.cat(nll_parts, dim=1), torch.cat(rank_parts, dim=1)


def _endpoint(records: list[dict[str, object]], methods: list[str]) -> dict[str, object]:
    denominator = float(sum(value["targets"] for value in records))
    return {
        method: {
            metric: float(
                sum(value["metric_sums"][method][metric] for value in records)
                / denominator
            )
            for metric in METRICS
        }
        for method in methods
    }


def _comparison(
    records: list[dict[str, object]],
    left: str,
    right: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    return _metric_comparison(
        records,
        left,
        right,
        lambda value: value["metric_sums"][left],
        lambda value: value["metric_sums"][right],
        bootstrap_samples,
        bootstrap_seed,
    )


def _aggregate_edge(
    records: list[dict[str, object]],
    methods: list[str],
    quality: dict[str, object],
    version: int,
    parent_edge: dict[str, object],
) -> dict[str, object]:
    records = sorted(records, key=lambda value: int(value["user_id"]))
    samples = int(quality["bootstrap_samples"])
    seed = int(quality["bootstrap_seed"]) + version * 1000003
    endpoints = _endpoint(records, methods)
    comparisons = {
        "compatible_update_value": _comparison(
            records,
            "compatible_previous",
            "compatible_current_reuse",
            samples,
            seed,
        ),
        "full_update_value": _comparison(
            records,
            "full_previous",
            "full_current",
            samples,
            seed + 101,
        ),
        "compatible_current_vs_full_current": _comparison(
            records,
            "compatible_current_reuse",
            "full_current",
            samples,
            seed + 202,
        ),
        "compatible_cumulative_over_base": _comparison(
            records,
            "base_anchor",
            "compatible_current_reuse",
            samples,
            seed + 303,
        ),
        "full_cumulative_over_base": _comparison(
            records,
            "base_anchor",
            "full_current",
            samples,
            seed + 404,
        ),
        "reuse_exact_equivalence": _comparison(
            records,
            "compatible_current_exact",
            "compatible_current_reuse",
            samples,
            seed + 505,
        ),
    }
    retention = {}
    for metric in METRICS:
        compatible = comparisons["compatible_update_value"][metric][
            "compatible_current_reuse_advantage_absolute"
        ]
        full = comparisons["full_update_value"][metric][
            "full_current_advantage_absolute"
        ]
        retention[metric] = {
            "compatible_update_advantage_absolute": compatible,
            "full_update_advantage_absolute": full,
            "fraction_of_full_update_percent": (
                100.0 * compatible / full if abs(full) > 1e-12 else None
            ),
        }
    cache_error = max(value["sanity"]["cache_maximum_absolute_error"] for value in records)
    hidden_error = max(value["sanity"]["hidden_maximum_absolute_error"] for value in records)
    nll_error = max(value["sanity"]["nll_maximum_absolute_error"] for value in records)
    ranks_equal = all(value["sanity"]["ranks_equal"] for value in records)
    parent_endpoint_error = {
        "full_previous": {
            metric: abs(
                endpoints["full_previous"][metric]
                - parent_edge["endpoints"]["previous_fresh"][metric]
            )
            for metric in METRICS
        },
        "full_current": {
            metric: abs(
                endpoints["full_current"][metric]
                - parent_edge["endpoints"]["fresh_full_a"][metric]
            )
            for metric in METRICS
        },
    }
    parent_parity = bool(
        parent_endpoint_error["full_previous"]["cross_entropy"] <= 1e-5
        and parent_endpoint_error["full_current"]["cross_entropy"] <= 1e-5
        and all(
            parent_endpoint_error[model][metric] <= 1e-9
            for model in ("full_previous", "full_current")
            for metric in METRICS
            if metric != "cross_entropy"
        )
    )
    sanity = {
        "cache_maximum_absolute_error": cache_error,
        "hidden_maximum_absolute_error": hidden_error,
        "nll_maximum_absolute_error": nll_error,
        "ranks_equal": ranks_equal,
        "parent_endpoint_absolute_error": parent_endpoint_error,
        "parent_endpoint_parity": parent_parity,
    }
    sanity["implementation_passed"] = bool(
        cache_error == 0.0
        and hidden_error == 0.0
        and nll_error <= 1e-7
        and ranks_equal
        and parent_parity
    )
    return {
        "users": len(records),
        "positive_targets": int(sum(value["targets"] for value in records)),
        "endpoints": endpoints,
        "comparisons": comparisons,
        "update_retention": retention,
        "sanity": sanity,
        "records_detail": records,
    }


@torch.no_grad()
def _evaluate_edge(
    document: dict[str, object],
    parent_result: dict[str, object],
    plan,
    models: list[HSTU],
    heads: list[OutputHead],
    version: int,
    update_date: str,
    eval_date: str,
    runtime,
) -> dict[str, object] | None:
    quality = document["quality"]
    methods = list(document["methods"])
    selected, eligible = _selected_users(
        plan,
        update_date,
        eval_date,
        int(quality["record_limit_per_rank"]) * runtime.world_size,
        int(quality["sampling_seed"]) + version * 1009,
    )
    selected_hash = _hash_int_array(np.asarray(selected))
    parent_edge = parent_result["edges"][version - 1]
    if selected_hash != parent_edge["selected_user_ids_sha256"]:
        raise ValueError("KuaiRand cache-compatible selected users differ")
    local_users = selected[runtime.rank::runtime.world_size]
    buffers = []
    suffix_chunk = int(quality["suffix_chunk"])
    base = models[0]
    previous = models[version - 1]
    current = models[version]
    for ordinal, user in enumerate(local_users):
        sequence = _evaluation_sequence(plan, user, eval_date)
        prefix = sequence["prefix"]
        reuse_cache = _stored_cache(base, prefix, runtime.device)
        exact_cache = _stored_cache(base, prefix, runtime.device)
        compatible_reuse = _run_suffix(
            base,
            reuse_cache,
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        compatible_exact = _run_suffix(
            base,
            exact_cache,
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        full_previous = _run_suffix(
            previous,
            _stored_cache(previous, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        full_current = _run_suffix(
            current,
            _stored_cache(current, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        targets = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
        if not (
            len(targets)
            == len(compatible_reuse)
            == len(compatible_exact)
            == len(full_previous)
            == len(full_current)
        ):
            raise RuntimeError("KuaiRand cache-compatible target alignment differs")
        buffers.append(
            {
                "user_id": user,
                "targets": targets,
                "hidden": {
                    "base_anchor": compatible_reuse,
                    "compatible_previous": compatible_reuse,
                    "compatible_current_reuse": compatible_reuse,
                    "compatible_current_exact": compatible_exact,
                    "full_previous": full_previous,
                    "full_current": full_current,
                },
                "sanity": {
                    "cache_maximum_absolute_error": float(
                        max(
                            (reuse_cache.k - exact_cache.k).abs().max().item(),
                            (reuse_cache.v - exact_cache.v).abs().max().item(),
                        )
                    ),
                    "hidden_maximum_absolute_error": float(
                        (compatible_reuse - compatible_exact).abs().max().item()
                    ),
                },
            }
        )
        print(
            f"phase=kuairand_compatible_edge{version}_hidden rank={runtime.rank} "
            f"user={ordinal + 1}/{len(local_users)} targets={len(targets)}",
            flush=True,
        )
    positives = torch.cat([value["targets"] for value in buffers])
    hidden = torch.stack(
        [
            torch.cat([value["hidden"][method] for value in buffers])
            for method in methods
        ]
    )
    weights = [
        heads[0].weight,
        heads[version - 1].weight,
        heads[version].weight,
        heads[version].weight,
        previous.item_emb.weight,
        current.item_emb.weight,
    ]
    nll, ranks = _full_catalog_multi(
        weights,
        hidden,
        positives,
        int(base.cfg.num_prediction_items),
        int(quality["target_chunk"]),
        int(quality["full_catalog_item_chunk"]),
        runtime.device,
        f"kuairand_compatible_edge{version}_rank{runtime.rank}",
    )
    records = []
    start = 0
    for value in buffers:
        stop = start + len(value["targets"])
        metric_sums = {
            method: _metric_sums(nll[index, start:stop], ranks[index, start:stop])
            for index, method in enumerate(methods)
        }
        sanity = value["sanity"]
        sanity.update(
            {
                "nll_maximum_absolute_error": float(
                    (
                        nll[methods.index("compatible_current_reuse"), start:stop]
                        - nll[methods.index("compatible_current_exact"), start:stop]
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "ranks_equal": bool(
                    torch.equal(
                        ranks[methods.index("compatible_current_reuse"), start:stop],
                        ranks[methods.index("compatible_current_exact"), start:stop],
                    )
                ),
            }
        )
        records.append(
            {
                "user_id": value["user_id"],
                "targets": len(value["targets"]),
                "metric_sums": metric_sums,
                "sanity": sanity,
            }
        )
        start = stop
    gathered: list[object] | None = [None] * runtime.world_size if runtime.is_primary else None
    dist.gather_object(records, gathered, dst=0)
    del buffers, hidden, weights, nll, ranks
    gc.collect()
    torch.cuda.empty_cache()
    if not runtime.is_primary:
        return None
    combined = [record for shard in gathered for record in shard]
    aggregate = _aggregate_edge(
        combined,
        methods,
        quality,
        version,
        parent_edge,
    )
    aggregate.update(
        {
            "edge": version,
            "update_date": update_date,
            "evaluation_date": eval_date,
            "eligible_same_user_population": eligible,
            "selected_user_ids_sha256": selected_hash,
        }
    )
    return aggregate


def _campaign_decision(
    edges: list[dict[str, object]], decision_rule: dict[str, object]
) -> dict[str, object]:
    full_ce = 0.0
    compatible_ce = 0.0
    targets = 0
    ce_positive = []
    ranking_positive = []
    for edge in edges:
        count = int(edge["positive_targets"])
        targets += count
        retention = edge["update_retention"]["cross_entropy"]
        full_ce += count * float(retention["full_update_advantage_absolute"])
        compatible_ce += count * float(retention["compatible_update_advantage_absolute"])
        ce_positive.append(
            bool(
                edge["comparisons"]["compatible_update_value"]["cross_entropy"][
                    "compatible_current_reuse_advantage_positive_with_ci"
                ]
            )
        )
        ranking_positive.append(
            any(
                edge["comparisons"]["compatible_update_value"][metric][
                    "compatible_current_reuse_advantage_positive_with_ci"
                ]
                for metric in METRICS
                if metric != "cross_entropy"
            )
        )
    pooled_retention = 100.0 * compatible_ce / full_ce if abs(full_ce) > 1e-12 else None
    implementation = all(edge["sanity"]["implementation_passed"] for edge in edges)
    if (
        implementation
        and all(ce_positive)
        and all(ranking_positive)
        and pooled_retention is not None
        and pooled_retention
        >= float(decision_rule["primary_pooled_ce_retention_percent"])
    ):
        classification = "primary_cache_compatible_update_candidate"
        next_route = "freeze_cache_producing_backbone_and_stream_output_scorer"
    elif (
        implementation
        and all(ce_positive)
        and pooled_retention is not None
        and pooled_retention
        >= float(decision_rule["hybrid_pooled_ce_retention_percent"])
    ):
        classification = "hybrid_cache_compatible_update_candidate"
        next_route = "stream_output_scorer_with_periodic_exact_backbone_refresh"
    else:
        classification = "pure_output_only_revision_rejected"
        next_route = "periodic_backbone_update_with_exact_cache_renewal"
    return {
        "classification": classification,
        "next_route": next_route,
        "pooled_positive_targets": targets,
        "pooled_cross_entropy_update_advantage": {
            "compatible": compatible_ce / max(targets, 1),
            "full": full_ce / max(targets, 1),
            "retention_percent": pooled_retention,
        },
        "positive_cross_entropy_ci_by_edge": ce_positive,
        "positive_ranking_ci_by_edge": ranking_positive,
        "implementation_passed": implementation,
        "rule": decision_rule,
    }


def run_compatible_evaluation(config_path: Path) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    parent_document = validate_compatible_document(document)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != int(document["execution"]["evaluation_world_size"]):
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand cache-compatible evaluation world size differs")
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["evaluation_result"])
        config_sha256 = file_sha256(config_path)
        if output.exists():
            result = json.loads(output.read_text())
            if (
                result.get("status") != "complete_development_model_revision"
                or result.get("config", {}).get("sha256") != config_sha256
            ):
                raise FileExistsError("KuaiRand cache-compatible evaluation result differs")
            return result if runtime.is_primary else None
        _seed_everything(int(document["quality"]["sampling_seed"]) + runtime.rank)
        torch.set_float32_matmul_precision("high")
        plan, data_metadata = load_plan(parent_document)
        parent = document["parent"]
        parent_root = Path(parent["checkpoint_root"])
        for version in range(3):
            if not _valid_checkpoint(
                parent_root,
                version,
                parent["config_sha256"],
                data_metadata,
            ):
                raise ValueError(f"KuaiRand cache-compatible theta{version} differs")
        models = []
        for version in range(3):
            model = make_model(parent_document, plan, runtime.device)
            _load_checkpoint(model, parent_root, version)
            model.eval()
            models.append(model)
        heads = [
            OutputHead(models[0].item_emb.weight).to(runtime.device)
        ]
        head_root = Path(document["outputs"]["checkpoint_root"])
        parent_theta0_hash = document["checkpoints"][0]["manifest_sha256"]
        for version in (1, 2):
            if not _valid_head(
                head_root,
                version,
                config_sha256,
                parent_theta0_hash,
            ):
                raise ValueError(f"KuaiRand compatible scorer{version} differs")
            head = OutputHead(torch.empty_like(heads[0].weight)).to(runtime.device)
            _load_head(head, head_root, version)
            head.eval()
            heads.append(head)
        parent_result = json.loads(Path(parent["evaluation_result"]).read_text())
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        edges = []
        for version, (update_index, eval_index) in enumerate(
            zip(
                parent_document["schedule"]["update_date_indices"],
                parent_document["schedule"]["evaluation_date_indices"],
                strict=True,
            ),
            start=1,
        ):
            update_date = dates[int(update_index)]
            eval_date = dates[int(eval_index)]
            plan.ingest_day(update_date)
            edge = _evaluate_edge(
                document,
                parent_result,
                plan,
                models,
                heads,
                version,
                update_date,
                eval_date,
                runtime,
            )
            if runtime.is_primary:
                edges.append(edge)
        if not runtime.is_primary:
            dist.barrier()
            return None
        decision = _campaign_decision(edges, document["decision_rule"])
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_model_revision",
            "scope": document["scope"],
            "scientific_result": False,
            "formal_result": False,
            "round_id": document["round_id"],
            "config": {"path": str(config_path), "sha256": config_sha256},
            "programs": {
                "runner": {
                    "path": "src/hstu_kvcache/streaming/kuairand_cache_compatible.py",
                    "sha256": file_sha256(Path(__file__)),
                },
                "parent_runner": {
                    "path": parent["runner"],
                    "sha256": parent["runner_sha256"],
                },
            },
            "parent": parent,
            "attribution": document["attribution"],
            "data": data_metadata,
            "model_revision": {
                "input_item_embedding": "frozen_theta0",
                "cache_producing_backbone": "frozen_theta0",
                "candidate_output_head": "untied_sequential_scorer",
                "cache_compatibility": "exact_by_construction_and_measured",
            },
            "methods": document["methods"],
            "quality": document["quality"],
            "edges": edges,
            "decision": decision,
            "sanity": {
                "implementation_passed": all(
                    edge["sanity"]["implementation_passed"] for edge in edges
                )
            },
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
