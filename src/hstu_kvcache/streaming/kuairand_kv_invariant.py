from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..models import HSTU
from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_cache_compatible import (
    OutputHead,
    _aggregate_edge,
    _full_catalog_multi,
)
from .kuairand_root_cause import (
    METRICS,
    _atomic_json,
    _atomic_torch,
    _evaluation_sequence,
    _hash_int_array,
    _load_checkpoint,
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

PROTOCOL = "evokv_root_cause_kuairand_kv_invariant_v0"


def kv_invariant_parameter_names(model: HSTU) -> list[str]:
    last = len(model.blocks) - 1
    candidates = [
        f"blocks.{last}.attn.q_proj.weight",
        f"blocks.{last}.attn.out_proj.weight",
        f"blocks.{last}.gate_proj.weight",
        "final_norm.weight",
    ]
    available = dict(model.named_parameters())
    if any(name not in available for name in candidates):
        raise ValueError("HSTU KV-invariant native tail differs")
    return candidates


def _set_trainable_tail(model: HSTU) -> list[torch.nn.Parameter]:
    names = set(kv_invariant_parameter_names(model))
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
        if name in names:
            trainable.append(parameter)
    return trainable


def validate_invariant_document(document: dict[str, object]) -> dict[str, object]:
    campaign = document.get("campaign")
    parent = document.get("parent")
    precursor = document.get("precursor")
    training = document.get("training")
    revision = document.get("model_revision")
    quality = document.get("quality")
    execution = document.get("execution")
    decision = document.get("decision_rule")
    outputs = document.get("outputs")
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
                precursor,
                training,
                revision,
                quality,
                execution,
                decision,
                outputs,
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
        or revision.get("input_item_embedding") != "frozen_theta0"
        or revision.get("cache_producing_parameters") != "frozen_theta0"
        or revision.get("output_head") != "untied_all_context_items"
        or revision.get("ranking_catalog") != "base_prediction_items_only"
        or revision.get("trainable_backbone_parameters")
        != [
            "blocks.2.attn.q_proj.weight",
            "blocks.2.attn.out_proj.weight",
            "blocks.2.gate_proj.weight",
            "final_norm.weight",
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
    ):
        raise ValueError("KuaiRand KV-invariant config differs")
    bindings = (
        (campaign, "path", "sha256"),
        (parent, "config", "config_sha256"),
        (parent, "training_result", "training_sha256"),
        (parent, "evaluation_result", "evaluation_sha256"),
        (parent, "runner", "runner_sha256"),
        (precursor, "config", "config_sha256"),
        (precursor, "training_result", "training_sha256"),
        (precursor, "evaluation_result", "evaluation_sha256"),
        (precursor, "runner", "runner_sha256"),
    )
    for binding, path_name, hash_name in bindings:
        if file_sha256(Path(binding[path_name])) != binding[hash_name]:
            raise ValueError(f"KuaiRand KV-invariant binding differs: {path_name}")
    parent_document = json.loads(Path(parent["config"]).read_text())
    validate_document(parent_document)
    for checkpoint in checkpoints:
        manifest = (
            Path(parent["checkpoint_root"])
            / f"theta_{checkpoint['version']}"
            / "manifest.json"
        )
        if file_sha256(manifest) != checkpoint["manifest_sha256"]:
            raise ValueError("KuaiRand KV-invariant parent checkpoint differs")
    return parent_document


def _checkpoint_paths(root: Path, version: int) -> tuple[Path, Path]:
    directory = root / f"theta_{version}"
    return directory / "kv_invariant_model.pt", directory / "manifest.json"


def _tail_state(model: HSTU) -> dict[str, torch.Tensor]:
    names = set(kv_invariant_parameter_names(model))
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name in names
    }


def _load_tail(model: HSTU, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = set(kv_invariant_parameter_names(model))
    if set(state) != expected:
        raise ValueError("KuaiRand KV-invariant tail state differs")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def _valid_candidate(
    root: Path,
    version: int,
    config_sha256: str,
    parent_theta0_manifest_sha256: str,
) -> bool:
    checkpoint_path, manifest_path = _checkpoint_paths(root, version)
    if not checkpoint_path.exists() or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return bool(
        manifest.get("protocol") == PROTOCOL
        and manifest.get("version") == version
        and manifest.get("config_sha256") == config_sha256
        and manifest.get("parent_theta0_manifest_sha256")
        == parent_theta0_manifest_sha256
        and manifest.get("checkpoint_sha256") == file_sha256(checkpoint_path)
    )


def _save_candidate(
    model: HSTU,
    head: OutputHead,
    root: Path,
    version: int,
    config_path: Path,
    parent_theta0_manifest_sha256: str,
    training_record: dict[str, object],
) -> None:
    checkpoint_path, manifest_path = _checkpoint_paths(root, version)
    if checkpoint_path.exists() or manifest_path.exists():
        raise FileExistsError(f"partial or existing KV-invariant theta{version}")
    state = {
        "tail": _tail_state(model),
        "output_head": {"weight": head.weight.detach().cpu()},
    }
    _atomic_torch(checkpoint_path, state)
    trainable = kv_invariant_parameter_names(model)
    parameters = dict(model.named_parameters())
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_development_checkpoint",
        "scientific_result": False,
        "formal_result": False,
        "version": version,
        "config_sha256": file_sha256(config_path),
        "program_sha256": file_sha256(Path(__file__)),
        "parent_theta0_manifest_sha256": parent_theta0_manifest_sha256,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "trainable_backbone_parameters": trainable,
        "trainable_backbone_parameter_count": int(
            sum(parameters[name].numel() for name in trainable)
        ),
        "output_head_shape": list(head.weight.shape),
        "training": training_record,
    }
    _atomic_json(manifest_path, manifest)


def _load_candidate(model: HSTU, head: OutputHead, root: Path, version: int) -> None:
    checkpoint_path, _ = _checkpoint_paths(root, version)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _load_tail(model, state["tail"])
    head.load_state_dict(state["output_head"])


def _training_step(
    model: HSTU,
    head: OutputHead,
    trainable: list[torch.nn.Parameter],
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    negative_count: int,
) -> tuple[float, int]:
    model.train()
    head.train()
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
    logits = head.score(hidden[:, :-1], candidates)
    labels_zero = torch.zeros_like(targets)
    per_target = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        labels_zero.flatten(),
        reduction="none",
    ).view_as(targets)
    loss = (per_target * valid).sum() / valid.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([*trainable, *head.parameters()], 1.0)
    optimizer.step()
    return float(loss.item()), target_count


def _train_epoch(
    model: HSTU,
    head: OutputHead,
    trainable: list[torch.nn.Parameter],
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
        loss, count = _training_step(
            model,
            head,
            trainable,
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


def run_invariant_training(config_path: Path) -> dict[str, object]:
    document = json.loads(config_path.read_text())
    parent_document = validate_invariant_document(document)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand KV-invariant training is single-rank")
    output = Path(document["outputs"]["training_result"])
    config_sha256 = file_sha256(config_path)
    if output.exists():
        result = json.loads(output.read_text())
        if (
            result.get("status") != "complete_development_training"
            or result.get("config", {}).get("sha256") != config_sha256
        ):
            raise FileExistsError("KuaiRand KV-invariant training result differs")
        return result
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    training = document["training"]
    _seed_everything(int(training["seed"]))
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
        raise ValueError("KuaiRand KV-invariant theta0 differs")
    model = make_model(parent_document, plan, device)
    _load_checkpoint(model, parent_root, 0)
    trainable = _set_trainable_tail(model)
    head = OutputHead(model.item_emb.weight).to(device)
    optimizer = torch.optim.AdamW(
        [*trainable, *head.parameters()],
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
            or not _valid_candidate(
                root,
                candidate_version,
                config_sha256,
                parent_theta0_hash,
            )
        ):
            raise ValueError("KuaiRand KV-invariant resume differs")
        completed_version = candidate_version
        _load_tail(model, resume["tail_state"])
        head.load_state_dict(resume["head_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
    records = []
    for version in range(1, completed_version + 1):
        _, manifest_path = _checkpoint_paths(root, version)
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
                    model,
                    head,
                    trainable,
                    optimizer,
                    batches,
                    device,
                    int(training["negative_count"]),
                    training["maximum_update_batches_per_epoch"],
                    f"kuairand_kv_invariant_theta{version}_epoch{epoch + 1}",
                )
            )
        record = {
            "version": version,
            "role": "kv_invariant_native_tail_and_output_update",
            "date": date,
            "epochs": epochs,
        }
        _save_candidate(
            model,
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
                "tail_state": _tail_state(model),
                "head_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
        )
        records.append(record)
        completed_version = version
        gc.collect()
        torch.cuda.empty_cache()
    names = kv_invariant_parameter_names(model)
    parameters = dict(model.named_parameters())
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
                "path": "src/hstu_kvcache/streaming/kuairand_kv_invariant.py",
                "sha256": program_sha256,
            }
        },
        "parent": parent,
        "precursor": document["precursor"],
        "data": data_metadata,
        "model_revision": {
            **document["model_revision"],
            "trainable_backbone_parameter_count": int(
                sum(parameters[name].numel() for name in names)
            ),
            "output_head_shape": list(head.weight.shape),
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
def _evaluate_edge(
    document: dict[str, object],
    parent_result: dict[str, object],
    plan,
    full_models: list[HSTU],
    candidate_models: list[HSTU],
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
        raise ValueError("KuaiRand KV-invariant selected users differ")
    local_users = selected[runtime.rank::runtime.world_size]
    base = candidate_models[0]
    previous_candidate = candidate_models[version - 1]
    current_candidate = candidate_models[version]
    previous_full = full_models[version - 1]
    current_full = full_models[version]
    buffers = []
    suffix_chunk = int(quality["suffix_chunk"])
    for ordinal, user in enumerate(local_users):
        sequence = _evaluation_sequence(plan, user, eval_date)
        prefix = sequence["prefix"]
        reuse_cache = _stored_cache(base, prefix, runtime.device)
        exact_cache = _stored_cache(current_candidate, prefix, runtime.device)
        base_hidden = _run_suffix(
            base,
            reuse_cache,
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        previous_hidden = _run_suffix(
            previous_candidate,
            _stored_cache(base, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        current_reuse = _run_suffix(
            current_candidate,
            reuse_cache,
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        current_exact = _run_suffix(
            current_candidate,
            exact_cache,
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        full_previous = _run_suffix(
            previous_full,
            _stored_cache(previous_full, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        full_current = _run_suffix(
            current_full,
            _stored_cache(current_full, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        targets = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
        if not (
            len(targets)
            == len(base_hidden)
            == len(previous_hidden)
            == len(current_reuse)
            == len(current_exact)
            == len(full_previous)
            == len(full_current)
        ):
            raise RuntimeError("KuaiRand KV-invariant target alignment differs")
        buffers.append(
            {
                "user_id": user,
                "targets": targets,
                "hidden": {
                    "base_anchor": base_hidden,
                    "compatible_previous": previous_hidden,
                    "compatible_current_reuse": current_reuse,
                    "compatible_current_exact": current_exact,
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
                        (current_reuse - current_exact).abs().max().item()
                    ),
                },
            }
        )
        print(
            f"phase=kuairand_kv_invariant_edge{version}_hidden rank={runtime.rank} "
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
        previous_full.item_emb.weight,
        current_full.item_emb.weight,
    ]
    nll, ranks = _full_catalog_multi(
        weights,
        hidden,
        positives,
        int(base.cfg.num_prediction_items),
        int(quality["target_chunk"]),
        int(quality["full_catalog_item_chunk"]),
        runtime.device,
        f"kuairand_kv_invariant_edge{version}_rank{runtime.rank}",
    )
    records = []
    start = 0
    reuse_index = methods.index("compatible_current_reuse")
    exact_index = methods.index("compatible_current_exact")
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
                    (nll[reuse_index, start:stop] - nll[exact_index, start:stop])
                    .abs()
                    .max()
                    .item()
                ),
                "ranks_equal": bool(
                    torch.equal(
                        ranks[reuse_index, start:stop],
                        ranks[exact_index, start:stop],
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


def _decision(edges: list[dict[str, object]], rule: dict[str, object]) -> dict[str, object]:
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
    retention = 100.0 * compatible_ce / full_ce if abs(full_ce) > 1e-12 else None
    implementation = all(edge["sanity"]["implementation_passed"] for edge in edges)
    if (
        implementation
        and all(ce_positive)
        and all(ranking_positive)
        and retention is not None
        and retention >= float(rule["primary_pooled_ce_retention_percent"])
    ):
        classification = "primary_kv_invariant_update_candidate"
        next_route = "kv_invariant_stream_updates_with_periodic_full_refresh_control"
    elif (
        implementation
        and all(ce_positive)
        and retention is not None
        and retention >= float(rule["hybrid_pooled_ce_retention_percent"])
    ):
        classification = "hybrid_kv_invariant_update_candidate"
        next_route = "kv_invariant_stream_updates_with_periodic_exact_backbone_refresh"
    else:
        classification = "native_kv_invariant_revision_rejected"
        next_route = "periodic_full_update_with_exact_cache_renewal"
    return {
        "classification": classification,
        "next_route": next_route,
        "pooled_positive_targets": targets,
        "pooled_cross_entropy_update_advantage": {
            "compatible": compatible_ce / max(targets, 1),
            "full": full_ce / max(targets, 1),
            "retention_percent": retention,
        },
        "positive_cross_entropy_ci_by_edge": ce_positive,
        "positive_ranking_ci_by_edge": ranking_positive,
        "implementation_passed": implementation,
        "rule": rule,
    }


def run_invariant_evaluation(config_path: Path) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    parent_document = validate_invariant_document(document)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != int(document["execution"]["evaluation_world_size"]):
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand KV-invariant evaluation world size differs")
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
                raise FileExistsError("KuaiRand KV-invariant evaluation result differs")
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
                raise ValueError(f"KuaiRand KV-invariant theta{version} differs")
        full_models = []
        for version in range(3):
            model = make_model(parent_document, plan, runtime.device)
            _load_checkpoint(model, parent_root, version)
            model.eval()
            full_models.append(model)
        candidate_models = []
        heads = []
        root = Path(document["outputs"]["checkpoint_root"])
        parent_theta0_hash = document["checkpoints"][0]["manifest_sha256"]
        for version in range(3):
            model = make_model(parent_document, plan, runtime.device)
            _load_checkpoint(model, parent_root, 0)
            head = OutputHead(model.item_emb.weight).to(runtime.device)
            if version:
                if not _valid_candidate(
                    root,
                    version,
                    config_sha256,
                    parent_theta0_hash,
                ):
                    raise ValueError(f"KuaiRand KV-invariant candidate{version} differs")
                _load_candidate(model, head, root, version)
            model.eval()
            head.eval()
            candidate_models.append(model)
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
                full_models,
                candidate_models,
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
        decision = _decision(edges, document["decision_rule"])
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
                    "path": "src/hstu_kvcache/streaming/kuairand_kv_invariant.py",
                    "sha256": file_sha256(Path(__file__)),
                },
                "parent_runner": {
                    "path": parent["runner"],
                    "sha256": parent["runner_sha256"],
                },
            },
            "parent": parent,
            "precursor": document["precursor"],
            "data": data_metadata,
            "model_revision": document["model_revision"],
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
