from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .kuairand_query_transition import (
    _atomic_json,
    _atomic_torch,
    _collate,
    _evaluate,
    _forward,
    _score,
    _seed_everything,
    _summary,
    _training_candidates,
    build_workload,
    file_sha256,
    load_config,
    make_model,
)

PROTOCOL = "evokv_kuairand_query_update_scope_v0"


def load_scope_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    candidates = document.get("candidates")
    selection = document.get("selection")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(candidates, list)
        or not candidates
        or not isinstance(selection, dict)
        or selection.get("metrics") != ["mrr", "ndcg_at_10", "hit_rate_at_10"]
        or float(selection.get("minimum_relative_percent", 0)) not in (3.0, 5.0)
        or float(selection.get("maximum_relative_percent", 0)) != 15.0
        or int(selection.get("minimum_metrics", 0)) != 2
    ):
        raise ValueError("KuaiRand update-scope config differs")
    base = parent.get("base_config")
    if not isinstance(base, dict):
        raise ValueError("KuaiRand update-scope base config is absent")
    base_path = Path(base.get("path", ""))
    if not base_path.is_file() or file_sha256(base_path) != base.get("sha256"):
        raise ValueError("KuaiRand update-scope base config differs")
    theta0 = parent.get("theta0")
    if not isinstance(theta0, list) or not theta0:
        raise ValueError("KuaiRand update-scope theta0 list is absent")
    for value in theta0:
        artifact = Path(value.get("path", ""))
        if (
            not artifact.is_file()
            or file_sha256(artifact) != value.get("sha256")
            or int(value.get("seed", -1)) < 0
        ):
            raise ValueError("KuaiRand update-scope theta0 differs")
    names = set()
    for candidate in candidates:
        name = str(candidate.get("name", ""))
        milestones = candidate.get("milestones")
        if (
            not name
            or name in names
            or candidate.get("scope")
            not in ("full", "core_only", "cache_breaking_only")
            or float(candidate.get("learning_rate", 0)) <= 0
            or not isinstance(milestones, list)
            or not milestones
            or milestones != sorted(set(int(value) for value in milestones))
            or int(milestones[0]) < 1
        ):
            raise ValueError("KuaiRand update-scope candidate differs")
        names.add(name)
    return document


def _configure_scope(model, scope: str) -> list[torch.nn.Parameter]:
    final_block = model.cfg.num_layers - 1
    selected = []
    for name, parameter in model.named_parameters():
        if scope == "full":
            active = True
        elif scope == "core_only":
            active = not name.startswith("item_emb.")
        elif name.startswith("item_emb.") or name.startswith("final_norm."):
            active = False
        elif name.startswith(f"blocks.{final_block}.attn.q_proj.") or name.startswith(
            f"blocks.{final_block}.attn.out_proj."
        ):
            active = False
        else:
            active = True
        parameter.requires_grad_(active)
        if active:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("KuaiRand update scope selected no parameters")
    return selected


def _train_epoch(
    model,
    examples,
    negative_pool: torch.Tensor,
    rank_by_item: torch.Tensor,
    author_by_item: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    rng: np.random.Generator,
    base_config: dict[str, Any],
    trainable: list[torch.nn.Parameter],
) -> float:
    training = base_config["training"]
    device = next(model.parameters()).device
    model.train()
    order = rng.permutation(len(examples))
    loss_sum = 0.0
    count = 0
    batch_size = int(training["batch_size"])
    for start in range(0, len(order), batch_size):
        batch = [examples[int(index)] for index in order[start : start + batch_size]]
        items, behaviors, deltas, lengths, targets = _collate(batch, device)
        optimizer.zero_grad(set_to_none=True)
        hidden = _forward(model, items, behaviors, deltas, lengths, author_by_item)
        vectors = model.last_hidden(hidden, lengths)
        candidates = _training_candidates(
            targets,
            negative_pool,
            int(training["negative_samples"]),
            generator,
            str(training["negative_source"]),
            model.cfg.num_prediction_items,
            rank_by_item,
        )
        scores = _score(
            model,
            vectors,
            candidates,
            float(training["temperature"]),
            author_by_item,
        )
        loss = F.cross_entropy(
            scores,
            torch.zeros(len(batch), dtype=torch.long, device=device),
        )
        if not torch.isfinite(loss):
            raise RuntimeError("KuaiRand update-scope loss is nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        optimizer.step()
        loss_sum += float(loss.detach().item()) * len(batch)
        count += len(batch)
    model.eval()
    return loss_sum / count


def _trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _delta(
    initial: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> dict[str, float]:
    numerator = sum(
        float((current[name].double() - value.double()).pow(2).sum().item())
        for name, value in initial.items()
    )
    denominator = sum(
        float(value.double().pow(2).sum().item()) for value in initial.values()
    )
    return {
        "absolute_l2": numerator**0.5,
        "relative_l2": numerator**0.5 / max(denominator**0.5, 1e-12),
    }


def _selection_pass(summary: dict[str, Any], selection: dict[str, Any]) -> list[str]:
    stale = summary["comparisons"]["recompute_over_reuse"]
    return [
        metric
        for metric in selection["metrics"]
        if stale[metric]["positive_direction_with_ci"]
        and float(selection["minimum_relative_percent"])
        <= stale[metric]["relative_percent"]
        <= float(selection["maximum_relative_percent"])
    ]


def run_scope_sweep(config_path: str | Path) -> dict[str, Any]:
    document = load_scope_config(config_path)
    output_root = Path(document["outputs"]["root"])
    result_path = output_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        validate_scope_result(result, document)
        return result
    base_config = load_config(document["parent"]["base_config"]["path"])
    workload = build_workload(base_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KuaiRand update-scope sweep requires CUDA")
    negative_count = min(
        int(base_config["training"]["negative_pool_size"]),
        len(workload["popular_ids"]),
    )
    negative_pool = torch.as_tensor(
        np.asarray(workload["popular_ids"][:negative_count]).copy(),
        dtype=torch.long,
        device=device,
    )
    rank_by_item = torch.as_tensor(
        workload["rank_by_item"], dtype=torch.long, device=device
    )
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(),
        dtype=torch.long,
        device=device,
    )
    cells = []
    started = time.monotonic()
    for parent in document["parent"]["theta0"]:
        seed = int(parent["seed"])
        payload = torch.load(parent["path"], map_location="cpu", weights_only=True)
        previous = make_model(
            base_config, int(workload["metadata"]["embedding_rows"]), device
        )
        previous.load_state_dict(payload["state_dict"])
        previous.eval()
        for candidate in document["candidates"]:
            _seed_everything(seed + 2003)
            current = make_model(
                base_config, int(workload["metadata"]["embedding_rows"]), device
            )
            current.load_state_dict(payload["state_dict"])
            trainable = _configure_scope(current, str(candidate["scope"]))
            initial = _trainable_state(current)
            optimizer = torch.optim.AdamW(
                trainable,
                lr=float(candidate["learning_rate"]),
                weight_decay=float(base_config["training"]["weight_decay"]),
                foreach=False,
            )
            generator = torch.Generator(device=device).manual_seed(seed + 11176)
            rng = np.random.default_rng(seed + 2003)
            losses = []
            maximum_epoch = int(max(candidate["milestones"]))
            for epoch in range(1, maximum_epoch + 1):
                loss = _train_epoch(
                    current,
                    workload["update_examples"],
                    negative_pool,
                    rank_by_item,
                    author_by_item,
                    optimizer,
                    generator,
                    rng,
                    base_config,
                    trainable,
                )
                losses.append(loss)
                print(
                    f"phase=kuairand_query_scope_train seed={seed} "
                    f"candidate={candidate['name']} epoch={epoch}/{maximum_epoch} "
                    f"loss={loss:.6f}",
                    flush=True,
                )
                if epoch not in candidate["milestones"]:
                    continue
                evaluation = _evaluate(previous, current, workload, base_config)
                compact = _summary(evaluation, base_config)
                current_state = _trainable_state(current)
                checkpoint_path = (
                    output_root
                    / "checkpoints"
                    / f"seed_{seed}"
                    / f"{candidate['name']}_epoch_{epoch}.pt"
                )
                _atomic_torch(
                    checkpoint_path,
                    {
                        "protocol": PROTOCOL,
                        "parent": parent,
                        "candidate": candidate,
                        "epoch": epoch,
                        "trainable_state_dict": current_state,
                    },
                )
                cell_path = (
                    output_root
                    / "cells"
                    / f"seed_{seed}_{candidate['name']}_epoch_{epoch}.json"
                )
                passing = _selection_pass(compact, document["selection"])
                cell = {
                    "seed": seed,
                    "candidate": candidate["name"],
                    "scope": candidate["scope"],
                    "learning_rate": candidate["learning_rate"],
                    "epoch": epoch,
                    "losses": losses.copy(),
                    "parameter_delta": _delta(initial, current_state),
                    "summary": compact,
                    "selection_passing_metrics": passing,
                    "selection_passed": compact["gate"]["same_model_sanity"]
                    and compact["gate"]["fresh_update_ranking_positive"]
                    and compact["gate"]["history_ranking_positive"]
                    and compact["gate"]["stale_candidate_ce_positive_ci"]
                    and len(passing)
                    >= int(document["selection"]["minimum_metrics"]),
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": file_sha256(checkpoint_path),
                        "bytes": checkpoint_path.stat().st_size,
                    },
                    "records": evaluation["records"],
                }
                _atomic_json(cell_path, cell)
                cells.append(
                    {
                        key: value
                        for key, value in cell.items()
                        if key not in ("records",)
                    }
                    | {"path": str(cell_path), "sha256": file_sha256(cell_path)}
                )
                stale = compact["comparisons"]["recompute_over_reuse"]
                print(
                    f"phase=kuairand_query_scope_eval seed={seed} "
                    f"candidate={candidate['name']} epoch={epoch} "
                    f"mrr={stale['mrr']['relative_percent']:.3f}% "
                    f"ndcg10={stale['ndcg_at_10']['relative_percent']:.3f}% "
                    f"hr10={stale['hit_rate_at_10']['relative_percent']:.3f}% "
                    f"gate={compact['gate']['passed']}",
                    flush=True,
                )
            del current, optimizer, trainable, initial
            torch.cuda.empty_cache()
        del previous, payload
        torch.cuda.empty_cache()
    seeds = [int(value["seed"]) for value in document["parent"]["theta0"]]
    common = []
    for candidate in document["candidates"]:
        for epoch in candidate["milestones"]:
            matched = [
                value
                for value in cells
                if value["candidate"] == candidate["name"]
                and value["epoch"] == epoch
                and value["selection_passed"]
            ]
            if sorted(value["seed"] for value in matched) == sorted(seeds):
                common.append({"candidate": candidate["name"], "epoch": epoch})
    result = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "workload": workload["metadata"],
        "cells": cells,
        "decision": {
            "common_admitted": common,
            "selected": common[0] if common else None,
            "next": "capacity_scale" if common else "revise_update_trajectory",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_scope_result(result, document)
    _atomic_json(result_path, result)
    return result


def validate_scope_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    cells = result.get("cells")
    if (
        result.get("protocol") != PROTOCOL
        or result.get("round_id") != document["round_id"]
        or result.get("status") != "complete"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or not isinstance(cells, list)
        or not cells
    ):
        raise ValueError("KuaiRand update-scope result differs")
    for cell in cells:
        path = Path(cell["path"])
        checkpoint = Path(cell["checkpoint"]["path"])
        if (
            not path.is_file()
            or file_sha256(path) != cell["sha256"]
            or not checkpoint.is_file()
            or file_sha256(checkpoint) != cell["checkpoint"]["sha256"]
            or not cell.get("summary", {}).get("sanity", {}).get("passed")
        ):
            raise ValueError("KuaiRand update-scope cell binding differs")
