from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_kv_only_candidate_triangle import (
    _load_model,
    _verify_bindings,
    load_candidate_triangle_config,
)
from .kuairand_kv_only_chain import _effective_document, load_kv_only_chain_config
from .kuairand_root_cause import (
    _atomic_json,
    _evaluation_sequence,
    _parameter_distances,
    _stored_cache,
    file_sha256,
    load_plan,
)

PROTOCOL = "evokv_kuairand_kv_only_logged_window_screen_v0"
METRICS = (
    "window_cross_entropy",
    "roc_auc",
    "average_precision",
    "ndcg_at_10",
    "ndcg_at_50",
    "recall_at_10",
    "recall_at_50",
)
LOWER_IS_BETTER = {"window_cross_entropy"}
EXPECTED_PAIRS = [
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
    [6, 5],
    [7, 6],
    [8, 7],
    [8, 1],
]


def load_logged_window_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("candidate_triangle_config", {})
    execution = document.get("execution", {})
    outputs = document.get("outputs", {})
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("pairs") != EXPECTED_PAIRS
        or document.get("metrics") != list(METRICS)
        or document.get("candidate_construction")
        != "deduplicated in-catalog exposures from the untouched next natural day"
        or int(document.get("bootstrap_samples", 0)) != 2000
        or int(document.get("bootstrap_seed", 0)) < 1
        or execution.get("cuda_visible_devices") != "0,1"
        or execution.get("world_size") != 2
        or not isinstance(source.get("path"), str)
        or not isinstance(source.get("sha256"), str)
        or not all(isinstance(outputs.get(name), str) for name in ("result", "table", "log"))
    ):
        raise ValueError("KuaiRand logged-window screen config differs")
    return document


def _verify(config: dict[str, Any]) -> dict[str, Any]:
    source = config["candidate_triangle_config"]
    if file_sha256(source["path"]) != source["sha256"]:
        raise ValueError("KuaiRand candidate triangle config hash differs")
    candidate_config = load_candidate_triangle_config(source["path"])
    _verify_bindings(candidate_config)
    return candidate_config


def _logged_candidates(plan, user: int, date: str) -> tuple[torch.Tensor, torch.Tensor]:
    frame = plan.daily_segments[date]
    frame = frame[frame["user_idx"] == user].sort_values("time_ms")
    frame = frame[
        (frame["item_idx"] >= 1) & (frame["item_idx"] <= plan.num_prediction_items)
    ]
    grouped = frame.groupby("item_idx", sort=False)["label"].max()
    items = torch.from_numpy(grouped.index.to_numpy(dtype=np.int64, copy=True)).long()
    labels = torch.from_numpy(grouped.to_numpy(dtype=np.bool_, copy=True)).bool()
    return items, labels


def _eligible_users(plan, update_date: str, evaluation_date: str) -> list[int]:
    update = plan.daily_segments[update_date]
    update_users = set(
        update.loc[
            (update["label"] > 0)
            & (update["item_idx"] >= 1)
            & (update["item_idx"] <= plan.num_prediction_items),
            "user_idx",
        ].astype(int)
    )
    evaluation = plan.daily_segments[evaluation_date]
    first_timestamps = evaluation.groupby("user_idx")["time_ms"].min()
    output = []
    for user in sorted(update_users & set(first_timestamps.index.astype(int))):
        items, labels = _logged_candidates(plan, user, evaluation_date)
        history = plan._build_seq(user, as_of_timestamp=int(first_timestamps.loc[user]))
        if (
            len(items) >= 2
            and bool(labels.any())
            and bool((~labels).any())
            and history is not None
            and len(history["item_ids"]) >= 2
        ):
            output.append(user)
    return output


@torch.no_grad()
def _query_hidden(model, cache, suffix: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    items = torch.from_numpy(suffix["item_ids"][:1]).long().unsqueeze(0).to(device)
    behaviors = torch.from_numpy(suffix["behaviors"][:1]).long().unsqueeze(0).to(device)
    deltas = torch.from_numpy(suffix["time_deltas"][:1]).float().unsqueeze(0).to(device)
    hidden, _ = model.forward_with_cache(cache, items, behaviors, deltas)
    return hidden[0, -1]


def _window_metric_values(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if (
        scores.ndim != 1
        or labels.ndim != 1
        or len(scores) != len(labels)
        or not bool(labels.any())
        or not bool((~labels).any())
    ):
        raise ValueError("KuaiRand logged-window metric input differs")
    values = scores.double()
    positive_values = values[labels]
    negatives = values[~labels]
    window_ce = torch.logsumexp(values, dim=0) - torch.logsumexp(positive_values, dim=0)
    auc = (
        (positive_values[:, None] > negatives[None, :]).double()
        + 0.5 * (positive_values[:, None] == negatives[None, :]).double()
    ).mean()
    order = torch.argsort(values, descending=True, stable=True)
    ordered = labels[order].double()
    ranks = torch.arange(1, len(values) + 1, dtype=torch.float64, device=values.device)
    cumulative = torch.cumsum(ordered, dim=0)
    average_precision = ((cumulative / ranks) * ordered).sum() / labels.sum()
    output = [window_ce, auc, average_precision]
    positives = int(labels.sum().item())
    for cutoff in (10, 50):
        width = min(cutoff, len(values))
        discounts = torch.reciprocal(
            torch.log2(
                torch.arange(2, width + 2, dtype=torch.float64, device=values.device)
            )
        )
        dcg = (ordered[:width] * discounts).sum()
        ideal = discounts[: min(width, positives)].sum()
        output.append(dcg / ideal.clamp_min(1e-12))
    for cutoff in (10, 50):
        width = min(cutoff, len(values))
        output.append(ordered[:width].sum() / positives)
    return torch.stack(output)


@torch.no_grad()
def _score_window(model, hidden, items, labels, device) -> np.ndarray:
    vectors = model.prediction_item_weight[items.to(device)]
    scores = torch.einsum("h,nh->n", hidden, vectors)
    return _window_metric_values(scores, labels.to(device)).cpu().numpy()


def _summarize(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    values = np.stack([np.asarray(value["metrics"]) for value in records])
    fresh = values[:, 0]
    reuse = values[:, 1]
    generator = np.random.default_rng(int(config["bootstrap_seed"]))
    weights = generator.multinomial(
        len(records),
        np.full(len(records), 1.0 / len(records), dtype=np.float64),
        size=int(config["bootstrap_samples"]),
    )
    metrics = {}
    for index, metric in enumerate(METRICS):
        oriented = (
            reuse[:, index] - fresh[:, index]
            if metric in LOWER_IS_BETTER
            else fresh[:, index] - reuse[:, index]
        )
        interval = np.quantile((weights @ oriented) / len(records), [0.025, 0.975])
        fresh_mean = float(fresh[:, index].mean())
        reuse_mean = float(reuse[:, index].mean())
        advantage = float(oriented.mean())
        metrics[metric] = {
            "recompute": fresh_mean,
            "reuse": reuse_mean,
            "recompute_advantage_absolute": advantage,
            "recompute_advantage_relative_percent": (
                100.0 * advantage / abs(reuse_mean) if reuse_mean else None
            ),
            "user_cluster_95_interval": interval.tolist(),
            "positive_with_ci": bool(interval[0] > 0.0),
            "negative_with_ci": bool(interval[1] < 0.0),
        }
    records = sorted(records, key=lambda value: int(value["user_id"]))
    return {
        "users": len(records),
        "candidate_items": {
            "minimum": min(value["candidate_items"] for value in records),
            "median": float(np.median([value["candidate_items"] for value in records])),
            "p95": float(np.percentile([value["candidate_items"] for value in records], 95)),
            "maximum": max(value["candidate_items"] for value in records),
        },
        "positive_items": sum(value["positive_items"] for value in records),
        "negative_items": sum(value["negative_items"] for value in records),
        "selected_user_ids_sha256": hashlib.sha256(
            np.asarray([value["user_id"] for value in records], dtype="<i8").tobytes()
        ).hexdigest(),
        "metrics": metrics,
    }


def _table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "# KuaiRand logged next-window Recompute-over-Reuse screen",
        "",
        "Each user ranks the deduplicated in-catalog impressions from the untouched next natural day using one fixed pre-window query. Positives are engaged impressions and negatives are unengaged impressions.",
        "",
        "| current | cache | age | users | candidates median/p95 | AUC | AP | NDCG@10 | NDCG@50 | window CE |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in cells:
        metrics = cell["metrics"]
        sizes = cell["candidate_items"]
        lines.append(
            "| θ{target} | θ{source} | {age} | {users} | {median:.0f}/{p95:.0f} | {auc:+.3f}% | {ap:+.3f}% | {ndcg10:+.3f}% | {ndcg50:+.3f}% | {ce:+.3f}% |".format(
                target=cell["target_version"],
                source=cell["source_version"],
                age=cell["cache_age"],
                users=cell["users"],
                median=sizes["median"],
                p95=sizes["p95"],
                auc=metrics["roc_auc"]["recompute_advantage_relative_percent"],
                ap=metrics["average_precision"]["recompute_advantage_relative_percent"],
                ndcg10=metrics["ndcg_at_10"]["recompute_advantage_relative_percent"],
                ndcg50=metrics["ndcg_at_50"]["recompute_advantage_relative_percent"],
                ce=metrics["window_cross_entropy"]["recompute_advantage_relative_percent"],
            )
        )
    return "\n".join(lines) + "\n"


def run_logged_window_screen(config_path: str | Path) -> dict[str, Any] | None:
    config = load_logged_window_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    try:
        if runtime.world_size != int(config["execution"]["world_size"]):
            raise ValueError("KuaiRand logged-window screen world size differs")
        candidate_config_holder: list[Any] = [None]
        error_holder: list[str | None] = [None]
        if runtime.is_primary:
            try:
                candidate_config_holder[0] = _verify(config)
            except Exception as error:
                error_holder[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(error_holder, src=0)
        if error_holder[0] is not None:
            raise ValueError(error_holder[0])
        source_path = config["candidate_triangle_config"]["path"]
        candidate_config = load_candidate_triangle_config(source_path)
        output = Path(config["outputs"]["result"])
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        chain = load_kv_only_chain_config(candidate_config["chain_config"]["path"])
        document = _effective_document(chain)
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in (14, 15):
            plan.ingest_day(dates[date_index])
        started = time.perf_counter()
        cells = []
        ingested_through = 15
        for target, source in config["pairs"]:
            required = 13 + int(target)
            for date_index in range(ingested_through + 1, required + 1):
                plan.ingest_day(dates[date_index])
            ingested_through = max(ingested_through, required)
            update_date = dates[13 + int(target)]
            evaluation_date = dates[14 + int(target)]
            eligible = _eligible_users(plan, update_date, evaluation_date)
            local_users = eligible[runtime.rank :: runtime.world_size]
            current = _load_model(
                candidate_config, int(target), document, plan, runtime.device
            )
            source_model = _load_model(
                candidate_config, int(source), document, plan, runtime.device
            )
            distances = _parameter_distances(source_model, current)
            if any(
                distances[name]["relative_l2_update"] != 0.0
                for name in ("item_embedding", "input_encoders", "other_core")
            ):
                raise RuntimeError("KuaiRand logged-window parameter isolation differs")
            records = []
            for ordinal, user in enumerate(local_users):
                sequence = _evaluation_sequence(plan, int(user), evaluation_date)
                items, labels = _logged_candidates(plan, int(user), evaluation_date)
                fresh_cache = _stored_cache(current, sequence["prefix"], runtime.device)
                stale_cache = _stored_cache(source_model, sequence["prefix"], runtime.device)
                fresh_hidden = _query_hidden(
                    current, fresh_cache, sequence["suffix"], runtime.device
                )
                stale_hidden = _query_hidden(
                    current, stale_cache, sequence["suffix"], runtime.device
                )
                records.append(
                    {
                        "user_id": int(user),
                        "candidate_items": len(items),
                        "positive_items": int(labels.sum().item()),
                        "negative_items": int((~labels).sum().item()),
                        "metrics": np.stack(
                            (
                                _score_window(
                                    current,
                                    fresh_hidden,
                                    items,
                                    labels,
                                    runtime.device,
                                ),
                                _score_window(
                                    current,
                                    stale_hidden,
                                    items,
                                    labels,
                                    runtime.device,
                                ),
                            )
                        ).tolist(),
                    }
                )
                if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(local_users):
                    print(
                        f"phase=kuairand_logged_window target={target} source={source} "
                        f"rank={runtime.rank} users={ordinal + 1}/{len(local_users)}",
                        flush=True,
                    )
            gathered: list[Any] | None = (
                [None] * runtime.world_size if runtime.is_primary else None
            )
            dist.gather_object(records, gathered, dst=0)
            if runtime.is_primary:
                combined = [value for shard in gathered for value in shard]
                cell = _summarize(combined, config)
                cell.update(
                    {
                        "target_version": int(target),
                        "source_version": int(source),
                        "cache_age": int(target) - int(source),
                        "update_date": update_date,
                        "evaluation_date": evaluation_date,
                        "eligible_users": len(eligible),
                        "parameter_group_distances": distances,
                    }
                )
                cells.append(cell)
            del current, source_model, records
            gc.collect()
            torch.cuda.empty_cache()
        if not runtime.is_primary:
            dist.barrier()
            return None
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "source_candidate_triangle": config["candidate_triangle_config"],
            "data": metadata,
            "serving_semantics": "one current model; old-version prefix K/V plus the same latest real pre-window item under the current model",
            "candidate_construction": config["candidate_construction"],
            "cells": cells,
            "decision": {
                metric: {
                    "positive_cells": sum(
                        cell["metrics"][metric]["recompute_advantage_relative_percent"]
                        > 0.0
                        for cell in cells
                    ),
                    "positive_with_ci_cells": sum(
                        cell["metrics"][metric]["positive_with_ci"] for cell in cells
                    ),
                    "negative_with_ci_cells": sum(
                        cell["metrics"][metric]["negative_with_ci"] for cell in cells
                    ),
                    "total_cells": len(cells),
                }
                for metric in METRICS
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        table = Path(config["outputs"]["table"])
        table.parent.mkdir(parents=True, exist_ok=True)
        temporary = table.with_suffix(table.suffix + ".tmp")
        temporary.write_text(_table(cells))
        temporary.replace(table)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)
