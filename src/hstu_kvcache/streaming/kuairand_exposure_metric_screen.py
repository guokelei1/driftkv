from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_next_item_rolling_chain import (
    EXPOSURE_CHAIN_PROTOCOL,
    effective_document,
    load_rolling_chain_config,
    make_rolling_model,
)
from .kuairand_next_item_rolling_screen import _checkpoint_manifest
from .kuairand_next_item_rollout import _recursive_cache
from .kuairand_root_cause import (
    _atomic_json,
    _empty_cache,
    _evaluation_sequence,
    _stored_cache,
    file_sha256,
    load_plan,
)
from .qk_stream_version import cache_relative_error

EXPOSURE_METRIC_PROTOCOL = "evokv_kuairand_exposure_metric_screen_v0"
METHODS = ("recursive_reuse", "fresh_recompute", "no_prefix", "theta7_recompute")
METRICS = ("log_loss", "brier", "roc_auc", "average_precision", "ndcg_at_10", "ndcg_at_50")
COMPARISONS = {
    "recompute_over_reuse": (0, 1),
    "theta8_over_theta7": (3, 1),
    "full_history_over_no_prefix": (2, 1),
}


def load_exposure_metric_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    chain_binding = document.get("chain_config", {})
    chain = load_rolling_chain_config(chain_binding.get("path", ""))
    if (
        document.get("protocol") != EXPOSURE_METRIC_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or chain.get("protocol") != EXPOSURE_CHAIN_PROTOCOL
        or [value.get("candidate_id") for value in chain.get("candidates", [])]
        != ["legacy_exposure_t010", "dense_exposure_t010"]
        or file_sha256(chain_binding.get("path", "")) != chain_binding.get("sha256")
        or document.get("evaluation_date_index") != 22
        or document.get("prefix_tokens") != 192
        or document.get("maximum_exposures") != 64
        or document.get("in_catalog_targets_only") is not True
        or int(document.get("bootstrap_samples", 0)) < 1000
        or int(document.get("bootstrap_seed", 0)) < 1
    ):
        raise ValueError("KuaiRand exposure-metric screen config differs")
    return document


def _eligible_users(plan, update_date: str, eval_date: str, maximum_exposures: int) -> list[int]:
    update = plan.daily_segments[update_date]
    update_labels = (update["label"] > 0) & (update["item_idx"] <= plan.num_prediction_items)
    update_users = set(update.loc[update_labels, "user_idx"].astype(int))
    output = []
    for user, user_day in plan.daily_segments[eval_date].groupby("user_idx", sort=False):
        user = int(user)
        first = user_day.sort_values("time_ms").iloc[:maximum_exposures]
        valid = first["item_idx"] <= plan.num_prediction_items
        labels = first.loc[valid, "label"].to_numpy(dtype=np.bool_)
        if (
            user in update_users
            and bool(labels.any())
            and bool((~labels).any())
            and plan._build_seq(user, as_of_timestamp=int(first["time_ms"].min())) is not None
        ):
            output.append(user)
    return sorted(output)


def _run_hidden(model, cache, suffix: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    items = torch.from_numpy(suffix["item_ids"]).long().unsqueeze(0).to(device)
    behaviors = torch.from_numpy(suffix["behaviors"]).long().unsqueeze(0).to(device)
    deltas = torch.from_numpy(suffix["time_deltas"]).float().unsqueeze(0).to(device)
    hidden, _ = model.forward_with_cache(cache, items, behaviors, deltas)
    return hidden[0]


def _logits(model, hidden: torch.Tensor, targets: torch.Tensor, candidate: dict[str, Any]):
    vectors = model.item_emb.weight[targets]
    if candidate["normalize_scores"]:
        hidden = F.normalize(hidden, dim=-1)
        vectors = F.normalize(vectors, dim=-1)
    return torch.einsum("mth,th->mt", hidden, vectors) / float(candidate["temperature"])


def _metric_values(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    values = []
    labels_float = labels.to(logits.dtype)
    for method_logits in logits:
        probabilities = torch.sigmoid(method_logits)
        log_loss = F.binary_cross_entropy_with_logits(method_logits, labels_float).item()
        brier = torch.mean((probabilities - labels_float) ** 2).item()
        positives = method_logits[labels]
        negatives = method_logits[~labels]
        auc = (
            (positives[:, None] > negatives[None, :]).double()
            + 0.5 * (positives[:, None] == negatives[None, :]).double()
        ).mean().item()
        order = torch.argsort(method_logits, descending=True)
        ordered_labels = labels_float[order]
        cumulative = torch.cumsum(ordered_labels, dim=0)
        ranks = torch.arange(1, len(labels) + 1, device=logits.device, dtype=logits.dtype)
        average_precision = ((cumulative / ranks) * ordered_labels).sum() / labels_float.sum()
        ndcgs = []
        for cutoff in (10, 50):
            width = min(cutoff, len(labels))
            discounts = torch.reciprocal(
                torch.log2(torch.arange(2, width + 2, device=logits.device, dtype=logits.dtype))
            )
            dcg = torch.sum(ordered_labels[:width] * discounts)
            ideal_width = min(width, int(labels.sum().item()))
            idcg = discounts[:ideal_width].sum()
            ndcgs.append((dcg / idcg.clamp_min(1e-12)).item())
        values.append([log_loss, brier, auc, average_precision.item(), *ndcgs])
    return np.asarray(values, dtype=np.float64)


def _summarize(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    values = np.stack([record["metrics"] for record in records])
    generator = np.random.default_rng(int(config["bootstrap_seed"]))
    weights = generator.multinomial(
        len(records),
        np.full(len(records), 1.0 / len(records)),
        size=int(config["bootstrap_samples"]),
    ).astype(np.float64)
    output = {}
    for candidate_index, candidate in enumerate(records[0]["candidate_names"]):
        absolute = values[:, candidate_index].mean(axis=0)
        comparisons = {}
        for comparison, (baseline_index, improved_index) in COMPARISONS.items():
            oriented = values[:, candidate_index, improved_index] - values[:, candidate_index, baseline_index]
            oriented[:, :2] *= -1.0
            bootstrap = weights @ oriented / len(records)
            metrics = {}
            for metric_index, metric in enumerate(METRICS):
                advantage = float(oriented[:, metric_index].mean())
                interval = np.quantile(bootstrap[:, metric_index], [0.025, 0.975])
                baseline = float(absolute[baseline_index, metric_index])
                improved = float(absolute[improved_index, metric_index])
                metrics[metric] = {
                    "baseline": baseline,
                    "improved": improved,
                    "advantage_absolute": advantage,
                    "relative_to_improved_percent": float(
                        100.0 * advantage / max(abs(improved), 1e-12)
                    ),
                    "relative_to_baseline_percent": float(
                        100.0 * advantage / max(abs(baseline), 1e-12)
                    ),
                    "user_cluster_95_interval": interval.tolist(),
                    "positive_with_ci": bool(interval[0] > 0.0),
                }
            comparisons[comparison] = {"metrics": metrics}
        output[candidate] = {
            "endpoints": {
                method: {
                    metric: float(absolute[method_index, metric_index])
                    for metric_index, metric in enumerate(METRICS)
                }
                for method_index, method in enumerate(METHODS)
            },
            "comparisons": comparisons,
        }
    return output


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for candidate, candidate_values in summary.items():
        metrics = candidate_values["comparisons"]["recompute_over_reuse"]["metrics"]
        ranking = [metrics[name] for name in ("roc_auc", "average_precision", "ndcg_at_10")]
        rows.append(
            {
                "candidate": candidate,
                "log_loss_positive": metrics["log_loss"]["positive_with_ci"],
                "ranking_metrics_over_five_percent": sum(
                    value["positive_with_ci"]
                    and value["relative_to_improved_percent"] >= 5.0
                    for value in ranking
                ),
            }
        )
    for value in rows:
        value["passes"] = bool(
            value["log_loss_positive"] and value["ranking_metrics_over_five_percent"] >= 2
        )
    return {
        "criterion": "log-loss positive and at least two of AUC/AP/NDCG@10 >=5% relative to recompute with positive user-cluster CI",
        "passes": [value for value in rows if value["passes"]],
        "diagnostics": rows,
    }


@torch.no_grad()
def run_exposure_metric_screen(config_path: str | Path) -> dict[str, Any] | None:
    config_path = Path(config_path)
    config = load_exposure_metric_config(config_path)
    chain_path = Path(config["chain_config"]["path"])
    chain = load_rolling_chain_config(chain_path)
    chain["config_path"] = str(chain_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand exposure-metric screen requires two ranks")
    output = Path(config["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = effective_document(chain)
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in chain["update_date_indices"]:
            plan.ingest_day(dates[int(date_index)])
        models = {}
        bindings = {}
        for candidate in chain["candidates"]:
            candidate_models = {}
            bindings[candidate["candidate_id"]] = []
            for version in range(2, 9):
                model_path, manifest = _checkpoint_manifest(chain, candidate, version)
                model = make_rolling_model(chain, candidate, plan, runtime.device)
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                model.eval()
                candidate_models[version] = model
                bindings[candidate["candidate_id"]].append(
                    {"version": version, "path": str(model_path), "sha256": manifest["model_sha256"]}
                )
            models[candidate["candidate_id"]] = candidate_models
        update_date = dates[21]
        eval_date = dates[int(config["evaluation_date_index"])]
        eligible = _eligible_users(plan, update_date, eval_date, int(config["maximum_exposures"]))
        generator = np.random.default_rng(int(config["sampling_seed"]))
        selected = sorted(
            np.asarray(eligible)[generator.permutation(len(eligible))[: int(config["selected_users"])]].tolist()
        )
        local_users = selected[runtime.rank :: runtime.world_size]
        boundaries = [
            (int(plan.daily_segments[dates[index]]["time_ms"].min()), version)
            for version, index in zip(range(3, 8), range(17, 22), strict=True)
        ]
        candidate_names = [value["candidate_id"] for value in chain["candidates"]]
        records = []
        started = time.perf_counter()
        for ordinal, user in enumerate(local_users):
            sequence = _evaluation_sequence(plan, int(user), eval_date)
            prefix_length = min(len(sequence["prefix"]["item_ids"]), int(config["prefix_tokens"]))
            sequence["prefix"] = {
                name: values[-prefix_length:] for name, values in sequence["prefix"].items()
            }
            limit = int(config["maximum_exposures"])
            sequence["suffix"] = {
                name: values[:limit] for name, values in sequence["suffix"].items()
            }
            targets = sequence["targets"][:limit]
            labels = sequence["labels"][:limit]
            valid = targets <= plan.num_prediction_items
            valid_tensor = torch.from_numpy(valid).bool().to(runtime.device)
            targets_tensor = torch.from_numpy(targets[valid]).long().to(runtime.device)
            labels_tensor = torch.from_numpy(labels[valid]).bool().to(runtime.device)
            user_day = plan.daily_segments[eval_date]
            user_day = user_day[user_day["user_idx"] == user].sort_values("time_ms").iloc[:limit]
            history = plan._build_seq(user, as_of_timestamp=int(user_day["time_ms"].min()))
            timestamps = history["timestamps"][:-1][-prefix_length:].copy()
            candidate_metrics = []
            candidate_errors = []
            for candidate in chain["candidates"]:
                candidate_models = models[candidate["candidate_id"]]
                current = candidate_models[8]
                old = candidate_models[7]
                recursive, _ = _recursive_cache(
                    candidate_models,
                    sequence["prefix"],
                    timestamps,
                    boundaries,
                    runtime.device,
                )
                fresh = _stored_cache(current, sequence["prefix"], runtime.device)
                old_fresh = _stored_cache(old, sequence["prefix"], runtime.device)
                hidden = torch.stack(
                    (
                        _run_hidden(current, recursive, sequence["suffix"], runtime.device)[
                            valid_tensor
                        ],
                        _run_hidden(current, fresh, sequence["suffix"], runtime.device)[valid_tensor],
                        _run_hidden(current, _empty_cache(fresh), sequence["suffix"], runtime.device)[
                            valid_tensor
                        ],
                        _run_hidden(old, old_fresh, sequence["suffix"], runtime.device)[
                            valid_tensor
                        ],
                    )
                )
                scores = _logits(current, hidden, targets_tensor, candidate)
                scores[3:4] = _logits(old, hidden[3:4], targets_tensor, candidate)
                candidate_metrics.append(_metric_values(scores, labels_tensor))
                candidate_errors.append(
                    {
                        "cache_relative_error": cache_relative_error(recursive, fresh),
                        "hidden_relative_error": float(
                            torch.linalg.vector_norm((hidden[0] - hidden[1]).double())
                            / torch.linalg.vector_norm(hidden[1].double()).clamp_min(1e-12)
                        ),
                    }
                )
            records.append(
                {
                    "user_id": int(user),
                    "valid_exposures": int(valid.sum()),
                    "positives": int(labels_tensor.sum().item()),
                    "metrics": np.stack(candidate_metrics),
                    "errors": candidate_errors,
                    "candidate_names": candidate_names,
                }
            )
            if (ordinal + 1) % 8 == 0 or ordinal + 1 == len(local_users):
                print(
                    f"phase=exposure_metric_screen rank={runtime.rank} "
                    f"users={ordinal + 1}/{len(local_users)}",
                    flush=True,
                )
        gathered: list[Any] | None = [None] * runtime.world_size if runtime.is_primary else None
        dist.gather_object(records, gathered, dst=0)
        if not runtime.is_primary:
            return None
        combined = sorted(
            [record for shard in gathered for record in shard],
            key=lambda value: value["user_id"],
        )
        summary = _summarize(combined, config)
        result = {
            "protocol": EXPOSURE_METRIC_PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "chain_config": config["chain_config"],
            "checkpoint_bindings": bindings,
            "data": metadata,
            "evaluation": {
                "update_date": update_date,
                "evaluation_date": eval_date,
                "eligible_users": len(eligible),
                "selected_users": len(combined),
                "valid_exposures": sum(value["valid_exposures"] for value in combined),
                "engaged_exposures": sum(value["positives"] for value in combined),
                "selected_user_ids_sha256": hashlib.sha256(
                    np.asarray([value["user_id"] for value in combined], dtype="<i8").tobytes()
                ).hexdigest(),
                "candidate_models": candidate_names,
                "methods": list(METHODS),
            },
            "mechanism": {
                candidate: {
                    name: float(
                        np.mean([value["errors"][index][name] for value in combined])
                    )
                    for name in ("cache_relative_error", "hidden_relative_error")
                }
                for index, candidate in enumerate(candidate_names)
            },
            "candidate_quality": summary,
            "decision": _decision(summary),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        return result
    finally:
        close_distributed_runtime(runtime)


def validate_exposure_metric_result(result: dict[str, Any]) -> None:
    evaluation = result.get("evaluation", {})
    if (
        result.get("protocol") != EXPOSURE_METRIC_PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or int(evaluation.get("selected_users", 0)) < 16
        or int(evaluation.get("valid_exposures", 0)) < 128
        or int(evaluation.get("engaged_exposures", 0)) < 16
        or len(evaluation.get("candidate_models", [])) != 2
        or evaluation.get("methods") != list(METHODS)
    ):
        raise ValueError("KuaiRand exposure-metric result differs")
