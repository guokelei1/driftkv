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
    ROLLING_CHAIN_PROTOCOL,
    effective_document,
    load_rolling_chain_config,
    make_rolling_model,
)
from .kuairand_next_item_rollout import (
    _candidate_metric_values,
    _horizon_name,
    _recursive_cache,
    _run_positive_hidden,
    nested_logged_unengaged_candidate_ids,
)
from .kuairand_root_cause import (
    _atomic_json,
    _empty_cache,
    _evaluation_sequence,
    _stored_cache,
    file_sha256,
    load_plan,
)
from .qk_protocol_sweep_runner import (
    METRICS,
    nested_popular_candidate_ids,
    nested_uniform_candidate_ids,
)
from .qk_stream_version import cache_relative_error

ROLLING_SCREEN_PROTOCOL = "evokv_kuairand_next_item_rolling_context_screen_v0"
EXPOSURE_SCREEN_PROTOCOL = "evokv_kuairand_next_item_exposure_objective_screen_v0"
METHODS = ("recursive_reuse", "fresh_recompute", "no_prefix", "theta7_recompute")
HORIZONS = (1, 4, 16, None)
COMPARISONS = {
    "recompute_over_reuse": (0, 1),
    "theta8_over_theta7": (3, 1),
    "full_history_over_no_prefix": (2, 1),
}


def load_rolling_screen_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    quality = document.get("quality", {})
    chain_binding = document.get("chain_config", {})
    chain = load_rolling_chain_config(chain_binding.get("path", ""))
    protocol = document.get("protocol")
    expected_chain_protocol = (
        ROLLING_CHAIN_PROTOCOL
        if protocol == ROLLING_SCREEN_PROTOCOL
        else EXPOSURE_CHAIN_PROTOCOL
    )
    expected_candidates = (
        ["legacy_raw", "legacy_norm_t005", "dense_raw", "dense_norm_t005"]
        if protocol == ROLLING_SCREEN_PROTOCOL
        else ["legacy_exposure_t010", "dense_exposure_t010"]
    )
    if (
        protocol not in (ROLLING_SCREEN_PROTOCOL, EXPOSURE_SCREEN_PROTOCOL)
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or file_sha256(chain_binding.get("path", "")) != chain_binding.get("sha256")
        or chain_binding.get("sha256") != file_sha256(chain_binding.get("path", ""))
        or chain.get("protocol") != expected_chain_protocol
        or [value.get("candidate_id") for value in chain.get("candidates", [])]
        != expected_candidates
        or (
            protocol == EXPOSURE_SCREEN_PROTOCOL
            and document.get("in_catalog_targets_only") is not True
        )
        or document.get("evaluation_date_index") != 22
        or document.get("prefix_tokens") != 192
        or document.get("maximum_exposures") != 64
        or quality.get("negative_count") != 99
        or quality.get("uniform_candidate_seeds") != [61031, 14929, 29063]
        or quality.get("horizons") != [1, 4, 16, "all"]
        or int(quality.get("bootstrap_samples", 0)) < 1000
        or int(quality.get("bootstrap_seed", 0)) < 1
    ):
        raise ValueError("KuaiRand rolling-context screen config differs")
    return document


def _checkpoint_manifest(
    config: dict[str, Any],
    candidate: dict[str, Any],
    version: int,
) -> tuple[Path, dict[str, Any]]:
    root = Path(config["checkpoint_parent"]) / candidate["candidate_id"] / f"theta_{version}"
    model_path = root / "model.pt"
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("protocol") != config["protocol"]
        or manifest.get("version") != version
        or manifest.get("candidate") != candidate
        or manifest.get("config_sha256") != file_sha256(config["config_path"])
        or manifest.get("model_sha256") != file_sha256(model_path)
    ):
        raise ValueError("KuaiRand rolling-context checkpoint binding differs")
    return model_path, manifest


def _eligible_users(
    plan,
    update_date: str,
    eval_date: str,
    maximum_exposures: int,
    in_catalog_targets_only: bool = False,
) -> list[int]:
    update = plan.daily_segments[update_date]
    evaluation = plan.daily_segments[eval_date]
    update_labels = update["label"] > 0
    if in_catalog_targets_only:
        update_labels = update_labels & (update["item_idx"] <= plan.num_prediction_items)
    update_users = set(update.loc[update_labels, "user_idx"].astype(int))
    output = []
    for user, user_day in evaluation.groupby("user_idx", sort=False):
        user = int(user)
        first = user_day.sort_values("time_ms").iloc[:maximum_exposures]
        labels = first["label"] > 0
        if in_catalog_targets_only:
            labels = labels & (first["item_idx"] <= plan.num_prediction_items)
        if user not in update_users or not bool(labels.any()):
            continue
        history = plan._build_seq(user, as_of_timestamp=int(first["time_ms"].min()))
        if history is not None and len(history["item_ids"]) >= 2:
            output.append(user)
    return sorted(output)


def _candidate_sets(
    positives: torch.Tensor,
    day_items: np.ndarray,
    labels: np.ndarray,
    exposure_popular: torch.Tensor,
    engaged_popular: torch.Tensor,
    num_prediction_items: int,
    config: dict[str, Any],
) -> tuple[list[str], list[torch.Tensor]]:
    count = int(config["quality"]["negative_count"])
    seeds = [int(value) for value in config["quality"]["uniform_candidate_seeds"]]
    names = [f"uniform_seed_{seed}" for seed in seeds]
    candidates = [
        nested_uniform_candidate_ids(
            positives,
            num_prediction_items=num_prediction_items,
            maximum_negative_count=count,
            seed=seed,
        )
        for seed in seeds
    ]
    names.extend(
        [
            "base_period_exposure_popular",
            "base_period_engaged_popular",
            "logged_unengaged_nearest",
        ]
    )
    candidates.extend(
        [
            nested_popular_candidate_ids(
                positives,
                exposure_popular,
                maximum_negative_count=count,
            ),
            nested_popular_candidate_ids(
                positives,
                engaged_popular,
                maximum_negative_count=count,
            ),
            nested_logged_unengaged_candidate_ids(
                positives,
                day_items,
                labels,
                exposure_popular,
                num_prediction_items=num_prediction_items,
                maximum_negative_count=count,
                seed=None,
            ),
        ]
    )
    return names, candidates


def _scores(model, hidden: torch.Tensor, candidates: torch.Tensor, candidate: dict[str, Any]):
    vectors = model.item_emb.weight[candidates]
    if candidate["normalize_scores"]:
        hidden = F.normalize(hidden, dim=-1)
        vectors = F.normalize(vectors, dim=-1)
    return torch.einsum("mth,tch->mtc", hidden, vectors) / float(candidate["temperature"])


def _bootstrap_weights(records: int, samples: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.multinomial(
        records,
        np.full(records, 1.0 / records, dtype=np.float64),
        size=samples,
    ).astype(np.float64)


def _summarize(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    sums = np.stack([value["sums"] for value in records])
    targets = np.stack([value["targets"] for value in records]).astype(np.float64)
    weights = _bootstrap_weights(
        len(records),
        int(config["quality"]["bootstrap_samples"]),
        int(config["quality"]["bootstrap_seed"]),
    )
    candidates = records[0]["candidate_names"]
    variants = records[0]["variant_names"]
    output = {}
    for candidate_index, candidate in enumerate(candidates):
        candidate_output = {}
        for variant_index, variant in enumerate(variants):
            horizon_output = {}
            for horizon_index, horizon in enumerate(HORIZONS):
                denominator = float(targets[:, horizon_index].sum())
                bootstrap_denominator = weights @ targets[:, horizon_index]
                absolute = (
                    sums[:, candidate_index, variant_index, horizon_index].sum(axis=0)
                    / denominator
                )
                comparisons = {}
                for comparison, (baseline_index, candidate_method_index) in COMPARISONS.items():
                    baseline = sums[
                        :,
                        candidate_index,
                        variant_index,
                        horizon_index,
                        baseline_index,
                    ]
                    improved = sums[
                        :,
                        candidate_index,
                        variant_index,
                        horizon_index,
                        candidate_method_index,
                    ]
                    oriented = improved - baseline
                    oriented[:, 0] *= -1.0
                    bootstrap = (weights @ oriented) / bootstrap_denominator[:, None]
                    metrics = {}
                    for metric_index, metric in enumerate(METRICS):
                        advantage = float(oriented[:, metric_index].sum() / denominator)
                        interval = np.quantile(bootstrap[:, metric_index], [0.025, 0.975])
                        baseline_value = float(absolute[baseline_index, metric_index])
                        improved_value = float(absolute[candidate_method_index, metric_index])
                        metrics[metric] = {
                            "baseline": baseline_value,
                            "improved": improved_value,
                            "advantage_absolute": advantage,
                            "relative_to_improved_percent": float(
                                100.0 * advantage / max(abs(improved_value), 1e-12)
                            ),
                            "relative_to_baseline_percent": float(
                                100.0 * advantage / max(abs(baseline_value), 1e-12)
                            ),
                            "user_cluster_95_interval": interval.tolist(),
                            "positive_with_ci": bool(interval[0] > 0.0),
                        }
                    comparisons[comparison] = {"metrics": metrics}
                horizon_output[_horizon_name(horizon)] = {
                    "targets": int(denominator),
                    "candidate_count": int(config["quality"]["negative_count"]) + 1,
                    "endpoints": {
                        method: {
                            metric: float(absolute[method_index, metric_index])
                            for metric_index, metric in enumerate(METRICS)
                        }
                        for method_index, method in enumerate(METHODS)
                    },
                    "comparisons": comparisons,
                }
            candidate_output[variant] = {"horizons": horizon_output}
        output[candidate] = {"candidate_variants": candidate_output}
    return output


def _passes(metrics: dict[str, Any]) -> bool:
    ranking = [metrics[name] for name in ("mrr", "ndcg_at_5", "ndcg_at_10")]
    return bool(
        metrics["cross_entropy"]["positive_with_ci"]
        and sum(
            value["positive_with_ci"]
            and value["relative_to_improved_percent"] >= 5.0
            for value in ranking
        )
        >= 2
    )


def _decision(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    seeds = [f"uniform_seed_{value}" for value in config["quality"]["uniform_candidate_seeds"]]
    rows = []
    for candidate, candidate_values in summary.items():
        for horizon in (_horizon_name(value) for value in HORIZONS):
            variants = candidate_values["candidate_variants"]
            stale = {
                variant: _passes(
                    variants[variant]["horizons"][horizon]["comparisons"]
                    ["recompute_over_reuse"]["metrics"]
                )
                for variant in variants
            }
            rows.append(
                {
                    "candidate": candidate,
                    "horizon": horizon,
                    "uniform_seed_robust": all(stale[value] for value in seeds),
                    "variant_passes": stale,
                }
            )
    return {
        "criterion": "CE positive and at least two of MRR/NDCG@5/NDCG@10 >=5% relative to recompute with positive user-cluster CI",
        "uniform_seed_robust_opportunities": [
            value for value in rows if value["uniform_seed_robust"]
        ],
        "diagnostics": rows,
    }


@torch.no_grad()
def run_rolling_screen(config_path: str | Path) -> dict[str, Any] | None:
    config_path = Path(config_path)
    screen = load_rolling_screen_config(config_path)
    chain_path = Path(screen["chain_config"]["path"])
    chain = load_rolling_chain_config(chain_path)
    chain["config_path"] = str(chain_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand rolling-context screen requires two ranks")
    output = Path(screen["evaluation_result"])
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
            candidate_bindings = []
            for version in range(2, 9):
                model_path, manifest = _checkpoint_manifest(chain, candidate, version)
                model = make_rolling_model(chain, candidate, plan, runtime.device)
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                model.eval()
                candidate_models[version] = model
                candidate_bindings.append(
                    {
                        "version": version,
                        "path": str(model_path),
                        "sha256": manifest["model_sha256"],
                    }
                )
            models[candidate["candidate_id"]] = candidate_models
            bindings[candidate["candidate_id"]] = candidate_bindings
        update_date = dates[21]
        eval_date = dates[int(screen["evaluation_date_index"])]
        in_catalog_targets_only = screen["protocol"] == EXPOSURE_SCREEN_PROTOCOL
        eligible = _eligible_users(
            plan,
            update_date,
            eval_date,
            int(screen["maximum_exposures"]),
            in_catalog_targets_only,
        )
        generator = np.random.default_rng(int(screen["quality"]["sampling_seed"]))
        selected = sorted(
            np.asarray(eligible)[generator.permutation(len(eligible))[: int(screen["selected_users"])]].tolist()
        )
        local_users = selected[runtime.rank :: runtime.world_size]
        boundaries = [
            (int(plan.daily_segments[dates[date_index]]["time_ms"].min()), version)
            for version, date_index in zip(range(3, 8), range(17, 22), strict=True)
        ]
        exposure_counts = np.zeros(plan.num_prediction_items + 1, dtype=np.int64)
        engaged_counts = np.zeros(plan.num_prediction_items + 1, dtype=np.int64)
        for date in plan.base_dates:
            frame = plan.daily_segments[date]
            exposed = frame["item_idx"].to_numpy(dtype=np.int64)
            exposed = exposed[(exposed >= 1) & (exposed <= plan.num_prediction_items)]
            engaged = frame.loc[frame["label"] > 0, "item_idx"].to_numpy(dtype=np.int64)
            engaged = engaged[(engaged >= 1) & (engaged <= plan.num_prediction_items)]
            np.add.at(exposure_counts, exposed, 1)
            np.add.at(engaged_counts, engaged, 1)
        ids = np.arange(1, plan.num_prediction_items + 1, dtype=np.int64)
        exposure_popular = torch.from_numpy(
            ids[np.lexsort((ids, -exposure_counts[1:]))].copy()
        )
        engaged_popular = torch.from_numpy(
            ids[np.lexsort((ids, -engaged_counts[1:]))].copy()
        )
        candidate_names = [value["candidate_id"] for value in chain["candidates"]]
        records = []
        digest = hashlib.sha256()
        started = time.perf_counter()
        for ordinal, user in enumerate(local_users):
            sequence = _evaluation_sequence(plan, int(user), eval_date)
            prefix_length = min(len(sequence["prefix"]["item_ids"]), int(screen["prefix_tokens"]))
            sequence["prefix"] = {
                name: values[-prefix_length:]
                for name, values in sequence["prefix"].items()
            }
            limit = int(screen["maximum_exposures"])
            sequence["suffix"] = {
                name: values[:limit] for name, values in sequence["suffix"].items()
            }
            sequence["targets"] = sequence["targets"][:limit]
            sequence["labels"] = sequence["labels"][:limit]
            if in_catalog_targets_only:
                sequence["labels"] = sequence["labels"] & (
                    sequence["targets"] <= plan.num_prediction_items
                )
            user_day = plan.daily_segments[eval_date]
            user_day = user_day[user_day["user_idx"] == user].sort_values("time_ms").iloc[:limit]
            history = plan._build_seq(user, as_of_timestamp=int(user_day["time_ms"].min()))
            timestamps = history["timestamps"][:-1][-prefix_length:].copy()
            positives = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
            variant_names, candidate_sets = _candidate_sets(
                positives,
                sequence["targets"],
                sequence["labels"],
                exposure_popular,
                engaged_popular,
                plan.num_prediction_items,
                screen,
            )
            target_counts = np.asarray(
                [min(len(positives), value) if value is not None else len(positives) for value in HORIZONS],
                dtype=np.int64,
            )
            sums = np.zeros(
                (
                    len(candidate_names),
                    len(candidate_sets),
                    len(HORIZONS),
                    len(METHODS),
                    len(METRICS),
                ),
                dtype=np.float64,
            )
            cache_errors = []
            hidden_errors = []
            for candidate_index, candidate in enumerate(chain["candidates"]):
                chain_models = models[candidate["candidate_id"]]
                current = chain_models[8]
                old = chain_models[7]
                recursive, _ = _recursive_cache(
                    chain_models,
                    sequence["prefix"],
                    timestamps,
                    boundaries,
                    runtime.device,
                )
                fresh = _stored_cache(current, sequence["prefix"], runtime.device)
                old_fresh = _stored_cache(old, sequence["prefix"], runtime.device)
                hidden = torch.stack(
                    (
                        _run_positive_hidden(
                            current,
                            recursive,
                            sequence["suffix"],
                            sequence["labels"],
                            limit,
                            runtime.device,
                        ),
                        _run_positive_hidden(
                            current,
                            fresh,
                            sequence["suffix"],
                            sequence["labels"],
                            limit,
                            runtime.device,
                        ),
                        _run_positive_hidden(
                            current,
                            _empty_cache(fresh),
                            sequence["suffix"],
                            sequence["labels"],
                            limit,
                            runtime.device,
                        ),
                        _run_positive_hidden(
                            old,
                            old_fresh,
                            sequence["suffix"],
                            sequence["labels"],
                            limit,
                            runtime.device,
                        ),
                    )
                ).to(runtime.device)
                cache_errors.append(cache_relative_error(recursive, fresh))
                hidden_errors.append(
                    float(
                        torch.linalg.vector_norm((hidden[0] - hidden[1]).double())
                        / torch.linalg.vector_norm(hidden[1].double()).clamp_min(1e-12)
                    )
                )
                for variant_index, candidate_ids in enumerate(candidate_sets):
                    if candidate_index == 0:
                        digest.update(candidate_ids.numpy().astype("<i8", copy=False).tobytes())
                    scores = _scores(
                        current,
                        hidden,
                        candidate_ids.to(runtime.device),
                        candidate,
                    )
                    if candidate["normalize_scores"]:
                        old_scores = _scores(
                            old,
                            hidden[3:4],
                            candidate_ids.to(runtime.device),
                            candidate,
                        )
                        scores[3:4] = old_scores
                    else:
                        scores[3:4] = _scores(
                            old,
                            hidden[3:4],
                            candidate_ids.to(runtime.device),
                            candidate,
                        )
                    values = _candidate_metric_values(scores)
                    for horizon_index, target_count in enumerate(target_counts):
                        sums[candidate_index, variant_index, horizon_index] = (
                            values[:, :target_count].sum(dim=1).numpy()
                        )
            records.append(
                {
                    "user_id": int(user),
                    "targets": target_counts,
                    "sums": sums,
                    "candidate_names": candidate_names,
                    "variant_names": variant_names,
                    "cache_errors": cache_errors,
                    "hidden_errors": hidden_errors,
                }
            )
            if (ordinal + 1) % 8 == 0 or ordinal + 1 == len(local_users):
                print(
                    f"phase=rolling_context_screen rank={runtime.rank} "
                    f"users={ordinal + 1}/{len(local_users)}",
                    flush=True,
                )
        gathered: list[Any] | None = [None] * runtime.world_size if runtime.is_primary else None
        dist.gather_object(
            {"records": records, "candidate_sha256": digest.hexdigest()},
            gathered,
            dst=0,
        )
        if not runtime.is_primary:
            dist.barrier()
            return None
        combined = sorted(
            [record for shard in gathered for record in shard["records"]],
            key=lambda value: value["user_id"],
        )
        summary = _summarize(combined, screen)
        result = {
            "protocol": screen["protocol"],
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "chain_config": screen["chain_config"],
            "checkpoint_bindings": bindings,
            "data": metadata,
            "evaluation": {
                "update_date": update_date,
                "evaluation_date": eval_date,
                "eligible_users": len(eligible),
                "selected_users": len(combined),
                "prefix_tokens": screen["prefix_tokens"],
                "maximum_exposures": screen["maximum_exposures"],
                "in_catalog_targets_only": in_catalog_targets_only,
                "selected_user_ids_sha256": hashlib.sha256(
                    np.asarray([value["user_id"] for value in combined], dtype="<i8").tobytes()
                ).hexdigest(),
                "candidate_sha256_by_rank": [value["candidate_sha256"] for value in gathered],
                "candidate_models": candidate_names,
                "candidate_variants": combined[0]["variant_names"],
            },
            "mechanism": {
                candidate: {
                    "cache_relative_error_mean": float(
                        np.mean([value["cache_errors"][index] for value in combined])
                    ),
                    "hidden_relative_error_mean": float(
                        np.mean([value["hidden_errors"][index] for value in combined])
                    ),
                }
                for index, candidate in enumerate(candidate_names)
            },
            "candidate_quality": summary,
            "decision": _decision(summary, screen),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)


def validate_rolling_screen_result(result: dict[str, Any]) -> None:
    evaluation = result.get("evaluation", {})
    if (
        result.get("protocol") not in (ROLLING_SCREEN_PROTOCOL, EXPOSURE_SCREEN_PROTOCOL)
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or int(evaluation.get("selected_users", 0)) < 16
        or int(evaluation.get("selected_users", 0)) > int(evaluation.get("eligible_users", 0))
        or len(evaluation.get("candidate_models", []))
        != (4 if result.get("protocol") == ROLLING_SCREEN_PROTOCOL else 2)
        or len(evaluation.get("candidate_variants", [])) != 6
    ):
        raise ValueError("KuaiRand rolling-context screen result differs")
