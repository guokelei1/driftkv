from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_next_item_rolling_chain import (
    effective_document,
    load_rolling_chain_config,
    make_rolling_model,
)
from .kuairand_next_item_rolling_screen import (
    HORIZONS,
    _bootstrap_weights,
    _candidate_sets,
    _checkpoint_manifest,
    _eligible_users,
    _scores,
)
from .kuairand_next_item_rollout import (
    _candidate_metric_values,
    _horizon_name,
    _recursive_cache,
    _run_positive_hidden,
)
from .kuairand_root_cause import (
    _atomic_json,
    _atomic_torch,
    _evaluation_sequence,
    _stored_cache,
    file_sha256,
    load_plan,
)
from .qk_protocol_sweep_runner import METRICS
from .qk_stream_version import cache_relative_error

CACHE_AGE_PROTOCOL = "evokv_kuairand_next_item_cache_age_screen_v0"
METHODS = ("lag1_theta7", "lag2_theta6", "lag4_theta4", "lag6_theta2", "recursive", "fresh")
SOURCE_VERSIONS = (7, 6, 4, 2)


def load_cache_age_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    chain_binding = document.get("chain_config", {})
    quality = document.get("quality", {})
    if (
        document.get("protocol") != CACHE_AGE_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("candidate_ids") != ["legacy_raw", "dense_norm_t005"]
        or document.get("source_versions") != list(SOURCE_VERSIONS)
        or document.get("methods") != list(METHODS)
        or document.get("evaluation_date_index") != 22
        or document.get("prefix_tokens") != 192
        or document.get("maximum_exposures") != 64
        or quality.get("negative_count") != 99
        or quality.get("uniform_candidate_seeds") != [61031, 14929, 29063]
        or quality.get("horizons") != [1, 4, 16, "all"]
        or int(quality.get("bootstrap_samples", 0)) < 1000
        or file_sha256(chain_binding.get("path", "")) != chain_binding.get("sha256")
    ):
        raise ValueError("KuaiRand cache-age screen config differs")
    load_rolling_chain_config(chain_binding["path"])
    return document


def _summarize(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    sums = np.stack([value["sums"] for value in records])
    targets = np.stack([value["targets"] for value in records]).astype(np.float64)
    weights = _bootstrap_weights(
        len(records),
        int(config["quality"]["bootstrap_samples"]),
        int(config["quality"]["bootstrap_seed"]),
    )
    output = {}
    fresh_index = METHODS.index("fresh")
    for candidate_index, candidate in enumerate(records[0]["candidate_names"]):
        variants = {}
        for variant_index, variant in enumerate(records[0]["variant_names"]):
            horizons = {}
            for horizon_index, horizon in enumerate(HORIZONS):
                denominator = float(targets[:, horizon_index].sum())
                bootstrap_denominator = weights @ targets[:, horizon_index]
                methods = {}
                for method_index, method in enumerate(METHODS[:-1]):
                    stale = sums[:, candidate_index, variant_index, horizon_index, method_index]
                    fresh = sums[:, candidate_index, variant_index, horizon_index, fresh_index]
                    oriented = fresh - stale
                    oriented[:, 0] *= -1.0
                    bootstrap = (weights @ oriented) / bootstrap_denominator[:, None]
                    metrics = {}
                    for metric_index, metric in enumerate(METRICS):
                        stale_value = float(stale[:, metric_index].sum() / denominator)
                        fresh_value = float(fresh[:, metric_index].sum() / denominator)
                        advantage = float(oriented[:, metric_index].sum() / denominator)
                        interval = np.quantile(bootstrap[:, metric_index], [0.025, 0.975])
                        metrics[metric] = {
                            "reuse": stale_value,
                            "recompute": fresh_value,
                            "recompute_advantage_absolute": advantage,
                            "relative_to_recompute_percent": float(
                                100.0 * advantage / max(abs(fresh_value), 1e-12)
                            ),
                            "relative_to_reuse_percent": float(
                                100.0 * advantage / max(abs(stale_value), 1e-12)
                            ),
                            "user_cluster_95_interval": interval.tolist(),
                            "positive_with_ci": bool(interval[0] > 0.0),
                        }
                    methods[method] = {"metrics": metrics}
                horizons[_horizon_name(horizon)] = {
                    "targets": int(denominator),
                    "candidate_count": int(config["quality"]["negative_count"]) + 1,
                    "reuse_methods": methods,
                }
            variants[variant] = {"horizons": horizons}
        output[candidate] = {"candidate_variants": variants}
    return output


def _decision(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    seeds = [f"uniform_seed_{value}" for value in config["quality"]["uniform_candidate_seeds"]]
    rows = []
    for candidate, candidate_values in summary.items():
        for method in METHODS[:-1]:
            for horizon in (_horizon_name(value) for value in HORIZONS):
                variants = candidate_values["candidate_variants"]
                passes = {
                    variant: _age_passes(
                        variants[variant]["horizons"][horizon]["reuse_methods"][method][
                            "metrics"
                        ]
                    )
                    for variant in variants
                }
                rows.append(
                    {
                        "candidate": candidate,
                        "reuse_method": method,
                        "horizon": horizon,
                        "uniform_seed_robust": all(passes[value] for value in seeds),
                        "variant_passes": passes,
                    }
                )
    return {
        "criterion": "CE positive and at least two of MRR/NDCG@5/NDCG@10 >=5% relative to recompute with positive user-cluster CI",
        "uniform_seed_robust_opportunities": [
            value for value in rows if value["uniform_seed_robust"]
        ],
        "diagnostics": rows,
    }


def _age_passes(metrics: dict[str, Any]) -> bool:
    ranking = [metrics[name] for name in ("mrr", "ndcg_at_5", "ndcg_at_10")]
    return bool(
        metrics["cross_entropy"]["positive_with_ci"]
        and sum(
            value["positive_with_ci"]
            and value["relative_to_recompute_percent"] >= 5.0
            for value in ranking
        )
        >= 2
    )


@torch.no_grad()
def run_cache_age_screen(config_path: str | Path) -> dict[str, Any] | None:
    config_path = Path(config_path)
    config = load_cache_age_config(config_path)
    chain_path = Path(config["chain_config"]["path"])
    chain = load_rolling_chain_config(chain_path)
    chain["config_path"] = str(chain_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand cache-age screen requires two ranks")
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
        candidate_by_id = {value["candidate_id"]: value for value in chain["candidates"]}
        models = {}
        bindings = {}
        for candidate_id in config["candidate_ids"]:
            candidate = candidate_by_id[candidate_id]
            candidate_models = {}
            candidate_bindings = []
            for version in range(2, 9):
                model_path, manifest = _checkpoint_manifest(chain, candidate, version)
                model = make_rolling_model(chain, candidate, plan, runtime.device)
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                model.eval()
                candidate_models[version] = model
                candidate_bindings.append(
                    {"version": version, "path": str(model_path), "sha256": manifest["model_sha256"]}
                )
            models[candidate_id] = candidate_models
            bindings[candidate_id] = candidate_bindings
        update_date = dates[21]
        eval_date = dates[int(config["evaluation_date_index"])]
        eligible = _eligible_users(plan, update_date, eval_date, int(config["maximum_exposures"]))
        generator = np.random.default_rng(int(config["quality"]["sampling_seed"]))
        selected = sorted(
            np.asarray(eligible)[generator.permutation(len(eligible))[: int(config["selected_users"])]].tolist()
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
        exposure_popular = torch.from_numpy(ids[np.lexsort((ids, -exposure_counts[1:]))].copy())
        engaged_popular = torch.from_numpy(ids[np.lexsort((ids, -engaged_counts[1:]))].copy())
        records = []
        digest = hashlib.sha256()
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
            sequence["targets"] = sequence["targets"][:limit]
            sequence["labels"] = sequence["labels"][:limit]
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
                config,
            )
            target_counts = np.asarray(
                [min(len(positives), value) if value is not None else len(positives) for value in HORIZONS],
                dtype=np.int64,
            )
            sums = np.zeros(
                (
                    len(config["candidate_ids"]),
                    len(candidate_sets),
                    len(HORIZONS),
                    len(METHODS),
                    len(METRICS),
                ),
                dtype=np.float64,
            )
            errors = {}
            for candidate_index, candidate_id in enumerate(config["candidate_ids"]):
                candidate = candidate_by_id[candidate_id]
                candidate_models = models[candidate_id]
                current = candidate_models[8]
                fresh = _stored_cache(current, sequence["prefix"], runtime.device)
                stale_caches = [
                    _stored_cache(candidate_models[version], sequence["prefix"], runtime.device)
                    for version in SOURCE_VERSIONS
                ]
                recursive, _ = _recursive_cache(
                    candidate_models,
                    sequence["prefix"],
                    timestamps,
                    boundaries,
                    runtime.device,
                )
                all_caches = [*stale_caches, recursive, fresh]
                hidden = torch.stack(
                    tuple(
                        _run_positive_hidden(
                            current,
                            cache,
                            sequence["suffix"],
                            sequence["labels"],
                            limit,
                            runtime.device,
                        )
                        for cache in all_caches
                    )
                ).to(runtime.device)
                errors[candidate_id] = {
                    method: {
                        "cache_relative_error": cache_relative_error(cache, fresh),
                        "hidden_relative_error": float(
                            torch.linalg.vector_norm((hidden[index] - hidden[-1]).double())
                            / torch.linalg.vector_norm(hidden[-1].double()).clamp_min(1e-12)
                        ),
                    }
                    for index, (method, cache) in enumerate(zip(METHODS[:-1], all_caches[:-1], strict=True))
                }
                for variant_index, candidate_ids in enumerate(candidate_sets):
                    if candidate_index == 0:
                        digest.update(candidate_ids.numpy().astype("<i8", copy=False).tobytes())
                    scores = _scores(
                        current,
                        hidden,
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
                    "candidate_names": config["candidate_ids"],
                    "variant_names": variant_names,
                    "errors": errors,
                }
            )
            if (ordinal + 1) % 8 == 0 or ordinal + 1 == len(local_users):
                print(
                    f"phase=cache_age_screen rank={runtime.rank} "
                    f"users={ordinal + 1}/{len(local_users)}",
                    flush=True,
                )
        shard_path = output.parent / f"rank_{runtime.rank}_records.pt"
        _atomic_torch(
            shard_path,
            {"records": records, "candidate_sha256": digest.hexdigest()},
        )
        dist.barrier(device_ids=[runtime.local_rank])
        if not runtime.is_primary:
            dist.barrier(device_ids=[runtime.local_rank])
            return None
        gathered = [
            torch.load(
                output.parent / f"rank_{rank}_records.pt",
                map_location="cpu",
                weights_only=False,
            )
            for rank in range(runtime.world_size)
        ]
        combined = sorted(
            [record for shard in gathered for record in shard["records"]],
            key=lambda value: value["user_id"],
        )
        summary = _summarize(combined, config)
        mechanism = {}
        for candidate in config["candidate_ids"]:
            mechanism[candidate] = {}
            for method in METHODS[:-1]:
                mechanism[candidate][method] = {
                    name: float(
                        np.mean([value["errors"][candidate][method][name] for value in combined])
                    )
                    for name in ("cache_relative_error", "hidden_relative_error")
                }
        result = {
            "protocol": CACHE_AGE_PROTOCOL,
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
                "selected_user_ids_sha256": hashlib.sha256(
                    np.asarray([value["user_id"] for value in combined], dtype="<i8").tobytes()
                ).hexdigest(),
                "candidate_sha256_by_rank": [value["candidate_sha256"] for value in gathered],
                "candidate_models": config["candidate_ids"],
                "candidate_variants": combined[0]["variant_names"],
                "methods": list(METHODS),
            },
            "mechanism": mechanism,
            "candidate_quality": summary,
            "decision": _decision(summary, config),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier(device_ids=[runtime.local_rank])
        return result
    finally:
        close_distributed_runtime(runtime)


def summarize_cache_age_shards(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_cache_age_config(config_path)
    output = Path(config["evaluation_result"])
    if output.is_file():
        return json.loads(output.read_text())
    started = time.perf_counter()
    gathered = [
        torch.load(
            output.parent / f"rank_{rank}_records.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(2)
    ]
    combined = sorted(
        [record for shard in gathered for record in shard["records"]],
        key=lambda value: value["user_id"],
    )
    chain_path = Path(config["chain_config"]["path"])
    chain = load_rolling_chain_config(chain_path)
    chain["config_path"] = str(chain_path)
    document = effective_document(chain)
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for date_index in chain["update_date_indices"]:
        plan.ingest_day(dates[int(date_index)])
    update_date = dates[21]
    eval_date = dates[int(config["evaluation_date_index"])]
    eligible = _eligible_users(plan, update_date, eval_date, int(config["maximum_exposures"]))
    generator = np.random.default_rng(int(config["quality"]["sampling_seed"]))
    selected = sorted(
        np.asarray(eligible)[generator.permutation(len(eligible))[: int(config["selected_users"])]].tolist()
    )
    if [value["user_id"] for value in combined] != selected:
        raise ValueError("KuaiRand cache-age shard users differ")
    candidate_by_id = {value["candidate_id"]: value for value in chain["candidates"]}
    bindings = {}
    for candidate_id in config["candidate_ids"]:
        bindings[candidate_id] = []
        for version in range(2, 9):
            model_path, manifest = _checkpoint_manifest(
                chain,
                candidate_by_id[candidate_id],
                version,
            )
            bindings[candidate_id].append(
                {"version": version, "path": str(model_path), "sha256": manifest["model_sha256"]}
            )
    summary = _summarize(combined, config)
    mechanism = {}
    for candidate in config["candidate_ids"]:
        mechanism[candidate] = {}
        for method in METHODS[:-1]:
            mechanism[candidate][method] = {
                name: float(
                    np.mean([value["errors"][candidate][method][name] for value in combined])
                )
                for name in ("cache_relative_error", "hidden_relative_error")
            }
    result = {
        "protocol": CACHE_AGE_PROTOCOL,
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
            "selected_user_ids_sha256": hashlib.sha256(
                np.asarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
            "candidate_sha256_by_rank": [value["candidate_sha256"] for value in gathered],
            "candidate_models": config["candidate_ids"],
            "candidate_variants": combined[0]["variant_names"],
            "methods": list(METHODS),
        },
        "mechanism": mechanism,
        "candidate_quality": summary,
        "decision": _decision(summary, config),
        "elapsed_seconds": time.perf_counter() - started,
        "shard_paths": [
            str(output.parent / f"rank_{rank}_records.pt") for rank in range(2)
        ],
    }
    _atomic_json(output, result)
    return result


def validate_cache_age_result(result: dict[str, Any]) -> None:
    evaluation = result.get("evaluation", {})
    if (
        result.get("protocol") != CACHE_AGE_PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or int(evaluation.get("selected_users", 0)) < 16
        or evaluation.get("methods") != list(METHODS)
        or len(evaluation.get("candidate_models", [])) != 2
        or len(evaluation.get("candidate_variants", [])) != 6
    ):
        raise ValueError("KuaiRand cache-age result differs")
