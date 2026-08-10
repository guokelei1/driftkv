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
from .kuairand_next_item_chain import _effective_document
from .kuairand_next_item_rollout import (
    _candidate_metric_values,
    _horizon_name,
    _recursive_cache,
    _run_positive_hidden,
)
from .kuairand_next_item_update_audit import _candidate_sets
from .kuairand_root_cause import (
    _atomic_json,
    _evaluation_sequence,
    _load_checkpoint,
    _selected_users,
    _stored_cache,
    file_sha256,
    load_plan,
    make_model,
)
from .qk_protocol_sweep_runner import METRICS
from .qk_stream_version import cache_relative_error

HARD_SCREEN_PROTOCOL = "evokv_kuairand_next_item_hard_negative_screen_v0"
LOGGED_SCREEN_PROTOCOL = "evokv_kuairand_next_item_logged_negative_screen_v1"
LONG_SCREEN_PROTOCOL = "evokv_kuairand_next_item_long_context_screen_v0"
METHODS = ("recursive_rollout", "fresh_recompute")
HORIZONS = (4, None)


def load_next_item_hard_screen_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    chain = document.get("chain_bindings", [])
    candidates = document.get("candidate_bindings", [])
    quality = document.get("quality", {})
    hard_candidates = [
        "uniform_high_e3_control",
        "mixed_exposure_high_e3",
        "mixed_engaged_high_e3",
        "mixed_exposure_mid_e3",
        "hard_exposure_high_e3",
    ]
    logged_candidates = [
        "uniform_high_e3_control",
        "mixed_logged_high_e3",
        "logged_heavy_high_e3",
        "mixed_logged_mid_e3",
        "logged_heavy_mid_e3",
    ]
    protocol = document.get("protocol")
    if protocol == LONG_SCREEN_PROTOCOL:
        candidate_chains = document.get("candidate_chain_bindings", [])
        flat_bindings = [
            binding
            for candidate_chain in candidate_chains
            for binding in candidate_chain.get("checkpoint_bindings", [])
        ]
        if (
            document.get("status") != "ready_for_autonomous_execution"
            or document.get("scientific_result") is not False
            or document.get("formal_result") is not False
            or document.get("total_num_days") != 23
            or document.get("update_date_indices") != list(range(16, 22))
            or document.get("evaluation_date_index") != 22
            or document.get("evaluation_history_window_days") != 7
            or [value.get("candidate_id") for value in candidate_chains]
            != ["long7_high_e3", "long7_mid_e3"]
            or any(
                [binding.get("version") for binding in value.get("checkpoint_bindings", [])]
                != list(range(2, 9))
                for value in candidate_chains
            )
            or quality.get("negative_count") != 99
            or quality.get("horizons") != [4, "all"]
            or quality.get("uniform_candidate_seed") != 61031
            or int(quality.get("record_limit_per_rank", 0)) < 1
            or quality.get("cap_user_limit_to_eligible") is not True
            or int(quality.get("bootstrap_samples", 0)) < 1000
            or int(quality.get("bootstrap_seed", 0)) < 1
            or file_sha256(document.get("source_config", {}).get("path", ""))
            != document.get("source_config", {}).get("sha256")
            or any(
                file_sha256(value.get("path", "")) != value.get("sha256")
                for value in flat_bindings
            )
        ):
            raise ValueError("KuaiRand long-context screen config differs")
        return document
    expected_candidates = (
        hard_candidates if protocol == HARD_SCREEN_PROTOCOL else logged_candidates
    )
    if (
        protocol not in (HARD_SCREEN_PROTOCOL, LOGGED_SCREEN_PROTOCOL)
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("total_num_days") != 23
        or document.get("update_date_indices") != list(range(16, 22))
        or document.get("evaluation_date_index") != 22
        or document.get("evaluation_history_window_days") != 7
        or [value.get("version") for value in chain] != list(range(2, 8))
        or [value.get("candidate_id") for value in candidates] != expected_candidates
        or quality.get("negative_count") != 99
        or quality.get("horizons") != [4, "all"]
        or quality.get("uniform_candidate_seed") != 61031
        or int(quality.get("record_limit_per_rank", 0)) < 1
        or quality.get("cap_user_limit_to_eligible") is not True
        or int(quality.get("bootstrap_samples", 0)) < 1000
        or int(quality.get("bootstrap_seed", 0)) < 1
        or file_sha256(document.get("source_config", {}).get("path", ""))
        != document.get("source_config", {}).get("sha256")
        or any(
            file_sha256(value.get("path", "")) != value.get("sha256")
            for value in [*chain, *candidates]
        )
    ):
        raise ValueError("KuaiRand hard-negative screen config differs")
    return document


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
    candidate_names = records[0]["candidate_names"]
    variant_names = records[0]["variant_names"]
    result = {}
    for candidate_index, candidate in enumerate(candidate_names):
        variants = {}
        for variant_index, variant in enumerate(variant_names):
            horizons = {}
            for horizon_index, horizon in enumerate(HORIZONS):
                denominator = float(targets[:, horizon_index].sum())
                bootstrap_denominator = weights @ targets[:, horizon_index]
                absolute = (
                    sums[:, candidate_index, variant_index, horizon_index].sum(axis=0)
                    / denominator
                )
                oriented = (
                    sums[:, candidate_index, variant_index, horizon_index, 1]
                    - sums[:, candidate_index, variant_index, horizon_index, 0]
                )
                oriented[:, 0] *= -1.0
                bootstrap = (weights @ oriented) / bootstrap_denominator[:, None]
                metrics = {}
                for metric_index, metric in enumerate(METRICS):
                    reuse = float(absolute[0, metric_index])
                    recompute = float(absolute[1, metric_index])
                    advantage = float(oriented[:, metric_index].sum() / denominator)
                    interval = np.quantile(bootstrap[:, metric_index], [0.025, 0.975])
                    metrics[metric] = {
                        "reuse": reuse,
                        "recompute": recompute,
                        "recompute_advantage_absolute": advantage,
                        "relative_to_recompute_percent": float(
                            100.0 * advantage / max(abs(recompute), 1e-12)
                        ),
                        "relative_to_reuse_percent": float(
                            100.0 * advantage / max(abs(reuse), 1e-12)
                        ),
                        "user_cluster_95_interval": interval.tolist(),
                        "positive_with_ci": bool(interval[0] > 0.0),
                    }
                horizons[_horizon_name(horizon)] = {
                    "targets": int(denominator),
                    "candidate_count": int(config["quality"]["negative_count"]) + 1,
                    "metrics": metrics,
                }
            variants[variant] = {"horizons": horizons}
        result[candidate] = {"candidate_variants": variants}
    return result


def _decide(summary: dict[str, Any]) -> dict[str, Any]:
    decisions = []
    for candidate, candidate_values in summary.items():
        for variant, variant_values in candidate_values["candidate_variants"].items():
            for horizon, horizon_values in variant_values["horizons"].items():
                metrics = horizon_values["metrics"]
                ranking = [metrics[name] for name in ("mrr", "ndcg_at_5", "ndcg_at_10")]
                pass_count = sum(
                    value["positive_with_ci"]
                    and value["relative_to_recompute_percent"] >= 5.0
                    for value in ranking
                )
                decisions.append(
                    {
                        "candidate": candidate,
                        "variant": variant,
                        "horizon": horizon,
                        "cross_entropy_positive": metrics["cross_entropy"]["positive_with_ci"],
                        "ranking_metrics_over_five_percent": pass_count,
                        "passes": bool(
                            metrics["cross_entropy"]["positive_with_ci"] and pass_count >= 2
                        ),
                    }
                )
    return {
        "criterion": "CE positive and at least two of MRR/NDCG@5/NDCG@10 >=5% with positive user-cluster CI",
        "passes": [value for value in decisions if value["passes"]],
        "diagnostics": decisions,
    }


@torch.no_grad()
def run_next_item_hard_screen(config_path: str | Path) -> dict[str, Any] | None:
    config_path = Path(config_path)
    config = load_next_item_hard_screen_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand hard-negative screen requires two ranks")
    output = Path(config["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = _effective_document(
            {
                **config,
                "source": {"config": config["source_config"]},
                "evaluation_methods": ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"],
                "record_limit_per_rank": config["quality"]["record_limit_per_rank"],
                "cap_user_limit_to_eligible": True,
            }
        )
        document["data"]["history_window_days"] = config["evaluation_history_window_days"]
        torch.set_float32_matmul_precision("high")
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in [14, 15, *config["update_date_indices"]]:
            plan.ingest_day(dates[int(date_index)])
        chain_models = {}
        candidate_chain_models = {}
        candidate_models = {}
        if "candidate_chain_bindings" in config:
            loaded_models = {}
            for candidate_chain in config["candidate_chain_bindings"]:
                models = {}
                for binding in candidate_chain["checkpoint_bindings"]:
                    path = binding["path"]
                    if path not in loaded_models:
                        model = make_model(document, plan, runtime.device)
                        model.load_state_dict(
                            torch.load(path, map_location="cpu", weights_only=True)
                        )
                        model.eval()
                        loaded_models[path] = model
                    models[int(binding["version"])] = loaded_models[path]
                candidate = candidate_chain["candidate_id"]
                candidate_chain_models[candidate] = models
                candidate_models[candidate] = models[8]
        else:
            for binding in config["chain_bindings"]:
                version = int(binding["version"])
                model = make_model(document, plan, runtime.device)
                checkpoint = Path(binding["path"])
                _load_checkpoint(model, checkpoint.parents[1], version)
                model.eval()
                chain_models[version] = model
            for binding in config["candidate_bindings"]:
                model = make_model(document, plan, runtime.device)
                model.load_state_dict(
                    torch.load(binding["path"], map_location="cpu", weights_only=True)
                )
                model.eval()
                candidate_models[binding["candidate_id"]] = model
        update_date = dates[int(config["update_date_indices"][-1])]
        eval_date = dates[int(config["evaluation_date_index"])]
        selected, eligible = _selected_users(
            plan,
            update_date,
            eval_date,
            int(config["quality"]["record_limit_per_rank"]) * runtime.world_size,
            int(document["quality"]["sampling_seed"]) + 8 * 1009,
            True,
        )
        local_users = selected[runtime.rank :: runtime.world_size]
        boundaries = [
            (
                int(plan.daily_segments[dates[int(date_index)]]["time_ms"].min()),
                version,
            )
            for version, date_index in zip(
                range(3, 8), config["update_date_indices"][1:], strict=True
            )
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
        candidate_names = list(candidate_models)
        records = []
        digest = hashlib.sha256()
        started = time.perf_counter()
        for ordinal, user in enumerate(local_users):
            sequence = _evaluation_sequence(plan, user, eval_date)
            user_day = plan.daily_segments[eval_date]
            user_day = user_day[user_day["user_idx"] == user].sort_values("time_ms")
            history = plan._build_seq(user, as_of_timestamp=int(user_day["time_ms"].min()))
            timestamps = history["timestamps"][:-1].copy()
            shared_recursive = None
            if chain_models:
                shared_recursive, _ = _recursive_cache(
                    chain_models,
                    sequence["prefix"],
                    timestamps,
                    boundaries,
                    runtime.device,
                )
            positives = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
            variant_names, candidate_sets = _candidate_sets(
                positives,
                sequence["targets"],
                sequence["labels"],
                exposure_popular,
                engaged_popular,
                next(iter(candidate_models.values())).cfg.num_prediction_items,
                config,
            )
            target_counts = np.asarray(
                [
                    min(len(positives), value)
                    if value is not None
                    else len(positives)
                    for value in HORIZONS
                ],
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
            for candidate_index, candidate in enumerate(candidate_names):
                model = candidate_models[candidate]
                if candidate_chain_models:
                    recursive, _ = _recursive_cache(
                        candidate_chain_models[candidate],
                        sequence["prefix"],
                        timestamps,
                        boundaries,
                        runtime.device,
                    )
                else:
                    recursive = shared_recursive
                if recursive is None:
                    raise RuntimeError("KuaiRand recursive cache is missing")
                fresh = _stored_cache(model, sequence["prefix"], runtime.device)
                reuse_hidden = _run_positive_hidden(
                    model,
                    recursive,
                    sequence["suffix"],
                    sequence["labels"],
                    int(document["quality"]["suffix_chunk"]),
                    runtime.device,
                )
                fresh_hidden = _run_positive_hidden(
                    model,
                    fresh,
                    sequence["suffix"],
                    sequence["labels"],
                    int(document["quality"]["suffix_chunk"]),
                    runtime.device,
                )
                hidden = torch.stack((reuse_hidden, fresh_hidden)).to(runtime.device)
                cache_errors.append(cache_relative_error(recursive, fresh))
                hidden_errors.append(
                    float(
                        torch.linalg.vector_norm((reuse_hidden - fresh_hidden).double())
                        / torch.linalg.vector_norm(fresh_hidden.double()).clamp_min(1e-12)
                    )
                )
                for variant, candidates in enumerate(candidate_sets):
                    if candidate_index == 0:
                        digest.update(candidates.numpy().astype("<i8", copy=False).tobytes())
                    vectors = model.item_emb.weight[candidates.to(runtime.device)]
                    scores = torch.einsum("mth,tch->mtc", hidden, vectors)
                    values = _candidate_metric_values(scores)
                    for horizon_index, target_count in enumerate(target_counts):
                        sums[candidate_index, variant, horizon_index] = (
                            values[:, :target_count].sum(dim=1).numpy()
                        )
            records.append(
                {
                    "user_id": user,
                    "targets": target_counts,
                    "sums": sums,
                    "candidate_names": candidate_names,
                    "variant_names": variant_names,
                    "cache_errors": cache_errors,
                    "hidden_errors": hidden_errors,
                }
            )
            if (ordinal + 1) % 32 == 0 or ordinal + 1 == len(local_users):
                print(
                    f"phase=kuairand_hard_screen rank={runtime.rank} "
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
            key=lambda value: int(value["user_id"]),
        )
        summary = _summarize(combined, config)
        result = {
            "protocol": config["protocol"],
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data": metadata,
            "evaluation": {
                "update_date": update_date,
                "evaluation_date": eval_date,
                "eligible_users": eligible,
                "selected_users": len(combined),
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
            "decision": _decide(summary),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)


def validate_next_item_hard_screen_result(result: dict[str, Any]) -> None:
    evaluation = result.get("evaluation", {})
    if (
        result.get("protocol")
        not in (HARD_SCREEN_PROTOCOL, LOGGED_SCREEN_PROTOCOL, LONG_SCREEN_PROTOCOL)
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or int(evaluation.get("selected_users", 0)) != 400
        or int(evaluation.get("selected_users", 0)) > int(evaluation.get("eligible_users", 0))
        or len(evaluation.get("candidate_models", []))
        != (2 if result.get("protocol") == LONG_SCREEN_PROTOCOL else 5)
        or len(evaluation.get("candidate_variants", [])) != 5
    ):
        raise ValueError("KuaiRand hard-negative screen result differs")
