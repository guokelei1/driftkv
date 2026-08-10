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
    _run_positive_hidden,
    nested_logged_unengaged_candidate_ids,
)
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
from .qk_protocol_sweep_runner import (
    METRICS,
    nested_popular_candidate_ids,
    nested_uniform_candidate_ids,
)

AUDIT_PROTOCOL = "evokv_kuairand_next_item_update_endpoint_audit_v0"
METHODS = ("theta7_full_recompute", "theta8_full_recompute")
HORIZONS = (4, None)


def load_next_item_update_audit_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    bindings = document.get("checkpoint_bindings", [])
    quality = document.get("quality", {})
    if (
        document.get("protocol") != AUDIT_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("total_num_days") != 23
        or document.get("ingested_date_indices") != list(range(14, 22))
        or document.get("update_date_index") != 21
        or document.get("evaluation_date_index") != 22
        or document.get("evaluation_history_window_days") != 7
        or quality.get("negative_count") != 99
        or quality.get("horizons") != [4, "all"]
        or int(quality.get("record_limit_per_rank", 0)) < 1
        or quality.get("cap_user_limit_to_eligible") is not True
        or int(quality.get("bootstrap_samples", 0)) < 1000
        or int(quality.get("bootstrap_seed", 0)) < 1
        or quality.get("uniform_candidate_seed") != 61031
        or [value.get("version") for value in bindings] != [7, 8]
        or file_sha256(document.get("source_config", {}).get("path", ""))
        != document.get("source_config", {}).get("sha256")
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand update endpoint audit config differs")
    return document


def _candidate_sets(
    positives: torch.Tensor,
    day_items: np.ndarray,
    labels: np.ndarray,
    exposure_popular: torch.Tensor,
    engaged_popular: torch.Tensor,
    num_prediction_items: int,
    config: dict[str, Any],
) -> tuple[list[str], list[torch.Tensor]]:
    quality = config["quality"]
    count = int(quality["negative_count"])
    seed = int(quality["uniform_candidate_seed"])
    names = [
        f"uniform_seed_{seed}",
        "base_period_exposure_popular",
        "base_period_engaged_popular",
        "logged_unengaged_nearest",
        f"logged_unengaged_seed_{seed}",
    ]
    candidates = [
        nested_uniform_candidate_ids(
            positives,
            num_prediction_items=num_prediction_items,
            maximum_negative_count=count,
            seed=seed,
        ),
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
        nested_logged_unengaged_candidate_ids(
            positives,
            day_items,
            labels,
            exposure_popular,
            num_prediction_items=num_prediction_items,
            maximum_negative_count=count,
            seed=seed,
        ),
    ]
    return names, candidates


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
    names = records[0]["variant_names"]
    result = {}
    for variant_index, variant in enumerate(names):
        horizons = {}
        for horizon_index, horizon in enumerate(HORIZONS):
            denominator = float(targets[:, horizon_index].sum())
            bootstrap_denominator = weights @ targets[:, horizon_index]
            absolute = sums[:, variant_index, horizon_index].sum(axis=0) / denominator
            oriented = (
                sums[:, variant_index, horizon_index, 1]
                - sums[:, variant_index, horizon_index, 0]
            )
            oriented[:, 0] *= -1.0
            bootstrap = (weights @ oriented) / bootstrap_denominator[:, None]
            metrics = {}
            for metric_index, metric in enumerate(METRICS):
                old = float(absolute[0, metric_index])
                new = float(absolute[1, metric_index])
                advantage = float(oriented[:, metric_index].sum() / denominator)
                interval = np.quantile(bootstrap[:, metric_index], [0.025, 0.975])
                metrics[metric] = {
                    "theta7": old,
                    "theta8": new,
                    "theta8_advantage_absolute": advantage,
                    "relative_to_theta7_percent": float(
                        100.0 * advantage / max(abs(old), 1e-12)
                    ),
                    "user_cluster_95_interval": interval.tolist(),
                    "positive_with_ci": bool(interval[0] > 0.0),
                }
            horizons[_horizon_name(horizon)] = {
                "targets": int(denominator),
                "candidate_count": int(config["quality"]["negative_count"]) + 1,
                "metrics": metrics,
            }
        result[variant] = {"horizons": horizons}
    return result


@torch.no_grad()
def run_next_item_update_audit(config_path: str | Path) -> dict[str, Any] | None:
    config_path = Path(config_path)
    config = load_next_item_update_audit_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand update endpoint audit requires two ranks")
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
        for date_index in config["ingested_date_indices"]:
            plan.ingest_day(dates[int(date_index)])
        models = {}
        for binding in config["checkpoint_bindings"]:
            version = int(binding["version"])
            model = make_model(document, plan, runtime.device)
            checkpoint = Path(binding["path"])
            _load_checkpoint(model, checkpoint.parents[1], version)
            model.eval()
            models[version] = model
        update_date = dates[int(config["update_date_index"])]
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
        records = []
        digest = hashlib.sha256()
        started = time.perf_counter()
        for ordinal, user in enumerate(local_users):
            sequence = _evaluation_sequence(plan, user, eval_date)
            positives = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
            hidden = []
            for version in (7, 8):
                cache = _stored_cache(models[version], sequence["prefix"], runtime.device)
                hidden.append(
                    _run_positive_hidden(
                        models[version],
                        cache,
                        sequence["suffix"],
                        sequence["labels"],
                        int(document["quality"]["suffix_chunk"]),
                        runtime.device,
                    )
                )
            hidden_by_method = torch.stack(hidden).to(runtime.device)
            variant_names, candidate_sets = _candidate_sets(
                positives,
                sequence["targets"],
                sequence["labels"],
                exposure_popular,
                engaged_popular,
                models[8].cfg.num_prediction_items,
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
                (len(candidate_sets), len(HORIZONS), len(METHODS), len(METRICS)),
                dtype=np.float64,
            )
            for variant, candidates in enumerate(candidate_sets):
                digest.update(candidates.numpy().astype("<i8", copy=False).tobytes())
                scores = []
                for method_index, version in enumerate((7, 8)):
                    vectors = models[version].item_emb.weight[
                        candidates.to(runtime.device)
                    ]
                    scores.append(
                        torch.einsum(
                            "th,tch->tc", hidden_by_method[method_index], vectors
                        )
                    )
                values = _candidate_metric_values(torch.stack(scores))
                for horizon_index, target_count in enumerate(target_counts):
                    sums[variant, horizon_index] = values[:, :target_count].sum(dim=1).numpy()
            records.append(
                {
                    "user_id": user,
                    "targets": target_counts,
                    "sums": sums,
                    "variant_names": variant_names,
                }
            )
            if (ordinal + 1) % 32 == 0 or ordinal + 1 == len(local_users):
                print(
                    f"phase=kuairand_update_audit rank={runtime.rank} "
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
        result = {
            "protocol": AUDIT_PROTOCOL,
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
                "methods": list(METHODS),
            },
            "candidate_quality": _summarize(combined, config),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)


def validate_next_item_update_audit_result(result: dict[str, Any]) -> None:
    evaluation = result.get("evaluation", {})
    if (
        result.get("protocol") != AUDIT_PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or int(evaluation.get("selected_users", 0)) < 300
        or int(evaluation.get("selected_users", 0)) > int(evaluation.get("eligible_users", 0))
        or evaluation.get("methods") != list(METHODS)
    ):
        raise ValueError("KuaiRand update endpoint audit result differs")
