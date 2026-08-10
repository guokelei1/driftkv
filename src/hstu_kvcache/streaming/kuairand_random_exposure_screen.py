from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_exposure_metric_screen import (
    METHODS,
    _decision,
    _logits,
    _metric_values,
    _summarize,
)
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
    _stored_cache,
    file_sha256,
    load_plan,
)
from .qk_stream_version import cache_relative_error

RANDOM_EXPOSURE_PROTOCOL = "evokv_kuairand_random_exposure_screen_v0"


def load_random_exposure_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    chain_binding = document.get("chain_config", {})
    random_binding = document.get("random_log", {})
    chain = load_rolling_chain_config(chain_binding.get("path", ""))
    if (
        document.get("protocol") != RANDOM_EXPOSURE_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or chain.get("protocol") != EXPOSURE_CHAIN_PROTOCOL
        or [value.get("candidate_id") for value in chain.get("candidates", [])]
        != ["legacy_exposure_t010", "dense_exposure_t010"]
        or file_sha256(chain_binding.get("path", "")) != chain_binding.get("sha256")
        or file_sha256(random_binding.get("path", "")) != random_binding.get("sha256")
        or document.get("evaluation_date") != "20220430"
        or document.get("evaluation_max_users") != 1000
        or document.get("prefix_tokens") != 192
        or document.get("random_log_excluded_from_training") is not True
        or document.get("in_catalog_targets_only") is not True
        or int(document.get("bootstrap_samples", 0)) < 1000
        or int(document.get("bootstrap_seed", 0)) < 1
    ):
        raise ValueError("KuaiRand random-exposure screen config differs")
    return document


def _random_frame(config: dict[str, Any], plan) -> pd.DataFrame:
    frame = pd.read_csv(config["random_log"]["path"])
    frame = frame[frame["date"].astype(str) == config["evaluation_date"]].copy()
    frame["user_idx"] = frame["user_id"].map(plan.trace.user_map)
    frame["item_idx"] = frame["video_id"].map(plan.trace.item_map)
    frame = frame[frame["user_idx"].notna() & frame["item_idx"].notna()].copy()
    frame["user_idx"] = frame["user_idx"].astype(np.int64)
    frame["item_idx"] = frame["item_idx"].astype(np.int64)
    frame["label"] = frame[
        ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view"]
    ].astype(bool).any(axis=1)
    return frame.sort_values(["user_idx", "time_ms"]).reset_index(drop=True)


def _query_sequence(plan, user: int, timestamp: int, prefix_tokens: int):
    frame = plan.trace.interactions
    user_frame = frame[(frame["user_idx"] == user) & (frame["time_ms"] < timestamp)]
    if len(user_frame) < 2:
        return None
    cutoff = timestamp - 7 * 86400 * 1000
    user_frame = user_frame[user_frame["time_ms"] >= cutoff].sort_values("time_ms").iloc[-(prefix_tokens + 1) :]
    if len(user_frame) < 2:
        return None
    history = plan._frame_sequence(user, user_frame)
    return {
        "prefix": {
            "item_ids": history["item_ids"][:-1],
            "behaviors": history["behaviors"][:-1],
            "time_deltas": history["time_deltas"][:-1],
        },
        "suffix": {
            "item_ids": history["item_ids"][-1:],
            "behaviors": history["behaviors"][-1:],
            "time_deltas": history["time_deltas"][-1:],
        },
        "timestamps": history["timestamps"][:-1],
    }


def _hidden(model, cache, suffix: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    items = torch.from_numpy(suffix["item_ids"]).long().unsqueeze(0).to(device)
    behaviors = torch.from_numpy(suffix["behaviors"]).long().unsqueeze(0).to(device)
    deltas = torch.from_numpy(suffix["time_deltas"]).float().unsqueeze(0).to(device)
    hidden, _ = model.forward_with_cache(cache, items, behaviors, deltas)
    return hidden[0, 0]


@torch.no_grad()
def run_random_exposure_screen(config_path: str | Path) -> dict[str, Any] | None:
    config_path = Path(config_path)
    config = load_random_exposure_config(config_path)
    chain_path = Path(config["chain_config"]["path"])
    chain = load_rolling_chain_config(chain_path)
    chain["config_path"] = str(chain_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand random-exposure screen requires two ranks")
    output = Path(config["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = effective_document(chain)
        training_users = int(document["data"]["max_users"])
        document["data"]["max_users"] = int(config["evaluation_max_users"])
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in chain["update_date_indices"]:
            plan.ingest_day(dates[int(date_index)])
        random_frame = _random_frame(config, plan)
        eligible = []
        for user, frame in random_frame.groupby("user_idx", sort=False):
            labels = frame["label"].to_numpy(dtype=np.bool_)
            if bool(labels.any()) and bool((~labels).any()):
                eligible.append(int(user))
        generator = np.random.default_rng(int(config["sampling_seed"]))
        selected = sorted(
            np.asarray(eligible)[generator.permutation(len(eligible))[: int(config["selected_users"])]].tolist()
        )
        local_users = selected[runtime.rank :: runtime.world_size]
        candidate_names = [value["candidate_id"] for value in chain["candidates"]]
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
        boundaries = [
            (int(plan.daily_segments[dates[index]]["time_ms"].min()), version)
            for version, index in zip(range(3, 9), range(17, 23), strict=True)
        ]
        records = []
        started = time.perf_counter()
        for ordinal, user in enumerate(local_users):
            user_random = random_frame[random_frame["user_idx"] == user]
            labels = []
            logits_by_candidate = [[] for _ in chain["candidates"]]
            errors_by_candidate = [[] for _ in chain["candidates"]]
            for row in user_random.itertuples(index=False):
                sequence = _query_sequence(
                    plan,
                    int(user),
                    int(row.time_ms),
                    int(config["prefix_tokens"]),
                )
                if sequence is None:
                    continue
                target = torch.tensor([int(row.item_idx)], dtype=torch.int64, device=runtime.device)
                for candidate_index, candidate in enumerate(chain["candidates"]):
                    candidate_models = models[candidate["candidate_id"]]
                    current = candidate_models[8]
                    old = candidate_models[7]
                    recursive, _ = _recursive_cache(
                        candidate_models,
                        sequence["prefix"],
                        sequence["timestamps"],
                        boundaries,
                        runtime.device,
                    )
                    fresh = _stored_cache(current, sequence["prefix"], runtime.device)
                    old_fresh = _stored_cache(old, sequence["prefix"], runtime.device)
                    hidden = torch.stack(
                        (
                            _hidden(current, recursive, sequence["suffix"], runtime.device),
                            _hidden(current, fresh, sequence["suffix"], runtime.device),
                            _hidden(current, _empty_cache(fresh), sequence["suffix"], runtime.device),
                            _hidden(old, old_fresh, sequence["suffix"], runtime.device),
                        )
                    ).unsqueeze(1)
                    scores = _logits(current, hidden, target, candidate).squeeze(1)
                    scores[3:4] = _logits(old, hidden[3:4], target, candidate).squeeze(1)
                    logits_by_candidate[candidate_index].append(scores.cpu())
                    errors_by_candidate[candidate_index].append(
                        {
                            "cache_relative_error": cache_relative_error(recursive, fresh),
                            "hidden_relative_error": float(
                                torch.linalg.vector_norm((hidden[0] - hidden[1]).double())
                                / torch.linalg.vector_norm(hidden[1].double()).clamp_min(1e-12)
                            ),
                        }
                    )
                labels.append(bool(row.label))
            labels_tensor = torch.tensor(labels, dtype=torch.bool)
            if not bool(labels_tensor.any()) or not bool((~labels_tensor).any()):
                continue
            metrics = []
            errors = []
            for candidate_index in range(len(chain["candidates"])):
                metrics.append(
                    _metric_values(torch.stack(logits_by_candidate[candidate_index], dim=1), labels_tensor)
                )
                errors.append(
                    {
                        name: float(
                            np.mean(
                                [value[name] for value in errors_by_candidate[candidate_index]]
                            )
                        )
                        for name in ("cache_relative_error", "hidden_relative_error")
                    }
                )
            records.append(
                {
                    "user_id": int(user),
                    "exposures": len(labels),
                    "positives": int(labels_tensor.sum().item()),
                    "metrics": np.stack(metrics),
                    "errors": errors,
                    "candidate_names": candidate_names,
                }
            )
            if (ordinal + 1) % 8 == 0 or ordinal + 1 == len(local_users):
                print(
                    f"phase=random_exposure_screen rank={runtime.rank} "
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
            "protocol": RANDOM_EXPOSURE_PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "chain_config": config["chain_config"],
            "checkpoint_bindings": bindings,
            "data": metadata,
            "evaluation": {
                "training_max_users": training_users,
                "evaluation_max_users": config["evaluation_max_users"],
                "evaluation_date": config["evaluation_date"],
                "eligible_users": len(eligible),
                "selected_users": len(combined),
                "exposures": sum(value["exposures"] for value in combined),
                "engaged_exposures": sum(value["positives"] for value in combined),
                "random_log_excluded_from_training": True,
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


def validate_random_exposure_result(result: dict[str, Any]) -> None:
    evaluation = result.get("evaluation", {})
    if (
        result.get("protocol") != RANDOM_EXPOSURE_PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or evaluation.get("random_log_excluded_from_training") is not True
        or int(evaluation.get("selected_users", 0)) < 64
        or int(evaluation.get("exposures", 0)) < 256
        or int(evaluation.get("engaged_exposures", 0)) < 64
        or len(evaluation.get("candidate_models", [])) != 2
        or evaluation.get("methods") != list(METHODS)
    ):
        raise ValueError("KuaiRand random-exposure result differs")
