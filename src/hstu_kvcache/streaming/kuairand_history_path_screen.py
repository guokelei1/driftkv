from __future__ import annotations

import copy
import gc
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_root_cause import (
    _atomic_json,
    _atomic_torch,
    _evaluate_edge,
    _popularity_ranks,
    _train_epoch,
    file_sha256,
    load_plan,
    make_model,
)

PROTOCOL = "evokv_kuairand_history_path_screen_v0"
EXPECTED_CANDIDATES = (
    ("all_parameters", "all"),
    ("core_only", "core"),
    ("cache_path_only", "cache_path"),
    ("kv_only", "kv"),
)


def load_history_path_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    candidates = document.get("candidates", [])
    execution = document.get("execution", {})
    outputs = document.get("outputs", {})
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("source_version") != 1
        or document.get("target_version") != 2
        or document.get("update_date_index") != 15
        or document.get("evaluation_date_index") != 16
        or [(value.get("id"), value.get("parameter_group")) for value in candidates]
        != list(EXPECTED_CANDIDATES)
        or any(
            int(value.get("epochs", 0)) != 2
            or float(value.get("lr", 0.0)) != 0.0001
            or float(value.get("weight_decay", -1.0)) != 0.0001
            for value in candidates
        )
        or execution.get("training_cuda_visible_devices") != "0"
        or execution.get("evaluation_cuda_visible_devices") != "0,1"
        or execution.get("evaluation_world_size") != 2
        or not all(isinstance(outputs.get(name), str) for name in ("checkpoint_root", "training_result", "evaluation_result"))
        or file_sha256(source.get("base_config", {}).get("path", ""))
        != source.get("base_config", {}).get("sha256")
        or file_sha256(source.get("theta1", {}).get("path", ""))
        != source.get("theta1", {}).get("sha256")
    ):
        raise ValueError("KuaiRand history-path screen config differs")
    return document


def _effective_document(config: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(config["source"]["base_config"]["path"]).read_text())
    document["interventions"]["methods"] = [
        "fresh_full_a",
        "fresh_full_b",
        "stale_previous",
        "no_prefix",
    ]
    document["quality"]["record_limit_per_rank"] = int(
        config["evaluation"]["record_limit_per_rank"]
    )
    document["quality"]["cap_user_limit_to_eligible"] = False
    document["quality"]["bootstrap_samples"] = int(
        config["evaluation"]["bootstrap_samples"]
    )
    document["quality"]["bootstrap_seed"] = int(
        config["evaluation"]["bootstrap_seed"]
    )
    return document


def _select_parameters(model, group: str) -> tuple[list[torch.nn.Parameter], list[str]]:
    last = model.cfg.num_layers - 1
    selected = []
    names = []
    for name, parameter in model.named_parameters():
        if group == "all":
            active = True
        elif group == "core":
            active = name != "item_emb.weight"
        elif group == "kv":
            active = ".attn.k_proj." in name or ".attn.v_proj." in name
        elif group == "cache_path":
            cache_safe = name == "item_emb.weight" or name == "final_norm.weight"
            cache_safe = cache_safe or name.startswith(
                (
                    f"blocks.{last}.attn.q_proj.",
                    f"blocks.{last}.attn.out_proj.",
                    f"blocks.{last}.gate_proj.",
                )
            )
            active = not cache_safe
        else:
            raise ValueError("KuaiRand history-path parameter group differs")
        parameter.requires_grad_(active)
        if active:
            selected.append(parameter)
            names.append(name)
    if not selected:
        raise RuntimeError("KuaiRand history-path parameter group is empty")
    return selected, names


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _checkpoint_path(config: dict[str, Any], candidate_id: str) -> Path:
    return Path(config["outputs"]["checkpoint_root"]) / candidate_id / "theta_2" / "model.pt"


def run_history_path_training(config_path: str | Path) -> dict[str, Any]:
    config = load_history_path_config(config_path)
    output = Path(config["outputs"]["training_result"])
    if output.is_file():
        return json.loads(output.read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("KuaiRand history-path training requires one rank")
    document = _effective_document(config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    plan.ingest_day(dates[14])
    source_histories = copy.deepcopy(plan.user_histories)
    source_state = torch.load(
        config["source"]["theta1"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    started = time.perf_counter()
    records = []
    for candidate_index, candidate in enumerate(config["candidates"]):
        plan.user_histories = copy.deepcopy(source_histories)
        plan.ingest_day(dates[int(config["update_date_index"])])
        model = make_model(document, plan, device)
        model.load_state_dict(source_state)
        parameters, parameter_names = _select_parameters(
            model, str(candidate["parameter_group"])
        )
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(candidate["lr"]),
            weight_decay=float(candidate["weight_decay"]),
        )
        epochs = []
        for epoch in range(int(candidate["epochs"])):
            seed = int(config["training_seed"]) + epoch
            _seed(seed)
            batches = plan.iter_train_batches(
                dates[int(config["update_date_index"])],
                int(config["batch_size"]),
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
            epochs.append(
                _train_epoch(
                    model,
                    optimizer,
                    batches,
                    device,
                    int(config["negative_count"]),
                    None,
                    f"kuairand_history_path_{candidate['id']}_epoch{epoch + 1}",
                )
            )
        model_path = _checkpoint_path(config, str(candidate["id"]))
        if model_path.exists():
            raise FileExistsError(f"KuaiRand history-path checkpoint exists: {model_path}")
        _atomic_torch(model_path, model.state_dict())
        manifest = {
            "protocol": PROTOCOL,
            "status": "complete_development_checkpoint",
            "scientific_result": False,
            "formal_result": False,
            "config_sha256": file_sha256(config_path),
            "source_model_sha256": config["source"]["theta1"]["sha256"],
            "candidate": candidate,
            "candidate_index": candidate_index,
            "model": asdict(model.cfg),
            "model_sha256": file_sha256(model_path),
            "trainable_parameter_names": parameter_names,
            "trainable_parameters": sum(value.numel() for value in parameters),
            "training": epochs,
        }
        _atomic_json(model_path.with_name("manifest.json"), manifest)
        records.append(manifest)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "protocol": PROTOCOL,
        "status": "complete_development_training",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data": metadata,
        "source_version": 1,
        "target_version": 2,
        "update_date": dates[int(config["update_date_index"])],
        "candidates": records,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    _atomic_json(output, result)
    return result


def _candidate_summary(edge: dict[str, Any]) -> dict[str, Any]:
    stale = edge["fresh_current_comparisons"]["stale_previous"]
    update = edge["previous_to_current_fresh_update_value"]
    stale_ranking = {
        metric: stale[metric]
        for metric in ("mrr", "ndcg_at_10", "hit_rate_at_50", "hit_rate_at_200")
    }
    positive_rankings = [
        metric
        for metric, value in stale_ranking.items()
        if value["fresh_current_advantage_positive_with_ci"]
        and value["fresh_current_advantage_relative_percent"] > 0.0
    ]
    return {
        "fresh_update_cross_entropy_positive_with_ci": update["cross_entropy"][
            "current_fresh_advantage_positive_with_ci"
        ],
        "fresh_update_ndcg_at_10_relative_percent": update["ndcg_at_10"][
            "current_fresh_advantage_relative_percent"
        ],
        "stale_cross_entropy_positive_with_ci": stale["cross_entropy"][
            "fresh_current_advantage_positive_with_ci"
        ],
        "stale_positive_ranking_metrics_with_ci": positive_rankings,
        "stale_ndcg_at_10_relative_percent": stale["ndcg_at_10"][
            "fresh_current_advantage_relative_percent"
        ],
        "stale_mrr_relative_percent": stale["mrr"][
            "fresh_current_advantage_relative_percent"
        ],
    }


def run_history_path_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_history_path_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand history-path evaluation requires two ranks")
    output = Path(config["outputs"]["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = _effective_document(config)
        torch.set_float32_matmul_precision("high")
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in (14, int(config["update_date_index"])):
            plan.ingest_day(dates[date_index])
        source = make_model(document, plan, runtime.device)
        source.load_state_dict(
            torch.load(
                config["source"]["theta1"]["path"],
                map_location="cpu",
                weights_only=True,
            )
        )
        source.eval()
        popularity = _popularity_ranks(plan)
        started = time.perf_counter()
        candidates = []
        for candidate in config["candidates"]:
            current = make_model(document, plan, runtime.device)
            current.load_state_dict(
                torch.load(
                    _checkpoint_path(config, str(candidate["id"])),
                    map_location="cpu",
                    weights_only=True,
                )
            )
            current.eval()
            edge = _evaluate_edge(
                document,
                plan,
                source,
                current,
                int(config["target_version"]),
                dates[int(config["update_date_index"])],
                dates[int(config["evaluation_date_index"])],
                popularity,
                runtime,
            )
            if runtime.is_primary:
                assert edge is not None
                candidates.append(
                    {
                        "candidate": candidate,
                        "summary": _candidate_summary(edge),
                        "edge": edge,
                    }
                )
            del current
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
            "data": metadata,
            "update_date": dates[int(config["update_date_index"])],
            "evaluation_date": dates[int(config["evaluation_date_index"])],
            "candidates": candidates,
            "same_user_set_across_candidates": len(
                {value["edge"]["selected_user_ids_sha256"] for value in candidates}
            )
            == 1,
            "all_implementation_checks_passed": all(
                value["edge"]["sanity"]["implementation_passed"]
                for value in candidates
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)

