from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.models import apply_attention_coordinate_scale_

from .kuairand_projected_persistent import (
    _distributed,
    _evaluation_batches,
    _initialize_model,
    _load_checkpoint,
    load_persistent_config,
)
from .kuairand_projected_scale import _capture_old, _evaluate_captured
from .kuairand_query_multiversion import _edge_config
from .kuairand_query_transition import (
    _atomic_json,
    _summary,
    build_workload,
    file_sha256,
    load_config,
)
from .kuairand_stationary_coordinate_control import (
    _fidelity_summary,
    _records_for_split,
)

PROTOCOL = "evokv_kuairand_natural_path_attribution_v0"
VARIANTS = (
    "embedding_only",
    "embedding_projection",
    "embedding_projection_plus_q",
    "embedding_projection_plus_kv",
    "embedding_projection_plus_qkv",
    "embedding_projection_plus_qkvo",
    "full_without_qkvo",
    "full_current",
    "frozen_coordinate_positive_control",
)
EXTRA_FIDELITY_METRICS = (
    "hidden_history_projection",
    "hidden_history_orthogonal_relative_error",
    "score_history_projection",
    "score_history_orthogonal_relative_error",
)


def load_natural_path_attribution_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source")
    evaluation = document.get("evaluation")
    control = document.get("positive_control")
    output = document.get("output")
    config_path = Path(source.get("config", {}).get("path", "")) if isinstance(source, dict) else Path()
    source_manifest = Path(source.get("source_manifest", {}).get("path", "")) if isinstance(source, dict) else Path()
    target_manifest = Path(source.get("target_manifest", {}).get("path", "")) if isinstance(source, dict) else Path()
    control_path = Path(control.get("result", {}).get("path", "")) if isinstance(control, dict) else Path()
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(isinstance(value, dict) for value in (source, evaluation, control))
        or not config_path.is_file()
        or file_sha256(config_path) != source.get("config", {}).get("sha256")
        or not source_manifest.is_file()
        or file_sha256(source_manifest) != source.get("source_manifest", {}).get("sha256")
        or not target_manifest.is_file()
        or file_sha256(target_manifest) != source.get("target_manifest", {}).get("sha256")
        or int(source.get("source_version", -1)) + 1 != int(source.get("target_version", -1))
        or int(evaluation.get("workload_transition_version", -1)) != int(source.get("target_version", -2))
        or int(evaluation.get("candidate_count", 0)) != 100
        or int(evaluation.get("targets_per_user", 0)) != 8
        or float(evaluation.get("tuning_fraction", 0.0)) != 0.25
        or int(evaluation.get("split_seed", 0)) < 1
        or document.get("variants") != list(VARIANTS)
        or not control_path.is_file()
        or file_sha256(control_path) != control.get("result", {}).get("sha256")
        or not isinstance(output, str)
    ):
        raise ValueError("KuaiRand natural-path attribution config differs")
    source_document = load_persistent_config(config_path)
    if int(source["target_version"]) > int(source_document["checkpoint"]["versions"]):
        raise ValueError("KuaiRand natural-path attribution version differs")
    control_result = json.loads(control_path.read_text())
    transform = control_result.get("selected", {}).get("transform")
    if (
        control_result.get("status") != "complete_development_causal_control"
        or not control_result.get("fresh_function_invariance", {}).get("passed")
        or not isinstance(transform, dict)
        or set(transform) != {"key_log_step", "name", "value_log_step"}
    ):
        raise ValueError("KuaiRand natural-path positive control differs")
    return document


def _is_projection(name: str, projection: str) -> bool:
    return f".attn.{projection}." in name


def _dense_state_for_variant(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, torch.Tensor]:
    if set(source) != set(target) or variant not in VARIANTS[:-1]:
        raise ValueError("KuaiRand natural-path dense state differs")
    if variant == "full_current":
        return {name: value.clone() for name, value in target.items()}
    if variant == "full_without_qkvo":
        return {
            name: (
                source[name].clone()
                if any(_is_projection(name, projection) for projection in ("q_proj", "k_proj", "v_proj", "out_proj"))
                else target[name].clone()
            )
            for name in source
        }
    selected = {
        "embedding_only": (),
        "embedding_projection": (),
        "embedding_projection_plus_q": ("q_proj",),
        "embedding_projection_plus_kv": ("k_proj", "v_proj"),
        "embedding_projection_plus_qkv": ("q_proj", "k_proj", "v_proj"),
        "embedding_projection_plus_qkvo": (
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ),
    }[variant]
    return {
        name: (
            target[name].clone()
            if any(_is_projection(name, projection) for projection in selected)
            else value.clone()
        )
        for name, value in source.items()
    }


def _relative_change(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    predicate,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, source_value in source.items():
        if predicate(name):
            target_value = target[name]
            numerator += float((target_value.double() - source_value.double()).square().sum().item())
            denominator += float(source_value.double().square().sum().item())
    if denominator == 0.0:
        raise RuntimeError("KuaiRand natural-path parameter group is empty")
    return math.sqrt(numerator / denominator)


def _extra_fidelity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[int(record["user_id"])].append(record["fidelity"])
    output = {}
    for metric in EXTRA_FIDELITY_METRICS:
        values = np.asarray(
            [
                statistics.fmean(float(record[metric]) for record in user_records)
                for user_records in grouped.values()
            ],
            dtype=np.float64,
        )
        output[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return output


def _evaluate_variant(
    name: str,
    dense,
    embedding,
    captured,
    workload,
    edge_document,
    split_seed: int,
    tuning_fraction: float,
    rank: int,
    world_size: int,
    device,
) -> dict[str, Any] | None:
    summary, evaluation = _evaluate_captured(
        dense,
        embedding,
        captured,
        workload,
        edge_document,
        rank,
        world_size,
        device,
    )
    if rank != 0:
        return None
    assert summary is not None and evaluation is not None
    partitions = {}
    for split in ("tuning", "holdout"):
        records = _records_for_split(evaluation, split, split_seed, tuning_fraction)
        ranking = _summary(
            {"records": records, "sanity": evaluation["sanity"]}, edge_document
        )
        fresh = ranking["endpoints"]["recompute"]
        reuse = ranking["endpoints"]["reuse"]
        partitions[split] = {
            "ranking": ranking,
            "fidelity": _fidelity_summary(records),
            "history_direction": _extra_fidelity_summary(records),
            "derived": {
                "mrr_loss_percent": 100.0 * (fresh["mrr"] - reuse["mrr"]) / fresh["mrr"],
                "ndcg_at_5_loss_percent": 100.0 * (fresh["ndcg_at_5"] - reuse["ndcg_at_5"]) / fresh["ndcg_at_5"],
                "hit_rate_at_5_loss_percent": 100.0 * (fresh["hit_rate_at_5"] - reuse["hit_rate_at_5"]) / fresh["hit_rate_at_5"],
            },
        }
    return {"name": name, "all_users": summary, **partitions}


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# KuaiRand natural update-path attribution",
        "",
        "| intervention | NDCG@5 loss | MRR loss | HR@5 loss | Top-10 changed | score cosine loss | hidden history projection | score history projection |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in result["variants"]:
        holdout = variant["holdout"]
        derived = holdout["derived"]
        fidelity = holdout["fidelity"]
        direction = holdout["history_direction"]
        lines.append(
            "| {name} | {ndcg:+.3f}% | {mrr:+.3f}% | {hr:+.3f}% | {top10:.3f}% | {score:.3f}% | {hidden_projection:.4f} | {score_projection:.4f} |".format(
                name=variant["name"],
                ndcg=derived["ndcg_at_5_loss_percent"],
                mrr=derived["mrr_loss_percent"],
                hr=derived["hit_rate_at_5_loss_percent"],
                top10=100.0 * fidelity["top10_changed_fraction"],
                score=100.0 * (1.0 - fidelity["user_mean"]["score_cosine"]["mean"]),
                hidden_projection=direction["hidden_history_projection"]["mean"],
                score_projection=direction["score_history_projection"]["mean"],
            )
        )
    return "\n".join(lines) + "\n"


def run_natural_path_attribution(config_path: str | Path) -> dict[str, Any] | None:
    path = Path(config_path)
    config = load_natural_path_attribution_config(path)
    output_path = Path(config["output"])
    if output_path.is_file():
        result = json.loads(output_path.read_text())
        return result if int(os.environ.get("RANK", "0")) == 0 else None
    source_path = Path(config["source"]["config"]["path"])
    document = load_persistent_config(source_path)
    document["config_path"] = str(source_path)
    config_sha256 = file_sha256(source_path)
    rank, world_size, device = _distributed(document)
    started = time.monotonic()
    try:
        base_config = load_config(document["parent"]["base_config"]["path"])
        target_version = int(config["source"]["target_version"])
        transition = document["transitions"][target_version - 1]
        edge_document = _edge_config(base_config, transition, 1.0)
        edge_document["data"]["update_dates"] = transition["update_dates"]
        edge_document["data"]["evaluation_targets_per_user"] = int(
            config["evaluation"]["targets_per_user"]
        )
        edge_document["data"]["user_limit"] = document["data"].get("user_limit")
        edge_document["evaluation"]["candidate_count"] = int(
            config["evaluation"]["candidate_count"]
        )
        workload = build_workload(edge_document)
        dense, embedding, tracker, geometry = _initialize_model(
            document,
            base_config,
            int(workload["metadata"]["embedding_rows"]),
            rank,
            world_size,
            device,
        )
        root = Path(document["outputs"]["checkpoint_root"])
        _load_checkpoint(
            root,
            int(config["source"]["source_version"]),
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
        )
        source_dense = {
            name: value.detach().clone() for name, value in dense.state_dict().items()
        }
        source_projection = embedding.projection_weight.detach().clone()
        batches = _evaluation_batches(
            workload,
            int(document["evaluation"]["local_batch_size"]),
            rank,
            world_size,
        )
        natural_source = _capture_old(
            dense, embedding, batches, workload, base_config, device
        )
        _load_checkpoint(
            root,
            target_version,
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
        )
        target_dense = {
            name: value.detach().clone() for name, value in dense.state_dict().items()
        }
        target_projection = embedding.projection_weight.detach().clone()
        coordinate_source = _capture_old(
            dense, embedding, batches, workload, base_config, device
        )
        split_seed = int(config["evaluation"]["split_seed"])
        tuning_fraction = float(config["evaluation"]["tuning_fraction"])
        variants = []
        for name in VARIANTS[:-1]:
            dense.load_state_dict(_dense_state_for_variant(source_dense, target_dense, name))
            with torch.no_grad():
                embedding.projection_weight.copy_(
                    source_projection if name == "embedding_only" else target_projection
                )
            value = _evaluate_variant(
                name,
                dense,
                embedding,
                natural_source,
                workload,
                edge_document,
                split_seed,
                tuning_fraction,
                rank,
                world_size,
                device,
            )
            if rank == 0:
                assert value is not None
                variants.append(value)
                print(
                    f"phase=kuairand_natural_path variant={name} "
                    f"holdout_ndcg={value['holdout']['derived']['ndcg_at_5_loss_percent']:.3f}%",
                    flush=True,
                )
        dense.load_state_dict(target_dense)
        with torch.no_grad():
            embedding.projection_weight.copy_(target_projection)
        control_result = json.loads(
            Path(config["positive_control"]["result"]["path"]).read_text()
        )
        transform = control_result["selected"]["transform"]
        certificate = apply_attention_coordinate_scale_(
            dense.core,
            key_log_scale=float(transform["key_log_step"]),
            value_log_scale=float(transform["value_log_step"]),
        )
        value = _evaluate_variant(
            VARIANTS[-1],
            dense,
            embedding,
            coordinate_source,
            workload,
            edge_document,
            split_seed,
            tuning_fraction,
            rank,
            world_size,
            device,
        )
        if rank != 0:
            return None
        assert value is not None
        value["coordinate_certificate"] = certificate
        variants.append(value)
        parameter_change = {
            projection: _relative_change(
                source_dense,
                target_dense,
                lambda name, projection=projection: _is_projection(name, projection),
            )
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
        }
        parameter_change["other_dense"] = _relative_change(
            source_dense,
            target_dense,
            lambda name: not any(
                _is_projection(name, projection)
                for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
            ),
        )
        parameter_change["embedding_projection"] = float(
            torch.linalg.vector_norm(
                (target_projection - source_projection).double()
            ).item()
            / torch.linalg.vector_norm(source_projection.double()).clamp_min(1e-12).item()
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_attribution",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": file_sha256(path)},
            "source": config["source"],
            "positive_control": config["positive_control"],
            "geometry": geometry,
            "parameter_relative_l2_change": parameter_change,
            "variants": variants,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output_path, result)
        table_path = output_path.with_suffix(".md")
        table_path.write_text(_render(result))
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
