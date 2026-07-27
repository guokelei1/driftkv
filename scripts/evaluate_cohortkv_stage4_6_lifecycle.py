from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from compile_cohortkv_stage4_6_edges import (
    DEFAULT_OUTPUT as DEFAULT_COMPILER_OUTPUT,
)
from compile_cohortkv_stage4_6_edges import (
    DEFAULT_RUNTIME_DIR,
    EXPERIMENT_PROTOCOL,
    direct_program_path,
)
from evaluate_cohortkv_stage1_frontier import (
    DEFAULT_BLUEPRINT,
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST,
    DEFAULT_PREPARED,
    DEFAULT_PROGRAM_DIR,
    DEFAULT_PROGRAM_RESULT,
    DEFAULT_TRAINING,
    label_free_eval_batches,
    semantic_values,
    sha256,
    validate_frozen_inputs,
)
from motivation_validity import move_batch, seed_everything
from scipy.stats import spearmanr

from hstu_kvcache.migration import (
    BalancedLifecyclePolicy,
    CacheLifecycleState,
    JaggedMigratedKVBatch,
    LinearSketchRiskCalibration,
    SketchLifecyclePolicy,
    absolute_log_norm_ratio_values,
    aggregate_layer_values,
    fit_monotone_risk_calibration,
    pack_padded_cache,
    relative_cache_values,
    transition_sketch_values,
    unpack_jagged_cache,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    DirectOldKVProgram,
    load_direct_oldkv_program,
)
from hstu_kvcache.streaming import load_checkpoint_model
from hstu_kvcache.utils import save_json

PROTOCOL = "cohortkv_single_config_stage4_6_lifecycle_search_v1"
DEFAULT_FIT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_6_fit_trajectory_seed0.json"
)
DEFAULT_FIT_TRANSITION_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_6_fit_transitions_seed0.json"
)
DEFAULT_SELECTION_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_6_selection_transitions_seed0.json"
)
DEFAULT_POLICY_SEARCH_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_6_policy_search_seed0.json"
)
LAYER_QUANTILES = (0.5, 0.75, 0.9, 1.0)
LAUNCH = {
    "block_m": 64,
    "block_n": 128,
    "block_k": 64,
    "num_warps": 8,
    "num_stages": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-result", default=DEFAULT_PROGRAM_RESULT)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--adjacent-compiler", default=DEFAULT_COMPILER_OUTPUT)
    parser.add_argument("--runtime-dir", default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--fit-output", default=DEFAULT_FIT_OUTPUT)
    parser.add_argument(
        "--fit-transition-output",
        default=DEFAULT_FIT_TRANSITION_OUTPUT,
    )
    parser.add_argument("--selection-output", default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument(
        "--policy-search-output",
        default=DEFAULT_POLICY_SEARCH_OUTPUT,
    )
    parser.add_argument(
        "--phase",
        choices=("fit", "fit-transitions", "selection", "policy"),
        default="fit",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.seed != 0 or args.batch_size != 4:
        raise ValueError("Stage 4.6 freezes seed 0 and batch size 4")
    if torch.device(args.device).type != "cuda":
        raise ValueError("Stage 4.6 requires CUDA")


def record_map(manifest: dict) -> dict[int, int]:
    return {
        int(record["user_id"]): int(record["record_id"])
        for record in manifest["records"]
    }


def ordered_samples(
    samples: list[dict],
    records_by_user: dict[int, int],
) -> list[dict]:
    return sorted(
        samples,
        key=lambda sample: records_by_user[
            int(sample["history"]["user_id"])
        ],
    )


def record_ids(
    selected: list[dict],
    records_by_user: dict[int, int],
) -> tuple[int, ...]:
    return tuple(
        records_by_user[int(sample["history"]["user_id"])]
        for sample in selected
    )


def output_batch(
    source: JaggedMigratedKVBatch,
    target_version: int,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version=source.migration_anchor_version,
        served_kv_target=f"theta{target_version}",
        k=torch.empty_like(source.k),
        v=torch.empty_like(source.v),
        lengths=source.lengths.clone(),
        offsets=source.offsets.clone(),
    )


def relabel_source(
    source: JaggedMigratedKVBatch,
    version: int,
) -> JaggedMigratedKVBatch:
    return JaggedMigratedKVBatch(
        record_ids=source.record_ids,
        migration_anchor_version=f"theta{version}",
        served_kv_target=f"theta{version}",
        k=source.k,
        v=source.v,
        lengths=source.lengths,
        offsets=source.offsets,
    )


def execute_direct(
    operator: DirectOldKVFusedOperator,
    program: DirectOldKVProgram,
    source: JaggedMigratedKVBatch,
    target_version: int,
) -> JaggedMigratedKVBatch:
    prepared_source = relabel_source(source, target_version - 1)
    destination = output_batch(prepared_source, target_version)
    return operator.execute_into(
        program,
        prepared_source,
        destination,
    )


def exact_batch(
    model,
    prefix: dict,
    batch_record_ids: tuple[int, ...],
    version: int,
) -> JaggedMigratedKVBatch:
    cache = model.compute_kv(
        prefix["item_ids"],
        prefix["behaviors"],
        prefix["time_deltas"],
        lengths=prefix["lengths"],
    )
    return pack_padded_cache(
        cache,
        prefix["lengths"],
        batch_record_ids,
        f"theta{version}",
        f"theta{version}",
    )


def aggregate(values: torch.Tensor) -> dict[str, list[float]]:
    return {
        f"q{int(quantile * 100):03d}": (
            aggregate_layer_values(values, quantile).cpu().tolist()
        )
        for quantile in LAYER_QUANTILES
    }


def timed_cuda(action, device: torch.device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = action()
    end.record()
    end.synchronize()
    return value, float(start.elapsed_time(end))


def aggregate_sketch(
    values: dict[str, torch.Tensor],
) -> dict[str, dict[str, list[float]]]:
    return {
        feature: aggregate(feature_values)
        for feature, feature_values in values.items()
    }


def row_sketch(
    values: dict[str, dict[str, list[float]]],
    row: int,
) -> dict[str, dict[str, float]]:
    return {
        feature: {
            quantile: float(rows[row])
            for quantile, rows in quantiles.items()
        }
        for feature, quantiles in values.items()
    }


@torch.inference_mode()
def exact_semantic_reference(
    model,
    exact: JaggedMigratedKVBatch,
    suffix: dict,
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = unpack_jagged_cache(exact)
    hidden, _ = model.forward_with_cache(
        cache,
        suffix["item_ids"],
        suffix["behaviors"],
        suffix["time_deltas"],
    )
    hidden = hidden[:, 0]
    return hidden, model.item_emb.score(hidden, candidate_ids)


@torch.inference_mode()
def run_transition_dag(
    args: argparse.Namespace,
    cfg,
    manifest: dict,
    samples: list[dict],
    role: str,
    include_semantics: bool,
) -> dict:
    device = torch.device(args.device)
    records_by_user = record_map(manifest)
    ordered = ordered_samples(samples, records_by_user)
    batches = list(
        label_free_eval_batches(
            ordered,
            cfg.max_seq_len,
            args.batch_size,
        )
    )
    operator = DirectOldKVFusedOperator(**LAUNCH)
    all_items = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    )
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        0,
        device,
    )
    states_by_anchor: dict[int, list[JaggedMigratedKVBatch]] = {0: []}
    for selected, _, prefix_cpu, _ in batches:
        prefix = move_batch(prefix_cpu, device)
        states_by_anchor[0].append(
            exact_batch(
                source_model,
                prefix,
                record_ids(selected, records_by_user),
                0,
            )
        )
    transitions = []
    exact_costs = []
    started = time.perf_counter()
    for source_version in range(11):
        target_version = source_version + 1
        target_model = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            target_version,
            device,
        )
        program_cpu, descriptor = load_direct_oldkv_program(
            direct_program_path(
                args.runtime_dir,
                source_version,
                target_version,
            ),
            expected_source_version=f"theta{source_version}",
            expected_target_version=f"theta{target_version}",
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.num_heads * cfg.head_dim,
        )
        program = operator.prepare_program(program_cpu, device)
        exact_targets = []
        semantic_references = []
        for selected, _, prefix_cpu, suffix_cpu in batches:
            prefix = move_batch(prefix_cpu, device)
            suffix = move_batch(suffix_cpu, device)
            ids = record_ids(selected, records_by_user)
            exact_target, exact_ms = timed_cuda(
                partial(
                    exact_batch,
                    target_model,
                    prefix,
                    ids,
                    target_version,
                ),
                device,
            )
            exact_targets.append(exact_target)
            token_counts = exact_target.lengths.cpu().tolist()
            token_total = sum(token_counts)
            for current_id, tokens in zip(
                ids,
                token_counts,
                strict=True,
            ):
                exact_costs.append(
                    {
                        "record_id": current_id,
                        "target_version": target_version,
                        "prefix_tokens": int(tokens),
                        "measured_batch_gpu_ms": exact_ms,
                        "attributed_gpu_ms": (
                            exact_ms * tokens / token_total
                        ),
                    }
                )
            if include_semantics:
                candidate_ids = all_items.unsqueeze(0).expand(
                    len(ids),
                    -1,
                )
                semantic_references.append(
                    exact_semantic_reference(
                        target_model,
                        exact_target,
                        suffix,
                        candidate_ids,
                    )
                )
            else:
                semantic_references.append(None)
        next_states: dict[int, list[JaggedMigratedKVBatch]] = {}
        warmed = False
        sketch_warmed = False
        for anchor_version in sorted(states_by_anchor):
            depth = source_version - anchor_version
            if not (
                0 <= depth < 4
                or anchor_version == 0
            ):
                raise RuntimeError("transition DAG depth differs")
            next_states[anchor_version] = []
            for batch_index, (
                _selected,
                _,
                _,
                suffix_cpu,
            ) in enumerate(batches):
                source = states_by_anchor[anchor_version][batch_index]
                if not warmed:
                    warmup = execute_direct(
                        operator,
                        program,
                        source,
                        target_version,
                    )
                    del warmup
                    warmed = True
                candidate, migration_ms = timed_cuda(
                    partial(
                        execute_direct,
                        operator,
                        program,
                        source,
                        target_version,
                    ),
                    device,
                )
                sketch_values = transition_sketch_values(
                    source,
                    candidate,
                )
                if not sketch_warmed:
                    absolute_log_norm_ratio_values(
                        source,
                        candidate,
                    )
                    torch.cuda.synchronize(device)
                    sketch_warmed = True
                _, router_ms = timed_cuda(
                    partial(
                        absolute_log_norm_ratio_values,
                        source,
                        candidate,
                    ),
                    device,
                )
                sketch = aggregate_sketch(sketch_values)
                errors = aggregate(
                    relative_cache_values(
                        candidate,
                        exact_targets[batch_index],
                    )
                )
                semantics = None
                if include_semantics:
                    suffix = move_batch(suffix_cpu, device)
                    ids = source.record_ids
                    candidate_ids = all_items.unsqueeze(0).expand(
                        len(ids),
                        -1,
                    )
                    reference = semantic_references[batch_index]
                    if reference is None:
                        raise RuntimeError("semantic reference is missing")
                    semantics = semantic_values(
                        target_model,
                        unpack_jagged_cache(candidate),
                        suffix,
                        candidate_ids,
                        reference[0],
                        reference[1],
                    )
                tokens_by_row = source.lengths.cpu().tolist()
                token_total = sum(tokens_by_row)
                for row, current_id in enumerate(source.record_ids):
                    value = {
                        "record_id": current_id,
                        "anchor_version": anchor_version,
                        "source_version": source_version,
                        "target_version": target_version,
                        "migration_depth_before": depth,
                        "migration_depth_after": depth + 1,
                        "program_sha256": descriptor["sha256"],
                        "prefix_tokens": int(tokens_by_row[row]),
                        "sketch": row_sketch(sketch, row),
                        "cache_error": {
                            name: float(rows[row])
                            for name, rows in errors.items()
                        },
                        "measured_batch_gpu_ms": {
                            "migration": migration_ms,
                            "router": router_ms,
                        },
                        "attributed_gpu_ms": {
                            "migration": (
                                migration_ms
                                * tokens_by_row[row]
                                / token_total
                            ),
                            "router": (
                                router_ms
                                * tokens_by_row[row]
                                / token_total
                            ),
                        },
                    }
                    if semantics is not None:
                        value["semantics"] = {
                            name: float(rows[row])
                            for name, rows in semantics.items()
                        }
                    transitions.append(value)
                next_states[anchor_version].append(candidate)
        next_states[target_version] = exact_targets
        states_by_anchor = {
            anchor: values
            for anchor, values in next_states.items()
            if anchor == 0 or target_version - anchor <= 3
        }
        print(
            json.dumps(
                {
                    "phase": f"{role}-transitions",
                    "edge": (
                        f"theta{source_version}->theta{target_version}"
                    ),
                    "active_anchors": sorted(states_by_anchor),
                    "transitions": len(transitions),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del source_model, program_cpu, program
        source_model = target_model
        gc.collect()
        torch.cuda.empty_cache()
    del source_model, states_by_anchor
    expected = len(samples) * (
        sum(min(source + 1, 4) for source in range(11)) + 7
    )
    if len(transitions) != expected or len(exact_costs) != len(samples) * 11:
        raise RuntimeError("transition DAG coverage differs")
    return {
        "protocol": PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "phase": f"{role}-transitions",
        "status": "complete",
        "role": role,
        "labels_used": False,
        "semantic_reference": (
            "current-model exact FP16 K/V boundary" if include_semantics else None
        ),
        "records": len(samples),
        "edges": 11,
        "maximum_candidate_depth": 11,
        "transitions": transitions,
        "exact_costs": exact_costs,
        "launch": LAUNCH,
        "elapsed_seconds": time.perf_counter() - started,
    }


def fit_linear_sketch_calibration(
    transitions: list[dict],
    feature_name: str,
    layer_quantile: str,
    ridge: float = 0.1,
) -> LinearSketchRiskCalibration:
    selected = [
        value
        for value in transitions
        if int(value["migration_depth_after"]) <= 4
    ]
    feature = np.asarray(
        [
            value["sketch"][feature_name][layer_quantile]
            for value in selected
        ],
        dtype=np.float64,
    )
    groups = np.zeros((len(selected), 44), dtype=np.float64)
    for row, value in enumerate(selected):
        index = (
            int(value["source_version"]) * 4
            + int(value["migration_depth_after"])
            - 1
        )
        groups[row, index] = 1.0
    values = np.column_stack((feature, groups))
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales < 1e-12] = 1.0
    design = np.column_stack(
        (
            np.ones(len(values)),
            (values - means) / scales,
        )
    )
    target = np.log(
        np.asarray(
            [
                value["cache_error"]["q090"]
                for value in selected
            ],
            dtype=np.float64,
        )
        + 1e-8
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return LinearSketchRiskCalibration(
        feature_name=feature_name,
        layer_quantile=int(layer_quantile[1:]) / 100,
        intercept=float(coefficients[0]),
        feature_mean=float(means[0]),
        feature_scale=float(scales[0]),
        feature_coefficient=float(coefficients[1]),
        group_means=tuple(float(value) for value in means[1:]),
        group_scales=tuple(float(value) for value in scales[1:]),
        group_coefficients=tuple(
            float(value) for value in coefficients[2:]
        ),
        num_edges=11,
        maximum_depth=4,
        ridge=ridge,
        target="log_exact_relative_cache_error_q090",
    )


def transition_metric_values(value: dict) -> tuple[float, float, float]:
    semantic = value.get("semantics")
    if not isinstance(semantic, dict):
        raise ValueError("selection transition semantics are missing")
    return (
        max(0.0, 1.0 - float(value["cache_error"]["q090"])),
        float(semantic["score_cosine"]),
        float(semantic["top100_overlap"]),
    )


def summarize_policy_steps(
    steps: list[dict],
    costs: dict[str, float],
    all_exact_ms: float,
    records: int,
) -> dict:
    metric_names = (
        "cache_fidelity_q090",
        "score_cosine",
        "top100_overlap",
    )
    worst = {
        name: min(float(step["metrics"][name]) for step in steps)
        for name in metric_names
    }
    total_actions = records * len(steps)
    exact_records = sum(int(step["exact_records"]) for step in steps)
    return {
        "steps": steps,
        "worst_step": worst,
        "worst_view_fidelity": min(worst.values()),
        "exact_records": exact_records,
        "exact_fraction": exact_records / total_actions,
        "candidate_records": sum(
            int(step["candidate_records"]) for step in steps
        ),
        "discarded_candidate_records": sum(
            int(step["discarded_candidate_records"]) for step in steps
        ),
        "gpu_cost_ms": costs,
        "mixed_policy_gpu_ms": sum(costs.values()),
        "all_exact_gpu_ms": all_exact_ms,
        "cost_ratio_to_all_exact": sum(costs.values()) / all_exact_ms,
    }


def simulate_sketch_policy(
    transitions: list[dict],
    exact_costs: list[dict],
    policy: SketchLifecyclePolicy,
) -> dict:
    index = {
        (
            int(value["record_id"]),
            int(value["anchor_version"]),
            int(value["source_version"]),
        ): value
        for value in transitions
    }
    exact = {
        (int(value["record_id"]), int(value["target_version"])): float(
            value["attributed_gpu_ms"]
        )
        for value in exact_costs
    }
    ids = sorted({value[0] for value in index})
    states = {
        record_id: CacheLifecycleState.exact(record_id, 0)
        for record_id in ids
    }
    costs = {"migration": 0.0, "router": 0.0, "exact": 0.0}
    steps = []
    quantile = f"q{int(policy.calibration.layer_quantile * 100):03d}"
    for source_version in range(11):
        target_version = source_version + 1
        metrics = []
        exact_records = 0
        candidate_records = 0
        discarded_records = 0
        depth_after = []
        for record_id in ids:
            state = states[record_id]
            if not policy.requires_candidate(state):
                decision = policy.decide(state, target_version)
                costs["exact"] += exact[(record_id, target_version)]
                exact_records += 1
                metrics.append((1.0, 1.0, 1.0))
            else:
                value = index[
                    (
                        record_id,
                        state.last_exact_version,
                        source_version,
                    )
                ]
                action_cost = value["attributed_gpu_ms"]
                costs["migration"] += float(action_cost["migration"])
                costs["router"] += float(action_cost["router"])
                feature = float(
                    value["sketch"][policy.calibration.feature_name][
                        quantile
                    ]
                )
                decision = policy.decide(
                    state,
                    target_version,
                    feature,
                )
                candidate_records += 1
                if decision.action == "exact":
                    costs["exact"] += exact[(record_id, target_version)]
                    exact_records += 1
                    discarded_records += 1
                    metrics.append((1.0, 1.0, 1.0))
                else:
                    metrics.append(transition_metric_values(value))
            next_state = policy.advance(state, decision)
            states[record_id] = next_state
            depth_after.append(next_state.migration_depth)
        means = np.asarray(metrics).mean(axis=0)
        steps.append(
            {
                "target_version": target_version,
                "exact_records": exact_records,
                "candidate_records": candidate_records,
                "discarded_candidate_records": discarded_records,
                "metrics": {
                    "cache_fidelity_q090": float(means[0]),
                    "score_cosine": float(means[1]),
                    "top100_overlap": float(means[2]),
                },
                "migration_depth": {
                    str(depth): depth_after.count(depth)
                    for depth in sorted(set(depth_after))
                },
            }
        )
    return summarize_policy_steps(
        steps,
        costs,
        sum(exact.values()),
        len(ids),
    )


def simulate_periodic_policy(
    transitions: list[dict],
    exact_costs: list[dict],
    max_depth: int,
) -> dict:
    index = {
        (
            int(value["record_id"]),
            int(value["anchor_version"]),
            int(value["source_version"]),
        ): value
        for value in transitions
    }
    exact = {
        (int(value["record_id"]), int(value["target_version"])): float(
            value["attributed_gpu_ms"]
        )
        for value in exact_costs
    }
    ids = sorted({value[0] for value in index})
    anchors = {record_id: 0 for record_id in ids}
    costs = {"migration": 0.0, "router": 0.0, "exact": 0.0}
    steps = []
    for source_version in range(11):
        target_version = source_version + 1
        metrics = []
        exact_records = 0
        candidate_records = 0
        depths = []
        for record_id in ids:
            depth = source_version - anchors[record_id]
            if depth >= max_depth:
                anchors[record_id] = target_version
                costs["exact"] += exact[(record_id, target_version)]
                exact_records += 1
                metrics.append((1.0, 1.0, 1.0))
                depths.append(0)
            else:
                value = index[
                    (record_id, anchors[record_id], source_version)
                ]
                costs["migration"] += float(
                    value["attributed_gpu_ms"]["migration"]
                )
                candidate_records += 1
                metrics.append(transition_metric_values(value))
                depths.append(depth + 1)
        means = np.asarray(metrics).mean(axis=0)
        steps.append(
            {
                "target_version": target_version,
                "exact_records": exact_records,
                "candidate_records": candidate_records,
                "discarded_candidate_records": 0,
                "metrics": {
                    "cache_fidelity_q090": float(means[0]),
                    "score_cosine": float(means[1]),
                    "top100_overlap": float(means[2]),
                },
                "migration_depth": {
                    str(depth): depths.count(depth)
                    for depth in sorted(set(depths))
                },
            }
        )
    return summarize_policy_steps(
        steps,
        costs,
        sum(exact.values()),
        len(ids),
    )


def derive_balanced_policy(
    fit_transitions: list[dict],
    base_fraction: float,
    severity_amplitude: float,
    scheduler_seed: int = 0,
) -> BalancedLifecyclePolicy:
    severities = []
    for source_version in range(11):
        values = [
            float(value["cache_error"]["q090"])
            for value in fit_transitions
            if int(value["source_version"]) == source_version
            and int(value["migration_depth_after"]) == 1
        ]
        if not values:
            raise ValueError("balanced policy edge calibration is missing")
        severities.append(float(np.median(values)))
    order = np.argsort(np.argsort(np.asarray(severities)))
    percentiles = order / max(len(severities) - 1, 1)
    fractions = np.clip(
        base_fraction + severity_amplitude * (2 * percentiles - 1),
        0.15,
        0.25,
    )
    return BalancedLifecyclePolicy(
        max_migration_depth=4,
        exact_fractions=tuple(float(value) for value in fractions),
        edge_severities=tuple(severities),
        scheduler_seed=scheduler_seed,
    )


def simulate_balanced_policy(
    transitions: list[dict],
    exact_costs: list[dict],
    policy: BalancedLifecyclePolicy,
) -> dict:
    index = {
        (
            int(value["record_id"]),
            int(value["anchor_version"]),
            int(value["source_version"]),
        ): value
        for value in transitions
    }
    exact = {
        (int(value["record_id"]), int(value["target_version"])): float(
            value["attributed_gpu_ms"]
        )
        for value in exact_costs
    }
    ids = sorted({value[0] for value in index})
    states = {
        record_id: CacheLifecycleState.exact(record_id, 0)
        for record_id in ids
    }
    costs = {"migration": 0.0, "router": 0.0, "exact": 0.0}
    steps = []
    for source_version in range(11):
        target_version = source_version + 1
        ordered_states = tuple(states[record_id] for record_id in ids)
        decisions = policy.plan(ordered_states, target_version)
        metrics = []
        depths = []
        for state, decision in zip(
            ordered_states,
            decisions,
            strict=True,
        ):
            record_id = state.record_id
            if decision.action == "exact":
                costs["exact"] += exact[(record_id, target_version)]
                metrics.append((1.0, 1.0, 1.0))
            else:
                value = index[
                    (
                        record_id,
                        state.last_exact_version,
                        source_version,
                    )
                ]
                costs["migration"] += float(
                    value["attributed_gpu_ms"]["migration"]
                )
                metrics.append(transition_metric_values(value))
            next_state = policy.advance(state, decision)
            states[record_id] = next_state
            depths.append(next_state.migration_depth)
        means = np.asarray(metrics).mean(axis=0)
        exact_records = sum(
            value.action == "exact" for value in decisions
        )
        steps.append(
            {
                "target_version": target_version,
                "configured_exact_fraction": policy.exact_fraction(
                    source_version
                ),
                "exact_records": exact_records,
                "candidate_records": len(ids) - exact_records,
                "discarded_candidate_records": 0,
                "metrics": {
                    "cache_fidelity_q090": float(means[0]),
                    "score_cosine": float(means[1]),
                    "top100_overlap": float(means[2]),
                },
                "migration_depth": {
                    str(depth): depths.count(depth)
                    for depth in sorted(set(depths))
                },
            }
        )
    output = summarize_policy_steps(
        steps,
        costs,
        sum(exact.values()),
        len(ids),
    )
    exact_fractions = [
        value["exact_records"] / len(ids) for value in steps
    ]
    output["balance"] = {
        "minimum_step_exact_fraction": min(exact_fractions),
        "maximum_step_exact_fraction": max(exact_fractions),
        "step_exact_fraction_range": (
            max(exact_fractions) - min(exact_fractions)
        ),
        "exact_records_by_step": [
            value["exact_records"] for value in steps
        ],
    }
    return output


def simulate_all_migrate(
    transitions: list[dict],
    exact_costs: list[dict],
) -> dict:
    index = {
        (
            int(value["record_id"]),
            int(value["anchor_version"]),
            int(value["source_version"]),
        ): value
        for value in transitions
    }
    exact_ms = sum(
        float(value["attributed_gpu_ms"]) for value in exact_costs
    )
    ids = sorted({value[0] for value in index})
    costs = {"migration": 0.0, "router": 0.0, "exact": 0.0}
    steps = []
    for source_version in range(11):
        values = [
            index[(record_id, 0, source_version)]
            for record_id in ids
        ]
        metrics = np.asarray(
            [transition_metric_values(value) for value in values]
        ).mean(axis=0)
        costs["migration"] += sum(
            float(value["attributed_gpu_ms"]["migration"])
            for value in values
        )
        steps.append(
            {
                "target_version": source_version + 1,
                "exact_records": 0,
                "candidate_records": len(ids),
                "discarded_candidate_records": 0,
                "metrics": {
                    "cache_fidelity_q090": float(metrics[0]),
                    "score_cosine": float(metrics[1]),
                    "top100_overlap": float(metrics[2]),
                },
                "migration_depth": {
                    str(source_version + 1): len(ids)
                },
            }
        )
    return summarize_policy_steps(
        steps,
        costs,
        exact_ms,
        len(ids),
    )


def simulate_matched_random(
    transitions: list[dict],
    exact_costs: list[dict],
    max_depth: int,
    exact_counts: list[int],
    seed: int,
) -> dict:
    index = {
        (
            int(value["record_id"]),
            int(value["anchor_version"]),
            int(value["source_version"]),
        ): value
        for value in transitions
    }
    exact = {
        (int(value["record_id"]), int(value["target_version"])): float(
            value["attributed_gpu_ms"]
        )
        for value in exact_costs
    }
    ids = sorted({value[0] for value in index})
    anchors = {record_id: 0 for record_id in ids}
    rng = np.random.default_rng(seed)
    costs = {"migration": 0.0, "router": 0.0, "exact": 0.0}
    steps = []
    for source_version, target_exact in enumerate(exact_counts):
        target_version = source_version + 1
        mandatory = [
            record_id
            for record_id in ids
            if source_version - anchors[record_id] >= max_depth
        ]
        optional = [
            record_id
            for record_id in ids
            if record_id not in set(mandatory)
        ]
        needed = max(0, target_exact - len(mandatory))
        chosen = set(mandatory)
        if needed:
            chosen.update(
                int(value)
                for value in rng.choice(
                    optional,
                    min(needed, len(optional)),
                    replace=False,
                )
            )
        metrics = []
        candidate_records = 0
        discarded_records = 0
        depths = []
        for record_id in ids:
            depth = source_version - anchors[record_id]
            if record_id in chosen:
                if depth < max_depth:
                    value = index[
                        (record_id, anchors[record_id], source_version)
                    ]
                    costs["migration"] += float(
                        value["attributed_gpu_ms"]["migration"]
                    )
                    candidate_records += 1
                    discarded_records += 1
                costs["exact"] += exact[(record_id, target_version)]
                anchors[record_id] = target_version
                metrics.append((1.0, 1.0, 1.0))
                depths.append(0)
            else:
                value = index[
                    (record_id, anchors[record_id], source_version)
                ]
                costs["migration"] += float(
                    value["attributed_gpu_ms"]["migration"]
                )
                candidate_records += 1
                metrics.append(transition_metric_values(value))
                depths.append(depth + 1)
        means = np.asarray(metrics).mean(axis=0)
        steps.append(
            {
                "target_version": target_version,
                "exact_records": len(chosen),
                "candidate_records": candidate_records,
                "discarded_candidate_records": discarded_records,
                "metrics": {
                    "cache_fidelity_q090": float(means[0]),
                    "score_cosine": float(means[1]),
                    "top100_overlap": float(means[2]),
                },
                "migration_depth": {
                    str(depth): depths.count(depth)
                    for depth in sorted(set(depths))
                },
            }
        )
    return summarize_policy_steps(
        steps,
        costs,
        sum(exact.values()),
        len(ids),
    )


def run_policy_search(
    fit_payload: dict,
    selection_payload: dict,
) -> dict:
    started = time.perf_counter()
    fit_transitions = fit_payload["transitions"]
    transitions = selection_payload["transitions"]
    exact_costs = selection_payload["exact_costs"]
    candidates = []
    calibrations = {}
    feature_grid = (
        ("relative_correction", "q050"),
        ("absolute_log_norm_ratio", "q050"),
        ("absolute_log_norm_ratio", "q075"),
        ("absolute_log_norm_ratio", "q090"),
        ("absolute_log_norm_ratio", "q100"),
    )
    for feature_name, quantile in feature_grid:
        key = f"{feature_name}_{quantile}"
        calibration = fit_linear_sketch_calibration(
            fit_transitions,
            feature_name,
            quantile,
        )
        calibrations[key] = calibration.to_dict()
        values = [
            calibration.predict(
                int(value["source_version"]),
                int(value["migration_depth_after"]),
                float(value["sketch"][feature_name][quantile]),
            )
            for value in transitions
            if int(value["migration_depth_after"]) <= 4
        ]
        thresholds = sorted(
            {
                float(np.quantile(values, quantile_value))
                for quantile_value in (
                    0.5,
                    0.6,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                    1.0,
                )
            }
        )
        for max_depth in (2, 3, 4):
            for threshold in thresholds:
                policy = SketchLifecyclePolicy(
                    max_migration_depth=max_depth,
                    risk_threshold=threshold,
                    calibration=calibration,
                )
                result = simulate_sketch_policy(
                    transitions,
                    exact_costs,
                    policy,
                )
                candidates.append(
                    {
                        "selector": key,
                        "max_migration_depth": max_depth,
                        "risk_threshold": threshold,
                        "policy": policy.to_dict(),
                        "result": result,
                    }
                )
    periodic = []
    for max_depth in (2, 3, 4):
        periodic.append(
            {
                "selector": "periodic_depth_only",
                "max_migration_depth": max_depth,
                "result": simulate_periodic_policy(
                    transitions,
                    exact_costs,
                    max_depth,
                ),
            }
        )
    balanced_candidates = []
    for name, base_fraction, amplitude in (
        ("fixed_0.20", 0.20, 0.0),
        ("severity_bounded_0.02", 0.20, 0.02),
        ("severity_bounded_0.03", 0.20, 0.03),
        ("severity_bounded_0.05", 0.20, 0.05),
        ("fixed_0.25", 0.25, 0.0),
    ):
        balanced_policy = derive_balanced_policy(
            fit_transitions,
            base_fraction,
            amplitude,
        )
        balanced_candidates.append(
            {
                "selector": "balanced_age_severity_quota",
                "configuration": {
                    "name": name,
                    "base_exact_fraction": base_fraction,
                    "severity_amplitude": amplitude,
                    "minimum_exact_fraction": 0.15,
                    "maximum_exact_fraction": 0.25,
                    "priority": (
                        "maximum-depth mandatory, then greater migration "
                        "age, then stable hash"
                    ),
                },
                "policy": balanced_policy.to_dict(),
                "result": simulate_balanced_policy(
                    transitions,
                    exact_costs,
                    balanced_policy,
                ),
            }
        )
    eligible = [
        value
        for value in candidates
        if value["selector"].startswith("absolute_log_norm_ratio")
        and value["result"]["cost_ratio_to_all_exact"] <= 0.25
    ]
    if not eligible:
        raise RuntimeError("no adaptive selector satisfies the cost cap")
    selected = max(
        eligible,
        key=lambda value: (
            value["result"]["worst_view_fidelity"],
            -value["result"]["cost_ratio_to_all_exact"],
            -value["max_migration_depth"],
            value["selector"],
        ),
    )
    exact_counts = [
        int(value["exact_records"])
        for value in selected["result"]["steps"]
    ]
    random_trials = [
        simulate_matched_random(
            transitions,
            exact_costs,
            int(selected["max_migration_depth"]),
            exact_counts,
            seed,
        )
        for seed in range(1000)
    ]
    random_worst = np.asarray(
        [value["worst_view_fidelity"] for value in random_trials]
    )
    adaptive_pass = bool(
        selected["result"]["worst_view_fidelity"]
        > float(np.quantile(random_worst, 0.95))
    )
    balanced_eligible = [
        value
        for value in balanced_candidates
        if value["result"]["cost_ratio_to_all_exact"] <= 0.25
        and value["result"]["worst_view_fidelity"] >= 0.95
        and value["result"]["balance"][
            "maximum_step_exact_fraction"
        ]
        <= 0.25
        and value["result"]["balance"][
            "step_exact_fraction_range"
        ]
        <= 0.10
    ]
    if not balanced_eligible:
        raise RuntimeError("no balanced lifecycle policy satisfies its gate")
    recommended = max(
        balanced_eligible,
        key=lambda value: (
            value["result"]["worst_view_fidelity"],
            -value["result"]["cost_ratio_to_all_exact"],
            -value["result"]["balance"][
                "step_exact_fraction_range"
            ],
            value["configuration"]["name"],
        ),
    )
    return {
        "protocol": PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "phase": "policy-search",
        "status": "complete",
        "roles": {
            "fit": fit_payload["role"],
            "selection": selection_payload["role"],
        },
        "labels_used": False,
        "selection_rule": (
            "require 0.15-0.25 per-step exact refresh, no more than 0.10 "
            "step-fraction spread, at least 0.95 minimum step-wise "
            "cache/score/top100 fidelity, and no more than 0.25x "
            "measured-attributed all-exact GPU cost; among qualifying "
            "balanced points maximize worst-view fidelity then lower cost"
        ),
        "calibrations": calibrations,
        "adaptive_candidates": candidates,
        "periodic_candidates": periodic,
        "balanced_candidates": balanced_candidates,
        "all_migrate": simulate_all_migrate(
            transitions,
            exact_costs,
        ),
        "all_exact": {
            "cost_ratio_to_all_exact": 1.0,
            "worst_view_fidelity": 1.0,
            "exact_fraction": 1.0,
        },
        "selected_adaptive": selected,
        "matched_random": {
            "trials": 1000,
            "exact_counts_by_step": exact_counts,
            "worst_view_fidelity": {
                "median": float(np.median(random_worst)),
                "p95": float(np.quantile(random_worst, 0.95)),
                "maximum": float(random_worst.max()),
            },
            "cost_ratio_to_all_exact": {
                "median": float(
                    np.median(
                        [
                            value["cost_ratio_to_all_exact"]
                            for value in random_trials
                        ]
                    )
                )
            },
        },
        "adaptive_beats_matched_random_p95": adaptive_pass,
        "selected_threshold_diagnostic": selected,
        "recommended": recommended,
        "risk_selector_status": (
            "not frozen: its cumulative objective passed matched random, "
            "but it produced unacceptable per-step refresh waves"
        ),
        "fallback_reason": (
            "per-cache threshold routing is replaced by bounded "
            "age/deadline scheduling with program-level edge severity"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def correlations(
    correction: list[float],
    error: list[float],
) -> dict[str, float]:
    correction_array = np.asarray(correction)
    error_array = np.asarray(error)
    result = spearmanr(correction_array, error_array)
    rho = float(result.statistic)
    if not math.isfinite(rho):
        rho = 0.0
    return {
        "spearman_rho": rho,
        "pearson_r": float(np.corrcoef(correction_array, error_array)[0, 1]),
        "correction_min": float(correction_array.min()),
        "correction_median": float(np.median(correction_array)),
        "correction_max": float(correction_array.max()),
        "error_min": float(error_array.min()),
        "error_median": float(np.median(error_array)),
        "error_max": float(error_array.max()),
    }


@torch.inference_mode()
def run_fit_trajectory(
    args: argparse.Namespace,
    cfg,
    manifest: dict,
    fit_samples: list[dict],
) -> dict:
    device = torch.device(args.device)
    records_by_user = record_map(manifest)
    ordered = ordered_samples(fit_samples, records_by_user)
    batches = list(
        label_free_eval_batches(
            ordered,
            cfg.max_seq_len,
            args.batch_size,
        )
    )
    operator = DirectOldKVFusedOperator(**LAUNCH)
    recursive: list[JaggedMigratedKVBatch] = []
    observations = []
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        0,
        device,
    )
    started = time.perf_counter()
    for source_version in range(11):
        target_version = source_version + 1
        target_model = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            target_version,
            device,
        )
        program_cpu, descriptor = load_direct_oldkv_program(
            direct_program_path(
                args.runtime_dir,
                source_version,
                target_version,
            ),
            expected_source_version=f"theta{source_version}",
            expected_target_version=f"theta{target_version}",
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.num_heads * cfg.head_dim,
        )
        program = operator.prepare_program(program_cpu, device)
        next_recursive = []
        edge_records = 0
        for batch_index, (
            selected,
            _,
            prefix_cpu,
            _,
        ) in enumerate(batches):
            prefix = move_batch(prefix_cpu, device)
            ids = record_ids(selected, records_by_user)
            exact_source = exact_batch(
                source_model,
                prefix,
                ids,
                source_version,
            )
            exact_target = exact_batch(
                target_model,
                prefix,
                ids,
                target_version,
            )
            recursive_source = (
                exact_source if source_version == 0 else recursive[batch_index]
            )
            exact_candidate = execute_direct(
                operator,
                program,
                exact_source,
                target_version,
            )
            recursive_candidate = execute_direct(
                operator,
                program,
                recursive_source,
                target_version,
            )
            one_hop_correction = aggregate(
                relative_cache_values(exact_candidate, exact_source)
            )
            one_hop_error = aggregate(
                relative_cache_values(exact_candidate, exact_target)
            )
            recursive_correction = aggregate(
                relative_cache_values(
                    recursive_candidate,
                    recursive_source,
                )
            )
            recursive_error = aggregate(
                relative_cache_values(
                    recursive_candidate,
                    exact_target,
                )
            )
            propagated = aggregate(
                relative_cache_values(
                    recursive_candidate,
                    exact_candidate,
                )
            )
            previous_error = (
                {
                    name: [0.0] * len(ids)
                    for name in one_hop_error
                }
                if source_version == 0
                else aggregate(
                    relative_cache_values(
                        recursive_source,
                        exact_source,
                    )
                )
            )
            for row, record_id in enumerate(ids):
                observations.append(
                    {
                        "record_id": record_id,
                        "source_version": source_version,
                        "target_version": target_version,
                        "program_sha256": descriptor["sha256"],
                        "one_hop_correction": {
                            name: values[row]
                            for name, values in one_hop_correction.items()
                        },
                        "one_hop_error": {
                            name: values[row]
                            for name, values in one_hop_error.items()
                        },
                        "recursive_correction": {
                            name: values[row]
                            for name, values in recursive_correction.items()
                        },
                        "recursive_error": {
                            name: values[row]
                            for name, values in recursive_error.items()
                        },
                        "previous_error": {
                            name: values[row]
                            for name, values in previous_error.items()
                        },
                        "propagated_error": {
                            name: values[row]
                            for name, values in propagated.items()
                        },
                    }
                )
            next_recursive.append(recursive_candidate)
            edge_records += len(ids)
            del (
                prefix,
                exact_source,
                exact_target,
                exact_candidate,
            )
        recursive = next_recursive
        print(
            json.dumps(
                {
                    "phase": "fit",
                    "edge": (
                        f"theta{source_version}->theta{target_version}"
                    ),
                    "records": edge_records,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del source_model, program_cpu, program
        source_model = target_model
        gc.collect()
        torch.cuda.empty_cache()
    del source_model, recursive
    calibration = {}
    diagnostics = {}
    for quantile in LAYER_QUANTILES:
        name = f"q{int(quantile * 100):03d}"
        correction = [
            float(value["one_hop_correction"][name])
            for value in observations
        ]
        one_hop_error = [
            float(value["one_hop_error"][name])
            for value in observations
        ]
        ratios = [
            float(value["propagated_error"][name])
            / max(float(value["previous_error"][name]), 1e-8)
            for value in observations
            if float(value["previous_error"][name]) >= 1e-5
        ]
        fitted = fit_monotone_risk_calibration(
            correction,
            one_hop_error,
            ratios,
            bins=8,
            quantile=0.9,
        )
        calibration[name] = fitted.to_dict()
        diagnostics[name] = {
            "one_hop": correlations(correction, one_hop_error),
            "recursive_runtime_correction": correlations(
                [
                    float(value["recursive_correction"][name])
                    for value in observations
                ],
                [
                    float(value["recursive_error"][name])
                    for value in observations
                ],
            ),
            "propagation_ratio_samples": len(ratios),
            "propagation_ratio_median": (
                float(np.median(ratios)) if ratios else None
            ),
            "propagation_ratio_p90": (
                float(np.quantile(ratios, 0.9)) if ratios else None
            ),
            "propagation_ratio_max": max(ratios) if ratios else None,
        }
    return {
        "protocol": PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "phase": "fit",
        "status": "complete",
        "role": "fit",
        "labels_used": False,
        "records": len(fit_samples),
        "edges": 11,
        "observations": observations,
        "calibrations": calibration,
        "diagnostics": diagnostics,
        "layer_quantiles": list(LAYER_QUANTILES),
        "launch": LAUNCH,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.cuda.set_device(torch.device(args.device))
    seed_everything(args.seed)
    (
        blueprint,
        manifest,
        training,
        cfg,
        _,
        roles,
    ) = validate_frozen_inputs(args)
    compiler = json.loads(Path(args.adjacent_compiler).read_text())
    if (
        compiler.get("status") != "complete"
        or compiler.get("experiment_protocol") != EXPERIMENT_PROTOCOL
        or len(compiler.get("pairs", [])) != 11
    ):
        raise ValueError("Stage 4.6 adjacent compiler artifact differs")
    if args.phase == "fit":
        payload = run_fit_trajectory(
            args,
            cfg,
            manifest,
            roles["fit"],
        )
        output = args.fit_output
    elif args.phase == "fit-transitions":
        payload = run_transition_dag(
            args,
            cfg,
            manifest,
            roles["fit"],
            "fit",
            False,
        )
        output = args.fit_transition_output
    elif args.phase == "selection":
        payload = run_transition_dag(
            args,
            cfg,
            manifest,
            roles["program_selection"],
            "program_selection",
            True,
        )
        output = args.selection_output
    else:
        fit_transitions = json.loads(
            Path(args.fit_transition_output).read_text()
        )
        selection_transitions = json.loads(
            Path(args.selection_output).read_text()
        )
        if (
            fit_transitions.get("phase") != "fit-transitions"
            or selection_transitions.get("phase")
            != "program_selection-transitions"
            or fit_transitions.get("status") != "complete"
            or selection_transitions.get("status") != "complete"
        ):
            raise ValueError("Stage 4.6 transition artifacts differ")
        payload = run_policy_search(
            fit_transitions,
            selection_transitions,
        )
        payload["transition_inputs"] = {
            "fit": {
                "path": args.fit_transition_output,
                "sha256": sha256(args.fit_transition_output),
            },
            "program_selection": {
                "path": args.selection_output,
                "sha256": sha256(args.selection_output),
            },
        }
        output = args.policy_search_output
    payload["repository_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    payload["inputs"] = {
        "blueprint": {
            "path": args.blueprint,
            "sha256": sha256(args.blueprint),
            "protocol": blueprint["protocol"],
        },
        "workload_manifest": {
            "path": args.workload_manifest,
            "sha256": sha256(args.workload_manifest),
            "content_sha256": manifest["content_sha256"],
        },
        "training_result": {
            "path": args.training_result,
            "sha256": sha256(args.training_result),
            "protocol": training["protocol"],
        },
        "adjacent_compiler": {
            "path": args.adjacent_compiler,
            "sha256": sha256(args.adjacent_compiler),
            "protocol": compiler["protocol"],
        },
    }
    save_json(payload, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "phase": payload["phase"],
                "output": output,
                "elapsed_seconds": payload["elapsed_seconds"],
                "diagnostics": payload.get("diagnostics"),
            }
        )
    )


if __name__ == "__main__":
    main()
