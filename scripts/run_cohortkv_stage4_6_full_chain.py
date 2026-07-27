from __future__ import annotations

import argparse
import gc
import json
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
    sha256,
    validate_frozen_inputs,
)
from evaluate_cohortkv_stage4_6_lifecycle import (
    DEFAULT_POLICY_SEARCH_OUTPUT,
    LAUNCH,
    exact_batch,
    execute_direct,
    ordered_samples,
    record_ids,
    record_map,
    timed_cuda,
)
from motivation_validity import move_batch, ranking_metrics, seed_everything

from hstu_kvcache.migration import (
    BalancedLifecyclePolicy,
    CacheLifecycleState,
    JaggedMigratedKVBatch,
    aggregate_layer_values,
    assemble_jagged_rows,
    relative_cache_values,
    select_jagged_rows,
    unpack_jagged_cache,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.streaming import load_checkpoint_model
from hstu_kvcache.utils import save_json

PROTOCOL = "cohortkv_single_config_stage4_6_recursive_chain_v1"
DEFAULT_CERTIFICATE_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_6_certificate_chain_seed0.json"
)
DEFAULT_FULL_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_6_full_chain_seed0.json"
)
CERTIFICATE_CONTRACT = {
    "minimum_step_cache_fidelity_q090": 0.90,
    "minimum_step_score_cosine": 0.995,
    "minimum_step_top100_overlap": 0.95,
    "maximum_cumulative_cost_ratio": 0.30,
    "minimum_step_exact_fraction": 0.15,
    "maximum_step_exact_fraction": 0.25,
    "maximum_step_exact_fraction_range": 0.10,
    "maximum_step_cost_ratio": 0.35,
    "maximum_migration_depth": 4,
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
    parser.add_argument("--policy-search", default=DEFAULT_POLICY_SEARCH_OUTPUT)
    parser.add_argument(
        "--certificate-output",
        default=DEFAULT_CERTIFICATE_OUTPUT,
    )
    parser.add_argument(
        "--certificate-result",
        default=DEFAULT_CERTIFICATE_OUTPUT,
    )
    parser.add_argument("--full-output", default=DEFAULT_FULL_OUTPUT)
    parser.add_argument(
        "--role",
        choices=("certificate", "all"),
        default="certificate",
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


def subset_prefix(
    prefix: dict,
    rows: tuple[int, ...],
    device: torch.device,
) -> dict:
    if not rows:
        raise ValueError("exact refresh subset is empty")
    index = torch.tensor(
        rows,
        dtype=torch.long,
        device=prefix["lengths"].device,
    )
    lengths = prefix["lengths"].index_select(0, index)
    width = int(lengths.max())
    return {
        "item_ids": prefix["item_ids"].index_select(0, index)[
            :, :width
        ].to(device),
        "behaviors": prefix["behaviors"].index_select(0, index)[
            :, :width
        ].to(device),
        "time_deltas": prefix["time_deltas"].index_select(0, index)[
            :, :width
        ].to(device),
        "lengths": lengths.to(device),
    }


def publish_target(
    layout: JaggedMigratedKVBatch,
    candidate: JaggedMigratedKVBatch | None,
    accepted_candidate_rows: tuple[int, ...],
    exact: JaggedMigratedKVBatch | None,
    target_version: int,
) -> JaggedMigratedKVBatch:
    if (
        candidate is not None
        and len(accepted_candidate_rows) == layout.batch_size
        and exact is None
        and candidate.record_ids == layout.record_ids
    ):
        return JaggedMigratedKVBatch(
            record_ids=candidate.record_ids,
            migration_anchor_version=f"theta{target_version}",
            served_kv_target=f"theta{target_version}",
            k=candidate.k,
            v=candidate.v,
            lengths=candidate.lengths,
            offsets=candidate.offsets,
        )
    if (
        exact is not None
        and exact.record_ids == layout.record_ids
        and not accepted_candidate_rows
    ):
        return exact
    sources = []
    if accepted_candidate_rows:
        if candidate is None:
            raise RuntimeError("migration candidate is missing")
        sources.append(
            select_jagged_rows(
                candidate,
                accepted_candidate_rows,
            )
        )
    if exact is not None:
        sources.append(exact)
    return assemble_jagged_rows(
        layout,
        tuple(sources),
        target_version,
    )


@torch.inference_mode()
def cache_hidden_scores(
    model,
    cache: JaggedMigratedKVBatch,
    suffix: dict,
    all_items: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden, _ = model.forward_with_cache(
        unpack_jagged_cache(cache),
        suffix["item_ids"],
        suffix["behaviors"],
        suffix["time_deltas"],
    )
    hidden = hidden[:, 0]
    candidates = all_items.unsqueeze(0).expand(cache.batch_size, -1)
    return hidden, model.item_emb.score(hidden, candidates)


def semantic_pair(
    actual_hidden: torch.Tensor,
    actual_scores: torch.Tensor,
    exact_hidden: torch.Tensor,
    exact_scores: torch.Tensor,
) -> dict[str, list[float]]:
    hidden_cosine = torch.nn.functional.cosine_similarity(
        actual_hidden.float(),
        exact_hidden.float(),
        dim=-1,
    )
    score_cosine = torch.nn.functional.cosine_similarity(
        actual_scores.float(),
        exact_scores.float(),
        dim=-1,
    )
    topk = min(100, actual_scores.shape[1])
    actual_top = torch.topk(actual_scores, topk, dim=1).indices
    exact_top = torch.topk(exact_scores, topk, dim=1).indices
    overlap = (
        (actual_top.unsqueeze(2) == exact_top.unsqueeze(1))
        .any(dim=2)
        .float()
        .mean(dim=1)
    )
    return {
        "hidden_cosine": hidden_cosine.cpu().tolist(),
        "score_cosine": score_cosine.cpu().tolist(),
        "top100_overlap": overlap.cpu().tolist(),
    }


def task_metrics(
    scores: torch.Tensor,
    positives: list[int],
) -> dict[str, float]:
    values = ranking_metrics(scores, positives)
    return {
        "mean_rank": float(values["mean_rank"]),
        "catalog_auc": float(values["auc"]),
        "ndcg@100": float(values["ndcg@100"]),
        "hit@100": float(values["hit@100"]),
    }


def summarize_tasks(records: list[dict]) -> dict | None:
    selected = [
        value
        for value in records
        if value["evaluation_role"] == "final_test"
        and "task" in value
    ]
    if not selected:
        return None
    metrics = ("mean_rank", "catalog_auc", "ndcg@100", "hit@100")
    output = {
        "records": len(selected),
        "mixed": {},
        "all_exact": {},
        "reuse": {},
        "paired_difference_mixed_minus_exact": {},
        "recovery_from_reuse_to_exact": {},
    }
    for metric in metrics:
        mixed = np.asarray(
            [value["task"]["mixed"][metric] for value in selected]
        )
        exact = np.asarray(
            [value["task"]["all_exact"][metric] for value in selected]
        )
        reuse = np.asarray(
            [value["task"]["reuse"][metric] for value in selected]
        )
        output["mixed"][metric] = float(mixed.mean())
        output["all_exact"][metric] = float(exact.mean())
        output["reuse"][metric] = float(reuse.mean())
        output["paired_difference_mixed_minus_exact"][metric] = float(
            (mixed - exact).mean()
        )
        denominator = float(
            (reuse - exact).mean()
            if metric == "mean_rank"
            else (exact - reuse).mean()
        )
        numerator = float(
            (reuse - mixed).mean()
            if metric == "mean_rank"
            else (mixed - reuse).mean()
        )
        output["recovery_from_reuse_to_exact"][metric] = (
            numerator / denominator
            if abs(denominator) >= 1e-6
            else None
        )
    return output


def summarize_step(
    target_version: int,
    records: list[dict],
    costs: dict[str, float],
    exact_reference_ms: float,
    scheduler_cpu_ms: float,
) -> dict:
    metrics = (
        "cache_error_q090",
        "cache_fidelity_q090",
        "hidden_cosine",
        "score_cosine",
        "top100_overlap",
    )
    states = [value["state_after"] for value in records]
    return {
        "target_version": target_version,
        "records": len(records),
        "actions": {
            "migrate": sum(
                value["decision"]["action"] == "migrate"
                for value in records
            ),
            "exact": sum(
                value["decision"]["action"] == "exact"
                for value in records
            ),
            "discarded_migration_candidates": sum(
                value["decision"]["action"] == "exact"
                and value["decision"]["candidate_evaluated"]
                for value in records
            ),
        },
        "migration_depth": {
            str(depth): sum(
                int(value["migration_depth"]) == depth
                for value in states
            )
            for depth in sorted(
                {int(value["migration_depth"]) for value in states}
            )
        },
        "label_free_metrics": {
            metric: float(
                np.mean([value["metrics"][metric] for value in records])
            )
            for metric in metrics
        },
        "task_metrics": summarize_tasks(records),
        "scheduler_cpu_ms": scheduler_cpu_ms,
        "gpu_cost_ms": {
            **costs,
            "mixed_policy": sum(costs.values()),
            "all_exact_reference": exact_reference_ms,
            "ratio_to_all_exact": (
                sum(costs.values()) / exact_reference_ms
            ),
        },
        "lineage": [
            {
                "record_id": value["record_id"],
                "decision": value["decision"],
                "state_before": value["state_before"],
                "state_after": value["state_after"],
            }
            for value in records
        ],
    }


def certificate_gate(steps: list[dict]) -> dict:
    cumulative_mixed = sum(
        value["gpu_cost_ms"]["mixed_policy"] for value in steps
    )
    cumulative_exact = sum(
        value["gpu_cost_ms"]["all_exact_reference"] for value in steps
    )
    exact_fractions = [
        value["actions"]["exact"] / value["records"] for value in steps
    ]
    observed = {
        "minimum_step_cache_fidelity_q090": min(
            value["label_free_metrics"]["cache_fidelity_q090"]
            for value in steps
        ),
        "minimum_step_score_cosine": min(
            value["label_free_metrics"]["score_cosine"]
            for value in steps
        ),
        "minimum_step_top100_overlap": min(
            value["label_free_metrics"]["top100_overlap"]
            for value in steps
        ),
        "cumulative_cost_ratio": cumulative_mixed / cumulative_exact,
        "minimum_step_exact_fraction": min(exact_fractions),
        "maximum_step_exact_fraction": max(exact_fractions),
        "step_exact_fraction_range": (
            max(exact_fractions) - min(exact_fractions)
        ),
        "maximum_step_cost_ratio": max(
            value["gpu_cost_ms"]["ratio_to_all_exact"]
            for value in steps
        ),
        "maximum_migration_depth": max(
            int(depth)
            for value in steps
            for depth in value["migration_depth"]
        ),
    }
    checks = {
        "cache_fidelity": (
            observed["minimum_step_cache_fidelity_q090"]
            >= CERTIFICATE_CONTRACT[
                "minimum_step_cache_fidelity_q090"
            ]
        ),
        "score_cosine": (
            observed["minimum_step_score_cosine"]
            >= CERTIFICATE_CONTRACT["minimum_step_score_cosine"]
        ),
        "top100_overlap": (
            observed["minimum_step_top100_overlap"]
            >= CERTIFICATE_CONTRACT["minimum_step_top100_overlap"]
        ),
        "cumulative_cost": (
            observed["cumulative_cost_ratio"]
            <= CERTIFICATE_CONTRACT["maximum_cumulative_cost_ratio"]
        ),
        "minimum_step_exact_fraction": (
            observed["minimum_step_exact_fraction"]
            >= CERTIFICATE_CONTRACT["minimum_step_exact_fraction"]
        ),
        "maximum_step_exact_fraction": (
            observed["maximum_step_exact_fraction"]
            <= CERTIFICATE_CONTRACT["maximum_step_exact_fraction"]
        ),
        "step_exact_fraction_range": (
            observed["step_exact_fraction_range"]
            <= CERTIFICATE_CONTRACT[
                "maximum_step_exact_fraction_range"
            ]
        ),
        "maximum_step_cost_ratio": (
            observed["maximum_step_cost_ratio"]
            <= CERTIFICATE_CONTRACT["maximum_step_cost_ratio"]
        ),
        "migration_depth": (
            observed["maximum_migration_depth"]
            <= CERTIFICATE_CONTRACT["maximum_migration_depth"]
        ),
    }
    return {
        "contract": CERTIFICATE_CONTRACT,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }


@torch.inference_mode()
def run_chain(
    args: argparse.Namespace,
    cfg,
    manifest: dict,
    samples: list[dict],
    policy: BalancedLifecyclePolicy,
    role: str,
) -> dict:
    device = torch.device(args.device)
    records_by_user = record_map(manifest)
    manifest_by_id = {
        int(value["record_id"]): value
        for value in manifest["records"]
    }
    ordered = ordered_samples(samples, records_by_user)
    batches = list(
        label_free_eval_batches(
            ordered,
            cfg.max_seq_len,
            args.batch_size,
        )
    )
    all_items = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    )
    initial_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        0,
        device,
    )
    cache_batches = []
    states = {}
    for selected, _, prefix_cpu, _ in batches:
        prefix = move_batch(prefix_cpu, device)
        ids = record_ids(selected, records_by_user)
        cache_batches.append(
            exact_batch(initial_model, prefix, ids, 0)
        )
        for record_id in ids:
            states[record_id] = CacheLifecycleState.exact(record_id, 0)
    del initial_model
    gc.collect()
    torch.cuda.empty_cache()
    operator = DirectOldKVFusedOperator(**LAUNCH)
    steps = []
    started = time.perf_counter()
    for source_version in range(11):
        target_version = source_version + 1
        model = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            target_version,
            device,
        )
        program_cpu, program_descriptor = load_direct_oldkv_program(
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
        scheduler_started = time.perf_counter()
        planned = policy.plan(
            tuple(states[record_id] for record_id in sorted(states)),
            target_version,
        )
        scheduler_cpu_ms = (
            time.perf_counter() - scheduler_started
        ) * 1000
        planned_by_record = {
            value.record_id: value for value in planned
        }
        warm_rows = tuple(
            row
            for row, record_id in enumerate(cache_batches[0].record_ids)
            if planned_by_record[record_id].action == "migrate"
        )
        if warm_rows:
            warm_source = select_jagged_rows(
                cache_batches[0],
                warm_rows,
            )
            warm_candidate = execute_direct(
                operator,
                program,
                warm_source,
                target_version,
            )
            torch.cuda.synchronize(device)
            del warm_source, warm_candidate
        step_records = []
        costs = {
            "migration": 0.0,
            "router": 0.0,
            "exact_refresh": 0.0,
            "publication": 0.0,
        }
        exact_reference_ms = 0.0
        for batch_index, (
            selected,
            _,
            prefix_cpu,
            suffix_cpu,
        ) in enumerate(batches):
            source = cache_batches[batch_index]
            source_states = [states[value] for value in source.record_ids]
            decisions = {
                row: planned_by_record[state.record_id]
                for row, state in enumerate(source_states)
            }
            candidate_rows = tuple(
                row
                for row, decision in decisions.items()
                if decision.action == "migrate"
            )
            candidate = None
            candidate_row_by_batch_row = {}
            if candidate_rows:
                if candidate_rows == tuple(range(source.batch_size)):
                    candidate_source = source
                    prepare_ms = 0.0
                else:
                    candidate_source, prepare_ms = timed_cuda(
                        partial(
                            select_jagged_rows,
                            source,
                            candidate_rows,
                        ),
                        device,
                    )
                candidate, migration_ms = timed_cuda(
                    partial(
                        execute_direct,
                        operator,
                        program,
                        candidate_source,
                        target_version,
                    ),
                    device,
                )
                costs["migration"] += prepare_ms + migration_ms
                for candidate_row, batch_row in enumerate(candidate_rows):
                    candidate_row_by_batch_row[batch_row] = candidate_row
                if candidate_source is not source:
                    del candidate_source
            exact_rows = tuple(
                row
                for row, decision in decisions.items()
                if decision.action == "exact"
            )
            prefix = move_batch(prefix_cpu, device)
            exact_selected = None
            if exact_rows:
                exact_prefix, exact_prepare_ms = timed_cuda(
                    partial(
                        subset_prefix,
                        prefix,
                        exact_rows,
                        device,
                    ),
                    device,
                )
                exact_ids = tuple(
                    source.record_ids[row] for row in exact_rows
                )
                exact_selected, exact_ms = timed_cuda(
                    partial(
                        exact_batch,
                        model,
                        exact_prefix,
                        exact_ids,
                        target_version,
                    ),
                    device,
                )
                costs["exact_refresh"] += exact_prepare_ms + exact_ms
                del exact_prefix
            accepted_candidate_rows = tuple(
                candidate_row_by_batch_row[row]
                for row, decision in decisions.items()
                if decision.action == "migrate"
            )

            target, publication_ms = timed_cuda(
                partial(
                    publish_target,
                    source,
                    candidate,
                    accepted_candidate_rows,
                    exact_selected,
                    target_version,
                ),
                device,
            )
            costs["publication"] += publication_ms
            exact_reference, reference_ms = timed_cuda(
                partial(
                    exact_batch,
                    model,
                    prefix,
                    source.record_ids,
                    target_version,
                ),
                device,
            )
            exact_reference_ms += reference_ms
            suffix = move_batch(suffix_cpu, device)
            exact_hidden, exact_scores = cache_hidden_scores(
                model,
                exact_reference,
                suffix,
                all_items,
            )
            mixed_hidden, mixed_scores = cache_hidden_scores(
                model,
                target,
                suffix,
                all_items,
            )
            semantic = semantic_pair(
                mixed_hidden,
                mixed_scores,
                exact_hidden,
                exact_scores,
            )
            cache_error = aggregate_layer_values(
                relative_cache_values(target, exact_reference),
                0.9,
            ).cpu().tolist()
            reuse_scores = None
            if role == "all":
                _, reuse_scores = cache_hidden_scores(
                    model,
                    source,
                    suffix,
                    all_items,
                )
            for row, sample in enumerate(selected):
                record_id = source.record_ids[row]
                decision = decisions[row]
                next_state = policy.advance(
                    source_states[row],
                    decision,
                )
                states[record_id] = next_state
                value = {
                    "record_id": record_id,
                    "user_id": int(sample["history"]["user_id"]),
                    "evaluation_role": manifest_by_id[record_id][
                        "evaluation_role"
                    ],
                    "decision": {
                        **decision.to_dict(),
                        "edge_severity": policy.edge_severities[
                            source_version
                        ],
                        "configured_exact_fraction": (
                            policy.exact_fractions[source_version]
                        ),
                        "program_sha256": program_descriptor["sha256"],
                    },
                    "state_before": source_states[row].to_dict(),
                    "state_after": next_state.to_dict(),
                    "metrics": {
                        "cache_error_q090": float(cache_error[row]),
                        "cache_fidelity_q090": max(
                            0.0,
                            1.0 - float(cache_error[row]),
                        ),
                        "hidden_cosine": float(
                            semantic["hidden_cosine"][row]
                        ),
                        "score_cosine": float(
                            semantic["score_cosine"][row]
                        ),
                        "top100_overlap": float(
                            semantic["top100_overlap"][row]
                        ),
                    },
                }
                if (
                    role == "all"
                    and value["evaluation_role"] == "final_test"
                    and reuse_scores is not None
                ):
                    value["task"] = {
                        "mixed": task_metrics(
                            mixed_scores[row],
                            sample["pos_items"],
                        ),
                        "all_exact": task_metrics(
                            exact_scores[row],
                            sample["pos_items"],
                        ),
                        "reuse": task_metrics(
                            reuse_scores[row],
                            sample["pos_items"],
                        ),
                    }
                step_records.append(value)
            cache_batches[batch_index] = target
            del (
                source,
                target,
                exact_reference,
                exact_hidden,
                exact_scores,
                mixed_hidden,
                mixed_scores,
                prefix,
                suffix,
            )
            if reuse_scores is not None:
                del reuse_scores
            if candidate is not None:
                del candidate
            if exact_selected is not None:
                del exact_selected
        step = summarize_step(
            target_version,
            step_records,
            costs,
            exact_reference_ms,
            scheduler_cpu_ms,
        )
        step["configured_exact_fraction"] = policy.exact_fractions[
            source_version
        ]
        steps.append(step)
        print(
            json.dumps(
                {
                    "role": role,
                    "target_version": target_version,
                    "actions": step["actions"],
                    "label_free_metrics": step["label_free_metrics"],
                    "cost_ratio": step["gpu_cost_ms"][
                        "ratio_to_all_exact"
                    ],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del model, program_cpu, program
        gc.collect()
        torch.cuda.empty_cache()
    cumulative_mixed = sum(
        value["gpu_cost_ms"]["mixed_policy"] for value in steps
    )
    cumulative_exact = sum(
        value["gpu_cost_ms"]["all_exact_reference"] for value in steps
    )
    return {
        "protocol": PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "status": "complete",
        "role": role,
        "labels_used_for_routing": False,
        "records": len(samples),
        "edges": 11,
        "policy": policy.to_dict(),
        "steps": steps,
        "cumulative_gpu_cost": {
            "mixed_policy_ms": cumulative_mixed,
            "all_exact_reference_ms": cumulative_exact,
            "ratio_to_all_exact": cumulative_mixed / cumulative_exact,
        },
        "terminal_state": {
            "served_versions": sorted(
                {value.served_version for value in states.values()}
            ),
            "maximum_migration_depth": max(
                value.migration_depth for value in states.values()
            ),
            "state_kind_counts": {
                kind: sum(
                    value.state_kind == kind for value in states.values()
                )
                for kind in ("exact", "migrated")
            },
        },
        "scheduler_cpu_ms": sum(
            value["scheduler_cpu_ms"] for value in steps
        ),
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
        samples,
        roles,
    ) = validate_frozen_inputs(args)
    compiler = json.loads(Path(args.adjacent_compiler).read_text())
    policy_search = json.loads(Path(args.policy_search).read_text())
    recommended = policy_search.get("recommended")
    if (
        compiler.get("status") != "complete"
        or compiler.get("experiment_protocol") != EXPERIMENT_PROTOCOL
        or len(compiler.get("pairs", [])) != 11
        or policy_search.get("phase") != "policy-search"
        or policy_search.get("status") != "complete"
        or not isinstance(recommended, dict)
        or recommended.get("selector")
        != "balanced_age_severity_quota"
        or "policy" not in recommended
    ):
        raise ValueError("Stage 4.6 compiler or policy artifact differs")
    policy = BalancedLifecyclePolicy.from_dict(recommended["policy"])
    certificate = None
    if args.role == "all":
        certificate = json.loads(
            Path(args.certificate_result).read_text()
        )
        if (
            certificate.get("role") != "certificate"
            or certificate.get("status") != "complete"
            or certificate.get("certificate", {}).get("passed") is not True
            or certificate.get("policy") != policy.to_dict()
        ):
            raise ValueError("Stage 4.6 certificate does not freeze this policy")
    selected_samples = (
        roles["certificate"] if args.role == "certificate" else samples
    )
    payload = run_chain(
        args,
        cfg,
        manifest,
        selected_samples,
        policy,
        args.role,
    )
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
        "policy_search": {
            "path": args.policy_search,
            "sha256": sha256(args.policy_search),
            "protocol": policy_search["protocol"],
        },
    }
    if certificate is not None:
        payload["inputs"]["certificate"] = {
            "path": args.certificate_result,
            "sha256": sha256(args.certificate_result),
            "protocol": certificate["protocol"],
        }
    if args.role == "certificate":
        payload["certificate"] = certificate_gate(payload["steps"])
        output = args.certificate_output
    else:
        payload["certificate"] = None
        output = args.full_output
    save_json(payload, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "role": args.role,
                "output": output,
                "cumulative_gpu_cost": payload["cumulative_gpu_cost"],
                "certificate": payload["certificate"],
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
