from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from layerwise_validity import load_model, reconstruct_eval_samples
from low_rank_migration_search import (
    evaluate,
    fit_adapter,
    split_samples,
)
from motivation_validity import build_streaming_plan, seed_everything

from hstu_kvcache.migration import (
    compile_low_rank_cache_adapter,
    compile_projection_cache_adapter,
)
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json

FIDELITY_TARGETS = (0.5, 0.75, 0.9)
BUDGET_TARGETS = (0.15, 0.3, 0.6, 0.85)
STRUCTURAL_MIN_SAVINGS = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-t", type=int, default=11)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--fit-users", type=int, default=40)
    parser.add_argument("--probe-users", type=int, default=60)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 96],
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--protocol",
        default="cohort_tiered_migration_discovery_v1",
    )
    parser.add_argument(
        "--study-stage",
        default="motivation_selected_seed_discovery",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def production_names(configs: dict[str, dict]) -> list[str]:
    return [
        name
        for name in configs
        if name in {"cheap_prepacked", "recompute"}
        or name.startswith("adapter_rank_")
        or name.startswith("residual_p")
    ]


def rank_value(name: str) -> int:
    if name == "cheap_prepacked":
        return 0
    return int(name.rsplit("_", 1)[1])


def projection_candidate(
    configs: dict[str, dict],
    target: float,
) -> tuple[str, float] | None:
    projection_names = [
        name
        for name in production_names(configs)
        if name == "cheap_prepacked" or name.startswith("adapter_rank_")
    ]
    eligible = [
        name
        for name in projection_names
        if configs[name]["cache_fidelity_recovery"] >= target
    ]
    if not eligible:
        return None
    selected = min(eligible, key=rank_value)
    class_cost = statistics.median(
        configs[name]["migration_ratio_to_recompute"]
        for name in projection_names
    )
    return selected, class_cost


def select_fidelity_actions(probe: dict, test: dict) -> dict[str, dict]:
    output = {}
    probe_configs = probe["configs"]
    test_configs = test["configs"]
    structural_names = [
        name
        for name in production_names(probe_configs)
        if name.startswith("residual_p")
        or name == "recompute"
    ]
    full_cost = probe_configs["recompute"][
        "migration_ratio_to_recompute"
    ]
    for target in FIDELITY_TARGETS:
        eligible = [
            (
                name,
                probe_configs[name]["migration_ratio_to_recompute"],
            )
            for name in structural_names
            if probe_configs[name]["cache_fidelity_recovery"] >= target
            and (
                name == "recompute"
                or probe_configs[name]["migration_ratio_to_recompute"]
                <= (1.0 - STRUCTURAL_MIN_SAVINGS) * full_cost
            )
        ]
        projection = projection_candidate(probe_configs, target)
        if projection is not None:
            eligible.append(projection)
        selected, planning_cost = min(
            eligible,
            key=lambda value: (
                value[1],
                -probe_configs[value[0]]["cache_fidelity_recovery"],
                value[0],
            ),
        )
        output[str(target)] = {
            "selected": selected,
            "target": target,
            "planning_cost_ratio": planning_cost,
            "probe": probe_configs[selected],
            "test": test_configs[selected],
        }
    return output


def select_budget_actions(probe: dict, test: dict) -> dict[str, dict]:
    output = {}
    probe_configs = probe["configs"]
    test_configs = test["configs"]
    candidates = production_names(probe_configs)
    projection_names = [
        name
        for name in candidates
        if name == "cheap_prepacked" or name.startswith("adapter_rank_")
    ]
    projection_cost = statistics.median(
        probe_configs[name]["migration_ratio_to_recompute"]
        for name in projection_names
    )
    effective_cost = {
        name: (
            projection_cost
            if name in projection_names
            else probe_configs[name]["migration_ratio_to_recompute"]
        )
        for name in candidates
    }
    for budget in BUDGET_TARGETS:
        eligible = [
            name
            for name in candidates
            if effective_cost[name] <= budget
        ]
        selected = max(
            eligible,
            key=lambda name: (
                probe_configs[name]["cache_fidelity_recovery"],
                -effective_cost[name],
                name,
            ),
            default="reuse",
        )
        output[str(budget)] = {
            "selected": selected,
            "budget": budget,
            "planning_cost_ratio": effective_cost.get(selected, 0.0),
            "probe": probe_configs[selected],
            "test": test_configs[selected],
        }
    return output


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    source = json.loads(Path(args.run_result).read_text())
    metadata = source["args"]
    max_supported_rank = min(
        metadata["hidden_size"],
        2 * metadata["num_heads"] * metadata["head_dim"],
    )
    ranks = sorted(
        {
            rank
            for rank in args.ranks
            if 0 < rank <= max_supported_rank
        }
    )
    if not ranks:
        raise ValueError(f"no rank is at most {max_supported_rank}")
    plan, _ = build_streaming_plan(metadata)
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        [args.model_t],
        metadata["stream_window_days"],
        args.max_users,
    )[args.model_t]
    fit_samples, probe_samples, test_samples = split_samples(
        samples,
        args.fit_users,
        args.probe_users,
        args.seed,
    )
    current = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        args.model_t,
    )
    old = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        0,
    )
    basis, fit_summary = fit_adapter(
        current,
        old,
        fit_samples,
        metadata,
        device,
        max(ranks),
        args.ridge,
    )
    compiled_cheap = compile_projection_cache_adapter(current)
    adapters = {
        rank: (
            basis.truncate(rank),
            compile_low_rank_cache_adapter(
                current,
                basis.truncate(rank),
            ),
        )
        for rank in ranks
    }
    probe = evaluate(
        current,
        old,
        probe_samples,
        metadata,
        device,
        args.timing_repeats,
        compiled_cheap,
        adapters,
        include_prefix_ladder=True,
        include_residual_ladder=True,
    )
    test = evaluate(
        current,
        old,
        test_samples,
        metadata,
        device,
        args.timing_repeats,
        compiled_cheap,
        adapters,
        include_prefix_ladder=True,
        include_residual_ladder=True,
    )
    current_vec = model_params_vec(current).detach()
    old_vec = model_params_vec(old).detach()
    result = {
        "protocol": args.protocol,
        "study_stage": args.study_stage,
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "stale_t": 0,
        "dtheta_rel": float(
            (current_vec - old_vec).norm() / old_vec.norm()
        ),
        "split": {
            "rule": (
                "seeded fit/probe/test user partition independent of "
                "labels and outcomes"
            ),
            "fit_users": len(fit_samples),
            "probe_users": len(probe_samples),
            "test_users": len(test_samples),
            "split_seed": 9151 + args.seed,
        },
        "operator": {
            "name": "cohort_tiered_compiled_and_structural_migration",
            "fast_tier": (
                "version-calibrated residual compiled into one K/V "
                "projection"
            ),
            "quality_tier": (
                "exact current prefix replay with residual-delta "
                "transport to deeper projections"
            ),
            "fallback": "exact full recomputation",
            "selection_scope": (
                "one homogeneous action per old/current version cohort"
            ),
            "task_labels_used_for_selection": False,
        },
        "ranks": ranks,
        "fit": fit_summary,
        "probe": probe,
        "test": test,
        "selection": {
            "signal": (
                "probe K/V cache fidelity and resident-GPU kernel cost"
            ),
            "projection_cost_rule": (
                "all compiled ranks share one kernel shape; use their "
                "median measured cost and choose the smallest eligible rank"
            ),
            "structural_minimum_savings_to_full": (
                STRUCTURAL_MIN_SAVINGS
            ),
            "fidelity_targets": select_fidelity_actions(probe, test),
            "budget_targets": select_budget_actions(probe, test),
        },
    }
    save_json(result, args.output)
    for target, value in result["selection"]["fidelity_targets"].items():
        test_value = value["test"]
        gains = test_value["gain_over_reuse"]
        print(
            f"target={target} selected={value['selected']} "
            f"cost={test_value['migration_ratio_to_recompute']:.3f} "
            f"cache={test_value['cache_fidelity_recovery']:.3f} "
            f"rank={gains['best_rank']:+.2f} "
            f"utility={gains['rank_utility']:+.5f} "
            f"ndcg100={gains['ndcg@100']:+.5f}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
