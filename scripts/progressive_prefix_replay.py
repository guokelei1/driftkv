from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from layerwise_validity import load_model, reconstruct_eval_samples
from motivation_validity import build_streaming_plan, seed_everything
from structural_replay_search import evaluate, split_samples

from hstu_kvcache.migration import select_minimum_cost_actions
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-t", type=int, default=11)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--probe-users", type=int, default=60)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--study-stage",
        choices=("discovery_replay", "frozen_rule_replication"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def prefix_actions(num_layers: int) -> dict[str, dict]:
    actions = {
        "cheap_all": {
            "kind": "recent_suffix",
            "top_n_full": 0,
            "depth_fraction": 0.0,
            "recent_fraction": 0.0,
        }
    }
    for depth in range(1, num_layers):
        actions[f"prefix_p{depth}"] = {
            "kind": "interval",
            "start_layer": 0,
            "end_layer": depth - 1,
            "depth_fraction": depth / num_layers,
            "recent_fraction": 1.0,
        }
    actions["recompute"] = {
        "kind": "recent_suffix",
        "top_n_full": num_layers,
        "depth_fraction": 1.0,
        "recent_fraction": 1.0,
    }
    return actions


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    source = json.loads(Path(args.run_result).read_text())
    metadata = source["args"]
    plan, _ = build_streaming_plan(metadata)
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        [args.model_t],
        metadata["stream_window_days"],
        args.max_users,
    )[args.model_t]
    probe_samples, test_samples, split_seed = split_samples(
        samples,
        args.max_users,
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
    current_vec = model_params_vec(current).detach()
    old_vec = model_params_vec(old).detach()
    actions = prefix_actions(len(current.blocks))
    probe = evaluate(
        current,
        old,
        probe_samples,
        metadata,
        device,
        args.timing_repeats,
        actions,
    )
    test = evaluate(
        current,
        old,
        test_samples,
        metadata,
        device,
        args.timing_repeats,
        actions,
    )
    selected = select_minimum_cost_actions(
        probe["summary"]["configs"],
        (0.2,),
    )["0.2"]
    selected_name = selected["selected"]
    selected["probe"] = probe["summary"]["configs"][selected_name]
    selected["test"] = test["summary"]["configs"][selected_name]
    result = {
        "protocol": "progressive_prefix_replay_v1",
        "study_stage": args.study_stage,
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "stale_t": 0,
        "dtheta_rel": float((current_vec - old_vec).norm() / old_vec.norm()),
        "operator": {
            "name": "version_cohort_progressive_prefix_replay",
            "actions": (
                "cheap projection, current native blocks over layers 1 through "
                "p with projection-only terminal execution, or full recompute"
            ),
            "candidate_count": len(actions),
            "planning_complexity_in_layers": "O(L)",
            "selection_scope": "one action per old/current model-version cohort",
            "learned_parameters": 0,
        },
        "selection_rule": {
            "signal": "relative K/V cache fidelity on probe users",
            "target": 0.2,
            "decision": "minimum measured GPU cost meeting the target",
            "task_labels_used_for_selection": False,
            "fallback": "full recompute",
        },
        "action_space": actions,
        "split": {
            "rule": "seeded user permutation independent of labels and outcomes",
            "probe_users": len(probe_samples),
            "test_users": len(test_samples),
            "split_seed": split_seed,
        },
        "probe": probe,
        "test": test,
        "selected": selected,
    }
    save_json(result, args.output)
    value = selected["test"]
    print(
        f"selected={selected_name} "
        f"cost={value['migration_ratio_to_recompute']:.3f} "
        f"cache={value['cache_fidelity_recovery']:.3f} "
        f"rank={value['gain_over_reuse']['best_rank']:.2f} "
        f"utility={value['gain_over_reuse']['rank_utility']:.5f} "
        f"ndcg100={value['gain_over_reuse']['ndcg@100']:.5f}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
