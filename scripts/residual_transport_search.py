from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from layerwise_validity import load_model, reconstruct_eval_samples
from motivation_validity import build_streaming_plan, seed_everything
from structural_replay_search import evaluate, select_actions, split_samples

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
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def residual_actions(num_layers: int) -> dict[str, dict]:
    actions = {}
    for depth in range(num_layers + 1):
        if depth == 0:
            name = "cheap_all"
        elif depth == num_layers:
            name = "recompute"
        else:
            name = f"residual_p{depth}"
        actions[name] = {
            "kind": "prefix_residual",
            "prefix_depth": depth,
            "depth_fraction": depth / num_layers,
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
    actions = residual_actions(len(current.blocks))
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
    selection = select_actions(
        probe,
        test,
        (0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.9),
        (0.2, 0.4, 0.6, 0.8),
    )
    result = {
        "protocol": "residual_transport_discovery_v1",
        "study_stage": "motivation_selected_seed_discovery",
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "stale_t": 0,
        "dtheta_rel": float((current_vec - old_vec).norm() / old_vec.norm()),
        "operator": {
            "name": "prefix_replay_with_residual_delta_transport",
            "exact_prefix": "current native blocks over layers 1 through p",
            "transport": (
                "add the current-minus-old boundary hidden delta to cached "
                "old hidden states at every deeper layer"
            ),
            "deep_refresh": "current Norm plus Wk/Wv",
            "candidate_count": len(actions),
            "planning_complexity_in_layers": "O(L)",
            "learned_parameters": 0,
        },
        "action_space": actions,
        "split": {
            "rule": "seeded user permutation independent of labels and outcomes",
            "probe_users": len(probe_samples),
            "test_users": len(test_samples),
            "split_seed": split_seed,
        },
        "selection_signal": (
            "relative K/V cache fidelity on probe users; task labels are "
            "used only for offline evaluation"
        ),
        "probe": probe,
        "test": test,
        "selection": selection,
    }
    save_json(result, args.output)
    for target, value in selection["fidelity_targets"].items():
        selected = value["selected"]
        test_value = value["test"]
        print(
            f"fidelity={target} selected={selected} "
            f"cost={test_value['migration_ratio_to_recompute']:.3f} "
            f"cache={test_value['cache_fidelity_recovery']:.3f} "
            f"rank={test_value['gain_over_reuse']['best_rank']:.2f} "
            f"utility={test_value['gain_over_reuse']['rank_utility']:.5f} "
            f"ndcg100={test_value['gain_over_reuse']['ndcg@100']:.5f}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
