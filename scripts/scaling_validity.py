from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from interval_oracle import evaluate_pair, suffix_config_name
from layerwise_validity import load_model, reconstruct_eval_samples
from motivation_validity import build_streaming_plan, seed_everything

from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="scaling_v1_fixed_optimized_suffix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-result")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument(
        "--axis",
        choices=["fixed_depth", "sequence_length", "update_magnitude"],
        required=True,
    )
    parser.add_argument("--model-t", type=int, default=5)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-eval-users", type=int, default=300)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.run_result is None:
        args.run_result = f"results/validity/core6l_seed{args.seed}.json"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/validity/core6l_seed{args.seed}"
    if args.output is None:
        args.output = f"results/scaling/{args.axis}_seed{args.seed}.json"


def selected_suffix_depths(num_layers: int) -> list[int]:
    depths = {
        max(1, round(num_layers / 3)),
        max(1, round(2 * num_layers / 3)),
        max(1, num_layers - 1),
    }
    return sorted(depth for depth in depths if depth < num_layers)


def blend_model(old, target, alpha: float) -> None:
    old_state = old.state_dict()
    target_state = target.state_dict()
    blended = {}
    for name, value in old_state.items():
        if torch.is_floating_point(value):
            blended[name] = value + alpha * (target_state[name] - value)
        else:
            blended[name] = target_state[name]
    target.load_state_dict(blended)


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    source = json.loads(Path(args.run_result).read_text())
    metadata = source["args"]
    plan_seq_len = metadata["seq_len"]
    if args.axis == "sequence_length":
        plan_seq_len = max(plan_seq_len, max(args.seq_lens))
    plan_metadata = dict(metadata)
    plan_metadata["seq_len"] = plan_seq_len
    plan, _ = build_streaming_plan(plan_metadata)
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        [args.model_t],
        metadata["stream_window_days"],
        args.max_eval_users,
    )[args.model_t]
    old = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        args.checkpoint_dir,
        0,
    )
    num_layers = len(old.blocks)
    suffix_depths = selected_suffix_depths(num_layers)
    config_names = [suffix_config_name(depth, num_layers) for depth in suffix_depths]
    if args.axis == "sequence_length":
        axis_values = [("sequence_length", value) for value in args.seq_lens]
    elif args.axis == "update_magnitude":
        axis_values = [("alpha", value) for value in args.alphas]
    else:
        axis_values = [("num_layers", num_layers)]
    result = {
        "protocol": args.protocol,
        "axis": args.axis,
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_t": args.model_t,
        "stale_t": 0,
        "max_eval_users": args.max_eval_users,
        "batch_size": args.batch_size,
        "suffix_depths": suffix_depths,
        "operator": "optimized deepest suffix with projection-only terminal layer",
        "points": [],
    }
    for axis_name, axis_value in axis_values:
        current = load_model(
            metadata,
            plan.num_items,
            plan.num_behaviors,
            args.device,
            args.checkpoint_dir,
            args.model_t,
        )
        if args.axis == "update_magnitude":
            blend_model(old, current, float(axis_value))
        eval_metadata = dict(metadata)
        eval_metadata["batch_size"] = args.batch_size
        if args.axis == "sequence_length":
            eval_metadata["seq_len"] = int(axis_value)
        pair = evaluate_pair(
            current,
            old,
            samples,
            eval_metadata,
            device,
            args.timing_repeats,
            config_names,
            False,
        )
        current_vec = model_params_vec(current).detach()
        old_vec = model_params_vec(old).detach()
        pair.update(
            {
                "axis_name": axis_name,
                "axis_value": axis_value,
                "num_layers": num_layers,
                "seq_len": eval_metadata["seq_len"],
                "dtheta_rel": float((current_vec - old_vec).norm() / old_vec.norm()),
                "n_users": len(pair["per_user"]),
            }
        )
        result["points"].append(pair)
        save_json(result, args.output)
        configs = pair["summary"]["configs"]
        print(
            f"{axis_name}={axis_value} dtheta={pair['dtheta_rel']:.6f} "
            f"users={pair['n_users']}",
            flush=True,
        )
        for name, value in configs.items():
            print(
                f"  {name:>18} cost={value['migration_ratio_to_recompute']:.3f} "
                f"rank_gain={value['gain_over_reuse']['best_rank']:.2f} "
                f"ndcg100_gain={value['gain_over_reuse']['ndcg@100']:.5f}",
                flush=True,
            )
        del current
    save_json(result, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
