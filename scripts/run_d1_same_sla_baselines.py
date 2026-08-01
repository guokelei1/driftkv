from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from layerwise_validity import load_model, reconstruct_eval_samples
from motivation_validity import build_streaming_plan, seed_everything
from structural_replay_search import evaluate, split_samples

from hstu_kvcache.migration.d1_same_sla import (
    D1_SAME_SLA_PROTOCOL,
    build_d1_same_sla_candidate_manifest,
    int_sequence_sha256,
    select_d1_same_sla_families,
)
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json

ROOT = Path(__file__).resolve().parents[1]
CELLS = tuple(
    f"{dataset}_{tier}"
    for dataset in ("kuai", "qb", "qk")
    for tier in ("small", "medium", "large")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=CELLS, default="qk_small")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--selection",
        default="results/motivation_scale/design_discovery_seeds.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-result")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--model-t", type=int, default=11)
    parser.add_argument("--stale-t", type=int, default=0)
    parser.add_argument("--probe-users", type=int, default=16)
    parser.add_argument("--test-users", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--rectangle-depth-fractions",
        type=float,
        nargs="+",
        default=[1 / 3, 2 / 3, 1.0],
    )
    parser.add_argument(
        "--rectangle-recent-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5],
    )
    parser.add_argument("--output")
    return parser.parse_args()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolve(args: argparse.Namespace) -> dict[str, object]:
    selection_path = _path(args.selection)
    selection = json.loads(selection_path.read_text())
    if args.seed is None:
        seed = int(selection["cells"][args.cell]["selected_seed"])
    else:
        seed = args.seed
    run_result = _path(
        args.run_result
        or f"results/motivation_scale/{args.cell}_v2_core_seed{seed}.json"
    )
    checkpoint_dir = _path(
        args.checkpoint_dir
        or f"checkpoints/motivation_capacity_v2/{args.cell}_seed{seed}"
    )
    output = _path(
        args.output
        or (
            "results/motivation_scale/d1_same_sla_development/"
            f"{args.cell}_seed{seed}_p{args.probe_users}_"
            f"t{args.test_users}_r{args.timing_repeats}.json"
        )
    )
    if (
        seed < 0
        or args.model_t < 1
        or args.stale_t < 0
        or args.stale_t >= args.model_t
        or args.probe_users < 1
        or args.test_users < 1
        or args.timing_repeats < 1
    ):
        raise ValueError("D1 same-SLA runner arguments are invalid")
    current_checkpoint = checkpoint_dir / f"theta_{args.model_t}.pt"
    stale_checkpoint = checkpoint_dir / f"theta_{args.stale_t}.pt"
    for path in (run_result, current_checkpoint, stale_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "seed": seed,
        "selection_path": selection_path,
        "run_result": run_result,
        "checkpoint_dir": checkpoint_dir,
        "current_checkpoint": current_checkpoint,
        "stale_checkpoint": stale_checkpoint,
        "output": output,
    }


def _sample_user_ids(samples: list[dict]) -> list[int]:
    return [
        int(sample["history"]["user_id"])
        for sample in samples
    ]


def main() -> None:
    args = parse_args()
    resolved = _resolve(args)
    seed = int(resolved["seed"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(seed)
    run_result = resolved["run_result"]
    checkpoint_dir = resolved["checkpoint_dir"]
    assert isinstance(run_result, Path)
    assert isinstance(checkpoint_dir, Path)
    source = json.loads(run_result.read_text())
    metadata = source["args"]
    plan, _ = build_streaming_plan(metadata)
    plan.init_base()
    requested_users = args.probe_users + args.test_users
    samples = reconstruct_eval_samples(
        plan,
        [args.model_t],
        metadata["stream_window_days"],
        requested_users,
    )[args.model_t]
    probe_samples, test_samples, split_seed = split_samples(
        samples,
        requested_users,
        args.probe_users,
        seed,
    )
    if (
        len(probe_samples) != args.probe_users
        or len(test_samples) != args.test_users
    ):
        raise RuntimeError("requested common probe/test split is incomplete")
    current = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        str(checkpoint_dir),
        args.model_t,
    )
    stale = load_model(
        metadata,
        plan.num_items,
        plan.num_behaviors,
        args.device,
        str(checkpoint_dir),
        args.stale_t,
    )
    manifest, manifest_sha256 = build_d1_same_sla_candidate_manifest(
        len(current.blocks),
        rectangle_depth_fractions=args.rectangle_depth_fractions,
        rectangle_recent_fractions=args.rectangle_recent_fractions,
    )
    actions = manifest["actions"]
    assert isinstance(actions, dict)
    probe = evaluate(
        current,
        stale,
        probe_samples,
        metadata,
        device,
        args.timing_repeats,
        actions,
    )
    test = evaluate(
        current,
        stale,
        test_samples,
        metadata,
        device,
        args.timing_repeats,
        actions,
    )
    selection = select_d1_same_sla_families(
        probe["summary"]["configs"],
        test["summary"]["configs"],
        manifest,
    )
    current_vec = model_params_vec(current).detach()
    stale_vec = model_params_vec(stale).detach()
    probe_user_ids = _sample_user_ids(probe_samples)
    test_user_ids = _sample_user_ids(test_samples)
    output = resolved["output"]
    current_checkpoint = resolved["current_checkpoint"]
    stale_checkpoint = resolved["stale_checkpoint"]
    selection_path = resolved["selection_path"]
    assert isinstance(output, Path)
    assert isinstance(current_checkpoint, Path)
    assert isinstance(stale_checkpoint, Path)
    assert isinstance(selection_path, Path)
    result = {
        "protocol": D1_SAME_SLA_PROTOCOL,
        "protocol_status": "development",
        "scientific_result": False,
        "formal_result": False,
        "study": "D1 same-SLA structural baseline comparison",
        "cell": args.cell,
        "seed": seed,
        "source": {
            "run_result": str(run_result),
            "run_result_sha256": _file_sha256(run_result),
            "source_protocol": source.get("protocol"),
            "discovery_selection": str(selection_path),
            "discovery_selection_sha256": _file_sha256(selection_path),
        },
        "models": {
            "stale": {
                "version": args.stale_t,
                "path": str(stale_checkpoint),
                "sha256": _file_sha256(stale_checkpoint),
            },
            "current": {
                "version": args.model_t,
                "path": str(current_checkpoint),
                "sha256": _file_sha256(current_checkpoint),
            },
            "relative_parameter_delta": float(
                (current_vec - stale_vec).norm()
                / stale_vec.norm().clamp_min(1e-12)
            ),
        },
        "common_split": {
            "rule": (
                "one seeded user permutation shared by every endpoint "
                "and structural family"
            ),
            "split_seed": split_seed,
            "probe_users": len(probe_user_ids),
            "test_users": len(test_user_ids),
            "probe_user_ids_sha256": int_sequence_sha256(
                probe_user_ids
            ),
            "test_user_ids_sha256": int_sequence_sha256(test_user_ids),
            "disjoint": not set(probe_user_ids).intersection(
                test_user_ids
            ),
        },
        "timing": {
            "device": args.device,
            "repeats_per_length_bucket": args.timing_repeats,
            "summary": "median measured migration GPU time",
            "selection_uses_probe_timing_only": True,
        },
        "candidate_manifest": manifest,
        "candidate_manifest_sha256": manifest_sha256,
        "selection": selection,
        "probe": probe,
        "test": test,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)
    for family, value in selection["families"].items():
        print(
            f"family={family} selected={value['selected']} "
            f"fallback={value['fallback_used']} "
            f"probe_recovery="
            f"{value['probe']['cache_fidelity_recovery']:.4f} "
            f"probe_cost="
            f"{value['probe']['migration_ratio_to_recompute']:.4f}"
        )
    print(output)


if __name__ == "__main__":
    main()
