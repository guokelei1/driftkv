from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch
from cohortkv_stage4_7_common import (
    CHECKPOINT_DIR,
    COMPILER_OUTPUT,
    COMPILER_PROTOCOL,
    EXPERIMENT_PROTOCOL,
    PREPARED_PATH,
    RUNTIME_DIR,
    TRAINING_PATH,
    direct_program_path,
    history_view_sha256,
    load_inputs,
    samples_for_users,
    sha256,
)
from motivation_validity import seed_everything
from search_kuairand_long_context_attention_weighted import (
    fit_attention_family,
    mix_name,
)

from hstu_kvcache.migration import MigrationProgram
from hstu_kvcache.migration.stage45_oldkv import (
    compile_direct_oldkv_program,
    load_direct_oldkv_program,
    write_direct_oldkv_program,
)
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--runtime-dir", default=RUNTIME_DIR)
    parser.add_argument("--output", default=COMPILER_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--attention-weight-cap", type=float, default=8.0)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.seed != 0
        or args.batch_size != 4
        or args.max_fit_tokens != 8192
        or args.attention_weight_cap != 8.0
        or args.ridge != 0.001
    ):
        raise ValueError("organic compiler settings differ")
    if torch.device(args.device).type != "cuda":
        raise ValueError("organic compiler requires CUDA")


def validate_existing(
    args,
    cfg,
    checkpoints,
    manifest,
    windows,
    fit_ids,
) -> list[dict]:
    hashes = {
        value["version"]: value["sha256"]
        for value in checkpoints
    }
    output = []
    for source in range(11):
        target = source + 1
        program, descriptor = load_direct_oldkv_program(
            direct_program_path(args.runtime_dir, source, target),
            expected_source_version=f"theta{source}",
            expected_target_version=f"theta{target}",
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.num_heads * cfg.head_dim,
        )
        provenance = descriptor["provenance"]
        if (
            provenance.get("experiment_protocol") != EXPERIMENT_PROTOCOL
            or provenance.get("compiler_protocol") != COMPILER_PROTOCOL
            or provenance.get("labels_used") is not False
            or provenance.get("history_version") != f"theta{target}"
            or provenance.get("history_view_sha256")
            != history_view_sha256(windows[target], fit_ids)
            or provenance.get("source_checkpoint_sha256")
            != hashes[f"theta{source}"]
            or provenance.get("target_checkpoint_sha256")
            != hashes[f"theta{target}"]
            or provenance.get("manifest_content_sha256")
            != manifest["content_sha256"]
        ):
            raise ValueError("organic adjacent program provenance differs")
        output.append(descriptor)
        del program
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(record["user_id"] for record in manifest["records"])
    windows = reconstruct_organic_windows(plan, user_ids)
    fit_ids = tuple(
        record["user_id"]
        for record in manifest["records"]
        if record["evaluation_role"] == "fit"
    )
    if args.validate_only:
        programs = validate_existing(
            args,
            cfg,
            checkpoints,
            manifest,
            windows,
            fit_ids,
        )
        print(
            json.dumps(
                {
                    "status": "validated",
                    "programs": len(programs),
                }
            )
        )
        return
    checkpoint_hashes = {
        value["version"]: value["sha256"]
        for value in checkpoints
    }
    fit_args = argparse.Namespace(
        seq_len=cfg.max_seq_len,
        batch_size=args.batch_size,
        max_fit_tokens=args.max_fit_tokens,
        attention_mixes=[1.0],
        attention_weight_cap=args.attention_weight_cap,
        ridge=args.ridge,
        seed=args.seed,
    )
    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        0,
        device,
    )
    pairs = []
    started = time.perf_counter()
    for source in range(11):
        target = source + 1
        target_model = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            target,
            device,
        )
        fit_samples = samples_for_users(
            windows[target],
            fit_ids,
            include_positives=False,
        )
        fit_samples = [
            sample
            for sample in fit_samples
            if len(sample["history"]["item_ids"]) >= 2
        ]
        family, fit = fit_attention_family(
            target_model,
            source_model,
            fit_samples,
            fit_args,
            device,
        )
        adapter = family[mix_name(1.0)]
        parent = MigrationProgram(
            source_version=f"theta{source}",
            target_version=f"theta{target}",
            adapter=adapter,
        )
        direct, compile_metrics = compile_direct_oldkv_program(
            source_model,
            parent,
        )
        descriptor = write_direct_oldkv_program(
            direct,
            direct_program_path(runtime_dir, source, target),
            {
                "experiment_protocol": EXPERIMENT_PROTOCOL,
                "compiler_protocol": COMPILER_PROTOCOL,
                "labels_used": False,
                "fit_role": "fit",
                "fit_records": len(fit_samples),
                "history_version": f"theta{target}",
                "history_target_date": windows[target].target_date,
                "history_view_sha256": history_view_sha256(
                    windows[target],
                    fit_ids,
                ),
                "attention_mix": 1.0,
                "attention_weight_cap": args.attention_weight_cap,
                "ridge": args.ridge,
                "source_checkpoint_sha256": checkpoint_hashes[
                    f"theta{source}"
                ],
                "target_checkpoint_sha256": checkpoint_hashes[
                    f"theta{target}"
                ],
                "manifest_content_sha256": manifest["content_sha256"],
                "future_history_used": False,
                "derivation": (
                    "label-free adjacent residual fit over the history "
                    "available at the target update boundary"
                ),
            },
            compile_metrics,
        )
        loaded, loaded_descriptor = load_direct_oldkv_program(
            descriptor["path"],
            expected_sha256=descriptor["sha256"],
            expected_source_version=f"theta{source}",
            expected_target_version=f"theta{target}",
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.num_heads * cfg.head_dim,
        )
        pairs.append(
            {
                "source_version": f"theta{source}",
                "target_version": f"theta{target}",
                "history_target_date": windows[target].target_date,
                "history_view_sha256": history_view_sha256(
                    windows[target],
                    fit_ids,
                ),
                "fit": fit,
                "direct_program": descriptor,
                "load_validation": {
                    "passed": True,
                    "provenance": loaded_descriptor["provenance"],
                },
            }
        )
        print(
            json.dumps(
                {
                    "status": "compiled",
                    "edge": f"theta{source}->theta{target}",
                    "history_target_date": windows[target].target_date,
                    "fit_records": len(fit_samples),
                    "fit_elapsed_ms": fit["elapsed_ms"],
                    "program_sha256": descriptor["sha256"],
                }
            ),
            flush=True,
        )
        del source_model, family, adapter, parent, direct, loaded
        source_model = target_model
        gc.collect()
        torch.cuda.empty_cache()
    del source_model
    payload = {
        "protocol": COMPILER_PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "status": "complete",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "inputs": {
            "prepared_data": {
                "path": args.prepared_data,
                "sha256": sha256(args.prepared_data),
                "protocol": metadata["protocol"],
            },
            "training_result": {
                "path": args.training_result,
                "sha256": sha256(args.training_result),
                "protocol": training["protocol"],
            },
            "checkpoints": checkpoints,
        },
        "manifest": manifest,
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "training_seed": 0,
            "model": training["model"],
            "device": str(device),
            "cohort_records": len(user_ids),
            "fit_role_records": len(fit_ids),
            "labels_used": False,
            "attention_mix": 1.0,
            "attention_weight_cap": args.attention_weight_cap,
            "ridge": args.ridge,
            "max_fit_tokens_per_layer": args.max_fit_tokens,
        },
        "windows": [
            {
                "version": window.version,
                "target_date": window.target_date,
                "history_view_sha256": history_view_sha256(
                    window,
                    user_ids,
                ),
            }
            for window in windows
        ],
        "pairs": pairs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(payload, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": args.output,
                "pairs": len(pairs),
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
