from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch
from evaluate_cohortkv_stage1_frontier import (
    DEFAULT_BLUEPRINT,
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST,
    DEFAULT_PREPARED,
    DEFAULT_PROGRAM_DIR,
    DEFAULT_PROGRAM_RESULT,
    DEFAULT_TRAINING,
    sha256,
    validate_frozen_inputs,
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
from hstu_kvcache.streaming import load_checkpoint_model
from hstu_kvcache.utils import save_json

PROTOCOL = "cohortkv_single_config_stage4_6_adjacent_compiler_v1"
EXPERIMENT_PROTOCOL = "cohortkv_single_config_stage4_6_lifecycle_development_v1"
DEFAULT_RUNTIME_DIR = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/stage4_6_runtime"
)
DEFAULT_OUTPUT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_6_adjacent_compiler_seed0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--program-result", default=DEFAULT_PROGRAM_RESULT)
    parser.add_argument("--program-dir", default=DEFAULT_PROGRAM_DIR)
    parser.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    parser.add_argument("--workload-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-dir", default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--attention-weight-cap", type=float, default=8.0)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def direct_program_path(runtime_dir: str | Path, source: int, target: int) -> Path:
    return Path(runtime_dir) / (
        f"theta{source}_to_theta{target}_direct_oldkv_fp16.pt"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.seed != 0:
        raise ValueError("Stage 4.6 freezes seed 0")
    if args.max_fit_tokens != 8192:
        raise ValueError("Stage 4.6 freezes 8192 fit tokens per layer")
    if args.attention_weight_cap != 8.0 or args.ridge != 0.001:
        raise ValueError("Stage 4.6 freezes the Stage 1 compiler settings")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Stage 4.6 requires measured CUDA execution")


def checkpoint_descriptors(
    checkpoint_dir: str | Path,
) -> list[dict[str, object]]:
    output = []
    for version in range(12):
        path = Path(checkpoint_dir) / f"theta_{version}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append(
            {
                "version": f"theta{version}",
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return output


def validate_existing(
    args: argparse.Namespace,
    cfg,
    descriptors: list[dict[str, object]],
) -> list[dict[str, object]]:
    programs = []
    hashes = {
        str(value["version"]): str(value["sha256"])
        for value in descriptors
    }
    for source in range(11):
        target = source + 1
        path = direct_program_path(args.runtime_dir, source, target)
        program, descriptor = load_direct_oldkv_program(
            path,
            expected_source_version=f"theta{source}",
            expected_target_version=f"theta{target}",
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.num_heads * cfg.head_dim,
        )
        provenance = descriptor["provenance"]
        if (
            provenance.get("experiment_protocol") != EXPERIMENT_PROTOCOL
            or provenance.get("labels_used") is not False
            or provenance.get("source_checkpoint_sha256")
            != hashes[f"theta{source}"]
            or provenance.get("target_checkpoint_sha256")
            != hashes[f"theta{target}"]
        ):
            raise ValueError("Stage 4.6 adjacent program provenance differs")
        programs.append(descriptor)
        del program
    return programs


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
    checkpoints = checkpoint_descriptors(args.checkpoint_dir)
    if args.validate_only:
        programs = validate_existing(args, cfg, checkpoints)
        print(
            json.dumps(
                {
                    "status": "validated",
                    "checkpoints": len(checkpoints),
                    "programs": len(programs),
                }
            )
        )
        return
    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_hashes = {
        str(value["version"]): str(value["sha256"])
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
    started = time.perf_counter()
    pairs = []
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        0,
        device,
    )
    for source in range(11):
        target = source + 1
        target_model = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            target,
            device,
        )
        family, fit = fit_attention_family(
            target_model,
            source_model,
            roles["fit"],
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
                "compiler_protocol": PROTOCOL,
                "labels_used": False,
                "fit_role": "fit",
                "fit_records": len(roles["fit"]),
                "attention_mix": 1.0,
                "attention_weight_cap": args.attention_weight_cap,
                "ridge": args.ridge,
                "source_checkpoint_sha256": checkpoint_hashes[
                    f"theta{source}"
                ],
                "target_checkpoint_sha256": checkpoint_hashes[
                    f"theta{target}"
                ],
                "workload_manifest_content_sha256": manifest[
                    "content_sha256"
                ],
                "derivation": (
                    "label-free adjacent residual fit compiled through the "
                    "source stacked K/V projection"
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
        "protocol": PROTOCOL,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "status": "complete",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "frozen_inputs": {
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
            "checkpoints": checkpoints,
        },
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "training_seed": 0,
            "model": training["model"],
            "device": str(device),
            "fit_records": len(roles["fit"]),
            "labels_used": False,
            "attention_mix": 1.0,
            "attention_weight_cap": args.attention_weight_cap,
            "ridge": args.ridge,
            "max_fit_tokens_per_layer": args.max_fit_tokens,
        },
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
