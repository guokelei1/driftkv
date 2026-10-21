#!/usr/bin/env python3
"""Deferred two-GPU sequence: Large V0-V5, paired evaluation, then Max V0-V5."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def tasks(config, config_path, output, resume):
    base = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={len(config['physical_gpus'])}"]
    for scale in config["order"]:
        training = output / scale / "training"
        command = base + ["scripts/recflow/train_feedback.py", "--mode", "train", "--scale", scale,
            "--config", str(config_path), "--output", str(training)]
        checkpoints = list(training.glob("v[0-5][12]/checkpoint.json")) + list((training / "recovery").glob("step_*.json"))
        if checkpoints:
            if not resume:
                raise FileExistsError(f"Existing training checkpoints: {training}; use --resume")
            command.append("--resume")
        yield f"{scale}_train_v0_v5", command
        for edge, day in zip(config["evaluation"]["edges"], config["evaluation"]["days"], strict=True):
            yield f"{scale}_{edge}_d{day}", base + ["scripts/recflow/evaluate_feedback.py", "--scale", scale,
                "--edge", edge, "--config", str(config_path), "--training", str(training),
                "--output", str(output / scale / "evaluation" / f"{edge}_d{day}")]


def summarize(config, output, binding):
    models = {}
    lines = ["# RecFlow Large / Max adjacent-version feedback", "",
        "Two independent foundation chains, two epochs per version, two GPUs.", "",
        "| Scale | Edge | Day | Parent AUC | Current AUC | AUC gain | LogLoss delta | NDCG@5 delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scale in config["order"]:
        edges = []
        for edge, day in zip(config["evaluation"]["edges"], config["evaluation"]["days"], strict=True):
            path = output / scale / "evaluation" / f"{edge}_d{day}" / "summary.json"
            result = json.loads(path.read_text())
            parent, current = edge.split("_")
            delta = result[f"{current}_minus_{parent}"]
            edges.append(dict(edge=edge, day=day, summary=str(path), sha256=digest(path),
                differences=delta, paired_uid_auc_gain_ci95=result["paired_uid_auc_gain_ci95"]))
            lines.append(f"| {scale} | {edge} | {day} | {result['metrics'][parent]['all']['ROC_AUC']:.6f} | "
                f"{result['metrics'][current]['all']['ROC_AUC']:.6f} | {delta['ROC_AUC']:+.6f} | "
                f"{delta['log_loss']:+.6f} | {delta['within_exposure_request_NDCG@5']:+.6f} |")
        mean = {key:sum(row["differences"][key] for row in edges) / len(edges)
            for key in edges[0]["differences"] if all(key in row["differences"] for row in edges)}
        models[scale] = dict(edges=edges, equal_release_mean_differences=mean)
    atomic_json(output / "summary.json", dict(status="complete", models=models, binding=binding,
        scope="all eligible fixed final-role users; adjacent native-model feedback gains, not cache-correction results"))
    lines.extend(["", "Positive AUC/NDCG delta and negative LogLoss delta indicate improvement. All outcomes are retained.",
        "Per-edge reports contain the remaining metrics, known/hash strata and paired user confidence intervals."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def ensure_storage(config, output):
    remaining = 0
    checkpoint_sizes = []
    for scale in config["order"]:
        size = 12 * {"large": 730211201, "max": 1149285121}[scale]
        checkpoint_sizes.append(size)
        training = output / scale / "training"
        missing = sum(not (training / label / "checkpoint.json").exists()
            for label in config["permanent_epoch_checkpoints"])
        rolling = sum(path.stat().st_size for path in (training / "recovery").glob("step_*.pt"))
        remaining += missing * size + max(0, config["keep_recovery_generations"] * size - rolling)
    # One full temporary payload, small score files and filesystem headroom.
    required = remaining + max(checkpoint_sizes) + 10_000_000_000
    free = shutil.disk_usage(output).free
    if free < required:
        raise RuntimeError(f"Insufficient free disk: {free / 1e9:.1f} GB; need about {required / 1e9:.1f} GB for remaining work")
    return dict(free_disk_gb=free / 1e9, estimated_remaining_with_headroom_gb=required / 1e9)


def ensure_idle_gpus(config):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    devices = {}
    for line in output.splitlines():
        index, total, used = map(int, line.split(","))
        devices[index] = (total, used)
    free = {}
    for gpu in config["physical_gpus"]:
        total, used = devices[gpu]
        if used > 1024:
            raise RuntimeError(f"GPU {gpu} is already using {used} MiB; this long run requires its two allocated GPUs to be idle")
        free[str(gpu)] = (total - used) * 1024**2 / 1e9
    return free


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
        default=Path("experiments/foundation_recflow/configs/chain_large_max_2gpu_202.json"))
    parser.add_argument("--output", type=Path,
        default=Path("experiments/foundation_recflow/runs/chain_2gpu/attempt_1"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print the sequence without launching any process")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    config = json.loads(args.config.read_text())
    if args.plan:
        print(json.dumps(dict(training_authorized=config["training_authorized"],
            physical_gpus=config["physical_gpus"], tasks=list(tasks(config, args.config, args.output, args.resume))), indent=2))
        return
    if not config.get("training_authorized", False):
        raise SystemExit("Prepared only: waiting for the user's start instruction; no formal training started.")
    if socket.gethostname() != "g202" or config["order"] != ["large", "max"] or len(config["physical_gpus"]) != 2:
        raise ValueError("This sequence requires machine 202, two GPUs, Large then Max")
    if config["stop_after"] != "v5" or config["base_epochs"] != 2 or config["update_epochs"] != 2:
        raise ValueError("The user fixed six versions and two epochs per version")
    source_paths = [__file__, "scripts/recflow/train_feedback.py", "scripts/recflow/evaluate_feedback.py",
        "src/hstu_kvcache/recflow/feedback.py", "src/hstu_kvcache/recflow/feedback_embedding.py",
        "src/hstu_kvcache/recflow/feedback_embedding_cuda.py", "src/hstu_kvcache/models/hstu.py",
        "src/hstu_kvcache/models/block.py", "src/hstu_kvcache/models/attention.py",
        "src/hstu_kvcache/models/embeddings.py", "src/hstu_kvcache/models/rmsnorm.py",
        "src/hstu_kvcache/models/triton_attention.py", "scripts/experiments_2026_10/statistics.py",
        "src/hstu_kvcache/evaluation/binary_metrics.py", "src/hstu_kvcache/training/recovery.py"]
    binding = dict(config_sha256=digest(args.config), design_sha256=digest(config["design"]),
        data_manifest_sha256={scale:digest(Path(spec["data"]) / "manifest.json") for scale,spec in config["models"].items()},
        source_sha256={path:digest(path) for path in source_paths})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = args.output / "configuration.json"
        if frozen.exists():
            if not args.resume or json.loads(frozen.read_text())["binding"] != binding:
                raise ValueError("Existing sequence requires --resume and unchanged inputs/settings/source")
        else:
            atomic_json(frozen, dict(config=config, binding=binding))
        if args.resume:
            # Only this locked run's uncommitted atomic-write temporaries.
            for path in args.output.rglob("*.partial"):
                if path.is_file():
                    path.unlink()
        storage = ensure_storage(config, args.output)
        storage["gpu_free_gb"] = ensure_idle_gpus(config)
        atomic_json(args.output / "resource_preflight.json", storage)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, config["physical_gpus"])),
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH="src")
        started = time.time()
        for name, command in tasks(config, args.config, args.output, args.resume):
            log = args.output / f"{name}.log"
            progress = dict(status="running", task=name, command=command, log=str(log), started_unix=started)
            atomic_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            try:
                with log.open("a") as stream:
                    subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            except BaseException:
                atomic_json(args.output / "progress.json", dict(progress, status="interrupted_or_failed"))
                raise
        summarize(config, args.output, binding)
        atomic_json(args.output / "progress.json", dict(status="complete_large_and_max_v0_v5",
            elapsed_seconds=time.time() - started, binding=binding))


if __name__ == "__main__":
    main()
