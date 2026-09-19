#!/usr/bin/env python3
"""Run a contract-bound HSTU release and its Full-only E14 evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda_medium_v2_2epoch_4gpu_speed_20260915.yaml"
EXECUTION = ROOT / "configs/unified_training_2026_09/medium_v2_4gpu_cpu14_execution.yaml"
EVALUATION_SCOPE = ROOT / "configs/unified_training_2026_09/evaluation_e14_only.yaml"
OUT = ROOT / "results/unified_training_2026_09/medium/seed17/v2_2epoch_4gpu_cpu14"
MANIFEST = ROOT / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1"
PARENT = ROOT / "results/unified_training_2026_09/medium/seed17/checkpoints/v1/checkpoint_100.pt"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    global CONTRACT, EXECUTION, OUT, PARENT, MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True,
                        help="Select the frozen runtime and its separate output directory.")
    parser.add_argument("--resume-evaluation", action="store_true",
                        help="Reuse the sealed checkpoint and existing sealed raw scores; never retrain.")
    parser.add_argument("--resume-training", action="store_true",
                        help="Resume from the latest complete same-layout recovery generation.")
    args = parser.parse_args()
    resume = args.resume_evaluation
    assert not (resume and args.resume_training)
    EXECUTION = args.execution_config.resolve()
    execution = yaml.safe_load(EXECUTION.read_text())
    CONTRACT = ROOT / execution["frozen_parent"]["contract"]
    OUT = ROOT / execution["output_amendment"]["root"]
    contract = yaml.safe_load(CONTRACT.read_text())
    scale = contract['scope']['scale']
    MANIFEST = (ROOT / contract['frozen_inputs']['request_manifest']).parent
    version = contract['scope']['versions'][0]
    parent_version = contract['scope']['expected_parent_version']
    PARENT = ROOT / contract['frozen_inputs'][f'parent_{parent_version}_checkpoint']
    start, end = contract['scope']['training_days_half_open']
    epochs = contract['training']['passes']
    evaluation_scope = yaml.safe_load(EVALUATION_SCOPE.read_text())
    assert evaluation_scope['evaluation_horizon_days'] == [14]
    assert contract['evaluation']['windows_days_half_open'] == {'E14': [end, end + 14]}
    canary = json.loads((OUT / "canary.pass.json").read_text())
    estimate = json.loads((OUT / "resource_estimate.json").read_text())
    assert canary["passed"] and estimate["formal_training_steps"] > 0
    assert canary["contract_sha256"] == sha(CONTRACT)
    assert canary["execution_contract_sha256"] == sha(EXECUTION)
    for key, value in contract["frozen_inputs"].items():
        if not key.endswith("_sha256"):
            assert sha(ROOT / value) == contract["frozen_inputs"][key + "_sha256"], key
    for path, digest in canary["code_sha256"].items():
        assert sha(ROOT / path) == digest, path
    if not resume and not args.resume_training:
        assert not (OUT / "checkpoint").exists(), "Refusing to overwrite a training run"
    topology = execution["execution_amendment"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ",".join(map(str, topology["physical_gpus"])),
           "OMP_NUM_THREADS": str(execution["training_runtime"]["omp_num_threads"]), "PYTHONUNBUFFERED": "1"}
    distributed = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={topology['world_size']}"]
    events = json.loads((OUT / "runtime.json").read_text()) if (OUT / 'runtime.json').exists() and (resume or args.resume_training) else []

    def run(name, command, extra_env=None):
        started = time.time()
        print(f"Starting {name}", flush=True)
        write(OUT / "progress.json", {"stage": name, "started_at_unix": started})
        log_path = OUT / "logs" / f"{name}.log"
        if (resume or args.resume_training) and log_path.exists():
            log_path = OUT / "logs" / f"{name}.resume_{time.time_ns()}.log"
        with log_path.open("x") as log:
            result = subprocess.run(command, cwd=ROOT, env={**env, **(extra_env or {})},
                                    stdout=log, stderr=subprocess.STDOUT)
        events.append({"stage": name, "command": command, "exit_code": result.returncode,
                       "started_at_unix": started, "elapsed_seconds": time.time() - started})
        write(OUT / "runtime.json", events)
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; see its log")

    endpoints = contract['training'].get('checkpoint_epochs', [])
    assert not endpoints or endpoints in ([epochs], [1, 2])
    multiple = len(endpoints) > 1
    candidates = ({f'{version}_e{e}': (e, OUT / 'checkpoint' / f'checkpoint_epoch_{e}.pt') for e in endpoints}
                  if multiple else {})
    checkpoint = OUT / "checkpoint" / (f"checkpoint_epoch_{epochs}.pt" if endpoints else "checkpoint_100.pt")
    train = [*distributed, "scripts/train_yambda500m_foundation_fsdp.py",
             "--version", version, "--branch", "D14", "--launch-contract", str(CONTRACT),
             "--execution-contract", str(EXECUTION), "--manifest-dir", str(MANIFEST),
             "--training-block", "matrix_horizon", "--parent", str(PARENT),
             "--output", str(checkpoint.parent), "--oov-buckets", "256", "--passes", str(epochs),
             "--global-batch-size", str(topology["global_train_batch_size"]), "--train-start-day", str(start), "--train-end-day", str(end),
             "--progress-interval", "250"]
    for flag, key in [("history-threads", "history_threads"), ("arrow-cpu-threads", "arrow_cpu_threads"),
                      ("arrow-io-threads", "arrow_io_threads"), ("torch-cpu-threads", "torch_cpu_threads"),
                      ("cpu-affinity-by-rank", "cpu_affinity_by_rank")]:
        train.extend(["--" + flag, str(execution["training_runtime"][key])])
    recovery = contract['training'].get('recovery')
    if recovery:
        train += ['--recovery-interval-steps', str(recovery['interval_steps']),
                  '--recovery-first-step', str(recovery['first_step'])]
    if args.resume_training:
        generations = sorted((checkpoint.parent / 'recovery').glob('step_*/complete.json'))
        assert generations, 'No complete recovery checkpoint'
        train += ['--resume-recovery', str(generations[-1].parent)]
    config_name = (f'resume_configuration_{time.time_ns()}.json' if resume or args.resume_training else 'configuration.json')
    write(OUT / config_name, {
        "contract": str(CONTRACT.relative_to(ROOT)), "contract_sha256": sha(CONTRACT),
        "execution_contract": str(EXECUTION.relative_to(ROOT)), "execution_contract_sha256": sha(EXECUTION),
        "canary_sha256": sha(OUT / "canary.pass.json"),
        "resource_estimate_sha256": sha(OUT / "resource_estimate.json"),
        "runner_sha256": sha(__file__), "interpretation": contract["scope"]["interpretation"],
        "evaluation_scope": str(EVALUATION_SCOPE.relative_to(ROOT)),
        "evaluation_scope_sha256": sha(EVALUATION_SCOPE),
    })
    if not resume:
        run("train", train)
    import torch
    torch.set_num_threads(execution['training_runtime']['torch_cpu_threads'])
    if not candidates:
        candidates = {version: (epochs, checkpoint)}
    checkpoint_seals = {}
    prior_seals = json.loads((checkpoint.parent / ('checkpoints.seal.json' if multiple else 'checkpoint.seal.json')).read_text()) if resume else None
    for name, (epoch, path) in candidates.items():
        if resume:
            prior = prior_seals[name] if multiple else prior_seals
            assert sha(path) == prior['checkpoint_sha256'] and sha(CONTRACT) == prior['contract_sha256']
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["parent_checkpoint_sha256"] == sha(PARENT)
        assert payload["training_day_range"] == [start, end] and payload["training_epochs_completed"] == epoch
        assert all(torch.isfinite(value).all().item() for value in payload["model"].values())
        del payload
        checkpoint_seals[name] = {
            "status": f"{scale.lower()}_release_checkpoint_sealed", "version": version, "seed": 17,
            "checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": sha(path),
            "parent_checkpoint_sha256": sha(PARENT), "contract_sha256": sha(CONTRACT),
            "training_day_range": [start, end], "epochs": epoch,
        }
    if not resume:
        write(checkpoint.parent / ("checkpoints.seal.json" if multiple else "checkpoint.seal.json"),
              checkpoint_seals if multiple else checkpoint_seals[version])
        recovery_root = checkpoint.parent / 'recovery'
        if recovery_root.exists():
            write(OUT / 'recovery_retirement.json', {
                'reason': 'all requested epoch endpoints validated and sealed',
                'generations': [json.loads(p.read_text()) for p in sorted(recovery_root.glob('step_*/complete.json'))],
            })
            shutil.rmtree(recovery_root)
    runtime = execution["evaluation_runtime"]
    rows = []
    primary_report = None
    for horizon in evaluation_scope["evaluation_horizon_days"]:
        output = OUT / "full_only" / f"E{horizon}"
        command = [*distributed, "scripts/evaluate_yambda500m_release_candidates_raw.py",
                   "--stage", f"{scale.lower()}_D14_E{horizon}_{parent_version}_to_{version}_full_only",
                   "--block", "matrix_horizon", "--training-block", "matrix_horizon",
                   "--manifest-dir", str(MANIFEST), "--dataset-manifest", contract["frozen_inputs"]["dataset_manifest"],
                   "--parent", f"{parent_version}={PARENT}",
                   "--start-day", str(end), "--end-day", str(end + horizon),
                   "--training-start-day", str(start), "--training-end-day", str(end),
                   "--batch-size", "64", "--output", str(output)]
        for name, (_, path) in candidates.items():
            command.extend(["--current", f"{name}={path}"])
        for flag, key in [("history-threads", "history_threads"), ("arrow-cpu-threads", "arrow_cpu_threads"),
                          ("arrow-io-threads", "arrow_io_threads"), ("torch-cpu-threads", "torch_cpu_threads"),
                          ("cpu-affinity-by-rank", "cpu_affinity_by_rank")]:
            command.extend(["--" + flag, str(runtime[key])])
        if not (resume and (output / "raw.seal.json").exists()):
            run(f"full_E{horizon}", command, {"OMP_NUM_THREADS": "4"})
        raw_seal = json.loads((output / "raw.seal.json").read_text())
        assert sha(output / "raw.parquet") == raw_seal["raw_sha256"]
        if not (resume and (output / "adjudication.json").exists()):
            run(f"adjudicate_E{horizon}", [sys.executable, "scripts/adjudicate_yambda500m_release_candidates.py",
                "--raw", str(output / "raw.parquet"), "--seal", str(output / "raw.seal.json"),
                "--labels", str(MANIFEST / "requests_quality.parquet"),
                "--output", str(output / "adjudication.json")])
        report = json.loads((output / "adjudication.json").read_text())
        assert report["raw_sha256"] == raw_seal["raw_sha256"]
        parent = report["parent_absolute"]["hstu_native"]
        admissions = {}
        for name, (epoch, path) in candidates.items():
            current = report["candidates"][name]["absolute"]["hstu_native"]
            pair = report["candidates"][name]["paired_release_gain"]["parent_minus_current_log_loss"]
            gain = 100 * (current["ROC_AUC"] / parent["ROC_AUC"] - 1)
            gates = {
                "current_auc_greater_than_parent": current["ROC_AUC"] > parent["ROC_AUC"],
                "current_log_loss_less_than_parent": current["log_loss"] < parent["log_loss"],
                "current_brier_not_greater_than_parent": current["Brier"] <= parent["Brier"],
                "paired_user_log_loss_bootstrap_lower_positive": pair["user_cluster_bootstrap_95CI"]["p2_5"] > 0,
            }
            admissions[name] = {"primary_horizon": "E14", "gates": gates,
                "original_metric_gates_pass": all(gates.values()),
                "research_target_relative_auc_gain_gt_1pct": gain > 1,
                "full_only_report_sha256": sha(output / "adjudication.json"),
                "contract_sha256": sha(CONTRACT), "serving_lineage_promoted": False, "reuse_executed": False}
            rows.append({"candidate": name, "epochs": epoch, "horizon": f"E{horizon}",
                "day_range": [end, end + horizon], "requests": pair["requests"], "users": pair["users"],
                "parent": parent, "current": current, "auc_delta_pp": 100 * (current["ROC_AUC"] - parent["ROC_AUC"]),
                "relative_auc_gain_percent": gain,
                "relative_log_loss_reduction_percent": 100 * (1-current["log_loss"]/parent["log_loss"]),
                "adjudication_sha256": sha(output / "adjudication.json")})
    admission = ({"candidates": admissions, "automatic_endpoint_selection": False} if multiple else admissions[version])
    write(OUT / "admission.seal.json", admission)
    write(OUT / "summary.json", {"status": "complete", "rows": rows, "admission": admission,
        "evaluation_completeness": contract['evaluation'].get('completeness', 'complete'),
        "interpretation": contract["scope"]["interpretation"]})
    table = [f"# {scale} {parent_version} → {version}：E14结果", "",
        f"训练窗口 [{start},{end})，评价窗口 [{end},{end+14})。",
        "评价完整性：" + contract['evaluation'].get('completeness', 'complete'), "",
        "| 端点 | Parent AUC | Current AUC | 相对提升 | 原四项准入 | >1%目标 |",
        "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        verdict = admissions[row['candidate']]
        table.append(f"| {row['candidate']} | {row['parent']['ROC_AUC']:.6f} | {row['current']['ROC_AUC']:.6f} | "
            f"{row['relative_auc_gain_percent']:+.3f}% | {verdict['original_metric_gates_pass']} | "
            f"{verdict['research_target_relative_auc_gain_gt_1pct']} |")
    table.extend(["", "完整指标见summary.json。所有约定端点均报告，不自动选择或推广模型。"])
    (OUT / "README.md").write_text("\n".join(table) + "\n")
    write(OUT / "progress.json", {"stage": "complete", "completed_at_unix": time.time()})
    print(json.dumps({"status": "complete", "admission": admission}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write(OUT / "progress.json", {"stage": "failed", "error": str(error), "time_unix": time.time()})
        (OUT / "exit_status.txt").write_text("1\n")
        raise
    else:
        (OUT / "exit_status.txt").write_text("0\n")
