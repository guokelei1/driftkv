#!/usr/bin/env python3
"""Authorized six-layer expanded-population base and five fixed daily updates."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_epoch(directory, setting, phase, epochs, before, source, canary):
    import torch

    record = read(directory / "summary.json")
    cfg = record["configuration"]
    count = 263 if canary else setting["windows"][phase]["eligible_requests_per_epoch"]
    steps = math.ceil(count / setting["arguments"]["batch_size"])
    assert record["complete"] and cfg["canary"] == canary
    assert cfg["phase"] == phase and cfg["requested_complete_epochs"] == epochs
    assert cfg["optimizer_state_steps_before"] == ([] if before == 0 else [before])
    assert cfg["optimizer_learning_rates"] == [setting["arguments"]["learning_rate"]]
    assert len(record["epochs"]) == epochs
    for number, entry in enumerate(record["epochs"], 1):
        train = entry["training"]
        assert train["complete"] and train["epoch"] == number
        assert train["processed_requests"] == train["eligible_requests"] == count
        assert train["optimizer_steps"] == steps
        assert math.isfinite(train["request_mean_loss"])
        assert math.isfinite(entry["fixed_training_nll"]["mean_nll"])
    if source:
        assert Path(cfg["source_checkpoint"]).resolve() == source.resolve()
        assert cfg["source_checkpoint_sha256"] == digest(source)
    checkpoint = directory / f"{phase}_epoch{epochs}" / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    actual = sorted({int(v["step"].item()) for v in saved["optimizer"]["state"].values() if "step" in v})
    rates = [g["lr"] for g in saved["optimizer"]["param_groups"]]
    after = before + epochs * steps
    assert actual == [after] and rates == cfg["optimizer_learning_rates"]
    assert saved["phase"] == phase and saved["epoch"] == epochs
    check = dict(passed=True, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        phase=phase, epoch=epochs, saved_learning_rates=rates, saved_optimizer_steps=actual,
        optimizer_state_steps_before=cfg["optimizer_state_steps_before"],
        source_checkpoint_sha256=cfg["source_checkpoint_sha256"],
        method="Actual serialized model/AdamW checkpoint read with CPU mmap; no GPU allocation.")
    save(directory / "optimizer_check.json", check)
    return checkpoint, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true", help="Only tiny actual A/B pipeline checks; no quality qualification")
    parser.add_argument("--canary-report", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    config, out = args.config.resolve(), args.output.resolve()
    setting = read(config)
    panels = Path(setting["arguments"]["panels"])
    assert digest(panels) == setting["panels_sha256"]
    if out.exists():
        raise SystemExit("Output exists; inspect retained progress instead of overwriting or duplicating a run.")
    if not args.canary:
        if args.canary_report is None:
            parser.error("Supply the completed focused --canary-report before the long run.")
        proof = read(args.canary_report)
        assert proof["passed"] and proof["settings_sha256"] == digest(config)
        assert proof["panels_sha256"] == digest(panels)
        assert proof["final_optimizer_steps"] == 6
        assert proof["cohort_users"] == setting["arguments"]["cohort_users"]
        assert all(digest(ROOT / name) == sha for name, sha in proof["source_sha256"].items())
    scripts = ("window_chain", "parallel_evaluate", "development_probe", "distributed_probe",
               "evaluation_pair", "window_comparison", "random_baseline", "daily_comparison", "expanded_chain_report")
    sources = [Path(__file__).resolve(), *[ROOT / f"scripts/recflow/{name}.py" for name in scripts],
               *sorted((ROOT / "src/hstu_kvcache/recflow").glob("*.py")),
               *sorted((ROOT / "src/hstu_kvcache/models").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out / "launcher_source.py")
    save(out / "launch.json", dict(started_unix=time.time(), canary=args.canary,
        settings=str(config), settings_sha256=digest(config), panels_sha256=digest(panels),
        canary_report=str(args.canary_report) if args.canary_report else None,
        authorization=setting["authorization"], resource_estimate=setting["resource_estimate"],
        source_sha256=hashes, interpretation=setting["continuation_rule"]))

    def run(name, command):
        assert digest(config) == read(out / "launch.json")["settings_sha256"]
        assert all(digest(ROOT / path) == sha for path, sha in hashes.items()), "Execution source changed during the sealed run"
        started = time.time()
        execution = dict(command=command, started_unix=started)
        save(out / f"{name}.execution.json", execution)
        save(out / "progress.json", dict(stage=name, **execution))
        print(json.dumps(dict(stage=name, started_unix=started)), flush=True)
        with (out / f"{name}.log").open("x") as log:
            result = subprocess.run(command, cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"), stdout=log, stderr=subprocess.STDOUT)
        execution.update(elapsed_seconds=time.time() - started, exit_code=result.returncode)
        save(out / f"{name}.execution.json", execution)
        (out / f"{name}.exit_status.txt").write_text(str(result.returncode) + "\n")
        if result.returncode:
            save(out / "progress.json", dict(failed=True, stage=name, **execution))
            raise SystemExit(result.returncode)

    phases = ("A", "B") if args.canary else tuple("ABCDEF")
    source, before = None, 0
    for phase in phases:
        epochs = 1 if args.canary or phase != "A" else setting["base_epochs"]
        command = ["torchrun", "--standalone", "--nproc-per-node=4", "scripts/recflow/window_chain.py",
            "--config", str(config), "--phase", phase, "--epochs", str(epochs),
            "--output", str(out / phase), "--eval-devices", "0", "1", "2", "3"]
        if source:
            command += ["--checkpoint", str(source)]
        if args.canary:
            command.append("--canary")
        run(phase, command)
        source, before = verify_epoch(out / phase, setting, phase, epochs, before, source, args.canary)
        evaluation = out / phase / f"{phase}_epoch{epochs}" / "evaluation"
        if phase == "A":
            run("A_random", [sys.executable, "scripts/recflow/random_baseline.py", "--runs", str(evaluation),
                "--output", str(out / "A_random"), "--draws", "5000", "--seed", "20260918"])
        else:
            predecessor = chr(ord(phase) - 1)
            day = setting["windows"][phase]["compare_parent_and_current"][0]
            pair = predecessor + phase + "_comparison"
            run(pair, [sys.executable, "scripts/recflow/evaluation_pair.py",
                "--parent", str(out / phase / f"parent_{predecessor}_d{day}_{day}"),
                "--current", str(evaluation), "--output", str(out / pair),
                "--primary-scope", setting["primary_scope"], "--primary-metric", setting["primary_metric"]])
        if not args.canary:
            run(f"{phase}_report", [sys.executable, "scripts/recflow/expanded_chain_report.py",
                "--root", str(out), "--config", str(config)])
    if args.canary:
        save(out / "canary_check.json", dict(passed=True, settings_sha256=digest(config),
            panels_sha256=digest(panels), cohort_users=setting["arguments"]["cohort_users"],
            phases=list(phases), complete_requests_per_phase=263, final_optimizer_steps=before,
            source_sha256=hashes, quality_interpretation="Execution/numerical/lineage canary only. Tiny sampled/free scores do not establish recommendation quality.",
            existing_numerical_reference="results/recflow/development/ddp_6l_h192_heads6_c1024_k1m_seed17_gpu0123_retry/summary.json",
            existing_known_positive_evaluation_reference="results/recflow/development/gloo_resident_6l_parallel_eval_canary_seed17/summary.json"))
    save(out / "progress.json", dict(complete=True, canary=args.canary, phases=list(phases), completed_unix=time.time()))
    print(json.dumps(dict(complete=True, canary=args.canary, output=str(out))), flush=True)


if __name__ == "__main__":
    main()
