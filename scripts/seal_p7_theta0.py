#!/usr/bin/env python3
"""Audit and seal completed P7.7 theta0 development runs.

This script is deliberately post-training only.  It reads the twelve frozen
run directories, verifies their immutable inputs and budgets, applies the
pre-registered development sanity gates, and writes reference manifests.  It
does not instantiate a dataset loader and therefore cannot read qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/p7/theta0_training/runs"
OUT = ROOT / "results/p7/theta0_training"
BUNDLE = OUT / "frozen_theta0_bundle_v1"

MODELS = ("m0_n", "m0_r", "m0_f", "m1")
SEEDS = (17, 37, 71)
MODEL_TASKS = {"m0_n": ("N",), "m0_r": ("R",), "m0_f": ("F",), "m1": ("N", "R", "F")}
NO_INFORMATION_LOSS = {
    "N": 4.605170185988093,
    "R": 5.534747289891543,
    "F": 0.3320749273830682,
}
EXPECTED = {
    "contract_hash": "8ead5b0ee25ad2aaafa5c8fa174d59ceadea88cba96f97a1c367067496492837",
    "base_bundle_hash": "4221f0b95245bb6b3a735c86d4cdb2603c6a7d8b173c7cac7c3dd5e28a64e42c",
    "train_manifest_hash": "5a24def8a2ef9e51fc78f52b5a0882564e2661bb291392ff40e02b36eb3645ea",
    "development_manifest_hash": "7685c1fb42b445e1c8f31cd1b8e852630d87e68e9e40d3b668c3f382f8a20e93",
    "qualification_manifest_hash": "61f86afefe2dd32eca33b625fd7af1338fdd1f4850550ca4a3d968d1170a3857",
    "training_code_hash": "4adbb7565174b87a733a9b6cc59591625e5ddac821b41db4d79f72e79f65e434",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sum_budget(parts: list[dict[str, int]]) -> dict[str, int]:
    keys = ("unique_queries", "query_presentations", "candidate_rows", "history_tokens", "token_layer_work", "optimizer_steps")
    return {key: sum(int(part[key]) for part in parts) for key in keys}


def main() -> None:
    runs: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    artifact_refs: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    for model in MODELS:
        for seed in SEEDS:
            run_dir = RUN_ROOT / f"{model}_seed{seed}"
            train_path = run_dir / "train.json"
            selection_path = run_dir / "checkpoint_selection.json"
            dev_path = run_dir / "dev_sanity.json"
            checkpoint_path = run_dir / "theta0_selected.pt"
            for path in (train_path, selection_path, dev_path, checkpoint_path):
                require(path.is_file(), f"missing required run artifact: {path}")

            train = load(train_path)
            selection = load(selection_path)
            dev = load(dev_path)
            require(train["model_name"] == model and train["seed"] == seed, f"run identity mismatch: {run_dir}")
            for key, value in EXPECTED.items():
                require(train[key] == value, f"{model}/{seed} immutable input mismatch: {key}")
            require(train["qualification_scored"] is False, f"{model}/{seed} scored qualification")
            require(dev["qualification_locked"] is True, f"{model}/{seed} qualification unlocked")
            require(dev["recent_or_H_evaluated"] is False, f"{model}/{seed} evaluated H")
            require(selection["selected_epoch"] == train["selected_epoch"], f"{model}/{seed} selection mismatch")
            require(selection["checkpoint_hash"] == train["checkpoint_hash"], f"{model}/{seed} selection hash mismatch")
            require(dev["checkpoint_hash"] == train["checkpoint_hash"], f"{model}/{seed} dev hash mismatch")
            require(sha256(checkpoint_path) == train["checkpoint_hash"], f"{model}/{seed} checkpoint bytes mismatch")

            task_gates: dict[str, Any] = {}
            for task in MODEL_TASKS[model]:
                metrics = dev["tasks"][task]
                deployment = float(metrics["deployment_loss"])
                base = float(metrics["base_only_loss"])
                residual_std = float(metrics["residual_score_std"])
                task_gates[task] = {
                    "deployment_loss": deployment,
                    "base_only_loss": base,
                    "increment_over_base_loss_reduction": base - deployment,
                    "no_information_loss": NO_INFORMATION_LOSS[task],
                    "better_than_no_information": deployment < NO_INFORMATION_LOSS[task],
                    "better_than_base": deployment < base,
                    "finite_nonzero_residual": residual_std > 0.0,
                    "residual_score_std": residual_std,
                }
                require(task_gates[task]["better_than_no_information"], f"{model}/{seed}/{task} fails no-information gate")
                require(task_gates[task]["finite_nonzero_residual"], f"{model}/{seed}/{task} has zero residual")

            if model != "m1":
                only_task = MODEL_TASKS[model][0]
                require(task_gates[only_task]["better_than_base"], f"{model}/{seed} fails M0 base increment gate")
            else:
                require(
                    task_gates["R"]["better_than_base"] or task_gates["F"]["better_than_base"],
                    f"m1/{seed} fails R-or-F positive increment gate",
                )
                logged_tasks = {row["task"] for row in train["step_logs"]}
                require(logged_tasks == {"N", "R", "F"}, f"m1/{seed} task routing incomplete")
                for task in ("N", "R", "F"):
                    rows = [row for row in train["step_logs"] if row["task"] == task]
                    require(any(float(row["gradient_norm"]) > 0 for row in rows), f"m1/{seed}/{task} has no gradient")

            runs[model].append({"train": train, "selection": selection, "dev": dev})
            gate_rows.append({"model": model, "seed": seed, "tasks": task_gates})
            artifact_refs.append(
                {
                    "model": model,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "checkpoint_sha256": train["checkpoint_hash"],
                    "selected_epoch": train["selected_epoch"],
                    "selection_objective": train["selected_development_objective"],
                    "train_json_sha256": sha256(train_path),
                    "selection_json_sha256": sha256(selection_path),
                    "dev_json_sha256": sha256(dev_path),
                }
            )

    budget_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_model = {model: next(row for row in runs[model] if row["train"]["seed"] == seed)["train"]["budget"] for model in MODELS}
        aggregate_m0 = sum_budget([per_model["m0_n"]["N"], per_model["m0_r"]["R"], per_model["m0_f"]["F"]])
        aggregate_m1 = sum_budget([per_model["m1"][task] for task in ("N", "R", "F")])
        # Optimizer steps differ by design: one shared optimizer versus three task-specific optimizers.
        for key in ("unique_queries", "query_presentations", "candidate_rows", "history_tokens", "token_layer_work"):
            require(aggregate_m0[key] == aggregate_m1[key], f"seed {seed} M1-vs-aggregate-M0 budget mismatch: {key}")
        for task, model in (("N", "m0_n"), ("R", "m0_r"), ("F", "m0_f")):
            for key in ("unique_queries", "query_presentations", "candidate_rows", "history_tokens", "token_layer_work"):
                require(per_model[model][task][key] == per_model["m1"][task][key], f"seed {seed} M1 task budget mismatch: {task}/{key}")
        budget_rows.append(
            {
                "seed": seed,
                "per_model": per_model,
                "three_m0_aggregate": aggregate_m0,
                "m1_aggregate": aggregate_m1,
                "query_candidate_history_token_work_matched": True,
                "optimizer_steps_matched": aggregate_m0["optimizer_steps"] == aggregate_m1["optimizer_steps"],
            }
        )

    for model in MODELS:
        compact_train = []
        compact_selection = []
        for row in runs[model]:
            train = row["train"]
            selection = row["selection"]
            compact_train.append(
                {
                    "seed": train["seed"],
                    "selected_epoch": train["selected_epoch"],
                    "selected_development_objective": train["selected_development_objective"],
                    "checkpoint_hash": train["checkpoint_hash"],
                    "budget": train["budget"],
                    "optimizer": train["optimizer"],
                }
            )
            compact_selection.append(
                {
                    "seed": train["seed"],
                    "rule": selection["rule"],
                    "selected_epoch": selection["selected_epoch"],
                    "trace": [
                        {"epoch": item["epoch"], "checkpoint_objective": item["checkpoint_objective"]}
                        for item in selection["trace"]
                    ],
                    "checkpoint_hash": selection["checkpoint_hash"],
                }
            )
        dump(OUT / f"{model}_theta0_train_v1.json", {"model": model, "runs": compact_train})
        dump(OUT / f"{model}_checkpoint_selection_v1.json", {"model": model, "runs": compact_selection})

    sanity = {
        "status": "passed_development_only_not_H",
        "qualification_scored": False,
        "recent32_or_H_evaluated": False,
        "gates": gate_rows,
        "interpretation": "Full-512 theta0 training/dev sanity only; not long-state qualification.",
    }
    budget = {
        "status": "passed",
        "comparison": {
            "m1_vs_each_task_specific_m0_compute_matched": False,
            "m1_vs_three_m0_aggregate_query_candidate_history_token_work_matched": True,
            "optimizer_step_count_matched": all(
                row["three_m0_aggregate"]["optimizer_steps"] == row["m1_aggregate"]["optimizer_steps"]
                for row in budget_rows
            ),
        },
        "seeds": budget_rows,
    }
    dump(OUT / "theta0_dev_sanity_v1.json", sanity)
    dump(OUT / "theta0_training_budget_audit_v1.json", budget)
    bundle_manifest = {
        "bundle": "frozen_theta0_bundle_v1",
        "status": "frozen_development_theta0_only",
        "contract_hash": EXPECTED["contract_hash"],
        "base_bundle_hash": EXPECTED["base_bundle_hash"],
        "qualification_manifest_hash_recorded_not_read": EXPECTED["qualification_manifest_hash"],
        "qualification_scored": False,
        "artifacts": artifact_refs,
    }
    dump(BUNDLE / "bundle_manifest.json", bundle_manifest)
    print(json.dumps({"status": "passed_and_sealed", "runs": len(artifact_refs), "bundle_manifest_sha256": sha256(BUNDLE / "bundle_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
